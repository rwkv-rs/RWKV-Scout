"""RWKV-native agent-loop planner.

The local RWKV service uses the online G1i function-calling transcript. Keep
the wire format explicit:

    System: Tools: [...]
    Return only a JSON function call.
    User: <instructions, task, or observation>
    Assistant: ```json
    {"name":"tool","arguments":{...}}
    User: Function output: <tool result>

The planner owns the conversation history.  The application only executes
the name and arguments returned by the model; it does not manufacture a
search provider or a rewritten query when parsing fails.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

from clients.llm_client import LLMClient
from config import (
    get_llm_context_length,
    get_model_replan_temperature,
    get_model_stage_temperature,
    is_local_provider,
    model_sampling_parameters,
)
from tools.builtin import load_builtin_tools
from tools.registry import ToolRegistry
from utils.model_events import visible_model_text
from utils.rwkv_json_protocol import normalize_json_object_envelope
from utils.chunker import get_token_count
from utils.retrieval_ledger import bounded_request_arguments, route_id
from utils.context_budget import (
    observation_chars,
    planner_prompt_tokens,
)
from utils.error_policy import classify_error
from utils.hard_literals import hard_literal_keys, untrusted_hard_literals
from utils.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    assistant_json_prefix,
    render_tool_transcript,
)
from runtime.transcript import render_rwkv_transcript
from agent.tool_protocol import canonicalize_tool_call
from agent.runtime_contracts import (
    EVIDENCE_RECORD_SET_CONTRACT,
    EVIDENCE_REVIEW_CONTRACT,
    TASK_PLAN_CONTRACT,
    require_runtime_contract,
)
from agent.task_plan_contract import (
    MAX_MODEL_TASK_RECORDS,
    compact_task_plan,
    field_ids,
    normalize_task_plan,
    record_question,
    record_id,
    task_records,
)


def _same_nonempty_route_target(left: str, right: str) -> bool:
    """Compare normalized route targets without interpreting their content."""

    return bool(left) and left in {right}


def _declared_task_record_id(value: Any, planned_task_record_ids: Iterable[str]) -> str:
    """Resolve only a task-record ID explicitly present in RWKV output.

    G1i sometimes emits ``"P2: missing date"`` or ``{"id": "P2", ...}``
    even when the protocol asks for the bare ID.  This is protocol
    canonicalization: it never guesses a Task Record from evidence or from the
    missing-fact description.
    """

    planned = [
        str(task_record_id).strip()
        for task_record_id in planned_task_record_ids
        if str(task_record_id).strip()
    ]
    if isinstance(value, dict):
        text = str(
            value.get("id")
            or value.get("task_record_id")
            or value.get("task_point_id")
            or value.get("point_id")
            or ""
        ).strip()
    else:
        text = str(value or "").strip()
    if not text:
        return ""
    exact = next(
        (task_record_id for task_record_id in planned if text.casefold() == task_record_id.casefold()),
        "",
    )
    if exact:
        return exact
    matches = [
        task_record_id
        for task_record_id in planned
        if re.search(
            rf"(?<![\w-]){re.escape(task_record_id)}(?![\w-])",
            text,
            flags=re.IGNORECASE,
        )
    ]
    return matches[0] if len(matches) == 1 else ""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Decode one structured object without repairing model-authored content."""

    return normalize_json_object_envelope(text).payload


def _canonicalize_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the standalone transport adapter."""

    return canonicalize_tool_call(payload)


# RWKV declares its next retrieval target's object family in a dedicated
# single-purpose step; the declaration only filters how the tool catalog is
# presented on the following decision. web_search, deterministic tools and
# finish_task remain available for every family, so the declaration can never
# force or forbid a route — both steps stay model-authored.
_ROUTE_FAMILY_CONNECTOR_OPERATIONS: dict[str, tuple[str, ...]] = {
    "target_weather": ("weather_current", "weather_alerts"),
    "target_github": ("github_repository", "github_code", "github_release"),
    "target_package_registry": ("crates_release", "pypi_release", "npm_release"),
    "target_scholarly": ("paper", "paper_series"),
    "target_security_advisories": ("security_advisories",),
    "target_general_web": (),
}


class Planner:
    """Own the model-facing plan, tool decisions, and global retrieval state."""

    def __init__(self):
        load_builtin_tools()
        self.llm = LLMClient()
        self._messages: list[dict[str, Any]] = []
        self._task_plan: dict[str, Any] | None = None
        # ``_messages`` is an audit trace, not recurrent model context. Every
        # decision is an independent online-G1i request built from the current
        # state projection and the latest compact outcome. Replaying a prior
        # Assistant function call anchors RWKV to the completed path.
        self._latest_routing_observation = ""
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed: int | None = None
        self._replan_generation = 0
        self._active_replan_review: dict[str, Any] | None = None
        self._trusted_runtime_context = ""

    def reset(self) -> None:
        self._messages = []
        self._task_plan = None
        self._latest_routing_observation = ""
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed = None
        self._replan_generation = 0
        self._active_replan_review = None
        self._trusted_runtime_context = ""

    def execution_transcript(self) -> str:
        """Return the visible routing transcript for final summarization."""
        return self._render_transcript(self._messages)

    @staticmethod
    def _validate_task_plan(payload: dict[str, Any]) -> dict[str, Any]:
        require_runtime_contract(payload, TASK_PLAN_CONTRACT)
        return normalize_task_plan(payload, max_records=MAX_MODEL_TASK_RECORDS)

    def create_task_plan(self, user_query: str, env_context: str = "") -> dict[str, Any]:
        """Ask RWKV for factual obligations, never a prewritten workflow."""

        prompt = (
            "System: Return only one JSON object.\n\n"
            "User: You are the RWKV factual-record planner. Preserve the user's exact goal. "
            "Create one record for each distinct answer record the user requests. Start with exactly one record and "
            "split it only when the question truly contains different subjects, a comparison, or different "
            "historical/current identities. A record is a row, not one field or one clause. A record has one stable subject and one historical/current/version/advisory "
            "identity. Put every requested field belonging to that same record in the same fields array. "
            "For example, a current release's identifier, title, date and theme are one record; a latest batch's "
            "item IDs, products and deadlines are one collection record. Do not split those fields into separate "
            "records. Preserve explicit answer cardinality and shape in the record question: words such as each, "
            "respectively, both, two dates, two sentences, and minimal snippet are user requirements, not disposable "
            "wording. Never collapse two requested dates into one generic date field. A minor release line and a "
            "specific patch release are different version identities even when they share one project. Pronouns such "
            "as that version, that release, that batch, those items, or 该批 refer back to the same record and must not create new records. "
            "subject names only the stable entity or object and must not repeat a field label. relation briefly "
            "names the shared record identity, such as current release, historical launch, latest batch, or "
            "procedure. A short lookup or how-to uses one record. "
            "Never create records for search, discovery, reading, extraction, verification, cross-checking, "
            "citation, formatting, or answer writing. Do not choose a tool, provider, query, URL, domain, source "
            "policy, status, completion gate, or answer. Fields are short labels in the user's language, never "
            "values. A source constraint such as according to official documentation changes where evidence should "
            "come from; it is not an answer field. Do not add a source, citation, official page, documentation link, "
            "or URL field unless the user explicitly asks the answer to contain that link or URL. "
            "set_semantics is single for one record, collection for a required non-empty list, and "
            "possibly_empty when an empty set is a valid answer. "
            "Use at most four records. Return exactly one compact object in this schema and no explanation: "
            f'{{"contract":"{TASK_PLAN_CONTRACT}","goal":"...",'
            '"records":[{"record_id":"P1","question":"...","subject":"...",'
            '"relation":"...","fields":[{"field_id":"P1:F1","name":"..."}],'
            '"time_scope":"current|historical|timeless|unspecified",'
            '"set_semantics":"single|collection|possibly_empty",'
            '"premise_requires_verification":false}]}.\n\n'
            f"User goal: {user_query}\n"
            f"Current environment summary: {env_context[:1200]}\n"
        )
        raw = ""
        last_nonempty_raw = ""
        last_error = ""
        # Structured protocol errors receive one correction at the same
        # low-entropy sampling profile.  Additional high-temperature repairs
        # previously expanded malformed plans instead of stabilising JSON.
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                request_prompt += (
                    "\nCorrection: the previous output was truncated or invalid. Return one compact, complete "
                    f"{TASK_PLAN_CONTRACT} object only. Merge repeated or overlapping records; use no more than four factual "
                    "records. Default to one record. For a short how-to, use one record. Do not invent values, APIs, flags, URLs, sources, "
                    "source-link fields, workflow steps, examples, or variants. A request to use official evidence is not a request to output a link. "
                    "Do not add explanation, Markdown, or a second object. "
                    "The prior validation problem was: "
                    + last_error[:700]
                )
            # Keep repair instructions in the user-side prompt.  Appending
            # them after the Assistant continuation marker makes RWKV copy
            # the repair text instead of regenerating the JSON object.
            request_prompt += f"\n{assistant_json_prefix(enable_think=False, prefill_object=False)}"
            try:
                completion_budget = self._completion_budget(request_prompt)
                if attempt:
                    # A repair must be short enough to finish after an overlong
                    # first continuation; the initial budget remains dynamic up
                    # to the configured 10K ceiling.
                    completion_budget = min(completion_budget, 4096)
                sampling_stage = "task_plan_repair" if attempt else "task_plan"
                sampling_temperature = get_model_stage_temperature(sampling_stage)
                with model_sampling_parameters(
                    sampling_temperature,
                    stage=sampling_stage,
                    policy_reason=(
                        "task_plan_protocol_or_semantic_repair"
                        if attempt
                        else "structured_goal_decomposition"
                    ),
                ):
                    response = self.llm.text_completion(
                        request_prompt,
                        max_tokens=completion_budget,
                        stop=JSON_CALL_STOP_SUFFIXES,
                    )
                raw = str(response.content or "")
                if raw.strip():
                    last_nonempty_raw = raw
                normalized_response = normalize_json_object_envelope(raw)
                task_plan = self._validate_task_plan(normalized_response.payload)
                semantic_warnings = self._task_plan_semantic_warnings(
                    task_plan,
                    user_query=user_query,
                    env_context=env_context,
                )
                if semantic_warnings:
                    raise ValueError(
                        "task plan introduced concrete anchors absent from the user/environment: "
                        + "; ".join(semantic_warnings[:6])
                    )
                task_plan["sampling_temperature"] = sampling_temperature
                task_plan["sampling_seed"] = None
                task_plan["plan_attempts"] = attempt + 1
                task_plan["protocol_input_format"] = normalized_response.input_format
                task_plan["protocol_normalized"] = normalized_response.normalized
                return task_plan
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if classify_error(exc) == "timeout":
                    break
        # A malformed plan must not erase an otherwise answerable user request.
        # This fallback preserves the exact goal as one factual Task Record; it does
        # not choose retrieval actions, facts, sources, or the final answer.
        fallback = normalize_task_plan(
            {
                "goal": user_query,
                "records": [
                    {
                        "record_id": "P1",
                        "question": user_query,
                        "fields": [],
                        "time_scope": "unspecified",
                    }
                ],
            },
            fallback_goal=user_query,
        )
        fallback.update(
            {
                "plan_fallback": True,
                "plan_error": last_error or "task plan generation failed",
                "raw_model_output": visible_model_text(last_nonempty_raw or raw),
                "plan_attempts": 2,
            }
        )
        return fallback

    @staticmethod
    def _task_plan_semantic_warnings(
        task_plan: dict[str, Any],
        *,
        user_query: str,
        env_context: str,
    ) -> list[str]:
        """Detect guessed concrete anchors and ask RWKV itself for one repair.

        This checker never inserts, deletes, or replaces a plan value. It
        compares typed concrete anchors against immutable user/runtime input.
        After the bounded repair fails, ``create_task_plan`` uses its exact-goal
        fallback instead of legitimising the guessed value downstream.
        """

        plan_text = json.dumps(task_plan, ensure_ascii=False)
        allowed_text = f"{str(user_query or '')}\n{str(env_context or '')}"
        allowed_keys = hard_literal_keys([allowed_text])
        warnings = [
            f"unexpected_{literal.kind}:{literal.surface_text}"
            for literal in untrusted_hard_literals(
                plan_text,
                allowed_keys=allowed_keys,
                provenance="task_plan",
            )
        ]
        example_markers = ("例如", "（如", "(e.g.", "(for example")
        if any(marker.casefold() in plan_text.casefold() for marker in example_markers) and not any(
            marker.casefold() in allowed_text.casefold() for marker in example_markers
        ):
            warnings.append("invented_example_marker")
        return warnings

    def begin_task(
        self,
        user_query: str,
        env_context: str,
        task_plan: dict[str, Any],
        phase: str = "DISCOVERY",
    ) -> None:
        """Start the tool transcript with the model-generated plan as data."""
        self._task_plan = task_plan
        self._trusted_runtime_context = str(env_context or "")[:2400]
        self._messages = []
        self._latest_routing_observation = ""
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed = None
        self._replan_generation = 0
        self._active_replan_review = None
        self._messages = [
            {
                "role": "system",
                "content": self._system_prompt(phase),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {str(user_query or '').strip()[:3000]}\n"
                    "Research brief (model-generated advisory context; you still decide every action and finish point):\n"
                    f"{json.dumps(self._compact_task_plan(task_plan), ensure_ascii=False, separators=(',', ':'))}\n"
                    f"Runtime state: {str(env_context or '').strip()}\n"
                    f"Phase: {phase}\n\n"
                    f"{self._decision_guidance_for_current_plan(phase)}"
                ),
            },
        ]
        self._trim_conversation()

    @staticmethod
    def _validate_evidence_review(
        payload: dict[str, Any],
        *,
        task_plan: dict[str, Any] | None = None,
        allowed_evidence_record_ids: Iterable[str] = (),
        allowed_route_ids: Iterable[str] = (),
        allow_replan: bool = True,
    ) -> dict[str, Any]:
        """Validate the binary Evidence Review contract without changing its action."""

        if not isinstance(payload, dict):
            raise ValueError("evidence-review output must be a JSON object")
        if set(payload) != {"name", "arguments"}:
            raise ValueError(
                "evidence-review output must contain exactly name and arguments"
            )
        action = str(payload.get("name") or "").strip().casefold()
        if action not in {"continue_retrieval", "write_answer"}:
            raise ValueError(
                "evidence-review name must be continue_retrieval or write_answer"
            )
        if action == "continue_retrieval" and not allow_replan:
            raise ValueError(
                "continue_retrieval is unavailable after the retrieval resource boundary"
            )
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("evidence-review arguments must be an object")
        if arguments != {}:
            raise ValueError(f"{action} arguments must be empty")
        if action == "write_answer":
            return {
                "contract": EVIDENCE_REVIEW_CONTRACT,
                "decision": "finish",
                "selected_action": action,
                "review_owner": "rwkv",
                "arguments_normalized": False,
            }

        return {
            "contract": EVIDENCE_REVIEW_CONTRACT,
            "decision": "replan",
            "selected_action": action,
            "review_owner": "rwkv",
            "arguments_normalized": False,
        }

    @classmethod
    def _validate_evidence_review_continuation(
        cls,
        raw: Any,
        *,
        task_plan: dict[str, Any] | None = None,
        allowed_evidence_record_ids: Iterable[str] = (),
        allowed_route_ids: Iterable[str] = (),
        allow_replan: bool = True,
    ) -> dict[str, Any]:
        """Losslessly normalize and validate one model-authored review action."""

        normalized = normalize_json_object_envelope(raw)
        payload = normalized.payload
        canonical = canonicalize_tool_call(payload)
        if canonical.get("task_record_id"):
            raise ValueError("evidence-review output must not bind a task record")
        review = cls._validate_evidence_review(
            {
                "name": canonical["name"],
                "arguments": canonical["arguments"],
            },
            task_plan=task_plan,
            allowed_evidence_record_ids=allowed_evidence_record_ids,
            allowed_route_ids=allowed_route_ids,
            allow_replan=allow_replan,
        )
        review["protocol_input_format"] = normalized.input_format
        review["protocol_normalized"] = bool(
            normalized.normalized
            or payload
            != {
                "name": canonical["name"],
                "arguments": canonical["arguments"],
            }
            or review.get("arguments_normalized")
        )
        return review

    @staticmethod
    def _evidence_review_tools(
        task_plan: dict[str, Any] | None = None,
        *,
        allowed_evidence_record_ids: Iterable[str] = (),
        allowed_route_ids: Iterable[str] = (),
        allow_replan: bool = True,
    ) -> list[dict[str, Any]]:
        """Return the complete two-action catalog shown to online G1i."""

        del task_plan, allowed_evidence_record_ids, allowed_route_ids

        empty_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        tools = []
        if allow_replan:
            tools.append({
                "name": "continue_retrieval",
                "description": (
                    "Choose only when another targeted search is needed before an "
                    "accurate, useful answer: the central requested fact is absent, "
                    "current/latest is supported only by stale material, or relevant "
                    "records conflict. The rebuilt Planner will diagnose and choose "
                    "the next route from authoritative state."
                ),
                "arguments": empty_arguments,
            })
        tools.append({
                "name": "write_answer",
                "description": (
                    "Choose when the retained source spans are sufficient for an "
                    "accurate, useful answer. The answer may state that a minor "
                    "requested detail is not established instead of inventing it."
                ),
                "arguments": empty_arguments,
            })
        return tools

    def review_evidence(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        exact_evidence_text: str,
        routing_context: str = "",
        *,
        resolution_advisory: str = "",
        allowed_evidence_record_ids: Iterable[str] = (),
        allowed_route_ids: Iterable[str] = (),
        terminal: bool = False,
    ) -> dict[str, Any]:
        """Ask RWKV for one explicit evidence-to-writer routing action."""

        allowed_evidence_record_ids = list(allowed_evidence_record_ids)
        allowed_route_ids = list(allowed_route_ids)

        projected_target = json.dumps(
            self._evidence_review_plan_projection(task_plan),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if terminal:
            review_instruction = (
                "The retrieval resource boundary has been reached, so no additional "
                "search route is available. Hand the retained source spans to the Writer "
                "by calling write_answer. The Writer may answer the supported portion and "
                "state that an unsupported detail is not established; do not invent it. "
            )
        else:
            review_instruction = (
                "Before final synthesis, perform one minimal evidence check from the retained "
                "source spans. Choose continue_retrieval only if the central answer would be "
                "materially wrong or unusable without another targeted search. Otherwise "
                "choose write_answer. Do not require perfect completeness: the Writer can "
                "state that a minor detail is not established instead of inventing it. "
            )
        user_prompt = (
            review_instruction
            + "Body spans "
            "are the factual material; titles and URLs identify their source records. "
            "If an Evidence Resolution advisory is present, treat it only as a prior "
            "RWKV attention map: verify every selected or conflicting E-* against its exact "
            "body span and reject the map when the text does not support it. "
            "Do not diagnose or serialize a gap in this call. "
            + (
                "If you choose continue_retrieval, the rebuilt Planner will inspect the "
                "original goal, retained evidence and routing state and choose the next "
                "action. Both functions take empty arguments. "
                if not terminal
                else "The write_answer function takes an empty arguments object. "
            )
            + "Do not write a query, prose reason, "
            "answer, URL, source ranking, ID, or any extra key.\n\n"
            f"USER QUESTION:\n{str(user_query or '')}\n\n"
            f"TASK PLAN RECORDS TO REVIEW:\n{projected_target}\n\n"
            + (
                "EVIDENCE RESOLUTION ADVISORY (control metadata only; never factual evidence):\n"
                f"{str(resolution_advisory or '')}\n\n"
                if str(resolution_advisory or "").strip()
                else ""
            )
            + (
                "RETRIEVAL ROUTING STATE (controller observations only; not factual evidence):\n"
                f"{str(routing_context or '')}\n\n"
                + (
                    "A frozen route has already been executed and cannot be repeated. Choose "
                    "continue_retrieval only if a materially different route can still target "
                    "a central missing fact; otherwise choose write_answer from retained "
                    "evidence.\n\n"
                    if not terminal
                    else "Retrieval is closed at this resource boundary; use write_answer.\n\n"
                )
                if str(routing_context or "").strip()
                else ""
            )
            + "RETAINED SOURCE SPANS (immutable factual evidence only):\n"
            f"{str(exact_evidence_text or '')}\n\n"
            + "FINAL ACTION CONTRACT:\n"
            + (
                "Call exactly one of the two functions defined in the Tool Schema. "
                "Both calls use an empty arguments object. "
                if not terminal
                else "Call the write_answer function defined in the Tool Schema with an "
                "empty arguments object. "
            )
            + "Do not write the answer, "
            "reason, analysis, IDs, or any extra key."
        )
        tools = self._evidence_review_tools(
            task_plan,
            allowed_evidence_record_ids=allowed_evidence_record_ids,
            allowed_route_ids=allowed_route_ids,
            allow_replan=not terminal,
        )
        prompt = render_rwkv_transcript(
            [{"role": "user", "content": user_prompt}],
            tools=tools,
        )
        raw = ""
        sampling_temperature = get_model_stage_temperature("evidence_review")
        try:
            with model_sampling_parameters(
                sampling_temperature,
                stage="evidence_review",
                policy_reason=(
                    "terminal_evidence_handoff"
                    if terminal
                    else "binary_evidence_review"
                ),
            ):
                response = self.llm.text_completion(
                    prompt,
                    max_tokens=min(256, self._completion_budget(prompt)),
                    stop=JSON_CALL_STOP_SUFFIXES,
                )
            raw = str(response.content or "")
            review = self._validate_evidence_review_continuation(
                raw,
                task_plan=task_plan,
                allowed_evidence_record_ids=allowed_evidence_record_ids,
                allowed_route_ids=allowed_route_ids,
                allow_replan=not terminal,
            )
            review["raw_model_output"] = raw
            review["prompt"] = prompt
            review["sampling_temperature"] = sampling_temperature
            review["sampling_seed"] = None
            review["review_attempts"] = 1
            review["review_mode"] = "terminal" if terminal else "binary"
            return review
        except Exception as exc:
            error_class = classify_error(exc)
            if error_class in {
                "timeout",
                "network",
                "auth",
                "quota",
                "rate_limit",
                "provider",
            }:
                raise
            return {
                "contract": EVIDENCE_REVIEW_CONTRACT,
                "decision": "protocol_error",
                "error_class": "evidence_review_protocol",
                "message": f"{type(exc).__name__}: {exc}"[:1000],
                "raw_model_output": raw,
                "prompt": prompt,
                "review_attempts": 1,
                "sampling_temperature": sampling_temperature,
                "sampling_seed": None,
                "review_owner": "rwkv",
                "review_mode": "terminal" if terminal else "binary",
            }

    def rebuild_session_after_review(
        self,
        user_query: str,
        env_context: str,
        review: dict[str, Any],
        phase: str = "DISCOVERY",
        routing_observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Rebuild context and let RWKV produce the next replan action."""

        compact_review = {
            key: value
            for key, value in review.items()
            if key not in {"prompt", "raw_model_output"}
        }
        observation = dict(routing_observation or {})
        observation["status"] = "evidence_review_replan"
        observation["evidence_review"] = compact_review
        self.rebuild_session(
            user_query,
            env_context,
            observation,
            phase,
        )
        return self._request_replan_action(
            user_query,
            env_context,
            phase,
        )

    def rebuild_session(
        self,
        user_query: str,
        env_context: str,
        observation: dict[str, Any],
        phase: str = "DISCOVERY",
    ) -> None:
        """Drop prior action history while retaining task plan and global state."""

        self._messages = []
        # A rebuilt Planner request must not replay the failed Assistant call.
        # Keeping this pointer made the supposedly fresh G1i session start with
        # the exact duplicate query that triggered recovery, anchoring RWKV to
        # the frozen path even though the audit message history was cleared.
        self._latest_routing_observation = self._compact_rebuild_observation(
            observation
        )
        self._next_decision_sampling_stage = "planner_replan"
        self._replan_generation += 1
        # Replanning deliberately changes only the request-level temperature.
        # A generated seed made repeated replans reproducible but also locked
        # RWKV into recurring trajectories on unchanged evidence.
        self._next_decision_seed = None
        for key in ("evidence_review", "pending_replan"):
            review = observation.get(key)
            if isinstance(review, dict):
                review_with_routes = dict(review)
                for route_key in ("frozen_path", "frozen_paths"):
                    if route_key in observation:
                        review_with_routes[route_key] = observation[route_key]
                self._active_replan_review = self._compact_replan_review(
                    review_with_routes
                )
                break
        # A replan is a new G1i session, not a malformed continuation that
        # starts with a Function output lacking its prior Assistant call.
        # The frozen session remains in the audit trace; the new User task is
        # reconstructed from authoritative state and compact recovery data.
        self._messages = [
            {
                "role": "system",
                "content": self._system_prompt(phase),
            },
            {
                "role": "user",
                "content": (
                    self._build_isolated_decision_body(user_query, env_context, phase)
                    + "\nRecovery state:\n"
                    + self._latest_routing_observation
                ),
            },
        ]
        self._trim_conversation()

    def _request_replan_action(
        self,
        user_query: str,
        env_context: str,
        phase: str,
    ) -> dict[str, Any]:
        """Ask RWKV for exactly one function call in the rebuilt G1i session."""

        self._next_decision_sampling_stage = "planner_replan"
        self._next_decision_seed = None
        return self.plan_next_action(
            user_query,
            {},
            env_context,
            phase,
        )

    @staticmethod
    def _retrieval_action_requires_task_record(action: str) -> bool:
        metadata = ToolRegistry.metadata(action)
        return str(metadata.get("retrieval_role") or "").casefold() in {
            "discovery",
            "evidence",
            "retrieval",
        } or action in {"web_search", "connector_lookup"}

    def mark_replan_progress(self, task_record_id: str = "") -> None:
        """Return to the normal planner profile after any material new evidence.

        The binary reviewer no longer invents a missing-record hypothesis. A
        rebuilt Planner owns that diagnosis from the original goal, retained
        evidence and frozen paths.  Once a new retrieval materially advances
        state, keeping every later call at exploratory replan sampling only
        adds variance.
        """

        del task_record_id
        if not self._active_replan_review:
            return
        self._active_replan_review = None
        self._next_decision_sampling_stage = "planner"

    @staticmethod
    def _system_prompt(
        phase: str,
        task_mode: str = "",
        connector_operations: tuple[str, ...] | None = None,
    ) -> str:
        # Keep the verified prompt shape, but make the public catalog stable
        # across internal workflow phases. The model should re-enter the same
        # research loop rather than learn a recovery tool catalog.
        catalog_phase = "ALL"
        catalog_rows = json.loads(
            ToolRegistry.get_json_catalog(catalog_phase, model_visible_only=True)
        )
        # Keep the public descriptions and argument contracts, but remove
        # backend/plugin metadata.  Provider selection, fetching, cleaning,
        # chunking and evidence extraction are separate controller stages.
        #
        # connector_operations narrows only the presentation of the catalog to
        # the object family RWKV itself declared one step earlier. web_search,
        # deterministic tools and finish_task are always present, so no route
        # is ever forced; None keeps the full catalog (single-step fallback).
        projected_rows = []
        for row in catalog_rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", ""))
            arguments = row.get("arguments") or {"type": "object"}
            if name == "connector_lookup" and connector_operations is not None:
                if not connector_operations:
                    continue
                arguments = json.loads(json.dumps(arguments))
                operation_schema = (
                    arguments.get("properties", {}).get("operation")
                    if isinstance(arguments.get("properties"), dict)
                    else None
                )
                if isinstance(operation_schema, dict) and operation_schema.get("enum"):
                    operation_schema["enum"] = [
                        value
                        for value in operation_schema["enum"]
                        if value in connector_operations
                    ]
            projected_rows.append(
                {
                    "name": name,
                    "description": str(row.get("description") or "").splitlines()[0],
                    "arguments": arguments,
                }
            )
        catalog = json.dumps(projected_rows, ensure_ascii=False, separators=(",", ":"))
        return f"Tools: {catalog}\nReturn only a JSON function call."

    @staticmethod
    def _decision_guidance(
        phase: str,
        *,
        task_record_example: str = "P1",
        allow_unbound: bool = False,
    ) -> str:
        """Return task-local guidance outside the fixed G1i System envelope."""

        response_example: dict[str, Any] = {
            "name": "tool_name",
            "arguments": {},
        }
        if not allow_unbound:
            response_example["task_record_id"] = str(task_record_example or "P1")
            # Keep the key order used by the verified G1i transcript.
            response_example = {
                "name": response_example["name"],
                "task_record_id": response_example["task_record_id"],
                "arguments": response_example["arguments"],
            }

        if allow_unbound:
            task_record_guidance = (
                "For a retrieval call, task_record_id is optional routing metadata. Include an existing id only when the route clearly targets "
                "one factual record; omit it when the route can support several records or the target is uncertain. Evidence binding is performed "
                "later by RWKV from exact page spans. Never invent an id.\n"
            )
        else:
            task_record_guidance = (
                "For every retrieval call, task_record_id must copy the existing id of the one factual record the route primarily targets. "
                "This focuses extraction but does not bind a fetched page as evidence. Never invent an id. If it is omitted, the Controller "
                "attaches the only existing task record id without changing the model-authored tool or arguments.\n"
            )

        return (
            "Retrieval decision agent.\n"
            "Choose exactly one next action for the current question and research brief.\n"
            "Return one JSON object only: "
            + json.dumps(response_example, ensure_ascii=False, separators=(",", ":"))
            + ".\n"
            + task_record_guidance
            + "Use only the tool names and argument contracts in the catalog. Never emit an answer in tool arguments.\n"
            "Tool choice contract:\n"
            "- connector_lookup: structured weather, alerts, a specific GitHub repository/code/release, an exact crates.io/PyPI/npm package release, or a scholarly record. Choose one complete operation from its enum; operation already includes both source and action, so there is no separate scope.\n"
            "- web_search: general Web research, exact URLs, documentation, and product or service status pages; also use it as connector fallback.\n"
            "- calculator/date_diff/current_time: deterministic calculation or clock observations after required operands are known.\n"
            "- finish_task: stop retrieval and ask RWKV to synthesize from all retained sources.\n"
            "Retrieval tools internally handle provider selection, URL fetching, cleaning, chunking and evidence extraction.\n"
            "Tool Output is context for your next decision. You decide whether to retrieve again, use another tool, or finish.\n"
            "Choose calculator for arithmetic, current_time for the clock, date_diff for exact date distance, connector_lookup for structured sources, and web_search for general web research when useful. "
            "When one connector exactly matches the request (weather, weather alerts, a named GitHub repository/code/release, an exact crates.io/PyPI/npm package release, or an arXiv/DOI/scholarly paper), use that structured connector before general-web fallback. A company's product or service status page is general web, not a repository lookup.\n"
            "The shared ledger reports candidate Evidence Record counts for every factual Task Record, including zero. These are RWKV extraction observations, not completion judgements. "
            "Before repeating an already-covered route, compare all Task Records and inspect Evidence Record identities; consider a materially different retrieval when central evidence is still absent, while keeping the choice and query model-authored.\n"
            "For a current or latest request, use the current UTC date visible in Runtime state to disambiguate the search; do not assume that an older record is current.\n"
            "If the latest observation reports no new evidence or a frozen path, choose a materially different query, "
            "source family, tool, or finish from retained sources. Changing only generic suffixes such as official, "
            "latest, or primary source is not a materially different route. The controller never supplies a replacement query.\n"
            "Avoid needless exact repeats, but make every next-step and finish decision yourself.\n"
            "For date_diff, use only exact YYYY-MM-DD values already present in the question or visible evidence.\n"
            f"Current phase: {phase}"
        )

    def _decision_guidance_for_current_plan(self, phase: str) -> str:
        """Select the route-binding contract from factual-record cardinality.

        A one-record plan has exactly one structurally valid routing id, so the
        verified G1i example keeps that id explicit.  A multi-record retrieval
        may support several records and is therefore allowed to remain unbound
        until RWKV sees exact page spans.  Neither branch chooses a fact,
        source, query, tool or answer.
        """

        task_record_ids = [record_id(record) for record in task_records(self._task_plan)]
        return self._decision_guidance(
            phase,
            task_record_example=(task_record_ids[0] if task_record_ids else "P1"),
            allow_unbound=len(task_record_ids) != 1,
        )

    @staticmethod
    def _compact_task_plan(task_plan: dict[str, Any] | None) -> dict[str, Any]:
        """Project only RWKV's goal and factual focuses for later decisions.

        Source policies, mutable Task Record statuses and acceptance gates are not
        part of the decision brief.  They previously contradicted retrieved
        material when an otherwise useful source was not explicitly bound to a
        task-record ID, which repeatedly anchored RWKV on an already-run query.
        The complete plan remains in the trace for provenance.
        """

        return compact_task_plan(task_plan)

    @staticmethod
    def _evidence_review_plan_projection(
        task_plan: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Project only answer obligations, without replaying task-plan protocol.

        Evidence review asks RWKV for a different JSON schema.  Replaying a
        complete canonical Task Plan object in the same prompt made the recurrent
        model continue the embedded planner schema instead of emitting the
        requested review.  This projection keeps every user-facing evidence
        obligation but removes planner-only protocol and workflow fields.
        """

        compact = Planner._compact_task_plan(task_plan)
        return {
            "goal": str(compact.get("goal") or "")[:500],
            "records_to_check": [
                {
                    "record_id": record_id(record),
                    "question": record_question(record)[:500],
                    "subject": str(record.get("subject") or "")[:240],
                    "relation": str(record.get("relation") or "")[:160],
                    "fields": [
                        {"field_id": field["field_id"], "name": field["name"]}
                        for field in record.get("fields") or []
                        if isinstance(field, dict)
                        and str(field.get("field_id") or "").strip()
                        and str(field.get("name") or "").strip()
                    ][:16],
                    "time_scope": record.get("time_scope") or "unspecified",
                    "set_semantics": record.get("set_semantics") or "single",
                    "premise_requires_verification": bool(
                        record.get("premise_requires_verification")
                    ),
                }
                for record in compact.get("records") or []
                if isinstance(record, dict)
            ],
        }

    @staticmethod
    def _retrieval_attempts_from_environment(env_context: str) -> list[dict[str, Any]]:
        """Read exact attempted routes from the shared state projection.

        This is protocol validation, not query generation: a candidate that
        exactly repeats a completed request would be blocked by the executor
        on the next step anyway.  Rejecting it inside the same RWKV replan
        request gives the one allowed correction a chance to produce a real
        strategy change instead of spending another controller round.
        """

        lines = str(env_context or "").splitlines()
        ledger: dict[str, Any] = {}
        for index, line in enumerate(lines[:-1]):
            if not line.startswith("Shared retrieval ledger"):
                continue
            try:
                value = json.loads(lines[index + 1])
            except (TypeError, json.JSONDecodeError):
                break
            if isinstance(value, dict):
                ledger = value
            break
        output: list[dict[str, Any]] = []
        for row in ledger.get("queries") or []:
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            output.append(
                {
                    "action": str(row.get("action") or "web_search")[:120],
                    "operation": str(row.get("operation") or "")[:120],
                    "query": query[:500],
                    "task_record_id": str(row.get("task_record_id") or "")[:120],
                    "status": str(row.get("status") or "")[:80],
                }
            )
        return output[-16:]

    @staticmethod
    def _replan_environment_projection(
        env_context: str,
        *,
        include_evidence: bool = True,
    ) -> str:
        """Build valid bounded state with explicit completed/frozen routes.

        Replan is a fresh RWKV request. RWKV must see which exact requests ran
        and which path was frozen; hiding their text made "choose a different
        route" impossible to follow. The route history is bounded and clearly
        labelled as completed/frozen, while the next query remains RWKV-owned.

        Every field is bounded before serialization.  Never slice serialized
        JSON: doing that produced malformed Runtime state in real traces.
        """

        text = str(env_context or "").strip()
        if not text:
            return ""
        lines = text.splitlines()
        labelled: dict[str, Any] = {}
        labels = {
            "Question time/freshness policy": "freshness",
            "Shared retrieval ledger": "retrieval_ledger",
            "RWKV-selected grounded evidence records": "evidence_records",
            # Legacy trace label retained only so old saved runs remain
            # inspectable after the object-contract migration.
            "RWKV-bound exact evidence records": "evidence_records",
            "Bounded original source locators": "source_locators",
            "Minimal original source locators": "source_locators",
            "Persistent deterministic tool results": "deterministic_results",
        }
        for index, line in enumerate(lines[:-1]):
            key = next(
                (name for prefix, name in labels.items() if line.startswith(prefix)),
                "",
            )
            if not key:
                continue
            try:
                labelled[key] = json.loads(lines[index + 1])
            except (TypeError, json.JSONDecodeError):
                continue

        if not include_evidence:
            labelled.pop("evidence_records", None)
            labelled.pop("source_locators", None)

        freshness = labelled.get("freshness")
        if isinstance(freshness, dict):
            labelled["freshness"] = {
                key: str(freshness.get(key) or "")[:160]
                for key in (
                    "as_of",
                    "now",
                    "mode",
                    "unknown_date_policy",
                )
                if key in freshness
            }

        ledger = labelled.get("retrieval_ledger")
        if isinstance(ledger, dict):
            queries = [row for row in ledger.get("queries") or [] if isinstance(row, dict)]
            frozen_paths = [
                row for row in ledger.get("frozen_paths") or [] if isinstance(row, dict)
            ]
            route_status_counts: dict[str, int] = {}
            for row in queries:
                status = str(row.get("status") or "unknown")[:80]
                route_status_counts[status] = route_status_counts.get(status, 0) + 1
            frozen_reason_counts: dict[str, int] = {}
            for row in frozen_paths:
                reason = str(row.get("reason") or "unknown")[:120]
                frozen_reason_counts[reason] = frozen_reason_counts.get(reason, 0) + 1
            infrastructure = ledger.get("retrieval_infrastructure")
            compact_infrastructure = {}
            if isinstance(infrastructure, dict):
                compact_infrastructure = {
                    key: infrastructure.get(key)
                    for key in (
                        "complete",
                        "event_count",
                        "affected_task_record_ids",
                        "unresolved_chunk_count",
                        "transport_error_count",
                    )
                    if key in infrastructure
                }
            labelled["retrieval_ledger"] = {
                key: ledger.get(key)
                for key in (
                    "contract",
                    "schema_version",
                    "evidence_revision",
                    "round_count",
                    "source_count",
                    "attempted_url_count",
                    "replan_count",
                )
                if key in ledger
            }
            record_progress = []
            for row in ledger.get("task_record_progress") or []:
                if not isinstance(row, dict) or not str(row.get("id") or "").strip():
                    continue
                record_progress.append(
                    {
                        "id": str(row.get("id") or "")[:120],
                        "task_bound_evidence_record_count": int(
                            row.get("task_bound_evidence_record_count")
                            or row.get("candidate_evidence_record_count")
                            or 0
                        ),
                        "candidate_evidence_record_count": int(
                            row.get("candidate_evidence_record_count") or 0
                        ),
                        "total_evidence_record_count": int(
                            row.get("total_evidence_record_count") or 0
                        ),
                        "attempt_count": int(row.get("attempt_count") or 0),
                        "retrieval_state": str(row.get("retrieval_state") or "")[:80],
                    }
                )
            labelled["retrieval_ledger"]["task_record_progress"] = record_progress[:8]
            labelled["retrieval_ledger"]["unassigned_source_count"] = int(
                ledger.get("unassigned_source_count") or 0
            )
            labelled["retrieval_ledger"]["connector_runtime"] = [
                {
                    key: row.get(key)
                    for key in (
                        "provider",
                        "operation",
                        "status",
                        "available",
                        "cooldown_active",
                        "retry_after_seconds",
                        "error_class",
                        "message",
                    )
                    if key in row
                }
                for row in ledger.get("connector_runtime") or []
                if isinstance(row, dict)
            ][:8]
            # Present each route exactly once. Earlier projections repeated the
            # same query in both recent_routes and recent_frozen_routes (and a
            # third time in recovery state), which strongly anchored G1i to the
            # path the controller had just rejected.
            route_order: list[str] = []
            route_by_key: dict[str, dict[str, Any]] = {}
            for row in queries:
                tool = str(row.get("action") or row.get("tool") or "")[:120]
                query = str(row.get("query") or "")[:500]
                arguments = bounded_request_arguments(
                    row.get("arguments")
                    if isinstance(row.get("arguments"), dict)
                    else {"query": query} if query else {}
                )
                task_record_id = str(row.get("task_record_id") or "")[:120]
                key = str(row.get("route_id") or "") or route_id(
                    tool,
                    arguments,
                    task_record_id=task_record_id,
                )
                if key not in route_by_key:
                    route_order.append(key)
                route_by_key[key] = {
                    "route_id": key,
                    "tool": tool,
                    "operation": str(
                        row.get("operation") or arguments.get("operation") or ""
                    )[:120],
                    "arguments": arguments,
                    "task_record_id": task_record_id,
                    "status": str(row.get("status") or "")[:80],
                    "error_class": str(row.get("error_class") or "")[:160],
                    "error_message": str(row.get("error_message") or "")[:600],
                    "completed": True,
                    "frozen": False,
                    "step": row.get("step"),
                }
            for row in frozen_paths:
                tool = str(row.get("action") or row.get("tool") or "")[:120]
                query = str(row.get("query") or "")[:500]
                arguments = bounded_request_arguments(
                    row.get("arguments")
                    if isinstance(row.get("arguments"), dict)
                    else {"query": query} if query else {}
                )
                task_record_id = str(row.get("task_record_id") or "")[:120]
                key = str(row.get("route_id") or "") or route_id(
                    tool,
                    arguments,
                    task_record_id=task_record_id,
                )
                route = route_by_key.get(key)
                if route is None:
                    route_order.append(key)
                    route = {
                        "route_id": key,
                        "tool": tool,
                        "operation": str(
                            row.get("operation") or arguments.get("operation") or ""
                        )[:120],
                        "arguments": arguments,
                        "task_record_id": task_record_id,
                        "status": "",
                        "completed": False,
                        "step": row.get("step"),
                    }
                    route_by_key[key] = route
                route["frozen"] = True
                route["frozen_reason"] = str(row.get("reason") or "")[:120]
                if not route.get("task_record_id"):
                    route["task_record_id"] = str(
                        row.get("task_record_id") or ""
                    )[:120]
            route_history = [route_by_key[key] for key in route_order[-8:]]
            labelled["retrieval_ledger"].update(
                {
                    "previous_route_count": len(queries),
                    "frozen_route_count": len(frozen_paths),
                    "visible_unique_route_count": len(route_history),
                    "route_status_counts": route_status_counts,
                    "frozen_reason_counts": frozen_reason_counts,
                    "route_history": route_history,
                }
            )
            if compact_infrastructure:
                labelled["retrieval_ledger"][
                    "retrieval_infrastructure"
                ] = compact_infrastructure

        locators = labelled.get("source_locators")
        if isinstance(locators, dict):
            projected_sources = []
            for source in list(locators.get("sources") or [])[:3]:
                if not isinstance(source, dict):
                    continue
                projected_source: dict[str, Any] = {}
                for key, limit in (
                    ("source_id", 80),
                    ("title", 240),
                    ("url", 500),
                    ("published", 80),
                    ("updated", 80),
                    ("date", 80),
                    ("provider", 120),
                ):
                    if key in source:
                        projected_source[key] = str(source.get(key) or "")[:limit]
                if isinstance(source.get("task_record_ids"), list):
                    projected_source["task_record_ids"] = [
                        str(value)[:80] for value in source["task_record_ids"][:8]
                    ]
                source_freshness = source.get("freshness")
                if isinstance(source_freshness, dict):
                    projected_source["freshness"] = {
                        key: source_freshness.get(key)
                        for key in (
                            "policy",
                            "as_of",
                            "source_date",
                            "source_date_origin",
                            "state",
                        )
                        if key in source_freshness
                    }
                spans = [
                    span for span in source.get("spans") or [] if isinstance(span, dict)
                ]
                projected_source["source_object"] = (
                    dict(source.get("source_object") or {})
                    if isinstance(source.get("source_object"), dict)
                    else {}
                )
                projected_source["object_alignment"] = (
                    dict(source.get("object_alignment") or {})
                    if isinstance(source.get("object_alignment"), dict)
                    else {}
                )
                projected_source["object_alignments"] = [
                    dict(value)
                    for value in source.get("object_alignments") or []
                    if isinstance(value, dict)
                ][:8]
                projected_source["retrieval_bindings"] = [
                    dict(value)
                    for value in source.get("retrieval_bindings") or []
                    if isinstance(value, dict)
                ][:8]
                projected_source["source_span_count"] = int(
                    source.get("source_span_count") or len(spans)
                )
                projected_source["spans"] = []
                for span in spans[:2]:
                    text_value = str(span.get("text") or "")
                    projected_source["spans"].append(
                        {
                            "chunk_id": str(span.get("chunk_id") or "")[:120],
                            "start_char": span.get("start_char"),
                            "text": text_value[:350],
                            "truncated": bool(span.get("truncated"))
                            or len(text_value) > 350,
                        }
                    )
                projected_sources.append(projected_source)
            labelled["source_locators"] = {
                "source_count": locators.get("source_count"),
                "visible_source_count": len(projected_sources),
                "sources": projected_sources,
            }

        evidence_records = labelled.get("evidence_records")
        if isinstance(evidence_records, dict):
            projected_records = []
            for record in list(evidence_records.get("records") or [])[:8]:
                if not isinstance(record, dict):
                    continue
                projected_records.append(
                    {
                        "evidence_record_id": str(
                            record.get("evidence_record_id") or ""
                        )[:80],
                        "task_record_id": str(
                            record.get("task_record_id") or ""
                        )[:80],
                        "subject_key": str(record.get("subject_key") or "")[:240],
                        "record_key": str(record.get("record_key") or "")[:240],
                        "source_object": dict(record.get("source_object") or {})
                        if isinstance(record.get("source_object"), dict)
                        else {},
                        "object_alignment": dict(
                            record.get("object_alignment") or {}
                        )
                        if isinstance(record.get("object_alignment"), dict)
                        else {},
                        "object_alignments": [
                            dict(value)
                            for value in record.get("object_alignments") or []
                            if isinstance(value, dict)
                        ][:8],
                        "rwkv_subject_alignment": dict(
                            record.get("rwkv_subject_alignment") or {}
                        )
                        if isinstance(record.get("rwkv_subject_alignment"), dict)
                        else {},
                        "retrieval_bindings": [
                            dict(value)
                            for value in record.get("retrieval_bindings") or []
                            if isinstance(value, dict)
                        ][:8],
                        "field_ids": [
                            str(value)[:120]
                            for value in record.get("field_ids") or []
                            if str(value).strip()
                        ][:16],
                        "title": str(record.get("title") or "")[:240],
                        "url": str(record.get("url") or "")[:500],
                        "published": str(record.get("published") or "")[:80],
                        "updated": str(record.get("updated") or "")[:80],
                        "quote": str(record.get("quote") or "")[:520],
                        "quote_truncated": bool(record.get("quote_truncated"))
                        or len(str(record.get("quote") or "")) > 520,
                    }
                )
            labelled["evidence_records"] = {
                "contract": EVIDENCE_RECORD_SET_CONTRACT,
                "record_count": int(evidence_records.get("record_count") or 0),
                "visible_record_count": len(projected_records),
                "truncated": bool(evidence_records.get("truncated"))
                or len(projected_records)
                < int(evidence_records.get("record_count") or 0),
                "records": projected_records,
            }

        deterministic = labelled.get("deterministic_results")
        if isinstance(deterministic, list):
            labelled["deterministic_results"] = [
                {
                    key: (str(row.get(key) or "")[:240] if isinstance(row.get(key), str) else row.get(key))
                    for key in (
                        "tool",
                        "status",
                        "expression",
                        "value",
                        "result",
                        "days",
                        "iso",
                        "date",
                        "timezone",
                    )
                    if key in row
                }
                for row in deterministic[:4]
                if isinstance(row, dict)
            ]

        if not labelled and include_evidence:
            labelled = {"unstructured_state_summary": text[:1200]}

        return json.dumps(labelled, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _compact_replan_review(review: dict[str, Any] | None) -> dict[str, Any] | None:
        """Retain only the prior RWKV review fields needed for routing."""

        if not isinstance(review, dict):
            return None
        compact: dict[str, Any] = {}
        for key in ("decision", "trigger"):
            value = str(review.get(key) or "").strip()
            if value:
                compact[key] = value[:1000]
        gap = review.get("gap")
        if isinstance(gap, dict):
            compact["gap"] = {
                "task_record_id": str(gap.get("task_record_id") or "")[:80],
                "gap_type": str(gap.get("gap_type") or "")[:40],
                "field_ids": [str(value)[:120] for value in gap.get("field_ids") or []],
                "evidence_record_ids": [
                    str(value)[:160] for value in gap.get("evidence_record_ids") or []
                ],
                "conflict_record_ids": [
                    str(value)[:160] for value in gap.get("conflict_record_ids") or []
                ],
                "excluded_route_ids": [
                    str(value)[:160] for value in gap.get("excluded_route_ids") or []
                ],
            }
        if review.get("evidence_revision") is not None:
            compact["evidence_revision"] = int(review.get("evidence_revision") or 0)
        return compact or None

    @staticmethod
    def _compact_rebuild_observation(result: Any) -> str:
        """Project recovery state without replaying prior route arguments.

        The complete observation remains in the execution trace.  This
        projection is only the fresh model request boundary, so it carries
        evidence/progress metadata but never the query or URL that just
        stalled.
        """

        value = result
        if isinstance(result, str):
            try:
                value = json.loads(result)
            except json.JSONDecodeError:
                value = {"status": "recovery", "observation_type": "unstructured"}
        if not isinstance(value, dict):
            value = {"status": "recovery", "observation_type": type(value).__name__}

        compact: dict[str, Any] = {}
        for key in (
            "contract",
            "schema_version",
            "protocol_version",
            "status",
            "tool",
            "error_class",
            "repeat_count",
            "count",
            "candidate_count",
            "fetched_count",
            "evidence_ready",
            "evidence_state",
            "missing_task_record_ids",
            "task_record_id",
            "task_record_state",
        ):
            if key not in value:
                continue
            item = value[key]
            if isinstance(item, str):
                item = item[:240]
            elif isinstance(item, list):
                item = item[:12]
            compact[key] = item

        retrieval_delta = value.get("retrieval_delta")
        if isinstance(retrieval_delta, dict):
            compact["retrieval_delta"] = {
                key: retrieval_delta.get(key)
                for key in (
                    "step",
                    "branch_id",
                    "task_record_id",
                    "action",
                    "phase",
                    "query_count",
                    "exact_repeat",
                    "new_url_count",
                    "evidence_count",
                    "status",
                    "evidence_ready",
                )
                if key in retrieval_delta
            }

        frozen_rows = value.get("frozen_paths") or []
        if isinstance(value.get("frozen_path"), dict):
            frozen_rows = [*list(frozen_rows or []), value["frozen_path"]]
        if isinstance(frozen_rows, dict):
            frozen_rows = [frozen_rows]
        if isinstance(frozen_rows, list):
            projected_frozen = [row for row in frozen_rows if isinstance(row, dict)][-8:]
            compact["frozen_route_count"] = len(projected_frozen)
            frozen_reason_counts: dict[str, int] = {}
            for row in projected_frozen:
                reason = str(row.get("reason") or "unknown")[:120]
                frozen_reason_counts[reason] = frozen_reason_counts.get(reason, 0) + 1
            compact["frozen_reason_counts"] = frozen_reason_counts
            compact["route_history"] = [
                {
                    "route_id": str(row.get("route_id") or "")[:200],
                    "tool": str(row.get("action") or row.get("tool") or "")[:120],
                    "arguments": bounded_request_arguments(
                        row.get("arguments")
                        if isinstance(row.get("arguments"), dict)
                        else {"query": str(row.get("query") or "")[:500]}
                    ),
                    "task_record_id": str(row.get("task_record_id") or "")[:120],
                    "frozen": True,
                    "frozen_reason": str(row.get("reason") or "")[:120],
                }
                for row in projected_frozen
            ]

        ledger = value.get("retrieval_ledger")
        if isinstance(ledger, dict):
            compact["retrieval_ledger"] = {
                key: ledger.get(key)
                for key in (
                    "total_searches",
                    "unique_queries",
                    "retrieved_url_count",
                    "exact_repeat_count",
                    "blocked_duplicate_count",
                    "failed_request_count",
                    "evidence_revision",
                    "round_count",
                    "source_count",
                    "attempted_url_count",
                    "replan_count",
                )
                if key in ledger
            }
            compact["retrieval_ledger"]["previous_route_count"] = len(
                [row for row in ledger.get("queries") or [] if isinstance(row, dict)]
            )
            compact["retrieval_ledger"]["frozen_route_count"] = len(
                [row for row in ledger.get("frozen_paths") or [] if isinstance(row, dict)]
            )
            # Exact route text is projected once by
            # ``_replan_environment_projection().route_history``. Recovery
            # metadata carries only counters so the same failed query does not
            # become a second model-visible anchor.

        review = value.get("evidence_review") or value.get("pending_replan")
        if isinstance(review, dict):
            projected_review = Planner._compact_replan_review(review)
            if projected_review:
                compact["evidence_review"] = projected_review

        return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _compact_routing_observation(result: Any) -> str:
        """Keep bounded routing state while retaining recent evidence semantics.

        Full page bodies remain in the audit transcript and final evidence
        stage, but dropping every extracted fact made follow-up queries blind
        to what the previous page actually said. Keep a small set of
        chunk-bound, source-backed facts and quotes for query refinement; this
        is routing context, never final-answer evidence by itself.
        """

        value = result
        if isinstance(result, str):
            try:
                value = json.loads(result)
            except json.JSONDecodeError:
                return result[:1600]
        if not isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:1600]

        compact: dict[str, Any] = {}
        for key in (
            "contract",
            "schema_version",
            "protocol_version",
            "status",
            "tool",
            "tool_call_id",
            "retrieval_role",
            "provider",
            "query",
            "count",
            "candidate_count",
            "fetched_count",
            "evidence_ready",
            "evidence_state",
            "deterministic",
            "timezone",
            "iso",
            "date",
            "utc_offset",
            "observed_at_utc",
            "expression",
            "value",
            "result",
            "days",
            "absolute_days",
            "signed_days",
            "error_class",
            "message",
            "missing_task_record_ids",
            "next_focus",
            "retrieval_delta",
            "connector",
            "freshness_policy",
            "repeat_count",
            "alternative_urls",
            "recovery_instruction",
            "task_record_id",
            "task_record_state",
            "request",
            "frozen_path",
        ):
            if key in value:
                item = value[key]
                if isinstance(item, str):
                    item = item[:500]
                elif isinstance(item, list):
                    item = item[:12]
                compact[key] = item

        retrieval_query_plan = value.get("retrieval_query_plan")
        if isinstance(retrieval_query_plan, dict):
            compact["retrieval_query_plan"] = {
                "status": str(retrieval_query_plan.get("status") or "")[:80],
                "expanded": bool(retrieval_query_plan.get("expanded")),
                "executed_queries": [
                    {
                        "query_id": str(row.get("query_id") or "")[:40],
                        "task_record_id": str(row.get("task_record_id") or "")[:120],
                        "intent": str(row.get("intent") or "")[:120],
                        "query": str(row.get("query") or "")[:500],
                    }
                    for row in retrieval_query_plan.get("queries") or []
                    if isinstance(row, dict) and str(row.get("query") or "").strip()
                ][:6],
            }

        candidates = []
        for item in list(value.get("candidate_urls") or [])[:4]:
            if not isinstance(item, dict):
                continue
            candidates.append(
                {
                    key: str(item.get(key) or "")[:360]
                    for key in ("candidate_rank", "title", "url", "source", "candidate_score")
                    if key in item
                }
            )
        for item in list(value.get("results") or [])[:4]:
            if not isinstance(item, dict):
                continue
            metadata = {
                key: str(item.get(key) or "")[:360]
                for key in ("title", "url", "source", "scope", "path", "project", "connector")
                if key in item
            }
            for metadata_key in ("source_object", "object_alignment"):
                if isinstance(item.get(metadata_key), dict):
                    metadata[metadata_key] = dict(item[metadata_key])
            for metadata_key in ("object_alignments", "retrieval_bindings"):
                if isinstance(item.get(metadata_key), list):
                    metadata[metadata_key] = [
                        dict(value)
                        for value in item.get(metadata_key) or []
                        if isinstance(value, dict)
                    ][:4]
            if metadata:
                candidates.append(metadata)
        if candidates:
            compact["candidates"] = candidates[:8]

        evidence_context = []
        for item in list(value.get("results") or [])[:2]:
            if not isinstance(item, dict):
                continue
            chunk_candidates = [
                candidate
                for candidate in list(item.get("chunk_candidates") or [])
                if isinstance(candidate, dict)
                and candidate.get("supported") is True
            ][:2]
            source_by_id = {
                str(chunk.get("chunk_id") or ""): str(chunk.get("text") or "")
                for chunk in list(item.get("source_chunks") or [])[:8]
                if isinstance(chunk, dict) and str(chunk.get("chunk_id") or "")
            }
            locators = []
            for candidate in chunk_candidates[:1]:
                chunk_id = str(candidate.get("chunk_id") or "")
                # Generated facts are retained in the execution trace only.
                # The recurrent planner may see the exact located source span
                # and bounded original chunk, never an extractor paraphrase.
                quote = str(candidate.get("quote") or "").strip()[:500]
                source_text = source_by_id.get(chunk_id, "")[:600]
                if quote or source_text:
                    locators.append(
                        {
                            "chunk_id": chunk_id,
                            "quote": quote,
                            "source_text": source_text,
                        }
                    )
            if locators:
                evidence_context.append(
                    {
                        "title": str(item.get("title") or "")[:240],
                        "url": str(item.get("url") or "")[:360],
                        "evidence_status": str(item.get("evidence_status") or "")[:80],
                        "source_object": dict(item.get("source_object") or {}),
                        "object_alignment": dict(item.get("object_alignment") or {}),
                        "object_alignments": [
                            dict(value)
                            for value in item.get("object_alignments") or []
                            if isinstance(value, dict)
                        ][:4],
                        "locators": locators,
                    }
                )
        if evidence_context:
            compact["evidence_context"] = evidence_context

        page_evidence = value.get("page_evidence")
        if isinstance(page_evidence, list):
            compact["page_evidence"] = [
                {
                    key: item.get(key, "")
                    for key in ("url", "title", "status", "chunk_count", "candidate_count")
                    if key in item
                }
                for item in page_evidence[:6]
                if isinstance(item, dict)
            ]

        review = value.get("evidence_review")
        if isinstance(review, dict):
            compact["evidence_review"] = {
                key: review.get(key)
                for key in (
                    "contract",
                    "schema_version",
                    "status",
                    "evidence_state",
                    "usable_evidence_count",
                    "task_record_count",
                    "covered_task_record_ids",
                    "missing_task_record_ids",
                    "conflict_task_record_ids",
                )
                if key in review
            }

        ledger = value.get("retrieval_ledger")
        if isinstance(ledger, dict):
            compact["retrieval_ledger"] = {
                key: ledger.get(key)
                for key in (
                    "total_searches",
                    "unique_queries",
                    "retrieved_url_count",
                    "exact_repeat_count",
                    "blocked_duplicate_count",
                    "failed_request_count",
                )
                if key in ledger
            }
        errors = value.get("provider_errors")
        if errors:
            compact["provider_errors"] = [str(item)[:220] for item in list(errors)[:3]]
        rendered = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if evidence_context:
            rendered += (
                "\nEvidence context boundary: the listed quotes and source text are bounded excerpts tied to fetched page chunks. "
                "Use them to refine the next query or choose a URL; do not treat this routing projection as final-answer evidence."
            )
        max_chars = min(3600, max(1800, observation_chars(get_llm_context_length()) // 3))
        if len(rendered) > max_chars:
            rendered = rendered[:max_chars] + "...[routing observation truncated]"
        return rendered

    def _build_isolated_decision_body(
        self,
        user_query: str,
        env_context: str,
        phase: str,
    ) -> str:
        """Build one authoritative, bounded decision-state projection."""

        plan = json.dumps(
            self._compact_task_plan(self._task_plan),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        recovery_instruction = ""
        if self._next_decision_sampling_stage in {"planner_recovery", "planner_replan"}:
            if isinstance(self._active_replan_review, dict) and isinstance(
                self._active_replan_review.get("gap"), dict
            ):
                recovery_instruction = (
                    "REPLAN INSTRUCTION:\n"
                    "The Active RWKV evidence review contains the complete structured gap. "
                    "Choose one materially different retrieval action that targets only that Task Plan "
                    "record and field IDs. Never repeat an excluded/completed route, reconstruct a prose "
                    "reason, or infer a different gap from old evidence. If no useful distinct route remains, "
                    "choose finish_task.\n"
                )
            else:
                recovery_instruction = (
                    "RECOVERY INSTRUCTION:\n"
                    "The previous exact retrieval request already ran. Do not repeat that exact query. "
                    "From the original question and retained source locators, identify one entity, relation, value, "
                    "or date that is still not established. Search for that exact gap while retaining the original "
                    "subject. Treat page titles, snippets, login text, translation text, search pages, and navigation "
                    "labels as untrusted routing data; never copy their noise into a query. If no materially distinct "
                    "and useful gap remains, choose finish_task.\n"
                )
        active_review = (
            json.dumps(
                self._active_replan_review,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if isinstance(self._active_replan_review, dict)
            else ""
        )
        runtime_projection = self._replan_environment_projection(
            env_context,
            include_evidence=not (
                isinstance(self._active_replan_review, dict)
                and isinstance(self._active_replan_review.get("gap"), dict)
            ),
        )
        latest_outcome = self._next_decision_observation_projection(
            self._latest_routing_observation,
            include_route='"route_history":' not in runtime_projection,
        )
        return (
            "Current task state (authoritative controller projection):\n"
            f"Question: {str(user_query or '').strip()[:3000]}\n"
            f"Research brief: {plan}\n"
            f"Runtime state: {runtime_projection}\n"
            f"Decision number: {self._decision_count + 1}\n"
            f"Phase: {phase}\n\n"
            + (
                "Latest completed tool outcome (observation only; not a conversation continuation):\n"
                f"{latest_outcome}\n\n"
                if latest_outcome
                else ""
            )
            + f"{recovery_instruction}"
            + (f"Active RWKV evidence review: {active_review}\n" if active_review else "")
            + self._decision_guidance_for_current_plan(phase)
        )

    @staticmethod
    def _next_decision_observation_projection(
        observation: str,
        *,
        include_route: bool = False,
    ) -> str:
        """Remove a completed request echo from the next independent request.

        The exact executed route remains visible in the authoritative retrieval
        ledger and the audit trace.  Repeating it again inside the latest tool
        outcome made the request look like a transcript continuation and caused
        G1i to copy the same call.  This projection retains only outcome,
        candidates, grounded locators, counters and errors.
        """

        text = str(observation or "").strip()
        if not text:
            return ""
        try:
            value, _ = json.JSONDecoder().raw_decode(text)
        except (TypeError, json.JSONDecodeError):
            return text[:1600]
        if not isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:1600]

        projected = dict(value)
        route_fallback: dict[str, Any] | None = None
        if include_route:
            frozen = value.get("frozen_routes")
            if isinstance(frozen, list):
                route_fallback = next(
                    (dict(row) for row in reversed(frozen) if isinstance(row, dict)),
                    None,
                )
            if route_fallback is None and isinstance(value.get("previous_request"), dict):
                request = dict(value["previous_request"])
                arguments = request.get("arguments") or {}
                route_fallback = {
                    "tool": request.get("tool", ""),
                    "query": (
                        arguments.get("query") or arguments.get("url") or ""
                        if isinstance(arguments, dict)
                        else ""
                    ),
                }
        for key in (
            "query",
            "request",
            "previous_request",
            "frozen_path",
            "frozen_paths",
            "frozen_routes",
        ):
            projected.pop(key, None)
        if not include_route:
            projected.pop("route_history", None)
        if route_fallback:
            projected["completed_or_frozen_route"] = {
                key: route_fallback.get(key)
                for key in (
                    "tool",
                    "query",
                    "task_record_id",
                    "reason",
                )
                if route_fallback.get(key) not in (None, "")
            }
        ledger = projected.get("retrieval_ledger")
        if isinstance(ledger, dict):
            projected["retrieval_ledger"] = {
                key: item
                for key, item in ledger.items()
                if key not in {"queries", "recent_routes", "recent_frozen_routes"}
            }
        rendered = json.dumps(
            projected,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return rendered[:2400]

    def _isolated_decision_messages(
        self,
        user_query: str,
        env_context: str,
        phase: str,
        connector_operations: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Build one independent G1i request without replaying history.

        The request contains one authoritative User state and one fresh
        Assistant JSON boundary. Completed routes live in the state ledger;
        their prior Assistant calls are never placed before generation.
        """

        system_message = {
            "role": "system",
            "content": self._system_prompt(
                phase, connector_operations=connector_operations
            ),
        }
        decision_body = self._build_isolated_decision_body(
            user_query,
            env_context,
            phase,
        )
        return [
            system_message,
            {"role": "user", "content": decision_body},
        ]

    @staticmethod
    def _render_transcript(messages: list[dict[str, Any]]) -> str:
        return render_tool_transcript(messages)

    def _ensure_conversation(self, user_query: str, env_context: str, phase: str) -> None:
        if self._messages:
            return
        self._messages = [
            {
                "role": "system",
                "content": self._system_prompt(phase),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {str(user_query or '').strip()[:3000]}\n"
                    f"Research brief: {json.dumps(self._compact_task_plan(self._task_plan), ensure_ascii=False, separators=(',', ':'))}\n"
                    f"Runtime state: {str(env_context or '').strip()}\n"
                    f"Phase: {phase}\n\n"
                    f"{self._decision_guidance_for_current_plan(phase)}"
                ),
            },
        ]

    @staticmethod
    def _compact_observation(result: Any) -> str:
        """Keep routing observations small and remove raw page bodies."""

        value = result
        if isinstance(result, str):
            try:
                value = json.loads(result)
            except json.JSONDecodeError:
                return result[:2800]
        if not isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:2800]

        compact: dict[str, Any] = {}
        candidate_urls = [
            {
                key: item.get(key, "")
                for key in (
                    "candidate_rank",
                    "title",
                    "url",
                    "source",
                    "candidate_score",
                    "source_object",
                    "object_alignment",
                )
                if key in item
            }
            for item in list(value.get("candidate_urls") or [])[:8]
            if isinstance(item, dict)
        ]
        # Put the model's next-step choices first.  Conversation trimming may
        # retain only the first 900 characters of a large observation.
        if candidate_urls:
            compact["candidate_urls"] = candidate_urls
        compact.update(
            {
                key: value.get(key)
                for key in (
                    "contract",
                    "schema_version",
                "protocol_version",
                "status",
                "tool",
                "tool_call_id",
                "retrieval_role",
                "provider",
                "query",
                "count",
                "candidate_count",
                "fetched_count",
                "evidence_ready",
                "evidence_policy",
                "authority_missing",
                "required_domains",
                "real_network",
                "error_class",
                "message",
                "allowed_tools",
                "action",
                "args",
                "request",
                "frozen_path",
                "replan_count",
            "repeat_count",
            "alternative_urls",
            "recovery_instruction",
            "task_record_id",
            "task_record_state",
            "missing_task_record_ids",
                "next_focus",
                "retrieval_delta",
                "retrieval_ledger",
                "connector",
                "freshness_policy",
                "matches",
                "match_count",
                )
                if key in value
            }
        )
        page_evidence = value.get("page_evidence")
        if isinstance(page_evidence, list):
            compact["page_evidence"] = [
                {
                    key: item.get(key, "")
                    for key in ("url", "title", "status", "chunk_count", "candidate_count")
                    if key in item
                }
                for item in page_evidence[:8]
                if isinstance(item, dict)
            ]
        provider_errors = value.get("provider_errors")
        if provider_errors:
            compact["provider_errors"] = [str(item)[:240] for item in list(provider_errors)[:3]]
        rows: list[dict[str, Any]] = []
        for item in list(value.get("results") or [])[:8]:
            if not isinstance(item, dict):
                continue
            row = {
                "discovery_metadata": {
                    key: item.get(key, "")
                    for key in (
                        "title",
                        "url",
                        "api_url",
                        "snippet",
                        "source",
                        "scope",
                        "path",
                        "project",
                        "language",
                        "connector",
                        "freshness",
                        "source_object",
                        "object_alignment",
                    )
                    if key in item
                },
                "evidence_status": item.get("evidence_status", ""),
                "chunk_count": item.get("chunk_count", 0),
                "date_candidates": list(item.get("date_candidates") or [])[:32],
            }
            candidates = item.get("chunk_candidates")
            if isinstance(candidates, list):
                row["evidence_candidates"] = [
                    {
                        "chunk_id": candidate.get("chunk_id", ""),
                        "facts": [],
                        "quote": str(candidate.get("quote") or "")[:400],
                        "task_record_ids": list(
                            candidate.get("task_record_ids")
                            or []
                        )[:8],
                        "field_ids": list(candidate.get("field_ids") or [])[:16],
                        "source_subject": str(
                            candidate.get("subject_key")
                            or candidate.get("source_subject")
                            or ""
                        )[:240],
                        "source_record_key": str(
                            candidate.get("record_key")
                            or candidate.get("source_record_key")
                            or ""
                        )[:240],
                        "object_alignment": dict(
                            candidate.get("object_alignment") or {}
                        ),
                        "rwkv_subject_alignment": dict(
                            candidate.get("rwkv_subject_alignment") or {}
                        ),
                    }
                    for candidate in candidates
                    if isinstance(candidate, dict)
                    and candidate.get("supported") is True
                ][:32]
            if "snippet" in row["discovery_metadata"]:
                row["discovery_metadata"]["snippet"] = str(row["discovery_metadata"]["snippet"] or "")[:260]
            rows.append(row)
        if rows:
            compact["results"] = rows
        evidence_review = value.get("evidence_review")
        if isinstance(evidence_review, dict):
            compact["evidence_review"] = evidence_review
        refs = []
        for item in list(value.get("citation_refs") or [])[:8]:
            if isinstance(item, dict):
                refs.append({key: item.get(key, "") for key in ("ref_id", "title", "url") if key in item})
        if refs:
            compact["citation_refs"] = refs
        # A full tool result is already persisted in the task trace.  The
        # recurrent planner only needs a small decision observation; keeping
        # this projection bounded prevents three Forks from filling the 12K
        # context before the next tool choice.
        rendered = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if rows:
            rendered += (
                "\nObservation boundary: discovery_metadata (title/url/snippet/provider fields) is for routing only, "
                "not factual evidence. Only evidence_candidates/page-body fields may support a later answer, and the final "
                "synthesis receives an even stricter evidence-only projection."
            )
        if (
            rows
            and str(value.get("retrieval_role") or "") == "discovery"
            and not value.get("evidence_ready")
        ):
            rendered += (
                "\nRetrieval observation: this is a discovery candidate list, not page evidence. "
                "Select a returned URL and choose the appropriate evidence-role tool from the complete catalog before answering. "
                "If the candidates are unrelated, refine the query or choose another listed retrieval capability; do not invent a URL."
            )
        if str(value.get("status") or "").casefold() in {"error", "failed", "unavailable", "unauthorized"}:
            rendered += (
                "\nController execution state: this tool execution failed and produced no evidence. "
                "Do not treat the failure as a successful empty search. Decide the next model-owned step; the shared ledger will prevent the same failed request from being reissued."
            )
        if value.get("alternative_urls"):
            rendered += (
                "\nController recovery state: the selected URL failed or had no evidence. "
                "Choose a different URL from alternative_urls with the matching evidence-role tool before searching again."
            )
        page_evidence = value.get("page_evidence")
        if isinstance(page_evidence, dict) and page_evidence.get("status") in {"no_evidence", "error"}:
            rendered += (
                "\nController evidence state: the previously selected page did not support the user question. "
                "Use the observation to decide the next model-owned step. Choose another URL or refine the search; the same failed request is not reissued."
            )
        if value.get("retrieval_ledger"):
            rendered += (
                "\nShared retrieval ledger: the following is progress context from the global retrieval episode. "
                "It is not a controller decision. Use it to avoid needless exact repeats when a better query, URL, "
                "or unfinished task record is available; an exact request that already failed will not be reissued."
            )
        # Allocate the routing observation from the configured model context.
        # The 13.3B/12K profile can retain substantially more than the old
        # fixed 1,800-character slice, while the older 10K profile still gets
        # a bounded observation.  Candidate URLs remain at the beginning and
        # the latest ledger/instructions remain at the tail if trimming is
        # necessary.
        max_chars = observation_chars(get_llm_context_length())
        if len(rendered) > max_chars:
            head_chars = int(max_chars * 0.78)
            tail_chars = max_chars - head_chars
            rendered = (
                rendered[:head_chars]
                + "\n[observation body truncated for routing; use the visible candidate URLs and ledger]\n"
                + rendered[-tail_chars:]
            )
        return rendered

    def _trim_conversation(self) -> None:
        """Bound recurrent history while retaining the initial task and latest turns."""

        if len(self._messages) > 6:
            self._messages = [*self._messages[:2], *self._messages[-4:]]

        # The model catalog is intentionally small; provider details stay in
        # the backend. Remove the oldest observations until the next request
        # has room for the model's short JSON decision. Full observations
        # remain in the trace.
        context_length = max(1024, int(get_llm_context_length()))
        prompt_limit = planner_prompt_tokens(context_length)
        while len(self._messages) > 3 and get_token_count(self._render_transcript(self._messages)) > prompt_limit:
            if (
                len(self._messages) > 4
                and str(self._messages[2].get("role") or "") == "assistant"
                and str(self._messages[3].get("role") or "") in {"tool", "function", "observation"}
            ):
                del self._messages[2:4]
            else:
                self._messages.pop(2)

        # A single unusually large observation should not make the request
        # exceed the server limit. Keep its status, URLs and final tail, while
        # the complete payload remains available through the execution trace.
        if self._messages and get_token_count(self._render_transcript(self._messages)) > prompt_limit:
            observation_chars_limit = observation_chars(context_length)
            for index in range(2, len(self._messages)):
                content = str(self._messages[index].get("content") or "")
                if len(content) > observation_chars_limit:
                    head_chars = int(observation_chars_limit * 0.78)
                    tail_chars = observation_chars_limit - head_chars
                    self._messages[index]["content"] = (
                        content[:head_chars]
                        + "\n[observation truncated for routing]\n"
                        + content[-tail_chars:]
                    )
            while len(self._messages) > 3 and get_token_count(self._render_transcript(self._messages)) > prompt_limit:
                if (
                    len(self._messages) > 4
                    and str(self._messages[2].get("role") or "") == "assistant"
                    and str(self._messages[3].get("role") or "") in {"tool", "function", "observation"}
                ):
                    del self._messages[2:4]
                else:
                    self._messages.pop(2)

    def observe_tool_result(self, result: Any) -> None:
        """Append the executor result in the rwkv-skills user-observation form."""

        rendered = self._compact_observation(result)
        self._latest_routing_observation = self._compact_routing_observation(result)
        self._messages.append(
            {
                "role": "tool",
                "content": rendered,
                "_routing_observation": True,
            }
        )
        self._trim_conversation()

    def request_recovery_turn(
        self,
        observation: Any,
        *,
        user_query: str = "",
        env_context: str = "",
        phase: str = "DISCOVERY",
    ) -> None:
        """Freeze a stalled transcript and rebuild one exploratory G1i session."""

        self._latest_routing_observation = self._compact_routing_observation(
            observation
        )
        self._next_decision_sampling_stage = (
            "planner_replan"
            if self._active_replan_review
            else "planner_recovery"
        )
        self._next_decision_seed = None
        if str(user_query or "").strip():
            value = observation if isinstance(observation, dict) else {
                "status": "recovery",
                "message": str(observation or ""),
            }
            self.rebuild_session(user_query, env_context, value, phase)
            self._next_decision_sampling_stage = (
                "planner_replan"
                if self._active_replan_review
                else "planner_recovery"
            )

    @staticmethod
    def _completion_budget(prompt: str) -> int:
        """Use the remaining model context, capped at a 10K planner response."""

        context_length = max(1024, int(get_llm_context_length()))
        prompt_tokens = get_token_count(prompt)
        remaining = context_length - prompt_tokens - 1024
        return max(256, min(10000, remaining))

    @staticmethod
    def _route_family_tools() -> list[dict[str, Any]]:
        """Zero-argument object-family declarations shown to online G1i."""

        empty_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        return [
            {
                "name": "target_weather",
                "description": (
                    "Choose when the question asks for current weather or weather alerts "
                    "for a location."
                ),
                "arguments": empty_arguments,
            },
            {
                "name": "target_github",
                "description": (
                    "Choose when the question names a GitHub repository (owner/repo, a project "
                    "hosted on github.com) or asks for a repository's latest release, tag, "
                    "release notes, or code. The structured GitHub connector reads the official "
                    "release record directly. Not for npm/PyPI/crates package names — those "
                    "belong to target_package_registry."
                ),
                "arguments": empty_arguments,
            },
            {
                "name": "target_package_registry",
                "description": (
                    "Choose when the question asks about one exact crates.io, PyPI, or npm "
                    "package (its current version or release)."
                ),
                "arguments": empty_arguments,
            },
            {
                "name": "target_scholarly",
                "description": (
                    "Choose when the question asks about a scholarly paper, an arXiv ID, a DOI, "
                    "an author's publications, or a research topic. The structured paper "
                    "connector reads official paper metadata directly."
                ),
                "arguments": empty_arguments,
            },
            {
                "name": "target_security_advisories",
                "description": (
                    "Choose when the question asks for a vendor's latest official security "
                    "advisories, bulletins, or CVE catalog entries and the vendor is one of: "
                    "CISA KEV (known exploited vulnerabilities 已知被利用漏洞目录), "
                    "Mozilla/Firefox (MFSA), Microsoft/微软 (MSRC/Patch Tuesday 安全更新), "
                    "Kubernetes, OpenSSL, GitHub package advisories (GHSA). The structured "
                    "connector reads the vendor's official advisory feed directly. Not for "
                    "Cisco, Android, Chrome, GitLab, or GitHub Enterprise Server advisories "
                    "— those belong to target_general_web."
                ),
                "arguments": empty_arguments,
            },
            {
                "name": "target_general_web",
                "description": (
                    "Choose when none of the structured families above matches: an explicit URL, "
                    "documentation, a product or service status page, game or vendor announcements, "
                    "security advisories of other vendors, news."
                ),
                "arguments": empty_arguments,
            },
        ]

    def _select_route_family(
        self,
        user_query: str,
        phase: str,
        *,
        sampling_temperature: float,
        sampling_stage: str,
    ) -> dict[str, Any]:
        """Ask RWKV to declare the object family of its next retrieval target.

        The reply filters only the catalog presentation of the following tool
        decision. Any protocol failure falls back to the full catalog, so this
        step can degrade but never block or redirect a route.
        """

        tools_json = json.dumps(
            self._route_family_tools(), ensure_ascii=False, separators=(",", ":")
        )
        system_content = f"Tools: {tools_json}\nReturn only a JSON function call."
        observation = str(self._latest_routing_observation or "").strip()[:1200]
        body = (
            f"Question: {str(user_query or '').strip()[:1500]}\n"
            "Research brief: "
            + json.dumps(
                self._compact_task_plan(self._task_plan),
                ensure_ascii=False,
                separators=(",", ":"),
            )[:1200]
            + "\n"
            + (f"Latest routing observation: {observation}\n" if observation else "")
            + "\nDeclare the object family of your NEXT retrieval target by calling exactly one function. "
            "This declaration only filters the tool catalog for your next decision; you still choose the "
            "tool and query yourself afterwards. Match the question's target object:\n"
            "- a named GitHub repository or its latest release/tag/code -> target_github\n"
            "- a paper, arXiv ID, DOI, author, or research topic -> target_scholarly\n"
            "- one exact crates.io/PyPI/npm package version -> target_package_registry\n"
            "- current weather or weather alerts -> target_weather\n"
            "- latest security advisories/bulletins of CISA KEV, Mozilla/Firefox, Microsoft, "
            "Kubernetes, OpenSSL, or GitHub -> target_security_advisories\n"
            "- anything else (explicit URLs, documentation, status pages, game or vendor "
            "announcements, other vendors' advisories) -> target_general_web.\n"
            "web_search stays available in every family, so a structured family declaration "
            "never removes the general-web fallback."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": body},
        ]
        raw = ""
        error = ""
        for attempt in range(2):
            request_messages = messages
            if attempt:
                request_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Correction: return exactly one complete JSON function call choosing one of the "
                            f"listed functions. Protocol error: {error[:300]}"
                        ),
                    },
                ]
            request_prompt = render_tool_transcript(request_messages)
            try:
                with model_sampling_parameters(
                    sampling_temperature,
                    seed=None,
                    stage=sampling_stage,
                    policy_reason="model_declared_route_object_family",
                ):
                    if is_local_provider(self.llm.provider):
                        response = self.llm.text_completion(
                            request_prompt,
                            max_tokens=220,
                            stop=JSON_CALL_STOP_SUFFIXES,
                        )
                    else:
                        response = self.llm.chat_completion(
                            [{"role": "user", "content": request_prompt}],
                            max_tokens=220,
                        )
                raw = str(response.content or "")
                payload = _canonicalize_tool_payload(
                    normalize_json_object_envelope(raw).payload
                )
                function_value = payload.get("function")
                if isinstance(function_value, dict):
                    function_value = function_value.get("name")
                family = str(payload.get("name") or function_value or "").strip()
                if family not in _ROUTE_FAMILY_CONNECTOR_OPERATIONS:
                    raise ValueError(f"unknown object family: {family!r}")
                return {
                    "family": family,
                    "connector_operations": _ROUTE_FAMILY_CONNECTOR_OPERATIONS[family],
                    "source": "model",
                    "raw_model_output": visible_model_text(raw),
                    "error": "",
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if classify_error(exc) == "timeout":
                    break
        return {
            "family": "",
            "connector_operations": None,
            "source": "fallback_full_catalog",
            "raw_model_output": visible_model_text(raw),
            "error": error,
        }

    def plan_next_action(
        self,
        user_query: str,
        analysis_result: dict | None,
        env_context: str,
        phase: str,
    ) -> dict[str, Any]:
        del analysis_result  # model sees the full environment, not a static route
        self._ensure_conversation(user_query, env_context, phase)
        # Every decision is a fresh online-G1i request. The audit transcript is
        # deliberately not replayed into the model.
        planned_task_record_ids = [record_id(record) for record in task_records(self._task_plan)]
        raw = ""
        planner_error = ""
        payload: dict[str, Any] = {}
        name = ""
        arguments: dict[str, Any] = {}
        task_record_id = ""
        task_record_binding_method = "not_required"
        task_record_binding_raw = ""
        task_record_binding_error = ""
        task_record_binding_temperature: float | None = None
        protocol_input_format = "invalid"
        protocol_normalized = False
        sampling_stage = self._next_decision_sampling_stage
        sampling_temperature = (
            get_model_replan_temperature(
                self._replan_generation,
                str(self._latest_routing_observation or ""),
            )
            if sampling_stage == "planner_replan"
            else get_model_stage_temperature(sampling_stage)
        )
        sampling_seed = self._next_decision_seed
        # Two-step model routing: RWKV first declares the object family of its
        # next retrieval target, then chooses the tool/query from a catalog
        # whose connector operations are narrowed to that declaration. Both
        # steps are model-authored; a protocol failure in step one falls back
        # to the full catalog.
        route_family_selection = self._select_route_family(
            user_query,
            phase,
            sampling_temperature=sampling_temperature,
            sampling_stage=sampling_stage,
        )
        history = self._isolated_decision_messages(
            user_query,
            env_context,
            phase,
            connector_operations=route_family_selection.get("connector_operations"),
        )
        prompt = render_tool_transcript(history)
        successful_prompt = prompt
        # A malformed tool decision gets one same-temperature protocol repair.
        # The repair changes no tool/query choice and never authors an action;
        # it only asks RWKV to serialize its own decision as valid JSON.
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                correction = (
                    "Correction: the previous continuation was not one complete executable JSON object. "
                    f"Protocol error: {planner_error[:500]}. "
                    "Return exactly one complete JSON object now; no Markdown, explanation, thinking, or extra text. "
                    'Use {"name":"tool_name","arguments":{}} and choose the tool from the current catalog. '
                    "A retrieval call may include one existing task_record_id, but omit it if the route spans several records or is uncertain."
                )
                request_prompt = render_tool_transcript(
                    [*history, {"role": "user", "content": correction}]
                )
            try:
                with model_sampling_parameters(
                    sampling_temperature,
                    seed=sampling_seed,
                    stage=sampling_stage,
                    policy_reason=(
                        "model_selected_retrieval_strategy"
                        if sampling_stage == "planner"
                        else "replan_follow_up_strategy"
                    ),
                ):
                    if is_local_provider(self.llm.provider):
                        response = self.llm.text_completion(
                            request_prompt,
                            max_tokens=self._completion_budget(request_prompt),
                            stop=JSON_CALL_STOP_SUFFIXES,
                        )
                    else:
                        response = self.llm.chat_completion(
                            [{"role": "user", "content": request_prompt}],
                            max_tokens=self._completion_budget(
                                request_prompt
                            ),
                        )
                self._next_decision_sampling_stage = (
                    "planner_replan" if self._active_replan_review else "planner"
                )
                self._next_decision_seed = None
                raw = str(response.content or "")
                normalized_response = normalize_json_object_envelope(raw)
                protocol_input_format = normalized_response.input_format
                protocol_normalized = normalized_response.normalized
                payload = _canonicalize_tool_payload(normalized_response.payload)
                function_value = payload.get("function")
                if isinstance(function_value, dict):
                    function_value = function_value.get("name")
                name = str(
                    payload.get("name")
                    or payload.get("tool_name")
                    or payload.get("action")
                    or payload.get("tool")
                    or function_value
                    or ""
                ).strip()
                parsed_arguments = payload.get("arguments") or payload.get("args") or payload.get("parameters") or {}
                # RWKV sometimes emits the arguments object as a JSON-encoded
                # string (and may add an internal call id).  rwkv-skills accepts
                # this representation and normalizes it before execution.
                if isinstance(parsed_arguments, str):
                    parsed_arguments = json.loads(parsed_arguments) if parsed_arguments.strip() else {}
                if not isinstance(parsed_arguments, dict):
                    raise ValueError("tool call arguments must be a JSON object")
                if not name:
                    raise ValueError("tool call name is empty")
                ToolRegistry.validate_model_call(name, parsed_arguments)
                if self._retrieval_action_requires_task_record(name):
                    # A literal is "invented" only when it appears in neither
                    # the user question, the trusted runtime state, nor the
                    # exact observation transcript RWKV just read (retrieved
                    # evidence, route history, tool feedback). Lines echoing a
                    # prior guard rejection are excluded so a rejected literal
                    # cannot launder itself into the trusted set.
                    observed_input_text = "\n".join(
                        line
                        for line in request_prompt.splitlines()
                        if "untrusted hard literal" not in line
                    )
                    trusted_keys = hard_literal_keys(
                        [user_query, self._trusted_runtime_context, observed_input_text]
                    )
                    route_text = json.dumps(
                        parsed_arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    untrusted = untrusted_hard_literals(
                        route_text,
                        allowed_keys=trusted_keys,
                        provenance="planner_route",
                    )
                    if untrusted:
                        rendered = ", ".join(
                            f"{literal.kind}:{literal.surface_text}"
                            for literal in untrusted[:6]
                        )
                        raise ValueError(
                            "retrieval route contains untrusted hard literal(s): "
                            + rendered
                        )
                arguments = parsed_arguments
                task_record_id = str(
                    payload.get("task_record_id") or ""
                ).strip()
                successful_prompt = request_prompt
                planner_error = ""
                break
            except Exception as exc:
                planner_error = f"{type(exc).__name__}: {exc}"
                if attempt == 1 or classify_error(exc) == "timeout":
                    return {
                        "action": "",
                        "args": {},
                        "router": "model_rwkv_json_parse_error",
                        "raw_model_output": visible_model_text(raw),
                        "planner_error": planner_error,
                        "planner_attempts": attempt + 1,
                        "protocol_input_format": protocol_input_format,
                        "protocol_normalized": protocol_normalized,
                        "sampling_temperature": sampling_temperature,
                        "sampling_seed": sampling_seed,
                        "route_family": route_family_selection.get("family", ""),
                        "route_family_source": route_family_selection.get("source", ""),
                    }

        if planner_error:
            return {
                "action": "",
                "args": {},
                "router": "model_rwkv_json_parse_error",
                "raw_model_output": visible_model_text(raw),
                "planner_error": planner_error,
                "planner_attempts": 2,
                "protocol_input_format": protocol_input_format,
                "protocol_normalized": protocol_normalized,
                "sampling_temperature": sampling_temperature,
                "sampling_seed": sampling_seed,
                "route_family": route_family_selection.get("family", ""),
                "route_family_source": route_family_selection.get("source", ""),
            }

        if task_record_id and task_record_id in planned_task_record_ids:
            task_record_binding_method = "inline_rwkv"
        elif self._retrieval_action_requires_task_record(name):
            # A singleton has no semantic routing choice. For a multi-record
            # goal, an omitted or invalid id remains unbound so page evidence
            # sees the complete user goal and RWKV binds exact spans to the
            # records they actually support. A second enum-classifier was
            # observed to select P1 for every R32/R33 multi-record route.
            if len(planned_task_record_ids) == 1:
                task_record_id = planned_task_record_ids[0]
                task_record_binding_method = "sole_task_record"
            elif planned_task_record_ids:
                task_record_id = ""
                task_record_binding_method = "multi_record_unbound"
            else:
                task_record_id = ""
                task_record_binding_method = "task_plan_has_no_record"
        elif task_record_id:
            task_record_id = ""
            task_record_binding_method = "invalid_non_retrieval_record_id_ignored"

        call = {"name": name, "arguments": arguments}
        call_id = str(payload.get("call_id") or "").strip()
        if task_record_id:
            call["task_record_id"] = task_record_id
        if call_id:
            call["call_id"] = call_id
        self._messages.append({"role": "assistant", "content": call})
        self._decision_count += 1
        return {
            "action": name,
            "args": arguments,
            "task_record_id": task_record_id,
            "call_id": call_id,
            "router": "model_rwkv_json",
            "raw_model_output": visible_model_text(raw),
            "planner_attempts": attempt + 1,
            "protocol_input_format": protocol_input_format,
            "protocol_normalized": protocol_normalized,
            "task_record_binding_method": task_record_binding_method,
            "task_record_binding_raw_model_output": task_record_binding_raw,
            "task_record_binding_error": task_record_binding_error,
            "task_record_binding_temperature": (
                task_record_binding_temperature
            ),
            "sampling_stage": sampling_stage,
            "sampling_temperature": sampling_temperature,
            "sampling_seed": sampling_seed,
            "route_family": route_family_selection.get("family", ""),
            "route_family_source": route_family_selection.get("source", ""),
        }
