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
from typing import Any, Iterable

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
from utils.chunker import get_token_count
from utils.retrieval_ledger import normalize_query
from utils.context_budget import (
    observation_chars,
    planner_prompt_tokens,
)
from utils.error_policy import classify_error
from utils.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    assistant_json_prefix,
    render_tool_transcript,
)
from runtime.transcript import render_rwkv_transcript
from agent.tool_protocol import canonicalize_tool_call
from agent.task_plan_contract import (
    compact_task_plan,
    normalize_task_plan,
    point_question,
)


def _same_nonempty_route_target(left: str, right: str) -> bool:
    """Compare normalized route targets without interpreting their content."""

    return bool(left) and left in {right}


def _declared_task_point_id(value: Any, planned_point_ids: Iterable[str]) -> str:
    """Resolve only a task-point ID explicitly present in RWKV output.

    G1i sometimes emits ``"P2: missing date"`` or ``{"id": "P2", ...}``
    even when the protocol asks for the bare ID.  This is protocol
    canonicalization: it never guesses a point from evidence or from the
    missing-fact description.
    """

    planned = [
        str(point_id).strip()
        for point_id in planned_point_ids
        if str(point_id).strip()
    ]
    if isinstance(value, dict):
        text = str(
            value.get("id")
            or value.get("point_id")
            or value.get("task_point_id")
            or value.get("claim_id")
            or ""
        ).strip()
    else:
        text = str(value or "").strip()
    if not text:
        return ""
    exact = next(
        (point_id for point_id in planned if text.casefold() == point_id.casefold()),
        "",
    )
    if exact:
        return exact
    matches = [
        point_id
        for point_id in planned
        if re.search(
            rf"(?<![\w-]){re.escape(point_id)}(?![\w-])",
            text,
            flags=re.IGNORECASE,
        )
    ]
    return matches[0] if len(matches) == 1 else ""


def _repair_unescaped_json_quotes(text: str) -> str:
    """Repair quotes inside a JSON string without changing its content.

    RWKV occasionally emits a valid-looking plan containing a shell example
    such as ``python -c "..."``.  Those inner quotes are not escaped, so the
    standard decoder rejects the entire outer plan.  A quote inside a string
    followed by ordinary text is unambiguously content in this protocol; a
    quote followed by JSON punctuation remains the string terminator.
    """
    repaired: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and in_string:
            repaired.append(char)
            escaped = not escaped
            index += 1
            continue
        if char == '"' and not escaped:
            if not in_string:
                in_string = True
                repaired.append(char)
            else:
                lookahead = index + 1
                while lookahead < len(text) and text[lookahead].isspace():
                    lookahead += 1
                next_char = text[lookahead] if lookahead < len(text) else ""
                if next_char and next_char not in ",:]}" and next_char != "":
                    repaired.extend(("\\", '"'))
                else:
                    in_string = False
                    repaired.append(char)
            index += 1
            escaped = False
            continue
        repaired.append(char)
        escaped = False
        index += 1
    return "".join(repaired)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object after removing hidden thinking."""

    cleaned = visible_model_text(text).strip()
    decoder = json.JSONDecoder()
    candidates = [cleaned]
    # rwkv-skills may prefill the opening ``{`` in the prompt, so the
    # completion can legitimately begin with the first JSON field.  Recreate
    # only that protocol delimiter; do not recover prose or missing values.
    prefixed = re.match(
        r'^"(?:name|schema_version|atomic_points|task_mode|goal|decision|candidates)"\s*:',
        cleaned,
    )
    if prefixed:
        candidates.insert(0, "{" + cleaned)
    repaired = _repair_unescaped_json_quotes(cleaned)
    if repaired != cleaned:
        candidates.append(repaired)
    fallback: dict[str, Any] | None = None
    tool_keys = {"name", "tool_name", "action", "tool", "function", "tool_calls"}
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                if fallback is None:
                    fallback = value
                # Prefer the outer plan/tool object. This prevents a nested
                # atomic point from being mistaken for the complete plan.
                if any(
                    key in value
                    for key in (
                        "schema_version",
                        "atomic_points",
                        "decision",
                        "candidates",
                        *tool_keys,
                    )
                ):
                    return value
    if fallback is not None:
        return fallback
    raise ValueError("model tool decision is not a JSON object")


def _canonicalize_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the standalone transport adapter."""

    return canonicalize_tool_call(payload)


class Planner:
    """Own the model-facing plan, tool decisions, and global retrieval state."""

    def __init__(self):
        load_builtin_tools()
        self.llm = LLMClient()
        self._messages: list[dict[str, Any]] = []
        self._task_plan: dict[str, Any] | None = None
        # ``_messages`` is an audit trace, not recurrent model context. Every
        # decision is an independent online-G1i request built from the current
        # state projection and, after the first turn, only the latest compact
        # Function output. Replaying Assistant calls anchors RWKV to old paths.
        self._latest_routing_observation = ""
        self._latest_tool_call: dict[str, Any] | None = None
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed: int | None = None
        self._replan_generation = 0
        self._active_replan_review: dict[str, Any] | None = None

    def reset(self) -> None:
        self._messages = []
        self._task_plan = None
        self._latest_routing_observation = ""
        self._latest_tool_call = None
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed = None
        self._replan_generation = 0
        self._active_replan_review = None

    def execution_transcript(self) -> str:
        """Return the visible routing transcript for final summarization."""
        return self._render_transcript(self._messages)

    @staticmethod
    def _validate_task_plan(payload: dict[str, Any]) -> dict[str, Any]:
        return normalize_task_plan(payload, max_points=4)

    def create_task_plan(self, user_query: str, env_context: str = "") -> dict[str, Any]:
        """Ask RWKV for factual obligations, never a prewritten workflow."""

        prompt = (
            "System: Return only one JSON object.\n\n"
            "User: You are the RWKV factual-record planner. Preserve the user's exact goal. "
            "Create one record for each distinct answer record the user requests. Start with exactly one record and "
            "split it only when the question truly contains different subjects, a comparison, or different "
            "historical/current identities. A record is a row, not one field or one clause. A record has one stable subject and one historical/current/version/advisory "
            "identity. Put every requested field belonging to that same record in the same fields array. "
            "For example, a current release's identifier, title, date and theme are one point; a latest batch's "
            "item IDs, products and deadlines are one collection point. Do not split those fields into separate "
            "records. Pronouns such as that version, that release, that batch, those items, or 该批 refer back to the same record and must not create new records. "
            "subject names only the stable entity or object and must not repeat a field label. relation briefly "
            "names the shared record identity, such as current release, historical launch, latest batch, or "
            "procedure. A short lookup or how-to uses one point. "
            "Never create points for search, discovery, reading, extraction, verification, cross-checking, "
            "citation, formatting, or answer writing. Do not choose a tool, provider, query, URL, domain, source "
            "policy, status, completion gate, or answer. Fields are short labels in the user's language, never "
            "values. set_semantics is single for one record, collection for a required non-empty list, and "
            "possibly_empty when an empty set is a valid answer. "
            "Use at most four records. Return exactly one compact object in this schema and no explanation: "
            '{"schema_version":"task_plan.v3","goal":"...",'
            '"records":[{"id":"P1","question":"...","subject":"...",'
            '"relation":"...","fields":["..."],'
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
                    "task_plan.v3 object only. Merge repeated or overlapping records; use no more than four factual "
                    "records. Default to one record. For a short how-to, use one record. Do not invent values, APIs, flags, URLs, sources, "
                    "workflow steps, examples, or variants. Do not add explanation, Markdown, or a second object. "
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
                task_plan = self._validate_task_plan(_extract_json_object(raw))
                semantic_warnings = self._task_plan_semantic_warnings(
                    task_plan,
                    user_query=user_query,
                    env_context=env_context,
                )
                if semantic_warnings and attempt < 1:
                    raise ValueError(
                        "task plan introduced concrete anchors absent from the user/environment: "
                        + "; ".join(semantic_warnings[:6])
                    )
                if semantic_warnings:
                    task_plan["semantic_warnings"] = semantic_warnings
                task_plan["sampling_temperature"] = sampling_temperature
                task_plan["sampling_seed"] = None
                task_plan["plan_attempts"] = attempt + 1
                return task_plan
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if classify_error(exc) == "timeout":
                    break
        # A malformed plan must not erase an otherwise answerable user request.
        # This fallback preserves the exact goal as one factual point; it does
        # not choose retrieval actions, facts, sources, or the final answer.
        fallback = normalize_task_plan(
            {
                "goal": user_query,
                "atomic_points": [
                    {
                        "id": "P1",
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

        This checker never inserts, deletes, or replaces a plan value.  It only
        compares concrete years/example markers against the user's text and the
        supplied runtime context, then gives the same RWKV bounded correction
        turns.  A final structurally valid plan remains usable with auditable
        warnings so this safeguard cannot turn a model answer into an
        engineering error.
        """

        plan_text = json.dumps(task_plan, ensure_ascii=False)
        allowed_text = f"{str(user_query or '')}\n{str(env_context or '')}"
        allowed_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", allowed_text))
        plan_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", plan_text))
        warnings = [f"unexpected_year:{year}" for year in sorted(plan_years - allowed_years)]
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
        self._messages = []
        self._latest_routing_observation = ""
        self._latest_tool_call = None
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed = None
        self._replan_generation = 0
        self._active_replan_review = None
        point_ids = [
            str(point.get("id") or "").strip()
            for point in task_plan.get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        self._messages = [
            {
                "role": "system",
                "content": self._system_prompt(
                    phase,
                    task_point_example=(point_ids[0] if point_ids else "P1"),
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {str(user_query or '').strip()[:3000]}\n"
                    "Research brief (model-generated advisory context; you still decide every action and finish point):\n"
                    f"{json.dumps(self._compact_task_plan(task_plan), ensure_ascii=False, separators=(',', ':'))}\n"
                    f"Runtime state: {str(env_context or '').strip()}\n"
                    f"Phase: {phase}\n\n"
                    f"{self._decision_guidance(phase)}"
                ),
            },
        ]
        self._trim_conversation()

    @staticmethod
    def _validate_cross_review(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate only RWKV's two-action G1i review protocol.

        The controller intentionally does not inspect evidence, infer missing
        fields, reconcile statuses, or override RWKV's decision.  A malformed
        object receives one protocol retry in :meth:`cross_validate_research`.
        """

        if not isinstance(payload, dict):
            raise ValueError("cross-validation output must be a JSON object")
        if set(payload) != {"name", "arguments"}:
            raise ValueError(
                "cross-validation output must contain exactly name and arguments"
            )
        action = str(payload.get("name") or "").strip().casefold()
        if action not in {"continue_retrieval", "write_answer"}:
            raise ValueError(
                "cross-validation name must be continue_retrieval or write_answer"
            )
        arguments = payload.get("arguments")
        if arguments != {}:
            raise ValueError("cross-validation arguments must be an empty object")
        return {
            "schema_version": "rwkv-cross-validation.v3",
            # Keep the controller's established state vocabulary internal.
            # This is a direct mapping of RWKV's tool choice, not a semantic
            # rule or evidence-derived override.
            "decision": (
                "replan" if action == "continue_retrieval" else "finish"
            ),
            "selected_action": action,
            "review_owner": "rwkv",
        }

    @classmethod
    def _validate_cross_review_continuation(cls, raw: Any) -> dict[str, Any]:
        """Validate one complete function call without semantic recovery."""

        return cls._validate_cross_review(
            _extract_json_object(visible_model_text(str(raw or "")).strip())
        )

    @staticmethod
    def _cross_review_tools() -> list[dict[str, Any]]:
        """Return the complete two-action catalog shown to online G1i."""

        empty_arguments = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        return [
            {
                "name": "continue_retrieval",
                "description": (
                    "Choose only when another targeted search is needed before an "
                    "accurate, useful answer: the central requested fact is absent, "
                    "current/latest is supported only by stale material, or relevant "
                    "records conflict."
                ),
                "arguments": empty_arguments,
            },
            {
                "name": "write_answer",
                "description": (
                    "Choose when the retained source spans are sufficient for an "
                    "accurate, useful answer. The answer may state that a minor "
                    "requested detail is not established instead of inventing it."
                ),
                "arguments": empty_arguments,
            },
        ]

    def cross_validate_research(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        evidence_context: str,
    ) -> dict[str, Any]:
        """Ask RWKV for one binary review action in the online G1i grammar."""

        projected_target = json.dumps(
            self._cross_validation_plan_projection(task_plan),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        user_prompt = (
            "Before final synthesis, perform one minimal evidence check from the retained "
            "source spans. Choose "
            "continue_retrieval only if the central answer would be materially wrong "
            "or unusable without another targeted search. Otherwise choose "
            "write_answer. Do not require perfect completeness: the Writer can state "
            "that a minor detail is not established instead of inventing it. Body spans "
            "are the factual material; titles and URLs identify their source records. "
            "Do not write the answer, reason, analysis, missing field, query, URL, source "
            "ranking, or any extra key.\n\n"
            f"USER QUESTION:\n{str(user_query or '')}\n\n"
            f"POINTS TO REVIEW:\n{projected_target}\n\n"
            f"RETAINED SOURCE SPANS:\n{str(evidence_context or '')}\n\n"
            "FINAL ACTION CONTRACT:\n"
            "Return exactly one complete JSON object now: "
            '{"name":"continue_retrieval","arguments":{}} or '
            '{"name":"write_answer","arguments":{}}. '
            "Do not write the answer, reason, analysis, or any extra key."
        )
        tools = self._cross_review_tools()
        prompt = render_rwkv_transcript(
            [{"role": "user", "content": user_prompt}],
            tools=tools,
        )
        raw = ""
        last_error = ""
        last_prompt = prompt
        sampling_temperature = get_model_stage_temperature("cross_validation")
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                repair_user_prompt = user_prompt + (
                    "\n\nProtocol correction: the previous continuation was not one "
                    f"valid function call ({last_error[:300]}). Return exactly "
                    '{"name":"continue_retrieval","arguments":{}} or '
                    '{"name":"write_answer","arguments":{}} with no Markdown, '
                    "explanation, answer, query, or extra text."
                )
                request_prompt = render_rwkv_transcript(
                    [{"role": "user", "content": repair_user_prompt}],
                    tools=tools,
                )
            last_prompt = request_prompt
            try:
                with model_sampling_parameters(
                    sampling_temperature,
                    stage="cross_validation",
                    policy_reason="binary_evidence_review",
                ):
                    response = self.llm.text_completion(
                        request_prompt,
                        max_tokens=min(96, self._completion_budget(request_prompt)),
                        stop=JSON_CALL_STOP_SUFFIXES,
                    )
                raw = str(response.content or "")
                review = self._validate_cross_review_continuation(raw)
                review["raw_model_output"] = raw
                review["prompt"] = request_prompt
                review["sampling_temperature"] = sampling_temperature
                review["sampling_seed"] = None
                review["review_attempts"] = attempt + 1
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
                last_error = f"{type(exc).__name__}: {exc}"
        return {
            "schema_version": "rwkv-cross-validation.v3",
            "decision": "protocol_error",
            "error_class": "cross_validation_protocol",
            "message": last_error[:1000],
            "raw_model_output": raw,
            "prompt": last_prompt,
            "review_attempts": 2,
            "sampling_temperature": sampling_temperature,
            "sampling_seed": None,
            "review_owner": "rwkv",
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
        observation["status"] = "cross_validation_replan"
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
        self._latest_tool_call = None
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
        point_ids = [
            str(point.get("id") or "").strip()
            for point in (self._task_plan or {}).get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        # A replan is a new G1i session, not a malformed continuation that
        # starts with a Function output lacking its prior Assistant call.
        # The frozen session remains in the audit trace; the new User task is
        # reconstructed from authoritative state and compact recovery data.
        self._messages = [
            {
                "role": "system",
                "content": self._system_prompt(
                    phase,
                    task_point_example=(point_ids[0] if point_ids else "P1"),
                ),
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

    def mark_replan_progress(self, task_point_id: str = "") -> None:
        """Return to the normal planner profile after any material new evidence.

        The binary reviewer no longer invents a missing-point hypothesis.  A
        rebuilt Planner owns that diagnosis from the original goal, retained
        evidence and frozen paths.  Once a new retrieval materially advances
        state, keeping every later call at exploratory replan sampling only
        adds variance.
        """

        del task_point_id
        if not self._active_replan_review:
            return
        self._active_replan_review = None
        self._next_decision_sampling_stage = "planner"

    @staticmethod
    def _system_prompt(
        phase: str,
        task_mode: str = "",
        task_point_example: str = "P1",
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
        catalog = json.dumps(
            [
                {
                    "name": row.get("name", ""),
                    "description": str(row.get("description") or "").splitlines()[0],
                    "arguments": row.get("arguments") or {"type": "object"},
                }
                for row in catalog_rows
                if isinstance(row, dict)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"Tools: {catalog}\nReturn only a JSON function call."

    @staticmethod
    def _decision_guidance(
        phase: str,
        task_point_example: str = "P1",
    ) -> str:
        """Return task-local guidance outside the fixed G1i System envelope."""

        return (
            "Retrieval decision agent.\n"
            "Choose exactly one next action for the current question and research brief.\n"
            "Return one JSON object only: "
            + json.dumps(
                {
                    "name": "tool_name",
                    "task_point_id": str(task_point_example or "P1"),
                    "arguments": {},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + ".\n"
            "task_point_id is optional route metadata. When a retrieval targets one factual record, copy that record's "
            "existing id so the extractor can focus on it. This does not bind a fetched page as evidence. Never invent an id; omitting it never "
            "invalidates an otherwise executable tool call.\n"
            "Use only the tool names and argument contracts in the catalog. Never emit an answer in tool arguments.\n"
            "Tool choice contract:\n"
            "- connector_lookup: structured weather/current alerts, a specific GitHub repository/code/release, and scholarly arXiv/DOI paper records. Its scope is one enum value, never a comma-separated list.\n"
            "- web_search: general Web research, exact URLs, documentation, and product or service status pages; also use it as connector fallback.\n"
            "- calculator/date_diff/current_time: deterministic calculation or clock observations after required operands are known.\n"
            "- finish_task: stop retrieval and ask RWKV to synthesize from all retained sources.\n"
            "Retrieval tools internally handle provider selection, URL fetching, cleaning, chunking and evidence extraction.\n"
            "Tool Output is context for your next decision. You decide whether to retrieve again, use another tool, or finish.\n"
            "Choose calculator for arithmetic, current_time for the clock, date_diff for exact date distance, connector_lookup for structured sources, and web_search for general web research when useful. "
            "When one connector exactly matches the request (weather, weather alerts, a named GitHub repository/code/release, or an arXiv/DOI/scholarly paper), use that structured connector before general-web fallback. A company's product or service status page is general web, not a repository lookup.\n"
            "The shared ledger reports exact bound and candidate evidence-record counts for every factual point, including zero. These are RWKV extraction observations, not completion judgements. "
            "Before repeating an already-covered route, compare all factual points and inspect candidate record identities; consider a materially different retrieval when exact evidence is still absent, while keeping the choice and query model-authored.\n"
            "For a current or latest request, use the current UTC date visible in Runtime state to disambiguate the search; do not assume that an older record is current.\n"
            "If the latest observation reports no new evidence or a frozen path, choose a materially different query, "
            "source, tool, or finish from retained sources. The controller never supplies a replacement query.\n"
            "Avoid needless exact repeats, but make every next-step and finish decision yourself.\n"
            "For date_diff, use only exact YYYY-MM-DD values already present in the question or visible evidence.\n"
            f"Current phase: {phase}"
        )

    @staticmethod
    def _compact_task_plan(task_plan: dict[str, Any] | None) -> dict[str, Any]:
        """Project only RWKV's goal and factual focuses for later decisions.

        Source policies, mutable Claim statuses and acceptance gates are not
        part of the decision brief.  They previously contradicted retrieved
        material when an otherwise useful source was not explicitly bound to a
        task-point ID, which repeatedly anchored RWKV on an already-run query.
        The complete plan remains in the trace for provenance.
        """

        return compact_task_plan(task_plan)

    @staticmethod
    def _cross_validation_plan_projection(
        task_plan: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Project only answer obligations, without replaying task-plan protocol.

        Cross-validation asks RWKV for a different JSON schema.  Replaying a
        complete ``task_plan.v2`` object in the same prompt made the recurrent
        model continue the embedded planner schema instead of emitting the
        requested review.  This projection keeps every user-facing evidence
        obligation but removes planner-only protocol and workflow fields.
        """

        compact = Planner._compact_task_plan(task_plan)
        return {
            "goal": str(compact.get("goal") or "")[:500],
            "points_to_check": [
                {
                    "id": point.get("id"),
                    "question": point_question(point)[:500],
                    "subject": str(point.get("subject") or "")[:240],
                    "relation": str(point.get("relation") or "")[:160],
                    "fields": list(point.get("fields") or [])[:16],
                    "time_scope": point.get("time_scope") or "unspecified",
                    "set_semantics": point.get("set_semantics") or "single",
                    "premise_requires_verification": bool(
                        point.get("premise_requires_verification")
                    ),
                }
                for point in compact.get("atomic_points") or []
                if isinstance(point, dict)
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
                    "action": "web_search",
                    "query": query[:500],
                    "task_point_id": str(row.get("task_point_id") or "")[:120],
                    "status": str(row.get("status") or "")[:80],
                }
            )
        return output[-16:]

    @staticmethod
    def _replan_environment_projection(env_context: str) -> str:
        """Build valid bounded state without replaying a stalled route.

        Replan is a fresh RWKV request.  Replaying exact query strings in both
        the ledger and frozen paths copy-anchors the recurrent model to the
        route that triggered recovery.  Keep observable progress counts and
        original source spans, but leave the next query entirely to RWKV.

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
                        "affected_claim_ids",
                        "unresolved_chunk_count",
                        "transport_error_count",
                    )
                    if key in infrastructure
                }
            labelled["retrieval_ledger"] = {
                key: ledger.get(key)
                for key in (
                    "schema_version",
                    "evidence_revision",
                    "round_count",
                    "source_count",
                    "attempted_url_count",
                    "replan_count",
                )
                if key in ledger
            }
            point_progress = []
            for row in ledger.get("factual_point_progress") or []:
                if not isinstance(row, dict) or not str(row.get("id") or "").strip():
                    continue
                point_progress.append(
                    {
                        "id": str(row.get("id") or "")[:120],
                        "bound_evidence_record_count": int(
                            row.get("bound_evidence_record_count") or 0
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
            labelled["retrieval_ledger"]["factual_point_progress"] = point_progress[:8]
            labelled["retrieval_ledger"]["unassigned_source_count"] = int(
                ledger.get("unassigned_source_count") or 0
            )
            labelled["retrieval_ledger"].update(
                {
                    "previous_route_count": len(queries),
                    "frozen_route_count": len(frozen_paths),
                    "route_status_counts": route_status_counts,
                    "frozen_reason_counts": frozen_reason_counts,
                    "route_text_withheld_from_replan": bool(queries or frozen_paths),
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
                if isinstance(source.get("claim_ids"), list):
                    projected_source["claim_ids"] = [
                        str(value)[:80] for value in source["claim_ids"][:8]
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
                projected_source["source_span_count"] = len(spans)
                projected_source["spans"] = [
                    {
                        "chunk_id": str(span.get("chunk_id") or "")[:120],
                        "start_char": span.get("start_char"),
                        "text": str(span.get("text") or "")[:520],
                        "truncated": bool(
                            span.get("truncated")
                            or len(str(span.get("text") or "")) > 520
                        ),
                    }
                    for span in spans[:2]
                ]
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
                        "field_keys": [
                            str(value)[:120]
                            for value in record.get("field_keys") or []
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
                "schema_version": "planner-records.v1",
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

        if not labelled:
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
        frozen_rows = review.get("frozen_paths") or []
        frozen_path = review.get("frozen_path")
        if isinstance(frozen_path, dict):
            frozen_rows = [*list(frozen_rows or []), frozen_path]
        if isinstance(frozen_rows, dict):
            frozen_rows = [frozen_rows]
        if isinstance(frozen_rows, list):
            projected_frozen = [value for value in frozen_rows[-8:] if isinstance(value, dict)]
            if projected_frozen:
                compact["frozen_route_count"] = len(projected_frozen)
                reasons: dict[str, int] = {}
                for value in projected_frozen:
                    reason = str(value.get("reason") or "unknown")[:120]
                    reasons[reason] = reasons.get(reason, 0) + 1
                compact["frozen_reason_counts"] = reasons
                compact["route_text_withheld_from_replan"] = True
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

        compact: dict[str, Any] = {
            "route_text_withheld_from_rebuild": True,
        }
        for key in (
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
            "missing_point_ids",
            "task_point_id",
            "task_point_state",
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
                    "task_point_id",
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

        request = value.get("request")
        if isinstance(request, dict):
            arguments = request.get("arguments") or request.get("args") or {}
            compact["previous_request"] = {
                "tool": str(
                    request.get("name")
                    or request.get("action")
                    or request.get("tool")
                    or ""
                )[:120],
                "argument_names": sorted(str(key)[:120] for key in arguments)[:12]
                if isinstance(arguments, dict)
                else [],
            }

        frozen_rows = value.get("frozen_paths") or []
        if isinstance(value.get("frozen_path"), dict):
            frozen_rows = [*list(frozen_rows or []), value["frozen_path"]]
        if isinstance(frozen_rows, dict):
            frozen_rows = [frozen_rows]
        if isinstance(frozen_rows, list):
            compact["frozen_route_count"] = len(
                [row for row in frozen_rows if isinstance(row, dict)]
            )

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
            "missing_point_ids",
            "next_focus",
            "retrieval_delta",
            "connector",
            "freshness_policy",
            "repeat_count",
            "alternative_urls",
            "recovery_instruction",
            "task_point_id",
            "task_point_state",
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
                    "schema_version",
                    "status",
                    "evidence_state",
                    "usable_evidence_count",
                    "task_point_count",
                    "covered_point_ids",
                    "missing_point_ids",
                    "conflict_point_ids",
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
        planned_point_ids = [
            str(point.get("id") or "").strip()
            for point in (self._task_plan or {}).get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        task_point_example = (
            planned_point_ids[0]
            if planned_point_ids
            else "P1"
        )
        recovery_instruction = ""
        if self._next_decision_sampling_stage in {"planner_recovery", "planner_replan"}:
            recovery_instruction = (
                "RECOVERY INSTRUCTION:\n"
                "The previous exact or equivalent retrieval path already ran. Do not repeat its query. "
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
        runtime_projection = self._replan_environment_projection(env_context)
        return (
            "Current task state (authoritative controller projection):\n"
            f"Question: {str(user_query or '').strip()[:3000]}\n"
            f"Research brief: {plan}\n"
            f"Runtime state: {runtime_projection}\n"
            f"Decision number: {self._decision_count + 1}\n"
            f"Phase: {phase}\n\n"
            f"{recovery_instruction}"
            + (f"Active RWKV evidence review: {active_review}\n" if active_review else "")
            + self._decision_guidance(phase, task_point_example)
        )

    def _isolated_decision_messages(
        self,
        user_query: str,
        env_context: str,
        phase: str,
    ) -> list[dict[str, Any]]:
        """Build one complete rolling G1i request without replaying history.

        The request always contains one authoritative User state.  A follow-up
        additionally contains only the immediately preceding Assistant call
        and its latest compact Function output.  This preserves the online
        G1i turn grammar without accumulating older action/result pairs.
        """

        point_ids = [
            str(point.get("id") or "").strip()
            for point in (self._task_plan or {}).get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        system_message = {
            "role": "system",
            "content": self._system_prompt(
                phase,
                task_point_example=(point_ids[0] if point_ids else "P1"),
            ),
        }
        decision_body = self._build_isolated_decision_body(
            user_query,
            env_context,
            phase,
        )
        messages: list[dict[str, Any]] = [
            system_message,
            {"role": "user", "content": decision_body},
        ]
        if self._latest_routing_observation:
            if self._latest_tool_call:
                messages.append(
                    {"role": "assistant", "content": dict(self._latest_tool_call)}
                )
                messages.append(
                    {"role": "tool", "content": self._latest_routing_observation}
                )
            else:
                # Recovery can be reconstructed without an executable prior
                # call.  In that case the observation belongs to the current
                # User state rather than a syntactically orphaned Function
                # output turn.
                messages[1]["content"] = (
                    f"{decision_body}\n\nRecovery observation:\n"
                    f"{self._latest_routing_observation}"
                )
        return messages

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
                    f"{self._decision_guidance(phase)}"
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
                for key in ("candidate_rank", "title", "url", "source", "candidate_score")
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
            "task_point_id",
            "task_point_state",
            "missing_point_ids",
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
                "or unfinished task point is available; an exact request that already failed will not be reissued."
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
        self._next_decision_sampling_stage = "planner_recovery"
        self._next_decision_seed = None
        if str(user_query or "").strip():
            value = observation if isinstance(observation, dict) else {
                "status": "recovery",
                "message": str(observation or ""),
            }
            self.rebuild_session(user_query, env_context, value, phase)
            self._next_decision_sampling_stage = "planner_recovery"

    @staticmethod
    def _completion_budget(prompt: str) -> int:
        """Use the remaining model context, capped at a 10K planner response."""

        context_length = max(1024, int(get_llm_context_length()))
        prompt_tokens = get_token_count(prompt)
        remaining = context_length - prompt_tokens - 1024
        return max(256, min(10000, remaining))

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
        planned_point_ids = [
            str(point.get("id") or "").strip()
            for point in (self._task_plan or {}).get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        history = self._isolated_decision_messages(
            user_query,
            env_context,
            phase,
        )
        prompt = render_tool_transcript(history)
        raw = ""
        planner_error = ""
        successful_prompt = prompt
        payload: dict[str, Any] = {}
        name = ""
        arguments: dict[str, Any] = {}
        task_point_id = ""
        task_point_binding_method = "not_required"
        task_point_binding_raw = ""
        task_point_binding_error = ""
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
                    "task_point_id is optional trace metadata."
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
                payload = _canonicalize_tool_payload(_extract_json_object(raw))
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
                arguments = parsed_arguments
                task_point_id = str(
                    payload.get("task_point_id") or payload.get("point_id") or ""
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
                        "sampling_temperature": sampling_temperature,
                        "sampling_seed": sampling_seed,
                    }

        if planner_error:
            return {
                "action": "",
                "args": {},
                "router": "model_rwkv_json_parse_error",
                "raw_model_output": visible_model_text(raw),
                "planner_error": planner_error,
                "planner_attempts": 2,
                "sampling_temperature": sampling_temperature,
                "sampling_seed": sampling_seed,
            }

        if task_point_id and task_point_id in planned_point_ids:
            task_point_binding_method = "inline_rwkv"
        elif task_point_id:
            # Invalid optional metadata is discarded without changing or
            # regenerating RWKV's otherwise executable tool call.
            task_point_id = ""
            task_point_binding_method = "invalid_optional_id_ignored"

        call = {"name": name, "arguments": arguments}
        call_id = str(payload.get("call_id") or "").strip()
        if task_point_id:
            call["task_point_id"] = task_point_id
        if call_id:
            call["call_id"] = call_id
        self._latest_tool_call = dict(call)
        self._messages.append({"role": "assistant", "content": call})
        self._decision_count += 1
        return {
            "action": name,
            "args": arguments,
            "task_point_id": task_point_id,
            "call_id": call_id,
            "router": "model_rwkv_json",
            "raw_model_output": visible_model_text(raw),
            "planner_attempts": attempt + 1,
            "task_point_binding_method": task_point_binding_method,
            "task_point_binding_raw_model_output": task_point_binding_raw,
            "task_point_binding_error": task_point_binding_error,
            "task_point_binding_temperature": (
                None
            ),
            "sampling_stage": sampling_stage,
            "sampling_temperature": sampling_temperature,
            "sampling_seed": sampling_seed,
        }
