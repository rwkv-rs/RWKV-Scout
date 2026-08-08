"""RWKV-native agent-loop planner.

The local RWKV service is trained around the same transcript used by the
rwkv-skills function-calling runner.  Keep the wire format explicit:

    ### User
    <instructions, task, or observation>
    ### Assistant
    **Tool Call:**
    ```json
    {"name":"tool","arguments":{...}}

The planner owns the conversation history.  The application only executes
the name and arguments returned by the model; it does not manufacture a
search provider or a rewritten query when parsing fails.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

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
from agent.tool_protocol import canonicalize_tool_call


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
        r'^"(?:name|schema_version|atomic_points|task_mode|goal|decision)"\s*:',
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
                    for key in ("schema_version", "atomic_points", "decision", *tool_keys)
                ):
                    return value
    if fallback is not None:
        return fallback
    raise ValueError("model tool decision is not a JSON object")


def _canonicalize_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the standalone transport adapter."""

    return canonicalize_tool_call(payload)


def _merge_unique(existing: list[Any], additions: list[Any]) -> list[Any]:
    """Append non-empty values while preserving the model's first-seen order."""
    merged = list(existing)
    seen = {str(item).casefold() for item in merged}
    for item in additions:
        key = str(item).casefold()
        if key and key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


class Planner:
    """Own the model-facing plan, tool decisions, and global retrieval state."""

    def __init__(self):
        load_builtin_tools()
        self.llm = LLMClient()
        self._messages: list[dict[str, Any]] = []
        self._task_plan: dict[str, Any] | None = None
        # The transcript is retained for auditability, but it is deliberately
        # not the model's recurrent decision context.  Each decision receives
        # only the latest routing projection below.
        self._latest_routing_observation = ""
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed: int | None = None
        self._replan_generation = 0

    def reset(self) -> None:
        self._messages = []
        self._task_plan = None
        self._latest_routing_observation = ""
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed = None
        self._replan_generation = 0

    def execution_transcript(self) -> str:
        """Return the visible routing transcript for final summarization."""
        return self._render_transcript(self._messages)

    @staticmethod
    def _validate_task_plan(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("task plan must be a JSON object")
        goal = str(payload.get("goal") or "").strip()
        points = payload.get("atomic_points")
        if not goal or not isinstance(points, list) or not points:
            raise ValueError("task plan requires goal and atomic_points")
        if len(points) > 32:
            raise ValueError("task plan contains too many atomic_points")
        normalized_points = []
        points_by_signature: dict[str, dict[str, Any]] = {}
        used_ids: set[str] = set()
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("each atomic point must be an object")
            point_id = str(point.get("id") or "").strip()
            task = str(point.get("task") or "").strip()
            objective = str(point.get("objective") or "").strip()
            evidence_needed = point.get("evidence_needed")
            acceptance_criteria = point.get("acceptance_criteria")
            if (
                not point_id
                or not task
                or not objective
                or not isinstance(evidence_needed, list)
                or not isinstance(acceptance_criteria, list)
                or not acceptance_criteria
            ):
                raise ValueError(
                    "each atomic point requires id, task, objective, evidence_needed and acceptance_criteria"
                )
            signature = "\n".join(
                re.sub(r"\s+", " ", value.casefold()).strip()
                for value in (task, objective)
            )
            evidence = [str(item).strip() for item in evidence_needed if str(item).strip()]
            criteria = [str(item).strip() for item in acceptance_criteria if str(item).strip()]
            existing = points_by_signature.get(signature)
            if existing is not None:
                # RWKV occasionally repeats a point while expanding a plan.
                # Merge the evidence and acceptance requirements instead of
                # discarding an otherwise usable plan.  The raw model output
                # remains in the execution record, so this normalization is
                # auditable and does not hide the model's mistake.
                existing["evidence_needed"] = _merge_unique(existing["evidence_needed"], evidence)
                existing["acceptance_criteria"] = _merge_unique(
                    existing["acceptance_criteria"], criteria
                )
                existing["source_ids"] = _merge_unique(existing.get("source_ids", []), [point_id])
                continue
            if point_id in used_ids:
                raise ValueError("task plan point ids must be unique")
            normalized_point = {
                "id": point_id,
                "task": task,
                "objective": objective,
                "evidence_needed": evidence,
                "acceptance_criteria": criteria,
                "output_format": str(point.get("output_format") or "prose").strip(),
                "status": str(point.get("status") or "pending"),
                "source_ids": [point_id],
                "required_domains": list(dict.fromkeys(
                    str(value).strip().casefold().removeprefix("www.").rstrip(".")
                    for value in (
                        [point.get("required_domains")]
                        if isinstance(point.get("required_domains"), str)
                        else point.get("required_domains") or []
                    )
                    if str(value).strip()
                ))[:4],
            }
            normalized_points.append(normalized_point)
            points_by_signature[signature] = normalized_point
            used_ids.add(point_id)
        if not normalized_points:
            raise ValueError("task plan contains no distinct atomic points")
        task_mode = str(payload.get("task_mode") or "lookup").strip().casefold()
        if task_mode not in {
            "lookup",
            "latest_list",
            "compare",
            "deep_research",
            "computation",
            "current_time",
            "date_arithmetic",
        }:
            task_mode = "lookup"
        source_policy = str(payload.get("source_policy") or "open_web").strip().casefold()
        if source_policy not in {"open_web", "primary_preferred", "official_required"}:
            source_policy = "open_web"
        required_domains = payload.get("required_domains") or []
        if isinstance(required_domains, str):
            required_domains = [required_domains]
        required_domains = list(dict.fromkeys(
            str(value).strip().casefold().removeprefix("www.").rstrip(".")
            for value in required_domains
            if str(value).strip()
        ))[:12]
        requested_fields = payload.get("requested_fields") or []
        if isinstance(requested_fields, str):
            requested_fields = [requested_fields]
        requested_fields = [str(value).strip() for value in requested_fields if str(value).strip()][:32]
        try:
            max_items = int(payload.get("max_items") or 0)
        except (TypeError, ValueError):
            max_items = 0
        max_items = max(0, min(max_items, 50))
        return {
            "schema_version": "task_plan.v1",
            "goal": goal,
            "atomic_points": normalized_points,
            "task_mode": task_mode,
            "source_policy": source_policy,
            "required_domains": required_domains,
            "requested_fields": requested_fields,
            "max_items": max_items,
            "completion_rule": str(payload.get("completion_rule") or "All atomic points are answered with supported evidence."),
        }

    def create_task_plan(self, user_query: str, env_context: str = "") -> dict[str, Any]:
        """Ask RWKV to make a generic atomic plan before any retrieval call."""
        prompt = (
            "### User\nYou are the task-planning RWKV. Decompose the user's goal into the smallest "
            "useful independently verifiable atomic points. Do not choose a provider, tool, query, "
            "or URL. Do not assume the local workspace is relevant unless the user explicitly asks about it. "
            "Describe evidence as the fact that must be verified, not as a preselected source. Do not add facts, "
            "API names, flags, values, examples, URLs, or implementation details that the user did not mention. "
            "For each point, state the task, evidence_needed, and concise acceptance_criteria. Preserve only the "
            "scope the user explicitly requested. Do not silently turn a request for the latest few items into a "
            "request for a complete archive, full document, or proof that no item is missing. "
            "For a short how-to or lookup question, use exactly one atomic point unless the user explicitly asks "
            "for separate comparisons or multiple deliverables. Do not create a separate example/code point unless "
            "the user requests an example or code. Classify the task as lookup, latest_list, compare, deep_research, computation, current_time, or date_arithmetic. "
            "Use computation for pure arithmetic, current_time for the current clock, and date_arithmetic only when exact dates are supplied or will be retrieved as facts before calculation. "
            "These deterministic modes do not require web sources. "
            "For lookup use 1-3 points only when the question genuinely contains multiple requested facts: "
            "authoritative source, requested facts, and one cross-check only when it materially reduces uncertainty. "
            "For latest_list set max_items to a small number (normally 3-5); only when the user explicitly asks for all "
            "or a complete list set max_items to 50. "
            "If the user names an organisation, regulator, standards body, or asks for an official source, set "
            "source_policy to official_required and infer its likely required_domains; otherwise use primary_preferred "
            "or open_web. Never treat a third-party summary as an official source. "
            "If the task requests a list or table, set output_format to list or table and preserve the requested "
            "fields and order, but do not require every row unless the user explicitly says complete/all. "
            "Keep distinct roles distinct, and give each requested paper or project its own exact URL. "
            "The fields evidence_needed, requested_fields, and acceptance_criteria must describe the user's words "
            "or generic verification needs; never propose a concrete API, command, variable, or example as a guess. "
            "For pure arithmetic, current clock, or exact date-distance questions, plan a deterministic operation rather than web research; "
            "the evidence_needed value may be the deterministic result and must not require a webpage. "
            "Inside JSON string values, escape every inner double quote as \\\"; prefer single quotes in shell examples "
            "and keep the plan compact. For a normal request use 1-8 atomic points; for a complex request group "
            "related facts and use no more than 16. Never create one point per URL, source, example, or repeated "
            "wording. Never emit more than 32 points. Never put raw unescaped double quotes inside a JSON string. "
            "Return exactly one JSON object and no explanation. "
            "Use this fixed format: "
            '{"schema_version":"task_plan.v1","goal":"...",'
            '"task_mode":"lookup|latest_list|compare|deep_research|computation|current_time|date_arithmetic",'
            '"source_policy":"open_web|primary_preferred|official_required",'
            '"required_domains":["..."],"requested_fields":["..."],"max_items":5,'
            '"atomic_points":[{"id":"P1","task":"...","objective":"...",'
            '"evidence_needed":["..."],"acceptance_criteria":["..."],'
            '"output_format":"prose|list|table|route|links|mixed","status":"pending"}],'
            '"completion_rule":"..."}. '
            "Every point must have a unique id, task, concrete objective, evidence_needed array, and at least one acceptance_criteria item.\n\n"
            f"\n\nUser goal: {user_query}\n"
            f"Current environment summary: {env_context[:2400]}\n\n"
        )
        raw = ""
        last_nonempty_raw = ""
        last_error = ""
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                request_prompt += (
                    "\nCorrection: the previous output was truncated or invalid. Return one compact, complete "
                    "task_plan.v1 JSON object only. Merge repeated or overlapping points; use no more than 8 "
                    "distinct atomic_points for this retry. For a short how-to, use one point. Do not invent API "
                    "names, flags, values, URLs, examples, or variants. Do not enumerate URLs, sources, examples, or variants. "
                    "Do not add explanation, markdown, or a second object."
                )
            # Keep repair instructions in the user-side prompt.  Appending
            # them after the Assistant continuation marker makes RWKV copy
            # the repair text instead of regenerating the JSON object.
            request_prompt += f"\n{assistant_json_prefix(enable_think=False, prefill_object=True)}"
            try:
                completion_budget = self._completion_budget(request_prompt)
                if attempt:
                    # A repair must be short enough to finish after an overlong
                    # first continuation; the initial budget remains dynamic up
                    # to the configured 10K ceiling.
                    completion_budget = min(completion_budget, 4096)
                response = self.llm.text_completion(
                    request_prompt,
                    max_tokens=completion_budget,
                    stop=JSON_CALL_STOP_SUFFIXES,
                )
                raw = str(response.content or "")
                if raw.strip():
                    last_nonempty_raw = raw
                return self._validate_task_plan(_extract_json_object(raw))
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if classify_error(exc) == "timeout":
                    break
        return {
            "schema_version": "task_plan.v1",
            "status": "error",
            "error_class": "task_plan_invalid",
            "message": last_error or "task plan generation failed",
            "raw_model_output": visible_model_text(last_nonempty_raw or raw),
        }

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
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed = None
        self._replan_generation = 0
        self._ensure_conversation(user_query, env_context, phase)
        self._messages.append(
            {
                "role": "user",
                "content": (
                    "Task plan (model-generated, fixed schema; use it to decide what to retrieve):\n"
                    f"{json.dumps(task_plan, ensure_ascii=False, separators=(',', ':'))}"
                ),
            }
        )
        self._trim_conversation()

    @staticmethod
    def _validate_cross_review(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate the RWKV review protocol without judging its conclusion."""

        if not isinstance(payload, dict):
            raise ValueError("cross-validation output must be a JSON object")
        decision = str(payload.get("decision") or "").strip().casefold()
        if decision not in {"finish", "replan"}:
            raise ValueError("cross-validation decision must be finish or replan")

        missing_points = payload.get("missing_points") or []
        if isinstance(missing_points, str):
            missing_points = [missing_points]
        if not isinstance(missing_points, list):
            raise ValueError("cross-validation missing_points must be an array")

        conflicts = payload.get("conflicts") or []
        if isinstance(conflicts, (str, dict)):
            conflicts = [conflicts]
        if not isinstance(conflicts, list):
            raise ValueError("cross-validation conflicts must be an array")

        return {
            "schema_version": "rwkv-cross-validation.v1",
            "decision": decision,
            "missing_points": [
                str(value)[:500] for value in missing_points[:32] if str(value).strip()
            ],
            "conflicts": conflicts[:32],
            "next_focus": str(payload.get("next_focus") or "")[:2000],
            "reason": str(payload.get("reason") or "")[:2000],
            "review_owner": "rwkv",
        }

    def cross_validate_research(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        evidence_context: str,
    ) -> dict[str, Any]:
        """Ask RWKV whether research should finish or return to planning."""

        user_prompt = (
            "You are the RWKV cross-validator inside a research loop. This is not the final answer. "
            "Inspect the user's requested scope, the model-generated task plan, deterministic tool results, "
            "and the original fetched source spans. Decide whether the collected material supports a useful "
            "answer or whether another retrieval round is needed. Choose replan only for a material missing "
            "fact or material source conflict that another search could reasonably resolve. Minor uncertainty "
            "does not require replan because the final RWKV writer can state uncertainty. The fetched source "
            "spans are already the extracted evidence: read their contents yourself, and choose finish when "
            "they explicitly contain the requested facts. There is no later page-summary or semantic-extraction "
            "tool. Do not choose replan merely because a checklist says retrieved instead of supported, or "
            "because no separate summary was produced. Do not generate a "
            "search query, URL, tool call, answer, refusal, or rewritten evidence. The next planner will choose "
            "all retrieval actions itself. Return exactly one JSON object with this schema: "
            '{"schema_version":"rwkv-cross-validation.v1","decision":"finish|replan",'
            '"missing_points":["P1 or a concise missing fact"],"conflicts":[], '
            '"next_focus":"concise evidence need, not a query","reason":"concise model judgement"}.\n\n'
            f"USER QUESTION:\n{str(user_query or '')}\n\n"
            "MODEL-GENERATED TASK PLAN:\n"
            f"{json.dumps(self._compact_task_plan(task_plan), ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"SHARED RESEARCH MATERIAL:\n{str(evidence_context or '')}\n\n"
            "Return the cross-validation JSON now."
        )
        prompt = (
            f"### User\n{user_prompt}\n"
            f"{assistant_json_prefix(enable_think=False, prefill_object=True)}"
        )
        raw = ""
        try:
            sampling_temperature = get_model_stage_temperature("cross_validation")
            with model_sampling_parameters(sampling_temperature):
                response = self.llm.text_completion(
                    prompt,
                    max_tokens=min(2048, self._completion_budget(prompt)),
                    stop=JSON_CALL_STOP_SUFFIXES,
                )
            raw = str(response.content or "")
            review = self._validate_cross_review(_extract_json_object(raw))
            review["raw_model_output"] = raw
            review["prompt"] = prompt
            review["sampling_temperature"] = sampling_temperature
            review["sampling_seed"] = None
            return review
        except Exception as exc:
            if classify_error(exc) in {
                "timeout",
                "network",
                "auth",
                "quota",
                "rate_limit",
                "provider",
            }:
                raise
            return {
                "schema_version": "rwkv-cross-validation.v1",
                "decision": "protocol_error",
                "error_class": "cross_validation_protocol",
                "message": f"{type(exc).__name__}: {exc}"[:1000],
                "raw_model_output": raw,
                "prompt": prompt,
                "review_owner": "rwkv",
            }

    def rebuild_session_after_review(
        self,
        user_query: str,
        env_context: str,
        review: dict[str, Any],
        phase: str = "DISCOVERY",
    ) -> None:
        """Rebuild planner short-term context while preserving global state."""

        compact_review = {
            key: value
            for key, value in review.items()
            if key not in {"prompt", "raw_model_output"}
        }
        self.rebuild_session(
            user_query,
            env_context,
            {
                "status": "cross_validation_replan",
                "evidence_review": compact_review,
            },
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
        self._latest_routing_observation = self._compact_routing_observation(observation)
        self._next_decision_sampling_stage = "planner_replan"
        self._replan_generation += 1
        seed_material = (
            f"{str(user_query or '')}\0planner_replan\0{self._replan_generation}"
        ).encode("utf-8")
        self._next_decision_seed = int.from_bytes(
            hashlib.blake2s(seed_material, digest_size=4).digest(),
            "big",
        )
        self._ensure_conversation(user_query, env_context, phase)
        self._messages.append(
            {
                "role": "tool",
                "content": self._compact_observation(observation),
                "_routing_observation": True,
            }
        )
        self._trim_conversation()

    @staticmethod
    def _system_prompt(phase: str, task_mode: str = "") -> str:
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
                    "description": row.get("description", ""),
                    "arguments": row.get("arguments") or {"type": "object"},
                }
                for row in catalog_rows
                if isinstance(row, dict)
            ],
            ensure_ascii=False,
            indent=2,
        )
        return (
            "Retrieval decision agent.\n"
            "Choose exactly one next action for the current question and current task plan.\n"
            "Return one JSON object only: {\"name\":\"tool_name\",\"task_point_id\":\"P1\",\"arguments\":{...}}.\n"
            "Choose task_point_id from the current task plan. Keep the same id while gathering evidence for one point; "
            "switch ids only when the next point is the actual retrieval target. For finish_task, retain the point id "
            "that the final decision is based on.\n"
            "Use only the tool names and argument contracts in the catalog. Never emit an answer in tool arguments.\n"
            "Tools:\n"
            f"{catalog}\n"
            "web_search/connector_lookup retrieve; calculator/date_diff/current_time compute or read time; "
            "finish_task requests an RWKV cross-validation review before final synthesis.\n"
            "Retrieval tools internally handle provider selection, URL fetching, cleaning, chunking and evidence extraction.\n"
            "Tool Output is context for your next decision. You decide whether to retrieve again, use another tool, or finish.\n"
            "Choose calculator for arithmetic, current_time for the clock, date_diff for exact date distance, connector_lookup for structured sources, and web_search for general web research when useful.\n"
            "Avoid needless exact repeats, but make every next-step and finish decision yourself.\n"
            "For date_diff, use only exact YYYY-MM-DD values already present in the question or visible evidence.\n"
            f"Current phase: {phase}"
        )

    @staticmethod
    def _compact_task_plan(task_plan: dict[str, Any] | None) -> dict[str, Any]:
        """Project the plan to routing fields instead of replaying its raw JSON."""

        if not isinstance(task_plan, dict):
            return {"status": "missing"}
        compact: dict[str, Any] = {}
        for key in (
            "schema_version",
            "goal",
            "task_mode",
            "source_policy",
            "required_domains",
            "requested_fields",
            "max_items",
            "completion_rule",
        ):
            if key in task_plan:
                value = task_plan[key]
                if isinstance(value, str):
                    value = value[:500]
                elif isinstance(value, list):
                    value = [str(item)[:240] for item in value[:12]]
                compact[key] = value
        points = []
        for point in list(task_plan.get("atomic_points") or [])[:8]:
            if not isinstance(point, dict):
                continue
            projected = {
                key: point.get(key)
                for key in (
                    "id",
                    "task",
                    "objective",
                    "evidence_needed",
                    "acceptance_criteria",
                    "output_format",
                    "status",
                )
                if key in point
            }
            for key in ("task", "objective", "output_format", "status"):
                if isinstance(projected.get(key), str):
                    projected[key] = projected[key][:400]
            for key in ("evidence_needed", "acceptance_criteria"):
                if isinstance(projected.get(key), list):
                    projected[key] = [str(item)[:300] for item in projected[key][:8]]
            points.append(projected)
        compact["atomic_points"] = points
        return compact

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
        for item in list(value.get("candidate_urls") or [])[:6]:
            if not isinstance(item, dict):
                continue
            candidates.append(
                {
                    key: str(item.get(key) or "")[:360]
                    for key in ("candidate_rank", "title", "url", "source", "candidate_score")
                    if key in item
                }
            )
        for item in list(value.get("results") or [])[:6]:
            if not isinstance(item, dict):
                continue
            metadata = {
                key: str(item.get(key) or "")[:360]
                for key in ("title", "url", "source", "scope", "path", "project", "connector")
                if key in item
            }
            snippet = str(item.get("snippet") or "").strip()
            if snippet:
                metadata["snippet"] = snippet[:220]
            if metadata:
                candidates.append(metadata)
        if candidates:
            compact["candidates"] = candidates[:8]

        evidence_context = []
        for item in list(value.get("results") or [])[:4]:
            if not isinstance(item, dict):
                continue
            chunk_candidates = [
                candidate
                for candidate in list(item.get("chunk_candidates") or [])
                if isinstance(candidate, dict)
                and candidate.get("supported") is True
            ][:4]
            source_by_id = {
                str(chunk.get("chunk_id") or ""): str(chunk.get("text") or "")
                for chunk in list(item.get("source_chunks") or [])[:8]
                if isinstance(chunk, dict) and str(chunk.get("chunk_id") or "")
            }
            locators = []
            for candidate in chunk_candidates[:2]:
                chunk_id = str(candidate.get("chunk_id") or "")
                # Generated facts are retained in the execution trace only.
                # The recurrent planner may see the exact located source span
                # and bounded original chunk, never an extractor paraphrase.
                facts: list[str] = []
                quote = str(candidate.get("quote") or "").strip()[:500]
                source_text = source_by_id.get(chunk_id, "")[:700]
                if facts or quote or source_text:
                    locators.append(
                        {
                            "chunk_id": chunk_id,
                            "facts": facts,
                            "quote": quote,
                            "source_text": source_text,
                        }
                    )
            compact_facts = str(item.get("model_extracted_facts") or "").strip()[:1200]
            if locators or compact_facts:
                evidence_context.append(
                    {
                        "title": str(item.get("title") or "")[:240],
                        "url": str(item.get("url") or "")[:360],
                        "evidence_status": str(item.get("evidence_status") or "")[:80],
                        "locators": locators,
                        "compact_facts": compact_facts,
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
                "\nEvidence context boundary: the listed facts/quotes are bounded excerpts tied to fetched page chunks. "
                "Use them to refine the next query or choose a URL; do not treat this routing projection as final-answer evidence."
            )
        max_chars = min(6000, max(2400, observation_chars(get_llm_context_length()) // 2))
        if len(rendered) > max_chars:
            rendered = rendered[:max_chars] + "...[routing observation truncated]"
        return rendered

    def _build_isolated_decision_body(
        self,
        user_query: str,
        env_context: str,
        phase: str,
    ) -> str:
        """Build one short decision input without replaying the tool transcript."""

        plan = json.dumps(
            self._compact_task_plan(self._task_plan),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        task_mode = str((self._task_plan or {}).get("task_mode") or "lookup").casefold()
        observation = self._latest_routing_observation or '{"status":"no_observation"}'
        return (
            f"Question: {str(user_query or '').strip()[:3000]}\n"
            f"Task plan: {plan}\n"
            f"Runtime state: {str(env_context or '').strip()}\n"
            f"Latest tool observation (routing only): {observation}\n"
            f"Decision number: {self._decision_count + 1}\n"
            f"Phase: {phase}\n\n"
            f"{self._system_prompt(phase, task_mode=task_mode)}"
        )

    @staticmethod
    def _render_transcript(messages: list[dict[str, Any]]) -> str:
        return render_tool_transcript(messages)

    def _ensure_conversation(self, user_query: str, env_context: str, phase: str) -> None:
        if self._messages:
            return
        self._messages = [
            {
                "role": "user",
                "content": (
                    f"{user_query}\n\n"
                    "Current environment state:\n"
                    f"{env_context[:5000]}"
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
        # Keep the bounded native transcript as the decision context.  The
        # current envelope refreshes phase/tool visibility, while compact
        # observations preserve the factual thread needed for precise follow-up
        # queries.  Raw audit entries are excluded by the routing marker.
        decision_body = self._build_isolated_decision_body(user_query, env_context, phase)
        history = [
            message
            for message in self._messages
            if not (
                str(message.get("role") or "").casefold() in {"tool", "function", "observation"}
                and not message.get("_routing_observation")
            )
        ]
        history.append({"role": "user", "content": decision_body})
        prompt_limit = planner_prompt_tokens(max(1024, int(get_llm_context_length())))
        while len(history) > 2 and get_token_count(render_tool_transcript(history)) > prompt_limit:
            # Preserve the initial goal and the live decision envelope; remove
            # the oldest bounded history entry first.
            history.pop(1)
        prompt = render_tool_transcript(history)
        raw = ""
        planner_error = ""
        successful_prompt = prompt
        payload: dict[str, Any] = {}
        name = ""
        arguments: dict[str, Any] = {}
        sampling_stage = self._next_decision_sampling_stage
        sampling_temperature = (
            get_model_replan_temperature(self._replan_generation)
            if sampling_stage == "planner_replan"
            else get_model_stage_temperature(sampling_stage)
        )
        sampling_seed = self._next_decision_seed
        # A malformed tool decision is a model/protocol failure. Do not
        # silently issue a controller-authored correction request; expose the
        # failure and let the caller decide what to do next.
        for attempt in range(1):
            request_prompt = prompt
            if attempt:
                request_prompt += (
                    "\n\n校正：上一轮没有返回可执行的完整 JSON。"
                    "现在只返回一个完整 JSON 对象，不要 Markdown、解释、思考过程或额外文字；"
                    '格式必须是 {"name":"工具名","arguments":{}}，工具名必须来自当前 Tools。'
                )
            try:
                with model_sampling_parameters(
                    sampling_temperature,
                    seed=sampling_seed,
                ):
                    if is_local_provider(self.llm.provider):
                        response = self.llm.text_completion(
                            request_prompt,
                            max_tokens=self._completion_budget(request_prompt),
                            stop=JSON_CALL_STOP_SUFFIXES,
                        )
                    else:
                        response = self.llm.chat_completion(
                            [{"role": "user", "content": prompt}],
                            max_tokens=self._completion_budget(
                                request_prompt
                            ),
                        )
                self._next_decision_sampling_stage = "planner"
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
                arguments = parsed_arguments
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
                        "planner_attempts": 1,
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
                "planner_attempts": 1,
                "sampling_temperature": sampling_temperature,
                "sampling_seed": sampling_seed,
            }

        task_point_id = str(payload.get("task_point_id") or payload.get("point_id") or "").strip()
        call = {"name": name, "arguments": arguments}
        call_id = str(payload.get("call_id") or "").strip()
        if task_point_id:
            call["task_point_id"] = task_point_id
        if call_id:
            call["call_id"] = call_id
        self._messages.append({"role": "assistant", "content": call})
        self._decision_count += 1
        return {
            "action": name,
            "args": arguments,
            "task_point_id": task_point_id,
            "call_id": call_id,
            "router": "model_rwkv_json",
            "raw_model_output": visible_model_text(raw),
            "planner_attempts": 1,
            "sampling_temperature": sampling_temperature,
            "sampling_seed": sampling_seed,
        }
