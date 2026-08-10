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
from agent.tool_protocol import canonicalize_tool_call


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
        self._active_replan_review: dict[str, Any] | None = None
        self._queued_replan_actions: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._messages = []
        self._task_plan = None
        self._latest_routing_observation = ""
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed = None
        self._replan_generation = 0
        self._active_replan_review = None
        self._queued_replan_actions = []

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
            "for separate comparisons or multiple deliverables. Search, source discovery, cross-checking, verification, "
            "citation, summarization, and formatting are research workflow stages, not user-requested facts: never "
            "create atomic points for those stages. Do not create a separate example/code point unless the user "
            "requests an example or code. Classify the task as lookup, latest_list, compare, deep_research, computation, current_time, or date_arithmetic. "
            "Use computation for pure arithmetic, current_time for the current clock, and date_arithmetic only when exact dates are supplied or will be retrieved as facts before calculation. "
            "These deterministic modes do not require web sources. "
            "When the user asks for current/latest/as-of-today information, use the current UTC date supplied in "
            "the environment summary. Never guess or copy a stale year from model memory; include a year in a "
            "search plan only when it is present in the user request or the supplied environment. "
            "For lookup use 1-3 points only when the question genuinely contains multiple requested facts: "
            "authoritative source, requested facts, and one cross-check only when it materially reduces uncertainty. "
            "When one question mixes historical events with a current/latest state, create separate atomic points "
            "for each time anchor so evidence for an old event is not reused as evidence for the current state. "
            "For latest_list set max_items to a small number (normally 3-5); only when the user explicitly asks for all "
            "or a complete list set max_items to 50. "
            "If the user names an organisation, regulator, standards body, or asks for an official source, set "
            "source_policy to official_required. Put a domain in required_domains only when the user explicitly names "
            "that domain or you are certain it is the organisation's actual official host; a bare brand-like domain may "
            "belong to another entity, so leave required_domains empty when uncertain. Never treat a third-party summary "
            "or a same-name website as an official source. "
            "If the task requests a list or table, set output_format to list or table and preserve the requested "
            "fields and order, but do not require every row unless the user explicitly says complete/all. "
            "Keep distinct roles distinct, and give each requested paper or project its own exact URL. "
            "The fields evidence_needed, requested_fields, and acceptance_criteria must describe the user's words "
            "or generic verification needs; never propose a concrete API, command, variable, or example as a guess. "
            "For pure arithmetic, current clock, or exact date-distance questions, plan a deterministic operation rather than web research; "
            "the evidence_needed value may be the deterministic result and must not require a webpage. "
            "Inside JSON string values, escape every inner double quote as \\\"; prefer single quotes in shell examples "
            "and keep the plan compact. Use one point per distinct user-requested deliverable, normally 1-3 points; "
            "only a genuinely complex multi-part request may use up to 8. Group larger related fact sets instead of "
            "expanding the research workflow into points. Never create one point per URL, source, example, or repeated "
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
        # Structured protocol errors receive one correction at the same
        # low-entropy sampling profile.  Additional high-temperature repairs
        # previously expanded malformed plans instead of stabilising JSON.
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                request_prompt += (
                    "\nCorrection: the previous output was truncated or invalid. Return one compact, complete "
                    "task_plan.v1 JSON object only. Merge repeated or overlapping points; use no more than 8 "
                    "distinct atomic_points for this retry. For a short how-to, use one point. Do not invent API "
                    "names, flags, values, URLs, examples, or variants. Do not enumerate URLs, sources, examples, or variants. "
                    "Do not add explanation, markdown, or a second object. The prior validation problem was: "
                    + last_error[:700]
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
        return {
            "schema_version": "task_plan.v1",
            "status": "error",
            "error_class": "task_plan_invalid",
            "message": last_error or "task plan generation failed",
            "raw_model_output": visible_model_text(last_nonempty_raw or raw),
        }

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
        self._decision_count = 0
        self._next_decision_sampling_stage = "planner"
        self._next_decision_seed = None
        self._replan_generation = 0
        self._active_replan_review = None
        self._queued_replan_actions = []
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
    def _validate_cross_review(
        payload: dict[str, Any],
        *,
        planned_point_ids: Iterable[str] = (),
        valid_evidence_refs: Iterable[str] | None = None,
        valid_evidence_text_by_ref: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Validate the RWKV review protocol without judging its conclusion."""

        if not isinstance(payload, dict):
            raise ValueError("cross-validation output must be a JSON object")
        decision = str(payload.get("decision") or "").strip().casefold()

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

        task_point_status = payload.get("task_point_status") or {}
        if not isinstance(task_point_status, dict):
            raise ValueError("cross-validation task_point_status must be an object")

        # Some G1i continuations express the same model judgement as a state
        # snapshot instead of duplicating it in ``decision``.  Canonicalize
        # only fields explicitly emitted by RWKV: no source text is inspected
        # and no controller-side evidence judgement is introduced here.
        atomic_points = payload.get("atomic_points") or []
        if atomic_points and not isinstance(atomic_points, (list, dict)):
            raise ValueError("cross-validation atomic_points must be an array or object")
        normalized_atomic_status: dict[str, Any] = {}
        atomic_rows = (
            list(atomic_points.items())
            if isinstance(atomic_points, dict)
            else [("", row) for row in atomic_points]
        )
        for fallback_id, row in atomic_rows[:32]:
            if not isinstance(row, dict):
                continue
            point_id = str(
                row.get("id")
                or row.get("point_id")
                or row.get("claim_id")
                or fallback_id
                or ""
            ).strip()
            if not point_id:
                continue
            normalized_atomic_status[point_id] = dict(row)
        if not task_point_status and normalized_atomic_status:
            task_point_status = normalized_atomic_status

        incomplete_statuses = {
            "blocked",
            "conflict",
            "conflicted",
            "incomplete",
            "missing",
            "missing_facts",
            "needs_more_evidence",
            "not_supported",
            "not_retrieved",
            "partial",
            "pending",
            "uncertain",
            "unsupported",
            "unverified",
        }
        complete_statuses = {
            "answered",
            "complete",
            "completed",
            "supported",
            "verified",
        }

        normalized_point_status: dict[str, dict[str, Any]] = {}
        for point_id, raw_status in task_point_status.items():
            point_key = str(point_id).strip()
            if not point_key:
                continue
            row = dict(raw_status) if isinstance(raw_status, dict) else {"status": raw_status}
            status = str(row.get("status") or "").strip().casefold()
            if status:
                row["status"] = status
            refs = row.get("evidence_refs") or []
            if isinstance(refs, str):
                refs = [refs]
            if not isinstance(refs, list):
                raise ValueError(
                    f"cross-validation evidence_refs for {point_key} must be an array"
                )
            row["evidence_refs"] = [
                str(value).strip().strip("[]")[:120]
                for value in refs[:16]
                if str(value).strip()
            ]
            supported_facts = row.get("supported_facts") or []
            if isinstance(supported_facts, (str, dict)):
                supported_facts = [supported_facts]
            if not isinstance(supported_facts, list):
                raise ValueError(
                    f"cross-validation supported_facts for {point_key} must be an array"
                )
            normalized_facts: list[dict[str, str]] = []
            for fact in supported_facts[:12]:
                value = dict(fact) if isinstance(fact, dict) else {"value": fact}
                normalized_facts.append(
                    {
                        "field": str(value.get("field") or "")[:160],
                        "value": str(value.get("value") or "")[:500],
                        "evidence_ref": str(value.get("evidence_ref") or "")
                        .strip()
                        .strip("[]")[:120],
                        "quote": str(value.get("quote") or "")[:1000],
                    }
                )
            row["supported_facts"] = normalized_facts
            missing_fields = row.get("missing_fields") or []
            if isinstance(missing_fields, str):
                missing_fields = [missing_fields]
            if not isinstance(missing_fields, list):
                raise ValueError(
                    f"cross-validation missing_fields for {point_key} must be an array"
                )
            row["missing_fields"] = [
                str(value)[:300]
                for value in missing_fields[:12]
                if str(value).strip()
            ]
            normalized_point_status[point_key] = row
        task_point_status = normalized_point_status

        expected_point_ids = list(
            dict.fromkeys(
                str(value).strip()
                for value in planned_point_ids
                if str(value).strip()
            )
        )
        enforce_point_bindings = valid_evidence_refs is not None and bool(
            expected_point_ids
        )
        normalized_valid_refs = {
            str(value).strip().strip("[]")
            for value in (valid_evidence_refs or [])
            if str(value).strip()
        }
        normalized_evidence_by_ref = {
            str(ref).strip().strip("[]"): re.sub(
                r"\s+", " ", str(text or "")
            )
            .strip()
            .casefold()
            for ref, text in (valid_evidence_text_by_ref or {}).items()
            if str(ref).strip()
        }
        enforce_fact_bindings = valid_evidence_text_by_ref is not None
        if enforce_point_bindings:
            omitted = [
                point_id
                for point_id in expected_point_ids
                if point_id not in task_point_status
            ]
            if omitted:
                raise ValueError(
                    "cross-validation task_point_status must cover every planned point; "
                    "missing: " + ", ".join(omitted)
                )
            known_statuses = incomplete_statuses | complete_statuses
            for point_id in expected_point_ids:
                row = task_point_status[point_id]
                status = str(row.get("status") or "").strip().casefold()
                if status not in known_statuses:
                    raise ValueError(
                        f"cross-validation status for {point_id} must be an explicit "
                        "supported/missing/conflict/uncertain state"
                    )
                refs = list(row.get("evidence_refs") or [])
                invalid_refs = [value for value in refs if value not in normalized_valid_refs]
                if invalid_refs:
                    raise ValueError(
                        f"cross-validation evidence_refs for {point_id} contain unavailable refs: "
                        + ", ".join(invalid_refs)
                    )
                if status in complete_statuses and not refs:
                    raise ValueError(
                        f"cross-validation completed status for {point_id} requires an "
                        "available source or tool evidence ref"
                    )
                if status in complete_statuses and enforce_fact_bindings:
                    facts = list(row.get("supported_facts") or [])
                    if not facts:
                        raise ValueError(
                            f"cross-validation completed status for {point_id} requires "
                            "answer-ready supported_facts"
                        )
                    for fact in facts:
                        fact_ref = str(fact.get("evidence_ref") or "")
                        value = str(fact.get("value") or "").strip()
                        quote = str(fact.get("quote") or "").strip()
                        if not value or not quote or fact_ref not in refs:
                            raise ValueError(
                                f"cross-validation supported fact for {point_id} must include "
                                "a value, verbatim quote, and one cited evidence_ref"
                            )
                        normalized_quote = re.sub(r"\s+", " ", quote).strip().casefold()
                        if normalized_quote not in normalized_evidence_by_ref.get(fact_ref, ""):
                            raise ValueError(
                                f"cross-validation quote for {point_id} is not present in "
                                f"cited evidence ref {fact_ref}"
                            )

        declared_status_by_point = {
            str(point_id): str(
                value.get("status") if isinstance(value, dict) else value
            )
            .strip()
            .casefold()
            for point_id, value in task_point_status.items()
            if str(value.get("status") if isinstance(value, dict) else value).strip()
        }

        progress = payload.get("progress") or {}
        if progress and not isinstance(progress, dict):
            raise ValueError("cross-validation progress must be an object")
        completed_count: int | None = None
        total_count: int | None = None
        if isinstance(progress, dict):
            try:
                if progress.get("completed") is not None:
                    completed_count = int(progress.get("completed"))
                if progress.get("total") is not None:
                    total_count = int(progress.get("total"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "cross-validation progress counts must be integers"
                ) from exc

        zero_source_points: list[str] = []
        retrieval_state = payload.get("retrieval_state") or {}
        if retrieval_state and not isinstance(retrieval_state, dict):
            raise ValueError("cross-validation retrieval_state must be an object")
        retrieved_rows = (
            retrieval_state.get("retrieved") or []
            if isinstance(retrieval_state, dict)
            else []
        )
        if isinstance(retrieved_rows, dict):
            retrieved_rows = [
                {"point_id": point_id, **(row if isinstance(row, dict) else {})}
                for point_id, row in retrieved_rows.items()
            ]
        if not isinstance(retrieved_rows, list):
            raise ValueError("cross-validation retrieval_state.retrieved must be an array")
        for row in retrieved_rows[:32]:
            if not isinstance(row, dict):
                continue
            point_id = str(
                row.get("point_id") or row.get("claim_id") or row.get("id") or ""
            ).strip()
            try:
                source_count = int(row.get("source_count"))
            except (TypeError, ValueError):
                continue
            if point_id and source_count == 0:
                zero_source_points.append(point_id)

        derived_missing_points = [
            point_id
            for point_id, status in declared_status_by_point.items()
            if status in incomplete_statuses
        ]
        derived_missing_points.extend(zero_source_points)
        normalized_missing_points: list[str] = []
        for value in [*missing_points, *derived_missing_points]:
            point_id = _declared_task_point_id(value, expected_point_ids)
            normalized = point_id or str(value).strip()
            if normalized and normalized not in normalized_missing_points:
                normalized_missing_points.append(normalized)
        missing_points = normalized_missing_points

        declared_statuses = set(declared_status_by_point.values())
        explicit_progress_incomplete = (
            completed_count is not None
            and total_count is not None
            and completed_count < total_count
        )
        explicit_progress_complete = (
            completed_count is not None
            and total_count is not None
            and total_count >= 0
            and completed_count >= total_count
        )
        has_progress_shape = bool(
            normalized_atomic_status
            or progress
            or retrieval_state
        )
        explicit_gap_fields = bool(
            missing_points
            or conflicts
            or declared_statuses.intersection(incomplete_statuses)
            or explicit_progress_incomplete
            or zero_source_points
        )

        # This is a protocol invariant, not an evidence gate.  RWKV remains
        # the reviewer, but its structured fields must express one coherent
        # decision.  Ask RWKV to correct a contradictory continuation instead
        # of silently privileging one of its fields in controller code.
        if decision == "finish" and explicit_gap_fields:
            raise ValueError(
                "cross-validation output is internally inconsistent: "
                "decision=finish cannot include explicit missing/conflict/incomplete fields"
            )

        decision_source = "explicit_decision"
        if decision not in {"finish", "replan"}:
            # G1i also emits the established cross-validation.v1 shape, where
            # its judgement is expressed through explicit point statuses and
            # missing_points rather than a duplicated decision field.  This
            # is protocol canonicalization only: the controller reads RWKV's
            # declared statuses and never evaluates evidence semantics here.
            if explicit_gap_fields:
                decision = "replan"
                decision_source = (
                    "rwkv_explicit_progress_fields"
                    if has_progress_shape
                    else "rwkv_explicit_gap_fields"
                )
            elif (
                declared_statuses
                and declared_statuses.issubset(complete_statuses)
                and (not progress or explicit_progress_complete)
            ) or (explicit_progress_complete and not declared_statuses):
                decision = "finish"
                decision_source = (
                    "rwkv_explicit_progress_fields"
                    if has_progress_shape
                    else "rwkv_explicit_completion_fields"
                )
            else:
                raise ValueError(
                    "cross-validation decision must be finish/replan or be expressed "
                    "through explicit status/progress fields"
                )

        return {
            "schema_version": "rwkv-cross-validation.v1",
            "decision": decision,
            "decision_source": decision_source,
            "missing_points": [
                str(value)[:500] for value in missing_points[:32] if str(value).strip()
            ],
            "conflicts": conflicts[:32],
            "task_point_status": {
                str(key)[:120]: value
                for key, value in list(task_point_status.items())[:32]
            },
            "next_focus": str(payload.get("next_focus") or "")[:2000],
            "reason": str(payload.get("reason") or "")[:2000],
            "review_owner": "rwkv",
        }

    def cross_validate_research(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        evidence_context: str,
        *,
        evidence_refs: Iterable[str] | None = None,
        evidence_text_by_ref: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Ask RWKV whether research should finish or return to planning."""

        planned_point_ids = [
            str(point.get("id") or "").strip()
            for point in task_plan.get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        available_evidence_refs = (
            None
            if evidence_refs is None
            else list(
                dict.fromkeys(
                    str(value).strip().strip("[]")
                    for value in evidence_refs
                    if str(value).strip()
                )
            )
        )

        point_status_contract = {
            point_id: {
                "status": "supported|missing|conflict|uncertain",
                "evidence_refs": ["S1"],
                "supported_facts": [
                    {
                        "field": "requested field",
                        "value": "answer-ready value explicitly supported by the quote",
                        "evidence_ref": "S1",
                        "quote": "short verbatim source text",
                    }
                ],
                "missing_fields": [],
            }
            for point_id in planned_point_ids
        } or {
            "P1": {
                "status": "supported|missing|conflict|uncertain",
                "evidence_refs": ["S1"],
                "supported_facts": [
                    {
                        "field": "requested field",
                        "value": "answer-ready value explicitly supported by the quote",
                        "evidence_ref": "S1",
                        "quote": "short verbatim source text",
                    }
                ],
                "missing_fields": [],
            }
        }
        review_contract = {
            "schema_version": "rwkv-cross-validation.v1",
            "decision": "finish|replan",
            "missing_points": planned_point_ids[:1],
            "conflicts": [],
            "task_point_status": point_status_contract,
            "next_focus": "concise evidence need, not a query",
            "reason": "concise model judgement",
        }
        projected_target = json.dumps(
            self._cross_validation_plan_projection(task_plan),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        user_prompt = (
            "You are the RWKV cross-validator inside a research loop. This is not the final answer. "
            "Inspect the user's requested scope, the model-generated task plan, deterministic tool results, "
            "and the original fetched source spans. Decide whether the collected material supports a useful "
            "answer or whether another retrieval round is needed. Choose replan only for a material missing "
            "fact or material source conflict that another search could reasonably resolve. Minor uncertainty "
            "does not require replan because the final RWKV writer can state uncertainty. The fetched source "
            "spans are already the extracted evidence: read their contents yourself, and choose finish when "
            "they explicitly contain the requested facts. Review every atomic point separately. For each planned "
            "point, treat every requested_fields value, evidence_needed item, and material acceptance_criteria "
            "item as a required sub-obligation. A point is supported only when its cited spans jointly satisfy "
            "all of those material sub-obligations; if even one requested field is absent, mark the owning point "
            "missing or uncertain and name that field in next_focus. For each point, emit task_point_status with "
            "status, evidence_refs, supported_facts, and missing_fields. For every supported point, "
            "supported_facts must state the exact answer-ready value for the requested field, name one "
            "of that point's evidence_refs, and copy a short verbatim quote from that same source. Do not "
            "paraphrase the quote or infer a date/version/theme that the quote does not state. If you cannot "
            "produce a grounded value and quote for a requested field, put that field in missing_fields and "
            "mark the point missing or uncertain. A supported/completed point must cite at "
            "least one available [S#] source or TOOL_RESULTS ref; never mark a point supported from the question, "
            "general knowledge, or evidence bound only to a different fact. There is no later page-summary or semantic-extraction "
            "tool. For current/latest questions, a historical page that says it was superseded, a prerelease, or a "
            "different version/entity does not by itself support the current answer. A date after the supplied "
            "current runtime cannot describe the current state unless the user explicitly asks for an upcoming "
            "event. For a repository-specific request, matching the host alone is insufficient: bind evidence to "
            "the exact owner/repository. Verify the exact requested entity/version/status/date combination from "
            "the cited source span; if the fields come from different records and cannot be bound safely, request "
            "a focused replan. "
            "Do not choose replan merely because a checklist says retrieved instead of supported, or "
            "because no separate summary was produced. Do not generate a "
            "search query, URL, tool call, answer, refusal, or rewritten evidence. The next planner will choose "
            "all retrieval actions itself. Keep the JSON internally consistent: decision=finish requires "
            "missing_points=[] and conflicts=[]; if any material point remains missing or conflicted, use "
            "decision=replan. Never combine decision=finish with a non-empty missing_points/conflicts list. "
            "missing_points must contain only exact IDs from the supplied task plan; put the factual detail "
            "in next_focus. Return exactly one JSON object with this schema: "
            + json.dumps(review_contract, ensure_ascii=False, separators=(",", ":"))
            + ".\n\n"
            f"USER QUESTION:\n{str(user_query or '')}\n\n"
            "USER-FACING FACTS TO CHECK (not a planner output schema):\n"
            f"{projected_target}\n\n"
            "AVAILABLE EVIDENCE REFS:\n"
            f"{json.dumps(available_evidence_refs or [], ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"SHARED RESEARCH MATERIAL:\n{str(evidence_context or '')}\n\n"
            "FINAL TARGET REMINDER (same user goal, repeated after the evidence for RWKV working memory):\n"
            f"{str(user_query or '')}\n"
            "ATOMIC OBLIGATIONS:\n"
            f"{projected_target}\n\n"
            "Return the cross-validation JSON now."
        )
        prompt = (
            f"### User\n{user_prompt}\n"
            f"{assistant_json_prefix(enable_think=False, prefill_object=True)}"
        )
        raw = ""
        last_error = ""
        last_prompt = prompt
        sampling_temperature = get_model_stage_temperature("cross_validation")
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                repair_user_prompt = user_prompt + (
                    "\n\nCorrection: the previous continuation was not one complete executable JSON object. "
                    f"Protocol error: {last_error[:500]}. "
                    "Return exactly one complete rwkv-cross-validation.v1 JSON object now; no Markdown, "
                    "explanation, answer, search query, or extra text."
                )
                request_prompt = (
                    f"### User\n{repair_user_prompt}\n"
                    f"{assistant_json_prefix(enable_think=False, prefill_object=True)}"
                )
            last_prompt = request_prompt
            try:
                with model_sampling_parameters(
                    sampling_temperature,
                    stage="cross_validation",
                    policy_reason="evidence_consistency_review",
                ):
                    response = self.llm.text_completion(
                        request_prompt,
                        max_tokens=min(2048, self._completion_budget(request_prompt)),
                        stop=JSON_CALL_STOP_SUFFIXES,
                    )
                raw = str(response.content or "")
                review = self._validate_cross_review(
                    _extract_json_object(raw),
                    planned_point_ids=planned_point_ids,
                    valid_evidence_refs=available_evidence_refs,
                    valid_evidence_text_by_ref=evidence_text_by_ref,
                )
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
            "schema_version": "rwkv-cross-validation.v1",
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
        replan_action = self._request_replan_action(
            user_query,
            env_context,
            phase,
        )
        if not replan_action.get("planner_error"):
            generated_candidates = [
                dict(value)
                for value in replan_action.get("candidate_actions") or []
                if isinstance(value, dict)
            ]
            self._queued_replan_actions = generated_candidates or [dict(replan_action)]
        return replan_action

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
        self._ensure_conversation(user_query, env_context, phase)
        if self._task_plan:
            self._messages.append(
                {
                    "role": "user",
                    "content": (
                        "Model-generated task plan retained for this replan:\n"
                        + json.dumps(
                            self._compact_task_plan(self._task_plan),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                }
            )
        self._messages.append(
            {
                "role": "tool",
                "content": self._compact_observation(observation),
                "_routing_observation": True,
            }
        )
        self._trim_conversation()

    def _request_replan_action(
        self,
        user_query: str,
        env_context: str,
        phase: str,
    ) -> dict[str, Any]:
        """Ask RWKV itself for one replacement action at replan temperature."""

        planned_point_ids = [
            str(point.get("id") or "").strip()
            for point in (self._task_plan or {}).get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        review = self._active_replan_review or {}
        missing_point_ids = list(
            dict.fromkeys(
                point_id
                for value in review.get("missing_points") or []
                if (point_id := _declared_task_point_id(value, planned_point_ids))
            )
        )
        frozen_paths = [
            value
            for value in review.get("frozen_paths") or []
            if isinstance(value, dict)
        ]
        attempted_paths = self._retrieval_attempts_from_environment(env_context)
        compact_catalog = [
            {
                "name": row.get("name", ""),
                "description": str(row.get("description") or "")[:500],
                "arguments": row.get("arguments") or {"type": "object"},
                "capabilities": row.get("capabilities") or [],
            }
            for row in json.loads(
                ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
            )
            if isinstance(row, dict) and row.get("name") != "finish_task"
        ]
        point_by_id = {
            str(point.get("id") or "").strip(): point
            for point in (self._task_plan or {}).get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        }
        replan_points = [
            {
                "id": point_id,
                "task": str((point_by_id.get(point_id) or {}).get("task") or "")[:500],
                "evidence_needed": [
                    str(value)[:300]
                    for value in (
                        (point_by_id.get(point_id) or {}).get("evidence_needed") or []
                    )[:6]
                ],
            }
            for point_id in (missing_point_ids or planned_point_ids)
        ]
        replan_state = {
            "missing_points": missing_point_ids,
            "next_focus": str(review.get("next_focus") or "")[:700],
            "reason": str(review.get("reason") or "")[:700],
            "conflicts": list(review.get("conflicts") or [])[:8],
            "frozen_paths": frozen_paths[-8:],
            "attempted_paths": attempted_paths[-12:],
        }
        prior_domain_hypotheses = [
            str(value)[:240]
            for value in (self._task_plan or {}).get("required_domains") or []
            if str(value).strip()
        ][:12]
        example_point_ids = missing_point_ids or planned_point_ids or ["P1"]
        output_contract = json.dumps(
            {
                "candidates": [
                    {
                        "name": "tool_name",
                        "task_point_id": example_point_ids[index % len(example_point_ids)],
                        "arguments": {},
                    }
                    for index in range(3)
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt_body = (
            "You are the RWKV replanner. The prior retrieval strategy did not produce usable evidence. "
            "Generate three materially different candidate tool calls. The application executes protocol-valid "
            "non-frozen calls exactly as RWKV writes them; it never invents a replacement query. Choose every task "
            "point, tool, query or URL, alias/language, and source strategy yourself. Keep the factual obligations, "
            "but treat prior queries and domain suggestions as fallible RWKV hypotheses, not hard restrictions. "
            "When a domain or wording already produced no evidence, test a genuinely different route instead of "
            "merely moving the same words or assigning the same query to another task point. Target missing_points. "
            "Do not answer the user, explain, copy the input objects, or reproduce a task plan/review.\n\n"
            f"USER QUESTION:\n{str(user_query or '').strip()[:3000]}\n\n"
            "CURRENT EVIDENCE STATE (bounded original spans and time):\n"
            + self._replan_environment_projection(env_context)
            + "\n\n"
            "MISSING FACT OBLIGATIONS:\n"
            + json.dumps(replan_points, ensure_ascii=False, separators=(",", ":"))
            + "\n\nACTIVE REVIEW AND FROZEN PATHS:\n"
            + json.dumps(replan_state, ensure_ascii=False, separators=(",", ":"))
            + "\n\nPRIOR DOMAIN HYPOTHESES (may be wrong):\n"
            + json.dumps(prior_domain_hypotheses, ensure_ascii=False, separators=(",", ":"))
            + "\n\nAVAILABLE TOOL CONTRACTS:\n"
            + json.dumps(compact_catalog, ensure_ascii=False, separators=(",", ":"))
            + f"\n\nCURRENT PHASE: {phase}"
            + "\nOUTPUT EXACTLY ONE JSON OBJECT WITH THIS SHAPE AND NO OTHER KEYS:\n"
            + output_contract
        )
        prompt = (
            f"### User\n{prompt_body}\n"
            f"{assistant_json_prefix(enable_think=False, prefill_object=True)}"
        )
        replan_policy_reason = " ".join(
            str(review.get(key) or "")
            for key in ("reason", "next_focus", "decision_source")
        ).strip()
        if self._latest_routing_observation:
            replan_policy_reason = (
                replan_policy_reason
                + " "
                + str(self._latest_routing_observation)[:1200]
            ).strip()
        sampling_temperature = get_model_replan_temperature(
            self._replan_generation,
            replan_policy_reason,
        )
        raw = ""
        last_error = ""
        last_prompt = prompt
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                request_prompt = (
                    "### User\n"
                    + prompt_body
                    + "\n\nCorrection: the prior replan continuation was not one executable, "
                    "materially different candidate-list JSON object. Protocol error: "
                    + last_error[:700]
                    + ". Return three different candidate calls and do not copy a frozen query.\n"
                    + assistant_json_prefix(enable_think=False, prefill_object=True)
                )
            last_prompt = request_prompt
            try:
                with model_sampling_parameters(
                    sampling_temperature,
                    stage="planner_replan",
                    policy_reason=replan_policy_reason or "material_evidence_gap",
                ):
                    response = self.llm.text_completion(
                        request_prompt,
                        max_tokens=min(1536, self._completion_budget(request_prompt)),
                        stop=JSON_CALL_STOP_SUFFIXES,
                    )
                raw = str(response.content or "")
                payload = _extract_json_object(raw)
                candidate_rows = payload.get("candidates")
                if not isinstance(candidate_rows, list) or not candidate_rows:
                    raise ValueError("replan output must contain a non-empty candidates list")
                candidate_errors: list[str] = []
                executable: list[dict[str, Any]] = []
                candidate_route_keys: set[tuple[str, str, str]] = set()
                for index, candidate in enumerate(candidate_rows[:6], start=1):
                    if not isinstance(candidate, dict):
                        candidate_errors.append(f"candidate {index} is not an object")
                        continue
                    candidate = _canonicalize_tool_payload(candidate)
                    name = str(
                        candidate.get("name")
                        or candidate.get("tool_name")
                        or candidate.get("action")
                        or candidate.get("tool")
                        or ""
                    ).strip()
                    arguments = candidate.get("arguments") or candidate.get("args") or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    task_point_id = str(
                        candidate.get("task_point_id")
                        or candidate.get("point_id")
                        or ""
                    ).strip()
                    if len(planned_point_ids) == 1 and not task_point_id:
                        task_point_id = planned_point_ids[0]
                    if not name or not isinstance(arguments, dict):
                        candidate_errors.append(
                            f"candidate {index} lacks a tool name or arguments object"
                        )
                        continue
                    if not ToolRegistry.has(name) or name == "finish_task":
                        candidate_errors.append(
                            f"candidate {index} is not a registered non-finish tool"
                        )
                        continue
                    if missing_point_ids and task_point_id not in missing_point_ids:
                        candidate_errors.append(
                            f"candidate {index} does not target a missing point"
                        )
                        continue
                    if len(planned_point_ids) > 1 and task_point_id not in planned_point_ids:
                        candidate_errors.append(
                            f"candidate {index} lacks a valid task_point_id"
                        )
                        continue
                    action_target = str(
                        arguments.get("query") or arguments.get("url") or ""
                    ).strip()
                    normalized_target = normalize_query(action_target)
                    blocked_paths = [*frozen_paths, *attempted_paths]
                    frozen = any(
                        name == str(path.get("action") or "web_search").strip()
                        and action_target
                        and _same_nonempty_route_target(
                            normalized_target,
                            normalize_query(str(path.get("query") or "").strip()),
                        )
                        for path in blocked_paths
                    )
                    if frozen:
                        candidate_errors.append(
                            f"candidate {index} repeats an attempted or explicitly frozen tool path"
                        )
                        continue
                    route_key = (
                        name,
                        normalized_target,
                        str(arguments.get("connector") or "").strip().casefold(),
                    )
                    if not normalized_target:
                        route_key = (
                            name,
                            json.dumps(
                                arguments,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                default=str,
                            ),
                            route_key[2],
                        )
                    if route_key in candidate_route_keys:
                        candidate_errors.append(
                            f"candidate {index} duplicates another candidate route"
                        )
                        continue
                    candidate_route_keys.add(route_key)
                    executable.append(
                        {
                            "action": name,
                            "args": dict(arguments),
                            "task_point_id": task_point_id,
                            "call_id": str(candidate.get("call_id") or "").strip(),
                            "router": "model_rwkv_replan_json",
                            "raw_model_output": visible_model_text(raw),
                            "planner_attempts": attempt + 1,
                            "planner_error": "",
                            "prompt": request_prompt,
                            "sampling_temperature": sampling_temperature,
                            "sampling_seed": None,
                            "sampling_policy_reason": replan_policy_reason,
                            "replan_generation": self._replan_generation,
                            "decision_owner": "rwkv",
                            "candidate_index": index,
                        }
                    )
                if not executable or (
                    len(candidate_rows) > 1 and len(executable) < 2
                ):
                    raise ValueError(
                        "replan candidates were not materially diverse: "
                        + "; ".join(candidate_errors[:6])
                    )
                return {**executable[0], "candidate_actions": executable}
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
            "action": "",
            "args": {},
            "task_point_id": "",
            "router": "model_rwkv_replan_protocol_error",
            "raw_model_output": visible_model_text(raw),
            "planner_attempts": 2,
            "planner_error": last_error[:1000],
            "prompt": last_prompt,
            "sampling_temperature": sampling_temperature,
            "sampling_seed": None,
            "sampling_policy_reason": replan_policy_reason,
            "replan_generation": self._replan_generation,
            "decision_owner": "rwkv",
        }

    def mark_replan_progress(self, task_point_id: str) -> None:
        """Clear a stale review only after its own missing point advances.

        The controller does not decide whether the evidence is sufficient. It
        only prevents a new source for an unrelated point from erasing the
        prior RWKV cross-validator's requested focus.
        """

        review = self._active_replan_review or {}
        missing_ids = {
            str(value).strip()
            for value in review.get("missing_points") or []
            if str(value).strip()
        }
        point_id = str(task_point_id or "").strip()
        if not review or (missing_ids and point_id not in missing_ids):
            return
        self._queued_replan_actions = []
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
            "When an active RWKV replan review is present, advance one of its missing_points before returning to an unrelated already-attempted point. Treat prior query wording and required_domains as revisable model hypotheses: if they produced no evidence, reconsider aliases, language, source class, domain, connector, or direct URL. If it lists frozen_paths, do not submit the same action and query or URL again; choose a materially different retrieval route yourself. The review never supplies a replacement query: choose the tool, query, and source yourself.\n"
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
    def _cross_validation_plan_projection(
        task_plan: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Project only answer obligations, without replaying task-plan protocol.

        Cross-validation asks RWKV for a different JSON schema.  Replaying a
        complete ``task_plan.v1`` object in the same prompt made the recurrent
        model continue the embedded planner schema instead of emitting the
        requested review.  This projection keeps every user-facing evidence
        obligation but removes planner-only protocol and workflow fields.
        """

        compact = Planner._compact_task_plan(task_plan)
        return {
            "goal": str(compact.get("goal") or "")[:500],
            "source_policy": str(compact.get("source_policy") or "")[:120],
            "required_domains": list(compact.get("required_domains") or [])[:12],
            "requested_fields": list(compact.get("requested_fields") or [])[:12],
            "points_to_check": [
                {
                    key: point.get(key)
                    for key in ("id", "task", "objective", "evidence_needed")
                    if key in point
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
        """Keep current time and original source spans visible to Replan.

        The shared state places feedback and query history before source
        locators. Taking only its first characters therefore repeated the
        failed route while hiding the evidence needed to change strategy.
        Parse the labelled state sections and preserve fetched spans verbatim.
        """

        text = str(env_context or "").strip()
        if not text:
            return ""
        lines = text.splitlines()
        labelled: dict[str, Any] = {}
        labels = {
            "Question time/freshness policy": "freshness",
            "Shared retrieval ledger": "retrieval_ledger",
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

        locators = labelled.get("source_locators")
        if isinstance(locators, dict):
            projected_sources = []
            for source in list(locators.get("sources") or [])[:4]:
                if not isinstance(source, dict):
                    continue
                projected_sources.append(
                    {
                        key: source.get(key)
                        for key in (
                            "source_id",
                            "title",
                            "url",
                            "claim_ids",
                            "published",
                            "updated",
                            "date",
                            "provider",
                            "freshness",
                            "spans",
                        )
                        if key in source
                    }
                )
            labelled["source_locators"] = {
                "source_count": locators.get("source_count"),
                "visible_source_count": len(projected_sources),
                "sources": projected_sources,
            }

        if not labelled:
            return text[:2000]
        rendered = json.dumps(labelled, ensure_ascii=False, separators=(",", ":"))
        return rendered[:6500]

    @staticmethod
    def _compact_replan_review(review: dict[str, Any] | None) -> dict[str, Any] | None:
        """Retain only the prior RWKV review fields needed for routing."""

        if not isinstance(review, dict):
            return None
        compact: dict[str, Any] = {}
        for key in ("decision", "decision_source", "next_focus", "reason"):
            value = str(review.get(key) or "").strip()
            if value:
                compact[key] = value[:800]
        for key in ("missing_points", "conflicts"):
            values = review.get(key) or []
            if isinstance(values, (str, dict)):
                values = [values]
            if isinstance(values, list):
                compact[key] = [
                    str(value)[:500]
                    for value in values[:16]
                    if str(value).strip()
                ]
        statuses = review.get("task_point_status") or {}
        if isinstance(statuses, dict):
            compact["task_point_status"] = {
                str(point_id)[:120]: {
                    "status": str(
                        value.get("status") if isinstance(value, dict) else value
                    )[:120],
                    "evidence_refs": [
                        str(ref)[:120]
                        for ref in (
                            value.get("evidence_refs") or []
                            if isinstance(value, dict)
                            else []
                        )[:8]
                    ],
                }
                for point_id, value in list(statuses.items())[:16]
            }
        frozen_rows = review.get("frozen_paths") or []
        frozen_path = review.get("frozen_path")
        if isinstance(frozen_path, dict):
            frozen_rows = [*list(frozen_rows or []), frozen_path]
        if isinstance(frozen_rows, dict):
            frozen_rows = [frozen_rows]
        if isinstance(frozen_rows, list):
            projected_frozen = [
                {
                    "action": str(value.get("action") or "")[:120],
                    "query": str(
                        value.get("query")
                        or (value.get("arguments") or {}).get("query")
                        or (value.get("arguments") or {}).get("url")
                        or ""
                    )[:500],
                    "task_point_id": str(value.get("task_point_id") or "")[:120],
                    "reason": str(value.get("reason") or "")[:240],
                }
                for value in frozen_rows[-8:]
                if isinstance(value, dict)
            ]
            if projected_frozen:
                compact["frozen_paths"] = projected_frozen
        return compact or None

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
        active_replan = json.dumps(
            self._active_replan_review or {"status": "none"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        planned_point_ids = [
            str(point.get("id") or "").strip()
            for point in (self._task_plan or {}).get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        active_missing_ids = [
            point_id
            for value in (self._active_replan_review or {}).get("missing_points") or []
            if (point_id := _declared_task_point_id(value, planned_point_ids))
        ]
        task_point_example = (
            active_missing_ids[0]
            if active_missing_ids
            else planned_point_ids[0]
            if planned_point_ids
            else "P1"
        )
        return (
            f"Question: {str(user_query or '').strip()[:3000]}\n"
            f"Task plan: {plan}\n"
            f"Active RWKV replan review: {active_replan}\n"
            f"Runtime state: {str(env_context or '').strip()}\n"
            f"Latest tool observation (routing only): {observation}\n"
            f"Decision number: {self._decision_count + 1}\n"
            f"Phase: {phase}\n\n"
            f"{self._system_prompt(phase, task_mode=task_mode, task_point_example=task_point_example)}"
        )

    @staticmethod
    def _render_transcript(messages: list[dict[str, Any]]) -> str:
        return render_tool_transcript(messages)

    def _ensure_conversation(self, user_query: str, env_context: str, phase: str) -> None:
        del env_context, phase
        if self._messages:
            return
        self._messages = [
            {
                "role": "user",
                # The live routing projection is supplied once in every
                # isolated decision body. Replaying a stale 5K prefix here
                # duplicated state and pushed replans to the 16K limit.
                "content": str(user_query or ""),
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

    def _bind_existing_action_to_task_point(
        self,
        user_query: str,
        *,
        action: str,
        arguments: dict[str, Any],
        planned_point_ids: list[str],
    ) -> dict[str, Any]:
        """Let RWKV bind its existing action without regenerating that action.

        ``task_point_id`` is routing metadata, not a reason to discard an
        otherwise valid model-selected tool call.  This small request asks
        RWKV to associate the exact action it already chose with one of its
        own task-plan points.  The controller never changes the action or its
        arguments, and a binding protocol failure remains non-fatal.
        """

        compact_points = [
            {
                key: point.get(key)
                for key in ("id", "task", "objective", "evidence_needed")
                if key in point
            }
            for point in list((self._task_plan or {}).get("atomic_points") or [])[:16]
            if isinstance(point, dict)
            and str(point.get("id") or "").strip() in planned_point_ids
        ]
        fixed_call = {"name": action, "arguments": arguments}
        prompt = (
            "### User\n"
            "You are the RWKV task-point binder. The RWKV planner has already chosen the tool call below. "
            "Do not change, regenerate, improve, or reject its name or arguments. Associate that exact call "
            "with the one atomic point it is trying to advance. Return exactly one JSON object in the form "
            '{"task_point_id":"P1"}; no Markdown, explanation, tool call, query, or extra fields.\n\n'
            f"Original user question: {str(user_query or '').strip()[:3000]}\n"
            "Model-generated atomic points: "
            f"{json.dumps(compact_points, ensure_ascii=False, separators=(',', ':'))}\n"
            "Fixed RWKV tool call (read-only): "
            f"{json.dumps(fixed_call, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Valid task_point_id values: {', '.join(planned_point_ids)}\n"
            f"{assistant_json_prefix(enable_think=False, prefill_object=False)}"
        )
        raw = ""
        sampling_temperature = get_model_stage_temperature("task_point_binding")
        try:
            with model_sampling_parameters(
                sampling_temperature,
                stage="task_point_binding",
                policy_reason="low_entropy_task_point_classification",
            ):
                if is_local_provider(self.llm.provider):
                    response = self.llm.text_completion(
                        prompt,
                        max_tokens=256,
                        stop=JSON_CALL_STOP_SUFFIXES,
                    )
                else:
                    response = self.llm.chat_completion(
                        [{"role": "user", "content": prompt}],
                        max_tokens=256,
                    )
            raw = str(response.content or "")
            payload = _extract_json_object(raw)
            task_point_id = str(
                payload.get("task_point_id") or payload.get("point_id") or ""
            ).strip()
            if task_point_id not in planned_point_ids:
                raise ValueError(
                    "task-point binding must choose one of: "
                    + ", ".join(planned_point_ids)
                )
            return {
                "task_point_id": task_point_id,
                "binding_method": "rwkv_binding_request",
                "raw_model_output": visible_model_text(raw),
                "prompt": prompt,
                "error": "",
                "sampling_temperature": sampling_temperature,
            }
        except Exception as exc:
            return {
                "task_point_id": "",
                "binding_method": "unassigned",
                "raw_model_output": visible_model_text(raw),
                "prompt": prompt,
                "error": f"{type(exc).__name__}: {exc}",
                "sampling_temperature": sampling_temperature,
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
        if self._queued_replan_actions:
            queued = dict(self._queued_replan_actions.pop(0))
            if not self._queued_replan_actions:
                self._next_decision_sampling_stage = (
                    "planner_replan" if self._active_replan_review else "planner"
                )
            self._next_decision_seed = None
            call = {
                "name": str(queued.get("action") or ""),
                "arguments": dict(queued.get("args") or {}),
            }
            if queued.get("task_point_id"):
                call["task_point_id"] = str(queued["task_point_id"])
            if queued.get("call_id"):
                call["call_id"] = str(queued["call_id"])
            self._messages.append({"role": "assistant", "content": call})
            self._decision_count += 1
            return queued
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
        task_point_id = ""
        task_point_binding_method = "not_required"
        task_point_binding_raw = ""
        task_point_binding_error = ""
        planned_point_ids = [
            str(point.get("id") or "").strip()
            for point in (self._task_plan or {}).get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
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
                point_protocol = (
                    " For this multi-point task, include task_point_id and choose exactly one of: "
                    + ", ".join(planned_point_ids)
                    + "."
                    if len(planned_point_ids) > 1
                    else ""
                )
                correction = (
                    "Correction: the previous continuation was not one complete executable JSON object. "
                    "Return exactly one complete JSON object now; no Markdown, explanation, thinking, or extra text. "
                    'Use {"name":"tool_name","task_point_id":"P1","arguments":{}} and choose the tool '
                    "from the current catalog."
                    + point_protocol
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

        if len(planned_point_ids) > 1 and name != "finish_task":
            if task_point_id in planned_point_ids:
                task_point_binding_method = "inline_rwkv"
            else:
                binding = self._bind_existing_action_to_task_point(
                    user_query,
                    action=name,
                    arguments=arguments,
                    planned_point_ids=planned_point_ids,
                )
                task_point_id = str(binding.get("task_point_id") or "").strip()
                task_point_binding_method = str(
                    binding.get("binding_method") or "unassigned"
                )
                task_point_binding_raw = str(binding.get("raw_model_output") or "")
                task_point_binding_error = str(binding.get("error") or "")
        elif task_point_id and task_point_id in planned_point_ids:
            task_point_binding_method = "inline_rwkv"

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
            "planner_attempts": attempt + 1,
            "task_point_binding_method": task_point_binding_method,
            "task_point_binding_raw_model_output": task_point_binding_raw,
            "task_point_binding_error": task_point_binding_error,
            "task_point_binding_temperature": (
                binding.get("sampling_temperature")
                if task_point_binding_method in {"rwkv_binding_request", "unassigned"}
                else None
            ),
            "sampling_temperature": sampling_temperature,
            "sampling_seed": sampling_seed,
        }
