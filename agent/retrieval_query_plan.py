"""Task-record-aware RWKV Retrieval Query Plan for one web transaction.

The Planner still owns the public ``web_search`` call and its primary query.
This module asks RWKV for a small set of complementary discovery queries for
the same unresolved factual record(s).  It does not select providers, URLs,
sources, facts, or answers; malformed output falls back to the primary query.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from agent.runtime_contracts import (
    RETRIEVAL_QUERY_PLAN_CONTRACT,
    require_runtime_contract,
)
from agent.task_plan_contract import record_fields, record_id, task_records
from config import (
    DATA_PIPELINE,
    get_llm_context_length,
    get_model_stage_sampling,
    get_model_stage_temperature,
    is_local_provider,
    model_sampling_parameters,
)
from utils.chunker import get_token_count
from utils.model_budget import bounded_completion_budget
from utils.model_events import visible_model_text
from utils.hard_literals import hard_literal_keys, untrusted_hard_literals
from utils.rwkv_json_protocol import normalize_json_object_envelope
from utils.rwkv_prompt import JSON_CALL_STOP_SUFFIXES, render_tool_transcript


RETRIEVAL_QUERY_INTENTS = frozenset(
    {
        "official_primary",
        "alias_cross_language",
        "temporal_version",
        "verification_counterevidence",
        "complementary",
    }
)


def _text(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[: max(0, int(limit))]


def _single_retrieval_query_plan(
    query: str,
    *,
    task_record_id: str = "",
    status: str = "disabled",
    error: str = "",
) -> dict[str, Any]:
    row = {
        "query_id": "Q1",
        "task_record_id": _text(task_record_id, 80),
        "intent": "planner_primary",
        "query": _text(query, 500),
        "origin": "planner",
    }
    return {
        "contract": RETRIEVAL_QUERY_PLAN_CONTRACT,
        "status": status,
        "expanded": False,
        "queries": [{key: value for key, value in row.items() if value != ""}],
        "attempts": 0,
        "error": _text(error, 1000),
        "raw_model_output": "",
        "prompt": "",
    }


def single_retrieval_query_plan(
    query: str,
    *,
    task_record_id: str = "",
    status: str = "disabled",
) -> dict[str, Any]:
    """Publish the formal contract when expansion is disabled or unnecessary."""

    return _single_retrieval_query_plan(
        query,
        task_record_id=task_record_id,
        status=status,
    )


def _record_progress(
    evidence_ledger_snapshot: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    snapshot = (
        evidence_ledger_snapshot
        if isinstance(evidence_ledger_snapshot, Mapping)
        else {}
    )
    return {
        str(row.get("task_record_id") or ""): {
            "retrieval_state": str(row.get("retrieval_state") or "not_recorded"),
            "attempt_count": int(row.get("attempt_count") or 0),
            "evidence_record_count": int(row.get("evidence_record_count") or 0),
        }
        for row in snapshot.get("task_records") or []
        if isinstance(row, Mapping) and str(row.get("task_record_id") or "").strip()
    }


def select_retrieval_query_records(
    task_plan: Mapping[str, Any] | None,
    *,
    query: str,
    task_record_id: str = "",
    evidence_ledger_snapshot: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select the active record or the still-uncovered records for expansion."""

    records = task_records(task_plan, fallback_query=query)
    requested = _text(task_record_id, 80)
    if requested:
        selected = [record for record in records if record_id(record) == requested]
        if selected:
            return selected

    progress = _record_progress(evidence_ledger_snapshot)
    unresolved = [
        record
        for record in records
        if progress.get(record_id(record), {}).get("retrieval_state")
        != "evidence_recorded"
    ]
    # When every record already has a candidate, the Planner's new search can
    # still be a conflict/currentness follow-up.  Do not suppress its targets.
    return unresolved or records


def _retrieval_query_plan_user_prompt(
    query: str,
    original_goal: str,
    selected_records: list[dict[str, Any]],
    *,
    evidence_ledger_snapshot: Mapping[str, Any] | None,
    recent_queries: list[str] | None,
    runtime_context: str,
    max_queries: int,
) -> str:
    progress = _record_progress(evidence_ledger_snapshot)
    record_rows = [
        {
            "task_record_id": record_id(record),
            "question": _text(record.get("question"), 800),
            "subject": _text(record.get("subject"), 300),
            "relation": _text(record.get("relation"), 240),
            "requested_fields": record_fields(record),
            "time_scope": str(record.get("time_scope") or "unspecified"),
            "progress": progress.get(record_id(record), {}),
        }
        for record in selected_records
    ]
    allowed_ids = [record_id(record) for record in selected_records]
    return (
        "Expand one Planner-selected web query into complementary discovery routes for the "
        "supplied unresolved factual task records. This is search strategy only: do not "
        "answer the question, select a source, assert facts, or invent an exact URL or "
        "hostname. The existing primary query will run automatically, so output only "
        "additional queries that target meaningfully different retrieval angles. Prefer "
        "orthogonal routes over cosmetic synonym changes: official/primary material, a "
        "cross-language or established-name alias, a current/latest route without adding a "
        "new concrete date/version, and an independent verification or counterevidence "
        "route. Preserve literal product names, versions, dates, jurisdictions and requested "
        "fields. A query must target exactly one supplied task_record_id. Do not repeat the primary "
        "query or any recent executed query. Never introduce a year, full date, version, CVE, "
        "or long numeric ID absent from USER GOAL, TRUSTED RUNTIME, or TARGET "
        "TASK RECORDS.\n\n"
        "Return exactly one JSON object with this shape:\n"
        f'{{"contract":"{RETRIEVAL_QUERY_PLAN_CONTRACT}","queries":['
        '{"task_record_id":"P1","intent":"official_primary|alias_cross_language|'
        'temporal_version|verification_counterevidence|complementary",'
        '"query":"concise search query"}]}\n'
        f"Return at most {max(0, max_queries - 1)} additional queries. The allowed Task Record IDs "
        f"are {allowed_ids}. An empty queries array is valid when no safe complementary route "
        "exists. Do not output query_id; the retrieval backend assigns stable IDs.\n\n"
        f"USER GOAL:\n{_text(original_goal, 2400)}\n\n"
        f"TRUSTED RUNTIME:\n{_text(runtime_context, 500)}\n\n"
        f"PLANNER PRIMARY QUERY:\n{_text(query, 500)}\n\n"
        "TARGET TASK RECORDS:\n"
        + json.dumps(record_rows, ensure_ascii=False, separators=(",", ":"))
        + "\n\nRECENT EXECUTED QUERIES (routing history only):\n"
        + json.dumps(
            [_text(value, 500) for value in (recent_queries or []) if _text(value, 500)][-8:],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def build_retrieval_query_plan_prompt(
    query: str,
    original_goal: str,
    selected_records: list[dict[str, Any]],
    *,
    evidence_ledger_snapshot: Mapping[str, Any] | None = None,
    recent_queries: list[str] | None = None,
    runtime_context: str = "",
    max_queries: int = 4,
    correction: str = "",
) -> tuple[str, str]:
    user_prompt = _retrieval_query_plan_user_prompt(
        query,
        original_goal,
        selected_records,
        evidence_ledger_snapshot=evidence_ledger_snapshot,
        recent_queries=recent_queries,
        runtime_context=runtime_context,
        max_queries=max_queries,
    )
    if correction:
        user_prompt += (
            "\n\nPROTOCOL CORRECTION: The previous continuation was invalid ("
            + _text(correction, 400)
            + "). Return only one complete JSON object in the required schema."
        )
    return (
        render_tool_transcript([{"role": "user", "content": user_prompt}], json_output=True),
        user_prompt,
    )


def parse_retrieval_query_plan_output(
    value: Any,
    *,
    primary_query: str,
    selected_records: list[dict[str, Any]],
    task_record_id: str = "",
    max_queries: int = 4,
    recent_queries: list[str] | None = None,
    allowed_hard_literal_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    envelope = normalize_json_object_envelope(value)
    payload = envelope.payload
    require_runtime_contract(
        payload,
        RETRIEVAL_QUERY_PLAN_CONTRACT,
    )
    raw_rows = payload.get("queries")
    if not isinstance(raw_rows, list):
        raise ValueError("query query expansion queries must be an array")

    allowed_ids = {record_id(record) for record in selected_records}
    rows = list(
        _single_retrieval_query_plan(
            primary_query,
            task_record_id=task_record_id if task_record_id in allowed_ids else "",
            status="ok",
        )["queries"]
    )
    seen = {
        _text(value, 500).casefold()
        for value in [primary_query, *(recent_queries or [])]
        if _text(value, 500)
    }
    limit = max(1, min(int(max_queries or 4), 6))
    trusted_keys = (
        set(allowed_hard_literal_keys)
        if allowed_hard_literal_keys is not None
        else hard_literal_keys([primary_query])
    )
    rejected_queries: list[dict[str, Any]] = []
    for raw in raw_rows:
        if len(rows) >= limit:
            break
        if not isinstance(raw, Mapping):
            raise ValueError("each query query expansion row must be an object")
        task_record_id = _text(raw.get("task_record_id"), 80)
        intent = _text(raw.get("intent"), 80).casefold()
        expanded_query = _text(raw.get("query"), 500)
        if task_record_id not in allowed_ids:
            raise ValueError("query query expansion row references an unknown task record")
        if intent not in RETRIEVAL_QUERY_INTENTS:
            raise ValueError("query query expansion row has an invalid intent")
        if not expanded_query:
            raise ValueError("query query expansion row has an empty query")
        untrusted = untrusted_hard_literals(
            expanded_query,
            allowed_keys=trusted_keys,
            provenance="retrieval_query_plan",
        )
        if untrusted:
            rejected_queries.append(
                {
                    "task_record_id": task_record_id,
                    "intent": intent,
                    "query": expanded_query,
                    "reason": "untrusted_hard_literal",
                    "hard_literals": [
                        {
                            "kind": literal.kind,
                            "value": literal.canonical_value,
                            "surface": literal.surface_text,
                        }
                        for literal in untrusted
                    ],
                }
            )
            continue
        signature = expanded_query.casefold()
        if signature in seen:
            continue
        seen.add(signature)
        rows.append(
            {
                "query_id": f"Q{len(rows) + 1}",
                "task_record_id": task_record_id,
                "intent": intent,
                "query": expanded_query,
                "origin": "rwkv_retrieval_query_plan",
            }
        )
    return {
        "contract": RETRIEVAL_QUERY_PLAN_CONTRACT,
        "status": "ok",
        "expanded": len(rows) > 1,
        "queries": rows,
        "input_format": envelope.input_format,
        "transport_normalized": envelope.normalized,
        "rejected_queries": rejected_queries,
        "rejected_query_count": len(rejected_queries),
    }


def generate_retrieval_query_plan(
    query: str,
    original_goal: str,
    task_plan: Mapping[str, Any] | None,
    llm: Any,
    *,
    task_record_id: str = "",
    evidence_ledger_snapshot: Mapping[str, Any] | None = None,
    recent_queries: list[str] | None = None,
    runtime_context: str = "",
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Generate a bounded route set, preserving the primary query on failure."""

    query_text = _text(query, 500)
    selected_records = select_retrieval_query_records(
        task_plan,
        query=query_text,
        task_record_id=task_record_id,
        evidence_ledger_snapshot=evidence_ledger_snapshot,
    )
    if not selected_records:
        return _single_retrieval_query_plan(
            query_text,
            task_record_id=task_record_id,
            status="no_task_records",
        )
    configured_max = (
        max_queries
        if max_queries is not None
        else DATA_PIPELINE.get("retrieval_query_plan_max_queries", 4)
    )
    try:
        query_limit = max(1, min(int(configured_max or 4), 6))
    except (TypeError, ValueError):
        query_limit = 4
    if query_limit <= 1:
        return _single_retrieval_query_plan(
            query_text,
            task_record_id=task_record_id,
            status="disabled",
        )

    try:
        retries = max(
            0,
            min(
                2,
                int(DATA_PIPELINE.get("retrieval_query_plan_protocol_retries", 1) or 0),
            ),
        )
    except (TypeError, ValueError):
        retries = 1
    try:
        requested_max = max(
            128,
            min(
                int(DATA_PIPELINE.get("retrieval_query_plan_max_tokens", 640) or 640),
                1280,
            ),
        )
    except (TypeError, ValueError):
        requested_max = 640

    sampling_temperature = get_model_stage_temperature("retrieval_query_plan")
    sampling_profile = get_model_stage_sampling("retrieval_query_plan")
    provider = str(getattr(llm, "provider", "") or "")
    raw = ""
    prompt = ""
    last_error = ""
    attempts = 0
    trusted_hard_literal_keys = hard_literal_keys(
        [
            original_goal,
            runtime_context,
            json.dumps(selected_records, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    for attempt in range(retries + 1):
        prompt, user_prompt = build_retrieval_query_plan_prompt(
            query_text,
            original_goal,
            selected_records,
            evidence_ledger_snapshot=evidence_ledger_snapshot,
            recent_queries=recent_queries,
            runtime_context=runtime_context,
            max_queries=query_limit,
            correction=last_error if attempt else "",
        )
        if get_token_count(prompt) + requested_max + 256 >= get_llm_context_length():
            last_error = "query query expansion prompt exceeds model context"
            break
        max_tokens_for_call = bounded_completion_budget(
            prompt,
            context_limit=get_llm_context_length(),
            requested_max=requested_max,
            safety_margin=256,
        )
        try:
            attempts = attempt + 1
            with model_sampling_parameters(
                sampling_temperature,
                stage="retrieval_query_plan",
                policy_reason="task_record_aware_retrieval_query_plan",
            ):
                if hasattr(llm, "text_completion") and (
                    not provider or is_local_provider(provider)
                ):
                    response = llm.text_completion(
                        prompt,
                        max_tokens=max_tokens_for_call,
                        stop=JSON_CALL_STOP_SUFFIXES,
                    )
                else:
                    response = llm.chat_completion(
                        [{"role": "user", "content": user_prompt}],
                        max_tokens=max_tokens_for_call,
                    )
            raw = visible_model_text(getattr(response, "content", response))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            break
        try:
            result = parse_retrieval_query_plan_output(
                raw,
                primary_query=query_text,
                selected_records=selected_records,
                task_record_id=task_record_id,
                max_queries=query_limit,
                recent_queries=recent_queries,
                allowed_hard_literal_keys=trusted_hard_literal_keys,
            )
            result.update(
                {
                    "attempts": attempts,
                    "raw_model_output": raw,
                    "prompt": prompt,
                    "sampling_temperature": sampling_temperature,
                    "sampling_parameters": sampling_profile,
                }
            )
            return result
        except ValueError as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    fallback = _single_retrieval_query_plan(
        query_text,
        task_record_id=task_record_id,
        status="unavailable",
        error=last_error,
    )
    fallback.update(
        {
            "attempts": attempts,
            "raw_model_output": raw,
            "prompt": prompt,
            "sampling_temperature": sampling_temperature,
            "sampling_parameters": sampling_profile,
        }
    )
    return fallback


def public_retrieval_query_plan(plan: Mapping[str, Any] | None) -> dict[str, Any]:
    """Remove model transcript payloads from the ordinary tool result."""

    value = plan if isinstance(plan, Mapping) else {}
    return {
        key: value[key]
        for key in (
            "contract",
            "status",
            "expanded",
            "queries",
            "attempts",
            "error",
            "sampling_temperature",
            "sampling_parameters",
            "rejected_queries",
            "rejected_query_count",
        )
        if key in value and value[key] not in ("", None, [])
    }


__all__ = [
    "RETRIEVAL_QUERY_INTENTS",
    "RETRIEVAL_QUERY_PLAN_CONTRACT",
    "build_retrieval_query_plan_prompt",
    "generate_retrieval_query_plan",
    "parse_retrieval_query_plan_output",
    "public_retrieval_query_plan",
    "select_retrieval_query_records",
    "single_retrieval_query_plan",
]
