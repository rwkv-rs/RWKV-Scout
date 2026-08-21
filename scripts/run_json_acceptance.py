"""Run real model-owned retrieval cases from a UTF-8 JSON file.

The query never comes from a shell literal.  This keeps Chinese input and the
resulting trace stable across PowerShell, WSL and CI environments.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import statistics
import re
import sys
import tempfile
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Make direct execution (`python scripts/run_json_acceptance.py`) use the
# repository root just like module execution does.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.orchestrator import Orchestrator
from config import get_analysis_timeout_seconds
from utils.runtime_gate import analysis_slot
from utils.error_policy import classify_error
from utils.token_tracker import current_task_id
from utils.task_events import append_task_event, get_task_events


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "case"))
    return cleaned.strip("._") or "case"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Durably replace one checkpoint without exposing a truncated JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, existing_mode)
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _case_timeout(seconds: float | None):
    """Bound one case while retaining any partial task events."""
    if seconds is None or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"case exceeded timeout of {seconds:.1f} seconds")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer and previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        cases = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
                cases.append(record)
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("input must contain a non-empty cases list")
    normalized = []
    for index, raw_case in enumerate(cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {index} must be an object")
        case = dict(raw_case)
        # Keep benchmark files in their native shape: gold suites use
        # ``question``/``id`` while acceptance suites use ``query``/``case_id``.
        if not str(case.get("query") or "").strip():
            case["query"] = case.get("question") or case.get("prompt") or ""
        if not str(case.get("query") or "").strip():
            raise ValueError(f"case {index} must contain a non-empty query")
        case.setdefault("case_id", case.get("id") or f"case_{index:03d}")
        normalized.append(case)
    return normalized


_CASE_RUNTIME_METADATA_KEYS: tuple[str, ...] = ()
_CASE_RUNTIME_BOOLEAN_KEYS: frozenset[str] = frozenset()


def _runtime_metadata_for_case(
    case: dict[str, Any],
    *,
    max_tool_steps_override: int | None = None,
) -> dict[str, Any]:
    """Project technical controls only; references never reach runtime."""

    metadata: dict[str, Any] = {}
    if max_tool_steps_override is not None:
        metadata["max_tool_steps"] = int(max_tool_steps_override)
    elif case.get("max_tool_steps") is not None:
        metadata["max_tool_steps"] = int(case["max_tool_steps"])
    for key in _CASE_RUNTIME_METADATA_KEYS:
        if key not in case:
            continue
        metadata[key] = bool(case[key]) if key in _CASE_RUNTIME_BOOLEAN_KEYS else case[key]
    return metadata


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _normalise_offline_fact(value: Any) -> str:
    """Canonicalise formatting for post-run fact comparison only."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(
        r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b",
        lambda match: (
            f"{int(match.group(3)):04d}-{_MONTHS[match.group(1)]:02d}-{int(match.group(2)):02d}"
        ),
        text,
    )
    text = re.sub(
        r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        lambda match: f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}",
        text,
    )
    text = re.sub(
        r"(?<!\d)(\d{4})[./-](\d{1,2})[./-](\d{1,2})(?!\d)",
        lambda match: f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}",
        text,
    )
    text = re.sub(r"\b(?:about|around|approximately|approx\.?|roughly)\b|大约|大概|约", "", text)
    return re.sub(r"[^0-9a-z\u3400-\u9fff.+/_-]+", "", text)


def _offline_quality(case: dict[str, Any], answer: str) -> tuple[str | None, dict[str, Any] | None]:
    """Compare a completed answer with explicit gold facts after runtime.

    This function is intentionally located in the evaluation runner.  Its
    result is never sent to the orchestrator, planner, retrieval tools, RWKV,
    task events, or the public API.
    """

    gold = case.get("gold") if isinstance(case.get("gold"), dict) else {}
    required = [str(value) for value in gold.get("required_facts") or [] if str(value).strip()]
    forbidden = [str(value) for value in gold.get("forbidden_facts") or [] if str(value).strip()]
    if not required and not forbidden:
        return None, None

    normalised_answer = _normalise_offline_fact(answer)
    required_rows = [
        {
            "fact": fact,
            "matched": bool(_normalise_offline_fact(fact))
            and _normalise_offline_fact(fact) in normalised_answer,
        }
        for fact in required
    ]
    forbidden_rows = [
        {
            "fact": fact,
            "matched": bool(_normalise_offline_fact(fact))
            and _normalise_offline_fact(fact) in normalised_answer,
        }
        for fact in forbidden
    ]
    passed = bool(str(answer or "").strip()) and all(
        row["matched"] for row in required_rows
    ) and not any(row["matched"] for row in forbidden_rows)
    return (
        "pass" if passed else "no-pass",
        {
            "method": "offline_explicit_fact_match.v1",
            "required": required_rows,
            "forbidden": forbidden_rows,
            "reference_answer": str(case.get("final_answer") or case.get("reference_answer") or ""),
        },
    )


def _json_result(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("result")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _numeric_summary(values: list[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return {"count": 0, "min": 0, "max": 0, "avg": 0}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "avg": round(statistics.fmean(numbers), 2),
    }


_RWKV_SEMANTIC_CONTROL_EVENTS = frozenset(
    {
        "evidence_review",
        "task_record_binding",
        "planner_session_rebuilt",
        "task_replan",
    }
)
_PROHIBITED_OUTPUT_INTERVENTION_EVENTS = frozenset(
    {
        "completion_judgement",
        "answer_repair",
        "answer_rewrite",
        "answer_translation",
        "answer_fallback",
        "refusal_fallback",
    }
)
_RWKV_SEMANTIC_CONTROL_STAGES = frozenset(
    {
        "evidence_review",
        "planner_replan",
        "task_record_binding",
    }
)
_PROHIBITED_OUTPUT_INTERVENTION_STAGES = frozenset(
    {
        "answer_repair",
        "answer_rewrite",
        "answer_translation",
        "answer_fallback",
    }
)


def _module_responsibility_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure whether online modules stay inside their declared boundaries.

    Retrieval may fetch, clean, chunk, deduplicate and verify an exact quote
    span.  It must not add a second semantic completion decision, replace an
    RWKV-selected action, or rewrite the public RWKV answer.  These metrics are
    offline observations only; they never affect execution.
    """

    rwkv_control_events = collections.Counter(
        str(event.get("type") or "")
        for event in events
        if str(event.get("type") or "") in _RWKV_SEMANTIC_CONTROL_EVENTS
    )
    prohibited_events = collections.Counter(
        str(event.get("type") or "")
        for event in events
        if str(event.get("type") or "") in _PROHIBITED_OUTPUT_INTERVENTION_EVENTS
    )
    rwkv_control_stages = collections.Counter()
    prohibited_stages = collections.Counter()
    controller_overrides = 0
    gateway_overrides = 0
    for event in events:
        if event.get("controller_override") is True or event.get("override") is True:
            controller_overrides += 1
        if event.get("gateway_override") is True:
            gateway_overrides += 1
        if event.get("type") != "model_call":
            continue
        stage = str(
            event.get("request_stage")
            or event.get("sampling_stage")
            or event.get("stage")
            or ""
        ).strip()
        if stage in _RWKV_SEMANTIC_CONTROL_STAGES:
            rwkv_control_stages[stage] += 1
        if stage in _PROHIBITED_OUTPUT_INTERVENTION_STAGES:
            prohibited_stages[stage] += 1

    synthesis_outputs = [
        str(event.get("content") or "")
        for event in events
        if event.get("type") == "synthesis" and str(event.get("content") or "")
    ]
    final_outputs = [
        str(event.get("content") or "")
        for event in events
        if event.get("type") == "final" and str(event.get("content") or "")
    ]
    final_output_mismatch = int(
        bool(synthesis_outputs)
        and bool(final_outputs)
        and synthesis_outputs[-1] != final_outputs[-1]
    )
    prohibited_interventions = sum(prohibited_events.values()) + sum(
        prohibited_stages.values()
    )
    violation_count = (
        prohibited_interventions
        + controller_overrides
        + gateway_overrides
        + final_output_mismatch
    )
    return {
        "schema_version": "module-responsibility-isolation.v2",
        "pass": violation_count == 0,
        "violation_count": violation_count,
        "rwkv_semantic_control_count": sum(rwkv_control_events.values())
        + sum(rwkv_control_stages.values()),
        "rwkv_semantic_control_events": dict(rwkv_control_events),
        "rwkv_semantic_control_model_stages": dict(rwkv_control_stages),
        "prohibited_output_intervention_count": prohibited_interventions,
        "prohibited_output_intervention_events": dict(prohibited_events),
        "prohibited_output_intervention_model_stages": dict(prohibited_stages),
        "controller_override_count": controller_overrides,
        "gateway_override_count": gateway_overrides,
        "final_output_mismatch_count": final_output_mismatch,
        "trace_bytes": len(
            json.dumps(events, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ),
    }


def _trace_summary(task_id: str) -> dict[str, Any]:
    """Summarize execution boundaries without judging task-specific content."""
    events = get_task_events(task_id)
    decisions = []
    tool_calls = []
    tool_results = []
    forks = []
    evidence = []
    plans = []
    strategy_selections = []
    judgements = []
    contexts = []
    validations = []
    finals = []
    model_calls = []
    step_limits = []
    ledger_events = []
    ledger_snapshot = {}
    web_search_stages = []
    web_search_chunks = []
    chunk_tokens = []
    chunk_chars = []
    chunk_windows = []
    prompt_chars = []
    candidate_output_chars = []
    candidate_finish_reasons: collections.Counter[str] = collections.Counter()
    candidate_retry_calls = 0
    action_counts: collections.Counter[str] = collections.Counter()
    phase_counts: collections.Counter[str] = collections.Counter()
    error_classes: collections.Counter[str] = collections.Counter()
    tool_result_statuses: collections.Counter[str] = collections.Counter()
    supported_candidates = 0
    unsupported_candidates = 0
    for event in events:
        event_type = event.get("type")
        if event_type == "model_call":
            model_calls.append(
                {
                    key: event.get(key)
                    for key in (
                        "seq",
                        "timestamp",
                        "phase",
                        "status",
                        "operation",
                        "provider",
                        "backend",
                        "model",
                        "duration_ms",
                        "prompt_tokens",
                        "completion_tokens",
                        "request_max_tokens",
                        "finish_reason",
                        "stop",
                        "request_stage",
                        "sampling_policy_reason",
                        "temperature",
                        "seed",
                        "sampling_parameters",
                        "input_messages",
                        "prompt",
                        "output",
                        "error",
                    )
                    if key in event
                }
            )
        elif event_type == "step_limit_reached":
            step_limits.append(
                {
                    key: event.get(key)
                    for key in ("seq", "timestamp", "step", "phase", "action", "max_steps", "evidence_rounds", "message")
                    if key in event
                }
            )
        elif event_type == "retrieval_ledger":
            ledger_events.append(
                {
                    "step": event.get("step"),
                    "phase": event.get("phase"),
                    "branch_id": event.get("branch_id") or "",
                    "task_record_id": event.get("task_record_id") or "",
                    "action": event.get("action") or "",
                    "query": event.get("query") or "",
                    "data": event.get("data") or {},
                    "snapshot": event.get("snapshot") or {},
                }
            )
            if isinstance(event.get("snapshot"), dict):
                ledger_snapshot = event.get("snapshot") or {}
        elif event_type == "web_search_stage":
            stage = str(event.get("stage") or "")
            page = event.get("page") or {}
            web_search_stages.append(
                {
                    "seq": event.get("seq"),
                    "step": event.get("step"),
                    "phase": event.get("phase"),
                    "stage": stage,
                    "status": event.get("status") or page.get("status") or "",
                    "query": event.get("query") or "",
                    "candidate_count": event.get("candidate_count"),
                    "fetched_count": event.get("fetched_count"),
                    "evidence_count": event.get("evidence_count"),
                    "url": page.get("url") or "",
                    "page_chars": page.get("page_chars"),
                    "chunk_count": page.get("chunk_count"),
                    "chunk_window_tokens": page.get("chunk_window_tokens"),
                    "parallel_candidate": page.get("parallel_candidate") or {},
                    "errors": page.get("errors") or [],
                }
            )
            if stage == "page_evidence" and page:
                evidence.append(
                    {
                        "step": event.get("step"),
                        "url": page.get("url") or "",
                        "data": page,
                        "compact_facts": "",
                        "source": "generic_web_search",
                    }
                )
                chunk_windows.append(page.get("chunk_window_tokens"))
        elif event_type == "web_search_chunk":
            chunk = event.get("chunk") or {}
            web_search_chunks.append(
                {
                    "seq": event.get("seq"),
                    "phase": event.get("phase"),
                    "url": event.get("url") or "",
                    "chunk_id": chunk.get("chunk_id") or "",
                    "index": chunk.get("index"),
                    "chars": chunk.get("chars") or len(str(chunk.get("text") or "")),
                    "token_count": chunk.get("token_count"),
                    "candidate": event.get("candidate") or {},
                }
            )
            chunk_tokens.append(chunk.get("token_count"))
            chunk_chars.append(chunk.get("chars") or len(str(chunk.get("text") or "")))
        elif event_type == "task_plan":
            plan = event.get("data") or {}
            plans.append(
                {
                    "contract": plan.get("contract"),
                    "status": plan.get("status", "ok"),
                    "error_class": plan.get("error_class", ""),
                    "message": str(plan.get("message") or "")[:1000],
                    "raw_model_output_chars": len(
                        str(plan.get("raw_model_output") or "")
                    ),
                    "record_count": len(plan.get("records") or [])
                    if isinstance(plan, dict)
                    else 0,
                    "record_ids": [
                        str(point.get("record_id") or "")
                        for point in (plan.get("records") or [])
                        if isinstance(point, dict)
                    ],
                }
            )
        elif event_type == "task_replan":
            plan = event.get("data") or {}
            plans.append(
                {
                    "contract": plan.get("contract")
                    if isinstance(plan, dict)
                    else None,
                    "status": plan.get("status", "ok") if isinstance(plan, dict) else "unknown",
                    "record_count": len(plan.get("records") or []) if isinstance(plan, dict) else 0,
                    "record_ids": [
                        str(point.get("record_id") or "")
                        for point in (plan.get("records") or [])
                        if isinstance(point, dict)
                    ],
                }
            )
        elif event_type == "retrieval_strategy_selected":
            strategy_selections.append(
                {
                    "step": event.get("step"),
                    "strategy": event.get("strategy") or "",
                    "record_count": event.get("record_count") or event.get("point_count") or 0,
                    "record_ids": event.get("record_ids") or event.get("point_ids") or [],
                    "source": event.get("source") or "",
                    "reason": event.get("reason") or "",
                    "override": bool(event.get("override")),
                }
            )
        elif event_type == "model_tool_decision":
            action = str(event.get("action") or "")
            phase = str(event.get("phase") or "")
            action_counts[action] += 1
            phase_counts[phase] += 1
            decisions.append(
                {
                    "step": event.get("step"),
                    "phase": phase,
                    "branch_id": event.get("branch_id") or "",
                    "branch_step": event.get("branch_step"),
                    "action": action,
                    "task_record_id": event.get("task_record_id") or "",
                    "args": event.get("args") or {},
                    "planner_error": event.get("planner_error") or "",
                }
            )
        elif event_type == "tool_call":
            tool_calls.append(
                {
                    key: event.get(key)
                    for key in (
                        "step",
                        "phase",
                        "branch_id",
                        "branch_step",
                        "action",
                        "args",
                        "decision_source",
                    )
                    if key in event
                }
            )
        elif event_type == "tool_result":
            result = _json_result(event)
            status = str(result.get("status") or "ok")
            tool_result_statuses[status] += 1
            if status in {"error", "failed", "unavailable", "unauthorized"}:
                error_classes[str(result.get("error_class") or "tool_result_error")] += 1
            tool_results.append(
                {
                    "step": event.get("step"),
                    "phase": event.get("phase"),
                    "branch_id": event.get("branch_id") or "",
                    "branch_step": event.get("branch_step"),
                    "action": event.get("action") or "",
                    "execution_status": event.get("execution_status") or "",
                    "retrieval_role": event.get("retrieval_role") or "",
                    "result": result,
                }
            )
        elif event_type in {"retrieval_fork_started", "retrieval_fork_completed"}:
            forks.append(
                {
                    "type": event_type,
                    "step": event.get("step"),
                    "phase": event.get("phase"),
                    "branch_width": event.get("branch_width"),
                    "max_tool_steps": event.get("max_tool_steps"),
                    "retrieval_phase": event.get("retrieval_phase"),
                    "generic_web_search_only": event.get("generic_web_search_only"),
                    "branch_count": event.get("branch_count"),
                    "evidence_rounds": event.get("evidence_rounds"),
                    "total_tool_steps": event.get("total_tool_steps"),
                    "branches": event.get("branches") or [],
                }
            )
        elif event_type in {"error", "provider_error"}:
            error_class = str(event.get("error_class") or event_type)
            error_classes[error_class] += 1
        elif event_type == "page_chunk":
            chunk_tokens.append(event.get("chunk_tokens"))
            chunk_chars.append(event.get("chunk_chars"))
        elif event_type == "page_chunk_candidate":
            candidate = event.get("candidate") or {}
            prompt_chars.append(event.get("prompt_chars"))
            candidate_output_chars.append(len(str(event.get("model_output") or "")))
            candidate_finish_reasons[str(event.get("finish_reason") or "unknown")] += 1
            candidate_retry_calls += int(event.get("retry_count") or 0)
            if candidate.get("supported"):
                supported_candidates += 1
            else:
                unsupported_candidates += 1
        elif event_type == "page_candidate_merge":
            page_data = event.get("data") or {}
            parallel = page_data.get("parallel_candidate") or {}
            chunk_windows.append(page_data.get("chunk_window_tokens"))
            evidence_row = {
                "step": event.get("step"),
                "url": event.get("url"),
                "data": page_data,
                "compact_facts": str(event.get("compact_facts") or "")[:6000],
            }
            evidence.append(evidence_row)
        elif event_type == "context_build":
            data = event.get("data") or {}
            contexts.append(data.get("context_stats") or {})
        elif event_type == "evidence_validation":
            data = event.get("data") or {}
            validations.append(
                {
                    "step": event.get("step"),
                    "validation": data.get("validation") or {},
                    "answer_alignment": data.get("answer_alignment") or {},
                }
            )
        elif event_type == "completion_judgement":
            data = event.get("data") or {}
            judgements.append(
                {
                    "step": event.get("step"),
                    "status": data.get("status", ""),
                    "missing_task_record_ids": data.get("missing_task_record_ids") or data.get("missing_point_ids") or [],
                    "reason": data.get("reason", ""),
                }
            )
        elif event_type == "final":
            finals.append(
                {
                    "status": event.get("status"),
                    "error_type": event.get("error_type") or "",
                    "content": event.get("content", ""),
                    "mode": event.get("mode", ""),
                    "action": event.get("action", ""),
                    "termination_reason": event.get("termination_reason", ""),
                    "model_output_available": event.get("model_output_available"),
                    "planner_error": event.get("planner_error") or "",
                    "round_count": event.get("round_count"),
                    "citation_refs": event.get("citation_refs") or [],
                    "validation": event.get("validation") or {},
                    "answer_alignment": event.get("answer_alignment") or {},
                    "answer_quality": event.get("answer_quality") or {},
                    "answer_requirement_validation": event.get("answer_requirement_validation") or {},
                }
            )

    evidence_statuses = collections.Counter(
        str((row.get("data") or {}).get("status") or "unknown") for row in evidence
    )
    context_token_values = [
        context.get("context_tokens")
        for context in contexts
        if isinstance(context, dict)
    ]
    module_responsibility = _module_responsibility_metrics(events)
    return {
        "event_count": len(events),
        "plans": plans,
        "strategy_selections": strategy_selections,
        "decisions": decisions,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "retrieval_forks": forks,
        "page_evidence": evidence,
        "completion_judgements": judgements,
        "contexts": contexts,
        "evidence_validations": validations,
        "finals": finals,
        "model_calls": model_calls,
        "step_limits": step_limits,
        "retrieval_ledger": {
            "events": ledger_events,
            "snapshot": ledger_snapshot,
        },
        "web_search_stages": web_search_stages,
        "web_search_chunks": web_search_chunks,
        "events": events,
        "final": finals[-1] if finals else None,
        "stats": {
            "action_counts": dict(action_counts),
            "phase_counts": dict(phase_counts),
            "tool_result_statuses": dict(tool_result_statuses),
            "error_class_counts": dict(error_classes),
            "task_record_selection": {
                "decision_count": len(decisions),
                "with_task_record_id": sum(bool(item.get("task_record_id")) for item in decisions),
            },
            "page_fetches": len(evidence),
            "web_search_page_evidence": sum(
                1 for item in web_search_stages if item.get("stage") == "page_evidence"
            ),
            "web_search_chunk_events": len(web_search_chunks),
            "model_call_count": len(model_calls),
            "step_limit_count": len(step_limits),
            "validation_event_count": len(validations),
            "event_type_counts": dict(collections.Counter(str(event.get("type") or "unknown") for event in events)),
            "ledger_event_count": len(ledger_events),
            "ledger_total_searches": int(ledger_snapshot.get("total_searches") or 0),
            "ledger_unique_queries": int(ledger_snapshot.get("unique_queries") or 0),
            "ledger_exact_repeat_count": int(ledger_snapshot.get("exact_repeat_count") or 0),
            "page_evidence_statuses": dict(evidence_statuses),
            "page_chars": _numeric_summary(
                [(row.get("data") or {}).get("page_chars") for row in evidence]
            ),
            "chunk_count": sum(int((row.get("data") or {}).get("chunk_count") or 0) for row in evidence),
            "chunk_input_tokens": _numeric_summary(chunk_tokens),
            "chunk_input_chars": _numeric_summary(chunk_chars),
            "chunk_window_tokens": _numeric_summary(chunk_windows),
            "candidate_outputs": {
                "supported": supported_candidates,
                "unsupported": unsupported_candidates,
                "output_chars": _numeric_summary(candidate_output_chars),
                "prompt_chars": _numeric_summary(prompt_chars),
                "finish_reasons": dict(candidate_finish_reasons),
                "retry_calls": candidate_retry_calls,
            },
            "context_tokens": _numeric_summary(context_token_values),
            "evidence_source_truncated_count": sum(
                int(context.get("source_truncated_count", context.get("truncated_count", 0)) or 0)
                for context in contexts
                if isinstance(context, dict)
            ),
            "final_context_truncated_count": sum(
                bool(context.get("final_context_truncated", context.get("context_truncated", False)))
                for context in contexts
                if isinstance(context, dict)
            ),
            # Backward-compatible alias; it now means final aggregate prompt
            # truncation rather than per-source truncation.
            "context_truncated_count": sum(
                bool(context.get("final_context_truncated", context.get("context_truncated", False)))
                for context in contexts
                if isinstance(context, dict)
            ),
            "completion_statuses": dict(
                collections.Counter(str(item.get("status") or "unknown") for item in judgements)
            ),
            "final_statuses": dict(collections.Counter(str(item.get("status") or "unknown") for item in finals)),
            "final_error_types": dict(
                collections.Counter(
                    str(item.get("error_type") or "none") for item in finals
                )
            ),
            "module_responsibility": module_responsibility,
        },
    }


def _aggregate_trace_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the same generic counters across a JSON acceptance suite."""
    aggregate: collections.Counter[str] = collections.Counter()
    for row in rows:
        stats = (row.get("trace") or {}).get("stats") or {}
        for key in (
            "page_fetches",
            "chunk_count",
            "evidence_source_truncated_count",
            "final_context_truncated_count",
            "context_truncated_count",
            "validation_event_count",
        ):
            aggregate[key] += int(stats.get(key) or 0)
        for namespace in (
            "tool_result_statuses",
            "error_class_counts",
            "page_evidence_statuses",
            "completion_statuses",
            "final_statuses",
            "final_error_types",
        ):
            for key, value in (stats.get(namespace) or {}).items():
                aggregate[f"{namespace}.{key}"] += int(value or 0)
    responsibility_rows = [
        ((row.get("trace") or {}).get("stats") or {}).get("module_responsibility") or {}
        for row in rows
    ]
    responsibility_passes = sum(value.get("pass") is True for value in responsibility_rows)
    responsibility_violations = sum(
        int(value.get("violation_count") or 0) for value in responsibility_rows
    )
    return {
        "case_count": len(rows),
        "returned_answer_cases": sum(row.get("delivery") == "answer" for row in rows),
        "network_error_cases": sum(row.get("runtime_error") == "network_error" for row in rows),
        "pass_cases": sum(row.get("quality") == "pass" for row in rows),
        "no_pass_cases": sum(row.get("quality") == "no-pass" for row in rows),
        "module_responsibility": {
            "schema_version": "module-responsibility-isolation.v2",
            "pass_cases": responsibility_passes,
            "pass_rate": round(responsibility_passes / len(rows), 4) if rows else 0.0,
            "violation_count": responsibility_violations,
            "rwkv_semantic_control_count": sum(
                int(value.get("rwkv_semantic_control_count") or 0)
                for value in responsibility_rows
            ),
            "prohibited_output_intervention_count": sum(
                int(value.get("prohibited_output_intervention_count") or 0)
                for value in responsibility_rows
            ),
            "controller_override_count": sum(
                int(value.get("controller_override_count") or 0)
                for value in responsibility_rows
            ),
            "gateway_override_count": sum(
                int(value.get("gateway_override_count") or 0)
                for value in responsibility_rows
            ),
            "final_output_mismatch_count": sum(
                int(value.get("final_output_mismatch_count") or 0)
                for value in responsibility_rows
            ),
            "trace_bytes": sum(int(value.get("trace_bytes") or 0) for value in responsibility_rows),
        },
        "counters": dict(aggregate),
    }


def run(
    input_path: Path,
    output_path: Path,
    case_timeout_seconds: float | None = None,
    max_tool_steps: int | None = None,
) -> dict[str, Any]:
    cases = _load_cases(input_path)
    started_at = datetime.now().isoformat(timespec="seconds")
    case_timeout = (
        get_analysis_timeout_seconds()
        if case_timeout_seconds is None
        else max(0.0, float(case_timeout_seconds))
    )
    if max_tool_steps is not None and int(max_tool_steps) < 1:
        raise ValueError("max_tool_steps must be positive")
    tool_step_override = int(max_tool_steps) if max_tool_steps is not None else None
    rows = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_checkpoint(*, finished: bool = False) -> dict[str, Any]:
        report = {
            "suite": "manual-real-web",
            "state": "ready" if finished else "running",
            "input": str(input_path),
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds") if finished else None,
            "processed_cases": len(rows),
            "total_cases": len(cases),
            "case_timeout_seconds": case_timeout,
            "max_tool_steps_override": tool_step_override,
            "cases": rows,
            "aggregate_trace_summary": _aggregate_trace_summaries(rows),
        }
        _atomic_write_json(output_path, report)
        return report

    for case in cases:
        case_id = _safe_id(case.get("case_id") or f"case_{len(rows) + 1}")
        task_id = f"JSON_ACCEPTANCE_{case_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        metadata = _runtime_metadata_for_case(
            case,
            max_tool_steps_override=tool_step_override,
        )
        query = str(case["query"])
        task_token = current_task_id.set(task_id)
        try:
            with _case_timeout(case_timeout):
                with analysis_slot(task_id):
                    answer = Orchestrator().run(query, task_id=task_id, run_metadata=metadata)
            error = ""
        except TimeoutError as exc:
            append_task_event(
                task_id,
                "case_timeout",
                phase="RUNTIME",
                error=str(exc),
                timeout_seconds=case_timeout,
            )
            answer = ""
            error = f"TimeoutError: {exc}"
        except Exception as exc:
            # evidence_review_protocol: the R53 contract refuses to enter the
            # Writer without a valid RWKV finish decision. That is a per-case
            # outcome (this case ends with no authorized answer), never a
            # batch-level crash — one case's protocol failure must not discard
            # the other cases in the same part runner.
            if classify_error(exc) not in {
                "network", "timeout", "provider", "auth", "quota",
                "evidence_review_protocol",
            }:
                raise
            answer = ""
            error = f"{type(exc).__name__}: {exc}"
        finally:
            current_task_id.reset(task_token)
        trace = _trace_summary(task_id)
        final_record = trace.get("final") if isinstance(trace.get("final"), dict) else {}
        final_answer = str(final_record.get("content") or answer or "")
        if not final_answer and str(final_record.get("status") or "") != "network_error":
            append_task_event(
                task_id,
                "final",
                status="network_error",
                content="",
                action="acceptance_runner",
                mode="runtime",
                error=error,
            )
            trace = _trace_summary(task_id)
            final_record = trace.get("final") if isinstance(trace.get("final"), dict) else {}
        answer = str(final_record.get("content") or answer or "")
        delivery = "answer" if answer else "none"
        runtime_error = "" if answer else "network_error"
        task_events = get_task_events(task_id) or []
        last_event = task_events[-1] if task_events else {}
        last_event_type = str(last_event.get("type") or "")
        failure_reason = ""
        if runtime_error:
            last_model_error = next(
                (
                    str(event.get("error"))
                    for event in reversed(task_events)
                    if event.get("type") == "model_call" and event.get("error")
                ),
                "",
            )
            failure_reason = error or last_model_error or (
                "orchestrator ended without RWKV output"
                f" (last_event_type={last_event_type or 'missing'})"
            )
        quality, quality_comparison = _offline_quality(case, answer)
        rows.append(
            {
                "case_id": case_id,
                "benchmark_id": case.get("id") or case_id,
                "task_id": task_id,
                "query": query,
                # Reference material is recorded for post-run comparison only;
                # it is deliberately never included in ``query`` or metadata.
                "reference_answer": case.get("final_answer") or case.get("reference_answer") or "",
                "gold": case.get("gold") if isinstance(case.get("gold"), dict) else None,
                "architecture": (
                    (trace.get("strategy_selections") or [{}])[-1].get("strategy")
                    or case.get("architecture")
                    or "single_loop"
                ),
                "validation_mode": "offline_reference_comparison_only",
                "max_tool_steps": metadata.get("max_tool_steps"),
                "delivery": delivery,
                "runtime_error": runtime_error,
                "quality": quality,
                "quality_comparison": quality_comparison,
                "answer": answer,
                "final_output": answer,
                "final_output_chars": len(answer),
                "error": error,
                "failure_reason": failure_reason,
                "trace": trace,
            }
        )
        write_checkpoint()

    return write_checkpoint(finished=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/evaluation/manual_real_queries.json",
        help="UTF-8 JSON input file",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/manual_real_results.json",
        help="UTF-8 JSON output file",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=None,
        help="Optional hard wall-clock limit per case; omitted means no single-case timeout.",
    )
    parser.add_argument(
        "--max-tool-steps",
        type=int,
        default=None,
        help="Optional suite-wide step budget override; takes precedence over values embedded in cases.",
    )
    args = parser.parse_args()
    if args.max_tool_steps is not None and args.max_tool_steps < 1:
        parser.error("--max-tool-steps must be positive")
    report = run(
        Path(args.input),
        Path(args.output),
        case_timeout_seconds=args.case_timeout_seconds,
        max_tool_steps=args.max_tool_steps,
    )
    # Keep stdout ASCII-safe on Windows; the full UTF-8 report is the file.
    print(json.dumps({"output": str(args.output), "case_count": len(report["cases"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
