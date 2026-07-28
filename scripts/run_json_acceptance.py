"""Run real model-owned retrieval cases from a UTF-8 JSON file.

The query never comes from a shell literal.  This keeps Chinese input and the
resulting trace stable across PowerShell, WSL and CI environments.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Make direct execution (`python scripts/run_json_acceptance.py`) use the
# repository root just like module execution does.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.orchestrator import Orchestrator
from utils.token_tracker import current_task_id
from utils.task_events import get_task_events


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "case"))
    return cleaned.strip("._") or "case"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("input JSON must contain a non-empty 'cases' list")
    normalized = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or not str(case.get("query") or "").strip():
            raise ValueError(f"case {index} must contain a non-empty query")
        normalized.append(case)
    return normalized


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


def _trace_summary(task_id: str) -> dict[str, Any]:
    """Summarize execution boundaries without judging task-specific content."""
    events = get_task_events(task_id)
    decisions = []
    evidence = []
    plans = []
    judgements = []
    contexts = []
    finals = []
    model_calls = []
    step_limits = []
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
        elif event_type == "task_plan":
            plan = event.get("data") or {}
            plans.append(
                {
                    "schema_version": plan.get("schema_version"),
                    "status": plan.get("status", "ok"),
                    "point_count": len(plan.get("atomic_points") or []) if isinstance(plan, dict) else 0,
                    "point_ids": [
                        str(point.get("id") or "")
                        for point in (plan.get("atomic_points") or [])
                        if isinstance(point, dict)
                    ],
                }
            )
        elif event_type == "task_replan":
            plan = event.get("data") or {}
            plans.append(
                {
                    "schema_version": plan.get("schema_version") if isinstance(plan, dict) else None,
                    "status": plan.get("status", "ok") if isinstance(plan, dict) else "unknown",
                    "point_count": len(plan.get("atomic_points") or []) if isinstance(plan, dict) else 0,
                    "point_ids": [
                        str(point.get("id") or "")
                        for point in (plan.get("atomic_points") or [])
                        if isinstance(point, dict)
                    ],
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
                    "action": action,
                    "task_point_id": event.get("task_point_id") or "",
                    "args": event.get("args") or {},
                    "planner_error": event.get("planner_error") or "",
                }
            )
        elif event_type == "tool_result":
            result = _json_result(event)
            status = str(result.get("status") or "ok")
            tool_result_statuses[status] += 1
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
        elif event_type == "completion_judgement":
            data = event.get("data") or {}
            judgements.append(
                {
                    "step": event.get("step"),
                    "status": data.get("status", ""),
                    "missing_point_ids": data.get("missing_point_ids") or [],
                    "reason": data.get("reason", ""),
                }
            )
        elif event_type == "final":
            finals.append(
                {
                    "status": event.get("status"),
                    "content": event.get("content", ""),
                    "mode": event.get("mode", ""),
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
    return {
        "event_count": len(events),
        "plans": plans,
        "decisions": decisions,
        "page_evidence": evidence,
        "completion_judgements": judgements,
        "contexts": contexts,
        "finals": finals,
        "model_calls": model_calls,
        "step_limits": step_limits,
        "final": finals[-1] if finals else None,
        "stats": {
            "action_counts": dict(action_counts),
            "phase_counts": dict(phase_counts),
            "tool_result_statuses": dict(tool_result_statuses),
            "error_class_counts": dict(error_classes),
            "task_point_selection": {
                "decision_count": len(decisions),
                "with_task_point_id": sum(bool(item.get("task_point_id")) for item in decisions),
            },
            "page_fetches": len(evidence),
            "model_call_count": len(model_calls),
            "step_limit_count": len(step_limits),
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
            "context_truncated_count": sum(
                int(context.get("truncated_count") or 0)
                for context in contexts
                if isinstance(context, dict)
            ),
            "completion_statuses": dict(
                collections.Counter(str(item.get("status") or "unknown") for item in judgements)
            ),
            "final_statuses": dict(collections.Counter(str(item.get("status") or "unknown") for item in finals)),
        },
    }


def _aggregate_trace_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the same generic counters across a JSON acceptance suite."""
    aggregate: collections.Counter[str] = collections.Counter()
    for row in rows:
        stats = (row.get("trace") or {}).get("stats") or {}
        for key in ("page_fetches", "chunk_count", "context_truncated_count"):
            aggregate[key] += int(stats.get(key) or 0)
        for namespace in ("tool_result_statuses", "error_class_counts", "page_evidence_statuses", "completion_statuses", "final_statuses"):
            for key, value in (stats.get(namespace) or {}).items():
                aggregate[f"{namespace}.{key}"] += int(value or 0)
    return {
        "case_count": len(rows),
        "completed_cases": sum(row.get("status") == "completed" for row in rows),
        "failed_cases": sum(row.get("status") != "completed" for row in rows),
        "counters": dict(aggregate),
    }


def run(input_path: Path, output_path: Path) -> dict[str, Any]:
    cases = _load_cases(input_path)
    started_at = datetime.now().isoformat(timespec="seconds")
    rows = []
    for case in cases:
        case_id = _safe_id(case.get("case_id") or f"case_{len(rows) + 1}")
        task_id = f"JSON_ACCEPTANCE_{case_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        metadata = {}
        if case.get("max_tool_steps") is not None:
            metadata["max_tool_steps"] = int(case["max_tool_steps"])
        query = str(case["query"])
        task_token = current_task_id.set(task_id)
        try:
            answer = Orchestrator().run(query, task_id=task_id, run_metadata=metadata)
            error = ""
        except Exception as exc:
            answer = ""
            error = f"{type(exc).__name__}: {exc}"
        finally:
            current_task_id.reset(task_token)
        trace = _trace_summary(task_id)
        final_status = str((trace.get("final") or {}).get("status") or "")
        status = "completed" if final_status.startswith("completed") else "failed"
        rows.append(
            {
                "case_id": case_id,
                "task_id": task_id,
                "query": query,
                "status": status,
                "answer": answer,
                "error": error,
                "trace": trace,
            }
        )

    report = {
        "suite": "manual-real-web",
        "input": str(input_path),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "cases": rows,
        "aggregate_trace_summary": _aggregate_trace_summaries(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


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
    args = parser.parse_args()
    report = run(Path(args.input), Path(args.output))
    # Keep stdout ASCII-safe on Windows; the full UTF-8 report is the file.
    print(json.dumps({"output": str(args.output), "case_count": len(report["cases"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
