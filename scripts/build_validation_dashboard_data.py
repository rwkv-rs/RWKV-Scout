"""Build a browser-sized projection of the validation JSON artifacts.

The full event artifact is intentionally kept as the source of truth.  This
projection preserves every event in order, but shortens very large prompt and
response fields so the browser can render the complete chain without loading
443 MB into the main thread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL = ROOT / "data/evaluation/validation_benchmark_full_events_fixed_20260730.json"
DEFAULT_COMPARISON = ROOT / "data/evaluation/validation_benchmark_comparison_fixed_20260730.json"
DEFAULT_AUDIT = ROOT / "data/evaluation/validation_record_audit_fixed_20260730.json"
DEFAULT_OUTPUT = ROOT / "frontend/public/validation-dashboard-data.json"

EXCERPT_LIMIT = 1400


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _excerpt(value: Any) -> dict[str, Any] | None:
    raw = _text(value)
    if not raw:
        return None
    return {
        "text": raw[:EXCERPT_LIMIT],
        "length": len(raw),
        "truncated": len(raw) > EXCERPT_LIMIT,
    }


def _event_projection(event: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "seq",
        "timestamp",
        "type",
        "phase",
        "step",
        "branch_id",
        "branch_step",
        "action",
        "decision_source",
        "operation",
        "status",
        "execution_status",
        "retrieval_role",
        "task_point_id",
        "url",
        "query",
        "completion_ready",
        "requires_replan",
        "missing_point_ids",
        "conflict_point_ids",
        "next_queries",
        "error_class",
        "timeout_seconds",
    )
    result = {key: event[key] for key in keep if key in event and event[key] not in (None, "", [])}
    for key in ("content", "prompt", "output", "model_output", "raw_model_output", "error"):
        excerpt = _excerpt(event.get(key))
        if excerpt:
            result[f"{key}_excerpt"] = excerpt

    for key in ("args", "result", "data", "snapshot", "validation", "answer_alignment"):
        value = event.get(key)
        if value in (None, "", []):
            continue
        serialized = _text(value)
        result[f"{key}_excerpt"] = {
            "text": serialized[:EXCERPT_LIMIT],
            "length": len(serialized),
            "truncated": len(serialized) > EXCERPT_LIMIT,
        }
    return result


def _comparison_projection(comparison: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "case_id",
        "query",
        "status",
        "answer_chars",
        "architecture",
        "validation_architecture",
        "model_calls",
        "tool_calls",
        "search_events",
        "fetch_events",
        "chunk_events",
        "context_builds",
        "evidence_validation_events",
        "evidence_verification_events",
        "verifier_statuses",
        "verifier_requires_replan",
        "replan_events",
        "replan_attempts",
        "replan_limit_events",
        "final_events",
        "synthesis_events",
        "usable_evidence_count",
        "page_evidence_statuses",
        "context_tokens",
        "event_count",
        "verifier_decisions",
    )
    return {key: comparison.get(key) for key in fields if key in comparison}


def build(full: dict[str, Any], comparison: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    comparison_by_case: dict[str, dict[str, Any]] = {}
    for suite in comparison.get("comparisons") or []:
        for row in suite.get("case_comparisons") or []:
            case_id = str(row.get("case_id") or "")
            if not case_id:
                continue
            comparison_by_case.setdefault(case_id, {})
            for architecture in ("engineering", "rwkv_verifier"):
                if isinstance(row.get(architecture), dict):
                    comparison_by_case[case_id][architecture] = _comparison_projection(row[architecture])
            comparison_by_case[case_id]["answer_changed"] = bool(row.get("answer_changed"))
            comparison_by_case[case_id]["verifier_added"] = bool(row.get("verifier_added"))

    audit_by_case: dict[str, dict[str, Any]] = {}
    for batch in audit.get("batches") or []:
        for row in batch.get("cases") or []:
            case_id = str(row.get("case_id") or "")
            if case_id:
                audit_by_case[case_id] = {
                    key: row.get(key)
                    for key in (
                        "case_id",
                        "status",
                        "event_count",
                        "model_call_count",
                        "tool_call_count",
                        "tool_result_count",
                        "page_fetch_count",
                        "chunk_count",
                        "duplicate_block_count",
                        "replan_count",
                        "verifier_event_count",
                        "raw_verifier_output_event_count",
                        "raw_verifier_output_in_final_prompt_count",
                        "missing_required_event_types",
                        "model_shape_errors",
                        "event_shape_errors",
                        "verifier_boundary_errors",
                        "ok",
                    )
                    if key in row
                }

    batches = []
    for batch in full.get("batches") or []:
        cases = []
        for case in batch.get("cases") or []:
            case_id = str(case.get("case_id") or "")
            trace = case.get("trace") if isinstance(case.get("trace"), dict) else {}
            events = trace.get("events") if isinstance(trace.get("events"), list) else []
            cases.append(
                {
                    "case_id": case_id,
                    "task_id": case.get("task_id"),
                    "query": case.get("query") or "",
                    "status": case.get("status") or "unknown",
                    "architecture": case.get("architecture") or batch.get("architecture") or "",
                    "validation_architecture": case.get("validation_architecture") or "",
                    "answer": case.get("answer") or "",
                    "final_output_chars": case.get("final_output_chars") or len(str(case.get("answer") or "")),
                    "error": case.get("error") or "",
                    "trace_stats": trace.get("stats") or {},
                    "events": [_event_projection(event) for event in events if isinstance(event, dict)],
                    "comparison": comparison_by_case.get(case_id, {}),
                    "audit": audit_by_case.get(case_id, {}),
                }
            )
        batches.append(
            {
                "batch_id": batch.get("batch_id") or "",
                "architecture": batch.get("architecture") or "",
                "suite": batch.get("suite") or "",
                "status": batch.get("status") or "",
                "completed_cases": batch.get("completed_cases") or 0,
                "total_cases": batch.get("total_cases") or len(cases),
                "failed_cases": batch.get("failed_cases") or 0,
                "aggregate_trace_summary": batch.get("aggregate_trace_summary") or {},
                "cases": cases,
            }
        )

    return {
        "schema_version": "validation_dashboard_projection.v1",
        "generated_at": full.get("generated_at"),
        "source_files": {
            "full_events": "data/evaluation/validation_benchmark_full_events_fixed_20260730.json",
            "comparison": "data/evaluation/validation_benchmark_comparison_fixed_20260730.json",
            "audit": "data/evaluation/validation_record_audit_fixed_20260730.json",
        },
        "full_events_meta": {
            key: full.get(key)
            for key in ("complete", "batch_count", "case_count", "completed_cases", "failed_cases", "event_count")
        },
        "comparison_meta": {
            "architectures": comparison.get("architectures") or {},
            "three_round_replanning": comparison.get("three_round_replanning"),
            "accuracy_scoring": comparison.get("accuracy_scoring") or {},
            "comparisons": [
                {
                    "suite": suite.get("suite"),
                    "case_count": suite.get("case_count"),
                    "case_comparisons": [
                        {
                            "case_id": row.get("case_id"),
                            "query": row.get("query"),
                            "answer_changed": row.get("answer_changed"),
                            "engineering": _comparison_projection(row.get("engineering") or {}),
                            "rwkv_verifier": _comparison_projection(row.get("rwkv_verifier") or {}),
                        }
                        for row in suite.get("case_comparisons") or []
                    ],
                }
                for suite in comparison.get("comparisons") or []
            ],
        },
        "audit_meta": {
            "complete": audit.get("complete"),
            "batch_count": audit.get("batch_count"),
            "accuracy_scoring": audit.get("accuracy_scoring") or {},
        },
        "batches": batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(_load(args.full), _load(args.comparison), _load(args.audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "batches": len(payload["batches"]), "cases": sum(len(b["cases"]) for b in payload["batches"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
