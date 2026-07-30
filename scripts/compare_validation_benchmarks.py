"""Compare the two validation architectures without inventing an accuracy score.

The acceptance runners keep the complete event stream in each batch output.
This report is deliberately a measurement layer: it compares execution,
evidence-boundary and verifier behavior, while leaving factual correctness to
gold answers or human review when a suite does not provide them.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BATCHES = (
    ("engineering_validator", "100"),
    ("engineering_validator", "50"),
    ("engineering_validator", "date"),
    ("rwkv_verifier", "100"),
    ("rwkv_verifier", "50"),
    ("rwkv_verifier", "date"),
)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _stats(values: list[Any]) -> dict[str, Any]:
    numbers = [_number(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return {"count": 0, "min": 0, "max": 0, "avg": 0}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "avg": round(statistics.fmean(numbers), 2),
    }


def _event_counts(events: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(event.get("type") or "unknown") for event in events)


def _case_measurement(case: dict[str, Any]) -> dict[str, Any]:
    trace = case.get("trace") if isinstance(case.get("trace"), dict) else {}
    events = [event for event in trace.get("events") or [] if isinstance(event, dict)]
    counts = _event_counts(events)
    verification_events = [
        event for event in events if event.get("type") == "evidence_verification"
    ]
    replans = [
        event
        for event in events
        if event.get("type") in {"task_replan_attempt", "task_replan"}
    ]
    model_events = [event for event in events if event.get("type") == "model_call"]
    search_events = [
        event
        for event in events
        if event.get("type") in {"retrieval_search", "web_search", "search"}
        or event.get("action") == "web_search"
    ]
    fetch_events = [
        event
        for event in events
        if event.get("type") in {"retrieval_fetch", "fetch_web_url", "page_fetch"}
        or event.get("action") in {"fetch_web_url", "fetch_url"}
    ]
    trace_stats = trace.get("stats") if isinstance(trace.get("stats"), dict) else {}
    verifier_statuses = Counter(
        str(event.get("status") or "unknown") for event in verification_events
    )
    verifier_replans = sum(bool(event.get("requires_replan")) for event in verification_events)
    limit_events = counts.get("task_replan_limit_reached", 0)
    return {
        "case_id": case.get("case_id"),
        "query": case.get("query"),
        "status": case.get("status"),
        "answer_chars": int(case.get("final_output_chars") or len(str(case.get("answer") or ""))),
        "architecture": case.get("architecture"),
        "validation_architecture": case.get("validation_architecture"),
        "model_calls": len(model_events),
        "tool_calls": counts.get("tool_call", 0) + counts.get("model_tool_decision", 0),
        "search_events": len(search_events),
        "fetch_events": len(fetch_events),
        "chunk_events": counts.get("page_chunk", 0) + counts.get("chunk", 0),
        "context_builds": counts.get("context_build", 0),
        "evidence_validation_events": counts.get("evidence_validation", 0),
        "evidence_verification_events": len(verification_events),
        "verifier_statuses": dict(verifier_statuses),
        "verifier_requires_replan": verifier_replans,
        "replan_events": len(replans),
        "replan_attempts": counts.get("task_replan_attempt", 0),
        "replan_limit_events": limit_events,
        "global_step_limit_events": counts.get("step_limit_reached", 0),
        "final_events": counts.get("final", 0),
        "synthesis_events": counts.get("synthesis", 0),
        "evidence_rounds": trace_stats.get("evidence_rounds", trace.get("evidence_rounds", 0)),
        "context_tokens": trace_stats.get("context_tokens", trace.get("context_tokens", 0)),
        "usable_evidence_count": trace_stats.get("usable_evidence_count", 0),
        "event_count": len(events),
        "event_type_counts": dict(counts),
        "verifier_decisions": [
            {
                "step": event.get("step"),
                "status": event.get("status"),
                "completion_ready": event.get("completion_ready"),
                "requires_replan": event.get("requires_replan"),
                "missing_point_ids": event.get("missing_point_ids") or [],
                "conflict_point_ids": event.get("conflict_point_ids") or [],
                "next_queries": event.get("next_queries") or [],
            }
            for event in verification_events
        ],
    }


def _batch_measurement(path: Path, architecture: str, suite: str) -> dict[str, Any]:
    payload = _load(path)
    if payload is None:
        return {
            "architecture": architecture,
            "suite": suite,
            "status": "pending",
            "output": str(path),
            "case_count": 0,
            "cases": [],
        }
    cases = [case for case in payload.get("cases") or [] if isinstance(case, dict)]
    measurements = [_case_measurement(case) for case in cases]
    return {
        "architecture": architecture,
        "suite": suite,
        "status": payload.get("status", "unknown"),
        "output": str(path),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "case_count": len(measurements),
        "completed_cases": payload.get("completed_cases", 0),
        "total_cases": payload.get("total_cases", len(measurements)),
        "cases": measurements,
        "aggregate_trace_summary": payload.get("aggregate_trace_summary") or {},
    }


def _compare_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    by_id = {str(row.get("case_id")): row for row in left.get("cases") or []}
    right_by_id = {str(row.get("case_id")): row for row in right.get("cases") or []}
    rows = []
    for case_id in sorted(set(by_id) | set(right_by_id)):
        a = by_id.get(case_id, {})
        b = right_by_id.get(case_id, {})
        rows.append(
            {
                "case_id": case_id,
                "query": a.get("query") or b.get("query"),
                "engineering": a,
                "rwkv_verifier": b,
                "answer_changed": a.get("answer_chars") != b.get("answer_chars"),
                "verifier_added": bool(b.get("evidence_verification_events")),
                "replan_delta": int(b.get("replan_attempts", 0)) - int(a.get("replan_attempts", 0)),
            }
        )
    return {
        "suite": left.get("suite") or right.get("suite"),
        "case_count": len(rows),
        "case_comparisons": rows,
        "quality_note": (
            "This is an execution/evidence-boundary comparison, not a factual accuracy score. "
            "No gold answer is assumed unless the input explicitly supplies one."
        ),
    }


def _batches_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = manifest_path.resolve().parents[2]
    batches = []
    for spec in manifest.get("batches") or []:
        if not isinstance(spec, dict):
            continue
        raw_output = Path(str(spec.get("output") or ""))
        output = raw_output if raw_output.is_absolute() else repo_root / raw_output
        batches.append(
            _batch_measurement(
                output,
                str(spec.get("architecture") or "unknown"),
                str(spec.get("suite") or "unknown"),
            )
        )
    return batches


def build_report(output_dir: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    batches = (
        _batches_from_manifest(manifest_path.resolve())
        if manifest_path is not None
        else [
            _batch_measurement(
                output_dir / f"validation_{architecture}_{suite}_20260730.json",
                architecture,
                suite,
            )
            for architecture, suite in DEFAULT_BATCHES
        ]
    )
    engineering = {row["suite"]: row for row in batches if row["architecture"] == "engineering_validator"}
    verifier = {row["suite"]: row for row in batches if row["architecture"] == "rwkv_verifier"}
    comparisons = [
        _compare_pair(engineering[suite], verifier[suite])
        for suite in ("100", "50", "date")
    ]
    return {
        "schema_version": "validation_benchmark_comparison.v1",
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path.resolve()) if manifest_path is not None else None,
        "architectures": {
            "engineering_validator": {
                "contract": "mechanical evidence boundary and existing controller rules",
                "model_may_output": "ordinary planner/final text only; no verifier call",
                "model_must_not_output": "controller-authored factual evidence",
            },
            "rwkv_verifier": {
                "contract": "control-only JSON: point status, evidence refs, missing fields, conflicts, next queries",
                "model_may_output": "control decisions and query strings",
                "model_must_not_output": "answer text or new factual claims",
            },
        },
        "three_round_replanning": {
            "enabled": True,
            "max_replan_attempts_default": 3,
            "counts_toward_global_tool_steps": False,
            "trigger": "missing or conflicting evidence, or repeated unusable evidence",
            "after_limit": "final synthesis is forced and must state evidence limits",
        },
        "batches": batches,
        "comparisons": comparisons,
        "accuracy_scoring": {
            "status": "not_automatically_scored",
            "reason": "The supplied fixed result files are prior outputs, not authoritative gold answers.",
            "required_for_accuracy": ["gold answers", "source-backed human review", "or a separately audited judge"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/evaluation")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Read architecture/suite output paths from a comparison manifest.",
    )
    parser.add_argument("--output", default="data/evaluation/validation_benchmark_comparison_20260730.json")
    args = parser.parse_args()
    report = build_report(
        Path(args.output_dir),
        Path(args.manifest) if args.manifest else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(output), "batches": len(report["batches"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
