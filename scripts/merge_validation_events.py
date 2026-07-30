"""Merge validation batch JSON files without dropping per-case event traces.

The resulting artifact is an inspection index plus the complete case records.
Each case keeps ``trace.events`` unchanged, including model prompts/outputs,
tool arguments/results, page evidence, chunks, context builds, validation,
replanning, and final synthesis events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _enrich_case(case: Any) -> Any:
    """Add missing failure metadata without changing the raw event trace."""
    if not isinstance(case, dict) or case.get("status") == "completed":
        return case
    if case.get("failure_reason"):
        return case
    trace = case.get("trace") if isinstance(case.get("trace"), dict) else {}
    events = [event for event in trace.get("events") or [] if isinstance(event, dict)]
    final = trace.get("final") if isinstance(trace.get("final"), dict) else {}
    final_status = str(case.get("final_status") or final.get("status") or "missing")
    last_event_type = str(events[-1].get("type") or "missing") if events else "missing"
    enriched = dict(case)
    enriched["failure_reason"] = str(case.get("error") or (
        "orchestrator ended without a completed final event"
        f" (final_status={final_status}, last_event_type={last_event_type})"
    ))
    enriched["failure_reason_derived"] = True
    return enriched


def merge(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    repo_root = manifest_path.parents[2]
    batches: list[dict[str, Any]] = []
    pending: list[str] = []
    total_cases = 0
    completed_cases = 0
    failed_cases = 0
    total_events = 0

    for spec in manifest.get("batches") or []:
        if not isinstance(spec, dict):
            continue
        raw_output = Path(str(spec.get("output") or ""))
        output = raw_output if raw_output.is_absolute() else repo_root / raw_output
        record: dict[str, Any] = {
            "batch_id": spec.get("batch_id") or "",
            "architecture": spec.get("architecture") or "",
            "suite": spec.get("suite") or "",
            "input": spec.get("input") or "",
            "output": spec.get("output") or "",
            "source_exists": output.is_file(),
            "source_sha256": _sha256(output) if output.is_file() else "",
            "status": "pending",
            "completed_cases": 0,
            "total_cases": 0,
            "aggregate_trace_summary": {},
            "cases": [],
        }
        if not output.is_file():
            pending.append(str(record["batch_id"]))
            batches.append(record)
            continue
        source = _read_json(output)
        raw_cases = source.get("cases") if isinstance(source.get("cases"), list) else []
        cases = [_enrich_case(case) for case in raw_cases]
        record.update(
            {
                "status": source.get("status") or spec.get("status") or "unknown",
                "started_at": source.get("started_at"),
                "finished_at": source.get("finished_at"),
                "completed_cases": source.get("completed_cases", len(cases)),
                "total_cases": source.get("total_cases", len(cases)),
                "case_timeout_seconds": source.get("case_timeout_seconds"),
                "aggregate_trace_summary": source.get("aggregate_trace_summary") or {},
                "cases": cases,
            }
        )
        total_cases += len(cases)
        completed_cases += sum(case.get("status") == "completed" for case in cases if isinstance(case, dict))
        failed_cases += sum(case.get("status") != "completed" for case in cases if isinstance(case, dict))
        total_events += sum(
            int(((case.get("trace") or {}).get("event_count") or 0))
            for case in cases
            if isinstance(case, dict)
        )
        batches.append(record)

    complete = not pending and all(batch.get("status") == "completed" for batch in batches)
    result = {
        "schema_version": "validation_benchmark_full_events.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "manifest_status": manifest.get("status"),
        "complete": complete,
        "pending_batches": pending,
        "batch_count": len(batches),
        "case_count": total_cases,
        "completed_cases": completed_cases,
        "failed_cases": failed_cases,
        "event_count": total_events,
        "accuracy_scoring": {
            "status": "not_automatically_scored",
            "reason": "The suites do not provide authoritative gold answers; this artifact records execution and evidence boundaries only.",
        },
        "batches": batches,
    }
    output_path = output_path if output_path.is_absolute() else repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/evaluation/validation_benchmark_queue_20260730.json")
    parser.add_argument("--output", default="data/evaluation/validation_benchmark_full_events_20260730.json")
    args = parser.parse_args()
    result = merge(Path(args.manifest), Path(args.output))
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("complete", "pending_batches", "batch_count", "case_count", "completed_cases", "failed_cases", "event_count")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
