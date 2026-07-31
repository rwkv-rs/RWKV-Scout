"""Audit validation runs without judging factual correctness.

The audit verifies that a comparison artifact contains the complete observable
execution chain and that the optional RWKV verifier stayed in its control-only
boundary.  It deliberately does not decide whether a retrieved fact is true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


CONTROL_FIELDS = {
    "step",
    "status",
    "completion_ready",
    "requires_replan",
    "missing_point_ids",
    "conflict_point_ids",
    "next_queries",
    "error",
}
FORBIDDEN_CONTROL_FIELDS = {"reason", "model_confidence", "answer", "facts", "claims"}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_expected_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.suffix.casefold() == ".jsonl":
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                values.append(str(value.get("case_id") or value.get("id") or ""))
        return [value for value in values if value]
    value = _read_json(path)
    rows = value.get("cases") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        rows = value.get("tasks") if isinstance(value, dict) else []
    return [
        str(row.get("case_id") or row.get("id") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("case_id") or row.get("id") or "")
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _events(case: dict[str, Any]) -> list[dict[str, Any]]:
    trace = case.get("trace") if isinstance(case.get("trace"), dict) else {}
    return [event for event in trace.get("events") or [] if isinstance(event, dict)]


def _event_types(events: list[dict[str, Any]]) -> set[str]:
    return {str(event.get("type") or "") for event in events}


def _case_audit(case: dict[str, Any], architecture: str) -> dict[str, Any]:
    events = _events(case)
    types = _event_types(events)
    model_events = [event for event in events if event.get("type") == "model_call"]
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    tool_results = [event for event in events if event.get("type") == "tool_result"]
    verifier_events = [event for event in events if event.get("type") == "evidence_verification"]
    trace = case.get("trace") if isinstance(case.get("trace"), dict) else {}
    normalized = [
        item for item in trace.get("evidence_verifications") or [] if isinstance(item, dict)
    ]

    model_shape_errors = []
    for index, event in enumerate(model_events):
        if not isinstance(event.get("prompt"), str):
            model_shape_errors.append(f"model_call[{index}] missing prompt")
        if not any(key in event for key in ("output", "model_output", "content", "error")):
            model_shape_errors.append(f"model_call[{index}] missing output/error")

    event_shape_errors = []
    if not tool_calls and "tool_call" in types:
        event_shape_errors.append("tool_call type has no materialized calls")
    for index, event in enumerate(tool_calls):
        if not isinstance(event.get("args"), dict):
            event_shape_errors.append(f"tool_call[{index}] missing args")
    for index, event in enumerate(tool_results):
        if not any(key in event for key in ("result", "status", "error")):
            event_shape_errors.append(f"tool_result[{index}] missing result/status/error")

    verifier_boundary_errors = []
    raw_verifier_output_count = 0
    for index, event in enumerate(verifier_events):
        if not isinstance(event.get("prompt"), str):
            verifier_boundary_errors.append(f"verifier_event[{index}] missing prompt")
        if any(key in event for key in ("model_output", "raw_model_output", "output")):
            raw_verifier_output_count += 1
        for key in event:
            if key in FORBIDDEN_CONTROL_FIELDS:
                verifier_boundary_errors.append(
                    f"verifier_event[{index}] exposes forbidden field {key}"
                )
        point_rows = event.get("points")
        if isinstance(point_rows, list):
            for point_index, point in enumerate(point_rows):
                if isinstance(point, dict):
                    invalid = sorted(set(point) - {"id", "status", "evidence", "missing", "next_queries"})
                    if invalid:
                        verifier_boundary_errors.append(
                            f"verifier_event[{index}].points[{point_index}] invalid fields: {invalid}"
                        )
    for index, item in enumerate(normalized):
        invalid = sorted(set(item) - CONTROL_FIELDS)
        forbidden = sorted(set(item) & FORBIDDEN_CONTROL_FIELDS)
        if invalid:
            verifier_boundary_errors.append(f"normalized_verifier[{index}] invalid fields: {invalid}")
        if forbidden:
            verifier_boundary_errors.append(f"normalized_verifier[{index}] forbidden fields: {forbidden}")

    raw_outputs = {
        str(event.get("model_output") or event.get("raw_model_output") or event.get("output") or "")
        for event in verifier_events
        if str(event.get("model_output") or event.get("raw_model_output") or event.get("output") or "")
    }
    final_prompts = [
        str(event.get("prompt") or "")
        for event in events
        if event.get("type") in {"synthesis", "final"}
    ]
    raw_leaks = sum(bool(raw and any(raw in prompt for prompt in final_prompts)) for raw in raw_outputs)

    required_types = {"user_input", "run_started", "model_call", "final"}
    missing_types = sorted(required_types - types)
    return {
        "case_id": case.get("case_id"),
        "status": case.get("status"),
        "event_count": len(events),
        "event_types": sorted(types),
        "missing_required_event_types": missing_types,
        "model_call_count": len(model_events),
        "tool_call_count": len(tool_calls),
        "tool_result_count": len(tool_results),
        "page_fetch_count": sum(event.get("type") == "page_fetch" for event in events),
        "chunk_count": sum(event.get("type") == "web_search_chunk" for event in events),
        "duplicate_block_count": sum(event.get("type") == "retrieval_duplicate_blocked" for event in events),
        "replan_count": sum(event.get("type") in {"task_replan", "task_replan_attempt"} for event in events),
        "verifier_event_count": len(verifier_events),
        "raw_verifier_output_event_count": raw_verifier_output_count,
        "raw_verifier_output_in_final_prompt_count": raw_leaks,
        "model_shape_errors": model_shape_errors,
        "event_shape_errors": event_shape_errors,
        "verifier_boundary_errors": verifier_boundary_errors,
        "ok": not (
            missing_types
            or model_shape_errors
            or event_shape_errors
            or verifier_boundary_errors
            or raw_leaks
            or (architecture == "rwkv_verifier" and verifier_events and raw_verifier_output_count == 0)
        ),
    }


def audit(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    repo_root = manifest_path.parents[2]
    batches = []
    all_ok = True
    for spec in manifest.get("batches") or []:
        if not isinstance(spec, dict):
            continue
        raw_output = Path(str(spec.get("output") or ""))
        output = raw_output if raw_output.is_absolute() else repo_root / raw_output
        raw_input = Path(str(spec.get("input") or ""))
        input_path = raw_input if raw_input.is_absolute() else repo_root / raw_input
        batch = {
            "batch_id": spec.get("batch_id") or "",
            "architecture": spec.get("architecture") or "",
            "suite": spec.get("suite") or "",
            "input": spec.get("input") or "",
            "output": spec.get("output") or "",
            "input_exists": input_path.is_file(),
            "input_sha256": _sha256(input_path) if input_path.is_file() else "",
            "output_exists": output.is_file(),
            "output_sha256": _sha256(output) if output.is_file() else "",
            "expected_case_count": len(_read_expected_ids(input_path)),
            "status": "pending",
            "case_count": 0,
            "completed_cases": 0,
            "failed_cases": 0,
            "duplicate_case_ids": [],
            "missing_case_ids": [],
            "unexpected_case_ids": [],
            "cases": [],
        }
        if output.is_file():
            payload = _read_json(output)
            cases = [case for case in payload.get("cases") or [] if isinstance(case, dict)]
            actual_ids = [str(case.get("case_id") or "") for case in cases]
            expected_ids = _read_expected_ids(input_path)
            batch.update(
                {
                    "status": payload.get("status") or "unknown",
                    "case_count": len(cases),
                    "completed_cases": sum(case.get("status") == "completed" for case in cases),
                    "failed_cases": sum(case.get("status") != "completed" for case in cases),
                    "duplicate_case_ids": sorted({case_id for case_id in actual_ids if actual_ids.count(case_id) > 1}),
                    "missing_case_ids": sorted(set(expected_ids) - set(actual_ids)),
                    "unexpected_case_ids": sorted(set(actual_ids) - set(expected_ids)),
                    "cases": [_case_audit(case, str(spec.get("architecture") or "")) for case in cases],
                }
            )
        batch["ok"] = bool(
            batch["input_exists"]
            and batch["status"] == "completed"
            and not batch["missing_case_ids"]
            and not batch["unexpected_case_ids"]
            and not batch["duplicate_case_ids"]
            and all(case["ok"] for case in batch["cases"])
        )
        all_ok = all_ok and batch["ok"]
        batches.append(batch)

    result = {
        "schema_version": "validation_record_audit.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "complete": all_ok and len(batches) == 6,
        "batch_count": len(batches),
        "batches": batches,
        "accuracy_scoring": {
            "status": "not_automatically_scored",
            "reason": "Execution completeness and protocol boundaries are not factual gold-answer scoring.",
        },
    }
    output_path = output_path if output_path.is_absolute() else repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(Path(args.manifest), Path(args.output))
    print(json.dumps({"complete": result["complete"], "batch_count": result["batch_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
