"""Export final answers from a concurrent evaluation run for later review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BATCHES = {
    "rwkv_search_100": 100,
    "rwkv_search_50": 50,
    "date_retrieval": 12,
    "url_summary_direct": 10,
}


def _read_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(path.glob("*.result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows.extend(item for item in payload.get("cases", []) if isinstance(item, dict))
    return rows


def _citation_urls(trace: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for ref in (trace.get("final", {}) or {}).get("citation_refs", []) or []:
        if not isinstance(ref, dict):
            continue
        url = str(ref.get("url") or "").strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _snapshot_case(batch: str, case: dict[str, Any]) -> dict[str, Any]:
    trace = case.get("trace", {}) or {}
    final = trace.get("final", {}) or {}
    context = (trace.get("contexts", []) or [{}])[-1]
    stats = trace.get("stats", {}) or {}
    answer = case.get("answer")
    if answer is None:
        answer = case.get("final_output")
    if answer is None:
        answer = final.get("content", "")
    return {
        "batch": batch,
        "case_id": case.get("case_id", ""),
        "task_id": case.get("task_id", ""),
        "query": case.get("query", ""),
        "status": case.get("status", ""),
        "final_answer": answer,
        "answer_chars": len(str(answer or "")),
        "citation_urls": _citation_urls(trace),
        "citation_refs": final.get("citation_refs", []),
        "evidence": {
            "usable_evidence_count": context.get(
                "final_usable_evidence_count",
                context.get("usable_evidence_count", 0),
            ),
            "selected_evidence_count": context.get("final_selected_evidence_count", 0),
            "page_evidence_statuses": stats.get("page_evidence_statuses", {}),
            "chunk_count": stats.get("chunk_count", 0),
            "page_fetches": stats.get("page_fetches", 0),
        },
        "context": {
            "final_context_tokens": context.get(
                "final_context_tokens", context.get("context_tokens", 0)
            ),
            "final_context_chars": context.get("final_context_chars", 0),
            "context_truncated": context.get(
                "final_context_truncated", context.get("context_truncated", False)
            ),
        },
        "execution": {
            "model_call_count": stats.get("model_call_count", 0),
            "termination_reason": final.get("termination_reason")
            or context.get("termination_reason", ""),
            "failure_reason": case.get("failure_reason", ""),
            "error": case.get("error", ""),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    batches: dict[str, dict[str, Any]] = {}
    for batch, expected in BATCHES.items():
        cases = _read_cases(args.run_dir / batch)
        rows.extend(_snapshot_case(batch, case) for case in cases)
        batches[batch] = {
            "expected_cases": expected,
            "case_count": len(cases),
            "completed_cases": sum(case.get("status") == "completed" for case in cases),
            "failed_cases": sum(case.get("status") == "failed" for case in cases),
            "complete": len(cases) == expected,
        }

    duplicate_ids = [
        case_id
        for case_id in {row["case_id"] for row in rows}
        if sum(row["case_id"] == case_id for row in rows) > 1
    ]
    complete = all(batch["complete"] for batch in batches.values()) and not duplicate_ids
    if not complete and not args.allow_partial:
        raise SystemExit(
            f"run is incomplete; use --allow-partial to export a partial snapshot: {batches}"
        )

    payload = {
        "schema_version": "answer_snapshot.v1",
        "source_run_dir": str(args.run_dir),
        "complete": complete,
        "expected_cases": sum(BATCHES.values()),
        "case_count": len(rows),
        "duplicate_case_ids": duplicate_ids,
        "batches": batches,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "complete": complete, "case_count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
