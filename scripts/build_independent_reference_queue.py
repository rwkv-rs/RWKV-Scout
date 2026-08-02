"""Build an isolated reference-answer review queue for a completed run.

The queue deliberately contains no model answer or retrieved evidence. It is
the input for independent source research, so a local run can never become its
own gold label by accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        yield from (item for item in value if isinstance(item, dict))
        return
    if not isinstance(value, dict):
        return
    for key in ("cases", "tasks", "items", "questions"):
        child = value.get(key)
        if isinstance(child, list):
            yield from (item for item in child if isinstance(item, dict))
            return


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    return list(_records(json.loads(path.read_text(encoding="utf-8"))))


def _by_query(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("query") or row.get("question") or "").strip(): row
        for row in rows
        if str(row.get("query") or row.get("question") or "").strip()
    }


def _by_case_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case_id") or row.get("id") or "").strip(): row
        for row in rows
        if str(row.get("case_id") or row.get("id") or "").strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--rwkv-search-root", type=Path, required=True)
    parser.add_argument("--questions-50", type=Path, required=True)
    parser.add_argument("--date-tasks", type=Path, required=True)
    parser.add_argument("--url-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    cases = snapshot.get("cases", []) if isinstance(snapshot, dict) else []
    source_specs = _by_query(
        _read_jsonl(args.rwkv_search_root / "bench" / "realtime_web_retrieval_dev_v2.jsonl")
    )
    spec_by_case = {
        "rwkv_search_50": _by_case_id(_load_records(args.questions_50)),
        "date_retrieval": _by_case_id(_load_records(args.date_tasks)),
        "url_summary_direct": _by_case_id(_load_records(args.url_tasks)),
    }

    output: list[dict[str, Any]] = []
    for case in cases:
        batch = str(case.get("batch") or "")
        case_id = str(case.get("case_id") or "")
        query = str(case.get("query") or "").strip()
        spec = source_specs.get(query) if batch == "rwkv_search_100" else spec_by_case.get(batch, {}).get(case_id)
        spec = spec or {}
        output.append(
            {
                "schema_version": "independent_reference_case.v1",
                "case_id": case_id,
                "batch": batch,
                "query": query,
                "source_policy": spec.get("source_policy", "independent_primary_or_official"),
                "expected_domains": spec.get("expected_domains_any", []),
                "target_url_patterns": spec.get("target_url_patterns_any", []),
                "category": spec.get("category", spec.get("task_type", "")),
                "task_family": spec.get("task_family", spec.get("category", "")),
                "notes": spec.get("notes", ""),
                "reference_status": "pending_independent_research",
                "reference_answer": "",
                "reference_citations": [],
                "key_facts": [],
                "comparison_status": "pending_reference",
            }
        )

    if len(output) != 172 or len({row["case_id"] for row in output}) != len(output):
        raise SystemExit(f"expected 172 unique cases, got {len(output)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "case_count": len(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
