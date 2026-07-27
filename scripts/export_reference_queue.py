"""Export a human reference-review queue from a dataset and recorded runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.evaluation_dataset import load_dataset


def _artifact_traces(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    output = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        case = row.get("case") or {}
        question_id = str(case.get("question_id") or "").strip()
        if question_id and isinstance(row.get("trace"), dict):
            output[question_id] = row["trace"]
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = _artifact_traces(args.artifact)
    rows = []
    for case in load_dataset(args.dataset):
        trace = traces.get(case["question_id"], {})
        rows.append(
            {
                "question_id": case["question_id"],
                "question": case["question"],
                "persona": case["persona"],
                "domain": case["domain"],
                "task_type": case["task_type"],
                "difficulty": case["difficulty"],
                "acceptance_criteria": case.get("acceptance_criteria", []),
                "rejection_criteria": case.get("rejection_criteria", []),
                "risk_checks": case.get("risk_checks", []),
                "expected_source_types": case.get("expected_source_types", []),
                "key_facts": case.get("key_facts", []),
                "search_queries": trace.get("search_queries", case.get("search_queries", [])),
                "search_results": trace.get("search_results", case.get("search_results", [])),
                "navigation_trace": trace.get("navigation_trace", case.get("navigation_trace", [])),
                "sources": trace.get("sources", case.get("sources", [])),
                "evidence": trace.get("evidence", case.get("evidence", [])),
                "model_draft_for_review_only": str(trace.get("final_answer") or ""),
                "reference_answer": case.get("reference_answer", ""),
                "reference_citations": case.get("reference_citations", []),
                "reference_metadata": case.get("reference_metadata", {}),
                "review_instructions": "人工核对原始来源后填写 reference_answer/reference_citations；不得直接接受 model_draft_for_review_only。",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_count": len(rows), "trace_count": len(traces)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
