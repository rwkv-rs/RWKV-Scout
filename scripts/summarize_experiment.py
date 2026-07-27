"""Write a trace-derived summary for one dynamic evaluation artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from utils.experiment_metrics import aggregate_scores
from utils.reference_validation import validate_reference_case
from utils.trace_validation import validate_replay_trace


def _failure_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in results:
        trace = row.get("trace") or {}
        score = row.get("score") or {}
        answer = score.get("answer") or {}
        engineering = score.get("engineering") or {}
        reasons: list[str] = []
        if row.get("trace_validation", {}).get("valid") is not True:
            reasons.append("invalid_trace")
        if trace.get("risk_validation", {}).get("valid") is False:
            reasons.append("risk_gate")
        if engineering.get("status") not in {"completed", "completed_with_citation_warnings", "completed_with_risk_warnings"}:
            reasons.append("engineering_failure")
        if not answer.get("non_empty"):
            reasons.append("empty_answer")
        citation = score.get("citation_validation") or {}
        if citation.get("invalid", 0):
            reasons.append("invalid_citation")
        if reasons:
            case = row.get("case") or {}
            output.append(
                {
                    "question_id": case.get("question_id"),
                    "domain": case.get("domain"),
                    "persona": case.get("persona"),
                    "task_type": case.get("task_type"),
                    "difficulty": case.get("difficulty"),
                    "reasons": reasons,
                    "status": trace.get("manifest", {}).get("status"),
                    "answer_preview": str(trace.get("final_answer") or "")[:500],
                }
            )
    return output


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    results = [row for row in payload.get("results", []) if isinstance(row, dict)]
    scores = [row.get("score") for row in results if isinstance(row.get("score"), dict)]
    traces = [row.get("trace") for row in results if isinstance(row.get("trace"), dict)]
    replay = [validate_replay_trace(trace) for trace in traces]
    references = [validate_reference_case(row.get("case") or {}) for row in results]
    status_counts = Counter(str(trace.get("manifest", {}).get("status") or "unknown") for trace in traces)
    return {
        "summary_version": "experiment-summary.v1",
        "experiment_id": payload.get("experiment_id", ""),
        "variant": payload.get("variant", ""),
        "dataset": payload.get("dataset", ""),
        "dataset_version": payload.get("dataset_version", ""),
        "sample_count": len(results),
        "model": payload.get("model", {}),
        "strategy_config": payload.get("strategy_config", {}),
        "status_counts": dict(status_counts),
        "trace_validation": {
            "valid_count": sum(item.get("valid") is True for item in replay),
            "invalid_count": sum(item.get("valid") is not True for item in replay),
            "issues": sorted({issue for item in replay for issue in item.get("issues") or []}),
        },
        "risk_validation": payload.get("risk_validation", {}),
        "reference_validation": {
            "ready_count": sum(item.get("ready_for_scoring") is True for item in references),
            "pending_count": sum(item.get("ready_for_scoring") is not True for item in references),
            "issues": sorted({issue for item in references for issue in item.get("issues") or []}),
        },
        "aggregate": aggregate_scores(scores),
        "groups": {
            group: aggregate_scores(scores, group_by=group)
            for group in ("domain", "persona", "task_type", "difficulty")
        },
        "failure_cases": _failure_cases(results),
        "decision": {
            "decision": "继续实验",
            "reason": "单实验汇总不替代同模型对照、人工参考答案和重复稳定性门禁",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    summary = build_summary(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_count": summary["sample_count"], "failure_count": len(summary["failure_cases"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
