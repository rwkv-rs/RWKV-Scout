"""Compare two recorded experiment artifacts with an auditable report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from utils.experiment_metrics import compare_scores, recommend_decision, score_trace
from utils.human_review import aggregate_reviews, load_reviews, paired_blind_summary
from utils.reference_validation import validate_reference_case


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    output = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        trace = row.get("trace") or row.get("normalized_trace")
        if isinstance(trace, dict):
            case = row.get("case") or {}
            output.append(
                {
                    "case": case,
                    "trace": trace,
                    "reference_validation": row.get("reference_validation") or validate_reference_case(case),
                    "score": score_trace(trace, case),
                }
            )
    return output


def load_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment artifact must be an object: {path}")
    return payload


def compatibility_report(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for key in ("dataset_version", "prompt_version"):
        checks[key] = {
            "baseline": baseline.get(key, ""),
            "candidate": candidate.get(key, ""),
            "match": baseline.get(key, "") == candidate.get(key, ""),
        }
    base_model = baseline.get("model") or {}
    cand_model = candidate.get("model") or {}
    for key in ("model", "endpoint", "context_length", "provider"):
        checks[f"model.{key}"] = {
            "baseline": base_model.get(key),
            "candidate": cand_model.get(key),
            "match": base_model.get(key) == cand_model.get(key),
        }
    checks["changed_variable"] = {
        "baseline": baseline.get("changed_variable", ""),
        "candidate": candidate.get("changed_variable", ""),
        "match": baseline.get("changed_variable", "") == candidate.get("changed_variable", ""),
    }
    return {"compatible": all(item["match"] for item in checks.values()), "checks": checks}


def _value(score: dict[str, Any], path: str) -> float | None:
    group, key = path.split(".", 1)
    value = (score.get(group) or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def case_differences(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    base = {item["score"].get("question_id"): item["score"] for item in baseline}
    cand = {item["score"].get("question_id"): item["score"] for item in candidate}
    successes: list[dict] = []
    failures: list[dict] = []
    quality_keys = (
        "retrieval.recall_at_5",
        "evidence.key_fact_recall",
        "answer.fact_accuracy",
        "answer.citation_accuracy",
    )
    for question_id in sorted(set(base) & set(cand)):
        before, after = base[question_id], cand[question_id]
        deltas = {key: (_value(after, key), _value(before, key)) for key in quality_keys}
        changes = {
            key: round(new - old, 4) if new is not None and old is not None else None
            for key, (new, old) in deltas.items()
        }
        known = [value for value in changes.values() if value is not None]
        row = {
            "question_id": question_id,
            "deltas": changes,
            "baseline": before,
            "candidate": after,
            "success_stage": _stage_for_score(after) if known and any(value > 0 for value in known) else "",
            "failure_stage": _stage_for_score(after) if known and any(value < 0 for value in known) else "",
        }
        if known and any(value > 0 for value in known) and not any(value < 0 for value in known):
            successes.append(row)
        if known and any(value < 0 for value in known):
            failures.append(row)
    return successes, failures


def _stage_for_score(score: dict[str, Any]) -> str:
    """Point a reviewer at the first likely failing stage, not only a score."""
    engineering = score.get("engineering") or {}
    retrieval = score.get("retrieval") or {}
    evidence = score.get("evidence") or {}
    context = score.get("context") or {}
    answer = score.get("answer") or {}
    if engineering.get("status") not in {"completed", "unknown"} or engineering.get("error_count", 0):
        return "engineering"
    if not answer.get("non_empty"):
        return "answer_generation"
    if answer.get("unsupported_citation_count", 0):
        return "citation_validation"
    if retrieval.get("result_count", 0) == 0 or retrieval.get("recall_at_5") == 0:
        return "retrieval_recall"
    if evidence.get("body_available_rate") == 0 or evidence.get("key_fact_recall") == 0:
        return "content_extraction"
    if context.get("important_information_truncation_rate"):
        return "context_building"
    return "answer_generation"


def failure_case_report(failures: list[dict[str, Any]]) -> dict[str, Any]:
    """Make regression cases directly actionable for review or rollback."""
    by_stage = Counter(str(row.get("failure_stage") or "unknown") for row in failures)
    cases = []
    for row in failures:
        candidate = row.get("candidate") or {}
        engineering = candidate.get("engineering") or {}
        cases.append(
            {
                "question_id": row.get("question_id", ""),
                "failure_stage": row.get("failure_stage") or "unknown",
                "deltas": row.get("deltas") or {},
                "candidate_status": engineering.get("status", "unknown"),
                "candidate_error_count": engineering.get("error_count", 0),
                "candidate_timed_out": bool(engineering.get("timed_out")),
            }
        )
    return {
        "count": len(failures),
        "by_stage": dict(sorted(by_stage.items())),
        "cases": cases,
    }


def reference_gate(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    existing = artifact.get("reference_validation")
    if isinstance(existing, dict) and "all_ready" in existing:
        return existing
    validations = [row.get("reference_validation") or {} for row in rows]
    counts = {
        "ready_count": sum(item.get("ready_for_scoring") is True for item in validations),
        "pending_count": sum(item.get("status") == "pending" for item in validations),
        "stale_count": sum(item.get("status") == "stale" for item in validations),
        "invalid_count": sum(item.get("status") == "invalid" for item in validations),
    }
    return {
        "sample_count": len(validations),
        **counts,
        "all_ready": bool(validations) and counts["ready_count"] == len(validations),
        "issues": sorted({issue for item in validations for issue in item.get("issues") or []}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--group-by", choices=["domain", "persona", "task_type", "difficulty"], default=None)
    parser.add_argument("--human-reviews", type=Path, default=None)
    parser.add_argument("--model-judge", type=Path, default=None, help="optional supplementary model-judge artifact; never used alone for promotion")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline_artifact = load_artifact(args.baseline)
    candidate_artifact = load_artifact(args.candidate)
    baseline_rows = load_results(args.baseline)
    candidate_rows = load_results(args.candidate)
    compatibility = compatibility_report(baseline_artifact, candidate_artifact)
    trace_validation = {
        "baseline": baseline_artifact.get("trace_validation", {}),
        "candidate": candidate_artifact.get("trace_validation", {}),
    }
    risk_validation = {
        "baseline": baseline_artifact.get("risk_validation", {}),
        "candidate": candidate_artifact.get("risk_validation", {}),
    }
    reference_validation = {
        "baseline": reference_gate(baseline_rows, baseline_artifact),
        "candidate": reference_gate(candidate_rows, candidate_artifact),
    }
    baseline = [row["score"] for row in baseline_rows]
    candidate = [row["score"] for row in candidate_rows]
    all_grouped = {
        dimension: compare_scores(baseline, candidate, group_by=dimension)
        for dimension in ("domain", "persona", "task_type", "difficulty")
    }
    comparison = compare_scores(baseline, candidate, group_by=args.group_by)
    successes, failures = case_differences(baseline_rows, candidate_rows)
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "sample_counts": {"baseline": len(baseline), "candidate": len(candidate)},
        "experiment_summary": {
            "baseline_experiment_id": baseline_artifact.get("experiment_id", ""),
            "candidate_experiment_id": candidate_artifact.get("experiment_id", ""),
            "baseline_variant": baseline_artifact.get("variant", ""),
            "candidate_variant": candidate_artifact.get("variant", ""),
            "dataset_version": baseline_artifact.get("dataset_version", ""),
            "model": baseline_artifact.get("model", {}),
            "changed_variable": baseline_artifact.get("changed_variable", ""),
            "compatibility": compatibility,
            "trace_validation": trace_validation,
            "risk_validation": risk_validation,
            "reference_validation": reference_validation,
        },
        "comparison": comparison,
        "grouped_results": all_grouped,
        "success_cases": successes,
        "failure_cases": failures,
        "decision": (
            {"decision": "继续实验", "reason": "运行轨迹结构校验未通过，不能进行生产归因"}
            if any((item.get("invalid_count") or 0) > 0 for item in trace_validation.values())
            else (
                {"decision": "继续实验", "reason": "高风险回答缺少风险边界，不能进入生产"}
                if any((item.get("invalid_count") or 0) > 0 for item in risk_validation.values())
                else (
                    {"decision": "继续实验", "reason": "baseline/candidate 的控制变量不一致，不能归因"}
                    if not compatibility["compatible"]
                    else recommend_decision(compare_scores(baseline, candidate))
                )
            )
        ),
    }
    if any(item.get("all_ready") is not True for item in reference_validation.values()):
        decision = {"decision": "继续实验", "reason": "参考答案尚未完成一致的人工审核或来源新鲜度校验"}
    elif any((item.get("invalid_count") or 0) > 0 for item in trace_validation.values()):
        decision = {"decision": "继续实验", "reason": "运行轨迹结构校验未通过，不能进行生产归因"}
    elif any((item.get("invalid_count") or 0) > 0 for item in risk_validation.values()):
        decision = {"decision": "继续实验", "reason": "高风险回答缺少风险边界，不能进入生产"}
    elif not compatibility["compatible"]:
        decision = {"decision": "继续实验", "reason": "baseline/candidate 的控制变量不一致，不能归因"}
    else:
        decision = recommend_decision(comparison)
    report["failure_case_report"] = failure_case_report(failures)
    report["reference_validation"] = reference_validation
    report["decision"] = decision
    report["rollback"] = {
        "triggered": decision.get("decision") == "回滚候选方案",
        "action": (
            "保留 baseline，停止 candidate 推广并回放失败案例"
            if decision.get("decision") == "回滚候选方案"
            else "不触发自动回滚"
        ),
        "automatic_mutation": False,
    }
    if args.human_reviews:
        reviews = load_reviews(args.human_reviews)
        report["human_evaluation"] = {
            "aggregate": aggregate_reviews(reviews),
            "paired_blind": paired_blind_summary(reviews),
        }
    if args.model_judge:
        judge = load_artifact(args.model_judge)
        judge_results = judge.get("results") or []
        report["model_evaluation"] = {
            "artifact": str(args.model_judge),
            "judge_version": judge.get("artifact_version", ""),
            "sample_count": len(judge_results),
            "completed_count": sum(item.get("status") == "completed" for item in judge_results if isinstance(item, dict)),
            "invalid_count": sum(item.get("status") in {"invalid_output", "failed"} for item in judge_results if isinstance(item, dict)),
            "supplementary_only": True,
        }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
