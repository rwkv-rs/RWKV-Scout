"""Stage-level metrics and controlled baseline/candidate comparisons."""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from utils.citation_validator import validate_citations


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _mean(values: Iterable[float | int | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return round(statistics.mean(numbers), 4) if numbers else None


def _percent(numerator: int, denominator: int) -> float | None:
    return round(numerator * 100 / denominator, 4) if denominator else None


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return round(len(set(retrieved[: max(0, k)]) & relevant) / len(relevant), 4)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if k <= 0:
        return None
    return round(len(set(retrieved[:k]) & relevant) / k, 4) if relevant else None


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return round(1 / index, 4)
    return 0.0


def ndcg(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    ranked = retrieved[: max(0, k)]
    dcg = sum((1 / math.log2(index + 2)) for index, item in enumerate(ranked) if item in relevant)
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), k)))
    return round(dcg / ideal, 4) if ideal else None


def _item_key(item: dict[str, Any]) -> str:
    return _norm(item.get("url") or item.get("title") or item.get("id"))


def _case_source_urls(case: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in case.get("sources") or case.get("reference_citations") or []:
        if isinstance(item, dict):
            value = _norm(item.get("url"))
        else:
            value = _norm(item)
        if value:
            values.add(value)
    return values


def _case_facts(case: dict[str, Any]) -> list[str]:
    facts = case.get("key_facts") or []
    return [str(value).strip() for value in facts if str(value).strip()]


def _fact_present(fact: str, text: str) -> bool:
    """Match a reference fact conservatively without inventing semantics."""
    expected = _norm(fact)
    observed = _norm(text)
    if not expected or not observed:
        return False
    if expected in observed:
        return True
    expected_tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", expected))
    observed_tokens = set(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", observed))
    if not expected_tokens:
        return False
    return len(expected_tokens & observed_tokens) / len(expected_tokens) >= 0.8


def _answer_overlap(reference: str, answer: str) -> float | None:
    """Return a transparent lexical F1; semantic correctness stays human-reviewed."""
    expected = set(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", _norm(reference)))
    observed = set(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", _norm(answer)))
    if not expected or not observed:
        return None
    overlap = len(expected & observed)
    precision = overlap / len(observed)
    recall = overlap / len(expected)
    return round(2 * precision * recall / max(0.000001, precision + recall), 4)


def _case_reference_answer(case: dict[str, Any]) -> str:
    return str(case.get("reference_answer") or "").strip()


def _duration_ms(events: list[dict[str, Any]], manifest: dict[str, Any]) -> float | None:
    if manifest.get("duration_ms") is not None:
        try:
            return float(manifest["duration_ms"])
        except (TypeError, ValueError):
            pass
    timestamps = []
    for event in events:
        try:
            timestamps.append(datetime.fromisoformat(str(event["timestamp"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(timestamps) < 2:
        return None
    return round((max(timestamps) - min(timestamps)).total_seconds() * 1000, 1)


def _multi_hop_gain(events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        if event.get("type") != "multi_hop_merge":
            continue
        data = event.get("data") or {}
        first_count = data.get("first_result_count")
        final_count = data.get("final_result_count")
        if isinstance(first_count, (int, float)) and first_count:
            if isinstance(final_count, (int, float)):
                return round((final_count - first_count) / first_count, 4)
    return None


def score_trace(trace: dict[str, Any], case: dict[str, Any] | None = None) -> dict[str, Any]:
    case = case or {}
    results = [item for item in trace.get("search_results") or [] if isinstance(item, dict)]
    ranking_events = trace.get("ranking_trace") or [event for event in trace.get("events") or [] if event.get("type") == "ranking"]
    final_ranking_event = ranking_events[-1] if ranking_events else {}
    ranking_rows = [
        row
        for row in ((final_ranking_event.get("data") or {}).get("results") or [])
        if isinstance(row, dict)
    ]
    if ranking_rows:
        retrieved = [_norm(row.get("url") or row.get("dedup_key")) for row in ranking_rows]
        retrieved = [item for item in retrieved if item]
        by_url = {_norm(item.get("url")): item for item in results if _norm(item.get("url"))}
        retrieval_results = [by_url[key] for key in retrieved if key in by_url]
    else:
        retrieval_results = results
        retrieved = [_item_key(item) for item in results if _item_key(item)]
    relevant = _case_source_urls(case)
    answer = str(trace.get("final_answer") or "")
    facts = _case_facts(case)
    evidence_text = " ".join(
        str(item.get("page_excerpt") or item.get("content") or item.get("abstract") or item.get("snippet") or "")
        for item in results
    ).casefold()
    answer_lower = answer.casefold()
    evidence_fact_hits = [fact for fact in facts if _fact_present(fact, evidence_text)]
    answer_fact_hits = [fact for fact in facts if _fact_present(fact, answer_lower)]
    supported_fact_hits = [
        fact for fact in facts
        if _fact_present(fact, evidence_text) and _fact_present(fact, answer_lower)
    ]
    reference_answer = _case_reference_answer(case)
    reference_sources = _case_source_urls({"reference_citations": case.get("reference_citations") or []})
    citations = trace.get("citations") or []
    answered_source_urls = {
        _norm(item.get("url"))
        for item in citations if isinstance(item, dict) and _norm(item.get("url"))
    }
    citation_validation = validate_citations(
        citations,
        answer=answer,
        evidence=results,
        check_remote=False,
    )
    events = trace.get("events") or []
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    tool_results = [event for event in events if event.get("type") == "tool_result"]
    retries = [event for event in events if event.get("type") == "retry"]
    errors = [
        event
        for event in events
        if event.get("type") in {"error", "provider_error"}
        or (event.get("type") == "model_call" and event.get("status") == "failed")
    ]
    model_calls = [event for event in events if event.get("type") == "model_call"]
    gate_waits = [
        event.get("wait_ms")
        for event in events
        if event.get("type") == "runtime_gate" and event.get("status") == "acquired"
    ]
    budget_events = [event for event in events if event.get("type") == "runtime_budget"]
    budget_start = next(
        (event for event in budget_events if event.get("status") == "started"),
        {},
    )
    budget_end = next(
        (event for event in reversed(budget_events) if event.get("status") in {"completed", "timed_out", "failed"}),
        {},
    )
    model_input_tokens = sum(int(event.get("prompt_tokens") or 0) for event in model_calls)
    model_output_tokens = sum(int(event.get("completion_tokens") or 0) for event in model_calls)
    duplicate_urls = len(retrieved) - len(set(retrieved))
    contents = [
        _norm(item.get("page_excerpt") or item.get("content") or item.get("abstract") or "")
        for item in results
    ]
    duplicate_contents = len(contents) - len(set(item for item in contents if item))
    quality_values = [float(item["content_quality"]) for item in results if item.get("content_quality") is not None]
    synthesis_prompt = "\n".join(str(item.get("prompt") or "") for item in trace.get("model_outputs") or [])
    relevant_positions = [index for index, item in enumerate(retrieved, start=1) if item in relevant]
    rounds = [event for event in events if event.get("type") in {"candidate_merge", "multi_hop_merge"}]
    query_rewrite_events = [event for event in events if event.get("type") == "query_candidates"]
    rewrite_success = any(event.get("queries") for event in query_rewrite_events)
    context_events = trace.get("context_trace") or [event for event in events if event.get("type") == "context_build"]
    context_data = ((context_events[-1].get("data") or {}) if context_events else {})
    context_text = str(context_data.get("context_text") or "")
    selected_evidence = [item for item in context_data.get("selected_evidence") or [] if isinstance(item, dict)]
    context_stats = context_data.get("context_stats") or {}
    risk_validation = trace.get("risk_validation") or {}
    context_length = trace.get("manifest", {}).get("config", {}).get("model", {}).get("context_length", 10240)
    def event_duration(event_type: str) -> float | None:
        return _mean(
            event.get("duration_ms")
            if isinstance(event.get("duration_ms"), (int, float))
            else (event.get("data") or {}).get("duration_ms")
            for event in events
            if event.get("type") == event_type
            and (
                isinstance(event.get("duration_ms"), (int, float))
                or isinstance((event.get("data") or {}).get("duration_ms"), (int, float))
            )
        )
    return {
        "question_id": case.get("question_id") or trace.get("manifest", {}).get("run_id", ""),
        "sample_metadata": {
            key: case.get(key, trace.get("manifest", {}).get("experiment", {}).get(key, ""))
            for key in ("persona", "domain", "task_type", "difficulty", "dataset_version")
        },
        "retrieval": {
            "result_count": len(retrieval_results),
            "recall_at_5": recall_at_k(retrieved, relevant, 5),
            "precision_at_5": precision_at_k(retrieved, relevant, 5),
            "mrr": mean_reciprocal_rank(retrieved, relevant),
            "ndcg_at_5": ndcg(retrieved, relevant, 5),
            "valid_source_rate": _percent(sum(bool(item.get("url")) for item in retrieval_results), len(retrieval_results)),
            "effective_source_recall": recall_at_k(retrieved, relevant, 5),
            "key_evidence_recall": _percent(len(evidence_fact_hits), len(facts)),
            "duplicate_result_rate": round(duplicate_urls / len(retrieved), 4) if retrieved else None,
            "invalid_result_rate": _percent(sum(not item.get("url") for item in retrieval_results), len(retrieval_results)),
            "query_rewrite_success": bool(rewrite_success) if query_rewrite_events else None,
            "query_count": len(trace.get("search_queries") or []),
            "navigation_count": len(trace.get("navigation_trace") or []),
            "average_retrieval_rounds": len(rounds) or None,
            "average_page_jumps": len(trace.get("navigation_trace") or []) or None,
            "ranking_event_count": len(ranking_events) or None,
            "average_rerank_score": _mean(row.get("rerank_score") for row in ranking_rows),
            "multi_hop_gain": _multi_hop_gain(events),
        },
        "evidence": {
            "evidence_count": len(trace.get("evidence") or []),
            "key_fact_recall": _percent(len(evidence_fact_hits), len(facts)),
            "body_available_rate": _percent(
                sum(bool(item.get("page_excerpt") or item.get("content") or item.get("abstract")) for item in results),
                len(results),
            ),
            "extraction_accuracy": None,
            "noise_ratio": round(1 - statistics.mean(quality_values), 4) if quality_values else None,
            "key_fragment_retention": _percent(len(evidence_fact_hits), len(facts)),
            "chunk_completeness": _percent(
                sum(item.get("selected_chars", 0) for item in selected_evidence),
                sum(item.get("source_chars", 0) for item in selected_evidence),
            ),
            "structured_info_retention": None,
            "duplicate_fragment_rate": round(duplicate_contents / len(contents), 4) if contents else None,
            "effective_information_density": round(sum(len(item) for item in contents) / max(1, len(contents)), 2)
            if contents
            else None,
        },
        "context": {
            "key_evidence_rank_position": min(relevant_positions) if relevant_positions else None,
            "evidence_coverage": _percent(
                sum(_fact_present(fact, context_text) for fact in facts), len(facts)
            ),
            "redundancy_rate": round(duplicate_contents / len(contents), 4) if contents else None,
            "conflict_recognition_rate": None,
            "length_utilization": round(
                float(context_stats.get("context_tokens")) / max(1, int(context_length)), 4
            )
            if context_stats.get("context_tokens") is not None
            else (round(len(synthesis_prompt) / max(1, int(context_length)), 4) if synthesis_prompt else None),
            "context_token_count": context_stats.get("context_tokens"),
            "context_chunk_count": context_stats.get("chunk_count"),
            "important_information_truncation_rate": _percent(
                sum(bool(item.get("truncated")) for item in selected_evidence),
                len(selected_evidence),
            )
            if selected_evidence
            else None,
        },
        "answer": {
            "non_empty": bool(answer.strip()),
            "answer_length": len(answer),
            "fact_accuracy": _percent(len(answer_fact_hits), len(facts)),
            "evidence_support_rate": _percent(len(supported_fact_hits), len(answer_fact_hits)),
            "citation_accuracy": _percent(citation_validation["supported"], citation_validation["total"]),
            "citation_completeness": _percent(citation_validation["referenced"], citation_validation["total"]),
            "citation_locator_coverage": _percent(citation_validation["located"], citation_validation["total"]),
            "citation_accessibility": _percent(citation_validation["accessible"], citation_validation["total"])
            if citation_validation["remote_check"]
            else None,
            "unsupported_citation_count": citation_validation["invalid"],
            "fact_hit_count": len(answer_fact_hits),
            "fact_count": len(facts),
            "question_coverage": _percent(len(answer_fact_hits), len(facts)),
            "reference_answer_overlap": _answer_overlap(reference_answer, answer),
            "reference_citation_recall": _percent(
                len(answered_source_urls & reference_sources), len(reference_sources)
            ),
            "instruction_adherence": None,
            "uncertainty_expression_accuracy": None,
            "hallucination_rate": None,
            "unsupported_claim_rate": None,
            "answer_usability": None,
            "risk_warning_present": risk_validation.get("warning_present"),
            "risk_validation_passed": risk_validation.get("valid"),
            "risk_issue_count": len(risk_validation.get("issues") or []),
        },
        "engineering": {
            "status": trace.get("manifest", {}).get("status", "unknown"),
            "duration_ms": _duration_ms(events, trace.get("manifest", {})),
            "tool_call_count": len(tool_calls),
            "tool_result_count": len(tool_results),
            "model_call_count": len(model_calls),
            "failed_model_call_count": sum(event.get("status") == "failed" for event in model_calls),
            "model_input_tokens": model_input_tokens,
            "model_output_tokens": model_output_tokens,
            "estimated_cost": None,
            "retry_count": len(retries),
            "error_count": len(errors),
            "concurrency_wait_ms": _mean(gate_waits),
            "time_budget_seconds": budget_start.get("timeout_seconds"),
            "budget_elapsed_ms": budget_end.get("elapsed_ms"),
            "timed_out": budget_end.get("status") == "timed_out" or trace.get("manifest", {}).get("status") == "timed_out",
            "first_success": not errors and bool(answer.strip()),
        },
        "timings": {
            "query_rewrite_ms": event_duration("query_candidates"),
            "retrieval_ms": event_duration("candidate_merge"),
            "page_fetch_ms": _mean(
                event.get("duration_ms")
                for event in events
                if event.get("type") == "page_fetch" and isinstance(event.get("duration_ms"), (int, float))
            ),
            "content_extraction_ms": _mean(
                event.get("duration_ms")
                for event in events
                if event.get("type") in {"content_extract", "page_extract"}
                and isinstance(event.get("duration_ms"), (int, float))
            ),
            "answer_generation_ms": event_duration("synthesis"),
            "total_ms": _duration_ms(events, trace.get("manifest", {})),
        },
        "citation_validation": citation_validation,
    }


def _flatten_metrics(score: dict[str, Any]) -> dict[str, float | int | bool | None]:
    output: dict[str, float | int | bool | None] = {}
    for group, values in score.items():
        if not isinstance(values, dict) or group in {"sample_metadata", "citation_validation"}:
            continue
        for key, value in values.items():
            if isinstance(value, (int, float, bool)) or value is None:
                output[f"{group}.{key}"] = value
    return output


def aggregate_scores(scores: list[dict[str, Any]], *, group_by: str | None = None) -> dict[str, Any]:
    if not group_by:
        groups = {"all": scores}
    else:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for score in scores:
            groups[str((score.get("sample_metadata") or {}).get(group_by) or "unknown")].append(score)
    rows: dict[str, Any] = {}
    for group, members in groups.items():
        flattened = [_flatten_metrics(item) for item in members]
        keys = sorted({key for row in flattened for key in row})
        metrics = {key: _mean(row.get(key) for row in flattened) for key in keys}
        durations = sorted(
            float(row["engineering.duration_ms"])
            for row in flattened
            if isinstance(row.get("engineering.duration_ms"), (int, float))
        )
        if durations:
            metrics["engineering.duration_p50_ms"] = durations[max(0, math.ceil(len(durations) * 0.50) - 1)]
            metrics["engineering.duration_p95_ms"] = durations[max(0, math.ceil(len(durations) * 0.95) - 1)]
            metrics["engineering.duration_p99_ms"] = durations[max(0, math.ceil(len(durations) * 0.99) - 1)]
        rows[group] = {
            "sample_count": len(members),
            "metrics": metrics,
        }
    return {"group_by": group_by or "all", "groups": rows}


def compare_scores(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    group_by: str | None = None,
) -> dict[str, Any]:
    base_agg = aggregate_scores(baseline, group_by=group_by)
    cand_agg = aggregate_scores(candidate, group_by=group_by)
    groups = sorted(set(base_agg["groups"]) | set(cand_agg["groups"]))
    result: dict[str, Any] = {"group_by": group_by or "all", "groups": {}}
    for group in groups:
        before = base_agg["groups"].get(group, {}).get("metrics", {})
        after = cand_agg["groups"].get(group, {}).get("metrics", {})
        metrics = {}
        for key in sorted(set(before) | set(after)):
            old, new = before.get(key), after.get(key)
            delta = round(new - old, 4) if isinstance(old, (int, float)) and isinstance(new, (int, float)) else None
            relative = round(delta / abs(old), 4) if delta is not None and old else None
            metrics[key] = {"baseline": old, "candidate": new, "absolute_change": delta, "relative_change": relative}
        result["groups"][group] = {
            "baseline_count": base_agg["groups"].get(group, {}).get("sample_count", 0),
            "candidate_count": cand_agg["groups"].get(group, {}).get("sample_count", 0),
            "metrics": metrics,
        }
    result["paired_statistics"] = paired_statistics(baseline, candidate)
    return result


def paired_statistics(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    metric_paths: tuple[str, ...] = (
        "retrieval.recall_at_5",
        "evidence.key_fact_recall",
        "answer.fact_accuracy",
        "answer.citation_accuracy",
        "engineering.duration_ms",
    ),
) -> dict[str, Any]:
    """Return paired deltas and an approximate 95% CI by question ID.

    The statistic is deliberately descriptive: it does not promote a
    candidate by itself, and it reports ``None`` when too few paired samples
    exist to support a confidence interval.
    """
    base_by_id = {score.get("question_id"): score for score in baseline}
    candidate_by_id = {score.get("question_id"): score for score in candidate}
    output: dict[str, Any] = {}
    for path in metric_paths:
        group, key = path.split(".", 1)
        deltas = []
        for question_id in sorted(set(base_by_id) & set(candidate_by_id)):
            old = (base_by_id[question_id].get(group) or {}).get(key)
            new = (candidate_by_id[question_id].get(group) or {}).get(key)
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                deltas.append(float(new) - float(old))
        if not deltas:
            output[path] = {"n": 0, "mean_delta": None, "ci95": None, "positive_fraction": None}
            continue
        mean_delta = statistics.mean(deltas)
        if len(deltas) > 1:
            margin = 1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas))
            ci95 = [round(mean_delta - margin, 4), round(mean_delta + margin, 4)]
        else:
            ci95 = None
        output[path] = {
            "n": len(deltas),
            "mean_delta": round(mean_delta, 4),
            "ci95": ci95,
            "positive_fraction": round(sum(delta > 0 for delta in deltas) / len(deltas), 4),
        }
    return output


def recommend_decision(comparison: dict[str, Any]) -> dict[str, Any]:
    """Apply conservative promotion rules; unknown metrics require more data."""
    all_group = comparison.get("groups", {}).get("all", {})
    metrics = all_group.get("metrics", {})
    sample_count = min(
        int(all_group.get("baseline_count", 0) or 0),
        int(all_group.get("candidate_count", 0) or 0),
    )
    quality_keys = (
        "retrieval.recall_at_5",
        "evidence.key_fact_recall",
        "answer.fact_accuracy",
        "answer.citation_accuracy",
    )
    latency = metrics.get("engineering.duration_ms", {})
    quality = [metrics.get(key, {}).get("absolute_change") for key in quality_keys]
    known = [value for value in quality if isinstance(value, (int, float))]
    if sample_count < 5:
        decision = "继续实验"
        reason = "样本量不足以支持生产替换决策"
    elif len(known) < 2:
        decision = "继续实验"
        reason = "缺少可比较的参考事实或引用指标"
    elif any(value < 0 for value in known) and any(value > 0 for value in known):
        decision = "局部采用"
        reason = "质量指标出现分化，需要按问题类型或领域切分"
    elif all(value >= 0 for value in known) and any(value > 0 for value in known):
        if isinstance(latency.get("absolute_change"), (int, float)) and latency["absolute_change"] > 0:
            decision = "局部采用"
            reason = "质量提升但延迟增加，需要限制到受益场景"
        else:
            decision = "全量采用"
            reason = "整体质量指标稳定提升且未观察到延迟退化"
    elif all(value <= 0 for value in known) and any(value < 0 for value in known):
        decision = "回滚候选方案"
        reason = "候选方案在可比较质量指标上退化"
    else:
        decision = "继续实验"
        reason = "提升不足以支持生产替换"
    return {"decision": decision, "reason": reason, "quality_deltas": dict(zip(quality_keys, quality))}
