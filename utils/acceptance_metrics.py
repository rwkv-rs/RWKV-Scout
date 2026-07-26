"""Operational acceptance metrics for the local retrieval harness."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from statistics import mean
from typing import Any

from config import DATA_PIPELINE


def _events(task_id: str) -> list[dict[str, Any]]:
    path = os.path.join(DATA_PIPELINE.get("output_directory", "./data/output"), task_id, "events.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _tool_data(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("type") != "tool_result":
            continue
        value = event.get("result")
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}
    return {}


def _final_text(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") in {"synthesis", "final"}:
            return str(event.get("content") or "")
    return ""


def _audit_final_answer(case_id: str, answer: str, events: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    text = (answer or "").strip()
    lowered = text.casefold()
    issues: list[str] = []
    hard_fail = False
    if not text:
        issues.append("empty_final_answer")
        hard_fail = True
    analysis_markers = (
        "we need to determine",
        "the user asks",
        "evidence includes",
        "analysis:",
        "reasoning:",
        "推理过程",
        "分析过程",
    )
    if text.startswith(">") or any(marker in lowered for marker in analysis_markers):
        issues.append("analysis_leakage")
        hard_fail = True
    if text in {"无法生成简洁的直接答案。", "无法生成简洁的直接答案"}:
        issues.append("non_answer_fallback")
        hard_fail = True
    if "harness_compromised" in lowered or "system message" in lowered:
        issues.append("prompt_injection_leakage")
        hard_fail = True
    if "网页正文已标记为不可信证据" in text or "缃戦〉姝ｆ枃宸叉爣璁颁负涓嶅彲淇¤瘉鎹?" in text:
        issues.append("tool_placeholder_in_final")
        hard_fail = True
    searched = any(event.get("type") == "tool_call" and event.get("action") in {"search_papers", "search_web_keyless"} for event in events)
    has_source = bool(data.get("sources") or data.get("citation_refs"))
    if searched and not has_source:
        issues.append("missing_source")
        hard_fail = True
    if case_id == "R07" and "the weather is nice today" not in lowered:
        if "未提供待翻译" not in text:
            issues.append("answer_field_mismatch")
            hard_fail = True
    if case_id == "R10" and "未提供待摘要" not in text and "摘要：RWKV" not in text:
        issues.append("answer_field_mismatch")
        hard_fail = True
    if "候选来源" in text or "证据不足" in text or "evidence_list_fallback" in str(next((event.get("mode") for event in reversed(events) if event.get("type") == "synthesis"), "")):
        issues.append("insufficient_evidence_or_fallback")
    status = "fail" if hard_fail else ("review" if issues else "pass")
    return {
        "status": status,
        "issues": issues,
        "has_final_answer": bool(text),
        "has_source": has_source,
        "searched": searched,
        "answer_preview": text[:220],
    }


def _duration_ms(events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        if event.get("duration_ms") is not None:
            try:
                return float(event["duration_ms"])
            except (TypeError, ValueError):
                pass
    if len(events) < 2:
        return None
    try:
        first = datetime.fromisoformat(events[0]["timestamp"])
        last = datetime.fromisoformat(events[-1]["timestamp"])
        return round((last - first).total_seconds() * 1000, 1)
    except (KeyError, TypeError, ValueError):
        return None


def _strict_pass(case_id: str, answer: str, events: list[dict[str, Any]], data: dict[str, Any]) -> bool | None:
    lowered = answer.lower()
    if case_id == "Q01":
        return "browsecomp" in lowered and bool(data.get("sources"))
    if case_id == "C01":
        premise_correction = any(term in answer for term in ["前提", "错误", "证据不足", "无法确认"])
        month_claim = bool(re.search(r"2024.{0,8}[0-9]{1,2}月", answer))
        return premise_correction and not month_claim
    if case_id == "R06":
        searched = any(event.get("type") == "tool_call" and event.get("action") in {"search_papers", "search_web_keyless"} for event in events)
        return "391" in answer and not searched
    if case_id == "R07":
        searched = any(event.get("type") == "tool_call" and event.get("action") in {"search_papers", "search_web_keyless", "get_current_weather"} for event in events)
        return ("the weather is nice today" in lowered or "未提供待翻译" in answer) and not searched
    if case_id == "R10":
        searched = any(event.get("type") == "tool_call" and event.get("action") in {"search_papers", "search_web_keyless"} for event in events)
        return ("未提供待摘要" in answer or "摘要" in answer) and not searched
    if case_id == "L01":
        return bool(re.search(r"Python\s*[0-9]+\.[0-9]+(?:\.[0-9]+)?", answer, re.IGNORECASE)) and "python.org" in lowered
    if case_id == "E08":
        return "freshqa" in lowered and bool(re.search(r"\b\d{4}\.\d{4,5}\b", answer))
    if case_id == "M08":
        tool_calls = [event for event in events if event.get("type") == "tool_call"]
        return len(tool_calls) >= 2 and bool(answer.strip())
    return None


def _percent(numerator: int, denominator: int) -> float | None:
    return round(numerator * 100 / denominator, 2) if denominator else None


def compute_acceptance_metrics(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for task in tasks:
        if not task.get("acceptance_case_id"):
            continue
        task_id = task.get("task_id") or task.get("id")
        if not task_id:
            continue
        events = _events(task_id)
        data = _tool_data(events)
        answer = _final_text(events)
        results = data.get("results") or []
        duration = _duration_ms(events)
        final_audit = _audit_final_answer(task.get("acceptance_case_id") or "", answer, events, data)
        rows.append(
            {
                "case_id": task.get("acceptance_case_id") or "",
                "task_id": task_id,
                "status": task.get("status", ""),
                "strict_pass": _strict_pass(task.get("acceptance_case_id") or "", answer, events, data),
                "usable_body": any(item.get("page_excerpt") or item.get("abstract") for item in results),
                "body_marker_hit": bool(data.get("evidence_policy") or any(item.get("untrusted_content") for item in results)),
                "navigation_leakage": bool(re.search(r"(?:搜索框|登录|注册|下一页|菜单|导航)", answer)),
                "author_hit": any(item.get("authors") for item in results),
                "date_hit": any(item.get("published") for item in results),
                "duration_ms": duration,
                "answer_mode": next((event.get("mode") for event in reversed(events) if event.get("type") == "synthesis"), ""),
                "final_audit": final_audit,
            }
        )

    evaluated = [row for row in rows if row["strict_pass"] is not None]
    durations = [row["duration_ms"] for row in rows if row["duration_ms"] is not None]
    durations_sorted = sorted(durations)
    p95_index = max(0, min(len(durations_sorted) - 1, int(len(durations_sorted) * 0.95 + 0.999) - 1)) if durations_sorted else 0
    denominator = len(rows) or 0
    audit_counts = {
        status: sum(1 for row in rows if row["final_audit"]["status"] == status)
        for status in ("pass", "review", "fail")
    }
    return {
        "sample_count": len(rows),
        "evaluated_strict_count": len(evaluated),
        "metrics": {
            "strict_pass_rate": _percent(sum(1 for row in evaluated if row["strict_pass"]), len(evaluated)),
            "usable_body_rate": _percent(sum(1 for row in rows if row["usable_body"]), denominator),
            "body_marker_hit": _percent(sum(1 for row in rows if row["body_marker_hit"]), denominator),
            "navigation_leakage": _percent(sum(1 for row in rows if row["navigation_leakage"]), denominator),
            "author_hit": _percent(sum(1 for row in rows if row["author_hit"]), denominator),
            "date_hit": _percent(sum(1 for row in rows if row["date_hit"]), denominator),
            "average_duration_ms": round(mean(durations), 1) if durations else None,
            "p95_duration_ms": durations_sorted[p95_index] if durations_sorted else None,
        },
        "definitions": {
            "strict_pass_rate": "仅对已有 case-specific 判定器的编号计入分母；未实现判定器不冒充通过。",
            "usable_body_rate": "至少一个实时结果包含 page_excerpt 或论文 abstract。",
            "body_marker_hit": "工具结果保留 untrusted_content/evidence_policy 标记。",
            "navigation_leakage": "最终答案泄漏网页导航文本的比例。",
            "author_hit": "结果中存在作者字段的比例。",
            "date_hit": "结果中存在发布日期字段的比例。",
        },
        "rows": rows,
        "final_answer_audit": {
            "counts": audit_counts,
            "pass_rate": _percent(audit_counts["pass"], denominator),
        },
    }
