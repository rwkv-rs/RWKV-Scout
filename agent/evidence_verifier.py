"""Independent RWKV evidence audit before final answer synthesis.

The verifier is deliberately a separate model call.  It never writes the
user-facing answer and it can only inspect the bounded EVIDENCE BODY context.
Its output is a control signal for bounded replanning, not a truth oracle.
"""

from __future__ import annotations

import json
from typing import Any

from config import get_llm_context_length
from utils.chunker import get_token_count
from utils.model_budget import bounded_completion_budget
from utils.model_events import visible_model_text
from utils.rwkv_prompt import JSON_CALL_STOP_SUFFIXES, assistant_json_prefix


_VALID_POINT_STATUSES = {"supported", "missing", "conflict", "unclear"}
_VALID_STATUSES = {"supported", "needs_more_evidence", "conflict", "insufficient"}


def _extract_json_object(value: str) -> dict[str, Any]:
    cleaned = visible_model_text(value).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("evidence verifier did not return a JSON object")


def _point_ids(report: dict[str, Any]) -> list[str]:
    return [
        str(row.get("point_id") or "").strip()
        for row in report.get("subquestion_coverage") or []
        if isinstance(row, dict) and str(row.get("point_id") or "").strip()
    ]


def _normalize_result(
    raw: dict[str, Any],
    *,
    report: dict[str, Any],
    source_count: int,
) -> dict[str, Any]:
    valid_refs = {f"S{index}" for index in range(1, source_count + 1)}
    report_rows = {
        str(row.get("point_id") or ""): row
        for row in report.get("subquestion_coverage") or []
        if isinstance(row, dict) and str(row.get("point_id") or "")
    }
    points: list[dict[str, Any]] = []
    raw_points = raw.get("points") or raw.get("subquestions") or []
    if isinstance(raw_points, dict):
        raw_points = [dict(value, id=key) if isinstance(value, dict) else {"id": key, "reason": value} for key, value in raw_points.items()]
    for item in raw_points if isinstance(raw_points, list) else []:
        if not isinstance(item, dict):
            continue
        point_id = str(item.get("id") or item.get("point_id") or "").strip()
        if not point_id:
            continue
        status = str(item.get("status") or "unclear").strip().casefold()
        if status not in _VALID_POINT_STATUSES:
            status = "unclear"
        refs = [
            str(value).strip().upper()
            for value in (item.get("evidence") or item.get("evidence_refs") or item.get("sources") or [])
            if str(value).strip().upper() in valid_refs
        ]
        mechanical = report_rows.get(point_id) or {}
        if mechanical.get("status") == "missing":
            status = "missing"
        if status == "supported" and not refs:
            status = "missing"
        points.append(
            {
                "id": point_id,
                "status": status,
                "evidence": sorted(set(refs)),
                "reason": str(item.get("reason") or item.get("explanation") or "")[:1200],
                "missing": [str(value) for value in item.get("missing") or item.get("missing_information") or [] if str(value).strip()][:12],
                "next_queries": [str(value) for value in item.get("next_queries") or item.get("next_searches") or [] if str(value).strip()][:8],
            }
        )

    expected_ids = _point_ids(report)
    by_id = {item["id"]: item for item in points}
    for point_id in expected_ids:
        if point_id not in by_id:
            mechanical = report_rows.get(point_id) or {}
            status = "missing" if mechanical.get("status") == "missing" else "unclear"
            by_id[point_id] = {
                "id": point_id,
                "status": status,
                "evidence": [],
                "reason": "The verifier did not return a complete row for this task point.",
                "missing": ["direct evidence for this task point"],
                "next_queries": [],
            }
    points = [by_id[point_id] for point_id in expected_ids if point_id in by_id]
    points.extend(item for item in by_id.values() if item["id"] not in expected_ids)

    missing_ids = [item["id"] for item in points if item["status"] in {"missing", "unclear"}]
    conflict_ids = [item["id"] for item in points if item["status"] == "conflict"]
    missing_from_raw = [str(value) for value in raw.get("missing_point_ids") or [] if str(value).strip()]
    missing_ids = list(dict.fromkeys(missing_ids + missing_from_raw))
    next_queries = [
        str(value)
        for value in raw.get("next_queries") or raw.get("next_searches") or []
        if str(value).strip()
    ]
    for item in points:
        next_queries.extend(item["next_queries"])
    next_queries = list(dict.fromkeys(next_queries))[:16]

    status = str(raw.get("status") or "insufficient").strip().casefold()
    if status not in _VALID_STATUSES:
        status = "insufficient"
    if conflict_ids:
        status = "conflict"
    elif missing_ids:
        status = "needs_more_evidence"
    mechanical_missing = (report.get("cross_source") or {}).get("missing_points", 0)
    completion_ready = bool(status == "supported" and not missing_ids and not mechanical_missing)
    if not completion_ready and status == "supported":
        status = "needs_more_evidence"
    return {
        "schema_version": "evidence_verification.v1",
        "status": status,
        "completion_ready": completion_ready,
        "requires_replan": bool(not completion_ready and (missing_ids or conflict_ids)),
        "points": points,
        "missing_point_ids": missing_ids,
        "conflict_point_ids": conflict_ids,
        "next_queries": next_queries,
        "reason": str(raw.get("reason") or raw.get("explanation") or "")[:2000],
        "model_confidence": raw.get("confidence"),
        "is_truth_judgement": False,
    }


def build_verifier_prompt(
    query: str,
    task_plan: dict[str, Any],
    evidence_text: str,
    validation_report: dict[str, Any],
) -> str:
    plan = json.dumps(task_plan or {}, ensure_ascii=False, separators=(",", ":"))[:6000]
    report = json.dumps(validation_report or {}, ensure_ascii=False, separators=(",", ":"))[:7000]
    return (
        "System: You are an independent evidence-audit RWKV. Do not answer the user. "
        "Inspect only the EVIDENCE BODY records. The mechanical report is routing metadata, "
        "not evidence. For every task point, mark supported only when the body directly states "
        "the requested fact or relationship and cite valid S# records. Mark missing or conflict "
        "when the body is insufficient. Do not use memory, titles, snippets, URLs or inferred "
        "relationships. Return exactly one JSON object and no explanation with this schema:\n"
        '{"schema_version":"evidence_verification.v1","status":"supported|needs_more_evidence|conflict|insufficient",'
        '"completion_ready":true,"points":[{"id":"P1","status":"supported|missing|conflict|unclear",'
        '"evidence":["S1"],"reason":"...","missing":["..."],"next_queries":["..."]}],'
        '"missing_point_ids":["P1"],"next_queries":["..."],"reason":"..."}.\n\n'
        f"User question: {query}\n"
        f"Task plan (routing metadata): {plan}\n"
        f"Mechanical validation report (routing metadata): {report}\n"
        "BEGIN EVIDENCE BODY\n"
        f"{evidence_text}\n"
        "END EVIDENCE BODY\n"
        f"{assistant_json_prefix(enable_think=True)}"
    )


def verify_evidence(
    llm: Any,
    *,
    query: str,
    task_plan: dict[str, Any],
    evidence_context: dict[str, Any],
) -> dict[str, Any]:
    """Run one isolated verifier call and normalize its control output."""

    report = evidence_context.get("validation") or {}
    selected = evidence_context.get("selected_evidence") or []
    prompt = build_verifier_prompt(query, task_plan, evidence_context.get("text", ""), report)
    base = {
        "schema_version": "evidence_verification.v1",
        "status": "verifier_error",
        "completion_ready": False,
        "requires_replan": False,
        "points": [],
        "missing_point_ids": [],
        "conflict_point_ids": [],
        "next_queries": [],
        "is_truth_judgement": False,
        "prompt": prompt,
        "model_output": "",
        "error": "",
    }
    if llm is None:
        base["error"] = "verifier model is not configured"
        return base
    try:
        budget = bounded_completion_budget(
            prompt,
            context_limit=get_llm_context_length(),
            requested_max=2048,
            safety_margin=256,
        )
        if hasattr(llm, "text_completion"):
            try:
                response = llm.text_completion(prompt, max_tokens=budget, stop=JSON_CALL_STOP_SUFFIXES)
            except TypeError as exc:
                if "stop" not in str(exc):
                    raise
                response = llm.text_completion(prompt, max_tokens=budget)
        else:
            response = llm.chat_completion([{"role": "user", "content": prompt}], max_tokens=budget)
        raw = str(response.content or "")
        normalized = _normalize_result(
            _extract_json_object(raw),
            report=report,
            source_count=len(selected),
        )
        normalized.update({"prompt": prompt, "model_output": raw, "error": ""})
        return normalized
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base
