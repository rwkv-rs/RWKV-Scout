"""Single factual task-record contract shared by the retrieval pipeline.

RWKV writes the plan.  This module only validates and projects that model
output so every downstream component reads the same meaning.  An atomic point
is one user-requested factual *record*, not a search/verification workflow
stage.  Record identity fields are model-authored routing metadata; this module
never infers their values from the question.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


TASK_PLAN_SCHEMA_VERSION = "task_plan.v2"
TIME_SCOPES = {"current", "historical", "timeless", "unspecified"}
SET_SEMANTICS = {"single", "collection", "possibly_empty"}


def _strings(value: Any, *, limit: int = 16) -> list[str]:
    values = [value] if isinstance(value, str) else list(value or [])
    return list(
        dict.fromkeys(
            text
            for item in values
            if (text := re.sub(r"\s+", " ", str(item or "")).strip())
        )
    )[: max(0, int(limit))]


def point_question(point: Mapping[str, Any] | None, fallback: str = "") -> str:
    """Read one point through the v2 contract or the legacy trace boundary."""

    row = point if isinstance(point, Mapping) else {}
    return str(
        row.get("question")
        or row.get("task")
        or row.get("objective")
        or fallback
        or ""
    ).strip()


def point_fields(point: Mapping[str, Any] | None) -> list[str]:
    row = point if isinstance(point, Mapping) else {}
    fields = _strings(row.get("fields") or row.get("requested_fields"), limit=16)
    for requirement in row.get("answer_requirements") or []:
        if isinstance(requirement, Mapping):
            fields.extend(_strings(requirement.get("type"), limit=1))
    return list(dict.fromkeys(fields))[:16]


def point_time_scope(point: Mapping[str, Any] | None) -> str:
    row = point if isinstance(point, Mapping) else {}
    value = str(row.get("time_scope") or "unspecified").strip().casefold()
    return value if value in TIME_SCOPES else "unspecified"


def point_subject(point: Mapping[str, Any] | None) -> str:
    """Return RWKV's subject label without inventing a missing entity."""

    row = point if isinstance(point, Mapping) else {}
    return re.sub(r"\s+", " ", str(row.get("subject") or "")).strip()[:400]


def point_relation(point: Mapping[str, Any] | None) -> str:
    """Return RWKV's requested relation label."""

    row = point if isinstance(point, Mapping) else {}
    return re.sub(r"\s+", " ", str(row.get("relation") or "")).strip()[:240]


def point_set_semantics(point: Mapping[str, Any] | None) -> str:
    row = point if isinstance(point, Mapping) else {}
    value = str(row.get("set_semantics") or "single").strip().casefold()
    return value if value in SET_SEMANTICS else "single"


def point_premise_requires_verification(point: Mapping[str, Any] | None) -> bool:
    row = point if isinstance(point, Mapping) else {}
    value = row.get("premise_requires_verification", False)
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "1", "是"}
    return bool(value)


def task_points(
    task_plan: Mapping[str, Any] | None,
    *,
    fallback_query: str = "",
    max_points: int = 4,
) -> list[dict[str, Any]]:
    """Return canonical factual points without inventing missing semantics."""

    plan = task_plan if isinstance(task_plan, Mapping) else {}
    raw_points = [
        value
        for value in plan.get("records") or plan.get("atomic_points") or []
        if isinstance(value, Mapping)
    ]
    if not raw_points and str(fallback_query or "").strip():
        raw_points = [{"id": "P1", "question": str(fallback_query).strip()}]

    legacy_global_fields = _strings(plan.get("requested_fields"), limit=16)
    normalized: list[dict[str, Any]] = []
    by_signature: dict[str, dict[str, Any]] = {}
    used_ids: set[str] = set()
    for raw in raw_points[: max(1, int(max_points))]:
        question = point_question(raw, fallback_query)
        if not question:
            continue
        signature = re.sub(r"\s+", " ", question.casefold()).strip()
        fields = point_fields(raw)
        if not fields and len(raw_points) == 1:
            fields = list(legacy_global_fields)
        source_id = str(raw.get("id") or f"P{len(normalized) + 1}").strip()
        existing = by_signature.get(signature)
        if existing is not None:
            existing["fields"] = list(
                dict.fromkeys([*existing.get("fields", []), *fields])
            )[:16]
            existing["source_ids"] = list(
                dict.fromkeys([*existing.get("source_ids", []), source_id])
            )
            continue
        point_id = source_id or f"P{len(normalized) + 1}"
        if point_id in used_ids:
            raise ValueError("task plan point ids must be unique")
        row = {
            "id": point_id,
            "question": question[:1200],
            "subject": point_subject(raw),
            "relation": point_relation(raw),
            "fields": fields,
            "time_scope": point_time_scope(raw),
            "set_semantics": point_set_semantics(raw),
            "premise_requires_verification": point_premise_requires_verification(raw),
            "source_ids": [point_id],
        }
        normalized.append(row)
        by_signature[signature] = row
        used_ids.add(point_id)
    return normalized


def normalize_task_plan(
    payload: Mapping[str, Any],
    *,
    fallback_goal: str = "",
    max_points: int = 4,
) -> dict[str, Any]:
    """Validate RWKV output and return the only runtime plan representation."""

    if not isinstance(payload, Mapping):
        raise ValueError("task plan must be a JSON object")
    goal = str(payload.get("goal") or fallback_goal or "").strip()
    if not goal:
        raise ValueError("task plan requires goal")
    raw_points = payload.get("records")
    if raw_points is None:
        raw_points = payload.get("atomic_points")
    if raw_points is not None and not isinstance(raw_points, list):
        raise ValueError("task plan records must be an array")
    if isinstance(raw_points, list) and len(raw_points) > max_points:
        raise ValueError("task plan contains too many records")
    points = task_points(payload, fallback_query=goal, max_points=max_points)
    if not points:
        raise ValueError("task plan contains no factual points")
    return {
        "schema_version": TASK_PLAN_SCHEMA_VERSION,
        "goal": goal[:2400],
        "atomic_points": points,
    }


def plan_fields(task_plan: Mapping[str, Any] | None) -> list[str]:
    plan = task_plan if isinstance(task_plan, Mapping) else {}
    fields = _strings(plan.get("requested_fields"), limit=32)
    for requirement in plan.get("answer_requirements") or []:
        if isinstance(requirement, Mapping):
            fields.extend(_strings(requirement.get("type"), limit=1))
    for point in task_points(plan):
        fields.extend(point_fields(point))
    return list(dict.fromkeys(fields))[:32]


def compact_task_plan(task_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(task_plan, Mapping):
        return {"status": "missing"}
    goal = str(task_plan.get("goal") or "").strip()
    return {
        "schema_version": TASK_PLAN_SCHEMA_VERSION,
        "goal": goal[:800],
        "atomic_points": [
            {
                "id": point["id"],
                "question": point["question"][:500],
                "subject": point.get("subject") or "",
                "relation": point.get("relation") or "",
                "fields": list(point.get("fields") or [])[:16],
                "time_scope": point.get("time_scope") or "unspecified",
                "set_semantics": point.get("set_semantics") or "single",
                "premise_requires_verification": bool(
                    point.get("premise_requires_verification")
                ),
            }
            for point in task_points(task_plan, fallback_query=goal)
        ],
    }


__all__ = [
    "TASK_PLAN_SCHEMA_VERSION",
    "SET_SEMANTICS",
    "TIME_SCOPES",
    "compact_task_plan",
    "normalize_task_plan",
    "plan_fields",
    "point_fields",
    "point_question",
    "point_premise_requires_verification",
    "point_relation",
    "point_set_semantics",
    "point_subject",
    "point_time_scope",
    "task_points",
]
