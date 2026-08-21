"""Canonical Task Plan contract for the complete retrieval pipeline.

There is exactly one runtime representation. Historical version-labelled
payloads are accepted only by :func:`normalize_task_plan` and are immediately
rewritten. Downstream stages never need to know which legacy shape produced a
plan.

Record and field identifiers are transport identity owned by the controller.
RWKV supplies semantics (question, subject, relation and field names); it does
not get to preserve, reuse or invent runtime identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agent.runtime_contracts import TASK_PLAN_CONTRACT


LEGACY_TASK_PLAN_SCHEMA_VERSIONS = frozenset(
    {"task_plan.v1", "task_plan.v2", "task_plan.v3", "task_plan.v4"}
)
TIME_SCOPES = {"current", "historical", "timeless", "unspecified"}
SET_SEMANTICS = {"single", "collection", "possibly_empty"}
MAX_MODEL_TASK_RECORDS = 4
MAX_RECORD_FIELDS = 16


def _text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _strings(value: Any, *, limit: int = MAX_RECORD_FIELDS) -> list[str]:
    values = [value] if isinstance(value, str) else list(value or [])
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, Mapping):
            item = item.get("name") or item.get("field_name") or item.get("type")
        text = _text(item, 160)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
        if len(output) >= max(0, int(limit)):
            break
    return output


def record_id(record: Mapping[str, Any] | None) -> str:
    row = record if isinstance(record, Mapping) else {}
    return _text(row.get("record_id"), 120)


def record_question(record: Mapping[str, Any] | None, fallback: str = "") -> str:
    """Return the question carried by one Task Record."""

    row = record if isinstance(record, Mapping) else {}
    return str(
        row.get("question")
        or row.get("task")
        or row.get("objective")
        or fallback
        or ""
    ).strip()


def record_fields(record: Mapping[str, Any] | None) -> list[str]:
    """Return ordered semantic field names from a canonical record."""

    row = record if isinstance(record, Mapping) else {}
    fields = _strings(row.get("fields") or row.get("requested_fields"))
    fields.extend(_strings(row.get("evidence_needed")))
    for requirement in row.get("answer_requirements") or []:
        if isinstance(requirement, Mapping):
            fields.extend(_strings(requirement.get("type"), limit=1))
    return _strings(fields)


def record_field_records(record: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Return canonical field identities for one Task Plan record."""

    row = record if isinstance(record, Mapping) else {}
    parent_id = record_id(row) or "P1"
    raw_fields = row.get("fields")
    raw_rows = raw_fields if isinstance(raw_fields, list) else []
    output: list[dict[str, str]] = []
    for index, name in enumerate(record_fields(row), start=1):
        raw = raw_rows[index - 1] if index <= len(raw_rows) else None
        supplied_id = (
            _text(raw.get("field_id"), 160)
            if isinstance(raw, Mapping)
            else ""
        )
        output.append(
            {
                "field_id": supplied_id or f"{parent_id}:F{index}",
                "name": name,
            }
        )
    return output


def field_name_by_id(record: Mapping[str, Any] | None) -> dict[str, str]:
    return {
        field["field_id"].casefold(): field["name"]
        for field in record_field_records(record)
    }


def field_ids(record: Mapping[str, Any] | None) -> list[str]:
    return [field["field_id"] for field in record_field_records(record)]


def record_time_scope(record: Mapping[str, Any] | None) -> str:
    row = record if isinstance(record, Mapping) else {}
    value = str(row.get("time_scope") or "unspecified").strip().casefold()
    return value if value in TIME_SCOPES else "unspecified"


def record_subject(record: Mapping[str, Any] | None) -> str:
    row = record if isinstance(record, Mapping) else {}
    return _text(row.get("subject"), 400)


def record_relation(record: Mapping[str, Any] | None) -> str:
    row = record if isinstance(record, Mapping) else {}
    return _text(row.get("relation"), 240)


def record_set_semantics(record: Mapping[str, Any] | None) -> str:
    row = record if isinstance(record, Mapping) else {}
    value = str(row.get("set_semantics") or "single").strip().casefold()
    return value if value in SET_SEMANTICS else "single"


def record_premise_requires_verification(record: Mapping[str, Any] | None) -> bool:
    row = record if isinstance(record, Mapping) else {}
    value = row.get("premise_requires_verification", False)
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "1", "是"}
    return bool(value)


def _raw_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("records")
    if raw is None:
        raw = payload.get("atomic_points")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("task plan records must be an array")
    return [item for item in raw if isinstance(item, Mapping)]


def _canonical_records(
    payload: Mapping[str, Any],
    *,
    fallback_query: str,
    max_records: int | None,
    preserve_runtime_ids: bool,
) -> list[dict[str, Any]]:
    raw_records = _raw_records(payload)
    if max_records is not None and len(raw_records) > max_records:
        raise ValueError("task plan contains too many records")
    if not raw_records and fallback_query.strip():
        raw_records = [{"question": fallback_query.strip()}]

    global_fields = _strings(payload.get("requested_fields"))
    semantics: list[dict[str, Any]] = []
    for raw in raw_records:
        explicit_question = _text(raw.get("question"), 1200)
        legacy_task = _text(raw.get("task"), 800)
        legacy_objective = _text(raw.get("objective"), 800)
        if explicit_question:
            question = explicit_question
        elif legacy_task and legacy_objective and legacy_objective.casefold() != legacy_task.casefold():
            question = _text(f"{legacy_task} — {legacy_objective}", 1200)
        else:
            question = _text(legacy_task or legacy_objective or fallback_query, 1200)
        if not question:
            continue
        names = record_fields(raw)
        if not names and len(raw_records) == 1:
            names = list(global_fields)
        semantic = {
            "_record_id": _text(raw.get("record_id") or raw.get("id"), 120),
            "_field_ids": [
                _text(item.get("field_id"), 160)
                if isinstance(item, Mapping)
                else ""
                for item in (
                    raw.get("fields") if isinstance(raw.get("fields"), list) else []
                )
            ],
            "question": question,
            "subject": record_subject(raw),
            "relation": record_relation(raw),
            "_field_names": names,
            "time_scope": record_time_scope(raw),
            "set_semantics": record_set_semantics(raw),
            "premise_requires_verification": record_premise_requires_verification(raw),
        }
        # Contract normalization is a representation change, never a semantic
        # deduplicator.  Two rows that happen to have equal wording may carry
        # different historical identities, evidence and downstream bindings.
        semantics.append(semantic)

    output: list[dict[str, Any]] = []
    used_record_ids: set[str] = set()
    for record_index, semantic in enumerate(semantics, start=1):
        supplied_record_id = str(semantic.pop("_record_id") or "").strip()
        if preserve_runtime_ids:
            canonical_id = supplied_record_id or f"P{record_index}"
            if canonical_id in used_record_ids:
                raise ValueError("canonical Task Plan record IDs must be unique")
        else:
            canonical_id = f"P{record_index}"
        used_record_ids.add(canonical_id)
        names = list(semantic.pop("_field_names"))
        supplied_field_ids = list(semantic.pop("_field_ids"))
        fields: list[dict[str, str]] = []
        used_field_ids: set[str] = set()
        for field_index, name in enumerate(names, start=1):
            supplied_field_id = (
                supplied_field_ids[field_index - 1]
                if field_index <= len(supplied_field_ids)
                else ""
            )
            if preserve_runtime_ids:
                canonical_field_id = (
                    supplied_field_id or f"{canonical_id}:F{field_index}"
                )
                if canonical_field_id in used_field_ids:
                    raise ValueError("canonical Task Plan field IDs must be unique")
            else:
                canonical_field_id = f"{canonical_id}:F{field_index}"
            used_field_ids.add(canonical_field_id)
            fields.append({"field_id": canonical_field_id, "name": name})
        output.append(
            {
                "record_id": canonical_id,
                "question": semantic["question"],
                "subject": semantic["subject"],
                "relation": semantic["relation"],
                "fields": fields,
                "time_scope": semantic["time_scope"],
                "set_semantics": semantic["set_semantics"],
                "premise_requires_verification": semantic[
                    "premise_requires_verification"
                ],
            }
        )
    return output


def normalize_task_plan(
    payload: Mapping[str, Any],
    *,
    fallback_goal: str = "",
    max_records: int | None = None,
    preserve_runtime_ids: bool = False,
) -> dict[str, Any]:
    """Rewrite a plan into the sole canonical runtime representation.

    ``max_records`` is a model-output admission limit, not a runtime storage
    limit.  Historical and already-admitted plans therefore default to no
    record-count cap so migration cannot silently erase state.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("task plan must be a JSON object")
    supplied_contract = str(payload.get("contract") or "").strip()
    legacy_schema = str(payload.get("schema_version") or "").strip()
    if supplied_contract and legacy_schema:
        raise ValueError("task plan cannot contain contract and schema_version together")
    if supplied_contract and supplied_contract != TASK_PLAN_CONTRACT:
        raise ValueError("unsupported Task Plan contract")
    if legacy_schema and legacy_schema not in LEGACY_TASK_PLAN_SCHEMA_VERSIONS:
        raise ValueError("unsupported historical Task Plan schema")
    goal = str(payload.get("goal") or fallback_goal or "").strip()
    if not goal:
        raise ValueError("task plan requires goal")
    records = _canonical_records(
        payload,
        fallback_query=goal,
        max_records=(max(1, int(max_records)) if max_records is not None else None),
        preserve_runtime_ids=bool(preserve_runtime_ids),
    )
    if not records:
        raise ValueError("task plan contains no factual records")
    return {
        "contract": TASK_PLAN_CONTRACT,
        "goal": goal[:2400],
        "records": records,
    }


def task_records(
    task_plan: Mapping[str, Any] | None,
    *,
    fallback_query: str = "",
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Read canonical records, migrating a historical boundary if necessary."""

    plan = task_plan if isinstance(task_plan, Mapping) else {}
    goal = str(plan.get("goal") or fallback_query or "").strip()
    if not goal:
        return []
    return normalize_task_plan(
        plan,
        fallback_goal=goal,
        max_records=max_records,
        preserve_runtime_ids=str(plan.get("contract") or "") == TASK_PLAN_CONTRACT,
    )["records"]


def plan_fields(task_plan: Mapping[str, Any] | None) -> list[str]:
    plan = task_plan if isinstance(task_plan, Mapping) else {}
    fields = _strings(plan.get("requested_fields"), limit=32)
    for requirement in plan.get("answer_requirements") or []:
        if isinstance(requirement, Mapping):
            fields.extend(_strings(requirement.get("type"), limit=1))
    for record in task_records(plan):
        fields.extend(record_fields(record))
    return _strings(fields, limit=32)


def compact_task_plan(task_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(task_plan, Mapping):
        return {"status": "missing"}
    goal = str(task_plan.get("goal") or "").strip()
    if not goal:
        return {"status": "missing"}
    try:
        canonical = normalize_task_plan(
            task_plan,
            fallback_goal=goal,
            preserve_runtime_ids=str(task_plan.get("contract") or "")
            == TASK_PLAN_CONTRACT,
        )
    except ValueError:
        return {"status": "invalid"}
    return {
        "contract": TASK_PLAN_CONTRACT,
        "goal": canonical["goal"][:800],
        "records": [
            {
                "record_id": record["record_id"],
                "question": record["question"][:500],
                "subject": record["subject"],
                "relation": record["relation"],
                "fields": list(record["fields"]),
                "time_scope": record["time_scope"],
                "set_semantics": record["set_semantics"],
                "premise_requires_verification": bool(
                    record["premise_requires_verification"]
                ),
            }
            for record in canonical["records"]
        ],
    }


__all__ = [
    "LEGACY_TASK_PLAN_SCHEMA_VERSIONS",
    "MAX_RECORD_FIELDS",
    "MAX_MODEL_TASK_RECORDS",
    "SET_SEMANTICS",
    "TASK_PLAN_CONTRACT",
    "TIME_SCOPES",
    "compact_task_plan",
    "field_ids",
    "field_name_by_id",
    "normalize_task_plan",
    "plan_fields",
    "record_field_records",
    "record_fields",
    "record_premise_requires_verification",
    "record_question",
    "record_relation",
    "record_set_semantics",
    "record_subject",
    "record_time_scope",
    "record_id",
    "task_records",
]
