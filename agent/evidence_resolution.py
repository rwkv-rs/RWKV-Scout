"""Global RWKV resolution over the exact evidence records sent to Writer.

The resolver emits only an attention/control map.  It cannot rewrite evidence,
promote packet aliases to identity, or declare a record resolved when any
candidate span was hidden from its request.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from agent.runtime_contracts import (
    EVIDENCE_RESOLUTION_CONTRACT,
    require_runtime_contract,
)
from agent.task_plan_contract import (
    compact_task_plan,
    field_ids,
    record_field_records,
    record_id,
    task_records,
)
from config import (
    DATA_PIPELINE,
    get_llm_context_length,
    get_model_stage_sampling,
    get_model_stage_temperature,
    is_local_provider,
    model_sampling_parameters,
)
from utils.chunker import get_token_count
from utils.model_budget import bounded_completion_budget
from utils.model_events import visible_model_text
from utils.rwkv_json_protocol import normalize_json_object_envelope
from utils.rwkv_prompt import JSON_CALL_STOP_SUFFIXES, render_tool_transcript


_VALID_STATUSES = frozenset({"resolved", "conflict", "missing"})


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[: max(0, int(limit))]


def _unique_strings(value: Any, *, limit: int = 16, char_limit: int = 300) -> list[str]:
    rows = value if isinstance(value, (list, tuple, set)) else [value]
    return list(
        dict.fromkeys(
            text for row in rows if (text := _text(row, char_limit))
        )
    )[: max(0, int(limit))]


def _source_object(item: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, str]:
    value = metadata.get("source_object")
    if not isinstance(value, Mapping):
        value = item.get("source_object")
    source = value if isinstance(value, Mapping) else {}
    return {
        key: text
        for key, text in {
            "source_object_id": _text(source.get("source_object_id"), 500),
            "source_object_type": _text(source.get("source_object_type"), 120),
            "source_record_id": _text(source.get("source_record_id"), 300),
        }.items()
        if text
    }


def _object_alignments(
    item: Mapping[str, Any], metadata: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw_rows: list[Any] = []
    for owner in (metadata, item):
        for key in ("object_alignments", "object_alignment"):
            value = owner.get(key)
            raw_rows.extend(value if isinstance(value, list) else [value])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        row = {
            "relation": _text(raw.get("relation"), 80),
            "source_object_id": _text(raw.get("source_object_id"), 300),
            "requested_object_ids": _unique_strings(
                raw.get("requested_object_ids"), limit=4, char_limit=300
            ),
        }
        row = {key: value for key, value in row.items() if value not in ("", [])}
        signature = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if row and signature not in seen:
            seen.add(signature)
            rows.append(row)
    return rows[:4]


def _stable_evidence_record_id(item: Mapping[str, Any], index: int) -> str:
    existing = _text(item.get("evidence_record_id"), 160)
    if existing:
        return existing
    metadata = item.get("record_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    identity = {
        "url": str(item.get("url") or ""),
        "record_key": str(metadata.get("record_key") or ""),
        "record_span_id": str(metadata.get("record_span_id") or ""),
        "chunks": [
            {
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "text": str(chunk.get("text") or ""),
            }
            for chunk in item.get("packed_chunks") or []
            if isinstance(chunk, Mapping)
        ],
    }
    if not any(
        (
            identity["url"],
            identity["record_key"],
            identity["record_span_id"],
            identity["chunks"],
        )
    ):
        # Only a truly empty synthetic slot needs positional disambiguation.
        # Normal evidence identity must remain stable when ranking order moves.
        identity["empty_slot"] = index
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"E-{digest}"


def _object_group_identity(card: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Create one transport identity group without deciding semantic correctness."""

    source_object = (
        card.get("source_object")
        if isinstance(card.get("source_object"), Mapping)
        else {}
    )
    source_object_id = _text(source_object.get("source_object_id"), 500)
    source_object_type = _text(source_object.get("source_object_type"), 120)
    source_record_id = _text(source_object.get("source_record_id"), 300)
    record_key = _text(card.get("record_key"), 300)
    subject_key = _text(card.get("subject_key"), 300)
    evidence_record_id = _text(card.get("evidence_record_id"), 160)
    if source_object_id and source_record_id:
        method = "source_object_record"
        identity = {
            "source_object_id": source_object_id,
            "source_object_type": source_object_type,
            "source_record_id": source_record_id,
        }
    elif record_key and (source_object_id or subject_key):
        method = "record_key"
        identity = {
            "source_object_id": source_object_id,
            "source_object_type": source_object_type,
            "subject_key": subject_key,
            "record_key": record_key,
        }
    else:
        method = "evidence_record"
        identity = {"evidence_record_id": evidence_record_id}
    identity = {key: value for key, value in identity.items() if value}
    digest = hashlib.sha256(
        json.dumps(
            {"method": method, "identity": identity},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"O-{digest}", {"identity_method": method, **identity}


def build_evidence_object_groups(
    evidence_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project deterministic record-identity groups for RWKV selection."""

    order: list[str] = []
    groups: dict[str, dict[str, Any]] = {}
    for card in evidence_records:
        if not isinstance(card, Mapping):
            continue
        group_id = _text(card.get("object_group_id"), 160)
        evidence_id = _text(card.get("evidence_record_id"), 160)
        if not group_id or not evidence_id:
            continue
        if group_id not in groups:
            order.append(group_id)
            identity = (
                dict(card.get("object_group_identity") or {})
                if isinstance(card.get("object_group_identity"), Mapping)
                else {}
            )
            groups[group_id] = {
                "object_group_id": group_id,
                "identity": identity,
                "evidence_record_ids": [],
                "declared_task_record_ids": [],
            }
        group = groups[group_id]
        group["evidence_record_ids"].append(evidence_id)
        group["declared_task_record_ids"].extend(
            _unique_strings(
                card.get("declared_task_record_ids"), limit=8, char_limit=80
            )
        )
    for group in groups.values():
        group["evidence_record_ids"] = list(
            dict.fromkeys(group["evidence_record_ids"])
        )
        group["declared_task_record_ids"] = list(
            dict.fromkeys(group["declared_task_record_ids"])
        )
    return [groups[group_id] for group_id in order]


def build_evidence_record_set(
    selected_evidence: list[dict[str, Any]] | None,
    *,
    max_records: int = 24,
    max_chars_per_record: int = 12000,
) -> list[dict[str, Any]]:
    """Project exact spans and explicitly report anything not shown to RWKV."""

    record_limit = max(1, min(int(max_records or 24), 24))
    char_limit = max(160, min(int(max_chars_per_record or 12000), 24000))
    evidence_records: list[dict[str, Any]] = []
    for index, raw in enumerate(selected_evidence or []):
        if not isinstance(raw, Mapping) or len(evidence_records) >= record_limit:
            continue
        item = dict(raw)
        evidence_id = _stable_evidence_record_id(item, index)
        packet_alias = _text(item.get("ref_id") or f"S{index + 1}", 32)
        metadata = (
            item.get("record_metadata")
            if isinstance(item.get("record_metadata"), Mapping)
            else {}
        )
        source_object = _source_object(item, metadata)
        task_ids = _unique_strings(
            [
                *(item.get("task_record_ids") or []),
                *(metadata.get("task_record_ids") or []),
                metadata.get("task_record_id"),
            ],
            limit=8,
            char_limit=80,
        )

        exact_spans: list[dict[str, str]] = []
        unviewed_span_ids: list[str] = []
        remaining = char_limit
        raw_chunks = [
            chunk
            for chunk in item.get("packed_chunks") or []
            if isinstance(chunk, Mapping) and str(chunk.get("text") or "").strip()
        ]
        if not raw_chunks and str(item.get("evidence_text") or "").strip():
            raw_chunks = [
                {
                    "chunk_id": f"{packet_alias}-body",
                    "text": str(item.get("evidence_text") or ""),
                }
            ]
        for chunk_index, chunk in enumerate(raw_chunks, start=1):
            span_id = _text(
                chunk.get("chunk_id") or f"{packet_alias}-{chunk_index}", 160
            )
            literal = str(chunk.get("text") or "").replace("\x00", "").strip()
            if not literal:
                continue
            if len(literal) > remaining:
                unviewed_span_ids.append(span_id)
                continue
            exact_spans.append({"span_id": span_id, "text": literal})
            remaining -= len(literal)

        card: dict[str, Any] = {
            "evidence_record_id": evidence_id,
            "packet_alias": packet_alias,
            "coverage_complete": not unviewed_span_ids,
            "unviewed_span_ids": unviewed_span_ids,
            "context_role": _text(item.get("context_role"), 80),
            "declared_task_record_ids": task_ids,
            "title": _text(item.get("title"), 300),
            "url": _text(item.get("url"), 800),
            "subject_key": _text(
                metadata.get("subject_key") or item.get("source_subject"), 300
            ),
            "record_key": _text(
                metadata.get("record_key")
                or source_object.get("source_record_id")
                or item.get("source_record_key"),
                300,
            ),
            "record_span_id": _text(metadata.get("record_span_id"), 120),
            "source_object": source_object,
            "object_alignments": _object_alignments(item, metadata),
            "published": _text(item.get("published") or item.get("published_at"), 120),
            "updated": _text(item.get("updated") or item.get("updated_at"), 120),
            "source_date": _text(item.get("date"), 120),
            "exact_spans": exact_spans,
        }
        object_group_id, object_group_identity = _object_group_identity(card)
        card["object_group_id"] = object_group_id
        card["object_group_identity"] = object_group_identity
        evidence_records.append(
            {key: value for key, value in card.items() if value not in ("", [], {})}
        )
    return evidence_records


def evidence_resolution_signature(
    query: str,
    task_plan: Mapping[str, Any] | None,
    evidence_records: list[dict[str, Any]],
) -> str:
    payload = {
        "query": str(query or ""),
        "task_plan": compact_task_plan(task_plan),
        "evidence_records": evidence_records,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _evidence_resolution_output_schema(
    query: str,
    task_plan: Mapping[str, Any] | None,
    evidence_records: list[dict[str, Any]],
) -> dict[str, Any]:
    records = task_records(task_plan, fallback_query=query)
    task_record_ids = [record_id(record) for record in records]
    all_field_ids = [
        field_id for record in records for field_id in field_ids(record)
    ]
    evidence_record_ids = [
        _text(card.get("evidence_record_id"), 160)
        for card in evidence_records
        if _text(card.get("evidence_record_id"), 160)
    ]
    object_group_ids = [
        _text(group.get("object_group_id"), 160)
        for group in build_evidence_object_groups(evidence_records)
        if _text(group.get("object_group_id"), 160)
    ]
    decision_properties = {
        "task_record_id": {"type": "string", "enum": task_record_ids},
        "status": {
            "type": "string",
            "enum": ["resolved", "conflict", "missing"],
        },
        "selected_object_group_id": {
            "type": "string",
            "enum": ["", *object_group_ids],
        },
        "conflicting_object_group_ids": {
            "type": "array",
            "items": {"type": "string", "enum": object_group_ids},
        },
        "selected_evidence_record_ids": {
            "type": "array",
            "items": {"type": "string", "enum": evidence_record_ids},
        },
        "conflicting_evidence_record_ids": {
            "type": "array",
            "items": {"type": "string", "enum": evidence_record_ids},
        },
        "field_evidence_record_ids": {
            "type": "object",
            "propertyNames": {"enum": all_field_ids},
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "enum": evidence_record_ids},
            },
        },
        "missing_field_ids": {
            "type": "array",
            "items": {"type": "string", "enum": all_field_ids},
        },
        "conflict_field_ids": {
            "type": "array",
            "items": {"type": "string", "enum": all_field_ids},
        },
        "needs_more_evidence": {"type": "boolean"},
    }
    return {
        "type": "object",
        "properties": {
            "contract": {"const": EVIDENCE_RESOLUTION_CONTRACT},
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": decision_properties,
                    "required": list(decision_properties),
                    "additionalProperties": False,
                },
            },
        },
        "required": ["contract", "decisions"],
        "additionalProperties": False,
    }


def _resolution_user_prompt(
    query: str,
    task_plan: Mapping[str, Any] | None,
    evidence_records: list[dict[str, Any]],
) -> str:
    object_groups = build_evidence_object_groups(evidence_records)
    output_schema = _evidence_resolution_output_schema(
        query,
        task_plan,
        evidence_records,
    )
    return (
        "Resolve the complete evidence-record set globally before answer writing. "
        "Use only controller-owned IDs present in the Output Schema. Packet display aliases "
        "are not identities and must never be output. For each Task Record, first select one "
        "object_group_id, then bind fields only to Evidence Records in that selected group. "
        "Bind a field only when an exact span contains its value. Never combine different "
        "versions, dates, rows, repositories, platforms or objects into one resolved record. "
        "Object-group metadata is identity context, not proof. "
        "Every requested field_id must appear in exactly one of field_evidence_record_ids, "
        "missing_field_ids or conflict_field_ids. Bindings must use selected evidence. A "
        "multi-object conflict must use status=conflict and name at least two competing "
        "object groups and Evidence Records. If any card says "
        "coverage_complete=false, resolved is forbidden because some exact spans were not "
        "shown. Output no facts, reasoning, prose, scores, new query, or invented IDs.\n\n"
        "Return exactly one JSON object conforming to this Output Schema, with exactly one "
        "decision for every Task Plan record:\n"
        + json.dumps(output_schema, ensure_ascii=False, separators=(",", ":"))
        + "\n\n"
        f"USER QUESTION:\n{str(query or '')}\n\n"
        "TASK PLAN:\n"
        + json.dumps(compact_task_plan(task_plan), ensure_ascii=False, separators=(",", ":"))
        + "\n\nOBJECT GROUPS:\n"
        + json.dumps(object_groups, ensure_ascii=False, separators=(",", ":"))
        + "\n\nEVIDENCE RECORDS:\n"
        + json.dumps(evidence_records, ensure_ascii=False, separators=(",", ":"))
    )


def build_evidence_resolution_prompt(
    query: str,
    task_plan: Mapping[str, Any] | None,
    evidence_records: list[dict[str, Any]],
    *,
    correction: str = "",
) -> tuple[str, str]:
    user_prompt = _resolution_user_prompt(query, task_plan, evidence_records)
    if correction:
        user_prompt += (
            "\n\nPROTOCOL CORRECTION: The previous continuation was invalid ("
            + _text(correction, 400)
            + "). Return only the complete JSON object in the required schema."
        )
    prompt = render_tool_transcript(
        [{"role": "user", "content": user_prompt}], json_output=True
    )
    return prompt, user_prompt


def _record_object_selection_schema(
    task_record_id: str,
    object_group_ids: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "contract": {"const": EVIDENCE_RESOLUTION_CONTRACT},
            "task_record_id": {"const": task_record_id},
            "selection": {
                "type": "string",
                "enum": ["selected", "conflict", "missing"],
            },
            "object_group_ids": {
                "type": "array",
                "items": {"type": "string", "enum": object_group_ids},
            },
        },
        "required": [
            "contract",
            "task_record_id",
            "selection",
            "object_group_ids",
        ],
        "additionalProperties": False,
    }


def _record_field_binding_schema(
    task_record_id: str,
    requested_field_ids: list[str],
    evidence_record_ids: list[str],
) -> dict[str, Any]:
    field_properties = {
        "field_id": {"type": "string", "enum": requested_field_ids},
        "state": {
            "type": "string",
            "enum": ["supported", "missing", "conflict"],
        },
        "evidence_record_ids": {
            "type": "array",
            "items": {"type": "string", "enum": evidence_record_ids},
        },
    }
    return {
        "type": "object",
        "properties": {
            "contract": {"const": EVIDENCE_RESOLUTION_CONTRACT},
            "task_record_id": {"const": task_record_id},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": field_properties,
                    "required": list(field_properties),
                    "additionalProperties": False,
                },
            },
        },
        "required": ["contract", "task_record_id", "fields"],
        "additionalProperties": False,
    }


def _ordered_cards_for_record(
    task_record_id: str,
    evidence_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer declared record bindings without hiding any candidate card."""

    matching: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    folded = task_record_id.casefold()
    for raw in evidence_records:
        card = dict(raw)
        declared = {
            str(value).casefold()
            for value in card.get("declared_task_record_ids") or []
            if str(value)
        }
        (matching if folded in declared else remaining).append(card)
    return [*matching, *remaining]


def _build_record_object_selection_prompt(
    query: str,
    record: Mapping[str, Any],
    evidence_records: list[dict[str, Any]],
    *,
    correction: str = "",
) -> tuple[str, str]:
    task_record_id = record_id(record)
    ordered_cards = _ordered_cards_for_record(task_record_id, evidence_records)
    object_groups = build_evidence_object_groups(ordered_cards)
    object_group_ids = [
        str(group.get("object_group_id") or "")
        for group in object_groups
        if str(group.get("object_group_id") or "")
    ]
    schema = _record_object_selection_schema(task_record_id, object_group_ids)
    user_prompt = (
        "Select the evidence object identity for exactly one Task Record. This call "
        "does not answer the question and does not bind fields. Use selected with exactly "
        "one object group when one version, release, row, repository, platform or other "
        "record matches the requested object. Use conflict with at least two groups only "
        "when their exact spans make materially incompatible claims about that requested "
        "object. Use missing with an empty group list when no shown group establishes the "
        "requested object. Historical and current records are alternatives, not automatically "
        "a conflict. Object metadata locates records; exact spans are the factual material. "
        "Use only IDs in the schema. Output no facts, reasoning, scores, query, prose or new IDs.\n\n"
        "Return exactly one JSON object conforming to this Output Schema:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        + "\n\nUSER QUESTION:\n"
        + str(query or "")
        + "\n\nTASK RECORD:\n"
        + json.dumps(dict(record), ensure_ascii=False, separators=(",", ":"))
        + "\n\nOBJECT GROUPS:\n"
        + json.dumps(object_groups, ensure_ascii=False, separators=(",", ":"))
        + "\n\nEVIDENCE RECORDS:\n"
        + json.dumps(ordered_cards, ensure_ascii=False, separators=(",", ":"))
    )
    if correction:
        user_prompt += (
            "\n\nPROTOCOL CORRECTION: The previous continuation was invalid ("
            + _text(correction, 400)
            + "). Return only the complete JSON object in the required schema."
        )
    return (
        render_tool_transcript(
            [{"role": "user", "content": user_prompt}], json_output=True
        ),
        user_prompt,
    )


def _parse_record_object_selection(
    value: Any,
    *,
    task_record_id: str,
    allowed_object_group_ids: set[str],
) -> dict[str, Any]:
    envelope = normalize_json_object_envelope(value)
    payload = envelope.payload
    require_runtime_contract(payload, EVIDENCE_RESOLUTION_CONTRACT)
    if set(payload) != {
        "contract",
        "task_record_id",
        "selection",
        "object_group_ids",
    }:
        raise ValueError("object selection contains unexpected or missing keys")
    if _text(payload.get("task_record_id"), 80).casefold() != task_record_id.casefold():
        raise ValueError("object selection changed task_record_id")
    selection = _text(payload.get("selection"), 40).casefold()
    if selection not in {"selected", "conflict", "missing"}:
        raise ValueError("object selection must be selected, conflict or missing")
    group_ids = _valid_ids(payload.get("object_group_ids"), allowed_object_group_ids)
    if selection == "selected" and len(group_ids) != 1:
        raise ValueError("selected requires exactly one object group")
    if selection == "conflict" and len(group_ids) < 2:
        raise ValueError("conflict requires at least two object groups")
    if selection == "missing" and group_ids:
        raise ValueError("missing requires an empty object group list")
    return {
        "selection": selection,
        "object_group_ids": group_ids,
        "input_format": envelope.input_format,
        "transport_normalized": envelope.normalized,
    }


def _build_record_field_binding_prompt(
    query: str,
    record: Mapping[str, Any],
    selection: Mapping[str, Any],
    evidence_records: list[dict[str, Any]],
    *,
    correction: str = "",
) -> tuple[str, str]:
    task_record_id = record_id(record)
    selected_groups = set(selection.get("object_group_ids") or [])
    selected_cards = [
        dict(card)
        for card in evidence_records
        if str(card.get("object_group_id") or "") in selected_groups
    ]
    requested_field_ids = field_ids(record)
    allowed_evidence_ids = [
        str(card.get("evidence_record_id") or "")
        for card in selected_cards
        if str(card.get("evidence_record_id") or "")
    ]
    schema = _record_field_binding_schema(
        task_record_id,
        requested_field_ids,
        allowed_evidence_ids,
    )
    user_prompt = (
        "Bind exact evidence for the requested fields of exactly one Task Record. The "
        "object-selection call has already limited the candidate object groups. For every "
        "requested field, output exactly one row. Use supported only when the cited Evidence "
        "Record has an exact span containing that field value for the selected object. Use "
        "missing with an empty evidence list when the shown spans do not establish it. Use "
        "conflict only with at least two cited records whose exact spans disagree; when the "
        "object selection is conflict, a conflict binding must cover competing groups. Never "
        "move a date, version, name, status or other value between object groups. Metadata and "
        "IDs are routing context, not proof. Use only IDs in the schema. Output no facts, "
        "reasoning, prose, query, scores or new IDs.\n\n"
        "Return exactly one JSON object conforming to this Output Schema:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        + "\n\nUSER QUESTION:\n"
        + str(query or "")
        + "\n\nTASK RECORD:\n"
        + json.dumps(dict(record), ensure_ascii=False, separators=(",", ":"))
        + "\n\nOBJECT SELECTION:\n"
        + json.dumps(dict(selection), ensure_ascii=False, separators=(",", ":"))
        + "\n\nSELECTED EVIDENCE RECORDS:\n"
        + json.dumps(selected_cards, ensure_ascii=False, separators=(",", ":"))
    )
    if correction:
        user_prompt += (
            "\n\nPROTOCOL CORRECTION: The previous continuation was invalid ("
            + _text(correction, 400)
            + "). Return only the complete JSON object in the required schema."
        )
    return (
        render_tool_transcript(
            [{"role": "user", "content": user_prompt}], json_output=True
        ),
        user_prompt,
    )


def _parse_record_field_bindings(
    value: Any,
    *,
    record: Mapping[str, Any],
    selection: Mapping[str, Any],
    evidence_records: list[dict[str, Any]],
) -> dict[str, Any]:
    envelope = normalize_json_object_envelope(value)
    payload = envelope.payload
    require_runtime_contract(payload, EVIDENCE_RESOLUTION_CONTRACT)
    if set(payload) != {"contract", "task_record_id", "fields"}:
        raise ValueError("field binding contains unexpected or missing keys")
    task_record_id = record_id(record)
    if _text(payload.get("task_record_id"), 80).casefold() != task_record_id.casefold():
        raise ValueError("field binding changed task_record_id")
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, list):
        raise ValueError("fields must be an array")

    selected_groups = set(selection.get("object_group_ids") or [])
    card_by_id = {
        str(card.get("evidence_record_id") or ""): card
        for card in evidence_records
        if str(card.get("evidence_record_id") or "")
        and str(card.get("object_group_id") or "") in selected_groups
    }
    allowed_evidence = set(card_by_id)
    required_fields = field_ids(record)
    field_lookup = {field_id.casefold(): field_id for field_id in required_fields}
    evidence_group = {
        evidence_id: str(card.get("object_group_id") or "")
        for evidence_id, card in card_by_id.items()
    }
    normalized: dict[str, dict[str, Any]] = {}
    for raw in raw_fields:
        if not isinstance(raw, Mapping) or set(raw) != {
            "field_id",
            "state",
            "evidence_record_ids",
        }:
            raise ValueError("every field row must contain exactly the required keys")
        supplied_field_id = _text(raw.get("field_id"), 160)
        field_id = field_lookup.get(supplied_field_id.casefold())
        if field_id is None:
            raise ValueError(f"unknown field_id: {supplied_field_id}")
        if field_id in normalized:
            raise ValueError(f"duplicate field binding: {field_id}")
        state = _text(raw.get("state"), 40).casefold()
        if state not in {"supported", "missing", "conflict"}:
            raise ValueError("field state must be supported, missing or conflict")
        evidence_ids = _valid_ids(raw.get("evidence_record_ids"), allowed_evidence)
        if evidence_ids and any(
            not (card_by_id[evidence_id].get("exact_spans") or [])
            for evidence_id in evidence_ids
        ):
            raise ValueError("field evidence must contain at least one shown exact span")
        if state == "missing" and evidence_ids:
            raise ValueError("missing field must have an empty evidence list")
        if state == "supported":
            if not evidence_ids:
                raise ValueError("supported field requires evidence")
            if len({evidence_group[evidence_id] for evidence_id in evidence_ids}) != 1:
                raise ValueError("supported field evidence must stay inside one object group")
        if state == "conflict":
            if len(evidence_ids) < 2:
                raise ValueError("conflict field requires at least two evidence records")
            if selection.get("selection") == "conflict" and len(
                {evidence_group[evidence_id] for evidence_id in evidence_ids}
            ) < 2:
                raise ValueError("object conflict evidence must cover competing groups")
        normalized[field_id] = {
            "field_id": field_id,
            "state": state,
            "evidence_record_ids": evidence_ids,
        }
    if set(normalized) != set(required_fields):
        raise ValueError("field binding must contain exactly one row per requested field")
    if selection.get("selection") == "conflict" and not any(
        row["state"] == "conflict" for row in normalized.values()
    ):
        raise ValueError("object conflict requires at least one conflicting requested field")
    return {
        "fields": [normalized[field_id] for field_id in required_fields],
        "input_format": envelope.input_format,
        "transport_normalized": envelope.normalized,
    }


def _call_resolution_stage(
    llm: Any,
    prompt_builder: Any,
    parser: Any,
    *,
    requested_max: int,
    retries: int,
    sampling_temperature: float,
) -> dict[str, Any]:
    """Run one narrow RWKV resolution stage with one protocol-only retry."""

    provider = str(getattr(llm, "provider", "") or "")
    last_error = ""
    raw = ""
    prompt = ""
    attempts = 0
    for attempt in range(retries + 1):
        prompt, user_prompt = prompt_builder(last_error if attempt else "")
        if get_token_count(prompt) + requested_max + 256 >= get_llm_context_length():
            last_error = "evidence resolution prompt exceeds model context"
            break
        max_tokens = bounded_completion_budget(
            prompt,
            context_limit=get_llm_context_length(),
            requested_max=requested_max,
            safety_margin=256,
        )
        try:
            attempts = attempt + 1
            with model_sampling_parameters(
                sampling_temperature,
                stage="evidence_resolution",
                policy_reason="record_scoped_evidence_resolution",
            ):
                if hasattr(llm, "text_completion") and (
                    not provider or is_local_provider(provider)
                ):
                    response = llm.text_completion(
                        prompt,
                        max_tokens=max_tokens,
                        stop=JSON_CALL_STOP_SUFFIXES,
                    )
                else:
                    response = llm.chat_completion(
                        [{"role": "user", "content": user_prompt}],
                        max_tokens=max_tokens,
                    )
            raw = visible_model_text(getattr(response, "content", response))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            break
        try:
            return {
                "status": "ok",
                "value": parser(raw),
                "attempts": attempts,
                "prompt": prompt,
                "raw_model_output": raw,
                "error": "",
                "completion_token_budget": requested_max,
            }
        except ValueError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {
        "status": "unavailable",
        "value": None,
        "attempts": attempts,
        "prompt": prompt,
        "raw_model_output": raw,
        "error": last_error[:1000],
        "completion_token_budget": requested_max,
    }


def _valid_ids(value: Any, allowed: set[str], *, limit: int = 24) -> list[str]:
    lookup = {item.casefold(): item for item in allowed}
    if not isinstance(value, list):
        raise ValueError("ID collection must be an array")
    output: list[str] = []
    for raw in value:
        text = _text(raw, 160)
        key = text.casefold()
        if not key or key not in lookup:
            raise ValueError(f"unknown controller-owned ID: {text}")
        if lookup[key] not in output:
            output.append(lookup[key])
    return output[: max(0, int(limit))]


def _empty_decision(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_record_id": record_id(record),
        "status": "missing",
        "selected_object_group_id": "",
        "conflicting_object_group_ids": [],
        "selected_evidence_record_ids": [],
        "conflicting_evidence_record_ids": [],
        "field_evidence_record_ids": {},
        "missing_field_ids": field_ids(record),
        "conflict_field_ids": [],
        "needs_more_evidence": True,
    }


def parse_evidence_resolution_output(
    value: Any,
    *,
    query: str,
    task_plan: Mapping[str, Any] | None,
    evidence_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate complete field closure and all controller-owned identities."""

    envelope = normalize_json_object_envelope(value)
    payload = envelope.payload
    require_runtime_contract(
        payload,
        EVIDENCE_RESOLUTION_CONTRACT,
    )
    if set(payload) != {"contract", "decisions"}:
        raise ValueError("resolution output must contain exactly contract and decisions")
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("resolution decisions must be an array")

    records = task_records(task_plan, fallback_query=query)
    record_lookup = {record_id(row).casefold(): row for row in records}
    card_by_id = {
        str(card.get("evidence_record_id")): card
        for card in evidence_records
        if str(card.get("evidence_record_id") or "")
    }
    allowed_evidence = set(card_by_id)
    object_groups = build_evidence_object_groups(evidence_records)
    group_by_id = {
        str(group.get("object_group_id")): group
        for group in object_groups
        if str(group.get("object_group_id") or "")
    }
    allowed_groups = set(group_by_id)
    evidence_group_by_id = {
        evidence_id: str(card.get("object_group_id") or "")
        for evidence_id, card in card_by_id.items()
    }
    globally_complete = all(
        card.get("coverage_complete") is True for card in card_by_id.values()
    )
    normalized: dict[str, dict[str, Any]] = {}
    for raw in raw_decisions:
        if not isinstance(raw, Mapping):
            raise ValueError("every resolution decision must be an object")
        required_decision_keys = {
            "task_record_id",
            "status",
            "selected_object_group_id",
            "conflicting_object_group_ids",
            "selected_evidence_record_ids",
            "conflicting_evidence_record_ids",
            "field_evidence_record_ids",
            "missing_field_ids",
            "conflict_field_ids",
            "needs_more_evidence",
        }
        if set(raw) != required_decision_keys:
            raise ValueError(
                "every resolution decision must contain exactly the required keys"
            )
        supplied_id = _text(raw.get("task_record_id"), 80)
        record = record_lookup.get(supplied_id.casefold())
        if record is None:
            raise ValueError(f"unknown task_record_id: {supplied_id}")
        canonical_record_id = record_id(record)
        if canonical_record_id in normalized:
            raise ValueError(f"duplicate resolution decision: {canonical_record_id}")
        status = _text(raw.get("status"), 40).casefold()
        if status not in _VALID_STATUSES:
            raise ValueError(f"invalid resolution status for {canonical_record_id}")

        supplied_group_id = _text(raw.get("selected_object_group_id"), 160)
        if supplied_group_id:
            group_lookup = {group_id.casefold(): group_id for group_id in allowed_groups}
            selected_group_id = group_lookup.get(supplied_group_id.casefold(), "")
            if not selected_group_id:
                raise ValueError(f"unknown object_group_id: {supplied_group_id}")
        else:
            selected_group_id = ""
        conflicting_group_ids = _valid_ids(
            raw.get("conflicting_object_group_ids"), allowed_groups
        )

        selected = _valid_ids(raw.get("selected_evidence_record_ids"), allowed_evidence)
        conflicts = _valid_ids(
            raw.get("conflicting_evidence_record_ids"), allowed_evidence
        )
        if not set(conflicts).issubset(selected):
            raise ValueError("conflicting evidence must also be selected")

        required_fields = set(field_ids(record))
        field_bindings: dict[str, list[str]] = {}
        raw_bindings = raw.get("field_evidence_record_ids")
        if not isinstance(raw_bindings, Mapping):
            raise ValueError("field_evidence_record_ids must be an object")
        field_lookup = {value.casefold(): value for value in required_fields}
        for raw_field_id, raw_evidence_ids in raw_bindings.items():
            canonical_field_id = field_lookup.get(_text(raw_field_id, 160).casefold())
            if canonical_field_id is None:
                raise ValueError(f"unknown field_id: {raw_field_id}")
            evidence_ids = _valid_ids(raw_evidence_ids, allowed_evidence)
            if not evidence_ids or not set(evidence_ids).issubset(selected):
                raise ValueError("field bindings must be non-empty selected evidence IDs")
            if not selected_group_id or any(
                evidence_group_by_id.get(evidence_id) != selected_group_id
                for evidence_id in evidence_ids
            ):
                raise ValueError(
                    "field bindings must stay inside the selected object group"
                )
            field_bindings[canonical_field_id] = evidence_ids
        missing = set(_valid_ids(raw.get("missing_field_ids"), required_fields))
        conflict_fields = set(_valid_ids(raw.get("conflict_field_ids"), required_fields))
        bound = set(field_bindings)
        if bound & missing or bound & conflict_fields or missing & conflict_fields:
            raise ValueError("field states must be disjoint")
        if bound | missing | conflict_fields != required_fields:
            raise ValueError("every requested field_id must have exactly one state")

        if status == "resolved":
            if not globally_complete:
                raise ValueError("resolved is forbidden with unviewed evidence spans")
            if (
                not selected_group_id
                or conflicting_group_ids
                or not selected
                or missing
                or conflict_fields
                or conflicts
            ):
                raise ValueError("resolved requires selected evidence and complete field bindings")
            if any(
                evidence_group_by_id.get(evidence_id) != selected_group_id
                for evidence_id in selected
            ):
                raise ValueError(
                    "resolved evidence must stay inside one selected object group"
                )
        elif status == "conflict":
            if len(conflicts) < 2:
                raise ValueError("conflict requires at least two evidence records")
            if required_fields and not conflict_fields:
                raise ValueError("conflict requires at least one conflict_field_id")
            if (
                not selected_group_id
                or len(conflicting_group_ids) < 2
                or selected_group_id not in conflicting_group_ids
            ):
                raise ValueError(
                    "conflict requires a selected group and at least two competing object groups"
                )
            conflict_evidence_groups = {
                evidence_group_by_id.get(evidence_id, "") for evidence_id in conflicts
            }
            if (
                "" in conflict_evidence_groups
                or len(conflict_evidence_groups) < 2
                or not conflict_evidence_groups.issubset(set(conflicting_group_ids))
            ):
                raise ValueError(
                    "conflicting evidence must represent at least two competing object groups"
                )
            if any(
                evidence_group_by_id.get(evidence_id) not in conflicting_group_ids
                for evidence_id in selected
            ):
                raise ValueError(
                    "conflict evidence must stay inside the declared competing groups"
                )
        else:
            if conflicts or conflict_fields or conflicting_group_ids:
                raise ValueError("missing status cannot carry conflict state")
            if required_fields and not missing:
                raise ValueError("missing status requires at least one missing_field_id")
            if selected and not selected_group_id:
                raise ValueError("selected evidence requires one selected object group")
            if selected_group_id and any(
                evidence_group_by_id.get(evidence_id) != selected_group_id
                for evidence_id in selected
            ):
                raise ValueError(
                    "missing decision evidence must stay inside one selected object group"
                )

        needs_more = raw.get("needs_more_evidence") is True
        if status != "resolved" and not needs_more:
            raise ValueError("missing/conflict decisions require needs_more_evidence=true")
        if status == "resolved" and needs_more:
            raise ValueError("resolved decision cannot need more evidence")
        normalized[canonical_record_id] = {
            "task_record_id": canonical_record_id,
            "status": status,
            "selected_object_group_id": selected_group_id,
            "conflicting_object_group_ids": conflicting_group_ids,
            "selected_evidence_record_ids": selected,
            "conflicting_evidence_record_ids": conflicts,
            "field_evidence_record_ids": field_bindings,
            "missing_field_ids": sorted(missing),
            "conflict_field_ids": sorted(conflict_fields),
            "needs_more_evidence": needs_more,
        }

    expected = {record_id(row) for row in records}
    if set(normalized) != expected:
        raise ValueError("resolution must contain exactly one decision per Task Plan record")
    return {
        "contract": EVIDENCE_RESOLUTION_CONTRACT,
        "status": "ok",
        "input_format": envelope.input_format,
        "transport_normalized": envelope.normalized,
        "evidence_record_count": len(evidence_records),
        "coverage_complete": globally_complete,
        "unviewed_span_ids": sorted(
            {
                str(span_id)
                for card in evidence_records
                for span_id in card.get("unviewed_span_ids") or []
                if str(span_id)
            }
        ),
        "packet_aliases": {
            str(card.get("evidence_record_id")): str(card.get("packet_alias") or "")
            for card in evidence_records
            if str(card.get("evidence_record_id") or "")
        },
        "object_groups": object_groups,
        "decisions": [normalized[record_id(row)] for row in records],
    }


def _empty_evidence_resolution(
    query: str,
    task_plan: Mapping[str, Any] | None,
    *,
    status: str = "no_evidence_records",
) -> dict[str, Any]:
    records = task_records(task_plan, fallback_query=query)
    return {
        "contract": EVIDENCE_RESOLUTION_CONTRACT,
        "status": status,
        "evidence_record_count": 0,
        "coverage_complete": True,
        "unviewed_span_ids": [],
        "packet_aliases": {},
        "object_groups": [],
        "decisions": [_empty_decision(record) for record in records],
        "attempts": 0,
        "raw_model_output": "",
        "prompt": "",
    }


def evidence_resolution_completion_token_budget(
    query: str,
    task_plan: Mapping[str, Any] | None,
) -> int:
    """Size the closed JSON continuation from Task Record and Field cardinality."""

    records = task_records(task_plan, fallback_query=query)
    field_count = sum(len(field_ids(record)) for record in records)
    estimated = 256 + len(records) * 192 + field_count * 56
    configured_cap = DATA_PIPELINE.get("evidence_resolution_completion_token_cap")
    if configured_cap is None:
        # Historical local configs may still carry the fixed-budget key.
        configured_cap = DATA_PIPELINE.get("evidence_resolution_max_tokens", 4096)
    cap = max(384, int(configured_cap or 4096))
    return min(cap, max(384, estimated))


def resolve_evidence(
    query: str,
    task_plan: Mapping[str, Any] | None,
    evidence_records: list[dict[str, Any]],
    llm: Any,
) -> dict[str, Any]:
    """Resolve each Task Record with narrow object-then-field RWKV calls.

    RWKV owns both semantic choices.  Deterministic code only scopes the second
    request to the groups selected in the first request, validates
    controller-owned IDs, and projects the decisions into the stable runtime
    shape consumed by the Writer packet.
    """

    bounded_evidence_records = [
        dict(record) for record in evidence_records if isinstance(record, Mapping)
    ][:24]
    if not bounded_evidence_records:
        return _empty_evidence_resolution(query, task_plan)
    input_digest = evidence_resolution_signature(
        query,
        task_plan,
        bounded_evidence_records,
    )
    retries_value = DATA_PIPELINE.get("evidence_resolution_protocol_retries", 1)
    retries = max(0, min(2, int(retries_value if retries_value is not None else 1)))
    sampling_temperature = get_model_stage_temperature("evidence_resolution")
    sampling_profile = get_model_stage_sampling("evidence_resolution")
    records = task_records(task_plan, fallback_query=query)
    object_groups = build_evidence_object_groups(bounded_evidence_records)
    allowed_groups = {
        str(group.get("object_group_id") or "")
        for group in object_groups
        if str(group.get("object_group_id") or "")
    }
    group_cards: dict[str, list[dict[str, Any]]] = {
        group_id: [
            card
            for card in bounded_evidence_records
            if str(card.get("object_group_id") or "") == group_id
        ]
        for group_id in allowed_groups
    }

    decisions: list[dict[str, Any]] = []
    stage_calls: list[dict[str, Any]] = []
    errors: list[str] = []
    total_attempts = 0
    partial = False
    accepted_stage_count = 0

    for record in records:
        task_record_id = record_id(record)
        object_budget = 256
        object_call = _call_resolution_stage(
            llm,
            lambda correction, record=record: _build_record_object_selection_prompt(
                query,
                record,
                bounded_evidence_records,
                correction=correction,
            ),
            lambda raw, task_record_id=task_record_id: _parse_record_object_selection(
                raw,
                task_record_id=task_record_id,
                allowed_object_group_ids=allowed_groups,
            ),
            requested_max=object_budget,
            retries=retries,
            sampling_temperature=sampling_temperature,
        )
        total_attempts += int(object_call.get("attempts") or 0)
        stage_calls.append(
            {
                "task_record_id": task_record_id,
                "stage": "object_selection",
                **{key: value for key, value in object_call.items() if key != "value"},
            }
        )
        if object_call["status"] != "ok":
            partial = True
            error = f"{task_record_id}/object_selection: {object_call.get('error') or 'unavailable'}"
            errors.append(error)
            decision = _empty_decision(record)
            decision["resolver_fallback"] = "object_selection_unavailable"
            decisions.append(decision)
            continue

        selection = dict(object_call["value"])
        accepted_stage_count += 1
        selection_kind = str(selection.get("selection") or "")
        selected_group_ids = [
            str(value) for value in selection.get("object_group_ids") or []
        ]
        selected_cards = [
            card
            for group_id in selected_group_ids
            for card in group_cards.get(group_id, [])
        ]
        selected_evidence_ids = [
            str(card.get("evidence_record_id") or "")
            for card in selected_cards
            if str(card.get("evidence_record_id") or "")
        ]

        if selection_kind == "missing":
            decision = _empty_decision(record)
            decisions.append(decision)
            continue

        requested_fields = field_ids(record)
        if not requested_fields:
            decisions.append(
                {
                    "task_record_id": task_record_id,
                    "status": "conflict" if selection_kind == "conflict" else "resolved",
                    "selected_object_group_id": selected_group_ids[0],
                    "conflicting_object_group_ids": (
                        selected_group_ids if selection_kind == "conflict" else []
                    ),
                    "selected_evidence_record_ids": selected_evidence_ids,
                    "conflicting_evidence_record_ids": (
                        selected_evidence_ids if selection_kind == "conflict" else []
                    ),
                    "field_evidence_record_ids": {},
                    "missing_field_ids": [],
                    "conflict_field_ids": [],
                    "needs_more_evidence": selection_kind == "conflict",
                }
            )
            continue

        field_budget = min(2048, max(384, 192 + len(requested_fields) * 96))
        field_call = _call_resolution_stage(
            llm,
            lambda correction, record=record, selection=selection: _build_record_field_binding_prompt(
                query,
                record,
                selection,
                bounded_evidence_records,
                correction=correction,
            ),
            lambda raw, record=record, selection=selection: _parse_record_field_bindings(
                raw,
                record=record,
                selection=selection,
                evidence_records=bounded_evidence_records,
            ),
            requested_max=field_budget,
            retries=retries,
            sampling_temperature=sampling_temperature,
        )
        total_attempts += int(field_call.get("attempts") or 0)
        stage_calls.append(
            {
                "task_record_id": task_record_id,
                "stage": "field_binding",
                **{key: value for key, value in field_call.items() if key != "value"},
            }
        )
        if field_call["status"] != "ok":
            partial = True
            error = f"{task_record_id}/field_binding: {field_call.get('error') or 'unavailable'}"
            errors.append(error)
            decision = _empty_decision(record)
            decision.update(
                {
                    "selected_object_group_id": selected_group_ids[0],
                    "conflicting_object_group_ids": (
                        selected_group_ids if selection_kind == "conflict" else []
                    ),
                    "selected_evidence_record_ids": selected_evidence_ids,
                    "resolver_fallback": "field_binding_unavailable",
                }
            )
            decisions.append(decision)
            continue

        field_rows = list(field_call["value"]["fields"])
        accepted_stage_count += 1
        bindings: dict[str, list[str]] = {}
        missing: list[str] = []
        conflict_fields: list[str] = []
        conflict_evidence: list[str] = []
        used_evidence: list[str] = []
        for field_row in field_rows:
            field_id = str(field_row["field_id"])
            evidence_ids = [str(value) for value in field_row["evidence_record_ids"]]
            state = str(field_row["state"])
            for evidence_id in evidence_ids:
                if evidence_id not in used_evidence:
                    used_evidence.append(evidence_id)
            if state == "supported":
                bindings[field_id] = evidence_ids
            elif state == "conflict":
                conflict_fields.append(field_id)
                for evidence_id in evidence_ids:
                    if evidence_id not in conflict_evidence:
                        conflict_evidence.append(evidence_id)
            else:
                missing.append(field_id)

        status = "conflict" if conflict_fields else "missing" if missing else "resolved"
        selected_group_id = selected_group_ids[0]
        if bindings:
            first_bound_id = next(iter(bindings.values()))[0]
            selected_group_id = str(
                next(
                    (
                        card.get("object_group_id")
                        for card in selected_cards
                        if str(card.get("evidence_record_id") or "") == first_bound_id
                    ),
                    selected_group_id,
                )
            )
        decisions.append(
            {
                "task_record_id": task_record_id,
                "status": status,
                "selected_object_group_id": selected_group_id,
                "conflicting_object_group_ids": (
                    selected_group_ids if conflict_fields else []
                ),
                "selected_evidence_record_ids": used_evidence,
                "conflicting_evidence_record_ids": conflict_evidence,
                "field_evidence_record_ids": bindings,
                "missing_field_ids": missing,
                "conflict_field_ids": conflict_fields,
                "needs_more_evidence": status != "resolved",
            }
        )

    coverage_complete = all(
        card.get("coverage_complete") is True for card in bounded_evidence_records
    )
    prompt_log = "\n\n".join(
        "===== "
        + str(call.get("task_record_id") or "")
        + "/"
        + str(call.get("stage") or "")
        + " =====\n"
        + str(call.get("prompt") or "")
        for call in stage_calls
    )
    raw_log = json.dumps(
        [
            {
                "task_record_id": call.get("task_record_id"),
                "stage": call.get("stage"),
                "status": call.get("status"),
                "raw_model_output": call.get("raw_model_output"),
                "error": call.get("error"),
            }
            for call in stage_calls
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "contract": EVIDENCE_RESOLUTION_CONTRACT,
        "status": (
            "unavailable"
            if partial and accepted_stage_count == 0
            else "partial"
            if partial
            else "ok"
        ),
        "evidence_record_count": len(bounded_evidence_records),
        "coverage_complete": coverage_complete,
        "unviewed_span_ids": sorted(
            {
                str(span_id)
                for card in bounded_evidence_records
                for span_id in card.get("unviewed_span_ids") or []
                if str(span_id)
            }
        ),
        "packet_aliases": {
            str(card.get("evidence_record_id")): str(card.get("packet_alias") or "")
            for card in bounded_evidence_records
            if str(card.get("evidence_record_id") or "")
        },
        "object_groups": object_groups,
        "decisions": decisions,
        "attempts": total_attempts,
        "input_digest": input_digest,
        "raw_model_output": raw_log,
        "prompt": prompt_log,
        "stage_calls": stage_calls,
        "error": " | ".join(errors)[:2000],
        "sampling_temperature": sampling_temperature,
        "sampling_parameters": sampling_profile,
        "completion_token_budget": sum(
            int(call.get("completion_token_budget") or 0) for call in stage_calls
        ),
    }


def render_evidence_resolution_view(
    resolution: Mapping[str, Any], *, max_tokens: int = 768
) -> str:
    """Render a bounded control lane; exact evidence remains a separate lane."""

    if not isinstance(resolution, Mapping) or not resolution.get("decisions"):
        return ""
    if str(resolution.get("status") or "") in {"disabled", "no_evidence_records", "unavailable"}:
        return ""
    lines = [
        "EVIDENCE RESOLUTION CONTROL MAP (attention only; E-* identifies evidence, S# is display only):"
    ]
    for raw in resolution.get("decisions") or []:
        if not isinstance(raw, Mapping):
            continue
        bindings = ",".join(
            f"{field_id}->{'+'.join(ids)}"
            for field_id, ids in (raw.get("field_evidence_record_ids") or {}).items()
            if isinstance(ids, list) and ids
        )
        values = [
            f"{raw.get('task_record_id')} status={raw.get('status')}",
            "object_group="
            + (str(raw.get("selected_object_group_id") or "") or "none"),
            "competing_groups="
            + (",".join(raw.get("conflicting_object_group_ids") or []) or "none"),
            "selected="
            + (",".join(raw.get("selected_evidence_record_ids") or []) or "none"),
            "conflicts="
            + (",".join(raw.get("conflicting_evidence_record_ids") or []) or "none"),
        ]
        if bindings:
            values.append("bindings=" + bindings)
        if raw.get("missing_field_ids"):
            values.append("missing=" + ",".join(raw["missing_field_ids"]))
        if raw.get("conflict_field_ids"):
            values.append("conflict_fields=" + ",".join(raw["conflict_field_ids"]))
        values.append(
            "needs_more_evidence="
            + ("true" if raw.get("needs_more_evidence") else "false")
        )
        line = "; ".join(values)
        if get_token_count("\n".join([*lines, line])) > max(64, int(max_tokens)):
            break
        lines.append(line)
    return "\n".join(lines) if len(lines) > 1 else ""


def build_record_first_writer_packet(
    context: Mapping[str, Any],
    resolution: Mapping[str, Any],
    task_plan: Mapping[str, Any] | None,
    *,
    max_tokens: int = 6000,
) -> dict[str, Any] | None:
    """Pack RWKV-selected object groups without rewriting any evidence span."""

    if str(resolution.get("status") or "") not in {"ok", "partial"}:
        return None
    records = task_records(task_plan)
    decisions = {
        str(row.get("task_record_id") or ""): row
        for row in resolution.get("decisions") or []
        if isinstance(row, Mapping) and str(row.get("task_record_id") or "")
    }
    if not records or set(decisions) != {record_id(record) for record in records}:
        return None
    selected_evidence = [
        dict(row)
        for row in context.get("selected_evidence") or []
        if isinstance(row, Mapping)
    ]
    evidence_by_id = {
        str(row.get("evidence_record_id") or ""): row
        for row in selected_evidence
        if str(row.get("evidence_record_id") or "")
    }
    if not evidence_by_id:
        return None
    group_by_id = {
        str(group.get("object_group_id") or ""): dict(group)
        for group in resolution.get("object_groups") or []
        if isinstance(group, Mapping) and str(group.get("object_group_id") or "")
    }
    evidence_group = {
        str(evidence_id): group_id
        for group_id, group in group_by_id.items()
        for evidence_id in group.get("evidence_record_ids") or []
        if str(evidence_id)
    }

    candidates: dict[str, dict[str, Any]] = {}
    field_candidates: dict[tuple[str, str], list[str]] = {}
    record_candidates: dict[str, list[str]] = {}

    def source_span_candidates(
        evidence_ids: list[str], *, limit: int = 2
    ) -> list[str]:
        per_evidence: list[list[str]] = []
        for evidence_id in evidence_ids:
            source = evidence_by_id.get(str(evidence_id))
            if source is None:
                continue
            rows: list[str] = []
            for index, chunk in enumerate(source.get("packed_chunks") or [], start=1):
                if not isinstance(chunk, Mapping):
                    continue
                literal = str(chunk.get("text") or "").replace("\x00", "")
                if not literal.strip():
                    continue
                chunk_id = _text(chunk.get("chunk_id") or f"span-{index}", 160)
                candidate_key = f"{evidence_id}\x1f{chunk_id}\x1f{hashlib.sha256(literal.encode('utf-8')).hexdigest()[:12]}"
                if candidate_key not in candidates:
                    candidates[candidate_key] = {
                        "span_ref": f"{evidence_id}:{chunk_id}",
                        "evidence_record_id": str(evidence_id),
                        "object_group_id": evidence_group.get(str(evidence_id), ""),
                        "ref_id": str(source.get("ref_id") or ""),
                        "title": str(source.get("title") or "")[:300],
                        "url": str(source.get("url") or "")[:800],
                        "chunk_id": chunk_id,
                        "text": literal,
                    }
                rows.append(candidate_key)
            if rows:
                per_evidence.append(rows)
        ordered: list[str] = []
        depth = 0
        bounded_limit = max(1, int(limit))
        while len(ordered) < bounded_limit:
            added = False
            for rows in per_evidence:
                if depth < len(rows):
                    ordered.append(rows[depth])
                    added = True
                    if len(ordered) >= bounded_limit:
                        break
            if not added:
                break
            depth += 1
        return ordered

    for record in records:
        task_record_id = record_id(record)
        decision = decisions[task_record_id]
        resolver_fallback = str(decision.get("resolver_fallback") or "")
        if resolver_fallback:
            fallback_evidence_ids = [
                str(value)
                for value in decision.get("selected_evidence_record_ids") or []
                if str(value) in evidence_by_id
            ]
            if not fallback_evidence_ids:
                fallback_evidence_ids = list(evidence_by_id)
            per_group: dict[str, list[str]] = {}
            for evidence_id in fallback_evidence_ids:
                per_group.setdefault(evidence_group.get(evidence_id, ""), []).append(
                    evidence_id
                )
            grouped_candidates: list[str] = []
            for evidence_ids in per_group.values():
                grouped_candidates.extend(
                    source_span_candidates(evidence_ids, limit=1)
                )
            record_candidates[task_record_id] = list(
                dict.fromkeys(grouped_candidates)
            )[:8]
            continue
        bindings = decision.get("field_evidence_record_ids") or {}
        for field in record_field_records(record):
            field_id = field["field_id"]
            evidence_ids = [
                str(value)
                for value in bindings.get(field_id) or []
                if str(value) in evidence_by_id
            ]
            if field_id in set(decision.get("conflict_field_ids") or []):
                conflict_ids = [
                    str(value)
                    for value in decision.get("conflicting_evidence_record_ids") or []
                    if str(value) in evidence_by_id
                ]
                by_group: dict[str, str] = {}
                for evidence_id in conflict_ids:
                    group_id = evidence_group.get(evidence_id, "")
                    if group_id and group_id not in by_group:
                        by_group[group_id] = evidence_id
                evidence_ids = list(by_group.values())[:2]
            field_candidates[(task_record_id, field_id)] = source_span_candidates(
                evidence_ids
            )
        if not record_field_records(record):
            record_candidates[task_record_id] = source_span_candidates(
                [
                    str(value)
                    for value in decision.get("selected_evidence_record_ids") or []
                    if str(value) in evidence_by_id
                ]
            )

    field_names = {
        (record_id(record), field["field_id"]): field["name"]
        for record in records
        for field in record_field_records(record)
    }

    def render(admitted: set[str]) -> str:
        sections = [
            "RECORD-FIRST EXACT EVIDENCE PACKET",
            "Only literal text inside <span-ref> blocks and TOOL RESULTS is factual material; all IDs, field maps, object identities, labels and URLs are routing metadata.",
        ]
        calculations = [
            dict(row)
            for row in context.get("calculation_results") or []
            if isinstance(row, Mapping)
        ]
        if calculations:
            sections.append(
                "TOOL RESULTS:\n"
                + json.dumps(calculations, ensure_ascii=False, separators=(",", ":"))
            )
        for record in records:
            task_record_id = record_id(record)
            decision = decisions[task_record_id]
            resolver_fallback = str(decision.get("resolver_fallback") or "")
            lines = [
                f"TASK RECORD {task_record_id}",
                "Question: " + str(record.get("question") or "")[:500],
                (
                    "RWKV resolution was unavailable; exact candidates remain grouped "
                    "without a controller-selected answer: "
                    if resolver_fallback
                    else "RWKV-selected object identity (control metadata, not proof): "
                )
                + json.dumps(
                    {
                        "object_group_id": decision.get("selected_object_group_id") or "",
                        "identity": (
                            group_by_id.get(
                                str(decision.get("selected_object_group_id") or ""), {}
                            ).get("identity")
                            or {}
                        ),
                        "status": decision.get("status") or "",
                        **(
                            {"fallback": resolver_fallback}
                            if resolver_fallback
                            else {}
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ]
            field_map: list[dict[str, Any]] = []
            span_keys: list[str] = []
            for field in record_field_records(record):
                field_id = field["field_id"]
                keys = [
                    key
                    for key in field_candidates.get((task_record_id, field_id), [])
                    if key in admitted
                ][:2]
                span_keys.extend(keys)
                state = "unresolved" if resolver_fallback else (
                    "conflict"
                    if field_id in set(decision.get("conflict_field_ids") or [])
                    else "missing"
                    if field_id in set(decision.get("missing_field_ids") or [])
                    else "bound"
                )
                field_map.append(
                    {
                        "field_id": field_id,
                        "name": field_names.get((task_record_id, field_id), ""),
                        "state": state,
                        "span_refs": [candidates[key]["span_ref"] for key in keys],
                    }
                )
            if field_map:
                lines.append(
                    "FIELD TO EXACT-SPAN MAP (RWKV attention metadata): "
                    + json.dumps(field_map, ensure_ascii=False, separators=(",", ":"))
                )
            if resolver_fallback or not field_map:
                span_keys.extend(
                    key
                    for key in record_candidates.get(task_record_id, [])
                    if key in admitted
                )
            unique_span_keys = list(dict.fromkeys(span_keys))
            selected_group_id = str(decision.get("selected_object_group_id") or "")
            selected_keys = [
                key
                for key in unique_span_keys
                if candidates[key]["object_group_id"] == selected_group_id
            ]
            conflict_keys = [key for key in unique_span_keys if key not in selected_keys]

            def append_spans(label: str, keys: list[str]) -> None:
                if not keys:
                    return
                lines.append(label)
                current_group = ""
                for key in keys:
                    span = candidates[key]
                    if span["object_group_id"] != current_group:
                        current_group = span["object_group_id"]
                        identity = group_by_id.get(current_group, {}).get("identity") or {}
                        lines.append(
                            "Object group: "
                            + json.dumps(
                                {
                                    "object_group_id": current_group,
                                    "identity": identity,
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                    lines.extend(
                        [
                            f"[{span['ref_id']}] evidence_record_id={span['evidence_record_id']}",
                            f"Source label: {span['title']}",
                            f"URL: {span['url']}",
                            f"<{span['span_ref']}>\n{span['text']}",
                        ]
                    )

            if resolver_fallback:
                append_spans(
                    "UNRESOLVED OBJECT-GROUP CANDIDATES (kept separate):",
                    unique_span_keys,
                )
            else:
                append_spans("SELECTED OBJECT EXACT SPANS:", selected_keys)
                append_spans(
                    "COMPETING OBJECT EXACT SPANS (kept separate):", conflict_keys
                )
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    ordered_candidates: list[str] = []
    field_keys = list(field_candidates)
    record_keys = list(record_candidates)
    candidate_depth = max(
        [
            len(rows)
            for rows in [*field_candidates.values(), *record_candidates.values()]
        ]
        or [0]
    )
    for ordinal in range(candidate_depth):
        for key in field_keys:
            rows = field_candidates[key]
            if ordinal < len(rows) and rows[ordinal] not in ordered_candidates:
                ordered_candidates.append(rows[ordinal])
        for key in record_keys:
            rows = record_candidates[key]
            if ordinal < len(rows) and rows[ordinal] not in ordered_candidates:
                ordered_candidates.append(rows[ordinal])

    admitted: set[str] = set()
    budget = max(512, min(int(max_tokens or 6000), 6000))
    for candidate_key in ordered_candidates:
        trial = {*admitted, candidate_key}
        if get_token_count(render(trial)) <= budget:
            admitted = trial
    required_candidate_sets = [
        rows
        for rows in [*field_candidates.values(), *record_candidates.values()]
        if rows
    ]
    if any(not admitted.intersection(rows) for rows in required_candidate_sets):
        # The pre-resolution evidence packet already obeys the same global
        # budget. Falling back preserves literal evidence instead of emitting
        # a structurally tidy packet whose central field has no factual span.
        return None
    text = render(admitted)
    used_evidence_ids = list(
        dict.fromkeys(candidates[key]["evidence_record_id"] for key in ordered_candidates if key in admitted)
    )
    used_ref_ids = list(
        dict.fromkeys(candidates[key]["ref_id"] for key in ordered_candidates if key in admitted and candidates[key]["ref_id"])
    )
    return {
        "text": text,
        "context_tokens": get_token_count(text),
        "used_evidence_record_ids": used_evidence_ids,
        "used_ref_ids": used_ref_ids,
        "candidate_span_count": len(ordered_candidates),
        "included_span_count": len(admitted),
        "omitted_span_count": len(ordered_candidates) - len(admitted),
        "token_budget": budget,
    }


def attach_evidence_resolution_to_context(
    context: Mapping[str, Any],
    resolution: Mapping[str, Any],
    task_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach control state and mechanically pack its selected literal spans."""

    output = deepcopy(dict(context))
    resolution_copy = deepcopy(dict(resolution))
    view = render_evidence_resolution_view(
        resolution_copy,
        max_tokens=int(
            DATA_PIPELINE.get("evidence_resolution_view_max_tokens", 768)
            or 768
        ),
    )
    output["evidence_resolution"] = resolution_copy
    output["evidence_resolution_view"] = view
    packet = build_record_first_writer_packet(
        output,
        resolution_copy,
        task_plan,
        max_tokens=int(
            DATA_PIPELINE.get("record_first_writer_packet_token_cap", 6000)
            or 6000
        ),
    )
    if packet is not None:
        used_evidence_ids = set(packet["used_evidence_record_ids"])
        used_ref_ids = set(packet["used_ref_ids"])
        output["selected_evidence"] = [
            row
            for row in output.get("selected_evidence") or []
            if isinstance(row, Mapping)
            and str(row.get("evidence_record_id") or "") in used_evidence_ids
        ]
        output["citation_refs"] = [
            row
            for row in output.get("citation_refs") or []
            if isinstance(row, Mapping)
            and str(row.get("ref_id") or "") in used_ref_ids
        ]
        output["record_first_writer_packet"] = {
            key: value for key, value in packet.items() if key != "text"
        }
        output["usable_evidence_count"] = len(output["selected_evidence"])
        output["chunk_count"] = int(packet.get("included_span_count") or 0)
        output["evidence_text"] = packet["text"]
    else:
        output["evidence_text"] = str(
            output.get("evidence_text") or output.get("text") or ""
        )
    output["text"] = output["evidence_text"]
    output["context_tokens"] = get_token_count(output["text"])
    stats = dict(output.get("context_stats") or {})
    if packet is not None:
        stats["source_count"] = len(output.get("selected_evidence") or [])
        stats["chunk_count"] = int(packet.get("included_span_count") or 0)
    stats.update(
        {
            "context_tokens": output["context_tokens"],
            "evidence_resolution_status": str(resolution_copy.get("status") or ""),
            "evidence_resolution_evidence_record_count": int(
                resolution_copy.get("evidence_record_count") or 0
            ),
            "evidence_resolution_view_tokens": get_token_count(view),
            "evidence_resolution_coverage_complete": bool(
                resolution_copy.get("coverage_complete")
            ),
            "record_first_writer_packet_active": packet is not None,
            "record_first_writer_packet_tokens": (
                int(packet.get("context_tokens") or 0) if packet else 0
            ),
            "record_first_writer_packet_included_spans": (
                int(packet.get("included_span_count") or 0) if packet else 0
            ),
            "record_first_writer_packet_omitted_spans": (
                int(packet.get("omitted_span_count") or 0) if packet else 0
            ),
        }
    )
    output["context_stats"] = stats
    return output


__all__ = [
    "EVIDENCE_RESOLUTION_CONTRACT",
    "attach_evidence_resolution_to_context",
    "build_evidence_object_groups",
    "build_evidence_record_set",
    "build_evidence_resolution_prompt",
    "build_record_first_writer_packet",
    "parse_evidence_resolution_output",
    "render_evidence_resolution_view",
    "evidence_resolution_signature",
    "evidence_resolution_completion_token_budget",
    "resolve_evidence",
]
