"""Global RWKV resolution over the exact evidence records sent to Writer.

The resolver emits only an attention/control map.  It cannot rewrite evidence,
promote packet aliases to identity, or declare a record resolved when any
candidate span was hidden from its request.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
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
from utils.concurrency import shutdown_pool, submit_with_context, task_wait_timeout
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


def build_balanced_order_views(
    values: list[str], *, max_views: int = 10
) -> list[list[str]]:
    """Return paired cyclic views that reduce position and direction bias.

    For five values and ten views, every value occupies every position twice
    and every pair appears in both relative orders equally often. Larger sets
    use evenly spaced rotations so the request count stays bounded.
    """

    items = list(dict.fromkeys(str(value) for value in values if str(value)))
    if len(items) <= 1:
        return [items]
    limit = max(1, min(int(max_views or 1), 2 * len(items)))
    rotation_count = min(len(items), max(1, (limit + 1) // 2))
    offsets = list(
        dict.fromkeys(
            (index * len(items)) // rotation_count
            for index in range(rotation_count)
        )
    )
    views: list[list[str]] = []
    for offset in offsets:
        forward = [*items[offset:], *items[:offset]]
        for candidate in (forward, list(reversed(forward))):
            if candidate not in views:
                views.append(candidate)
            if len(views) >= limit:
                return views
    return views or [items]


def build_overlapping_candidate_batches(
    values: list[str], *, group_size: int = 5, coverage: int = 2
) -> list[list[str]]:
    """Cover every candidate in small, differently composed nomination batches."""

    items = list(dict.fromkeys(str(value) for value in values if str(value)))
    if not items:
        return []
    size = max(2, min(int(group_size or 5), len(items)))
    passes = max(1, min(int(coverage or 2), 4))
    batches: list[list[str]] = []
    for pass_index in range(passes):
        stride = 1 + pass_index * 2
        while math.gcd(stride, len(items)) != 1:
            stride += 1
        if pass_index % 2:
            stride *= -1
        start = (pass_index * max(1, size // 2)) % len(items)
        sequence = [
            items[(start + position * stride) % len(items)]
            for position in range(len(items))
        ]
        batches.extend(
            sequence[index : index + size]
            for index in range(0, len(sequence), size)
        )
    return batches


def _bounded_confidence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a number from 0 to 100")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("confidence must be a finite number from 0 to 100")
    confidence = int(round(numeric))
    if confidence < 0 or confidence > 100:
        raise ValueError("confidence must be a number from 0 to 100")
    return confidence


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
            "confidence": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
            },
        },
        "required": [
            "contract",
            "task_record_id",
            "selection",
            "object_group_ids",
            "confidence",
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
    previous_selection: Mapping[str, Any] | None = None,
    object_group_order: list[str] | None = None,
) -> tuple[str, str]:
    task_record_id = record_id(record)
    ordered_cards = _ordered_cards_for_record(task_record_id, evidence_records)
    if object_group_order:
        order_index = {
            group_id: index for index, group_id in enumerate(object_group_order)
        }
        ordered_cards = [
            card
            for card in ordered_cards
            if str(card.get("object_group_id") or "") in order_index
        ]
        ordered_cards = sorted(
            ordered_cards,
            key=lambda card: (
                order_index.get(
                    str(card.get("object_group_id") or ""), len(order_index)
                ),
                str(card.get("evidence_record_id") or ""),
            ),
        )
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
        "Use only IDs in the schema. The confidence is your 0-100 estimate that this selection correctly classifies the shown exact spans for the requested object; it is not source authority and is not a fact. Output no facts, reasoning, scores other than confidence, query, prose or new IDs.\n\n"
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
    if isinstance(previous_selection, Mapping) and str(
        previous_selection.get("selection") or ""
    ).strip() in {"selected", "conflict"}:
        # Monotonicity observation only: RWKV sees its own earlier decision for
        # this record and still makes the choice itself. This guards against
        # silently regressing an already-selected correct object when new,
        # weaker candidates arrive (observed as correct_evidence_discarded).
        prior_compact = {
            "selection": str(previous_selection.get("selection") or ""),
            "object_group_ids": list(previous_selection.get("object_group_ids") or [])[:4],
        }
        user_prompt += (
            "\n\nPREVIOUS RESOLUTION FOR THIS RECORD (your own earlier decision; "
            "re-evaluate against the current groups, do not blindly repeat): "
            + json.dumps(prior_compact, ensure_ascii=False, separators=(",", ":"))[:400]
            + "\nIf you move away from a previously selected group, the replacement "
            "group's exact spans must explicitly supersede it for the requested "
            "object (a newer date or a more specific matching identity)."
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
        "confidence",
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
        "confidence": _bounded_confidence(payload.get("confidence")),
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


def _selection_semantic_key(selection: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "selection": str(selection.get("selection") or ""),
            "object_group_ids": sorted(
                str(value) for value in selection.get("object_group_ids") or []
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _aggregate_object_selection_calls(
    calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choose the RWKV plurality and expose agreement separately from self-confidence."""

    valid = [
        (index, call, call.get("value"))
        for index, call in enumerate(calls)
        if call.get("status") == "ok" and isinstance(call.get("value"), Mapping)
    ]
    if not valid:
        return None
    buckets: dict[str, list[tuple[int, dict[str, Any], Mapping[str, Any]]]] = {}
    for index, call, value in valid:
        buckets.setdefault(_selection_semantic_key(value), []).append(
            (index, call, value)
        )
    winner = max(
        buckets.values(),
        key=lambda rows: (
            len(rows),
            sum(int(row[2].get("confidence") or 0) for row in rows),
            -rows[0][0],
        ),
    )
    representative = dict(winner[0][2])
    model_confidence = round(
        sum(int(row[2].get("confidence") or 0) for row in winner) / len(winner)
    )
    agreement = len(winner) / len(valid)
    completion_ratio = len(valid) / len(calls)
    combined_confidence = round(model_confidence * agreement * completion_ratio)
    representative["model_confidence"] = model_confidence
    representative["confidence"] = combined_confidence
    representative["consensus"] = {
        "order_design": "paired_cyclic_rotations.v1",
        "requested_view_count": len(calls),
        "valid_view_count": len(valid),
        "winning_view_count": len(winner),
        "distinct_decision_count": len(buckets),
        "agreement": round(agreement, 4),
        "completion_ratio": round(completion_ratio, 4),
        "model_confidence_mean": model_confidence,
        "combined_confidence": combined_confidence,
        "unanimous": len(winner) == len(valid),
    }
    return representative


def _run_object_selection_ensemble(
    query: str,
    record: Mapping[str, Any],
    evidence_records: list[dict[str, Any]],
    llm: Any,
    *,
    allowed_object_group_ids: set[str],
    retries: int,
    sampling_temperature: float,
    previous_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_record_id = record_id(record)
    ordered_cards = _ordered_cards_for_record(task_record_id, evidence_records)
    base_group_ids = [
        str(group.get("object_group_id") or "")
        for group in build_evidence_object_groups(ordered_cards)
        if str(group.get("object_group_id") or "")
    ]
    ensemble_enabled = bool(
        DATA_PIPELINE.get("evidence_resolution_order_ensemble_enabled", True)
    )
    configured_views = int(
        DATA_PIPELINE.get("evidence_resolution_order_ensemble_max_views", 10) or 10
    )
    configured_workers = int(
        DATA_PIPELINE.get("evidence_resolution_order_ensemble_parallelism", 10)
        or 10
    )

    def run_view(
        view_index: int, group_order: list[str], ensemble_stage: str
    ) -> dict[str, Any]:
        visible_groups = set(group_order)
        prior_groups = set(
            str(value)
            for value in (previous_selection or {}).get("object_group_ids") or []
            if str(value)
        )
        visible_prior = (
            previous_selection
            if previous_selection and prior_groups.issubset(visible_groups)
            else None
        )
        call = _call_resolution_stage(
            llm,
            lambda correction: _build_record_object_selection_prompt(
                query,
                record,
                evidence_records,
                correction=correction,
                previous_selection=visible_prior,
                object_group_order=group_order,
            ),
            lambda raw: _parse_record_object_selection(
                raw,
                task_record_id=task_record_id,
                allowed_object_group_ids=visible_groups,
            ),
            requested_max=256,
            retries=retries,
            sampling_temperature=sampling_temperature,
        )
        call["order_view_index"] = view_index
        call["object_group_order"] = group_order
        call["ensemble_stage"] = ensemble_stage
        return call

    def run_views(
        view_orders: list[list[str]], ensemble_stage: str
    ) -> list[dict[str, Any]]:
        view_orders = view_orders or [[]]
        calls: list[dict[str, Any]] = [{} for _ in view_orders]
        if len(view_orders) == 1:
            calls[0] = run_view(0, view_orders[0], ensemble_stage)
            return calls
        worker_count = max(1, min(configured_workers, len(view_orders)))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
        futures = {
            submit_with_context(
                executor, run_view, index, group_order, ensemble_stage
            ): index
            for index, group_order in enumerate(view_orders)
        }
        cancelled = False
        try:
            for future in concurrent.futures.as_completed(
                futures, timeout=task_wait_timeout()
            ):
                index = futures[future]
                try:
                    calls[index] = future.result()
                except Exception as exc:
                    calls[index] = {
                        "status": "unavailable",
                        "value": None,
                        "attempts": 0,
                        "prompt": "",
                        "raw_model_output": "",
                        "error": f"{type(exc).__name__}: {exc}"[:1000],
                        "completion_token_budget": 256,
                        "order_view_index": index,
                        "object_group_order": view_orders[index],
                        "ensemble_stage": ensemble_stage,
                    }
        except concurrent.futures.TimeoutError:
            cancelled = True
            raise
        finally:
            shutdown_pool(executor, list(futures), cancelled=cancelled)
        return calls

    hierarchy_enabled = bool(
        DATA_PIPELINE.get("evidence_resolution_hierarchy_enabled", True)
    )
    group_size = max(
        2,
        min(
            int(DATA_PIPELINE.get("evidence_resolution_hierarchy_group_size", 5) or 5),
            8,
        ),
    )
    coverage = max(
        1,
        min(
            int(DATA_PIPELINE.get("evidence_resolution_hierarchy_coverage", 2) or 2),
            4,
        ),
    )
    finalist_limit = max(
        2,
        min(
            int(DATA_PIPELINE.get("evidence_resolution_hierarchy_finalists", 5) or 5),
            8,
        ),
    )
    finalist_group_ids = list(base_group_ids)
    nomination_calls: list[dict[str, Any]] = []
    hierarchy: dict[str, Any] = {
        "enabled": False,
        "input_group_count": len(base_group_ids),
        "finalist_group_ids": list(base_group_ids),
    }
    if hierarchy_enabled and len(base_group_ids) > finalist_limit:
        batches = build_overlapping_candidate_batches(
            base_group_ids,
            group_size=group_size,
            coverage=coverage,
        )
        nomination_calls = run_views(batches, "object_nomination")
        base_index = {group_id: index for index, group_id in enumerate(base_group_ids)}
        stats = {
            group_id: {
                "appearances": 0,
                "nominations": 0,
                "confidence_total": 0,
            }
            for group_id in base_group_ids
        }
        for batch, call in zip(batches, nomination_calls):
            for group_id in batch:
                stats[group_id]["appearances"] += 1
            value = call.get("value")
            if call.get("status") != "ok" or not isinstance(value, Mapping):
                continue
            confidence = int(value.get("confidence") or 0)
            for group_id in value.get("object_group_ids") or []:
                if group_id in stats:
                    stats[group_id]["nominations"] += 1
                    stats[group_id]["confidence_total"] += confidence

        nominated = [
            group_id
            for group_id in base_group_ids
            if stats[group_id]["nominations"] > 0
        ]
        nominated.sort(
            key=lambda group_id: (
                -stats[group_id]["nominations"],
                -(
                    stats[group_id]["confidence_total"]
                    / stats[group_id]["nominations"]
                ),
                base_index[group_id],
            )
        )
        prior_group_ids = [
            str(value)
            for value in (previous_selection or {}).get("object_group_ids") or []
            if str(value) in allowed_object_group_ids
        ]
        ranked = list(dict.fromkeys([*prior_group_ids, *nominated]))
        fallback_full_set = not ranked
        if ranked:
            finalist_group_ids = ranked[:finalist_limit]
        hierarchy = {
            "enabled": True,
            "method": "overlapping_id_nomination.v1",
            "input_group_count": len(base_group_ids),
            "group_size": group_size,
            "coverage": coverage,
            "batch_count": len(batches),
            "valid_batch_count": sum(
                call.get("status") == "ok" for call in nomination_calls
            ),
            "nominated_group_count": len(nominated),
            "fallback_full_set": fallback_full_set,
            "finalist_group_ids": list(finalist_group_ids),
            "nomination_stats": {
                group_id: {
                    "appearances": stats[group_id]["appearances"],
                    "nominations": stats[group_id]["nominations"],
                    "model_confidence_mean": (
                        round(
                            stats[group_id]["confidence_total"]
                            / stats[group_id]["nominations"]
                        )
                        if stats[group_id]["nominations"]
                        else 0
                    ),
                }
                for group_id in base_group_ids
            },
        }

    views = (
        build_balanced_order_views(
            finalist_group_ids, max_views=configured_views
        )
        if ensemble_enabled
        else [finalist_group_ids]
    )
    selection_calls = run_views(views, "object_selection")
    selection = _aggregate_object_selection_calls(selection_calls)
    if selection is not None:
        selection["consensus"]["hierarchy"] = hierarchy

    calls = [*nomination_calls, *selection_calls]
    errors = [str(call.get("error") or "") for call in calls if call.get("error")]
    return {
        "status": "ok" if selection is not None else "unavailable",
        "value": selection,
        "calls": calls,
        "attempts": sum(int(call.get("attempts") or 0) for call in calls),
        "error": " | ".join(errors)[:2000],
        "completion_token_budget": sum(
            int(call.get("completion_token_budget") or 0) for call in calls
        ),
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
    *,
    previous_resolution: Mapping[str, Any] | None = None,
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
    previous_decisions: dict[str, Mapping[str, Any]] = {}
    record_consensus: dict[str, dict[str, Any]] = {}
    if isinstance(previous_resolution, Mapping):
        for row in previous_resolution.get("decisions") or []:
            if isinstance(row, Mapping) and str(row.get("task_record_id") or ""):
                status = str(row.get("status") or "")
                selected_group = str(row.get("selected_object_group_id") or "")
                conflicting_groups = [
                    str(value)
                    for value in row.get("conflicting_object_group_ids") or []
                    if str(value)
                ]
                previous_decisions[str(row["task_record_id"])] = {
                    "selection": (
                        "selected"
                        if status == "resolved" and selected_group
                        else "conflict"
                        if status == "conflict" and conflicting_groups
                        else "missing"
                    ),
                    "object_group_ids": (
                        conflicting_groups
                        if status == "conflict"
                        else [selected_group]
                        if selected_group
                        else []
                    ),
                }

    for record in records:
        task_record_id = record_id(record)
        object_call = _run_object_selection_ensemble(
            query,
            record,
            bounded_evidence_records,
            llm,
            allowed_object_group_ids=allowed_groups,
            retries=retries,
            sampling_temperature=sampling_temperature,
            previous_selection=previous_decisions.get(task_record_id),
        )
        total_attempts += int(object_call.get("attempts") or 0)
        for view_call in object_call.get("calls") or []:
            parsed = view_call.get("value") if isinstance(view_call.get("value"), Mapping) else {}
            stage_calls.append(
                {
                    "task_record_id": task_record_id,
                    "stage": str(
                        view_call.get("ensemble_stage") or "object_selection"
                    ),
                    **{
                        key: value
                        for key, value in view_call.items()
                        if key != "value"
                    },
                    "parsed_selection": (
                        {
                            "selection": parsed.get("selection"),
                            "object_group_ids": parsed.get("object_group_ids") or [],
                            "confidence": parsed.get("confidence"),
                        }
                        if parsed
                        else {}
                    ),
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
        record_consensus[task_record_id] = {
            "object_selection": dict(selection.get("consensus") or {}),
            "model_confidence": int(selection.get("model_confidence") or 0),
            "combined_confidence": int(selection.get("confidence") or 0),
        }
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
        "consensus": record_consensus,
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
    consensus_by_record = (
        resolution.get("consensus")
        if isinstance(resolution.get("consensus"), Mapping)
        else {}
    )
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
        record_consensus = consensus_by_record.get(str(raw.get("task_record_id") or ""))
        if isinstance(record_consensus, Mapping):
            object_consensus = record_consensus.get("object_selection")
            if isinstance(object_consensus, Mapping):
                values.append(
                    "order_consensus="
                    + str(object_consensus.get("winning_view_count") or 0)
                    + "/"
                    + str(object_consensus.get("valid_view_count") or 0)
                )
                values.append(
                    "object_confidence="
                    + str(record_consensus.get("combined_confidence") or 0)
                    + "/100"
                )
        values.append(
            "needs_more_evidence="
            + ("true" if raw.get("needs_more_evidence") else "false")
        )
        line = "; ".join(values)
        if get_token_count("\n".join([*lines, line])) > max(64, int(max_tokens)):
            break
        lines.append(line)
    return "\n".join(lines) if len(lines) > 1 else ""


def attach_evidence_resolution_to_context(
    context: Mapping[str, Any],
    resolution: Mapping[str, Any],
    task_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach RWKV advisory state without mutating the factual evidence lane.

    Evidence Resolution is not a fact gate.  Its model-authored output may help
    the final RWKV Writer focus, but it must never delete, replace, filter, or
    reorder the exact spans already admitted by ``build_evidence_context``.
    ``task_plan`` remains in the signature for call-site compatibility only.
    """

    del task_plan

    immutable_lane = {
        key: deepcopy(context[key])
        for key in (
            "text",
            "evidence_text",
            "selected_evidence",
            "citation_refs",
        )
        if key in context
    }
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
    if "evidence_text" not in output:
        output["evidence_text"] = str(output.get("text") or "")
    if "text" not in output:
        output["text"] = str(output.get("evidence_text") or "")
    if "context_tokens" not in output:
        output["context_tokens"] = get_token_count(str(output.get("text") or ""))
    changed_lane_keys = [
        key for key, value in immutable_lane.items() if output.get(key) != value
    ]
    if changed_lane_keys:
        raise RuntimeError(
            "Evidence Resolution mutated the immutable evidence lane: "
            + ", ".join(changed_lane_keys)
        )
    evidence_lane_digest = hashlib.sha256(
        json.dumps(
            immutable_lane,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    stats = dict(output.get("context_stats") or {})
    stats.update(
        {
            "context_tokens": int(output.get("context_tokens") or 0),
            "evidence_resolution_status": str(resolution_copy.get("status") or ""),
            "evidence_resolution_evidence_record_count": int(
                resolution_copy.get("evidence_record_count") or 0
            ),
            "evidence_resolution_view_tokens": get_token_count(view),
            "evidence_resolution_coverage_complete": bool(
                resolution_copy.get("coverage_complete")
            ),
            "evidence_resolution_advisory_only": True,
            "evidence_lane_preserved": True,
            "evidence_lane_digest": evidence_lane_digest,
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
    "parse_evidence_resolution_output",
    "render_evidence_resolution_view",
    "evidence_resolution_signature",
    "evidence_resolution_completion_token_budget",
    "resolve_evidence",
]
