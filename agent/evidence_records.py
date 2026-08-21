"""Assemble grounded RWKV spans into atomic Evidence Record envelopes.

This module owns transport identity only. It never decides whether a span is
true, current, complete, authoritative, or sufficient, and it never creates
or edits answer text. Every assembled quote is the exact grounded quote that
entered the module.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[: max(1, int(limit))]


def _source_object_id(item: Mapping[str, Any]) -> str:
    source_object = item.get("source_object")
    if isinstance(source_object, Mapping):
        value = str(source_object.get("source_object_id") or "").strip()
        if value:
            return value
    return str(item.get("url") or item.get("content_sha256") or "").strip()


def _locator_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    locator = candidate.get("source_locator")
    locator = dict(locator) if isinstance(locator, Mapping) else {}
    return {
        "chunk_id": str(
            locator.get("chunk_id") or candidate.get("chunk_id") or ""
        ),
        "char_start": int(locator.get("char_start") or 0),
        "char_end": int(locator.get("char_end") or 0),
    }


def _opaque_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:20]


def _merge_text_list(current: Any, incoming: Any, *, limit: int) -> list[str]:
    values: list[Any] = []
    for raw in (current, incoming):
        if isinstance(raw, (list, tuple, set)):
            values.extend(raw)
        elif raw not in (None, ""):
            values.append(raw)
    return list(
        dict.fromkeys(
            _bounded_text(value, 160)
            for value in values
            if str(value or "").strip()
        )
    )[: max(1, int(limit))]


def assemble_grounded_candidates(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one atomic transport record per distinct grounded source span.

    ``chunk_candidates`` contains the pre-merge audit rows and is therefore
    authoritative when available. ``candidates`` remains a compatibility
    input for structured connectors and historical fixtures. Duplicate views
    of the same exact source span collapse to one row, but two locators from
    one URL/Task Record never collapse merely because their record key is
    empty.
    """

    source_id = _source_object_id(item)
    rows: list[dict[str, Any]] = []
    row_by_span_id: dict[str, dict[str, Any]] = {}
    candidates = [
        *list(item.get("chunk_candidates") or []),
        *list(item.get("candidates") or []),
    ]
    for raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        if raw.get("supported") is not True or raw.get("source_grounded") is not True:
            continue
        quote = str(raw.get("quote") or "").strip()
        if not quote:
            continue
        candidate = deepcopy(dict(raw))
        locator = _locator_identity(candidate)
        span_payload = {
            "source_object_id": source_id,
            "chunk_id": locator["chunk_id"],
            "char_start": locator["char_start"],
            "char_end": locator["char_end"],
            # A locator can be absent in archived/structured fixtures. Quote
            # identity keeps those records stable without interpreting text.
            "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        }
        record_span_id = _opaque_id("SPAN-", span_payload)
        existing = row_by_span_id.get(record_span_id)
        if existing is not None:
            # The same exact span can appear in both the pre-merge audit view
            # and compatibility view, or be bound by RWKV to several task
            # points. Collapse the duplicate text only; retain every model-
            # authored task/field binding instead of keeping whichever view
            # happened to arrive first.
            for key in ("task_record_ids",):
                merged = _merge_text_list(existing.get(key), candidate.get(key), limit=8)
                if merged:
                    existing[key] = merged
            merged_fields = _merge_text_list(
                existing.get("field_ids"), candidate.get("field_ids"), limit=16
            )
            if merged_fields:
                existing["field_ids"] = merged_fields
            continue
        parent_payload = {
            **span_payload,
            "task_record_ids": [
                _bounded_text(value, 80)
                for value in (candidate.get("task_record_ids") or [])
                if str(value or "").strip()
            ][:8],
            "record_key": _bounded_text(candidate.get("record_key"), 300),
        }
        candidate.update(
            {
                "record_span_id": record_span_id,
                "parent_candidate_id": _opaque_id("C-", parent_payload),
                "assembly_basis": "single_grounded_span",
                "atomic_quote_index": 0,
                "source_object_id": source_id,
            }
        )
        rows.append(candidate)
        row_by_span_id[record_span_id] = candidate
    return rows


__all__ = ["assemble_grounded_candidates"]
