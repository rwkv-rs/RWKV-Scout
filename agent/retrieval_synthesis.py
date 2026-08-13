"""Minimal evidence packing and the single RWKV final-writer call.

This module deliberately has no answer gate, repair pass, translation pass,
deterministic answer fallback, refusal generator, or quality status. Retrieval
code may organise source material, but only RWKV writes the public answer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from config import (
    DATA_PIPELINE,
    get_llm_context_length,
    get_model_stage_temperature,
    get_model_stage_sampling,
    is_local_provider,
    model_sampling_parameters,
)
from utils.chunker import get_token_count, semantic_chunk_text
from utils.context_budget import evidence_tokens
from utils.freshness import extract_explicit_date
from utils.model_budget import bounded_completion_budget
from utils.rwkv_prompt import (
    build_final_continuation_prompt,
    consume_final_prefill_boundary,
)
from agent.task_plan_contract import compact_task_plan, task_points
from agent.retrieval_object_contract import merge_mapping_rows, task_record_contract


def _clean_answer(value: Any) -> str:
    """Return the exact text emitted by RWKV.

    The historical function name is retained only for import compatibility.
    No whitespace trimming, protocol stripping, deduplication, citation
    rewriting, or other answer transformation is permitted here.
    """

    return "" if value is None else str(value)


def _source_body(item: dict[str, Any]) -> str:
    """Return the richest source text already produced by retrieval."""

    for key in (
        "source_excerpt",
        "page_excerpt",
        "structured_evidence_text",
        "content",
        "abstract",
    ):
        value = str(item.get(key) or "").replace("\x00", "").strip()
        if value:
            return value
    return ""


def _source_identity(item: dict[str, Any]) -> str:
    """Build a stable resource identity without judging source semantics."""

    record_metadata = item.get("record_metadata")
    if isinstance(record_metadata, dict):
        source_object = record_metadata.get("source_object")
        source_object_id = (
            str(source_object.get("source_object_id") or "").strip().casefold()
            if isinstance(source_object, dict)
            else ""
        )
        record_key = str(
            record_metadata.get("record_key")
            or (
                source_object.get("source_record_id")
                if isinstance(source_object, dict)
                else ""
            )
            or ""
        ).strip().casefold()
        if source_object_id and record_key:
            # Multiple grounded spans may describe different requested fields
            # of one observable source record. Keep them in one context block
            # instead of making RWKV reconstruct the tuple across unrelated
            # [S#] entries. Empty record keys intentionally do not merge.
            return f"source-record:{source_object_id}:{record_key}"
        task_record_ids = [
            str(value).strip()
            for value in (
                record_metadata.get("task_record_ids")
                or [record_metadata.get("task_record_id")]
            )
            if str(value or "").strip()
        ]
        if source_object_id and task_record_ids:
            # When RWKV did not emit a literal record key, group the grounded
            # spans only at the source-object + task-record boundary. The
            # context block explicitly keeps record identity unresolved, so
            # this removes duplicate page headers without claiming that all
            # spans describe one version/date/advisory.
            return (
                "source-object-task:"
                + source_object_id
                + ":"
                + ",".join(sorted(set(task_record_ids)))
            )
    evidence_record_id = str(item.get("evidence_record_id") or "").strip()
    if evidence_record_id:
        # Candidate records without an observable record key remain separate;
        # URL-level merging could conflate versions, advisories or table rows.
        return f"record:{evidence_record_id}"
    url = str(item.get("url") or "").strip().casefold()
    if url:
        url = re.sub(r"^https?://(?:www\.)?", "", url)
        url = url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
        return f"url:{url}"
    digest = str(item.get("content_sha256") or "").strip().casefold()
    if digest:
        return f"sha256:{digest}"
    title = " ".join(str(item.get("title") or "").split()).casefold()
    return f"title:{title}" if title else ""


def _has_grounded_locator(item: dict[str, Any]) -> bool:
    """Return whether a source carries a grounded RWKV-selected source span."""

    return any(
        isinstance(row, dict)
        and row.get("supported") is True
        and row.get("source_grounded") is True
        and str(row.get("quote") or "").strip()
        for row in item.get("chunk_candidates") or []
    )


def _source_chunks(
    item: dict[str, Any],
    *,
    chunk_tokens: int = 1000,
    grounded_span_limit: int = 3,
) -> list[dict[str, Any]]:
    """Pack query-focused spans and exact locators without duplicate context.

    Page extraction keeps ``selected_source_chunks`` as an attention-routing
    result and ``source_chunks`` as the complete provenance record. Exact
    locators and selected source windows often overlap. Round 19 packed
    both, creating hundreds of exact/containment duplicates and wasting RWKV's
    attention budget.  The highest-priority focused span now wins: a later
    full source chunk must not replace the exact window selected for RWKV.
    Locator claim ids are merged into that span. This is text deduplication
    only; no fact is accepted, rejected, rewritten, or selected as the answer.
    """

    # These fields are computed by the page-evidence attention router.  They
    # describe the literal record window; they do not decide whether a fact is
    # true or current.  Keep them attached while spans are deduplicated so the
    # final RWKV writer can receive the same minimal temporal context that the
    # extractor used.  Previously `_source_chunks` rebuilt each row from only
    # id/index/text and silently dropped this cross-layer state.
    routing_metadata_fields = (
        "record_date",
        "record_date_precision",
        "record_temporal_role",
    )

    def routing_metadata(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: row[key]
            for key in routing_metadata_fields
            if row.get(key) not in (None, "", [], {})
        }

    grounded_rows = [
        {
            "chunk_id": "locator-" + str(row.get("chunk_id") or index + 1),
            "index": int(row.get("chunk_index") or index),
            "text": str(row.get("quote") or "").strip(),
            "claim_ids": list(row.get("claim_ids") or []),
            "packing_role": "grounded_locator",
            "_packing_priority": 2,
            **routing_metadata(row),
        }
        for index, row in enumerate(item.get("chunk_candidates") or [])
        if isinstance(row, dict)
        and row.get("supported") is True
        and row.get("source_grounded") is True
        and str(row.get("quote") or "").strip()
    ][: max(1, int(grounded_span_limit))]
    selected_rows = [
        {
            **row,
            "packing_role": "selected_attention_window",
            "_packing_priority": 3,
        }
        for row in item.get("selected_source_chunks") or []
        if isinstance(row, dict) and str(row.get("text") or "").strip()
    ]
    selected_rows.sort(
        key=lambda row: (
            int(row.get("attention_rank") or 10**6),
            -int(row.get("attention_score") or 0),
            int(row.get("index") or 0),
        )
    )
    original_rows = [
        {
            **row,
            "packing_role": "source_neighbour",
            "_packing_priority": 1,
        }
        for row in item.get("source_chunks") or []
        if isinstance(row, dict) and str(row.get("text") or "").strip()
    ]
    focus_indexes = {
        int(row.get("index") or 0)
        for row in [*selected_rows, *grounded_rows]
    }
    neighbour_rows = [
        row
        for row in original_rows
        if any(abs(int(row.get("index") or 0) - focus) <= 1 for focus in focus_indexes)
    ]
    preferred_rows = [*selected_rows, *grounded_rows, *neighbour_rows]
    # Once RWKV has located grounded candidate evidence, the final writer needs that span
    # and its immediate source neighbourhood, not an unrelated page preamble
    # or every historical table row from the same document.  The complete
    # source_chunks record remains in state for recovery and later replans.
    chunks: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()

    def normalized(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().casefold()

    def merged_claim_ids(*values: Any) -> list[str]:
        return list(
            dict.fromkeys(
                str(item).strip()
                for value in values
                for item in (value or [])
                if str(item).strip()
            )
        )[:8]

    routed_rows = preferred_rows if preferred_rows else original_rows
    for index, row in enumerate(routed_rows):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        chunk_id = str(row.get("chunk_id") or f"chunk-{index + 1}")
        chunk_index = int(row.get("index", index) or index)
        identity = (chunk_id, chunk_index, text)
        normalized_text = normalized(text)
        if identity in seen:
            continue
        seen.add(identity)
        # Keep the already-routed span when it contains a later row. Locator
        # and task bindings are provenance, so merge them instead of retaining
        # a second copy of the same source text.
        contained_by = next(
            (
                existing
                for existing in chunks
                if len(normalized_text) >= 24
                and int(existing.get("index") or 0) == chunk_index
                and normalized_text in normalized(existing.get("text"))
            ),
            None,
        )
        if contained_by is not None:
            contained_by["claim_ids"] = merged_claim_ids(
                contained_by.get("claim_ids"), row.get("claim_ids")
            )
            locator_ids = list(contained_by.get("contained_locator_ids") or [])
            if str(chunk_id).startswith("locator-") and chunk_id not in locator_ids:
                locator_ids.append(chunk_id)
            if locator_ids:
                contained_by["contained_locator_ids"] = locator_ids
            continue

        contained_indexes = [
            existing_index
            for existing_index, existing in enumerate(chunks)
            if len(normalized(existing.get("text"))) >= 24
            and int(existing.get("index") or 0) == chunk_index
            and normalized(existing.get("text")) in normalized_text
        ]
        row_priority = int(row.get("_packing_priority") or 0)
        if contained_indexes:
            strongest_index = max(
                contained_indexes,
                key=lambda existing_index: int(
                    chunks[existing_index].get("_packing_priority") or 0
                ),
            )
            strongest = chunks[strongest_index]
            strongest_priority = int(strongest.get("_packing_priority") or 0)
            if row_priority <= strongest_priority:
                # A broad neighbour/full chunk arriving after a selected
                # exact window used to erase that window here. Preserve the
                # routing decision and merge provenance only.
                strongest["claim_ids"] = merged_claim_ids(
                    strongest.get("claim_ids"), row.get("claim_ids")
                )
                if str(chunk_id).startswith("locator-"):
                    locator_ids = list(strongest.get("contained_locator_ids") or [])
                    if chunk_id not in locator_ids:
                        locator_ids.append(chunk_id)
                    strongest["contained_locator_ids"] = locator_ids
                continue

        inherited_claim_ids: list[str] = []
        inherited_locator_ids: list[str] = []
        for existing_index in reversed(contained_indexes):
            existing = chunks.pop(existing_index)
            inherited_claim_ids = merged_claim_ids(
                inherited_claim_ids, existing.get("claim_ids")
            )
            inherited_locator_ids.extend(existing.get("contained_locator_ids") or [])
            if str(existing.get("chunk_id") or "").startswith("locator-"):
                inherited_locator_ids.append(str(existing.get("chunk_id") or ""))
        parts = (
            semantic_chunk_text(
                text,
                max_tokens=max(128, int(chunk_tokens)),
                overlap_ratio=0.04,
            )
            if get_token_count(text) > max(128, int(chunk_tokens))
            else [text]
        )
        for part_index, part in enumerate(parts, start=1):
            packed_row = {
                    "chunk_id": (
                        chunk_id
                        if len(parts) == 1
                        else f"{chunk_id}.part-{part_index}"
                    ),
                    "index": chunk_index,
                    "text": part,
                    "claim_ids": merged_claim_ids(
                        row.get("claim_ids"), inherited_claim_ids
                    ),
                    "packing_role": str(row.get("packing_role") or "source_chunk"),
                    "_packing_priority": row_priority,
                    "_routing_order": index,
                    "_part_index": part_index,
                    **routing_metadata(row),
                }
            locator_ids = list(dict.fromkeys(inherited_locator_ids))
            if locator_ids:
                packed_row["contained_locator_ids"] = locator_ids
            chunks.append(packed_row)
    if chunks:
        # A single very long source chunk must not consume every per-source
        # slot before another neighbouring record receives one. Interleave
        # semantic parts within the same routing tier while preserving focused
        # windows ahead of locators and broad neighbours.
        chunks.sort(
            key=lambda row: (
                -int(row.get("_packing_priority") or 0),
                int(row.get("_part_index") or 1),
                int(row.get("_routing_order") or 0),
            )
        )
        return chunks

    body = _source_body(item)
    return [
        {"chunk_id": f"chunk-{index + 1}", "index": index, "text": text}
        for index, text in enumerate(
            semantic_chunk_text(body, max_tokens=max(128, int(chunk_tokens)), overlap_ratio=0.08)
        )
        if text.strip()
    ]


def _claim_projection(value: Any) -> list[dict[str, Any]]:
    """Expose Claim Ledger progress as model context, never as a gate."""

    if not isinstance(value, dict):
        return []
    projected: list[dict[str, Any]] = []
    for row in value.get("claims") or []:
        if not isinstance(row, dict):
            continue
        spans = [
            span
            for span in (row.get("sources") or row.get("evidence") or [])
            if isinstance(span, dict)
        ]
        source_urls = list(
            dict.fromkeys(
                str(span.get("url") or "").strip()
                for span in spans
                if str(span.get("url") or "").strip()
            )
        )
        grounded_source_count = sum(
            bool(span.get("grounded_spans")) for span in spans
        )
        projected.append(
            {
                "claim_id": str(row.get("claim_id") or row.get("point_id") or ""),
                "question": str(
                    row.get("question") or row.get("task") or row.get("objective") or ""
                )[:600],
                "fields": [
                    str(item)[:400]
                    for item in row.get("fields") or row.get("evidence_needed") or []
                    if str(item).strip()
                ][:8],
                "time_scope": str(row.get("time_scope") or "unspecified"),
                "retrieval_state": str(row.get("retrieval_state") or ""),
                "retrieved_source_count": len(spans),
                "grounded_source_count": grounded_source_count,
                "bound_source_urls": source_urls[:8],
            }
        )
    unassigned = [
        row for row in value.get("unassigned_sources") or [] if isinstance(row, dict)
    ]
    if unassigned:
        projected.append(
            {
                "claim_id": "UNASSIGNED",
                "question": (
                    "Retrieved material not bound to a specific atomic point by RWKV; "
                    "inspect it directly and do not assume it covers every point."
                ),
                "retrieval_state": "retrieved_unassigned",
                "retrieved_source_count": len(unassigned),
                "source_urls": [
                    str(row.get("url") or "")
                    for row in unassigned[:8]
                    if str(row.get("url") or "")
                ],
            }
        )
    return projected


def _claim_grounded_source_items(
    value: Any,
    *,
    grounded_span_limit: int = 3,
) -> list[dict[str, Any]]:
    """Project model-grounded ledger spans into ordinary final source blocks.

    The Claim Ledger previously embedded every span, locator coordinate and
    source metadata object inside the checklist while final source selection
    could omit the very page carrying the span.  That made the checklist much
    larger than the actual evidence section.  This projection keeps each
    grounded quote bound to its URL and task point, then lets the normal source
    packer assign [S#] references and budgets.
    """

    if not isinstance(value, dict):
        return []
    projected_by_claim: list[list[dict[str, Any]]] = []
    for claim in value.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or claim.get("point_id") or "").strip()
        claim_items: list[dict[str, Any]] = []
        evidence_records = [
            row
            for row in claim.get("evidence_records") or []
            if isinstance(row, dict)
            and str(row.get("quote") or "").strip()
        ]
        if evidence_records:
            for record in evidence_records[: max(2, int(grounded_span_limit) * 3)]:
                quote = str(record.get("quote") or "").strip()
                context_role = "candidate_evidence_record"
                item: dict[str, Any] = {
                    "evidence_record_id": str(
                        record.get("evidence_record_id") or ""
                    ),
                    "evidence_kind": "evidence_record",
                    "context_role": context_role,
                    "title": str(
                        record.get("title")
                        or record.get("url")
                        or "Grounded evidence record"
                    ),
                    "url": str(record.get("url") or ""),
                    "claim_ids": [claim_id] if claim_id else [],
                    "content": quote,
                    "record_metadata": {
                        "task_record_id": claim_id,
                        "task_record_ids": [claim_id] if claim_id else [],
                        "task_record_relation": str(claim.get("relation") or ""),
                        "task_record_time_scope": str(
                            claim.get("time_scope") or "unspecified"
                        ),
                        "subject_key": str(record.get("subject_key") or ""),
                        "record_key": str(record.get("record_key") or ""),
                        "field_keys": list(record.get("field_keys") or [])[:16],
                        "support_state": str(record.get("support_state") or ""),
                        "record_match": str(record.get("record_match") or ""),
                        "field_contract_valid": bool(
                            record.get("field_contract_valid", True)
                        ),
                        "binding_origin": str(record.get("binding_origin") or ""),
                        "source_object": dict(record.get("source_object") or {}),
                        "object_alignment": dict(
                            record.get("object_alignment") or {}
                        ),
                        "object_alignments": merge_mapping_rows(
                            record.get("object_alignments"),
                            record.get("object_alignment"),
                        )[:8],
                        "rwkv_subject_alignment": dict(
                            record.get("rwkv_subject_alignment") or {}
                        ),
                        "task_object_alignments": [
                            dict(value)
                            for value in record.get("task_object_alignments") or []
                            if isinstance(value, dict)
                        ][:8],
                        "retrieval_request": dict(record.get("retrieval_request") or {}),
                        "retrieval_requests": merge_mapping_rows(
                            record.get("retrieval_requests"),
                            record.get("retrieval_request"),
                        )[:8],
                        "retrieval_bindings": merge_mapping_rows(
                            record.get("retrieval_bindings")
                        )[:8],
                    },
                    "chunk_candidates": [
                        {
                            "supported": True,
                            "source_grounded": True,
                            "chunk_id": str(record.get("chunk_id") or ""),
                            "chunk_index": int(record.get("chunk_index") or 0),
                            "claim_ids": [claim_id] if claim_id else [],
                            "field_keys": list(record.get("field_keys") or [])[:16],
                            "subject_key": str(record.get("subject_key") or ""),
                            "record_key": str(record.get("record_key") or ""),
                            "quote": quote,
                            "source_locator": dict(record.get("source_locator") or {}),
                            "grounding_basis": str(record.get("grounding_basis") or ""),
                        }
                    ],
                }
                for key in (
                    "source",
                    "provider",
                    "source_type",
                    "published",
                    "published_at",
                    "updated",
                    "updated_at",
                    "date",
                    "retrieved_at",
                    "freshness",
                    "source_kind",
                    "authority",
                    "connector",
                    "operation",
                    "source_object",
                    "retrieval_request",
                    "object_alignments",
                    "retrieval_requests",
                    "retrieval_bindings",
                ):
                    if record.get(key) not in (None, "", [], {}):
                        item[key] = record[key]
                claim_items.append(item)
            projected_by_claim.append(claim_items)
            continue
        for source in claim.get("sources") or claim.get("evidence") or []:
            if not isinstance(source, dict):
                continue
            grounded = [
                row
                for row in source.get("grounded_spans") or []
                if isinstance(row, dict) and str(row.get("text") or "").strip()
            ][: max(1, int(grounded_span_limit))]
            if not grounded:
                continue
            item: dict[str, Any] = {
                "title": str(source.get("title") or source.get("url") or "Grounded source"),
                "url": str(source.get("url") or ""),
                "claim_ids": [claim_id] if claim_id else [],
                "context_role": "legacy_bound_span",
                "content": "\n\n".join(str(row.get("text") or "") for row in grounded),
                "source_chunks": [
                    dict(row)
                    for row in source.get("chunks") or []
                    if isinstance(row, dict) and str(row.get("text") or "").strip()
                ],
                "selected_source_chunks": [
                    dict(row)
                    for row in source.get("selected_chunks") or []
                    if isinstance(row, dict) and str(row.get("text") or "").strip()
                ],
                "chunk_candidates": [
                    {
                        "supported": True,
                        "source_grounded": True,
                        "chunk_id": str(row.get("chunk_id") or ""),
                        "chunk_index": int(row.get("index") or 0),
                        "quote": str(row.get("text") or "")[:1200],
                        "source_locator": dict(row.get("source_locator") or {}),
                        "grounding_basis": str(row.get("grounding_basis") or ""),
                    }
                    for row in grounded
                ],
            }
            for key in (
                "source",
                "provider",
                "source_type",
                "published",
                "published_at",
                "updated",
                "updated_at",
                "date",
                "retrieved_at",
                "freshness",
            ):
                if source.get(key) not in (None, "", [], {}):
                    item[key] = source[key]
            claim_items.append(item)
        if claim_items:
            projected_by_claim.append(claim_items)
    projected: list[dict[str, Any]] = []
    cursor = 0
    while any(cursor < len(items) for items in projected_by_claim):
        for items in projected_by_claim:
            if cursor < len(items):
                projected.append(items[cursor])
        cursor += 1
    return projected


def _unbound_fallback_items(
    items: Iterable[dict[str, Any]],
    *,
    excluded_urls: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Keep a small isolated escape hatch when extraction missed a page.

    This does not bind a source to a task record and does not decide that the
    page is relevant. It only preserves bounded fetched text for RWKV to judge
    instead of turning an extractor false-negative into an empty answer.
    """

    output: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        if not _source_body(item):
            continue
        url_identity = re.sub(
            r"^https?://(?:www\.)?",
            "",
            str(item.get("url") or "").strip().casefold(),
        ).split("#", 1)[0].split("?", 1)[0].rstrip("/")
        if url_identity and url_identity in excluded_urls:
            continue
        item["claim_ids"] = []
        item["context_role"] = "unbound_source_fallback"
        output.append(item)
        if len(output) >= max(0, int(limit)):
            break
    return output


def _merge_source_records(items: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge only identical observable records and exact duplicate resources."""

    merged: list[dict[str, Any]] = []
    index_by_identity: dict[str, int] = {}
    duplicate_count = 0
    for raw_item in items:
        item = dict(raw_item)
        identity = _source_identity(item)
        if not identity or identity not in index_by_identity:
            if identity:
                index_by_identity[identity] = len(merged)
            merged.append(item)
            continue
        duplicate_count += 1
        current = merged[index_by_identity[identity]]
        for key, value in item.items():
            if key in {
                "object_alignments",
                "retrieval_requests",
                "retrieval_bindings",
            }:
                current[key] = merge_mapping_rows(current.get(key), value)
            elif key in {
                "claim_ids",
                "source_chunks",
                "selected_source_chunks",
                "chunk_candidates",
            }:
                existing_rows = list(current.get(key) or [])
                incoming_rows = list(value or []) if isinstance(value, list) else []
                combined: list[Any] = []
                seen_rows: set[str] = set()
                for row in [*existing_rows, *incoming_rows]:
                    marker = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
                    if marker in seen_rows:
                        continue
                    seen_rows.add(marker)
                    combined.append(row)
                current[key] = combined
            elif key == "record_metadata" and isinstance(value, dict):
                existing_metadata = current.get("record_metadata")
                if not isinstance(existing_metadata, dict):
                    current["record_metadata"] = dict(value)
                    continue
                for metadata_key, metadata_value in value.items():
                    if metadata_key in {"field_keys", "task_record_ids"}:
                        existing_values = list(existing_metadata.get(metadata_key) or [])
                        incoming_values = (
                            list(metadata_value or [])
                            if isinstance(metadata_value, list)
                            else [metadata_value]
                        )
                        existing_metadata[metadata_key] = list(
                            dict.fromkeys(
                                str(row)
                                for row in [*existing_values, *incoming_values]
                                if str(row).strip()
                            )
                        )[:16]
                    elif metadata_key in {
                        "object_alignments",
                        "retrieval_requests",
                        "retrieval_bindings",
                        "task_object_alignments",
                    }:
                        existing_metadata[metadata_key] = merge_mapping_rows(
                            existing_metadata.get(metadata_key),
                            metadata_value,
                        )
                    elif existing_metadata.get(metadata_key) in (None, "", [], {}):
                        existing_metadata[metadata_key] = metadata_value
            elif current.get(key) in (None, "", [], {}):
                current[key] = value
    return merged, duplicate_count


def _citation_refs(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        url = str(item.get("url") or "").strip()
        chunks = [
            row
            for row in item.get("packed_chunks") or []
            if isinstance(row, dict) and str(row.get("text") or "").strip()
        ]
        refs.append(
            {
                "ref_id": f"S{index}",
                "title": str(item.get("title") or url or f"Source {index}"),
                "url": url,
                "quote": str(chunks[0].get("text") or "")[:1600] if chunks else "",
                "chunk_ids": [str(row.get("chunk_id") or "") for row in chunks],
                "published": item.get("published") or item.get("published_at"),
                "updated": item.get("updated") or item.get("updated_at"),
                "authority": item.get("authority") or {},
                "evidence_record_id": str(
                    item.get("evidence_record_id") or ""
                ),
                "context_role": str(item.get("context_role") or ""),
                "record_metadata": dict(item.get("record_metadata") or {}),
                "source_object": dict(item.get("source_object") or {}),
                "task_point_ids": [
                    str(value)
                    for value in item.get("claim_ids") or []
                    if str(value).strip()
                ],
            }
        )
    return refs


_OBSERVED_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+(?:19|20)\d{2}\b|"
    r"(?:19|20)\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日",
    re.IGNORECASE,
)
_OBSERVED_VERSION_RE = re.compile(
    r"(?<![\d-])(?:v(?:ersion)?\s*)?\d+\.\d+(?:\.\d+)?"
    r"(?:[-+._][A-Za-z0-9.-]+)?\b",
    re.IGNORECASE,
)


def _observed_record_markers(text: Any) -> dict[str, list[str]]:
    """Expose literal marker order without selecting a current/true record."""

    value = str(text or "")
    return {
        "dates": list(dict.fromkeys(match.group(0) for match in _OBSERVED_DATE_RE.finditer(value)))[:12],
        "versions": list(dict.fromkeys(match.group(0) for match in _OBSERVED_VERSION_RE.finditer(value)))[:12],
    }


_TEMPORAL_IDENTITY_FIELD_RE = re.compile(
    r"(?:date|time|month|year|version|release|published|updated|tag|build|"
    r"patch[_ -]?level|版本|日期|时间|月份|年份|补丁|构建号)",
    re.IGNORECASE,
)


def _task_plan_requests_temporal_identity(task_plan: dict[str, Any] | None) -> bool:
    """Whether RWKV explicitly requested a dated or versioned record field.

    ``time_scope=current`` alone is intentionally insufficient. Current
    documentation and direct-page procedure tasks also receive that scope,
    while dates in their navigation bars are unrelated to the requested
    command or procedure. This controls metadata display only; it neither
    ranks facts nor decides an answer.
    """

    for point in task_points(task_plan):
        fields = " ".join(str(value) for value in point.get("fields") or [])
        if _TEMPORAL_IDENTITY_FIELD_RE.search(fields):
            return True
    return False


def _query_url_identities(query: str) -> set[str]:
    """Return stable identities for URLs explicitly supplied by the user."""

    identities: set[str] = set()
    for value in re.findall(r"https?://[^\s<>\]\[()]+", str(query or ""), re.IGNORECASE):
        identity = _source_identity({"url": value.rstrip(".,;:!?，。；：！？")})
        if identity:
            identities.add(identity)
    return identities


def _source_selection_metadata(
    item: dict[str, Any],
    *,
    query_url_identities: set[str],
    prefer_recent: bool,
    original_index: int,
) -> dict[str, Any]:
    """Describe deterministic packing priority without judging source truth."""

    identity = _source_identity(item)
    direct = bool(identity and identity in query_url_identities)
    origin = str(item.get("evidence_origin") or "").casefold()
    kind = str(item.get("evidence_kind") or "").casefold()
    content_type = str(item.get("content_type") or "").casefold()
    structured = bool(
        origin == "structured_api_record"
        or kind == "structured_record"
        or content_type in {"release", "weather", "paper"}
    )
    authority = item.get("authority") if isinstance(item.get("authority"), dict) else {}
    authority_rank = int(authority.get("rank") or 0)
    if str(item.get("source_kind") or "").casefold() == "official":
        authority_rank = max(authority_rank, 3)
    grounded = _has_grounded_locator(item)
    record_metadata = (
        item.get("record_metadata")
        if isinstance(item.get("record_metadata"), dict)
        else {}
    )
    alignment = (
        record_metadata.get("object_alignment")
        if isinstance(record_metadata.get("object_alignment"), dict)
        else item.get("object_alignment")
        if isinstance(item.get("object_alignment"), dict)
        else {}
    )
    alignment_candidates = [
        row
        for row in (
            record_metadata.get("object_alignments")
            or item.get("object_alignments")
            or []
        )
        if isinstance(row, dict)
    ]
    if alignment:
        alignment_candidates.insert(0, alignment)
    alignment_priority = {
        "exact": 4,
        "not_explicitly_scoped": 3,
        "source_identity_unavailable": 2,
        "unresolved": 1,
        "conflict": 0,
    }
    if alignment_candidates:
        alignment = max(
            alignment_candidates,
            key=lambda row: alignment_priority.get(
                str(row.get("relation") or "").casefold(),
                1,
            ),
        )
    alignment_relation = str(alignment.get("relation") or "").casefold()
    object_alignment_rank = alignment_priority.get(alignment_relation, 1)
    subject_alignment = (
        record_metadata.get("rwkv_subject_alignment")
        if isinstance(record_metadata.get("rwkv_subject_alignment"), dict)
        else {}
    )
    subject_alignment_relation = str(
        subject_alignment.get("relation") or ""
    ).casefold()
    subject_alignment_rank = {
        "exact": 2,
        "unresolved": 1,
        "conflict": 0,
    }.get(subject_alignment_relation, 1)
    source_date = ""
    source_date_origin = ""
    freshness = item.get("freshness") if isinstance(item.get("freshness"), dict) else {}
    for value in (
        freshness.get("source_date"),
        item.get("published"),
        item.get("published_at"),
        item.get("updated"),
        item.get("updated_at"),
        item.get("date"),
    ):
        source_date = extract_explicit_date(value)
        if source_date:
            source_date_origin = "source_metadata"
            break
    if not source_date and kind == "evidence_record":
        # EvidenceRecord content is already an exact grounded source quote.
        # Expose its first literal date for attention order so paired
        # version/date rows do not arrive at RWKV in arbitrary lexical order.
        # This metadata never asserts currentness or correctness.
        source_date = extract_explicit_date(_source_body(item))
        if source_date:
            source_date_origin = "grounded_record_quote"
    date_rank = int(source_date.replace("-", "")) if prefer_recent and source_date else 0
    return {
        "direct_user_url": direct,
        "structured_record": structured,
        "authority_rank": authority_rank,
        "grounded_locator": grounded,
        "object_alignment_relation": alignment_relation or None,
        "object_alignment_rank": object_alignment_rank,
        "rwkv_subject_alignment_relation": subject_alignment_relation or None,
        "rwkv_subject_alignment_rank": subject_alignment_rank,
        "source_date": source_date or None,
        "source_date_origin": source_date_origin or None,
        "date_priority_active": prefer_recent,
        "original_index": original_index,
        "sort_key": (
            int(direct),
            object_alignment_rank,
            subject_alignment_rank,
            int(structured),
            authority_rank,
            int(grounded),
            date_rank,
            -original_index,
        ),
    }


def _order_sources_for_context(
    items: Iterable[dict[str, Any]],
    query: str,
    task_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Order useful source records for the bounded final context.

    This is attention routing, not an answer gate: no source is declared true,
    false, supported, or unsupported.  Existing structured provenance,
    authority metadata, exact locators, and explicit dates only decide which
    records reach RWKV first when the context has a source limit.
    """

    query_urls = _query_url_identities(query)
    prefer_recent = bool(
        re.search(
            r"(?:\bcurrent\b|\blatest\b|\bnewest\b|\brecent\b|\btoday\b|"
            r"当前|最新|目前|现行|截至)",
            str(query or ""),
            flags=re.IGNORECASE,
        )
    )
    ranked: list[dict[str, Any]] = []
    for index, raw in enumerate(items):
        item = dict(raw)
        metadata = _source_selection_metadata(
            item,
            query_url_identities=query_urls,
            prefer_recent=prefer_recent,
            original_index=index,
        )
        item["context_selection"] = {
            key: value for key, value in metadata.items() if key != "sort_key"
        }
        ranked.append(item)
    ranked.sort(
        key=lambda item: _source_selection_metadata(
            item,
            query_url_identities=query_urls,
            prefer_recent=prefer_recent,
            original_index=int(
                (item.get("context_selection") or {}).get("original_index") or 0
            ),
        )["sort_key"],
        reverse=True,
    )
    point_ids = [
        str(point.get("id") or "").strip()
        for point in task_points(task_plan)
        if str(point.get("id") or "").strip()
    ]
    if not point_ids:
        return ranked

    # Attention projection only: reserve up to two already-ranked records for
    # every RWKV factual point before filling the remaining slots.  No source is
    # declared true or sufficient and no claim binding is invented here.
    ordered: list[dict[str, Any]] = []
    selected: set[int] = set()
    per_point_count = {point_id: 0 for point_id in point_ids}

    def add(item: dict[str, Any], role: str = "") -> None:
        marker = id(item)
        claim_ids = [
            str(value).strip()
            for value in item.get("claim_ids") or []
            if str(value).strip() in per_point_count
        ]
        if marker not in selected:
            selected.add(marker)
            ordered.append(item)
            for claim_id in claim_ids:
                per_point_count[claim_id] += 1
        if role:
            metadata = item.setdefault("context_selection", {})
            roles = list(metadata.get("task_point_roles") or [])
            if role not in roles:
                roles.append(role)
            metadata["task_point_roles"] = roles

    for item in ranked:
        if bool((item.get("context_selection") or {}).get("direct_user_url")):
            add(item, "direct_user_url")

    for ordinal, label in ((1, "primary"), (2, "secondary")):
        for point_id in point_ids:
            if per_point_count[point_id] >= ordinal:
                continue
            candidate = next(
                (
                    item
                    for item in ranked
                    if id(item) not in selected
                    and point_id
                    in {
                        str(value).strip()
                        for value in item.get("claim_ids") or []
                    }
                ),
                None,
            )
            if candidate is not None:
                add(candidate, f"{point_id}:{label}")

    for item in ranked:
        add(item)
    return ordered


def build_evidence_context(
    data: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    *,
    query: str = "",
    max_sources: int | None = None,
    token_budget: int | None = None,
    max_chunks_per_source: int | None = None,
    source_locator_char_limit: int = 2400,
) -> dict[str, Any]:
    """Pack fetched chunks into a bounded, source-labelled RWKV context.

    Packing is a resource operation only.  It does not decide whether a source
    supports a claim and it does not suppress RWKV's ability to answer.
    """

    configured_source_limit = int(
        ((constraints or {}).get("strategy_config") or {}).get("context_source_count")
        or (constraints or {}).get("context_source_count")
        or DATA_PIPELINE.get("context_source_count", 8)
        or 8
    )
    source_limit = (
        configured_source_limit if max_sources is None else int(max_sources)
    )
    source_limit = max(1, min(source_limit, 24))
    ledger_value = data.get("claim_ledger") or (constraints or {}).get("claim_ledger")
    task_plan = (constraints or {}).get("task_plan")
    if not task_points(task_plan) and isinstance(ledger_value, dict):
        task_plan = {
            "goal": query,
            "atomic_points": [
                {
                    "id": row.get("claim_id") or row.get("point_id"),
                    "question": row.get("question") or row.get("task") or row.get("objective"),
                    "subject": row.get("subject") or "",
                    "relation": row.get("relation") or "",
                    "fields": row.get("fields") or row.get("evidence_needed") or [],
                    "time_scope": row.get("time_scope") or "unspecified",
                    "set_semantics": row.get("set_semantics") or "single",
                    "premise_requires_verification": bool(
                        row.get("premise_requires_verification")
                    ),
                }
                for row in ledger_value.get("claims") or []
                if isinstance(row, dict)
            ],
        }
    grounded_span_limit = max(
        1,
        min(
            8,
            int(DATA_PIPELINE.get("final_context_max_grounded_spans_per_source", 3) or 3),
        ),
    )
    # RWKV-selected grounded candidate records are the primary semantic
    # context. Raw fetched pages remain available only through a small,
    # explicitly unbound fallback lane so an extraction miss does not become
    # an empty answer.
    fetched_items = [
        item for item in data.get("results") or [] if isinstance(item, dict)
    ]
    record_items = _claim_grounded_source_items(
        ledger_value,
        grounded_span_limit=grounded_span_limit,
    )
    merged_records, duplicate_record_count = _merge_source_records(record_items)
    ordered_records = _order_sources_for_context(
        [item for item in merged_records if _source_body(item)],
        query,
        task_plan,
    )
    record_urls = {
        re.sub(
            r"^https?://(?:www\.)?",
            "",
            str(item.get("url") or "").strip().casefold(),
        ).split("#", 1)[0].split("?", 1)[0].rstrip("/")
        for item in ordered_records
        if str(item.get("url") or "").strip()
    }
    configured_fallback_limit = int(
        DATA_PIPELINE.get("final_context_unbound_fallback_sources", 2) or 2
    )
    fallback_limit = (
        configured_fallback_limit
        if ordered_records
        else max(configured_fallback_limit, min(4, source_limit))
    )
    fallback_candidates = [
        {**dict(item), "claim_ids": []}
        for item in fetched_items
    ]
    fallback_items = _unbound_fallback_items(
        _order_sources_for_context(fallback_candidates, query, task_plan),
        excluded_urls=record_urls,
        # Scan beyond the visible limit so duplicate URLs can be merged and
        # still backfilled with distinct fallback sources.
        limit=max(fallback_limit, fallback_limit * 3),
    )
    merged_fallback, duplicate_fallback_count = _merge_source_records(
        fallback_items
    )
    merged_fallback = merged_fallback[:fallback_limit]
    unique_items = [*ordered_records, *merged_fallback]
    duplicate_source_count = duplicate_record_count + duplicate_fallback_count

    selected = unique_items[:source_limit]

    default_budget = evidence_tokens(get_llm_context_length())
    budget = (
        default_budget
        if token_budget is None
        else max(128, min(int(token_budget), default_budget))
    )
    factual_plan = compact_task_plan(task_plan)
    factual_points = [
        {
            **point,
            "object_contract": task_record_contract(
                task_plan,
                str(point.get("id") or ""),
            ),
        }
        for point in factual_plan.get("atomic_points") or []
        if isinstance(point, dict)
    ]
    show_temporal_routing = _task_plan_requests_temporal_identity(task_plan)
    calculations = [
        item for item in data.get("calculation_results") or [] if isinstance(item, dict)
    ]
    freshness_policy = (constraints or {}).get("freshness_policy")
    static_preview: list[str] = []
    if factual_points:
        static_preview.append(json.dumps(factual_points, ensure_ascii=False, indent=2))
    if calculations:
        static_preview.append(json.dumps(calculations, ensure_ascii=False, indent=2))
    if isinstance(freshness_policy, dict) and freshness_policy:
        static_preview.append(json.dumps(freshness_policy, ensure_ascii=False, indent=2))
    for item in selected:
        locator_preview = (
            ""
            if _has_grounded_locator(item)
            else str(item.get("model_locator_facts") or "").strip()
        )
        static_preview.append(
            "\n".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("url") or ""),
                    json.dumps(item.get("freshness") or {}, ensure_ascii=False),
                    locator_preview[: max(0, int(source_locator_char_limit or 0))],
                ]
            )
        )
    static_overhead_tokens = get_token_count("\n\n".join(static_preview)) + 160
    chunk_budget = max(128, budget - static_overhead_tokens)

    all_per_source_chunks = [
        _source_chunks(item, grounded_span_limit=grounded_span_limit)
        for item in selected
    ]
    if max_chunks_per_source is None:
        configured_chunk_limit = max(
            1,
            min(
                12,
                int(DATA_PIPELINE.get("final_context_max_chunks_per_source", 4) or 4),
            ),
        )
        per_source_chunks = [
            chunks[:configured_chunk_limit] for chunks in all_per_source_chunks
        ]
        effective_chunk_limit: int | None = configured_chunk_limit
    else:
        chunk_limit = max(1, int(max_chunks_per_source))
        per_source_chunks = [chunks[:chunk_limit] for chunks in all_per_source_chunks]
        effective_chunk_limit = chunk_limit
    packed: list[list[dict[str, Any]]] = [[] for _ in selected]
    used_tokens = 0
    cursor = 0
    while selected and used_tokens < chunk_budget:
        made_progress = False
        for source_index, chunks in enumerate(per_source_chunks):
            if cursor >= len(chunks):
                continue
            chunk = chunks[cursor]
            chunk_tokens = get_token_count(chunk["text"])
            if used_tokens + chunk_tokens > chunk_budget:
                continue
            packed[source_index].append(chunk)
            used_tokens += chunk_tokens
            made_progress = True
        if not made_progress:
            break
        cursor += 1

    blocks: list[str] = []
    projected_sources: list[dict[str, Any]] = []
    for item, chunks in zip(selected, packed):
        if not chunks:
            continue
        ref_index = len(projected_sources) + 1
        title = str(item.get("title") or item.get("url") or f"Source {ref_index}")
        url = str(item.get("url") or "")
        context_role = str(item.get("context_role") or "")
        if context_role in {"exact_evidence_record", "candidate_evidence_record"}:
            # Old traces may still carry exact_evidence_record.  At the live
            # context boundary all chunk-local observations are candidates;
            # only RWKV may compare the full record set and decide currentness.
            record_metadata = dict(item.get("record_metadata") or {})
            unresolved_record_identity = not str(
                record_metadata.get("record_key") or ""
            ).strip()
            role_label = (
                "RWKV-CANDIDATE SOURCE OBJECT SPANS (record identity unresolved)"
                if unresolved_record_identity
                else "RWKV-CANDIDATE SOURCE RECORD"
            )
            lines = [
                f"[S{ref_index}] {role_label}: {title}",
                f"URL: {url}",
            ]
            visible_record_metadata = {
                key: value
                for key, value in record_metadata.items()
                if key != "retrieval_request"
            }
            lines.append(
                "Record grouping metadata (RWKV routing labels, not extra facts): "
                + json.dumps(
                    visible_record_metadata,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            source_object = record_metadata.get("source_object")
            if isinstance(source_object, dict) and source_object:
                lines.append(
                    "Observable source object identity (transport metadata): "
                    + json.dumps(
                        source_object,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            alignment = record_metadata.get("object_alignment")
            if isinstance(alignment, dict) and alignment:
                lines.append(
                    "Literal object-identifier alignment (transport comparison only): "
                    + json.dumps(
                        alignment,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            subject_alignment = record_metadata.get("rwkv_subject_alignment")
            if isinstance(subject_alignment, dict) and subject_alignment:
                lines.append(
                    "RWKV-authored subject-label alignment: "
                    + json.dumps(
                        subject_alignment,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            if unresolved_record_identity:
                lines.append(
                    "Record identity note: these grounded spans share one source object and task record, "
                    "but may describe different rows, dates or versions. Compare each <chunk-id> literally."
                )
        elif context_role == "unbound_source_fallback":
            lines = [
                f"[S{ref_index}] UNBOUND SOURCE EXCERPT: {title}",
                f"URL: {url}",
                (
                    "Binding note: no exact span from this source was bound to a task record; "
                    "RWKV must inspect the excerpt directly and must not combine it with another "
                    "record merely because the topic is similar."
                ),
            ]
        else:
            lines = [
                f"[S{ref_index}] Source label (routing metadata only): {title}",
                f"URL: {url}",
            ]
        for label, value in (
            ("Published", item.get("published") or item.get("published_at")),
            ("Updated", item.get("updated") or item.get("updated_at")),
            ("Source date", item.get("date")),
            ("Provider", item.get("provider") or item.get("source")),
        ):
            if value not in (None, "", [], {}):
                lines.append(f"{label}: {value}")
        if isinstance(item.get("freshness"), dict) and item.get("freshness"):
            lines.append(
                "Freshness metadata: "
                + json.dumps(item["freshness"], ensure_ascii=False, separators=(",", ":"))
            )
        claim_ids = [str(value) for value in item.get("claim_ids") or [] if str(value).strip()]
        if claim_ids:
            lines.append("RWKV task-point bindings: " + ", ".join(claim_ids))
        for chunk in chunks:
            temporal_role = str(chunk.get("record_temporal_role") or "").strip()
            record_date = str(chunk.get("record_date") or "").strip()
            if show_temporal_routing and (temporal_role or record_date):
                lines.append(
                    "Same-page temporal routing metadata "
                    "(literal ordering only; RWKV must judge identity, stability and currentness): "
                    + json.dumps(
                        {
                            "record_date": record_date or None,
                            "date_precision": chunk.get("record_date_precision") or None,
                            "role": temporal_role or None,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            observed = _observed_record_markers(chunk.get("text"))
            if observed["dates"] or observed["versions"]:
                lines.append(
                    "Observed literal record markers in occurrence order "
                    "(routing metadata only; no truth/currentness judgment): "
                    + json.dumps(observed, ensure_ascii=False, separators=(",", ":"))
                )
            lines.append(f"<{chunk['chunk_id']}>\n{chunk['text']}")
        source_locators = (
            ""
            if _has_grounded_locator(item)
            else str(item.get("model_locator_facts") or "").strip()
        )
        locator_limit = max(0, int(source_locator_char_limit or 0))
        if source_locators and locator_limit:
            lines.append(
                f"VERBATIM SOURCE LOCATORS:\n{source_locators[:locator_limit]}"
            )
        blocks.append("\n".join(lines))
        projected_sources.append(
            {
                **item,
                "ref_id": f"S{ref_index}",
                "packed_chunks": chunks,
                "chunk_count": len(chunks),
                "evidence_text": "\n\n".join(chunk["text"] for chunk in chunks),
            }
        )

    sections: list[str] = []
    if factual_points:
        sections.append(
            "RWKV FACTUAL PLAN (user-requested records; not a completion gate):\n"
            + json.dumps(factual_points, ensure_ascii=False, indent=2)
        )
    if calculations:
        sections.append(
            "TOOL RESULTS:\n" + json.dumps(calculations, ensure_ascii=False, indent=2)
        )
    if isinstance(freshness_policy, dict) and freshness_policy:
        sections.append(
            "QUESTION TIME/FRESHNESS POLICY (metadata for RWKV, not an answer gate):\n"
            + json.dumps(freshness_policy, ensure_ascii=False, indent=2)
        )
    sections.append(
        "RETRIEVED SOURCES:\n" + ("\n\n".join(blocks) if blocks else "No source text was retrieved.")
    )
    text = "\n\n".join(sections)
    refs = _citation_refs(projected_sources)
    return {
        "text": text,
        "evidence_text": text,
        "selected_evidence": projected_sources,
        "citation_refs": refs,
        "calculation_results": calculations,
        "usable_evidence_count": len(projected_sources),
        "duplicate_source_count": duplicate_source_count,
        "chunk_count": sum(len(chunks) for chunks in packed),
        "context_chars": len(text),
        "context_tokens": get_token_count(text),
        "context_truncated": any(
            len(chunks) < len(all_chunks)
            for chunks, all_chunks in zip(packed, all_per_source_chunks)
        ),
        "validation": {},
        "context_stats": {
            "source_count": len(projected_sources),
            "chunk_count": sum(len(chunks) for chunks in packed),
            "duplicate_source_count": duplicate_source_count,
            "bound_evidence_record_count": sum(
                str(item.get("context_role") or "")
                == "exact_evidence_record"
                for item in projected_sources
            ),
            "candidate_evidence_record_count": sum(
                str(item.get("context_role") or "")
                == "candidate_evidence_record"
                for item in projected_sources
            ),
            "unbound_fallback_source_count": sum(
                str(item.get("context_role") or "")
                == "unbound_source_fallback"
                for item in projected_sources
            ),
            "unbound_fallback_source_limit": fallback_limit,
            "calculation_count": len(calculations),
            "factual_point_count": len(factual_points),
            "factual_points_with_selected_sources": sum(
                any(
                    str(point.get("id") or "")
                    in {
                        str(value).strip()
                        for value in source.get("claim_ids") or []
                    }
                    for source in projected_sources
                )
                for point in factual_points
                if str(point.get("id") or "").strip()
            ),
            "context_tokens": get_token_count(text),
            "evidence_budget_tokens": budget,
            "source_chunk_budget_tokens": chunk_budget,
            "static_overhead_tokens": static_overhead_tokens,
            "configured_source_limit": source_limit,
            "max_chunks_per_source": effective_chunk_limit,
            "max_grounded_spans_per_source": grounded_span_limit,
            "cross_validation_review_included": False,
            "source_locator_char_limit": max(0, int(source_locator_char_limit or 0)),
            "exact_or_containment_dedup_active": True,
        },
    }


def _writer_prompt(
    query: str,
    context: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> str:
    runtime_environment = str((constraints or {}).get("runtime_environment") or "").strip()
    runtime_section = (
        f"CURRENT RUNTIME:\n{runtime_environment}\n\n" if runtime_environment else ""
    )
    return (
        "You are the final RWKV answer writer. Answer the user's question yourself using the retrieved "
        "material and tool results below. The material can be incomplete or conflicting; judge it directly. "
        "If some information is missing, answer the supported parts and clearly state what remains uncertain. "
        "For commands, dates, versions, identifiers, names, statuses, quoted output, and examples, use only "
        "values explicitly present in the material; do not substitute a plausible value or invent an example. "
        "Source labels, page titles, provider names and URLs identify records but are not factual evidence; bind "
        "the answer to the original text inside each <chunk-id> block. "
        "Treat each RWKV-CANDIDATE SOURCE RECORD as an independent subject/version/time record. "
        "A RWKV-CANDIDATE SOURCE OBJECT SPANS block only groups quotes from one page/object; when its record identity is unresolved, its chunks may still describe different versions, dates or rows. "
        "No candidate has been declared correct, current, latest, or authoritative by the controller. Compare its literal source-object identity, record identity and date yourself. Do not take a field "
        "from one record and attach it to another record unless the source text explicitly establishes that identity. "
        "UNBOUND SOURCE EXCERPT blocks are fallback material, not pre-established support. "
        "For a current/latest request, do not present a future-dated or explicitly historical record as current; "
        "state the time conflict or uncertainty instead. For a repository-specific request, bind claims to the "
        "exact owner/repository rather than another project on the same host. When a literal object-identifier "
        "alignment is conflict, that source is about a different explicitly named object and cannot supply that "
        "object's fields. This metadata compares identifiers only; it does not decide which version or fact is "
        "correct. Satisfy every requested field when "
        "the material supports it, and explicitly identify any requested field that remains unsupported. "
        "Answer only the fields the user requested. Do not append adjacent limitations, examples, commands, or "
        "background merely because they occur near the supporting span. Use one clean answer path and keep it concise. "
        "Do not restart, repeat, enumerate duplicate support, "
        "or re-check fields already answered. If the answer starts to loop, stop immediately and return the "
        "best answer already written. Do not describe controller rules or the research process. Use [S1], [S2], ... "
        "when citing a retrieved source.\n\n"
        + runtime_section
        + f"USER QUESTION:\n{query}\n\n"
        + f"RESEARCH MATERIAL:\n{context['text']}\n\n"
        + "Write the final answer now."
    )


def _call_local_or_chat(
    llm,
    *,
    prompt: str,
    user_prompt: str,
    max_tokens: int,
    stage: str,
    policy_reason: str,
):
    provider = str(getattr(llm, "provider", "") or "")
    sampling_temperature = get_model_stage_temperature(stage)
    with model_sampling_parameters(
        sampling_temperature,
        stage=stage,
        policy_reason=policy_reason,
    ):
        if hasattr(llm, "text_completion") and (not provider or is_local_provider(provider)):
            response = llm.text_completion(prompt, max_tokens=max_tokens)
        else:
            response = llm.chat_completion(
                [{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
            )
    return response, sampling_temperature, get_model_stage_sampling(stage)


def synthesize_retrieval_answer(
    query: str,
    data: dict[str, Any],
    llm=None,
    constraints: dict[str, Any] | None = None,
    execution_context: str = "",
    termination_reason: str = "model_requested_finish",
) -> dict[str, Any]:
    """Return the single public answer written by RWKV without post-processing."""

    del execution_context
    context = build_evidence_context(data, constraints=constraints, query=query)
    user_prompt = _writer_prompt(query, context, constraints)
    prompt = build_final_continuation_prompt(user_prompt)
    if llm is None:
        raise ConnectionError("RWKV final writer is unavailable")

    requested_max_tokens = max(
        1,
        int(DATA_PIPELINE.get("final_answer_max_tokens", 5120) or 5120),
    )
    max_tokens = bounded_completion_budget(
        prompt,
        context_limit=get_llm_context_length(),
        requested_max=requested_max_tokens,
        safety_margin=256,
    )
    context["context_stats"].update(
        {
            "final_prompt_tokens": get_token_count(prompt),
            "requested_output_tokens": requested_max_tokens,
            "effective_output_tokens": max_tokens,
            "completion_safety_margin_tokens": 256,
        }
    )
    response, sampling_temperature, sampling_profile = _call_local_or_chat(
        llm,
        prompt=prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        stage="final_writer",
        policy_reason="grounded_public_answer_generation",
    )
    sampling_profile["temperature"] = sampling_temperature
    context["context_stats"].update(
        {
            "sampling_temperature": sampling_temperature,
            "sampling_seed": None,
            "sampling_parameters": sampling_profile,
        }
    )
    raw_model_output = _clean_answer(getattr(response, "content", ""))
    answer = consume_final_prefill_boundary(raw_model_output)
    if not answer.strip():
        raise ConnectionError("RWKV final writer returned no output")
    generation_attempts = [
        {
            "stage": "rwkv_final",
            "output": answer,
            "termination_reason": termination_reason,
            "sampling_temperature": sampling_temperature,
            "sampling_seed": None,
            "sampling_parameters": sampling_profile,
        }
    ]

    return {
        "content": answer,
        "mode": "rwkv_final",
        "evidence_count": len(context["selected_evidence"]),
        "citation_refs": context["citation_refs"],
        "prompt": prompt,
        "model_output": answer,
        "raw_model_output": raw_model_output,
        "generation_attempts": generation_attempts,
        "context_text": context["text"],
        "selected_evidence": context["selected_evidence"],
        "context_stats": context["context_stats"],
        "answer_alignment": {},
        "answer_quality": {},
    }


__all__ = ["build_evidence_context", "synthesize_retrieval_answer"]
