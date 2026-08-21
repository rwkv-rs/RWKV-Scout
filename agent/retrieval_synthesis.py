"""Minimal evidence packing and the single RWKV final-writer call.

This module deliberately has no answer gate, repair pass, translation pass,
deterministic answer fallback, refusal generator, or quality status. Retrieval
code may organise source material, but only RWKV writes the public answer.
"""

from __future__ import annotations

import json
import hashlib
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
    FINAL_ANSWER_STOP_SUFFIXES,
    build_final_continuation_prompt,
    consume_final_prefill_boundary,
)
from utils.web_retrieval import retrieval_url_identity
from agent.task_plan_contract import (
    compact_task_plan,
    record_fields,
    record_id,
    task_records,
)
from agent.retrieval_object_contract import merge_mapping_rows


def _clean_answer(value: Any) -> str:
    """Return the exact text emitted by RWKV.

    The historical function name is retained only for import compatibility.
    No whitespace trimming, protocol stripping, deduplication, citation
    rewriting, or other answer transformation is permitted here.
    """

    return "" if value is None else str(value)


def _bounded_unique_text_values(
    value: Any,
    *,
    limit: int = 16,
    char_limit: int = 160,
) -> list[str]:
    rows = value if isinstance(value, (list, tuple, set)) else [value]
    return list(
        dict.fromkeys(
            str(row).strip()[: max(1, int(char_limit))]
            for row in rows
            if str(row or "").strip()
        )
    )[: max(1, int(limit))]


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
        record_span_id = str(record_metadata.get("record_span_id") or "").strip()
        if record_span_id:
            # An empty literal record key means identity is unresolved. Keep
            # every exact source span atomic; URL/task-level merging would
            # silently turn competing rows into fields of one record.
            return f"record-span:{record_span_id}"
    evidence_record_id = str(item.get("evidence_record_id") or "").strip()
    if evidence_record_id:
        # Evidence Records without an observable record key remain separate;
        # URL-level merging could conflate versions, advisories or table rows.
        return f"record:{evidence_record_id}"
    url = _canonical_url_identity(item.get("url"))
    if url:
        return f"url:{url}"
    digest = str(item.get("content_sha256") or "").strip().casefold()
    if digest:
        return f"sha256:{digest}"
    title = " ".join(str(item.get("title") or "").split()).casefold()
    return f"title:{title}" if title else ""


def _canonical_url_identity(value: Any) -> str:
    """Use the retrieval transport's conservative, query-preserving identity."""

    return retrieval_url_identity(str(value or ""))


def _retrieval_ranking_by_url(
    items: Iterable[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Index final retrieval ranks so later modules consume the same order.

    This is a state-transfer join, not a new ranker.  It copies the ranking
    already produced by ``merge_retrieval_results`` onto RWKV-grounded records
    that refer to the same fetched URL.
    """

    ranking: dict[str, dict[str, Any]] = {}
    for fallback_rank, raw in enumerate(items or [], start=1):
        if not isinstance(raw, dict):
            continue
        identity = _canonical_url_identity(raw.get("url"))
        if not identity:
            continue
        try:
            retrieval_rank = max(1, int(raw.get("retrieval_rank") or fallback_rank))
        except (TypeError, ValueError):
            retrieval_rank = fallback_rank
        try:
            rerank_score = float(raw.get("rerank_score") or 0.0)
        except (TypeError, ValueError):
            rerank_score = 0.0
        candidate = {
            "retrieval_rank": retrieval_rank,
            "rerank_score": rerank_score,
            "ranking_method": str(raw.get("ranking_method") or ""),
        }
        previous = ranking.get(identity)
        if previous is None or (
            retrieval_rank,
            -rerank_score,
        ) < (
            int(previous.get("retrieval_rank") or 10**9),
            -float(previous.get("rerank_score") or 0.0),
        ):
            ranking[identity] = candidate
    return ranking


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
    Locator task_record ids are merged into that span. This is text deduplication
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

    item_record_metadata = (
        item.get("record_metadata")
        if isinstance(item.get("record_metadata"), dict)
        else {}
    )

    def merged_field_ids(*values: Any) -> list[str]:
        return list(
            dict.fromkeys(
                str(field).strip()[:160]
                for value in values
                for field in (
                    value
                    if isinstance(value, (list, tuple, set))
                    else [value]
                )
                if str(field or "").strip()
            )
        )[:16]

    def routing_metadata(row: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            key: row[key]
            for key in routing_metadata_fields
            if row.get(key) not in (None, "", [], {})
        }
        selected_field_ids = merged_field_ids(
            item_record_metadata.get("field_ids"),
            item.get("field_ids"),
            row.get("field_ids"),
        )
        if selected_field_ids:
            metadata["field_ids"] = selected_field_ids
        return metadata

    grounded_rows = [
        {
            "chunk_id": "locator-" + str(row.get("chunk_id") or index + 1),
            # Zero is a valid first-chunk index. Falling back with ``or``
            # changed duplicate views of chunk 0 into indexes 0, 1, ... and
            # defeated exact-span deduplication after task-record merging.
            "index": int(
                row.get("chunk_index")
                if row.get("chunk_index") is not None
                else index
            ),
            "text": str(row.get("quote") or "").strip(),
            "task_record_ids": list(row.get("task_record_ids") or []),
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

    def merged_task_record_ids(*values: Any) -> list[str]:
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
        chunk_index = int(
            row.get("index") if row.get("index") is not None else index
        )
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
            contained_by["task_record_ids"] = merged_task_record_ids(
                contained_by.get("task_record_ids"), row.get("task_record_ids")
            )
            contained_by["field_ids"] = merged_field_ids(
                contained_by.get("field_ids"),
                routing_metadata(row).get("field_ids"),
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
                strongest["task_record_ids"] = merged_task_record_ids(
                    strongest.get("task_record_ids"), row.get("task_record_ids")
                )
                strongest["field_ids"] = merged_field_ids(
                    strongest.get("field_ids"),
                    routing_metadata(row).get("field_ids"),
                )
                if str(chunk_id).startswith("locator-"):
                    locator_ids = list(strongest.get("contained_locator_ids") or [])
                    if chunk_id not in locator_ids:
                        locator_ids.append(chunk_id)
                    strongest["contained_locator_ids"] = locator_ids
                continue

        inherited_task_record_ids: list[str] = []
        inherited_field_ids: list[str] = []
        inherited_locator_ids: list[str] = []
        for existing_index in reversed(contained_indexes):
            existing = chunks.pop(existing_index)
            inherited_task_record_ids = merged_task_record_ids(
                inherited_task_record_ids, existing.get("task_record_ids")
            )
            inherited_field_ids = merged_field_ids(
                inherited_field_ids,
                existing.get("field_ids"),
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
                    "task_record_ids": merged_task_record_ids(
                        row.get("task_record_ids"), inherited_task_record_ids
                    ),
                    "field_ids": merged_field_ids(
                        routing_metadata(row).get("field_ids"),
                        inherited_field_ids,
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
    fallback_field_ids = merged_field_ids(
        item_record_metadata.get("field_ids"),
        item.get("field_ids"),
    )
    return [
        {
            "chunk_id": f"chunk-{index + 1}",
            "index": index,
            "text": text,
            **({"field_ids": fallback_field_ids} if fallback_field_ids else {}),
        }
        for index, text in enumerate(
            semantic_chunk_text(body, max_tokens=max(128, int(chunk_tokens)), overlap_ratio=0.08)
        )
        if text.strip()
    ]


def _evidence_ledger_projection(value: Any) -> list[dict[str, Any]]:
    """Expose Evidence Ledger progress as model context, never as a gate."""

    if not isinstance(value, dict):
        return []
    projected: list[dict[str, Any]] = []
    for row in value.get("task_records") or []:
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
                "task_record_id": str(row.get("task_record_id") or ""),
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
                "task_record_id": "UNASSIGNED",
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


def _evidence_record_source_items(
    value: Any,
    *,
    grounded_span_limit: int = 3,
    ranked_results: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Project model-grounded ledger spans into ordinary final source blocks.

    The Evidence Ledger previously embedded every span, locator coordinate and
    source metadata object inside the checklist while final source selection
    could omit the very page carrying the span.  That made the checklist much
    larger than the actual evidence section.  This projection keeps each
    grounded quote bound to its URL and task record, then lets the normal source
    packer assign [S#] references and budgets.
    """

    if not isinstance(value, dict):
        return []
    ranking_by_url = _retrieval_ranking_by_url(ranked_results)
    projected_by_task_record: list[list[dict[str, Any]]] = []
    for task_record in value.get("task_records") or []:
        if not isinstance(task_record, dict):
            continue
        task_record_id = str(task_record.get("task_record_id") or "").strip()
        task_record_items: list[dict[str, Any]] = []
        evidence_records = [
            row
            for row in task_record.get("evidence_records") or []
            if isinstance(row, dict)
            and str(row.get("quote") or "").strip()
        ]
        if evidence_records:
            # The Evidence Ledger snapshot already bounds each Task Record to 32
            # records.  Project all of them here, then deduplicate and order
            # them using the final retrieval rank.  Truncating this append-order
            # list first used to discard later, better-ranked official records.
            for record in evidence_records:
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
                    "task_record_ids": [task_record_id] if task_record_id else [],
                    "content": quote,
                    "record_metadata": {
                        "task_record_id": task_record_id,
                        "task_record_ids": [task_record_id] if task_record_id else [],
                        "task_record_relation": str(task_record.get("relation") or ""),
                        "task_record_time_scope": str(
                            task_record.get("time_scope") or "unspecified"
                        ),
                        "subject_key": str(record.get("subject_key") or ""),
                        "record_key": str(record.get("record_key") or ""),
                        "record_span_id": str(record.get("record_span_id") or ""),
                        "parent_candidate_id": str(
                            record.get("parent_candidate_id") or ""
                        ),
                        "assembly_basis": str(record.get("assembly_basis") or ""),
                        "field_ids": _bounded_unique_text_values(
                            record.get("field_ids")
                        ),
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
                            "task_record_ids": [task_record_id] if task_record_id else [],
                            "field_ids": _bounded_unique_text_values(
                                record.get("field_ids")
                            ),
                            "subject_key": str(record.get("subject_key") or ""),
                            "record_key": str(record.get("record_key") or ""),
                            "record_span_id": str(
                                record.get("record_span_id") or ""
                            ),
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
                    "evidence_origin",
                    "source_evidence_kind",
                    "content_type",
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
                ranking = ranking_by_url.get(
                    _canonical_url_identity(record.get("url"))
                )
                if ranking:
                    item.update(ranking)
                task_record_items.append(item)
            projected_by_task_record.append(task_record_items)
            continue
        for source in task_record.get("sources") or task_record.get("evidence") or []:
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
                "task_record_ids": [task_record_id] if task_record_id else [],
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
                        "task_record_ids": [task_record_id] if task_record_id else [],
                        "field_ids": _bounded_unique_text_values(
                            row.get("field_ids")
                        ),
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
                "evidence_origin",
                "evidence_kind",
                "content_type",
                "source_kind",
                "authority",
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
            task_record_items.append(item)
        if task_record_items:
            projected_by_task_record.append(task_record_items)
    projected: list[dict[str, Any]] = []
    cursor = 0
    while any(cursor < len(items) for items in projected_by_task_record):
        for items in projected_by_task_record:
            if cursor < len(items):
                projected.append(items[cursor])
        cursor += 1
    return projected


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
                "task_record_ids",
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
                    if metadata_key in {"field_ids", "task_record_ids"}:
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
                "task_record_ids": [
                    str(value)
                    for value in item.get("task_record_ids") or []
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

    for point in task_records(task_plan):
        fields = " ".join(record_fields(point))
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
        or str(item.get("source_evidence_kind") or "").casefold()
        == "structured_record"
        or content_type in {"release", "weather", "paper"}
    )
    authority = item.get("authority") if isinstance(item.get("authority"), dict) else {}
    authority_rank = int(authority.get("rank") or 0)
    if str(item.get("source_kind") or "").casefold() == "official":
        authority_rank = max(authority_rank, 3)
    # A generic institutional-host heuristic is useful trace metadata, but it
    # must not outrank the retrieval pipeline when no explicit source/domain
    # requirement was satisfied.  Only an observable required-domain match
    # activates authority as a packing priority.
    authority_priority_active = bool(
        str(item.get("source_kind") or "").casefold() == "official"
        or (authority.get("required") and authority.get("satisfied"))
    )
    effective_authority_rank = authority_rank if authority_priority_active else 0
    try:
        retrieval_rank = max(1, int(item.get("retrieval_rank")))
    except (TypeError, ValueError):
        retrieval_rank = 0
    try:
        retrieval_score = float(item.get("rerank_score") or 0.0)
    except (TypeError, ValueError):
        retrieval_score = 0.0
    retrieval_rank_priority = -retrieval_rank if retrieval_rank else -(10**9)
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
    # Do not derive source freshness from the first date in a grounded quote.
    # A release page may mention publication, maintenance and EOL dates in one
    # span; choosing one of those dates would be a factual interpretation, not
    # neutral packing metadata. Only explicit source-level metadata participates
    # in date ordering. Literal dates remain visible inside the source span for
    # RWKV to compare.
    date_rank = int(source_date.replace("-", "")) if prefer_recent and source_date else 0
    return {
        "direct_user_url": direct,
        "structured_record": structured,
        "authority_rank": authority_rank,
        "authority_priority_active": authority_priority_active,
        "effective_authority_rank": effective_authority_rank,
        "retrieval_rank": retrieval_rank or None,
        "retrieval_rerank_score": retrieval_score,
        "retrieval_ranking_method": str(item.get("ranking_method") or "") or None,
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
            effective_authority_rank,
            retrieval_rank_priority,
            retrieval_score,
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
    task_record_ids = [record_id(point) for point in task_records(task_plan)]
    if not task_record_ids:
        return ranked

    # Attention projection only: reserve up to two already-ranked records for
    # every RWKV factual point before filling the remaining slots.  No source is
    # declared true or sufficient and no task_record binding is invented here.
    ordered: list[dict[str, Any]] = []
    selected: set[int] = set()
    per_task_record_count = {task_record_id: 0 for task_record_id in task_record_ids}

    def add(item: dict[str, Any], role: str = "") -> None:
        marker = id(item)
        task_record_ids = [
            str(value).strip()
            for value in item.get("task_record_ids") or []
            if str(value).strip() in per_task_record_count
        ]
        if marker not in selected:
            selected.add(marker)
            ordered.append(item)
            for task_record_id in task_record_ids:
                per_task_record_count[task_record_id] += 1
        if role:
            metadata = item.setdefault("context_selection", {})
            roles = list(metadata.get("task_record_roles") or [])
            if role not in roles:
                roles.append(role)
            metadata["task_record_roles"] = roles

    for item in ranked:
        if bool((item.get("context_selection") or {}).get("direct_user_url")):
            add(item, "direct_user_url")

    for ordinal, label in ((1, "primary"), (2, "secondary")):
        for task_record_id in task_record_ids:
            if per_task_record_count[task_record_id] >= ordinal:
                continue
            candidate = next(
                (
                    item
                    for item in ranked
                    if id(item) not in selected
                    and task_record_id
                    in {
                        str(value).strip()
                        for value in item.get("task_record_ids") or []
                    }
                ),
                None,
            )
            if candidate is not None:
                add(candidate, f"{task_record_id}:{label}")

    for item in ranked:
        add(item)
    return ordered


def _writer_task_records(task_plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Project only user-facing factual obligations into the Writer packet.

    Retrieval object contracts, request bindings and controller bookkeeping
    remain in Trace. Repeating them beside every source consumed several
    thousand tokens without adding a user-requested fact.
    """

    compact = compact_task_plan(task_plan)
    return [
        {
            "record_id": record_id(point),
            "question": str(point.get("question") or "")[:500],
            "subject": str(point.get("subject") or "")[:240],
            "relation": str(point.get("relation") or "")[:160],
            "fields": [dict(field) for field in point.get("fields") or []],
            "time_scope": str(point.get("time_scope") or "unspecified")[:40],
            "set_semantics": str(point.get("set_semantics") or "single")[:40],
            "premise_requires_verification": bool(
                point.get("premise_requires_verification")
            ),
        }
        for point in compact.get("records") or []
        if isinstance(point, dict)
    ]


def _compact_record_routing_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Keep one compact source/record identity for RWKV, not controller logs."""

    metadata = (
        item.get("record_metadata")
        if isinstance(item.get("record_metadata"), dict)
        else {}
    )
    source_object = (
        metadata.get("source_object")
        if isinstance(metadata.get("source_object"), dict)
        else item.get("source_object")
        if isinstance(item.get("source_object"), dict)
        else {}
    )
    alignment_rows = merge_mapping_rows(
        metadata.get("object_alignments"),
        metadata.get("object_alignment"),
        item.get("object_alignments"),
        item.get("object_alignment"),
    )
    alignments = [
        {
            "relation": str(row.get("relation") or "")[:80],
            "source_object_id": str(row.get("source_object_id") or "")[:300],
            "requested_object_ids": [
                str(value)[:300]
                for value in list(row.get("requested_object_ids") or [])[:4]
            ],
        }
        for row in alignment_rows[:4]
        if isinstance(row, dict)
    ]
    subject_alignment = (
        metadata.get("rwkv_subject_alignment")
        if isinstance(metadata.get("rwkv_subject_alignment"), dict)
        else {}
    )
    compact: dict[str, Any] = {
        "subject_key": str(
            metadata.get("subject_key") or item.get("source_subject") or ""
        )[:300],
        "record_key": str(
            metadata.get("record_key")
            or source_object.get("source_record_id")
            or item.get("source_record_key")
            or ""
        )[:300],
        "source_object_id": str(source_object.get("source_object_id") or "")[:500],
        "object_alignments": alignments,
    }
    if subject_alignment:
        compact["rwkv_subject_alignment"] = {
            "relation": str(subject_alignment.get("relation") or "")[:80],
            "requested_subject": str(
                subject_alignment.get("requested_subject") or ""
            )[:300],
            "observed_source_subject": str(
                subject_alignment.get("observed_source_subject") or ""
            )[:300],
        }
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _source_context_block(
    item: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    ref_index: int,
    show_temporal_routing: bool,
    source_locator_char_limit: int,
) -> str:
    """Render exactly the bounded material that RWKV receives for one source."""

    title = str(item.get("title") or item.get("url") or f"Source {ref_index}")
    url = str(item.get("url") or "")
    context_role = str(item.get("context_role") or "")
    if context_role in {"exact_evidence_record", "candidate_evidence_record"}:
        record_metadata = dict(item.get("record_metadata") or {})
        source_object = (
            record_metadata.get("source_object")
            if isinstance(record_metadata.get("source_object"), dict)
            else {}
        )
        unresolved_record_identity = not str(
            record_metadata.get("record_key")
            or source_object.get("source_record_id")
            or item.get("source_record_key")
            or ""
        ).strip()
        atomic_span = bool(str(record_metadata.get("record_span_id") or "").strip())
        role_label = (
            "RWKV-CANDIDATE SOURCE SPAN (record identity unresolved)"
            if unresolved_record_identity and atomic_span
            else "RWKV-CANDIDATE SOURCE OBJECT SPANS (record identity unresolved)"
            if unresolved_record_identity
            else "RWKV-CANDIDATE SOURCE RECORD"
        )
        lines = [f"[S{ref_index}] {role_label}: {title}", f"URL: {url}"]
        compact_metadata = _compact_record_routing_metadata(item)
        if compact_metadata:
            # One compact identity line. The former nested alignment JSON is
            # planner-lane routing data the Writer is instructed to ignore;
            # rendering it drowned the literal spans for a fixed-state RNN.
            identity_bits = []
            if compact_metadata.get("record_key"):
                identity_bits.append(f"key={compact_metadata['record_key']}")
            if compact_metadata.get("source_object_id"):
                identity_bits.append(f"object={compact_metadata['source_object_id']}")
            # R56 ablation showed the subject-alignment pair materially helps
            # the Writer keep object identities apart (game_live/procedure
            # regressed without it); keep it as compact text, not nested JSON.
            alignment = compact_metadata.get("rwkv_subject_alignment") or {}
            requested_subject = str(alignment.get("requested_subject") or "")[:80]
            observed_subject = str(alignment.get("observed_source_subject") or "")[:80]
            if requested_subject:
                identity_bits.append(f"requested_subject={requested_subject}")
            if observed_subject:
                identity_bits.append(f"observed_subject={observed_subject}")
            if identity_bits:
                lines.append(
                    "Record identity (routing only): " + "; ".join(identity_bits)[:400]
                )
        if unresolved_record_identity and atomic_span:
            # The global Writer contract already defines the atomic boundary.
            # Repeating it for every span consumes evidence budget without
            # adding source facts or a new model decision.
            pass
        elif unresolved_record_identity:
            lines.append(
                "Record identity note: these spans share one source object and task record, "
                "but may describe different rows, dates or versions. Compare each <chunk-id> literally."
            )
    else:
        lines = [
            f"[S{ref_index}] Source label (routing metadata only): {title}",
            f"URL: {url}",
        ]

    evidence_record_id = str(item.get("evidence_record_id") or "").strip()
    if evidence_record_id:
        lines.append(f"Evidence id: {evidence_record_id}")

    for label, value in (
        ("Published", item.get("published") or item.get("published_at")),
        ("Updated", item.get("updated") or item.get("updated_at")),
        ("Source date", item.get("date")),
        ("Provider", item.get("provider") or item.get("source")),
    ):
        if value not in (None, "", [], {}):
            lines.append(f"{label}: {value}")
    freshness = item.get("freshness")
    if isinstance(freshness, dict) and freshness:
        # Render freshness only when it carries information: an
        # unknown-date/no-date row adds noise without a routing signal.
        state = str(freshness.get("state") or "")
        source_date = str(freshness.get("source_date") or "")
        if source_date or (state and state != "unknown_date"):
            bits = [b for b in (state, source_date) if b]
            lines.append("Freshness (routing only): " + " ".join(bits))
    task_record_ids = [
        str(value)
        for value in item.get("task_record_ids") or []
        if str(value).strip()
    ]
    if task_record_ids:
        lines.append("RWKV task-record bindings: " + ", ".join(task_record_ids))
    for chunk in chunks:
        temporal_role = str(chunk.get("record_temporal_role") or "").strip()
        record_date = str(chunk.get("record_date") or "").strip()
        if show_temporal_routing and (temporal_role or record_date):
            lines.append(
                "Temporal routing (literal order only): "
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
                "Observed markers (routing only): "
                + json.dumps(observed, ensure_ascii=False, separators=(",", ":"))
            )
        selected_field_ids = _bounded_unique_text_values(chunk.get("field_ids"))
        if selected_field_ids:
            lines.append(
                "Candidate fields (routing only): "
                + json.dumps(selected_field_ids, ensure_ascii=False, separators=(",", ":"))
            )
        lines.append(f"<{chunk['chunk_id']}>\n{chunk['text']}")

    source_locators = (
        ""
        if _has_grounded_locator(item)
        else str(item.get("model_locator_facts") or "").strip()
    )
    locator_limit = max(0, int(source_locator_char_limit or 0))
    if source_locators and locator_limit:
        lines.append(f"VERBATIM SOURCE LOCATORS:\n{source_locators[:locator_limit]}")
    return "\n".join(lines)


def _render_evidence_packet(
    *,
    factual_task_records: list[dict[str, Any]],
    calculations: list[dict[str, Any]],
    freshness_policy: Any,
    selected: list[dict[str, Any]],
    packed: list[list[dict[str, Any]]],
    show_temporal_routing: bool,
    source_locator_char_limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Render the packet and its contiguous citation projection together."""

    blocks: list[str] = []
    projected_sources: list[dict[str, Any]] = []
    for item, chunks in zip(selected, packed):
        if not chunks:
            continue
        if not str(item.get("evidence_record_id") or "").strip():
            metadata = (
                item.get("record_metadata")
                if isinstance(item.get("record_metadata"), dict)
                else {}
            )
            identity = {
                "url": str(item.get("url") or ""),
                "record_key": str(metadata.get("record_key") or ""),
                "record_span_id": str(metadata.get("record_span_id") or ""),
                "chunks": [
                    {
                        "chunk_id": str(chunk.get("chunk_id") or ""),
                        "text": str(chunk.get("text") or ""),
                    }
                    for chunk in chunks
                ],
            }
            item["evidence_record_id"] = "E-" + hashlib.sha256(
                json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
        ref_index = len(projected_sources) + 1
        blocks.append(
            _source_context_block(
                item,
                chunks,
                ref_index=ref_index,
                show_temporal_routing=show_temporal_routing,
                source_locator_char_limit=source_locator_char_limit,
            )
        )
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
    if factual_task_records:
        sections.append(
            "RWKV FACTUAL PLAN (user-requested records; not a completion gate):\n"
            + json.dumps(factual_task_records, ensure_ascii=False, separators=(",", ":"))
        )
    if calculations:
        sections.append(
            "TOOL RESULTS:\n"
            + json.dumps(calculations, ensure_ascii=False, separators=(",", ":"))
        )
    if isinstance(freshness_policy, dict) and freshness_policy:
        sections.append(
            "QUESTION TIME/FRESHNESS POLICY (metadata for RWKV, not an answer gate):\n"
            + json.dumps(freshness_policy, ensure_ascii=False, separators=(",", ":"))
        )
    sections.append(
        "RETRIEVED SOURCES:\n"
        + ("\n\n".join(blocks) if blocks else "No source text was retrieved.")
    )
    return "\n\n".join(sections), projected_sources


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
    supports a task_record and it does not suppress RWKV's ability to answer.
    """

    strategy_config = (constraints or {}).get("strategy_config") or {}
    explicit_source_limit = (
        strategy_config.get("context_source_count")
        if isinstance(strategy_config, dict)
        and strategy_config.get("context_source_count") is not None
        else (constraints or {}).get("context_source_count")
    )
    configured_source_limit = int(
        explicit_source_limit
        if explicit_source_limit is not None
        else DATA_PIPELINE.get("context_source_count", 8) or 8
    )
    configured_source_limit = max(1, min(configured_source_limit, 24))
    evidence_record_source_limit = max(
        configured_source_limit,
        min(
            24,
            int(
                DATA_PIPELINE.get("final_context_max_evidence_records", 24)
                or 24
            ),
        ),
    )
    ledger_value = data.get("evidence_ledger") or (constraints or {}).get("evidence_ledger")
    task_plan = (constraints or {}).get("task_plan")
    if not task_records(task_plan) and isinstance(ledger_value, dict):
        task_plan = {
            "goal": query,
            "records": [
                {
                    "record_id": row.get("task_record_id"),
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
                for row in ledger_value.get("task_records") or []
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
    # Only RWKV-selected, source-located evidence records enter the factual
    # context. Raw fetched pages remain retrieval artifacts and can never be
    # promoted by a deterministic fallback after extraction fails.
    fetched_items = [
        item for item in data.get("results") or [] if isinstance(item, dict)
    ]
    record_items = _evidence_record_source_items(
        ledger_value,
        grounded_span_limit=grounded_span_limit,
        ranked_results=fetched_items,
    )
    merged_records, duplicate_record_count = _merge_source_records(record_items)
    ordered_records = _order_sources_for_context(
        [item for item in merged_records if _source_body(item)],
        query,
        task_plan,
    )
    # ``context_source_count`` historically counted page-sized blocks. Once
    # unresolved records become one exact span per block, retaining that old
    # cap silently discards evidence before the existing token budget runs.
    # Production therefore lets evidence records use a larger hard safety
    # cap while the exact rendered packet remains bounded by ``token_budget``.
    # Explicit caller limits remain authoritative for experiments and tests.
    source_limit = (
        max(1, min(int(max_sources), 24))
        if max_sources is not None
        else configured_source_limit
        if explicit_source_limit is not None or not ordered_records
        else evidence_record_source_limit
    )
    # Raw fetched pages never enter the factual Writer lane. An extraction
    # miss remains an explicit missing-evidence state; any future recovery must
    # first produce a normally grounded EvidenceSpan with source offsets.
    unique_items = list(ordered_records)
    duplicate_source_count = duplicate_record_count

    selected = unique_items[:source_limit]

    default_budget = evidence_tokens(get_llm_context_length())
    budget = (
        default_budget
        if token_budget is None
        else max(128, min(int(token_budget), default_budget))
    )
    factual_task_records = _writer_task_records(task_plan)
    show_temporal_routing = _task_plan_requests_temporal_identity(task_plan)
    calculations = [
        item for item in data.get("calculation_results") or [] if isinstance(item, dict)
    ]
    freshness_policy = (constraints or {}).get("freshness_policy")
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
    empty_text, _ = _render_evidence_packet(
        factual_task_records=factual_task_records,
        calculations=calculations,
        freshness_policy=freshness_policy,
        selected=selected,
        packed=packed,
        show_temporal_routing=show_temporal_routing,
        source_locator_char_limit=source_locator_char_limit,
    )
    static_overhead_tokens = get_token_count(empty_text)
    chunk_budget = max(0, budget - static_overhead_tokens)

    # Enforce the budget against the exact text RWKV will receive, including
    # headers and routing metadata. The former estimate counted only chunk
    # bodies and allowed the real packet to grow from 6K to 9–13.5K tokens.
    cursor = 0
    while selected:
        made_progress = False
        for source_index, chunks in enumerate(per_source_chunks):
            if cursor >= len(chunks):
                continue
            candidate_packed = [list(rows) for rows in packed]
            candidate_packed[source_index].append(chunks[cursor])
            candidate_text, _ = _render_evidence_packet(
                factual_task_records=factual_task_records,
                calculations=calculations,
                freshness_policy=freshness_policy,
                selected=selected,
                packed=candidate_packed,
                show_temporal_routing=show_temporal_routing,
                source_locator_char_limit=source_locator_char_limit,
            )
            if get_token_count(candidate_text) > budget:
                continue
            packed = candidate_packed
            made_progress = True
        if not made_progress:
            break
        cursor += 1

    # Budget packing above runs in strength order so the strongest records win
    # marginal chunks. The final render is then reversed so the strongest
    # record sits LAST in the packet, nearest the continuation point: RWKV is
    # a fixed-state RNN and the most recently read span dominates its state
    # when writing begins. Attention routing only — identical records and
    # chunk selections reach the Writer either way.
    selected = list(reversed(selected))
    packed = list(reversed(packed))
    text, projected_sources = _render_evidence_packet(
        factual_task_records=factual_task_records,
        calculations=calculations,
        freshness_policy=freshness_policy,
        selected=selected,
        packed=packed,
        show_temporal_routing=show_temporal_routing,
        source_locator_char_limit=source_locator_char_limit,
    )
    context_tokens = get_token_count(text)
    refs = _citation_refs(projected_sources)
    return {
        "text": text,
        "evidence_text": text,
        "selected_evidence": projected_sources,
        "citation_refs": refs,
        "calculation_results": calculations,
        "factual_task_records": factual_task_records,
        "usable_evidence_count": len(projected_sources),
        "duplicate_source_count": duplicate_source_count,
        "chunk_count": sum(len(chunks) for chunks in packed),
        "context_chars": len(text),
        "context_tokens": context_tokens,
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
            "calculation_count": len(calculations),
            "factual_task_record_count": len(factual_task_records),
            "factual_task_records_with_selected_sources": sum(
                any(
                    record_id(task_record)
                    in {
                        str(value).strip()
                        for value in source.get("task_record_ids") or []
                    }
                    for source in projected_sources
                )
                for task_record in factual_task_records
                if record_id(task_record)
            ),
            "context_tokens": context_tokens,
            "evidence_budget_tokens": budget,
            "source_chunk_budget_tokens": chunk_budget,
            "static_overhead_tokens": static_overhead_tokens,
            "configured_source_limit": source_limit,
            "legacy_page_source_limit": configured_source_limit,
            "evidence_record_source_limit": evidence_record_source_limit,
            "max_chunks_per_source": effective_chunk_limit,
            "max_grounded_spans_per_source": grounded_span_limit,
            "evidence_review_included": False,
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
    evidence_resolution_view = str(context.get("evidence_resolution_view") or "").strip()
    resolution_section = (
        f"EVIDENCE RESOLUTION CONTROL LANE:\n{evidence_resolution_view}\n\n"
        if evidence_resolution_view
        else ""
    )
    evidence_text = str(context.get("evidence_text") or context.get("text") or "")
    # Evidence spans are rendered as <span-ref>, legacy <chunk-N>, or the
    # current <locator-chunk-N>/<locator-structured-record> blocks. The
    # availability banner must only appear when none of these exist; keying it
    # off a stale tag subset silently ordered the Writer to refuse packets
    # that did contain literal spans.
    has_literal_facts = (
        "<span-ref" in evidence_text
        or "<chunk-" in evidence_text
        or "<locator-" in evidence_text
        or "TOOL RESULTS:" in evidence_text
    )
    availability_section = (
        "FINAL FACT AVAILABILITY:\n"
        "This packet contains no literal factual span or tool result. Do not supply a concrete "
        "date, version, identifier, name, status, command, or example from model memory. State "
        "concisely that the retrieved evidence is insufficient for the requested fields.\n\n"
        if not has_literal_facts
        else ""
    )
    return (
        "You are the final RWKV answer writer. Answer the user's question yourself using the retrieved "
        "material and tool results below. The material can be incomplete or conflicting; judge it directly. "
        "If some information is missing, answer the supported parts and clearly state what remains uncertain. "
        "Only tool results and literal text inside <span-ref>, <locator-...>, or compatibility <chunk-id> blocks "
        "are factual material. Source labels, "
        "titles, URLs, provider labels, routing metadata, and the Evidence Resolution control lane "
        "only locate or prioritize candidates; verify every P#/E-* binding against the chunk text. Treat each "
        "S# as an independent record: compare its literal object, version, and date, and do not move a field "
        "between records unless their text explicitly establishes the same identity. "
        "For commands, dates, versions, identifiers, names, statuses, quoted output, and examples, use only "
        "values explicitly present in the material; do not substitute a plausible value or invent an example. "
        "For a current/latest request, do not present a future-dated or explicitly historical record as current; "
        "state the time conflict or uncertainty instead. For a repository-specific request, bind task_records to the "
        "exact owner/repository rather than another project on the same host. Satisfy every requested field when "
        "the material supports it, and explicitly identify any requested field that remains unsupported. "
        "Answer only the fields the user requested. Do not append adjacent limitations, examples, commands, or "
        "background merely because they occur near the supporting span. Use one clean answer path and keep it concise. "
        "Do not restart, repeat, enumerate duplicate support, "
        "or re-check fields already answered. If the answer starts to loop, stop immediately and return the "
        "best answer already written. Do not describe controller rules or the research process. Use [S1], [S2], ... "
        "when citing a retrieved source.\n\n"
        + runtime_section
        + f"USER QUESTION:\n{query}\n\n"
        + resolution_section
        + f"EXACT EVIDENCE LANE:\n{evidence_text}\n\n"
        + availability_section
        + _writer_obligation_tail(query, context)
        + "Write the final answer now."
    )


def _writer_obligation_tail(query: str, context: dict[str, Any]) -> str:
    """One obligation line rendered LAST, nearest the continuation point.

    RWKV is a fixed-state RNN: the most recently read text dominates its state
    when writing begins, so the current obligation (which record and fields to
    answer) is restated at the tail. This adds no new rule — it compresses the
    already-stated task into one line at the position where it is most salient.
    """

    records = [
        record
        for record in context.get("factual_task_records") or []
        if isinstance(record, dict)
    ]
    if not records:
        return ""
    parts = []
    for record in records[:4]:
        rid = str(record.get("record_id") or "").strip()
        fields = [
            str(field.get("name") or "").strip()
            for field in record.get("fields") or []
            if isinstance(field, dict) and str(field.get("name") or "").strip()
        ]
        if rid and fields:
            parts.append(f"{rid}: {', '.join(fields[:6])}")
        elif rid:
            parts.append(rid)
    if not parts:
        return ""
    return (
        "CURRENT OBLIGATION (restated): answer exactly these requested fields — "
        + "; ".join(parts)
        + " — every concrete value verbatim from the spans above.\n\n"
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
            response = llm.text_completion(
                prompt,
                max_tokens=max_tokens,
                stop=FINAL_ANSWER_STOP_SUFFIXES,
            )
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
    prebuilt_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the single public answer written by RWKV without post-processing."""

    del execution_context
    context = (
        dict(prebuilt_context)
        if isinstance(prebuilt_context, dict)
        else build_evidence_context(data, constraints=constraints, query=query)
    )
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
        "evidence_resolution": context.get(
            "evidence_resolution"
        )
        or {},
        "answer_alignment": {},
        "answer_quality": {},
    }


__all__ = [
    "build_evidence_context",
    "synthesize_retrieval_answer",
]
