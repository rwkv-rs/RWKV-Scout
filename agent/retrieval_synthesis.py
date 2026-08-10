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
from utils.model_budget import bounded_completion_budget
from utils.rwkv_prompt import (
    build_final_continuation_prompt,
    consume_final_prefill_boundary,
)


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
    """Return whether a source carries an exact RWKV-selected source span."""

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
    """Put query-focused original spans before short model locators.

    Page extraction keeps ``selected_source_chunks`` as an attention routing
    result and ``source_chunks`` as the complete provenance record.  Packing
    model-grounded quotes first made a merely related short quote consume the
    per-source chunk cap before the exact requested field later in the page.
    Attention-ranked original spans are verbatim source text and therefore
    receive first priority; grounded locators and neighbouring chunks remain
    available as context.  Complete original chunks stay in persistent state
    for later replans.
    """

    grounded_rows = [
        {
            "chunk_id": "locator-" + str(row.get("chunk_id") or index + 1),
            "index": int(row.get("chunk_index") or index),
            "text": str(row.get("quote") or "").strip(),
        }
        for index, row in enumerate(item.get("chunk_candidates") or [])
        if isinstance(row, dict)
        and row.get("supported") is True
        and row.get("source_grounded") is True
        and str(row.get("quote") or "").strip()
    ][: max(1, int(grounded_span_limit))]
    selected_rows = [
        row
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
    preferred_rows = [*selected_rows, *grounded_rows]
    original_rows = [
        row
        for row in item.get("source_chunks") or []
        if isinstance(row, dict) and str(row.get("text") or "").strip()
    ]
    # Once RWKV has located exact evidence, the final writer needs that span
    # and its immediate source neighbourhood, not an unrelated page preamble
    # or every historical table row from the same document.  The complete
    # source_chunks record remains in state for recovery and later replans.
    routed_indices = {
        int(row.get("index", row.get("chunk_index", 0)) or 0)
        for row in [*grounded_rows, *selected_rows]
    }
    if routed_indices:
        original_rows = [
            row
            for row in original_rows
            if any(
                abs(int(row.get("index", 0) or 0) - selected_index) <= 1
                for selected_index in routed_indices
            )
        ]
    chunks: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    seen_text: set[str] = set()
    for index, row in enumerate([*preferred_rows, *original_rows]):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        chunk_id = str(row.get("chunk_id") or f"chunk-{index + 1}")
        chunk_index = int(row.get("index", index) or index)
        identity = (chunk_id, chunk_index, text)
        normalized_text = re.sub(r"\s+", " ", text).strip().casefold()
        if identity in seen or normalized_text in seen_text:
            continue
        seen.add(identity)
        seen_text.add(normalized_text)
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
            chunks.append(
                {
                    "chunk_id": (
                        chunk_id
                        if len(parts) == 1
                        else f"{chunk_id}.part-{part_index}"
                    ),
                    "index": chunk_index,
                    "text": part,
                }
            )
    if chunks:
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
                "task": str(row.get("task") or row.get("objective") or "")[:600],
                "evidence_needed": [
                    str(item)[:400]
                    for item in row.get("evidence_needed") or []
                    if str(item).strip()
                ][:8],
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
                "task": (
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
    projected: list[dict[str, Any]] = []
    for claim in value.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or claim.get("point_id") or "").strip()
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
            projected.append(item)
    return projected


def _merge_source_records(items: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge duplicate URLs while preserving all model-routed source spans."""

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
            elif current.get(key) in (None, "", [], {}):
                current[key] = value
    return merged, duplicate_count


def _citation_refs(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        url = str(item.get("url") or "").strip()
        refs.append(
            {
                "ref_id": f"S{index}",
                "title": str(item.get("title") or url or f"Source {index}"),
                "url": url,
            }
        )
    return refs


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
    grounded_span_limit = max(
        1,
        min(
            8,
            int(DATA_PIPELINE.get("final_context_max_grounded_spans_per_source", 3) or 3),
        ),
    )
    raw_items = [
        *_claim_grounded_source_items(
            ledger_value,
            grounded_span_limit=grounded_span_limit,
        ),
        *[item for item in data.get("results") or [] if isinstance(item, dict)],
    ]
    merged_items, duplicate_source_count = _merge_source_records(raw_items)
    unique_items = [item for item in merged_items if _source_body(item)]

    # Preserve the retrieval ranking while reserving one slot for every
    # explicitly bound RWKV task point. This prevents one high-volume aspect
    # from evicting all evidence for another aspect of a mixed question.
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def select(item: dict[str, Any]) -> None:
        identity = _source_identity(item)
        if len(selected) >= source_limit or (identity and identity in selected_ids):
            return
        selected.append(item)
        if identity:
            selected_ids.add(identity)

    # A completed RWKV cross-validation may bind supported task points to
    # concrete [S#] sources. Preserve that model-owned choice as attention
    # priority for the final writer; it is not a controller-side relevance
    # judgement and every other source remains eligible for budget backfill.
    validated_urls = {
        str(value).strip().casefold().rstrip("/")
        for value in (
            (constraints or {}).get("validated_source_urls")
            or data.get("validated_source_urls")
            or []
        )
        if str(value).strip()
    }
    # Reserve one exact model-grounded source for each claim before any
    # high-volume claim or validated set can consume every context slot.
    if isinstance(ledger_value, dict):
        claim_buckets: list[list[dict[str, Any]]] = []
        for claim in ledger_value.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id") or claim.get("point_id") or "")
            bucket = [
                item
                for item in unique_items
                if claim_id
                and claim_id
                in {
                    str(value)
                    for value in item.get("claim_ids") or []
                    if str(value).strip()
                }
                and _has_grounded_locator(item)
            ]
            claim_buckets.append(bucket)
        for bucket in claim_buckets:
            if bucket:
                select(bucket[0])

        for item in unique_items:
            if str(item.get("url") or "").strip().casefold().rstrip("/") in validated_urls:
                select(item)

        cursor = 1
        while len(selected) < source_limit and any(cursor < len(bucket) for bucket in claim_buckets):
            for bucket in claim_buckets:
                if cursor < len(bucket):
                    select(bucket[cursor])
            cursor += 1

        # A claim without an exact grounded locator still receives one generic
        # bound source when space remains, preserving recall without declaring
        # that source supportive.
        for claim in ledger_value.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id") or claim.get("point_id") or "")
            bound = next(
                (
                    item
                    for item in unique_items
                    if claim_id
                    and claim_id
                    in {
                        str(value)
                        for value in item.get("claim_ids") or []
                        if str(value).strip()
                    }
                ),
                None,
            )
            if bound is not None:
                select(bound)
    else:
        for item in unique_items:
            if str(item.get("url") or "").strip().casefold().rstrip("/") in validated_urls:
                select(item)
    for item in unique_items:
        if _has_grounded_locator(item):
            select(item)
    for item in unique_items:
        select(item)

    default_budget = evidence_tokens(get_llm_context_length())
    budget = (
        default_budget
        if token_budget is None
        else max(128, min(int(token_budget), default_budget))
    )
    claims = _claim_projection(ledger_value)
    calculations = [
        item for item in data.get("calculation_results") or [] if isinstance(item, dict)
    ]
    freshness_policy = (constraints or {}).get("freshness_policy") or data.get(
        "freshness_policy"
    )
    cross_validation_review = (constraints or {}).get("last_cross_validation") or data.get(
        "last_cross_validation"
    )
    static_preview: list[str] = []
    if claims:
        static_preview.append(json.dumps(claims, ensure_ascii=False, indent=2))
    if calculations:
        static_preview.append(json.dumps(calculations, ensure_ascii=False, indent=2))
    if isinstance(freshness_policy, dict) and freshness_policy:
        static_preview.append(json.dumps(freshness_policy, ensure_ascii=False, indent=2))
    if isinstance(cross_validation_review, dict) and cross_validation_review:
        static_preview.append(
            json.dumps(cross_validation_review, ensure_ascii=False, indent=2)
        )
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
        lines = [f"[S{ref_index}] {title}", f"URL: {url}"]
        claim_ids = [
            str(value) for value in item.get("claim_ids") or [] if str(value).strip()
        ]
        if claim_ids:
            lines.append(f"RWKV task-point bindings: {', '.join(claim_ids)}")
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
        for chunk in chunks:
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
    if claims:
        sections.append(
            "RESEARCH CHECKLIST (progress information for RWKV, not an answer gate):\n"
            + json.dumps(claims, ensure_ascii=False, indent=2)
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
    if isinstance(cross_validation_review, dict) and cross_validation_review:
        sections.append(
            "LATEST RWKV CROSS-VALIDATION REVIEW (model-authored coverage notes, not an answer gate):\n"
            + json.dumps(cross_validation_review, ensure_ascii=False, indent=2)
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
            "calculation_count": len(calculations),
            "context_tokens": get_token_count(text),
            "evidence_budget_tokens": budget,
            "source_chunk_budget_tokens": chunk_budget,
            "static_overhead_tokens": static_overhead_tokens,
            "configured_source_limit": source_limit,
            "max_chunks_per_source": effective_chunk_limit,
            "max_grounded_spans_per_source": grounded_span_limit,
            "cross_validation_review_included": bool(
                isinstance(cross_validation_review, dict) and cross_validation_review
            ),
            "source_locator_char_limit": max(0, int(source_locator_char_limit or 0)),
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
        "For a current/latest request, do not present a future-dated or explicitly historical record as current; "
        "state the time conflict or uncertainty instead. For a repository-specific request, bind claims to the "
        "exact owner/repository rather than another project on the same host. Satisfy every requested field when "
        "the material supports it, and explicitly identify any requested field that remains unsupported. "
        "Use one clean answer path and keep it concise. Do not restart, repeat, enumerate duplicate support, "
        "or re-check fields already answered. If the answer starts to loop, stop immediately and return the "
        "best answer already written. Do not describe controller rules or the research process. Use [S1], [S2], ... "
        "when citing a retrieved source.\n\n"
        + runtime_section
        + f"USER QUESTION:\n{query}\n\n"
        + f"RESEARCH MATERIAL:\n{context['text']}\n\n"
        + "FINAL EVIDENCE CHECK:\n"
        + "A requested field marked missing by the latest RWKV cross-validation remains unsupported "
        + "in the final answer; do not fill it from another source snippet or from model memory. "
        + "For a current/latest field, an explicitly historical record cannot resolve a missing current value. "
        + "If no source text was retrieved, do not supply names, dates, versions, identifiers, statuses, or themes "
        + "from memory. This does not prevent an answer: answer supported fields and clearly state which fields "
        + "could not be verified.\n\n"
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
