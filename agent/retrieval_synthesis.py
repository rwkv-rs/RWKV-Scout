"""Minimal evidence packing and the single RWKV final-writer call.

This module deliberately has no answer gate, repair pass, translation pass,
deterministic answer fallback, refusal generator, or quality status.  Retrieval
code may organise source material, but only RWKV writes the public answer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from config import DATA_PIPELINE, get_llm_context_length, is_local_provider
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


def _source_chunks(item: dict[str, Any], *, chunk_tokens: int = 800) -> list[dict[str, Any]]:
    """Keep retrieval chunks when present, otherwise chunk the retained body."""

    rows = item.get("source_chunks") or []
    chunks = [
        {
            "chunk_id": str(row.get("chunk_id") or f"chunk-{index + 1}"),
            "index": int(row.get("index", index) or index),
            "text": str(row.get("text") or "").strip(),
        }
        for index, row in enumerate(rows)
        if isinstance(row, dict) and str(row.get("text") or "").strip()
    ]
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
        projected.append(
            {
                "claim_id": str(row.get("claim_id") or row.get("point_id") or ""),
                "task": str(row.get("task") or row.get("objective") or "")[:600],
                "retrieved_source_count": len(spans),
                "source_urls": source_urls[:8],
            }
        )
    return projected


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
) -> dict[str, Any]:
    """Pack fetched chunks into a bounded, source-labelled RWKV context.

    Packing is a resource operation only.  It does not decide whether a source
    supports a claim and it does not suppress RWKV's ability to answer.
    """

    del query
    source_limit = int(
        ((constraints or {}).get("strategy_config") or {}).get("context_source_count")
        or (constraints or {}).get("context_source_count")
        or DATA_PIPELINE.get("context_source_count", 8)
        or 8
    )
    source_limit = max(1, min(source_limit, 24))
    raw_items = [item for item in data.get("results") or [] if isinstance(item, dict)]

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_source_count = 0
    for item in raw_items:
        identity = _source_identity(item)
        if identity and identity in seen:
            duplicate_source_count += 1
            continue
        if identity:
            seen.add(identity)
        if _source_body(item):
            selected.append(dict(item))
        if len(selected) >= source_limit:
            break

    budget = evidence_tokens(get_llm_context_length())
    per_source_chunks = [_source_chunks(item) for item in selected]
    packed: list[list[dict[str, Any]]] = [[] for _ in selected]
    used_tokens = 0
    cursor = 0
    while selected and used_tokens < budget:
        made_progress = False
        for source_index, chunks in enumerate(per_source_chunks):
            if cursor >= len(chunks):
                continue
            chunk = chunks[cursor]
            chunk_tokens = get_token_count(chunk["text"])
            if used_tokens + chunk_tokens > budget:
                continue
            packed[source_index].append(chunk)
            used_tokens += chunk_tokens
            made_progress = True
        if not made_progress:
            break
        cursor += 1

    blocks: list[str] = []
    projected_sources: list[dict[str, Any]] = []
    for index, (item, chunks) in enumerate(zip(selected, packed), start=1):
        if not chunks:
            continue
        title = str(item.get("title") or item.get("url") or f"Source {index}")
        url = str(item.get("url") or "")
        lines = [f"[S{index}] {title}", f"URL: {url}"]
        for chunk in chunks:
            lines.append(f"<{chunk['chunk_id']}>\n{chunk['text']}")
        # Current retrieval records populate ``model_locator_facts`` only from
        # quotes that were mapped back to exact fetched-source spans.  Repeat
        # those compact spans near the end of the source block as an attention
        # aid for RWKV.  Never fall back to a model-authored fact field.
        source_locators = str(item.get("model_locator_facts") or "").strip()
        if source_locators:
            lines.append(f"VERBATIM SOURCE LOCATORS:\n{source_locators[:2400]}")
        blocks.append("\n".join(lines))
        projected_sources.append(
            {
                **item,
                "ref_id": f"S{index}",
                "packed_chunks": chunks,
                "chunk_count": len(chunks),
                "evidence_text": "\n\n".join(chunk["text"] for chunk in chunks),
            }
        )

    calculations = [
        item for item in data.get("calculation_results") or [] if isinstance(item, dict)
    ]
    sections: list[str] = []
    claims = _claim_projection(data.get("claim_ledger") or (constraints or {}).get("claim_ledger"))
    if claims:
        sections.append(
            "RESEARCH CHECKLIST (progress information for RWKV, not an answer gate):\n"
            + json.dumps(claims, ensure_ascii=False, indent=2)
        )
    if calculations:
        sections.append(
            "TOOL RESULTS:\n" + json.dumps(calculations, ensure_ascii=False, indent=2)
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
            for chunks, all_chunks in zip(packed, per_source_chunks)
        ),
        "validation": {},
        "context_stats": {
            "source_count": len(projected_sources),
            "chunk_count": sum(len(chunks) for chunks in packed),
            "duplicate_source_count": duplicate_source_count,
            "calculation_count": len(calculations),
            "context_tokens": get_token_count(text),
            "evidence_budget_tokens": budget,
        },
    }


def _writer_prompt(query: str, context: dict[str, Any]) -> str:
    return (
        "You are the final RWKV answer writer. Answer the user's question yourself using the retrieved "
        "material and tool results below. The material can be incomplete or conflicting; judge it directly. "
        "If some information is missing, answer the supported parts and clearly state what remains uncertain. "
        "For commands, dates, versions, identifiers, names, statuses, quoted output, and examples, use only "
        "values explicitly present in the material; do not substitute a plausible value or invent an example. "
        "Do not describe controller rules or the research process. Use [S1], [S2], ... "
        "when citing a retrieved source.\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"RESEARCH MATERIAL:\n{context['text']}\n\n"
        "Write the final answer now."
    )


def synthesize_retrieval_answer(
    query: str,
    data: dict[str, Any],
    llm=None,
    constraints: dict[str, Any] | None = None,
    execution_context: str = "",
    termination_reason: str = "model_requested_finish",
) -> dict[str, Any]:
    """Call RWKV once and return its text unchanged as the public answer."""

    del execution_context
    context = build_evidence_context(data, constraints=constraints, query=query)
    user_prompt = _writer_prompt(query, context)
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
    provider = str(getattr(llm, "provider", "") or "")
    if hasattr(llm, "text_completion") and (not provider or is_local_provider(provider)):
        response = llm.text_completion(
            prompt,
            max_tokens=max_tokens,
        )
    else:
        response = llm.chat_completion(
            [{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
        )

    raw_model_output = _clean_answer(getattr(response, "content", ""))
    answer = consume_final_prefill_boundary(raw_model_output)
    if not answer.strip():
        raise ConnectionError("RWKV final writer returned no output")

    return {
        "content": answer,
        "mode": "rwkv_final",
        "evidence_count": len(context["selected_evidence"]),
        "citation_refs": context["citation_refs"],
        "prompt": prompt,
        "model_output": answer,
        "raw_model_output": raw_model_output,
        "generation_attempts": [
            {
                "stage": "rwkv_final",
                "output": answer,
                "termination_reason": termination_reason,
            }
        ],
        "context_text": context["text"],
        "selected_evidence": context["selected_evidence"],
        "context_stats": context["context_stats"],
        "validation": {},
        "answer_alignment": {},
        "answer_quality": {},
    }


__all__ = ["build_evidence_context", "synthesize_retrieval_answer"]
