"""Shared evidence-quality checks for retrieval results.

Discovery metadata and a fetched page are not automatically answer evidence.
This module keeps that distinction small and reusable: it does not decide what
the answer is, it only prevents empty, title-only, or navigation-only payloads
from being advertised as usable evidence.
"""

from __future__ import annotations

import re
from typing import Any


MIN_SUBSTANTIVE_EVIDENCE_CHARS = 48
MIN_PAGE_BODY_CHARS = MIN_SUBSTANTIVE_EVIDENCE_CHARS
BODY_EVIDENCE_ORIGINS = frozenset(
    {
        "fetched_page_body",
        "fetched_page_chunk",
        "fetched_page_body_with_model_locator",
        "local_document",
    }
)
STRUCTURED_EVIDENCE_ORIGINS = frozenset(
    {
        "structured_api_record",
        "structured_record",
        "api_record",
    }
)
DISCOVERY_ONLY_ORIGINS = frozenset({"discovery", "search_result", "snippet"})
_DATE_PATTERNS = (
    re.compile(r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b"),
    re.compile(r"\b(?:19|20)\d{2}[-/.]\d{1,2}\b"),
    re.compile(r"(?:19|20)\d{2}\u5e74\d{1,2}\u6708(?:\d{1,2}\u65e5)?"),
    re.compile(r"\d{1,2}\u6708\d{1,2}\u65e5"),
)


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def evidence_origin(item: dict[str, Any]) -> str:
    """Return the explicit provenance class for one retrieval record."""

    return _normalized(item.get("evidence_origin")).casefold()


def evidence_kind(item: dict[str, Any]) -> str:
    """Classify a record without treating discovery metadata as evidence."""

    origin = evidence_origin(item)
    if origin in BODY_EVIDENCE_ORIGINS:
        return "page_body"
    if origin in STRUCTURED_EVIDENCE_ORIGINS:
        return "structured_record"
    if origin in DISCOVERY_ONLY_ORIGINS:
        return "discovery"

    # Backward-compatible records created before provenance was added may still
    # carry an explicit page excerpt.  A bare snippet is never promoted by this
    # fallback.  New writers should always provide evidence_origin explicitly.
    if _normalized(item.get("source_excerpt")) or _normalized(item.get("page_excerpt")):
        return "page_body_legacy"
    if _normalized(item.get("structured_evidence_text")):
        return "structured_record_legacy"
    if _normalized(item.get("content")) and not _normalized(item.get("snippet")):
        return "page_body_legacy"
    if _normalized(item.get("abstract")) and not _normalized(item.get("snippet")):
        return "structured_record_legacy"
    return "discovery"


def evidence_text(item: dict[str, Any]) -> str:
    """Return only source-body or explicit structured-record text.

    Search snippets, titles, URLs, provider summaries, and model-extracted
    chunk facts are intentionally excluded.  They remain useful in traces and
    ranking, but cannot satisfy the final-answer evidence gate.
    """

    kind = evidence_kind(item)
    if kind in {"page_body", "page_body_legacy"}:
        for key in ("source_excerpt", "page_excerpt", "content"):
            value = str(item.get(key) or "").replace("\x00", "").replace("\r\n", "\n").strip()
            if value:
                return value
        return ""
    if kind in {"structured_record", "structured_record_legacy"}:
        for key in ("structured_evidence_text", "abstract", "content"):
            value = str(item.get(key) or "").replace("\x00", "").replace("\r\n", "\n").strip()
            if value:
                return value
    return ""


def has_substantive_evidence(item: dict[str, Any]) -> bool:
    """Return whether a result contains enough source text for synthesis.

    The threshold is deliberately applied to the aggregate evidence from one
    source, so short but legitimate date/entity fields can contribute together.
    A status flag alone is never evidence.
    """

    text = re.sub(r"\s+", " ", evidence_text(item)).strip()
    return evidence_kind(item) not in {"discovery", ""} and len(text) >= MIN_SUBSTANTIVE_EVIDENCE_CHARS


def substantive_evidence_items(items: Any) -> list[dict[str, Any]]:
    """Return only records whose body/structured evidence can enter synthesis.

    Search titles, snippets, candidate URLs and provider metadata intentionally
    do not pass this boundary.  They remain available in the execution trace
    for routing and diagnostics.
    """

    if not isinstance(items, list):
        return []
    return [
        item
        for item in items
        if isinstance(item, dict) and has_substantive_evidence(item)
    ]


def evidence_provenance(item: dict[str, Any]) -> dict[str, Any]:
    """Return serializable provenance metadata for traces and answer context."""

    text = evidence_text(item)
    kind = evidence_kind(item)
    origin = evidence_origin(item)
    return {
        "origin": origin or ("legacy_page_body" if kind == "page_body_legacy" else "legacy_structured_record" if kind == "structured_record_legacy" else "discovery"),
        "kind": kind,
        "body_verified": kind in {"page_body", "page_body_legacy"},
        "content_chars": len(text),
        "source_locator": item.get("source_locator") or item.get("evidence_locator") or {},
    }


def date_mentions(value: Any) -> list[str]:
    """Extract date-shaped strings for ranking/inspection, never as facts."""

    text = _normalized(value)
    found: list[str] = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.findall(text):
            value = _normalized(match)
            if value and value not in found:
                found.append(value)
    return [item for item in found if not any(item != other and item in other for other in found)]
