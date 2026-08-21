"""Shared evidence-quality checks for retrieval results.

Discovery metadata and a fetched page are not automatically answer evidence.
This module keeps that distinction small and reusable: it does not decide what
the answer is, it only prevents empty, title-only, or navigation-only payloads
from being advertised as usable evidence.
"""

from __future__ import annotations

import re
from typing import Any


MIN_SUBSTANTIVE_EVIDENCE_CHARS = 80
# Chinese/Japanese/Korean pages carry substantially more information per
# character than whitespace-tokenised Latin prose.  A fixed 80-character gate
# discards short encyclopedia entries, release notices and government alerts.
MIN_CJK_EVIDENCE_CHARS = 36
MIN_LATIN_EVIDENCE_WORDS = 10
MIN_PAGE_BODY_CHARS = MIN_SUBSTANTIVE_EVIDENCE_CHARS
# A failed locator may fall back only to a compact, cleaned entry. This keeps
# small encyclopedia/API pages usable without admitting a whole article.
MAX_SHORT_BODY_FALLBACK_CHARS = 3000
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

_IMAGE_PLACEHOLDER_RE = re.compile(
    r"\[\s*(?:image|图片|鍥剧墖)[^\]]*(?:filtered|过滤|杩囨护)[^\]]*\]",
    flags=re.IGNORECASE,
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)", flags=re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")
_URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_SCRIPT_URL_RE = re.compile(r"\b(?:javascript|data):\S+", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]{1,400}>")
_NAVIGATION_MARKERS = (
    "skip to",
    "skip navigation",
    "log in",
    "login",
    "sign up",
    "signup",
    "reload to refresh",
    "you signed out",
    "table of contents",
    "navigation",
    "main menu",
    "menu",
    "search",
    "copyright",
    "cookie policy",
    "privacy policy",
    "terms of use",
    "breadcrumb",
    "[home]",
    "[登录]",
    "[注册]",
    "[首页]",
    "登录",
    "注册",
    "导航",
    "菜单",
    "搜索",
)


def _navigation_marker_hit(value: str) -> bool:
    """Recognize standalone/page-chrome labels without matching prose words."""

    lowered = re.sub(r"\s+", " ", str(value or "")).strip(" -:;,.[]()\t\n").casefold()
    if not lowered:
        return False
    exact = {
        "home", "menu", "search", "login", "signup", "copyright",
        "breadcrumb", "navigation", "登录", "注册", "首页", "导航", "菜单", "搜索",
    }
    if lowered in exact:
        return True
    prefixes = (
        "skip to ", "skip navigation", "log in ", "sign up ", "reload to refresh",
        "you signed out", "table of contents", "main menu", "copyright ",
        "cookie policy", "privacy policy", "terms of use", "breadcrumb ",
    )
    return lowered.startswith(prefixes) or lowered in _NAVIGATION_MARKERS


def body_query_signal(text: Any, query: Any) -> dict[str, Any]:
    """Measure lightweight query anchors without rejecting short pages."""

    body = str(text or "").casefold()
    raw_query = str(query or "").casefold()
    terms: set[str] = set(re.findall(r"[a-z][a-z0-9_-]{2,}|[\u3400-\u9fff]{2,}", raw_query))
    terms.difference_update(
        {
            "what", "which", "when", "where", "who", "how", "why", "the",
            "official", "page", "open", "read", "summarize", "summary",
            "describe", "description", "content", "main", "key", "project",
            "product", "web", "website", "this",
        }
    )
    matched = sorted(term for term in terms if term in body)
    return {
        "query_term_count": len(terms),
        "query_terms_matched": matched,
        "query_signal": round(len(matched) / max(1, len(terms)), 4),
    }


def _substantive_text(value: Any) -> bool:
    """Accept compact factual prose without admitting navigation labels."""

    text = str(value or "").strip()
    if len(text) >= MIN_SUBSTANTIVE_EVIDENCE_CHARS:
        return True
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9_+./-]*", text))
    sentence_markers = len(re.findall(r"[。！？.!?;；:]", text))
    technical_tokens = re.findall(
        r"--[A-Za-z0-9][\w-]*|"
        r"\$[A-Za-z_][A-Za-z0-9_]*|"
        r"\b[A-Za-z][A-Za-z0-9.-]*(?:_[A-Za-z0-9.-]+)+\b|"
        r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/\S*|"
        r"\b[A-Za-z][A-Za-z0-9_.-]*\s*=",
        text,
    )
    return (
        cjk_chars >= MIN_CJK_EVIDENCE_CHARS
        and sentence_markers >= 1
    ) or (
        latin_words >= MIN_LATIN_EVIDENCE_WORDS
        and sentence_markers >= 1
    ) or (
        len(text) >= 24
        and len(technical_tokens) >= 2
        and sentence_markers >= 1
    )


def _project_markdown_link(match: re.Match[str]) -> str:
    """Keep safe factual links and strip executable/non-web targets."""

    label = str(match.group(1) or "").strip()
    target = str(match.group(2) or "").strip()
    if re.match(r"https?://", target, flags=re.IGNORECASE):
        return f"[{label}]({target})"
    return label


def clean_page_body(value: Any) -> dict[str, Any]:
    """Remove deterministic page chrome while preserving short factual pages.

    This is intentionally a conservative line-level projection.  It is not a
    relevance judge and it must not require a large character count: compact
    encyclopedia entries and API landing pages can be valid evidence with only
    a few substantive sentences.
    """

    raw = str(value or "").replace("\x00", "").replace("\r\n", "\n")
    raw_lines = raw.splitlines()
    output: list[str] = []
    seen: set[str] = set()
    removed_navigation = 0
    removed_images = 0
    removed_link_only = 0

    in_code_fence = False
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            output.append(stripped)
            continue
        if in_code_fence:
            # Exact whitespace, punctuation and line breaks are factual for
            # commands, source code, YAML and configuration examples.
            line = raw_line.rstrip()
            if line or (output and output[-1] != ""):
                output.append(line)
            continue
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        line = _IMAGE_PLACEHOLDER_RE.sub("", line)
        line = _MARKDOWN_IMAGE_RE.sub("", line)
        if not line.strip():
            removed_images += 1
            continue
        projected = _MARKDOWN_LINK_RE.sub(_project_markdown_link, line)
        projected = _SCRIPT_URL_RE.sub(" ", projected)
        projected = _HTML_TAG_RE.sub(" ", projected)
        projected = re.sub(r"\[\s*\]", " ", projected)
        plain = _MARKDOWN_LINK_RE.sub(r"\1", projected)
        plain = _URL_RE.sub("", plain)
        plain = re.sub(r"[`*_#|<>]", " ", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        marker_hit = _navigation_marker_hit(plain)
        # Link density is not a navigation signal: CVE indexes, release lists
        # and repository pages are often composed almost entirely of links.
        short_ui_line = len(plain) <= 180 and (marker_hit or not plain)
        if short_ui_line:
            removed_navigation += len(line)
            removed_link_only += 1
            continue
        clean_line = re.sub(r"\s+", " ", projected).strip()
        key = re.sub(r"\s+", " ", clean_line).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(clean_line)

    text = "\n".join(output).strip()
    raw_chars = len(raw)
    clean_chars = len(text)
    content_lines = len(output)
    navigation_ratio = removed_navigation / max(1, raw_chars)
    # A short factual page is valid when it has at least one real content line;
    # callers can add query-specific relevance checks on top of this result.
    body_eligible = _substantive_text(text) and content_lines >= 1
    return {
        "text": text,
        "raw_chars": raw_chars,
        "clean_chars": clean_chars,
        "content_lines": content_lines,
        "removed_navigation_chars": removed_navigation,
        "removed_navigation_lines": removed_link_only,
        "removed_image_markers": removed_images,
        "navigation_ratio": round(navigation_ratio, 4),
        "body_eligible": body_eligible,
    }


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

    # ``build_evidence_context`` projects a fetched source into a compact
    # metadata record.  That projection intentionally stores the canonical
    # body under ``evidence_text`` instead of pretending it is a new page
    # field.  Read it only when the projection carries our explicit boundary
    # marker; arbitrary model facts must never pass this gate.
    if item.get("evidence_boundary") == "fetched_page_or_structured_record_only":
        projected = str(item.get("evidence_text") or "").replace("\x00", "").replace("\r\n", "\n").strip()
        if projected:
            return projected

    kind = evidence_kind(item)
    if kind in {"page_body", "page_body_legacy"}:
        for key in ("source_excerpt", "page_excerpt", "content"):
            value = str(item.get(key) or "").replace("\x00", "").replace("\r\n", "\n").strip()
            if value:
                # Fetched bodies are cleaned at the fetch boundary.  Re-running
                # the line cleaner here can mutate canonical punctuation and
                # links and makes source-span offsets impossible to audit.
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
    return evidence_kind(item) not in {"discovery", ""} and _substantive_text(text)


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
