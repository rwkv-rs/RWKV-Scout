"""Deterministic citation and source-integrity checks.

The validator does not decide whether a model answer is true.  It checks the
properties that can be established mechanically: URL shape, search-page
leakage, evidence presence, answer linkage, and optional live accessibility.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable
from urllib.parse import urlparse

from utils.network_fetch import NetworkFetchError, fetch_text


_SEARCH_HOSTS = {
    "google.com",
    "bing.com",
    "duckduckgo.com",
    "search.brave.com",
    "search.yahoo.com",
    "baidu.com",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u3400-\u9fff]{2,}")


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").casefold().removeprefix("www.")


def _is_search_page(url: str) -> bool:
    host = _host(url)
    path = (urlparse(url).path or "").casefold()
    return host in _SEARCH_HOSTS or any(host.endswith(f".{item}") for item in _SEARCH_HOSTS) and (
        path in {"", "/", "/search"} or "search" in path
    )


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(value or "")}


def _evidence_text(ref: dict[str, Any]) -> str:
    span = ref.get("source_span")
    if isinstance(span, str):
        return span
    parts = [
        ref.get("evidence_text"),
        ref.get("content"),
        ref.get("page_excerpt"),
        ref.get("snippet"),
        ref.get("abstract"),
    ]
    if isinstance(span, dict):
        parts.extend(str(value) for value in span.values() if isinstance(value, str))
    return " ".join(str(value) for value in parts if value)


def _normalized_url(value: Any) -> str:
    return str(value or "").strip().casefold().rstrip("/")


def _locator(ref: dict[str, Any], evidence_text: str) -> dict[str, Any] | None:
    value = ref.get("evidence_locator") or ref.get("source_span")
    if isinstance(value, dict) and value:
        return value
    if isinstance(value, str) and value.strip():
        return {"type": "source_span", "value": value.strip()}
    # A captured excerpt is itself a locator: it can be searched in the saved
    # page snapshot even when the provider did not return paragraph offsets.
    if evidence_text.strip():
        return {"type": "text_excerpt", "value": evidence_text[:240]}
    return None


def _retrieved_evidence(evidence: Iterable[dict[str, Any]] | None) -> dict[str, str]:
    """Index captured page text so provider citation refs can be checked.

    Search providers commonly return citation metadata separately from the
    result body.  Joining on the normalized URL lets the validator check the
    actual captured source without asking a provider to duplicate large text
    fields in every citation object.
    """
    indexed: dict[str, str] = {}
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        url = _normalized_url(item.get("url"))
        if not url:
            continue
        text = " ".join(
            str(item.get(key) or "")
            for key in ("page_excerpt", "content", "abstract", "snippet")
        ).strip()
        if text and len(text) > len(indexed.get(url, "")):
            indexed[url] = text
    return indexed


def _referenced(answer: str, ref: dict[str, Any], index: int) -> bool:
    lowered = (answer or "").casefold()
    ref_id = str(ref.get("ref_id") or "").casefold()
    url = str(ref.get("url") or "").casefold()
    return bool(
        (ref_id and ref_id in lowered)
        or (url and url in lowered)
        or f"[s{index}]" in lowered
        or f"[source {index}]" in lowered
        or bool(re.search(rf"(?:source\s*[:：]?\s*)?\bs{index}\b", lowered))
        or f"【{index}】" in answer
    )


def validate_citations(
    citations: Iterable[dict[str, Any]] | None,
    *,
    answer: str = "",
    evidence: Iterable[dict[str, Any]] | None = None,
    check_remote: bool = False,
    timeout: int = 8,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_urls: Counter[str] = Counter()
    retrieved_by_url = _retrieved_evidence(evidence)
    for index, raw in enumerate(citations or [], start=1):
        ref = dict(raw) if isinstance(raw, dict) else {"value": str(raw)}
        url = str(ref.get("url") or "").strip()
        parsed = urlparse(url)
        valid_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        normalized = _normalized_url(url)
        seen_urls[normalized] += int(bool(normalized))
        inline_evidence = _evidence_text(ref).strip()
        captured_evidence = retrieved_by_url.get(normalized, "")
        evidence_text = inline_evidence or captured_evidence
        locator = _locator(ref, evidence_text)
        row: dict[str, Any] = {
            "index": index,
            "ref_id": str(ref.get("ref_id") or f"S{index}"),
            "title": str(ref.get("title") or ""),
            "url": url,
            "url_valid": valid_url,
            "search_result_page": bool(valid_url and _is_search_page(url)),
            "duplicate_url": False,
            "referenced_in_answer": _referenced(answer, ref, index),
            "evidence_present": bool(evidence_text),
            "evidence_source": "citation" if inline_evidence else ("retrieved_result" if captured_evidence else "none"),
            "locator_present": bool(locator),
            "locator_type": locator.get("type") if locator else "",
            "accessible": None,
            "support_overlap": None,
            "issues": [],
        }
        if not valid_url:
            row["issues"].append("invalid_url")
        if row["search_result_page"]:
            row["issues"].append("search_result_page")
        if not row["evidence_present"]:
            row["issues"].append("missing_evidence")
        if row["evidence_present"] and not row["locator_present"]:
            row["issues"].append("missing_locator")
        if check_remote and valid_url:
            try:
                body = fetch_text(url, timeout=timeout)
                row["accessible"] = bool(body.strip())
                if not row["accessible"]:
                    row["issues"].append("empty_page")
            except (NetworkFetchError, OSError, ValueError) as exc:
                row["accessible"] = False
                row["issues"].append(f"inaccessible:{type(exc).__name__}")
        body_tokens = _tokens(evidence_text)
        answer_tokens = _tokens(answer)
        if body_tokens and answer_tokens:
            overlap = len(body_tokens & answer_tokens) / max(1, min(len(body_tokens), 12))
            row["support_overlap"] = round(min(1.0, overlap), 4)
        rows.append(row)

    for row in rows:
        normalized = _normalized_url(row["url"])
        row["duplicate_url"] = bool(normalized and seen_urls[normalized] > 1)
        if row["duplicate_url"]:
            row["issues"].append("duplicate_url")
        row["valid"] = not row["issues"]

    def count(predicate) -> int:
        return sum(1 for row in rows if predicate(row))

    return {
        "validator_version": "citation-validator.v1",
        "checked_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "remote_check": check_remote,
        "total": len(rows),
        "valid_url": count(lambda row: row["url_valid"]),
        "accessible": count(lambda row: row["accessible"] is True),
        "referenced": count(lambda row: row["referenced_in_answer"]),
        "evidence_present": count(lambda row: row["evidence_present"]),
        "located": count(lambda row: row["locator_present"]),
        "supported": count(lambda row: (row["support_overlap"] or 0) >= 0.15),
        "invalid": count(lambda row: not row["valid"]),
        "rows": rows,
    }
