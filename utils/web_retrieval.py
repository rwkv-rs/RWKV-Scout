"""Bounded public-web discovery and evidence extraction for RWKV-ECRA.

This module is deliberately self-contained.  It borrows the useful ideas of
bounded candidate admission, domain pivots, one-hop expansion and an explicit
evidence record, but it does not depend on or copy the rwkv-search package.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urldefrag, urlencode, urljoin, urlparse

from utils.network_fetch import NetworkFetchError, fetch_text


_SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "template"}
_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "msclkid")
_SEARCH_HOSTS = {
    "bing.com",
    "cn.bing.com",
    "duckduckgo.com",
    "html.duckduckgo.com",
}


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self.meta: dict[str, str] = {}
        self._title_depth = 0
        self._skip_depth = 0
        self._link: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.casefold()
        values = {key.casefold(): value or "" for key, value in attrs}
        if name in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if name == "title":
            self._title_depth += 1
        if name == "meta":
            key = values.get("name") or values.get("property") or values.get("itemprop")
            value = values.get("content", "").strip()
            if key and value:
                self.meta[key.casefold()] = value
        if name == "a" and values.get("href"):
            self._link = {"href": values["href"], "text": ""}

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        name = tag.casefold()
        if name in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if name == "title" and self._title_depth:
            self._title_depth -= 1
        if name == "a" and self._link is not None:
            self.links.append(self._link)
            self._link = None

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if not value or self._skip_depth:
            return
        if self._title_depth:
            self.title_parts.append(value)
        if self._link is not None:
            self._link["text"] += f" {value}"
        self.text_parts.append(value)


def normalize_url(value: str, base: str = "") -> str:
    candidate = urljoin(base, (value or "").strip())
    candidate, _ = urldefrag(candidate)
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    query = [
        (key, item)
        for key, values in parse_qs(parsed.query, keep_blank_values=True).items()
        if not key.casefold().startswith(_TRACKING_PREFIXES)
        for item in values
    ]
    path = parsed.path or "/"
    return parsed._replace(query=urlencode(query), path=path).geturl()


def hostname(value: str) -> str:
    return (urlparse(value).hostname or "").casefold().removeprefix("www.")


def registrable_hint(value: str) -> str:
    host = hostname(value)
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _tokens(value: str) -> list[str]:
    latin = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", value or "")
    han = re.findall(r"[\u3400-\u9fff]{2,}", value or "")
    return [item.casefold() for item in [*latin, *han] if len(item) > 1]


def _query_terms(query: str) -> set[str]:
    ignored = {
        "the", "and", "for", "from", "with", "find", "search", "look", "latest",
        "official", "give", "report", "according", "what", "which", "that", "this",
        "截至", "查找", "搜索", "根据", "给出", "报告", "当前", "最新", "是否",
    }
    return {token for token in _tokens(query) if token not in ignored and len(token) >= 2}


def _is_search_host(url: str) -> bool:
    host = hostname(url)
    return host in _SEARCH_HOSTS or any(host.endswith(f".{item}") for item in _SEARCH_HOSTS)


def _looks_like_noise(item: dict[str, Any], query: str) -> bool:
    haystack = " ".join(str(item.get(key) or "") for key in ("title", "snippet", "url")).casefold()
    terms = _query_terms(query)
    if not terms:
        return False
    noise_markers = ("dictionary", "translate", "grammar", "word meaning", "linux find", "array.prototype.find")
    if any(marker in haystack for marker in noise_markers) and not any(term in haystack for term in terms if len(term) > 4):
        return True
    return False


def candidate_score(query: str, item: dict[str, Any]) -> float:
    terms = _query_terms(query)
    title = str(item.get("title") or "").casefold()
    snippet = str(item.get("snippet") or "").casefold()
    url = str(item.get("url") or "").casefold()
    score = 0.0
    for term in terms:
        if term in title:
            score += 4.0
        if term in snippet:
            score += 1.0
        if term in url:
            score += 1.5
    if item.get("source_kind") in {"github", "arxiv", "official"}:
        score += 1.5
    if item.get("body_available"):
        score += 0.5
    return round(score, 4)


def admit_candidates(
    query: str,
    candidates: Iterable[dict[str, Any]],
    *,
    limit: int = 8,
    per_domain: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    domains: dict[str, int] = {}
    for raw in candidates:
        item = dict(raw)
        item["url"] = normalize_url(str(item.get("url") or ""))
        if not item["url"]:
            rejected.append({"url": "", "reason": "invalid_url"})
            continue
        key = item["url"].casefold()
        if key in seen:
            rejected.append({"url": item["url"], "reason": "duplicate"})
            continue
        seen.add(key)
        if _is_search_host(item["url"]):
            rejected.append({"url": item["url"], "reason": "search_result_page"})
            continue
        if _looks_like_noise(item, query):
            rejected.append({"url": item["url"], "reason": "low_relevance_noise"})
            continue
        domain = registrable_hint(item["url"])
        if domains.get(domain, 0) >= per_domain:
            rejected.append({"url": item["url"], "reason": "domain_budget"})
            continue
        item["domain"] = domain
        item["score"] = candidate_score(query, item)
        item["source_kind"] = item.get("source_kind") or infer_source_kind(item["url"], query)
        domains[domain] = domains.get(domain, 0) + 1
        admitted.append(item)
    admitted.sort(key=lambda value: (float(value.get("score") or 0.0), bool(value.get("snippet"))), reverse=True)
    return admitted[: max(1, limit)], rejected


def infer_source_kind(url: str, query: str = "") -> str:
    host = hostname(url)
    if host == "github.com" or host.endswith(".github.com"):
        return "github"
    if "arxiv.org" in host:
        return "arxiv"
    query_lower = (query or "").casefold()
    if any(term in query_lower for term in ("official", "官网", "官方", "release", "文档", "docs", "pep")):
        return "official"
    return "web"


def extract_page(url: str, *, max_chars: int = 14000, timeout: int = 15) -> dict[str, Any]:
    html = fetch_text(url, timeout=timeout)
    parser = _PageParser()
    parser.feed(html[:300_000])
    text = " ".join(parser.text_parts)
    title = " ".join(parser.title_parts).strip()
    meta = parser.meta
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for link in parser.links:
        target = normalize_url(link.get("href", ""), url)
        if not target or target in seen:
            continue
        seen.add(target)
        links.append({"url": target, "text": " ".join(link.get("text", "").split())[:180]})
    body = text[:max_chars]
    published = next((meta.get(key) for key in ("article:published_time", "datepublished", "publishdate", "date") if meta.get(key)), "")
    modified = next((meta.get(key) for key in ("article:modified_time", "datemodified", "last-modified") if meta.get(key)), "")
    return {
        "title": title,
        "body": body,
        "body_chars": len(body),
        "links": links,
        "description": meta.get("description", ""),
        "published": published,
        "modified": modified,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_quality": round(min(1.0, len(body) / 1800.0), 4),
    }


def attach_page(item: dict[str, Any], *, timeout: int = 15) -> dict[str, Any]:
    value = dict(item)
    try:
        page = extract_page(value["url"], timeout=timeout)
        value["page_excerpt"] = page["body"]
        value["content"] = page["body"]
        value["body_available"] = bool(page["body"])
        value["body_chars"] = page["body_chars"]
        value["page_title"] = page["title"]
        value["links"] = page["links"]
        value["published"] = page["published"] or value.get("published", "")
        value["updated"] = page["modified"] or value.get("updated", "")
        value["captured_at"] = page["captured_at"]
        value["content_quality"] = page["content_quality"]
        value["score"] = round(float(value.get("score") or 0.0) + (0.5 if value["body_available"] else 0.0), 4)
    except (NetworkFetchError, ValueError, OSError) as exc:
        value["body_available"] = False
        value["body_error"] = str(exc)[-300:]
        value.setdefault("page_excerpt", "")
        value.setdefault("content", "")
    return value


def select_pivot_domains(query: str, candidates: Iterable[dict[str, Any]], *, limit: int = 2) -> list[str]:
    precision_terms = ("official", "官网", "官方", "论文", "paper", "arxiv", "github", "release", "文档", "docs")
    if not any(term in (query or "").casefold() for term in precision_terms):
        return []
    domains: list[str] = []
    for item in candidates:
        domain = registrable_hint(str(item.get("url") or ""))
        if domain and domain not in domains and domain not in _SEARCH_HOSTS:
            domains.append(domain)
        if len(domains) >= limit:
            break
    return domains


def select_one_hop_links(
    query: str,
    pages: Iterable[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in pages:
        base = str(page.get("url") or "")
        base_domain = registrable_hint(base)
        for link in page.get("links") or []:
            url = normalize_url(str(link.get("url") or ""))
            if not url or url in seen or registrable_hint(url) != base_domain:
                continue
            text = str(link.get("text") or "")
            if not text or url.rstrip("/") == base.rstrip("/"):
                continue
            item = {
                "title": text,
                "url": url,
                "snippet": text,
                "source": "same_site_one_hop",
                "retrieval_stage": "one_hop",
            }
            relevance = sum(1 for term in terms if term in f"{text} {url}".casefold())
            if relevance == 0 and len(candidates) >= limit:
                continue
            item["score"] = float(relevance) + 0.25
            candidates.append(item)
            seen.add(url)
    candidates.sort(key=lambda value: float(value.get("score") or 0.0), reverse=True)
    return candidates[:limit]

