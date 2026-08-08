"""Bounded first-party URL discovery for official-source tasks.

Public SERPs are useful discovery hints, but they are not a dependable routing
primitive: HTML endpoints can ignore ``site:`` or return a bot challenge under
concurrency.  This adapter follows a small number of links on the required
official host and returns candidate URLs only.  Page evidence is still fetched,
cleaned, chunked, and admitted by the shared web pipeline.
"""

from __future__ import annotations

import concurrent.futures
import re
import threading
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from typing import Any, Iterable, Mapping
from urllib.parse import urldefrag, unquote, urljoin, urlparse

from utils.concurrency import shutdown_pool, submit_with_context, task_wait_timeout
from utils.network_fetch import NetworkFetchError, fetch_text
from utils.query_constraints import candidate_relevance, explicit_fact_anchors, meaningful_query_terms


_STOP_TERMS = {
    "site", "official", "source", "page", "direct", "exact", "latest", "recent",
    "what", "which", "when", "where", "who", "how", "the", "and", "for", "from",
    "with", "user", "question", "fact", "facts", "answer", "identifier",
}
_SKIP_PATH_MARKERS = (
    "/login", "/logout", "/signup", "/register", "/share", "facebook.com",
    "twitter.com", "linkedin.com", "mailto:", "javascript:",
)
_SITEMAP_CACHE_LOCK = threading.Lock()
_SITEMAP_CACHE: dict[str, tuple[str, ...]] = {}
_SITEMAP_FETCH_LOCKS: dict[str, threading.Lock] = {}


class _LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.rows: list[dict[str, str]] = []
        self._href = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a" or self._href:
            return
        href = str(dict(attrs).get("href") or "").strip()
        if href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "a" or not self._href:
            return
        url = urldefrag(urljoin(self.base_url, self._href))[0]
        title = " ".join(" ".join(self._text).split())
        self.rows.append({"url": url, "title": title})
        self._href = ""
        self._text = []


def _host_matches(url: str, domain: str) -> bool:
    host = (urlparse(url).hostname or "").casefold().removeprefix("www.").rstrip(".")
    expected = str(domain or "").casefold().removeprefix("www.").rstrip(".")
    return bool(host and expected and (host == expected or host.endswith("." + expected)))


def _dominant_sitemap_domains(domain: str, urls: Iterable[str]) -> list[str]:
    """Resolve a canonical host from a sitemap reached through an official host.

    Official projects sometimes redirect an old or singular hostname to a new
    canonical hostname.  Trust only a dominant HTTPS host represented by at
    least two sitemap entries; an isolated external URL is never promoted.
    The result is request-scoped and does not mutate the global authority map.
    """

    original = str(domain or "").casefold().removeprefix("www.").rstrip(".")
    counts: Counter[str] = Counter()
    for value in urls:
        parsed = urlparse(str(value or ""))
        host = (parsed.hostname or "").casefold().removeprefix("www.").rstrip(".")
        if parsed.scheme.casefold() == "https" and host:
            counts[host] += 1
    total = sum(counts.values())
    resolved = [original] if original else []
    for host, count in counts.most_common(2):
        if host in resolved:
            continue
        if count >= 2 and count / max(1, total) >= 0.8:
            resolved.append(host)
    return resolved


def _host_matches_any(url: str, domains: Iterable[str]) -> bool:
    return any(_host_matches(url, domain) for domain in domains)


def _query_terms(query: str, domain: str) -> list[str]:
    return [
        value for value in meaningful_query_terms(query, domain=domain)
        if value not in _STOP_TERMS
    ][:24]


def _requirement_types(values: Iterable[Mapping[str, Any]] | None) -> set[str]:
    return {
        str(value.get("type") or "").casefold()
        for value in values or []
        if isinstance(value, Mapping) and str(value.get("type") or "").strip()
    }


def _score_link(
    row: Mapping[str, Any],
    *,
    terms: list[str],
    requirements: set[str],
    query: str = "",
    domain: str = "",
    constraint_query: str = "",
) -> float:
    url = unquote(str(row.get("url") or "")).casefold()
    title = str(row.get("title") or "").casefold()
    haystack = f"{title} {url}"
    normalized_haystack = re.sub(r"[._]", "-", haystack)
    score = 0.0
    for term in terms:
        variants = {term, re.sub(r"[._]", "-", term), "v" + re.sub(r"[._]", "-", term)}
        if any(variant in title for variant in variants):
            score += 3.0
        elif any(variant in normalized_haystack for variant in variants):
            score += 1.5
    relevance = candidate_relevance(
        row,
        query,
        domain=domain,
        constraint_query=constraint_query or query,
    )
    score += 12.0 * len(relevance.get("anchor_hits") or [])
    if relevance.get("anchor_satisfied"):
        score += 4.0
    if relevance.get("version_conflict"):
        score -= 20.0
    # Task-class path names are tie-breakers.  They must never outweigh a
    # direct entity, identifier, or version match from the RWKV query.
    if "procedure" in requirements and any(marker in haystack for marker in ("howto", "how-to", "guide", "install", "build", "config")):
        score += 1.5
    if "cve_id" in requirements and any(marker in haystack for marker in ("known-exploited", "kev", "cve-", "vulnerabilit", "advisories", "alerts")):
        score += 3.0
    if "date" in requirements and any(marker in haystack for marker in ("release", "news", "announc", "changelog")):
        score += 1.5
    if "version" in requirements and any(marker in haystack for marker in ("release", "version", "download", "changelog")):
        score += 1.5
    if requirements.intersection({"date", "version"}):
        if any(marker in haystack for marker in ("released", "release-notes", "release_notes", "changelog")):
            score += 1.0
        if any(marker in haystack for marker in ("alpha", "beta", "preview", "release-candidate", "release_candidate", "schedule-update", "schedule_update")):
            score -= 3.0
    if "extension" not in terms and "extension" in haystack:
        score -= 3.0
    path = urlparse(str(row.get("url") or "")).path or "/"
    if path in {"", "/"}:
        score -= 3.0
    return score


def _generic_seed_urls(domain: str, query: str, requirements: set[str]) -> list[str]:
    """Build capability-shaped first-party routes for any domain."""

    host = str(domain or "").casefold().removeprefix("www.").rstrip(".")
    canonical_host = host
    seeds = [f"https://{canonical_host}/"]
    version_match = re.search(r"(?<!\d)(\d+\.\d+)(?!\d)", str(query or ""))
    version = version_match.group(1) if version_match else ""
    if "procedure" in requirements and version:
        seeds.extend(
            (
                f"https://{canonical_host}/docs/{version}/",
                f"https://{canonical_host}/{version}/howto/",
                f"https://{canonical_host}/{version}/",
            )
        )
    if requirements.intersection({"version", "date"}):
        for path in ("downloads/", "blog/", "news/", "changelog/", "releases/", "release-notes/"):
            seeds.append(f"https://{canonical_host}/{path}")
        if version:
            seeds.extend(
                (
                    f"https://{canonical_host}/releases/{version}/",
                    f"https://{canonical_host}/release/{version}/",
                    f"https://{canonical_host}/v{version}/",
                )
            )
    if "procedure" in requirements:
        for path in ("guide/", "guides/", "how-to/", "howto/", "documentation/", "docs/"):
            seeds.append(f"https://{canonical_host}/{path}")
    if "cve_id" in requirements:
        for path in ("security/", "security/advisories/", "advisories/", "vulnerabilities/"):
            seeds.append(f"https://{canonical_host}/{path}")
    return list(dict.fromkeys(seeds))


def _seed_urls(
    domain: str,
    query: str,
    requirements: set[str],
) -> list[str]:
    return _generic_seed_urls(domain, query, requirements)


def _sitemap_urls(domain: str) -> tuple[str, ...]:
    """Read one official sitemap index and cache only successful discoveries.

    A timeout or temporary provider failure must not poison every later replan
    in the process.  Per-domain locks provide single-flight behavior under
    case concurrency; only a non-empty URL set enters the shared cache.
    """

    host = str(domain or "").casefold().removeprefix("www.").rstrip(".")
    with _SITEMAP_CACHE_LOCK:
        cached = _SITEMAP_CACHE.get(host)
        fetch_lock = _SITEMAP_FETCH_LOCKS.setdefault(host, threading.Lock())
    if cached:
        return cached

    with fetch_lock:
        with _SITEMAP_CACHE_LOCK:
            cached = _SITEMAP_CACHE.get(host)
        if cached:
            return cached
        discovered = _fetch_sitemap_urls(host)
        if discovered:
            with _SITEMAP_CACHE_LOCK:
                _SITEMAP_CACHE[host] = discovered
        return discovered


def _fetch_sitemap_urls(host: str) -> tuple[str, ...]:
    """Fetch and parse one sitemap attempt without mutating shared state."""

    root = f"https://{host}/sitemap.xml"
    try:
        body = fetch_text(root, timeout=20)
    except (NetworkFetchError, ValueError):
        return ()
    locations = [unescape(value.strip()) for value in re.findall(r"<loc>\s*([^<]+?)\s*</loc>", body, re.I)]
    child_maps = [value for value in locations if value.casefold().endswith(("sitemap.xml", "sitemap.xml.gz"))]
    urls = [value for value in locations if value not in child_maps]
    if child_maps:
        child_maps.sort(
            key=lambda value: (
                0 if re.search(r"/(?:en|en-us)/sitemap\.xml$", value, re.I) else 1,
                len(value),
                value,
            )
        )
        for child in child_maps[:2]:
            try:
                child_body = fetch_text(child, timeout=25)
            except (NetworkFetchError, ValueError):
                continue
            urls.extend(
                unescape(value.strip())
                for value in re.findall(r"<loc>\s*([^<]+?)\s*</loc>", child_body, re.I)
            )
    return tuple(dict.fromkeys(urls))


def _clear_sitemap_cache() -> None:
    with _SITEMAP_CACHE_LOCK:
        _SITEMAP_CACHE.clear()
        _SITEMAP_FETCH_LOCKS.clear()


_sitemap_urls.cache_clear = _clear_sitemap_cache  # type: ignore[attr-defined]


def discover_official_urls(
    query: str,
    required_domains: Iterable[str],
    *,
    answer_requirements: Iterable[Mapping[str, Any]] | None = None,
    max_results: int = 8,
    fetch_budget: int = 5,
    prefer_recent: bool = False,
    constraint_query: str = "",
) -> dict[str, Any]:
    """Return same-host candidates from a bounded, two-hop official crawl."""

    domains = [str(value).casefold().removeprefix("www.").rstrip(".") for value in required_domains if str(value).strip()]
    limit = max(1, min(int(max_results or 8), 8))
    budget = max(1, min(int(fetch_budget or 5), 8))
    requirements = _requirement_types(answer_requirements)
    hard_query = constraint_query or query
    anchors = explicit_fact_anchors(hard_query)
    candidates: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    resolved_domains: list[str] = []
    domain_aliases: list[dict[str, str]] = []

    for domain in domains[:2]:
        terms = _query_terms(query, domain)
        sitemap_urls = _sitemap_urls(domain)
        accepted_domains = _dominant_sitemap_domains(domain, sitemap_urls)
        for accepted in accepted_domains:
            if accepted not in resolved_domains:
                resolved_domains.append(accepted)
            if accepted != domain:
                domain_aliases.append(
                    {
                        "requested_domain": domain,
                        "canonical_domain": accepted,
                        "verification": "dominant_https_sitemap_host",
                    }
                )
        for url in sitemap_urls:
            if not _host_matches_any(url, accepted_domains):
                continue
            row = {"url": url, "title": unquote(urlparse(url).path.replace("/", " "))}
            # A sitemap entry is a verified first-party route.  Keep it ahead
            # of an equally relevant common-path guess that may be a 404.
            relevance = candidate_relevance(
                row,
                query,
                domain=domain,
                constraint_query=hard_query,
            )
            score = _score_link(
                row,
                terms=terms,
                requirements=requirements,
                query=query,
                domain=domain,
                constraint_query=hard_query,
            ) + 1.0
            if score <= 0:
                continue
            candidates[url] = {
                "title": str(row["title"] or url),
                "url": url,
                "snippet": "Official URL discovered from the site's sitemap.",
                "source": "official sitemap adapter",
                "page_excerpt": "",
                "untrusted_content": True,
                "discovery_score": score,
                "query_relevance": relevance,
                "anchor_satisfied": bool(relevance.get("anchor_satisfied")),
                "literal_satisfied": bool(relevance.get("literal_satisfied")),
                "anchor_hit_count": len(relevance.get("anchor_hits") or []),
                "term_hit_count": len(relevance.get("term_hits") or []),
            }
        visited: set[str] = set()
        seed_urls = _seed_urls(
            domain,
            query,
            requirements,
        )
        frontier = [(url, 0) for url in seed_urls]
        for url in seed_urls:
            path = (urlparse(url).path or "/").rstrip("/") or "/"
            if path == "/" or not _host_matches_any(url, accepted_domains):
                continue
            row = {"url": url, "title": unquote(path.replace("/", " "))}
            relevance = candidate_relevance(
                row,
                query,
                domain=domain,
                constraint_query=hard_query,
            )
            score = _score_link(
                row,
                terms=terms,
                requirements=requirements,
                query=query,
                domain=domain,
                constraint_query=hard_query,
            )
            candidates.setdefault(
                url,
                {
                    "title": str(row["title"] or url),
                    "url": url,
                    "snippet": "Deterministic official route for this source class.",
                    "source": "official route adapter",
                    "page_excerpt": "",
                    "untrusted_content": True,
                    "discovery_score": score,
                    "query_relevance": relevance,
                    "anchor_satisfied": bool(relevance.get("anchor_satisfied")),
                    "literal_satisfied": bool(relevance.get("literal_satisfied")),
                    "anchor_hit_count": len(relevance.get("anchor_hits") or []),
                    "term_hit_count": len(relevance.get("term_hits") or []),
                },
            )
        fetched_count = 0
        while frontier and fetched_count < budget:
            batch = []
            while frontier and len(batch) < min(3, budget - fetched_count):
                url, depth = frontier.pop(0)
                if url in visited:
                    continue
                visited.add(url)
                batch.append((url, depth))
            if not batch:
                break

            def fetch(entry: tuple[str, int]) -> tuple[str, int, str, str]:
                url, depth = entry
                try:
                    return url, depth, fetch_text(url, timeout=15), ""
                except (NetworkFetchError, ValueError) as exc:
                    return url, depth, "", f"{type(exc).__name__}: {exc}"

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(batch))
            future_map = {submit_with_context(pool, fetch, entry): entry for entry in batch}
            completed: list[tuple[str, int, str, str]] = []
            cancelled = False
            try:
                for future in concurrent.futures.as_completed(future_map, timeout=task_wait_timeout()):
                    completed.append(future.result())
            except concurrent.futures.TimeoutError:
                cancelled = True
                raise
            finally:
                shutdown_pool(pool, list(future_map), cancelled=cancelled)
            fetched_count += len(batch)

            next_rows: list[tuple[float, str]] = []
            for page_url, depth, html, error in completed:
                if error:
                    errors.append(f"{page_url}: {error}"[:500])
                    continue
                parser = _LinkParser(page_url)
                parser.feed(html)
                for row in parser.rows:
                    url = str(row.get("url") or "")
                    lowered = url.casefold()
                    if not url.startswith(("http://", "https://")) or not _host_matches_any(url, accepted_domains):
                        continue
                    if any(marker in lowered for marker in _SKIP_PATH_MARKERS):
                        continue
                    if re.search(r"(?:^|&)(?:page|offset|sort|filter)=", urlparse(url).query, flags=re.IGNORECASE):
                        continue
                    relevance = candidate_relevance(
                        row,
                        query,
                        domain=domain,
                        constraint_query=hard_query,
                    )
                    score = _score_link(
                        row,
                        terms=terms,
                        requirements=requirements,
                        query=query,
                        domain=domain,
                        constraint_query=hard_query,
                    )
                    existing = candidates.get(url)
                    if existing is None or score > float(existing.get("discovery_score") or 0.0):
                        candidates[url] = {
                            "title": str(row.get("title") or url),
                            "url": url,
                            "snippet": "Official-site link discovered from a first-party page.",
                            "source": "official site adapter",
                            "page_excerpt": "",
                            "untrusted_content": True,
                            "discovery_score": score,
                            "query_relevance": relevance,
                            "anchor_satisfied": bool(relevance.get("anchor_satisfied")),
                            "literal_satisfied": bool(relevance.get("literal_satisfied")),
                            "anchor_hit_count": len(relevance.get("anchor_hits") or []),
                            "term_hit_count": len(relevance.get("term_hits") or []),
                        }
                    if depth < 1 and url not in visited:
                        next_rows.append((score, url))
            next_rows.sort(key=lambda item: (-item[0], item[1]))
            discovered_frontier = []
            for _, url in next_rows[: max(0, budget - fetched_count)]:
                if url not in visited and all(url != queued for queued, _ in frontier):
                    discovered_frontier.append((url, 1))
            # First-party links discovered from a real page are stronger than
            # unverified common-path guesses, so they receive the remaining
            # crawl budget first.
            frontier = discovered_frontier + frontier

    def url_date(row: Mapping[str, Any]) -> tuple[int, int, int]:
        match = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)", str(row.get("url") or ""))
        return tuple(int(value) for value in match.groups()) if match else (0, 0, 0)

    def recent_group(row: Mapping[str, Any]) -> int:
        if not prefer_recent:
            return 0
        if url_date(row) != (0, 0, 0):
            return 0
        return 1

    rows = sorted(
        candidates.values(),
        key=lambda item: (
            recent_group(item),
            *((-value for value in url_date(item)) if prefer_recent else ()),
            -int(bool(item.get("anchor_satisfied"))) if anchors else 0,
            -int(bool(item.get("literal_satisfied"))),
            -int(item.get("anchor_hit_count") or 0),
            -int(item.get("term_hit_count") or 0),
            -float(item.get("discovery_score") or 0.0),
            str(item.get("url") or ""),
        ),
    )[:limit]
    return {
        "status": "ok" if rows else "no_results",
        "real_network": True,
        "provider": "official site adapter",
        "retrieval_role": "discovery",
        "query": query,
        "count": len(rows),
        "results": rows,
        "sources": [row["url"] for row in rows],
        "provider_errors": errors,
        "fetch_count": len(visited) if domains else 0,
        "resolved_domains": resolved_domains or domains,
        "domain_aliases": domain_aliases,
    }


__all__ = ["discover_official_urls"]
