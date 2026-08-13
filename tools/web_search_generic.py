"""Provider-agnostic web retrieval for the model-owned search episode.

The model chooses whether to call ``web_search`` and supplies the query.  The
tool owns only the mechanical retrieval transaction: discovery, candidate
admission, page fetching, Markdown extraction, chunking and evidence merging.
It deliberately does not produce an answer or decide whether the user's goal
is complete.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import threading
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from agent.page_evidence import build_page_chunks, extract_single_page_evidence, select_grounded_source_chunks
from agent.task_plan_contract import point_question, task_points
from clients.llm_client import LLMClient
from config import DATA_PIPELINE, get_llm_context_length
from tools.registry import ToolRegistry
from tools.web_search_keyless import _is_search_result_url, search_web_keyless
from tools.official_site_discovery import discover_official_urls
from tools.web_search_tavily import search_web_tavily
from utils.task_events import append_task_event
from utils.evidence_quality import (
    MAX_SHORT_BODY_FALLBACK_CHARS,
    body_query_signal,
    clean_page_body,
    has_substantive_evidence,
)
from utils.concurrency import shutdown_pool, submit_with_context, task_wait_timeout
from utils.source_authority import (
    annotate_source,
    explicit_domains,
    infer_candidate_authority_domains,
    required_domains_for_task_point,
    resolve_source_policy,
)
from utils.web_retrieval import candidate_score, normalize_url, retrieval_url_identity
from utils.freshness import annotate_freshness, build_freshness_policy
from utils.query_constraints import candidate_relevance, meaningful_query_terms
from utils.retrieval_ranking import rrf_fuse


_HOST_FETCH_GATE_LOCK = threading.Lock()
_HOST_FETCH_GATES: dict[tuple[str, int], threading.BoundedSemaphore] = {}


def _host_fetch_gate(url: Any) -> threading.BoundedSemaphore:
    """Limit same-origin pressure without reducing cross-origin concurrency."""

    host = _host(str(url or "")) or "unknown"
    per_host = _pipeline_concurrency("web_page_fetch_per_host_concurrency", 2, maximum=8)
    key = (host, per_host)
    with _HOST_FETCH_GATE_LOCK:
        gate = _HOST_FETCH_GATES.get(key)
        if gate is None:
            gate = threading.BoundedSemaphore(per_host)
            _HOST_FETCH_GATES[key] = gate
        return gate


def _transient_fetch_error(result: dict[str, Any]) -> bool:
    if str(result.get("status") or "").casefold() != "error":
        return False
    message = str(result.get("message") or " ".join(result.get("provider_errors") or [])).casefold()
    return bool(
        re.search(
            r"ssl|eof|timed?\s*out|timeout|connection|temporar|reset|remote end|"
            r"too many requests|\b429\b|\b502\b|\b503\b|\b504\b",
            message,
        )
    )


def _parse_result(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"status": "error", "message": value[:500], "results": []}
        return parsed if isinstance(parsed, dict) else {"status": "error", "results": []}
    return {"status": "error", "results": []}


def _pipeline_concurrency(name: str, default: int, *, maximum: int = 64) -> int:
    try:
        return max(1, min(int(DATA_PIPELINE.get(name, default) or default), maximum))
    except (TypeError, ValueError):
        return default


def _shared_source_urls(agent_state: Any) -> set[str]:
    retrieval = getattr(agent_state, "retrieval", None)
    sources = getattr(retrieval, "sources", {}) if retrieval is not None else {}
    attempted = getattr(retrieval, "attempted_urls", set()) if retrieval is not None else set()
    attempt_counts = getattr(retrieval, "url_attempt_counts", {}) if retrieval is not None else {}
    accepted = {
        retrieval_url_identity(str(item.get("url") or ""))
        for item in (sources.values() if isinstance(sources, dict) else [])
        if isinstance(item, dict) and str(item.get("url") or "").strip()
    }
    try:
        failed_retry_limit = max(
            1,
            min(int(DATA_PIPELINE.get("web_failed_url_retry_limit", 2) or 2), 3),
        )
    except (TypeError, ValueError):
        failed_retry_limit = 2
    frozen_failures = {
        retrieval_url_identity(str(url or ""))
        for url in (attempted if isinstance(attempted, (set, list, tuple)) else [])
        if str(url or "").strip()
        and int((attempt_counts or {}).get(url, 0) or 0) >= failed_retry_limit
    }
    return accepted | frozen_failures


def _host(url: str) -> str:
    match = re.match(r"https?://([^/]+)", str(url or "").strip(), re.I)
    return (match.group(1) if match else "").casefold().removeprefix("www.")


def _extract_direct_url(query: str) -> str:
    """Return a complete URL query as a direct-fetch target.

    A URL supplied as the whole model-selected query is an instruction to
    retrieve that page, not a search phrase. Natural-language queries that
    merely mention a URL continue through provider discovery.
    """
    value = str(query or "").strip().strip("<>[]()")
    if any(char.isspace() for char in value):
        return ""
    parsed = urlparse(value)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return ""
    return normalize_url(value.rstrip(".,;:!?"))


def _extract_single_goal_url(goal: str) -> str:
    """Recover one URL explicitly supplied in the original user goal."""
    candidates = re.findall(r"https?://[^\s<>\[\]()\"']+", str(goal or ""), flags=re.I)
    normalized = [_extract_direct_url(item) for item in candidates]
    normalized = [item for item in normalized if item]
    return normalized[0] if len(normalized) == 1 else ""


_UNSUPPORTED_DOWNLOAD_SUFFIXES = {
    ".7z",
    ".doc",
    ".docx",
    ".gz",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".tar",
    ".xls",
    ".xlsx",
    ".zip",
    ".pdf",
}


def _is_unsupported_download_url(value: str) -> bool:
    """Return whether the URL is a binary download for this HTML pipeline.

    A landing page may link to a report PDF, but the generic search tool only
    admits pages that its HTML/Markdown fetcher can turn into evidence.  Keep
    this check at candidate admission so binary downloads do not consume the
    page-fetch budget or masquerade as short/empty evidence.
    """

    path = urlparse(str(value or "")).path.casefold().rstrip("/")
    return any(path.endswith(suffix) for suffix in _UNSUPPORTED_DOWNLOAD_SUFFIXES)


def _direct_candidate(url: str) -> dict[str, Any]:
    return {
        "url": url,
        "title": url,
        "snippet": "direct URL requested by the model",
        "source": "direct_url",
        "candidate_score": 100.0,
        "candidate_rank": 1,
        "discovery_providers": ["direct_url"],
    }


def _dated_url_rank(value: Any) -> tuple[int, int, int]:
    """Extract a publication date encoded in a conventional URL path."""

    path = urlparse(str(value or "")).path
    match = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)", path)
    if not match:
        return (0, 0, 0)
    year, month, day = (int(part) for part in match.groups())
    try:
        datetime(year, month, day)
    except ValueError:
        return (0, 0, 0)
    return (year, month, day)


def _is_community_detail_url(value: Any) -> bool:
    parsed = urlparse(str(value or ""))
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    path = parsed.path.casefold()
    if host == "v2ex.com" or host.endswith(".v2ex.com"):
        return bool(re.fullmatch(r"/t/\d+/?", path))
    if host == "reddit.com" or host.endswith(".reddit.com"):
        return "/comments/" in path
    if host == "stackoverflow.com" or host.endswith(".stackoverflow.com"):
        return bool(re.match(r"/questions/\d+", path))
    return False


def _official_cross_script_relevance(
    row: dict[str, Any],
    relevance: dict[str, Any],
    *,
    query: str,
    constraint_query: str,
    source_policy: dict[str, Any],
) -> dict[str, Any]:
    """Admit one strong sitemap topic across a script boundary for fetching.

    An English official URL cannot lexically contain the Chinese procedural
    half of a mixed-language query. A verified sitemap route with one exact
    Latin topic token is therefore enough to fetch the page, but never enough
    by itself to support a Claim. The fetched body must still pass RWKV span
    extraction, source authority, Claim binding, and relation validation.
    """

    result = dict(relevance or {})
    if result.get("related") is True or not source_policy.get("required"):
        return result
    if str(row.get("source") or "").casefold() != "official sitemap adapter":
        return result
    if not bool((row.get("authority") or {}).get("satisfied")):
        return result
    hard_query = str(constraint_query or query or "")
    locator = " ".join(str(row.get(key) or "") for key in ("title", "url"))
    if not re.search(r"[\u3400-\u9fff]{2,}", hard_query):
        return result
    if not re.search(r"[A-Za-z]{3,}", locator):
        return result
    strong_hits = [
        str(value).casefold()
        for value in result.get("term_hits") or []
        if re.fullmatch(r"[a-z][a-z0-9+_.-]{2,}", str(value).casefold())
    ]
    if not strong_hits:
        return result
    if result.get("version_conflict") or result.get("literal_satisfied") is False:
        return result
    if result.get("anchors") and result.get("anchor_satisfied") is not True:
        return result
    result.update(
        {
            "related": True,
            "raw_threshold_satisfied": True,
            "original_required_term_hits": result.get("required_term_hits"),
            "required_term_hits": 1,
            "effective_required_term_hits": 1,
            "admission_basis": "official_sitemap_cross_script_topic",
            "cross_script_topic_hits": strong_hits,
        }
    )
    return result


def _merge_candidates(
    query: str,
    provider_results: list[dict[str, Any]],
    limit: int = 8,
    *,
    task_plan: dict[str, Any] | None = None,
    constraint_query: str = "",
    policy_query: str = "",
) -> list[dict[str, Any]]:
    """Deduplicate and rank candidates without classifying the task domain."""

    # Search-query text is RWKV-owned strategy, not user policy.  A domain
    # inserted into a follow-up query must therefore remain a retrieval hint
    # instead of silently becoming a hard source requirement.
    authority_query = str(policy_query or query)
    source_policy = resolve_source_policy(
        authority_query,
        {"task_plan": task_plan or {}},
    )
    merged: dict[str, dict[str, Any]] = {}
    for result in provider_results:
        for item in result.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = normalize_url(str(item.get("url") or ""))
            if not url or _is_search_result_url(url):
                continue
            if _is_unsupported_download_url(url):
                continue
            title = " ".join(str(item.get("title") or "").split())
            snippet = " ".join(str(item.get("snippet") or "").split())
            if not title and not snippet:
                continue
            row = dict(item)
            row.update(
                {
                    "url": url,
                    "title": title or url,
                    "snippet": snippet[:1000],
                    "source": str(item.get("source") or result.get("provider") or "web"),
                }
            )
            row = annotate_source(
                row,
                authority_query,
                {"task_plan": task_plan or {}},
            )
            relevance = candidate_relevance(
                row,
                query,
                domain=str((source_policy.get("required_domains") or [""])[0]),
                constraint_query=constraint_query or query,
            )
            relevance = _official_cross_script_relevance(
                row,
                relevance,
                query=query,
                constraint_query=constraint_query,
                source_policy=source_policy,
            )
            # Discovery relevance is a soft ranking feature.  Search snippets
            # are frequently sparse, translated, or generated from navigation
            # text; rejecting here can discard the useful page before its body
            # is fetched.  Only protocol/safety exclusions above are hard.
            row["query_relevance"] = relevance
            row["candidate_score"] = candidate_score(query, row)
            row["discovery_score"] = float(item.get("discovery_score") or 0.0)
            row["discovery_providers"] = list(
                dict.fromkeys(
                    [
                        *[
                            str(value)
                            for value in item.get("discovery_providers") or []
                            if str(value).strip()
                        ],
                        row["source"],
                    ]
                )
            )
            identity = retrieval_url_identity(url)
            existing = merged.get(identity)
            if existing is None:
                merged[identity] = row
                continue
            existing["candidate_score"] = max(
                float(existing.get("candidate_score") or 0),
                float(row.get("candidate_score") or 0),
            )
            existing["discovery_score"] = max(
                float(existing.get("discovery_score") or 0),
                float(row.get("discovery_score") or 0),
            )
            sources = list(existing.get("discovery_providers") or [])
            source = str(row.get("source") or "")
            if source and source not in sources:
                sources.append(source)
            existing["discovery_providers"] = sources
            if len(row.get("snippet") or "") > len(existing.get("snippet") or ""):
                existing["snippet"] = row["snippet"]

    # A URL can be discovered first through a sparse official sitemap row and
    # later through a search provider with a much richer snippet.  Relevance
    # belongs to the final merged candidate, not whichever provider happened
    # to arrive first.  Recompute it after deduplication so provider order
    # cannot leave a canonical page carrying stale term hits.
    required_domain = str((source_policy.get("required_domains") or [""])[0])
    for item in merged.values():
        relevance = candidate_relevance(
            item,
            query,
            domain=required_domain,
            constraint_query=constraint_query or query,
        )
        item["query_relevance"] = _official_cross_script_relevance(
            item,
            relevance,
            query=query,
            constraint_query=constraint_query,
            source_policy=source_policy,
        )
        item["candidate_score"] = candidate_score(query, item)

    generic_ranking_terms = {
        "document", "documentation", "docs", "example", "examples", "guide",
        "information", "official", "page", "reference", "result", "source",
    }
    constraint_topic_terms = {
        term
        for term in meaningful_query_terms(
            constraint_query or query,
            domain=required_domain,
        )
        if re.fullmatch(r"[a-z][a-z0-9+_.-]*", term)
        and term not in generic_ranking_terms
    }
    topic_document_frequency: dict[str, int] = {
        term: sum(
            term
            in " ".join(
                str(item.get(key) or "")
                for key in ("title", "snippet", "url")
            ).casefold()
            for item in merged.values()
        )
        for term in constraint_topic_terms
    }

    route_query = str(constraint_query or query or "").casefold()
    release_record_requested = bool(
        re.search(
            r"\b(?:release|released|version|changelog|current|latest|stable)\b|"
            r"(?:发布|发行|版本|更新日志|当前|最新|稳定)",
            route_query,
            flags=re.IGNORECASE,
        )
    )
    documentation_requested = bool(
        re.search(
            r"\b(?:api\s+)?(?:docs?|documentation|reference)\b|"
            r"(?:api|接口)\s*文档|官方文档|文档",
            route_query,
            flags=re.IGNORECASE,
        )
    )

    def canonical_path_rank(value: Any) -> int:
        parsed = urlparse(str(value or ""))
        host = parsed.netloc.casefold().removeprefix("www.")
        path = parsed.path.casefold()
        if documentation_requested:
            if (
                host.startswith(("docs.", "documentation.", "developer."))
                or re.search(r"/(?:api|docs?|documentation|reference)(?:/|$)", path)
            ):
                return 3
            if re.search(
                r"/(?:blog|news|announcements?|migrations?|releases?)(?:/|$)",
                path,
            ):
                return 0
        if release_record_requested:
            if re.search(
                r"/(?:releases?|release-notes?|release_notes|changelog|downloads?)(?:/|$)",
                path,
            ):
                return 3
            if re.search(
                r"/(?:advanced|tutorial|examples?|community)(?:/|$)",
                path,
            ):
                return 0
        if re.search(r"/(?:current|latest|stable)(?:/|$)", path):
            return 2
        if re.search(r"/(?:docs?|documentation)/v?\d+(?:\.\d+){0,2}(?:/|$)", path):
            return 0
        return 1

    for item in merged.values():
        locator = " ".join(
            str(item.get(key) or "")
            for key in ("title", "snippet", "url")
        ).casefold()
        topic_hits = sorted(term for term in constraint_topic_terms if term in locator)
        topic_score = sum(
            1.0 / max(1, topic_document_frequency.get(term, 1))
            for term in topic_hits
        )
        relevance = dict(item.get("query_relevance") or {})
        relevance.update(
            {
                "constraint_topic_terms": sorted(constraint_topic_terms),
                "constraint_topic_hits": topic_hits,
                "constraint_topic_score": round(topic_score, 6),
                "canonical_path_rank": canonical_path_rank(item.get("url")),
            }
        )
        item["query_relevance"] = relevance

    latest_list = _requests_recency(constraint_query or query)
    community_required = str(source_policy.get("mode") or "").casefold() == "community_required"
    def common_relevance_rank(item: dict[str, Any]) -> tuple[int, int, float]:
        """Compare providers on one query-derived scale.

        Provider-private scores are not calibrated against each other.  The
        official sitemap adapter's discovery score, for example, must not beat
        a search result that actually contains the requested subtopic.  Keep
        discovery permissive, but rank first by the raw topic threshold that
        every provider candidate receives from ``candidate_relevance``.
        """

        relevance = item.get("query_relevance") or {}
        hits = {
            str(value or "").casefold().strip()
            for value in relevance.get("term_hits") or []
            if str(value or "").strip()
        }
        try:
            required = max(0, int(relevance.get("required_term_hits") or 0))
        except (TypeError, ValueError):
            required = 0
        raw_satisfied = bool(
            relevance.get("raw_threshold_satisfied")
            if "raw_threshold_satisfied" in relevance
            else len(hits) >= required
        )
        try:
            relevance_score = float(relevance.get("score") or 0.0)
        except (TypeError, ValueError):
            relevance_score = 0.0
        return int(raw_satisfied), len(hits), relevance_score

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            int(bool((item.get("authority") or {}).get("satisfied"))),
            int((item.get("authority") or {}).get("rank") or 0),
            int(_is_community_detail_url(item.get("url"))) if community_required else 0,
            int(_is_community_detail_url(item.get("url"))),
            common_relevance_rank(item)[0],
            int(bool((item.get("query_relevance") or {}).get("anchor_satisfied"))),
            int(bool((item.get("query_relevance") or {}).get("constraint_topic_hits"))),
            int(bool(_dated_url_rank(item.get("url")))) if latest_list else 0,
            _dated_url_rank(item.get("url")) if latest_list else (0, 0, 0),
            int((item.get("query_relevance") or {}).get("canonical_path_rank") or 0),
            float((item.get("query_relevance") or {}).get("constraint_topic_score") or 0.0),
            len((item.get("query_relevance") or {}).get("constraint_topic_hits") or []),
            common_relevance_rank(item)[1],
            common_relevance_rank(item)[2],
            # If no common topical threshold was met, retain the adapter's
            # own ordering as a bounded fallback (for example release notes
            # ahead of a generic guide in the same official sitemap).
            float(item.get("discovery_score") or 0)
            if source_policy.get("required") and not common_relevance_rank(item)[0]
            else 0.0,
            float(item.get("candidate_score") or 0),
            len(item.get("discovery_providers") or []),
            # Provider-specific scores have no common calibration and are
            # therefore only a final tie-break within otherwise equal rows.
            float(item.get("discovery_score") or 0) if source_policy.get("required") else 0.0,
        ),
        reverse=True,
    )
    # For an official-source task, do not spend the fetch budget on unrelated
    # third-party pages when an admitted required-domain candidate exists.
    required_candidates = [
        item for item in ranked
        if (item.get("authority") or {}).get("satisfied")
    ]
    if source_policy.get("required") and required_candidates:
        ranked = required_candidates + [
            item for item in ranked if item not in required_candidates
        ]
    if community_required:
        detail_candidates = [item for item in ranked if _is_community_detail_url(item.get("url"))]
        if detail_candidates:
            ranked = detail_candidates

    # Keep a bounded amount of domain diversity.  This is a fetch-budget
    # boundary, not a source/domain policy for the user's question.
    selected: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    per_domain_cap = max(1, int(limit)) if source_policy.get("required") else 3
    for item in ranked:
        domain = _host(item.get("url", ""))
        if domain_counts.get(domain, 0) >= per_domain_cap:
            continue
        item["candidate_rank"] = len(selected) + 1
        selected.append(item)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def _fetch_candidate(candidate: dict[str, Any], task_id: str) -> dict[str, Any]:
    from tools.web_search_keyless import fetch_web_url

    try:
        page_max_chars = max(
            20_000,
            min(int(DATA_PIPELINE.get("web_page_max_chars", 120_000) or 120_000), 500_000),
        )
    except (TypeError, ValueError):
        page_max_chars = 120_000
    try:
        attempts = max(
            1,
            min(int(DATA_PIPELINE.get("web_page_fetch_attempts", 2) or 2), 3),
        )
    except (TypeError, ValueError):
        attempts = 2
    retry_errors: list[str] = []
    result: dict[str, Any] = {"status": "error", "message": "page fetch did not run", "results": []}
    with _host_fetch_gate(candidate["url"]):
        for attempt in range(1, attempts + 1):
            raw = fetch_web_url(
                candidate["url"],
                max_chars=page_max_chars,
                task_id=task_id,
                agentic_tool_loop=True,
            )
            result = _parse_result(raw)
            result["fetch_attempt_count"] = attempt
            if not _transient_fetch_error(result) or attempt >= attempts:
                break
            retry_errors.append(str(result.get("message") or "transient page fetch error")[:500])
            time.sleep(0.2 * attempt)
    if retry_errors:
        result["fetch_retry_errors"] = retry_errors
    return result


def _has_usable_fetched_page(result: dict[str, Any]) -> bool:
    pages = [item for item in result.get("results") or [] if isinstance(item, dict)]
    if len(pages) != 1:
        return False
    page = pages[0]
    text = str(page.get("page_excerpt") or page.get("content") or "").strip()
    quality = page.get("body_quality") if isinstance(page.get("body_quality"), dict) else {}
    return bool(text and (page.get("body_verified") or quality.get("body_eligible")))


def _resolve_failed_page_fetches(
    query: str,
    fetched: list[tuple[dict[str, Any], dict[str, Any]]],
    task_id: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Try optional extractors for exact URLs whose primary fetch had no body.

    The boundary is capability-based rather than provider-based. A deployment
    with no extractor plugin keeps the primary fetch result unchanged, and a
    failed plugin never turns a retrieval miss into an engineering failure.
    """

    try:
        fallback_limit = max(
            1,
            min(
                int(DATA_PIPELINE.get("web_page_content_fallback_max_urls", 4) or 4),
                20,
            ),
        )
    except (TypeError, ValueError):
        fallback_limit = 4

    unresolved = [
        index
        for index, (_, result) in enumerate(fetched)
        if not _has_usable_fetched_page(result)
    ][:fallback_limit]
    adapters = ToolRegistry.capability_names("page_content_extract", phase="ALL")
    if not unresolved or not adapters:
        return fetched

    resolved = list(fetched)
    attempts: list[dict[str, Any]] = []
    remaining = set(unresolved)
    for adapter in adapters:
        requested_urls = [
            str(resolved[index][0].get("url") or "").strip()
            for index in sorted(remaining)
            if str(resolved[index][0].get("url") or "").strip()
        ]
        if not requested_urls:
            break
        payload = _parse_result(
            ToolRegistry.execute(
                adapter,
                {"urls": requested_urls, "query": query},
                {"task_id": task_id},
                phase="ALL",
            )
        )
        pages_by_url = {
            retrieval_url_identity(str(page.get("url") or "")): page
            for page in payload.get("results") or []
            if isinstance(page, dict)
            and retrieval_url_identity(str(page.get("url") or ""))
            and _has_usable_fetched_page({"results": [page]})
        }
        recovered: list[str] = []
        for index in sorted(remaining):
            candidate, primary_result = resolved[index]
            normalized = retrieval_url_identity(str(candidate.get("url") or ""))
            page = pages_by_url.get(normalized)
            if page is None:
                continue
            page = dict(page)
            page.setdefault("content_resolver", str(payload.get("provider") or adapter))
            page.setdefault("content_transport", "provider_extract")
            resolved[index] = (
                candidate,
                {
                    "status": "ok",
                    "real_network": True,
                    "provider": str(payload.get("provider") or adapter),
                    "query": str(candidate.get("url") or ""),
                    "results": [page],
                    "sources": [str(page.get("url") or candidate.get("url") or "")],
                    "primary_fetch": {
                        "status": str(primary_result.get("status") or "error"),
                        "message": str(primary_result.get("message") or "")[:500],
                    },
                    "content_resolver": adapter,
                },
            )
            remaining.discard(index)
            recovered.append(str(candidate.get("url") or ""))
        attempts.append(
            {
                "adapter": adapter,
                "status": str(payload.get("status") or "error"),
                "requested_count": len(requested_urls),
                "recovered_count": len(recovered),
                "provider_errors": [
                    str(value)[:500] for value in payload.get("provider_errors") or []
                ],
            }
        )
        if not remaining:
            break

    for index in sorted(remaining):
        candidate, result = resolved[index]
        result = dict(result)
        result["content_resolver_attempts"] = attempts
        resolved[index] = (candidate, result)
    append_task_event(
        task_id,
        "web_search_stage",
        phase="EXTRACTION",
        action="web_search",
        stage="page_content_resolution",
        query=query,
        unresolved_count=len(unresolved),
        recovered_count=len(unresolved) - len(remaining),
        attempts=attempts,
    )
    return resolved


def _claim_evidence_focus(
    query: str,
    task_plan: dict[str, Any] | None,
    task_point_id: str,
) -> str:
    """Bind the unchanged extractor prompt to the active atomic claim."""

    point_id = str(task_point_id or "").strip()
    if not point_id:
        return str(query or "").strip()
    plan = task_plan if isinstance(task_plan, dict) else {}
    point = next(
        (
            item
            for item in task_points(plan)
            if isinstance(item, dict) and str(item.get("id") or "").strip() == point_id
        ),
        None,
    )
    if not isinstance(point, dict):
        return str(query or "").strip()
    return (point_question(point) or str(query or "").strip())[:1600]


def _requests_recency(value: Any) -> bool:
    """Detect an explicit recency request for attention ordering only."""

    return bool(
        re.search(
            r"(?:\bcurrent\b|\blatest\b|\bnewest\b|\brecent\b|\btoday\b|"
            r"当前|最新|目前|现行|截至)",
            str(value or ""),
            flags=re.IGNORECASE,
        )
    )


def _cached_page_result(record: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct one cleaned page from the task-scoped chunk cache."""

    chunks = [
        item
        for item in record.get("source_chunks") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    chunks.sort(key=lambda item: int(item.get("index") or 0))
    seen_chunk_ids: set[str] = set()
    text_parts: list[str] = []
    for index, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("chunk_id") or f"chunk-{index + 1}")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        text_parts.append(str(chunk.get("text") or "").strip())
    page_text = "\n\n".join(text_parts).strip()
    if not page_text:
        page_text = str(
            record.get("page_excerpt")
            or record.get("source_excerpt")
            or record.get("content")
            or ""
        ).strip()
    quality = dict(record.get("body_quality") or {})
    quality.update(
        {
            "text": page_text,
            "clean_chars": len(page_text),
            "raw_chars": max(len(page_text), int(quality.get("raw_chars") or 0)),
            "body_eligible": bool(page_text),
        }
    )
    return {
        "status": "ok" if page_text else "no_evidence",
        "results": [
            {
                "url": str(record.get("url") or ""),
                "title": str(record.get("title") or record.get("url") or ""),
                "content": page_text,
                "page_excerpt": page_text,
                "body_cleaned": True,
                "body_verified": bool(page_text),
                "body_quality": quality,
                "raw_page_chars": int(quality.get("raw_chars") or len(page_text)),
            }
        ] if page_text else [],
    }


def _compact_page(
    query: str,
    candidate: dict[str, Any],
    fetched: dict[str, Any],
    llm: LLMClient,
    task_id: str,
    task_plan: dict[str, Any] | None = None,
    task_point_id: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    pages = [item for item in fetched.get("results") or [] if isinstance(item, dict)]
    if len(pages) != 1:
        return None, {
            "url": candidate["url"],
            "status": str(fetched.get("status") or "error"),
            "error": str(fetched.get("message") or "page fetch did not return one page"),
            "chunk_count": 0,
        }

    page = pages[0]
    url = str(page.get("url") or candidate["url"])
    evidence_focus = _claim_evidence_focus(query, task_plan, task_point_id)
    raw_page_text = str(page.get("page_excerpt") or page.get("content") or "").strip()
    if page.get("body_cleaned") is True:
        page_text = raw_page_text
        page_quality = dict(page.get("body_quality") or {})
        page_quality.setdefault("text", page_text)
        page_quality.setdefault("clean_chars", len(page_text))
        page_quality.setdefault("raw_chars", int(page.get("raw_page_chars") or len(page_text)))
        page_quality.setdefault("body_eligible", bool(page.get("body_verified")))
    else:
        page_quality = clean_page_body(raw_page_text)
        page_text = str(page_quality.get("text") or "").strip()
    query_signal = body_query_signal(page_text, evidence_focus)
    if not page_quality.get("body_eligible"):
        return None, {
            "url": url,
            "title": str(page.get("title") or candidate.get("title") or url),
            "status": "no_evidence",
            "page_chars": len(page_text),
            "raw_page_chars": len(raw_page_text),
            "body_quality": {**page_quality, **query_signal},
            "chunk_count": 0,
            "candidate_count": 0,
            "errors": ["cleaned page body is below the substantive evidence threshold"],
        }

    def on_chunk(*, chunk: dict[str, Any], prompt: str, candidate: dict[str, Any], task_id: str) -> None:
        append_task_event(
            task_id,
            "web_search_chunk",
            phase="EXTRACTION",
            action="web_search",
            url=url,
            chunk=chunk,
            prompt=prompt,
            candidate=candidate,
        )

    try:
        evidence = extract_single_page_evidence(
            query=evidence_focus,
            page=page,
            llm=llm,
            task_id=task_id,
            on_chunk=on_chunk,
            task_plan=task_plan,
        )
    except Exception as exc:
        evidence = {
            "status": "error",
            "error_class": "chunk_extraction_failed",
            "url": url,
            "title": page.get("title") or url,
            "page_chars": len(page_text),
            "raw_page_chars": len(raw_page_text),
            "body_quality": {**page_quality, **query_signal},
            "chunk_count": 0,
            "candidates": [],
            "compact_facts": "",
            "extraction_degraded": True,
            "invalid_response_count": 1,
            "unresolved_chunk_indexes": [0],
            "transport_error_count": 1,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }

    post_gate_candidates = [
        dict(item)
        for item in evidence.get("chunk_candidates") or []
        if isinstance(item, dict)
    ]
    merged_candidates = [
        dict(item)
        for item in evidence.get("candidates") or []
        if isinstance(item, dict)
    ]
    append_task_event(
        task_id,
        "page_candidate_merge",
        phase="EXTRACTION",
        action="web_search",
        url=url,
        data={
            "url": url,
            "page_chars": int(evidence.get("page_chars") or len(page_text)),
            "chunk_count": int(evidence.get("chunk_count") or 0),
            "inspected_chunk_count": int(evidence.get("inspected_chunk_count") or 0),
            "chunk_window_tokens": int(evidence.get("chunk_window_tokens") or 0),
            "chunk_candidate_count": len(post_gate_candidates),
            "candidate_count": len(merged_candidates),
            "grounded_candidate_count": sum(
                item.get("supported") is True and item.get("source_grounded") is True
                for item in post_gate_candidates
            ),
            "rejected_ungrounded_count": sum(
                str(item.get("rejection_reason") or "")
                in {"model_quote_not_grounded", "deterministic_quote_not_grounded"}
                for item in post_gate_candidates
            ),
            "parallel_candidate": evidence.get("parallel_candidate") or {},
        },
        compact_facts=str(evidence.get("compact_facts") or ""),
        candidates=merged_candidates,
        chunk_candidates=post_gate_candidates,
    )

    compact_facts = str(evidence.get("compact_facts") or "").strip()
    selected_source_chunks = [
        item for item in evidence.get("selected_source_chunks") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    source_chunks = [
        item for item in evidence.get("source_chunks") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    if not source_chunks:
        # The model call itself may have raised before returning its chunk
        # metadata.  Recreate the same deterministic source chunks locally so
        # a model outage cannot erase an already fetched body.
        source_chunks = build_page_chunks(page_text)
    if not selected_source_chunks:
        selected_source_chunks = select_grounded_source_chunks(
            evidence_focus,
            source_chunks,
            [
                item
                for item in (evidence.get("chunk_candidates") or evidence.get("candidates") or [])
                if isinstance(item, dict)
            ],
            max_chunks=int(DATA_PIPELINE.get("web_fallback_chunks_per_source", 3) or 3),
            task_plan=task_plan,
        )
    source_excerpt = "\n\n".join(
        str(item.get("text") or "").strip()
        for item in selected_source_chunks
        if str(item.get("text") or "").strip()
    ).strip()
    if not source_excerpt:
        source_excerpt = str(evidence.get("source_excerpt") or "").strip()
    if not source_excerpt:
        source_excerpt = str(evidence.get("first_chunk_text") or "").strip()
    if not source_excerpt:
        source_excerpt = "\n".join(
            str(item.get("text") or "").strip()
            for item in evidence.get("source_chunks") or []
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ).strip()
    if not source_excerpt and len(page_text) <= MAX_SHORT_BODY_FALLBACK_CHARS:
        # This is the only permitted direct page-body fallback: a compact,
        # already-cleaned entry whose locator produced no usable output.
        source_excerpt = page_text
    page_evidence = {
        "url": url,
        "title": str(page.get("title") or candidate.get("title") or url),
        "content_resolver": str(page.get("content_resolver") or "direct_http"),
        "content_transport": str(page.get("content_transport") or "direct_fetch"),
        "status": str(evidence.get("status") or "no_evidence"),
        "error_class": str(evidence.get("error_class") or ""),
        "source_body_available": bool(source_excerpt),
        "page_chars": int(evidence.get("page_chars") or 0),
        "raw_page_chars": int(evidence.get("raw_page_chars") or len(raw_page_text)),
        "body_quality": {**page_quality, **query_signal, **(evidence.get("body_quality") or {})},
        "chunk_count": int(evidence.get("chunk_count") or 0),
        "inspected_chunk_count": int(evidence.get("inspected_chunk_count") or 0),
        "chunk_window_tokens": int(evidence.get("chunk_window_tokens") or 0),
        "candidate_count": len(evidence.get("candidates") or []),
        "valid_json_count": int(evidence.get("valid_json_count") or 0),
        "valid_contract_count": int(evidence.get("valid_contract_count") or 0),
        "invalid_response_count": int(evidence.get("invalid_response_count") or 0),
        "negative_response_count": int(evidence.get("negative_response_count") or 0),
        "all_chunks_valid_negative": bool(evidence.get("all_chunks_valid_negative")),
        "all_selected_chunks_valid_negative": bool(evidence.get("all_selected_chunks_valid_negative")),
        "all_selected_chunks_semantically_rejected": bool(
            evidence.get("all_selected_chunks_semantically_rejected")
        ),
        "extraction_degraded": bool(evidence.get("extraction_degraded")),
        "unresolved_chunk_indexes": [
            int(value)
            for value in evidence.get("unresolved_chunk_indexes") or []
            if isinstance(value, int)
        ],
        "recovered_chunk_indexes": [
            int(value)
            for value in evidence.get("recovered_chunk_indexes") or []
            if isinstance(value, int)
        ],
        "transport_error_count": int(evidence.get("transport_error_count") or 0),
        "parallel_candidate": evidence.get("parallel_candidate") or {},
        "model_chunks": evidence.get("model_chunks") or [],
        "errors": evidence.get("errors") or [],
    }
    if (
        page_evidence["all_chunks_valid_negative"]
        or page_evidence["all_selected_chunks_valid_negative"]
        or page_evidence["all_selected_chunks_semantically_rejected"]
    ):
        # The extractor's negative decision is useful routing metadata, not a
        # source admission gate.  Preserve only the bounded original spans so
        # the final RWKV writer can judge the fetched material itself.
        page_evidence["source_body_available"] = bool(source_excerpt)
        page_evidence["model_extraction_status"] = (
            "semantically_rejected"
            if page_evidence["all_selected_chunks_semantically_rejected"]
            and not page_evidence["all_selected_chunks_valid_negative"]
            else "negative"
        )
        page_evidence["deterministic_chunk_fallback"] = bool(source_excerpt)
        page_evidence["evidence_origin"] = "bounded_fetched_page_body"
    if not source_excerpt:
        return None, page_evidence

    rwkv_bound_claim_ids = list(
        dict.fromkeys(
            str(claim_id).strip()
            for candidate_row in (evidence.get("candidates") or [])
            if isinstance(candidate_row, dict)
            for claim_id in (candidate_row.get("claim_ids") or [])
            if str(claim_id).strip()
        )
    )

    extraction_status = str(page_evidence.get("status") or "no_evidence").casefold()
    # The locator improves chunk selection; it is never the admission gate for
    # a successfully fetched source.  When it fails, keep only the bounded
    # deterministic original spans selected above, never the complete page.
    deterministic_fallback = extraction_status in {"error", "no_evidence"} and not compact_facts
    explicit_negative_status = str(
        page_evidence.get("model_extraction_status") or ""
    ).casefold() in {"negative", "semantically_rejected"}
    if not explicit_negative_status:
        page_evidence["evidence_origin"] = "fetched_page_body"
        page_evidence["model_extraction_status"] = (
            "ok"
            if compact_facts
            else ("empty" if extraction_status == "no_evidence" else "error")
        )
        page_evidence["deterministic_chunk_fallback"] = deterministic_fallback

    record = {
        "title": page_evidence["title"],
        "url": url,
        "snippet": str(candidate.get("snippet") or "")[:800],
        "source": str(candidate.get("source") or "web"),
        "candidate_score": candidate.get("candidate_score", 0.0),
        "discovery_score": candidate.get("discovery_score", 0.0),
        "query_relevance": candidate.get("query_relevance") or {},
        "content": source_excerpt[:14000],
        "page_excerpt": source_excerpt[:14000],
        "source_excerpt": source_excerpt[:14000],
        "model_extracted_facts": compact_facts[:14000],
        "untrusted_content": True,
        "evidence_origin": page_evidence["evidence_origin"],
        "evidence_kind": "page_body",
        "evidence_boundary": "page_body_only",
        "body_verified": True,
        "body_quality": page_evidence["body_quality"],
        "content_sha256": hashlib.sha256(source_excerpt[:14000].encode("utf-8")).hexdigest(),
        "source_locator": {
            "type": "page_excerpt",
            "char_start": 0,
            "char_end": len(source_excerpt[:14000]),
        },
        "content_resolver": page_evidence["content_resolver"],
        "content_transport": page_evidence["content_transport"],
        "evidence_status": page_evidence["status"],
        "chunk_count": page_evidence["chunk_count"],
        "inspected_chunk_count": page_evidence["inspected_chunk_count"],
        "model_chunks": page_evidence["model_chunks"],
        "chunk_candidates": evidence.get("chunk_candidates") or evidence.get("candidates") or [],
        "source_chunks": source_chunks,
        "selected_source_chunks": selected_source_chunks,
        "claim_ids": rwkv_bound_claim_ids[:8],
        # The Planner-selected point scopes the retrieval attempt and the
        # extractor prompt; it is not evidence that every fetched page proves
        # that point. Only RWKV chunk bindings enter ``claim_ids`` above.
        "attempt_task_point_id": task_point_id,
        "locator_claim_id": (
            rwkv_bound_claim_ids[0] if len(rwkv_bound_claim_ids) == 1 else ""
        ),
        "claim_binding_origin": (
            "rwkv_chunk_binding"
            if any(
                candidate_row.get("claim_ids")
                for candidate_row in (evidence.get("candidates") or [])
                if isinstance(candidate_row, dict)
            )
            else "unbound"
        ),
        "locator_query": evidence_focus,
        "candidate_rank": candidate.get("candidate_rank"),
        "discovery_providers": candidate.get("discovery_providers") or [],
        "authority": candidate.get("authority") or {},
        "model_extraction_status": page_evidence["model_extraction_status"],
        "model_extraction_degraded": page_evidence["extraction_degraded"],
        "model_extraction_unresolved_chunks": page_evidence["invalid_response_count"],
        "model_extraction_transport_errors": page_evidence["transport_error_count"],
    }
    if compact_facts:
        record["model_locator_facts"] = compact_facts[:14000]
    if not has_substantive_evidence(record):
        # Keep a fetched bounded body even when lexical heuristics consider it
        # weak.  The metadata remains visible for ranking and audit, but only
        # RWKV decides whether the text supports the final answer.
        page_evidence["low_signal_body"] = True
        record["low_signal_body"] = True
    # A fetched body may still be retained for deterministic routing after
    # some chunk calls failed.  Keep that source available, but do not rewrite
    # the extraction outcome to ``ok``: callers need to know that part of the
    # page was never inspected by RWKV.
    page_evidence["status"] = (
        "partial" if page_evidence["extraction_degraded"] else "ok"
    )
    record["evidence_status"] = (
        "body_fallback"
        if deterministic_fallback
        else "partial_extraction"
        if page_evidence["extraction_degraded"]
        else "ok"
    )
    return record, page_evidence


def _model_extraction_diagnostics(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize unresolved RWKV chunk work without hiding usable sources."""

    rows: list[dict[str, Any]] = []
    recovered_chunks = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        parallel = page.get("parallel_candidate") or {}
        error_class = str(page.get("error_class") or "")
        is_extraction_page = bool(
            parallel
            or "valid_json_count" in page
            or "invalid_response_count" in page
            or error_class in {"chunk_extraction_failed", "page_evidence_failed", "cached_page_evidence_failed"}
        )
        if not is_extraction_page:
            continue
        unresolved = int(
            page.get("invalid_response_count")
            or parallel.get("unresolved_calls")
            or 0
        )
        recovered = len(page.get("recovered_chunk_indexes") or []) or int(
            parallel.get("recovered_calls") or 0
        )
        transport_errors = int(
            page.get("transport_error_count")
            or parallel.get("transport_error_count")
            or 0
        )
        recovered_chunks += recovered
        if not (page.get("extraction_degraded") or unresolved):
            continue
        rows.append(
            {
                "url": str(page.get("url") or ""),
                "status": str(page.get("status") or ""),
                "error_class": error_class or "chunk_extraction_incomplete",
                "unresolved_chunk_count": unresolved,
                "transport_error_count": transport_errors,
                "errors": [str(value)[:500] for value in (page.get("errors") or [])[:4]],
            }
        )
    return {
        "schema_version": "model-extraction-diagnostics.v1",
        "complete": not rows,
        "degraded_page_count": len(rows),
        "unresolved_chunk_count": sum(int(row["unresolved_chunk_count"]) for row in rows),
        "transport_error_count": sum(int(row["transport_error_count"]) for row in rows),
        "recovered_chunk_count": recovered_chunks,
        "pages": rows[:16],
    }


@ToolRegistry.register(
    name="web_search",
    phase="ALL",
    plugin="web.generic",
    capabilities=("url_discovery", "candidate_admission", "page_fetch", "markdown", "chunk_evidence"),
    retrieval_role="discovery",
    model_visible=True,
    category="retrieval",
    description="Search/fetch an exact URL, documentation, product or service status page, or the general web; also use it when no structured connector matches or connector evidence is insufficient.",
    signature="""[Tool] web_search
- Function: perform one bounded general-web retrieval transaction.
- Parameters: query (one concise search query or one complete http/https URL), max_results (optional, capped at 8).
- Pipeline: discovery, candidate admission/ranking, bounded page fetch, Markdown extraction, adaptive evidence extraction (cleaned pages up to the configured threshold stay single-pass; longer pages are chunked).
- Provider selection, URL fetching, page cleaning and chunk aggregation are internal backend steps; do not invent a provider-specific tool name.
- The result is evidence only. It is not a final answer and does not decide whether the user's task is complete.""",
)
def web_search(query: str, max_results: int = 8, **kwargs: Any) -> str:
    query = " ".join(str(query or "").split()).strip()
    task_id = str(kwargs.get("task_id") or "")
    agent_state = kwargs.get("agent_state")
    task_plan = kwargs.get("task_plan") if isinstance(kwargs.get("task_plan"), dict) else {}
    effective_task_plan = dict(task_plan)
    task_point_id = str(kwargs.get("task_point_id") or "").strip()
    original_goal = str(kwargs.get("original_goal") or query).strip()
    point_count = len(task_points(task_plan, fallback_query=original_goal))
    constraint_query = (
        _claim_evidence_focus(query, task_plan, task_point_id)
        if task_point_id and point_count > 1
        else original_goal
    )
    scoped_domains = required_domains_for_task_point(
        effective_task_plan,
        task_point_id,
        fallback_query=query,
    )
    # A hostname invented by the task-plan model is useful as a discovery
    # hypothesis, but it must not become a hard admission gate.  Only a domain
    # explicitly present in the user's goal is mandatory before retrieval;
    # otherwise the generic providers and official-site adapter verify or
    # replace the hypothesis from observable web results.
    explicit_goal_domains = explicit_domains(original_goal)
    model_domain_hypotheses = list(scoped_domains)
    explicit_scoped_domains = [
        domain
        for domain in scoped_domains
        if any(
            domain == explicit
            or domain.endswith("." + explicit)
            or explicit.endswith("." + domain)
            for explicit in explicit_goal_domains
        )
    ]
    if explicit_scoped_domains:
        effective_task_plan["required_domains"] = explicit_scoped_domains
    elif scoped_domains:
        effective_task_plan["required_domains"] = []
        effective_task_plan["preferred_domains"] = model_domain_hypotheses
        effective_task_plan["source_resolution"] = {
            "domain_source": "model_hypothesis_unverified",
            "required_domains": [],
            "preferred_domains": model_domain_hypotheses,
        }
    freshness_policy = build_freshness_policy(
        kwargs.get("original_goal") or query,
        effective_task_plan,
    )
    # Evidence locators need the same observable time boundary that is stored
    # in the retrieval result.  This prevents a future, explicitly labelled
    # prerelease row from being selected as the current release while keeping
    # the fetched source body intact for RWKV.
    effective_task_plan["freshness_policy"] = freshness_policy
    source_policy = resolve_source_policy(
        original_goal,
        {"task_plan": effective_task_plan},
    )
    if explicit_scoped_domains:
        source_policy = resolve_source_policy(
            constraint_query,
            {
                "task_plan": effective_task_plan,
                "required_domains": explicit_scoped_domains,
                "source_policy": source_policy.get("mode"),
            },
        )
    if not query:
        return json.dumps({"status": "error", "message": "query is empty", "results": []}, ensure_ascii=False)

    def source_resolution_payload() -> dict[str, Any]:
        value = effective_task_plan.get("source_resolution") or {}
        if isinstance(value, dict) and value:
            return dict(value)
        return {
            "domain_source": source_policy.get("domain_source") or "unresolved",
            "required_domains": list(source_policy.get("required_domains") or []),
        }

    try:
        requested_candidates = max(1, min(int(max_results or 8), 32))
    except (TypeError, ValueError):
        requested_candidates = 8
    fetch_limit = _pipeline_concurrency("web_fetch_limit", 8, maximum=32)
    max_candidates = min(requested_candidates, fetch_limit)
    provider_result_limit = max(
        max_candidates,
        _pipeline_concurrency("web_provider_result_limit", 20, maximum=50),
    )
    candidate_pool_limit = _pipeline_concurrency(
        "web_candidate_pool_limit", 48, maximum=100
    )
    rrf_k = _pipeline_concurrency("web_rrf_k", 60, maximum=1000)
    rrf_shadow_enabled = bool(DATA_PIPELINE.get("web_enable_rrf_shadow", True))
    max_pages = min(
        _pipeline_concurrency("web_search_max_pages", 8, maximum=16),
        fetch_limit,
    )
    append_task_event(
        task_id,
        "web_search_stage",
        phase="DISCOVERY",
        action="web_search",
        stage="start",
        query=query,
        budget={
            "max_candidates": max_candidates,
            "provider_result_limit": provider_result_limit,
            "candidate_pool_limit": candidate_pool_limit,
            "fetch_limit": fetch_limit,
            "rrf_k": rrf_k,
            "rrf_shadow_only": rrf_shadow_enabled,
            "max_pages": max_pages,
            "context_length": get_llm_context_length(),
            "chunk_mode": DATA_PIPELINE.get("web_chunk_mode", "adaptive"),
            "single_pass_threshold_tokens": DATA_PIPELINE.get("web_chunk_single_pass_tokens", 7000),
            "chunk_target_tokens": DATA_PIPELINE.get("web_chunk_tokens", 4096),
            "chunk_max_tokens": DATA_PIPELINE.get("web_chunk_max_tokens", 2400),
            "page_evidence_concurrency": DATA_PIPELINE.get("web_page_evidence_concurrency", 8),
        },
    )

    def run_keyless() -> dict[str, Any]:
        return _parse_result(
            search_web_keyless(
                query,
                max_results=provider_result_limit,
                fetch_pages=0,
                task_id=task_id,
                agentic_tool_loop=True,
                constraint_query=constraint_query,
            )
        )

    def run_tavily() -> dict[str, Any]:
        return _parse_result(
            search_web_tavily(
                query,
                max_results=provider_result_limit,
                search_depth="advanced",
                topic="general",
                task_id=task_id,
            )
        )

    def run_official_site() -> dict[str, Any]:
        return discover_official_urls(
            query,
            source_policy.get("required_domains") or [],
            max_results=provider_result_limit,
            prefer_recent=_requests_recency(original_goal),
            constraint_query=constraint_query,
        )

    def run_model_domain_hypothesis() -> dict[str, Any]:
        return discover_official_urls(
            query,
            model_domain_hypotheses,
            max_results=provider_result_limit,
            prefer_recent=_requests_recency(original_goal),
            constraint_query=constraint_query,
        )

    direct_url = _extract_direct_url(query) or _extract_single_goal_url(kwargs.get("original_goal"))
    if direct_url:
        provider_results: list[dict[str, Any]] = []
        candidate_pool_shadow: list[dict[str, Any]] = []
        provider_statuses = [
            {"provider": "direct_url", "status": "ok", "count": 1, "errors": []}
        ]
        candidates = (
            []
            if _is_unsupported_download_url(direct_url)
            else [
                annotate_source(
                    _direct_candidate(direct_url),
                    original_goal,
                    {"task_plan": effective_task_plan},
                )
            ]
        )
    else:
        provider_jobs = (
            [run_official_site, run_keyless, run_tavily]
            if source_policy.get("required") and source_policy.get("required_domains")
            else [run_model_domain_hypothesis, run_keyless, run_tavily]
            if model_domain_hypotheses
            else [run_keyless, run_tavily]
        )
        provider_results: list[dict[str, Any]] = [
            {"status": "error", "provider_errors": ["provider did not complete"], "results": []}
            for _ in provider_jobs
        ]
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(
                max(
                    _pipeline_concurrency("web_provider_concurrency", 2),
                    3 if len(provider_jobs) == 3 else 2,
                ),
                len(provider_jobs),
            )
        )
        futures = {
            submit_with_context(pool, job): index
            for index, job in enumerate(provider_jobs)
        }
        cancelled = False
        try:
            for future in concurrent.futures.as_completed(futures, timeout=task_wait_timeout()):
                index = futures[future]
                try:
                    provider_results[index] = future.result()
                except Exception as exc:
                    provider_results[index] = {
                        "status": "error",
                        "provider_errors": [f"{type(exc).__name__}: {exc}"],
                        "results": [],
                    }
        except concurrent.futures.TimeoutError:
            cancelled = True
            # This exception can be the acceptance runner's SIGALRM case
            # boundary as well as an executor deadline.  Do not turn a hard
            # case timeout into a successful-looking partial search; cancel
            # pending work in ``finally`` and let the outer runner emit the
            # answer-shaped timeout result.
            raise
        finally:
            shutdown_pool(pool, list(futures), cancelled=cancelled)

        # Preserve Round 19's active per-provider admission slice while
        # retaining the wider raw lists for offline recall/RRF analysis.  The
        # shadow pool cannot affect authority inference, fetching or RWKV.
        shadow_provider_results = [dict(result) for result in provider_results]
        provider_results = [
            {
                **dict(result),
                "results": list(result.get("results") or [])[:max_candidates],
                "count": min(
                    int(result.get("count") or len(result.get("results") or [])),
                    max_candidates,
                ),
            }
            for result in provider_results
        ]

        # For an unseen project the user may require an official source without
        # supplying its hostname. Provider results can suggest domains, but a
        # lexical ownership guess is not proof and must not open a sitemap or
        # consume fetch slots. Round 24 therefore records this inference only
        # as a shadow diagnostic. Ordinary providers remain the active first
        # pass; if RWKV later chooses a site/domain in its next exact query,
        # that model-owned hypothesis goes through `run_model_domain_hypothesis`.
        authority_candidates: list[dict[str, Any]] = []
        if (
            str(source_policy.get("mode") or "").casefold() == "official_required"
            and not source_policy.get("required_domains")
        ):
            authority_candidates = infer_candidate_authority_domains(
                constraint_query or query,
                provider_results,
            )
            if authority_candidates:
                append_task_event(
                    task_id,
                    "web_search_stage",
                    phase="DISCOVERY",
                    action="web_search",
                    stage="authority_resolution",
                    query=query,
                    source="provider_bootstrap_shadow",
                    candidates=authority_candidates,
                    required_domains=[],
                    preferred_domains=[],
                    affects_admission=False,
                    affects_fetch=False,
                    affects_rwkv=False,
                )

        # A first-party sitemap fetched through the planned official domain
        # may prove that the project has redirected to a canonical hostname.
        # Publish that alias to every downstream layer before candidate
        # annotation; otherwise discovery succeeds but the authority gate
        # rejects the very page that the official host redirected us to.
        planned_domains = list(source_policy.get("required_domains") or [])
        preferred_domains = list(
            effective_task_plan.get("preferred_domains") or []
        )
        resolved_domains = list(planned_domains)
        resolved_preferred_domains = list(preferred_domains)
        verified_aliases: list[dict[str, Any]] = []
        for provider_result in provider_results:
            if str(provider_result.get("provider") or "") != "official site adapter":
                continue
            aliases = [
                dict(value)
                for value in provider_result.get("domain_aliases") or []
                if isinstance(value, dict)
                and value.get("verification") == "dominant_https_sitemap_host"
            ]
            verified_aliases.extend(aliases)
            for value in provider_result.get("resolved_domains") or []:
                domain = str(value or "").casefold().removeprefix("www.").rstrip(".")
                if not domain:
                    continue
                if planned_domains and domain not in resolved_domains:
                    resolved_domains.append(domain)
                elif not planned_domains and domain not in resolved_preferred_domains:
                    resolved_preferred_domains.append(domain)
        if planned_domains and verified_aliases and resolved_domains != planned_domains:
            effective_task_plan["required_domains"] = resolved_domains
            effective_task_plan["source_resolution"] = {
                "domain_source": "official_sitemap_canonicalization",
                "required_domains": resolved_domains,
                "aliases": verified_aliases,
            }
            source_policy = resolve_source_policy(
                constraint_query or query,
                {"task_plan": effective_task_plan},
            )
            append_task_event(
                task_id,
                "web_search_stage",
                phase="DISCOVERY",
                action="web_search",
                stage="authority_resolution",
                query=query,
                source="official_sitemap_canonicalization",
                aliases=verified_aliases,
                required_domains=resolved_domains,
            )
        elif (
            not planned_domains
            and verified_aliases
            and resolved_preferred_domains != preferred_domains
        ):
            effective_task_plan["preferred_domains"] = resolved_preferred_domains
            effective_task_plan["source_resolution"] = {
                "domain_source": "official_sitemap_hypothesis_canonicalization",
                "required_domains": [],
                "preferred_domains": resolved_preferred_domains,
                "aliases": verified_aliases,
            }
            append_task_event(
                task_id,
                "web_search_stage",
                phase="DISCOVERY",
                action="web_search",
                stage="authority_resolution",
                query=query,
                source="official_sitemap_hypothesis_canonicalization",
                aliases=verified_aliases,
                required_domains=[],
                preferred_domains=resolved_preferred_domains,
                affects_admission=False,
            )

        provider_statuses = [
            {
                "provider": result.get("provider") or "unknown",
                "status": result.get("status") or "error",
                "count": int(result.get("count") or len(result.get("results") or [])),
                "errors": result.get("provider_errors") or [],
            }
            for result in provider_results
        ]
        candidates = _merge_candidates(
            query,
            provider_results,
            limit=max_candidates,
            task_plan=effective_task_plan,
            constraint_query=constraint_query,
            policy_query=original_goal,
        )
        candidate_pool_shadow = (
            rrf_fuse(
                shadow_provider_results,
                rrf_k=rrf_k,
                pool_limit=candidate_pool_limit,
            )
            if rrf_shadow_enabled
            else []
        )
        legacy_rank_by_url = {
            retrieval_url_identity(str(item.get("url") or "")): int(
                item.get("candidate_rank") or index
            )
            for index, item in enumerate(candidates, start=1)
        }
        for item in candidate_pool_shadow:
            item["legacy_rank"] = legacy_rank_by_url.get(
                retrieval_url_identity(str(item.get("url") or ""))
            )
            # Provider snippets remain in provider responses and are not
            # needed in the compact shadow record.
            item.pop("snippet", None)
        append_task_event(
            task_id,
            "web_candidate_pool_shadow",
            phase="DISCOVERY",
            action="web_search",
            query=query,
            active_ranking="legacy",
            provider_result_limit=provider_result_limit,
            active_provider_slice=max_candidates,
            candidate_pool_limit=candidate_pool_limit,
            fetch_limit=max_pages,
            rrf_k=rrf_k,
            providers=[
                {
                    "provider": result.get("provider") or "unknown",
                    "raw_count": len(result.get("results") or []),
                    "status": result.get("status") or "error",
                }
                for result in shadow_provider_results
            ],
            legacy_candidates=[
                {
                    "rank": item.get("candidate_rank"),
                    "url": item.get("url"),
                    "title": item.get("title"),
                }
                for item in candidates
            ],
            rrf_candidates=candidate_pool_shadow,
            affects_fetch=False,
            affects_rwkv=False,
        )

    # Authority remains auditable metadata and a soft ranking signal.  A
    # model- or adapter-derived domain hypothesis must never suppress every
    # fetched page.  Explicit user URLs are already handled by direct_url.

    # A follow-up research round must add novelty to the shared evidence
    # store. Reuse is handled from the store; do not refetch the same page as
    # a second recovery path.
    seen_urls = _shared_source_urls(agent_state)
    novel_candidates = [
        item for item in candidates
        if retrieval_url_identity(str(item.get("url") or "")) not in seen_urls
    ]
    reused_records = []
    retrieval = getattr(agent_state, "retrieval", None)
    stored_sources = getattr(retrieval, "sources", {}) if retrieval is not None else {}
    claim_sources = (
        getattr(retrieval, "sources_by_claim", {}).get(task_point_id, {})
        if retrieval is not None and task_point_id
        else {}
    )
    if not novel_candidates and stored_sources:
        discovered_urls = {
            retrieval_url_identity(str(item.get("url") or ""))
            for item in candidates
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        }
        claim_by_url = {
            retrieval_url_identity(str(item.get("url") or "")): item
            for item in claim_sources.values()
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        }
        for item in stored_sources.values():
            if not isinstance(item, dict):
                continue
            normalized = retrieval_url_identity(str(item.get("url") or ""))
            if normalized not in discovered_urls:
                continue
            reused_records.append(dict(claim_by_url.get(normalized) or item))
            if len(reused_records) >= max_pages:
                break
    if novel_candidates:
        candidates = novel_candidates
    elif not reused_records and seen_urls:
        result = {
            "status": "no_new_evidence",
            "real_network": False,
            "provider": "evidence_store",
            "retrieval_role": "discovery",
            "query": query,
            "count": 0,
            "candidate_count": len(candidates),
            "fetched_count": 0,
            "results": [],
            "sources": [],
            "citation_refs": [],
            "page_evidence": [],
            "evidence_ready": False,
            "reused_sources": False,
            "novel_source_count": 0,
            "model_extraction": _model_extraction_diagnostics([]),
            "source_resolution": source_resolution_payload(),
            "evidence_policy": "all discovered URLs are already in the shared evidence store; refine the query",
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    append_task_event(
        task_id,
        "web_search_stage",
        phase="DISCOVERY",
        action="web_search",
        stage="candidate_admission",
        query=query,
        providers=provider_statuses,
        candidates=[
            {
                key: item.get(key)
                for key in (
                    "candidate_rank",
                    "title",
                    "url",
                    "snippet",
                    "source",
                    "candidate_score",
                    "discovery_score",
                    "query_relevance",
                    "discovery_providers",
                    "authority",
                )
            }
            for item in candidates
        ],
    )

    if not candidates:
        errors = [
            str(error)
            for result in provider_results
            for error in (result.get("provider_errors") or [])
            if str(error).strip()
        ]
        return json.dumps(
            {
                "status": "error" if any(item.get("status") == "error" for item in provider_results) else "no_results",
                "real_network": True,
                "provider": "web.generic",
                "retrieval_role": "discovery",
                "query": query,
                "count": 0,
                "results": [],
                "sources": [],
                "citation_refs": [],
                "provider_statuses": provider_statuses,
                "provider_errors": errors,
                "candidate_count": 0,
                "page_evidence": [],
                "evidence_ready": False,
                "source_resolution": source_resolution_payload(),
                "freshness_policy": freshness_policy,
                "evidence_policy": "no candidate URL was admitted; the model must decide whether to refine the query",
            },
            ensure_ascii=False,
            indent=2,
        )

    if reused_records:
        bound_records = [
            dict(item)
            for item in reused_records
            if not task_point_id
            or str(item.get("locator_claim_id") or "").strip() == task_point_id
        ]
        foreign_records = [
            dict(item)
            for item in reused_records
            if task_point_id
            and str(item.get("locator_claim_id") or "").strip() != task_point_id
        ]
        page_evidence: list[dict[str, Any]] = []
        if foreign_records:
            llm = LLMClient()

            def reextract_cached(record: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
                candidate = annotate_source(
                    dict(record),
                    original_goal,
                    {"task_plan": effective_task_plan},
                )
                return _compact_page(
                    query,
                    candidate,
                    _cached_page_result(record),
                    llm,
                    task_id,
                    task_plan=effective_task_plan,
                    task_point_id=task_point_id,
                )

            workers = min(
                _pipeline_concurrency("web_page_evidence_concurrency", 8, maximum=32),
                len(foreign_records),
            )
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers))
            futures = {
                submit_with_context(pool, reextract_cached, item): index
                for index, item in enumerate(foreign_records)
            }
            processed_cached: list[tuple[int, dict[str, Any] | None, dict[str, Any]]] = []
            cancelled = False
            try:
                for future in concurrent.futures.as_completed(
                    futures,
                    timeout=task_wait_timeout(),
                ):
                    index = futures[future]
                    try:
                        record, evidence = future.result()
                    except Exception as exc:
                        record, evidence = None, {
                            "status": "error",
                            "error_class": "cached_page_evidence_failed",
                            "errors": [f"{type(exc).__name__}: {exc}"],
                        }
                    processed_cached.append((index, record, evidence))
            except concurrent.futures.TimeoutError:
                cancelled = True
                raise
            finally:
                shutdown_pool(pool, list(futures), cancelled=cancelled)
            for _, record, evidence in sorted(processed_cached, key=lambda value: value[0]):
                page_evidence.append(evidence)
                if record:
                    bound_records.append(annotate_freshness(record, freshness_policy))

        usable_records = [item for item in bound_records if has_substantive_evidence(item)]
        refs = [
            {
                "ref_id": f"WEB_CACHE_{index}",
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "evidence_text": item.get("content", "")[:14000],
                "evidence_origin": item.get("evidence_origin", "fetched_page_body"),
                "evidence_boundary": item.get("evidence_boundary", "page_body_only"),
                "source_locator": item.get("source_locator") or {},
            }
            for index, item in enumerate(usable_records, start=1)
        ]
        extraction_diagnostics = _model_extraction_diagnostics(page_evidence)
        result = {
            "status": "ok" if usable_records else "no_evidence",
            "real_network": False,
            "provider": "evidence_store",
            "retrieval_role": "discovery",
            "query": query,
            "count": len(bound_records),
            "candidate_count": len(candidates),
            "fetched_count": 0,
            "results": bound_records,
            "sources": [item.get("url", "") for item in bound_records],
            "citation_refs": refs,
            "page_evidence": page_evidence,
            "evidence_ready": bool(usable_records),
            "reused_sources": True,
            "reextracted_cached_sources": bool(foreign_records),
            "novel_source_count": 0,
            "model_extraction": extraction_diagnostics,
            "source_resolution": source_resolution_payload(),
            "freshness_policy": freshness_policy,
            "evidence_policy": (
                "claim-scoped chunk re-extraction from the shared page cache; no duplicate network fetch"
                if foreign_records
                else "existing claim-scoped evidence projection; no duplicate network fetch"
            ),
        }
        append_task_event(
            task_id,
            "web_search_stage",
            phase="EXTRACTION" if foreign_records else "RESEARCH",
            action="web_search",
            stage="cache_reextract" if foreign_records else "reuse",
            query=query,
            count=len(bound_records),
            task_point_id=task_point_id,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    llm = LLMClient()
    selected = candidates[:max_pages]
    fetched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(_pipeline_concurrency("web_page_fetch_concurrency", 2), len(selected))
    )
    future_map = {
        submit_with_context(pool, _fetch_candidate, item, task_id): item
        for item in selected
    }
    cancelled = False
    try:
        for future in concurrent.futures.as_completed(future_map, timeout=task_wait_timeout()):
            candidate = future_map[future]
            try:
                fetched.append((candidate, future.result()))
            except Exception as exc:
                fetched.append((candidate, {"status": "error", "message": f"{type(exc).__name__}: {exc}", "results": []}))
    except concurrent.futures.TimeoutError:
        cancelled = True
        # Preserve the hard case boundary instead of continuing with a
        # partially fetched page set while the original workers still hold
        # task/model leases.
        raise
    finally:
        shutdown_pool(pool, list(future_map), cancelled=cancelled)

    fetched.sort(key=lambda item: int(item[0].get("candidate_rank") or 10**6))
    fetched = _resolve_failed_page_fetches(
        constraint_query,
        fetched,
        task_id,
    )
    # Page extraction is independent once fetches have completed. Run pages
    # concurrently, while page_evidence.py applies the separate global model
    # request gate to every chunk call.
    page_workers = min(
        _pipeline_concurrency("web_page_evidence_concurrency", 8, maximum=32),
        max(1, len(fetched)),
    )

    def process_page(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        candidate, fetched_result = pair
        # When Planner omits the optional task_point_id, extraction must retain
        # the complete user request rather than inherit a narrow search query.
        # Otherwise a query for P1 makes RWKV's chunk locator blind to P2 even
        # when the same fetched paper/page contains both requested facts.
        return _compact_page(
            constraint_query,
            candidate,
            fetched_result,
            llm,
            task_id,
            task_plan=effective_task_plan,
            task_point_id=task_point_id,
        )

    processed: list[tuple[int, dict[str, Any] | None, dict[str, Any]]] = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=page_workers)
    future_map = {
        submit_with_context(pool, process_page, pair): index
        for index, pair in enumerate(fetched)
    }
    cancelled = False
    try:
        for future in concurrent.futures.as_completed(future_map, timeout=task_wait_timeout()):
            index = future_map[future]
            try:
                record, page_evidence = future.result()
            except Exception as exc:
                record, page_evidence = None, {
                    "status": "error",
                    "error_class": "page_evidence_failed",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            processed.append((index, record, page_evidence))
    except concurrent.futures.TimeoutError:
        cancelled = True
        # The outer case runner owns the answer-shaped timeout fallback.  A
        # search layer must not swallow the signal and keep the analysis slot
        # occupied beyond the case boundary.
        raise
    finally:
        shutdown_pool(pool, list(future_map), cancelled=cancelled)

    evidence_pages: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for _, record, page_evidence in sorted(processed, key=lambda item: item[0]):
        evidence_pages.append(page_evidence)
        append_task_event(
            task_id,
            "web_search_stage",
            phase="EXTRACTION",
            action="web_search",
            stage="page_evidence",
            query=query,
            page=page_evidence,
        )
        if record:
            records.append(annotate_freshness(record, freshness_policy))

    refs = [
        {
            "ref_id": f"WEB_GENERIC_{index}",
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "source": item.get("source", ""),
            "evidence_text": item.get("content", "")[:14000],
            "evidence_origin": item.get("evidence_origin", "fetched_page_body"),
            "evidence_boundary": item.get("evidence_boundary", "page_body_only"),
            "source_locator": item.get("source_locator") or {},
        }
        for index, item in enumerate(records, start=1)
    ]
    usable_records = [item for item in records if has_substantive_evidence(item)]
    missing_page_count = sum(
        1
        for page in evidence_pages
        if str(page.get("status") or "").casefold() != "ok"
    )
    extraction_diagnostics = _model_extraction_diagnostics(evidence_pages)
    result = {
        "status": "ok" if usable_records else "no_evidence",
        "real_network": True,
        "provider": "web.generic",
        "retrieval_role": "discovery",
        "query": query,
        "count": len(records),
        "candidate_count": len(candidates),
        "fetched_count": len(fetched),
        "results": records,
        "sources": [item.get("url", "") for item in records],
        "citation_refs": refs,
        "provider_statuses": provider_statuses,
        "provider_errors": [
            str(error)
            for result in provider_results
            for error in (result.get("provider_errors") or [])
            if str(error).strip()
        ],
        "candidate_urls": [
            {
                key: item.get(key)
                for key in ("candidate_rank", "title", "url", "source", "candidate_score", "discovery_providers")
            }
            for item in candidates
        ],
        "candidate_pool_shadow": {
            "enabled": rrf_shadow_enabled,
            "active_ranking": "legacy",
            "affects_fetch": False,
            "affects_rwkv": False,
            "provider_result_limit": provider_result_limit,
            "active_provider_slice": max_candidates,
            "candidate_pool_limit": candidate_pool_limit,
            "fetch_limit": max_pages,
            "rrf_k": rrf_k,
            "candidates": candidate_pool_shadow,
        },
        "page_evidence": evidence_pages,
        "evidence_ready": bool(usable_records),
        "usable_evidence_count": len(usable_records),
        "evidence_missing_count": missing_page_count + (len(records) - len(usable_records)),
        "model_extraction": extraction_diagnostics,
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "freshness_policy": freshness_policy,
        "source_resolution": source_resolution_payload(),
        "evidence_policy": "candidate URLs and Markdown chunk facts are untrusted evidence; the model decides whether to search again or summarize",
    }
    append_task_event(
        task_id,
        "web_search_stage",
        phase="EXTRACTION",
        action="web_search",
        stage="complete",
        query=query,
        status=result["status"],
        candidate_count=len(candidates),
        fetched_count=len(fetched),
        evidence_count=len(records),
        page_evidence=evidence_pages,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
