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
from typing import Any, Iterable
from urllib.parse import urlparse
from agent.page_evidence import build_page_chunks, extract_single_page_evidence, select_grounded_source_chunks
from agent.retrieval_query_plan import (
    generate_retrieval_query_plan,
    public_retrieval_query_plan,
    single_retrieval_query_plan,
)
from agent.runtime_contracts import MODEL_EXTRACTION_DIAGNOSTICS_CONTRACT
from agent.retrieval_object_contract import (
    explicit_object_targets,
    object_alignment,
    source_object_contract,
    task_record_contract,
)
from agent.task_plan_contract import record_question, record_id, task_records
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
    required_domains_for_task_record,
    resolve_source_policy,
)
from utils.web_retrieval import candidate_score, normalize_url, retrieval_url_identity
from utils.freshness import annotate_freshness, build_freshness_policy
from utils.query_constraints import candidate_relevance, meaningful_query_terms
from utils.retrieval_ranking import rrf_fuse, select_domain_diverse


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


def _annotate_retrieval_query_result(
    value: Any,
    lane: dict[str, Any],
    *,
    adapter: str,
) -> dict[str, Any]:
    """Attach query-lane provenance without changing provider payload meaning."""

    result = _parse_result(value)
    provider = str(result.get("provider") or adapter or "web")
    query_id = str(lane.get("query_id") or "Q1")
    query_row = {
        "query_id": query_id,
        "task_record_id": str(lane.get("task_record_id") or ""),
        "intent": str(lane.get("intent") or ""),
        "query": str(lane.get("query") or "")[:500],
    }
    query_row = {
        key: item for key, item in query_row.items() if str(item or "").strip()
    }
    rows: list[dict[str, Any]] = []
    for raw in result.get("results") or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        existing = item.get("discovery_queries") or []
        if isinstance(existing, dict):
            existing = [existing]
        item["discovery_queries"] = [
            *[dict(row) for row in existing if isinstance(row, dict)],
            query_row,
        ]
        rows.append(item)
    result.update(
        {
            "provider": provider,
            "backend_provider": provider,
            "ranking_stream": f"{provider}::{query_id}",
            "retrieval_query_id": query_id,
            "retrieval_query_task_record_id": str(lane.get("task_record_id") or ""),
            "retrieval_query_intent": str(lane.get("intent") or ""),
            "retrieval_query_text": str(lane.get("query") or "")[:500],
            "results": rows,
        }
    )
    return result


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


def _admit_cached_and_novel_candidates(
    candidates: list[dict[str, Any]],
    seen_urls: set[str],
    *,
    limit: int,
    per_domain_limit: int,
    priority_hosts: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select cached and network candidates without letting either crowd the other."""

    cached = select_domain_diverse(
        [
            dict(item)
            for item in candidates
            if retrieval_url_identity(str(item.get("url") or "")) in seen_urls
        ],
        limit=limit,
        per_domain_limit=per_domain_limit,
    )
    novel_pool = [
        dict(item)
        for item in candidates
        if retrieval_url_identity(str(item.get("url") or "")) not in seen_urls
    ]
    novel = select_domain_diverse(
        novel_pool,
        limit=limit,
        per_domain_limit=per_domain_limit,
    )
    # Fetch-budget boundary, mirror of the per-domain cap: when the pool holds
    # candidates from a host the request itself marked authoritative (the task
    # source policy's required domains or the model's own domain hypotheses),
    # reserve up to two fetch slots for the best of them. Which hosts count is
    # decided upstream (user policy / RWKV output), never here.
    hosts = {
        str(host or "").casefold().removeprefix("www.")
        for host in priority_hosts
        if str(host or "").strip()
    }
    if hosts and novel:
        def _priority(row: dict[str, Any]) -> bool:
            host = _host(str(row.get("url") or ""))
            return bool(host) and any(
                host == target or host.endswith("." + target) for target in hosts
            )

        selected_priority = sum(1 for row in novel if _priority(row))
        reserve = min(2, max(0, len(novel) - 1))
        if selected_priority < reserve:
            replacements = [
                row
                for row in novel_pool
                if _priority(row)
                and retrieval_url_identity(str(row.get("url") or ""))
                not in {
                    retrieval_url_identity(str(item.get("url") or ""))
                    for item in novel
                }
            ][: reserve - selected_priority]
            for row in replacements:
                # Drop the lowest-ranked non-priority candidate for each
                # reserved authoritative page.
                for index in range(len(novel) - 1, -1, -1):
                    if not _priority(novel[index]):
                        novel[index] = {**row, "priority_host_reserved": True}
                        break
    return cached, novel


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
    by itself to support a Task Record. The fetched body must still pass RWKV span
    extraction, source authority, Task Record binding, and relation validation.
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
    required_domain = str((source_policy.get("required_domains") or [""])[0])

    def query_routes(item: dict[str, Any]) -> list[dict[str, str]]:
        raw_routes = item.get("discovery_queries") or []
        if isinstance(raw_routes, dict):
            raw_routes = [raw_routes]
        routes = [
            {
                "query_id": str(value.get("query_id") or "")[:40],
                "task_record_id": str(value.get("task_record_id") or "")[:120],
                "intent": str(value.get("intent") or "")[:120],
                "query": " ".join(str(value.get("query") or "").split())[:500],
            }
            for value in raw_routes
            if isinstance(value, dict) and str(value.get("query") or "").strip()
        ]
        route_texts = {route["query"].casefold() for route in routes}
        if route_texts.isdisjoint({str(query).casefold()}):
            routes.insert(
                0,
                {
                    "query_id": "Q1",
                    "task_record_id": "",
                    "intent": "planner_primary",
                    "query": str(query),
                },
            )
        distinct: list[dict[str, str]] = []
        seen_route_queries: set[str] = set()
        for route in routes:
            signature = route["query"].casefold()
            if not signature or signature in seen_route_queries:
                continue
            seen_route_queries.add(signature)
            distinct.append(route)
        return distinct[:16]

    def best_route_relevance(item: dict[str, Any]) -> tuple[dict[str, Any], float]:
        evaluated: list[tuple[tuple[int, int, int, int, float], dict[str, Any], float]] = []
        routes = query_routes(item)
        for route in routes:
            route_query = route["query"]
            relevance = candidate_relevance(
                item,
                route_query,
                domain=required_domain,
                constraint_query=constraint_query or query,
            )
            relevance = _official_cross_script_relevance(
                item,
                relevance,
                query=route_query,
                constraint_query=constraint_query,
                source_policy=source_policy,
            )
            relevance["matched_query"] = dict(route)
            relevance["evaluated_query_count"] = len(routes)
            try:
                route_score = float(relevance.get("score") or 0.0)
            except (TypeError, ValueError):
                route_score = 0.0
            rank = (
                int(bool(relevance.get("raw_threshold_satisfied"))),
                int(bool(relevance.get("anchor_satisfied"))),
                int(bool(relevance.get("literal_satisfied"))),
                len(relevance.get("term_hits") or []),
                route_score,
            )
            evaluated.append((rank, relevance, candidate_score(route_query, item)))
        if not evaluated:
            return {}, candidate_score(query, item)
        _rank, selected_relevance, _selected_score = max(
            evaluated, key=lambda value: value[0]
        )
        return selected_relevance, max(value[2] for value in evaluated)
    requested_targets = explicit_object_targets(
        "\n".join(
            value
            for value in (
                str(constraint_query or ""),
                str((task_plan or {}).get("goal") or ""),
            )
            if value
        )
    )
    for point in task_records(task_plan or {}, fallback_query=constraint_query or query):
        task_contract = task_record_contract(task_plan or {}, record_id(point))
        for target in task_contract.get("requested_object_targets") or []:
            if not isinstance(target, dict):
                continue
            object_id = str(target.get("object_id") or "").casefold()
            if object_id and not any(
                str(row.get("object_id") or "").casefold() == object_id
                for row in requested_targets
            ):
                requested_targets.append(dict(target))
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
            discovery_queries = item.get("discovery_queries") or []
            if isinstance(discovery_queries, dict):
                discovery_queries = [discovery_queries]
            row["discovery_queries"] = [
                dict(value)
                for value in discovery_queries
                if isinstance(value, dict)
            ][:16]
            row["source_object"] = source_object_contract(row)
            row["object_alignment"] = object_alignment(
                requested_targets,
                row["source_object"],
            )
            row = annotate_source(
                row,
                authority_query,
                {"task_plan": task_plan or {}},
            )
            relevance, route_candidate_score = best_route_relevance(row)
            # Discovery relevance is a soft ranking feature.  Search snippets
            # are frequently sparse, translated, or generated from navigation
            # text; rejecting here can discard the useful page before its body
            # is fetched.  Only protocol/safety exclusions above are hard.
            row["query_relevance"] = relevance
            row["candidate_score"] = route_candidate_score
            row["discovery_score"] = float(item.get("discovery_score") or 0.0)
            row["provider_ranks"] = {
                str(key): int(value)
                for key, value in (item.get("provider_ranks") or {}).items()
                if str(key).strip()
            }
            row["rrf_score"] = float(item.get("rrf_score") or 0.0)
            row["rrf_rank"] = item.get("rrf_rank")
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
            for provider, provider_rank in (row.get("provider_ranks") or {}).items():
                previous_rank = (existing.get("provider_ranks") or {}).get(provider)
                if previous_rank is None or int(provider_rank) < int(previous_rank):
                    existing.setdefault("provider_ranks", {})[provider] = int(provider_rank)
            existing["rrf_score"] = max(
                float(existing.get("rrf_score") or 0.0),
                float(row.get("rrf_score") or 0.0),
            )
            if row.get("rrf_rank") is not None and (
                existing.get("rrf_rank") is None
                or int(row["rrf_rank"]) < int(existing["rrf_rank"])
            ):
                existing["rrf_rank"] = int(row["rrf_rank"])
            sources = list(existing.get("discovery_providers") or [])
            source = str(row.get("source") or "")
            if source and source not in sources:
                sources.append(source)
            existing["discovery_providers"] = sources
            known_queries = {
                (
                    str(value.get("query_id") or ""),
                    str(value.get("query") or "").casefold(),
                )
                for value in existing.get("discovery_queries") or []
                if isinstance(value, dict)
            }
            for value in row.get("discovery_queries") or []:
                if not isinstance(value, dict):
                    continue
                query_identity = (
                    str(value.get("query_id") or ""),
                    str(value.get("query") or "").casefold(),
                )
                if query_identity in known_queries:
                    continue
                known_queries.add(query_identity)
                existing.setdefault("discovery_queries", []).append(dict(value))
                if len(existing["discovery_queries"]) >= 16:
                    break
            if len(row.get("snippet") or "") > len(existing.get("snippet") or ""):
                existing["snippet"] = row["snippet"]

    # A URL can be discovered first through a sparse official sitemap row and
    # later through a search provider with a much richer snippet.  Relevance
    # belongs to the final merged candidate, not whichever provider happened
    # to arrive first.  Recompute it after deduplication so provider order
    # cannot leave a canonical page carrying stale term hits.
    for item in merged.values():
        relevance, route_candidate_score = best_route_relevance(item)
        item["query_relevance"] = relevance
        item["candidate_score"] = route_candidate_score

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
            {
                "exact": 3,
                "not_explicitly_scoped": 2,
                "source_identity_unavailable": 1,
                "unresolved": 1,
                "conflict": 0,
            }.get(
                str((item.get("object_alignment") or {}).get("relation") or ""),
                1,
            ),
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
            # RRF is a provider-neutral consensus tie-break after every
            # request/object/authority/topic feature. It cannot promote a
            # generic high-frequency page over a materially better match.
            float(item.get("rrf_score") or 0.0),
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


def _task_record_evidence_focus(
    query: str,
    task_plan: dict[str, Any] | None,
    task_record_id: str,
) -> str:
    """Bind the unchanged extractor prompt to the active atomic task_record."""

    task_record_id = str(task_record_id or "").strip()
    if not task_record_id:
        return str(query or "").strip()
    plan = task_plan if isinstance(task_plan, dict) else {}
    point = next(
        (
            item
            for item in task_records(plan)
            if isinstance(item, dict) and record_id(item) == task_record_id
        ),
        None,
    )
    if not isinstance(point, dict):
        return str(query or "").strip()
    return (record_question(point) or str(query or "").strip())[:1600]


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


def _cached_extraction_matches_scope(
    record: dict[str, Any],
    *,
    task_record_id: str,
    evidence_focus: str,
) -> bool:
    """Return whether a cached projection was extracted for this exact route."""

    requested_scope = str(task_record_id or "").strip()
    if "attempt_task_record_id" not in record:
        # Historical records did not persist extraction scope.  A task_record-bound
        # request can use its locator binding as a conservative compatibility
        # signal; an unbound request must re-extract because empty is not a
        # wildcard over those old task_record-specific projections.
        return bool(
            requested_scope
            and str(record.get("locator_task_record_id") or "").strip() == requested_scope
        )
    cached_scope = str(record.get("attempt_task_record_id") or "").strip()
    if cached_scope != requested_scope:
        return False
    cached_focus = " ".join(str(record.get("locator_query") or "").split()).casefold()
    requested_focus = " ".join(str(evidence_focus or "").split()).casefold()
    return bool(cached_focus and requested_focus and cached_focus == requested_focus)


def _compact_page(
    query: str,
    candidate: dict[str, Any],
    fetched: dict[str, Any],
    llm: LLMClient,
    task_id: str,
    task_plan: dict[str, Any] | None = None,
    task_record_id: str = "",
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
    evidence_focus = _task_record_evidence_focus(query, task_plan, task_record_id)
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
        "source_object": dict(evidence.get("source_object") or {}),
        "object_alignment": dict(candidate.get("object_alignment") or {}),
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

    rwkv_bound_task_record_ids = list(
        dict.fromkeys(
            str(task_record_id).strip()
            for candidate_row in (evidence.get("candidates") or [])
            if isinstance(candidate_row, dict)
            for task_record_id in (candidate_row.get("task_record_ids") or [])
            if str(task_record_id).strip()
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
        "source_object": dict(page_evidence.get("source_object") or {}),
        "object_alignment": dict(page_evidence.get("object_alignment") or {}),
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
        "task_record_ids": rwkv_bound_task_record_ids[:8],
        # The Planner-selected point scopes the retrieval attempt and the
        # extractor prompt; it is not evidence that every fetched page proves
        # that point. Only RWKV chunk bindings enter ``task_record_ids`` above.
        "attempt_task_record_id": task_record_id,
        "locator_task_record_id": (
            rwkv_bound_task_record_ids[0] if len(rwkv_bound_task_record_ids) == 1 else ""
        ),
        "task_record_binding_origin": (
            "rwkv_chunk_binding"
            if any(
                candidate_row.get("task_record_ids")
                for candidate_row in (evidence.get("candidates") or [])
                if isinstance(candidate_row, dict)
            )
            else "unbound"
        ),
        "locator_query": evidence_focus,
        "candidate_rank": candidate.get("candidate_rank"),
        "discovery_providers": candidate.get("discovery_providers") or [],
        "discovery_queries": [
            dict(value)
            for value in candidate.get("discovery_queries") or []
            if isinstance(value, dict)
        ][:16],
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
        "contract": MODEL_EXTRACTION_DIAGNOSTICS_CONTRACT,
        "complete": not rows,
        "degraded_page_count": len(rows),
        "unresolved_chunk_count": sum(int(row["unresolved_chunk_count"]) for row in rows),
        "transport_error_count": sum(int(row["transport_error_count"]) for row in rows),
        "recovered_chunk_count": recovered_chunks,
        "pages": rows[:16],
    }


def _recent_executed_queries(
    retrieval: Any,
    *,
    history_limit: int = 8,
    query_limit: int = 32,
) -> list[str]:
    """Flatten Planner and query expansion lanes into next-round query history."""

    history = getattr(retrieval, "query_history", []) if retrieval is not None else []
    recent: list[str] = []
    seen: set[str] = set()
    for item in list(history)[-max(1, int(history_limit)) :]:
        if not isinstance(item, dict):
            continue
        values: list[Any] = [item.get("query")]
        values.extend(
            row.get("query")
            for row in item.get("executed_queries") or []
            if isinstance(row, dict)
        )
        for value in values:
            query_value = " ".join(str(value or "").split()).strip()[:500]
            marker = query_value.casefold()
            if not query_value or marker in seen:
                continue
            seen.add(marker)
            recent.append(query_value)
    return recent[-max(1, int(query_limit)) :]


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
- Parameters: query (one concise search query or one complete http/https URL), max_results (optional output-size hint). The backend independently over-recalls provider candidates and may fetch up to its configured evidence budget.
- For a search phrase, the backend may ask RWKV for a few Task-Record-aware complementary queries and execute them concurrently; supply the best primary query instead of manually listing cosmetic variants. A complete URL is fetched directly and is never expanded.
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
    task_record_id = str(kwargs.get("task_record_id") or "").strip()
    original_goal = str(kwargs.get("original_goal") or query).strip()
    task_record_count = len(task_records(task_plan, fallback_query=original_goal))
    constraint_query = (
        _task_record_evidence_focus(query, task_plan, task_record_id)
        if task_record_id and task_record_count > 1
        else original_goal
    )
    scoped_domains = required_domains_for_task_record(
        effective_task_plan,
        task_record_id,
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

    direct_url = _extract_direct_url(query) or _extract_single_goal_url(
        kwargs.get("original_goal")
    )
    evidence_ledger_snapshot: dict[str, Any] = {}
    retrieval = getattr(agent_state, "retrieval", None)
    evidence_ledger = (
        getattr(retrieval, "evidence_ledger", None)
        if retrieval is not None
        else None
    )
    if evidence_ledger is not None and hasattr(evidence_ledger, "snapshot"):
        try:
            evidence_ledger_snapshot = evidence_ledger.snapshot(max_spans_per_record=0)
        except Exception:
            evidence_ledger_snapshot = {}
    recent_queries = _recent_executed_queries(retrieval)
    query_plan_enabled = bool(
        DATA_PIPELINE.get("retrieval_query_plan_enabled", True)
        and kwargs.get("agentic_tool_loop")
        and not direct_url
    )
    if query_plan_enabled:
        runtime_metadata = getattr(agent_state, "run_metadata", {})
        runtime_metadata = runtime_metadata if isinstance(runtime_metadata, dict) else {}
        runtime_context = " ".join(
            str(runtime_metadata.get(key) or "").strip()
            for key in ("current_utc_datetime", "current_utc_date")
            if str(runtime_metadata.get(key) or "").strip()
        )
        retrieval_query_plan = generate_retrieval_query_plan(
            query,
            original_goal,
            effective_task_plan,
            LLMClient(),
            task_record_id=task_record_id,
            evidence_ledger_snapshot=evidence_ledger_snapshot,
            recent_queries=recent_queries,
            runtime_context=runtime_context,
        )
    else:
        retrieval_query_plan = single_retrieval_query_plan(
            query,
            task_record_id=task_record_id,
            status="direct_url" if direct_url else "disabled",
        )
    query_lanes = [
        dict(item)
        for item in retrieval_query_plan.get("queries") or []
        if isinstance(item, dict) and str(item.get("query") or "").strip()
    ] or single_retrieval_query_plan(query, task_record_id=task_record_id)["queries"]
    retrieval_query_plan_view = public_retrieval_query_plan(retrieval_query_plan)
    append_task_event(
        task_id,
        "retrieval_query_plan",
        phase="DISCOVERY",
        action="web_search",
        query=query,
        task_record_id=task_record_id,
        status=retrieval_query_plan.get("status", ""),
        expanded=bool(retrieval_query_plan.get("expanded")),
        queries=query_lanes,
        attempts=int(retrieval_query_plan.get("attempts") or 0),
        error=str(retrieval_query_plan.get("error") or "")[:1000],
        raw_model_output=str(retrieval_query_plan.get("raw_model_output") or ""),
        prompt=str(retrieval_query_plan.get("prompt") or ""),
        decision_owner="rwkv" if query_plan_enabled else "backend",
    )

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
    # Provider recall, fused candidate-pool size and fetched-page count are
    # separate resource budgets.  A model-authored max_results=1 must not
    # discard the correct page before body retrieval; it remains an audit and
    # output-size hint rather than a provider-admission gate.
    max_candidates = fetch_limit
    provider_result_limit = max(
        max_candidates,
        _pipeline_concurrency("web_provider_result_limit", 20, maximum=50),
    )
    candidate_pool_limit = _pipeline_concurrency(
        "web_candidate_pool_limit", 48, maximum=100
    )
    rrf_k = _pipeline_concurrency("web_rrf_k", 60, maximum=1000)
    rrf_shadow_enabled = bool(DATA_PIPELINE.get("web_enable_rrf_shadow", True))
    candidate_ranking_mode = str(
        DATA_PIPELINE.get("web_candidate_ranking_mode", "legacy") or "legacy"
    ).strip().casefold()
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
            "model_requested_result_limit": requested_candidates,
            "provider_result_limit": provider_result_limit,
            "candidate_pool_limit": candidate_pool_limit,
            "fetch_limit": fetch_limit,
            "rrf_k": rrf_k,
            "candidate_ranking_mode": candidate_ranking_mode,
            "rrf_shadow_only": candidate_ranking_mode != "hybrid_rrf",
            "max_pages": max_pages,
            "context_length": get_llm_context_length(),
            "chunk_mode": DATA_PIPELINE.get("web_chunk_mode", "adaptive"),
            "single_pass_threshold_tokens": DATA_PIPELINE.get("web_chunk_single_pass_tokens", 7000),
            "chunk_target_tokens": DATA_PIPELINE.get("web_chunk_tokens", 4096),
            "chunk_max_tokens": DATA_PIPELINE.get("web_chunk_max_tokens", 2400),
            "page_evidence_concurrency": DATA_PIPELINE.get("web_page_evidence_concurrency", 8),
            "retrieval_query_plan_enabled": query_plan_enabled,
            "retrieval_query_plan_count": len(query_lanes),
            "retrieval_query_plan_concurrency": DATA_PIPELINE.get("retrieval_query_plan_concurrency", 4),
        },
    )

    def run_keyless(lane: dict[str, Any]) -> dict[str, Any]:
        return _annotate_retrieval_query_result(
            search_web_keyless(
                str(lane.get("query") or query),
                max_results=provider_result_limit,
                fetch_pages=0,
                task_id=task_id,
                agentic_tool_loop=True,
                constraint_query=constraint_query,
            ),
            lane,
            adapter="web.keyless",
        )

    def run_tavily(lane: dict[str, Any]) -> dict[str, Any]:
        return _annotate_retrieval_query_result(
            search_web_tavily(
                str(lane.get("query") or query),
                max_results=provider_result_limit,
                search_depth="advanced",
                topic="general",
                task_id=task_id,
            ),
            lane,
            adapter="web.tavily",
        )

    def run_official_site(lane: dict[str, Any]) -> dict[str, Any]:
        return _annotate_retrieval_query_result(
            discover_official_urls(
                str(lane.get("query") or query),
                source_policy.get("required_domains") or [],
                max_results=provider_result_limit,
                prefer_recent=_requests_recency(original_goal),
                constraint_query=constraint_query,
            ),
            lane,
            adapter="official site adapter",
        )

    def run_model_domain_hypothesis(lane: dict[str, Any]) -> dict[str, Any]:
        return _annotate_retrieval_query_result(
            discover_official_urls(
                str(lane.get("query") or query),
                model_domain_hypotheses,
                max_results=provider_result_limit,
                prefer_recent=_requests_recency(original_goal),
                constraint_query=constraint_query,
            ),
            lane,
            adapter="official site adapter",
        )

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
        provider_jobs: list[
            tuple[Any, dict[str, Any], str]
        ] = []
        primary_lane = query_lanes[0]
        # Sitemap discovery is domain-scoped and can be expensive; run it once
        # for the primary lane. General providers receive every query lane.
        if source_policy.get("required") and source_policy.get("required_domains"):
            provider_jobs.append((run_official_site, primary_lane, "official_site"))
        elif model_domain_hypotheses:
            provider_jobs.append(
                (run_model_domain_hypothesis, primary_lane, "official_site_hypothesis")
            )
        for lane in query_lanes:
            provider_jobs.extend(
                (
                    (run_keyless, lane, "keyless"),
                    (run_tavily, lane, "tavily"),
                )
            )
        provider_results: list[dict[str, Any]] = [
            {"status": "error", "provider_errors": ["provider did not complete"], "results": []}
            for _ in provider_jobs
        ]
        query_concurrency = _pipeline_concurrency(
            "retrieval_query_plan_concurrency", 4, maximum=6
        )
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(
                max(
                    _pipeline_concurrency("web_provider_concurrency", 2),
                    min(len(query_lanes), query_concurrency) * 2,
                ),
                len(provider_jobs),
            )
        )
        futures = {
            submit_with_context(pool, job, lane): index
            for index, (job, lane, _adapter) in enumerate(provider_jobs)
        }
        cancelled = False
        try:
            for future in concurrent.futures.as_completed(futures, timeout=task_wait_timeout()):
                index = futures[future]
                try:
                    provider_results[index] = future.result()
                except Exception as exc:
                    _job, lane, adapter = provider_jobs[index]
                    provider_results[index] = _annotate_retrieval_query_result(
                        {
                            "status": "error",
                            "provider_errors": [f"{type(exc).__name__}: {exc}"],
                            "results": [],
                        },
                        lane,
                        adapter=adapter,
                    )
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

        # Preserve a narrow legacy slice for audit comparison, while the live
        # hybrid path keeps the full provider recall until after fusion and
        # task-aware ranking.
        shadow_provider_results = [dict(result) for result in provider_results]
        legacy_provider_results = [
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
                legacy_provider_results,
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
            if str(
                provider_result.get("backend_provider")
                or provider_result.get("provider")
                or ""
            ) != "official site adapter":
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
                "ranking_stream": result.get("ranking_stream") or "",
                "query_id": result.get("retrieval_query_id") or "",
                "task_record_id": result.get("retrieval_query_task_record_id") or "",
                "intent": result.get("retrieval_query_intent") or "",
                "query": result.get("retrieval_query_text") or result.get("query") or "",
                "status": result.get("status") or "error",
                "count": int(result.get("count") or len(result.get("results") or [])),
                "errors": result.get("provider_errors") or [],
            }
            for result in shadow_provider_results
        ]
        legacy_candidates = _merge_candidates(
            query,
            legacy_provider_results,
            limit=max_candidates,
            task_plan=effective_task_plan,
            constraint_query=constraint_query,
            policy_query=original_goal,
        )
        # Hybrid ranking is a live retrieval stage, not a side effect of trace
        # logging. Disabling the shadow/audit payload must never silently turn
        # RRF off while the configured active ranking mode still requires it.
        rrf_required = rrf_shadow_enabled or candidate_ranking_mode == "hybrid_rrf"
        candidate_pool_shadow = (
            rrf_fuse(
                shadow_provider_results,
                rrf_k=rrf_k,
                pool_limit=candidate_pool_limit,
            )
            if rrf_required
            else []
        )
        rrf_by_identity = {
            retrieval_url_identity(str(item.get("url") or "")): item
            for item in candidate_pool_shadow
        }
        fused_provider_results = []
        for result in shadow_provider_results:
            fused_rows = []
            for raw in result.get("results") or []:
                if not isinstance(raw, dict):
                    continue
                item = dict(raw)
                fused = rrf_by_identity.get(
                    retrieval_url_identity(str(item.get("url") or "")),
                    {},
                )
                if fused:
                    item.update(
                        {
                            "provider_ranks": dict(fused.get("provider_ranks") or {}),
                            "rrf_score": float(fused.get("rrf_score") or 0.0),
                            "rrf_rank": fused.get("rrf_rank"),
                        }
                    )
                fused_rows.append(item)
            fused_provider_results.append({**dict(result), "results": fused_rows})

        if candidate_ranking_mode == "hybrid_rrf":
            candidate_pool = _merge_candidates(
                query,
                fused_provider_results,
                limit=candidate_pool_limit,
                task_plan=effective_task_plan,
                constraint_query=constraint_query,
                policy_query=original_goal,
            )
            for rank, item in enumerate(candidate_pool, start=1):
                item["candidate_rank"] = rank
            # Novelty is not known until the shared evidence store is read.
            # Keep the complete fused pool here; applying the fetch slice now
            # would let cached Top-N URLs permanently hide a novel rank N+1.
            candidates = list(candidate_pool)
        else:
            candidate_pool = list(legacy_candidates)
            candidates = list(legacy_candidates)
        legacy_rank_by_url = {
            retrieval_url_identity(str(item.get("url") or "")): int(
                item.get("candidate_rank") or index
            )
            for index, item in enumerate(legacy_candidates, start=1)
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
            active_ranking=candidate_ranking_mode,
            provider_result_limit=provider_result_limit,
            active_provider_slice=(
                provider_result_limit
                if candidate_ranking_mode == "hybrid_rrf"
                else max_candidates
            ),
            candidate_pool_limit=candidate_pool_limit,
            fetch_limit=max_pages,
            rrf_k=rrf_k,
            providers=[
                {
                    "provider": result.get("provider") or "unknown",
                    "ranking_stream": result.get("ranking_stream") or "",
                    "query_id": result.get("retrieval_query_id") or "",
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
                for item in legacy_candidates
            ],
            rrf_candidates=candidate_pool_shadow,
            hybrid_candidates=[
                {
                    "rank": item.get("candidate_rank"),
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "rrf_rank": item.get("rrf_rank"),
                    "rrf_score": item.get("rrf_score"),
                    "provider_ranks": item.get("provider_ranks") or {},
                    "discovery_queries": item.get("discovery_queries") or [],
                }
                for item in candidates
            ],
            affects_fetch=candidate_ranking_mode == "hybrid_rrf",
            affects_rwkv=candidate_ranking_mode == "hybrid_rrf",
        )

    # Authority remains auditable metadata and a soft ranking signal.  A
    # model- or adapter-derived domain hypothesis must never suppress every
    # fetched page.  Explicit user URLs are already handled by direct_url.

    # A follow-up research round must add novelty to the shared evidence
    # store. Reuse is handled from the store; do not refetch the same page as
    # a second recovery path.
    seen_urls = _shared_source_urls(agent_state)
    discovered_candidates = [dict(item) for item in candidates]
    per_domain_fetch_limit = _pipeline_concurrency(
        "web_per_domain_fetch_limit", 3, maximum=16
    )
    processing_candidates, novel_candidates = _admit_cached_and_novel_candidates(
        discovered_candidates,
        seen_urls,
        limit=max_pages,
        per_domain_limit=per_domain_fetch_limit,
        priority_hosts=[
            *(source_policy.get("required_domains") or []),
            *(model_domain_hypotheses or []),
            # Hosts the request itself names via explicit URLs are an
            # upstream (user/model) decision, not a code judgment.
            *re.findall(
                r"https?://([A-Za-z0-9.-]+)",
                f"{original_goal or ''} {query or ''} {constraint_query or ''}",
            ),
        ],
    )
    reused_records = []
    retrieval = getattr(agent_state, "retrieval", None)
    stored_sources = getattr(retrieval, "sources", {}) if retrieval is not None else {}
    task_record_sources = (
        getattr(retrieval, "sources_by_task_record", {}).get(task_record_id, {})
        if retrieval is not None and task_record_id
        else {}
    )
    if stored_sources:
        stored_by_url = {
            retrieval_url_identity(str(item.get("url") or "")): item
            for item in stored_sources.values()
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        }
        task_record_by_url = {
            retrieval_url_identity(str(item.get("url") or "")): item
            for item in task_record_sources.values()
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        }
        # Preserve candidate ranking while projecting already fetched URLs
        # back to their cached page bodies.  Cached and novel candidates are
        # two inputs to the same round, not mutually exclusive branches.
        for candidate in processing_candidates:
            normalized = retrieval_url_identity(str(candidate.get("url") or ""))
            item = task_record_by_url.get(normalized) or stored_by_url.get(normalized)
            if not isinstance(item, dict):
                continue
            reused_records.append(dict(item))
    candidates = novel_candidates
    if not reused_records and not novel_candidates and seen_urls:
        result = {
            "status": "no_new_evidence",
            "real_network": False,
            "provider": "evidence_store",
            "retrieval_role": "discovery",
            "query": query,
            "retrieval_query_plan": retrieval_query_plan_view,
            "count": 0,
            "candidate_count": len(discovered_candidates),
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
                    "provider_ranks",
                    "rrf_rank",
                    "rrf_score",
                    "discovery_score",
                    "query_relevance",
                    "discovery_providers",
                    "discovery_queries",
                    "authority",
                    "source_object",
                    "object_alignment",
                )
            }
            for item in discovered_candidates
        ],
    )

    if not candidates and not reused_records:
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
                "retrieval_query_plan": retrieval_query_plan_view,
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

    if reused_records and not novel_candidates:
        cache_evidence_focus = _task_record_evidence_focus(
            constraint_query,
            effective_task_plan,
            task_record_id,
        )
        bound_records = [
            dict(item)
            for item in reused_records
            if _cached_extraction_matches_scope(
                item,
                task_record_id=task_record_id,
                evidence_focus=cache_evidence_focus,
            )
        ]
        foreign_records = [
            dict(item)
            for item in reused_records
            if not _cached_extraction_matches_scope(
                item,
                task_record_id=task_record_id,
                evidence_focus=cache_evidence_focus,
            )
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
                    constraint_query,
                    candidate,
                    _cached_page_result(record),
                    llm,
                    task_id,
                    task_plan=effective_task_plan,
                    task_record_id=task_record_id,
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
            "retrieval_query_plan": retrieval_query_plan_view,
            "count": len(bound_records),
            "candidate_count": len(discovered_candidates),
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
                "task_record-scoped chunk re-extraction from the shared page cache; no duplicate network fetch"
                if foreign_records
                else "existing task_record-scoped evidence projection; no duplicate network fetch"
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
            task_record_id=task_record_id,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    mixed_bound_records: list[dict[str, Any]] = []
    mixed_foreign_records: list[dict[str, Any]] = []
    if reused_records:
        cache_evidence_focus = _task_record_evidence_focus(
            constraint_query,
            effective_task_plan,
            task_record_id,
        )
        mixed_bound_records = [
            dict(item)
            for item in reused_records
            if _cached_extraction_matches_scope(
                item,
                task_record_id=task_record_id,
                evidence_focus=cache_evidence_focus,
            )
        ]
        mixed_foreign_records = [
            dict(item)
            for item in reused_records
            if not _cached_extraction_matches_scope(
                item,
                task_record_id=task_record_id,
                evidence_focus=cache_evidence_focus,
            )
        ]

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

    network_fetched_count = len(fetched)
    fetched.extend(
        (
            annotate_source(
                dict(record),
                original_goal,
                {"task_plan": effective_task_plan},
            ),
            _cached_page_result(record),
        )
        for record in mixed_foreign_records
    )
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
        # When Planner omits the optional task_record_id, extraction must retain
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
            task_record_id=task_record_id,
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
    records: list[dict[str, Any]] = [
        annotate_freshness(item, freshness_policy)
        for item in mixed_bound_records
    ]
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
        "retrieval_query_plan": retrieval_query_plan_view,
        "count": len(records),
        "candidate_count": len(discovered_candidates),
        "fetched_count": network_fetched_count,
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
                for key in (
                    "candidate_rank",
                    "title",
                    "url",
                    "source",
                    "candidate_score",
                    "discovery_providers",
                    "discovery_queries",
                    "source_object",
                    "object_alignment",
                )
            }
            for item in discovered_candidates
        ],
        "candidate_pool_shadow": {
            "enabled": rrf_shadow_enabled,
            "active_ranking": candidate_ranking_mode,
            "affects_fetch": candidate_ranking_mode == "hybrid_rrf",
            "affects_rwkv": candidate_ranking_mode == "hybrid_rrf",
            "model_requested_result_limit": requested_candidates,
            "provider_result_limit": provider_result_limit,
            "active_provider_slice": max_candidates,
            "candidate_pool_limit": candidate_pool_limit,
            "fetch_limit": max_pages,
            "rrf_k": rrf_k,
            "candidates": candidate_pool_shadow,
        },
        "page_evidence": evidence_pages,
        "evidence_ready": bool(usable_records),
        "reused_sources": bool(reused_records),
        "reextracted_cached_sources": bool(mixed_foreign_records),
        "novel_source_count": len(
            [
                item
                for item in records
                if retrieval_url_identity(str(item.get("url") or "")) not in seen_urls
            ]
        ),
        "usable_evidence_count": len(usable_records),
        "evidence_missing_count": missing_page_count + (len(records) - len(usable_records)),
        "model_extraction": extraction_diagnostics,
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "freshness_policy": freshness_policy,
        "source_resolution": source_resolution_payload(),
        "evidence_policy": (
            "mixed cached-scope re-extraction and novel page retrieval; candidate URLs and Markdown chunks remain untrusted evidence"
            if reused_records
            else "candidate URLs and Markdown chunk facts are untrusted evidence; the model decides whether to search again or summarize"
        ),
    }
    append_task_event(
        task_id,
        "web_search_stage",
        phase="EXTRACTION",
        action="web_search",
        stage="complete",
        query=query,
        status=result["status"],
        candidate_count=len(discovered_candidates),
        fetched_count=network_fetched_count,
        evidence_count=len(records),
        page_evidence=evidence_pages,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
