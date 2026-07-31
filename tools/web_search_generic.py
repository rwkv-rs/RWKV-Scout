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
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from agent.page_evidence import extract_single_page_evidence
from clients.llm_client import LLMClient
from config import DATA_PIPELINE, get_llm_context_length
from tools.registry import ToolRegistry
from tools.web_search_keyless import _is_search_result_url, search_web_keyless
from tools.web_search_tavily import search_web_tavily
from utils.task_events import append_task_event
from utils.evidence_quality import MIN_PAGE_BODY_CHARS, has_substantive_evidence
from utils.web_retrieval import candidate_score, normalize_url


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


def _merge_candidates(query: str, provider_results: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Deduplicate and rank candidates without classifying the task domain."""

    merged: dict[str, dict[str, Any]] = {}
    for result in provider_results:
        for item in result.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = normalize_url(str(item.get("url") or ""))
            if not url or _is_search_result_url(url):
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
            row["candidate_score"] = candidate_score(query, row)
            row["discovery_providers"] = [row["source"]]
            existing = merged.get(url)
            if existing is None:
                merged[url] = row
                continue
            existing["candidate_score"] = max(
                float(existing.get("candidate_score") or 0),
                float(row.get("candidate_score") or 0),
            )
            sources = list(existing.get("discovery_providers") or [])
            source = str(row.get("source") or "")
            if source and source not in sources:
                sources.append(source)
            existing["discovery_providers"] = sources
            if len(row.get("snippet") or "") > len(existing.get("snippet") or ""):
                existing["snippet"] = row["snippet"]

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("candidate_score") or 0),
            len(item.get("discovery_providers") or []),
        ),
        reverse=True,
    )
    # Keep a bounded amount of domain diversity.  This is a fetch-budget
    # boundary, not a source/domain policy for the user's question.
    selected: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    for item in ranked:
        domain = _host(item.get("url", ""))
        if domain_counts.get(domain, 0) >= 3:
            continue
        item["candidate_rank"] = len(selected) + 1
        selected.append(item)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def _fetch_candidate(candidate: dict[str, Any], task_id: str) -> dict[str, Any]:
    from tools.web_search_keyless import fetch_web_url

    raw = fetch_web_url(
        candidate["url"],
        max_chars=20000,
        task_id=task_id,
        agentic_tool_loop=True,
    )
    return _parse_result(raw)


def _compact_page(
    query: str,
    candidate: dict[str, Any],
    fetched: dict[str, Any],
    llm: LLMClient,
    task_id: str,
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
    page_text = str(page.get("page_excerpt") or page.get("content") or "").strip()
    if len(page_text) < MIN_PAGE_BODY_CHARS:
        return None, {
            "url": url,
            "title": str(page.get("title") or candidate.get("title") or url),
            "status": "no_evidence",
            "page_chars": len(page_text),
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
            query=query,
            page=page,
            llm=llm,
            task_id=task_id,
            on_chunk=on_chunk,
        )
    except Exception as exc:
        evidence = {
            "status": "error",
            "url": url,
            "title": page.get("title") or url,
            "page_chars": len(page_text),
            "chunk_count": 0,
            "candidates": [],
            "compact_facts": "",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }

    compact_facts = str(evidence.get("compact_facts") or "").strip()
    source_excerpt = str(evidence.get("source_excerpt") or page_text[:14000]).strip()
    page_evidence = {
        "url": url,
        "title": str(page.get("title") or candidate.get("title") or url),
        "status": str(evidence.get("status") or "no_evidence"),
        "page_chars": int(evidence.get("page_chars") or 0),
        "chunk_count": int(evidence.get("chunk_count") or 0),
        "chunk_window_tokens": int(evidence.get("chunk_window_tokens") or 0),
        "candidate_count": len(evidence.get("candidates") or []),
        "parallel_candidate": evidence.get("parallel_candidate") or {},
        "errors": evidence.get("errors") or [],
    }
    if not source_excerpt:
        return None, page_evidence

    if not compact_facts:
        page_evidence["status"] = "ok"
        page_evidence["evidence_origin"] = "fetched_page_body"
        page_evidence["model_extraction_status"] = "no_evidence"
    else:
        page_evidence["evidence_origin"] = "fetched_page_body_with_model_locator"
        page_evidence["model_extraction_status"] = "ok"

    record = {
        "title": page_evidence["title"],
        "url": url,
        "snippet": str(candidate.get("snippet") or "")[:800],
        "source": str(candidate.get("source") or "web"),
        "candidate_score": candidate.get("candidate_score", 0.0),
        "content": source_excerpt[:14000],
        "page_excerpt": source_excerpt[:14000],
        "source_excerpt": source_excerpt[:14000],
        "model_extracted_facts": compact_facts[:14000],
        "untrusted_content": True,
        "evidence_origin": "fetched_page_body",
        "evidence_kind": "page_body",
        "evidence_boundary": "page_body_only",
        "body_verified": True,
        "content_sha256": hashlib.sha256(source_excerpt[:14000].encode("utf-8")).hexdigest(),
        "source_locator": {
            "type": "page_excerpt",
            "char_start": 0,
            "char_end": len(source_excerpt[:14000]),
        },
        "evidence_status": page_evidence["status"],
        "chunk_count": page_evidence["chunk_count"],
        "chunk_candidates": evidence.get("candidates") or [],
        "source_chunks": evidence.get("source_chunks") or [],
        "candidate_rank": candidate.get("candidate_rank"),
        "discovery_providers": candidate.get("discovery_providers") or [],
    }
    if not has_substantive_evidence(record):
        page_evidence["status"] = "no_evidence"
        page_evidence.setdefault("errors", []).append("extracted body did not meet the substantive evidence threshold")
        return None, page_evidence
    return record, page_evidence


@ToolRegistry.register(
    name="web_search",
    phase="ALL",
    plugin="web.generic",
    capabilities=("url_discovery", "candidate_admission", "page_fetch", "markdown", "chunk_evidence"),
    retrieval_role="discovery",
    model_visible=True,
    category="retrieval",
    signature="""[Tool] web_search
- Function: perform one bounded general-web retrieval transaction.
- Parameters: query (one concise search query or one complete http/https URL).
- Pipeline: discovery, candidate admission/ranking, bounded page fetch, Markdown extraction, adaptive evidence extraction (cleaned pages up to the configured threshold stay single-pass; longer pages are chunked).
- Provider selection, URL fetching, page cleaning and chunk aggregation are internal backend steps; do not invent a provider-specific tool name.
- The result is evidence only. It is not a final answer and does not decide whether the user's task is complete.""",
)
def web_search(query: str, **kwargs: Any) -> str:
    query = " ".join(str(query or "").split()).strip()
    task_id = str(kwargs.get("task_id") or "")
    if not query:
        return json.dumps({"status": "error", "message": "query is empty", "results": []}, ensure_ascii=False)

    max_candidates = 8
    max_pages = 4
    append_task_event(
        task_id,
        "web_search_stage",
        phase="DISCOVERY",
        action="web_search",
        stage="start",
        query=query,
        budget={
            "max_candidates": max_candidates,
            "max_pages": max_pages,
            "context_length": get_llm_context_length(),
            "chunk_mode": DATA_PIPELINE.get("web_chunk_mode", "adaptive"),
            "single_pass_threshold_tokens": DATA_PIPELINE.get("web_chunk_single_pass_tokens", 7000),
            "chunk_target_tokens": DATA_PIPELINE.get("web_chunk_tokens", 4096),
            "chunk_max_tokens": DATA_PIPELINE.get("web_chunk_max_tokens", 4096),
        },
    )

    def run_keyless() -> dict[str, Any]:
        return _parse_result(
            search_web_keyless(
                query,
                max_results=max_candidates,
                fetch_pages=0,
                task_id=task_id,
                agentic_tool_loop=True,
            )
        )

    def run_tavily() -> dict[str, Any]:
        return _parse_result(
            search_web_tavily(
                query,
                max_results=max_candidates,
                search_depth="advanced",
                topic="general",
                task_id=task_id,
            )
        )

    direct_url = _extract_direct_url(query) or _extract_single_goal_url(kwargs.get("original_goal"))
    if direct_url:
        provider_results: list[dict[str, Any]] = []
        provider_statuses = [
            {"provider": "direct_url", "status": "ok", "count": 1, "errors": []}
        ]
        candidates = [_direct_candidate(direct_url)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_keyless), pool.submit(run_tavily)]
            provider_results = [future.result() for future in futures]

        provider_statuses = [
            {
                "provider": result.get("provider") or "unknown",
                "status": result.get("status") or "error",
                "count": int(result.get("count") or len(result.get("results") or [])),
                "errors": result.get("provider_errors") or [],
            }
            for result in provider_results
        ]
        candidates = _merge_candidates(query, provider_results, limit=max_candidates)
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
                for key in ("candidate_rank", "title", "url", "snippet", "source", "candidate_score", "discovery_providers")
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
                "evidence_policy": "no candidate URL was admitted; the model must decide whether to refine the query",
            },
            ensure_ascii=False,
            indent=2,
        )

    llm = LLMClient()
    selected = candidates[:max_pages]
    fetched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(selected))) as pool:
        future_map = {pool.submit(_fetch_candidate, item, task_id): item for item in selected}
        for future in concurrent.futures.as_completed(future_map):
            candidate = future_map[future]
            try:
                fetched.append((candidate, future.result()))
            except Exception as exc:
                fetched.append((candidate, {"status": "error", "message": f"{type(exc).__name__}: {exc}", "results": []}))

    fetched.sort(key=lambda item: int(item[0].get("candidate_rank") or 10**6))
    evidence_pages: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for candidate, fetched_result in fetched:
        record, page_evidence = _compact_page(query, candidate, fetched_result, llm, task_id)
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
            records.append(record)

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
        "page_evidence": evidence_pages,
        "evidence_ready": bool(usable_records),
        "usable_evidence_count": len(usable_records),
        "evidence_missing_count": missing_page_count + (len(records) - len(usable_records)),
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
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
