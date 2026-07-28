"""Provider-agnostic web retrieval for the model-owned search episode.

The model chooses whether to call ``web_search`` and supplies the query.  The
tool owns only the mechanical retrieval transaction: discovery, candidate
admission, page fetching, Markdown extraction, chunking and evidence merging.
It deliberately does not produce an answer or decide whether the user's goal
is complete.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from datetime import datetime
from typing import Any

from agent.page_evidence import extract_single_page_evidence
from clients.llm_client import LLMClient
from tools.registry import ToolRegistry
from tools.web_search_keyless import _is_search_result_url, search_web_keyless
from tools.web_search_tavily import search_web_tavily
from utils.task_events import append_task_event
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
            "page_chars": len(str(page.get("page_excerpt") or page.get("content") or "")),
            "chunk_count": 0,
            "candidates": [],
            "compact_facts": "",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }

    compact_facts = str(evidence.get("compact_facts") or "").strip()
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
    if not compact_facts:
        return None, page_evidence

    record = {
        "title": page_evidence["title"],
        "url": url,
        "snippet": str(candidate.get("snippet") or "")[:800],
        "source": str(candidate.get("source") or "web"),
        "content": compact_facts[:14000],
        "page_excerpt": compact_facts[:14000],
        "untrusted_content": True,
        "evidence_status": page_evidence["status"],
        "chunk_count": page_evidence["chunk_count"],
        "chunk_candidates": evidence.get("candidates") or [],
        "candidate_rank": candidate.get("candidate_rank"),
        "discovery_providers": candidate.get("discovery_providers") or [],
    }
    return record, page_evidence


@ToolRegistry.register(
    name="web_search",
    phase="ALL",
    plugin="web.generic",
    capabilities=("url_discovery", "candidate_admission", "page_fetch", "markdown", "chunk_evidence"),
    retrieval_role="discovery",
    signature="""[Tool] web_search
- Function: perform one bounded general-web retrieval transaction.
- Parameters: query (one concise search query).
- Pipeline: discovery, candidate admission/ranking, bounded page fetch, Markdown extraction, 2048-token chunk evidence.
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
        budget={"max_candidates": max_candidates, "max_pages": max_pages, "chunk_window_tokens": 2048},
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
        }
        for index, item in enumerate(records, start=1)
    ]
    result = {
        "status": "ok" if records else "no_evidence",
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
        "evidence_ready": bool(records),
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
