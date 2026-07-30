"""Model-owned Tavily discovery search.

The tool returns URLs and short snippets only. The model must choose one URL
and call ``fetch_web_url``; page bodies are handled by the single-page
chunk/candidate pipeline in the orchestrator.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from config import get_search_api_keys
from tools.registry import ToolRegistry
from utils.network_fetch import create_network_session


@ToolRegistry.register(
    name="search_web_tavily",
    phase="ALL",
    plugin="web.tavily",
    capabilities=("url_discovery", "web_search"),
    retrieval_role="discovery",
    signature="""[Tool] search_web_tavily
- Function: search the public web with Tavily and return candidate URLs plus short snippets.
- Parameters: query, max_results (1-10), search_depth (basic|advanced), topic (general|news), time_range (optional).
- Safety: discovery only; choose one returned URL and call fetch_web_url for page evidence.""",
)
def search_web_tavily(
    query: str,
    max_results: int = 8,
    search_depth: str = "advanced",
    topic: str = "general",
    time_range: str | None = None,
    working_memory: dict | None = None,
    agent_state=None,
    **kwargs: Any,
) -> str:
    del working_memory, agent_state, kwargs
    query = " ".join(str(query or "").split())
    if not query:
        return json.dumps({"status": "error", "message": "query is empty", "results": []}, ensure_ascii=False)

    api_keys = get_search_api_keys("tavily")
    if not api_keys:
        return json.dumps(
            {
                "status": "error",
                "provider": "Tavily API",
                "message": "TAVILY_API_KEY is not configured",
                "results": [],
                "provider_errors": ["missing TAVILY_API_KEY"],
            },
            ensure_ascii=False,
        )

    normalized_depth = str(search_depth or "advanced").strip().lower()
    if normalized_depth not in {"basic", "advanced"}:
        normalized_depth = "advanced"
    normalized_topic = str(topic or "general").strip().lower()
    if normalized_topic not in {"general", "news", "finance"}:
        normalized_topic = "general"
    normalized_time_range = str(time_range or "").strip().lower()
    if normalized_time_range not in {"day", "week", "month", "year"}:
        normalized_time_range = ""
    params: dict[str, Any] = {
        "query": query,
        "search_depth": normalized_depth,
        "max_results": max(1, min(int(max_results or 8), 10)),
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "topic": normalized_topic,
    }
    if normalized_time_range:
        params["time_range"] = normalized_time_range

    payload: dict[str, Any] | None = None
    provider_errors: list[str] = []
    for api_key in api_keys:
        try:
            # The desktop Windows process may inherit a proxy that aborts HTTPS
            # connections to api.tavily.com. Use a direct session, like the
            # local RWKV bridge client does, and keep the key in the header.
            session = create_network_session(
                {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
            )
            response = session.post(
                "https://api.tavily.com/search",
                json=params,
                timeout=(15, 45),
            )
            response.raise_for_status()
            payload = response.json() or {}
            break
        except Exception as exc:
            response_error = getattr(exc, "response", None)
            response_body = ""
            if response_error is not None:
                try:
                    response_body = str(response_error.text or "").strip()[:300]
                except Exception:
                    response_body = ""
            detail = f"{type(exc).__name__}: {exc}"
            if response_body:
                detail = f"{detail}; response_body={response_body}"
            provider_errors.append(detail[:800])

    if payload is None:
        return json.dumps(
            {
                "status": "error",
                "provider": "Tavily API",
                "query": query,
                "results": [],
                "provider_attempts": len(api_keys),
                "provider_errors": provider_errors,
            },
            ensure_ascii=False,
        )

    rows: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict) or not str(item.get("url") or "").strip():
            continue
        rows.append(
            {
                "title": str(item.get("title") or item.get("url") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "snippet": str(item.get("content") or item.get("snippet") or "").strip()[:800],
                "source": "Tavily API",
                "score": item.get("score"),
                "published_date": item.get("published_date"),
                "page_excerpt": "",
                "untrusted_content": True,
            }
        )

    return json.dumps(
        {
            "status": "ok" if rows else "no_results",
            "real_network": True,
            "provider": "Tavily API",
            "query": query,
            "provider_query": query,
            "retrieved_at": datetime.now().isoformat(timespec="seconds"),
            "count": len(rows),
            "results": rows,
            "sources": [row["url"] for row in rows],
            "citation_refs": [
                {
                    "ref_id": f"TAVILY_{index}",
                    "title": row["title"],
                    "url": row["url"],
                    "source": "Tavily API",
                }
                for index, row in enumerate(rows, start=1)
            ],
            "provider_attempts": len(provider_errors) + 1,
            "provider_errors": provider_errors,
            "evidence_policy": "search results are discovery metadata; fetch one selected URL for page evidence",
        },
        ensure_ascii=False,
    )
