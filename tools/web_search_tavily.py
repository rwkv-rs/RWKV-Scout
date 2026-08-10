"""Model-owned Tavily discovery search.

The tool returns URLs and short snippets only. The model must choose one URL
and call ``fetch_web_url``; page bodies are handled by the single-page
chunk/candidate pipeline in the orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from config import get_search_api_keys, retire_search_api_key
from tools.registry import ToolRegistry
from utils.evidence_quality import clean_page_body
from utils.network_fetch import create_network_session


_HEALTH_LOCK = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}
_PERMANENT_KEY_FAILURES: dict[str, str] = {}
_KEY_CURSOR = 0


def _key_identity(api_key: str) -> str:
    """Return a non-secret process-local identity for provider health state."""

    return hashlib.sha256(api_key.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _ordered_key_entries(api_keys: list[str]) -> list[tuple[str, str, threading.Lock]]:
    """Rotate healthy keys and give each key a shared probe lock.

    The per-key lock prevents concurrent research cases from all probing the
    same exhausted credential.  Permanently failed keys are skipped, while a
    newly configured key is automatically eligible because it has a different
    identity.
    """

    global _KEY_CURSOR
    with _HEALTH_LOCK:
        entries: list[tuple[str, str, threading.Lock]] = []
        for api_key in api_keys:
            identity = _key_identity(api_key)
            if identity in _PERMANENT_KEY_FAILURES:
                continue
            entries.append(
                (api_key, identity, _KEY_LOCKS.setdefault(identity, threading.Lock()))
            )
        if not entries:
            return []
        offset = _KEY_CURSOR % len(entries)
        _KEY_CURSOR += 1
        return entries[offset:] + entries[:offset]


def _key_is_permanently_failed(identity: str) -> bool:
    with _HEALTH_LOCK:
        return identity in _PERMANENT_KEY_FAILURES


def _has_available_key(api_keys: list[str]) -> bool:
    with _HEALTH_LOCK:
        return any(
            _key_identity(api_key) not in _PERMANENT_KEY_FAILURES
            for api_key in api_keys
        )


def _mark_key_permanently_failed(identity: str, reason: str) -> None:
    with _HEALTH_LOCK:
        _PERMANENT_KEY_FAILURES[identity] = reason[:200]


def _permanent_provider_failure(exc: Exception, response_body: str) -> tuple[bool, str]:
    """Classify credential/account failures that another retry cannot fix."""

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    folded = f"{exc} {response_body}".casefold()
    quota_markers = (
        "usage limit",
        "quota exceeded",
        "quota exhausted",
        "plan's set usage limit",
        "plan limit",
        "insufficient credit",
        "credits exhausted",
        "payment required",
    )
    auth_markers = (
        "invalid api key",
        "invalid token",
        "unauthorized",
        "authentication failed",
    )
    if status_code in {402, 432} or any(marker in folded for marker in quota_markers):
        return True, "provider quota exhausted"
    if status_code == 401 or any(marker in folded for marker in auth_markers):
        return True, "provider credential rejected"
    return False, ""


def _reset_provider_health_for_tests() -> None:
    """Clear process health state; intended only for isolated unit tests."""

    global _KEY_CURSOR
    with _HEALTH_LOCK:
        _KEY_LOCKS.clear()
        _PERMANENT_KEY_FAILURES.clear()
        _KEY_CURSOR = 0


def _request_tavily(
    endpoint: str,
    params: dict[str, Any],
    *,
    timeout: tuple[int, int],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Execute one Tavily endpoint with the shared credential-health policy."""

    api_keys = get_search_api_keys("tavily")
    if not api_keys:
        return None, {
            "provider_attempts": 0,
            "removed_credentials": 0,
            "provider_disabled": True,
            "error_class": "provider_not_configured",
            "provider_errors": ["missing TAVILY_API_KEY"],
            "message": "TAVILY_API_KEY is not configured",
        }

    key_entries = _ordered_key_entries(api_keys)
    if not key_entries:
        return None, {
            "provider_attempts": 0,
            "removed_credentials": 0,
            "provider_disabled": True,
            "error_class": "provider_quota_or_auth_unavailable",
            "provider_errors": [
                "all configured Tavily credentials are unavailable for this process"
            ],
        }

    payload: dict[str, Any] | None = None
    provider_errors: list[str] = []
    provider_attempts = 0
    removed_credentials = 0
    for api_key, identity, key_lock in key_entries:
        with key_lock:
            if _key_is_permanently_failed(identity):
                continue
            try:
                session = create_network_session(
                    {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                )
                provider_attempts += 1
                response = session.post(
                    f"https://api.tavily.com/{str(endpoint).strip('/')}",
                    json=params,
                    timeout=timeout,
                )
                response.raise_for_status()
                value = response.json() or {}
                payload = value if isinstance(value, dict) else {}
                break
            except Exception as exc:
                response_error = getattr(exc, "response", None)
                response_body = ""
                if response_error is not None:
                    try:
                        response_body = str(response_error.text or "").strip()[:300]
                    except Exception:
                        response_body = ""
                permanent, reason = _permanent_provider_failure(exc, response_body)
                if permanent:
                    _mark_key_permanently_failed(identity, reason)
                    try:
                        if retire_search_api_key("tavily", api_key):
                            removed_credentials += 1
                    except OSError:
                        provider_errors.append(
                            "permanently unavailable credential could not be removed from local storage"
                        )
                detail = f"{type(exc).__name__}: {exc}"
                if response_body:
                    detail = f"{detail}; response_body={response_body}"
                provider_errors.append(detail[:800])

    all_keys_unavailable = not _has_available_key(api_keys)
    return payload, {
        "provider_attempts": provider_attempts,
        "removed_credentials": removed_credentials,
        "provider_disabled": all_keys_unavailable,
        "error_class": (
            "provider_quota_or_auth_unavailable"
            if all_keys_unavailable
            else "provider_request_failed"
        ),
        "provider_errors": provider_errors,
    }


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

    payload, provider_meta = _request_tavily(
        "search",
        params,
        timeout=(15, 45),
    )
    if payload is None:
        return json.dumps(
            {
                "status": "error",
                "provider": "Tavily API",
                "query": query,
                "results": [],
                **provider_meta,
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
            "provider_attempts": provider_meta["provider_attempts"],
            "removed_credentials": provider_meta["removed_credentials"],
            "provider_errors": provider_meta["provider_errors"],
            "evidence_policy": "search results are discovery metadata; fetch one selected URL for page evidence",
        },
        ensure_ascii=False,
    )


@ToolRegistry.register(
    name="extract_web_urls_tavily",
    phase="ALL",
    plugin="web.tavily",
    capabilities=("page_content_extract",),
    retrieval_role="evidence",
    signature="Internal Tavily adapter for extracting source text from already selected URLs.",
)
def extract_web_urls_tavily(
    urls: list[str],
    query: str = "",
    working_memory: dict | None = None,
    agent_state=None,
    task_id: str = "",
    **kwargs: Any,
) -> str:
    """Extract literal source chunks for selected URLs; never generate an answer."""

    del working_memory, agent_state, task_id, kwargs
    selected: list[str] = []
    for value in urls or []:
        url = str(value or "").strip()
        parsed = urlparse(url)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            continue
        if url not in selected:
            selected.append(url)
        if len(selected) >= 20:
            break
    if not selected:
        return json.dumps(
            {
                "status": "error",
                "provider": "Tavily Extract",
                "error_class": "invalid_urls",
                "message": "no valid http(s) URL was provided",
                "results": [],
                "provider_errors": ["no valid http(s) URL was provided"],
            },
            ensure_ascii=False,
        )

    focused_query = " ".join(str(query or "").split())
    params: dict[str, Any] = {
        "urls": selected,
        "extract_depth": "advanced",
        "include_images": False,
        "include_favicon": False,
        "format": "markdown",
        "include_usage": False,
    }
    if focused_query:
        params.update({"query": focused_query, "chunks_per_source": 5})

    payload, provider_meta = _request_tavily(
        "extract",
        params,
        timeout=(15, 120),
    )
    if payload is None:
        return json.dumps(
            {
                "status": "error",
                "provider": "Tavily Extract",
                "query": focused_query,
                "requested_urls": selected,
                "results": [],
                **provider_meta,
            },
            ensure_ascii=False,
        )

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        raw_content = str(item.get("raw_content") or "")
        body_quality = clean_page_body(raw_content)
        page_excerpt = str(body_quality.get("text") or "").strip()
        if not url or not body_quality.get("body_eligible"):
            rejected.append(
                {
                    "url": url,
                    "error": "extracted content did not meet the substantive body threshold",
                }
            )
            continue
        rows.append(
            {
                "title": str(item.get("title") or urlparse(url).hostname or url),
                "url": url,
                "snippet": "",
                "page_excerpt": page_excerpt,
                "source_excerpt": page_excerpt,
                "content": page_excerpt,
                "source": "Tavily Extract",
                "untrusted_content": True,
                "evidence_origin": "fetched_page_body",
                "evidence_kind": "page_body",
                "evidence_boundary": "page_body_only",
                "body_verified": True,
                "body_cleaned": True,
                "body_quality": body_quality,
                "raw_page_chars": len(raw_content),
                "content_resolver": "web.tavily",
                "content_transport": "provider_extract",
            }
        )

    failed_results = [
        {
            "url": str(item.get("url") or ""),
            "error": str(item.get("error") or "provider extraction failed")[:500],
        }
        for item in payload.get("failed_results") or []
        if isinstance(item, dict)
    ]
    failed_results.extend(rejected)
    return json.dumps(
        {
            "status": "ok" if rows else "no_results",
            "real_network": True,
            "provider": "Tavily Extract",
            "query": focused_query,
            "retrieval_role": "evidence",
            "retrieved_at": datetime.now().isoformat(timespec="seconds"),
            "requested_urls": selected,
            "count": len(rows),
            "results": rows,
            "sources": [row["url"] for row in rows],
            "failed_results": failed_results,
            "provider_attempts": provider_meta["provider_attempts"],
            "removed_credentials": provider_meta["removed_credentials"],
            "provider_errors": provider_meta["provider_errors"],
            "evidence_policy": (
                "literal text extracted from the requested URLs; content must still pass "
                "the normal chunk and RWKV evidence-extraction pipeline"
            ),
        },
        ensure_ascii=False,
    )
