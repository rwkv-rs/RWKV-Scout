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

from config import get_search_api_keys, retire_search_api_key
from tools.registry import ToolRegistry
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

    key_entries = _ordered_key_entries(api_keys)
    if not key_entries:
        return json.dumps(
            {
                "status": "error",
                "provider": "Tavily API",
                "query": query,
                "results": [],
                "provider_attempts": 0,
                "provider_disabled": True,
                "error_class": "provider_quota_or_auth_unavailable",
                "provider_errors": [
                    "all configured Tavily credentials are unavailable for this process"
                ],
            },
            ensure_ascii=False,
        )

    payload: dict[str, Any] | None = None
    provider_errors: list[str] = []
    provider_attempts = 0
    permanent_failures = 0
    removed_credentials = 0
    for api_key, identity, key_lock in key_entries:
        with key_lock:
            # Another concurrent case may have completed the only necessary
            # probe while this case waited for the key lock.
            if _key_is_permanently_failed(identity):
                continue
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
                provider_attempts += 1
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
                permanent, reason = _permanent_provider_failure(exc, response_body)
                if permanent:
                    permanent_failures += 1
                    _mark_key_permanently_failed(identity, reason)
                    try:
                        if retire_search_api_key("tavily", api_key):
                            removed_credentials += 1
                    except OSError:
                        # Runtime quarantine still prevents another probe. A
                        # storage error is intentionally sanitized so a path or
                        # credential cannot leak into provider output.
                        provider_errors.append(
                            "permanently unavailable credential could not be removed from local storage"
                        )
                detail = f"{type(exc).__name__}: {exc}"
                if response_body:
                    detail = f"{detail}; response_body={response_body}"
                provider_errors.append(detail[:800])

    if payload is None:
        all_keys_unavailable = not _has_available_key(api_keys)
        return json.dumps(
            {
                "status": "error",
                "provider": "Tavily API",
                "query": query,
                "results": [],
                "provider_attempts": provider_attempts,
                "removed_credentials": removed_credentials,
                "provider_disabled": all_keys_unavailable,
                "error_class": (
                    "provider_quota_or_auth_unavailable"
                    if all_keys_unavailable
                    else "provider_request_failed"
                ),
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
            "provider_attempts": provider_attempts,
            "removed_credentials": removed_credentials,
            "provider_errors": provider_errors,
            "evidence_policy": "search results are discovery metadata; fetch one selected URL for page evidence",
        },
        ensure_ascii=False,
    )
