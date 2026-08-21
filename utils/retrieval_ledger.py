"""Shared, model-visible retrieval progress for the retrieval episode.

The ledger records what the retrieval system has attempted and what changed.
It is shared by the active global decision loop. It does not choose a
replacement query. The runtime blocks only an exact repeated action+arguments
request within the same task point. Query similarity remains an offline
telemetry helper and never controls execution.
"""

from __future__ import annotations

import re
import threading
import json
import hashlib
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from agent.runtime_contracts import RETRIEVAL_EVENT_LEDGER_CONTRACT

_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_CHINESE_DIGITS = str.maketrans(
    "\u96f6\u3007\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d",
    "00123456789",
)


def _query_tokens(value: Any) -> tuple[str, ...]:
    """Tokenize a query for conservative equivalent-query detection."""

    text = normalize_query(value).translate(_CHINESE_DIGITS)
    return tuple(sorted(set(_QUERY_TOKEN_RE.findall(text))))


def query_similarity(left: Any, right: Any) -> float:
    """Return a conservative similarity score for two search queries."""

    left_key = normalize_query(left).translate(_CHINESE_DIGITS)
    right_key = normalize_query(right).translate(_CHINESE_DIGITS)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    left_tokens = set(_query_tokens(left_key))
    right_tokens = set(_query_tokens(right_key))
    if len(left_tokens) < 5 or len(right_tokens) < 5:
        return 0.0
    union = left_tokens | right_tokens
    shared = left_tokens & right_tokens
    containment = len(shared) / min(len(left_tokens), len(right_tokens))
    # A shared entity alone does not make two routes equivalent. A current
    # theme, historical release date and support deadline may all mention the
    # same product while targeting different requested fields. Require near
    # containment before applying the 0.60 route threshold.
    if containment < 0.80:
        return 0.0
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    ordered = SequenceMatcher(
        None,
        " ".join(sorted(left_tokens)),
        " ".join(sorted(right_tokens)),
    ).ratio()
    return max(jaccard, ordered)


def normalize_query(value: Any) -> str:
    """Create an exact-repeat key without semantic query rewriting."""

    text = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    return text.strip(" \t\r\n?？!！。.,，")


def canonical_url(value: Any) -> str:
    """Canonicalize URLs for duplicate accounting while retaining meaning."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    if not parts.scheme or not parts.netloc:
        return raw
    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith(("utm_", "fbclid", "gclid"))
    ]
    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path or "/",
            urlencode(query, doseq=True),
            "",
        )
    )


def request_key(action: Any, arguments: Mapping[str, Any] | None) -> str:
    """Return the canonical, lossless identity of one executable request."""

    payload = arguments if isinstance(arguments, Mapping) else {}
    return json.dumps(
        {
            "action": str(action or "").strip(),
            "arguments": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def route_id(
    action: Any,
    arguments: Mapping[str, Any] | None,
    *,
    task_record_id: str = "",
) -> str:
    """Return a compact identity for one request within one task record."""

    identity = json.dumps(
        {
            "request_key": request_key(action, arguments),
            "task_record_id": str(task_record_id or ""),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def result_error_observation(result: Mapping[str, Any] | None) -> dict[str, str]:
    """Project only the failure semantics already returned by a tool.

    This preserves information needed by RWKV on a later replan.  It does not
    classify the failure or prescribe a replacement route.
    """

    payload = result if isinstance(result, Mapping) else {}
    error_class = str(
        payload.get("error_class")
        or payload.get("error_type")
        or payload.get("code")
        or ""
    ).strip()
    error_message = str(payload.get("message") or payload.get("error") or "").strip()
    provider_errors = payload.get("provider_errors") or []
    if isinstance(provider_errors, Mapping):
        provider_errors = [provider_errors]
    if isinstance(provider_errors, list) and provider_errors:
        first = provider_errors[0]
        if isinstance(first, Mapping):
            error_class = error_class or str(
                first.get("error_class")
                or first.get("error_type")
                or first.get("code")
                or ""
            ).strip()
            error_message = error_message or str(
                first.get("message") or first.get("error") or ""
            ).strip()
        else:
            error_message = error_message or str(first).strip()
    return {
        "error_class": error_class[:160],
        "error_message": " ".join(error_message.split())[:600],
    }


def bounded_request_arguments(
    arguments: Mapping[str, Any] | None,
    *,
    max_items: int = 16,
    max_text: int = 500,
    max_depth: int = 3,
) -> dict[str, Any]:
    """Project model-authored arguments without changing request identity."""

    def compact(value: Any, depth: int) -> Any:
        if isinstance(value, Mapping):
            if depth >= max_depth:
                return str(dict(value))[:max_text]
            return {
                str(key)[:120]: compact(item, depth + 1)
                for key, item in list(value.items())[:max_items]
            }
        if isinstance(value, (list, tuple)):
            if depth >= max_depth:
                return str(list(value))[:max_text]
            return [compact(item, depth + 1) for item in list(value)[:max_items]]
        if isinstance(value, str):
            return value[:max_text]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)[:max_text]

    compacted = compact(arguments if isinstance(arguments, Mapping) else {}, 0)
    return compacted if isinstance(compacted, dict) else {}


class RetrievalLedger:
    """Track shared retrieval progress and optional legacy branch views.

    ``record`` accepts discovery, evidence, and generic bounded-search results.
    The stored data is intentionally compact: full pages and chunk text remain
    in the existing task events and are not duplicated in every model prompt.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._searches: list[dict[str, Any]] = []
        self._query_counts: Counter[str] = Counter()
        self._urls: dict[str, dict[str, Any]] = {}
        self._branch_queries: dict[str, list[str]] = {}
        self._requests: dict[str, dict[str, Any]] = {}
        self._blocked_duplicates: Counter[str] = Counter()

    def reset(self) -> None:
        """Clear one task episode while preserving the ledger instance."""

        with self._lock:
            self._searches.clear()
            self._query_counts.clear()
            self._urls.clear()
            self._branch_queries.clear()
            self._requests.clear()
            self._blocked_duplicates.clear()

    @staticmethod
    def request_key(action: Any, arguments: Mapping[str, Any] | None) -> str:
        """Build a stable identity for one model-selected tool request."""

        return request_key(action, arguments)

    @staticmethod
    def route_id(
        action: Any,
        arguments: Mapping[str, Any] | None,
        *,
        task_record_id: str = "",
    ) -> str:
        return route_id(action, arguments, task_record_id=task_record_id)

    def request_status(
        self,
        action: Any,
        arguments: Mapping[str, Any] | None,
        *,
        task_record_id: str = "",
    ) -> dict[str, Any] | None:
        """Return an exact request status within the requested task point."""

        key = self.request_key(action, arguments)
        scope = str(task_record_id or "").strip()
        with self._lock:
            value = self._requests.get(key)
            if not value:
                return None
            request_scopes = set(
                str(item or "").strip()
                for item in (
                    value.get("request_scopes")
                    or value.get("task_record_ids")
                    or []
                )
            )
            # The empty scope is a whole-task extraction route, not a wildcard
            # for every task point that has previously executed this request.
            if scope not in request_scopes:
                return None
            scoped = (value.get("scope_statuses") or {}).get(scope)
            if not isinstance(scoped, Mapping):
                return deepcopy(value)
            return {
                **deepcopy(value),
                **deepcopy(dict(scoped)),
                "task_record_id": scope,
            }

    def query_status(
        self,
        query: Any,
        *,
        task_record_id: str = "",
        threshold: float = 0.88,
        action: str = "",
    ) -> dict[str, Any]:
        """Describe exact/similar history as route telemetry.

        This method never decides whether a request may execute.  ``action``
        optionally keeps the observation within one tool boundary.
        """

        query_key = normalize_query(query)
        similarity_threshold = max(0.0, min(float(threshold), 1.0))
        with self._lock:
            candidates = [
                item
                for item in self._searches
                if (
                    (not task_record_id or str(item.get("task_record_id") or "") == task_record_id)
                    and (not action or str(item.get("action") or "") == str(action))
                )
            ]
            exact_matches = [
                item
                for item in candidates
                if query_key and item.get("query_key") == query_key
            ]
            if exact_matches:
                last = deepcopy(exact_matches[-1])
                return {
                    "query": str(query or "").strip(),
                    "query_key": query_key,
                    "attempted": True,
                    "count": len(exact_matches),
                    "match_type": "exact",
                    "exact_match": True,
                    "matched_query": last.get("query", ""),
                    "similarity": 1.0,
                    "threshold": similarity_threshold,
                    "last": last,
                    "blocked_count": int(self._blocked_duplicates.get(query_key, 0)),
                }

            best: tuple[float, dict[str, Any]] | None = None
            for item in candidates:
                score = query_similarity(query, item.get("query", ""))
                if score >= similarity_threshold and (best is None or score > best[0]):
                    best = (score, item)
            if best:
                score, matched = best
                return {
                    "query": str(query or "").strip(),
                    "query_key": query_key,
                    "attempted": True,
                    "count": 1,
                    "match_type": "equivalent",
                    "exact_match": False,
                    "matched_query": matched.get("query", ""),
                    "similarity": round(score, 4),
                    "threshold": similarity_threshold,
                    "last": deepcopy(matched),
                    "blocked_count": int(self._blocked_duplicates.get(query_key, 0)),
                }
            return {
                "query": str(query or "").strip(),
                "query_key": query_key,
                "attempted": False,
                "count": 0,
                "match_type": "none",
                "exact_match": False,
                "matched_query": "",
                "similarity": 0.0,
                "threshold": similarity_threshold,
                "last": None,
                "blocked_count": int(self._blocked_duplicates.get(query_key, 0)),
            }

    def record_duplicate_block(
        self,
        query: Any,
        *,
        step: int = 0,
        task_record_id: str = "",
        branch_id: str = "",
    ) -> dict[str, Any]:
        """Record a blocked duplicate without pretending a search ran."""

        raw_query = str(query or "").strip()
        query_key = normalize_query(raw_query)
        with self._lock:
            if query_key:
                self._blocked_duplicates[query_key] += 1
            return {
                "query": raw_query,
                "query_key": query_key,
                "blocked_count": int(self._blocked_duplicates.get(query_key, 0)),
                "step": step,
                "task_record_id": task_record_id,
                "branch_id": branch_id,
            }

    def record_request(
        self,
        action: Any,
        arguments: Mapping[str, Any] | None,
        result: Mapping[str, Any] | None,
        *,
        step: int = 0,
        task_record_id: str = "",
    ) -> dict[str, Any]:
        """Record one exact request and keep its compact latest status."""

        key = self.request_key(action, arguments)
        payload = result if isinstance(result, Mapping) else {}
        status = str(payload.get("status") or "ok").casefold()
        failed = status in {"error", "failed", "unavailable", "unauthorized"}
        error_observation = result_error_observation(payload)
        scope = str(task_record_id or "").strip()
        with self._lock:
            previous = self._requests.get(key) or {
                "request_key": key,
                "action": str(action or ""),
                "arguments": deepcopy(dict(arguments or {})),
                "attempts": 0,
                "failed_attempts": 0,
                "task_record_ids": [],
                "request_scopes": [],
                "scope_statuses": {},
            }
            previous.setdefault("request_scopes", list(previous.get("task_record_ids") or []))
            previous.setdefault("scope_statuses", {})
            previous["attempts"] = int(previous.get("attempts") or 0) + 1
            if failed:
                previous["failed_attempts"] = int(previous.get("failed_attempts") or 0) + 1
            previous["last_status"] = status
            previous["last_error_class"] = error_observation["error_class"]
            previous["last_error"] = error_observation["error_message"]
            previous["last_step"] = step
            previous["task_record_id"] = scope
            if scope and scope not in previous["task_record_ids"]:
                previous["task_record_ids"].append(scope)
            if scope not in previous["request_scopes"]:
                previous["request_scopes"].append(scope)
            previous["failed"] = failed
            scoped = previous["scope_statuses"].get(scope) or {
                "attempts": 0,
                "failed_attempts": 0,
            }
            scoped["attempts"] = int(scoped.get("attempts") or 0) + 1
            if failed:
                scoped["failed_attempts"] = int(scoped.get("failed_attempts") or 0) + 1
            scoped.update(
                {
                    "last_status": status,
                    "last_error_class": error_observation["error_class"],
                    "last_error": error_observation["error_message"],
                    "last_step": step,
                    "failed": failed,
                }
            )
            previous["scope_statuses"][scope] = scoped
            self._requests[key] = previous
            return deepcopy(previous)

    @staticmethod
    def _result_urls(result: Mapping[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("results", "candidate_urls", "page_evidence"):
            rows = result.get(key) or []
            if isinstance(rows, Mapping):
                rows = [rows]
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                for field in ("url", "page_url", "source_url"):
                    url = canonical_url(row.get(field))
                    if url:
                        values.append(url)
                        break
        return list(dict.fromkeys(values))

    @staticmethod
    def _evidence_count(result: Mapping[str, Any]) -> int:
        page_evidence = result.get("page_evidence")
        if isinstance(page_evidence, list):
            return sum(
                1
                for row in page_evidence
                if isinstance(row, Mapping)
                and str(row.get("status") or "").casefold() in {"ok", "supported", "completed"}
            )
        if isinstance(page_evidence, Mapping):
            return int(str(page_evidence.get("status") or "").casefold() in {"ok", "supported", "completed"})
        return sum(
            1
            for row in (result.get("results") or [])
            if isinstance(row, Mapping)
            and str(row.get("evidence_status") or row.get("status") or "").casefold()
            in {"ok", "supported", "completed"}
        )

    def record(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Record a result while keeping shared branch observations atomic."""

        with self._lock:
            return self._record_unlocked(*args, **kwargs)

    def _record_unlocked(
        self,
        query: Any,
        result: Mapping[str, Any] | None,
        *,
        step: int = 0,
        branch_id: str = "",
        task_record_id: str = "",
        action: str = "",
        arguments: Mapping[str, Any] | None = None,
        phase: str = "",
    ) -> dict[str, Any]:
        """Record one retrieval result and return its new-information delta."""

        payload = result if isinstance(result, Mapping) else {}
        error_observation = result_error_observation(payload)
        raw_query = str(query or "").strip()
        query_key = normalize_query(raw_query)
        route_arguments = dict(arguments or {})
        route_operation = str(route_arguments.get("operation") or "").strip()
        previous_count = self._query_counts[query_key] if query_key else 0
        self._query_counts[query_key] += 1 if query_key else 0
        urls = self._result_urls(payload)
        new_urls = [url for url in urls if url not in self._urls]
        for url in urls:
            record = self._urls.setdefault(
                url,
                {"url": url, "first_step": step, "branches": [], "query_keys": [], "evidence_count": 0},
            )
            if branch_id and branch_id not in record["branches"]:
                record["branches"].append(branch_id)
            if query_key and query_key not in record["query_keys"]:
                record["query_keys"].append(query_key)
            record["evidence_count"] = max(record.get("evidence_count", 0), self._evidence_count(payload))

        evidence_count = self._evidence_count(payload)
        entry = {
            "step": step,
            "branch_id": branch_id,
            "task_record_id": task_record_id,
            "action": action,
            "operation": route_operation,
            "arguments": route_arguments,
            "phase": phase,
            "query": raw_query,
            "query_key": query_key,
            "query_count": previous_count + 1 if query_key else 0,
            "exact_repeat": previous_count > 0,
            "urls": urls[:32],
            "new_urls": new_urls[:32],
            "new_url_count": len(new_urls),
            "evidence_count": evidence_count,
            "status": str(payload.get("status") or "ok"),
            "error_class": error_observation["error_class"],
            "error_message": error_observation["error_message"],
            "evidence_ready": bool(payload.get("evidence_ready")),
        }
        self._searches.append(entry)
        if branch_id and query_key:
            self._branch_queries.setdefault(branch_id, []).append(query_key)
        return deepcopy(entry)

    def observation(
        self,
        *,
        branch_id: str = "",
        task_record_id: str = "",
        limit: int = 8,
    ) -> dict[str, Any]:
        """Return one consistent snapshot for a concurrent branch decision."""

        with self._lock:
            return self._observation_unlocked(
                branch_id=branch_id,
                task_record_id=task_record_id,
                limit=limit,
            )

    def _observation_unlocked(
        self,
        *,
        branch_id: str = "",
        task_record_id: str = "",
        limit: int = 8,
    ) -> dict[str, Any]:
        """Return a compact shared+branch view for the next RWKV decision."""

        recent = self._searches[-max(1, int(limit)) :]
        branch_keys = set(self._branch_queries.get(branch_id, [])) if branch_id else set()
        branch_recent = [item for item in self._searches if item.get("query_key") in branch_keys][-limit:]
        query_counts = [
            {
                "query": item.get("query", ""),
                "count": item.get("query_count", 0),
                "last_step": item.get("step", 0),
                "new_urls": item.get("new_url_count", 0),
                "evidence": item.get("evidence_count", 0),
                "exact_repeat": item.get("exact_repeat", False),
            }
            for item in recent
        ]
        return {
            "contract": RETRIEVAL_EVENT_LEDGER_CONTRACT,
            "branch_id": branch_id,
            "task_record_id": task_record_id,
            "total_searches": len(self._searches),
            "unique_queries": len(self._query_counts),
            "exact_repeat_count": sum(max(0, count - 1) for count in self._query_counts.values()),
            "blocked_duplicate_count": sum(self._blocked_duplicates.values()),
            "retrieved_url_count": len(self._urls),
            "retrieved_urls": list(self._urls.keys())[-32:],
            "failed_requests": [
                {
                    "action": item.get("action", ""),
                    "arguments": item.get("arguments", {}),
                    "failed_attempts": item.get("failed_attempts", 0),
                    "last_step": item.get("last_step", 0),
                    "last_error_class": item.get("last_error_class", ""),
                    "last_error": item.get("last_error", ""),
                }
                for item in self._requests.values()
                if item.get("failed")
            ][-8:],
            "recent_searches": query_counts,
            "branch_searches": [
                {
                    "query": item.get("query", ""),
                    "count": item.get("query_count", 0),
                    "step": item.get("step", 0),
                    "new_urls": item.get("new_url_count", 0),
                    "evidence": item.get("evidence_count", 0),
                }
                for item in branch_recent
            ],
            "decision_guidance": (
                "Already-searched information is observational context. Exact web_search repeats "
                "are blocked before network execution. You decide whether to refine the query, use existing "
                "evidence, search a different aspect, or finish."
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the complete compact ledger for trace/report persistence."""

        with self._lock:
            return {
                "contract": RETRIEVAL_EVENT_LEDGER_CONTRACT,
                "searches": deepcopy(self._searches),
                "queries": dict(self._query_counts),
                "urls": deepcopy(self._urls),
                "branches": deepcopy(self._branch_queries),
                "requests": deepcopy(self._requests),
                "summary": self._observation_unlocked(limit=16),
            }
