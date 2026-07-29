"""Shared, model-visible retrieval progress for Single-loop and Fork runs.

The ledger is an observation layer.  It records what the retrieval system has
already attempted and what changed, but it never rejects a model-selected
query or chooses a replacement query on the model's behalf.
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


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


class RetrievalLedger:
    """Track shared retrieval progress and branch-local views.

    ``record`` accepts discovery, evidence, and generic bounded-search results.
    The stored data is intentionally compact: full pages and chunk text remain
    in the existing task events and are not duplicated in every model prompt.
    """

    VERSION = "retrieval_ledger.v1"

    def __init__(self) -> None:
        self._searches: list[dict[str, Any]] = []
        self._query_counts: Counter[str] = Counter()
        self._urls: dict[str, dict[str, Any]] = {}
        self._branch_queries: dict[str, list[str]] = {}

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

    def record(
        self,
        query: Any,
        result: Mapping[str, Any] | None,
        *,
        step: int = 0,
        branch_id: str = "",
        task_point_id: str = "",
        action: str = "",
        phase: str = "",
    ) -> dict[str, Any]:
        """Record one retrieval result and return its new-information delta."""

        payload = result if isinstance(result, Mapping) else {}
        raw_query = str(query or "").strip()
        query_key = normalize_query(raw_query)
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
            "task_point_id": task_point_id,
            "action": action,
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
            "evidence_ready": bool(payload.get("evidence_ready")),
        }
        self._searches.append(entry)
        if branch_id and query_key:
            self._branch_queries.setdefault(branch_id, []).append(query_key)
        return deepcopy(entry)

    def observation(self, *, branch_id: str = "", task_point_id: str = "", limit: int = 8) -> dict[str, Any]:
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
            "schema_version": self.VERSION,
            "branch_id": branch_id,
            "task_point_id": task_point_id,
            "total_searches": len(self._searches),
            "unique_queries": len(self._query_counts),
            "exact_repeat_count": sum(max(0, count - 1) for count in self._query_counts.values()),
            "retrieved_url_count": len(self._urls),
            "retrieved_urls": list(self._urls.keys())[-32:],
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
                "Already-searched information is observational context. You decide whether to refine the query, "
                "use a returned URL, search a different aspect, or finish. Repeating is allowed but consumes one step."
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the complete compact ledger for trace/report persistence."""

        return {
            "schema_version": self.VERSION,
            "searches": deepcopy(self._searches),
            "queries": dict(self._query_counts),
            "urls": deepcopy(self._urls),
            "branches": deepcopy(self._branch_queries),
            "summary": self.observation(limit=16),
        }
