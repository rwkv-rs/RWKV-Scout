"""Canonical retrieval envelope and provider-response conversion.

Provider adapters may speak Tavily JSON, HTML search rows, scholarly metadata,
or a local service protocol. The rest of RWKV-Scout consumes only this stable
envelope, so adding a provider does not add another orchestrator branch.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping


SCHEMA_VERSION = "retrieval.v1"
DISCOVERY_ROLE = "discovery"
EVIDENCE_ROLE = "evidence"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return {}


def _clean_rows(value: Any) -> list[dict[str, Any]]:
    rows = []
    for item in value or []:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        row["title"] = str(row.get("title") or row.get("name") or row.get("url") or "").strip()
        row["url"] = str(row.get("url") or row.get("link") or "").strip()
        row["snippet"] = str(row.get("snippet") or row.get("excerpt") or row.get("description") or "").strip()
        row["page_excerpt"] = str(row.get("page_excerpt") or row.get("content") or "").strip()
        row["source"] = str(row.get("source") or "").strip()
        if row["url"] or row["title"]:
            rows.append(row)
    return rows


def normalize_result(
    value: Any,
    *,
    provider: str = "",
    query: str = "",
    role: str = DISCOVERY_ROLE,
    real_network: bool | None = None,
) -> dict[str, Any]:
    """Convert one provider response to the stable retrieval envelope."""

    payload = _as_dict(value)
    if not payload:
        return error_result(
            provider=provider,
            query=query,
            role=role,
            message="provider returned a non-object response",
        )

    rows = _clean_rows(payload.get("results"))
    errors = payload.get("provider_errors") or payload.get("errors") or []
    if isinstance(errors, str):
        errors = [errors]
    errors = [str(item)[:500] for item in errors if str(item).strip()]
    status = str(payload.get("status") or "").strip().casefold()
    if status in {"error", "failed", "unavailable", "unauthorized"}:
        normalized_status = "error"
    elif rows:
        normalized_status = "ok"
    else:
        normalized_status = "no_results"

    normalized = dict(payload)
    normalized.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": normalized_status,
            "retrieval_role": str(payload.get("retrieval_role") or role),
            "provider": str(payload.get("provider") or provider),
            "query": str(payload.get("query") or query),
            "retrieved_at": str(payload.get("retrieved_at") or datetime.now().isoformat(timespec="seconds")),
            "count": len(rows),
            "results": rows,
            "sources": [row["url"] for row in rows if row.get("url")],
            "provider_errors": errors,
            "real_network": bool(payload.get("real_network")) if real_network is None else bool(real_network),
        }
    )
    refs = []
    for index, ref in enumerate(payload.get("citation_refs") or [], start=1):
        if not isinstance(ref, Mapping):
            continue
        item = dict(ref)
        item.setdefault("ref_id", f"{provider or 'provider'}_{index}")
        item.setdefault("title", "")
        item.setdefault("url", "")
        item.setdefault("source", normalized["provider"])
        refs.append(item)
    normalized["citation_refs"] = refs
    return normalized


def error_result(
    *,
    provider: str = "",
    query: str = "",
    role: str = DISCOVERY_ROLE,
    message: str,
    error_class: str = "provider_error",
    real_network: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "retrieval_role": role,
        "error_class": error_class,
        "message": str(message)[:500],
        "provider": provider,
        "query": query,
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "count": 0,
        "results": [],
        "sources": [],
        "citation_refs": [],
        "provider_errors": [str(message)[:500]],
        "real_network": real_network,
    }


def is_error(value: Any) -> bool:
    payload = _as_dict(value)
    return str(payload.get("status") or "").casefold() in {
        "error",
        "failed",
        "unavailable",
        "unauthorized",
    }
