"""Observable, provider-neutral retrieval ranking primitives.

The functions accept only live provider order and resource URLs.  They never
accept reference answers or benchmark metadata and never decide which source
contains a true answer.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from utils.web_retrieval import normalize_url, retrieval_url_identity


def rrf_fuse(
    provider_results: Iterable[Mapping[str, Any]],
    *,
    rrf_k: int = 60,
    pool_limit: int = 48,
) -> list[dict[str, Any]]:
    """Fuse provider-local ranks without comparing private provider scores."""

    k = max(1, int(rrf_k or 60))
    limit = max(1, int(pool_limit or 48))
    merged: dict[str, dict[str, Any]] = {}
    for provider_index, result in enumerate(provider_results, start=1):
        # One backend may return independent rankings for several RWKV-authored
        # query lanes.  Treat each lane as its own RRF list while retaining the
        # physical provider name in the surrounding result for audit/UI use.
        ranking_stream = str(
            result.get("ranking_stream")
            or result.get("provider")
            or f"provider_{provider_index}"
        )
        physical_provider = str(
            result.get("provider") or f"provider_{provider_index}"
        )
        result_query = {
            "query_id": str(result.get("retrieval_query_id") or ""),
            "task_record_id": str(result.get("retrieval_query_task_record_id") or ""),
            "intent": str(result.get("retrieval_query_intent") or ""),
            "query": str(
                result.get("retrieval_query_text") or result.get("query") or ""
            )[:500],
        }
        result_query = {
            key: value for key, value in result_query.items() if str(value or "").strip()
        }
        for rank, raw in enumerate(result.get("results") or [], start=1):
            if not isinstance(raw, Mapping):
                continue
            url = normalize_url(str(raw.get("url") or ""))
            if not url:
                continue
            identity = retrieval_url_identity(url)
            row = merged.setdefault(
                identity,
                {
                    "title": str(raw.get("title") or url),
                    "url": url,
                    "snippet": str(raw.get("snippet") or "")[:1000],
                    "provider_ranks": {},
                    "discovery_providers": [],
                    "ranking_streams": [],
                    "discovery_queries": [],
                },
            )
            previous = row["provider_ranks"].get(ranking_stream)
            if previous is None or rank < previous:
                row["provider_ranks"][ranking_stream] = rank
            if ranking_stream not in row["ranking_streams"]:
                row["ranking_streams"].append(ranking_stream)
            if physical_provider not in row["discovery_providers"]:
                row["discovery_providers"].append(physical_provider)
            raw_queries = raw.get("discovery_queries") or []
            if isinstance(raw_queries, Mapping):
                raw_queries = [raw_queries]
            query_rows = [
                dict(value) for value in raw_queries if isinstance(value, Mapping)
            ]
            if result_query:
                query_rows.append(result_query)
            seen_queries = {
                (
                    str(value.get("query_id") or ""),
                    str(value.get("query") or "").casefold(),
                )
                for value in row["discovery_queries"]
            }
            for query_row in query_rows:
                identity_row = (
                    str(query_row.get("query_id") or ""),
                    str(query_row.get("query") or "").casefold(),
                )
                if identity_row in seen_queries:
                    continue
                seen_queries.add(identity_row)
                row["discovery_queries"].append(query_row)
            if len(str(raw.get("snippet") or "")) > len(str(row.get("snippet") or "")):
                row["snippet"] = str(raw.get("snippet") or "")[:1000]
            if not str(row.get("title") or "").strip() and raw.get("title"):
                row["title"] = str(raw["title"])

    fused: list[dict[str, Any]] = []
    for row in merged.values():
        score = sum(1.0 / (k + rank) for rank in row["provider_ranks"].values())
        fused.append({**row, "rrf_score": round(score, 9)})
    fused.sort(
        key=lambda row: (
            float(row.get("rrf_score") or 0.0),
            len(row.get("provider_ranks") or {}),
            -min((row.get("provider_ranks") or {"": 10**6}).values()),
            str(row.get("url") or ""),
        ),
        reverse=True,
    )
    for rank, row in enumerate(fused[:limit], start=1):
        row["rrf_rank"] = rank
    return fused[:limit]


def select_domain_diverse(
    candidates: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    per_domain_limit: int,
) -> list[dict[str, Any]]:
    """Apply a bounded domain quota to an already ranked candidate sequence."""

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    cap = max(1, int(per_domain_limit or 1))
    for raw in candidates:
        row = dict(raw)
        url = str(row.get("url") or "")
        host = url.split("/", 3)[2].casefold() if "://" in url else url.casefold()
        if counts.get(host, 0) >= cap:
            continue
        selected.append(row)
        counts[host] = counts.get(host, 0) + 1
        if len(selected) >= max(1, int(limit or 1)):
            break
    return selected
