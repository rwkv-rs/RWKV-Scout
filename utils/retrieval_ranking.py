"""Observable, provider-neutral retrieval ranking primitives.

Round 20 uses these functions in shadow mode only.  They record what a wider
candidate pool and reciprocal-rank fusion would have selected without
changing the URLs fetched for RWKV.  No reference answer or benchmark datum is
accepted by this module.
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
        provider = str(result.get("provider") or f"provider_{provider_index}")
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
                },
            )
            previous = row["provider_ranks"].get(provider)
            if previous is None or rank < previous:
                row["provider_ranks"][provider] = rank
            if provider not in row["discovery_providers"]:
                row["discovery_providers"].append(provider)
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
