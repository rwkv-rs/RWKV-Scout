"""Evidence merging for the single shared research episode."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from utils.experiment_strategies import normalize_strategy
from utils.evidence_quality import date_mentions, evidence_text, has_substantive_evidence, substantive_evidence_items


def _record_key(item: Mapping[str, Any]) -> str:
    doi = str(item.get("doi") or "").strip().casefold()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi).rstrip("/")
    if doi.startswith("10."):
        return f"doi:{doi}"
    url = str(item.get("url") or "").strip().casefold()
    url = re.sub(r"^https?://(?:www\.)?", "", url)
    url = url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    if url:
        doi_match = re.search(r"(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s]+)", url)
        if doi_match:
            return f"doi:{doi_match.group(1).rstrip('/')}"
        return f"url:{url}"
    title = " ".join(str(item.get("title") or "").split()).casefold()
    return f"title:{title}" if title else ""


def _quality_terms(query: str, candidate_queries: Sequence[str]) -> set[str]:
    stopwords = {
        "the", "and", "for", "find", "then", "from", "with", "official", "source",
        "search", "query", "information", "answer", "page", "please", "current",
    }
    terms: set[str] = set()
    for value in (query, *candidate_queries):
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(value or "")):
            token = token.casefold()
            if token not in stopwords:
                terms.add(token)
        for run in re.findall(r"[\u3400-\u9fff]+", str(value or "")):
            if len(run) >= 2:
                terms.add(run.casefold())
                for size in (2, 3, 4):
                    terms.update(run[index : index + size].casefold() for index in range(len(run) - size + 1))
    return terms


def _evidence_quality_score(item: Mapping[str, Any], terms: set[str], candidate_count: int) -> float:
    body = evidence_text(dict(item))
    haystack = " ".join(
        [str(item.get("title") or ""), body, " ".join(str(value) for value in (item.get("authors") or []))]
    ).casefold()
    hits = sum(term in haystack for term in terms)
    lexical = min(1.0, hits / max(1, min(len(terms), 8)))
    support = min(1.0, len(set(item.get("candidate_queries") or [])) / max(1, candidate_count))
    best_rank = min(item.get("candidate_ranks") or [999])
    rank_score = 1 / max(1, best_rank)
    substantive = has_substantive_evidence(dict(item))
    body_score = min(1.0, len(body) / 1200) if substantive else 0.0
    date_requested = bool(terms.intersection({"\u65e5\u671f", "\u65f6\u95f4", "\u53d1\u5e03", "\u5468\u5e74", "\u7248\u672c", "date", "year", "release"}))
    date_score = min(1.0, len(date_mentions(body)) / 3) if date_requested else 0.0
    return round(0.35 * lexical + 0.25 * body_score + 0.20 * support + 0.10 * rank_score + 0.10 * date_score, 4)


def merge_retrieval_results(
    query: str,
    action: str,
    rounds: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    scope: str = "",
    ranking_strategy: str = "evidence_quality.v1",
) -> dict[str, Any]:
    """Merge candidate/round results without inventing fields."""
    strategy = normalize_strategy({"ranking_strategy": ranking_strategy})["ranking_strategy"]
    merged: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    discovery_sources: list[str] = []
    citation_refs: list[dict[str, Any]] = []
    citation_by_key: dict[str, int] = {}
    errors: list[str] = []
    real_network_values: list[bool] = []
    candidate_queries: list[str] = []
    evidence_missing_count = 0
    extraction_events: list[dict[str, Any]] = []
    for candidate_query, data in rounds:
        candidate_queries.append(candidate_query)
        real_network_values.append(bool(data.get("real_network", True)))
        errors.extend(str(value) for value in data.get("provider_errors") or [])
        raw_items = [item for item in data.get("results") or [] if isinstance(item, dict)]
        valid_items = substantive_evidence_items(raw_items)
        evidence_missing_count += len(raw_items) - len(valid_items)
        extraction = data.get("model_extraction") or {}
        if isinstance(extraction, Mapping) and not extraction.get("complete", True):
            extraction_events.append(
                {
                    "query": candidate_query,
                    "degraded_page_count": int(extraction.get("degraded_page_count") or 0),
                    "unresolved_chunk_count": int(extraction.get("unresolved_chunk_count") or 0),
                    "transport_error_count": int(extraction.get("transport_error_count") or 0),
                    "recovered_chunk_count": int(extraction.get("recovered_chunk_count") or 0),
                    "pages": [
                        dict(value)
                        for value in (extraction.get("pages") or [])[:16]
                        if isinstance(value, Mapping)
                    ],
                }
            )
        valid_keys = {_record_key(item) for item in valid_items if _record_key(item)}
        for source in data.get("sources") or []:
            if source and source not in discovery_sources:
                discovery_sources.append(source)
        for ref in data.get("citation_refs") or []:
            if not isinstance(ref, dict):
                continue
            key = _record_key(ref) or str(ref.get("ref_id") or "").strip().casefold()
            if not key or key not in valid_keys:
                continue
            existing_index = citation_by_key.get(key)
            if existing_index is None:
                citation_by_key[key] = len(citation_refs)
                citation_refs.append(dict(ref))
            else:
                current = citation_refs[existing_index]
                current_evidence = str(current.get("evidence_text") or current.get("content") or "")
                new_evidence = str(ref.get("evidence_text") or ref.get("content") or "")
                if len(new_evidence) > len(current_evidence):
                    citation_refs[existing_index] = {**current, **ref}
        for rank, item in enumerate(valid_items, start=1):
            if item.get("url") and item.get("url") not in sources:
                sources.append(str(item.get("url")))
            key = _record_key(item)
            if not key:
                continue
            if key not in merged:
                value = dict(item)
                value["candidate_queries"] = [candidate_query]
                value["candidate_ranks"] = [rank]
                merged[key] = value
            else:
                current = merged[key]
                current["candidate_queries"] = list(dict.fromkeys([*current.get("candidate_queries", []), candidate_query]))
                current["candidate_ranks"] = [*current.get("candidate_ranks", []), rank]
                current["claim_ids"] = list(
                    dict.fromkeys(
                        [
                            *[
                                str(value)
                                for value in current.get("claim_ids") or []
                                if str(value).strip()
                            ],
                            *[
                                str(value)
                                for value in item.get("claim_ids") or []
                                if str(value).strip()
                            ],
                        ]
                    )
                )
                for field in (
                    "source_excerpt",
                    "page_excerpt",
                    "structured_evidence_text",
                    "content",
                    "abstract",
                    "snippet",
                    "model_locator_facts",
                ):
                    if len(str(item.get(field) or "")) > len(str(current.get(field) or "")):
                        current[field] = item.get(field)
                current_chunks = current.get("source_chunks") or []
                item_chunks = item.get("source_chunks") or []
                current_chunk_chars = sum(
                    len(str(row.get("text") or ""))
                    for row in current_chunks
                    if isinstance(row, Mapping)
                )
                item_chunk_chars = sum(
                    len(str(row.get("text") or ""))
                    for row in item_chunks
                    if isinstance(row, Mapping)
                )
                if item_chunk_chars > current_chunk_chars:
                    current["source_chunks"] = list(item_chunks)
                # A focused follow-up on the same page can nominate different
                # original chunks for a newly missing task point. Keep the
                # newest attention selection first while retaining prior
                # selections and the complete source_chunks provenance.
                selected_chunks: list[dict[str, Any]] = []
                selected_seen: set[tuple[str, int, str]] = set()
                for selected in [
                    *list(item.get("selected_source_chunks") or []),
                    *list(current.get("selected_source_chunks") or []),
                ]:
                    if not isinstance(selected, Mapping):
                        continue
                    text = str(selected.get("text") or "").strip()
                    if not text:
                        continue
                    identity = (
                        str(selected.get("chunk_id") or ""),
                        int(selected.get("index") or 0),
                        text,
                    )
                    if identity in selected_seen:
                        continue
                    selected_seen.add(identity)
                    selected_chunks.append(dict(selected))
                if selected_chunks:
                    current["selected_source_chunks"] = selected_chunks[:12]
                for field in (
                    "source",
                    "provider",
                    "source_type",
                    "published",
                    "published_at",
                    "updated",
                    "updated_at",
                    "date",
                    "retrieved_at",
                    "freshness",
                ):
                    if current.get(field) in (None, "", [], {}) and item.get(field) not in (
                        None,
                        "",
                        [],
                        {},
                    ):
                        current[field] = item.get(field)
    results = list(merged.values())
    quality_terms = _quality_terms(query, candidate_queries)
    if strategy == "best_rank.v1":
        results.sort(key=lambda item: (min(item.get("candidate_ranks") or [999]), -len(item.get("candidate_queries") or [])))
    elif strategy == "dedup_order.v1":
        # Deliberately weak but reproducible ablation: preserve first-seen order.
        results.sort(key=lambda item: min(item.get("candidate_ranks") or [999]))
    elif strategy == "evidence_quality.v1":
        results.sort(
            key=lambda item: _evidence_quality_score(item, quality_terms, len(dict.fromkeys(candidate_queries))),
            reverse=True,
        )
    else:
        results.sort(key=lambda item: (len(item.get("candidate_queries") or []), -min(item.get("candidate_ranks") or [999])), reverse=True)
    candidate_count = max(1, len(dict.fromkeys(candidate_queries)))
    for rank, item in enumerate(results, start=1):
        support_count = len(dict.fromkeys(item.get("candidate_queries") or []))
        best_candidate_rank = min(item.get("candidate_ranks") or [999])
        support_score = support_count / candidate_count
        rank_score = 1 / max(1, best_candidate_rank)
        item["dedup_key"] = _record_key(item)
        item["ranking_method"] = strategy
        item["rerank_score"] = round(
            rank_score
            if strategy == "best_rank.v1"
            else 0.0
            if strategy == "dedup_order.v1"
            else _evidence_quality_score(item, quality_terms, candidate_count)
            if strategy == "evidence_quality.v1"
            else 0.7 * support_score + 0.3 * rank_score,
            4,
        )
        item["retrieval_rank"] = rank
    return {
        "status": "ok" if results else ("no_evidence" if evidence_missing_count else "no_results"),
        "real_network": all(real_network_values) if real_network_values else True,
        "scope": scope,
        "query": query,
        "provider_query": " | ".join(candidate_queries),
        "candidate_queries": candidate_queries,
        "round_count": len(rounds),
        "count": len(results),
        "results": results,
        "sources": sources,
        "discovery_sources": discovery_sources,
        "citation_refs": citation_refs,
        "evidence_missing_count": evidence_missing_count,
        "model_extraction": {
            "schema_version": "model-extraction-diagnostics.v1",
            "complete": not extraction_events,
            "event_count": len(extraction_events),
            "degraded_page_count": sum(int(item["degraded_page_count"]) for item in extraction_events),
            "unresolved_chunk_count": sum(int(item["unresolved_chunk_count"]) for item in extraction_events),
            "transport_error_count": sum(int(item["transport_error_count"]) for item in extraction_events),
            "recovered_chunk_count": sum(int(item["recovered_chunk_count"]) for item in extraction_events),
            "events": extraction_events[:24],
        },
        "provider_errors": errors,
        "evidence_policy": next((data.get("evidence_policy") for _, data in rounds if data.get("evidence_policy")), ""),
        "ranking_strategy": strategy,
    }
