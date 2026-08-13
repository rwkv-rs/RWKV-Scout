"""State and context projection for one research task."""

from __future__ import annotations

import os
import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Set

from agent.claim_ledger import ClaimLedger
from agent.retrieval_object_contract import (
    merge_candidate_observations,
    merge_mapping_rows,
)
from config import get_llm_context_length
from utils.chunker import get_token_count
from utils.context_budget import routing_observation_tokens
from utils.retrieval_ledger import RetrievalLedger, canonical_url


_SOURCE_BODY_FIELDS = (
    "source_excerpt",
    "page_excerpt",
    "structured_evidence_text",
    "content",
    "abstract",
)


def _original_source_chunks(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Return fetched source text without extractor paraphrases or snippets."""

    # Put the original chunks selected for the active evidence focus first,
    # then retain bounded surrounding page chunks. Using only either side was
    # brittle: a page preamble can hide the relevant section, while a narrow
    # selected span can hide the version/date context that disambiguates it.
    selected_rows = list(item.get("selected_source_chunks") or [])
    source_rows = list(item.get("source_chunks") or [])
    rows = list(selected_rows)
    if selected_rows:
        selected_indices = {
            int(row.get("index"))
            for row in selected_rows
            if isinstance(row, dict)
            and isinstance(row.get("index"), int)
        }
        neighbour_indices = {
            value + offset
            for value in selected_indices
            for offset in (-1, 0, 1)
            if value + offset >= 0
        }
        rows.extend(
            row
            for row in source_rows
            if isinstance(row, dict)
            and isinstance(row.get("index"), int)
            and int(row.get("index")) in neighbour_indices
        )
    else:
        rows.extend(source_rows)
    chunks = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").replace("\x00", "")
        if not text:
            continue
        chunk_id = str(row.get("chunk_id") or f"chunk-{index + 1}")
        identity = (chunk_id, text)
        if identity in seen:
            continue
        seen.add(identity)
        chunks.append({"chunk_id": chunk_id, "text": text})
    if chunks:
        return chunks

    for field_name in _SOURCE_BODY_FIELDS:
        body = str(item.get(field_name) or "").replace("\x00", "")
        if body:
            return [{"chunk_id": "body-1", "text": body}]
    return []


@dataclass
class RetrievalEpisodeState:
    """Single shared evidence state for one model-owned retrieval episode.

    The planner, orchestrator, validator and final synthesizer must observe
    the same evidence ledger.  Keeping these collections on the task state
    prevents a replan or phase transition from creating a fresh local view
    and losing the already collected page/point bindings.
    """

    rounds: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    rounds_by_point: Dict[str, list[tuple[str, dict[str, Any]]]] = field(default_factory=dict)
    point_state: Dict[str, dict[str, Any]] = field(default_factory=dict)
    last_discovery_results: list[dict[str, Any]] = field(default_factory=list)
    last_retrieval_failure: dict[str, Any] | None = None
    # The evidence store is task-scoped and shared by the planner, tools and
    # final synthesizer.  It is deliberately external to the model
    # transcript: a follow-up search adds to this store instead of creating a
    # second recovery conversation.
    sources: Dict[str, dict[str, Any]] = field(default_factory=dict)
    sources_by_claim: Dict[str, Dict[str, dict[str, Any]]] = field(default_factory=dict)
    deterministic_results: list[dict[str, Any]] = field(default_factory=list)
    attempted_urls: Set[str] = field(default_factory=set)
    url_attempt_counts: Dict[str, int] = field(default_factory=dict)
    query_history: list[dict[str, Any]] = field(default_factory=list)
    coverage: Dict[str, dict[str, Any]] = field(default_factory=dict)
    frozen_paths: list[dict[str, Any]] = field(default_factory=list)
    infrastructure_events: list[dict[str, Any]] = field(default_factory=list)
    replan_count: int = 0
    evidence_revision: int = 0
    progress: RetrievalLedger = field(default_factory=RetrievalLedger)
    claims: ClaimLedger = field(default_factory=ClaimLedger)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @staticmethod
    def source_key(item: dict[str, Any]) -> str:
        url = str(item.get("url") or "").strip().casefold().rstrip("/")
        if url:
            return url
        digest = str(item.get("content_sha256") or "").strip()
        if digest:
            return f"sha256:{digest}"
        body = str(item.get("source_excerpt") or item.get("content") or "")
        return f"sha256:{hashlib.sha256(body.encode('utf-8')).hexdigest()}"

    def record_query(
        self,
        query: str,
        result: dict[str, Any],
        *,
        step: int = 0,
        task_point_id: str = "",
        strategy: str = "",
    ) -> dict[str, Any]:
        """Merge one research round into the shared evidence store."""

        query_text = " ".join(str(query or "").split()).strip()
        added: list[str] = []
        material_changed = False
        canonical_result_items: dict[str, dict[str, Any]] = {}
        with self._lock:
            self.query_history.append(
                {
                    "query": query_text,
                    "step": int(step or 0),
                    "status": str(result.get("status") or ""),
                    "new_source_count": 0,
                    "task_point_id": str(task_point_id or ""),
                    "strategy": str(strategy or ""),
                }
            )
            extraction = result.get("model_extraction") or {}
            if isinstance(extraction, dict):
                unresolved_chunks = int(extraction.get("unresolved_chunk_count") or 0)
                degraded_pages = int(extraction.get("degraded_page_count") or 0)
                recovered_chunks = int(extraction.get("recovered_chunk_count") or 0)
                self.query_history[-1]["model_extraction_unresolved_chunks"] = unresolved_chunks
                self.query_history[-1]["model_extraction_degraded_pages"] = degraded_pages
                if unresolved_chunks or degraded_pages:
                    self.infrastructure_events.append(
                        {
                            "kind": "model_extraction_incomplete",
                            "query": query_text,
                            "step": int(step or 0),
                            "task_point_id": str(task_point_id or ""),
                            "unresolved_chunk_count": unresolved_chunks,
                            "degraded_page_count": degraded_pages,
                            "transport_error_count": int(extraction.get("transport_error_count") or 0),
                            "recovered_chunk_count": recovered_chunks,
                            "pages": [
                                dict(value)
                                for value in (extraction.get("pages") or [])[:16]
                                if isinstance(value, dict)
                            ],
                        }
                    )
            attempted_this_round: set[str] = set()
            for key in ("results", "candidate_urls", "page_evidence"):
                rows = result.get(key) or []
                if isinstance(rows, dict):
                    rows = [rows]
                for row in rows if isinstance(rows, list) else []:
                    if not isinstance(row, dict):
                        continue
                    url = canonical_url(row.get("url") or row.get("page_url") or row.get("source_url"))
                    if url:
                        self.attempted_urls.add(url)
                        attempted_this_round.add(url)
            for url in attempted_this_round:
                self.url_attempt_counts[url] = self.url_attempt_counts.get(url, 0) + 1
            for item in result.get("results") or []:
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                # Route scope and evidence binding are separate contracts.
                # ``task_point_id`` records what RWKV tried to retrieve; only
                # the chunk extractor may bind an ordinary web span to a
                # factual record via ``claim_ids``.
                item.setdefault("attempt_task_point_id", str(task_point_id or ""))
                item.setdefault("retrieval_query", query_text)
                item.setdefault("retrieval_strategy", str(strategy or ""))
                request = (
                    dict(item.get("retrieval_request") or {})
                    if isinstance(item.get("retrieval_request"), dict)
                    else {}
                )
                alignment = (
                    dict(item.get("object_alignment") or {})
                    if isinstance(item.get("object_alignment"), dict)
                    else {}
                )
                binding = {
                    "task_record_id": str(
                        request.get("task_record_id") or task_point_id or ""
                    ),
                    "request_id": str(request.get("request_id") or ""),
                    "object_alignment": alignment,
                }
                item["retrieval_bindings"] = merge_mapping_rows(
                    item.get("retrieval_bindings"),
                    binding
                    if any(
                        value not in (None, "", {}, [])
                        for value in binding.values()
                    )
                    else None,
                )
                item["object_alignments"] = merge_mapping_rows(
                    item.get("object_alignments"),
                    alignment,
                )
                item["retrieval_requests"] = merge_mapping_rows(
                    item.get("retrieval_requests"),
                    request,
                )
                key = self.source_key(item)
                if key not in self.sources:
                    self.sources[key] = dict(item)
                    added.append(key)
                    material_changed = True
                else:
                    # Keep the richest representation when the same URL is
                    # encountered by a focused follow-up search.
                    current = self.sources[key]
                    prior_claim_ids = list(current.get("claim_ids") or [])
                    prior_selected = list(current.get("selected_source_chunks") or [])
                    prior_candidates = list(current.get("chunk_candidates") or [])
                    prior_bindings = list(current.get("retrieval_bindings") or [])
                    prior_requests = list(current.get("retrieval_requests") or [])
                    prior_alignments = [
                        dict(value)
                        for value in [
                            *list(current.get("object_alignments") or []),
                            current.get("object_alignment"),
                        ]
                        if isinstance(value, dict) and value
                    ]
                    if len(str(item.get("content") or "")) > len(str(current.get("content") or "")):
                        self.sources[key] = {**current, **item}
                        material_changed = True
                    current = self.sources[key]
                    current["claim_ids"] = list(dict.fromkeys([
                        *[
                            str(value)
                            for value in prior_claim_ids
                            if str(value).strip()
                        ],
                        *[
                            str(value)
                            for value in item.get("claim_ids") or []
                            if str(value).strip()
                        ],
                    ]))
                    selected_chunks: list[dict[str, Any]] = []
                    selected_seen: set[tuple[str, str]] = set()
                    for selected in [
                        *list(item.get("selected_source_chunks") or []),
                        *prior_selected,
                    ]:
                        if not isinstance(selected, dict):
                            continue
                        text = str(selected.get("text") or "").strip()
                        if not text:
                            continue
                        identity = (str(selected.get("chunk_id") or ""), text)
                        if identity in selected_seen:
                            continue
                        selected_seen.add(identity)
                        selected_chunks.append(dict(selected))
                    if selected_chunks:
                        prior_selected_identities = {
                            (
                                str(value.get("chunk_id") or ""),
                                str(value.get("text") or "").strip(),
                            )
                            for value in prior_selected
                            if isinstance(value, dict)
                            and str(value.get("text") or "").strip()
                        }
                        selected_identities = {
                            (
                                str(value.get("chunk_id") or ""),
                                str(value.get("text") or "").strip(),
                            )
                            for value in selected_chunks
                        }
                        if selected_identities != prior_selected_identities:
                            material_changed = True
                            current["selected_source_chunks"] = selected_chunks[:12]
                        elif prior_selected:
                            # Provider/model completion order may vary. Keep
                            # stable state when the observable spans are the
                            # same so no false evidence revision is emitted.
                            current["selected_source_chunks"] = prior_selected[:12]
                    locator_candidates = merge_candidate_observations(
                        item.get("chunk_candidates"),
                        prior_candidates,
                    )
                    if locator_candidates:
                        def candidate_map(values: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
                            return {
                                (
                                    str(value.get("chunk_id") or ""),
                                    str(value.get("quote") or "").strip(),
                                ): json.dumps(
                                    value,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    default=str,
                                )
                                for value in values
                                if isinstance(value, dict)
                                and str(value.get("quote") or "").strip()
                            }

                        if candidate_map(locator_candidates) != candidate_map(
                            prior_candidates
                        ):
                            material_changed = True
                            current["chunk_candidates"] = locator_candidates[:64]
                        elif prior_candidates:
                            current["chunk_candidates"] = prior_candidates[:64]
                    locator_text = str(item.get("model_locator_facts") or "").strip()
                    if locator_text and locator_text != str(current.get("model_locator_facts") or ""):
                        current["model_locator_facts"] = locator_text
                        material_changed = True
                    bindings = merge_mapping_rows(
                        item.get("retrieval_bindings"),
                        prior_bindings,
                    )
                    if bindings:
                        prior_binding_markers = {
                            json.dumps(
                                value,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                default=str,
                            )
                            for value in prior_bindings
                            if isinstance(value, dict)
                        }
                        binding_markers = {
                            json.dumps(
                                value,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                default=str,
                            )
                            for value in bindings
                        }
                        if binding_markers != prior_binding_markers:
                            material_changed = True
                            current["retrieval_bindings"] = bindings[:16]
                        elif prior_bindings:
                            current["retrieval_bindings"] = prior_bindings[:16]

                    alignments = merge_mapping_rows(
                        item.get("object_alignments"),
                        item.get("object_alignment"),
                        prior_alignments,
                    )
                    prior_alignment_markers = {
                        json.dumps(
                            value,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        )
                        for value in prior_alignments
                    }
                    if alignments:
                        alignment_markers = {
                            json.dumps(
                                value,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                default=str,
                            )
                            for value in alignments
                        }
                        if alignment_markers != prior_alignment_markers:
                            material_changed = True
                            current["object_alignments"] = alignments[:16]
                        elif prior_alignments:
                            current["object_alignments"] = prior_alignments[:16]
                    requests = merge_mapping_rows(
                        item.get("retrieval_requests"),
                        item.get("retrieval_request"),
                        prior_requests,
                    )
                    request_markers = {
                        json.dumps(
                            value,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        )
                        for value in requests
                    }
                    prior_request_markers = {
                        json.dumps(
                            value,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        )
                        for value in prior_requests
                        if isinstance(value, dict)
                    }
                    if request_markers != prior_request_markers:
                        current["retrieval_requests"] = requests[:16]
                        material_changed = True
                    elif prior_requests:
                        current["retrieval_requests"] = prior_requests[:16]
                canonical_source = dict(self.sources[key])
                canonical_result_items[key] = canonical_source
                bound_claim_ids = [
                    str(value).strip()
                    for value in canonical_source.get("claim_ids") or []
                    if str(value).strip()
                ]
                for claim_id in bound_claim_ids:
                    point_sources = self.sources_by_claim.setdefault(claim_id, {})
                    current_point_source = point_sources.get(key)
                    if current_point_source != canonical_source:
                        material_changed = True
                        point_sources[key] = canonical_source
            self.query_history[-1]["new_source_count"] = len(added)
            self.rounds.append((query_text, result))
            self.last_discovery_results[:] = [
                item for item in result.get("results") or [] if isinstance(item, dict)
            ]
        source_resolution = result.get("source_resolution") or {}
        if isinstance(source_resolution, dict):
            self.claims.update_required_domains(source_resolution.get("required_domains") or [])
        claim_result = {
            **result,
            "results": list(canonical_result_items.values()),
        }
        claim_delta = self.claims.ingest(
            query_text,
            claim_result,
            task_point_id=task_point_id,
            strategy=strategy,
            step=step,
        )
        material_changed = material_changed or int(
            claim_delta.get("added_source_bindings") or 0
        ) > 0 or int(claim_delta.get("added_evidence_records") or 0) > 0 or int(
            claim_delta.get("added_unassigned_sources") or 0
        ) > 0 or int(claim_delta.get("updated_evidence_records") or 0) > 0
        with self._lock:
            if material_changed:
                self.evidence_revision += 1
            evidence_revision = self.evidence_revision
        return {
            "new_source_keys": added,
            "new_source_count": len(added),
            "total_sources": len(self.sources),
            "claim_delta": claim_delta,
            "evidence_revision": evidence_revision,
            "material_changed": material_changed,
        }

    def source_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self.sources.values()]

    def record_deterministic_result(
        self,
        action: str,
        result: dict[str, Any],
        *,
        step: int = 0,
        task_point_id: str = "",
    ) -> None:
        """Persist an exact successful calculator/time result for replanning."""

        if str(result.get("status") or "").casefold() != "ok":
            return
        record = {
            "tool": str(action or ""),
            "step": int(step or 0),
            "task_point_id": str(task_point_id or ""),
            "result": dict(result),
        }
        with self._lock:
            self.deterministic_results.append(record)
            self.evidence_revision += 1
            if len(self.deterministic_results) > 32:
                del self.deterministic_results[:-32]

    def planner_evidence_snapshot(
        self,
        *,
        max_sources: int = 6,
        max_chars_per_source: int = 3000,
        max_total_chars: int = 9000,
    ) -> dict[str, Any]:
        """Project bounded original source spans into every planner decision.

        This is task memory, not an answerability judgement.  It deliberately
        ignores extractor ``supported`` flags and never includes generated
        fact paraphrases.  Character caps only protect the RWKV context window.
        """

        max_sources = max(1, int(max_sources or 1))
        max_chars_per_source = max(256, int(max_chars_per_source or 256))
        max_total_chars = max(512, int(max_total_chars or 512))
        with self._lock:
            all_sources = [dict(item) for item in self.sources.values()]
            sources_by_claim = {
                point_id: [dict(item) for item in values.values()]
                for point_id, values in self.sources_by_claim.items()
            }

        # Give every explicitly bound task point one source before using the
        # remaining budget for recent material. This is context packing, not a
        # judgement that the selected source proves the point.
        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set()

        def select(item: dict[str, Any]) -> None:
            key = self.source_key(item)
            if key in selected_keys or len(selected) >= max_sources:
                return
            selected.append(item)
            selected_keys.add(key)

        for point_sources in sources_by_claim.values():
            if point_sources:
                select(point_sources[0])
        for item in reversed(all_sources):
            select(item)
        if len(selected) < max_sources:
            for item in all_sources:
                select(item)

        projected: list[dict[str, Any]] = []
        remaining = max_total_chars
        for source_index, item in enumerate(selected, start=1):
            sources_left = len(selected) - source_index + 1
            source_budget = min(
                max_chars_per_source,
                max(0, remaining // max(1, sources_left)),
            )
            if source_budget <= 0:
                break

            spans: list[dict[str, Any]] = []
            source_chars = 0
            included_chars = 0
            for chunk in _original_source_chunks(item):
                text = str(chunk.get("text") or "")
                source_chars += len(text)
                if included_chars >= source_budget:
                    continue
                visible = text[: source_budget - included_chars]
                if not visible:
                    continue
                spans.append(
                    {
                        "chunk_id": str(chunk.get("chunk_id") or ""),
                        "start_char": 0,
                        "text": visible,
                        "truncated": len(visible) < len(text),
                    }
                )
                included_chars += len(visible)

            if not spans:
                continue
            remaining -= included_chars
            projected.append(
                {
                    "source_id": f"R{source_index}",
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "claim_ids": [
                        str(value)
                        for value in item.get("claim_ids") or []
                        if str(value).strip()
                    ],
                    "published": item.get("published") or item.get("published_at") or "",
                    "updated": item.get("updated") or item.get("updated_at") or "",
                    "date": item.get("date") or "",
                    "provider": item.get("provider") or item.get("source") or "",
                    "freshness": dict(item.get("freshness") or {})
                    if isinstance(item.get("freshness"), dict)
                    else {},
                    "source_object": dict(item.get("source_object") or {}),
                    "object_alignment": dict(item.get("object_alignment") or {}),
                    "object_alignments": merge_mapping_rows(
                        item.get("object_alignments"),
                        item.get("object_alignment"),
                    )[:8],
                    "retrieval_bindings": [
                        dict(value)
                        for value in item.get("retrieval_bindings") or []
                        if isinstance(value, dict)
                    ][:8],
                    "spans": spans,
                    "source_chars": source_chars,
                    "visible_chars": included_chars,
                    "truncated": included_chars < source_chars,
                }
            )

        return {
            "schema_version": "planner-evidence.v1",
            "source_count": len(all_sources),
            "visible_source_count": len(projected),
            "visible_chars": sum(item["visible_chars"] for item in projected),
            "truncated": len(projected) < len(all_sources)
            or any(bool(item.get("truncated")) for item in projected),
            "sources": projected,
        }

    def planner_record_snapshot(
        self,
        *,
        max_records: int = 8,
        max_quote_chars: int = 600,
        max_total_chars: int = 3200,
    ) -> dict[str, Any]:
        """Project grounded RWKV-selected candidate records for the next decision.

        Records are interleaved across task points. This is persistent working
        memory, not a deterministic finish decision: RWKV still decides
        whether the visible records satisfy the user's requested fields.
        """

        snapshot = self.claims.snapshot(max_spans_per_claim=0)
        per_point: list[list[dict[str, Any]]] = []
        for claim in snapshot.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            rows = [
                {
                    "evidence_record_id": str(
                        record.get("evidence_record_id") or ""
                    )[:80],
                    "task_record_id": str(
                        record.get("task_record_id")
                        or claim.get("claim_id")
                        or ""
                    )[:80],
                    "subject_key": str(record.get("subject_key") or "")[:240],
                    "record_key": str(record.get("record_key") or "")[:240],
                    "field_keys": [
                        str(value)[:120]
                        for value in record.get("field_keys") or []
                        if str(value).strip()
                    ][:16],
                    "support_state": str(record.get("support_state") or "")[:80],
                    "record_match": str(record.get("record_match") or "")[:80],
                    "field_contract_valid": bool(
                        record.get("field_contract_valid", True)
                    ),
                    "source_object": dict(record.get("source_object") or {}),
                    "object_alignment": dict(
                        record.get("object_alignment") or {}
                    ),
                    "object_alignments": merge_mapping_rows(
                        record.get("object_alignments"),
                        record.get("object_alignment"),
                    )[:8],
                    "rwkv_subject_alignment": dict(
                        record.get("rwkv_subject_alignment") or {}
                    ),
                    "retrieval_request": dict(record.get("retrieval_request") or {}),
                    "retrieval_requests": merge_mapping_rows(
                        record.get("retrieval_requests"),
                        record.get("retrieval_request"),
                    )[:8],
                    "retrieval_bindings": merge_mapping_rows(
                        record.get("retrieval_bindings")
                    )[:8],
                    "title": str(record.get("title") or "")[:240],
                    "url": str(record.get("url") or "")[:500],
                    "published": str(
                        record.get("published")
                        or record.get("published_at")
                        or ""
                    )[:80],
                    "updated": str(
                        record.get("updated")
                        or record.get("updated_at")
                        or ""
                    )[:80],
                    "quote": str(record.get("quote") or ""),
                }
                for record in claim.get("evidence_records") or []
                if isinstance(record, dict)
                and str(record.get("quote") or "").strip()
            ]
            if rows:
                per_point.append(rows)

        selected: list[dict[str, Any]] = []
        cursor = 0
        remaining = max(256, int(max_total_chars or 256))
        while (
            len(selected) < max(1, int(max_records or 1))
            and any(cursor < len(rows) for rows in per_point)
            and remaining > 0
        ):
            for rows in per_point:
                if cursor >= len(rows) or len(selected) >= max_records:
                    continue
                row = dict(rows[cursor])
                quote = str(row.get("quote") or "")
                visible = quote[: min(max_quote_chars, remaining)]
                if not visible:
                    continue
                row["quote"] = visible
                row["quote_truncated"] = len(visible) < len(quote)
                selected.append(row)
                remaining -= len(visible)
            cursor += 1

        total_records = sum(len(rows) for rows in per_point)
        return {
            "schema_version": "planner-records.v1",
            "record_count": total_records,
            "visible_record_count": len(selected),
            "truncated": len(selected) < total_records,
            "records": selected,
        }

    def infrastructure_report(self, *, claim_ids: list[str] | None = None) -> dict[str, Any]:
        """Return unresolved retrieval/model failures, optionally by Claim."""

        requested = {str(value) for value in (claim_ids or []) if str(value).strip()}
        with self._lock:
            events = [
                dict(item)
                for item in self.infrastructure_events
                if not requested
                or not str(item.get("task_point_id") or "")
                or str(item.get("task_point_id") or "") in requested
            ]
        affected_claim_ids = sorted(
            {
                str(item.get("task_point_id") or "")
                for item in events
                if str(item.get("task_point_id") or "")
            }
        )
        return {
            "schema_version": "retrieval-infrastructure.v1",
            "complete": not events,
            "event_count": len(events),
            "affected_claim_ids": affected_claim_ids,
            "unresolved_chunk_count": sum(int(item.get("unresolved_chunk_count") or 0) for item in events),
            "degraded_page_count": sum(int(item.get("degraded_page_count") or 0) for item in events),
            "transport_error_count": sum(int(item.get("transport_error_count") or 0) for item in events),
            "recovered_chunk_count": sum(int(item.get("recovered_chunk_count") or 0) for item in events),
            "events": events[:24],
        }

    def routing_snapshot(self, *, max_sources: int = 8, max_queries: int = 8) -> dict[str, Any]:
        with self._lock:
            return {
                "round_count": len(self.query_history),
                "evidence_revision": self.evidence_revision,
                "source_count": len(self.sources),
                "queries": [
                    {
                        "query": item.get("query", ""),
                        "status": item.get("status", ""),
                        "new_source_count": item.get("new_source_count", 0),
                    }
                    for item in self.query_history[-max_queries:]
                ],
                "sources": [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "chunk_count": item.get("chunk_count", 0),
                    }
                    for item in list(self.sources.values())[-max_sources:]
                ],
                "coverage": dict(self.coverage),
                "claim_ledger": self.claims.snapshot(max_spans_per_claim=0),
                "attempted_url_count": len(self.attempted_urls),
                "claim_source_counts": {
                    point_id: len(values)
                    for point_id, values in self.sources_by_claim.items()
                },
                "frozen_path_count": len(self.frozen_paths),
                "frozen_paths": [dict(item) for item in self.frozen_paths[-8:]],
                "replan_count": self.replan_count,
                "deterministic_result_count": len(self.deterministic_results),
                "retrieval_infrastructure": self.infrastructure_report(),
            }

    def planner_routing_snapshot(
        self,
        *,
        max_sources: int = 6,
        max_queries: int = 6,
        max_frozen_paths: int = 4,
    ) -> dict[str, Any]:
        """Return a small routing projection for the next RWKV decision.

        Full source bodies, Claim source records, infrastructure events and
        URL histories remain in persistent state and the audit trace.  A
        planner only needs progress, point bindings, recent requests and the
        frozen paths it must avoid; replaying the complete ledger crowds the
        actual replan instruction out of a 16K context window.
        """

        with self._lock:
            infrastructure = self.infrastructure_report()
            claim_snapshot = self.claims.snapshot(max_spans_per_claim=0)
            factual_point_progress = [
                {
                    "id": str(row.get("claim_id") or "")[:120],
                    "task_bound_candidate_record_count": int(
                        row.get("evidence_record_count") or 0
                    ),
                    "candidate_evidence_record_count": int(
                        row.get("evidence_record_count") or 0
                    ),
                    "total_evidence_record_count": int(
                        row.get("evidence_record_count") or 0
                    ),
                    "attempt_count": int(row.get("attempt_count") or 0),
                    "retrieval_state": str(row.get("retrieval_state") or "")[:80],
                }
                for row in claim_snapshot.get("claims") or []
                if isinstance(row, dict) and str(row.get("claim_id") or "").strip()
            ]
            return {
                "schema_version": "planner-routing.v1",
                "evidence_revision": self.evidence_revision,
                "round_count": len(self.query_history),
                "source_count": len(self.sources),
                # Observable grounded candidate-span counts only. Zero is not
                # a truth, sufficiency, or completion decision.
                "factual_point_progress": factual_point_progress,
                "unassigned_source_count": int(
                    claim_snapshot.get("unassigned_source_count") or 0
                ),
                "queries": [
                    {
                        "query": str(item.get("query") or "")[:500],
                        "status": str(item.get("status") or "")[:80],
                        "new_source_count": int(item.get("new_source_count") or 0),
                        "task_point_id": str(item.get("task_point_id") or "")[:120],
                    }
                    for item in self.query_history[-max(1, int(max_queries or 1)) :]
                ],
                "sources": [
                    {
                        "title": str(item.get("title") or "")[:300],
                        "url": str(item.get("url") or "")[:500],
                        "claim_ids": [
                            str(value)[:120]
                            for value in item.get("claim_ids") or []
                            if str(value).strip()
                        ][:8],
                        "published": str(
                            item.get("published") or item.get("published_at") or ""
                        )[:120],
                        "updated": str(
                            item.get("updated") or item.get("updated_at") or ""
                        )[:120],
                        "source_object": dict(item.get("source_object") or {}),
                        "object_alignment": dict(
                            item.get("object_alignment") or {}
                        ),
                        "object_alignments": merge_mapping_rows(
                            item.get("object_alignments"),
                            item.get("object_alignment"),
                        )[:8],
                        "retrieval_bindings": [
                            dict(value)
                            for value in item.get("retrieval_bindings") or []
                            if isinstance(value, dict)
                        ][:8],
                    }
                    for item in list(self.sources.values())[
                        -max(1, int(max_sources or 1)) :
                    ]
                ],
                "attempted_url_count": len(self.attempted_urls),
                "frozen_paths": [
                    {
                        "route_id": str(item.get("route_id") or "")[:40],
                        "query": str(item.get("query") or "")[:500],
                        "action": str(item.get("action") or "")[:120],
                        "task_point_id": str(item.get("task_point_id") or "")[:120],
                        "step": int(item.get("step") or 0),
                        "first_step": int(item.get("first_step") or item.get("step") or 0),
                        "last_step": int(item.get("last_step") or item.get("step") or 0),
                        "blocked_count": int(item.get("blocked_count") or 1),
                        "reason": str(item.get("reason") or "")[:240],
                    }
                    for item in self.frozen_paths[
                        -max(1, int(max_frozen_paths or 1)) :
                    ]
                ],
                "replan_count": self.replan_count,
                "retrieval_infrastructure": {
                    "complete": bool(infrastructure.get("complete")),
                    "event_count": int(infrastructure.get("event_count") or 0),
                    "affected_claim_ids": list(
                        infrastructure.get("affected_claim_ids") or []
                    )[:16],
                    "unresolved_chunk_count": int(
                        infrastructure.get("unresolved_chunk_count") or 0
                    ),
                    "transport_error_count": int(
                        infrastructure.get("transport_error_count") or 0
                    ),
                },
            }

    def freeze_path(
        self,
        query: str,
        *,
        action: str = "",
        arguments: dict[str, Any] | None = None,
        task_point_id: str = "",
        step: int = 0,
        reason: str = "",
    ) -> dict[str, Any]:
        """Freeze one exact retrieval route without duplicating route state.

        Repeated attempts of the same model-authored request are still counted
        for audit, but they remain one frozen path.  Treating every blocked
        repeat as a new path made the routing revision advance even though no
        strategy changed, which in turn caused redundant validation/replan
        cycles on identical state.
        """

        record = {
            "query": " ".join(str(query or "").split()).strip(),
            "action": str(action or ""),
            "arguments": dict(arguments or {}),
            "task_point_id": str(task_point_id or ""),
            "first_step": int(step or 0),
            "last_step": int(step or 0),
            # Historical readers use ``step``. Keep it as the most recent
            # blocked attempt while publishing explicit first/last fields.
            "step": int(step or 0),
            "reason": str(reason or "")[:500],
            "blocked_count": 1,
        }
        identity = json.dumps(
            {
                "action": record["action"],
                "arguments": record["arguments"],
                "task_point_id": record["task_point_id"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        record["route_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        with self._lock:
            existing = next(
                (
                    item
                    for item in reversed(self.frozen_paths)
                    if str(item.get("route_id") or "") == record["route_id"]
                ),
                None,
            )
            if existing is not None:
                existing["last_step"] = int(step or 0)
                existing["step"] = int(step or 0)
                existing["blocked_count"] = int(existing.get("blocked_count") or 1) + 1
                if reason:
                    existing["reason"] = str(reason)[:500]
                return dict(existing)
            self.frozen_paths.append(record)
        return dict(record)

    def record_replan(self) -> int:
        """Increment and return the task-scoped recovery count."""

        with self._lock:
            self.replan_count += 1
            return self.replan_count

    def reset(self) -> None:
        self.rounds.clear()
        self.rounds_by_point.clear()
        self.point_state.clear()
        self.last_discovery_results.clear()
        self.last_retrieval_failure = None
        self.sources.clear()
        self.sources_by_claim.clear()
        self.deterministic_results.clear()
        self.attempted_urls.clear()
        self.url_attempt_counts.clear()
        self.query_history.clear()
        self.coverage.clear()
        self.frozen_paths.clear()
        self.infrastructure_events.clear()
        self.replan_count = 0
        self.progress.reset()
        self.claims.reset()


@dataclass
class AgentState:
    task_id: str = ""
    task_output_dir: str = ""
    user_query: str = ""
    refined_query: str = ""
    id_to_path: Dict[str, str] = field(default_factory=dict)
    path_to_id: Dict[str, str] = field(default_factory=dict)
    working_memory: Dict[str, str] = field(default_factory=dict)
    memory_catalog: Dict[str, str] = field(default_factory=dict)
    last_feedback: str = ""
    entity_audit: Dict[str, str] = field(default_factory=dict)
    abandoned_file_ids: Set[str] = field(default_factory=set)
    is_finished: bool = False
    final_result: str = ""
    run_metadata: dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalEpisodeState = field(default_factory=RetrievalEpisodeState)

    def _mount_global_env(self) -> str:
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_query = self.refined_query or self.user_query
        env_lines = [
            "【挂载模块: 任务环境】",
            f"- 系统时间: {current_time_str}",
            f"- 当前执行目标: {active_query}",
        ]
        if self.entity_audit:
            env_lines.append("- 🎯 当前实体校验状态 (Entity Audit):")
            for entity, status in self.entity_audit.items():
                env_lines.append(f"  * {entity}: {status}")
        return "\n".join(env_lines)

    def _mount_memory_catalog(self) -> str:
        filtered_memory = {}
        for key, value in self.working_memory.items():
            if any(file_id in key for file_id in self.abandoned_file_ids):
                continue
            if not key.startswith(("__", "AbsPath_", "Path_", "Category_")):
                filtered_memory[key] = value

        if not filtered_memory:
            return "【挂载模块: 情报目录大纲】\n*(记忆区当前为空)*"

        memory_total_tokens = sum(get_token_count(str(value)) for value in filtered_memory.values())
        memory_details = [
            f"- `{key}`: [{self.memory_catalog.get(key, '已存储有效结构化数据')}]"
            for key in filtered_memory
        ]
        lines = [
            "【挂载模块: 情报目录大纲】",
            f"[系统状态] 当前可用知识库缓存已挂载（体积估算: {memory_total_tokens} Tokens）。",
            "【工作指引】: 请继续检查并收集其他缺漏情报；如果所有核心事实均已齐备，请立即调用 generate_final_aggregate_reports 进入最终聚合。",
            "\n以下是已获取的可用情报，请据此决定下一步：",
            *memory_details,
        ]
        return "\n".join(lines)

    def _mount_local_workspace(self) -> str:
        pending_preview = []
        pending_extract = []
        for file_id, path in self.id_to_path.items():
            if file_id in self.abandoned_file_ids or f"Summary_{file_id}" in self.memory_catalog:
                continue
            if f"Preview_{file_id}" in self.memory_catalog:
                pending_extract.append(
                    f"- {file_id}: {os.path.basename(path)} [已试读判定为相关，等待进行全文深度提炼]"
                )
            else:
                pending_preview.append(
                    f"- {file_id}: {os.path.basename(path)} [未读，可试读或直接全文提取]"
                )

        pending_items = pending_preview + pending_extract
        if not pending_items:
            return ""

        lines = [
            "【挂载模块: 本地工作区文件 (Local Workspace)】",
            "核心防幻觉红线：本地工作区中的文件可能是【完全独立、毫无关联】的，不要在无原文依据时捏造它们的合作关系！",
            f"📊 静态代码审计提醒：总共 {len(self.id_to_path)} 份文件中，仍有 {len(pending_items)} 份未被处理（系统防漏缺扫描）！",
            "💡 快捷通配符：如果缺口是需要处理大量未读文件，在调用工具的 file_ids 时可直接传入 [\"ALL\"]，底层引擎会自动将所有剩余未处理文件安全映射展开！",
            f"\n剩余清单 (共 {len(pending_items)} 项)：",
            *pending_items,
        ]
        return "\n".join(lines)

    def _mount_feedback(self) -> str:
        if not self.last_feedback:
            return ""
        return f"【挂载模块: 最新执行反馈】\n{self.last_feedback}"

    def to_markdown_context(self) -> str:
        modules = [
            self._mount_global_env(),
            self._mount_memory_catalog(),
            self._mount_local_workspace(),
            self._mount_feedback(),
        ]
        return "\n\n".join(module for module in modules if module)

    def to_retrieval_context(self) -> str:
        """Return persistent task state for the next RWKV tool decision.

        The legacy Markdown context describes the local workspace and its
        file-processing workflow.  Sending that context to an ordinary web
        retrieval turn creates a false "environment ready" signal and spends
        model tokens on an unrelated task.  File research still uses
        ``to_markdown_context``; the model-owned web loop gets this narrow
        routing view instead.
        """
        def compact_review(value: Any) -> dict[str, Any]:
            review = value if isinstance(value, dict) else {}
            return {
                key: review.get(key)
                for key in (
                    "decision",
                    "missing_point_id",
                    "evidence_needed",
                    "trigger",
                )
                if key in review
            }

        def compact_feedback(raw: str) -> dict[str, Any] | str:
            try:
                value = json.loads(str(raw or ""))
            except json.JSONDecodeError:
                return str(raw or "")[:1200]
            if not isinstance(value, dict):
                return str(raw or "")[:1200]
            output: dict[str, Any] = {
                key: value.get(key)
                for key in (
                    "status",
                    "error_class",
                    "repeat_count",
                    "replan_count",
                    "missing_point_id",
                    "evidence_needed",
                    "stalled_actions_after_replan",
                    "max_stalled_actions_after_replan",
                    "replan_rebuilds_without_progress",
                )
                if key in value
            }
            if value.get("message"):
                output["message"] = str(value.get("message") or "")[:600]
            request = value.get("request") or {}
            if isinstance(request, dict):
                arguments = request.get("arguments") or {}
                output["request"] = {
                    "action": str(request.get("action") or "")[:120],
                    "arguments": dict(arguments) if isinstance(arguments, dict) else {},
                }
            frozen = value.get("frozen_path") or {}
            if isinstance(frozen, dict):
                output["frozen_path"] = {
                    "query": str(frozen.get("query") or "")[:500],
                    "action": str(frozen.get("action") or "")[:120],
                    "task_point_id": str(frozen.get("task_point_id") or "")[:120],
                    "step": int(frozen.get("step") or 0),
                    "reason": str(frozen.get("reason") or "")[:240],
                }
            previous = value.get("previous_request_status") or {}
            if isinstance(previous, dict):
                output["previous_request_status"] = {
                    key: previous.get(key)
                    for key in (
                        "attempted",
                        "count",
                        "match_type",
                        "matched_query",
                        "similarity",
                        "threshold",
                    )
                    if key in previous
                }
            for review_key in ("evidence_review", "pending_replan"):
                review = value.get(review_key)
                if isinstance(review, dict):
                    output[review_key] = compact_review(review)
            return output

        lines = ["Retrieval task state:", f"Task: {self.refined_query or self.user_query}"]
        if self.last_feedback:
            lines.append("Latest controller feedback (compact routing projection):")
            lines.append(
                json.dumps(
                    compact_feedback(self.last_feedback),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        freshness_policy = self.run_metadata.get("freshness_policy")
        if isinstance(freshness_policy, dict) and freshness_policy:
            lines.append("Question time/freshness policy (observable metadata, not a gate):")
            lines.append(
                json.dumps(
                    freshness_policy,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        lines.append("Shared retrieval ledger (compact progress metadata; not a finish gate):")
        lines.append(
            json.dumps(
                self.retrieval.planner_routing_snapshot(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        lines.append(
            "RWKV-selected grounded candidate records "
            "(working memory; no record is pre-declared correct/current):"
        )
        lines.append(
            json.dumps(
                self.retrieval.planner_record_snapshot(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        lines.append("Bounded original source locators for routing (not final-answer evidence):")
        lines.append(
            json.dumps(
                self.retrieval.planner_evidence_snapshot(
                    max_sources=2,
                    max_chars_per_source=350,
                    max_total_chars=700,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        if self.retrieval.deterministic_results:
            lines.append("Persistent deterministic tool results:")
            lines.append(
                json.dumps(
                    self.retrieval.deterministic_results[-16:],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        rendered = "\n".join(lines)
        token_budget = routing_observation_tokens(get_llm_context_length())
        if get_token_count(rendered) <= token_budget:
            return rendered

        # A huge feedback object or unusually dense CJK source may still
        # exceed the routing budget. Rebuild with much smaller locators rather
        # than slicing through JSON and hiding the missing-point/frozen-path
        # records at the end.
        compact_lines: list[str] = []
        skip_evidence_payload = False
        for line in lines:
            if line.startswith("Bounded original source locators"):
                skip_evidence_payload = True
                continue
            if skip_evidence_payload:
                skip_evidence_payload = False
                continue
            compact_lines.append(line)
        compact_lines.append("Minimal original source locators for routing:")
        compact_lines.append(
            json.dumps(
                self.retrieval.planner_evidence_snapshot(
                    max_sources=3,
                    max_chars_per_source=300,
                    max_total_chars=900,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return "\n".join(compact_lines)
