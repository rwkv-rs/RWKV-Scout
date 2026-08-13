"""Minimal task-scoped evidence index for RWKV.

The ledger records which planned task points have retrieved source material.
It never decides whether RWKV may answer, whether an answer is correct, or
whether a source proves a semantic claim.
"""

from __future__ import annotations

import hashlib
import re
import threading
from copy import deepcopy
from typing import Any, Mapping

from agent.task_plan_contract import task_points


def _status_entity_terms(value: Any) -> set[str]:
    """Return lightweight entity tokens used only by page candidate helpers."""

    text = str(value or "").casefold()
    terms = set(re.findall(r"[a-z][a-z0-9_.+-]{1,}", text))
    terms.update(run for run in re.findall(r"[\u3400-\u9fff]{2,}", text))
    return terms


def _normalized_with_positions(value: str) -> tuple[str, list[int]]:
    output: list[str] = []
    positions: list[int] = []
    pending_space = False
    for index, char in enumerate(str(value or "")):
        if char.isspace():
            pending_space = bool(output)
            continue
        if pending_space:
            output.append(" ")
            positions.append(index)
            pending_space = False
        output.append(char.casefold())
        positions.append(index)
    return "".join(output).strip(), positions


def locate_grounded_quote_span(
    candidate: Mapping[str, Any],
    source_text: str,
) -> dict[str, Any] | None:
    """Locate a model/deterministic quote in the fetched chunk text."""

    source = str(source_text or "")
    quote = str(candidate.get("quote") or candidate.get("source_span") or "").strip()
    if not quote:
        facts = candidate.get("facts") or []
        if isinstance(facts, str):
            facts = [facts]
        quote = next((str(value).strip() for value in facts if str(value).strip()), "")
    if not source or not quote:
        return None

    start = source.find(quote)
    basis = "exact"
    if start < 0:
        start = source.casefold().find(quote.casefold())
        basis = "casefold"
    if start < 0:
        normalized_source, positions = _normalized_with_positions(source)
        normalized_quote, _ = _normalized_with_positions(quote)
        normalized_start = normalized_source.find(normalized_quote)
        if normalized_start >= 0 and positions:
            normalized_end = normalized_start + len(normalized_quote) - 1
            if normalized_end >= len(positions):
                return None
            start = positions[normalized_start]
            end = positions[normalized_end] + 1
            basis = "normalized_whitespace"
        else:
            # Extractors often preserve page words while normalizing Markdown
            # headings and blank lines. Locate those verbatim fragments in
            # order, then expose only the original contiguous source span.
            fragments = [
                line.strip()
                for line in quote.splitlines()
                if len("".join(line.split())) >= 8
            ]
            matches: list[tuple[int, int, str]] = []
            cursor = 0
            source_folded = source.casefold()
            for fragment in fragments:
                fragment_start = source.find(fragment, cursor)
                fragment_basis = "exact"
                if fragment_start < 0:
                    fragment_start = source_folded.find(fragment.casefold(), cursor)
                    fragment_basis = "casefold"
                if fragment_start < 0:
                    normalized_tail, tail_positions = _normalized_with_positions(source[cursor:])
                    normalized_fragment, _ = _normalized_with_positions(fragment)
                    tail_start = normalized_tail.find(normalized_fragment)
                    if tail_start < 0 or not tail_positions:
                        continue
                    tail_end = tail_start + len(normalized_fragment) - 1
                    if tail_end >= len(tail_positions):
                        continue
                    fragment_start = cursor + tail_positions[tail_start]
                    fragment_end = cursor + tail_positions[tail_end] + 1
                    fragment_basis = "normalized_whitespace"
                else:
                    fragment_end = fragment_start + len(fragment)
                matches.append((fragment_start, fragment_end, fragment_basis))
                cursor = fragment_end

            matched_chars = sum(end_value - start_value for start_value, end_value, _ in matches)
            visible_quote_chars = len("".join(quote.split()))
            minimum_coverage = max(24, min(120, visible_quote_chars // 8))
            strongest_match = max(
                (end_value - start_value for start_value, end_value, _ in matches),
                default=0,
            )
            if matched_chars < minimum_coverage or (
                len(matches) < 2 and strongest_match < 60
            ):
                return None
            start = matches[0][0]
            end = matches[-1][1]
            if end - start > 5000:
                return None
            basis = "ordered_source_segments"
            grounded_segment_count = len(matches)
    else:
        end = start + len(quote)

    if basis != "ordered_source_segments":
        grounded_segment_count = 1

    context_start = max(0, start - 240)
    context_end = min(len(source), end + 240)
    return {
        "text": source[start:end],
        "char_start": start,
        "char_end": end,
        "context_char_start": context_start,
        "context_char_end": context_end,
        "grounding_basis": basis,
        "grounded_segment_count": grounded_segment_count,
    }


def _source_text(item: Mapping[str, Any]) -> str:
    for key in (
        "source_excerpt",
        "page_excerpt",
        "structured_evidence_text",
        "content",
        "abstract",
        "snippet",
    ):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _source_record(item: Mapping[str, Any]) -> dict[str, Any]:
    # The ledger is persistent state, not a model prompt.  Keep enough of each
    # exact source span to survive later context reconstruction; final and
    # validation packers apply their own token budgets.  The former 2,400-char
    # prefix silently cut query-focused examples located near the end of a
    # 1,600-token chunk.
    retained_span_chars = 6000
    chunks = [
        {
            "chunk_id": str(row.get("chunk_id") or ""),
            "index": int(row.get("index") or 0),
            "text": str(row.get("text") or "")[:retained_span_chars],
        }
        for row in item.get("source_chunks") or []
        if isinstance(row, Mapping) and str(row.get("text") or "").strip()
    ][:8]
    selected_chunks = [
        {
            "chunk_id": str(row.get("chunk_id") or ""),
            "index": int(row.get("index") or 0),
            "text": str(row.get("text") or "")[:retained_span_chars],
            "attention_rank": int(row.get("attention_rank") or 0),
            "attention_score": int(row.get("attention_score") or 0),
            "attention_reasons": [
                str(value)[:160]
                for value in row.get("attention_reasons") or []
                if str(value).strip()
            ][:12],
        }
        for row in item.get("selected_source_chunks") or []
        if isinstance(row, Mapping) and str(row.get("text") or "").strip()
    ][:8]
    grounded_spans = []
    for row in item.get("chunk_candidates") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("supported") is not True or row.get("source_grounded") is not True:
            continue
        quote = str(row.get("quote") or "").strip()
        if not quote:
            continue
        grounded_spans.append(
            {
                "chunk_id": str(row.get("chunk_id") or ""),
                "index": int(row.get("chunk_index") or 0),
                "text": quote[:1200],
                "source_locator": deepcopy(dict(row.get("source_locator") or {})),
                "grounding_basis": str(row.get("grounding_basis") or ""),
            }
        )
        if len(grounded_spans) >= 8:
            break
    record = {
        "title": str(item.get("title") or ""),
        "url": str(item.get("url") or ""),
        "retrieval_query": str(item.get("retrieval_query") or ""),
        "text_available": bool(_source_text(item)),
        "chunk_count": len(chunks) or int(item.get("chunk_count") or 0),
        "chunks": chunks,
        "selected_chunks": selected_chunks,
        "grounded_spans": grounded_spans,
    }
    # Preserve observable time/source metadata for the planner and final
    # writer. This is transport only: the ledger never decides whether a date
    # is current or whether a source is authoritative.
    for key in (
        "source",
        "provider",
        "source_type",
        "published",
        "published_at",
        "updated",
        "updated_at",
        "date",
        "retrieved_at",
    ):
        value = item.get(key)
        if value not in (None, "", [], {}):
            record[key] = deepcopy(value)
    freshness = item.get("freshness")
    if isinstance(freshness, Mapping):
        record["freshness"] = deepcopy(dict(freshness))
    return record


def _candidate_task_record_ids(candidate: Mapping[str, Any]) -> list[str]:
    values = (
        candidate.get("task_record_ids")
        or candidate.get("claim_ids")
        or candidate.get("task_point_ids")
        or []
    )
    if isinstance(values, str):
        values = [values]
    return list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )[:8]


def _grounded_candidates(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return exact RWKV-selected spans without page-level route bindings."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in [
        *list(item.get("chunk_candidates") or []),
        *list(item.get("candidates") or []),
    ]:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("supported") is not True or candidate.get("source_grounded") is not True:
            continue
        quote = str(candidate.get("quote") or "").strip()
        if not quote:
            continue
        identity = (str(candidate.get("chunk_id") or ""), quote)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(deepcopy(dict(candidate)))
    return rows


def _evidence_record(
    item: Mapping[str, Any],
    candidate: Mapping[str, Any],
    task_record_id: str,
    *,
    binding_origin: str = "rwkv_chunk_extractor",
) -> dict[str, Any]:
    """Build one immutable exact-span record for a model-selected task record."""

    quote = str(candidate.get("quote") or "").strip()
    url = str(item.get("url") or "")
    chunk_id = str(candidate.get("chunk_id") or "")
    digest_input = "\n".join((task_record_id, url, chunk_id, quote))
    evidence_record_id = "E-" + hashlib.sha256(
        digest_input.encode("utf-8")
    ).hexdigest()[:20]
    record_match = str(
        candidate.get("record_match") or "exact_requested_record"
    ).strip().casefold()
    field_contract_valid = bool(candidate.get("field_contract_valid", True))
    if record_match == "exact_requested_record" and field_contract_valid:
        support_state = "rwkv_exact_requested_record"
    elif record_match == "same_subject_other_record":
        support_state = "rwkv_candidate_other_record"
    else:
        support_state = "rwkv_unmapped_candidate_record"
    record = {
        "evidence_record_id": evidence_record_id,
        "task_record_id": task_record_id,
        "subject_key": str(candidate.get("subject_key") or "")[:300],
        "record_key": str(candidate.get("record_key") or "")[:300],
        "field_keys": [
            str(value)[:160]
            for value in candidate.get("field_keys") or []
            if str(value).strip()
        ][:16],
        "source_id": url or str(item.get("content_sha256") or ""),
        "title": str(item.get("title") or ""),
        "url": url,
        "chunk_id": chunk_id,
        "chunk_index": int(candidate.get("chunk_index") or 0),
        "quote": quote[:1200],
        "source_locator": deepcopy(dict(candidate.get("source_locator") or {})),
        "grounding_basis": str(candidate.get("grounding_basis") or ""),
        "support_state": support_state,
        "record_match": record_match,
        "field_contract_valid": field_contract_valid,
        "binding_origin": binding_origin,
    }
    for key in (
        "source",
        "provider",
        "source_type",
        "published",
        "published_at",
        "updated",
        "updated_at",
        "date",
        "retrieved_at",
    ):
        value = item.get(key)
        if value not in (None, "", [], {}):
            record[key] = deepcopy(value)
    freshness = item.get("freshness")
    if isinstance(freshness, Mapping):
        record["freshness"] = deepcopy(dict(freshness))
    return record


def _source_from_evidence_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility source view containing only the exact bound record."""

    source = {
        "title": str(record.get("title") or ""),
        "url": str(record.get("url") or ""),
        "text_available": bool(str(record.get("quote") or "").strip()),
        "chunk_count": 1,
        "grounded_spans": [
            {
                "evidence_record_id": str(record.get("evidence_record_id") or ""),
                "chunk_id": str(record.get("chunk_id") or ""),
                "index": int(record.get("chunk_index") or 0),
                "text": str(record.get("quote") or ""),
                "field_keys": list(record.get("field_keys") or []),
                "subject_key": str(record.get("subject_key") or ""),
                "record_key": str(record.get("record_key") or ""),
                "source_locator": deepcopy(dict(record.get("source_locator") or {})),
                "grounding_basis": str(record.get("grounding_basis") or ""),
            }
        ],
        "evidence_records": [deepcopy(dict(record))],
    }
    for key in (
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
        if record.get(key) not in (None, "", [], {}):
            source[key] = deepcopy(record[key])
    return source


class ClaimLedger:
    """Record exact evidence spans per RWKV-planned factual record.

    The historical class name is retained for API compatibility. Search-route
    metadata never binds a whole page to a factual record. A web span is bound
    only when RWKV's chunk extractor names that record and the quote maps back
    to fetched text. Structured harness output may use the RWKV-selected tool
    route because it is already an exact typed record rather than a web page.
    """

    VERSION = "evidence-ledger.v1"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._claims: dict[str, dict[str, Any]] = {}
        self._source_policy = "open_web"
        self._required_domains: list[str] = []
        self._query = ""
        self._unassigned_sources: list[dict[str, Any]] = []
        self._unassigned_attempts: list[dict[str, Any]] = []

    def reset(self) -> None:
        with self._lock:
            self._claims.clear()
            self._source_policy = "open_web"
            self._required_domains = []
            self._query = ""
            self._unassigned_sources.clear()
            self._unassigned_attempts.clear()

    def initialize(self, task_plan: Mapping[str, Any] | None, query: str) -> None:
        plan = task_plan if isinstance(task_plan, Mapping) else {}
        points = task_points(plan, fallback_query=query)
        with self._lock:
            self.reset()
            self._query = str(query or "")
            # Source routing is resolved from the user request and runtime
            # connectors. The factual plan is deliberately not a policy gate.
            self._source_policy = "open_web"
            self._required_domains = []
            for index, point in enumerate(points, start=1):
                claim_id = str(point.get("id") or f"P{index}").strip() or f"P{index}"
                self._claims[claim_id] = {
                    "claim_id": claim_id,
                    "question": str(point.get("question") or query or ""),
                    "subject": str(point.get("subject") or ""),
                    "relation": str(point.get("relation") or ""),
                    "fields": [
                        str(value)
                        for value in point.get("fields") or []
                        if str(value).strip()
                    ],
                    "time_scope": str(point.get("time_scope") or "unspecified"),
                    "set_semantics": str(point.get("set_semantics") or "single"),
                    "premise_requires_verification": bool(
                        point.get("premise_requires_verification")
                    ),
                    "attempts": [],
                    "sources": [],
                    "evidence_records": [],
                }

    def claim_ids(self) -> list[str]:
        with self._lock:
            return list(self._claims)

    def update_required_domains(self, domains: Any) -> None:
        values = [domains] if isinstance(domains, str) else list(domains or [])
        with self._lock:
            self._required_domains = list(
                dict.fromkeys(
                    [
                        *self._required_domains,
                        *[
                            str(value).casefold().strip().removeprefix("www.").rstrip(".")
                            for value in values
                            if str(value).strip()
                        ],
                    ]
                )
            )

    def ingest(
        self,
        query: str,
        result: Mapping[str, Any],
        *,
        task_point_id: str = "",
        strategy: str = "",
        step: int = 0,
    ) -> dict[str, Any]:
        items = [item for item in result.get("results") or [] if isinstance(item, Mapping)]
        with self._lock:
            route_target = task_point_id if task_point_id in self._claims else ""
            added = 0
            added_records = 0
            added_unassigned = 0
            touched: set[str] = set()

            if route_target:
                self._claims[route_target]["attempts"].append(
                    {
                        "query": str(query or ""),
                        "strategy": str(strategy or ""),
                        "step": int(step or 0),
                        "url": "",
                        "kind": "rwkv_selected_route",
                    }
                )

            for item in items:
                source_record = _source_record(item)
                candidates = _grounded_candidates(item)
                assigned_records: list[dict[str, Any]] = []

                for candidate in candidates:
                    targets = [
                        value
                        for value in _candidate_task_record_ids(candidate)
                        if value in self._claims
                    ]
                    for claim_id in targets:
                        assigned_records.append(
                            _evidence_record(item, candidate, claim_id)
                        )

                # A typed connector is already one deterministic record. Its
                # binding remains RWKV-authored because RWKV selected both the
                # tool request and task_record_id. Ordinary web pages never use
                # this route-level fallback.
                if (
                    not assigned_records
                    and route_target
                    and str(item.get("evidence_kind") or "").casefold()
                    == "structured_record"
                    and _source_text(item)
                ):
                    text = _source_text(item)[:1200]
                    assigned_records.append(
                        _evidence_record(
                            item,
                            {
                                "chunk_id": "structured-record",
                                "chunk_index": 0,
                                "quote": text,
                                "source_locator": dict(item.get("source_locator") or {}),
                                "grounding_basis": "structured_record",
                                "field_keys": [],
                            },
                            route_target,
                            binding_origin="rwkv_structured_tool_route",
                        )
                    )

                if not assigned_records:
                    attempt = {
                        "query": str(query or ""),
                        "strategy": str(strategy or ""),
                        "step": int(step or 0),
                        "url": source_record["url"],
                        "task_record_id": route_target,
                    }
                    self._unassigned_attempts.append(attempt)
                    identity = source_record["url"] or source_record["title"]
                    if identity and not any(
                        (row.get("url") or row.get("title")) == identity
                        for row in self._unassigned_sources
                    ):
                        self._unassigned_sources.append(deepcopy(source_record))
                        added_unassigned += 1
                    continue

                assigned_identity = source_record["url"] or source_record["title"]
                if assigned_identity:
                    self._unassigned_sources = [
                        row
                        for row in self._unassigned_sources
                        if (row.get("url") or row.get("title"))
                        != assigned_identity
                    ]

                for evidence_record in assigned_records:
                    claim_id = str(evidence_record["task_record_id"])
                    claim = self._claims[claim_id]
                    record_id = str(evidence_record["evidence_record_id"])
                    if any(
                        str(row.get("evidence_record_id") or "") == record_id
                        for row in claim["evidence_records"]
                    ):
                        continue
                    claim["evidence_records"].append(deepcopy(evidence_record))
                    source = _source_from_evidence_record(evidence_record)
                    identity = source["url"] or source["title"]
                    existing_source = next(
                        (
                            row
                            for row in claim["sources"]
                            if identity
                            and (row.get("url") or row.get("title")) == identity
                        ),
                        None,
                    )
                    if existing_source is None:
                        claim["sources"].append(source)
                        added += 1
                    else:
                        existing_source["grounded_spans"] = [
                            *list(existing_source.get("grounded_spans") or []),
                            *list(source.get("grounded_spans") or []),
                        ][:16]
                        existing_source["evidence_records"] = [
                            *list(existing_source.get("evidence_records") or []),
                            deepcopy(evidence_record),
                        ][:16]
                    added_records += 1
                    touched.add(claim_id)

            return {
                "added_source_bindings": added,
                "added_evidence_records": added_records,
                "added_unassigned_sources": added_unassigned,
                "touched_claim_ids": sorted(touched),
                "claim_count": len(self._claims),
                "unassigned_source_count": len(self._unassigned_sources),
            }

    def snapshot(self, *, max_spans_per_claim: int = 4) -> dict[str, Any]:
        span_limit = max(0, int(max_spans_per_claim or 0))
        with self._lock:
            claims = []
            for claim in self._claims.values():
                source_records = deepcopy(claim.get("sources") or [])
                sources = []
                for source in source_records[:8]:
                    projected = {
                        key: value
                        for key, value in source.items()
                        if key not in {"chunks", "selected_chunks"}
                    }
                    if span_limit:
                        span_rows = [
                            *list(source.get("grounded_spans") or []),
                            *list(source.get("selected_chunks") or []),
                            *list(source.get("chunks") or []),
                        ]
                        distinct_spans: list[dict[str, Any]] = []
                        seen_spans: set[tuple[str, str]] = set()
                        for span in span_rows:
                            if not isinstance(span, Mapping):
                                continue
                            identity = (
                                str(span.get("chunk_id") or ""),
                                str(span.get("text") or ""),
                            )
                            if not identity[1] or identity in seen_spans:
                                continue
                            seen_spans.add(identity)
                            distinct_spans.append(deepcopy(dict(span)))
                        projected["spans"] = distinct_spans[:span_limit]
                    sources.append(projected)
                claims.append(
                    {
                        "claim_id": claim["claim_id"],
                        "question": claim["question"],
                        "subject": claim.get("subject") or "",
                        "relation": claim.get("relation") or "",
                        "fields": deepcopy(claim["fields"]),
                        "time_scope": claim["time_scope"],
                        "set_semantics": claim.get("set_semantics") or "single",
                        "premise_requires_verification": bool(
                            claim.get("premise_requires_verification")
                        ),
                        "retrieval_state": (
                            "evidence_recorded"
                            if claim.get("evidence_records")
                            else "not_recorded"
                        ),
                        "attempt_count": len(claim.get("attempts") or []),
                        "source_count": len(sources),
                        "evidence_record_count": len(
                            claim.get("evidence_records") or []
                        ),
                        "exact_record_count": sum(
                            str(row.get("support_state") or "")
                            == "rwkv_exact_requested_record"
                            for row in claim.get("evidence_records") or []
                            if isinstance(row, Mapping)
                        ),
                        "candidate_record_count": sum(
                            str(row.get("support_state") or "")
                            != "rwkv_exact_requested_record"
                            for row in claim.get("evidence_records") or []
                            if isinstance(row, Mapping)
                        ),
                        "evidence_records": deepcopy(
                            claim.get("evidence_records") or []
                        )[:32],
                        "sources": sources[:8],
                        # Compatibility alias for old trace readers. It is an
                        # evidence index, not a semantic support decision.
                        "evidence": sources[:8],
                    }
                )
            return {
                "schema_version": self.VERSION,
                "source_policy": self._source_policy,
                "required_domains": list(self._required_domains),
                "claim_count": len(claims),
                "claims": claims,
                "unassigned_source_count": len(self._unassigned_sources),
                "unassigned_attempt_count": len(self._unassigned_attempts),
                "unassigned_sources": [
                    {
                        key: deepcopy(value)
                        for key, value in source.items()
                        if key != "chunks"
                    }
                    for source in self._unassigned_sources[:8]
                ],
                "advisory_only": True,
            }


__all__ = ["ClaimLedger", "locate_grounded_quote_span"]
