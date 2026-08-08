"""Minimal task-scoped evidence index for RWKV.

The ledger records which planned task points have retrieved source material.
It never decides whether RWKV may answer, whether an answer is correct, or
whether a source proves a semantic claim.
"""

from __future__ import annotations

import re
import threading
from copy import deepcopy
from typing import Any, Mapping


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
    chunks = [
        {
            "chunk_id": str(row.get("chunk_id") or ""),
            "index": int(row.get("index") or 0),
            "text": str(row.get("text") or "")[:2400],
        }
        for row in item.get("source_chunks") or []
        if isinstance(row, Mapping) and str(row.get("text") or "").strip()
    ][:8]
    return {
        "title": str(item.get("title") or ""),
        "url": str(item.get("url") or ""),
        "retrieval_query": str(item.get("retrieval_query") or ""),
        "text_available": bool(_source_text(item)),
        "chunk_count": len(chunks) or int(item.get("chunk_count") or 0),
        "chunks": chunks,
    }


class ClaimLedger:
    """Record retrieved material per RWKV-planned task point."""

    VERSION = "claim-ledger.v2"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._claims: dict[str, dict[str, Any]] = {}
        self._source_policy = "open_web"
        self._required_domains: list[str] = []
        self._query = ""

    def reset(self) -> None:
        with self._lock:
            self._claims.clear()
            self._source_policy = "open_web"
            self._required_domains = []
            self._query = ""

    def initialize(self, task_plan: Mapping[str, Any] | None, query: str) -> None:
        plan = task_plan if isinstance(task_plan, Mapping) else {}
        points = [point for point in plan.get("atomic_points") or [] if isinstance(point, Mapping)]
        if not points:
            points = [{"id": "P1", "task": str(query or ""), "objective": str(query or "")}]
        with self._lock:
            self.reset()
            self._query = str(query or "")
            self._source_policy = str(plan.get("source_policy") or "open_web")
            self._required_domains = [
                str(value).casefold().strip().removeprefix("www.").rstrip(".")
                for value in plan.get("required_domains") or []
                if str(value).strip()
            ]
            for index, point in enumerate(points, start=1):
                claim_id = str(point.get("id") or f"P{index}").strip() or f"P{index}"
                self._claims[claim_id] = {
                    "claim_id": claim_id,
                    "task": str(point.get("task") or point.get("objective") or query or ""),
                    "objective": str(point.get("objective") or point.get("task") or ""),
                    "evidence_needed": [
                        str(value) for value in point.get("evidence_needed") or [] if str(value).strip()
                    ],
                    "attempts": [],
                    "sources": [],
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
            explicit_targets = [task_point_id] if task_point_id in self._claims else []
            added = 0
            touched: set[str] = set()
            for item in items:
                item_claim_ids = [
                    str(value) for value in item.get("claim_ids") or [] if str(value) in self._claims
                ]
                targets = list(dict.fromkeys([*explicit_targets, *item_claim_ids]))
                if not targets:
                    targets = list(self._claims)
                record = _source_record(item)
                for claim_id in targets:
                    claim = self._claims[claim_id]
                    claim["attempts"].append(
                        {
                            "query": str(query or ""),
                            "strategy": str(strategy or ""),
                            "step": int(step or 0),
                            "url": record["url"],
                        }
                    )
                    identity = record["url"] or record["title"]
                    if identity and any(
                        (row.get("url") or row.get("title")) == identity
                        for row in claim["sources"]
                    ):
                        continue
                    claim["sources"].append(deepcopy(record))
                    touched.add(claim_id)
                    added += 1
            if not items and explicit_targets:
                for claim_id in explicit_targets:
                    self._claims[claim_id]["attempts"].append(
                        {
                            "query": str(query or ""),
                            "strategy": str(strategy or ""),
                            "step": int(step or 0),
                            "url": "",
                        }
                    )
            return {
                "added_source_bindings": added,
                "touched_claim_ids": sorted(touched),
                "claim_count": len(self._claims),
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
                        if key != "chunks"
                    }
                    if span_limit:
                        projected["spans"] = list(source.get("chunks") or [])[:span_limit]
                    sources.append(projected)
                claims.append(
                    {
                        "claim_id": claim["claim_id"],
                        "task": claim["task"],
                        "objective": claim["objective"],
                        "evidence_needed": deepcopy(claim["evidence_needed"]),
                        "retrieval_state": "retrieved" if sources else "not_retrieved",
                        "attempt_count": len(claim.get("attempts") or []),
                        "source_count": len(sources),
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
                "advisory_only": True,
            }


__all__ = ["ClaimLedger", "locate_grounded_quote_span"]
