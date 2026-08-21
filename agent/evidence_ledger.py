"""Minimal task-scoped evidence index for RWKV.

The ledger records which planned Task Records have retrieved source material.
It never decides whether RWKV may answer, whether an answer is correct, or
whether a source proves a semantic task_record.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from copy import deepcopy
from typing import Any, Mapping

from agent.evidence_records import assemble_grounded_candidates
from agent.runtime_contracts import EVIDENCE_LEDGER_CONTRACT

from agent.retrieval_object_contract import merge_mapping_rows
from agent.task_plan_contract import record_field_records, record_id, task_records


def _status_entity_terms(value: Any) -> set[str]:
    """Return lightweight entity tokens used only by page candidate helpers."""

    text = str(value or "").casefold()
    terms = set(re.findall(r"[a-z][a-z0-9_.+-]{1,}", text))
    terms.update(run for run in re.findall(r"[\u3400-\u9fff]{2,}", text))
    return terms


def _normalized_with_positions(value: str) -> tuple[str, list[int]]:
    output: list[str] = []
    positions: list[int] = []
    pending_space: int | None = None
    for index, char in enumerate(str(value or "")):
        if char.isspace():
            if output and pending_space is None:
                pending_space = index
            continue
        if pending_space is not None:
            output.append(" ")
            positions.append(pending_space)
            pending_space = None
        for folded in char.casefold():
            output.append(folded)
            positions.append(index)
    return "".join(output).strip(), positions


def _casefold_with_positions(value: str) -> tuple[str, list[int]]:
    """Case-fold text while retaining one raw offset per folded code point."""

    output: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(str(value or "")):
        for folded in char.casefold():
            output.append(folded)
            positions.append(index)
    return "".join(output), positions


def source_quote_view_with_positions(source_text: str) -> tuple[str, list[int]]:
    """Render Markdown as visible text while retaining raw-source positions.

    This is a transport projection for verbatim quote selection.  It removes
    only common presentation syntax (for example a link destination) and does
    not summarize, reorder, or semantically compare source text.
    """

    source = str(source_text or "")
    single_emphasis_positions: set[int] = set()
    for pattern in (
        re.compile(r"(?<!\*)\*(?=\S)(.+?)(?<=\S)\*(?!\*)", re.DOTALL),
        re.compile(r"(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])", re.DOTALL),
    ):
        for match in pattern.finditer(source):
            single_emphasis_positions.update((match.start(), match.end() - 1))
    visible: list[str] = []
    positions: list[int] = []
    index = 0
    line_start = True

    def append_range(start: int, end: int) -> None:
        cursor = start
        while cursor < end:
            if source.startswith(("**", "__", "~~"), cursor):
                cursor += 2
                continue
            if cursor in single_emphasis_positions:
                cursor += 1
                continue
            char = source[cursor]
            if char == "`":
                cursor += 1
                continue
            if char == "\\" and cursor + 1 < end:
                visible.append(source[cursor + 1])
                positions.append(cursor + 1)
                cursor += 2
                continue
            visible.append(char)
            positions.append(cursor)
            cursor += 1

    while index < len(source):
        char = source[index]
        if char == "\n":
            visible.append(char)
            positions.append(index)
            index += 1
            line_start = True
            continue

        if line_start:
            prefix = re.match(r"[ \t]{0,3}(?:#{1,6}|>|[-+*])(?:[ \t]+)", source[index:])
            if prefix:
                index += prefix.end()
                line_start = False
                continue
            fence = re.match(r"[ \t]{0,3}```[^\n]*", source[index:])
            if fence:
                index += fence.end()
                line_start = False
                continue
            if not char.isspace():
                line_start = False

        link_start = index + 1 if char == "!" and index + 1 < len(source) and source[index + 1] == "[" else index
        if source[link_start : link_start + 1] == "[":
            close = source.find("](", link_start + 1)
            if close >= 0:
                depth = 1
                cursor = close + 2
                while cursor < len(source) and depth:
                    if source[cursor] == "(":
                        depth += 1
                    elif source[cursor] == ")":
                        depth -= 1
                    cursor += 1
                if depth == 0:
                    append_range(link_start + 1, close)
                    index = cursor
                    continue

        if char == "<":
            close = source.find(">", index + 1)
            if close >= 0:
                inner = source[index + 1 : close]
                # CommonMark autolinks render the URI/address between the
                # angle brackets.  They are visible text, not HTML tags.
                if re.fullmatch(
                    r"[A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\s]*", inner
                ) or re.fullmatch(
                    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
                    r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?",
                    inner,
                ):
                    append_range(index + 1, close)
                    index = close + 1
                    continue
                if re.fullmatch(r"/?[A-Za-z][^>]*", inner):
                    index = close + 1
                    continue
        if source.startswith(("**", "__", "~~"), index):
            index += 2
            continue
        if index in single_emphasis_positions:
            index += 1
            continue
        if char == "`":
            index += 1
            continue
        if char == "\\" and index + 1 < len(source):
            visible.append(source[index + 1])
            positions.append(index + 1)
            index += 2
            continue
        if char in {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}:
            index += 1
            continue
        visible.append(char)
        positions.append(index)
        index += 1
    return "".join(visible), positions


def source_quote_view(source_text: str) -> str:
    """Return an exact visible-text projection used only by the quote locator."""

    return source_quote_view_with_positions(source_text)[0]


def _normalized_projection_with_positions(
    text: str,
    source_positions: list[int],
) -> tuple[str, list[int]]:
    output: list[str] = []
    positions: list[int] = []
    pending_space: int | None = None
    for index, char in enumerate(text):
        source_index = source_positions[index]
        if char.isspace():
            if output and pending_space is None:
                pending_space = source_index
            continue
        if pending_space is not None:
            output.append(" ")
            positions.append(pending_space)
            pending_space = None
        for folded in char.casefold():
            output.append(folded)
            positions.append(source_index)
    return "".join(output), positions


def _without_whitespace_with_positions(text: str) -> tuple[str, list[int]]:
    """Remove Unicode whitespace only and retain every raw character index."""

    output: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(str(text or "")):
        if char.isspace():
            continue
        output.append(char)
        positions.append(index)
    return "".join(output), positions


def _unique_without_whitespace_span(
    source: str,
    quote: str,
    *,
    minimum_characters: int = 16,
    maximum_raw_span: int = 5000,
) -> tuple[int, int] | None:
    """Locate one whitespace-only transport variant as a continuous raw span."""

    compact_source, positions = _without_whitespace_with_positions(source)
    compact_quote, _ = _without_whitespace_with_positions(quote)
    if len(compact_quote) < max(1, int(minimum_characters)) or not positions:
        return None

    matches: list[int] = []
    cursor = 0
    while True:
        match = compact_source.find(compact_quote, cursor)
        if match < 0:
            break
        matches.append(match)
        if len(matches) > 1:
            return None
        cursor = match + 1
    if len(matches) != 1:
        return None

    compact_start = matches[0]
    compact_end = compact_start + len(compact_quote) - 1
    if compact_end >= len(positions):
        return None
    start = positions[compact_start]
    end = positions[compact_end] + 1
    if end <= start or end - start > max(1, int(maximum_raw_span)):
        return None
    raw_span = source[start:end]
    if "".join(char for char in raw_span if not char.isspace()) != compact_quote:
        return None
    return start, end


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
        folded_source, folded_positions = _casefold_with_positions(source)
        folded_quote, _ = _casefold_with_positions(quote)
        folded_start = folded_source.find(folded_quote)
        if folded_start >= 0 and folded_positions and folded_quote:
            folded_end = folded_start + len(folded_quote) - 1
            if folded_end >= len(folded_positions):
                return None
            start = folded_positions[folded_start]
            end = folded_positions[folded_end] + 1
            # Expansion folds such as ß -> ss and İ -> i + combining dot
            # make folded-string offsets incompatible with raw slicing.  The
            # position map above is accepted only when the mapped raw span is
            # exactly the same text under Unicode case folding.
            if source[start:end].casefold() != quote.casefold():
                start = -1
            else:
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
            visible_source, visible_positions = source_quote_view_with_positions(source)
            normalized_visible, projected_positions = _normalized_projection_with_positions(
                visible_source,
                visible_positions,
            )
            visible_start = normalized_visible.find(normalized_quote)
            if visible_start >= 0 and projected_positions:
                visible_end = visible_start + len(normalized_quote) - 1
                if visible_end >= len(projected_positions):
                    return None
                start = projected_positions[visible_start]
                end = projected_positions[visible_end] + 1
                basis = "markdown_visible_exact"
            else:
                compact_span = _unique_without_whitespace_span(source, quote)
                if compact_span is None:
                    return None
                start, end = compact_span
                basis = "unique_without_whitespace"
    elif basis == "exact":
        end = start + len(quote)

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
        "evidence_origin": str(item.get("evidence_origin") or ""),
        "source_evidence_kind": str(item.get("evidence_kind") or ""),
        "content_type": str(item.get("content_type") or ""),
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
        "evidence_origin",
        "source_evidence_kind",
        "content_type",
        "published",
        "published_at",
        "updated",
        "updated_at",
        "date",
        "retrieved_at",
        "source_kind",
        "authority",
        "connector",
        "operation",
        "source_object",
        "object_alignment",
        "retrieval_request",
        "retrieval_bindings",
        "object_alignments",
        "retrieval_requests",
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
        or []
    )
    if isinstance(values, str):
        values = [values]
    return list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )[:8]


def _grounded_candidates(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return grounded spans with opaque, non-semantic record identities."""

    return assemble_grounded_candidates(item)


def _evidence_record(
    item: Mapping[str, Any],
    candidate: Mapping[str, Any],
    task_record_id: str,
    *,
    binding_origin: str = "rwkv_chunk_extractor",
) -> dict[str, Any]:
    """Build one immutable grounded candidate for a model-selected task record."""

    quote = str(candidate.get("quote") or "").strip()
    url = str(item.get("url") or "")
    chunk_id = str(candidate.get("chunk_id") or "")
    record_span_id = str(candidate.get("record_span_id") or "").strip()[:80]
    if record_span_id:
        span_identity = record_span_id
    else:
        # Compatibility for archived/structured candidates that predate the
        # assembler's opaque span ID.  Locator coordinates are part of the
        # atomic identity, so repeated text at two positions remains two
        # EvidenceRecords.
        locator = dict(candidate.get("source_locator") or {})
        span_identity = json.dumps(
            {
                "source_id": url or str(item.get("content_sha256") or ""),
                "chunk_id": str(locator.get("chunk_id") or chunk_id),
                "char_start": int(locator.get("char_start") or 0),
                "char_end": int(locator.get("char_end") or 0),
                "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    digest_input = "\n".join((task_record_id, span_identity))
    evidence_record_id = "E-" + hashlib.sha256(
        digest_input.encode("utf-8")
    ).hexdigest()[:20]
    record_match = str(
        candidate.get("record_match") or "evidence_record_candidate"
    ).strip().casefold()
    field_contract_valid = bool(candidate.get("field_contract_valid", True))
    # A chunk-local extractor observes one source record but cannot decide
    # whether it is globally current/latest or the unique requested record.
    # Preserve it as a candidate for the later full-set RWKV comparison.
    support_state = "rwkv_evidence_record_candidate"
    primary_alignment = deepcopy(
        dict(
            candidate.get("object_alignment")
            or item.get("object_alignment")
            or {}
        )
    )
    record = {
        "evidence_record_id": evidence_record_id,
        "task_record_id": task_record_id,
        "subject_key": str(candidate.get("subject_key") or "")[:300],
        "record_key": str(candidate.get("record_key") or "")[:300],
        "record_span_id": record_span_id,
        "parent_candidate_id": str(candidate.get("parent_candidate_id") or "")[:80],
        "assembly_basis": str(candidate.get("assembly_basis") or "")[:80],
        "atomic_quote_index": int(candidate.get("atomic_quote_index") or 0),
        "field_ids": [
            str(value)[:160]
            for value in candidate.get("field_ids") or []
            if str(value).strip()
        ][:16],
        "source_id": url or str(item.get("content_sha256") or ""),
        "title": str(item.get("title") or ""),
        "url": url,
        "evidence_origin": str(item.get("evidence_origin") or ""),
        "source_evidence_kind": str(item.get("evidence_kind") or ""),
        "content_type": str(item.get("content_type") or ""),
        "chunk_id": chunk_id,
        "chunk_index": int(candidate.get("chunk_index") or 0),
        "quote": quote[:1200],
        "source_locator": deepcopy(dict(candidate.get("source_locator") or {})),
        "grounding_basis": str(candidate.get("grounding_basis") or ""),
        "support_state": support_state,
        "record_match": record_match,
        "field_contract_valid": field_contract_valid,
        "binding_origin": binding_origin,
        "object_alignment": primary_alignment,
        "object_alignments": merge_mapping_rows(
            candidate.get("object_alignments"),
            candidate.get("object_alignment"),
            item.get("object_alignments"),
            item.get("object_alignment"),
        ),
        "rwkv_subject_alignment": deepcopy(
            dict(candidate.get("rwkv_subject_alignment") or {})
        ),
        "task_object_alignments": [
            deepcopy(dict(value))
            for value in candidate.get("task_object_alignments") or []
            if isinstance(value, Mapping)
        ][:8],
    }
    record["retrieval_bindings"] = merge_mapping_rows(
        item.get("retrieval_bindings")
    )
    record["retrieval_requests"] = merge_mapping_rows(
        item.get("retrieval_requests"),
        item.get("retrieval_request"),
    )
    for key in ("source_object", "retrieval_request"):
        value = item.get(key)
        if isinstance(value, Mapping):
            record[key] = deepcopy(dict(value))
    for key in (
        "source",
        "provider",
        "source_type",
        "evidence_origin",
        "source_evidence_kind",
        "content_type",
        "published",
        "published_at",
        "updated",
        "updated_at",
        "date",
        "retrieved_at",
        "source_kind",
        "authority",
        "connector",
        "operation",
    ):
        value = item.get(key)
        if value not in (None, "", [], {}):
            record[key] = deepcopy(value)
    freshness = item.get("freshness")
    if isinstance(freshness, Mapping):
        record["freshness"] = deepcopy(dict(freshness))
    return record


def _merge_evidence_record(
    current: dict[str, Any],
    incoming: Mapping[str, Any],
) -> bool:
    """Merge later transport observations into one immutable source span."""

    changed = False
    for key, singular_key in (
        ("object_alignments", "object_alignment"),
        ("retrieval_requests", "retrieval_request"),
        ("retrieval_bindings", ""),
        ("task_object_alignments", ""),
    ):
        merged = merge_mapping_rows(
            current.get(key),
            current.get(singular_key) if singular_key else None,
            incoming.get(key),
            incoming.get(singular_key) if singular_key else None,
        )
        if merged and merged != list(current.get(key) or []):
            current[key] = merged
            changed = True
    for key, value in incoming.items():
        if current.get(key) in (None, "", [], {}) and value not in (
            None,
            "",
            [],
            {},
        ):
            current[key] = deepcopy(value)
            changed = True
    return changed


def _merge_task_record_source(
    current: dict[str, Any],
    incoming: Mapping[str, Any],
) -> bool:
    """Keep one source view synchronized with its evidence records."""

    changed = False
    for key in ("grounded_spans", "evidence_records"):
        existing_rows = list(current.get(key) or [])
        incoming_rows = list(incoming.get(key) or [])
        combined: list[dict[str, Any]] = []
        seen: set[str] = set()
        # The incoming evidence-record view was rebuilt from the canonical
        # ledger record and therefore carries any newly observed route/object
        # metadata. Grounded text spans retain their stable first-seen order.
        ordered_rows = (
            [*incoming_rows, *existing_rows]
            if key == "evidence_records"
            else [*existing_rows, *incoming_rows]
        )
        for row in ordered_rows:
            if not isinstance(row, Mapping):
                continue
            marker = str(
                row.get("evidence_record_id")
                or "\n".join(
                    (
                        str(row.get("chunk_id") or ""),
                        str(row.get("text") or row.get("quote") or ""),
                    )
                )
            )
            if not marker or marker in seen:
                continue
            seen.add(marker)
            combined.append(deepcopy(dict(row)))
        if combined != existing_rows:
            current[key] = combined[:16]
            changed = True
    for key, singular_key in (
        ("object_alignments", "object_alignment"),
        ("retrieval_requests", "retrieval_request"),
        ("retrieval_bindings", ""),
    ):
        merged = merge_mapping_rows(
            current.get(key),
            current.get(singular_key) if singular_key else None,
            incoming.get(key),
            incoming.get(singular_key) if singular_key else None,
        )
        if merged and merged != list(current.get(key) or []):
            current[key] = merged
            changed = True
    for key, value in incoming.items():
        if current.get(key) in (None, "", [], {}) and value not in (
            None,
            "",
            [],
            {},
        ):
            current[key] = deepcopy(value)
            changed = True
    return changed


def _source_from_evidence_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility source view containing one grounded Evidence Record."""

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
                "field_ids": list(record.get("field_ids") or []),
                "subject_key": str(record.get("subject_key") or ""),
                "record_key": str(record.get("record_key") or ""),
                "record_span_id": str(record.get("record_span_id") or ""),
                "parent_candidate_id": str(record.get("parent_candidate_id") or ""),
                "assembly_basis": str(record.get("assembly_basis") or ""),
                "object_alignment": deepcopy(
                    dict(record.get("object_alignment") or {})
                ),
                "rwkv_subject_alignment": deepcopy(
                    dict(record.get("rwkv_subject_alignment") or {})
                ),
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
        "evidence_origin",
        "source_evidence_kind",
        "content_type",
        "published",
        "published_at",
        "updated",
        "updated_at",
        "date",
        "retrieved_at",
        "freshness",
        "source_kind",
        "authority",
        "connector",
        "operation",
        "source_object",
        "object_alignment",
        "retrieval_request",
        "retrieval_bindings",
        "object_alignments",
        "retrieval_requests",
    ):
        if record.get(key) not in (None, "", [], {}):
            source[key] = deepcopy(record[key])
    return source


class EvidenceLedger:
    """Record grounded candidate spans per RWKV-planned factual record.

    Search-route metadata never binds a whole page to a factual record. A web
    span is bound only when RWKV's chunk extractor names that record and the
    quote maps back to fetched text. Structured harness output may use the
    RWKV-selected tool route because it is already an exact typed record rather
    than a web page.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._task_records: dict[str, dict[str, Any]] = {}
        self._source_policy = "open_web"
        self._required_domains: list[str] = []
        self._query = ""
        self._unassigned_sources: list[dict[str, Any]] = []
        self._unassigned_attempts: list[dict[str, Any]] = []

    def reset(self) -> None:
        with self._lock:
            self._task_records.clear()
            self._source_policy = "open_web"
            self._required_domains = []
            self._query = ""
            self._unassigned_sources.clear()
            self._unassigned_attempts.clear()

    def initialize(self, task_plan: Mapping[str, Any] | None, query: str) -> None:
        plan = task_plan if isinstance(task_plan, Mapping) else {}
        records = task_records(plan, fallback_query=query)
        with self._lock:
            self.reset()
            self._query = str(query or "")
            # Source routing is resolved from the user request and runtime
            # connectors. The factual plan is deliberately not a policy gate.
            self._source_policy = "open_web"
            self._required_domains = []
            for task_record in records:
                task_record_id = record_id(task_record)
                self._task_records[task_record_id] = {
                    "task_record_id": task_record_id,
                    "question": str(task_record.get("question") or query or ""),
                    "subject": str(task_record.get("subject") or ""),
                    "relation": str(task_record.get("relation") or ""),
                    "fields": record_field_records(task_record),
                    "time_scope": str(task_record.get("time_scope") or "unspecified"),
                    "set_semantics": str(task_record.get("set_semantics") or "single"),
                    "premise_requires_verification": bool(
                        task_record.get("premise_requires_verification")
                    ),
                    "attempts": [],
                    "sources": [],
                    "evidence_records": [],
                }

    def task_record_ids(self) -> list[str]:
        with self._lock:
            return list(self._task_records)

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
        task_record_id: str = "",
        strategy: str = "",
        step: int = 0,
    ) -> dict[str, Any]:
        items = [item for item in result.get("results") or [] if isinstance(item, Mapping)]
        with self._lock:
            route_target = task_record_id if task_record_id in self._task_records else ""
            added = 0
            added_records = 0
            updated_records = 0
            added_unassigned = 0
            touched: set[str] = set()

            if route_target:
                self._task_records[route_target]["attempts"].append(
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
                        if value in self._task_records
                    ]
                    for task_record_id in targets:
                        assigned_records.append(
                            _evidence_record(item, candidate, task_record_id)
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
                                "field_ids": [],
                                "record_match": "evidence_record_candidate",
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
                    task_record_id = str(evidence_record["task_record_id"])
                    task_record = self._task_records[task_record_id]
                    record_id = str(evidence_record["evidence_record_id"])
                    existing_record = next(
                        (
                            row
                            for row in task_record["evidence_records"]
                            if str(row.get("evidence_record_id") or "") == record_id
                        ),
                        None,
                    )
                    if existing_record is None:
                        task_record["evidence_records"].append(deepcopy(evidence_record))
                        canonical_record = evidence_record
                        added_records += 1
                    else:
                        if _merge_evidence_record(existing_record, evidence_record):
                            updated_records += 1
                            touched.add(task_record_id)
                        canonical_record = existing_record
                    source = _source_from_evidence_record(canonical_record)
                    identity = source["url"] or source["title"]
                    existing_source = next(
                        (
                            row
                            for row in task_record["sources"]
                            if identity
                            and (row.get("url") or row.get("title")) == identity
                        ),
                        None,
                    )
                    if existing_source is None:
                        task_record["sources"].append(source)
                        added += 1
                    else:
                        if _merge_task_record_source(existing_source, source):
                            touched.add(task_record_id)
                    touched.add(task_record_id)

            return {
                "added_source_bindings": added,
                "added_evidence_records": added_records,
                "updated_evidence_records": updated_records,
                "added_unassigned_sources": added_unassigned,
                "touched_task_record_ids": sorted(touched),
                "task_record_count": len(self._task_records),
                "unassigned_source_count": len(self._unassigned_sources),
            }

    def snapshot(self, *, max_spans_per_record: int = 4) -> dict[str, Any]:
        span_limit = max(0, int(max_spans_per_record or 0))
        with self._lock:
            task_record_rows = []
            for task_record in self._task_records.values():
                source_records = deepcopy(task_record.get("sources") or [])
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
                task_record_rows.append(
                    {
                        "task_record_id": task_record["task_record_id"],
                        "question": task_record["question"],
                        "subject": task_record.get("subject") or "",
                        "relation": task_record.get("relation") or "",
                        "fields": deepcopy(task_record["fields"]),
                        "time_scope": task_record["time_scope"],
                        "set_semantics": task_record.get("set_semantics") or "single",
                        "premise_requires_verification": bool(
                            task_record.get("premise_requires_verification")
                        ),
                        "retrieval_state": (
                            "evidence_recorded"
                            if task_record.get("evidence_records")
                            else "not_recorded"
                        ),
                        "attempt_count": len(task_record.get("attempts") or []),
                        "source_count": len(sources),
                        "evidence_record_count": len(
                            task_record.get("evidence_records") or []
                        ),
                        # All page/structured Evidence Records remain candidates
                        # until a full-set RWKV comparison.
                        "exact_record_count": 0,
                        "evidence_records": deepcopy(
                            task_record.get("evidence_records") or []
                        )[:32],
                        "sources": sources[:8],
                        # Compatibility alias for old trace readers. It is an
                        # evidence index, not a semantic support decision.
                        "evidence": sources[:8],
                    }
                )
            return {
                "contract": EVIDENCE_LEDGER_CONTRACT,
                "source_policy": self._source_policy,
                "required_domains": list(self._required_domains),
                "task_record_count": len(task_record_rows),
                "task_records": task_record_rows,
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


__all__ = [
    "EvidenceLedger",
    "locate_grounded_quote_span",
    "source_quote_view",
    "source_quote_view_with_positions",
]
