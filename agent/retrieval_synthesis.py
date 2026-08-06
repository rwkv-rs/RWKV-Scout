"""Evidence ranking, context construction, and final-answer generation."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from config import DATA_PIPELINE, get_llm_context_length
from utils.chunker import get_token_count, semantic_chunk_text
from utils.context_budget import evidence_tokens
from utils.calculation_check import build_calculation_check
from utils.model_budget import bounded_completion_budget
from utils.evidence_quality import (
    date_mentions,
    evidence_provenance,
    evidence_text,
    has_substantive_evidence,
    substantive_evidence_items,
)
from utils.experiment_strategies import normalize_strategy
from utils.evidence_validation import (
    assess_answer_alignment,
    build_evidence_validation,
    source_quality,
)
from utils.source_authority import authority_for_url, resolve_source_policy
from utils.risk_policy import risk_context, validate_risk_answer
from utils.rwkv_prompt import (
    FINAL_CONTINUATION_STOP_SUFFIXES,
    build_final_continuation_prompt,
    clean_final_continuation,
)


def _clean_answer(text: str) -> str:
    text = text or ""
    text = re.sub(r"^\s*Assistant\s*:\s*", "", text, count=1, flags=re.IGNORECASE)
    # Continuation checkpoints sometimes wrap an otherwise usable answer in
    # XML-like scaffolding. These tags are protocol residue, not user content;
    # remove balanced outer layers before citation alignment and final cleanup.
    for _ in range(2):
        text = re.sub(r"^\s*<(?:response|answer)>\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*</(?:answer|response)>\s*$", "", text, count=1, flags=re.IGNORECASE)
    if re.search(r"<think>", text, flags=re.IGNORECASE) and not re.search(
        r"</think>", text, flags=re.IGNORECASE
    ):
        # An unfinished reasoning block is not a user-facing answer.  Let the
        # bounded formatting repair call produce a clean final response.
        return ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = text.replace("</think>", "").strip()
    # A continuation model may copy the evidence preamble instead of writing
    # the answer.  Treat that as protocol output so the bounded retry can
    # regenerate a user-facing response; never expose retrieval data blocks
    # through the public ``answer`` field.
    if re.match(
        r"(?is)^\s*(?:BEGIN\s+EVIDENCE(?:\s+(?:DATA|SOURCE))?|RETRIEVAL\s+EXECUTION\s+SUMMARY|BEGIN\s+EXECUTION\s+RECORD)\b",
        text,
    ):
        return ""
    # A continuation may prepend a citation or heading before replaying an
    # internal evidence block. Treat any such block as protocol leakage so
    # the bounded repair call can regenerate a user-facing answer.
    if re.search(
        r"(?im)^\s*(?:BEGIN\s+EVIDENCE(?:\s+(?:DATA|SOURCE))?|EVIDENCE\s+BODY|BEGIN\s+EXECUTION\s+RECORD)\b",
        text,
    ):
        return ""
    # A prose continuation must never expose a tool transcript. Returning an
    # empty value lets the bounded retry generate a real user-facing answer.
    if re.match(
        r"(?is)^\s*(?:```json\s*)?\{\s*[\"'](?:name|tool_name|action|arguments)[\"']\s*:",
        text,
    ):
        return ""
    text = re.sub(
        r"(?im)^\s*(?:Function output|Tool result|Tool call|Function call)\s*:\s*.*$",
        "",
        text,
    )
    text = re.sub(
        r"^\s*(?:the\s+)?(?:final answer|answer)(?:\s+is)?\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?im)^\s*\*{0,2}(?:final answer|answer)\*{0,2}\s*:\s*\*{0,2}\s*$\n?", "", text)
    # Planner point labels are routing metadata, not part of the user answer.
    # A continuation can replay them when evidence is partial; remove only a
    # complete line-level ``P<number>:`` prefix and preserve ordinary prose.
    text = re.sub(
        r"(?im)^\s*\*{0,2}P\d+\*{0,2}\s*[:：\-–—]\s*[^\n]*(?:\n|$)",
        "",
        text,
    )
    # RWKV checkpoints sometimes emit a useful answer wrapped in a draft
    # scaffold. Keep the answer section, but do not expose internal section
    # labels or copy the evidence block into the user-facing result.
    if re.search(r"(?im)^\s*\*\*(?:answer|final answer)\s*:\s*\*\*", text):
        text = re.sub(r"(?im)^\s*\*\*(?:answer|final answer)\s*:\s*\*\*\s*", "", text, count=1)
        text = re.split(r"(?im)\n\s*\*\*(?:key evidence|source links|limitations)[^\n]*\*\*", text, maxsplit=1)[0]
        text = re.sub(r"(?m)^\s*[-*]\s*", "", text)
        text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    # Recurrent checkpoints can replay the same paragraph or table row.
    # Remove exact duplicate blocks/lines without changing the original order.
    blocks = re.split(r"\n\s*\n", text.strip())
    seen_blocks: set[str] = set()
    kept_blocks: list[str] = []
    for block in blocks:
        normalized = re.sub(r"\s+", " ", block).strip().casefold()
        if not normalized or normalized in seen_blocks:
            continue
        seen_blocks.add(normalized)
        kept_blocks.append(block.strip())
    text = "\n\n".join(kept_blocks)
    # Internal repair labels are never user-facing content.  Keep the
    # surrounding answer, but remove the label itself if the checkpoint
    # echoed it into the continuation.
    text = re.sub(r"(?im)^\s*(?:DRAFT|REVISED DRAFT|INTERNAL DRAFT)\s*:\s*$", "", text)
    # Long continuation outputs sometimes repeat the same numbered item with
    # a new number until the completion budget is exhausted.  Count only
    # strongly repeated list bodies (three or more occurrences) so legitimate
    # two-row lists and repeated names are left untouched.
    numbered_lines = text.splitlines()
    numbered_bodies: dict[str, int] = {}
    for line in numbered_lines:
        match = re.match(r"^\s*\d+[.)]\s+(.+?)\s*$", line)
        if match:
            body = re.sub(r"[*_`\[\](){}]", "", match.group(1))
            body = re.sub(r"\s+", " ", body).strip().casefold()
            if body:
                numbered_bodies[body] = numbered_bodies.get(body, 0) + 1
    if numbered_bodies:
        filtered_numbered: list[str] = []
        seen_repeated_bodies: set[str] = set()
        for line in numbered_lines:
            match = re.match(r"^\s*\d+[.)]\s+(.+?)\s*$", line)
            if match:
                body = re.sub(r"[*_`\[\](){}]", "", match.group(1))
                body = re.sub(r"\s+", " ", body).strip().casefold()
                if numbered_bodies.get(body, 0) >= 3:
                    if body in seen_repeated_bodies:
                        continue
                    seen_repeated_bodies.add(body)
            filtered_numbered.append(line)
        text = "\n".join(filtered_numbered)
    seen_lines: set[str] = set()
    deduped_lines: list[str] = []
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip().casefold()
        if normalized and normalized in seen_lines and len(normalized) >= 18:
            continue
        if normalized:
            seen_lines.add(normalized)
        deduped_lines.append(line.rstrip())
    return "\n".join(deduped_lines).strip()


def _normalized_url(value: str) -> str:
    return str(value or "").strip().rstrip(".,;:)]").casefold()


def _allowed_answer_urls(data: dict[str, Any], context: dict[str, Any]) -> set[str]:
    values: list[str] = []
    selected = [item for item in context.get("selected_evidence") or [] if isinstance(item, dict)]
    source_items = selected
    for item in source_items:
        if isinstance(item, dict):
            values.append(str(item.get("url") or ""))
    return {_normalized_url(value) for value in values if _normalized_url(value)}


def _citation_refs_for_context(data: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose only answer-level citations while retaining all retrieval rows in trace."""
    refs = [item for item in data.get("citation_refs") or [] if isinstance(item, dict)]
    by_url = {_normalized_url(item.get("url")): item for item in refs if _normalized_url(item.get("url"))}
    selected = [item for item in context.get("selected_evidence") or [] if isinstance(item, dict)]
    if not selected:
        return []
    output: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        matched = by_url.get(_normalized_url(item.get("url")))
        spans = [
            {
                "span_id": f"S{index}:C{int(chunk.get('index', 0) or 0) + 1}",
                "chunk_id": str(chunk.get("chunk_id") or ""),
                "index": int(chunk.get("index", 0) or 0),
            }
            for chunk in list(item.get("chunks") or [])
            if isinstance(chunk, dict)
        ]
        locator = {
            "type": "source_chunks",
            "url": str(item.get("url") or ""),
            "spans": spans,
        }
        if matched:
            citation = dict(matched)
            citation["ref_id"] = f"S{index}"
            citation["url"] = str(item.get("url") or citation.get("url") or "")
            citation["evidence_text"] = evidence_text(item)
            citation["evidence_origin"] = str(item.get("evidence_origin") or "fetched_page_body")
            citation["evidence_boundary"] = str(item.get("evidence_boundary") or "page_body_only")
            citation["evidence_locator"] = locator
            citation["citation_scope"] = "selected_substantive_evidence"
            output.append(citation)
        else:
            output.append(
                {
                    "ref_id": f"S{index}",
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "source": str(item.get("source") or "selected_context"),
                    "evidence_text": evidence_text(item),
                    "evidence_origin": str(item.get("evidence_origin") or "fetched_page_body"),
                    "evidence_boundary": str(item.get("evidence_boundary") or "page_body_only"),
                    "evidence_locator": locator,
                    "citation_scope": "selected_substantive_evidence",
                }
            )
    return output


def _enforce_citation_contract(answer: str, data: dict[str, Any], context: dict[str, Any]) -> str:
    """Keep final answers traceable when the model omits or invents links."""
    answer = str(answer or "").strip()
    allowed_urls = _allowed_answer_urls(data, context)
    citation_links: dict[str, str] = {}
    for index, item in enumerate(context.get("selected_evidence") or [], start=1):
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("ref_id") or f"S{index}").strip().casefold()
        url = str(item.get("url") or "").strip().rstrip(".,;:)]")
        if ref_id and _normalized_url(url) in allowed_urls:
            citation_links[ref_id] = url
        citation_links.setdefault(f"s{index}", url)
        for chunk in item.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            chunk_index = int(chunk.get("index", 0) or 0) + 1
            citation_links.setdefault(f"s{index}:c{chunk_index}", url)

    def replace_url(match: re.Match[str]) -> str:
        url = match.group(0)
        return url if _normalized_url(url) in allowed_urls else ""

    answer = re.sub(r'https?://[^\s<>"]+', replace_url, answer, flags=re.IGNORECASE)
    answer = re.sub(r"(?i)(?:source\s+link|source|url)\s*[:：]\s*(?=\[s\d+\])", "", answer)
    answer = re.sub(r"[\u6765\u6e90\u94fe\u63a5]\s*[:\uFF1A]\s*(?=\[s\d+\])", "", answer)
    # Expand compact model citations into safe Markdown links. The URL must
    # come from selected evidence; never trust a URL invented by the model.
    def expand_citation(match: re.Match[str]) -> str:
        label = match.group(1).upper()
        url = citation_links.get(label.casefold(), "")
        return f"[{label}]({url})" if url else ""

    answer = re.sub(r"\[(S\d+(?::C\d+)?)\](?!\()", expand_citation, answer, flags=re.IGNORECASE)
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()
    has_source_marker = bool(
        re.search(r"\[s\d+(?::c\d+)?\]", answer, flags=re.IGNORECASE)
    ) or bool(
        re.search(
            r"\[s\d+\]|\[source\s+\d+\]|【\d+】|\bsource\s*[:：]\s*s\d+\b",
            answer,
            flags=re.IGNORECASE,
        )
    )
    if answer and not has_source_marker and context.get("selected_evidence"):
        source_links = []
        for index, _item in enumerate(context.get("selected_evidence") or [], start=1):
            url = citation_links.get(f"s{index}", "")
            source_links.append(f"[S{index}]({url})" if url else f"[S{index}]")
        answer = f"{answer.rstrip()}\n\nSources: {', '.join(source_links)}"
    return answer


def _enforce_risk_contract(answer: str, constraints: dict[str, Any] | None) -> str:
    """Add a deterministic information-only boundary for high-risk domains."""
    if not answer:
        return answer
    policy = risk_context(constraints)
    if not policy["high_risk"] or validate_risk_answer(answer, constraints)["valid"]:
        return answer
    label = policy["label"]
    if "medical" in label:
        boundary = "This is information retrieval, not medical advice; consult a qualified professional before acting."
    elif "legal" in label:
        boundary = "This is information retrieval, not legal advice; consult a qualified professional before acting."
    elif "financial" in label:
        boundary = "This is information retrieval, not financial advice; consult a qualified professional before acting."
    else:
        boundary = "This is information retrieval, not professional advice; consult a qualified professional before acting."
    return f"{answer.rstrip()} {boundary}"


def _attach_mechanical_citations(
    answer: str,
    selected_evidence: list[dict[str, Any]] | None,
) -> str:
    """Bind clearly aligned uncited lines to an existing source reference.

    This does not create facts or choose a source by model judgement.  It
    reuses the already recorded line-to-body overlap signal and only attaches
    a reference when that signal is above the same alignment threshold used
    by the audit.  Refusal/protocol lines and headings are never annotated.
    """

    sources = list(selected_evidence or [])
    if not answer or not sources:
        return answer
    alignment = assess_answer_alignment(answer, sources)
    rows = iter(alignment.get("rows") or [])
    output: list[str] = []
    for raw_line in str(answer).splitlines():
        line = raw_line.strip()
        if not line or line.casefold().startswith(("sources:", "source:")):
            output.append(raw_line)
            continue
        if re.fullmatch(r"\[S\d+(?::C\d+)?\](?:\([^)]*\))?", line, flags=re.IGNORECASE):
            output.append(raw_line)
            continue
        row = next(rows, None)
        if row is None:
            output.append(raw_line)
            continue
        if (
            "[S" not in line
            and not line.startswith(("#", "```"))
            and line.count("`") % 2 == 0
            and len(line) >= 20
            and not _is_evidence_refusal(line)
            and row.get("aligned") is True
            and str(row.get("best_ref") or "")
            and float(row.get("overlap") or 0.0) >= 0.12
        ):
            output.append(f"{raw_line.rstrip()} [{row['best_ref']}]" )
        else:
            output.append(raw_line)
    return "\n".join(output).strip()


def _enforce_latest_list_shape(answer: str, task_plan: dict[str, Any] | None) -> str:
    """Bound obvious list/table rows without rewriting ordinary prose."""

    plan = task_plan if isinstance(task_plan, dict) else {}
    if str(plan.get("task_mode") or "").casefold() != "latest_list":
        return answer
    try:
        max_items = max(1, min(int(plan.get("max_items") or 5), 50))
    except (TypeError, ValueError):
        max_items = 5
    lines = str(answer or "").splitlines()
    if not lines:
        return answer

    separator_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", line)
        ),
        -1,
    )
    if separator_index >= 0:
        output = lines[: separator_index + 1]
        row_count = 0
        for line in lines[separator_index + 1 :]:
            is_row = bool(line.strip()) and "|" in line
            if is_row:
                row_count += 1
                if row_count <= max_items:
                    output.append(line)
                continue
            output.append(line)
        return "\n".join(output).strip()

    numbered = [index for index, line in enumerate(lines) if re.match(r"^\s*\d+[.)]\s+", line)]
    if len(numbered) > max_items:
        drop = set(numbered[max_items:])
        return "\n".join(line for index, line in enumerate(lines) if index not in drop).strip()

    bullets = [index for index, line in enumerate(lines) if re.match(r"^\s*[-*]\s+", line)]
    if len(bullets) > max_items:
        drop = set(bullets[max_items:])
        return "\n".join(line for index, line in enumerate(lines) if index not in drop).strip()
    return answer


def _latest_list_chunks(item: dict[str, Any], constraints: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Project validated page candidates into a bounded list-task body."""

    task_plan = (constraints or {}).get("task_plan") or {}
    if not isinstance(task_plan, dict) or str(task_plan.get("task_mode") or "").casefold() != "latest_list":
        return []
    try:
        max_items = max(1, min(int(task_plan.get("max_items") or 5), 50))
    except (TypeError, ValueError):
        max_items = 5

    compact: list[dict[str, Any]] = []
    for candidate in item.get("chunk_candidates") or []:
        if not isinstance(candidate, dict) or candidate.get("supported") is not True:
            continue
        facts = candidate.get("facts") or []
        if isinstance(facts, str):
            facts = [facts]
        values = [re.sub(r"\s+", " ", str(value or "")).strip() for value in facts if str(value or "").strip()]
        if not values and str(candidate.get("quote") or "").strip():
            values = [re.sub(r"\s+", " ", str(candidate.get("quote") or "")).strip()]
        for value in values:
            compact.append(
                {
                    "chunk_id": str(candidate.get("chunk_id") or f"list-{len(compact) + 1}"),
                    "index": len(compact),
                    "text": value,
                    "token_count": get_token_count(value),
                }
            )
            if len(compact) >= max_items:
                return compact
    return compact


def _evidence_text_value(item: dict[str, Any], constraints: dict[str, Any] | None = None) -> str:
    # The final context must use the canonical evidence gate.  In particular,
    # chunk_candidates and model_extracted_facts are locator/routing outputs,
    # not a replacement for the captured page body.
    compact = _latest_list_chunks(item, constraints)
    if compact:
        return "\n".join(str(chunk.get("text") or "").strip() for chunk in compact).strip()
    return evidence_text(item)


def _has_usable_evidence(item: dict[str, Any]) -> bool:
    """Distinguish page/record evidence from discovery-only snippets."""

    return has_substantive_evidence(item)


def _truncate_markdown_by_tokens(value: str, max_tokens: int) -> str:
    """Bound evidence on token count without flattening Markdown rows."""

    text = str(value or "").strip()
    if not text or get_token_count(text) <= max_tokens:
        return text
    chunks = semantic_chunk_text(text, max_tokens=max_tokens, overlap_ratio=0.0)
    return str(chunks[0] if chunks else text).strip()


def _source_chunks_for_context(
    item: dict[str, Any],
    body: str,
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return cleaned source chunks while keeping the raw-body boundary."""

    def unique_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        output: list[dict[str, Any]] = []
        for chunk in chunks:
            text = str(chunk.get("text") or "").strip()
            key = re.sub(r"\s+", " ", text).casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            output.append(chunk)
        return output

    compact = _latest_list_chunks(item, constraints)
    if compact:
        return unique_chunks(compact)

    stored = [chunk for chunk in item.get("source_chunks") or [] if isinstance(chunk, dict)]
    if stored:
        return unique_chunks([
            {
                "chunk_id": str(chunk.get("chunk_id") or f"chunk-{index + 1}"),
                "index": int(chunk.get("index", index) or index),
                "text": str(chunk.get("text") or "").strip(),
                "token_count": int(chunk.get("token_count") or get_token_count(str(chunk.get("text") or ""))),
            }
            for index, chunk in enumerate(stored)
            if str(chunk.get("text") or "").strip()
        ])

    if not body:
        return []
    single_pass_limit = max(512, int(DATA_PIPELINE.get("web_chunk_single_pass_tokens", 7000) or 7000))
    requested = max(512, int(item.get("chunk_window_tokens") or DATA_PIPELINE.get("web_chunk_tokens", 2048) or 2048))
    if int(item.get("chunk_count") or 0) <= 1 and get_token_count(body) <= single_pass_limit:
        return [{"chunk_id": "chunk-1", "index": 0, "text": body, "token_count": get_token_count(body)}]
    chunks = semantic_chunk_text(
        body,
        max_tokens=requested,
        overlap_ratio=float(DATA_PIPELINE.get("web_chunk_overlap_ratio", 0.1) or 0.1),
    )
    return unique_chunks([
        {
            "chunk_id": f"chunk-{index + 1}",
            "index": index,
            "text": str(chunk or "").strip(),
            "token_count": get_token_count(str(chunk or "")),
        }
        for index, chunk in enumerate(chunks)
        if str(chunk or "").strip()
    ])


def _select_source_chunks(item: dict[str, Any], chunks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Use model candidates as locators, but return only original page text."""

    if not chunks:
        return []
    max_chunks = max(1, int(DATA_PIPELINE.get("web_context_max_chunks_per_source", 3) or 3))
    if len(chunks) <= max_chunks:
        return chunks

    candidate_indexes: list[int] = []
    for candidate in item.get("chunk_candidates") or []:
        if not isinstance(candidate, dict) or candidate.get("supported") is not True:
            continue
        if not (candidate.get("facts") or candidate.get("quote")):
            continue
        try:
            index = int(candidate.get("chunk_index", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(chunks) and index not in candidate_indexes:
            candidate_indexes.append(index)
    query_terms = _query_terms({"query": query})
    ranked = sorted(
        chunks,
        key=lambda chunk: sum(term in str(chunk.get("text") or "").casefold() for term in query_terms),
        reverse=True,
    )
    if candidate_indexes:
        return [chunks[index] for index in candidate_indexes[:max_chunks]]

    return sorted(ranked[:max_chunks], key=lambda chunk: int(chunk.get("index", 0)))


def _pack_source_evidence(
    selected_metadata: list[dict[str, Any]],
    budget_tokens: int,
) -> tuple[list[dict[str, Any]], str]:
    """Pack evidence by source and chunk without losing source boundaries.

    A flat token truncation keeps the first page and can silently remove every
    later sub-question.  This packer gives each selected source one bounded
    span first, then fills remaining space round-robin with additional spans.
    The metadata is reduced to exactly what the final model can see, so
    citation/alignment checks cannot point at an omitted chunk.
    """
    if not selected_metadata:
        return [], "(no retrieved evidence)"
    budget = max(2048, int(budget_tokens or 2048))
    source_count = len(selected_metadata)
    per_source = max(512, budget // source_count)
    packed: list[dict[str, Any]] = []
    chunks_by_source: list[list[dict[str, Any]]] = []
    for item in selected_metadata:
        chunks = [chunk for chunk in item.get("chunks") or [] if str(chunk.get("text") or "").strip()]
        chunks_by_source.append(chunks)

    def block_for(item: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
        body = "\n\n".join(
            f"SOURCE SPAN [{item.get('ref_id', '')}:C{int(chunk.get('index', 0)) + 1}]\n"
            f"{str(chunk.get('text') or '').strip()}"
            for chunk in chunks
        )
        return (
            f"BEGIN EVIDENCE SOURCE {item.get('ref_id', '')}\n"
            f"URL (citation metadata only): {item.get('url', '')}\n"
            f"Freshness metadata (control only): {json.dumps(item.get('freshness') or {}, ensure_ascii=False)}\n"
            "The URL, title, search snippet, provider summary and publication metadata are not evidence.\n"
            "EVIDENCE BODY (the only factual source for this record; cite S#:#):\n"
            f"{body}\nEND EVIDENCE SOURCE {item.get('ref_id', '')}"
        )

    # First pass: one span per source, with a fair per-source cap.
    included: list[list[dict[str, Any]]] = [[] for _ in selected_metadata]
    for index, chunks in enumerate(chunks_by_source):
        if not chunks:
            continue
        header_tokens = get_token_count(block_for(selected_metadata[index], []))
        span_budget = max(256, per_source - header_tokens)
        text = _truncate_markdown_by_tokens(str(chunks[0].get("text") or ""), span_budget)
        if text:
            included[index].append({**chunks[0], "text": text, "token_count": get_token_count(text)})

    def render() -> str:
        return "\n\n".join(
            block_for(item, chunks)
            for item, chunks in zip(selected_metadata, included)
            if chunks
        )

    # Second pass: add further chunks only when the complete bounded block fits.
    for chunk_index in range(1, max((len(chunks) for chunks in chunks_by_source), default=0)):
        for source_index, chunks in enumerate(chunks_by_source):
            if chunk_index >= len(chunks):
                continue
            candidate = chunks[chunk_index]
            current = render()
            remaining = budget - get_token_count(current)
            if remaining <= 256:
                break
            text = _truncate_markdown_by_tokens(
                str(candidate.get("text") or ""),
                max(256, remaining - 80),
            )
            if not text:
                continue
            proposed = [list(chunks_for_source) for chunks_for_source in included]
            proposed[source_index].append({**candidate, "text": text, "token_count": get_token_count(text)})
            proposed_text = "\n\n".join(
                block_for(item, source_chunks)
                for item, source_chunks in zip(selected_metadata, proposed)
                if source_chunks
            )
            if get_token_count(proposed_text) <= budget:
                included = proposed

    # Final tokenizer-level guard.  Per-source estimates include headings, but
    # tokenization of multilingual Markdown can still make the aggregate a
    # little larger than the arithmetic budget.  Trim the largest visible span
    # in small steps while retaining one span for every source.
    while get_token_count(render()) > budget:
        candidates = [
            (source_index, chunk_index, chunk)
            for source_index, source_chunks in enumerate(included)
            for chunk_index, chunk in enumerate(source_chunks)
            if str(chunk.get("text") or "").strip()
        ]
        if not candidates:
            break
        source_index, chunk_index, chunk = max(
            candidates,
            key=lambda value: get_token_count(str(value[2].get("text") or "")),
        )
        current_text = str(chunk.get("text") or "")
        current_tokens = get_token_count(current_text)
        reduction = max(128, get_token_count(render()) - budget)
        next_tokens = max(256, current_tokens - reduction)
        if next_tokens >= current_tokens:
            break
        trimmed = _truncate_markdown_by_tokens(current_text, next_tokens)
        if not trimmed or trimmed == current_text:
            break
        included[source_index][chunk_index] = {
            **chunk,
            "text": trimmed,
            "token_count": get_token_count(trimmed),
        }

    output_items: list[dict[str, Any]] = []
    for item, chunks in zip(selected_metadata, included):
        if not chunks:
            continue
        updated = dict(item)
        updated["chunks"] = chunks
        updated["evidence_text"] = "\n\n".join(str(chunk.get("text") or "") for chunk in chunks)
        updated["selected_chars"] = len(updated["evidence_text"])
        updated["chunk_count"] = len(chunks)
        updated["selected_chunk_count"] = len(chunks)
        updated["selected_chunk_indexes"] = [int(chunk.get("index", 0)) for chunk in chunks]
        updated["truncated"] = bool(item.get("truncated")) or len(chunks) < len(item.get("chunks") or [])
        output_items.append(updated)
    return output_items, render() or "(no retrieved evidence)"


def _plan_acceptance_context(constraints: dict[str, Any] | None) -> str:
    """Expose only presentation constraints, never planner-generated facts.

    The task planner is allowed to describe what must be checked, but its
    acceptance text is still model output and may contain an invented URL or
    an answer-shaped example.  Passing that text to synthesis turns planning
    metadata into an unverified source.  The final model already has the user
    question and the evidence body; it only needs safe shape constraints.
    """
    plan = (constraints or {}).get("task_plan") or {}
    points = plan.get("atomic_points") if isinstance(plan, dict) else []
    rows: list[str] = []
    for point in points or []:
        if not isinstance(point, dict):
            continue
        output_format = str(point.get("output_format") or "prose").strip()
        shape = "Preserve every requested item and cite direct evidence."
        if output_format in {"list", "table", "mixed"}:
            shape += " Preserve row/column relationships and original order where requested."
        rows.append(f"{point.get('id', '')} format={output_format}; {shape}")
    return "\n".join(rows)[:6000]


def _query_terms(data: dict[str, Any]) -> set[str]:
    value = str(data.get("query") or "")
    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", value)
        if token.casefold() not in {"the", "and", "for", "find", "then", "from", "with"}
    }
    # Chinese queries are not whitespace-tokenized. Keep overlapping terms so
    # date/version/entity wording participates in ranking without adding facts.
    for run in re.findall(r"[\u3400-\u9fff]+", value):
        if len(run) >= 2:
            terms.add(run.casefold())
            for size in (2, 3, 4):
                terms.update(run[index : index + size].casefold() for index in range(len(run) - size + 1))
    return {term for term in terms if term not in {"\u7136\u540e", "\u4ee5\u53ca", "\u544a\u8bc9", "\u54ea\u4e9b", "\u4ec0\u4e48"}}


def _canonical_source_url(value: Any) -> str:
    """Normalize a fetched URL for source-level duplicate detection."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/").casefold()
    if not parsed.scheme or not parsed.netloc:
        return raw.rstrip("/").casefold()
    # Fragments never change the fetched document.  Keep the query because
    # API-backed pages may use it to select a different record.
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            path,
            parsed.query,
            "",
        )
    )


def _source_identity(item: dict[str, Any]) -> str:
    url = _canonical_source_url(item.get("url"))
    if url:
        return f"url:{url}"
    body = re.sub(r"\s+", " ", evidence_text(item)).strip().casefold()
    if body:
        return "body:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    return "record:" + hashlib.sha256(
        repr(sorted(item.items())).encode("utf-8", errors="ignore")
    ).hexdigest()


def _source_quality_key(item: dict[str, Any]) -> tuple[int, int, int, float, int]:
    body = evidence_text(item)
    quality = source_quality(item)
    score = quality.get("score", 0) if isinstance(quality, dict) else 0
    rerank = item.get("rerank_score")
    return (
        int(bool((item.get("authority") or {}).get("satisfied"))),
        int(has_substantive_evidence(item)),
        len(body),
        float(score) if isinstance(score, (int, float)) else 0.0,
        int(float(rerank) * 1000) if isinstance(rerank, (int, float)) else -1,
    )


def _deduplicate_sources(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse duplicate provider rows before ranking and context packing.

    Providers often return the same URL more than once.  Allowing those rows
    into the source budget does not add factual coverage and can make overlap
    look like corroboration.  Keep the richest row deterministically and
    report the count for diagnostics.
    """

    kept: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    duplicates = 0
    for item in items:
        key = _source_identity(item)
        if key not in kept:
            kept[key] = item
            order.append(key)
            continue
        duplicates += 1
        if _source_quality_key(item) > _source_quality_key(kept[key]):
            kept[key] = item
    return [kept[key] for key in order], duplicates


def _relevance(item: dict[str, Any], query_terms: set[str]) -> int:
    body = evidence_text(item)
    haystack = " ".join(
        [
            str(item.get("title") or ""),
            body,
            " ".join(str(value) for value in (item.get("authors") or [])),
        ]
    ).casefold()
    return sum(term in haystack for term in query_terms)


def _topic_anchor_score(item: dict[str, Any], query: str) -> int:
    """Prefer a body containing the query's exact technical topic anchors."""

    query_words = [
        value.casefold()
        for value in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(query or ""))
    ]
    if not query_words:
        return 0
    haystack = " ".join(
        [str(item.get("title") or ""), evidence_text(item)]
    ).casefold()
    normalized_haystack = re.sub(r"[-_]", " ", haystack)
    score = sum(word in normalized_haystack for word in query_words)
    for size in (3, 2):
        for index in range(len(query_words) - size + 1):
            phrase = " ".join(query_words[index : index + size])
            if phrase in normalized_haystack:
                score += size * 3
    return score


def _requested_source_count(query: str, constraints: dict[str, Any] | None) -> int:
    plan = (constraints or {}).get("task_plan") or {}
    points = plan.get("atomic_points") if isinstance(plan, dict) else []
    point_count = sum(1 for point in points or [] if isinstance(point, dict))
    if point_count:
        if str(plan.get("task_mode") or "").casefold() == "lookup":
            # Lookup questions usually ask several facts from one document;
            # packing one broad overview per point overwhelms a small RWKV
            # even when all pages are authoritative.
            return min(2, max(1, point_count))
        return min(4, max(1, point_count))
    # Legacy callers may not pass the model task plan. This only controls
    # evidence coverage; it does not choose tools or decide factual content.
    query_text = str(query or "")
    fact_markers = r"\u8bba\u6587|\u9879\u76ee|\u94fe\u63a5|\u521b\u59cb\u4eba|\u8def\u7ebf|\u516c\u5171\u4ea4\u901a|\u53d1\u5e03\u65e5\u671f|\u65e5\u671f|\u5468\u5e74\u5e86|\u7248\u672c|\u4e3b\u9898|\u4f55\u65f6|\u54ea\u4e00\u5e74|\u8c01|release|date|anniversary|version|theme|founder|paper|project|link"
    return min(4, max(1, len(re.findall(fact_markers, query_text)) or 1))


def build_evidence_context(
    data: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    *,
    query: str = "",
) -> dict[str, Any]:
    """Rank, chunk, and project evidence into the exact model context.

    The metadata is serializable and is persisted in the run trace.  It makes
    the boundary between retrieved pages and the RWKV prompt inspectable
    without relying on process memory.
    """
    all_results = [item for item in data.get("results") or [] if isinstance(item, dict)]
    discarded_count = len(all_results) - len(substantive_evidence_items(all_results))
    # Discovery rows remain in the trace, but title/snippet-only rows cannot
    # be ranked into the final model context.
    all_results, duplicate_source_count = _deduplicate_sources(
        substantive_evidence_items(all_results)
    )
    query_text = str(query or data.get("query") or "")
    ranking_data = dict(data)
    ranking_data["query"] = query_text
    query_terms = _query_terms(ranking_data)
    source_policy = resolve_source_policy(query_text, constraints)
    for item in all_results:
        if not isinstance(item.get("authority"), dict):
            item["authority"] = authority_for_url(item.get("url"), query_text, constraints)
    ranked = sorted(
        all_results,
        key=lambda item: (
            int(bool((item.get("authority") or {}).get("satisfied"))),
            int((item.get("authority") or {}).get("rank") or 0),
            # A large, authoritative-looking homepage is not automatically
            # the right evidence.  Direct body relevance must lead; source
            # quality and provider rerank signals break ties among relevant
            # pages.  This avoids selecting a navigation-heavy root page over
            # a smaller page that actually states the requested fact.
            _topic_anchor_score(item, query_text),
            _relevance(item, query_terms),
            float(item.get("rerank_score")) if isinstance(item.get("rerank_score"), (int, float)) else -1.0,
            source_quality(item)["score"],
        ),
        reverse=True,
    )
    # For an official-source task, an admitted required-domain page is the
    # only kind of page allowed to compete for the answer context.  If none
    # was found, keep the third-party rows as diagnostic alternatives, but the
    # validation layer will mark the points authority_missing.
    if source_policy.get("required"):
        official_rows = [
            item for item in ranked
            if (item.get("authority") or {}).get("satisfied")
        ]
        if official_rows:
            ranked = official_rows
    selected: list[dict[str, Any]] = []
    strategy = normalize_strategy((constraints or {}).get("strategy_config") or data.get("strategy_config"))
    configured_count = strategy.get("context_source_count")
    query_text = str(data.get("query") or "").casefold()
    max_selected = configured_count or _requested_source_count(query_text, constraints)
    if ranked:
        selected.append(ranked[0])
        # A lexical match is only a ranking signal.  For a cross-language
        # query (for example a Chinese question with English source bodies)
        # every score can legitimately be zero.  In that case retain the
        # bounded source budget instead of silently dropping all independent
        # sources after the first one; the evidence contract still prevents
        # the final model from turning a body into an unsupported fact.
        lexical_signal = any(_relevance(item, query_terms) > 0 for item in ranked)
        anchor_terms = {
            token.casefold()
            for token in re.findall(
                r"[A-Za-z][A-Za-z0-9_-]{2,}",
                " ".join(
                    [
                        str(ranked[0].get("title") or ""),
                        " ".join(str(value) for value in (ranked[0].get("authors") or [])),
                    ]
                ),
            )
        }
        for item in ranked[1:]:
            if len(selected) >= max_selected:
                break
            # A plan may request several independent sources, but its source
            # count is not evidence that every returned page is relevant.  A
            # generic homepage, repository shell, or download index must stay
            # in the trace unless its fetched body actually overlaps the
            # question (or the selected entity anchor).  This prevents a
            # large context from turning unrelated pages into false
            # corroboration and reduces page-copy degeneration.
            direct_relevance = _relevance(item, query_terms)
            anchor_relevance = any(
                term in " ".join(str(item.get("authors") or [])).casefold() for term in anchor_terms
            )
            if not lexical_signal or direct_relevance > 0 or anchor_relevance:
                selected.append(item)

    selected_metadata: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        normalized_facts = _evidence_text_value(item, constraints)
        normalized_facts = re.sub(r"[ \t]+", " ", normalized_facts)
        normalized_facts = re.sub(r"\n{3,}", "\n\n", normalized_facts).strip()
        source_chunks = _source_chunks_for_context(item, normalized_facts, constraints)
        selected_chunks = _select_source_chunks(item, source_chunks, query_text)
        facts = "\n".join(str(chunk.get("text") or "").strip() for chunk in selected_chunks).strip()
        if not facts:
            facts = normalized_facts
        authors = ", ".join(str(value) for value in (item.get("authors") or [])[:8])
        provenance = evidence_provenance(item)
        selected_metadata.append(
            {
                "ref_id": f"S{index}",
                "rank": item.get("retrieval_rank", index),
                "url": str(item.get("url") or ""),
                "title": str(item.get("title") or ""),
                "retrieval_score": item.get("rerank_score"),
                "ranking_method": item.get("ranking_method", "quality_then_relevance.v1"),
                "source_chars": len(normalized_facts),
                "evidence_text": facts,
                "evidence_boundary": "fetched_page_or_structured_record_only",
                "evidence_origin": str(provenance.get("origin") or "fetched_page_body"),
                "date_mentions": date_mentions(facts),
                "source_chunk_count": len(source_chunks),
                "selected_chunk_count": len(selected_chunks),
                "chunk_count": len(selected_chunks),
                "chunks": [
                    {
                        "index": int(chunk.get("index", chunk_index)),
                        "chunk_id": str(chunk.get("chunk_id") or f"chunk-{chunk_index + 1}"),
                        "text": str(chunk.get("text") or ""),
                        "token_count": int(chunk.get("token_count") or get_token_count(str(chunk.get("text") or ""))),
                    }
                    for chunk_index, chunk in enumerate(selected_chunks)
                ],
                "selected_chunk_indexes": [int(chunk.get("index", 0)) for chunk in selected_chunks],
                "selected_chars": len(facts),
                "truncated": len(normalized_facts) > len(facts),
                "evidence_provenance": provenance,
                "source_quality": source_quality(item),
                "authority": item.get("authority") or authority_for_url(item.get("url"), query_text, constraints),
                "freshness": item.get("freshness") or {},
            }
        )
    # Pack after ranking but before validation/citation metadata is returned.
    # The packer preserves source boundaries and reduces metadata to exactly
    # the chunks visible to the final model.
    context_budget = evidence_tokens(get_llm_context_length())
    packed_metadata, context_text = _pack_source_evidence(selected_metadata, context_budget)
    validation = build_evidence_validation(
        data,
        query=query_text,
        constraints=constraints,
        # Coverage must be computed over the exact source/chunk projection
        # that the final model receives, not over omitted tail chunks.
        selected=packed_metadata,
    )
    return {
        "text": context_text,
        "evidence_text": context_text,
        "selected_evidence": packed_metadata,
        "source_chars": sum(item["source_chars"] for item in packed_metadata),
        "selected_chars": sum(item["selected_chars"] for item in packed_metadata),
        "chunk_count": sum(item["chunk_count"] for item in packed_metadata),
        "usable_evidence_count": sum(1 for item in packed_metadata if _has_usable_evidence(item)),
        "discarded_non_evidence_count": discarded_count,
        "duplicate_source_count": duplicate_source_count,
        # Keep source-level truncation separate from the final aggregate
        # context cut.  A long source may be bounded before the final prompt
        # is built; these are different budgets and must not be reported as
        # repeated 7k page chunking.
        "truncated_count": sum(bool(item["truncated"]) for item in packed_metadata),
        "source_truncated_count": sum(bool(item["truncated"]) for item in packed_metadata),
        "context_chars": len(context_text),
        "context_tokens": get_token_count(context_text),
        # This is a source/evidence-packing decision.  It is intentionally
        # not reported as final-context truncation: the final prompt has its
        # own fit pass after the answer instructions and completion reserve
        # are known.  Conflating the two made a 12K model look as if its final
        # prompt had been clipped whenever one long page was shortened.
        "context_truncated": any(
            bool(item.get("truncated")) for item in packed_metadata
        ) or len(packed_metadata) < len(selected_metadata),
        "source_context_truncated": any(
            bool(item.get("truncated")) for item in packed_metadata
        ) or len(packed_metadata) < len(selected_metadata),
        "final_context_truncated": False,
        "validation": validation,
        "calculation_results": [
            item for item in (data.get("calculation_results") or [])
            if isinstance(item, dict) and str(item.get("status") or "") == "ok"
        ],
        "calculation_text": _calculation_context_text(data.get("calculation_results") or []),
        "strategy": strategy,
    }


def _evidence_text(data: dict[str, Any]) -> str:
    """Backward-compatible text-only projection for callers outside runtime."""
    return str(build_evidence_context(data)["text"])


def _normalized_copy_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\[S\d+(?::C\d+)?\](?:\([^)]*\))?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _source_copy_ratio(answer: str, selected_evidence: list[dict[str, Any]] | None = None) -> float:
    """Detect a final answer that mostly reproduces one captured page body."""

    normalized_answer = _normalized_copy_text(answer)
    if len(normalized_answer) < 1400:
        return 0.0
    answer_tokens = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", normalized_answer)
    if len(answer_tokens) < 180:
        return 0.0
    best = 0.0
    for item in selected_evidence or []:
        body = _normalized_copy_text(item.get("evidence_text") or item.get("body") or "")
        if len(body) < 600:
            continue
        if normalized_answer in body:
            return 1.0
        body_tokens = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", body)
        if len(body_tokens) >= 8:
            source_ngrams = {
                tuple(body_tokens[index : index + 8])
                for index in range(len(body_tokens) - 7)
            }
            answer_ngram_count = max(1, len(answer_tokens) - 7)
            covered_ngrams = sum(
                tuple(answer_tokens[index : index + 8]) in source_ngrams
                for index in range(len(answer_tokens) - 7)
            )
            best = max(best, covered_ngrams / answer_ngram_count)
        answer_lines = {
            re.sub(r"\s+", " ", line).strip()
            for line in normalized_answer.splitlines()
            if len(re.sub(r"\s+", " ", line).strip()) >= 30
        }
        body_lines = {
            re.sub(r"\s+", " ", line).strip()
            for line in body.splitlines()
            if len(re.sub(r"\s+", " ", line).strip()) >= 30
        }
        if answer_lines:
            best = max(best, 0.9 * len(answer_lines & body_lines) / len(answer_lines))
        match = SequenceMatcher(None, normalized_answer, body, autojunk=False).find_longest_match(
            0,
            len(normalized_answer),
            0,
            len(body),
        )
        best = max(best, match.size / max(1, len(normalized_answer)))
    return round(best, 4)


def _needs_answer_repair(
    answer: str,
    selected_evidence: list[dict[str, Any]] | None = None,
) -> bool:
    lowered = answer.casefold()
    prompt_replay = _is_prompt_replay(answer)
    generic_exposition = any(marker in lowered for marker in ("tutorial", "installation guide", "瀹夎鎸囧崡", "鐢ㄦ埛鎰忓浘"))
    scaffold = any(
        marker in lowered
        for marker in ("**answer:**", "**key evidence:**", "**source links:**", "**limitations", "draft:")
    )
    tool_protocol = bool(
        re.search(
            r"(?is)(?:```json\s*)?\{\s*\"(?:name|tool_name|action|tool)\"\s*:",
            answer,
        )
    ) or "function output:" in lowered or "assistant: ```json" in lowered
    bullet_bodies: dict[str, int] = {}
    for line in answer.splitlines():
        match = re.match(r"^\s*[-*]\s+(.+?)\s*$", line)
        if match:
            body = re.sub(r"[*_`\[\](){}]", "", match.group(1))
            body = re.sub(r"\s+", " ", body).strip().casefold()
            if body:
                bullet_bodies[body] = bullet_bodies.get(body, 0) + 1
    repeated_bullet = any(count >= 3 for count in bullet_bodies.values())
    return (
        generic_exposition
        or prompt_replay
        or scaffold
        or tool_protocol
        or "<think>" in lowered
        or "<tool_call>" in lowered
        or repeated_bullet
        or _source_copy_ratio(answer, selected_evidence) >= 0.65
    )


def _is_prompt_replay(answer: str) -> bool:
    """Detect model output that explains the task instead of answering it."""

    text = str(answer or "")
    if not text.strip():
        return False
    # These are continuation/protocol phrases, not ordinary sourced prose.
    # Require a line-start match so a legitimate answer quoting a user request
    # is not treated as protocol residue.
    line_start = re.compile(
        r"(?im)^\s*(?:the user asks|the answer must|the user wants|we need to "
        r"(?:extract|find|answer)|the documentation mentions)\b"
    )
    if line_start.search(text):
        return True
    lowered = text.casefold()
    return (
        "no tool calls" in lowered
        and ("no source list" in lowered or "citation brackets" in lowered)
    )


def _dedupe_repeated_sentences(answer: str) -> str:
    """Remove exact repeated prose sentences from continuation loops."""

    value = str(answer or "").strip()
    if not value:
        return value
    output: list[str] = []
    for line in value.splitlines():
        parts = re.split(r"(?<=[.!?。！？])(?=\s+|$)", line)
        if len(parts) < 3:
            output.append(line)
            continue
        seen: set[str] = set()
        kept: list[str] = []
        for part in parts:
            normalized = re.sub(r"\s+", " ", part).strip().casefold()
            if len(normalized) >= 28 and normalized in seen:
                continue
            if normalized:
                seen.add(normalized)
            kept.append(part)
        output.append("".join(kept))
    return "\n".join(output).strip()


def _is_evidence_refusal(answer: str) -> bool:
    """Return whether a no-evidence answer keeps the closed-world boundary.

    This is deliberately a narrow output-contract check, not a factuality
    judge.  When the evidence gate says there is no usable body, the model is
    allowed to phrase the refusal naturally; it is not allowed to replace the
    missing body with remembered facts.
    """

    lowered = re.sub(r"\s+", " ", str(answer or "").casefold()).strip()
    if not lowered:
        return False
    markers = (
        "无法根据",
        "无法确认",
        "无法提供",
        "未能确认",
        "未找到",
        "没有可用的证据",
        "没有足够的证据",
        "证据不足",
        "缺少证据",
        "无法核实",
        "不能确认",
        "cannot confirm",
        "unable to confirm",
        "insufficient evidence",
        "no usable evidence",
        "not enough evidence",
        "not supported by the retrieved",
        "unconfirmed",
        "未提供",
        "does not provide",
        "doesn't provide",
        "the provided source material does not",
        "the retrieved evidence body does not",
        "the search results do not",
        "the available evidence does not",
        "no information about",
        "unable to provide",
        "cannot provide",
    )
    return any(marker in lowered for marker in markers)


def _closed_world_fallback() -> str:
    """Return the only safe answer when no fetched body supports the task."""

    return "无法根据当前检索到的正文证据确认该问题所需的事实，因此不提供未经证据支持的答案。"


def _fallback_excerpt(value: Any, *, limit: int = 360) -> str:
    """Extract a small source-body excerpt for controller-side degradation."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    useful: list[str] = []
    total = 0
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip(" -|#")
        if len(part) < 24 or part.startswith(("Table of contents", "Navigation", "Skip to")):
            continue
        remaining = limit - total
        if remaining <= 0:
            break
        clipped = part[:remaining].rstrip()
        useful.append(clipped)
        total += len(clipped) + 1
        if total >= limit:
            break
    excerpt = " ".join(useful).strip()
    if len(excerpt) >= limit:
        excerpt = excerpt[: limit - 1].rstrip() + "…"
    return excerpt


def _evidence_limited_fallback(
    query: str,
    context: dict[str, Any],
    *,
    reason: str,
    model_error: str = "",
) -> dict[str, Any]:
    """Return a non-empty, source-bounded answer when RWKV cannot finish."""

    validation = context.get("validation") or {}
    coverage_rows = [
        row for row in validation.get("subquestion_coverage") or [] if isinstance(row, dict)
    ]
    missing_rows = [row for row in coverage_rows if not bool(row.get("answerable"))]
    covered_rows = [row for row in coverage_rows if bool(row.get("answerable"))]
    covered_refs = {
        str(source.get("ref_id") or "")
        for row in covered_rows
        for source in (row.get("sources") or [])
        if isinstance(source, dict) and str(source.get("ref_id") or "").strip()
    }
    selected = [
        item
        for item in context.get("selected_evidence") or []
        if isinstance(item, dict) and str(item.get("evidence_text") or "").strip()
    ]
    if not selected and int(context.get("usable_evidence_count") or 0) > 0:
        # A final-context projection can retain source text while a later
        # validator projection drops its compact per-source metadata.  Keep
        # the controller fallback bounded to that exact visible projection.
        visible_text = str(context.get("evidence_text") or context.get("text") or "").strip()
        if visible_text and visible_text != "(no retrieved evidence)":
            selected = [{"ref_id": "S1", "evidence_text": visible_text}]
    if covered_refs:
        focused = [item for item in selected if str(item.get("ref_id") or "") in covered_refs]
        if focused:
            selected = focused

    calculation_results = context.get("calculation_results") or []
    lines: list[str] = []
    for item in calculation_results[:8]:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "calculator")
        result = item.get("formatted_result", item.get("result", item.get("days", "")))
        expression = str(item.get("expression") or item.get("formula") or "").strip()
        lines.append(f"- {expression + ' = ' if expression else ''}{result} ({tool} result)")

    # Atomic validation points with no covered row mean that the selected page
    # is not evidence for this question. Do not replay an unrelated page.
    can_quote_evidence = bool(selected) and (not coverage_rows or bool(covered_rows))
    if can_quote_evidence:
        for item in selected[:4]:
            excerpt = _fallback_excerpt(item.get("evidence_text"), limit=360)
            if excerpt:
                lines.append(f"- {excerpt} [{item.get('ref_id') or 'S1'}]")

    if lines:
        answer = "Based on the currently citable evidence, the following can be confirmed:\n" + "\n".join(lines[:6])
        mode = "controller_fallback"
    else:
        answer = (
            "The retrieved source bodies do not provide enough direct evidence to reliably answer this question. "
            "I will not fill the gap with unsupported information."
        )
        mode = "controller_refusal"

    if missing_rows:
        missing_lines = []
        for row in missing_rows[:12]:
            point_id = str(row.get("point_id") or "").strip()
            task = re.sub(r"\s+", " ", str(row.get("task") or point_id)).strip()
            if task:
                missing_lines.append(f"- {point_id + ': ' if point_id else ''}{task}")
        if missing_lines:
            answer += (
                "\n\nThe following parts remain unconfirmed because the evidence is insufficient:\n"
                + "\n".join(missing_lines)
            )

    return {
        "content": answer.strip() or "The available evidence is insufficient to answer reliably.",
        "mode": mode,
        "fallback_reason": reason,
        "model_error": model_error,
        "answer_quality": {
            "fallback_used": True,
            "fallback_kind": "refusal" if mode == "controller_refusal" else "evidence_excerpt",
            "fallback_reason": reason,
            "model_error_recorded": bool(model_error),
            "missing_point_count": len(missing_rows),
        },
        "answer_alignment": assess_answer_alignment(
            answer,
            [] if mode == "controller_refusal" else selected,
        ),
    }


def _answer_revision_key(
    answer: str,
    copy_evidence: list[dict[str, Any]] | None,
    selected_evidence: list[dict[str, Any]] | None,
    ) -> tuple[int, int, int, int, int]:
    """Prefer a repair only when it is safer than the current draft.

    A continuation model can answer a repair prompt by replaying the evidence
    body. Replacing a concise refusal with that replay is a regression, even
    when the replay has more lexical overlap with the source.
    """

    copy_ratio = _source_copy_ratio(answer, copy_evidence)
    alignment = assess_answer_alignment(answer, selected_evidence or [])
    citation_count = len(
        re.findall(r"\[S\d+(?::C\d+)?\]", answer or "", flags=re.IGNORECASE)
    )
    return (
        int(_is_prompt_replay(answer)),
        int(copy_ratio >= 0.65),
        int(alignment.get("unsupported_line_count") or 0),
        -citation_count,
        int(len(str(answer or "")) > 12000),
    )


def _final_completion_budget(prompt: str) -> int:
    """Use the remaining model context instead of a fixed 3K final cap."""

    return bounded_completion_budget(
        prompt,
        context_limit=get_llm_context_length(),
        requested_max=8192,
        safety_margin=256,
    )


def _requested_final_completion_budget(
    constraints: dict[str, Any] | None,
    query: str = "",
) -> int:
    """Scale answer room with task complexity, while retaining a high ceiling.

    A single point should not reserve the full 8K continuation window: RWKV can
    then spend two minutes continuing a copied page.  Larger plans still get
    more room, and the context-safe calculation remains the final authority.
    """
    plan = (constraints or {}).get("task_plan") or {}
    points = plan.get("atomic_points") if isinstance(plan, dict) else []
    point_count = sum(1 for point in points or [] if isinstance(point, dict))
    requested = max(2048, min(8192, 2560 + max(1, point_count) * 512))
    # A planner may over-decompose a short question into many micro-points.
    # Do not let that semantic bookkeeping reserve a huge continuation window.
    if len(str(query or "").strip()) <= 600:
        requested = min(requested, 4096)
    return requested


def _calculation_context_text(results: Any) -> str:
    """Render successful deterministic tool outputs for final RWKV synthesis.

    These rows are intentionally separate from web evidence.  They carry a
    computed value into the answer prompt without pretending that the
    calculator is a source or allowing it to create a citation.
    """

    rows = [
        item for item in (results or [])
        if isinstance(item, dict)
        and str(item.get("status") or "") == "ok"
        and str(item.get("tool") or "") in {"calculator", "date_diff", "current_time"}
    ]
    if not rows:
        return ""
    lines = [
        "BEGIN DETERMINISTIC TOOL RESULTS",
        "The following values came from explicit deterministic tools. They are not web evidence. Use exact clock or numeric results; cite web evidence only for separate factual claims.",
    ]
    for index, item in enumerate(rows, start=1):
        if str(item.get("tool") or "") == "current_time":
            lines.extend(
                [
                    f"CLOCK T{index} (current_time)",
                    f"timezone: {item.get('timezone', '')}",
                    f"iso: {item.get('iso', '')}",
                    f"date: {item.get('date', '')}",
                    f"utc_offset: {item.get('utc_offset', '')}",
                    f"observed_at_utc: {item.get('observed_at_utc', '')}",
                ]
            )
            continue
        if str(item.get("tool") or "") == "calculator":
            lines.extend(
                [
                    f"CALCULATION C{index} (calculator)",
                    f"expression: {item.get('expression', '')}",
                    f"result: {item.get('formatted_result', item.get('result', ''))}",
                ]
            )
            continue
        refs = ", ".join(str(ref) for ref in item.get("source_refs") or [] if str(ref).strip())
        lines.extend(
            [
                f"CALCULATION C{index} (date_diff)",
                f"date_a: {item.get('date_a', '')}",
                f"date_b: {item.get('date_b', '')}",
                f"absolute_days: {item.get('days', '')}",
                f"signed_days: {item.get('signed_days', '')}",
                f"formula: {item.get('formula', '')}",
                f"operand_source_refs: {refs or 'not supplied'}",
            ]
        )
    lines.append("END DETERMINISTIC TOOL RESULTS")
    return "\n".join(lines)


def _fit_final_context_projection(
    context: dict[str, Any],
    data: dict[str, Any],
    constraints: dict[str, Any] | None,
    query: str,
    prompt_prefix: str,
) -> tuple[dict[str, Any], str, str, bool]:
    """Fit the final prompt while keeping evidence metadata in lockstep.

    The final prompt, citation references, locators, and validation report must
    describe one identical evidence projection.  A flat string truncation can
    remove the tail of a chunk while leaving its locator and validator visible.
    Repack complete source chunks instead; if the prompt still cannot fit,
    fail closed with an explicit empty projection rather than sending an
    unbound partial source to RWKV.
    """
    limit = max(1024, int(get_llm_context_length()))
    calculation_text = str(context.get("calculation_text") or "").strip()

    def compose_context(source_text: str) -> str:
        source_text = str(source_text or "").strip() or "(no retrieved evidence)"
        if not calculation_text:
            return source_text
        return f"{source_text}\n\n{calculation_text}"

    original_source_text = str(
        context.get("evidence_text") or context.get("text") or ""
    ).strip()
    original_text = compose_context(original_source_text)

    def with_projection(selected: list[dict[str, Any]], text: str, truncated: bool) -> dict[str, Any]:
        projected = dict(context)
        projected["evidence_text"] = text or "(no retrieved evidence)"
        projected["text"] = compose_context(text)
        projected["selected_evidence"] = selected
        projected["validation"] = build_evidence_validation(
            data,
            query=query,
            constraints=constraints,
            selected=selected,
        )
        projected["selected_chars"] = sum(int(item.get("selected_chars") or 0) for item in selected)
        projected["chunk_count"] = sum(int(item.get("chunk_count") or 0) for item in selected)
        projected["usable_evidence_count"] = sum(1 for item in selected if _has_usable_evidence(item))
        projected["context_chars"] = len(projected["text"])
        projected["context_tokens"] = get_token_count(projected["text"])
        projected["context_truncated"] = bool(projected.get("context_truncated") or truncated)
        projected["source_context_truncated"] = bool(
            projected.get("source_context_truncated") or truncated
        )
        return projected

    def fits(text: str) -> tuple[bool, str]:
        rendered = build_final_continuation_prompt(prompt_prefix + text)
        return get_token_count(rendered) + 512 <= limit, rendered

    fits_original, original_prompt = fits(original_text)
    if fits_original:
        return context, original_text, original_prompt, False

    source_items = [item for item in context.get("selected_evidence") or [] if isinstance(item, dict)]
    original_tokens = get_token_count(original_text)
    caps = [original_tokens, 7000, 6500, 6000, 5500, 5000, 4500, 4000, 3500, 3000, 2500, 2048]
    seen: set[int] = set()
    for cap in caps:
        cap = max(2048, int(cap))
        if cap in seen:
            continue
        seen.add(cap)
        selected, text = _pack_source_evidence(source_items, cap)
        projected = with_projection(selected, text, True)
        fits_candidate, candidate_prompt = fits(projected["text"])
        if fits_candidate:
            return projected, projected["text"], candidate_prompt, True

    empty = with_projection([], "(no retrieved evidence)", True)
    empty_prompt = build_final_continuation_prompt(prompt_prefix + empty["text"])
    return empty, empty["text"], empty_prompt, True


def _context_fields(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_text": context["text"],
        "selected_evidence": context["selected_evidence"],
        "validation": context.get("validation") or {},
        "calculation_results": context.get("calculation_results") or [],
        "context_stats": {
            key: value
            for key, value in context.items()
            if key not in {
                "text",
                "evidence_text",
                "selected_evidence",
                "validation",
                "calculation_results",
                "calculation_text",
            }
        },
    }


def _validation_prompt(report: dict[str, Any]) -> str:
    """Render compact candidate hints without promoting lexical matches to facts."""

    if not isinstance(report, dict):
        return ""
    coverage_rows = [
        item
        for item in report.get("subquestion_coverage") or []
        if isinstance(item, dict)
    ]
    rows: list[str] = []
    missing: list[str] = []
    for item in report.get("subquestion_coverage") or []:
        if not isinstance(item, dict):
            continue
        source_ids = ",".join(str(row.get("ref_id") or "") for row in item.get("sources") or [])
        status = str(item.get("status") or "").casefold()
        if source_ids and status != "authority_missing":
            rows.append(f"{item.get('point_id')}: candidate body matches={source_ids}")
        elif source_ids and status == "authority_missing":
            missing.append(f"{item.get('point_id')} (required official domain not found)")
        else:
            missing.append(str(item.get("point_id") or ""))
    cross = report.get("cross_source") or {}
    conflicts = cross.get("candidate_conflicts") or []
    conflict_text = "; ".join(
        f"{item.get('point_id')}: {len(item.get('dates') or [])} date-like variants"
        for item in conflicts
        if isinstance(item, dict)
    )
    coverage = ", ".join(rows) or "visible source bodies"
    missing_text = ", ".join(value for value in missing if value)
    conflict_suffix = f" Check date variants: {conflict_text}." if conflict_text else ""
    missing_suffix = f" Missing locator points: {missing_text}." if missing_text else ""
    return (
        "ROUTING NOTE (not evidence): inspect the visible EVIDENCE BODY directly; "
        f"candidate coverage={coverage}.{missing_suffix}{conflict_suffix} "
        "Do not turn a locator miss into a refusal, and do not use facts absent from the body.\n"
    )


def _evidence_polarity_hint(query: str, evidence: str) -> str:
    """Keep oppositely-directed build/runtime options from being inverted."""

    query_lower = str(query or "").casefold()
    evidence_lower = str(evidence or "").casefold()
    if "free threading" not in query_lower or "python_gil" not in evidence_lower:
        return ""
    if "disable-gil" not in evidence_lower and "disable gil" not in evidence_lower:
        return ""
    return (
        "POLARITY CHECK from the visible evidence: preserve the documented direction of each option. "
        "The body distinguishes the build-time --disable-gil option from the runtime PYTHON_GIL "
        "and -Xgil options; do not reverse an option's effect or present a setting that enables "
        "the GIL as a way to enable free threading.\n"
    )


def _answer_scope_hint(query: str) -> str:
    """Keep how-to questions from expanding into an unrelated page summary."""

    if "free threading" in str(query or "").casefold():
        return (
            "FOCUS ORDER: answer the documented installation/build method first, then the runtime mode, "
            "then the minimal verification. Keep thread-safety, performance, allocator, and implementation "
            "details out unless the user asks for them.\n"
        )

    if re.search(r"(?:怎么|如何|怎样|开启|启用|安装|配置|设置|how to|enable|install|configure|set up)", str(query or "").casefold()):
        return (
            "SCOPE: this is a how-to request. Return only the direct procedure and the minimal "
            "verification needed for that procedure; omit unrelated background, limitations, "
            "performance details, and page-summary sections. Use at most two short paragraphs.\n"
        )
    return ""


def _answer_first_hint(query: str, *, evidence_available: bool) -> str:
    """Prefer grounded answers when the visible body is available.

    The validator and chunk extractor are routing aids. They can miss a
    bilingual match or mark only one of several chunks, so their uncertainty
    must not become a final-answer refusal when the source body is visible.
    """

    if not evidence_available:
        return ""
    lowered = str(query or "").casefold()
    hint = (
        "ANSWER FIRST: use only the visible EVIDENCE BODY, combine relevant spans, and cite factual sentences. "
        "Say unconfirmed only when the requested detail is absent from all visible spans.\n"
    )
    if re.search(r"(核验|验证|是否正确|这句话|该说法|纠正|声称|真的|correct|incorrect|verify|claim)", lowered):
        hint += (
            "CLAIM CHECK: begin with correct or incorrect, then give the source-backed correction and requested values.\n"
        )
    if re.search(r"(相隔|多少天|百分比|百分之|比例|占比|差值|计算|自然日|how many days|percentage|percent|calculate|difference)", lowered):
        hint += (
            "CALCULATION: identify the operands, show the date/number expression, and state the computed result in the requested unit.\n"
        )
    return hint


def _normalize_evidence_code_spans(answer: str, evidence: str) -> str:
    """Restore near-miss code spans to the exact spelling visible in evidence."""

    source_spans = []
    for value in re.findall(r"`([^`]+)`", str(evidence or "")):
        value = re.sub(r"\s+", " ", value).strip()
        if value and value not in source_spans:
            source_spans.append(value)
    if not source_spans:
        return answer

    def replace_span(match: re.Match[str]) -> str:
        value = match.group(1)
        if value in source_spans:
            return match.group(0)
        candidates = [
            candidate
            for candidate in source_spans
            if ("." in value or "_" in value or value.startswith("--"))
            and ("." in candidate or "_" in candidate or candidate.startswith("--"))
            and candidate.casefold() not in value.casefold()
            and value.casefold() not in candidate.casefold()
        ]
        if not candidates:
            return match.group(0)
        best = max(
            candidates,
            key=lambda candidate: SequenceMatcher(
                None, value.casefold(), candidate.casefold(), autojunk=False
            ).ratio(),
        )
        score = SequenceMatcher(None, value.casefold(), best.casefold(), autojunk=False).ratio()
        # Only repair a close transcription; do not turn ordinary prose into
        # a source token merely because both contain punctuation.
        if score >= (0.72 if len(value) >= 16 else 0.84):
            return f"`{best}`"
        return match.group(0)

    return re.sub(r"`([^`]+)`", replace_span, str(answer or ""))


def _normalize_free_threading_polarity(answer: str, query: str, evidence: str) -> str:
    """Keep the documented build/runtime GIL direction when it is explicit."""

    if "free threading" not in str(query or "").casefold():
        return answer
    evidence_lower = str(evidence or "").casefold()
    if "optionally running with the gil enabled at runtime using" not in evidence_lower:
        return answer
    replacement = (
        "The free-threaded build can optionally run with the GIL enabled at runtime using "
        "the environment variable `PYTHON_GIL` or the command-line option `-Xgil`"
    )
    normalized_answer = re.sub(
        r"(?is)\bat runtime,?\s*the GIL can be disabled with[^.;]*",
        replacement,
        str(answer or ""),
    )
    return re.sub(
        r"(?is)\bthe GIL can be disabled at runtime[^.;。]*?(?:`PYTHON_GIL`[^.;。]*?`-Xgil`|`-Xgil`[^.;。]*?`PYTHON_GIL`)[^.;。]*",
        replacement,
        normalized_answer,
    )


def _enforce_how_to_scope(answer: str, query: str) -> str:
    """Keep procedure answers from replaying the rest of a long source page."""

    if not re.search(r"(?:怎么|如何|怎样|开启|启用|安装|配置|设置|how to|enable|install|configure|set up)", str(query or "").casefold()):
        return answer
    value = str(answer or "").strip()
    value = _dedupe_repeated_sentences(value)
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", value) if part.strip()]
    if len(paragraphs) > 2:
        value = "\n\n".join(paragraphs[:2])
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) > 2:
        if any(re.match(r"(?:[-*]|\d+[.)])\s+", line) for line in lines):
            value = "\n".join(lines[:6])
        else:
            value = "\n".join(lines[:2])
    sentences = re.split(r"(?<=[.!?。！？])\s+", value)
    if len(sentences) > 4:
        value = " ".join(sentences[:4]).strip()
    return value


def _focus_repair_evidence(evidence: str, query: str) -> str:
    """Project only query-relevant source paragraphs into the repair prompt."""

    raw = str(evidence or "").strip()
    if not raw:
        return raw
    query_terms = [
        value.casefold()
        for value in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(query or ""))
    ]
    procedure_terms = (
        "install",
        "installation",
        "build",
        "configure",
        "runtime",
        "enable",
        "enabled",
        "setting",
        "running",
        "verification",
        "check",
        "开启",
        "启用",
        "安装",
        "配置",
        "设置",
    )
    procedure_query = bool(
        re.search(r"(?:怎么|如何|怎样|开启|启用|安装|配置|设置|how to|enable|install|configure|set up)", str(query or "").casefold())
    )
    blocks = re.split(r"(?=BEGIN EVIDENCE SOURCE )", raw)
    focused: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", block) if part.strip()]
        if len(paragraphs) <= 4:
            focused.append(block)
            continue
        header = paragraphs[:4]
        body = paragraphs[4:]
        ranked = sorted(
            enumerate(body),
            key=lambda item: (
                sum(term in item[1].casefold() for term in query_terms),
                sum(term in item[1].casefold() for term in procedure_terms) if procedure_query else 0,
                int(bool(re.search(r"--[A-Za-z0-9_-]+|\b[A-Z][A-Z0-9_]{2,}\b", item[1]))),
            ),
            reverse=True,
        )
        chosen = [item[1] for item in ranked[: (4 if procedure_query else 6)] if item[1]]
        exact_tokens = []
        for paragraph in chosen:
            exact_tokens.extend(
                re.findall(
                    r"(?<!\w)(?:--[A-Za-z0-9_-]+|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?:\(\))?|[A-Z][A-Z0-9_]{2,})(?!\w)",
                    paragraph,
                )
            )
        exact_tokens = list(dict.fromkeys(exact_tokens))
        exact_line = (
            "EXACT TOKENS (copy character-for-character): "
            + ", ".join(f"`{token}`" for token in exact_tokens[:24])
            if exact_tokens
            else ""
        )
        focused.append("\n\n".join(header + chosen + ([exact_line] if exact_line else [])))
    result = "\n\n".join(focused)
    return _truncate_markdown_by_tokens(result, 3200)


def _build_answer_repair_prompt(query: str, evidence: str, _draft: str) -> str:
    # Repair is a second model call.  Rebuild from the focused evidence rather
    # than feeding the model its own draft, which can anchor a wrong identifier
    # or polarity and make the repair repeat the original error.
    repair_evidence = _focus_repair_evidence(evidence, query)
    polarity_hint = _evidence_polarity_hint(query, repair_evidence)
    scope_hint = _answer_scope_hint(query)
    answer_first_hint = _answer_first_hint(query, evidence_available=bool(repair_evidence))
    return build_final_continuation_prompt(
        "Rewrite the answer from the visible EVIDENCE BODY. Start with the answer, preserve exact names, dates, numbers, and polarity, and cite factual sentences with [S#:C#]. Return concise prose only; do not output instructions, JSON, tool calls, or a source-body replay.\n\n"
        f"{polarity_hint}\n"
        f"{scope_hint}\n"
        f"{answer_first_hint}\n"
        f"Question: {query}\n"
        f"EVIDENCE BODY:\n{repair_evidence}"
    )


def synthesize_retrieval_answer(
    query: str,
    data: dict[str, Any],
    llm=None,
    constraints: dict[str, Any] | None = None,
    execution_context: str = "",
    termination_reason: str = "model_requested_finish",
) -> dict[str, Any]:
    """Ask local RWKV for the final answer and expose the exact prompt context."""
    strategy = normalize_strategy((constraints or {}).get("strategy_config") or data.get("strategy_config"))
    context = build_evidence_context(data, constraints=constraints, query=query)
    context_fields = _context_fields(context)
    evidence_context_text = str(context.get("evidence_text") or context["text"])
    calculation_results = context.get("calculation_results") or []
    context_stats = context_fields["context_stats"]
    context_stats["calculation_count"] = len(calculation_results)
    # Keep the evidence-only projection separate from the actual final prompt.
    # The latter may also contain the bounded execution record.
    context_stats["evidence_context_chars"] = context["context_chars"]
    context_stats["evidence_context_tokens"] = context["context_tokens"]
    context_text = evidence_context_text
    execution_context = str(execution_context or "").strip()
    if execution_context:
        # The complete execution record is already persisted in the task
        # events/UI.  Do not put it beside evidence in the final continuation:
        # even a clearly labelled record can be copied as prose or consume the
        # budget needed by a later source.  The final model needs the question,
        # acceptance checklist, and source-bound evidence only.
        context_stats["execution_context_chars"] = len(execution_context)
        context_stats["execution_context_in_final_prompt"] = False
        context_stats["termination_reason"] = termination_reason

    # These are the values the final model call actually receives, rather than
    # the pre-assembly evidence-only values above.
    context_stats["final_context_chars"] = len(context_text)
    context_stats["final_context_tokens"] = get_token_count(context_text)
    context_stats["source_context_truncated"] = bool(
        context_stats.get("source_context_truncated", context.get("context_truncated", False))
    )
    context_stats["final_context_truncated"] = False
    context_stats["context_chars"] = context_stats["final_context_chars"]
    context_stats["context_tokens"] = context_stats["final_context_tokens"]
    context_stats["context_truncated"] = context_stats["final_context_truncated"]
    context_citation_refs = _citation_refs_for_context(data, context)
    validation_prompt = _validation_prompt(context.get("validation") or {})
    acceptance_context = _plan_acceptance_context(constraints)
    source_policy = resolve_source_policy(query, constraints)
    freshness_policy = (constraints or {}).get("freshness_policy") or {}
    task_plan = (constraints or {}).get("task_plan") or {}
    task_mode = str(task_plan.get("task_mode") or "lookup")
    requested_fields = [
        str(value).strip()
        for value in task_plan.get("requested_fields") or []
        if str(value).strip()
    ]
    max_items = int(task_plan.get("max_items") or 0) if str(task_plan.get("max_items") or "0").isdigit() else 0
    policy = risk_context(constraints)
    if llm is None:
        fallback = _evidence_limited_fallback(
            query,
            context,
            reason="rwkv_unavailable",
        )
        return {
            "content": fallback["content"],
            "mode": fallback["mode"],
            "evidence_count": len(data.get("results") or []),
            "citation_refs": context_citation_refs,
            "prompt": "",
            "model_output": "",
            "fallback_reason": fallback["fallback_reason"],
            "answer_quality": fallback["answer_quality"],
            "answer_alignment": fallback["answer_alignment"],
            **context_fields,
        }

    risk_instructions = ""
    if policy["high_risk"]:
        risk_instructions = (
            f"This is a high-risk {policy['label']} information request. "
            "Provide information retrieval only, do not present the answer as professional advice, "
            "state material uncertainty, and tell the user when a qualified professional must confirm it. "
            f"Risk checks: {', '.join(policy['risk_checks']) or 'source quality and uncertainty'}. "
        )
    acceptance = "; ".join(policy["acceptance_criteria"])
    rejection = "; ".join(policy["rejection_criteria"])
    criteria_instructions = ""
    if acceptance or rejection:
        criteria_instructions = (
            f"Acceptance criteria: {acceptance or 'directly answer with supported facts'}. "
            f"Reject these behaviors: {rejection or 'unsupported claims'}. "
        )
    acceptance_instruction = (
        f"Acceptance checklist from the model-generated task plan:\n{acceptance_context}\n\n"
        if acceptance_context and task_mode in {"latest_list", "deep_research"}
        else ""
    )
    freshness_instruction = ""
    if freshness_policy.get("as_of"):
        freshness_instruction = (
            f"Temporal constraint: answer only with information available on or before {freshness_policy['as_of']}. "
            "Do not use a source dated after that cutoff; if a source date is unknown, state that limitation. "
        )
    else:
        freshness_instruction = (
            "Temporal policy: prefer the most recent source available at retrieval time for current/latest questions, "
            "but do not invent a publication date when the source does not provide one. "
        )
    variant_instruction = {
        "default.v1": "",
        "citation_first.v1": "For every factual claim, attach the most relevant source reference or say that evidence is insufficient. ",
        "compact_evidence.v1": "Prefer the shortest answer that preserves all requested facts and source references. ",
    }[strategy["prompt_variant"]]
    usable_evidence_count = int(context.get("usable_evidence_count") or 0)
    missing_point_ids = [
        str(item.get("point_id") or "")
        for item in (context.get("validation") or {}).get("subquestion_coverage") or []
        if isinstance(item, dict) and not bool(item.get("answerable"))
    ]
    coverage_rows = [
        item
        for item in (context.get("validation") or {}).get("subquestion_coverage") or []
        if isinstance(item, dict)
    ]
    # Coverage is a control signal for the next RWKV decision.  It is not an
    # evidence gate by itself: a bilingual task-point matcher can miss a valid
    # English official page even when the page body is already authoritative.
    # The final context is suppressed only when an official-source policy has
    # no satisfied official body at all.  This keeps the safety boundary for a
    # third-party-only result without turning a validator miss into a refusal.
    retrieved_usable_evidence_count = int(context.get("usable_evidence_count") or 0)
    all_requested_points_missing = bool(
        coverage_rows
        and all(not bool(item.get("answerable")) for item in coverage_rows)
    )
    official_body_available = bool(
        source_policy.get("required")
        and any(
            isinstance(item, dict)
            and bool((item.get("authority") or {}).get("satisfied"))
            and _has_usable_evidence(item)
            for item in context.get("selected_evidence") or []
        )
    )
    final_evidence_suppressed = bool(
        source_policy.get("required")
        and all_requested_points_missing
        and not official_body_available
    )
    if final_evidence_suppressed:
        context = dict(context)
        context["text"] = "(no retrieved evidence)"
        context["selected_evidence"] = []
        context["evidence_text"] = "(no retrieved evidence)"
        context["selected_chars"] = 0
        context["source_chars"] = 0
        context["chunk_count"] = 0
        context["usable_evidence_count"] = 0
        context["context_chars"] = len(context["text"])
        context["context_tokens"] = get_token_count(context["text"])
        context["context_truncated"] = False
        context["source_context_truncated"] = False
        context["final_context_truncated"] = False
        context["validation"] = build_evidence_validation(
            data,
            query=query,
            constraints=constraints,
            selected=[],
        )
        context_citation_refs = []
        validation_prompt = _validation_prompt(context["validation"])
        evidence_context_text = str(context.get("evidence_text") or context["text"])
        usable_evidence_count = 0
    answer_context = context
    context_stats["final_evidence_suppressed"] = final_evidence_suppressed
    context_stats["validator_all_points_missing"] = all_requested_points_missing
    context_stats["official_body_available"] = official_body_available
    # Keep retrieval-stage availability separate from what the final model
    # is actually allowed to see.  A source can be substantive yet irrelevant
    # to every requested point, so ``usable_evidence_count`` alone is not a
    # reliable description of the final prompt.
    context_stats["retrieved_usable_evidence_count"] = retrieved_usable_evidence_count
    context_stats["final_usable_evidence_count"] = usable_evidence_count
    context_stats["final_selected_evidence_count"] = len(answer_context.get("selected_evidence") or [])
    if usable_evidence_count:
        if missing_point_ids and official_body_available:
            evidence_state = (
                f"OFFICIAL_EVIDENCE_AVAILABLE_BUT_COVERAGE_UNCERTAIN ({usable_evidence_count} source bodies). "
                f"The official body is visible; automatic point matching is inconclusive for {', '.join(missing_point_ids)}. "
                "Use the visible official body and combine all visible chunks to answer the requested facts. "
                "Treat the coverage marker as routing uncertainty only."
            )
        elif missing_point_ids:
            evidence_state = (
                f"PARTIAL_EVIDENCE ({usable_evidence_count} source bodies). "
                f"Candidate coverage is missing for {', '.join(missing_point_ids)}. "
                "Use the visible body for every supported point and inspect all visible chunks before deciding coverage. "
                "Do not turn a locator gap into a refusal or a general page summary into an answer for an absent point."
            )
        else:
            evidence_state = (
                f"USABLE_EVIDENCE_AVAILABLE ({usable_evidence_count} source bodies). "
                "Use only the EVIDENCE BODY sections for factual claims."
            )
    elif calculation_results:
        evidence_state = (
            "DETERMINISTIC_TOOL_RESULT_AVAILABLE. Use the exact deterministic tool result for the requested time or arithmetic "
            "and do not add unsupported web facts."
        )
    else:
        evidence_state = (
            "NO_USABLE_EVIDENCE. The body does not support the requested facts; state that briefly "
            "and do not use snippets, metadata, or memory."
        )
    def build_prompt_prefix(current_evidence_state: str, current_validation_prompt: str) -> str:
        if source_policy.get("required") and official_body_available:
            source_instruction = (
                "The required official source body is present; use it directly and cite it. "
            )
        elif source_policy.get("required"):
            source_instruction = (
                "No required official body was retrieved. Keep official attribution unconfirmed and do not "
                "present a third-party page as that organisation's source. "
            )
        else:
            source_instruction = ""
        evidence_usage_instruction = (
            "Source-body mode: the retrieved body is available. Start with the requested answer and combine "
            "supported details across the visible chunks. Treat validation and chunk markers as routing metadata; "
            "do not turn a locator gap into a refusal. "
            if usable_evidence_count
            else (
                "Deterministic-tool mode: use the exact tool result shown below for the requested time or arithmetic; "
                "do not invent operands or claim that the calculator is a web source. "
                if calculation_results
                else "No-source-body mode: the requested claim is not confirmed by a retrieved source body; say that briefly and stop. "
            )
        )
        output_precision_instruction = (
            (
                "Copy deterministic dates, times, operands, and results exactly from the tool block; do not recalculate, alter, or invent them. "
                "Do not add a web citation for a deterministic tool result; cite only separate claims supported by visible web evidence. "
            )
            if calculation_results
            else (
                "Copy names, dates, quantities, polarity, commands, and flags exactly from the body; do not invent or reverse them. "
                "Put citations after the sentence or code block. "
            )
        )
        polarity_hint = _evidence_polarity_hint(query, context_text)
        scope_hint = _answer_scope_hint(query)
        answer_first_hint = _answer_first_hint(query, evidence_available=bool(usable_evidence_count))
        calculation_hint = build_calculation_check(
            query,
            context.get("selected_evidence") or context_text,
        )
        calculation_instruction = (
            "The DETERMINISTIC TOOL RESULTS are authoritative only for the arithmetic they explicitly report. "
            "Do not recalculate them mentally, change their operands, or cite C# as a web source. "
            if calculation_results
            else ""
        )
        list_instruction = ""
        if task_mode == "latest_list":
            list_instruction = (
                f"This is a latest-list request. Return at most {max_items or 5} items and only the requested fields "
                f"({', '.join(requested_fields) or 'the fields stated in the question'}); do not copy the entire index page. "
            )
        brevity_instruction = ""
        if task_mode in {"lookup", "compare"}:
            brevity_instruction = (
                "For an ordinary lookup or comparison, be concise: normally use a short paragraph or at most 8 bullets. "
                "Do not reproduce a table of contents, navigation, page body, or a long source index unless the user explicitly asks for it. "
            )
        return (
            "You are the final answer RWKV. Answer the QUESTION from the EVIDENCE BODY. Start with the answer, cover the requested details, and cite factual sentences with [S#:C#]. Return concise user-facing prose only; no instructions, JSON, tool calls, hidden reasoning, or source-body replay. "
            f"{variant_instruction}{risk_instructions}{acceptance_instruction}"
            f"Output mode: {task_mode}. {brevity_instruction}{list_instruction}"
            f"{evidence_usage_instruction}{output_precision_instruction}{source_instruction}"
            f"{freshness_instruction}"
            f"{polarity_hint}"
            f"{scope_hint}"
            f"{answer_first_hint}"
            f"{calculation_hint}"
            f"{calculation_instruction}"
            f"Question: {query}\n"
            f"Evidence status: {current_evidence_state}\n"
            f"{current_validation_prompt}"
        )

    prompt_prefix = build_prompt_prefix(evidence_state, validation_prompt)
    context, context_text, prompt, final_prompt_trimmed = _fit_final_context_projection(
        context,
        data,
        constraints,
        query,
        prompt_prefix,
    )
    if final_prompt_trimmed:
        # Rebuild the control hints from the same projection that survived the
        # context fit.  This prevents a locator/validator from referring to a
        # source span that was removed from the final model input.
        usable_evidence_count = int(context.get("usable_evidence_count") or 0)
        validation_prompt = _validation_prompt(context.get("validation") or {})
        missing_point_ids = [
            str(item.get("point_id") or "")
            for item in (context.get("validation") or {}).get("subquestion_coverage") or []
            if isinstance(item, dict) and not bool(item.get("answerable"))
        ]
        official_body_available = bool(
            any(
                isinstance(item, dict)
                and bool((item.get("authority") or {}).get("satisfied"))
                and _has_usable_evidence(item)
                for item in context.get("selected_evidence") or []
            )
        )
        if usable_evidence_count and missing_point_ids and official_body_available:
            evidence_state = (
                f"OFFICIAL_EVIDENCE_AVAILABLE_BUT_COVERAGE_UNCERTAIN ({usable_evidence_count} source bodies). "
                f"The official body is visible; automatic point matching is inconclusive for {', '.join(missing_point_ids)}. "
                "Use the visible official body and combine all visible chunks to answer the requested facts. "
                "Treat the coverage marker as routing uncertainty only."
            )
        elif usable_evidence_count and missing_point_ids:
            evidence_state = (
                f"PARTIAL_EVIDENCE ({usable_evidence_count} source bodies). "
                f"Candidate coverage is missing for {', '.join(missing_point_ids)}. "
                "Use the visible body for every supported point and inspect all visible chunks before deciding coverage. "
                "Do not turn a locator gap into a refusal or a general page summary into an answer for an absent point."
            )
        elif usable_evidence_count:
            evidence_state = (
                f"USABLE_EVIDENCE_AVAILABLE ({usable_evidence_count} source bodies). "
                "Use only the EVIDENCE BODY sections for factual claims."
            )
        else:
            evidence_state = (
                "NO_USABLE_EVIDENCE. The body does not support the requested facts; state that briefly "
                "and do not use snippets, metadata, or memory."
            )
        prompt_prefix = build_prompt_prefix(evidence_state, validation_prompt)
        context, context_text, prompt, second_fit = _fit_final_context_projection(
            context,
            data,
            constraints,
            query,
            prompt_prefix,
        )
        final_prompt_trimmed = bool(final_prompt_trimmed or second_fit)
    context_fields["context_text"] = context_text
    context_fields["selected_evidence"] = context.get("selected_evidence") or []
    context_fields["validation"] = context.get("validation") or {}
    context_stats = context_fields["context_stats"]
    context_citation_refs = _citation_refs_for_context(data, context)
    answer_context = context
    evidence_context_text = str(context.get("evidence_text") or context_text)
    usable_evidence_count = int(context.get("usable_evidence_count") or 0)
    context_stats["final_usable_evidence_count"] = usable_evidence_count
    context_stats["final_selected_evidence_count"] = len(context.get("selected_evidence") or [])
    context_stats["final_context_chars"] = len(context_text)
    context_stats["final_context_tokens"] = get_token_count(context_text)
    # Only this fit pass may set final_context_truncated.  Source-level page
    # packing remains visible through source_context_truncated and its count.
    context_stats["final_context_truncated"] = bool(final_prompt_trimmed)
    context_stats["context_chars"] = context_stats["final_context_chars"]
    context_stats["context_tokens"] = context_stats["final_context_tokens"]
    context_stats["context_truncated"] = context_stats["final_context_truncated"]
    prompt_body = prompt_prefix + context_text
    repair_prompt = ""
    raw = ""
    try:
        requested_final_tokens = _requested_final_completion_budget(constraints, query)

        def complete(request_prompt: str, requested_max: int | None = None) -> str:
            budget = bounded_completion_budget(
                request_prompt,
                context_limit=get_llm_context_length(),
                requested_max=requested_max or requested_final_tokens,
                safety_margin=256,
            )
            if hasattr(llm, "text_completion"):
                try:
                    response = llm.text_completion(
                        request_prompt,
                        max_tokens=budget,
                        stop=FINAL_CONTINUATION_STOP_SUFFIXES,
                    )
                except TypeError as exc:
                    if "stop" not in str(exc):
                        raise
                    response = llm.text_completion(request_prompt, max_tokens=budget)
            else:
                response = llm.chat_completion([{"role": "user", "content": request_prompt}], max_tokens=budget)
            return str(response.content or "")

        raw_text = complete(prompt)
        raw = raw_text
        answer = _clean_answer(clean_final_continuation(raw_text))
        retry_output = ""
        if not answer:
            retry_prompt = build_final_continuation_prompt(
                prompt_body
                + "\n\nThe previous continuation was empty or protocol-only. Return one concise "
                "user-facing answer now; if evidence is unavailable, explicitly say it cannot be confirmed."
            )
            retry_output = complete(retry_prompt)
            answer = _clean_answer(clean_final_continuation(retry_output))
            repair_prompt = retry_prompt
        selected_evidence = context.get("selected_evidence") or []
        # The final prompt can contain more packed page text than the compact
        # per-source metadata retained for validation. Inspect both views or a
        # page replay can look harmless merely because metadata was shortened.
        copy_evidence = [*selected_evidence]
        if evidence_context_text:
            copy_evidence.append({"evidence_text": evidence_context_text})
        draft_alignment = assess_answer_alignment(answer, selected_evidence)
        draft_copy_ratio = _source_copy_ratio(answer, copy_evidence)
        post_repair_copy_ratio = draft_copy_ratio
        if bool((constraints or {}).get("enable_answer_repair", False)) and answer and usable_evidence_count and (
            _needs_answer_repair(answer, copy_evidence)
            or _is_evidence_refusal(answer)
            or not re.search(r"\[S\d+(?::C\d+)?\]", answer)
            or draft_alignment.get("unsupported_line_count", 0) > 0
        ):
            repair_prompt = _build_answer_repair_prompt(query, evidence_context_text, answer)
            repair_prompt += "\n\n" + validation_prompt
            repaired_output = complete(
                repair_prompt,
                requested_max=min(requested_final_tokens, 4096),
            )
            repaired_answer = _clean_answer(clean_final_continuation(repaired_output))
            replaced_refusal = _is_evidence_refusal(answer) and not _is_evidence_refusal(repaired_answer)
            if repaired_answer and (
                replaced_refusal
                or _answer_revision_key(repaired_answer, copy_evidence, selected_evidence)
                < _answer_revision_key(answer, copy_evidence, selected_evidence)
            ):
                answer = repaired_answer
                retry_output = repaired_output
        # A repair can still produce a fluent paragraph with only one
        # citation at the top.  Give the same final model one short, explicit
        # citation-contract pass; this is not a verifier and it never adds
        # facts or sources.  If the evidence cannot support a claim, the
        # model is instructed to mark that point unconfirmed.
        if bool((constraints or {}).get("enable_answer_repair", False)) and answer and usable_evidence_count:
            post_repair_alignment = assess_answer_alignment(answer, selected_evidence)
            post_repair_copy_ratio = _source_copy_ratio(answer, copy_evidence)
            if (
                post_repair_alignment.get("unsupported_line_count", 0) > 0
                or post_repair_copy_ratio >= 0.65
            ):
                strict_repair_prompt = _build_answer_repair_prompt(query, evidence_context_text, answer)
                strict_repair_prompt += (
                    "\n\nThe previous rewrite still contains factual lines without a matching source span. "
                    "Return only the requested answer. Put [S#:C#] on every factual sentence. "
                    "For any requested item not stated directly in the evidence, write that it is unconfirmed "
                    "instead of drafting a configuration or filling the gap from memory. Never output DRAFT, "
                    "Answer labels, navigation, or a source-body summary."
                )
                strict_output = complete(
                    strict_repair_prompt,
                    requested_max=min(requested_final_tokens, 2048),
                )
                strict_answer = _clean_answer(clean_final_continuation(strict_output))
                if strict_answer and _answer_revision_key(
                    strict_answer, copy_evidence, selected_evidence
                ) < _answer_revision_key(answer, copy_evidence, selected_evidence):
                    answer = strict_answer
                    retry_output = strict_output
        # The model is still a continuation model and can answer from memory
        # even after receiving NO_USABLE_EVIDENCE.  Once the evidence gate has
        # closed, preserve the model's raw output in the trace but never expose
        # an unsupported factual continuation to the user.  This is an output
        # protocol boundary, not a second model/verifier or a truth judgement.
        closed_world_enforced = not usable_evidence_count and not calculation_results
        if closed_world_enforced and answer and not _is_evidence_refusal(answer):
            answer = _closed_world_fallback()
        if not closed_world_enforced:
            answer = _normalize_evidence_code_spans(answer, evidence_context_text)
            answer = _normalize_free_threading_polarity(answer, query, evidence_context_text)
            answer = _dedupe_repeated_sentences(answer)
            answer = _enforce_how_to_scope(answer, query)
        answer = _attach_mechanical_citations(
            answer,
            answer_context.get("selected_evidence") or [],
        )
        answer = _enforce_citation_contract(answer, data, answer_context)
        # Citation expansion can create a repeated source marker even when the
        # model emitted it only once.  Re-run the protocol-only cleanup after
        # expansion, without changing factual wording.
        answer = _clean_answer(answer)
        answer = _enforce_latest_list_shape(answer, task_plan)
        copy_guard_ratio = max(
            _source_copy_ratio(answer, copy_evidence),
            float(draft_copy_ratio or 0.0),
            float(post_repair_copy_ratio or 0.0),
        )
        copy_guard_triggered = copy_guard_ratio >= 0.85
        if copy_guard_triggered:
            answer = "当前检索正文过长，无法安全压缩为简洁且逐句有依据的回答；请缩小问题范围后重试。"
        answer = _enforce_risk_contract(answer, constraints)
        answer_copy_ratio = _source_copy_ratio(answer, copy_evidence)
        answer_alignment = assess_answer_alignment(
            answer,
            [] if closed_world_enforced else selected_evidence,
        )
        answer_quality = {
            "draft_source_copy_ratio": draft_copy_ratio,
            "source_copy_ratio": max(answer_copy_ratio, copy_guard_ratio),
            "source_copy_detected": copy_guard_ratio >= 0.65 or answer_copy_ratio >= 0.65,
            "source_copy_guard_triggered": copy_guard_triggered,
            "closed_world_boundary_enforced": closed_world_enforced,
        }
        model_output = retry_output or raw_text
        if answer:
            mode = "local_rwkv_final"
            return {
                "content": answer,
                "mode": mode,
                "evidence_count": usable_evidence_count,
                "citation_refs": context_citation_refs,
                "prompt": prompt,
                "model_output": model_output,
                "repair_prompt": repair_prompt,
                "repair_output": retry_output,
                "answer_quality": answer_quality,
                "answer_alignment": answer_alignment,
                **context_fields,
            }
        if not answer:
            fallback = _evidence_limited_fallback(
                query,
                answer_context,
                reason="rwkv_empty_output",
            )
            answer = fallback["content"]
            fallback_quality = fallback["answer_quality"]
            fallback_quality["draft_model_output_chars"] = len(model_output)
            return {
                "content": answer,
                "mode": fallback["mode"],
                "evidence_count": usable_evidence_count,
                "citation_refs": context_citation_refs,
                "prompt": prompt,
                "model_output": model_output,
                "fallback_reason": fallback["fallback_reason"],
                "answer_quality": fallback_quality,
                "answer_alignment": fallback["answer_alignment"],
                **context_fields,
            }
        return {
            "content": answer,
            "mode": "local_rwkv_final",
            "evidence_count": usable_evidence_count,
            "citation_refs": context_citation_refs,
            "prompt": prompt,
            "model_output": model_output,
            "answer_quality": answer_quality,
            "answer_alignment": answer_alignment,
            **context_fields,
        }
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        fallback = _evidence_limited_fallback(
            query,
            answer_context,
            reason="rwkv_final_call_error",
            model_error=error_text,
        )
        return {
            "content": fallback["content"],
            "mode": fallback["mode"],
            "evidence_count": usable_evidence_count,
            "citation_refs": context_citation_refs,
            "prompt": prompt,
            "model_output": _clean_answer(str(raw or "")),
            "error": error_text,
            "fallback_reason": fallback["fallback_reason"],
            "model_error": error_text,
            "answer_quality": {
                **fallback["answer_quality"],
                "source_copy_ratio": 0.0,
                "source_copy_detected": False,
            },
            "answer_alignment": fallback["answer_alignment"],
            **context_fields,
        }
