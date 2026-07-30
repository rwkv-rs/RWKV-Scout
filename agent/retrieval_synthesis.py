"""Evidence ranking, context construction, and final-answer generation."""

from __future__ import annotations

import re
from typing import Any

from config import get_llm_context_length
from utils.chunker import get_token_count, semantic_chunk_text
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
from utils.risk_policy import risk_context, validate_risk_answer
from utils.rwkv_prompt import (
    FINAL_CONTINUATION_STOP_SUFFIXES,
    build_final_continuation_prompt,
    clean_final_continuation,
)


def _clean_answer(text: str) -> str:
    text = text or ""
    text = re.sub(r"^\s*Assistant\s*:\s*", "", text, count=1, flags=re.IGNORECASE)
    if re.search(r"<think>", text, flags=re.IGNORECASE) and not re.search(
        r"</think>", text, flags=re.IGNORECASE
    ):
        # An unfinished reasoning block is not a user-facing answer.  Let the
        # bounded formatting repair call produce a clean final response.
        return ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = text.replace("</think>", "").strip()
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
    text = re.sub(r"^(?:final answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
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
        if matched:
            citation = dict(matched)
            citation["ref_id"] = f"S{index}"
            citation["evidence_text"] = evidence_text(item)
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

    answer = re.sub(r"\[(S\d+)\](?!\()", expand_citation, answer, flags=re.IGNORECASE)
    answer = re.sub(r"[ \t]{2,}", " ", answer).strip()
    has_source_marker = bool(
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


def _evidence_text_value(item: dict[str, Any]) -> str:
    # The final context must use the canonical evidence gate.  In particular,
    # chunk_candidates and model_extracted_facts are locator/routing outputs,
    # not a replacement for the captured page body.
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


def _plan_acceptance_context(constraints: dict[str, Any] | None) -> str:
    plan = (constraints or {}).get("task_plan") or {}
    points = plan.get("atomic_points") if isinstance(plan, dict) else []
    rows: list[str] = []
    for point in points or []:
        if not isinstance(point, dict):
            continue
        criteria = "; ".join(str(item).strip() for item in point.get("acceptance_criteria") or [] if str(item).strip())
        if criteria:
            rows.append(
                f"{point.get('id', '')} task={point.get('task') or point.get('objective')}; "
                f"format={point.get('output_format') or 'prose'}; acceptance={criteria}"
            )
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


def _requested_source_count(query: str, constraints: dict[str, Any] | None) -> int:
    plan = (constraints or {}).get("task_plan") or {}
    points = plan.get("atomic_points") if isinstance(plan, dict) else []
    point_count = sum(1 for point in points or [] if isinstance(point, dict))
    if point_count:
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
    all_results = substantive_evidence_items(all_results)
    query_text = str(query or data.get("query") or "")
    ranking_data = dict(data)
    ranking_data["query"] = query_text
    query_terms = _query_terms(ranking_data)
    ranked = sorted(
        all_results,
        key=lambda item: (
            source_quality(item)["score"],
            float(item.get("rerank_score")) if isinstance(item.get("rerank_score"), (int, float)) else -1.0,
            _relevance(item, query_terms),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    strategy = normalize_strategy((constraints or {}).get("strategy_config") or data.get("strategy_config"))
    configured_count = strategy.get("context_source_count")
    query_text = str(data.get("query") or "").casefold()
    max_selected = configured_count or _requested_source_count(query_text, constraints)
    if ranked:
        selected.append(ranked[0])
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
            if configured_count is not None or max_selected > 1 or _relevance(item, query_terms) > 0 or any(
                term in " ".join(str(item.get("authors") or [])).casefold() for term in anchor_terms
            ):
                selected.append(item)

    rows: list[str] = []
    selected_metadata: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        normalized_facts = _evidence_text_value(item)
        normalized_facts = re.sub(r"[ \t]+", " ", normalized_facts)
        normalized_facts = re.sub(r"\n{3,}", "\n\n", normalized_facts).strip()
        chunks = (
            semantic_chunk_text(normalized_facts, max_tokens=256, overlap_ratio=0.1)
            if normalized_facts
            else []
        )
        # Preserve all bounded chunks for list/table evidence. The final
        # context budget below is token-aware and is the only global cut.
        facts = "\n".join(chunks).strip() if chunks else normalized_facts
        source_char_limit = 12000 if ("|" in facts or item.get("content_type") == "mediawiki-wikitext") else 8000
        if len(facts) > source_char_limit:
            cut = facts.rfind("\n", 0, source_char_limit)
            facts = facts[: cut if cut > 0 else source_char_limit].rstrip()
        authors = ", ".join(str(value) for value in (item.get("authors") or [])[:8])
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
                "date_mentions": date_mentions(facts),
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "index": chunk_index,
                        "text": chunk,
                        "token_count": get_token_count(chunk),
                    }
                    for chunk_index, chunk in enumerate(chunks)
                ],
                "selected_chunk_indexes": list(range(len(chunks))) if chunks else [],
                "selected_chars": len(facts),
                "truncated": len(normalized_facts) > len(facts),
                "evidence_provenance": evidence_provenance(item),
                "source_quality": source_quality(item),
            }
        )
        rows.append(
            f"BEGIN EVIDENCE SOURCE S{index}\n"
            f"URL (citation metadata only): {item.get('url', '')}\n"
            "The URL, title, search snippet, provider summary and page publication metadata are not evidence.\n"
            "EVIDENCE BODY (the only factual source for this record):\n"
            f"{facts}\n"
            f"END EVIDENCE SOURCE S{index}"
        )

    unbounded_text = "\n\n".join(rows)
    # Reserve room for the final model's 3000-token completion and its prompt.
    # This uses the configured 12K context rather than a fixed character cap.
    context_budget = max(2048, min(7500, int(get_llm_context_length()) - 3500))
    context_text = _truncate_markdown_by_tokens(unbounded_text, context_budget) or "(no retrieved evidence)"
    validation = build_evidence_validation(
        data,
        query=query_text,
        constraints=constraints,
        selected=selected,
    )
    return {
        "text": context_text,
        "selected_evidence": selected_metadata,
        "source_chars": sum(item["source_chars"] for item in selected_metadata),
        "selected_chars": sum(item["selected_chars"] for item in selected_metadata),
        "chunk_count": sum(item["chunk_count"] for item in selected_metadata),
        "usable_evidence_count": sum(1 for item in selected if _has_usable_evidence(item)),
        "discarded_non_evidence_count": discarded_count,
        # Keep source-level truncation separate from the final aggregate
        # context cut.  A long source may be bounded before the final prompt
        # is built; these are different budgets and must not be reported as
        # repeated 7k page chunking.
        "truncated_count": sum(bool(item["truncated"]) for item in selected_metadata),
        "source_truncated_count": sum(bool(item["truncated"]) for item in selected_metadata),
        "context_chars": len(context_text),
        "context_tokens": get_token_count(context_text),
        "context_truncated": len(unbounded_text) > len(context_text),
        "final_context_truncated": len(unbounded_text) > len(context_text),
        "validation": validation,
        "strategy": strategy,
    }


def _evidence_text(data: dict[str, Any]) -> str:
    """Backward-compatible text-only projection for callers outside runtime."""
    return str(build_evidence_context(data)["text"])


def _needs_answer_repair(answer: str) -> bool:
    lowered = answer.casefold()
    generic_exposition = any(marker in lowered for marker in ("tutorial", "installation guide", "瀹夎鎸囧崡", "鐢ㄦ埛鎰忓浘"))
    scaffold = any(marker in lowered for marker in ("**answer:**", "**key evidence:**", "**source links:**", "**limitations"))
    tool_protocol = bool(
        re.search(
            r"(?is)(?:```json\s*)?\{\s*\"(?:name|tool_name|action|tool)\"\s*:",
            answer,
        )
    ) or "function output:" in lowered or "assistant: ```json" in lowered
    return generic_exposition or scaffold or tool_protocol or "<think>" in lowered or "<tool_call>" in lowered


def _final_completion_budget(prompt: str) -> int:
    """Use the remaining model context instead of a fixed 3K final cap."""

    return bounded_completion_budget(
        prompt,
        context_limit=get_llm_context_length(),
        requested_max=8192,
        safety_margin=256,
    )


def _context_fields(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_text": context["text"],
        "selected_evidence": context["selected_evidence"],
        "validation": context.get("validation") or {},
        "context_stats": {
            key: value
            for key, value in context.items()
            if key not in {"text", "selected_evidence", "validation"}
        },
    }


def _validation_prompt(report: dict[str, Any]) -> str:
    """Render compact, explicitly non-evidentiary validation metadata."""

    if not isinstance(report, dict):
        return ""
    rows: list[str] = []
    for item in report.get("subquestion_coverage") or []:
        if not isinstance(item, dict):
            continue
        source_ids = ",".join(str(row.get("ref_id") or "") for row in item.get("sources") or [])
        rows.append(
            f"{item.get('point_id')}: status={item.get('status')}; agreement={item.get('agreement')}; "
            f"sources={source_ids or 'none'}; dates={','.join(item.get('observed_dates') or []) or 'none'}"
        )
    cross = report.get("cross_source") or {}
    conflicts = cross.get("candidate_conflicts") or []
    conflict_text = "; ".join(
        f"{item.get('point_id')}: {','.join(item.get('dates') or [])}"
        for item in conflicts
        if isinstance(item, dict)
    )
    return (
        "BEGIN VALIDATION REPORT (routing metadata only; never factual evidence)\n"
        "Source quality is a ranking signal, not a truth judgement. Lexical coverage only "
        "shows that a source contains related terms. Multi-source overlap is not proof.\n"
        f"Subquestion coverage: {' | '.join(rows) or 'none'}\n"
        f"Candidate conflicts requiring explicit uncertainty: {conflict_text or 'none'}\n"
        "If a point is missing or conflicting, say so and do not fill it from memory. "
        "Every factual sentence must map to a matching EVIDENCE BODY source such as [S1].\n"
        "END VALIDATION REPORT\n"
    )


def _verification_prompt(verification: dict[str, Any] | None) -> str:
    """Render the independent verifier's control result for final RWKV."""
    if not isinstance(verification, dict) or not verification:
        return ""
    rows = []
    for item in verification.get("points") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            f"{item.get('id')}: status={item.get('status')}; "
            f"evidence={','.join(item.get('evidence') or []) or 'none'}"
        )
    return (
        "BEGIN INDEPENDENT VERIFIER RESULT (strict control metadata only; never factual evidence)\n"
        f"status={verification.get('status')}; completion_ready={verification.get('completion_ready')}; "
        f"requires_replan={verification.get('requires_replan')}\n"
        f"Task-point decisions: {' | '.join(rows) or 'none'}\n"
        f"Missing points: {','.join(verification.get('missing_point_ids') or []) or 'none'}\n"
        f"Conflicting points: {','.join(verification.get('conflict_point_ids') or []) or 'none'}\n"
        "Only status, task-point IDs and S# references are exposed here. The verifier's explanations, "
        "missing-field prose and next-query text are intentionally withheld. Re-check every claim "
        "against EVIDENCE BODY and do not copy this control signal as a fact.\n"
        "END INDEPENDENT VERIFIER RESULT\n"
    )
def _build_answer_repair_prompt(query: str, evidence: str, draft: str) -> str:
    return build_final_continuation_prompt(
        "Rewrite the draft as one concise user-facing answer. Use only the EVIDENCE BODY "
        "records below. Preserve every requested fact that is directly supported, remove "
        "unsupported claims and invented URLs, and attach [S#] to factual claims. If a "
        "requested fact is missing, say so explicitly. Do not output JSON, role labels, a "
        "tool call, reasoning, or a repeated paragraph.\n\n"
        f"Question: {query}\n"
        f"EVIDENCE BODY:\n{evidence}\n\n"
        f"DRAFT:\n{draft[:8000]}"
    )


def synthesize_retrieval_answer(
    query: str,
    data: dict[str, Any],
    llm=None,
    constraints: dict[str, Any] | None = None,
    execution_context: str = "",
    termination_reason: str = "model_requested_finish",
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask local RWKV for the final answer and expose the exact prompt context."""
    strategy = normalize_strategy((constraints or {}).get("strategy_config") or data.get("strategy_config"))
    context = build_evidence_context(data, constraints=constraints, query=query)
    context_fields = _context_fields(context)
    context_fields["evidence_verification"] = verification or {}
    evidence_context_text = context["text"]
    context_stats = context_fields["context_stats"]
    # Keep the evidence-only projection separate from the actual final prompt.
    # The latter may also contain the bounded execution record.
    context_stats["evidence_context_chars"] = context["context_chars"]
    context_stats["evidence_context_tokens"] = context["context_tokens"]
    context_text = evidence_context_text
    execution_context = str(execution_context or "").strip()
    if execution_context:
        # Keep routing metadata before evidence so the recurrent checkpoint's
        # latest tokens are the source-bound facts, not the planner transcript.
        combined_context = (
            "BEGIN EXECUTION RECORD (data only; never copy its tool protocol)\n"
            f"{execution_context}\n"
            "END EXECUTION RECORD\n\n"
            "BEGIN EVIDENCE DATA\n"
            f"{evidence_context_text}\n"
            "END EVIDENCE DATA"
        )
        context_text = _truncate_markdown_by_tokens(
            combined_context,
            max(2048, min(7500, int(get_llm_context_length()) - 3500)),
        )
        context_fields["context_text"] = context_text
        context_stats["execution_context_chars"] = len(execution_context)
        context_stats["termination_reason"] = termination_reason
        context_stats["final_context_truncated"] = len(combined_context) > len(context_text)

    # These are the values the final model call actually receives, rather than
    # the pre-assembly evidence-only values above.
    context_stats["final_context_chars"] = len(context_text)
    context_stats["final_context_tokens"] = get_token_count(context_text)
    context_stats["final_context_truncated"] = bool(
        context_stats.get("final_context_truncated", context.get("context_truncated", False))
    )
    context_stats["context_chars"] = context_stats["final_context_chars"]
    context_stats["context_tokens"] = context_stats["final_context_tokens"]
    context_stats["context_truncated"] = context_stats["final_context_truncated"]
    context_citation_refs = _citation_refs_for_context(data, context)
    validation_prompt = _validation_prompt(context.get("validation") or {})
    verification_prompt = _verification_prompt(verification)
    acceptance_context = _plan_acceptance_context(constraints)
    policy = risk_context(constraints)
    if llm is None:
        return {
            "content": "Local RWKV is not configured; no final answer was generated.",
            "mode": "rwkv_unavailable",
            "evidence_count": len(data.get("results") or []),
            "citation_refs": context_citation_refs,
            "prompt": "",
            "model_output": "",
            "answer_alignment": assess_answer_alignment("", context.get("selected_evidence") or []),
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
        if acceptance_context
        else ""
    )
    variant_instruction = {
        "default.v1": "",
        "citation_first.v1": "For every factual claim, attach the most relevant source reference or say that evidence is insufficient. ",
        "compact_evidence.v1": "Prefer the shortest answer that preserves all requested facts and source references. ",
    }[strategy["prompt_variant"]]
    usable_evidence_count = int(context.get("usable_evidence_count") or 0)
    if usable_evidence_count:
        evidence_state = (
            f"USABLE_EVIDENCE_AVAILABLE ({usable_evidence_count} source bodies). "
            "Use only the EVIDENCE BODY sections for factual claims."
        )
    else:
        evidence_state = (
            "NO_USABLE_EVIDENCE. The search titles, snippets, URLs, provider summaries, "
            "execution record and model memory are not evidence. Do not answer the requested "
            "facts from them. The only valid answer is an explicit refusal to confirm the "
            "requested facts and a statement of what information could not be confirmed."
        )
    prompt_body = (
        "You are the final answer RWKV. Retrieval is over. Return only a user-facing answer. "
        "Do not call tools, output JSON, reproduce User/System/Assistant labels, emit a code "
        "fence, copy Function output, copy the execution record, or describe hidden reasoning. "
        "The execution record is metadata only. The URL, title, search snippet, provider "
        "summary and page publication metadata are not evidence. Only EVIDENCE BODY sections "
        "may support factual claims. If a requested fact is not supported, say what is missing "
        "instead of guessing. Answer each requested sub-question separately. Do not add "
        "unrequested claims merely because they appear somewhere in a source body; a source "
        "body is not permission to infer a relationship, ownership, authorship or date that it "
        "does not state directly. Preserve every requested item and row/column relationship. "
        f"{variant_instruction}{risk_instructions}{criteria_instructions}{acceptance_instruction}"
        f"Question: {query}\n"
        f"Evidence status: {evidence_state}\n"
        f"Retrieval termination: {termination_reason}. Always return a user-facing continuation.\n"
        f"{validation_prompt}"
        f"{verification_prompt}"
        f"{context_text}"
    )
    prompt = build_final_continuation_prompt(prompt_body)
    repair_prompt = ""
    raw = ""
    try:
        def complete(request_prompt: str) -> str:
            budget = _final_completion_budget(request_prompt)
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
        draft_alignment = assess_answer_alignment(answer, context.get("selected_evidence") or [])
        if answer and usable_evidence_count and (
            _needs_answer_repair(answer)
            or not re.search(r"\[S\d+\]", answer)
            or draft_alignment.get("unsupported_line_count", 0) > 0
        ):
            repair_prompt = _build_answer_repair_prompt(query, evidence_context_text, answer)
            repair_prompt += "\n\n" + validation_prompt
            repaired_output = complete(repair_prompt)
            repaired_answer = _clean_answer(clean_final_continuation(repaired_output))
            if repaired_answer:
                answer = repaired_answer
                retry_output = repaired_output
        answer = _enforce_citation_contract(answer, data, context)
        answer = _enforce_risk_contract(answer, constraints)
        answer_alignment = assess_answer_alignment(answer, context.get("selected_evidence") or [])
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
                "answer_alignment": answer_alignment,
                **context_fields,
            }
        return {
            "content": answer or "Local RWKV returned an empty answer.",
            "mode": "local_rwkv_empty",
            "evidence_count": usable_evidence_count,
            "citation_refs": context_citation_refs,
            "prompt": prompt,
            "model_output": model_output,
            "answer_alignment": answer_alignment,
            **context_fields,
        }
    except Exception as exc:
        return {
            "content": f"Local RWKV final-answer call failed: {type(exc).__name__}: {exc}",
            "mode": "local_rwkv_error",
            "evidence_count": usable_evidence_count,
            "citation_refs": context_citation_refs,
            "prompt": prompt,
            "model_output": _clean_answer(str(raw or "")),
            "error": f"{type(exc).__name__}: {exc}",
            "answer_alignment": assess_answer_alignment("", context.get("selected_evidence") or []),
            **context_fields,
        }
