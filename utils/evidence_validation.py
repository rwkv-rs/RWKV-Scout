"""Mechanical validation metadata for retrieved evidence.

This module never decides that a task_record is true.  It produces inspectable
signals for source ordering, task-point coverage, and cross-source agreement so
the final RWKV can state uncertainty instead of silently filling gaps.
"""

from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import urlparse
from typing import Any, Iterable

from utils.evidence_quality import date_mentions, evidence_kind, evidence_text
from utils.source_authority import authority_for_url, resolve_source_policy
from agent.task_plan_contract import record_fields, record_question, record_id, task_records


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u3400-\u9fff]{2,}")
_LATIN_TOPIC_RE = re.compile(r"(?i)c\+\+|c#|[a-z][a-z0-9_+-]{2,}")
_STOP_TERMS = {
    "什么",
    "哪些",
    "告诉",
    "请问",
    "根据",
    "是否",
    "如何",
    "以及",
    "分别",
    "主要",
    "信息",
    "链接",
    "问题",
    "answer",
    "find",
    "list",
    "tell",
    # Planning language is not fact-bearing evidence.  Without this second
    # group, a generic GitHub profile can appear to cover a point merely
    # because both the point and the page contain words such as
    # ``repositories`` or ``source``.
    "all",
    "projects",
    "project",
    "repositories",
    "repository",
    "github",
    "results",
    "search",
    "source",
    "sources",
    "stars",
    "forks",
    "with",
    "paper",
    "papers",
    "link",
    "links",
    "url",
    "metadata",
    "title",
    "authors",
    "author",
    "venue",
    "year",
    "number",
    "direct",
    "working",
    "official",
    "profile",
    "page",
    "primary",
    "authoritative",
    "reliable",
    "independent",
    "multiple",
    "verify",
    "verification",
    "completeness",
    "complete",
    "correctly",
    "present",
    "attributed",
    "include",
    "including",
    "provide",
    "every",
    "determine",
    "identify",
    "person",
    "people",
    "name",
    "names",
    "affiliation",
    "published",
    "owned",
    "created",
    "required",
    "requested",
    "fact",
    "facts",
    "information",
    "question",
    "answer",
    "format",
    "output",
    "body",
    "ownership",
    # Function words and request verbs do not identify the requested fact.
    "the",
    "for",
    "and",
    "that",
    "this",
    "current",
    "how",
    "can",
    "use",
    "using",
    "support",
    "configure",
    "configuration",
    "documentation",
    "reference",
    "identify",
    "locate",
    "extract",
    "syntax",
    "exact",
    "ensure",
    "prose",
    "code",
    "table",
}


def _host(item: dict[str, Any]) -> str:
    try:
        return (urlparse(str(item.get("url") or "")).hostname or "").casefold().removeprefix("www.")
    except ValueError:
        return ""


def _terms(value: Any) -> set[str]:
    terms: set[str] = set()
    for token in _TOKEN_RE.findall(str(value or "")):
        token = token.casefold()
        if token in _STOP_TERMS:
            continue
        terms.add(token)
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            for size in (2, 3, 4):
                terms.update(token[index : index + size] for index in range(len(token) - size + 1))
    return {term for term in terms if term and term not in _STOP_TERMS}


def _semantic_terms(value: Any) -> set[str]:
    """Return compact topic terms for coverage diagnostics.

    ``_terms`` intentionally emits overlapping CJK n-grams for ranking, but
    using every 2/3/4-gram as a hard requirement makes a long Chinese task
    description look uncovered even when its table or ordered list is present.
    This smaller set is only a routing signal; it never promotes metadata to
    evidence or decides that an answer is true.
    """

    terms: set[str] = set()
    for token in _TOKEN_RE.findall(str(value or "")):
        token = token.casefold()
        if token in _STOP_TERMS:
            continue
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            terms.update(
                token[index : index + 2]
                for index in range(max(0, len(token) - 1))
                if token[index : index + 2] not in _STOP_TERMS
            )
        else:
            terms.add(token)
    return {term for term in terms if term and term not in _STOP_TERMS}


def _latin_topic_terms(value: Any) -> set[str]:
    """Return stable English/product anchors for mixed-language requests.

    The planner commonly writes Chinese task descriptions while the retrieved
    official page is English.  Comparing every translated planning word makes
    a genuine page look uncovered.  This helper intentionally keeps only
    product/entity/topic tokens that can be compared across that language
    boundary; it is a routing signal, not a task_record verifier.
    """

    terms: set[str] = set()
    for token in _LATIN_TOPIC_RE.findall(str(value or "")):
        token = token.casefold()
        if token in _STOP_TERMS:
            continue
        terms.add(token)
    return terms


def source_quality(item: dict[str, Any]) -> dict[str, Any]:
    """Return explainable ranking signals; this is not a truth score."""

    kind = evidence_kind(item)
    host = _host(item)
    text = evidence_text(item)
    score = 0.0
    signals: list[str] = []
    if kind in {"structured_record", "structured_record_legacy"}:
        score += 3.0
        signals.append("structured_record")
    elif kind in {"page_body", "page_body_legacy"}:
        score += 2.0
        signals.append("fetched_page_body")
    if item.get("body_verified") is True:
        score += 1.0
        signals.append("body_verified")
    if len(text) >= 2000:
        score += 0.5
        signals.append("substantive_body")
    if host.endswith((".gov", ".edu", ".org")):
        score += 0.75
        signals.append("institutional_domain_signal")
    if str(item.get("source") or "").casefold().find("api") >= 0:
        score += 0.5
        signals.append("api_source_label")
    providers = item.get("discovery_providers") or []
    if isinstance(providers, list) and len(set(str(value) for value in providers if value)) > 1:
        score += 0.5
        signals.append("multi_provider_discovery")
    candidate_score = item.get("candidate_score")
    if isinstance(candidate_score, (int, float)):
        score += max(0.0, min(1.0, float(candidate_score))) * 0.25
        signals.append("candidate_rank_signal")
    authority = item.get("authority")
    if isinstance(authority, dict):
        authority_rank = int(authority.get("rank") or 0)
        score += min(1.5, authority_rank * 0.35)
        if authority.get("satisfied"):
            signals.append("required_authority_satisfied")
        elif authority.get("required"):
            signals.append("required_authority_missing")
        elif authority_rank:
            signals.append("institutional_authority_signal")
    return {
        "score": round(score, 4),
        "host": host,
        "kind": kind,
        "signals": signals,
        "body_chars": len(text),
        "authority": authority if isinstance(authority, dict) else {},
    }

def _point_rows(constraints: dict[str, Any] | None, query: str) -> list[dict[str, Any]]:
    plan = (constraints or {}).get("task_plan") or {}
    points = task_records(plan, fallback_query=query)
    if points:
        return [
            {
                "record_id": record_id(point),
                "text": " ".join(
                    [record_question(point), *record_fields(point)]
                ).strip(),
                "acceptance": [],
            }
            for index, point in enumerate(points, start=1)
        ]
    return [{"record_id": "P1", "text": str(query or ""), "acceptance": []}]


def _coverage(point_text: str, body: str, query: str = "") -> dict[str, Any]:
    required = _terms(point_text)
    observed = _terms(body)
    matched = sorted(required & observed)
    score = len(matched) / max(1, len(required))
    minimum_matches = 2 if len(required) <= 2 else 3
    compact_required = _semantic_terms(point_text)
    compact_matched = sorted(compact_required & _semantic_terms(body))
    compact_score = len(compact_matched) / max(1, len(compact_required))
    query_required = _semantic_terms(query)
    query_matched = sorted(query_required & _semantic_terms(body))
    query_score = len(query_matched) / max(1, len(query_required))
    # Keep Latin entity/topic anchors from the user's query when planner and
    # source body use different languages; translated prose need not share
    # n-grams with the retained page even when both identify the same topic.
    query_topic_required = _latin_topic_terms(query)
    point_topic_required = _latin_topic_terms(point_text)
    topic_required = query_topic_required & point_topic_required
    if len(topic_required) < 2:
        topic_required = query_topic_required
    topic_observed = _latin_topic_terms(body)
    topic_matched = sorted(topic_required & topic_observed)
    topic_score = len(topic_matched) / max(1, len(topic_required))
    qualifier_rules = (
        (
            re.compile(r"(?i)(performance|benchmark|throughput|latency|性能|基准|吞吐|延迟)"),
            re.compile(r"(?i)(performance|benchmark|throughput|latency|ops/?s|tokens?/?s|性能|基准|吞吐|延迟)"),
        ),
    )
    qualifiers_satisfied = all(
        not required.search(point_text) or observed.search(body)
        for required, observed in qualifier_rules
    )
    structured_list_point = bool(
        re.search(
            r"(?i)(list|table|order|整理|列表|顺序|清单|表格)",
            point_text,
        )
    )
    compact_threshold = max(3, min(10, (len(compact_required) + 1) // 2))
    compact_covered = len(compact_matched) >= compact_threshold and compact_score >= 0.50
    query_anchor_covered = (
        structured_list_point
        and len(query_matched) >= 3
        and query_score >= 0.20
        and bool(compact_matched)
    )
    bilingual_topic_covered = (
        len(topic_required) >= 2
        and len(topic_matched) == len(topic_required)
        and qualifiers_satisfied
    )
    covered = (
        len(matched) >= minimum_matches and score >= 0.60
    ) or compact_covered or query_anchor_covered or bilingual_topic_covered
    return {
        "score": round(score, 4),
        "required_term_count": len(required),
        "matched_terms": matched[:40],
        "compact_score": round(compact_score, 4),
        "compact_matched_terms": compact_matched[:40],
        "query_anchor_score": round(query_score, 4),
        "query_anchor_terms": query_matched[:40],
        "topic_required_terms": sorted(topic_required)[:40],
        "topic_matched_terms": topic_matched[:40],
        "topic_anchor_score": round(topic_score, 4),
        "coverage_basis": (
            "direct_terms"
            if len(matched) >= minimum_matches and score >= 0.60
            else "compact_topic_terms"
            if compact_covered
            else "query_anchor_and_structure"
            if query_anchor_covered
            else "bilingual_topic_anchors"
            if bilingual_topic_covered
            else "none"
        ),
        "covered": covered,
    }


def _fact_signatures(item: dict[str, Any]) -> set[str]:
    values: list[str] = []
    raw = item.get("model_extracted_facts")
    if isinstance(raw, str) and raw.strip():
        values.append(raw)
    for candidate in item.get("chunk_candidates") or []:
        if not isinstance(candidate, dict) or not candidate.get("supported"):
            continue
        quote = str(candidate.get("quote") or "").strip()
        if quote:
            values.append(quote)
    signatures: set[str] = set()
    for value in values:
        terms = sorted(_terms(value))
        if terms:
            signatures.add(" ".join(terms[:24]))
    return signatures


def build_evidence_validation(
    data: dict[str, Any],
    *,
    query: str,
    constraints: dict[str, Any] | None = None,
    selected: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build ranking, coverage and agreement metadata for the final prompt."""

    rows = [item for item in (selected if selected is not None else data.get("results") or []) if isinstance(item, dict)]
    profiles: list[dict[str, Any]] = []
    source_policy = resolve_source_policy(query, constraints)
    for index, item in enumerate(rows, start=1):
        authority = item.get("authority")
        if not isinstance(authority, dict):
            authority = authority_for_url(item.get("url"), query, constraints)
        quality = source_quality({**item, "authority": authority})
        profiles.append(
            {
                "ref_id": f"S{index}",
                "url": str(item.get("url") or ""),
                "host": quality["host"],
                "quality_score": quality["score"],
                "quality_signals": quality["signals"],
                "body_chars": quality["body_chars"],
                "date_mentions": date_mentions(evidence_text(item)),
                "fact_signatures": sorted(_fact_signatures(item))[:16],
                "authority": authority,
            }
        )

    coverage_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    points = _point_rows(constraints, query)
    for point in points:
        source_matches: list[dict[str, Any]] = []
        date_by_host: dict[str, set[str]] = defaultdict(set)
        for index, item in enumerate(rows, start=1):
            result = _coverage(point["text"], evidence_text(item), query)
            if result["covered"]:
                profile = profiles[index - 1]
                source_matches.append(
                    {
                        "ref_id": profile["ref_id"],
                        "host": profile["host"],
                        "score": result["score"],
                        "matched_terms": result["matched_terms"],
                        "coverage_basis": result["coverage_basis"],
                        "authority": profile["authority"],
                    }
                )
                for value in profile["date_mentions"]:
                    date_by_host[profile["host"] or profile["ref_id"]].add(value)
        observed_dates = sorted({value for values in date_by_host.values() for value in values})
        independent_hosts = sorted({row["host"] or row["ref_id"] for row in source_matches})
        date_conflict = len(observed_dates) > 1 and len(date_by_host) > 1
        if date_conflict:
            conflicts.append(
                {
                    "task_record_id": point["record_id"],
                    "type": "date_variants_observed",
                    "dates": observed_dates,
                    "sources": sorted(date_by_host),
                    "interpretation": "candidate conflict only; the final model must inspect the quoted body text",
                }
            )
        authority_satisfied = any(
            bool(row.get("authority", {}).get("satisfied"))
            for row in source_matches
        )
        answerable = bool(source_matches) and (
            not source_policy.get("required") or authority_satisfied
        )
        coverage_rows.append(
            {
                "task_record_id": point["record_id"],
                "task": point["text"],
                "status": (
                    "covered"
                    if answerable
                    else "authority_missing"
                    if source_matches
                    else "missing"
                ),
                "answerable": answerable,
                "authority_satisfied": authority_satisfied,
                "source_policy": source_policy,
                "source_count": len(source_matches),
                "independent_host_count": len(independent_hosts),
                "sources": source_matches,
                "observed_dates": observed_dates,
                "agreement": (
                    "multi_source_overlap"
                    if len(independent_hosts) >= 2
                    else "single_source"
                    if source_matches
                    else "no_source"
                ),
            }
        )

    shared_signatures: dict[str, set[str]] = defaultdict(set)
    for profile in profiles:
        for signature in profile["fact_signatures"]:
            shared_signatures[signature].add(profile["ref_id"])
    repeated_facts = [
        {"signature": signature, "sources": sorted(source_ids)}
        for signature, source_ids in shared_signatures.items()
        if len(source_ids) >= 2
    ]
    return {
        "validation_version": "evidence-validation.v1",
        "is_truth_judgement": False,
        "source_ranking": profiles,
        "task_record_coverage": coverage_rows,
        "cross_source": {
            "multi_source_task_records": sum(row["agreement"] == "multi_source_overlap" for row in coverage_rows),
            "single_source_task_records": sum(row["agreement"] == "single_source" for row in coverage_rows),
            "missing_task_records": sum(not row.get("answerable") for row in coverage_rows),
            "authority_missing_task_records": sum(row["status"] == "authority_missing" for row in coverage_rows),
            "repeated_fact_signatures": repeated_facts[:32],
            "candidate_conflicts": conflicts,
        },
        "policy": (
            "Signals are routing metadata only. They cannot promote snippets, titles, URLs, "
            "model facts, source agreement, or third-party pages into an official task_record without "
            "direct EVIDENCE BODY text and a satisfied source policy."
        ),
    }


def _alignment_script_family(value: Any) -> str:
    text = re.sub(r"`[^`\n]+`|https?://\S+|\[S\d+(?::C\d+)?\]", " ", str(value or ""), flags=re.IGNORECASE)
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk >= 4 and cjk * 2 >= latin:
        return "cjk"
    if latin >= 12 and latin > cjk * 2:
        return "latin"
    return "mixed"


def assess_answer_alignment(answer: str, selected_evidence: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure whether answer lines cite and overlap selected source bodies."""

    sources = list(selected_evidence or [])
    if not sources:
        return {
            "alignment_version": "answer-alignment.v1",
            "claim_line_count": 0,
            "aligned_line_count": 0,
            "unsupported_line_count": 0,
            "non_applicable_line_count": 0,
            "rows": [],
            "policy": "No selected evidence; alignment is not applicable.",
        }
    rows: list[dict[str, Any]] = []
    unsupported = 0
    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip()
        if not line or line.casefold().startswith(("sources:", "source:")):
            continue
        if re.fullmatch(r"\[S\d+\]", line, flags=re.IGNORECASE):
            continue
        if line.casefold().startswith(
            (
                "the provided source material does not",
                "the retrieved evidence body does not",
                "the search results do not",
                "the available evidence does not",
            )
        ):
            continue
        if any(marker in line for marker in ("无法确认", "证据不足", "缺少", "不能确定", "cannot confirm", "insufficient evidence")):
            continue
        refs = [int(value) - 1 for value in re.findall(r"\[S(\d+)(?::C\d+)?\]", line, flags=re.IGNORECASE)]
        targets = [sources[index] for index in refs if 0 <= index < len(sources)] if refs else sources
        line_script = _alignment_script_family(line)
        evidence_script = _alignment_script_family(
            "\n".join(evidence_text(item) for item in targets)
        )
        if (
            line_script in {"cjk", "latin"}
            and evidence_script in {"cjk", "latin"}
            and line_script != evidence_script
        ):
            rows.append(
                {
                    "text": line[:500],
                    "refs": refs,
                    "best_ref": f"S{refs[0] + 1}" if refs else "",
                    "overlap": None,
                    "aligned": None,
                    "applicable": False,
                    "basis": "cross_language_lexical_alignment_not_applicable",
                }
            )
            continue
        answer_terms = _terms(line)
        best = 0.0
        best_ref = ""
        for index, item in enumerate(targets):
            body_terms = _terms(evidence_text(item))
            overlap = len(answer_terms & body_terms) / max(1, len(answer_terms))
            if overlap > best:
                best = overlap
                best_ref = f"S{sources.index(item) + 1}" if item in sources else f"S{index + 1}"
        aligned = best >= 0.12
        if not aligned:
            unsupported += 1
        rows.append({"text": line[:500], "refs": refs, "best_ref": best_ref, "overlap": round(best, 4), "aligned": aligned, "applicable": True, "basis": "lexical_overlap"})
    return {
        "alignment_version": "answer-alignment.v1",
        "claim_line_count": len(rows),
        "aligned_line_count": sum(row.get("aligned") is True for row in rows),
        "unsupported_line_count": unsupported,
        "non_applicable_line_count": sum(row.get("applicable") is False for row in rows),
        "rows": rows[:64],
        "policy": "Diagnostic signal only; cross-language lexical overlap is not applicable and does not assert support or non-support.",
    }
