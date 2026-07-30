"""Mechanical validation metadata for retrieved evidence.

This module never decides that a claim is true.  It produces inspectable
signals for source ordering, task-point coverage, and cross-source agreement so
the final RWKV can state uncertainty instead of silently filling gaps.
"""

from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import urlparse
from typing import Any, Iterable

from utils.evidence_quality import date_mentions, evidence_kind, evidence_text


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u3400-\u9fff]{2,}")
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
    return {
        "score": round(score, 4),
        "host": host,
        "kind": kind,
        "signals": signals,
        "body_chars": len(text),
    }

def _point_rows(constraints: dict[str, Any] | None, query: str) -> list[dict[str, Any]]:
    plan = (constraints or {}).get("task_plan") or {}
    raw_points = plan.get("atomic_points") if isinstance(plan, dict) else []
    points = [point for point in raw_points or [] if isinstance(point, dict)]
    if points:
        return [
            {
                "id": str(point.get("id") or f"P{index}"),
                "text": " ".join(
                    str(point.get(key) or "")
                    for key in ("task", "objective", "question", "output_format")
                ).strip(),
                "acceptance": [str(value) for value in point.get("acceptance_criteria") or [] if str(value).strip()],
            }
            for index, point in enumerate(points, start=1)
        ]
    return [{"id": "P1", "text": str(query or ""), "acceptance": []}]


def _coverage(point_text: str, body: str) -> dict[str, Any]:
    required = _terms(point_text)
    observed = _terms(body)
    matched = sorted(required & observed)
    score = len(matched) / max(1, len(required))
    return {
        "score": round(score, 4),
        "required_term_count": len(required),
        "matched_terms": matched[:40],
        "covered": bool(matched) and (score >= 0.12 or len(matched) >= 3),
    }


def _fact_signatures(item: dict[str, Any]) -> set[str]:
    values: list[str] = []
    raw = item.get("model_extracted_facts")
    if isinstance(raw, str) and raw.strip():
        values.append(raw)
    for candidate in item.get("chunk_candidates") or []:
        if not isinstance(candidate, dict) or not candidate.get("supported"):
            continue
        values.extend(str(value) for value in candidate.get("facts") or [] if str(value).strip())
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
    for index, item in enumerate(rows, start=1):
        quality = source_quality(item)
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
            }
        )

    coverage_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    points = _point_rows(constraints, query)
    for point in points:
        source_matches: list[dict[str, Any]] = []
        date_by_host: dict[str, set[str]] = defaultdict(set)
        for index, item in enumerate(rows, start=1):
            result = _coverage(point["text"], evidence_text(item))
            if result["covered"]:
                profile = profiles[index - 1]
                source_matches.append(
                    {
                        "ref_id": profile["ref_id"],
                        "host": profile["host"],
                        "score": result["score"],
                        "matched_terms": result["matched_terms"],
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
                    "point_id": point["id"],
                    "type": "date_variants_observed",
                    "dates": observed_dates,
                    "sources": sorted(date_by_host),
                    "interpretation": "candidate conflict only; the final model must inspect the quoted body text",
                }
            )
        coverage_rows.append(
            {
                "point_id": point["id"],
                "task": point["text"],
                "status": "covered" if source_matches else "missing",
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
        "subquestion_coverage": coverage_rows,
        "cross_source": {
            "multi_source_points": sum(row["agreement"] == "multi_source_overlap" for row in coverage_rows),
            "single_source_points": sum(row["agreement"] == "single_source" for row in coverage_rows),
            "missing_points": sum(row["status"] == "missing" for row in coverage_rows),
            "repeated_fact_signatures": repeated_facts[:32],
            "candidate_conflicts": conflicts,
        },
        "policy": (
            "Signals are routing metadata only. They cannot promote snippets, titles, URLs, "
            "model facts, or a source agreement into truth without direct EVIDENCE BODY text."
        ),
    }


def assess_answer_alignment(answer: str, selected_evidence: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measure whether answer lines cite and overlap selected source bodies."""

    sources = list(selected_evidence or [])
    rows: list[dict[str, Any]] = []
    unsupported = 0
    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip()
        if not line or line.casefold().startswith(("sources:", "source:")):
            continue
        if re.fullmatch(r"\[S\d+\]", line, flags=re.IGNORECASE):
            continue
        if any(marker in line for marker in ("无法确认", "证据不足", "缺少", "不能确定", "cannot confirm", "insufficient evidence")):
            continue
        refs = [int(value) - 1 for value in re.findall(r"\[S(\d+)\]", line, flags=re.IGNORECASE)]
        targets = [sources[index] for index in refs if 0 <= index < len(sources)] if refs else sources
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
        rows.append({"text": line[:500], "refs": refs, "best_ref": best_ref, "overlap": round(best, 4), "aligned": aligned})
    return {
        "alignment_version": "answer-alignment.v1",
        "claim_line_count": len(rows),
        "aligned_line_count": sum(row["aligned"] for row in rows),
        "unsupported_line_count": unsupported,
        "rows": rows[:64],
        "policy": "Diagnostic signal only; it does not rewrite or assert the answer is true.",
    }
