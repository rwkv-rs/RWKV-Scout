"""Validation gates for human-maintained evaluation references."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import urlparse

from utils.network_fetch import NetworkFetchError, fetch_text


REFERENCE_VALIDATOR_VERSION = "reference-validator.v1"
TIME_SENSITIVE_TASKS = {"time_sensitive"}
TIME_SENSITIVE_DOMAINS = {"news_current_events", "finance_business", "public_policy"}


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _valid_url(value: Any) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_reference_case(
    case: dict[str, Any],
    *,
    now: datetime | None = None,
    default_stale_after_days: int = 30,
    check_remote: bool = False,
    remote_timeout: int = 8,
) -> dict[str, Any]:
    """Return a conservative status; only ``ready`` may support promotion."""
    metadata = case.get("reference_metadata") or {}
    answer = str(case.get("reference_answer") or "").strip()
    citations = [item for item in case.get("reference_citations") or [] if isinstance(item, dict)]
    issues: list[str] = []
    if not answer:
        issues.append("missing_reference_answer")
    if not citations:
        issues.append("missing_reference_citations")
    if metadata.get("status") != "human_reviewed":
        issues.append("not_human_reviewed")
    reviewer_id = str(metadata.get("reviewer_id") or "").strip()
    if not reviewer_id:
        issues.append("missing_human_reviewer")
    elif reviewer_id.casefold() in {"model", "auto", "self"}:
        issues.append("non_human_reviewer_id")

    accessibility: list[bool] = []
    for index, citation in enumerate(citations, start=1):
        url = str(citation.get("url") or "").strip()
        valid_url = _valid_url(url)
        if not valid_url:
            issues.append(f"invalid_citation_url:{index}")
        if not str(
            citation.get("evidence_text")
            or citation.get("content")
            or citation.get("source_span")
            or ""
        ).strip():
            issues.append(f"missing_citation_evidence:{index}")
        if check_remote and valid_url:
            try:
                accessibility.append(bool(fetch_text(url, timeout=remote_timeout).strip()))
                if not accessibility[-1]:
                    issues.append(f"empty_citation_source:{index}")
            except (NetworkFetchError, OSError, ValueError):
                accessibility.append(False)
                issues.append(f"inaccessible_citation_source:{index}")

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    reviewed_at = _parse_time(metadata.get("reviewed_at"))
    source_checked_at = _parse_time(metadata.get("source_checked_at"))
    if metadata.get("reviewed_at") and reviewed_at is None:
        issues.append("invalid_reviewed_at")
    if metadata.get("source_checked_at") and source_checked_at is None:
        issues.append("invalid_source_checked_at")
    if not reviewed_at:
        issues.append("missing_reviewed_at")
    if not source_checked_at:
        issues.append("missing_source_checked_at")

    is_time_sensitive = (
        str(case.get("task_type") or "") in TIME_SENSITIVE_TASKS
        or str(case.get("domain") or "") in TIME_SENSITIVE_DOMAINS
    )
    try:
        stale_after_days = max(1, int(metadata.get("stale_after_days") or default_stale_after_days))
    except (TypeError, ValueError):
        stale_after_days = max(1, int(default_stale_after_days))
        issues.append("invalid_stale_after_days")
    freshness_time = source_checked_at or reviewed_at
    has_reference = bool(answer or citations)
    stale = bool(
        is_time_sensitive
        and has_reference
        and (not freshness_time or now - freshness_time > timedelta(days=stale_after_days))
    )
    if stale:
        issues.append("stale_time_sensitive_reference")

    hard_invalid = any(issue.startswith("invalid_") for issue in issues)
    if hard_invalid:
        status = "invalid"
    elif stale:
        status = "stale"
    elif issues:
        status = "pending"
    else:
        status = "ready"
    return {
        "validator_version": REFERENCE_VALIDATOR_VERSION,
        "question_id": case.get("question_id", ""),
        "status": status,
        "ready_for_scoring": status == "ready",
        "is_time_sensitive": is_time_sensitive,
        "key_fact_count": len([item for item in case.get("key_facts") or [] if str(item).strip()]),
        "citation_count": len(citations),
        "citation_accessibility": {
            "checked": check_remote,
            "accessible_count": sum(accessibility),
            "sample_count": len(accessibility),
        },
        "reviewer_id_present": bool(str(metadata.get("reviewer_id") or "").strip()),
        "reviewed_at": metadata.get("reviewed_at", ""),
        "source_checked_at": metadata.get("source_checked_at", ""),
        "issues": issues,
    }


def validate_dataset_references(
    rows: Iterable[dict[str, Any]],
    *,
    now: datetime | None = None,
    default_stale_after_days: int = 30,
    check_remote: bool = False,
    remote_timeout: int = 8,
) -> dict[str, Any]:
    results = [
        validate_reference_case(
            row,
            now=now,
            default_stale_after_days=default_stale_after_days,
            check_remote=check_remote,
            remote_timeout=remote_timeout,
        )
        for row in rows
    ]
    counts = {status: sum(item["status"] == status for item in results) for status in ("ready", "pending", "stale", "invalid")}
    return {
        "validator_version": REFERENCE_VALIDATOR_VERSION,
        "sample_count": len(results),
        "ready_count": counts["ready"],
        "pending_count": counts["pending"],
        "stale_count": counts["stale"],
        "invalid_count": counts["invalid"],
        "all_ready": bool(results) and counts["ready"] == len(results),
        "results": results,
    }
