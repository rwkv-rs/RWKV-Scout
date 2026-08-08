"""Product-time and source-freshness metadata.

Freshness is a retrieval constraint, not an evaluation score.  The helpers
only derive explicit date limits and annotate what the connector knows; an
unknown publication date is never treated as fresh or stale by guesswork.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any


_ISO_DATE = re.compile(r"\b((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_ZH_DATE = re.compile(r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_EN_DATE = re.compile(
    r"\b(?:on\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_MONTHS = {name: index for index, name in enumerate(("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), start=1)}


def _date_value(year: str, month: str, day: str) -> str:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except (TypeError, ValueError):
        return ""


def extract_explicit_date(value: Any) -> str:
    """Find an explicit ISO/English calendar date without semantic guessing."""

    text = str(value or "")
    match = _ISO_DATE.search(text)
    if match:
        return _date_value(match.group(1), match.group(2), match.group(3))
    match = _ZH_DATE.search(text)
    if match:
        return _date_value(match.group(1), match.group(2), match.group(3))
    match = _EN_DATE.search(text)
    if match:
        return _date_value(match.group(3), str(_MONTHS[match.group(1).capitalize()]), match.group(2))
    return ""


def build_freshness_policy(goal: Any, task_plan: dict[str, Any] | None = None, *, now: datetime | None = None) -> dict[str, Any]:
    """Build an explicit as-of policy from user text or a plan field."""

    plan = task_plan if isinstance(task_plan, dict) else {}
    explicit = extract_explicit_date(plan.get("as_of"))
    if not explicit:
        text = str(goal or "")
        markers = re.compile(r"(?:截至|as\s+of|before|through|up\s+to|截止)\s*[:：]?\s*", re.IGNORECASE)
        marker = markers.search(text)
        if marker:
            explicit = extract_explicit_date(text[marker.end() :])
    current = now or datetime.now(timezone.utc)
    return {
        "as_of": explicit or None,
        "now": current.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "mode": "explicit_cutoff" if explicit else "retrieval_time_only",
        "unknown_date_policy": "preserve_unknown",
    }


def annotate_freshness(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Attach freshness metadata without dropping a source body."""

    item = dict(row)
    as_of = str(policy.get("as_of") or "")
    source_date = ""
    for key in ("published", "published_at", "publication_date", "updated", "updated_at", "date"):
        source_date = extract_explicit_date(item.get(key))
        if source_date:
            break
    if not source_date:
        source_date = extract_explicit_date(item.get("snippet"))
    if as_of and source_date:
        state = "within_cutoff" if source_date <= as_of else "after_cutoff"
    elif as_of:
        state = "unknown_date"
    else:
        state = "dated" if source_date else "unknown_date"
    item["freshness"] = {
        "policy": "explicit_cutoff" if as_of else "retrieval_time_only",
        "as_of": as_of or None,
        "source_date": source_date or None,
        "state": state,
    }
    return item


def annotate_result_freshness(result: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(result or {})
    enriched["freshness_policy"] = dict(policy)
    enriched["results"] = [
        annotate_freshness(item, policy)
        for item in enriched.get("results") or []
        if isinstance(item, dict)
    ]
    return enriched


__all__ = ["annotate_freshness", "annotate_result_freshness", "build_freshness_policy", "extract_explicit_date"]
