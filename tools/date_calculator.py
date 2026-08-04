"""Deterministic date arithmetic exposed as a model-selectable tool.

The model still owns retrieval and chooses the operands.  This module only
parses strict ISO dates and performs the subtraction, so date arithmetic does
not depend on RWKV's mental calculation.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from tools.registry import ToolRegistry


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SOURCE_REF = re.compile(r"^S\d+(?::C\d+)?$", re.IGNORECASE)
_DATE_CANDIDATE = re.compile(
    r"\b((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b"
    r"|\b((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日?"
)


def extract_date_candidates(text: Any) -> list[str]:
    """Extract valid calendar-date candidates from retrieved text.

    This is candidate extraction only.  It does not decide which date belongs
    to which event; that semantic binding remains the model's responsibility.
    """

    candidates: list[str] = []
    for match in _DATE_CANDIDATE.finditer(str(text or "")):
        year, month, day = match.group(1, 2, 3) if match.group(1) else match.group(4, 5, 6)
        try:
            normalized = date(int(year), int(month), int(day)).isoformat()
        except (TypeError, ValueError):
            continue
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _parse_iso_date(value: Any, field: str) -> date:
    text = str(value or "").strip()
    if not _ISO_DATE.fullmatch(text):
        raise ValueError(f"{field} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid calendar date") from exc


def _source_refs(*values: Any) -> list[str]:
    refs: list[str] = []
    for value in values:
        for item in str(value or "").split(","):
            item = item.strip()
            if item and _SOURCE_REF.fullmatch(item) and item.upper() not in refs:
                refs.append(item.upper())
    return refs


@ToolRegistry.register(
    name="date_diff",
    phase="ALL",
    model_visible=True,
    category="computation",
    description="Calculate exact calendar-day distance between two already-confirmed YYYY-MM-DD dates; it does not select event dates or retrieve facts.",
    signature="""[Tool] date_diff
- Function: calculate the exact number of calendar days between two dates.
- Parameters: date_a (YYYY-MM-DD), date_b (YYYY-MM-DD), source_a (optional S#:C#), source_b (optional S#:C#).
- Use only after the dates are visible in retrieved evidence or explicitly supplied by the user.
- Returns both absolute days and signed days (date_b - date_a). It never searches the web or invents dates.""",
)
def date_diff(
    date_a: str,
    date_b: str,
    source_a: str = "",
    source_b: str = "",
    **_: Any,
) -> str:
    """Return an auditable, deterministic date difference as JSON."""

    try:
        start = _parse_iso_date(date_a, "date_a")
        end = _parse_iso_date(date_b, "date_b")
    except ValueError as exc:
        return json.dumps(
            {
                "status": "error",
                "tool": "date_diff",
                "error_class": "invalid_date",
                "message": str(exc),
            },
            ensure_ascii=False,
        )

    signed_days = (end - start).days
    return json.dumps(
        {
            "status": "ok",
            "tool": "date_diff",
            "date_a": start.isoformat(),
            "date_b": end.isoformat(),
            "days": abs(signed_days),
            "signed_days": signed_days,
            "formula": f"{end.isoformat()} - {start.isoformat()} = {signed_days} days",
            "source_refs": _source_refs(source_a, source_b),
        },
        ensure_ascii=False,
    )
