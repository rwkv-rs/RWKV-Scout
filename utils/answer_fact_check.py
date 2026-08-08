"""Mechanical final-answer fact checks for the live Agent path.

This is not an evaluator and has no reference answer.  It checks only whether
literal dates, versions, percentages, and calculated day counts in the final
RWKV text appear in the captured evidence or deterministic tool results.  It
never edits, replaces, or regenerates the answer.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Iterable


_ISO_DATE = re.compile(r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b")
_CN_DATE = re.compile(r"\b((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日?\b")
_EN_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_VERSION = re.compile(
    r"(?<![A-Za-z0-9])v?\d+\.\d+(?:\.\d+){0,3}"
    r"(?:[-+][A-Za-z0-9.-]+)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_COMPOUND_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[A-Za-z0-9_.-]{4,}(?![A-Za-z0-9_.-]))"
    r"(?=[A-Za-z0-9_.-]*\d)"
    r"[A-Za-z][A-Za-z0-9_]*(?:[.-][A-Za-z0-9_]+)*"
)
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_DAY_COUNT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:calendar\s+)?days?\b", re.IGNORECASE)
_MONTHS = {name: index for index, name in enumerate(("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), start=1)}


def _normal_date(value: str) -> str:
    text = str(value or "").strip()
    match = _CN_DATE.fullmatch(text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
        except ValueError:
            return ""
    match = re.fullmatch(r"((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
        except ValueError:
            return ""
    match = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),?\s+((?:19|20)\d{2})", text)
    if match and match.group(1).capitalize() in _MONTHS:
        try:
            return date(int(match.group(3)), _MONTHS[match.group(1).capitalize()], int(match.group(2))).isoformat()
        except ValueError:
            return ""
    return ""


def _dates(text: str) -> set[str]:
    values = {_normal_date(item) for item in _ISO_DATE.findall(text or "")}
    values.update(_normal_date("-".join(match)) for match in _CN_DATE.findall(text or ""))
    values.update(_normal_date(item) for item in _EN_DATE.findall(text or ""))
    return {item for item in values if item}


def _evidence_text(items: Iterable[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        parts.extend(
            str(item.get(key) or "")
            for key in ("source_excerpt", "page_excerpt", "content", "structured_evidence_text", "abstract", "evidence_text")
        )
    return "\n".join(parts)


def _claim_rows(answer: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in str(answer or "").splitlines() or [str(answer or "")]:
        # Citation destinations and bare links are metadata.  Strip them before
        # every literal check so a version or date in a source URL is not
        # counted as a second answer claim.
        literal_line = re.sub(r"\]\(https?://[^)]+\)", "]", line, flags=re.IGNORECASE)
        literal_line = re.sub(r"https?://\S+", "", literal_line, flags=re.IGNORECASE)
        for value in sorted(_dates(literal_line)):
            rows.append({"kind": "date", "value": value, "line": line[:600]})
        for pattern, kind in ((_VERSION, "version"), (_PERCENT, "percentage"), (_DAY_COUNT, "day_count")):
            for match in pattern.findall(literal_line):
                rows.append({"kind": kind, "value": re.sub(r"\s+", " ", str(match)).strip(), "line": line[:600]})
        # An exact identifier written in prose is a factual value. Executable
        # URLs are checked separately by the synthesis code guard.
        for value in _COMPOUND_IDENTIFIER.findall(literal_line):
            if re.fullmatch(r"v\d+(?:\.\d+)+", value, flags=re.IGNORECASE):
                continue
            rows.append(
                {
                    "kind": "compound_identifier",
                    "value": value,
                    "line": line[:600],
                }
            )
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["kind"], row["value"].casefold(), row["line"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def check_answer_facts(
    answer: str,
    *,
    evidence: Iterable[dict[str, Any]] | None = None,
    calculation_results: Iterable[dict[str, Any]] | None = None,
    freshness_policy: dict[str, Any] | None = None,
    user_supplied_literals: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return inspectable support signals without changing ``answer``."""

    body = _evidence_text(evidence)
    calc_text = json.dumps(list(calculation_results or []), ensure_ascii=False)
    searchable = f"{body}\n{calc_text}"
    supplied_literals = [
        str(value or "")
        for value in user_supplied_literals or []
        if str(value or "").strip()
    ]
    supplied_text = "\n".join(supplied_literals)
    evidence_dates = _dates(searchable)
    supplied_dates = _dates(supplied_text)
    unsupported: list[dict[str, str]] = []
    supported: list[dict[str, str]] = []
    for claim in _claim_rows(answer):
        value = claim["value"]
        if claim["kind"] == "date":
            found = value in evidence_dates
            user_supplied = value in supplied_dates
        elif claim["kind"] == "day_count":
            number = re.search(r"\d+(?:\.\d+)?", value)
            found = bool(number and re.search(rf"\"(?:days|signed_days)\"\s*:\s*{re.escape(number.group(0))}\b", calc_text)) or value.casefold() in searchable.casefold()
            user_supplied = value.casefold() in supplied_text.casefold()
        else:
            found = value.casefold().replace(" ", "") in searchable.casefold().replace(" ", "")
            user_supplied = (
                value.casefold().replace(" ", "")
                in supplied_text.casefold().replace(" ", "")
            )
        accepted = bool(found or user_supplied)
        row = {
            **claim,
            "supported": accepted,
            "support_basis": (
                "selected_evidence_or_calculation"
                if found
                else "user_supplied_scope"
                if user_supplied
                else ""
            ),
        }
        (supported if accepted else unsupported).append(row)

    freshness_violations: list[dict[str, str]] = []
    as_of = str((freshness_policy or {}).get("as_of") or "")
    if as_of:
        for claim in list(supported):
            if claim.get("kind") != "date" or claim.get("value", "") <= as_of:
                continue
            violation = {**claim, "reason": "answer date is after the explicit as_of cutoff"}
            freshness_violations.append(violation)
            supported.remove(claim)
            unsupported.append(violation)

    return {
        "checker_version": "answer-fact-check.v1",
        "status": "needs_review" if unsupported else "supported" if supported else "no_literal_claims",
        "answer_changed": False,
        "scope": ["dates", "versions", "percentages", "day_counts", "compound_identifiers"],
        "claim_count": len(supported) + len(unsupported),
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "freshness_violations": freshness_violations[:64],
        "supported_claims": supported[:64],
        "unsupported_claims": unsupported[:64],
        "freshness_policy": dict(freshness_policy or {}),
        "evidence_body_count": sum(1 for item in evidence or [] if isinstance(item, dict)),
        "user_supplied_literal_count": len(supplied_literals),
    }


__all__ = ["check_answer_facts"]
