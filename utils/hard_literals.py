"""Typed hard-literal extraction for retrieval-state provenance checks.

The extractor recognizes concrete values that can materially redirect a
retrieval route.  It never decides whether a value is true or relevant; callers
compare the canonical ``(kind, value)`` identity against trusted input lanes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_CVE_RE = re.compile(r"\bCVE-(\d{4})-(\d{4,})\b", re.IGNORECASE)
_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)|"
    r"(?<!\d)((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日?|"
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+((?:19|20)\d{2})\b|"
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9])([vV]?\d+(?:\.\d+){1,3}"
    r"(?:[-+._][A-Za-z0-9][A-Za-z0-9.-]*)?)(?![A-Za-z0-9])"
)
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_NUMERIC_ID_RE = re.compile(r"(?<![\d.])(\d{5,})(?!\d|\.\d)")


@dataclass(frozen=True, slots=True)
class HardLiteral:
    kind: str
    canonical_value: str
    surface_text: str
    start: int
    end: int
    provenance: str = "unknown"

    @property
    def key(self) -> tuple[str, str]:
        return self.kind, self.canonical_value

    def with_provenance(self, provenance: str) -> "HardLiteral":
        return replace(self, provenance=str(provenance or "unknown"))


def _date_value(match: re.Match[str]) -> str:
    groups = match.groups()
    if groups[0]:
        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
    elif groups[3]:
        year, month, day = int(groups[3]), int(groups[4]), int(groups[5])
    elif groups[6]:
        day = int(groups[6])
        month = _MONTHS[groups[7].casefold()]
        year = int(groups[8])
    else:
        month = _MONTHS[groups[9].casefold()]
        day = int(groups[10])
        year = int(groups[11])
    return f"{year:04d}-{month:02d}-{day:02d}"


def _version_value(surface: str) -> str:
    value = str(surface or "").casefold()
    if value.startswith("v"):
        value = value[1:]
    match = re.match(r"(\d+(?:\.\d+){1,3})(.*)", value)
    if not match:
        return value
    numeric, suffix = match.groups()
    normalized_numeric = ".".join(str(int(part)) for part in numeric.split("."))
    return normalized_numeric + suffix


def extract_hard_literals(
    text: str,
    *,
    provenance: str = "unknown",
) -> list[HardLiteral]:
    """Return non-overlapping typed literals in source order.

    More specific identities win over their components: a full date suppresses
    its year, and a CVE suppresses its embedded numeric fragments.
    """

    source = str(text or "")
    rows: list[HardLiteral] = []
    occupied: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < prior_end and end > prior_start for prior_start, prior_end in occupied)

    def add(kind: str, canonical: str, match: re.Match[str]) -> None:
        start, end = match.span()
        if overlaps(start, end):
            return
        rows.append(
            HardLiteral(
                kind=kind,
                canonical_value=canonical,
                surface_text=source[start:end],
                start=start,
                end=end,
                provenance=str(provenance or "unknown"),
            )
        )
        occupied.append((start, end))

    for match in _CVE_RE.finditer(source):
        add("cve", f"CVE-{match.group(1)}-{match.group(2)}", match)
    for match in _DATE_RE.finditer(source):
        add("date", _date_value(match), match)
    for match in _VERSION_RE.finditer(source):
        # Magnitudes such as M5.0 are intentionally not versions.
        if match.start() > 0 and source[match.start() - 1 : match.start()].casefold() == "m":
            continue
        add("version", _version_value(match.group(1)), match)
    for match in _YEAR_RE.finditer(source):
        add("year", match.group(1), match)
    for match in _NUMERIC_ID_RE.finditer(source):
        add("numeric_id", match.group(1), match)
    return sorted(rows, key=lambda row: (row.start, row.end, row.kind))


def hard_literal_keys(texts: Iterable[str]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for text in texts:
        for literal in extract_hard_literals(str(text or "")):
            keys.add(literal.key)
            # A trusted full date also authorizes its literal year component.
            # Without this, exposing 2026-08-14 as runtime state paradoxically
            # rejects a safe "2026" current-year discovery route because the
            # non-overlap extractor correctly suppresses nested identities.
            if literal.kind == "date":
                keys.add(("year", literal.canonical_value[:4]))
    return keys


def untrusted_hard_literals(
    text: str,
    *,
    allowed_keys: set[tuple[str, str]],
    provenance: str = "model",
) -> list[HardLiteral]:
    return [
        literal
        for literal in extract_hard_literals(text, provenance=provenance)
        if literal.key not in allowed_keys
    ]


__all__ = [
    "HardLiteral",
    "extract_hard_literals",
    "hard_literal_keys",
    "untrusted_hard_literals",
]
