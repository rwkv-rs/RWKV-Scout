"""Small, deterministic text repairs for broken model/API boundaries."""

from __future__ import annotations

import re


_MOJIBAKE_MARKERS = ("Ã", "Â", "â", "æ", "å", "ç", "è", "é", "ï¿½", "�")
_SMART_PUNCTUATION_MOJIBAKE = (
    "\u00e2\u20ac\u2122",
    "\u00e2\u20ac\u02dc",
    "\u00e2\u20ac\u0153",
    "\u00e2\u20ac\u009d",
    "\u00e2\u20ac\u201c",
    "\u00e2\u20ac\u201d",
    "\u00e2\u20ac\u2018",
    "\u00e2\u20ac\u00a6",
)


def _quality(value: str) -> tuple[int, int, int, int]:
    """Lower is better; reward CJK text and penalize common mojibake."""

    replacement_count = value.count("\ufffd") + value.count("�")
    mojibake_count = sum(
        value.count(marker)
        for marker in (*_MOJIBAKE_MARKERS, *_SMART_PUNCTUATION_MOJIBAKE)
    )
    controls = sum(1 for char in value if ord(char) < 32 and char not in "\r\n\t")
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", value))
    return (
        replacement_count * 1000 + mojibake_count * 20 + controls * 10,
        -cjk_count,
        -len(value),
        0,
    )


def repair_mojibake(value: object) -> str:
    """Repair a whole response decoded as cp1252/Latin-1 instead of UTF-8.

    The repair is deliberately conservative: the original text is retained
    unless a reversible cp1252/Latin-1 round-trip has a materially better
    quality score. It is intended for the model HTTP boundary, not for
    semantic rewriting of model output.
    """

    original = str(value or "")
    best = original
    best_score = _quality(original)
    for source_encoding in ("cp1252", "latin-1"):
        try:
            candidate = original.encode(source_encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        candidate_score = _quality(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best
