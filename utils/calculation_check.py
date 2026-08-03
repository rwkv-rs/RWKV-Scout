"""Small, deterministic arithmetic checks for evidence-grounded synthesis."""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable


_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?(?![A-Za-z])")
_PERCENT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")


def _number(value: str) -> Decimal:
    return Decimal(str(value).replace(",", "").strip())


def _numbers_in(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for match in _NUMBER_RE.finditer(str(text or "")):
        raw = match.group(0)
        try:
            value = _number(raw)
        except Exception:
            continue
        # A four-digit year or a version component is not an arithmetic
        # operand for the percentage/subtraction forms handled here.
        if value >= 1900 and value <= 2200:
            continue
        if value not in values:
            values.append(value)
    return values


def _body_contains_number(body: str, value: Decimal) -> bool:
    target = format(value, "f").rstrip("0").rstrip(".")
    for match in _NUMBER_RE.finditer(str(body or "")):
        try:
            if _number(match.group(0)) == value:
                return True
        except Exception:
            continue
    return bool(target and target in str(body or ""))


def _rounded(value: Decimal, places: int = 1) -> str:
    quantum = Decimal("1").scaleb(-places)
    return format(value.quantize(quantum, rounding=ROUND_HALF_UP), "f")


def build_calculation_check(query: str, evidence: Iterable[dict[str, Any]] | str = "") -> str:
    """Return a bounded arithmetic cue, never an answer or a source claim.

    The cue is emitted only when the operands occur in the user question and
    the same numeric values are visible in fetched page bodies.  It therefore
    cannot manufacture missing evidence or decide which source is correct.
    """

    text = str(query or "")
    body_parts: list[str] = []
    if isinstance(evidence, str):
        body_parts.append(evidence)
    else:
        for item in evidence:
            if not isinstance(item, dict):
                continue
            body_parts.append(str(item.get("evidence_text") or item.get("content") or ""))
            for chunk in item.get("chunks") or []:
                if isinstance(chunk, dict):
                    body_parts.append(str(chunk.get("text") or ""))
    body = "\n".join(body_parts)
    lowered = text.casefold()
    numbers = _numbers_in(text)
    if len(numbers) < 2 or not all(_body_contains_number(body, value) for value in numbers[:2]):
        return ""

    first, second = numbers[0], numbers[1]
    if any(marker in lowered for marker in ("百分比", "百分之", "percentage", "percent")):
        if second == 0:
            return ""
        result = _rounded(first / second * Decimal("100"), 1)
        return (
            "ARITHMETIC CHECK (operands still require matching evidence): "
            f"{first:g} / {second:g} × 100 = {result}%; use the requested rounding rule.\n"
        )
    if any(marker in lowered for marker in ("相差多少", "多多少", "difference", "how many")):
        result = abs(first - second)
        return (
            "ARITHMETIC CHECK (operands still require matching evidence): "
            f"|{first:g} - {second:g}| = {result:g}; preserve the question's requested unit.\n"
        )
    return ""
