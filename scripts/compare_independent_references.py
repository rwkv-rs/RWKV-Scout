"""Compare local answers with independently checked reference records.

This report deliberately separates factual coverage from source quality.  A
local answer can contain the right terms while citing an unsuitable page, or
it can refuse even though an authoritative page was available.  Those are
different engineering failures and must not collapse into one score.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


def norm(value: Any) -> str:
    """Normalize prose/code for conservative fact matching."""

    text = str(value or "").casefold()
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", text)


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def dates(value: Any) -> set[tuple[int, int, int]]:
    """Extract dates so English and Chinese/HTML date formatting compare equally."""

    text = str(value or "")
    found: set[tuple[int, int, int]] = set()
    for year, month, day in re.findall(r"(20\d{2})\D{0,8}(\d{1,2})\D{0,8}(\d{1,2})", text):
        found.add((int(year), int(month), int(day)))
    for day, month, year in re.findall(
        r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})",
        text,
        flags=re.IGNORECASE,
    ):
        found.add((int(year), MONTHS[month.casefold()], int(day)))
    for month, day, year in re.findall(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(20\d{2})",
        text,
        flags=re.IGNORECASE,
    ):
        found.add((int(year), MONTHS[month.casefold()], int(day)))
    return found


def name_norm(value: Any) -> str:
    """Ignore middle initials when comparing a person's short name."""

    text = str(value or "").casefold()
    text = re.sub(r"\b[a-z]\b", "", text)
    return norm(text)


def fact_matches(answer: str, variants: list[str]) -> bool:
    answer_norm = norm(answer)
    if any(norm(variant) in answer_norm for variant in variants):
        return True
    expected_dates = {date for variant in variants for date in dates(variant)}
    if expected_dates and expected_dates.intersection(dates(answer)):
        return True
    expected_names = [name_norm(variant) for variant in variants if len(str(variant).split()) >= 2]
    answer_name = name_norm(answer)
    return any(value and value in answer_name for value in expected_names)


def host(value: Any) -> str:
    parsed = urlparse(str(value or ""))
    return (parsed.hostname or "").casefold().removeprefix("www.")


def urls_from_local(local: dict[str, Any]) -> list[str]:
    urls = list(local.get("citation_urls") or [])
    urls.extend(str(item.get("url")) for item in local.get("citation_refs") or [] if item.get("url"))
    return list(dict.fromkeys(urls))


def reference_hosts(reference: dict[str, Any]) -> set[str]:
    return {
        host(item.get("url"))
        for item in reference.get("reference_citations") or []
        if host(item.get("url"))
    }


def local_source_ok(local: dict[str, Any], reference: dict[str, Any]) -> bool:
    expected = reference_hosts(reference)
    return bool(expected.intersection({host(url) for url in urls_from_local(local)}))


def fact_groups(reference: dict[str, Any]) -> list[Any]:
    raw = reference.get("fact_groups")
    if raw:
        result: list[Any] = []
        for group in raw:
            if isinstance(group, list):
                variants = [str(variant) for variant in group if norm(variant)]
                if variants:
                    result.append(variants)
            elif isinstance(group, dict):
                normalized = {
                    key: [str(value) for value in values if norm(value)]
                    for key, values in group.items()
                    if key in {"all", "any"} and isinstance(values, list)
                }
                if any(normalized.values()):
                    result.append(normalized)
        return result
    return [[str(fact)] for fact in reference.get("key_facts", []) if norm(fact)]


def matched_groups(answer: str, groups: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    answer_norm = norm(answer)
    matched: list[Any] = []
    missing: list[Any] = []
    for group in groups:
        if isinstance(group, dict):
            all_values = group.get("all") or []
            any_values = group.get("any") or []
            is_match = (
                all(norm(value) in answer_norm for value in all_values)
                and (not any_values or any(norm(value) in answer_norm for value in any_values))
            )
        else:
            is_match = fact_matches(answer, [str(variant) for variant in group])
        if is_match:
            matched.append(group)
        else:
            missing.append(group)
    return matched, missing


def is_refusal(answer: str) -> bool:
    return bool(
        re.search(
            r"cannot\s+(?:confirm|answer|be\s+confirmed)|"
            r"does\s+not\s+contain|unable\s+to|no\s+(?:usable\s+)?evidence|"
            r"not\s+enough\s+evidence|cannot\s+be\s+answered|"
            r"provided\s+(?:source|evidence)",
            str(answer or ""),
            flags=re.IGNORECASE,
        )
    )


def pollution_flags(answer: str) -> dict[str, bool]:
    text = str(answer or "")
    return {
        "protocol_pollution": bool(
            re.search(r"(?:^|\n)\s*(?:User|Assistant|System):|<think>|BEGIN EXECUTION|\"name\"\s*:\s*\"(?:web_search|fetch_web_url)", text, re.I)
        ),
        "navigation_pollution": bool(
            re.search(r"Skip to content|Log in|Sign up|Copyright|\[首页\]|\[返回\]|javascript:void", text, re.I)
        ),
        "repeated_url_citation": any(count > 2 for count in Counter(re.findall(r"https?://[^)\\s]+", text)).values()),
    }


def classify(local: dict[str, Any], reference: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if str(local.get("status") or "") != "completed":
        return "engineering_failure", {"reason": "local_run_not_completed"}

    answer = str(local.get("final_answer") or "")
    if not answer.strip():
        return "empty_answer", {"reason": "empty_final_answer"}

    groups = fact_groups(reference)
    matched, missing = matched_groups(answer, groups)
    source_ok = local_source_ok(local, reference)
    refusal = is_refusal(answer)
    flags = pollution_flags(answer)
    details = {
        "matched_group_count": len(matched),
        "fact_group_count": len(groups),
        "missing_fact_groups": missing,
        "factual_coverage": (len(matched) / len(groups)) if groups else None,
        "reference_source_hosts": sorted(reference_hosts(reference)),
        "local_source_hosts": sorted({host(url) for url in urls_from_local(local) if host(url)}),
        "source_policy_ok": source_ok,
        "refusal_detected": refusal,
        **flags,
    }

    if groups and len(matched) == len(groups):
        if source_ok:
            return "fully_supported_facts", details
        return "factually_supported_unofficial_source", details
    if matched:
        return "partially_supported_facts", details
    if refusal:
        return "unsupported_refusal", details
    return "fact_mismatch", details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--facts", type=Path, help="Optional case_id -> fact_groups mapping")
    parser.add_argument(
        "--fact-alias-map",
        type=Path,
        help="Optional JSON object mapping a case id to an identical fact-group case id.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    local_by_id = {str(row.get("case_id")): row for row in snapshot.get("cases", [])}
    references = [
        json.loads(line)
        for line in args.references.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fact_overrides = json.loads(args.facts.read_text(encoding="utf-8")) if args.facts else {}
    fact_alias_map = json.loads(args.fact_alias_map.read_text(encoding="utf-8")) if args.fact_alias_map else {}
    rows: list[dict[str, Any]] = []
    for reference in references:
        if reference.get("reference_status") != "independently_checked":
            continue
        local = local_by_id.get(str(reference.get("case_id")), {})
        reference_for_scoring = copy.deepcopy(reference)
        reference_case_id = str(reference.get("case_id"))
        override = fact_overrides.get(reference_case_id)
        if override is None:
            source_case_id = fact_alias_map.get(reference_case_id)
            override = fact_overrides.get(str(source_case_id)) if source_case_id else None
        if override:
            reference_for_scoring["fact_groups"] = override
        classification, diagnostics = classify(local, reference_for_scoring)
        rows.append(
            {
                "case_id": reference.get("case_id"),
                "batch": reference.get("batch"),
                "query": reference.get("query"),
                "classification": classification,
                "diagnostics": diagnostics,
                "local_status": local.get("status", ""),
                "local_answer": local.get("final_answer", ""),
                "local_citations": urls_from_local(local),
                "reference_answer": reference.get("reference_answer", ""),
                "reference_citations": reference.get("reference_citations", []),
            }
        )

    summary = Counter(str(row["classification"]) for row in rows)
    payload = {
        "schema_version": "independent_reference_comparison.v2",
        "scoring_note": "Factual coverage, source policy, refusal and pollution are reported separately; this is not a fully human-verified accuracy score.",
        "reference_case_count": len(rows),
        "queue_case_count": len(references),
        "pending_reference_case_count": len(references) - len(rows),
        "summary": dict(sorted(summary.items())),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reference_case_count": len(rows), "summary": dict(sorted(summary.items()))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
