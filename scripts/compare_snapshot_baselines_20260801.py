"""Compare current 172 snapshot slices with completed historical baselines."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/full_172_rollback_c4_threshold_evidence80_20260801"
sys.path.insert(0, str(ROOT / "scripts"))
from compare_independent_references import classify  # noqa: E402


def urls(answer: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"https?://[^)\s]+", answer or "")))


def old_rows(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, object]] = {}
    for row in data.get("cases", []):
        answer = str(row.get("final_output") or row.get("answer") or "")
        result[str(row.get("case_id"))] = {
            "case_id": row.get("case_id"),
            "status": row.get("status") or row.get("final_status") or "",
            "final_answer": answer,
            "citation_urls": urls(answer),
            "citation_refs": [],
        }
    return result


def current_rows() -> dict[str, dict[str, object]]:
    data = json.loads((RUN / "answers_for_comparison.json").read_text(encoding="utf-8"))
    return {str(row.get("case_id")): row for row in data.get("cases", [])}


def references() -> dict[str, dict[str, object]]:
    result = {}
    for line in (RUN / "independent_reference_queue_seeded.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("reference_status") == "independently_checked":
            result[str(row["case_id"])] = row
    return result


def score(rows: dict[str, dict[str, object]], refs: dict[str, dict[str, object]], ids: list[str]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    coverage: list[float] = []
    source_ok = 0
    for case_id in ids:
        ref = refs[case_id]
        local = rows.get(case_id, {"case_id": case_id, "status": "missing", "final_answer": ""})
        classification, diagnostics = classify(local, ref)
        counts[classification] += 1
        if diagnostics.get("factual_coverage") is not None:
            coverage.append(float(diagnostics["factual_coverage"]))
        source_ok += int(bool(diagnostics.get("source_policy_ok")))
    return {
        "count": len(ids),
        "classification": dict(sorted(counts.items())),
        "mean_factual_coverage": (sum(coverage) / len(coverage)) if coverage else None,
        "full_factual_coverage": sum(value == 1.0 for value in coverage),
        "source_policy_ok": source_ok,
    }


def main() -> int:
    refs = references()
    current = current_rows()
    old_search = old_rows(ROOT / "data/evaluation/rwkv_search_20260729_50_global_results_v2.json")
    old_url = old_rows(ROOT / "data/evaluation/url_summary_results_direct_20260731.json")
    search_ids = [f"rwkv_search_{i:03d}" for i in range(1, 51)]
    url_ids = [f"url_{i:03d}" for i in range(1, 11)]
    payload = {
        "schema_version": "baseline_comparison.v1",
        "current_source": str(RUN / "answers_for_comparison.json"),
        "historical_sources": {
            "rwkv_search_50": "data/evaluation/rwkv_search_20260729_50_global_results_v2.json",
            "url_summary_direct": "data/evaluation/url_summary_results_direct_20260731.json",
        },
        "rwkv_search_50": {"current": score(current, refs, search_ids), "historical": score(old_search, refs, search_ids)},
        "url_summary_direct": {"current": score(current, refs, url_ids), "historical": score(old_url, refs, url_ids)},
    }
    out = RUN / "baseline_comparison_20260801.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
