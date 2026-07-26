"""Compare numbered acceptance runs with one operational schema."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def score(name: str, path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8")).get("results", [])
    local = any("report" in row for row in rows)
    durations = [float(row.get("duration_ms") or 0) for row in rows]
    answers = []
    source_counts = []
    evidence_counts = []
    tool_calls = []
    real_network = 0
    for row in rows:
        if local:
            report = row.get("report") or {}
            answers.append(str(report.get("final_answer") or ""))
            source_counts.append(int(report.get("source_count") or 0))
            evidence_counts.append(0)
            tool_calls.append(int(report.get("tool_call_count") or 0))
            real_network += int(bool(report.get("real_network")))
        else:
            answer = row.get("answer")
            answers.append(str(answer.get("answer") if isinstance(answer, dict) else answer or ""))
            source_counts.append(int(row.get("source_count") or 0))
            evidence_counts.append(int(row.get("evidence_count") or 0))
            tool_calls.append(0)
            real_network += int(bool(row.get("source_count") or row.get("evidence_count")))
    return {
        "run": name,
        "file": str(path),
        "n": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "answer_nonempty": sum(bool(text.strip()) for text in answers),
        "real_network_or_source": real_network,
        "cases_with_sources": sum(value > 0 for value in source_counts),
        "cases_with_evidence": sum(value > 0 for value in evidence_counts),
        "sources_avg": round(statistics.mean(source_counts), 2) if source_counts else 0,
        "evidence_avg": round(statistics.mean(evidence_counts), 2) if evidence_counts else 0,
        "tool_call_cases": sum(value > 0 for value in tool_calls),
        "latency_avg_ms": round(statistics.mean(durations), 1) if durations else 0,
        "latency_p95_ms": round(p95(durations), 1),
    }


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python scripts/compare_acceptance_runs.py <name=run.json> ...")
    report = {}
    for spec in sys.argv[1:]:
        name, raw_path = spec.split("=", 1)
        report[name] = score(name, Path(raw_path))
    output = Path("data/output/acceptance_runs/comparison_54-2026-07-26.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"ARTIFACT {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
