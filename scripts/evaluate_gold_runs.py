"""Score local and rwkv-search JSONL runs against the copied gold file."""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path


def norm(value: object) -> str:
    text = str(value or "").casefold()
    text = text.replace("：", ":").replace("－", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def answer_text(row: dict) -> str:
    if "report" in row:
        return str((row.get("report") or {}).get("final_answer") or "")
    answer = row.get("answer")
    if isinstance(answer, dict):
        return str(answer.get("answer") or answer.get("summary") or answer.get("final") or "")
    return str(answer or "")


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = max(0, math.ceil(0.95 * len(values)) - 1)
    return values[index]


def load_gold(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def score(name: str, path: Path, gold: dict[str, dict]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    passed = 0
    nonempty = 0
    body = 0
    forbidden = 0
    category: dict[str, dict[str, int]] = {}
    durations = []
    for row in rows:
        case = gold.get(row.get("id"), {})
        answer = norm(answer_text(row))
        required = [norm(item) for item in (case.get("gold") or {}).get("required_facts", [])]
        forbidden_facts = [norm(item) for item in (case.get("gold") or {}).get("forbidden_facts", [])]
        has_required = bool(required) and all(item in answer for item in required)
        has_forbidden = any(item and item in answer for item in forbidden_facts)
        strict = has_required and not has_forbidden
        if strict:
            passed += 1
        if answer:
            nonempty += 1
        local_report = row.get("report") or {}
        has_body = bool(local_report.get("source_count") or row.get("evidence_count") or local_report.get("real_network") or row.get("source_count"))
        if has_body:
            body += 1
        if has_forbidden:
            forbidden += 1
        durations.append(float(row.get("duration_ms") or 0))
        bucket = category.setdefault(case.get("category", "unknown"), {"n": 0, "strict": 0, "body": 0})
        bucket["n"] += 1
        bucket["strict"] += int(strict)
        bucket["body"] += int(has_body)
    return {
        "run": name,
        "file": str(path),
        "n": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "strict_pass": passed,
        "strict_pass_rate": round(100 * passed / len(rows), 2) if rows else 0,
        "answer_nonempty": nonempty,
        "usable_body": body,
        "usable_body_rate": round(100 * body / len(rows), 2) if rows else 0,
        "forbidden_fact_hits": forbidden,
        "avg_ms": round(statistics.mean(durations), 1) if durations else 0,
        "p95_ms": round(p95(durations), 1),
        "by_category": category,
    }


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python scripts/evaluate_gold_runs.py <gold.jsonl> <name=run.json> ...")
    gold = load_gold(Path(sys.argv[1]))
    report = {}
    for spec in sys.argv[2:]:
        name, raw_path = spec.split("=", 1)
        report[name] = score(name, Path(raw_path), gold)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
