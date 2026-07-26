"""Gold evaluator with exact body checks from raw local events / external evidence."""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(value: object) -> str:
    text = str(value or "").casefold().replace("：", ":").replace("－", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def answer_text(row: dict) -> str:
    if "report" in row:
        return str((row.get("report") or {}).get("final_answer") or "")
    value = row.get("answer")
    return str(value.get("answer") or value.get("summary") or value.get("final") or "") if isinstance(value, dict) else str(value or "")


def local_body(row: dict) -> bool:
    task_id = row.get("task_id")
    if not task_id:
        return False
    event_path = ROOT / "data/output" / task_id / "events.jsonl"
    if not event_path.exists():
        return False
    for line in event_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = event.get("result") if event.get("type") == "tool_result" else None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            for item in value.get("results") or []:
                if item.get("page_excerpt") or item.get("abstract"):
                    return True
    return False


def external_body(row: dict) -> bool:
    for event in row.get("events") or []:
        if event.get("type") != "evidence":
            continue
        if any(str(item.get("text") or "").strip() for item in event.get("evidence") or []):
            return True
    return False


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def load_gold(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def score(name: str, path: Path, gold: dict[str, dict]) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8")).get("results", [])
    external = any("events" in row for row in rows)
    strict = 0
    forbidden = 0
    body = 0
    durations = []
    for row in rows:
        case = gold.get(row.get("id"), {})
        answer = norm(answer_text(row))
        required = [norm(item) for item in (case.get("gold") or {}).get("required_facts", [])]
        forbidden_facts = [norm(item) for item in (case.get("gold") or {}).get("forbidden_facts", [])]
        if required and all(item in answer for item in required) and not any(item and item in answer for item in forbidden_facts):
            strict += 1
        forbidden += int(any(item and item in answer for item in forbidden_facts))
        body += int(external_body(row) if external else local_body(row))
        durations.append(float(row.get("duration_ms") or 0))
    n = len(rows)
    return {
        "run": name,
        "file": str(path),
        "n": n,
        "completed": sum(row.get("status") == "completed" for row in rows),
        "strict_pass": strict,
        "strict_pass_rate": round(100 * strict / n, 2) if n else 0,
        "answer_nonempty": sum(bool(answer_text(row).strip()) for row in rows),
        "usable_body": body,
        "usable_body_rate": round(100 * body / n, 2) if n else 0,
        "forbidden_fact_hits": forbidden,
        "avg_ms": round(statistics.mean(durations), 1) if durations else 0,
        "p95_ms": round(p95(durations), 1),
        "body_definition": "local page_excerpt/abstract; rwkv-search evidence.text",
    }


def main() -> int:
    gold = load_gold(Path(sys.argv[1]))
    result = {}
    for spec in sys.argv[2:]:
        name, raw_path = spec.split("=", 1)
        result[name] = score(name, Path(raw_path), gold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    output = ROOT / "data/output/acceptance_runs/gold-60-comparison-exact-2026-07-26.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ARTIFACT {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
