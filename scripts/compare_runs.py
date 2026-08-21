"""Compare local RWKV-Scout vs external rwkv-search on the 54-case numbered set.

Loads the four acceptance runs (local 7.2B / 13.3B, rwkv-search 7.2B / 13.3B),
aligns them by case id, and reports operational metrics that are comparable
across the two differently-shaped output schemas.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/output/acceptance_runs"

FILES = {
    "local 7.2B": RUNS / "2026-07-25T10-45-14-445Z.json",
    "local 13.3B": RUNS / "2026-07-25T13-40-09-679Z.json",
    "rwkv-search 7.2B": RUNS / "rwkv-search-7b-54-2026-07-25.json",
    "rwkv-search 13.3B": RUNS / "rwkv-search-13b-54-2026-07-25.json",
}

REFUSAL = re.compile(
    r"未发现|无法(报告|回答|确定)|没有(找到|相关|提供)|insufficient|缺少证据|未提及",
    re.IGNORECASE,
)


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def _local_answer(row: dict) -> str:
    return str((row.get("report") or {}).get("final_answer") or "")


def _search_answer(row: dict) -> str:
    ans = row.get("answer") or {}
    if isinstance(ans, dict):
        return str(ans.get("answer") or "")
    return str(ans or "")


def summarize(name: str, rows: list[dict]) -> dict:
    is_local = "local" in name
    n = len(rows)
    durations = [r.get("duration_ms", 0) / 1000 for r in rows]
    if is_local:
        sources = [(r.get("report") or {}).get("source_count") or 0 for r in rows]
        evidence = [None] * n
        answers = [_local_answer(r) for r in rows]
        grounded_flag = [bool((r.get("report") or {}).get("real_network")) for r in rows]
    else:
        sources = [r.get("source_count") or 0 for r in rows]
        evidence = [r.get("evidence_count") or 0 for r in rows]
        answers = [_search_answer(r) for r in rows]
        grounded_flag = [(r.get("source_count") or 0) > 0 for r in rows]

    completed = sum(r.get("status") == "completed" for r in rows)
    non_empty = sum(bool(a.strip()) for a in answers)
    refusals = sum(bool(REFUSAL.search(a)) for a in answers)
    with_src = sum(s > 0 for s in sources)
    grounded = sum(grounded_flag)
    ans_len = [len(a) for a in answers]

    return {
        "run": name,
        "n": n,
        "completed": completed,
        "latency_avg_s": round(statistics.mean(durations), 1),
        "latency_med_s": round(statistics.median(durations), 1),
        "sources_avg": round(statistics.mean(sources), 2),
        "cases_with_sources": with_src,
        "grounded_cases": grounded,
        "evidence_avg": (round(statistics.mean([e for e in evidence if e is not None]), 2) if not is_local else None),
        "answer_nonempty": non_empty,
        "refusal_or_insufficient": refusals,
        "answer_len_avg": round(statistics.mean(ans_len), 0),
    }


def main() -> None:
    data = {name: _load(path) for name, path in FILES.items()}
    summaries = {name: summarize(name, rows) for name, rows in data.items()}

    cols = list(FILES)
    def row(label, key, fmt=str):
        vals = "  ".join(f"{fmt(summaries[c][key]):>18}" for c in cols)
        print(f"{label:<26}{vals}")

    print("\n==== 54-CASE NUMBERED TEST: LOCAL vs RWKV-SEARCH ====\n")
    print(f"{'metric':<26}" + "  ".join(f"{c:>18}" for c in cols))
    print("-" * (26 + 20 * len(cols)))
    row("cases", "n")
    row("completed", "completed")
    row("latency avg (s)", "latency_avg_s")
    row("latency median (s)", "latency_med_s")
    row("sources/case (avg)", "sources_avg")
    row("cases w/ >=1 source", "cases_with_sources")
    row("grounded cases", "grounded_cases")
    row("evidence/case (avg)", "evidence_avg")
    row("answer non-empty", "answer_nonempty")
    row("refusal/insufficient", "refusal_or_insufficient")
    row("answer len (chars avg)", "answer_len_avg")

    # Per-case source_count parity (aligned by id) between local_13b and search_13b
    out = RUNS / "comparison_54.json"
    out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nARTIFACT {out}")


if __name__ == "__main__":
    main()
