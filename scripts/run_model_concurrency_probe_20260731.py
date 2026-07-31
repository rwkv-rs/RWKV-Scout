"""Probe real RWKV task concurrency without running the full benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.run_concurrent_json_suite_20260731 import load_cases, run_dataset


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/evaluation/concurrency_model_probe_20260731.json"
OUTPUT = ROOT / "data/evaluation/concurrency_model_probe_20260731"


def main() -> int:
    cases = [
        {
            "case_id": f"model_probe_{index:02d}",
            "query": "请直接回答：2+2等于多少？只输出答案和一句简短解释，不需要联网。",
            "max_tool_steps": 2,
        }
        for index in range(1, 7)
    ]
    INPUT.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for workers in (2, 3, 4, 6):
        target = OUTPUT / f"workers_{workers}"
        started = time.perf_counter()
        report = run_dataset(f"workers_{workers}", INPUT, target, workers)
        duration = round(time.perf_counter() - started, 1)
        rows = report.get("cases") or []
        summary.append(
            {
                "workers": workers,
                "duration_seconds": duration,
                "status": report.get("status"),
                "completed": sum(row.get("status") == "completed" for row in rows),
                "failed": sum(row.get("status") != "completed" for row in rows),
                "timeouts": sum("timeout" in str(row.get("failure_reason", "")).lower() for row in rows),
                "cases": len(rows),
            }
        )
        (OUTPUT / "progress.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if all(row["failed"] == 0 for row in summary) else 2


if __name__ == "__main__":
    raise SystemExit(main())
