"""Repeat the highest successful endpoint concurrency to check stability."""

from __future__ import annotations

import json
import time
from pathlib import Path

import config
from scripts.probe_openai_concurrency_20260731 import run_level


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/evaluation/openai_concurrency_probe_repeat_20260731.json"


def main() -> int:
    rows = []
    for round_number in range(1, 6):
        result = run_level(
            128,
            str(config.get_slm_endpoint()),
            str(config.EXPERIMENT_MODEL_CONTRACT.get("model")),
            str(config.get_slm_password()),
            timeout=180.0,
        )
        result["round"] = round_number
        rows.append(result)
        OUTPUT.write_text(json.dumps({"concurrency": 128, "rounds": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if all(row["failed"] == 0 for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
