"""Probe the configured OpenAI-compatible RWKV endpoint at stepped concurrency."""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import requests

import config


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/evaluation/openai_concurrency_probe_20260731.json"


def one_request(index: int, url: str, model: str, api_key: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "只回答：4"}],
                "max_tokens": 8,
                "temperature": 0,
                "stream": False,
            },
            timeout=(10, timeout),
        )
        elapsed = round(time.perf_counter() - started, 3)
        payload = response.json() if response.content else {}
        return {
            "index": index,
            "status_code": response.status_code,
            "ok": response.ok and isinstance(payload, dict) and bool(payload.get("choices")),
            "latency_seconds": elapsed,
            "error": "" if response.ok else response.text[:500],
        }
    except Exception as exc:
        return {
            "index": index,
            "status_code": None,
            "ok": False,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_level(level: int, url: str, model: str, api_key: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
        futures = [pool.submit(one_request, index, url, model, api_key, timeout) for index in range(level)]
        rows = [future.result() for future in futures]
    latencies = [row["latency_seconds"] for row in rows]
    return {
        "concurrency": level,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "requests": len(rows),
        "success": sum(bool(row["ok"]) for row in rows),
        "failed": sum(not row["ok"] for row in rows),
        "status_codes": sorted({row["status_code"] for row in rows}),
        "min_latency_seconds": min(latencies) if latencies else None,
        "avg_latency_seconds": round(mean(latencies), 3) if latencies else None,
        "max_latency_seconds": max(latencies) if latencies else None,
        "errors": [row["error"] for row in rows if row["error"]][:8],
    }


def main() -> int:
    url = str(config.get_slm_endpoint())
    model = str(config.EXPERIMENT_MODEL_CONTRACT.get("model"))
    api_key = str(config.get_slm_password())
    levels = (1, 2, 4, 8, 16, 32, 64, 128)
    results = []
    for level in levels:
        row = run_level(level, url, model, api_key, timeout=180.0)
        results.append(row)
        OUTPUT.write_text(json.dumps({"endpoint": url, "model": model, "levels": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row["failed"]:
            break
    return 0 if results and results[-1]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
