"""Run the external rwkv-search checkout on a JSONL case file.

This is a test-only adapter. The external checkout remains unchanged; the
answerer is pointed at one of the user's local OpenAI-compatible vLLM models.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import requests

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path(os.environ.get("RWKV_SEARCH_ROOT", "/tmp/rwkv-search-baseline"))
CASE_FILE = Path(os.environ["RWKV_SEARCH_CASE_FILE"])
OUTPUT = Path(os.environ["RWKV_SEARCH_OUTPUT"])
DB_PATH = os.environ.get("RWKV_SEARCH_DB", str(ROOT / "data/output/acceptance_runs/rwkv-search-gold.db"))
CONFIG_FILE = Path(os.environ.get("RWKV_SEARCH_CONFIG", str(EXTERNAL_ROOT / "configs/benchmark.json")))
RUN_LABEL = os.environ.get("RWKV_SEARCH_RUN_ID", "rwkv-search-jsonl")

sys.path.insert(0, str(EXTERNAL_ROOT / "src"))

from rwkv_search.config import AppConfig  # noqa: E402
from rwkv_search.db import SearchDatabase  # noqa: E402
from rwkv_search.rwkv_answerer import (  # noqa: E402
    build_rwkv_prompt,
    extract_last_json,
    natural_answer_envelope,
    valid_answer_schema,
)
from rwkv_search.service import SearchService  # noqa: E402


class VLLMAnswerer:
    def __init__(self) -> None:
        self.endpoint = os.environ.get("RWKV_SEARCH_ENDPOINT", "http://127.0.0.1:29572/v1/chat/completions")
        self.api_key = os.environ.get("RWKV_SEARCH_API_KEY", "rwkv-skills")
        self.model = os.environ.get("RWKV_SEARCH_MODEL", "rwkv7-g1h-7.2b-20260710-ctx10240")

    def answer(self, query, route, evidence, *, as_of, timezone, history=None, **_kwargs):
        prompt = build_rwkv_prompt(query, route, evidence, as_of=as_of, timezone=timezone, history=history)
        started = time.perf_counter()
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 640, "temperature": 0.1, "stream": False},
            timeout=(10, 180),
        )
        response.raise_for_status()
        payload = response.json()
        raw = str(((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        answer = extract_last_json(raw)
        if not valid_answer_schema(answer):
            answer = natural_answer_envelope(raw, evidence, as_of=as_of)
        usage = payload.get("usage") or {}
        return SimpleNamespace(answer=answer, raw=raw, latency_ms=round((time.perf_counter() - started) * 1000, 1), new_tokens=int(usage.get("completion_tokens") or 0), repaired=False, error=None if answer else "model output rejected")


def compact_event(event: dict) -> dict:
    result = {"type": event.get("type")}
    for key in ("count", "stats", "route", "answer", "message"):
        if key in event:
            result[key] = event[key]
    if event.get("type") == "sources":
        result["sources"] = [{"title": item.get("title"), "url": item.get("url"), "source_type": item.get("source_type")} for item in (event.get("sources") or [])[:10]]
    if event.get("type") == "evidence":
        result["evidence"] = [{"evidence_id": item.get("evidence_id"), "title": item.get("title"), "url": item.get("url"), "text": str(item.get("text") or "")[:1800]} for item in (event.get("evidence") or [])]
    return result


def run_case(service: SearchService, case: dict) -> dict:
    started = time.perf_counter()
    events: list[dict] = []
    try:
        for event in service.ask_events(case["prompt"], user_timezone="Asia/Shanghai", mode="deep", debug=False):
            events.append(event)
        source_event = next((event for event in events if event.get("type") == "sources"), {})
        evidence_event = next((event for event in events if event.get("type") == "evidence"), {})
        answer_event = next((event for event in events if event.get("type") == "answer"), {})
        return {
            "id": case["id"], "prompt": case["prompt"], "status": "completed",
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "source_count": int(source_event.get("count") or 0),
            "evidence_count": len(evidence_event.get("evidence") or []),
            "answer": answer_event.get("answer") or {},
            "stats": source_event.get("stats") or {},
            "events": [compact_event(event) for event in events],
        }
    except Exception as exc:
        return {"id": case["id"], "prompt": case["prompt"], "status": "failed", "duration_ms": round((time.perf_counter() - started) * 1000, 1), "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    rows = [json.loads(line) for line in CASE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [{"id": row["id"], "prompt": row.get("prompt") or row.get("question")} for row in rows]
    config = AppConfig.load(CONFIG_FILE)
    config.database = DB_PATH
    answerer = VLLMAnswerer()
    service = SearchService(SearchDatabase(config.database), search_config=config.search, answerer=answerer, realtime_config=config.realtime_search)
    results: list[dict] = []
    for index, case in enumerate(cases, start=1):
        row = run_case(service, case)
        results.append(row)
        print(json.dumps({"progress": f"{index}/{len(cases)}", "id": row["id"], "status": row["status"], "duration_ms": row.get("duration_ms"), "source_count": row.get("source_count", 0), "evidence_count": row.get("evidence_count", 0), "error": row.get("error", "")}, ensure_ascii=False), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"run_id": RUN_LABEL, "source_run": str(CASE_FILE), "model": answerer.model, "endpoint": answerer.endpoint.rsplit("/v1", 1)[0] + "/v1", "external_repo": str(EXTERNAL_ROOT), "total": len(results), "completed": sum(row["status"] == "completed" for row in results), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ARTIFACT {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
