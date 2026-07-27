"""Probe one fetched page through the chunk-parallel candidate pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.page_evidence import build_chunk_candidate_prompt, build_page_chunks, extract_single_page_evidence
from clients.llm_client import LLMClient
from tools.builtin import load_builtin_tools
from tools.registry import ToolRegistry


def run(input_path: Path, output_path: Path) -> dict:
    config = json.loads(input_path.read_text(encoding="utf-8"))
    query = str(config.get("query") or "").strip()
    url = str(config.get("url") or "").strip()
    if not query or not url:
        raise ValueError("input JSON must contain query and url")

    load_builtin_tools()
    fetch_raw = ToolRegistry.execute(
        "fetch_web_url",
        {
            "url": url,
            "max_chars": int(config.get("max_chars") or 14000),
        },
        {"agentic_tool_loop": True},
        phase="EXTRACTION",
    )
    fetched = json.loads(fetch_raw) if isinstance(fetch_raw, str) else fetch_raw
    pages = [item for item in fetched.get("results") or [] if isinstance(item, dict)]
    report: dict = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "url": url,
        "fetch": {
            "schema_version": fetched.get("schema_version", ""),
            "status": fetched.get("status", ""),
            "provider": fetched.get("provider", ""),
            "count": fetched.get("count", 0),
            "provider_errors": fetched.get("provider_errors") or [],
            "page_chars": len(str((pages[0] if pages else {}).get("page_excerpt") or "")),
        },
    }
    if not pages:
        report["evidence"] = {"status": "no_page", "candidates": []}
    else:
        evidence = extract_single_page_evidence(
            query=query,
            page=pages[0],
            llm=LLMClient(),
            max_chunk_tokens=int(config.get("max_chunk_tokens") or 800),
        )
        page_text = str(pages[0].get("page_excerpt") or pages[0].get("content") or "")
        chunks = build_page_chunks(
            page_text,
            max_tokens=int(config.get("max_chunk_tokens") or 800),
        )
        evidence["chunks_detail"] = [
            {
                "chunk_id": chunk["chunk_id"],
                "index": chunk["index"],
                "token_count": chunk["token_count"],
                "text": chunk["text"],
                "prompt": build_chunk_candidate_prompt(
                    query,
                    url,
                    str(pages[0].get("title") or url),
                    chunk,
                    len(chunks),
                ),
            }
            for chunk in chunks
        ]
        report["evidence"] = evidence
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/evaluation/page_evidence_probe.json")
    parser.add_argument("--output", default="data/evaluation/page_evidence_probe_result.json")
    args = parser.parse_args()
    report = run(Path(args.input), Path(args.output))
    evidence = report.get("evidence") or {}
    parallel = evidence.get("parallel_candidate") or {}
    print(json.dumps({
        "output": args.output,
        "fetch_status": (report.get("fetch") or {}).get("status"),
        "chunk_count": evidence.get("chunk_count", 0),
        "candidate_count": len(evidence.get("candidates") or []),
        "worker_count": parallel.get("worker_count", 0),
        "completed_calls": parallel.get("completed_calls", 0),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
