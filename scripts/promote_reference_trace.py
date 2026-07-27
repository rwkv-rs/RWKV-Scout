"""Promote a recorded retrieval trace into a human-reviewed eval sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.evaluation_dataset import update_reference_from_trace


def _load_trace(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("trace"), dict):
        return payload["trace"]
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("trace artifact must be a JSON object")


def _parse_citations(values: list[str]) -> list[dict[str, str]]:
    citations = []
    for value in values:
        parts = value.split("|", 2)
        url = parts[0]
        title = parts[1] if len(parts) > 1 else ""
        evidence = parts[2] if len(parts) > 2 else ""
        url = url.strip()
        if not url:
            raise ValueError("citation URL cannot be empty")
        citation = {"url": url, "title": title.strip(), "source": "human_review"}
        if evidence.strip():
            citation["evidence_text"] = evidence.strip()
        citations.append(citation)
    return citations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("question_id")
    parser.add_argument("trace", type=Path, help="normalized trace JSON or an artifact containing a trace field")
    parser.add_argument("--answer", required=True, help="human-reviewed reference answer")
    parser.add_argument("--reviewer-id", required=True, help="human reviewer identifier; model/auto/self are rejected")
    parser.add_argument("--source-checked-at", default="", help="ISO timestamp when the cited sources were checked")
    parser.add_argument("--citation", action="append", default=[], help="URL or URL|title|evidence; repeat for multiple sources")
    parser.add_argument("--key-fact", action="append", default=[], help="reviewed atomic fact used by deterministic scoring; repeatable")
    args = parser.parse_args()
    version = update_reference_from_trace(
        args.dataset,
        args.question_id,
        _load_trace(args.trace),
        reference_answer=args.answer,
        reference_citations=_parse_citations(args.citation),
        key_facts=args.key_fact,
        reviewer_id=args.reviewer_id,
        source_checked_at=args.source_checked_at,
    )
    print(json.dumps({"question_id": args.question_id, "dataset_version": version}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
