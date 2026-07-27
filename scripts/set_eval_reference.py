"""Update one evaluation case with a human-reviewed answer and citations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.evaluation_dataset import update_reference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("question_id")
    parser.add_argument("--answer", required=True)
    parser.add_argument("--reviewer-id", required=True, help="human reviewer identifier; model/auto/self are rejected")
    parser.add_argument("--source-checked-at", default="", help="ISO timestamp when the cited sources were checked")
    parser.add_argument("--citation", action="append", default=[], help="URL or URL|title|evidence; repeat for multiple sources")
    parser.add_argument("--key-fact", action="append", default=[], help="reviewed atomic fact used by deterministic scoring; repeatable")
    args = parser.parse_args()
    citations = []
    for value in args.citation:
        parts = [part.strip() for part in value.split("|", 2)]
        citations.append({
            "url": parts[0],
            "title": parts[1] if len(parts) > 1 else "",
            "evidence_text": parts[2] if len(parts) > 2 else "",
            "source": "human_review",
        })
    version = update_reference(
        args.dataset,
        args.question_id,
        reference_answer=args.answer,
        reference_citations=citations,
        key_facts=args.key_fact,
        reviewer_id=args.reviewer_id,
        source_checked_at=args.source_checked_at,
    )
    print(json.dumps({"question_id": args.question_id, "dataset_version": version}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
