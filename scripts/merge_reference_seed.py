"""Merge independently checked reference records into a JSONL queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument(
        "--alias-map",
        type=Path,
        help="Optional JSON object mapping a case id to an identical independently checked case id.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed_rows = json.loads(args.seed.read_text(encoding="utf-8"))
    seed_by_id = {str(row["case_id"]): row for row in seed_rows}
    alias_map = json.loads(args.alias_map.read_text(encoding="utf-8")) if args.alias_map else {}
    rows: list[dict[str, Any]] = []
    for line in args.queue.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row.get("case_id"))
        patch = seed_by_id.get(case_id)
        if patch is None:
            source_case_id = alias_map.get(case_id)
            patch = seed_by_id.get(str(source_case_id)) if source_case_id else None
        if patch:
            for key in ("reference_status", "reference_answer", "reference_citations", "key_facts", "fact_groups"):
                if key in patch:
                    row[key] = patch[key]
            row["comparison_status"] = "ready_for_comparison"
        rows.append(row)
    if len(rows) != 172:
        raise SystemExit(f"expected 172 queue rows, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "case_count": len(rows), "seed_count": len(seed_by_id), "alias_count": len(alias_map)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
