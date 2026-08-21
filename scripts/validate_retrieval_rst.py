"""Validate retrieval-RST bundles and export only accepted training records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.retrieval_training_factory import (
    export_training_records,
    select_diverse_tasks,
    validate_pool,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATIONS = (
    ROOT / "data/evaluation/retrieval_required_100_20260808.json",
    ROOT / "data/evaluation/rwkv_search_100_fixed_20260729.json",
    ROOT / "data/evaluation/interference_40_query_only_20260808.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=ROOT / "training/retrieval_rst/pilot_v1/tasks",
        help="Task-bundle root.",
    )
    parser.add_argument(
        "--evaluation",
        action="append",
        type=Path,
        default=None,
        help="Held-out evaluation JSON/JSONL. Repeat for multiple files.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "training/retrieval_rst/pilot_v1/validation",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=ROOT / "training/retrieval_rst/pilot_v1/exports",
    )
    parser.add_argument("--selection-limit", type=int, default=10_000)
    args = parser.parse_args()

    evaluation_paths = tuple(args.evaluation or DEFAULT_EVALUATIONS)
    manifest = validate_pool(
        args.root.resolve(),
        evaluation_paths=evaluation_paths,
        report_dir=args.report_dir.resolve(),
    )
    selected = select_diverse_tasks(
        manifest,
        limit=max(0, args.selection_limit),
    )
    selection = {
        "schema_version": "rwkv-retrieval-rst-selection.v1",
        "requested_limit": max(0, args.selection_limit),
        "selected_count": len(selected),
        "tasks": selected,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "selected_pool.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    export_manifest: dict[str, object] | None = None
    if manifest["rejected_count"] == 0:
        selected_ids = {row["task_id"] for row in selected}
        export_source = {
            **manifest,
            "tasks": [
                row
                for row in manifest["tasks"]
                if row.get("task_id") in selected_ids
            ],
        }
        export_manifest = export_training_records(
            export_source,
            args.export_dir.resolve(),
        )

    summary = {
        "root": str(args.root.resolve()),
        "evaluation_files": [str(path.resolve()) for path in evaluation_paths],
        "task_count": manifest["task_count"],
        "accepted_count": manifest["accepted_count"],
        "rejected_count": manifest["rejected_count"],
        "selected_count": len(selected),
        "export": export_manifest,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if manifest["rejected_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
