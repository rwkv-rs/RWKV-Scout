"""Validate one persisted run trace and return a CI-friendly exit status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.experiment_manifest import reconstruct_run
from utils.trace_validation import validate_replay_trace


def _load_trace(
    value: str,
    output_directory: Path | None,
    *,
    result_index: int = 0,
    question_id: str = "",
) -> dict[str, Any]:
    path = Path(value)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("trace"), dict):
            return payload["trace"]
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            rows = [row for row in payload["results"] if isinstance(row, dict) and isinstance(row.get("trace"), dict)]
            if question_id:
                rows = [row for row in rows if (row.get("case") or {}).get("question_id") == question_id]
            if not rows:
                raise ValueError("experiment artifact contains no matching normalized trace")
            if result_index < 0 or result_index >= len(rows):
                raise ValueError(f"result index {result_index} is outside artifact trace count {len(rows)}")
            return rows[result_index]["trace"]
        if isinstance(payload, dict):
            return payload
        raise ValueError("trace JSON must be an object")
    if path.is_dir():
        return reconstruct_run(path.name, path.parent)
    return reconstruct_run(value, output_directory)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="task id, run directory, normalized trace JSON, or experiment artifact")
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--result-index", type=int, default=0, help="sample index when validating an experiment artifact")
    parser.add_argument("--question-id", default="", help="sample question_id when validating an experiment artifact")
    args = parser.parse_args()
    try:
        trace = _load_trace(
            args.trace,
            args.output_directory,
            result_index=args.result_index,
            question_id=args.question_id,
        )
        result = validate_replay_trace(trace)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "validator_version": "trace-validator.v1",
            "valid": False,
            "issues": [f"load_error:{type(exc).__name__}:{exc}"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
