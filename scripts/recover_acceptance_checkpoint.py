"""Recover a sequential JSON acceptance run from its last checkpoint.

The acceptance runner checkpoints after every case, but intentionally does not
resume a partially written case file by itself.  This utility is for a stopped
run: it verifies that the checkpoint is a prefix of the original input, runs
only the remaining cases, and atomically replaces the checkpoint with the
combined completed report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_json_acceptance import _aggregate_trace_summaries, _load_cases


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def recover(
    input_path: Path,
    partial_output: Path,
    *,
    recovery_output: Path,
    backup_output: Path,
) -> dict[str, object]:
    cases = _load_cases(input_path)
    partial = json.loads(partial_output.read_text(encoding="utf-8"))
    rows = partial.get("cases") if isinstance(partial, dict) else None
    if not isinstance(rows, list):
        raise ValueError("checkpoint does not contain a cases list")
    if len(rows) > len(cases):
        raise ValueError("checkpoint contains more rows than the input")

    # The runner is sequential.  Verify the saved rows are the exact input
    # prefix before trusting row count as the resume offset.
    for index, row in enumerate(rows):
        expected = str(cases[index].get("query") or "")
        actual = str((row or {}).get("query") or "")
        if expected != actual:
            raise ValueError(f"checkpoint diverges from input at index {index + 1}")

    remaining = cases[len(rows) :]
    if not remaining:
        return {"status": "already_complete", "completed_cases": len(rows), "total_cases": len(cases)}

    subset_path = recovery_output.with_name(recovery_output.stem + "_input.json")
    _write_json(subset_path, {"cases": remaining})
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_json_acceptance.py"),
        "--input",
        str(subset_path),
        "--output",
        str(recovery_output),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"recovery runner failed with return code {completed.returncode}")

    recovery = json.loads(recovery_output.read_text(encoding="utf-8"))
    recovery_rows = recovery.get("cases") if isinstance(recovery, dict) else None
    if recovery.get("status") != "completed" or not isinstance(recovery_rows, list):
        raise RuntimeError("recovery output is not a completed acceptance report")
    if len(recovery_rows) != len(remaining):
        raise RuntimeError("recovery output does not contain every remaining case")

    # Preserve a recoverable copy of the partial checkpoint before replacement.
    shutil.copy2(partial_output, backup_output)
    combined_rows = rows + recovery_rows
    combined = dict(partial)
    combined.update(
        {
            "status": "completed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "completed_cases": len(combined_rows),
            "total_cases": len(cases),
            "cases": combined_rows,
            "aggregate_trace_summary": _aggregate_trace_summaries(combined_rows),
            "recovered_from_checkpoint": True,
            "recovery_input": str(subset_path),
        }
    )
    temporary = partial_output.with_name(partial_output.name + ".recovered.tmp")
    _write_json(temporary, combined)
    temporary.replace(partial_output)
    return {
        "status": "completed",
        "completed_cases": len(combined_rows),
        "total_cases": len(cases),
        "backup": str(backup_output),
        "recovery_output": str(recovery_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--partial-output", required=True)
    parser.add_argument("--recovery-output", required=True)
    parser.add_argument("--backup-output", required=True)
    args = parser.parse_args()
    result = recover(
        Path(args.input),
        Path(args.partial_output),
        recovery_output=Path(args.recovery_output),
        backup_output=Path(args.backup_output),
    )
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
