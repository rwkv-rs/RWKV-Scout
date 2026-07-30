"""Finalize the fixed validation comparison after all six runs finish.

This watcher is intentionally read-only with respect to the running acceptance
runs. It waits for every manifest output to be a complete, parseable batch and
then invokes the existing merge, comparison, and audit scripts exactly once.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("data/evaluation/validation_comparison_manifest_fixed_20260730.json")
DEFAULT_LOG = Path("data/evaluation/validation_finalization_20260730.log")
DEFAULT_OUTPUTS = {
    "merge": Path("data/evaluation/validation_benchmark_full_events_fixed_20260730.json"),
    "compare": Path("data/evaluation/validation_benchmark_comparison_fixed_20260730.json"),
    "audit": Path("data/evaluation/validation_record_audit_fixed_20260730.json"),
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _manifest_outputs(manifest_path: Path) -> list[Path]:
    manifest = _read_json(manifest_path) or {}
    repo_root = manifest_path.resolve().parents[2]
    outputs: list[Path] = []
    for spec in manifest.get("batches") or []:
        if not isinstance(spec, dict):
            continue
        raw = Path(str(spec.get("output") or ""))
        outputs.append(raw if raw.is_absolute() else repo_root / raw)
    return outputs


def _all_batches_complete(outputs: list[Path]) -> tuple[bool, str]:
    if len(outputs) != 6:
        return False, f"manifest outputs={len(outputs)}, expected=6"
    for output in outputs:
        payload = _read_json(output)
        if payload is None:
            return False, f"not_parseable:{output}"
        if payload.get("status") != "completed":
            return False, f"status:{output}={payload.get('status')}"
        total = payload.get("total_cases")
        completed = payload.get("completed_cases")
        cases = payload.get("cases")
        if not isinstance(cases, list) or completed != total or len(cases) != total:
            return False, f"incomplete:{output} cases={len(cases or [])} completed={completed} total={total}"
    return True, "all six batches complete"


def _run(repo_root: Path, manifest: Path, script: str, output: Path, log) -> None:
    command = [
        "/home/chase/.local/bin/uv",
        "run",
        "--project",
        str(repo_root),
        "python",
        f"scripts/{script}",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    ]
    log.write(f"{datetime.now().isoformat(timespec='seconds')} run {' '.join(command)}\n")
    log.flush()
    completed = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    log.write(completed.stdout)
    log.write(completed.stderr)
    log.write(f"exit_code={completed.returncode}\n")
    log.flush()
    if completed.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    log_path = args.log if args.log.is_absolute() else repo_root / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{datetime.now().isoformat(timespec='seconds')} watcher_started\n")
        log.flush()
        while True:
            outputs = _manifest_outputs(manifest)
            ready, reason = _all_batches_complete(outputs)
            log.write(f"{datetime.now().isoformat(timespec='seconds')} ready={ready} reason={reason}\n")
            log.flush()
            if ready:
                try:
                    _run(repo_root, manifest, "merge_validation_events.py", DEFAULT_OUTPUTS["merge"], log)
                    _run(repo_root, manifest, "compare_validation_benchmarks.py", DEFAULT_OUTPUTS["compare"], log)
                    _run(repo_root, manifest, "audit_validation_records.py", DEFAULT_OUTPUTS["audit"], log)
                except Exception as exc:  # pragma: no cover - exercised by a failed external command
                    log.write(f"{datetime.now().isoformat(timespec='seconds')} finalization_failed={exc}\n")
                    return 1
                log.write(f"{datetime.now().isoformat(timespec='seconds')} finalization_complete\n")
                return 0
            time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    sys.exit(main())
