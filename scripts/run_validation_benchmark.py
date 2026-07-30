"""Run the two validation architectures over the fixed evaluation suites.

Each acceptance output contains the full event trace.  This queue only adds
batch-level lifecycle metadata so a long run can be resumed and audited.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SUITES = (
    ("100", "data/evaluation/rwkv_search_100_fixed_20260729.json"),
    ("50", "data/evaluation/rwkv_search_50_fixed_20260729.json"),
    ("date", "data/evaluation/date_retrieval_fixed_20260729.json"),
)
ARCHITECTURES = ("engineering_validator", "rwkv_verifier")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(output_dir: Path, manifest_path: Path, *, resume: bool = True) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": "validation_benchmark_queue.v1",
            "status": "queued",
            "started_at": None,
            "finished_at": None,
            "architectures": list(ARCHITECTURES),
            "suites": [name for name, _ in SUITES],
            "batches": [],
        }
    existing = {row.get("batch_id"): row for row in manifest.get("batches") or [] if isinstance(row, dict)}
    batches = []
    for architecture in ARCHITECTURES:
        for suite_name, input_path in SUITES:
            batch_id = f"{architecture}_{suite_name}"
            row = existing.get(batch_id) or {
                "batch_id": batch_id,
                "architecture": architecture,
                "suite": suite_name,
                "input": input_path,
                "output": str(output_dir / f"validation_{architecture}_{suite_name}_20260730.json"),
                "log": str(output_dir / f"validation_{architecture}_{suite_name}_20260730.log"),
                "status": "queued",
            }
            batches.append(row)
    manifest["batches"] = batches
    manifest["status"] = "running"
    manifest["started_at"] = manifest.get("started_at") or _now()
    _write(manifest_path, manifest)

    for row in batches:
        if resume and row.get("status") == "completed" and Path(row["output"]).exists():
            continue
        row["status"] = "running"
        row["started_at"] = _now()
        row["returncode"] = None
        _write(manifest_path, manifest)
        command = [
            sys.executable,
            "scripts/run_json_acceptance.py",
            "--input",
            row["input"],
            "--output",
            row["output"],
            "--validation-architecture",
            row["architecture"],
        ]
        with Path(row["log"]).open("w", encoding="utf-8") as log:
            log.write(f"started_at={row['started_at']}\ncommand={json.dumps(command, ensure_ascii=False)}\n")
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            row["returncode"] = completed.returncode
        row["finished_at"] = _now()
        row["status"] = "completed" if completed.returncode == 0 else "failed"
        if Path(row["output"]).exists():
            try:
                result = json.loads(Path(row["output"]).read_text(encoding="utf-8"))
                row["completed_cases"] = result.get("completed_cases", 0)
                row["total_cases"] = result.get("total_cases", 0)
                row["aggregate_trace_summary"] = result.get("aggregate_trace_summary", {})
            except Exception as exc:
                row["output_parse_error"] = f"{type(exc).__name__}: {exc}"
        _write(manifest_path, manifest)

    manifest["finished_at"] = _now()
    manifest["status"] = "completed" if all(row.get("status") == "completed" for row in batches) else "failed"
    _write(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/evaluation")
    parser.add_argument("--manifest", default="data/evaluation/validation_benchmark_queue_20260730.json")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    manifest = run(Path(args.output_dir), Path(args.manifest), resume=not args.no_resume)
    print(json.dumps({"status": manifest["status"], "manifest": args.manifest}, ensure_ascii=True))


if __name__ == "__main__":
    main()
