"""Run the real acceptance suites with bounded cross-case concurrency.

Each case keeps the production per-case timeout by running in its own
``run_json_acceptance.py`` process.  Two processes are launched per dataset;
the workspace runtime gate remains the final safety boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_experiment_max_parallel_cases

ROOT = Path(__file__).resolve().parents[1]
UV = Path("/home/chase/.local/bin/uv")
RUNNER = ROOT / "scripts" / "run_json_acceptance.py"


def load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty or invalid case file: {path}")
    normalized = [row for row in rows if isinstance(row, dict) and str(row.get("query") or "").strip()]
    for index, row in enumerate(normalized, start=1):
        row = dict(row)
        row.setdefault("case_id", f"case_{index:03d}")
        normalized[index - 1] = row
    return normalized


def write_part(path: Path, cases: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_part(label: str, part: int, input_path: Path, output_path: Path, log_path: Path) -> int:
    command = [
        str(UV), "run", "--project", str(ROOT), "python", str(RUNNER),
        "--input", str(input_path), "--output", str(output_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"START {label} part={part} {datetime.now().isoformat(timespec='seconds')}\n")
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"DONE {label} part={part} rc={completed.returncode} {datetime.now().isoformat(timespec='seconds')}\n")
    return completed.returncode


def merge_parts(label: str, parts: list[Path], output_path: Path, input_path: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in parts]
    rows = []
    for report in reports:
        rows.extend(report.get("cases") or [])
    order = {str(row.get("case_id")): index for index, row in enumerate(cases)}
    rows.sort(key=lambda row: order.get(str(row.get("case_id")), len(order)))
    try:
        from scripts.run_json_acceptance import _aggregate_trace_summaries

        aggregate = _aggregate_trace_summaries(rows)
    except Exception:
        aggregate = {}
    report = {
        "suite": "manual-real-web-concurrent",
        "status": "completed" if all(item.get("status") == "completed" for item in reports) else "failed",
        "input": str(input_path),
        "started_at": min((item.get("started_at") for item in reports if item.get("started_at")), default=None),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "completed_cases": len(rows),
        "total_cases": len(cases),
        "workers": len(parts),
        "case_timeout_seconds": reports[0].get("case_timeout_seconds") if reports else None,
        "cases": rows,
        "aggregate_trace_summary": aggregate,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_dataset(label: str, input_path: Path, out_dir: Path, workers: int) -> dict[str, Any]:
    cases = load_cases(input_path)
    dataset_dir = out_dir / label
    dataset_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    output_paths: list[Path] = []
    log_paths: list[Path] = []
    for part in range(workers):
        subset = cases[part::workers]
        part_input = dataset_dir / f"part_{part + 1:02d}.json"
        part_output = dataset_dir / f"part_{part + 1:02d}.json.result.json"
        part_log = dataset_dir / f"part_{part + 1:02d}.log"
        write_part(part_input, subset)
        part_paths.append(part_input)
        output_paths.append(part_output)
        log_paths.append(part_log)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_part, label, index + 1, part_paths[index], output_paths[index], log_paths[index]) for index in range(workers)]
        return_codes = [future.result() for future in futures]
    if any(code != 0 for code in return_codes):
        raise RuntimeError(f"{label} part runner failed: {return_codes}")
    return merge_parts(label, output_paths, out_dir / f"{label}_chunk2_concurrent_20260731.json", input_path, cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/evaluation/chunk2_concurrent_20260731")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional benchmark override; omitted means config.json EXPERIMENT.max_parallel_cases.",
    )
    args = parser.parse_args()
    workers = get_experiment_max_parallel_cases() if args.workers is None else args.workers
    if workers < 1:
        raise SystemExit("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        ("rwkv_search_100", ROOT / "data/evaluation/rwkv_search_100_fixed_20260729.json"),
        ("rwkv_search_50", ROOT / "data/evaluation/rwkv_search_50_fixed_20260729.json"),
        ("date_retrieval", ROOT / "data/evaluation/date_retrieval_tasks_20260729.jsonl"),
        ("url_summary_direct", ROOT / "data/evaluation/url_summary_tasks_20260731.json"),
    ]
    queue_log = args.output_dir / "queue.log"
    with queue_log.open("w", encoding="utf-8") as log:
        source = "config.json" if args.workers is None else "--workers override"
        log.write(f"START concurrent suite workers={workers} source={source} {datetime.now().isoformat(timespec='seconds')}\n")
    for label, input_path in datasets:
        report = run_dataset(label, input_path, args.output_dir, workers)
        with queue_log.open("a", encoding="utf-8") as log:
            log.write(f"DONE {label} completed={report['completed_cases']} total={report['total_cases']} status={report['status']} {datetime.now().isoformat(timespec='seconds')}\n")
    with queue_log.open("a", encoding="utf-8") as log:
        log.write(f"ALL_DONE {datetime.now().isoformat(timespec='seconds')}\n")
    print(json.dumps({"output_dir": str(args.output_dir), "status": "completed"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
