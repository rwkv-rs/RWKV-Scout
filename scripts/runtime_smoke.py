"""Exercise the production runtime gate without requiring the RWKV service."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from uuid import uuid4

import config
from utils.experiment_manifest import finalize_manifest, reconstruct_run
from utils.task_events import append_task_event
from utils.task_manager import record_task
from utils.time_budget import TaskTimeoutError, check_time_budget, task_time_budget
from utils.trace_validation import validate_replay_trace
from utils.runtime_gate import analysis_slot


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile + 0.999) - 1))
    return round(ordered[index], 1)


def _run_case(index: int, hold_ms: float, timeout_seconds: float) -> dict:
    task_id = f"runtime-smoke-{uuid4().hex[:12]}-{index:03d}"
    output_dir = str(config.DATA_PIPELINE["output_directory"])
    task_dir = str(Path(output_dir) / task_id)
    record_task(task_id, f"runtime smoke case {index}", "running", task_dir)
    started = time.perf_counter()
    status = "completed"
    error = ""
    try:
        with task_time_budget(task_id, timeout_seconds=timeout_seconds):
            append_task_event(task_id, "user_input", content=f"runtime smoke case {index}")
            with analysis_slot(task_id):
                remaining = max(0.0, hold_ms) / 1000.0
                while remaining > 0:
                    check_time_budget(minimum_seconds=0.001)
                    interval = min(0.01, remaining)
                    time.sleep(interval)
                    remaining -= interval
            append_task_event(task_id, "final", status="completed", content="runtime smoke completed")
    except TaskTimeoutError as exc:
        status = "timed_out"
        error = str(exc)
        append_task_event(task_id, "error", phase="RUNTIME", error=error, error_class="timeout")
        append_task_event(task_id, "final", status=status, content="", error_class="timeout")
    except Exception as exc:  # pragma: no cover - exercised by deployment failures
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        append_task_event(task_id, "error", phase="RUNTIME", error=error, error_class="runtime")
        append_task_event(task_id, "final", status=status, content="", error_class="runtime")
    record_task(task_id, f"runtime smoke case {index}", status, task_dir, error)
    finalize_manifest(task_id, status=status, error=error)
    trace = reconstruct_run(task_id)
    validation = validate_replay_trace(trace)
    gate_events = [event for event in trace["events"] if event.get("type") == "runtime_gate"]
    acquired = next((event for event in gate_events if event.get("status") == "acquired"), {})
    return {
        "task_id": task_id,
        "status": status,
        "error": error,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "wait_ms": acquired.get("wait_ms"),
        "trace_valid": validation.get("valid") is True,
        "trace_issues": validation.get("issues") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hold-ms", type=float, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/output/runtime-smoke.json"))
    args = parser.parse_args()
    if args.tasks < 1 or args.workers < 1:
        raise SystemExit("--tasks and --workers must be positive")
    timeout = args.timeout_seconds or config.get_analysis_timeout_seconds()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_case, index, args.hold_ms, timeout) for index in range(1, args.tasks + 1)]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda item: item["task_id"])
    waits = [float(row["wait_ms"]) for row in rows if row.get("wait_ms") is not None]
    payload = {
        "schema_version": "rwkv-ecra.runtime-smoke.v1",
        "configured_limit": config.get_experiment_max_parallel_cases(),
        "tasks": len(rows),
        "workers": args.workers,
        "hold_ms": args.hold_ms,
        "timeout_seconds": timeout,
        "summary": {
            "completed": sum(row["status"] == "completed" for row in rows),
            "timed_out": sum(row["status"] == "timed_out" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "trace_invalid": sum(not row["trace_valid"] for row in rows),
            "wait_p50_ms": _percentile(waits, 0.50),
            "wait_p95_ms": _percentile(waits, 0.95),
            "average_duration_ms": round(mean(row["duration_ms"] for row in rows), 1) if rows else None,
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["summary"]}, ensure_ascii=False))
    return 0 if not any(payload["summary"][key] for key in ("timed_out", "failed", "trace_invalid")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
