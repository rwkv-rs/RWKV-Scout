"""Read-only operational metrics derived from persisted task traces.

The API should expose health and metrics without depending on process-local
state.  This module therefore reads the task index and replay manifests, and
keeps the output deliberately aggregate: queries, prompts, URLs and secrets
are never included in the metrics response.
"""

from __future__ import annotations

import math
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config import DATA_PIPELINE
from utils.experiment_manifest import reconstruct_run
from utils.task_manager import get_all_tasks
from utils.trace_validation import validate_replay_trace


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return round(ordered[index], 1)


def _duration_ms(trace: dict[str, Any]) -> float | None:
    manifest = trace.get("manifest") or {}
    value = manifest.get("duration_ms")
    if isinstance(value, (int, float)):
        return float(value)
    timestamps: list[datetime] = []
    for event in trace.get("events") or []:
        try:
            parsed = datetime.fromisoformat(str(event.get("timestamp", "")).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        timestamps.append(parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc))
    if len(timestamps) < 2:
        return None
    return max(0.0, (max(timestamps) - min(timestamps)).total_seconds() * 1000)


def _iter_task_traces(tasks: Iterable[dict[str, Any]], output_directory: str | Path | None) -> Iterable[dict[str, Any]]:
    for task in tasks:
        task_id = str(task.get("task_id") or task.get("id") or "").strip()
        if not task_id:
            continue
        try:
            yield reconstruct_run(task_id, output_directory)
        except (OSError, ValueError, json.JSONDecodeError):
            # A partially deleted task should be visible as a counter, but
            # must not make the monitoring endpoint unavailable.
            yield {"manifest": {"status": "unreadable"}, "events": []}


def collect_operational_metrics(
    tasks: Iterable[dict[str, Any]] | None = None,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Return aggregate metrics suitable for JSON or Prometheus adapters."""
    task_rows = list(tasks if tasks is not None else get_all_tasks())
    traces = list(_iter_task_traces(task_rows, output_directory or DATA_PIPELINE.get("output_directory")))
    statuses = Counter(str((trace.get("manifest") or {}).get("status") or "unknown") for trace in traces)
    durations = [duration for trace in traces if (duration := _duration_ms(trace)) is not None]
    model_calls = 0
    failed_model_calls = 0
    retry_count = 0
    error_count = 0
    trace_valid = 0
    trace_invalid = 0
    gate_waits: list[float] = []
    for trace in traces:
        events = trace.get("events") or []
        model_events = [event for event in events if event.get("type") == "model_call"]
        model_calls += len(model_events)
        failed_model_calls += sum(event.get("status") == "failed" for event in model_events)
        retry_count += sum(event.get("type") == "retry" for event in events)
        error_count += sum(event.get("type") in {"error", "provider_error"} for event in events)
        gate_waits.extend(
            float(event["wait_ms"])
            for event in events
            if event.get("type") == "runtime_gate"
            and event.get("status") == "acquired"
            and isinstance(event.get("wait_ms"), (int, float))
        )
        validation = validate_replay_trace(trace)
        if validation.get("valid") is True:
            trace_valid += 1
        else:
            trace_invalid += 1
    return {
        "schema_version": "rwkv-ecra.operational-metrics.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": {
            "total": len(traces),
            "active": sum(statuses.get(status, 0) for status in ("running", "queued")),
            "by_status": dict(sorted(statuses.items())),
        },
        "latency_ms": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "p99": _percentile(durations, 0.99),
            "sample_count": len(durations),
        },
        "model": {
            "call_count": model_calls,
            "failed_call_count": failed_model_calls,
            "failure_rate": round(failed_model_calls / model_calls, 4) if model_calls else None,
        },
        "reliability": {
            "error_count": error_count,
            "retry_count": retry_count,
            "timeout_count": statuses.get("timed_out", 0),
            "first_success_count": statuses.get("completed", 0),
        },
        "concurrency": {
            "wait_p50_ms": _percentile(gate_waits, 0.50),
            "wait_p95_ms": _percentile(gate_waits, 0.95),
            "sample_count": len(gate_waits),
        },
        "trace_integrity": {
            "valid_count": trace_valid,
            "invalid_count": trace_invalid,
            "valid_rate": round(trace_valid / len(traces), 4) if traces else None,
        },
    }


def prometheus_text(metrics: dict[str, Any]) -> str:
    """Render the aggregate schema as stable Prometheus text metrics."""
    tasks = metrics.get("tasks") or {}
    by_status = tasks.get("by_status") or {}
    lines = [
        "# HELP rwkv_ecra_tasks_total Persisted task traces by terminal status.",
        "# TYPE rwkv_ecra_tasks_total gauge",
    ]
    for status, count in sorted(by_status.items()):
        safe_status = str(status).replace('"', "")
        lines.append(f'rwkv_ecra_tasks_total{{status="{safe_status}"}} {int(count)}')
    scalar_map = {
        "rwkv_ecra_latency_ms_p50": (metrics.get("latency_ms") or {}).get("p50"),
        "rwkv_ecra_latency_ms_p95": (metrics.get("latency_ms") or {}).get("p95"),
        "rwkv_ecra_latency_ms_p99": (metrics.get("latency_ms") or {}).get("p99"),
        "rwkv_ecra_model_calls_total": (metrics.get("model") or {}).get("call_count"),
        "rwkv_ecra_model_failures_total": (metrics.get("model") or {}).get("failed_call_count"),
        "rwkv_ecra_errors_total": (metrics.get("reliability") or {}).get("error_count"),
        "rwkv_ecra_retries_total": (metrics.get("reliability") or {}).get("retry_count"),
        "rwkv_ecra_trace_invalid_total": (metrics.get("trace_integrity") or {}).get("invalid_count"),
    }
    for name, value in scalar_map.items():
        if value is not None:
            lines.append(f"{name} {value}")
    return "\n".join(lines) + "\n"
