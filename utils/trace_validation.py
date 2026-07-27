"""Structural validation for replayable run traces."""

from __future__ import annotations

import json
import re
from typing import Any


REQUIRED_TRACE_FIELDS = (
    "manifest",
    "events",
    "query",
    "search_queries",
    "search_results",
    "navigation_trace",
    "sources",
    "evidence",
    "model_outputs",
    "final_answer",
    "citations",
    "errors",
)


def validate_replay_trace(trace: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not isinstance(trace, dict):
        return {"validator_version": "trace-validator.v1", "valid": False, "issues": ["trace_not_object"]}

    missing = [key for key in REQUIRED_TRACE_FIELDS if key not in trace]
    issues.extend(f"missing:{key}" for key in missing)
    manifest = trace.get("manifest") if isinstance(trace.get("manifest"), dict) else {}
    for key in (
        "schema_version",
        "run_id",
        "status",
        "config",
        "prompt_version",
        "code_revision",
        "config_version",
        "workspace_hash",
    ):
        if not manifest.get(key):
            issues.append(f"manifest_missing:{key}")
    model = ((manifest.get("config") or {}).get("model") or {}) if isinstance(manifest, dict) else {}
    for key in ("model", "endpoint", "context_length"):
        if not model.get(key):
            issues.append(f"model_missing:{key}")

    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    sequences = [event.get("seq") for event in events if isinstance(event, dict)]
    if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
        issues.append("events_not_monotonic")
    if not any(event.get("type") == "user_input" for event in events if isinstance(event, dict)):
        issues.append("missing:user_input_event")
    if not any(event.get("type") in {"final", "error", "provider_error"} for event in events if isinstance(event, dict)):
        issues.append("missing:terminal_event")

    gate_events = [event for event in events if isinstance(event, dict) and event.get("type") == "runtime_gate"]
    if any(event.get("status") == "acquired" for event in gate_events) and not any(
        event.get("status") == "released" for event in gate_events
    ):
        issues.append("runtime_gate_not_released")
    budget_events = [event for event in events if isinstance(event, dict) and event.get("type") == "runtime_budget"]
    if any(event.get("status") == "started" for event in budget_events) and not any(
        event.get("status") in {"completed", "timed_out", "failed"} for event in budget_events
    ):
        issues.append("runtime_budget_not_closed")

    serialized = json.dumps(trace, ensure_ascii=False, default=str)
    if (
        "rwkv-skills" in serialized
        or "Bearer " in serialized
        or re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{16,}", serialized)
    ):
        issues.append("possible_secret_leak")

    return {
        "validator_version": "trace-validator.v1",
        "valid": not issues,
        "issues": issues,
        "event_count": len(events),
        "search_result_count": len(trace.get("search_results") or []) if isinstance(trace.get("search_results"), list) else 0,
        "citation_count": len(trace.get("citations") or []) if isinstance(trace.get("citations"), list) else 0,
        "has_model_output": bool(trace.get("model_outputs")),
        "has_final_answer": bool(str(trace.get("final_answer") or "").strip()),
    }
