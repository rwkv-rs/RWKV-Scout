"""Best-effort model-call events for per-run engineering metrics."""

from __future__ import annotations

import re
from typing import Any


_HIDDEN_THOUGHT = re.compile(r"<think>[\s\S]*?</think>", flags=re.IGNORECASE)
_SENSITIVE_KEYS = {"api_key", "authorization", "password", "token"}


def visible_model_text(value: Any) -> str:
    """Keep the visible response while excluding hidden reasoning blocks."""
    text = str(value or "")
    return _HIDDEN_THOUGHT.sub("", text).replace("</think>", "").strip()


def _safe_payload(value: Any, key: str = "") -> Any:
    if key.casefold() in _SENSITIVE_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {str(name): _safe_payload(item, str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_payload(item, key) for item in value]
    if key.casefold() in {"prompt", "output", "model_output", "content", "error"}:
        return visible_model_text(value)
    return value


def record_model_event(task_id: str | None, *, status: str, **payload: Any) -> None:
    if not task_id or task_id == "UNKNOWN_TASK":
        return
    try:
        from utils.task_events import append_task_event

        safe_payload = _safe_payload(payload)
        append_task_event(str(task_id), "model_call", phase="MODEL", status=status, **safe_payload)
    except Exception:
        # Model observability must never turn a successful provider response
        # into an application failure.
        return
