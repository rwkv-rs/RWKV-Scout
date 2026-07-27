"""Best-effort page-level events shared by retrieval providers."""

from __future__ import annotations

from typing import Any


def record_retrieval_event(task_id: str | None, event_type: str, **payload: Any) -> None:
    if not task_id:
        return
    try:
        from utils.task_events import append_task_event

        append_task_event(str(task_id), event_type, phase="DISCOVERY", **payload)
    except Exception:
        # A provider must remain usable if observability is unavailable.
        return
