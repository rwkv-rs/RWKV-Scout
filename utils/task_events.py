"""Persistent, user-visible execution events for the local task UI.

The event stream intentionally stores structured inputs, tool calls, outputs and
state summaries. It does not store private chain-of-thought text.
"""

import json
import threading
from datetime import datetime
from typing import Any

from utils.experiment_manifest import (
    ensure_manifest,
    task_directory,
    update_manifest,
)
from utils.file_lock import atomic_file_lease


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _task_lock(task_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(task_id, threading.Lock())


def _events_path(task_id: str, *, create: bool = True):
    task_dir = task_directory(task_id)
    if create:
        task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir / "events.jsonl"


def append_task_event(task_id: str, event_type: str, **payload: Any) -> dict[str, Any]:
    """Append one ordered event and return the persisted record."""

    path = _events_path(task_id)
    lock = _task_lock(task_id)
    with lock:
        ensure_manifest(
            task_id,
            query=str(payload.get("content") or "") if event_type == "user_input" else "",
            metadata=payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else None,
        )
        with atomic_file_lease(path):
            next_seq = 1
            if path.exists():
                try:
                    with path.open("rb") as handle:
                        for line in handle:
                            if line.strip():
                                next_seq += 1
                except OSError:
                    next_seq = 1

            event = {
                "seq": next_seq,
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "type": event_type,
                **payload,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        if event_type == "user_input":
            update_manifest(task_id, query=str(payload.get("content") or ""))
        elif event_type == "run_started":
            update_manifest(
                task_id,
                experiment=payload.get("experiment") or {},
                config=payload.get("config") or {},
                prompt_version=payload.get("prompt_version") or "",
            )
        elif event_type == "final":
            manifest_status = (
                "network_error"
                if str(payload.get("status") or "") == "network_error"
                else "ready"
            )
            update_manifest(
                task_id,
                status=manifest_status,
                summary={
                    "answer_mode": payload.get("mode") or "",
                    "citation_count": len(payload.get("citation_refs") or []),
                    "round_count": payload.get("round_count", 1),
                },
            )
        elif event_type in {"error", "provider_error"}:
            update_manifest(task_id, last_error=str(payload.get("error") or payload.get("message") or "")[:1000])
        return event


def get_task_events(task_id: str, after: int = 0) -> list[dict[str, Any]]:
    path = _events_path(task_id, create=False)
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(event.get("seq", 0)) > after:
                events.append(event)
    return events
