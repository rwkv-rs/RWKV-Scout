"""Persistent, user-visible execution events for the local task UI.

The event stream intentionally stores structured inputs, tool calls, outputs and
state summaries. It does not store private chain-of-thought text.
"""

import json
import os
import threading
from datetime import datetime
from typing import Any

from config import DATA_PIPELINE


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _task_lock(task_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(task_id, threading.Lock())


def _events_path(task_id: str) -> str:
    task_dir = os.path.join(DATA_PIPELINE.get("output_directory", "./data/output"), task_id)
    os.makedirs(task_dir, exist_ok=True)
    return os.path.join(task_dir, "events.jsonl")


def append_task_event(task_id: str, event_type: str, **payload: Any) -> dict[str, Any]:
    """Append one ordered event and return the persisted record."""

    path = _events_path(task_id)
    lock = _task_lock(task_id)
    with lock:
        next_seq = 1
        if os.path.exists(path):
            try:
                with open(path, "rb") as handle:
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
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return event


def get_task_events(task_id: str, after: int = 0) -> list[dict[str, Any]]:
    path = _events_path(task_id)
    if not os.path.exists(path):
        return []

    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
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
