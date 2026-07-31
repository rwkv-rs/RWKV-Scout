"""Cooperative wall-clock budgets for long-running analysis tasks."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from config import get_analysis_timeout_seconds


class TaskTimeoutError(TimeoutError):
    """Raised when a task has exhausted its configured wall-clock budget."""


_deadline: ContextVar[float | None] = ContextVar("rwkv_ecra_task_deadline", default=None)
_task_id: ContextVar[str | None] = ContextVar("rwkv_ecra_budget_task_id", default=None)


def remaining_seconds() -> float | None:
    deadline = _deadline.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def check_time_budget(*, minimum_seconds: float = 0.0) -> float | None:
    """Return remaining seconds or fail closed when the budget is exhausted."""
    remaining = remaining_seconds()
    if remaining is not None and remaining <= max(0.0, float(minimum_seconds)):
        task_id = _task_id.get() or "unknown"
        raise TaskTimeoutError(f"analysis task {task_id} exceeded its wall-clock budget")
    return remaining


def bounded_timeout(requested_seconds: float, *, minimum_seconds: float = 0.1) -> float:
    """Cap an I/O timeout by the current task budget when one is active."""
    requested = max(float(minimum_seconds), float(requested_seconds))
    remaining = check_time_budget(minimum_seconds=minimum_seconds)
    if remaining is None:
        return requested
    return max(float(minimum_seconds), min(requested, remaining))


@contextmanager
def task_time_budget(task_id: str, timeout_seconds: float | None = None) -> Iterator[None]:
    """Install a task budget and persist its lifecycle in the replay trace."""
    from utils.task_events import append_task_event

    configured = get_analysis_timeout_seconds() if timeout_seconds is None else timeout_seconds
    budget = None if configured is None or float(configured) <= 0 else float(configured)
    started = time.monotonic()
    task_token = _task_id.set(str(task_id))
    deadline_token = _deadline.set(started + budget if budget is not None else None)
    append_task_event(
        str(task_id),
        "runtime_budget",
        phase="RUNTIME",
        status="started",
        timeout_seconds=budget,
        disabled=budget is None,
    )
    try:
        yield
        check_time_budget()
    except TaskTimeoutError as exc:
        append_task_event(
            str(task_id),
            "runtime_budget",
            phase="RUNTIME",
            status="timed_out",
            timeout_seconds=budget,
            elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            error=str(exc),
        )
        raise
    except Exception as exc:
        append_task_event(
            str(task_id),
            "runtime_budget",
            phase="RUNTIME",
            status="failed",
            timeout_seconds=budget,
            elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            error=f"{type(exc).__name__}: {exc}"[:1000],
        )
        raise
    else:
        append_task_event(
            str(task_id),
            "runtime_budget",
            phase="RUNTIME",
            status="completed",
            timeout_seconds=budget,
            elapsed_ms=round((time.monotonic() - started) * 1000, 1),
        )
    finally:
        _deadline.reset(deadline_token)
        _task_id.reset(task_token)
