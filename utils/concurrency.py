"""Small helpers for bounded, context-aware worker pools.

The retrieval pipeline uses worker pools for network and chunk operations.
``ThreadPoolExecutor`` does not copy ``ContextVar`` state into workers and a
``with`` block waits for every worker even after the parent task has timed out.
These helpers keep task identity/deadlines attached to child work and make
pending work cancellable at a task boundary.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
from typing import Any, Callable

from utils.time_budget import remaining_seconds


def submit_with_context(
    executor: concurrent.futures.Executor,
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> concurrent.futures.Future:
    """Submit one callable with a private copy of the caller context."""

    context = contextvars.copy_context()
    return executor.submit(context.run, function, *args, **kwargs)


def task_wait_timeout() -> float | None:
    """Return the remaining task budget for a future wait, if one is active."""

    remaining = remaining_seconds()
    if remaining is None:
        return None
    return max(0.05, remaining)


def shutdown_pool(
    executor: concurrent.futures.Executor,
    futures: list[concurrent.futures.Future],
    *,
    cancelled: bool,
) -> None:
    """Cancel pending futures and avoid waiting again after a timeout."""

    if cancelled:
        for future in futures:
            future.cancel()
    # ``cancel_futures`` is available on supported Python versions.  Keep a
    # fallback for environments that provide an older Executor implementation.
    try:
        executor.shutdown(wait=not cancelled, cancel_futures=cancelled)
    except TypeError:
        executor.shutdown(wait=not cancelled)

