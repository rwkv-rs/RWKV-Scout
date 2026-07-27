"""Retry policy with task-visible failure and retry events."""

from __future__ import annotations

import functools
import time

from utils.error_policy import classify_error


def _record_retry_event(
    event_type: str,
    *,
    function: str,
    attempt: int,
    max_retries: int,
    error: Exception,
    delay: float = 0.0,
) -> None:
    """Persist retry information when a task context is active.

    Observability is best-effort: a logging failure must never turn a provider
    error into a second application failure.
    """
    try:
        from utils.task_events import append_task_event
        from utils.token_tracker import current_task_id

        task_id = current_task_id.get()
        if task_id and task_id != "UNKNOWN_TASK":
            append_task_event(
                task_id,
                event_type,
                phase="RUNTIME",
                function=function,
                attempt=attempt,
                max_retries=max_retries,
                delay_seconds=delay,
                error=f"{type(error).__name__}: {error}"[:1000],
                error_class=classify_error(error),
            )
    except Exception:
        return


def retry_with_fallback(max_retries=3, delay=2, backoff=2):
    max_retries = max(1, int(max_retries))
    delay = max(0.0, float(delay))
    backoff = max(1.0, float(backoff))

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
                    error_class = classify_error(exc)
                    retryable = error_class not in {"auth", "validation", "filesystem"}
                    next_delay = delay * (backoff**attempt) if attempt < max_retries - 1 else 0.0
                    _record_retry_event(
                        "retry" if attempt < max_retries - 1 and retryable else "error",
                        function=f"{func.__module__}.{func.__qualname__}",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=exc,
                        delay=next_delay,
                    )
                    print(
                        f"[retry] {func.__name__} attempt {attempt + 1}/{max_retries}: {exc}"
                    )
                    if attempt < max_retries - 1 and retryable:
                        time.sleep(next_delay)
                    elif not retryable:
                        raise
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator
