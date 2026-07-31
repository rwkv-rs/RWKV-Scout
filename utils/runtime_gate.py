"""Workspace-wide concurrency gate for long-running analysis tasks."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from config import (
    DATA_PIPELINE,
    get_analysis_timeout_seconds,
    get_experiment_max_parallel_cases,
    get_model_read_timeout_seconds,
    get_model_request_concurrency,
)
from utils.task_events import append_task_event
from utils.time_budget import check_time_budget


_guard = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = None
_limit = 0


def _lease_directory() -> Path:
    root = Path(DATA_PIPELINE.get("output_directory", "./data/output")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / ".runtime_gate"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cleanup_stale_leases(directory: Path, stale_after: float, prefix: str) -> None:
    cutoff = time.time() - stale_after
    for path in directory.glob(f"{prefix}-*.lease"):
        try:
            stale = path.stat().st_mtime < cutoff
            if not stale:
                # A terminated benchmark worker can leave a fresh lease
                # behind.  Do not make the next run wait for the full stale
                # timeout when the recorded owner PID is already gone.
                owner_pid = None
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("pid="):
                        try:
                            owner_pid = int(line.partition("=")[2])
                        except ValueError:
                            owner_pid = None
                        break
                if owner_pid is None:
                    stale = True
                else:
                    try:
                        os.kill(owner_pid, 0)
                    except ProcessLookupError:
                        stale = True
                    except PermissionError:
                        # The process exists but belongs to another user;
                        # leave the lease in place rather than deleting it.
                        pass
            if stale:
                path.unlink()
        except (FileNotFoundError, OSError):
            continue


def _acquire_workspace_lease(task_id: str, limit: int, *, prefix: str, stale_after: float) -> str:
    directory = _lease_directory()
    while True:
        check_time_budget(minimum_seconds=0.2)
        for slot in range(limit):
            path = directory / f"{prefix}-{slot}.lease"
            try:
                descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(
                        f"task_id={task_id}\npid={os.getpid()}\n"
                        f"thread={threading.get_ident()}\nlease_id={uuid4().hex}\n"
                    )
                return str(path)
            except FileExistsError:
                continue
            except OSError:
                try:
                    path.unlink()
                except OSError:
                    pass
                raise
        _cleanup_stale_leases(directory, stale_after, prefix)
        time.sleep(0.1)


def _release_workspace_lease(lease_path: str) -> None:
    try:
        Path(lease_path).unlink(missing_ok=True)
    except OSError:
        # The process-local gate still releases in the caller's finally block.
        return


def _get_semaphore() -> tuple[threading.BoundedSemaphore, int]:
    global _semaphore, _limit
    configured = get_experiment_max_parallel_cases()
    with _guard:
        if _semaphore is None or _limit != configured:
            _semaphore = threading.BoundedSemaphore(configured)
            _limit = configured
        return _semaphore, _limit


@contextmanager
def analysis_slot(task_id: str) -> Iterator[None]:
    """Bound expensive tasks across threads and workers in one workspace.

    Waiting is intentional: a submitted task remains observable instead of
    being silently rejected when the local model is busy.
    """
    semaphore, limit = _get_semaphore()
    started = time.perf_counter()
    append_task_event(task_id, "runtime_gate", status="waiting", limit=limit, scope="workspace")
    while not semaphore.acquire(timeout=0.1):
        check_time_budget(minimum_seconds=0.2)
    lease_id = None
    try:
        analysis_timeout = get_analysis_timeout_seconds()
        lease_id = _acquire_workspace_lease(
            task_id,
            limit,
            prefix="slot",
            stale_after=max(60.0, (analysis_timeout or 1800.0) * 2.0),
        )
    except Exception:
        semaphore.release()
        raise
    waited_ms = round((time.perf_counter() - started) * 1000, 1)
    append_task_event(
        task_id,
        "runtime_gate",
        status="acquired",
        limit=limit,
        wait_ms=waited_ms,
        scope="workspace",
    )
    try:
        yield
    finally:
        if lease_id:
            _release_workspace_lease(lease_id)
        semaphore.release()
        append_task_event(task_id, "runtime_gate", status="released", limit=limit, scope="workspace")


@contextmanager
def model_request_slot(task_id: str) -> Iterator[None]:
    """Bound model HTTP requests across all local worker processes.

    The benchmark starts one Python process per worker, so a normal semaphore
    cannot coordinate them.  The lease files are atomic and shared by the
    workspace, which makes this limit effective for both the API and JSON
    acceptance runners.
    """
    limit = get_model_request_concurrency()
    started = time.perf_counter()
    append_task_event(task_id, "model_request_gate", status="waiting", limit=limit, scope="workspace")
    lease_id = _acquire_workspace_lease(
        task_id,
        limit,
        prefix="model-slot",
        stale_after=max(60.0, get_model_read_timeout_seconds() * 2.0),
    )
    waited_ms = round((time.perf_counter() - started) * 1000, 1)
    append_task_event(
        task_id,
        "model_request_gate",
        status="acquired",
        limit=limit,
        wait_ms=waited_ms,
        scope="workspace",
    )
    try:
        yield
    finally:
        _release_workspace_lease(lease_id)
        append_task_event(task_id, "model_request_gate", status="released", limit=limit, scope="workspace")
