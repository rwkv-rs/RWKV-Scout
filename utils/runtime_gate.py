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
    get_model_chunk_requests_per_task,
    get_model_read_timeout_seconds,
    get_model_request_concurrency,
    get_model_reserved_control_slots,
)
from utils.task_events import append_task_event
from utils.time_budget import check_time_budget
from utils.token_tracker import current_model_lane


_guard = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = None
_limit = 0
_chunk_semaphores: dict[str, threading.BoundedSemaphore] = {}
_chunk_semaphore_limit = 0


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


def _acquire_workspace_lease(
    task_id: str,
    limit: int,
    *,
    prefix: str,
    stale_after: float,
    max_wait_seconds: float | None = None,
    slot_indices: tuple[int, ...] | None = None,
) -> str:
    slots = tuple(range(limit)) if slot_indices is None else tuple(slot_indices)
    if not slots:
        raise ValueError("workspace lease requires at least one slot")
    directory = _lease_directory()
    deadline = (
        time.monotonic() + float(max_wait_seconds)
        if max_wait_seconds is not None and float(max_wait_seconds) > 0
        else None
    )
    while True:
        check_time_budget(minimum_seconds=0.2)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"{prefix} lease wait exceeded {max_wait_seconds:.1f}s")
        for slot in slots:
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
        sleep_seconds = 0.1
        if deadline is not None:
            sleep_seconds = min(sleep_seconds, max(0.01, deadline - time.monotonic()))
        time.sleep(sleep_seconds)


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


def _get_chunk_semaphore(task_id: str, limit: int) -> threading.BoundedSemaphore:
    """Limit chunk fan-out per task inside each worker process."""
    global _chunk_semaphore_limit
    key = str(task_id or "UNKNOWN_TASK")
    with _guard:
        if _chunk_semaphore_limit != limit:
            _chunk_semaphores.clear()
            _chunk_semaphore_limit = limit
        semaphore = _chunk_semaphores.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _chunk_semaphores[key] = semaphore
        return semaphore


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
def model_request_slot(task_id: str, *, lane: str | None = None) -> Iterator[None]:
    """Bound model HTTP requests across all local worker processes.

    The benchmark starts one Python process per worker, so a normal semaphore
    cannot coordinate them.  The lease files are atomic and shared by the
    workspace, which makes this limit effective for both the API and JSON
    acceptance runners.
    """
    limit = get_model_request_concurrency()
    selected_lane = str(lane or current_model_lane.get() or "control").casefold()
    is_chunk = selected_lane in {"chunk", "evidence", "extraction"}
    reserved = get_model_reserved_control_slots() if is_chunk else 0
    chunk_capacity = max(1, limit - reserved) if is_chunk else limit
    chunk_semaphore = (
        _get_chunk_semaphore(task_id, get_model_chunk_requests_per_task())
        if is_chunk
        else None
    )
    started = time.perf_counter()
    append_task_event(
        task_id,
        "model_request_gate",
        status="waiting",
        limit=limit,
        lane=selected_lane,
        reserved_control_slots=reserved,
        chunk_capacity=chunk_capacity if is_chunk else None,
        scope="workspace",
    )
    local_acquired = False
    try:
        if chunk_semaphore is not None:
            while not chunk_semaphore.acquire(timeout=0.1):
                check_time_budget(minimum_seconds=0.2)
            local_acquired = True
        lease_id = _acquire_workspace_lease(
            task_id,
            limit,
            prefix="model-slot",
            stale_after=max(60.0, get_model_read_timeout_seconds() * 2.0),
            # Chunk requests may only use the non-reserved part of the shared
            # slot namespace. Control requests may use every slot.
            slot_indices=tuple(range(chunk_capacity)) if is_chunk else tuple(range(limit)),
            max_wait_seconds=get_model_read_timeout_seconds(),
        )
    except Exception:
        if local_acquired and chunk_semaphore is not None:
            chunk_semaphore.release()
        raise
    waited_ms = round((time.perf_counter() - started) * 1000, 1)
    append_task_event(
        task_id,
        "model_request_gate",
        status="acquired",
        limit=limit,
        lane=selected_lane,
        wait_ms=waited_ms,
        scope="workspace",
    )
    try:
        yield
    finally:
        _release_workspace_lease(lease_id)
        if local_acquired and chunk_semaphore is not None:
            chunk_semaphore.release()
        append_task_event(
            task_id,
            "model_request_gate",
            status="released",
            limit=limit,
            lane=selected_lane,
            scope="workspace",
        )
