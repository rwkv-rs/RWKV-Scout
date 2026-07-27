"""Small cross-process atomic sidecar lock for append-only local artifacts."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4


@contextmanager
def atomic_file_lease(
    target: str | os.PathLike[str],
    *,
    timeout_seconds: float = 10.0,
    stale_after_seconds: float = 120.0,
) -> Iterator[None]:
    """Acquire an atomic sidecar lease that works across Python processes.

    It is intentionally filesystem based so Windows, WSL and Linux workers
    share the same mechanism.  A stale lease is recoverable after a process
    crash; normal release only removes the caller's own sidecar.
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_path.with_name(target_path.name + ".lock")
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    lease_id = uuid4().hex
    while True:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\nlease_id={lease_id}\n")
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > max(1.0, float(stale_after_seconds)):
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring file lease: {target_path}")
            time.sleep(0.02)
    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
