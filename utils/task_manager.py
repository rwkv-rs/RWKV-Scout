"""Thread-safe task metadata store backed by append-only JSONL events."""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DATA_PIPELINE
from utils.file_lock import atomic_file_lease


TASK_LOG_FILE = str(Path(DATA_PIPELINE.get("output_directory", "./data/output")) / "tasks.jsonl")
_store_lock = threading.Lock()


class TaskStore:
    def __init__(self, filepath: str, output_directory: str | None = None):
        self.filepath = Path(filepath)
        self.output_directory = Path(
            output_directory or DATA_PIPELINE.get("output_directory", "./data/output")
        )
        self._lock = threading.RLock()
        self._task_index: dict[str, dict[str, Any]] = {}
        self._ordered_keys: list[str] = []
        self._load()
        self._sync_with_filesystem()

    def _load(self) -> None:
        if not self.filepath.exists():
            return
        try:
            lines = self.filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                self._apply_record(record)

    def _apply_record(self, record: dict[str, Any]) -> None:
        task_id = str(record.get("task_id") or "").strip()
        if not task_id:
            return
        if task_id not in self._task_index:
            self._ordered_keys.append(task_id)
            self._task_index[task_id] = {}
        self._task_index[task_id].update(record)
        if self._task_index[task_id].get("status") == "deleted":
            self._task_index.pop(task_id, None)
            if task_id in self._ordered_keys:
                self._ordered_keys.remove(task_id)

    def _append_event(self, record: dict[str, Any]) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file_lease(self.filepath):
            with self.filepath.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _sync_with_filesystem(self) -> None:
        with self._lock:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            changed = False

            for task_id in list(self._ordered_keys):
                record = self._task_index[task_id]
                expected = self.output_directory / task_id
                recorded = Path(str(record.get("result_dir") or "")) if record.get("result_dir") else None
                if not expected.exists() and not (recorded and recorded.exists()):
                    self._ordered_keys.remove(task_id)
                    self._task_index.pop(task_id, None)
                    changed = True

            for item in self.output_directory.iterdir():
                if not item.is_dir() or item.name in self._task_index:
                    continue
                report_files = [
                    path for path in item.rglob("*")
                    if path.is_file() and path.suffix.casefold() in {".jsonl", ".md"}
                ]
                if not report_files:
                    continue
                target = max(report_files, key=lambda path: path.stat().st_mtime)
                timestamp = datetime.fromtimestamp(target.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                self._ordered_keys.append(item.name)
                self._task_index[item.name] = {
                    "task_id": item.name,
                    "timestamp": timestamp,
                    "query": item.name,
                    "status": "completed",
                    "result_dir": str(item),
                    "error": "",
                    "queued_at": timestamp,
                }
                changed = True

            if changed:
                self._ordered_keys.sort(
                    key=lambda task_id: self._task_index[task_id].get("timestamp", "")
                )
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                with atomic_file_lease(self.filepath):
                    self.filepath.write_text(
                        "".join(
                            json.dumps(self._task_index[task_id], ensure_ascii=False) + "\n"
                            for task_id in self._ordered_keys
                        ),
                        encoding="utf-8",
                    )

    def record_task(
        self,
        task_id: str,
        query: str,
        status: str,
        result_dir: str = "",
        error: str = "",
        queued_at: str | None = None,
        acceptance_case_id: str | None = None,
    ) -> None:
        with self._lock:
            existing = self._task_index.get(task_id, {})
            record: dict[str, Any] = {
                "task_id": task_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "status": status,
                "result_dir": result_dir,
                "error": error,
            }
            if queued_at is not None:
                record["queued_at"] = queued_at
            elif existing.get("queued_at") is not None:
                record["queued_at"] = existing["queued_at"]
            if acceptance_case_id is not None:
                record["acceptance_case_id"] = acceptance_case_id
            elif existing.get("acceptance_case_id") is not None:
                record["acceptance_case_id"] = existing["acceptance_case_id"]
            self._apply_record(record)
            self._append_event(self._task_index[task_id].copy())

    def update_task_progress(self, task_id: str, progress: str) -> None:
        with self._lock:
            if task_id not in self._task_index:
                return
            self._task_index[task_id]["progress"] = progress
            self._append_event(self._task_index[task_id].copy())

    def request_stop(self, task_id: str) -> None:
        with self._lock:
            task = self._task_index.get(task_id)
            if not task or task.get("status") != "running":
                return
            task["status"] = "stopped"
            self._append_event(task.copy())

    def delete_task(self, task_id: str) -> None:
        with self._lock:
            task = self._task_index.get(task_id)
            if not task:
                return
            self._append_event({**task, "status": "deleted"})
            result_dir_value = str(task.get("result_dir") or "").strip()
            if result_dir_value:
                result_dir = Path(result_dir_value).expanduser().resolve()
                try:
                    result_dir.relative_to(self.output_directory.resolve())
                except ValueError:
                    result_dir = None
                if result_dir is not None and result_dir.exists() and result_dir.is_dir():
                    try:
                        shutil.rmtree(result_dir)
                    except OSError:
                        pass
            self._task_index.pop(task_id, None)
            if task_id in self._ordered_keys:
                self._ordered_keys.remove(task_id)

    def is_task_stopped(self, task_id: str) -> bool:
        with self._lock:
            task = self._task_index.get(task_id)
            # A direct Orchestrator invocation may start before the API/task
            # queue has persisted a record.  Absence is not a stop request;
            # only an explicit terminal stop state should interrupt work.
            return bool(task and task.get("status") in {"stopped", "deleted"})

    def get_all_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._task_index[task_id].copy() for task_id in reversed(self._ordered_keys)]


_store: TaskStore | None = None


def _get_store() -> TaskStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = TaskStore(TASK_LOG_FILE)
    return _store


def record_task(
    task_id: str,
    query: str,
    status: str,
    result_dir: str = "",
    error: str = "",
    queued_at: str | None = None,
    acceptance_case_id: str | None = None,
) -> None:
    _get_store().record_task(
        task_id,
        query,
        status,
        result_dir,
        error,
        queued_at,
        acceptance_case_id,
    )


def update_task_progress(task_id: str, progress: str) -> None:
    _get_store().update_task_progress(task_id, progress)


def request_stop(task_id: str) -> None:
    _get_store().request_stop(task_id)


def delete_task(task_id: str) -> None:
    _get_store().delete_task(task_id)


def is_task_stopped(task_id: str) -> bool:
    return _get_store().is_task_stopped(task_id)


def get_all_tasks() -> list[dict[str, Any]]:
    return _get_store().get_all_tasks()
