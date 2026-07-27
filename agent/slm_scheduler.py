"""Bounded asynchronous batching for local SLM requests."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque

from clients.slm_client import SLMClient
from config import (
    get_slm_async_batch_wait_ms,
    get_slm_async_enabled,
    get_slm_async_parallelism,
    get_slm_concurrency,
)
from utils.chunker import get_token_count
from utils.token_tracker import global_token_tracker


class _SLMInputQueueItem:
    def __init__(
        self,
        request_id: str,
        task_id: str,
        index: int,
        content: str,
        tracker,
        endpoint: str,
        password: str,
    ):
        self.request_id = request_id
        self.task_id = task_id
        self.index = index
        self.content = content
        self.tracker = tracker
        self.endpoint = endpoint
        self.password = password
        self.result = ""
        self.error = None
        self.done = threading.Event()

    @property
    def backend_key(self):
        return self.endpoint, self.password


class SLMInputScheduler:
    """Queue prompts and batch compatible backend requests."""

    def __init__(self):
        self._queue = deque()
        self._condition = threading.Condition()
        self._worker_started = False
        self._active_batches = 0

    def submit(self, contents: list[str], tracker=None, task_id: str = "") -> list[str]:
        if not contents:
            return []
        if not get_slm_async_enabled():
            return SLMClient().batch_generate(contents, tracker=tracker, task_id=task_id)

        request_id = uuid.uuid4().hex
        client = SLMClient()
        items = [
            _SLMInputQueueItem(
                request_id,
                task_id or "UNKNOWN_TASK",
                index,
                content,
                tracker,
                client.endpoint,
                client.password,
            )
            for index, content in enumerate(contents)
        ]
        with self._condition:
            self._ensure_worker_locked()
            self._queue.extend(items)
            self._condition.notify()

        for item in items:
            item.done.wait()
            if item.error:
                raise item.error
        return [item.result for item in items]

    def _ensure_worker_locked(self):
        if self._worker_started:
            return
        worker = threading.Thread(target=self._run, name="SLMInputScheduler", daemon=True)
        worker.start()
        self._worker_started = True

    def _run(self):
        while True:
            batch = self._take_batch()
            worker = threading.Thread(
                target=self._process_batch,
                args=(batch,),
                name="SLMInputBatch",
                daemon=True,
            )
            worker.start()

    def _take_batch(self):
        with self._condition:
            while not self._queue or self._active_batches >= get_slm_async_parallelism():
                self._condition.wait()

            max_batch = get_slm_concurrency()
            first = self._queue.popleft()
            backend_key = first.backend_key
            batch = [first]
            wait_until = time.monotonic() + get_slm_async_batch_wait_ms() / 1000.0

            while len(batch) < max_batch:
                scan_index = 0
                matched = False
                while len(batch) < max_batch and scan_index < len(self._queue):
                    candidate = self._queue[scan_index]
                    if candidate.backend_key == backend_key:
                        batch.append(candidate)
                        del self._queue[scan_index]
                        matched = True
                    else:
                        scan_index += 1

                if len(batch) >= max_batch:
                    break
                remaining = wait_until - time.monotonic()
                if remaining <= 0:
                    break
                if not matched:
                    self._condition.wait(timeout=remaining)
                    if self._active_batches >= get_slm_async_parallelism():
                        break

            self._active_batches += 1
            return batch

    def _process_batch(self, batch):
        try:
            print(f"[SLM 输入队列] 发射 {len(batch)} 个片段 | 首任务: {batch[0].task_id}")
            for item in batch:
                global_token_tracker.add_slm(
                    get_token_count(item.content),
                    0,
                    task_id=item.task_id,
                )

            client = SLMClient(
                endpoint_override=batch[0].endpoint,
                password_override=batch[0].password,
            )
            results = client._batch_generate_direct([item.content for item in batch])
            for item, result in zip(batch, results):
                item.result = result
                global_token_tracker.add_slm(
                    0,
                    get_token_count(result),
                    task_id=item.task_id,
                )
                if item.tracker:
                    item.tracker.track_slm(
                        input_prompt=item.content,
                        output_text=result,
                        task_id=item.task_id,
                    )
        except Exception as exc:
            for item in batch:
                item.error = exc
        finally:
            for item in batch:
                item.done.set()
            with self._condition:
                self._active_batches = max(0, self._active_batches - 1)
                self._condition.notify_all()


GLOBAL_SLM_INPUT_SCHEDULER = SLMInputScheduler()
