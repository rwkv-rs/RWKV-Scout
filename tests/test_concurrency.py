from __future__ import annotations

import concurrent.futures
import contextvars
import threading
import unittest

from utils.concurrency import shutdown_pool, submit_with_context, task_wait_timeout
from utils.time_budget import child_time_budget, remaining_seconds, task_time_budget


class ConcurrencyHelperTests(unittest.TestCase):
    def test_worker_receives_task_context(self):
        marker = contextvars.ContextVar("test_concurrency_marker", default="missing")
        token = marker.set("task-123")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = submit_with_context(pool, marker.get)
                self.assertEqual(future.result(timeout=1), "task-123")
        finally:
            marker.reset(token)

    def test_task_wait_timeout_tracks_active_budget(self):
        with task_time_budget("CONCURRENCY_BUDGET", timeout_seconds=10.0):
            timeout = task_wait_timeout()
            self.assertIsNotNone(timeout)
            self.assertLessEqual(timeout, 10.0)
            self.assertGreater(timeout, 0.0)

    def test_child_budget_does_not_extend_parent_budget(self):
        with task_time_budget("PARENT_BUDGET", timeout_seconds=60.0):
            parent_remaining = remaining_seconds()
            with child_time_budget(1.0):
                self.assertLessEqual(remaining_seconds(), 1.0)
            self.assertLessEqual(remaining_seconds(), parent_remaining)

    def test_shutdown_pool_cancels_pending_work(self):
        started = threading.Event()
        release = threading.Event()

        def hold():
            started.set()
            release.wait(timeout=2)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        first = submit_with_context(pool, hold)
        self.assertTrue(started.wait(timeout=1))
        second = submit_with_context(pool, lambda: "should not run")
        shutdown_pool(pool, [first, second], cancelled=True)
        release.set()
        first.result(timeout=1)
        self.assertTrue(second.cancelled())


if __name__ == "__main__":
    unittest.main()
