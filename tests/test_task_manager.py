from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.task_manager import TaskStore


class TaskStoreTests(unittest.TestCase):
    def test_events_replay_and_status_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            task_dir = output / "TASK_1"
            task_dir.mkdir(parents=True)
            (task_dir / "report.md").write_text("done", encoding="utf-8")

            store = TaskStore(str(root / "tasks.jsonl"), str(output))
            store.record_task("TASK_1", "query", "running", str(task_dir), queued_at="now")
            store.update_task_progress("TASK_1", "working")
            self.assertEqual(store.get_all_tasks()[0]["progress"], "working")
            self.assertFalse(store.is_task_stopped("TASK_1"))

            store.request_stop("TASK_1")
            self.assertTrue(store.is_task_stopped("TASK_1"))

            reloaded = TaskStore(str(root / "tasks.jsonl"), str(output))
            self.assertEqual(reloaded.get_all_tasks()[0]["status"], "stopped")

    def test_delete_does_not_touch_an_empty_result_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            unrelated = root / "unrelated"
            unrelated.mkdir()
            store = TaskStore(str(root / "tasks.jsonl"), str(output))
            store.record_task("TASK_2", "query", "completed", "")
            store.delete_task("TASK_2")
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
