from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.orchestrator import Orchestrator


def _plan(count: int) -> dict:
    return {
        "schema_version": "task_plan.v1",
        "goal": "goal",
        "atomic_points": [
            {"id": f"P{index}", "task": f"task {index}"}
            for index in range(1, count + 1)
        ],
    }


class GlobalSharedLoopTests(unittest.TestCase):
    def test_all_task_point_counts_use_one_shared_loop(self):
        for count in (1, 2, 3, 8):
            orchestrator = Orchestrator()
            orchestrator.state.task_id = f"SINGLE_LOOP_{count}"
            plan = _plan(count)
            with (
                patch("agent.orchestrator.append_task_event"),
                patch.object(orchestrator.planner, "create_task_plan", return_value=plan),
                patch.object(orchestrator.planner, "begin_task"),
                patch.object(orchestrator, "_run_single_loop", return_value="ok") as run_loop,
            ):
                result = orchestrator._run_model_tool_loop("goal", {})

            self.assertEqual(result, "ok")
            run_loop.assert_called_once()
            self.assertEqual(
                orchestrator.state.run_metadata["retrieval_strategy"],
                "single_loop",
            )
            self.assertEqual(
                orchestrator.state.run_metadata["retrieval_strategy_decision"]["point_count"],
                count,
            )

if __name__ == "__main__":
    unittest.main()
