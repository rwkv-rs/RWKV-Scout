from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.execution_strategy import FORK, SINGLE_LOOP, select_strategy
from agent.orchestrator import Orchestrator
from agent.retrieval_runners import ForkRunner, SingleLoopRunner, build_runner


def _plan(count: int) -> dict:
    return {
        "schema_version": "task_plan.v1",
        "goal": "goal",
        "atomic_points": [
            {"id": f"P{index}", "task": f"task {index}"}
            for index in range(1, count + 1)
        ],
    }


class ExecutionStrategyTests(unittest.TestCase):
    def test_one_or_two_points_use_single_loop(self):
        self.assertEqual(select_strategy(_plan(1)).strategy, SINGLE_LOOP)
        self.assertEqual(select_strategy(_plan(2)).strategy, SINGLE_LOOP)

    def test_three_or_more_points_still_use_one_global_loop(self):
        decision = select_strategy(_plan(3))
        self.assertEqual(decision.strategy, SINGLE_LOOP)
        self.assertEqual(decision.point_ids, ("P1", "P2", "P3"))
        self.assertEqual(decision.source, "global_shared_state")

    def test_explicit_strategy_override_is_reserved_for_experiments(self):
        decision = select_strategy(_plan(1), {"retrieval_strategy": "fork"})
        self.assertEqual(decision.strategy, FORK)
        self.assertEqual(decision.source, "retrieval_strategy_override")

        legacy = select_strategy(_plan(3), {"retrieval_fork": False})
        self.assertEqual(legacy.strategy, SINGLE_LOOP)
        self.assertEqual(legacy.source, "legacy_retrieval_fork_override")

    def test_invalid_or_empty_plan_conservatively_uses_single_loop(self):
        decision = select_strategy({"goal": "goal", "atomic_points": []})
        self.assertEqual(decision.strategy, SINGLE_LOOP)
        self.assertEqual(decision.point_count, 0)

    def test_runner_factory_uses_global_loop_by_default(self):
        controller = object()
        self.assertIsInstance(build_runner(select_strategy(_plan(1)), controller), SingleLoopRunner)
        self.assertIsInstance(build_runner(select_strategy(_plan(3)), controller), SingleLoopRunner)
        self.assertIsInstance(
            build_runner(select_strategy(_plan(3), {"retrieval_strategy": "fork"}), controller),
            ForkRunner,
        )

    def test_orchestrator_routes_from_plan_without_a_strategy_field_in_the_plan(self):
        class FakeRunner:
            def __init__(self, decision):
                self.decision = decision

            def run(self, *_args):
                return self.decision.strategy

        for count, expected in ((1, SINGLE_LOOP), (3, SINGLE_LOOP)):
            orchestrator = Orchestrator()
            orchestrator.state.task_id = f"STRATEGY_{count}"
            plan = _plan(count)
            with (
                patch("agent.orchestrator.append_task_event"),
                patch.object(orchestrator.planner, "create_task_plan", return_value=plan),
                patch.object(orchestrator.planner, "begin_task"),
                patch(
                    "agent.orchestrator.build_runner",
                    side_effect=lambda decision, _controller: FakeRunner(decision),
                ),
            ):
                result = orchestrator._run_model_tool_loop("goal", {})

            self.assertEqual(result, expected)
            self.assertEqual(orchestrator.state.run_metadata["retrieval_strategy"], expected)


if __name__ == "__main__":
    unittest.main()
