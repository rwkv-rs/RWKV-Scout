from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.orchestrator import Orchestrator
from agent.retrieval_synthesis import synthesize_retrieval_answer
from tools.builtin import load_builtin_tools
from tools.date_calculator import extract_date_candidates
from tools.registry import ToolRegistry


class DateCalculatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_builtin_tools()

    def test_date_diff_returns_absolute_and_signed_days(self):
        result = json.loads(
            ToolRegistry.execute(
                "date_diff",
                {
                    "date_a": "2023-10-05",
                    "date_b": "2025-04-16",
                    "source_a": "S1:C3",
                    "source_b": "S1:C7",
                },
                {},
                phase="ALL",
            )
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["days"], 559)
        self.assertEqual(result["signed_days"], 559)
        self.assertEqual(result["source_refs"], ["S1:C3", "S1:C7"])

    def test_date_diff_preserves_direction(self):
        result = json.loads(
            ToolRegistry.execute(
                "date_diff",
                {"date_a": "2025-04-16", "date_b": "2023-10-05"},
                {},
                phase="ALL",
            )
        )
        self.assertEqual(result["days"], 559)
        self.assertEqual(result["signed_days"], -559)

    def test_date_diff_handles_leap_year(self):
        result = json.loads(
            ToolRegistry.execute(
                "date_diff",
                {"date_a": "2024-02-28", "date_b": "2024-03-01"},
                {},
                phase="ALL",
            )
        )
        self.assertEqual(result["days"], 2)

    def test_date_candidate_extraction_is_deterministic_but_not_semantic_binding(self):
        candidates = extract_date_candidates(
            "Release A: 2023年10月5日. Release B: 2025-04-16. Invalid: 2024-02-30."
        )
        self.assertEqual(candidates, ["2023-10-05", "2025-04-16"])

    def test_date_diff_rejects_ambiguous_or_invalid_dates(self):
        for value in ("2024/02/28", "2024-02-30", "today"):
            result = json.loads(
                ToolRegistry.execute(
                    "date_diff",
                    {"date_a": value, "date_b": "2024-03-01"},
                    {},
                    phase="ALL",
                )
            )
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_class"], "invalid_date")

    def test_date_diff_is_not_available_during_page_extraction(self):
        self.assertFalse(ToolRegistry.can_execute("date_diff", "DISCOVERY"))
        self.assertFalse(ToolRegistry.can_execute("date_diff", "EXTRACTION"))
        self.assertTrue(ToolRegistry.can_execute("date_diff", "SYNTHESIS"))

    def test_calculation_result_reaches_final_rwkv_prompt(self):
        class FakeLLM:
            def text_completion(self, prompt, max_tokens=1024, stop=None):
                self.prompt = prompt
                return SimpleNamespace(content="相隔 559 天。")

        llm = FakeLLM()
        result = synthesize_retrieval_answer(
            "2023-10-05 到 2025-04-16 相隔多少天？",
            {
                "query": "2023-10-05 到 2025-04-16 相隔多少天？",
                "results": [],
                "calculation_results": [
                    {
                        "status": "ok",
                        "tool": "date_diff",
                        "date_a": "2023-10-05",
                        "date_b": "2025-04-16",
                        "days": 559,
                        "signed_days": 559,
                        "formula": "2025-04-16 - 2023-10-05 = 559 days",
                        "source_refs": [],
                    }
                ],
            },
            llm=llm,
            constraints={"task_plan": {"task_mode": "lookup"}},
        )
        self.assertIn("559", result["content"])
        self.assertIn("TOOL RESULTS", result["prompt"])
        self.assertEqual(result["context_stats"]["calculation_count"], 1)
        self.assertEqual(result["answer_quality"], {})

    def test_orchestrator_records_model_selected_calculation(self):
        orchestrator = Orchestrator()
        orchestrator.state.task_id = "DATE_TOOL_LOOP_TEST"
        orchestrator.planner.plan_next_action = Mock(
            side_effect=[
                {
                    "action": "date_diff",
                    "args": {
                        "date_a": "2023-10-05",
                        "date_b": "2025-04-16",
                        "source_a": "S1:C3",
                        "source_b": "S1:C7",
                    },
                },
                {"action": "finish_task", "args": {}},
            ]
        )
        orchestrator.planner.observe_tool_result = Mock()
        orchestrator._cross_validate_research = Mock(return_value={"decision": "finish"})
        orchestrator._complete_model_tool_loop = Mock(return_value="done")
        with patch("agent.unified_research.append_task_event") as append_event:
            result = orchestrator._run_single_loop(
                "How many days?",
                {},
                {"atomic_points": []},
                max_steps=2,
            )

        self.assertEqual(result, "done")
        self.assertEqual(orchestrator._calculation_results[0]["days"], 559)
        calculation_events = [
            call for call in append_event.call_args_list
            if call.args[1] == "calculation_result"
        ]
        self.assertEqual(len(calculation_events), 1)


if __name__ == "__main__":
    unittest.main()
