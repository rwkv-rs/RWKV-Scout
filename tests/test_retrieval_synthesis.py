import unittest
from types import SimpleNamespace

from agent.retrieval_synthesis import synthesize_retrieval_answer


class _FakeLLM:
    provider = "local_13b"

    def __init__(self):
        self.calls = []

    def text_completion(self, prompt, max_tokens=0, **kwargs):
        self.calls.append((prompt, max_tokens, kwargs))
        return SimpleNamespace(content="完整答案 [S1]")


class RetrievalSynthesisTests(unittest.TestCase):
    def test_final_summary_uses_remaining_context_budget_and_acceptance_plan(self):
        llm = _FakeLLM()
        result = synthesize_retrieval_answer(
            "列出全部站点",
            {
                "query": "列出全部站点",
                "results": [
                    {
                        "title": "站点表",
                        "url": "https://example.com/stations",
                        "content": "| Name | Line |\n| --- | --- |\n| A | 1 |\n| B | 1 |",
                        "source": "test",
                        "chunk_candidates": [],
                    }
                ],
                "citation_refs": [{"ref_id": "S1", "title": "站点表", "url": "https://example.com/stations"}],
            },
            llm=llm,
            constraints={
                "task_plan": {
                    "atomic_points": [
                        {
                            "id": "P1",
                            "task": "列出全部站点",
                            "objective": "站点清单",
                            "acceptance_criteria": ["每一行都保留原始列关系"],
                            "output_format": "table",
                        }
                    ]
                }
            },
        )
        self.assertGreater(llm.calls[0][1], 3000)
        self.assertLessEqual(llm.calls[0][1], 8192)
        self.assertIn("Acceptance checklist", llm.calls[0][0])
        self.assertIn("row/column relationship", llm.calls[0][0])
        self.assertEqual(result["content"], "完整答案 [S1]")
        self.assertEqual(result["model_output"], "完整答案 [S1]")

    def test_final_summary_receives_visible_execution_context_at_step_limit(self):
        llm = _FakeLLM()
        result = synthesize_retrieval_answer(
            "查找事实",
            {"query": "查找事实", "results": [], "citation_refs": []},
            llm=llm,
            execution_context="RWKV planner transcript:\nAssistant: search_mediawiki\nFunction output: status=ok",
            termination_reason="max_steps_reached",
        )
        self.assertIn("max_steps_reached", llm.calls[0][0])
        self.assertIn("search_mediawiki", llm.calls[0][0])
        self.assertIn("status=ok", result["context_text"])


if __name__ == "__main__":
    unittest.main()
