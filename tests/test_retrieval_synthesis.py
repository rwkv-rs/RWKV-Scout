import unittest
from types import SimpleNamespace

from agent.retrieval_synthesis import _clean_answer, _final_completion_budget, synthesize_retrieval_answer
from config import get_llm_context_length
from utils.chunker import get_token_count


class _FakeLLM:
    provider = "local_13b"

    def __init__(self):
        self.calls = []

    def text_completion(self, prompt, max_tokens=0, **kwargs):
        self.calls.append((prompt, max_tokens, kwargs))
        return SimpleNamespace(content="Supported answer [S1]")


class RetrievalSynthesisTests(unittest.TestCase):
    def test_evidence_protocol_is_not_a_user_facing_answer(self):
        self.assertEqual(
            _clean_answer(
                "BEGIN EVIDENCE SOURCE S1\n"
                "URL (citation metadata only): https://example.com\n"
                "EVIDENCE BODY\nSome source text"
            ),
            "",
        )

    def test_final_budget_never_requests_more_than_remaining_context(self):
        prompt = "token " * 11265
        budget = _final_completion_budget(prompt)
        self.assertGreaterEqual(budget, 1)
        self.assertLessEqual(
            get_token_count(prompt) + budget + 256,
            get_llm_context_length(),
        )

    def test_final_summary_uses_remaining_context_budget_and_acceptance_plan(self):
        llm = _FakeLLM()
        result = synthesize_retrieval_answer(
            "\u5217\u51fa\u5168\u90e8\u7ad9\u70b9",
            {
                "query": "\u5217\u51fa\u5168\u90e8\u7ad9\u70b9",
                "results": [
                    {
                        "title": "\u7ad9\u70b9\u8868",
                        "url": "https://example.com/stations",
                        "content": "| Name | Line |\n| --- | --- |\n| A | 1 |\n| B | 1 | The source body preserves every requested row and column relationship.",
                        "source": "test",
                        "chunk_candidates": [],
                    }
                ],
                "citation_refs": [{"ref_id": "S1", "title": "\u7ad9\u70b9\u8868", "url": "https://example.com/stations"}],
            },
            llm=llm,
            constraints={
                "task_plan": {
                    "atomic_points": [
                        {
                            "id": "P1",
                            "task": "\u5217\u51fa\u5168\u90e8\u7ad9\u70b9",
                            "objective": "\u7ad9\u70b9\u6e05\u5355",
                            "acceptance_criteria": ["\u6bcf\u4e00\u884c\u90fd\u4fdd\u7559\u539f\u59cb\u5217\u5173\u7cfb"],
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
        self.assertIn("[S1](https://example.com/stations)", result["content"])

    def test_final_summary_receives_visible_execution_context_at_step_limit(self):
        llm = _FakeLLM()
        result = synthesize_retrieval_answer(
            "find a fact",
            {"query": "find a fact", "results": [], "citation_refs": []},
            llm=llm,
            execution_context="RWKV planner transcript:\nAssistant: search_mediawiki\nFunction output: status=ok",
            termination_reason="max_steps_reached",
        )
        self.assertIn("max_steps_reached", llm.calls[0][0])
        self.assertIn("search_mediawiki", llm.calls[0][0])
        self.assertIn("status=ok", result["context_text"])

    def test_no_usable_evidence_requires_explicit_refusal(self):
        llm = _FakeLLM()
        result = synthesize_retrieval_answer(
            "Who is the founder?",
            {
                "query": "Who is the founder?",
                "results": [
                    {
                        "title": "Search result title",
                        "url": "https://example.com/search",
                        "snippet": "possibly relevant summary",
                        "source": "search",
                        "evidence_status": "no_evidence",
                    }
                ],
                "citation_refs": [],
            },
            llm=llm,
        )
        self.assertIn("NO_USABLE_EVIDENCE", llm.calls[0][0])
        self.assertIn("only valid answer is an explicit refusal", llm.calls[0][0])
        self.assertEqual(result["citation_refs"], [])


if __name__ == "__main__":
    unittest.main()
