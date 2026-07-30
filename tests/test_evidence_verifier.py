import unittest
from types import SimpleNamespace

from agent.evidence_verifier import _normalize_result, build_verifier_prompt, verify_evidence


class _FakeVerifier:
    provider = "local_13b"

    def __init__(self, content):
        self.content = content
        self.calls = []

    def text_completion(self, prompt, max_tokens=0, **kwargs):
        self.calls.append((prompt, max_tokens, kwargs))
        return SimpleNamespace(content=self.content)


class EvidenceVerifierTests(unittest.TestCase):
    def _report(self, status="supported"):
        return {
            "subquestion_coverage": [{"point_id": "P1", "status": status}],
            "cross_source": {"missing_points": 0},
        }

    def test_normalized_result_drops_answer_and_marks_control_only(self):
        result = _normalize_result(
            {
                "status": "supported",
                "answer": "This must never be forwarded as the answer.",
                "points": [{"id": "P1", "status": "supported", "evidence": ["S1"]}],
            },
            report=self._report(),
            source_count=1,
        )
        self.assertNotIn("answer", result)
        self.assertFalse(result["is_truth_judgement"])
        self.assertEqual(result["points"][0]["evidence"], ["S1"])

    def test_mechanical_missing_overrides_model_supported(self):
        result = _normalize_result(
            {
                "status": "supported",
                "points": [{"id": "P1", "status": "supported", "evidence": ["S1"]}],
            },
            report=self._report(status="missing"),
            source_count=1,
        )
        self.assertEqual(result["status"], "needs_more_evidence")
        self.assertFalse(result["completion_ready"])
        self.assertEqual(result["missing_point_ids"], ["P1"])

    def test_verifier_prompt_and_transport_are_control_only(self):
        prompt = build_verifier_prompt(
            "Who is the founder?",
            {"atomic_points": [{"id": "P1", "task": "founder"}]},
            "BEGIN EVIDENCE BODY\nS1 body\nEND EVIDENCE BODY",
            self._report(),
        )
        self.assertIn("Do not answer the user", prompt)
        self.assertIn("Return exactly one JSON object", prompt)

        llm = _FakeVerifier(
            '{"status":"supported","answer":"secret factual answer",'
            '"points":[{"id":"P1","status":"supported","evidence":["S1"]}]}'
        )
        result = verify_evidence(
            llm,
            query="Who is the founder?",
            task_plan={"atomic_points": [{"id": "P1", "task": "founder"}]},
            evidence_context={
                "text": "BEGIN EVIDENCE BODY\nS1 body\nEND EVIDENCE BODY",
                "selected_evidence": [{"url": "https://example.com"}],
                "validation": self._report(),
            },
        )
        self.assertNotIn("answer", result)
        self.assertFalse(result["is_truth_judgement"])
        self.assertEqual(result["status"], "supported")
        self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
