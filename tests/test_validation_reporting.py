import unittest

from scripts.compare_validation_benchmarks import _case_measurement


class ValidationReportingTests(unittest.TestCase):
    def test_measurement_uses_trace_statistics_for_retrieval_counts(self):
        case = {
            "case_id": "case_1",
            "query": "example",
            "status": "completed",
            "trace": {
                "stats": {
                    "action_counts": {"web_search": 2},
                    "page_fetches": 3,
                    "chunk_count": 4,
                    "page_evidence_statuses": {"ok": 2, "error": 1},
                },
                "events": [
                    {"type": "tool_call"},
                    {"type": "model_tool_decision"},
                    {"type": "model_tool_decision"},
                    {"type": "web_search_chunk"},
                    {"type": "final"},
                ],
            },
        }

        measurement = _case_measurement(case)

        self.assertEqual(measurement["tool_calls"], 1)
        self.assertEqual(measurement["tool_decision_events"], 2)
        self.assertEqual(measurement["search_events"], 2)
        self.assertEqual(measurement["fetch_events"], 3)
        self.assertEqual(measurement["chunk_events"], 4)
        self.assertEqual(measurement["usable_evidence_count"], 2)


if __name__ == "__main__":
    unittest.main()
