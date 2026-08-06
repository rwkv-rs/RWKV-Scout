import unittest
from unittest.mock import patch

from agent.planner import Planner
from tools.web_search_generic import _compact_page


class ChunkContractTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "url": "https://example.com/fact",
            "title": "Fact page",
            "snippet": "fact",
            "source": "test",
            "candidate_rank": 1,
        }
        self.body = "The fetched page contains the requested fact. " * 20
        self.fetched = {
            "status": "ok",
            "results": [
                {
                    "url": self.candidate["url"],
                    "title": self.candidate["title"],
                    "page_excerpt": self.body,
                }
            ],
        }

    def test_empty_chunk_output_keeps_fetched_body_as_evidence(self):
        evidence = {
            "status": "error",
            "error_class": "chunk_extraction_failed",
            "page_chars": len(self.body),
            "chunk_count": 1,
            "candidates": [],
            "compact_facts": "",
            "source_chunks": [{"chunk_id": "chunk-1", "text": self.body}],
            "errors": ["chunk extraction returned no usable model output"],
        }
        with patch("tools.web_search_generic.extract_single_page_evidence", return_value=evidence):
            record, page_evidence = _compact_page(
                "find the fact",
                self.candidate,
                self.fetched,
                object(),
                "CHUNK_EMPTY_TEST",
            )

        self.assertIsNotNone(record)
        self.assertEqual(page_evidence["status"], "ok")
        self.assertEqual(page_evidence["error_class"], "chunk_extraction_failed")
        self.assertTrue(page_evidence["source_body_available"])
        self.assertEqual(page_evidence["model_extraction_status"], "error")

    def test_empty_locator_keeps_fetched_body_as_evidence(self):
        evidence = {
            "status": "no_evidence",
            "page_chars": len(self.body),
            "chunk_count": 1,
            "candidates": [],
            "compact_facts": "",
            "source_chunks": [{"chunk_id": "chunk-1", "text": self.body}],
            "errors": [],
        }
        with patch("tools.web_search_generic.extract_single_page_evidence", return_value=evidence):
            record, page_evidence = _compact_page(
                "find the fact",
                self.candidate,
                self.fetched,
                object(),
                "CHUNK_NO_EVIDENCE_TEST",
            )

        self.assertIsNotNone(record)
        self.assertEqual(page_evidence["status"], "ok")
        self.assertEqual(page_evidence["error_class"], "")
        self.assertEqual(page_evidence["model_extraction_status"], "empty")

    def test_successful_chunk_output_becomes_a_record(self):
        evidence = {
            "status": "ok",
            "page_chars": len(self.body),
            "chunk_count": 1,
            "candidates": [{"chunk_id": "chunk-1", "facts": ["the requested fact"]}],
            "compact_facts": "[chunk-1] the requested fact",
            "source_chunks": [{"chunk_id": "chunk-1", "text": self.body}],
            "errors": [],
        }
        with patch("tools.web_search_generic.extract_single_page_evidence", return_value=evidence):
            record, page_evidence = _compact_page(
                "find the fact",
                self.candidate,
                self.fetched,
                object(),
                "CHUNK_SUCCESS_TEST",
            )

        self.assertIsNotNone(record)
        self.assertEqual(page_evidence["status"], "ok")
        self.assertEqual(page_evidence["model_extraction_status"], "ok")
        self.assertEqual(record["model_extracted_facts"], "[chunk-1] the requested fact")

    def test_routing_projection_retains_bounded_chunk_context(self):
        routing = Planner._compact_routing_observation(
            {
                "status": "ok",
                "retrieval_role": "evidence",
                "results": [
                    {
                        "title": "Fact page",
                        "url": "https://example.com/fact",
                        "evidence_status": "ok",
                        "model_extracted_facts": "[chunk-1] release date is 2025-04-16",
                        "chunk_candidates": [
                            {
                                "chunk_id": "chunk-1",
                                "facts": ["release date is 2025-04-16"],
                                "quote": "The release date is 2025-04-16.",
                            }
                        ],
                        "source_chunks": [
                            {
                                "chunk_id": "chunk-1",
                                "text": "The release date is 2025-04-16.",
                            }
                        ],
                    }
                ],
            }
        )

        self.assertIn("evidence_context", routing)
        self.assertIn("2025-04-16", routing)
        self.assertIn("fetched page chunks", routing)


if __name__ == "__main__":
    unittest.main()
