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

    def test_explicit_negative_locator_rejects_page_body(self):
        evidence = {
            "status": "no_evidence",
            "page_chars": len(self.body),
            "chunk_count": 1,
            "candidates": [],
            "compact_facts": "",
            "source_chunks": [{"chunk_id": "chunk-1", "text": self.body}],
            "valid_json_count": 1,
            "negative_response_count": 1,
            "all_chunks_valid_negative": True,
            "errors": [],
        }
        with patch("tools.web_search_generic.extract_single_page_evidence", return_value=evidence):
            record, page_evidence = _compact_page(
                "a different fact that is absent",
                self.candidate,
                self.fetched,
                object(),
                "CHUNK_EXPLICIT_NEGATIVE_TEST",
            )

        self.assertIsNone(record)
        self.assertEqual(page_evidence["status"], "no_evidence")
        self.assertEqual(page_evidence["model_extraction_status"], "negative")
        self.assertFalse(page_evidence["deterministic_chunk_fallback"])
        self.assertTrue(page_evidence["source_body_available"])

    def test_selected_chunk_negative_rejects_uninspected_body_fallback(self):
        evidence = {
            "status": "no_evidence",
            "page_chars": len(self.body),
            "chunk_count": 4,
            "inspected_chunk_count": 2,
            "candidates": [],
            "compact_facts": "",
            "source_chunks": [{"chunk_id": "chunk-1", "text": self.body}],
            "valid_json_count": 2,
            "negative_response_count": 2,
            "all_chunks_valid_negative": False,
            "all_selected_chunks_valid_negative": True,
            "errors": [],
        }
        with patch("tools.web_search_generic.extract_single_page_evidence", return_value=evidence):
            record, page_evidence = _compact_page(
                "a fact absent from the selected relevant chunks",
                self.candidate,
                self.fetched,
                object(),
                "CHUNK_SELECTED_NEGATIVE_TEST",
            )

        self.assertIsNone(record)
        self.assertEqual(page_evidence["model_extraction_status"], "negative")
        self.assertFalse(page_evidence["deterministic_chunk_fallback"])

    def test_post_gate_semantic_rejection_does_not_restore_page_body(self):
        evidence = {
            "status": "no_evidence",
            "page_chars": len(self.body),
            "chunk_count": 1,
            "inspected_chunk_count": 1,
            "candidates": [],
            "compact_facts": "",
            "source_chunks": [{"chunk_id": "chunk-1", "text": self.body}],
            "valid_json_count": 1,
            "valid_contract_count": 1,
            "negative_response_count": 0,
            "all_selected_chunks_valid_negative": False,
            "all_selected_chunks_semantically_rejected": True,
            "errors": [],
        }
        with patch("tools.web_search_generic.extract_single_page_evidence", return_value=evidence):
            record, page_evidence = _compact_page(
                "a target product fact absent from this other product page",
                self.candidate,
                self.fetched,
                object(),
                "CHUNK_SEMANTIC_REJECTION_TEST",
            )

        self.assertIsNone(record)
        self.assertEqual(page_evidence["model_extraction_status"], "semantically_rejected")
        self.assertEqual(page_evidence["evidence_origin"], "rejected_fetched_page_body")
        self.assertFalse(page_evidence["deterministic_chunk_fallback"])

    def test_long_body_with_failed_extraction_keeps_bounded_original_chunks(self):
        evidence = {
            "status": "error",
            "error_class": "chunk_extraction_failed",
            "page_chars": len(self.body),
            "chunk_count": 2,
            "candidates": [],
            "compact_facts": "",
            "source_chunks": [],
            "errors": ["chunk extraction returned no usable model output"],
        }
        with patch("tools.web_search_generic.extract_single_page_evidence", return_value=evidence):
            record, page_evidence = _compact_page(
                "find the fact",
                self.candidate,
                self.fetched,
                object(),
                "CHUNK_LONG_FAILURE_TEST",
            )

        self.assertIsNotNone(record)
        self.assertEqual(page_evidence["status"], "ok")
        self.assertTrue(page_evidence["deterministic_chunk_fallback"])
        self.assertLessEqual(len(record["selected_source_chunks"]), 3)

    def test_long_single_chunk_failure_keeps_original_span(self):
        long_body = "The fetched page contains the requested fact and extensive surrounding navigation. " * 100
        evidence = {
            "status": "error",
            "error_class": "chunk_extraction_failed",
            "page_chars": len(long_body),
            "chunk_count": 1,
            "source_excerpt": long_body,
            "candidates": [],
            "compact_facts": "",
            "source_chunks": [{"chunk_id": "chunk-1", "text": long_body}],
            "errors": ["chunk extraction returned no usable model output"],
        }
        fetched = {"status": "ok", "results": [{**self.fetched["results"][0], "page_excerpt": long_body}]}
        with patch("tools.web_search_generic.extract_single_page_evidence", return_value=evidence):
            record, page_evidence = _compact_page(
                "find the fact",
                self.candidate,
                fetched,
                object(),
                "CHUNK_LONG_SINGLE_FAILURE_TEST",
            )

        self.assertIsNotNone(record)
        self.assertEqual(page_evidence["status"], "ok")
        self.assertEqual(record["evidence_status"], "body_fallback")

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

    def test_page_merge_event_records_post_grounding_candidates(self):
        evidence = {
            "status": "ok",
            "page_chars": len(self.body),
            "chunk_count": 1,
            "inspected_chunk_count": 1,
            "chunk_window_tokens": 128,
            "chunk_candidates": [
                {
                    "chunk_id": "chunk-1",
                    "supported": True,
                    "source_grounded": True,
                    "quote": "The fetched page contains the requested fact.",
                },
                {
                    "chunk_id": "chunk-2",
                    "supported": False,
                    "source_grounded": False,
                    "rejection_reason": "model_quote_not_grounded",
                    "model_quote": "invented fact",
                },
            ],
            "candidates": [
                {
                    "chunk_id": "chunk-1",
                    "supported": True,
                    "source_grounded": True,
                    "quote": "The fetched page contains the requested fact.",
                }
            ],
            "compact_facts": "[chunk-1] The fetched page contains the requested fact.",
            "source_chunks": [{"chunk_id": "chunk-1", "text": self.body}],
            "parallel_candidate": {"worker_count": 1},
            "errors": [],
        }
        with (
            patch("tools.web_search_generic.extract_single_page_evidence", return_value=evidence),
            patch("tools.web_search_generic.append_task_event") as append_event,
        ):
            record, _ = _compact_page(
                "find the fact",
                self.candidate,
                self.fetched,
                object(),
                "CHUNK_MERGE_EVENT_TEST",
            )

        self.assertIsNotNone(record)
        event_call = next(
            call
            for call in append_event.call_args_list
            if len(call.args) >= 2 and call.args[1] == "page_candidate_merge"
        )
        self.assertEqual(event_call.kwargs["data"]["grounded_candidate_count"], 1)
        self.assertEqual(event_call.kwargs["data"]["rejected_ungrounded_count"], 1)
        self.assertEqual(len(event_call.kwargs["chunk_candidates"]), 2)
        self.assertEqual(len(event_call.kwargs["candidates"]), 1)

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

    def test_routing_projection_excludes_rejected_model_facts_before_slicing(self):
        rejected = [
            {
                "chunk_id": f"chunk-{index}",
                "supported": False,
                "facts": [f"invented planner fact {index}"],
                "quote": f"invented quote {index}",
                "rejection_reason": "model_quote_not_grounded",
            }
            for index in range(1, 6)
        ]
        grounded = {
            "chunk_id": "chunk-6",
            "supported": True,
            "source_grounded": True,
            "facts": ["invented paraphrase must stay out"],
            "quote": "The exact source says fetch remained experimental.",
        }
        routing = Planner._compact_routing_observation(
            {
                "status": "ok",
                "results": [
                    {
                        "title": "Official source",
                        "url": "https://example.com/source",
                        "evidence_status": "ok",
                        "model_extracted_facts": "[chunk-6] The exact source says fetch remained experimental.",
                        "chunk_candidates": [*rejected, grounded],
                        "source_chunks": [
                            {
                                "chunk_id": "chunk-6",
                                "text": "The exact source says fetch remained experimental.",
                            }
                        ],
                    }
                ],
            }
        )

        self.assertIn("fetch remained experimental", routing)
        self.assertNotIn("invented planner fact", routing)
        self.assertNotIn("invented paraphrase", routing)


if __name__ == "__main__":
    unittest.main()
