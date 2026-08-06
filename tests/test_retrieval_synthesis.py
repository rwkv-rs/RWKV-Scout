import unittest
from types import SimpleNamespace

from agent.retrieval_synthesis import (
    _attach_mechanical_citations,
    _clean_answer,
    _final_completion_budget,
    _normalize_evidence_code_spans,
    _normalize_free_threading_polarity,
    _evidence_limited_fallback,
    build_evidence_context,
    synthesize_retrieval_answer,
)
from config import get_llm_context_length
from utils.chunker import get_token_count


class _FakeLLM:
    provider = "local_13b"

    def __init__(self):
        self.calls = []

    def text_completion(self, prompt, max_tokens=0, **kwargs):
        self.calls.append((prompt, max_tokens, kwargs))
        return SimpleNamespace(content="Supported answer [S1]")


class _EmptyLLM:
    provider = "local_13b"

    def text_completion(self, prompt, max_tokens=0, **kwargs):
        return SimpleNamespace(content="")


class _ErrorLLM:
    provider = "local_13b"

    def text_completion(self, prompt, max_tokens=0, **kwargs):
        raise TimeoutError("simulated RWKV final timeout")


class RetrievalSynthesisTests(unittest.TestCase):
    def test_empty_rwkv_output_becomes_a_nonempty_evidence_bounded_answer(self):
        result = synthesize_retrieval_answer(
            "Who founded the project?",
            {
                "query": "Who founded the project?",
                "results": [
                    {
                        "title": "Project source",
                        "url": "https://example.com/project",
                        "content": "The project was founded by Example Research in 2024. " * 8,
                        "evidence_origin": "fetched_page_body",
                    }
                ],
            },
            llm=_EmptyLLM(),
        )
        self.assertTrue(result["content"].strip())
        self.assertIn(result["mode"], {"controller_fallback", "controller_refusal"})
        self.assertTrue(result["answer_quality"]["fallback_used"])
        self.assertNotIn("Local RWKV", result["content"])

    def test_rwkv_final_exception_becomes_a_nonempty_answer_and_keeps_diagnostic(self):
        result = synthesize_retrieval_answer(
            "Who founded the project?",
            {
                "query": "Who founded the project?",
                "results": [
                    {
                        "title": "Project source",
                        "url": "https://example.com/project",
                        "content": "The project was founded by Example Research in 2024. " * 8,
                        "evidence_origin": "fetched_page_body",
                    }
                ],
            },
            llm=_ErrorLLM(),
        )
        self.assertTrue(result["content"].strip())
        self.assertIn(result["mode"], {"controller_fallback", "controller_refusal"})
        self.assertIn("TimeoutError", result["model_error"])
        self.assertNotIn("TimeoutError", result["content"])

    def test_empty_rwkv_output_without_evidence_is_an_explicit_nonempty_refusal(self):
        result = synthesize_retrieval_answer(
            "Who founded the project?",
            {
                "query": "Who founded the project?",
                "results": [],
            },
            llm=_EmptyLLM(),
        )
        self.assertTrue(result["content"].strip())
        self.assertEqual(result["mode"], "controller_refusal")
        self.assertEqual(result["answer_quality"]["fallback_kind"], "refusal")

    def test_controller_fallback_quotes_visible_evidence_when_model_is_unavailable(self):
        result = _evidence_limited_fallback(
            "What does the source state?",
            {
                "usable_evidence_count": 1,
                "selected_evidence": [
                    {
                        "ref_id": "S1",
                        "evidence_text": "The source explicitly states the project launched in 2024.",
                    }
                ],
                "validation": {"subquestion_coverage": []},
            },
            reason="rwkv_empty_output",
        )
        self.assertEqual(result["mode"], "controller_fallback")
        self.assertIn("launched in 2024", result["content"])

    def test_answer_first_contract_is_used_only_with_visible_evidence(self):
        llm = _FakeLLM()
        result = synthesize_retrieval_answer(
            "核验这句话是否正确，并计算相隔多少天",
            {
                "query": "核验这句话是否正确，并计算相隔多少天",
                "results": [
                    {
                        "title": "Grounded source",
                        "url": "https://example.com/source",
                        "content": "The source states the two dates and the corrected claim. " * 20,
                        "evidence_origin": "fetched_page_body",
                    }
                ],
            },
            llm=llm,
        )
        self.assertIn("ANSWER FIRST", llm.calls[0][0])
        self.assertIn("CLAIM CHECK", llm.calls[0][0])
        self.assertIn("CALCULATION:", llm.calls[0][0])
        self.assertIn("routing metadata", llm.calls[0][0])
        self.assertTrue(result["context_stats"]["final_usable_evidence_count"])

    def test_near_miss_code_span_is_restored_from_evidence(self):
        answer = "Use `sys.is_gil_enabled()` to inspect the interpreter."
        evidence = "The `sys._is_gil_enabled()` function checks whether the GIL is disabled."
        self.assertEqual(
            _normalize_evidence_code_spans(answer, evidence),
            "Use `sys._is_gil_enabled()` to inspect the interpreter.",
        )
        self.assertEqual(
            _normalize_evidence_code_spans(
                "Use `sys.version_info` for the version.",
                "Use `sys.version` or `sys.version_info` for the version.",
            ),
            "Use `sys.version_info` for the version.",
        )

    def test_explicit_free_threading_runtime_direction_is_preserved(self):
        answer = (
            "The GIL can be disabled at runtime with the environment variable `PYTHON_GIL` "
            "or the command-line option `-Xgil`."
        )
        evidence = (
            "Free-threaded builds support optionally running with the GIL enabled at runtime "
            "using the environment variable `PYTHON_GIL` or the command-line option `-Xgil`."
        )
        corrected = _normalize_free_threading_polarity(
            answer,
            "python free threading how to enable",
            evidence,
        )
        self.assertIn("run with the GIL enabled at runtime", corrected)

    def test_aligned_uncited_fact_gets_existing_source_reference(self):
        answer = _attach_mechanical_citations(
            "The source directly states that the release date is 2024-01-01.",
            [
                {
                    "url": "https://example.com/fact",
                    "content": "The source directly states that the release date is 2024-01-01.",
                    "evidence_origin": "fetched_page_body",
                }
            ],
        )
        self.assertIn("[S1]", answer)

    def test_evidence_protocol_is_not_a_user_facing_answer(self):
        self.assertEqual(
            _clean_answer(
                "BEGIN EVIDENCE SOURCE S1\n"
                "URL (citation metadata only): https://example.com\n"
                "EVIDENCE BODY\nSome source text"
            ),
            "",
        )

    def test_clean_answer_removes_continuation_xml_scaffolding(self):
        self.assertEqual(
            _clean_answer("<response>\n<answer>Actual answer [S1]</answer>\n</response>"),
            "Actual answer [S1]",
        )

    def test_clean_answer_collapses_budget_exhaustion_list_repeats(self):
        cleaned = _clean_answer(
            "1. Enable the module.\n"
            "2. Enable the module.\n"
            "3. Enable the module.\n"
            "4. Enable the module.\n"
            "5. Enable the module."
        )
        self.assertEqual(cleaned, "1. Enable the module.")
        self.assertEqual(
            _clean_answer(
                "[S1](https://example.com)\n\n"
                "EVIDENCE BODY (the only factual source)\n"
                "Copied page text"
            ),
            "",
        )
        self.assertEqual(
            _clean_answer("P1: internal routing point\nP2: another point\nUser-facing answer."),
            "User-facing answer.",
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
                    "task_mode": "latest_list",
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
        self.assertNotIn("The fixture fact is 42", llm.calls[0][0])
        self.assertIn("[S1](https://example.com/stations)", result["content"])

    def test_final_summary_keeps_execution_trace_out_of_evidence_prompt(self):
        llm = _FakeLLM()
        result = synthesize_retrieval_answer(
            "find a fact",
            {"query": "find a fact", "results": [], "citation_refs": []},
            llm=llm,
            execution_context="RWKV planner transcript:\nAssistant: search_mediawiki\nFunction output: status=ok",
            termination_reason="max_steps_reached",
        )
        self.assertNotIn("max_steps_reached", llm.calls[0][0])
        self.assertNotIn("search_mediawiki", llm.calls[0][0])
        self.assertNotIn("status=ok", result["context_text"])
        self.assertFalse(result["context_stats"]["execution_context_in_final_prompt"])

    def test_source_packing_is_not_reported_as_final_prompt_truncation(self):
        context = build_evidence_context(
            {
                "query": "find the fact",
                "results": [
                    {
                        "title": "Long source",
                        "url": "https://example.com/long",
                        "content": "A directly supported fact. " * 6000,
                        "source": "test",
                    }
                ],
            },
            query="find the fact",
        )
        self.assertTrue(context["source_context_truncated"])
        self.assertTrue(context["context_truncated"])
        self.assertFalse(context["final_context_truncated"])

    def test_no_usable_evidence_uses_short_closed_world_hint(self):
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
        self.assertIn("state that briefly", llm.calls[0][0])
        self.assertNotIn("only valid answer is an explicit refusal", llm.calls[0][0])
        self.assertEqual(result["citation_refs"], [])
        self.assertTrue(result["answer_quality"]["closed_world_boundary_enforced"])
        self.assertIn("无法根据当前检索到的正文证据确认", result["content"])

    def test_generic_plan_words_do_not_count_as_fact_coverage(self):
        context = build_evidence_context(
            {
                "query": "RWKV founder papers GitHub projects",
                "results": [
                    {
                        "title": "GitHub profile",
                        "url": "https://github.com/example",
                        "content": (
                            "RWKV GitHub projects repositories search results source stars forks. "
                            * 80
                        ),
                        "evidence_origin": "fetched_page_body",
                    }
                ],
            },
            query="RWKV founder papers GitHub projects",
            constraints={
                "task_plan": {
                    "atomic_points": [
                        {
                            "id": "P3",
                            "task": "List all GitHub projects created by the founder(s) of RWKV",
                            "objective": "repository ownership and metadata",
                            "evidence_needed": ["GitHub repository URL and metadata"],
                            "acceptance_criteria": ["Each repository is owned by the founder(s)"],
                        }
                    ]
                }
            },
        )
        row = context["validation"]["subquestion_coverage"][0]
        self.assertEqual(row["status"], "missing")

    def test_product_homepage_does_not_cover_a_specific_feature(self):
        context = build_evidence_context(
            {
                "query": "Docker Compose GPU configuration",
                "results": [
                    {
                        "title": "Docker documentation",
                        "url": "https://docs.docker.com/",
                        "content": (
                            "Docker documentation. Docker Compose helps define and run applications. "
                            "Browse the reference and guides for containers. "
                            * 80
                        ),
                        "evidence_origin": "fetched_page_body",
                    }
                ],
            },
            query="Docker Compose GPU configuration",
            constraints={
                "task_plan": {
                    "atomic_points": [
                        {
                            "id": "P1",
                            "task": "Find the exact Docker Compose GPU configuration syntax",
                            "objective": "Expose a GPU to a container",
                        }
                    ]
                }
            },
        )
        row = context["validation"]["subquestion_coverage"][0]
        self.assertEqual(row["status"], "missing")

    def test_irrelevant_substantive_pages_remain_visible_for_rwkv_decision(self):
        llm = _FakeLLM()
        result = synthesize_retrieval_answer(
            "specific requested fact",
            {
                "query": "specific requested fact",
                "results": [
                    {
                        "title": "Unrelated page",
                        "url": "https://example.com/unrelated",
                        "content": "This is a long unrelated page body about gardening and weather. " * 40,
                        "evidence_origin": "fetched_page_body",
                    }
                ],
                "citation_refs": [],
            },
            llm=llm,
            constraints={
                "task_plan": {
                    "atomic_points": [
                        {
                            "id": "P1",
                            "task": "find the requested fact",
                            "objective": "specific requested fact",
                            "evidence_needed": ["the exact fact"],
                            "acceptance_criteria": ["directly stated"],
                        }
                    ]
                }
            },
        )
        # The engineering validator may report that the requested point is
        # still missing, but it must not erase fetched page evidence before
        # RWKV gets to decide whether to cross-check or re-plan.
        self.assertIn("PARTIAL_EVIDENCE", llm.calls[0][0])
        self.assertIn("gardening and weather", llm.calls[0][0])
        self.assertFalse(result["context_stats"]["final_evidence_suppressed"])
        self.assertEqual(result["context_stats"]["retrieved_usable_evidence_count"], 1)
        self.assertEqual(result["context_stats"]["final_usable_evidence_count"], 1)
        self.assertEqual(result["context_stats"]["final_selected_evidence_count"], 1)
        self.assertEqual(len(result["citation_refs"]), 1)

    def test_citation_ref_binds_url_to_selected_chunk_locator(self):
        llm = _FakeLLM()
        result = synthesize_retrieval_answer(
            "specific requested fact",
            {
                "query": "specific requested fact",
                "results": [
                    {
                        "title": "Grounded page",
                        "url": "https://example.com/fact",
                        "content": "The requested fact is stated on this page. " * 20,
                        "evidence_origin": "fetched_page_body",
                        "evidence_boundary": "page_body_only",
                    }
                ],
                "citation_refs": [],
            },
            llm=llm,
        )
        ref = result["citation_refs"][0]
        locator = ref["evidence_locator"]
        self.assertEqual(ref["url"], "https://example.com/fact")
        self.assertEqual(locator["url"], ref["url"])
        self.assertTrue(locator["spans"])
        self.assertTrue(locator["spans"][0]["span_id"].startswith(f"{ref['ref_id']}:C"))


if __name__ == "__main__":
    unittest.main()
