from __future__ import annotations

import unittest

from agent.retrieval_loop import merge_retrieval_results
from agent.retrieval_synthesis import (
    _clean_answer,
    _enforce_citation_contract,
    _enforce_risk_contract,
    _needs_answer_repair,
    build_evidence_context,
    synthesize_retrieval_answer,
)
from tools.web_search_keyless import _is_search_result_url
from utils.experiment_strategies import normalize_strategy


class _FakeModel:
    def __init__(self) -> None:
        self.prompt = ""

    def text_completion(self, prompt: str, max_tokens: int = 384):
        from types import SimpleNamespace

        self.prompt = prompt
        return SimpleNamespace(content="Supported answer [S1].")


class ExperimentStrategyTests(unittest.TestCase):
    def test_search_page_urls_are_filtered_before_citation(self):
        self.assertTrue(_is_search_result_url("https://global.bing.com/dict/search?q=python"))
        self.assertTrue(_is_search_result_url("https://www.baidu.com/"))
        self.assertFalse(_is_search_result_url("https://docs.python.org/3/"))

    def test_answer_cleanup_preserves_inline_citations(self):
        cleaned = _clean_answer(
            "**Answer:** The fact is supported [S1].\n\n**Key Evidence:**\n- copied source block"
        )
        self.assertEqual(cleaned, "The fact is supported [S1].")
        self.assertFalse(_needs_answer_repair(cleaned))

    def test_final_answer_cannot_emit_unverified_url_without_a_source_marker(self):
        data = {
            "citation_refs": [{"ref_id": "S1", "url": "https://example.com/source"}],
        }
        context = {"selected_evidence": [{"ref_id": "S1", "url": "https://example.com/source"}]}
        answer = _enforce_citation_contract(
            "The source is https://untrusted.example.invalid/page.", data, context
        )
        self.assertNotIn("unverified URL", answer)
        self.assertTrue(answer.endswith("[S1](https://example.com/source)"))

    def test_high_risk_answer_gets_a_deterministic_boundary(self):
        answer = _enforce_risk_contract(
            "A retrieved medical fact [S1].",
            {"domain": "medicine_literacy"},
        )
        self.assertIn("not medical advice", answer)
        self.assertIn("consult a qualified professional", answer)

    def test_strategy_defaults_and_bounds_are_validated(self):
        self.assertEqual(normalize_strategy()["ranking_strategy"], "evidence_quality.v1")
        self.assertEqual(normalize_strategy({"context_source_count": 3})["context_source_count"], 3)
        with self.assertRaises(ValueError):
            normalize_strategy({"context_source_count": 5})
        with self.assertRaises(ValueError):
            normalize_strategy({"untracked_variable": "value"})

    def test_ranking_strategy_is_a_real_controlled_change(self):
        rounds = [
            (
                "query-a",
                {
                    "results": [
                        {"title": "A", "url": "https://example.com/a", "page_excerpt": "A supported evidence body with enough text for the source boundary."},
                        {"title": "B", "url": "https://example.com/b", "page_excerpt": "B supported evidence body with enough text for the source boundary."},
                    ],
                },
            ),
            (
                "query-b",
                {
                    "results": [
                        {"title": "C", "url": "https://example.com/c", "page_excerpt": "C supported evidence body with enough text for the source boundary."},
                        {"title": "B", "url": "https://example.com/b", "page_excerpt": "B supported evidence body with enough text for the source boundary."},
                    ],
                },
            ),
        ]
        support = merge_retrieval_results("query", "search_web_keyless", rounds, ranking_strategy="candidate_support_then_rank.v1")
        best_rank = merge_retrieval_results("query", "search_web_keyless", rounds, ranking_strategy="best_rank.v1")
        self.assertEqual(support["ranking_strategy"], "candidate_support_then_rank.v1")
        self.assertEqual(best_rank["ranking_strategy"], "best_rank.v1")
        self.assertEqual(support["results"][0]["url"], "https://example.com/b")
        self.assertEqual(best_rank["results"][0]["url"], "https://example.com/a")

    def test_evidence_quality_ranking_prefers_relevant_captured_body(self):
        rounds = [
            (
                "Python official documentation",
                {
                    "results": [
                {"title": "Unrelated welcome page", "url": "https://example.com/welcome", "page_excerpt": "A generic welcome page body with no requested documentation facts."},
                        {
                            "title": "Python documentation",
                            "url": "https://docs.python.org/",
                            "snippet": "Python documentation",
                            "page_excerpt": "Python documentation and library reference with the requested official API details.",
                        },
                    ],
                },
            )
        ]
        result = merge_retrieval_results(
            "Python documentation",
            "search_web_keyless",
            rounds,
            ranking_strategy="evidence_quality.v1",
        )
        self.assertEqual(result["results"][0]["url"], "https://docs.python.org/")
        self.assertEqual(result["results"][0]["ranking_method"], "evidence_quality.v1")

    def test_context_and_prompt_variants_are_recorded(self):
        data = {
            "query": "evidence",
            "results": [
                {"title": "A", "url": "https://example.com/a", "page_excerpt": "This is a supported evidence fact from source A with sufficient body text."},
                {"title": "B", "url": "https://example.com/b", "page_excerpt": "This is a supported evidence fact from source B with sufficient body text."},
                {"title": "C", "url": "https://example.com/c", "page_excerpt": "This is a supported evidence fact from source C with sufficient body text."},
            ],
            "citation_refs": [],
        }
        context = build_evidence_context(data, {"strategy_config": {"context_source_count": 2}})
        self.assertEqual(len(context["selected_evidence"]), 2)
        model = _FakeModel()
        result = synthesize_retrieval_answer(
            "evidence",
            data,
            llm=model,
            constraints={"strategy_config": {"prompt_variant": "citation_first.v1"}},
        )
        self.assertIn("For every factual claim", result["prompt"])
        self.assertEqual(result["context_stats"]["strategy"]["prompt_variant"], "citation_first.v1")

    def test_chinese_multi_part_query_keeps_multiple_sources_in_context(self):
        data = {
            "query": "RWKV创始人、论文和GitHub项目链接是什么？",
            "results": [
                {"title": "Founder", "url": "https://example.com/founder", "page_excerpt": "The founder evidence is directly stated in this source body."},
                {"title": "Papers", "url": "https://example.com/papers", "page_excerpt": "The paper evidence is directly stated in this source body."},
                {"title": "Projects", "url": "https://example.com/projects", "page_excerpt": "The project evidence is directly stated in this source body."},
            ],
            "citation_refs": [],
        }
        context = build_evidence_context(data)
        self.assertEqual(len(context["selected_evidence"]), 3)

    def test_answer_citations_are_limited_to_selected_context_sources(self):
        data = {
            "query": "evidence",
            "results": [
                {"title": "Selected", "url": "https://example.com/selected", "page_excerpt": "The selected source contains a directly supported evidence fact."},
                {"title": "Noise", "url": "https://example.com/noise", "page_excerpt": "The noise source contains an unrelated evidence paragraph."},
            ],
            "citation_refs": [
                {"ref_id": "S1", "title": "Selected", "url": "https://example.com/selected"},
                {"ref_id": "S2", "title": "Noise", "url": "https://example.com/noise"},
            ],
        }
        result = synthesize_retrieval_answer("evidence", data, llm=_FakeModel())
        self.assertEqual([item["url"] for item in result["citation_refs"]], ["https://example.com/selected"])


if __name__ == "__main__":
    unittest.main()
