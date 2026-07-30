from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.orchestrator import Orchestrator
from agent.planner import Planner
from utils.retrieval_ledger import RetrievalLedger, canonical_url, normalize_query, query_similarity


class RetrievalLedgerTests(unittest.TestCase):
    def test_normalization_is_exact_repeat_accounting_not_semantic_rewriting(self):
        self.assertEqual(normalize_query("  RWKV 公司地址？ "), "rwkv 公司地址")
        self.assertEqual(
            canonical_url("HTTPS://Example.com/page?utm_source=x&id=7#section"),
            "https://example.com/page?id=7",
        )

    def test_record_exposes_new_urls_and_exact_repeat_delta(self):
        ledger = RetrievalLedger()
        first = ledger.record(
            "RWKV 项目 GitHub 链接",
            {
                "status": "ok",
                "results": [{"url": "https://example.com/a"}],
                "page_evidence": [{"url": "https://example.com/a", "status": "ok"}],
            },
            step=1,
            action="web_search",
        )
        second = ledger.record(
            "RWKV 项目 GitHub 链接",
            {
                "status": "ok",
                "results": [{"url": "https://example.com/a#fragment"}],
                "page_evidence": [{"url": "https://example.com/a", "status": "ok"}],
            },
            step=2,
            branch_id="B1",
            action="web_search",
        )
        self.assertEqual(first["new_url_count"], 1)
        self.assertFalse(first["exact_repeat"])
        self.assertEqual(second["new_url_count"], 0)
        self.assertTrue(second["exact_repeat"])
        observation = ledger.observation(branch_id="B1")
        self.assertEqual(observation["total_searches"], 2)
        self.assertEqual(observation["unique_queries"], 1)
        self.assertEqual(observation["exact_repeat_count"], 1)
        self.assertEqual(observation["retrieved_url_count"], 1)

    def test_query_status_marks_successful_searches_as_reusable_duplicates(self):
        ledger = RetrievalLedger()
        ledger.record(
            "  RWKV official GitHub  ",
            {"status": "ok", "results": [{"url": "https://example.com/rwkv"}]},
            step=1,
            action="web_search",
        )
        status = ledger.query_status("rwkv official github")
        self.assertTrue(status["attempted"])
        self.assertEqual(status["count"], 1)
        self.assertEqual(status["last"]["new_url_count"], 1)

    def test_blocked_duplicate_is_observable_without_counting_as_a_search(self):
        ledger = RetrievalLedger()
        ledger.record(
            "official RWKV founder",
            {"status": "ok", "results": [{"url": "https://example.com/founder"}]},
            step=1,
            action="web_search",
        )
        blocked = ledger.record_duplicate_block("official RWKV founder", step=2)
        self.assertEqual(blocked["blocked_count"], 1)
        observation = ledger.observation()
        self.assertEqual(observation["total_searches"], 1)
        self.assertEqual(observation["blocked_duplicate_count"], 1)
        self.assertEqual(ledger.query_status("official RWKV founder")["blocked_count"], 1)

    def test_query_status_blocks_reordered_and_digit_variant_queries(self):
        ledger = RetrievalLedger()
        original = "深圳地铁一号线 站点 列表 官方"
        variant = "深圳地铁1号线 站点列表 官方 完整"
        ledger.record(
            original,
            {"status": "ok", "results": [{"url": "https://example.com/metro"}]},
            step=1,
            action="web_search",
        )
        status = ledger.query_status(variant)
        self.assertTrue(status["attempted"])
        self.assertEqual(status["match_type"], "equivalent")
        self.assertEqual(status["matched_query"], original)
        self.assertGreaterEqual(status["similarity"], 0.88)

    def test_query_similarity_does_not_collapse_different_aspects(self):
        self.assertLess(
            query_similarity(
                "深圳地铁一号线 站点 列表 官方",
                "深圳地铁一号线 首班车 末班车 时间 官方",
            ),
            0.88,
        )

    def test_branch_view_contains_shared_progress_and_branch_scope(self):
        ledger = RetrievalLedger()
        ledger.record(
            "founder",
            {"results": [{"url": "https://example.com/founder"}]},
            step=1,
            branch_id="B1",
            task_point_id="P1",
            action="web_search",
        )
        ledger.record(
            "papers",
            {"results": [{"url": "https://example.com/paper"}]},
            step=2,
            branch_id="B2",
            task_point_id="P2",
            action="web_search",
        )
        view = ledger.observation(branch_id="B2", task_point_id="P2")
        self.assertEqual(view["branch_id"], "B2")
        self.assertEqual(view["task_point_id"], "P2")
        self.assertEqual(view["retrieved_url_count"], 2)
        self.assertEqual(len(view["branch_searches"]), 1)
        self.assertEqual(view["branch_searches"][0]["query"], "papers")

    def test_orchestrator_attaches_ledger_to_model_observation(self):
        orchestrator = Orchestrator()
        orchestrator.state.task_id = "LEDGER_TRACE"
        with patch("agent.orchestrator.append_task_event") as append_event:
            enriched = orchestrator._record_retrieval_progress(
                {
                    "status": "ok",
                    "results": [{"url": "https://example.com/fact"}],
                },
                query="fact",
                step=1,
                action="web_search",
                phase="GENERIC_WEB",
                branch_id="B1",
                task_point_id="P1",
            )
        self.assertEqual(enriched["retrieval_delta"]["new_url_count"], 1)
        self.assertEqual(enriched["retrieval_ledger"]["total_searches"], 1)
        self.assertTrue(any(call.args[1] == "retrieval_ledger" for call in append_event.call_args_list))

    def test_failed_exact_request_is_shared_across_the_episode(self):
        ledger = RetrievalLedger()
        args = {"url": "https://example.com/unavailable", "max_chars": 20000}
        ledger.record_request(
            "fetch_web_url",
            args,
            {"status": "error", "message": "TLS failure", "results": []},
            step=4,
            task_point_id="P1",
        )
        status = ledger.request_status("fetch_web_url", dict(args))
        self.assertIsNotNone(status)
        self.assertTrue(status["failed"])
        self.assertEqual(status["failed_attempts"], 1)
        self.assertEqual(ledger.observation(task_point_id="P2")["failed_requests"][0]["action"], "fetch_web_url")

    def test_planner_compact_observation_preserves_ledger_context(self):
        rendered = Planner._compact_observation(
            {
                "status": "ok",
                "results": [],
                "retrieval_delta": {"exact_repeat": True, "new_url_count": 0},
                "retrieval_ledger": {
                    "total_searches": 2,
                    "unique_queries": 1,
                    "exact_repeat_count": 1,
                },
            }
        )
        self.assertIn("retrieval_ledger", rendered)
        self.assertIn("exact_repeat_count", rendered)
        self.assertNotIn("repeating is allowed", rendered)


if __name__ == "__main__":
    unittest.main()
