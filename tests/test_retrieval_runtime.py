from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from agent.planner import Planner
from retrieval_plugins import PluginRegistry, normalize_result
from tools.builtin import load_builtin_tools
from tools.registry import ToolRegistry


class RetrievalRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_builtin_tools()

    def test_provider_result_has_one_canonical_envelope(self):
        result = normalize_result(
            {
                "status": "ok",
                "provider": "test-provider",
                "results": [{"title": "Source", "url": "https://example.com"}],
            },
            provider="test.plugin",
            query="query",
            role="discovery",
        )
        self.assertEqual(result["schema_version"], "retrieval.v1")
        self.assertEqual(result["retrieval_role"], "discovery")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["sources"], ["https://example.com"])

    def test_model_plan_judge_and_replan_keep_fixed_generic_schemas(self):
        responses = iter(
            [
                '{"schema_version":"task_plan.v1","goal":"goal","atomic_points":[{"id":"P1","objective":"check fact","evidence_needed":["source"],"status":"pending"}],"completion_rule":"supported"}',
                '{"schema_version":"completion_judgement.v1","status":"incomplete","reason":"missing evidence","missing_point_ids":["P1"],"next_focus":["check fact"]}',
                '{"schema_version":"task_plan.v1","goal":"goal","atomic_points":[{"id":"P1a","objective":"check fact from a new source","evidence_needed":["source"],"status":"pending"}],"completion_rule":"supported"}',
            ]
        )

        class FakeLLM:
            def text_completion(self, prompt, max_tokens=1024, stop=None):
                return SimpleNamespace(content=next(responses))

        planner = Planner()
        planner.llm = FakeLLM()
        plan = planner.create_task_plan("goal")
        judgement = planner.judge_completion("goal", plan, "draft", "evidence")
        followup = planner.replan_task("goal", plan, judgement, "evidence")
        self.assertEqual(plan["schema_version"], "task_plan.v1")
        self.assertEqual(judgement["status"], "incomplete")
        self.assertEqual(followup["atomic_points"][0]["id"], "P1a")

    def test_legacy_provider_is_not_executable_in_agent_phase(self):
        self.assertFalse(ToolRegistry.can_execute("search_web_keyless", "EXTRACTION"))
        result = json.loads(
            ToolRegistry.execute(
                "search_web_keyless",
                {"query": "test"},
                {},
                phase="EXTRACTION",
            )
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_class"], "tool_not_allowed_in_phase")

    def test_unknown_arguments_are_rejected_before_provider_execution(self):
        result = json.loads(
            ToolRegistry.execute(
                "search_web_tavily",
                {"query": "test", "unknown_option": True},
                {},
                phase="DISCOVERY",
            )
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_class"], "tool_protocol")
        self.assertEqual(result["unknown_arguments"], ["unknown_option"])

    def test_unknown_tool_uses_the_same_error_envelope(self):
        result = json.loads(ToolRegistry.execute("not_registered", {}, {}, phase="DISCOVERY"))
        self.assertEqual(result["schema_version"], "retrieval.v1")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_class"], "unknown_tool")

    def test_current_catalog_contains_roles_not_provider_routing_rules(self):
        catalog = json.loads(ToolRegistry.get_json_catalog("DISCOVERY"))
        rows = {row["name"]: row for row in catalog}
        self.assertEqual(rows["search_web_tavily"]["retrieval_role"], "discovery")
        self.assertNotIn("fetch_web_url", rows)
        self.assertEqual(rows["search_web_keyless"]["retrieval_role"], "discovery")
        self.assertEqual(rows["search_crossref"]["plugin"], "crossref.rest")
        self.assertEqual(rows["search_github_rest"]["plugin"], "github.rest")
        self.assertEqual(rows["search_mediawiki"]["plugin"], "mediawiki.api")

        extraction = json.loads(ToolRegistry.get_json_catalog("EXTRACTION"))
        extraction_rows = {row["name"]: row for row in extraction}
        self.assertEqual(extraction_rows["fetch_web_url"]["retrieval_role"], "evidence")
        self.assertNotIn("search_web_tavily", extraction_rows)
        self.assertEqual(extraction_rows["fetch_crossref_record"]["retrieval_role"], "evidence")
        self.assertEqual(extraction_rows["fetch_github_rest"]["retrieval_role"], "evidence")
        self.assertEqual(extraction_rows["fetch_mediawiki_page"]["retrieval_role"], "evidence")

    def test_discovery_and_evidence_roles_are_phase_gated(self):
        self.assertTrue(ToolRegistry.can_execute("search_web_tavily", "DISCOVERY"))
        self.assertTrue(ToolRegistry.can_execute("search_web_keyless", "DISCOVERY"))
        self.assertTrue(ToolRegistry.can_execute("search_crossref", "DISCOVERY"))
        self.assertTrue(ToolRegistry.can_execute("search_github_rest", "DISCOVERY"))
        self.assertTrue(ToolRegistry.can_execute("search_mediawiki", "DISCOVERY"))
        self.assertFalse(ToolRegistry.can_execute("search_web_tavily", "EXTRACTION"))
        self.assertFalse(ToolRegistry.can_execute("search_github_rest", "EXTRACTION"))
        self.assertTrue(ToolRegistry.can_execute("fetch_web_url", "EXTRACTION"))
        self.assertTrue(ToolRegistry.can_execute("fetch_crossref_record", "EXTRACTION"))
        self.assertTrue(ToolRegistry.can_execute("fetch_github_rest", "EXTRACTION"))
        self.assertTrue(ToolRegistry.can_execute("fetch_mediawiki_page", "EXTRACTION"))
        self.assertTrue(ToolRegistry.can_execute("fetch_web_url", "SYNTHESIS"))
        self.assertNotIn("fetch_web_url", ToolRegistry.names("DISCOVERY"))
        self.assertIn("fetch_web_url", ToolRegistry.names("EXTRACTION"))

    def test_plugin_registration_merges_tools_for_one_provider_package(self):
        PluginRegistry.register("test.multi", capabilities=("first",), tools=("search",))
        PluginRegistry.register("test.multi", capabilities=("second",), tools=("fetch",))
        row = next(item for item in PluginRegistry.catalog() if item["plugin"] == "test.multi")
        self.assertEqual(row["capabilities"], ["first", "second"])
        self.assertEqual(row["tools"], ["search", "fetch"])


if __name__ == "__main__":
    unittest.main()
