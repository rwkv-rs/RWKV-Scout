from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.orchestrator import Orchestrator
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

    def test_model_plan_and_replan_keep_fixed_generic_schemas(self):
        responses = iter(
            [
                '{"schema_version":"task_plan.v1","goal":"goal","atomic_points":[{"id":"P1","task":"check the fact","objective":"check fact","evidence_needed":["source"],"acceptance_criteria":["the fact is directly supported"],"output_format":"prose","status":"pending"}],"completion_rule":"supported"}',
                '{"schema_version":"task_plan.v1","goal":"goal","atomic_points":[{"id":"P1a","task":"check the fact from a new source","objective":"check fact from a new source","evidence_needed":["source"],"acceptance_criteria":["the new source directly supports the fact"],"output_format":"prose","status":"pending"}],"completion_rule":"supported"}',
            ]
        )

        class FakeLLM:
            def text_completion(self, prompt, max_tokens=1024, stop=None):
                return SimpleNamespace(content=next(responses))

        planner = Planner()
        planner.llm = FakeLLM()
        plan = planner.create_task_plan("goal")
        retrieval_observation = {
            "schema_version": "retrieval.v1",
            "status": "error",
            "error_class": "page_fetch_failed",
            "message": "the selected page could not be fetched",
        }
        followup = planner.replan_task("goal", plan, retrieval_observation, "evidence")
        self.assertEqual(plan["schema_version"], "task_plan.v1")
        self.assertEqual(plan["atomic_points"][0]["task"], "check the fact")
        self.assertEqual(plan["atomic_points"][0]["acceptance_criteria"], ["the fact is directly supported"])
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
        self.assertTrue(rows["search_mediawiki"]["description"])
        self.assertIn("properties", rows["search_mediawiki"]["arguments"])

        extraction = json.loads(ToolRegistry.get_json_catalog("EXTRACTION"))
        extraction_rows = {row["name"]: row for row in extraction}
        self.assertEqual(extraction_rows["fetch_web_url"]["retrieval_role"], "evidence")
        self.assertNotIn("search_web_tavily", extraction_rows)
        self.assertEqual(extraction_rows["fetch_crossref_record"]["retrieval_role"], "evidence")
        self.assertEqual(extraction_rows["fetch_github_rest"]["retrieval_role"], "evidence")
        self.assertEqual(extraction_rows["fetch_mediawiki_page"]["retrieval_role"], "evidence")

        all_tools = json.loads(ToolRegistry.get_json_catalog("ALL"))
        all_rows = {row["name"]: row for row in all_tools}
        self.assertIn("search_web_keyless", all_rows)
        self.assertIn("fetch_web_url", all_rows)
        self.assertTrue(ToolRegistry.can_execute("search_web_keyless", "ALL"))
        self.assertTrue(ToolRegistry.can_execute("fetch_web_url", "ALL"))

    def test_model_catalog_matches_single_public_web_capability(self):
        catalog = json.loads(
            ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
        )
        rows = {row["name"]: row for row in catalog}
        self.assertEqual(set(rows), {"web_search", "finish_task"})
        self.assertEqual(ToolRegistry.model_visible_names("ALL"), ["finish_task", "web_search"])
        self.assertEqual(rows["web_search"]["category"], "retrieval")
        self.assertEqual(rows["finish_task"]["category"], "control")
        self.assertNotIn("search_web_tavily", rows)
        self.assertNotIn("fetch_web_url", rows)

    def test_planner_prompt_does_not_leak_provider_routing_matrix(self):
        prompt = Planner._system_prompt("ALL")
        self.assertIn('"name": "web_search"', prompt)
        self.assertNotIn('"name": "search_web_tavily"', prompt)
        self.assertNotIn('"name": "search_web_keyless"', prompt)
        self.assertNotIn('"name": "fetch_web_url"', prompt)
        self.assertNotIn("provider-specific search API", prompt)

    def test_global_loop_blocks_successful_duplicate_web_search_before_execution(self):
        orchestrator = Orchestrator()
        orchestrator.state.task_id = "DUPLICATE_SEARCH_TEST"
        orchestrator.planner.plan_next_action = Mock(
            side_effect=[
                {"action": "web_search", "args": {"query": "same query"}},
                {"action": "web_search", "args": {"query": "same query"}},
                {"action": "finish_task", "args": {}},
            ]
        )
        orchestrator.planner.observe_tool_result = Mock()
        orchestrator._complete_model_tool_loop = Mock(return_value="done")
        task_plan = {"atomic_points": []}
        tool_result = json.dumps(
            {
                "status": "ok",
                "results": [{"url": "https://example.com/fact"}],
                "evidence_ready": True,
            }
        )

        with patch("agent.orchestrator.ToolRegistry.execute", return_value=tool_result) as execute:
            with patch("agent.orchestrator.append_task_event") as append_event:
                result = orchestrator._run_single_loop("goal", {}, task_plan, max_steps=3)

        self.assertEqual(result, "done")
        self.assertEqual(execute.call_count, 1)
        rounds = orchestrator._complete_model_tool_loop.call_args.args[2]
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0][0], "same query")
        self.assertTrue(rounds[0][1]["evidence_ready"])
        duplicate_events = [
            call
            for call in append_event.call_args_list
            if call.args[1] == "retrieval_duplicate_blocked"
        ]
        self.assertEqual(len(duplicate_events), 1)
        blocked_results = [
            call
            for call in append_event.call_args_list
            if call.args[1] == "tool_result"
            and call.kwargs.get("execution_status") == "blocked_duplicate"
        ]
        self.assertEqual(len(blocked_results), 1)

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
