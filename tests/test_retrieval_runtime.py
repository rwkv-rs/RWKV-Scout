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

    def test_model_plan_keeps_fixed_generic_schema(self):
        class FakeLLM:
            def text_completion(self, prompt, max_tokens=1024, stop=None):
                return SimpleNamespace(
                    content=(
                        '{"schema_version":"task_plan.v1","goal":"goal",'
                        '"atomic_points":[{"id":"P1","task":"check the fact",'
                        '"objective":"check fact","evidence_needed":["source"],'
                        '"acceptance_criteria":["the fact is directly supported"],'
                        '"output_format":"prose","status":"pending"}],'
                        '"completion_rule":"supported"}'
                    )
                )

        planner = Planner()
        planner.llm = FakeLLM()
        plan = planner.create_task_plan("goal")
        self.assertEqual(plan["schema_version"], "task_plan.v1")
        self.assertEqual(plan["atomic_points"][0]["task"], "check the fact")
        self.assertEqual(
            plan["atomic_points"][0]["acceptance_criteria"],
            ["the fact is directly supported"],
        )

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
        self.assertEqual(
            set(rows),
            {
                "web_search",
                "connector_lookup",
                "calculator",
                "current_time",
                "date_diff",
                "finish_task",
            },
        )
        self.assertEqual(
            ToolRegistry.model_visible_names("ALL"),
            [
                "finish_task",
                "web_search",
                "connector_lookup",
                "calculator",
                "date_diff",
                "current_time",
            ],
        )
        self.assertEqual(rows["web_search"]["category"], "retrieval")
        self.assertEqual(rows["date_diff"]["category"], "computation")
        self.assertEqual(rows["finish_task"]["category"], "control")
        self.assertEqual(rows["calculator"]["category"], "computation")
        self.assertEqual(rows["connector_lookup"]["category"], "connector")
        self.assertNotIn("search_web_tavily", rows)
        self.assertNotIn("fetch_web_url", rows)

    def test_planner_prompt_does_not_leak_provider_routing_matrix(self):
        prompt = Planner._system_prompt("ALL")
        self.assertIn('"name": "web_search"', prompt)
        self.assertNotIn('"name": "search_web_tavily"', prompt)
        self.assertNotIn('"name": "search_web_keyless"', prompt)
        self.assertNotIn('"name": "fetch_web_url"', prompt)
        self.assertNotIn("provider-specific search API", prompt)
        self.assertIn('"name": "date_diff"', prompt)
        self.assertIn("YYYY-MM-DD", prompt)

    def test_planner_prompt_scopes_public_tools_by_phase_and_point(self):
        discovery = Planner._system_prompt("DISCOVERY")
        extraction = Planner._system_prompt("EXTRACTION")
        self.assertIn('"name": "web_search"', discovery)
        self.assertNotIn('"name": "open_page"', discovery)
        self.assertNotIn('"name": "open_page"', extraction)
        self.assertIn('"name": "web_search"', extraction)
        self.assertIn('"task_point_id":"P1"', discovery)

    def test_planner_decision_does_not_replay_audit_or_page_evidence_history(self):
        prompts = []

        class FakeLLM:
            provider = "local"

            def text_completion(self, prompt, max_tokens=1024, stop=None):
                prompts.append(prompt)
                return SimpleNamespace(content='{"name":"finish_task","arguments":{}}')

        planner = Planner()
        planner.llm = FakeLLM()
        planner.begin_task(
            "Find the requested fact",
            "Task: Find the requested fact",
            {
                "schema_version": "task_plan.v1",
                "goal": "Find the requested fact",
                "atomic_points": [
                    {
                        "id": "P1",
                        "task": "find the fact",
                        "objective": "verify the fact",
                        "evidence_needed": ["the direct fact"],
                        "acceptance_criteria": ["a source supports it"],
                    }
                ],
            },
        )
        planner._messages.append(
            {"role": "tool", "content": "HISTORICAL_CONTROLLER_ERROR_SHOULD_NOT_BE_SENT"}
        )
        planner.observe_tool_result(
            {
                "status": "ok",
                "results": [
                    {
                        "title": "Current page",
                        "url": "https://example.com/current",
                        "chunk_candidates": [
                            {"facts": ["SECRET_PAGE_FACT"], "quote": "SECRET_PAGE_QUOTE"}
                        ],
                    }
                ],
            }
        )

        action = planner.plan_next_action(
            "Find the requested fact",
            None,
            "Task: Find the requested fact",
            "DISCOVERY",
        )

        self.assertEqual(action["action"], "finish_task")
        self.assertEqual(len(prompts), 1)
        self.assertIn("Latest tool observation (routing only)", prompts[0])
        self.assertIn("https://example.com/current", prompts[0])
        self.assertNotIn("HISTORICAL_CONTROLLER_ERROR_SHOULD_NOT_BE_SENT", prompts[0])
        self.assertIn("SECRET_PAGE_FACT", prompts[0])
        self.assertIn("SECRET_PAGE_QUOTE", prompts[0])

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

        with patch("tools.registry.ToolRegistry.execute", return_value=tool_result) as execute:
            with patch("agent.unified_research.append_task_event") as append_event:
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

    def test_duplicate_after_empty_search_leaves_a_recovery_turn(self):
        orchestrator = Orchestrator()
        orchestrator.state.task_id = "EMPTY_DUPLICATE_RECOVERY_TEST"
        orchestrator.state.run_metadata = {"max_replan_attempts": 0}
        orchestrator.planner.plan_next_action = Mock(
            side_effect=[
                {"action": "web_search", "args": {"query": "same query"}},
                {"action": "web_search", "args": {"query": "same query"}},
                {"action": "finish_task", "args": {}},
            ]
        )
        orchestrator.planner.observe_tool_result = Mock()
        orchestrator._complete_model_tool_loop = Mock(return_value="done")
        task_plan = {
            "atomic_points": [
                {"id": "P1", "evidence_needed": ["official evidence"]},
            ]
        }
        empty_result = json.dumps(
            {
                "status": "no_evidence",
                "results": [],
                "evidence_ready": False,
                "candidate_urls": [{"url": "https://example.com/fact"}],
            }
        )

        with patch("tools.registry.ToolRegistry.execute", return_value=empty_result) as execute:
            with patch("agent.unified_research.append_task_event") as append_event:
                result = orchestrator._run_single_loop("goal", {}, task_plan, max_steps=3)

        self.assertEqual(result, "done")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(orchestrator.planner.plan_next_action.call_count, 2)
        duplicate_events = [
            call
            for call in append_event.call_args_list
            if call.args[1] == "retrieval_duplicate_blocked"
        ]
        self.assertEqual(len(duplicate_events), 1)

    def test_duplicate_freezes_path_rebuilds_planner_and_fans_out_replan(self):
        orchestrator = Orchestrator()
        orchestrator.state.task_id = "DUPLICATE_MICRO_REPLAN_TEST"
        orchestrator.planner.plan_next_action = Mock(
            side_effect=[
                {"action": "web_search", "args": {"query": "same query"}},
                {"action": "web_search", "args": {"query": "same query"}},
                {"action": "web_search", "args": {"query": "alternate focus"}},
                {"action": "finish_task", "args": {}},
            ]
        )
        orchestrator.planner.begin_replan = Mock()
        orchestrator.planner.observe_tool_result = Mock()
        orchestrator._complete_model_tool_loop = Mock(return_value="done")
        task_plan = {
            "atomic_points": [
                {
                    "id": "P1",
                    "task": "verify the official fact",
                    "objective": "verify the official fact",
                    "evidence_needed": ["direct source evidence"],
                }
            ]
        }
        result = json.dumps(
            {
                "status": "ok",
                "results": [
                    {
                        "title": "Source",
                        "url": "https://example.com/fact",
                        "content": "Direct source evidence.",
                    }
                ],
            }
        )

        with (
            patch("tools.registry.ToolRegistry.execute", return_value=result) as execute,
            patch(
                "agent.unified_research._coverage",
                side_effect=[
                    {"status": "insufficient_evidence", "missing": [{"point_id": "P1", "task": "verify the official fact"}]},
                    {"status": "complete", "missing": []},
                ],
            ),
            patch("agent.unified_research.append_task_event") as append_event,
        ):
            answer = orchestrator._run_single_loop("verify the fact", {}, task_plan, max_steps=4)

        self.assertEqual(answer, "done")
        self.assertEqual(execute.call_count, 4)  # initial search + three replan queries
        orchestrator.planner.begin_replan.assert_called_once()
        self.assertEqual(orchestrator.state.retrieval.replan_count, 1)
        self.assertEqual(len(orchestrator.state.retrieval.frozen_paths), 1)
        replan_events = [
            call for call in append_event.call_args_list if call.args[1] == "task_replan_attempt"
        ]
        self.assertEqual(len(replan_events), 1)

    def test_finish_with_missing_evidence_returns_to_same_shared_loop(self):
        orchestrator = Orchestrator()
        orchestrator.state.task_id = "SHARED_GAP_FEEDBACK_TEST"
        orchestrator.state.run_metadata = {"max_network_searches": 2}
        orchestrator.planner.plan_next_action = Mock(
            side_effect=[
                {"action": "web_search", "args": {"query": "first evidence direction"}},
                {"action": "finish_task", "args": {}},
                {"action": "web_search", "args": {"query": "second evidence direction"}},
                {"action": "finish_task", "args": {}},
            ]
        )
        orchestrator.planner.observe_tool_result = Mock()
        orchestrator._complete_model_tool_loop = Mock(return_value="done")
        task_plan = {
            "atomic_points": [
                {
                    "id": "P1",
                    "task": "verify the requested fact",
                    "objective": "verify the requested fact",
                    "evidence_needed": ["direct source evidence"],
                    "acceptance_criteria": ["the source supports the fact"],
                }
            ]
        }
        first = json.dumps({"status": "no_evidence", "results": []})
        second = json.dumps(
            {
                "status": "ok",
                "results": [
                    {
                        "title": "Fact source",
                        "url": "https://example.com/fact",
                        "content": "The requested fact is directly stated here. " * 20,
                        "page_excerpt": "The requested fact is directly stated here. " * 20,
                        "evidence_origin": "fetched_page_body",
                        "body_verified": True,
                    }
                ],
            }
        )

        with (
            patch(
                "tools.registry.ToolRegistry.execute",
                side_effect=[first, second],
            ) as execute,
            patch("agent.unified_research.append_task_event") as append_event,
        ):
            result = orchestrator._run_single_loop(
                "verify the requested fact",
                {},
                task_plan,
                max_steps=4,
            )

        self.assertEqual(result, "done")
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(
            [item["query"] for item in orchestrator.state.retrieval.query_history],
            ["first evidence direction", "second evidence direction"],
        )
        gap_events = [
            call for call in append_event.call_args_list if call.args[1] == "research_gap"
        ]
        self.assertEqual(len(gap_events), 1)

    def test_discovery_and_evidence_roles_are_phase_gated(self):
        self.assertTrue(ToolRegistry.can_execute("finish_task", "DISCOVERY"))
        self.assertIn("finish_task", ToolRegistry.model_visible_names("DISCOVERY"))
        self.assertTrue(ToolRegistry.can_execute("finish_task", "RECOVERY"))
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
