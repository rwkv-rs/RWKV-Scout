import json
import unittest
from unittest.mock import Mock, patch

import config
from agent.tool_protocol import canonicalize_tool_call, normalize_tool_result
from agent.planner import Planner
from agent.orchestrator import Orchestrator
from tools.builtin import load_builtin_tools
from tools.registry import ToolRegistry
from utils.answer_fact_check import check_answer_facts
from utils.freshness import annotate_freshness, build_freshness_policy


class AgentProductToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_builtin_tools()

    def test_public_agent_surface_contains_product_tools(self):
        names = ToolRegistry.model_visible_names("ALL")
        for name in ("web_search", "connector_lookup", "calculator", "date_diff", "current_time", "finish_task"):
            self.assertIn(name, names)

    def test_every_model_tool_has_description_and_argument_contract(self):
        catalog = json.loads(
            ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
        )
        self.assertEqual(
            {row["name"] for row in catalog},
            set(ToolRegistry.model_visible_names("ALL")),
        )
        for row in catalog:
            self.assertTrue(row["description"].strip(), row["name"])
            self.assertEqual(row["arguments"].get("type"), "object")
            properties = row["arguments"].get("properties") or {}
            required = set(row["arguments"].get("required") or [])
            self.assertIsInstance(properties, dict)
            self.assertTrue(required.issubset(properties), row["name"])

    def test_model_description_keeps_legacy_parameter_contract(self):
        catalog = json.loads(
            ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
        )
        web_search = next(row for row in catalog if row["name"] == "web_search")
        self.assertIn("Parameters: query", web_search["description"])
        self.assertEqual(
            set(web_search["arguments"]["properties"]),
            {"query", "max_results"},
        )

    def test_planner_prompt_contains_each_model_tool_description(self):
        catalog = json.loads(
            ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
        )
        prompt = Planner._system_prompt("ALL")
        for row in catalog:
            self.assertIn(row["description"].splitlines()[0], prompt)

    def test_rwkv_cross_validator_owns_finish_or_replan_decision(self):
        planner = Planner()
        raw = (
            '"decision":"replan","missing_points":["P2"],"conflicts":[],'
            '"next_focus":"confirm the current version","reason":"P2 is missing"}'
        )
        planner.llm = Mock()
        planner.llm.text_completion.return_value = Mock(content=raw)
        review = planner.cross_validate_research(
            "question",
            {
                "goal": "answer",
                "atomic_points": [
                    {
                        "id": "P2",
                        "task": "current version",
                        "objective": "confirm current version",
                    }
                ],
            },
            "ORIGINAL_SOURCE_SPAN",
        )

        self.assertEqual(review["decision"], "replan")
        self.assertEqual(review["missing_points"], ["P2"])
        self.assertEqual(review["raw_model_output"], raw)
        prompt = planner.llm.text_completion.call_args.args[0]
        self.assertIn("ORIGINAL_SOURCE_SPAN", prompt)
        self.assertIn("There is no later page-summary", prompt)
        self.assertIn("read their contents yourself", prompt)
        self.assertNotIn("search query for P2", review["next_focus"])

    def test_rebuilt_planner_uses_one_seeded_diversification_request(self):
        planner = Planner()
        seen = []

        class SamplingAwareRWKV:
            provider = "local_13b"

            def text_completion(self, _prompt, max_tokens=0, stop=None):
                self_max_tokens = max_tokens
                del stop
                seen.append(
                    {
                        "temperature": config.get_llm_temperature(),
                        "seed": config.get_llm_seed(),
                        "max_tokens": self_max_tokens,
                    }
                )
                return Mock(content='{"name":"finish_task","arguments":{}}')

        planner.llm = SamplingAwareRWKV()
        planner.rebuild_session(
            "same question",
            "shared evidence",
            {"status": "cross_validation_replan"},
            "REPLAN",
        )

        first = planner.plan_next_action("same question", {}, "shared evidence", "REPLAN")
        second = planner.plan_next_action("same question", {}, "shared evidence", "DISCOVERY")

        self.assertEqual(first["sampling_temperature"], 0.25)
        self.assertIsInstance(first["sampling_seed"], int)
        self.assertEqual(seen[0]["seed"], first["sampling_seed"])
        self.assertEqual(second["sampling_temperature"], 0.1)
        self.assertIsNone(second["sampling_seed"])
        self.assertIsNone(seen[1]["seed"])

    def test_repeated_replans_progress_temperature_and_change_only_the_seed(self):
        planner = Planner()
        seen = []

        class SamplingAwareRWKV:
            provider = "local_13b"

            def text_completion(self, _prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(
                    {
                        "temperature": config.get_llm_temperature(),
                        "seed": config.get_llm_seed(),
                    }
                )
                return Mock(content='{"name":"finish_task","arguments":{}}')

        planner.llm = SamplingAwareRWKV()
        decisions = []
        for generation in range(1, 6):
            planner.rebuild_session(
                "same question",
                "unchanged shared evidence",
                {"status": "cross_validation_replan", "generation": generation},
                "REPLAN",
            )
            decisions.append(
                planner.plan_next_action(
                    "same question", {}, "unchanged shared evidence", "REPLAN"
                )
            )

        self.assertEqual(
            [round(row["sampling_temperature"], 2) for row in decisions],
            [0.25, 0.35, 0.45, 0.55, 0.55],
        )
        seeds = [row["sampling_seed"] for row in decisions]
        self.assertTrue(all(isinstance(seed, int) for seed in seeds))
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertEqual([row["temperature"] for row in seen], [0.25, 0.35, 0.45, 0.55, 0.55])
        self.assertEqual([row["seed"] for row in seen], seeds)

    def test_multiple_deterministic_tools_run_in_one_model_owned_loop(self):
        orchestrator = Orchestrator()
        orchestrator.state.task_id = "MULTI_TOOL_LOOP_TEST"
        orchestrator.planner.plan_next_action = Mock(
            side_effect=[
                {
                    "action": "current_time",
                    "args": {"timezone": "UTC"},
                    "call_id": "clock-1",
                },
                {
                    "action": "calculator",
                    "args": {"expression": "(10 + 5) * 2"},
                    "call_id": "calc-1",
                },
                {
                    "action": "date_diff",
                    "args": {
                        "date_a": "2023-10-05",
                        "date_b": "2025-04-16",
                    },
                    "call_id": "date-1",
                },
                {"action": "finish_task", "args": {}, "call_id": "finish-1"},
            ]
        )
        orchestrator.planner.observe_tool_result = Mock()
        orchestrator._cross_validate_research = Mock(return_value={"decision": "finish"})
        orchestrator._complete_model_tool_loop = Mock(return_value="done")
        with patch("agent.orchestrator.append_task_event"):
            result = orchestrator._run_single_loop(
                "Use the tools to check the time, calculate 15*2, and find the date difference.",
                {},
                {"atomic_points": []},
                max_steps=4,
            )

        self.assertEqual(result, "done")
        self.assertEqual(orchestrator.planner.plan_next_action.call_count, 4)
        observed_tools = [
            item[0][0]["tool"]
            for item in orchestrator.planner.observe_tool_result.call_args_list
            if item[0] and isinstance(item[0][0], dict) and item[0][0].get("tool")
        ]
        self.assertEqual(
            observed_tools,
            ["current_time", "calculator", "date_diff"],
        )
        self.assertEqual(orchestrator._arithmetic_results[0]["result"], 30)
        self.assertEqual(orchestrator._calculation_results[0]["days"], 559)

    def test_current_time_is_sufficient_for_time_only_task(self):
        orchestrator = Orchestrator()
        orchestrator.state.task_id = "CURRENT_TIME_ONLY_TEST"
        orchestrator.planner.plan_next_action = Mock(
            side_effect=[
                {
                    "action": "current_time",
                    "args": {"timezone": "Asia/Shanghai"},
                    "call_id": "clock-only-1",
                },
                {"action": "finish_task", "args": {}, "call_id": "clock-only-finish"},
            ]
        )
        orchestrator.planner.observe_tool_result = Mock()
        orchestrator._cross_validate_research = Mock(return_value={"decision": "finish"})
        orchestrator._complete_model_tool_loop = Mock(return_value="current date")
        with patch("agent.orchestrator.append_task_event"):
            result = orchestrator._run_single_loop(
                "现在的日期是什么？",
                {},
                {
                    "task_mode": "current_time",
                    "atomic_points": [
                        {
                            "id": "P1",
                            "objective": "return the current date",
                            "evidence_needed": ["current clock observation"],
                        }
                    ]
                },
                max_steps=2,
            )

        self.assertEqual(result, "current date")
        self.assertEqual(orchestrator._complete_model_tool_loop.call_count, 1)
        self.assertEqual(orchestrator._time_results[0]["tool"], "current_time")

    def test_protocol_adapter_preserves_model_call_and_id(self):
        call = canonicalize_tool_call({"tool_calls": [{"id": "call-7", "function": {"name": "calculator", "arguments": '{"expression":"2+3"}'}}]})
        self.assertEqual(call["name"], "calculator")
        self.assertEqual(call["arguments"], {"expression": "2+3"})
        self.assertEqual(call["call_id"], "call-7")
        result = normalize_tool_result('{"status":"ok","result":5}', tool_name="calculator", call_id="call-7")
        self.assertEqual(result["tool_call_id"], "call-7")
        self.assertEqual(result["result"], 5)

    def test_calculator_is_deterministic_and_rejects_code(self):
        ok = json.loads(ToolRegistry.execute("calculator", {"expression": "(10 + 5) * 2"}, {}, phase="ALL"))
        self.assertEqual(ok["result"], 30)
        bad = json.loads(ToolRegistry.execute("calculator", {"expression": "__import__('os').getcwd()"}, {}, phase="ALL"))
        self.assertEqual(bad["status"], "error")

    def test_current_time_uses_requested_timezone(self):
        result = json.loads(ToolRegistry.execute("current_time", {"timezone": "UTC"}, {}, phase="ALL"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["timezone"], "UTC")
        self.assertRegex(result["date"], r"^\d{4}-\d{2}-\d{2}$")

    @patch("tools.connectors.get_current_weather")
    def test_connector_lookup_wraps_structured_provider_data(self, weather):
        weather.return_value = json.dumps(
            {
                "status": "ok",
                "location": "Shanghai",
                "current": {"temperature_c": 25},
                "sources": ["https://api.example/weather"],
            }
        )
        result = json.loads(
            ToolRegistry.execute(
                "connector_lookup",
                {"connector": "weather", "query": "Shanghai"},
                {"agentic_tool_loop": True},
                phase="ALL",
            )
        )
        self.assertEqual(result["connector"], "weather")
        self.assertEqual(result["results"][0]["evidence_origin"], "structured_api_record")
        self.assertEqual(result["freshness_policy"]["mode"], "retrieval_time_only")

    @patch("tools.connectors.get_current_weather")
    def test_weather_not_found_is_not_promoted_to_structured_evidence(self, weather):
        weather.return_value = json.dumps(
            {"status": "not_found", "location": "Missing Place"}
        )
        result = json.loads(
            ToolRegistry.execute(
                "connector_lookup",
                {"connector": "weather", "query": "Missing Place"},
                {"agentic_tool_loop": True},
                phase="ALL",
            )
        )
        self.assertEqual(result["status"], "no_results")
        self.assertEqual(result["results"], [])
        self.assertFalse(result["evidence_ready"])

    def test_freshness_policy_is_explicit_and_unknown_dates_are_not_guessed(self):
        policy = build_freshness_policy("请给出截至 2024-12-31 的信息")
        self.assertEqual(policy["as_of"], "2024-12-31")
        within = annotate_freshness({"url": "https://a", "published": "2024-10-01"}, policy)
        after = annotate_freshness({"url": "https://b", "published": "2025-01-01"}, policy)
        unknown = annotate_freshness({"url": "https://c"}, policy)
        self.assertEqual(within["freshness"]["state"], "within_cutoff")
        self.assertEqual(after["freshness"]["state"], "after_cutoff")
        self.assertEqual(unknown["freshness"]["state"], "unknown_date")
        chinese_policy = build_freshness_policy("截至 2026 年 7 月 29 日，查询最新稳定版本")
        self.assertEqual(chinese_policy["as_of"], "2026-07-29")
        self.assertEqual(chinese_policy["mode"], "explicit_cutoff")

    def test_answer_fact_check_does_not_modify_model_text(self):
        answer = "发布日期是 2024-10-07，版本为 3.13.0，间隔 559 days。"
        result = check_answer_facts(
            answer,
            evidence=[{"source_excerpt": "Python 3.13.0 was released on October 7, 2024."}],
            calculation_results=[{"tool": "date_diff", "days": 559}],
        )
        self.assertFalse(result["answer_changed"])
        self.assertEqual(result["status"], "supported")
        bad = check_answer_facts("发布日期是 2026-01-01。", evidence=[{"source_excerpt": "released on October 7, 2024"}])
        self.assertEqual(bad["status"], "needs_review")
        cutoff = check_answer_facts(
            "发布日期是 2025-01-01。",
            evidence=[{"source_excerpt": "released on January 1, 2025"}],
            freshness_policy={"as_of": "2024-12-31"},
        )
        self.assertEqual(cutoff["status"], "needs_review")
        self.assertEqual(len(cutoff["freshness_violations"]), 1)

    def test_answer_fact_check_flags_unseen_compound_build_identifiers(self):
        result = check_answer_facts(
            "The source shows cu118, not cu117 or cu121.",
            evidence=[{"source_excerpt": "Use the cu118 wheel index."}],
        )

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(
            {row["value"] for row in result["unsupported_claims"]},
            {"cu117", "cu121"},
        )
        self.assertIn(
            "cu118",
            {row["value"] for row in result["supported_claims"]},
        )

    def test_answer_fact_check_keeps_prefixed_versions_whole(self):
        result = check_answer_facts(
            "Fetch was experimental in Node.js v18.0.0 and stable in v21.0.0.",
            evidence=[
                {
                    "source_excerpt": (
                        "Node.js v18.0.0 had an experimental Fetch API; "
                        "Node.js v21.0.0 made it stable."
                    )
                }
            ],
        )

        self.assertEqual(result["status"], "supported")
        self.assertEqual(
            {row["value"] for row in result["supported_claims"]},
            {"v18.0.0", "v21.0.0"},
        )

    def test_answer_fact_check_ignores_versions_in_citation_destinations(self):
        result = check_answer_facts(
            (
                "Python 3.14 supports this procedure. "
                "[S1](https://docs.python.org/3.14/howto/example.html)"
            ),
            evidence=[{"source_excerpt": "Python 3.14 supports this procedure."}],
        )

        self.assertEqual(result["claim_count"], 1)
        self.assertEqual(result["supported_claims"][0]["value"], "3.14")

    def test_answer_fact_check_allows_a_literal_user_scope_anchor(self):
        result = check_answer_facts(
            "For Python 3.14, use the documented build option.",
            evidence=[{"source_excerpt": "Use the documented build option."}],
            user_supplied_literals=["3.14"],
        )

        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["unsupported_count"], 0)
        self.assertEqual(
            result["supported_claims"][0]["support_basis"],
            "user_supplied_scope",
        )

if __name__ == "__main__":
    unittest.main()
