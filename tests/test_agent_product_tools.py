import json
import unittest
from unittest.mock import Mock, patch

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
        for name in ("web_search", "open_page", "find_in_page", "connector_lookup", "calculator", "date_diff", "current_time", "finish_task"):
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
            {"query"},
        )

    def test_planner_prompt_contains_each_model_tool_description(self):
        catalog = json.loads(
            ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
        )
        prompt = Planner._system_prompt("ALL")
        for row in catalog:
            self.assertIn(row["description"].splitlines()[0], prompt)

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
        self.assertEqual(orchestrator._calculation_results[0]["days"], 559)

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

    def test_freshness_policy_is_explicit_and_unknown_dates_are_not_guessed(self):
        policy = build_freshness_policy("请给出截至 2024-12-31 的信息")
        self.assertEqual(policy["as_of"], "2024-12-31")
        within = annotate_freshness({"url": "https://a", "published": "2024-10-01"}, policy)
        after = annotate_freshness({"url": "https://b", "published": "2025-01-01"}, policy)
        unknown = annotate_freshness({"url": "https://c"}, policy)
        self.assertEqual(within["freshness"]["state"], "within_cutoff")
        self.assertEqual(after["freshness"]["state"], "after_cutoff")
        self.assertEqual(unknown["freshness"]["state"], "unknown_date")

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

    @patch("tools.page_tools.fetch_web_url")
    def test_find_in_page_returns_matches_and_page_body(self, fetch):
        fetch.return_value = json.dumps(
            {
                "status": "ok",
                "results": [
                    {
                        "title": "Example",
                        "url": "https://example.com/doc",
                        "content": "alpha\nRelease date: 2024-10-07\nomega",
                        "page_excerpt": "alpha\nRelease date: 2024-10-07\nomega",
                        "source_excerpt": "alpha\nRelease date: 2024-10-07\nomega",
                        "evidence_origin": "fetched_page_body",
                        "evidence_boundary": "page_body_only",
                    }
                ],
            },
            ensure_ascii=False,
        )
        from tools.page_tools import find_in_page

        result = json.loads(find_in_page("https://example.com/doc", "Release date"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["matches"][0]["line"], 2)
        self.assertIn("2024-10-07", result["results"][0]["content"])


if __name__ == "__main__":
    unittest.main()
