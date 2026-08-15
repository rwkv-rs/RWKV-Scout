import json
import unittest
from unittest.mock import Mock, patch

import config
from agent.tool_protocol import (
    canonicalize_tool_call,
    normalize_tool_call_format,
)
from agent.planner import Planner
from agent.orchestrator import Orchestrator
from agent.task_plan_contract import normalize_task_plan
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
        self.assertLess(names.index("connector_lookup"), names.index("web_search"))
        self.assertEqual(names[-1], "finish_task")

    def test_every_model_tool_has_description_and_argument_contract(self):
        catalog = json.loads(
            ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
        )
        catalog_names = [row["name"] for row in catalog]
        self.assertEqual(
            set(catalog_names),
            set(ToolRegistry.model_visible_names("ALL")),
        )
        self.assertEqual(catalog_names, ToolRegistry.model_visible_names("ALL"))
        self.assertEqual(catalog_names[0], "connector_lookup")
        self.assertEqual(catalog_names[-1], "finish_task")
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

    def test_connector_catalog_exposes_exact_rwkv_enum_contract(self):
        catalog = json.loads(
            ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
        )
        connector = next(row for row in catalog if row["name"] == "connector_lookup")
        schema = connector["arguments"]

        self.assertEqual(schema["required"], ["operation", "query"])
        self.assertEqual(
            schema["properties"]["operation"]["enum"],
            [
                "weather_current",
                "weather_alerts",
                "github_repository",
                "github_code",
                "github_release",
                "paper",
                "paper_series",
                "crates_release",
                "pypi_release",
                "npm_release",
            ],
        )
        self.assertNotIn("scope", schema["properties"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            connector["accepted_object_types"]["github_release"],
            ["github_repository"],
        )
        self.assertEqual(
            connector["accepted_object_types"]["pypi_release"],
            ["pypi_package"],
        )
        self.assertIn("does not read product status pages", connector["description"])
        self.assertIn("supported by web_search", connector["description"])

    def test_model_call_validation_rejects_combined_enum_without_rewriting_it(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            ToolRegistry.validate_model_call(
                "connector_lookup",
                {
                    "operation": "github_repository,github_code,github_release",
                    "query": "repository release",
                },
            )

    def test_planner_prompt_contains_each_model_tool_description(self):
        catalog = json.loads(
            ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
        )
        prompt = Planner._system_prompt("ALL")
        for row in catalog:
            self.assertIn(row["description"].splitlines()[0], prompt)
        self.assertLess(
            prompt.index('"name":"connector_lookup"'),
            prompt.index('"name":"web_search"'),
        )
        self.assertLess(
            prompt.index('"name":"current_time"'),
            prompt.index('"name":"finish_task"'),
        )

    def test_rwkv_cross_validator_owns_finish_or_replan_decision(self):
        planner = Planner()
        raw = '{"name":"continue_retrieval","arguments":{}}'
        planner.llm = Mock()
        planner.llm.text_completion.return_value = Mock(content=raw)
        review = planner.review_evidence(
            "question",
            normalize_task_plan({
                "goal": "answer",
                "records": [
                    {
                        "question": "current version",
                        "fields": ["version"],
                    }
                ],
            }),
            "ORIGINAL_SOURCE_SPAN",
        )

        self.assertEqual(review["decision"], "replan")
        self.assertEqual(review["selected_action"], "continue_retrieval")
        self.assertEqual(review["contract"], "rwkv.ecra.runtime.evidence-review")
        self.assertNotIn("gap", review)
        self.assertEqual(review["raw_model_output"], raw)
        prompt = planner.llm.text_completion.call_args.args[0]
        self.assertIn("ORIGINAL_SOURCE_SPAN", prompt)
        self.assertIn('"name":"continue_retrieval"', prompt)
        self.assertIn('"name":"write_answer"', prompt)
        self.assertIn("minimal evidence check", prompt)
        self.assertTrue(prompt.endswith("Assistant: ```json\n"))
        self.assertNotIn('{"decision":"', prompt)

    def test_evidence_review_schema_is_a_strict_binary_empty_argument_contract(self):
        task_plan = normalize_task_plan(
            {
                "goal": "compare two releases",
                "records": [
                    {
                        "question": "first release",
                        "fields": ["version", "date"],
                    },
                    {
                        "question": "second release",
                        "fields": ["version"],
                    },
                ],
            }
        )

        tools = Planner._evidence_review_tools(
            task_plan,
            allowed_evidence_record_ids=["E-7", "E-9", "E-7"],
            allowed_route_ids=["route-a", "route-b"],
        )
        self.assertEqual(
            [row["name"] for row in tools],
            ["continue_retrieval", "write_answer"],
        )
        for row in tools:
            self.assertEqual(row["arguments"]["properties"], {})
            self.assertFalse(row["arguments"]["additionalProperties"])

    def test_evidence_review_rejects_any_non_binary_payload(self):
        task_plan = normalize_task_plan(
            {
                "goal": "answer both records",
                "records": [
                    {"question": "first", "fields": ["version"]},
                    {"question": "second", "fields": ["date"]},
                ],
            }
        )
        base = {"name": "continue_retrieval", "arguments": {}}

        accepted = Planner._validate_evidence_review(
            base,
            task_plan=task_plan,
            allowed_evidence_record_ids=["E-7", "E-9"],
            allowed_route_ids=["route-a"],
        )
        self.assertEqual(accepted["decision"], "replan")

        with self.assertRaisesRegex(ValueError, "arguments must be empty"):
            Planner._validate_evidence_review(
                {
                    "name": "continue_retrieval",
                    "arguments": {"task_record_id": "P1"},
                },
                task_plan=task_plan,
            )

    def test_evidence_review_preserves_write_intent_but_discards_embedded_answer(self):
        review = Planner._validate_evidence_review_continuation(
            '{"name":"write_answer","arguments":{"answer":'
            '"Model-authored prose is not the final Writer output."}}'
        )

        self.assertEqual(review["decision"], "finish")
        self.assertEqual(review["selected_action"], "write_answer")
        self.assertTrue(review["arguments_normalized"])
        self.assertTrue(review["protocol_normalized"])
        self.assertNotIn("answer", review)

    def test_rebuilt_planner_uses_request_level_replan_profile(self):
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
                        "profile": config.get_llm_sampling_parameters(),
                        "seed": config.get_llm_seed(),
                        "max_tokens": self_max_tokens,
                    }
                )
                return Mock(content='{"name":"finish_task","arguments":{}}')

        planner.llm = SamplingAwareRWKV()
        planner.rebuild_session(
            "same question",
            "shared evidence",
            {"status": "evidence_review_replan"},
            "REPLAN",
        )

        first = planner.plan_next_action("same question", {}, "shared evidence", "REPLAN")
        second = planner.plan_next_action("same question", {}, "shared evidence", "DISCOVERY")

        self.assertEqual(first["sampling_temperature"], 0.3)
        self.assertIsNone(first["sampling_seed"])
        self.assertIsNone(seen[0]["seed"])
        self.assertEqual(seen[0]["profile"], {
            "temperature": 0.3,
            "top_p": 0.6,
            "presence_penalty": 0.65,
            "frequency_penalty": 0.25,
            "top_k": 50,
        })
        self.assertEqual(second["sampling_temperature"], 0.1)
        self.assertIsNone(second["sampling_seed"])
        self.assertIsNone(seen[1]["seed"])

    def test_repeated_replans_do_not_mechanically_raise_temperature(self):
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
                {"status": "evidence_review_replan", "generation": generation},
                "REPLAN",
            )
            decisions.append(
                planner.plan_next_action(
                    "same question", {}, "unchanged shared evidence", "REPLAN"
                )
            )

        self.assertEqual(
            [round(row["sampling_temperature"], 2) for row in decisions],
            [0.3, 0.4, 0.4, 0.4, 0.4],
        )
        seeds = [row["sampling_seed"] for row in decisions]
        self.assertEqual(seeds, [None, None, None, None, None])
        self.assertEqual(
            [row["temperature"] for row in seen],
            [0.3, 0.4, 0.4, 0.4, 0.4],
        )
        self.assertEqual([row["seed"] for row in seen], seeds)

    def test_planner_repairs_invalid_tool_json_once_at_the_same_temperature(self):
        planner = Planner()
        seen = []

        class RepairingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(
                    {
                        "prompt": prompt,
                        "temperature": config.get_llm_temperature(),
                        "seed": config.get_llm_seed(),
                    }
                )
                if len(seen) == 1:
                    return Mock(content="not valid JSON")
                return Mock(content='{"name":"finish_task","arguments":{}}')

        planner.llm = RepairingRWKV()
        decision = planner.plan_next_action("question", {}, "state", "DISCOVERY")

        self.assertEqual(decision["action"], "finish_task")
        self.assertEqual(decision["planner_attempts"], 2)
        self.assertEqual([row["temperature"] for row in seen], [0.1, 0.1])
        self.assertEqual([row["seed"] for row in seen], [None, None])
        self.assertIn("Correction:", seen[1]["prompt"])
        self.assertLess(
            seen[1]["prompt"].rfind("Correction:"),
            seen[1]["prompt"].rfind("Assistant: ```json"),
        )

    def test_planner_repairs_invalid_enum_once_without_selecting_a_replacement(self):
        planner = Planner()
        seen = []

        class RepairingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append((prompt, config.get_llm_temperature()))
                if len(seen) == 1:
                    return Mock(
                        content=(
                            '{"name":"connector_lookup","arguments":'
                            '{"operation":"github_repository,github_code,github_release",'
                            '"query":"release"}}'
                        )
                    )
                return Mock(
                    content=(
                        '{"name":"connector_lookup","arguments":'
                        '{"operation":"github_release","query":"release"}}'
                    )
                )

        planner.llm = RepairingRWKV()
        decision = planner.plan_next_action("question", {}, "state", "DISCOVERY")

        self.assertEqual(decision["action"], "connector_lookup")
        self.assertEqual(decision["args"]["operation"], "github_release")
        self.assertEqual(decision["planner_attempts"], 2)
        self.assertEqual([temperature for _, temperature in seen], [0.1, 0.1])
        self.assertIn("must be one of", seen[1][0])

    def test_planner_uses_independent_online_g1i_function_requests(self):
        planner = Planner()
        prompts = []
        outputs = [
            '{"name":"web_search","arguments":{"query":"RWKV release"}}',
            '{"name":"finish_task","arguments":{}}',
        ]

        class G1IRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                prompts.append(prompt)
                return Mock(content=outputs.pop(0))

        planner.llm = G1IRWKV()
        planner.begin_task(
            "Find the RWKV release.",
            "runtime",
            {
                "goal": "Find the RWKV release.",
                "records": [
                    {"id": "P1", "task": "release", "objective": "find release"}
                ],
            },
        )
        first = planner.plan_next_action(
            "Find the RWKV release.", {}, "runtime", "DISCOVERY"
        )
        planner.observe_tool_result(
            {"status": "ok", "query": "RWKV release", "results": [{"url": "https://example.test"}]}
        )
        second = planner.plan_next_action(
            "Find the RWKV release.", {}, "runtime", "DISCOVERY"
        )

        self.assertEqual(first["action"], "web_search")
        self.assertEqual(second["action"], "finish_task")
        self.assertTrue(prompts[0].startswith("System: Tools: ["))
        self.assertIn("\nReturn only a JSON function call.\n\nUser: ", prompts[0])
        self.assertTrue(prompts[0].endswith("Assistant: ```json\n"))
        self.assertEqual(prompts[1].count("System:"), 1)
        self.assertNotIn("\n\nUser: Function output: ", prompts[1])
        self.assertIn("Find the RWKV release.", prompts[1])
        self.assertIn("Latest completed tool outcome", prompts[1])
        self.assertIn('"status":"ok"', prompts[1])
        self.assertNotIn(
            'Assistant: ```json\n{"name":"web_search","arguments":{"query":"RWKV release"}}',
            prompts[1],
        )
        self.assertNotIn('"query":"RWKV release"', prompts[1])
        self.assertEqual(prompts[1].count("Assistant: ```json"), 1)
        self.assertTrue(prompts[1].endswith("Assistant: ```json\n"))
        for prompt in prompts:
            self.assertNotIn("### User", prompt)
            self.assertNotIn("### Assistant", prompt)
            self.assertNotIn("**Tool Call:**", prompt)
            self.assertNotIn("### Tool Output", prompt)

    def test_recovery_rebuild_drops_failed_assistant_call_and_keeps_g1i_contract(self):
        planner = Planner()
        prompts = []

        class G1IRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                prompts.append(prompt)
                if len(prompts) == 1:
                    return Mock(content='{"name":"web_search","arguments":{"query":"same route"}}')
                return Mock(content='{"name":"web_search","arguments":{"query":"different exact gap"}}')

        planner.llm = G1IRWKV()
        task_plan = {
            "goal": "Find two facts.",
            "records": [{"id": "P1", "task": "facts", "objective": "find facts"}],
        }
        authoritative_state = (
            'Shared retrieval ledger:\n'
            '{"queries":[{"action":"web_search","query":"same route",'
            '"task_record_id":"P1","status":"completed"}],'
            '"frozen_paths":[{"action":"web_search","query":"same route",'
            '"task_record_id":"P1","reason":"duplicate"}]}'
        )
        planner.begin_task("Find two facts.", "runtime", task_plan)
        planner.plan_next_action("Find two facts.", {}, "runtime", "DISCOVERY")
        planner.observe_tool_result({"status": "no_new_evidence", "query": "same route"})
        planner.request_recovery_turn(
            {
                "status": "no_new_evidence",
                "frozen_path": {"action": "web_search", "query": "same route"},
            },
            user_query="Find two facts.",
            env_context=authoritative_state,
            phase="DISCOVERY",
        )
        decision = planner.plan_next_action(
            "Find two facts.", {}, authoritative_state, "DISCOVERY"
        )

        self.assertEqual(decision["args"]["query"], "different exact gap")
        self.assertTrue(prompts[1].startswith("System: Tools: ["))
        self.assertIn("RECOVERY INSTRUCTION", prompts[1])
        self.assertIn('"query":"same route"', prompts[1])
        self.assertEqual(prompts[1].count('"query":"same route"'), 1)
        self.assertNotIn("route_text_withheld", prompts[1])
        self.assertNotIn(
            'Assistant: ```json\n{"name":"web_search","arguments":{"query":"same route"}}',
            prompts[1],
        )
        self.assertIn("Latest completed tool outcome", prompts[1])
        self.assertNotIn('"previous_request"', prompts[1])
        self.assertEqual(prompts[1].count("Assistant: ```json"), 1)
        self.assertTrue(prompts[1].endswith("Assistant: ```json\n"))

    def test_multi_point_tool_call_without_inline_id_remains_unbound(self):
        planner = Planner()
        planner.begin_task(
            "mixed question",
            "state",
            {
                "goal": "answer three time-anchored facts",
                "records": [
                    {"id": "P1", "task": "historical launch"},
                    {"id": "P2", "task": "first anniversary"},
                    {"id": "P3", "task": "current theme"},
                ],
            },
        )
        seen = []

        class PlannerRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(prompt)
                return Mock(content='{"name":"web_search","arguments":{"query":"launch date"}}')

        planner.llm = PlannerRWKV()
        decision = planner.plan_next_action(
            "mixed question",
            {},
            "shared state",
            "DISCOVERY",
        )

        self.assertEqual(decision["action"], "web_search")
        self.assertEqual(decision["args"], {"query": "launch date"})
        self.assertEqual(decision["task_record_id"], "")
        self.assertEqual(decision["planner_attempts"], 1)
        self.assertEqual(
            decision["task_record_binding_method"],
            "multi_record_unbound",
        )
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].endswith("Assistant: ```json\n"))
        self.assertNotIn("bind_task_record", seen[0])

    def test_invalid_inline_task_record_remains_unbound_without_regenerating_action(self):
        planner = Planner()
        planner.begin_task(
            "mixed question",
            "state",
            {
                "goal": "answer two facts",
                "records": [
                    {"id": "P1", "task": "historical launch"},
                    {"id": "P2", "task": "current theme"},
                ],
            },
        )
        seen = []

        class PlannerRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(prompt)
                return Mock(
                    content=(
                        '{"name":"web_search","task_record_id":"PX",'
                        '"arguments":{"query":"current theme"}}'
                    )
                )

        planner.llm = PlannerRWKV()
        decision = planner.plan_next_action(
            "mixed question",
            {},
            "shared state",
            "DISCOVERY",
        )

        self.assertEqual(decision["action"], "web_search")
        self.assertEqual(decision["args"], {"query": "current theme"})
        self.assertEqual(decision["task_record_id"], "")
        self.assertNotIn("planner_error", decision)
        self.assertEqual(
            decision["task_record_binding_method"],
            "multi_record_unbound",
        )
        self.assertEqual(decision["task_record_binding_error"], "")
        self.assertEqual(len(seen), 1)
        self.assertNotIn("bind_task_record", seen[0])

    def test_single_point_tool_call_uses_the_only_structural_task_record(self):
        planner = Planner()
        planner.begin_task(
            "single question",
            "state",
            {
                "goal": "find one release",
                "records": [{"id": "P1", "task": "current release"}],
            },
        )
        seen = []

        class PlannerRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(prompt)
                return Mock(
                    content='{"name":"web_search","arguments":{"query":"current release"}}'
                )

        planner.llm = PlannerRWKV()
        decision = planner.plan_next_action(
            "single question", {}, "shared state", "DISCOVERY"
        )

        self.assertEqual(decision["task_record_id"], "P1")
        self.assertEqual(decision["task_record_binding_method"], "sole_task_record")
        self.assertEqual(len(seen), 1)

    def test_cross_validator_repairs_invalid_json_once_at_the_same_temperature(self):
        planner = Planner()
        seen = []

        class RepairingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append((prompt, config.get_llm_temperature()))
                if len(seen) == 1:
                    return Mock(content="not valid JSON")
                return Mock(
                    content='{"name":"write_answer","arguments":{}}'
                )

        planner.llm = RepairingRWKV()
        review = planner.review_evidence("question", {"records": []}, "evidence")

        self.assertEqual(review["decision"], "finish")
        self.assertEqual(review["review_attempts"], 2)
        self.assertEqual([temperature for _, temperature in seen], [0.1, 0.1])
        self.assertIn("Protocol correction:", seen[1][0])
        self.assertNotIn('"task_record_id":"P1"', seen[0][0])
        self.assertNotIn('"field_ids":["P1:F1"]', seen[0][0])
        self.assertNotIn('"task_record_id":"P1"', seen[1][0])
        self.assertNotIn('"field_ids":["P1:F1"]', seen[1][0])
        self.assertLess(
            seen[1][0].rfind("Protocol correction:"),
            seen[1][0].rfind("Assistant: ```json"),
        )

    def test_cross_validator_repairs_extra_keys_at_the_same_temperature(self):
        planner = Planner()
        seen = []

        class ContradictoryThenConsistentRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append((prompt, config.get_llm_temperature()))
                if len(seen) == 1:
                    return Mock(
                        content=(
                            '{"name":"write_answer","arguments":{},'
                            '"reason":"P2 is not retrieved"}'
                        )
                    )
                return Mock(
                    content='{"name":"write_answer","arguments":{}}'
                )

        planner.llm = ContradictoryThenConsistentRWKV()
        review = planner.review_evidence(
            "question with two points",
            {"records": [{"id": "P1"}, {"id": "P2"}]},
            "P1 evidence only",
        )

        self.assertEqual(review["decision"], "finish")
        self.assertEqual(review["review_attempts"], 2)
        self.assertEqual([temperature for _, temperature in seen], [0.1, 0.1])
        self.assertEqual(len(seen), 2)

    def test_cross_validator_accepts_rwkv_owned_finish_without_hidden_gate(self):
        planner = Planner()
        seen = []

        class ShortcutThenBoundRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(prompt)
                if len(seen) == 1:
                    return Mock(
                        content='{"name":"write_answer","arguments":{}}'
                    )
                return Mock(
                    content=(
                        '{"name":"continue_retrieval","arguments":{},'
                        '"reason":"P2 has no bound evidence"}'
                    )
                )

        planner.llm = ShortcutThenBoundRWKV()
        review = planner.review_evidence(
            "two-point question",
            {"records": [{"id": "P1"}, {"id": "P2"}]},
            "[S1] evidence for P1 only",
        )

        self.assertEqual(review["decision"], "finish")
        self.assertEqual(review["review_attempts"], 1)
        self.assertEqual(len(seen), 1)
        self.assertNotIn("task_record_status", seen[0])

    def test_cross_validator_does_not_replay_task_plan_protocol_schema(self):
        planner = Planner()
        prompts = []

        class CapturingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                prompts.append(prompt)
                return Mock(
                    content='{"name":"write_answer","arguments":{}}'
                )

        planner.llm = CapturingRWKV()
        review = planner.review_evidence(
            "short how-to",
            normalize_task_plan({
                "goal": "complete the how-to",
                "records": [
                    {
                        "question": "steps",
                        "fields": [],
                        "time_scope": "timeless",
                    }
                ],
            }),
            "[S1] direct source steps",
        )

        self.assertEqual(review["decision"], "finish")
        self.assertIn("TASK PLAN RECORDS TO REVIEW", prompts[0])
        self.assertIn('"records_to_check"', prompts[0])
        self.assertIn("RETAINED SOURCE SPANS", prompts[0])
        self.assertIn("minimal evidence check", prompts[0])
        self.assertNotIn('"contract": "rwkv.ecra.runtime.task-plan"', prompts[0])
        self.assertEqual(prompts[0].count("short how-to"), 1)
        self.assertEqual(prompts[0].count('"records_to_check"'), 1)
        self.assertTrue(prompts[0].endswith("Assistant: ```json\n"))

    def test_rebuilt_planner_requests_one_rwkv_function_call_with_replan_state(self):
        planner = Planner()
        prompts = []
        sampling_profiles = []
        task_plan = {
            "goal": "answer both facts",
            "records": [
                {"id": "P1", "task": "historical fact"},
                {"id": "P2", "task": "current fact"},
            ],
        }

        class CapturingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                prompts.append(prompt)
                sampling_profiles.append(
                    {
                        "stage": config.get_model_request_stage(),
                        **config.get_llm_sampling_parameters(),
                    }
                )
                return Mock(
                    content=(
                        '{"name":"web_search","task_record_id":"P2",'
                        '"arguments":{"query":"model-selected P2 route"}}'
                    )
                )

        planner.llm = CapturingRWKV()
        planner.begin_task("mixed question", "", task_plan, "DISCOVERY")
        decision = planner.rebuild_session_after_review(
            "mixed question",
            "shared state",
            {
                "decision": "replan",
                "missing_point_id": "P2",
                "evidence_needed": "current fact",
            },
            "REPLAN",
            routing_observation={
                "frozen_path": {
                    "action": "web_search",
                    "query": "already attempted P1 query",
                    "task_record_id": "P1",
                    "reason": "exact_duplicate_request",
                }
            },
        )

        self.assertEqual(len(prompts), 1)
        self.assertTrue(prompts[0].startswith("System: Tools: ["))
        self.assertIn("\nReturn only a JSON function call.\n\nUser: ", prompts[0])
        self.assertIn("RECOVERY INSTRUCTION", prompts[0])
        self.assertIn('"decision":"replan"', prompts[0])
        self.assertNotIn('"missing_point_id"', prompts[0])
        self.assertNotIn('"evidence_needed"', prompts[0])
        self.assertIn("already attempted P1 query", prompts[0])
        self.assertEqual(prompts[0].count("already attempted P1 query"), 1)
        self.assertIn('"frozen_route_count":1', prompts[0])
        self.assertNotIn("route_text_withheld", prompts[0])
        self.assertNotIn('"candidates"', prompts[0])
        self.assertTrue(prompts[0].endswith("Assistant: ```json\n"))
        self.assertEqual(decision["router"], "model_rwkv_json")
        self.assertEqual(decision["task_record_id"], "P2")
        self.assertEqual(decision["args"]["query"], "model-selected P2 route")
        self.assertEqual(sampling_profiles[0]["stage"], "planner_replan")
        self.assertEqual(sampling_profiles[0]["temperature"], 0.3)
        self.assertEqual(sampling_profiles[0], {
            "stage": "planner_replan",
            "temperature": 0.3,
            "top_p": 0.6,
            "presence_penalty": 0.65,
            "frequency_penalty": 0.25,
            "top_k": 50,
        })
        planner.mark_replan_progress("P2")
        self.assertEqual(planner._next_decision_sampling_stage, "planner")

    def test_replan_protocol_repair_keeps_route_count_without_replaying_query(self):
        planner = Planner()
        outputs = [
            '{"candidates":[{"name":"web_search","arguments":{"query":"already searched route"}}]}',
            (
                '{"name":"web_search","task_record_id":"P1",'
                '"arguments":{"query":"novel official release archive"}}'
            ),
        ]
        prompts = []

        class CorrectingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                prompts.append(prompt)
                return Mock(content=outputs.pop(0))

        planner.llm = CorrectingRWKV()
        planner._task_plan = {
            "goal": "find the current release",
            "records": [{"id": "P1", "task": "current release"}],
        }
        planner.rebuild_session(
            "find the current release",
            "shared evidence state",
            {
                "status": "evidence_review_replan",
                "evidence_review": {
                    "decision": "replan",
                    "missing_point_id": "P1",
                    "evidence_needed": "missing current release evidence",
                },
                "frozen_path": {
                    "action": "web_search",
                    "query": "already searched route",
                },
            },
            "REPLAN",
        )

        decision = planner.plan_next_action(
            "find the current release", {}, "shared evidence state", "REPLAN"
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("already searched route", prompts[0])
        self.assertEqual(prompts[0].count("already searched route"), 1)
        self.assertIn('"frozen_route_count":1', prompts[0])
        self.assertIn("Correction:", prompts[1])
        self.assertTrue(prompts[1].endswith("Assistant: ```json\n"))
        self.assertEqual(decision["args"]["query"], "novel official release archive")
        self.assertEqual(decision["router"], "model_rwkv_json")
        self.assertEqual(decision["planner_attempts"], 2)
        self.assertNotIn("candidate_actions", decision)

    def test_regular_planner_schema_example_does_not_inherit_cv_hypothesis(self):
        planner = Planner()
        planner._task_plan = {
            "records": [
                {"id": "P1", "task": "historical fact"},
                {"id": "P2", "task": "current fact"},
            ]
        }
        planner._active_replan_review = {"decision": "replan"}

        body = planner._build_isolated_decision_body(
            "mixed question",
            "shared state",
            "REPLAN",
        )

        self.assertIn(
            'Return one JSON object only: {"name":"tool_name","arguments":{}}.',
            body,
        )
        self.assertIn("task_record_id is optional routing metadata", body)
        self.assertIn("exact crates.io/PyPI/npm package release", body)
        self.assertIn("Changing only generic suffixes", body)
        self.assertNotIn("FINAL REPLAN FOCUS", body)

    def test_single_record_planner_keeps_explicit_verified_task_record_shape(self):
        planner = Planner()
        planner._task_plan = normalize_task_plan(
            {"goal": "current fact", "records": [{"question": "current fact"}]}
        )

        body = planner._build_isolated_decision_body(
            "single-record question",
            "shared state",
            "DISCOVERY",
        )

        self.assertIn(
            'Return one JSON object only: {"name":"tool_name","task_record_id":"P1","arguments":{}}.',
            body,
        )
        self.assertIn(
            "task_record_id must copy the existing id of the one factual record",
            body,
        )
        self.assertIn("attaches the only existing task record id", body)
        self.assertNotIn("task_record_id is optional routing metadata", body)

    def test_replan_environment_projection_keeps_late_source_spans(self):
        context = "\n".join(
            [
                "Retrieval task state:",
                "Task: current release",
                "Latest controller feedback (compact routing projection):",
                json.dumps({"message": "x" * 5000}),
                "Question time/freshness policy (observable metadata, not a gate):",
                json.dumps({"now": "2026-08-10T00:00:00Z"}),
                "Shared retrieval ledger (compact progress metadata; not a finish gate):",
                json.dumps({"queries": [{"query": "old route"}]}),
                "Bounded original source locators for routing (not final-answer evidence):",
                json.dumps(
                    {
                        "source_count": 1,
                        "sources": [
                            {
                                "source_id": "R1",
                                "url": "https://example.com/current",
                                "task_record_ids": ["P2"],
                                "spans": [
                                    {
                                        "chunk_id": "chunk-9",
                                        "text": "LATE_CURRENT_SOURCE_SPAN",
                                    }
                                ],
                            }
                        ],
                    }
                ),
            ]
        )

        projected = Planner._replan_environment_projection(context)
        decoded = json.loads(projected)

        self.assertIn("LATE_CURRENT_SOURCE_SPAN", projected)
        self.assertIn("2026-08-10T00:00:00Z", projected)
        self.assertNotIn("x" * 100, projected)
        self.assertIn("old route", projected)
        self.assertEqual(decoded["retrieval_ledger"]["previous_route_count"], 1)
        self.assertEqual(
            decoded["retrieval_ledger"]["route_history"][0]["arguments"]["query"],
            "old route",
        )

    def test_rebuilt_runtime_state_is_valid_json_after_bounding_large_spans(self):
        planner = Planner()
        planner._task_plan = {
            "goal": "find current release",
            "records": [{"id": "P1", "task": "current release"}],
        }
        context = "\n".join(
            [
                "Shared retrieval ledger (compact progress metadata; not a finish gate):",
                json.dumps(
                    {
                        "queries": [{"query": "OLD_QUERY_SHOULD_NOT_REPLAY", "status": "ok"}],
                        "frozen_paths": [
                            {
                                "query": "OLD_QUERY_SHOULD_NOT_REPLAY",
                                "reason": "equivalent_duplicate_query",
                            }
                        ],
                        "source_count": 3,
                    }
                ),
                "Bounded original source locators for routing (not final-answer evidence):",
                json.dumps(
                    {
                        "source_count": 3,
                        "sources": [
                            {
                                "source_id": f"R{index}",
                                "title": f"source {index}",
                                "url": f"https://example.com/{index}",
                                "spans": [
                                    {
                                        "chunk_id": f"chunk-{index}",
                                        "text": "ORIGINAL_BOUND_SPAN " + ("x" * 5000),
                                    }
                                ],
                            }
                            for index in range(3)
                        ],
                    }
                ),
            ]
        )
        planner.rebuild_session(
            "find current release",
            context,
            {
                "status": "evidence_review_replan",
                "request": {
                    "name": "web_search",
                    "arguments": {"query": "OLD_QUERY_SHOULD_NOT_REPLAY"},
                },
                "retrieval_delta": {
                    "step": 2,
                    "action": "web_search",
                    "query": "OLD_QUERY_SHOULD_NOT_REPLAY",
                    "query_key": "old_query_should_not_replay",
                    "urls": ["https://example.com/old-route"],
                    "new_url_count": 0,
                    "evidence_count": 0,
                    "status": "no_new_evidence",
                },
                "frozen_path": {
                    "query": "OLD_QUERY_SHOULD_NOT_REPLAY",
                    "reason": "equivalent_duplicate_query",
                },
                "evidence_review": {"decision": "replan", "trigger": "duplicate"},
            },
            "REPLAN",
        )

        body = planner._build_isolated_decision_body(
            "find current release", context, "REPLAN"
        )
        runtime_text = body.split("Runtime state: ", 1)[1].split(
            "\nDecision number:", 1
        )[0]
        runtime = json.loads(runtime_text)

        self.assertIn("OLD_QUERY_SHOULD_NOT_REPLAY", body)
        self.assertEqual(body.count("OLD_QUERY_SHOULD_NOT_REPLAY"), 1)
        self.assertEqual(runtime["retrieval_ledger"]["previous_route_count"], 1)
        self.assertEqual(runtime["retrieval_ledger"]["frozen_route_count"], 1)
        self.assertTrue(runtime["retrieval_ledger"]["route_history"][0]["frozen"])
        self.assertEqual(len(runtime["source_locators"]["sources"]), 3)
        self.assertIn(
            "ORIGINAL_BOUND_SPAN",
            runtime["source_locators"]["sources"][0]["spans"][0]["text"],
        )
        self.assertTrue(
            runtime["source_locators"]["sources"][0]["spans"][0]["truncated"]
        )

    def test_replan_temperature_is_selected_by_failure_type(self):
        self.assertEqual(config.get_model_replan_temperature(1, "material missing"), 0.3)
        self.assertEqual(config.get_model_replan_temperature(1, "source conflict"), 0.4)
        self.assertEqual(config.get_model_replan_temperature(1, "protocol error"), 0.1)
        self.assertEqual(config.get_model_replan_temperature(1, "duplicate frozen path"), 0.3)
        self.assertEqual(config.get_model_replan_temperature(2, "replan stalled"), 0.4)

    def test_task_plan_repairs_invented_year_without_controller_rewrite(self):
        planner = Planner()
        outputs = [
            {
                "contract": "rwkv.ecra.runtime.task-plan",
                "goal": "answer current release",
                "records": [
                    {
                        "record_id": "P1",
                        "question": "find the 2024 release",
                        "fields": [{"field_id": "P1:F1", "name": "version"}],
                        "time_scope": "current",
                    }
                ],
            },
            {
                "contract": "rwkv.ecra.runtime.task-plan",
                "goal": "answer current release",
                "records": [
                    {
                        "record_id": "P1",
                        "question": "find the current release",
                        "fields": [{"field_id": "P1:F1", "name": "version"}],
                        "time_scope": "current",
                    }
                ],
            },
        ]
        seen = []

        class RepairingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(
                    {
                        "prompt": prompt,
                        "temperature": config.get_llm_temperature(),
                        "stage": config.get_model_request_stage(),
                        "sampling": config.get_llm_sampling_parameters(),
                    }
                )
                return Mock(content=json.dumps(outputs[len(seen) - 1]))

        planner.llm = RepairingRWKV()
        result = planner.create_task_plan(
            "What is the current release?",
            "Current UTC date: 2026-08-09",
        )

        self.assertEqual(result["records"][0]["question"], "find the current release")
        self.assertEqual(result["contract"], "rwkv.ecra.runtime.task-plan")
        self.assertEqual(result["plan_attempts"], 2)
        self.assertEqual([row["temperature"] for row in seen], [0.1, 0.1])
        self.assertEqual(
            [row["stage"] for row in seen],
            ["task_plan", "task_plan_repair"],
        )
        self.assertEqual(seen[1]["sampling"], {
            "temperature": 0.1,
            "top_p": 0.3,
            "presence_penalty": 0.00001,
            "frequency_penalty": 0.00001,
            "top_k": 40,
        })
        self.assertIn("unexpected_year:2024", seen[1]["prompt"])

    def test_task_plan_allows_one_same_temperature_repair_then_keeps_warning(self):
        planner = Planner()

        def plan(task: str) -> dict:
            return {
                "contract": "rwkv.ecra.runtime.task-plan",
                "goal": "answer current release",
                "records": [
                    {
                        "record_id": "P1",
                        "question": task,
                        "fields": [{"field_id": "P1:F1", "name": "version"}],
                        "time_scope": "current",
                    }
                ],
            }

        outputs = [plan("find the 2024 release"), plan("find the 2025 release"), plan("find the current release")]
        seen = []

        class TwiceRepairingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del prompt, max_tokens, stop
                seen.append(
                    (
                        config.get_model_request_stage(),
                        config.get_llm_temperature(),
                    )
                )
                return Mock(content=json.dumps(outputs[len(seen) - 1]))

        planner.llm = TwiceRepairingRWKV()
        result = planner.create_task_plan(
            "What is the current release?",
            "Current UTC date: 2026-08-09",
        )

        self.assertEqual(result["records"][0]["question"], "What is the current release?")
        self.assertEqual(result["plan_attempts"], 2)
        self.assertTrue(result["plan_fallback"])
        self.assertIn("unexpected_year:2025", result["plan_error"])
        self.assertEqual(
            seen,
            [
                ("task_plan", 0.1),
                ("task_plan_repair", 0.1),
            ],
        )

    def test_cross_validator_does_not_derive_decision_from_legacy_status_fields(self):
        planner = Planner()
        calls = []

        class LegacySchemaRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                calls.append(prompt)
                return Mock(
                    content=(
                        '{"task_record_status":{"P1":{"status":"completed"},'
                        '"P2":{"status":"not_retrieved"}},'
                        '"missing_points":["P2"]}'
                    )
                )

        planner.llm = LegacySchemaRWKV()
        review = planner.review_evidence(
            "mixed question",
            {"records": [{"id": "P1"}, {"id": "P2"}]},
            "P1 evidence only",
        )

        self.assertEqual(review["decision"], "protocol_error")
        self.assertEqual(review["review_attempts"], 2)
        self.assertEqual(len(calls), 2)
        self.assertIn("exactly name and arguments", calls[1])

    def test_cross_validator_accepts_minimal_binary_replan(self):
        planner = Planner()

        class ReplanRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del prompt, max_tokens, stop
                return Mock(content='{"name":"continue_retrieval","arguments":{}}')

        planner.llm = ReplanRWKV()
        review = planner.review_evidence(
            "mixed question",
            {"records": [{"id": "P1"}, {"id": "P2"}]},
            "P1 evidence only",
        )

        self.assertEqual(review["contract"], "rwkv.ecra.runtime.evidence-review")
        self.assertEqual(review["decision"], "replan")
        self.assertEqual(review["review_attempts"], 1)
        self.assertNotIn("gap", review)
        self.assertNotIn("task_record_status", review)

    def test_cross_validator_never_fact_checks_or_rewrites_rwkv_decision(self):
        planner = Planner()
        calls = []

        class FinishRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                calls.append(prompt)
                return Mock(content='{"name":"write_answer","arguments":{}}')

        planner.llm = FinishRWKV()
        review = planner.review_evidence(
            "What is the current version theme?",
            {"records": [{"id": "P1", "task": "current version theme"}]},
            "[S1] Version 3.1 theme: Long Goodbye",
        )

        self.assertEqual(review["decision"], "finish")
        self.assertEqual(review["review_attempts"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(review).intersection({"supported_facts", "reason"}), set())

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
        orchestrator._review_evidence = Mock(return_value={"decision": "finish"})
        orchestrator._complete_model_tool_loop = Mock(return_value="done")
        with patch("agent.orchestrator.append_task_event"):
            result = orchestrator._run_single_loop(
                "Use the tools to check the time, calculate 15*2, and find the date difference.",
                {},
                {"records": []},
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
        orchestrator._review_evidence = Mock(return_value={"decision": "finish"})
        orchestrator._complete_model_tool_loop = Mock(return_value="current date")
        with patch("agent.orchestrator.append_task_event"):
            result = orchestrator._run_single_loop(
                "现在的日期是什么？",
                {},
                {
                    "task_mode": "current_time",
                    "records": [
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

    def test_protocol_adapter_accepts_common_function_call_wrapper_losslessly(self):
        call = canonicalize_tool_call(
            {
                "function_call": {
                    "name": "web_search",
                    "arguments": '{"query":"RWKV current release"}',
                },
                "task_record_id": "P2",
            }
        )
        self.assertEqual(call["name"], "web_search")
        self.assertEqual(call["arguments"], {"query": "RWKV current release"})
        self.assertEqual(call["task_record_id"], "P2")

    def test_protocol_adapter_rejects_function_call_conflicts(self):
        with self.assertRaisesRegex(ValueError, "conflicting name aliases"):
            canonicalize_tool_call(
                {
                    "name": "finish_task",
                    "function_call": {"name": "web_search", "arguments": {}},
                }
            )

    def test_format_converter_only_maps_representation(self):
        call = normalize_tool_call_format(
            {
                "tool_name": "web_search_generic",
                "args": {"query": "RWKV"},
                "task_record_id": "P2",
            }
        )
        self.assertEqual(
            call,
            {
                "contract": "rwkv.ecra.runtime.tool-call",
                "name": "web_search_generic",
                "arguments": {"query": "RWKV"},
                "task_record_id": "P2",
            },
        )

    def test_format_converter_accepts_single_named_call_wrapper_losslessly(self):
        call = normalize_tool_call_format(
            {"select_evidence": {"candidate_ids": ["E1", "E3"]}}
        )
        self.assertEqual(
            call,
            {
                "contract": "rwkv.ecra.runtime.tool-call",
                "name": "select_evidence",
                "arguments": {"candidate_ids": ["E1", "E3"]},
            },
        )

    def test_format_converter_does_not_treat_scalar_single_key_as_call(self):
        with self.assertRaisesRegex(ValueError, "tool call name is empty"):
            normalize_tool_call_format({"answer": "not a tool call"})

    def test_protocol_adapter_rejects_multiple_or_conflicting_calls(self):
        with self.assertRaisesRegex(ValueError, "exactly one call"):
            canonicalize_tool_call(
                {
                    "tool_calls": [
                        {"function": {"name": "calculator", "arguments": {}}},
                        {"function": {"name": "web_search", "arguments": {}}},
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "conflicting name aliases"):
            canonicalize_tool_call(
                {
                    "name": "calculator",
                    "action": "web_search",
                    "arguments": {},
                }
            )
        with self.assertRaisesRegex(ValueError, "conflicting argument aliases"):
            canonicalize_tool_call(
                {
                    "name": "calculator",
                    "arguments": {"expression": "1+1"},
                    "args": {"expression": "2+2"},
                }
            )

    def test_protocol_adapter_rejects_conflicts_inside_native_tool_call(self):
        with self.assertRaisesRegex(ValueError, "conflicting name aliases"):
            canonicalize_tool_call(
                {
                    "name": "finish_task",
                    "arguments": {},
                    "tool_calls": [
                        {
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "RWKV"},
                            }
                        }
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "conflicting argument aliases"):
            canonicalize_tool_call(
                {
                    "tool_calls": [
                        {
                            "name": "web_search",
                            "arguments": {"query": "first"},
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "second"},
                            },
                        }
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "conflicting task-record aliases"):
            canonicalize_tool_call(
                {
                    "task_record_id": "P1",
                    "tool_calls": [
                        {
                            "task_record_id": "P2",
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "RWKV"},
                            },
                        }
                    ],
                }
            )

    def test_protocol_adapter_accepts_identical_native_aliases_losslessly(self):
        call = canonicalize_tool_call(
            {
                "name": "web_search",
                "arguments": {"query": "RWKV"},
                "task_record_id": "P1",
                "call_id": "call-9",
                "tool_calls": [
                    {
                        "id": "call-9",
                        "name": "web_search",
                        "task_record_id": "P1",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"RWKV"}',
                        },
                    }
                ],
            }
        )
        self.assertEqual(
            call,
            {
                "contract": "rwkv.ecra.runtime.tool-call",
                "name": "web_search",
                "arguments": {"query": "RWKV"},
                "call_id": "call-9",
                "task_record_id": "P1",
            },
        )

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
                {"operation": "weather_current", "query": "Shanghai"},
                {"agentic_tool_loop": True},
                phase="ALL",
            )
        )
        self.assertEqual(result["connector"], "weather")
        self.assertEqual(result["results"][0]["evidence_origin"], "structured_api_record")
        self.assertEqual(result["freshness_policy"]["mode"], "retrieval_time_only")

    @patch("tools.connectors.get_current_weather_alerts")
    def test_weather_alert_connector_accepts_natural_current_scope(self, alerts):
        alerts.return_value = json.dumps(
            {"status": "ok", "results": [], "sources": []}
        )
        result = json.loads(
            ToolRegistry.execute(
                "connector_lookup",
                {
                    "operation": "weather_alerts",
                    "query": "上海市",
                },
                {"agentic_tool_loop": True},
                phase="ALL",
            )
        )
        self.assertNotEqual(result.get("error_class"), "invalid_arguments")

    @patch("tools.connectors.get_current_weather")
    def test_weather_not_found_is_not_promoted_to_structured_evidence(self, weather):
        weather.return_value = json.dumps(
            {"status": "not_found", "location": "Missing Place"}
        )
        result = json.loads(
            ToolRegistry.execute(
                "connector_lookup",
                {"operation": "weather_current", "query": "Missing Place"},
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

    def test_search_snippet_date_is_not_promoted_to_source_date(self):
        policy = build_freshness_policy("What is the latest release?")
        row = annotate_freshness(
            {
                "url": "https://example.com/releases",
                "title": "Release history",
                "snippet": "A nearby record was published on 2026-07-31.",
            },
            policy,
        )

        self.assertIsNone(row["freshness"]["source_date"])
        self.assertIsNone(row["freshness"]["source_date_origin"])

    def test_dated_record_url_is_safe_source_date_metadata(self):
        policy = build_freshness_policy("What is the latest bulletin?")
        row = annotate_freshness(
            {
                "url": "https://source.example/bulletin/2026/2026-08-01",
                "snippet": "Patch level 2026-08-05 fixes the listed issues.",
            },
            policy,
        )

        self.assertEqual(row["freshness"]["source_date"], "2026-08-01")
        self.assertEqual(
            row["freshness"]["source_date_origin"], "url_record_identity"
        )

    def test_chinese_freshness_cutoff_is_parsed(self):
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
