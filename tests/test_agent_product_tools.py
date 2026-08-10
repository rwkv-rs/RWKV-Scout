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
            {"status": "cross_validation_replan"},
            "REPLAN",
        )

        first = planner.plan_next_action("same question", {}, "shared evidence", "REPLAN")
        second = planner.plan_next_action("same question", {}, "shared evidence", "DISCOVERY")

        self.assertEqual(first["sampling_temperature"], 0.8)
        self.assertIsNone(first["sampling_seed"])
        self.assertIsNone(seen[0]["seed"])
        self.assertEqual(seen[0]["profile"]["top_k"], 50)
        self.assertEqual(seen[0]["profile"]["top_p"], 0.6)
        self.assertEqual(seen[0]["profile"]["presence_penalty"], 0.65)
        self.assertEqual(seen[0]["profile"]["frequency_penalty"], 0.25)
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
            [0.8, 0.8, 0.8, 0.8, 0.8],
        )
        seeds = [row["sampling_seed"] for row in decisions]
        self.assertEqual(seeds, [None, None, None, None, None])
        self.assertEqual([row["temperature"] for row in seen], [0.8] * 5)
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
            seen[1]["prompt"].rfind("### Assistant"),
        )

    def test_multi_point_tool_call_repairs_missing_task_point_id(self):
        planner = Planner()
        planner.begin_task(
            "mixed question",
            "state",
            {
                "goal": "answer three time-anchored facts",
                "atomic_points": [
                    {"id": "P1", "task": "historical launch"},
                    {"id": "P2", "task": "first anniversary"},
                    {"id": "P3", "task": "current theme"},
                ],
            },
        )
        outputs = [
            '{"name":"web_search","arguments":{"query":"launch date"}}',
            '{"task_point_id":"P1"}',
        ]
        seen = []

        class RepairingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(prompt)
                return Mock(content=outputs[len(seen) - 1])

        planner.llm = RepairingRWKV()
        decision = planner.plan_next_action(
            "mixed question",
            {},
            "shared state",
            "DISCOVERY",
        )

        self.assertEqual(decision["action"], "web_search")
        self.assertEqual(decision["args"], {"query": "launch date"})
        self.assertEqual(decision["task_point_id"], "P1")
        self.assertEqual(decision["planner_attempts"], 1)
        self.assertEqual(decision["task_point_binding_method"], "rwkv_binding_request")
        self.assertIn("P1, P2, P3", seen[1])
        self.assertIn('"query":"launch date"', seen[1])
        self.assertNotIn("Correction:", seen[1])

    def test_multi_point_binding_failure_keeps_original_rwkv_action_unassigned(self):
        planner = Planner()
        planner.begin_task(
            "mixed question",
            "state",
            {
                "goal": "answer two facts",
                "atomic_points": [
                    {"id": "P1", "task": "historical launch"},
                    {"id": "P2", "task": "current theme"},
                ],
            },
        )
        outputs = [
            '{"name":"web_search","arguments":{"query":"current theme"}}',
            "not valid binding JSON",
        ]
        seen = []

        class BindingFailureRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(prompt)
                return Mock(content=outputs[len(seen) - 1])

        planner.llm = BindingFailureRWKV()
        decision = planner.plan_next_action(
            "mixed question",
            {},
            "shared state",
            "DISCOVERY",
        )

        self.assertEqual(decision["action"], "web_search")
        self.assertEqual(decision["args"], {"query": "current theme"})
        self.assertEqual(decision["task_point_id"], "")
        self.assertNotIn("planner_error", decision)
        self.assertEqual(decision["task_point_binding_method"], "unassigned")
        self.assertIn("ValueError", decision["task_point_binding_error"])

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
                    content=(
                        '{"decision":"finish","missing_points":[],"conflicts":[],'
                        '"next_focus":"","reason":"enough"}'
                    )
                )

        planner.llm = RepairingRWKV()
        review = planner.cross_validate_research("question", {"atomic_points": []}, "evidence")

        self.assertEqual(review["decision"], "finish")
        self.assertEqual(review["review_attempts"], 2)
        self.assertEqual([temperature for _, temperature in seen], [0.05, 0.05])
        self.assertIn("Correction:", seen[1][0])
        self.assertLess(
            seen[1][0].rfind("Correction:"),
            seen[1][0].rfind("### Assistant"),
        )

    def test_cross_validator_reasks_rwkv_for_a_consistent_finish_decision(self):
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
                            '{"decision":"finish","missing_points":["P2"],'
                            '"conflicts":[],"next_focus":"P2",'
                            '"reason":"P2 is not retrieved"}'
                        )
                    )
                return Mock(
                    content=(
                        '{"decision":"replan","missing_points":["P2"],'
                        '"conflicts":[],"next_focus":"P2",'
                        '"reason":"P2 is not retrieved"}'
                    )
                )

        planner.llm = ContradictoryThenConsistentRWKV()
        review = planner.cross_validate_research(
            "question with two points",
            {"atomic_points": [{"id": "P1"}, {"id": "P2"}]},
            "P1 evidence only",
        )

        self.assertEqual(review["decision"], "replan")
        self.assertEqual(review["missing_points"], ["P2"])
        self.assertEqual(review["review_attempts"], 2)
        self.assertEqual([temperature for _, temperature in seen], [0.05, 0.05])
        self.assertIn("internally inconsistent", seen[1][0])
        self.assertLess(
            seen[1][0].rfind("Protocol error:"),
            seen[1][0].rfind("### Assistant"),
        )

    def test_cross_validator_rejects_unbound_multi_point_finish_shortcut(self):
        planner = Planner()
        seen = []

        class ShortcutThenBoundRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                seen.append(prompt)
                if len(seen) == 1:
                    return Mock(
                        content=(
                            '{"decision":"finish","missing_points":[],"conflicts":[],'
                            '"next_focus":"","reason":"all points are covered"}'
                        )
                    )
                return Mock(
                    content=(
                        '{"decision":"replan","missing_points":["P2"],"conflicts":[],'
                        '"task_point_status":{'
                        '"P1":{"status":"supported","evidence_refs":["S1"]},'
                        '"P2":{"status":"missing","evidence_refs":[]}},'
                        '"next_focus":"P2","reason":"P2 has no bound evidence"}'
                    )
                )

        planner.llm = ShortcutThenBoundRWKV()
        review = planner.cross_validate_research(
            "two-point question",
            {"atomic_points": [{"id": "P1"}, {"id": "P2"}]},
            "[S1] evidence for P1 only",
            evidence_refs=["S1"],
        )

        self.assertEqual(review["decision"], "replan")
        self.assertEqual(review["missing_points"], ["P2"])
        self.assertEqual(review["review_attempts"], 2)
        self.assertIn("must cover every planned point", seen[1])
        self.assertEqual(
            review["task_point_status"]["P1"]["evidence_refs"], ["S1"]
        )

    def test_cross_validator_does_not_replay_task_plan_protocol_schema(self):
        planner = Planner()
        prompts = []

        class CapturingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                prompts.append(prompt)
                return Mock(
                    content=(
                        '{"schema_version":"rwkv-cross-validation.v1",'
                        '"decision":"finish","missing_points":[],"conflicts":[],'
                        '"task_point_status":{"P1":{"status":"supported",'
                        '"evidence_refs":["S1"]}},"next_focus":"",'
                        '"reason":"the requested fact is supported"}'
                    )
                )

        planner.llm = CapturingRWKV()
        review = planner.cross_validate_research(
            "short how-to",
            {
                "schema_version": "task_plan.v1",
                "goal": "complete the how-to",
                "atomic_points": [
                    {
                        "id": "P1",
                        "task": "steps",
                        "evidence_needed": ["direct steps"],
                    }
                ],
            },
            "[S1] direct source steps",
            evidence_refs=["S1"],
        )

        self.assertEqual(review["decision"], "finish")
        self.assertIn("USER-FACING FACTS TO CHECK", prompts[0])
        self.assertIn('"points_to_check"', prompts[0])
        self.assertIn("every requested_fields value", prompts[0])
        self.assertIn("date after the supplied current runtime", prompts[0])
        self.assertIn("exact owner/repository", prompts[0])
        self.assertNotIn('"schema_version":"task_plan.v1"', prompts[0])
        self.assertEqual(prompts[0].count("short how-to"), 2)
        self.assertEqual(prompts[0].count('"points_to_check"'), 2)
        self.assertLess(
            prompts[0].rfind("SHARED RESEARCH MATERIAL"),
            prompts[0].rfind("FINAL TARGET REMINDER"),
        )
        self.assertLess(
            prompts[0].rfind("FINAL TARGET REMINDER"),
            prompts[0].rfind("Return the cross-validation JSON now."),
        )

    def test_rebuilt_planner_mounts_rwkv_replan_review_and_compact_task_plan(self):
        planner = Planner()
        prompts = []
        sampling_profiles = []
        task_plan = {
            "goal": "answer both facts",
            "atomic_points": [
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
                        '{"candidates":['
                        '{"name":"web_search","task_point_id":"P2",'
                        '"arguments":{"query":"already attempted P1 query"}},'
                        '{"name":"web_search","task_point_id":"P2",'
                        '"arguments":{"query":"model-selected P2 route"}},'
                        '{"name":"connector_lookup","task_point_id":"P2",'
                        '"arguments":{"connector":"github","query":"P2"}}]}'
                    )
                )

        planner.llm = CapturingRWKV()
        planner.begin_task("mixed question", "", task_plan, "DISCOVERY")
        planner.rebuild_session_after_review(
            "mixed question",
            "shared state",
            {
                "decision": "replan",
                "missing_points": ["P2"],
                "conflicts": [],
                "next_focus": "current fact",
            },
            "REPLAN",
            routing_observation={
                "frozen_path": {
                    "action": "web_search",
                    "query": "already attempted P1 query",
                    "task_point_id": "P1",
                    "reason": "exact_duplicate_request",
                }
            },
        )
        decision = planner.plan_next_action(
            "mixed question", {}, "shared state", "REPLAN"
        )

        self.assertEqual(len(prompts), 1)
        self.assertIn("You are the RWKV replanner", prompts[0])
        self.assertIn("MISSING FACT OBLIGATIONS", prompts[0])
        self.assertIn("ACTIVE REVIEW AND FROZEN PATHS", prompts[0])
        self.assertIn("PRIOR DOMAIN HYPOTHESES (may be wrong)", prompts[0])
        self.assertIn('"missing_points":["P2"]', prompts[0])
        self.assertIn('"id":"P2"', prompts[0])
        self.assertIn('"frozen_paths":[', prompts[0])
        self.assertIn("already attempted P1 query", prompts[0])
        self.assertIn(
            '"candidates":[{"name":"tool_name","task_point_id":"P2","arguments":{}},'
            '{"name":"tool_name","task_point_id":"P2","arguments":{}},'
            '{"name":"tool_name","task_point_id":"P2","arguments":{}}]',
            prompts[0],
        )
        self.assertEqual(decision["router"], "model_rwkv_replan_json")
        self.assertEqual(decision["task_point_id"], "P2")
        self.assertEqual(decision["args"]["query"], "model-selected P2 route")
        second_candidate = planner.plan_next_action(
            "mixed question", {}, "no evidence from first candidate", "REPLAN"
        )
        self.assertEqual(len(prompts), 1)
        self.assertEqual(second_candidate["action"], "connector_lookup")
        self.assertEqual(second_candidate["args"]["connector"], "github")
        self.assertIn("CURRENT EVIDENCE STATE", prompts[0])
        self.assertIn('"description":', prompts[0])
        self.assertIn("weather|weather_alerts|github|papers", prompts[0])
        self.assertEqual(sampling_profiles[0]["stage"], "planner_replan")
        self.assertEqual(sampling_profiles[0]["temperature"], 0.8)
        self.assertEqual(sampling_profiles[0]["top_k"], 50)
        self.assertEqual(sampling_profiles[0]["top_p"], 0.6)
        self.assertEqual(sampling_profiles[0]["presence_penalty"], 0.65)
        self.assertEqual(sampling_profiles[0]["frequency_penalty"], 0.25)
        self.assertEqual(
            sampling_profiles[0]["no_penalty_token_ids"],
            (33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58),
        )
        self.assertEqual(planner._next_decision_sampling_stage, "planner_replan")
        planner.mark_replan_progress("P2")
        self.assertEqual(planner._next_decision_sampling_stage, "planner")

    def test_replan_rejects_attempted_ledger_paths_and_rwkv_corrects_once(self):
        planner = Planner()
        outputs = [
            (
                '{"candidates":['
                '{"name":"web_search","task_point_id":"P1",'
                '"arguments":{"query":"already searched route"}},'
                '{"name":"web_search","task_point_id":"P1",'
                '"arguments":{"query":"already searched route"}},'
                '{"name":"web_search","task_point_id":"P1",'
                '"arguments":{"query":"already searched route"}}]}'
            ),
            (
                '{"candidates":['
                '{"name":"web_search","task_point_id":"P1",'
                '"arguments":{"query":"novel official release archive"}},'
                '{"name":"web_search","task_point_id":"P1",'
                '"arguments":{"query":"alternate language release announcement"}},'
                '{"name":"web_search","task_point_id":"P1",'
                '"arguments":{"query":"structured version history source"}}]}'
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
        planner.begin_task(
            "find the current release",
            "",
            {
                "goal": "find the current release",
                "atomic_points": [{"id": "P1", "task": "current release"}],
            },
            "DISCOVERY",
        )
        planner._active_replan_review = {
            "missing_points": ["P1"],
            "reason": "missing current release evidence",
        }
        env_context = "\n".join(
            [
                "Shared retrieval ledger (compact progress metadata; not a finish gate):",
                json.dumps(
                    {
                        "queries": [
                            {
                                "query": "already searched route",
                                "task_point_id": "P1",
                                "status": "ok",
                            }
                        ]
                    }
                ),
            ]
        )

        decision = planner._request_replan_action(
            "find the current release", env_context, "REPLAN"
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("attempted_paths", prompts[0])
        self.assertIn("already searched route", prompts[0])
        self.assertIn("Correction:", prompts[1])
        self.assertEqual(decision["args"]["query"], "novel official release archive")
        self.assertEqual(decision["decision_owner"], "rwkv")
        self.assertEqual(decision["planner_attempts"], 2)
        self.assertEqual(len(decision["candidate_actions"]), 3)

    def test_regular_planner_schema_example_tracks_active_missing_point(self):
        planner = Planner()
        planner._task_plan = {
            "atomic_points": [
                {"id": "P1", "task": "historical fact"},
                {"id": "P2", "task": "current fact"},
            ]
        }
        planner._active_replan_review = {"missing_points": ["P2"]}

        body = planner._build_isolated_decision_body(
            "mixed question",
            "shared state",
            "REPLAN",
        )

        self.assertIn(
            'Return one JSON object only: {"name":"tool_name",'
            '"task_point_id":"P2","arguments":{}}.',
            body,
        )

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
                                "claim_ids": ["P2"],
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

        self.assertIn("LATE_CURRENT_SOURCE_SPAN", projected)
        self.assertIn("2026-08-10T00:00:00Z", projected)
        self.assertNotIn("x" * 100, projected)

    def test_replan_temperature_is_selected_by_failure_type(self):
        self.assertEqual(config.get_model_replan_temperature(1, "material missing"), 0.8)
        self.assertEqual(config.get_model_replan_temperature(1, "source conflict"), 0.5)
        self.assertEqual(config.get_model_replan_temperature(1, "protocol error"), 0.1)
        self.assertEqual(config.get_model_replan_temperature(1, "duplicate frozen path"), 0.8)
        self.assertEqual(config.get_model_replan_temperature(2, "replan stalled"), 0.9)

    def test_task_plan_repairs_invented_year_without_controller_rewrite(self):
        planner = Planner()
        outputs = [
            {
                "schema_version": "task_plan.v1",
                "goal": "answer current release",
                "task_mode": "lookup",
                "source_policy": "open_web",
                "required_domains": [],
                "requested_fields": ["version"],
                "max_items": 0,
                "atomic_points": [
                    {
                        "id": "P1",
                        "task": "find the 2024 release",
                        "objective": "identify the 2024 release",
                        "evidence_needed": ["release version"],
                        "acceptance_criteria": ["version is stated"],
                    }
                ],
                "completion_rule": "answer the question",
            },
            {
                "schema_version": "task_plan.v1",
                "goal": "answer current release",
                "task_mode": "lookup",
                "source_policy": "open_web",
                "required_domains": [],
                "requested_fields": ["version"],
                "max_items": 0,
                "atomic_points": [
                    {
                        "id": "P1",
                        "task": "find the current release",
                        "objective": "identify the current release",
                        "evidence_needed": ["release version"],
                        "acceptance_criteria": ["version is stated"],
                    }
                ],
                "completion_rule": "answer the question",
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

        self.assertEqual(result["atomic_points"][0]["task"], "find the current release")
        self.assertEqual(result["plan_attempts"], 2)
        self.assertEqual([row["temperature"] for row in seen], [0.1, 0.1])
        self.assertEqual(
            [row["stage"] for row in seen],
            ["task_plan", "task_plan_repair"],
        )
        self.assertEqual(seen[1]["sampling"]["top_k"], 40)
        self.assertEqual(seen[1]["sampling"]["top_p"], 0.3)
        self.assertEqual(seen[1]["sampling"]["presence_penalty"], 0.00001)
        self.assertEqual(seen[1]["sampling"]["frequency_penalty"], 0.00001)
        self.assertIn("unexpected_year:2024", seen[1]["prompt"])

    def test_task_plan_allows_one_same_temperature_repair_then_keeps_warning(self):
        planner = Planner()

        def plan(task: str) -> dict:
            return {
                "schema_version": "task_plan.v1",
                "goal": "answer current release",
                "task_mode": "lookup",
                "source_policy": "open_web",
                "required_domains": [],
                "requested_fields": ["version"],
                "max_items": 0,
                "atomic_points": [
                    {
                        "id": "P1",
                        "task": task,
                        "objective": task,
                        "evidence_needed": ["release version"],
                        "acceptance_criteria": ["version is stated"],
                    }
                ],
                "completion_rule": "answer the question",
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

        self.assertEqual(result["atomic_points"][0]["task"], "find the 2025 release")
        self.assertEqual(result["plan_attempts"], 2)
        self.assertEqual(result["semantic_warnings"], ["unexpected_year:2025"])
        self.assertEqual(
            seen,
            [
                ("task_plan", 0.1),
                ("task_plan_repair", 0.1),
            ],
        )

    def test_cross_validator_accepts_rwkv_explicit_point_status_schema(self):
        planner = Planner()

        class StatusSchemaRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del prompt, max_tokens, stop
                return Mock(
                    content=(
                        '{"schema_version":"cross-validation.v1",'
                        '"task_point_status":{'
                        '"P1":{"status":"completed"},'
                        '"P2":{"status":"not_retrieved"}},'
                        '"missing_points":["P2"],"conflicts":[],'
                        '"next_focus":"retrieve P2 evidence",'
                        '"reason":"P2 is missing"}'
                    )
                )

        planner.llm = StatusSchemaRWKV()
        review = planner.cross_validate_research(
            "mixed question",
            {"atomic_points": [{"id": "P1"}, {"id": "P2"}]},
            "P1 evidence only",
        )

        self.assertEqual(review["decision"], "replan")
        self.assertEqual(review["decision_source"], "rwkv_explicit_gap_fields")
        self.assertEqual(review["missing_points"], ["P2"])
        self.assertEqual(review["task_point_status"]["P1"]["status"], "completed")
        self.assertEqual(review["review_attempts"], 1)

    def test_cross_validator_canonicalizes_explicit_verbose_missing_point_id(self):
        planner = Planner()

        class VerboseMissingPointRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del prompt, max_tokens, stop
                return Mock(
                    content=(
                        '{"schema_version":"rwkv-cross-validation.v1",'
                        '"decision":"replan",'
                        '"missing_points":["P2: current release date is missing"],'
                        '"conflicts":[],"task_point_status":{'
                        '"P1":{"status":"supported","evidence_refs":["S1"]},'
                        '"P2":{"status":"missing","evidence_refs":[]}},'
                        '"next_focus":"current release date",'
                        '"reason":"P2 lacks a source"}'
                    )
                )

        planner.llm = VerboseMissingPointRWKV()
        review = planner.cross_validate_research(
            "mixed question",
            {"atomic_points": [{"id": "P1"}, {"id": "P2"}]},
            "P1 evidence only",
            evidence_refs=["S1"],
        )

        self.assertEqual(review["decision"], "replan")
        self.assertEqual(review["missing_points"], ["P2"])

    def test_cross_validator_requires_answer_ready_fact_with_verbatim_source_quote(self):
        planner = Planner()
        calls = []
        outputs = [
            {
                "schema_version": "rwkv-cross-validation.v1",
                "decision": "finish",
                "missing_points": [],
                "conflicts": [],
                "task_point_status": {
                    "P1": {
                        "status": "supported",
                        "evidence_refs": ["S1"],
                        "supported_facts": [
                            {
                                "field": "current version theme",
                                "value": "Long Goodbye",
                                "evidence_ref": "S1",
                                "quote": "invented quote not in the source",
                            }
                        ],
                        "missing_fields": [],
                    }
                },
                "next_focus": "",
                "reason": "the source states the value",
            },
            {
                "schema_version": "rwkv-cross-validation.v1",
                "decision": "finish",
                "missing_points": [],
                "conflicts": [],
                "task_point_status": {
                    "P1": {
                        "status": "supported",
                        "evidence_refs": ["S1"],
                        "supported_facts": [
                            {
                                "field": "current version theme",
                                "value": "Long Goodbye",
                                "evidence_ref": "S1",
                                "quote": "Version 3.1 theme: Long Goodbye",
                            }
                        ],
                        "missing_fields": [],
                    }
                },
                "next_focus": "",
                "reason": "the cited source states the exact value",
            },
        ]

        class QuoteCorrectingRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del max_tokens, stop
                calls.append(prompt)
                return Mock(content=json.dumps(outputs[len(calls) - 1]))

        planner.llm = QuoteCorrectingRWKV()
        review = planner.cross_validate_research(
            "What is the current version theme?",
            {"atomic_points": [{"id": "P1", "task": "current version theme"}]},
            "[S1] Official release\nVersion 3.1 theme: Long Goodbye",
            evidence_refs=["S1"],
            evidence_text_by_ref={
                "S1": "Official release\nVersion 3.1 theme: Long Goodbye"
            },
        )

        self.assertEqual(review["decision"], "finish")
        self.assertEqual(review["review_attempts"], 2)
        self.assertEqual(
            review["task_point_status"]["P1"]["supported_facts"][0]["value"],
            "Long Goodbye",
        )
        self.assertIn("not present in cited evidence ref S1", calls[1])
        self.assertIn("supported_facts", calls[0])

    def test_cross_validator_accepts_rwkv_explicit_progress_state_schema(self):
        planner = Planner()

        class ProgressSchemaRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del prompt, max_tokens, stop
                return Mock(
                    content=(
                        '{"schema_version":"rwkv-cross-validation.v1",'
                        '"atomic_points":['
                        '{"id":"P1","status":"pending"},'
                        '{"id":"P2","status":"pending"},'
                        '{"id":"P3","status":"pending"}],'
                        '"progress":{"completed":0,"total":3,"percent":0},'
                        '"retrieval_state":{"retrieved":['
                        '{"point_id":"P1","source_count":3},'
                        '{"point_id":"P2","source_count":0},'
                        '{"point_id":"P3","source_count":0}]},'
                        '"next_focus":"obtain the exact P1 date",'
                        '"reason":"research is unfinished","final_answer":null}'
                    )
                )

        planner.llm = ProgressSchemaRWKV()
        review = planner.cross_validate_research(
            "mixed historical and current question",
            {
                "atomic_points": [
                    {"id": "P1"},
                    {"id": "P2"},
                    {"id": "P3"},
                ]
            },
            "P1 source spans only",
        )

        self.assertEqual(review["decision"], "replan")
        self.assertEqual(
            review["decision_source"], "rwkv_explicit_progress_fields"
        )
        self.assertEqual(review["missing_points"], ["P1", "P2", "P3"])
        self.assertEqual(review["task_point_status"]["P2"]["status"], "pending")
        self.assertEqual(review["review_attempts"], 1)

    def test_cross_validator_accepts_rwkv_missing_facts_status(self):
        planner = Planner()

        class MissingFactsRWKV:
            provider = "local_13b"

            def text_completion(self, prompt, max_tokens=0, stop=None):
                del prompt, max_tokens, stop
                return Mock(
                    content=(
                        '"schema_version":"rwkv-cross-validation.v1",'
                        '"decision":"replan",'
                        '"missing_points":["P1"],"conflicts":[],'
                        '"task_point_status":{'
                        '"P1":{"status":"missing_facts","evidence_refs":[]}},'
                        '"next_focus":"find a source span",'
                        '"reason":"the current source does not bind the requested fact"}'
                    )
                )

        planner.llm = MissingFactsRWKV()
        review = planner.cross_validate_research(
            "question",
            {"atomic_points": [{"id": "P1"}]},
            "No usable source spans",
            evidence_refs=["S1"],
        )

        self.assertEqual(review["decision"], "replan")
        self.assertEqual(review["task_point_status"]["P1"]["status"], "missing_facts")
        self.assertEqual(review["review_attempts"], 1)

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
