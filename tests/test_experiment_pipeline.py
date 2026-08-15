from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from utils.citation_validator import validate_citations
from utils.evaluation_dataset import (
    DOMAINS,
    PERSONAS,
    TASK_TYPES,
    append_dataset,
    generate_cases,
    load_dataset,
    save_dataset,
    update_reference,
    update_reference_from_trace,
)
from utils.experiment_manifest import finalize_manifest, reconstruct_run
from utils.experiment_metrics import compare_scores, score_trace
from utils.human_review import aggregate_reviews, build_blind_packet, create_review, paired_blind_summary
from utils.model_judge import build_judge_prompt, parse_judge_output
from utils.operational_metrics import collect_operational_metrics, prometheus_text
from scripts.preflight import run_preflight
from scripts.run_json_acceptance import (
    _aggregate_trace_summaries,
    _atomic_write_json,
    _module_responsibility_metrics,
    _trace_summary,
)
from utils.retry import retry_with_fallback
from utils.runtime_gate import analysis_slot
from utils.time_budget import TaskTimeoutError, task_time_budget
from utils.model_events import record_model_event
from utils.risk_policy import validate_risk_answer
from utils.reference_validation import validate_dataset_references, validate_reference_case
from agent.retrieval_synthesis import build_evidence_context, synthesize_retrieval_answer
from agent.planner import Planner
from utils.task_events import append_task_event
from utils.token_tracker import current_task_id
from utils.trace_validation import validate_replay_trace
from config import LLM_ENDPOINTS, get_experiment_model_config, validate_experiment_model_contract
from agent.orchestrator import Orchestrator
from agent.unified_research import EvidenceReviewDecisionError
from tools.registry import ToolRegistry


class ExperimentPipelineTests(unittest.TestCase):
    def test_atomic_checkpoint_failure_preserves_previous_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "part.result.json"
            _atomic_write_json(output, {"revision": 1, "cases": ["stable"]})

            with (
                patch(
                    "scripts.run_json_acceptance.json.dump",
                    side_effect=RuntimeError("simulated serialization failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated serialization failure"),
            ):
                _atomic_write_json(output, {"revision": 2, "cases": ["partial"]})

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"revision": 1, "cases": ["stable"]},
            )
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_atomic_checkpoint_is_always_parseable_during_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "part.result.json"
            _atomic_write_json(output, {"revision": 0, "payload": "x" * 100_000})
            stopped = threading.Event()
            read_errors: list[BaseException] = []

            def read_repeatedly() -> None:
                while not stopped.is_set():
                    try:
                        value = json.loads(output.read_text(encoding="utf-8"))
                        if not isinstance(value.get("revision"), int):
                            raise AssertionError("checkpoint revision is missing")
                    except BaseException as exc:  # captured for the main test thread
                        read_errors.append(exc)
                        stopped.set()

            reader = threading.Thread(target=read_repeatedly)
            reader.start()
            try:
                for revision in range(1, 21):
                    _atomic_write_json(
                        output,
                        {"revision": revision, "payload": str(revision) * 100_000},
                    )
            finally:
                stopped.set()
                reader.join(timeout=2)

            self.assertEqual(read_errors, [])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["revision"], 20)

    def test_rwkv_replan_is_observed_but_not_a_module_responsibility_violation(self):
        metrics = _module_responsibility_metrics(
            [
                {"type": "planner_session_rebuilt", "reason": "duplicate_path"},
                {"type": "model_call", "request_stage": "planner_replan"},
                {"type": "synthesis", "content": "RWKV answer"},
                {"type": "final", "content": "RWKV answer"},
            ]
        )

        self.assertTrue(metrics["pass"])
        self.assertEqual(metrics["violation_count"], 0)
        self.assertEqual(metrics["rwkv_semantic_control_count"], 2)
        self.assertEqual(metrics["prohibited_output_intervention_count"], 0)

    def test_answer_mutation_and_output_mismatch_are_responsibility_violations(self):
        metrics = _module_responsibility_metrics(
            [
                {"type": "answer_rewrite"},
                {"type": "tool_call", "controller_override": True},
                {"type": "synthesis", "content": "RWKV answer"},
                {"type": "final", "content": "rewritten answer"},
            ]
        )
        aggregate = _aggregate_trace_summaries(
            [
                {
                    "delivery": "answer",
                    "runtime_error": "",
                    "trace": {"stats": {"module_responsibility": metrics}},
                }
            ]
        )

        self.assertFalse(metrics["pass"])
        self.assertEqual(metrics["violation_count"], 3)
        self.assertEqual(metrics["prohibited_output_intervention_count"], 1)
        self.assertEqual(metrics["controller_override_count"], 1)
        self.assertEqual(metrics["final_output_mismatch_count"], 1)
        self.assertEqual(aggregate["module_responsibility"]["violation_count"], 3)
        self.assertEqual(
            aggregate["module_responsibility"]["prohibited_output_intervention_count"],
            1,
        )

    def test_production_live_100_uses_natural_agent_visible_queries(self):
        dataset_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "evaluation"
            / "retrieval_production_live_100_v1_20260812.json"
        )
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        cases = payload["cases"]

        self.assertEqual(payload["suite"], "retrieval-production-live-100-v1")
        self.assertEqual(len(cases), 100)
        self.assertEqual(len({row["case_id"] for row in cases}), 100)
        benchmark_only_phrases = (
            "截至测试当天",
            "截至查询时刻",
            "当前测试",
            "本次测试",
            "在测试当天",
        )
        forbidden_agent_hints = {
            "answer",
            "final_answer",
            "reference_answer",
            "task_graph",
            "task_plan",
            "tool_route",
            "source_url",
            "replan_path",
            "benchmark_timestamp",
        }
        for row in cases:
            query = str(row.get("query") or "").strip()
            self.assertTrue(query, row.get("case_id"))
            self.assertFalse(
                any(phrase in query for phrase in benchmark_only_phrases),
                (row.get("case_id"), query),
            )
            self.assertFalse(forbidden_agent_hints.intersection(row), row)

    def test_task_plan_preserves_record_count_and_factual_fields_without_gates(self):
        plan = Planner._validate_task_plan(
            {
                "contract": "rwkv.ecra.runtime.task-plan",
                "goal": "verify one fact",
                "records": [
                    {
                        "id": "P1",
                        "task": "Find the fact",
                        "objective": "Verify the fact",
                        "evidence_needed": ["official page"],
                        "acceptance_criteria": ["date is present"],
                    },
                    {
                        "id": "P2",
                        "task": "  find the fact ",
                        "objective": "verify   the fact",
                        "evidence_needed": ["archived page", "official page"],
                        "acceptance_criteria": ["source URL is present"],
                    },
                ],
            }
        )
        self.assertEqual(plan["contract"], "rwkv.ecra.runtime.task-plan")
        self.assertEqual(len(plan["records"]), 2)
        first, second = plan["records"]
        self.assertEqual(first["record_id"], "P1")
        self.assertEqual(second["record_id"], "P2")
        self.assertEqual(first["question"], "Find the fact — Verify the fact")
        self.assertEqual(second["question"], "find the fact — verify the fact")
        self.assertEqual(
            first["fields"],
            [{"field_id": "P1:F1", "name": "official page"}],
        )
        self.assertEqual(
            second["fields"],
            [
                {"field_id": "P2:F1", "name": "archived page"},
                {"field_id": "P2:F2", "name": "official page"},
            ],
        )
        for record in plan["records"]:
            self.assertNotIn("evidence_needed", record)
            self.assertNotIn("acceptance_criteria", record)

    def test_routing_observation_keeps_candidate_urls_before_evidence_rows(self):
        observation = Planner._compact_observation(
            {
                "status": "ok",
                "retrieval_role": "discovery",
                "candidate_urls": [
                    {
                        "candidate_rank": 1,
                        "title": "Official documentation",
                        "url": "https://example.org/docs/relevant-page",
                        "source": "search",
                        "candidate_score": 9.5,
                    }
                ],
                "results": [
                    {
                        "title": "Large page",
                        "url": "https://example.org/",
                        "snippet": "x" * 260,
                        "chunk_candidates": [{"facts": ["y" * 400], "quote": "z" * 400}],
                    }
                ],
            }
        )
        self.assertIn("candidate_urls", observation)
        self.assertIn("relevant-page", observation)
        self.assertLess(observation.index("candidate_urls"), 100)

    def test_evidence_context_packs_each_source_inside_configured_budget(self):
        results = []
        for index in range(1, 4):
            body = (f"Source {index} directly states the requested fact. " * 260).strip()
            results.append(
                {
                    "title": f"Source {index}",
                    "url": f"https://example.org/source-{index}",
                    "page_excerpt": body,
                    "content": body,
                    "source_excerpt": body,
                    "body_verified": True,
                    "evidence_origin": "fetched_page_body",
                    "evidence_boundary": "page_body_only",
                    "source_chunks": [
                        {
                            "chunk_id": f"source-{index}-chunk-1",
                            "index": 0,
                            "text": body,
                            "token_count": len(body.split()),
                        }
                    ],
                    "chunk_count": 1,
                }
            )
        context = build_evidence_context(
            {"query": "requested fact", "results": results},
            constraints={"strategy_config": {"context_source_count": 3}},
            query="requested fact",
        )
        self.assertEqual(context["selected_evidence"], [])
        self.assertLessEqual(context["context_tokens"], 10000)
        self.assertNotIn("UNBOUND SOURCE EXCERPT", context["text"])
        self.assertEqual(context["usable_evidence_count"], 0)
        self.assertEqual(
            sum(item["chunk_count"] for item in context["selected_evidence"]),
            context["chunk_count"],
        )

    def test_context_builder_does_not_override_upstream_source_ordering(self):
        generic = "nginx homepage navigation and unrelated release notes. " * 700
        relevant = (
            "nginx WebSocket reverse proxy configuration uses proxy_http_version 1.1, "
            "Upgrade, and Connection headers. "
        ) * 30
        context = build_evidence_context(
            {
                "query": "nginx WebSocket reverse proxy configuration",
                "results": [
                    {
                        "title": "NGINX home",
                        "url": "https://nginx.org/en/",
                        "content": generic,
                        "page_excerpt": generic,
                        "source_excerpt": generic,
                        "body_verified": True,
                        "evidence_origin": "fetched_page_body",
                    },
                    {
                        "title": "WebSocket proxying",
                        "url": "https://nginx.org/en/docs/http/websocket.html",
                        "content": relevant,
                        "page_excerpt": relevant,
                        "source_excerpt": relevant,
                        "body_verified": True,
                        "evidence_origin": "fetched_page_body",
                    },
                ],
            },
            constraints={"strategy_config": {"context_source_count": 1}},
        )
        self.assertEqual(context["selected_evidence"], [])

    def test_duplicate_source_rows_do_not_consume_context_slots(self):
        body = "The primary source states the release date is 2026-07-30. " * 12
        results = [
            {
                "title": "Primary source",
                "url": "https://example.org/fact/#section",
                "content": body,
                "page_excerpt": body,
                "source_excerpt": body,
                "body_verified": True,
                "evidence_origin": "fetched_page_body",
            },
            {
                "title": "The same source from another provider",
                "url": "HTTPS://EXAMPLE.ORG/fact/",
                "content": body,
                "page_excerpt": body,
                "source_excerpt": body,
                "body_verified": True,
                "evidence_origin": "fetched_page_body",
            },
            {
                "title": "Independent source",
                "url": "https://example.net/confirmation",
                "content": body,
                "page_excerpt": body,
                "source_excerpt": body,
                "body_verified": True,
                "evidence_origin": "fetched_page_body",
            },
        ]
        context = build_evidence_context(
            {"query": "release date", "results": results},
            constraints={"strategy_config": {"context_source_count": 3}},
            query="release date",
        )
        self.assertEqual(context["duplicate_source_count"], 0)
        self.assertEqual(context["selected_evidence"], [])

    def test_configured_source_count_is_only_a_resource_cap(self):
        relevant = "The WebSocket reverse proxy uses the Upgrade header. " * 8
        unrelated = "This page is a generic download index with release archives. " * 30
        context = build_evidence_context(
            {
                "query": "WebSocket reverse proxy Upgrade header",
                "results": [
                    {
                        "title": "Relevant documentation",
                        "url": "https://example.org/websocket",
                        "content": relevant,
                        "page_excerpt": relevant,
                        "source_excerpt": relevant,
                        "evidence_origin": "fetched_page_body",
                    },
                    {
                        "title": "Unrelated downloads",
                        "url": "https://example.org/downloads",
                        "content": unrelated,
                        "page_excerpt": unrelated,
                        "source_excerpt": unrelated,
                        "evidence_origin": "fetched_page_body",
                    },
                ],
            },
            constraints={"strategy_config": {"context_source_count": 2}},
            query="WebSocket reverse proxy Upgrade header",
        )
        self.assertEqual(context["selected_evidence"], [])
        self.assertNotIn("unbound_fallback_source_limit", context["context_stats"])

    def test_experiment_model_contract_is_the_active_rwkv_13b_profile(self):
        profile = get_experiment_model_config()
        self.assertEqual(profile["model"], "rwkv7-g1i-13.3b-20260805-ctx16384")
        self.assertEqual(profile["endpoint"], LLM_ENDPOINTS["local_13b"]["base_url"])
        self.assertEqual(profile["context_length"], 16384)
        self.assertNotIn("api_key", profile)
        self.assertNotIn("api_key", validate_experiment_model_contract())

    def test_historical_rwkv_7b_contract_remains_explicitly_validatable(self):
        profile = get_experiment_model_config("local_7b")
        self.assertEqual(profile["model"], "rwkv7-g1h-7.2b-20260710-ctx10240")
        self.assertEqual(validate_experiment_model_contract("local_7b")["provider"], "local_7b")

    def test_task_events_create_replayable_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                append_task_event("TRACE_1", "user_input", content="What is the answer?")
                append_task_event(
                    "TRACE_1",
                    "tool_result",
                    action="search_web_wigolo",
                    query="answer",
                    result=json.dumps(
                        {
                            "results": [
                                {
                                    "title": "Primary source",
                                    "url": "https://example.com/source",
                                    "page_excerpt": "The answer is 42.",
                                }
                            ],
                            "sources": ["https://example.com/source"],
                            "citation_refs": [
                                {
                                    "ref_id": "S1",
                                    "url": "https://example.com/source",
                                    "content": "The answer is 42.",
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                )
                append_task_event(
                    "TRACE_1",
                    "synthesis",
                    content="The answer is 42 [S1].",
                    prompt="Answer with evidence.",
                    model_output="The answer is 42 [S1].",
                    citation_refs=[{"ref_id": "S1", "url": "https://example.com/source", "content": "The answer is 42."}],
                )
                append_task_event(
                    "TRACE_1",
                    "final",
                    status="completed",
                    content="The answer is 42 [S1].",
                    round_count=2,
                )
                finalize_manifest("TRACE_1", status="completed")

            trace = reconstruct_run("TRACE_1", output)
            self.assertEqual(trace["query"], "What is the answer?")
            self.assertEqual(trace["search_queries"], [])
            self.assertEqual(trace["search_results"][0]["title"], "Primary source")
            self.assertEqual(trace["model_outputs"][0]["prompt"], "Answer with evidence.")
            self.assertIn("42", trace["final_answer"])
            self.assertEqual(trace["manifest"]["status"], "completed")
            self.assertEqual(trace["manifest"]["summary"]["round_count"], 2)
            self.assertIn("pipeline", trace["manifest"]["config"])
            self.assertNotIn("password", trace["manifest"]["config"]["runtime"]["slm"])
            self.assertTrue(trace["manifest"].get("workspace_hash"))

    def test_replay_trace_validator_checks_structure_and_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                append_task_event("VALID_TRACE", "user_input", content="question")
                append_task_event("VALID_TRACE", "final", status="completed", content="answer")
            trace = reconstruct_run("VALID_TRACE", output)
            result = validate_replay_trace(trace)
            self.assertTrue(result["valid"])
            self.assertNotIn("possible_secret_leak", result["issues"])

    def test_dynamic_dataset_is_schema_complete_and_versioned(self):
        rows = generate_cases(12, seed=11)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["dataset_version"] for row in rows}), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            version = save_dataset(rows, path)
            loaded = load_dataset(path)
            self.assertEqual(version, loaded[0]["dataset_version"])
            self.assertTrue({row["domain"] for row in loaded})
            self.assertTrue(all(row["acceptance_criteria"] and row["risk_checks"] for row in loaded))
            revised = update_reference(
                path,
                loaded[0]["question_id"],
                reference_answer="Reviewed answer",
                reference_citations=[{"url": "https://example.com/reference"}],
            )
            reviewed = load_dataset(path)[0]
            self.assertEqual(reviewed["reference_answer"], "Reviewed answer")
            self.assertEqual(reviewed["dataset_version"], revised)

    def test_default_dynamic_matrix_covers_domains_personas_and_task_types(self):
        rows = generate_cases(60, seed=20260726)
        self.assertTrue(set(DOMAINS).issubset({row["domain"] for row in rows}))
        self.assertTrue(set(PERSONAS).issubset({row["persona"] for row in rows}))
        self.assertTrue(set(TASK_TYPES).issubset({row["task_type"] for row in rows}))
        self.assertEqual({row["difficulty"] for row in rows}, {"L1", "L2", "L3", "L4", "L5"})

    def test_dynamic_dataset_can_expand_without_overwriting_existing_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            original = generate_cases(1, seed=31)
            save_dataset(original, path)
            update_reference(
                path,
                original[0]["question_id"],
                reference_answer="Reviewed",
                reference_citations=[{"url": "https://example.com/source", "evidence_text": "Fact"}],
                key_facts=["Fact"],
                reviewer_id="human-1",
            )
            append_dataset(path, 2, seed=32)
            loaded = load_dataset(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0]["reference_answer"], "Reviewed")
            self.assertEqual(loaded[0]["key_facts"], ["Fact"])
            self.assertEqual(len({row["question_id"] for row in loaded}), 3)

    def test_reference_promotion_copies_trace_but_keeps_answer_human_supplied(self):
        rows = generate_cases(1, seed=19)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            save_dataset(rows, path)
            trace = {
                "search_queries": ["official source"],
                "search_results": [{"url": "https://example.com/source", "page_excerpt": "Fact."}],
                "navigation_trace": [{"type": "page_fetch", "url": "https://example.com/source"}],
                "sources": [{"url": "https://example.com/source"}],
                "evidence": [{"url": "https://example.com/source", "content": "Fact."}],
                "final_answer": "model answer that is not automatically accepted",
            }
            update_reference_from_trace(
                path,
                rows[0]["question_id"],
                trace,
                reference_answer="Human reviewed answer",
                reference_citations=[{"url": "https://example.com/source"}],
            )
            reviewed = load_dataset(path)[0]
            self.assertEqual(reviewed["reference_answer"], "Human reviewed answer")
            self.assertNotEqual(reviewed["reference_answer"], trace["final_answer"])
            self.assertEqual(reviewed["search_queries"], ["official source"])
            self.assertEqual(reviewed["reference_citations"][0]["evidence_text"], "Fact.")

    def test_reference_gate_requires_human_review_evidence_and_freshness(self):
        rows = generate_cases(1, seed=23)
        pending = validate_dataset_references(rows)
        self.assertFalse(pending["all_ready"])
        self.assertEqual(pending["pending_count"], 1)

        reviewed = dict(rows[0])
        reviewed.update(
            {
                "reference_answer": "Reviewed answer",
                "reference_citations": [
                    {"url": "https://example.com/source", "evidence_text": "Supporting fact."}
                ],
                "reference_metadata": {
                    "status": "human_reviewed",
                    "reviewer_id": "human-1",
                    "reviewed_at": "2026-07-01T00:00:00+00:00",
                    "source_checked_at": "2026-07-01T00:00:00+00:00",
                    "stale_after_days": 30,
                },
            }
        )
        ready = validate_reference_case(reviewed, now=datetime(2026, 7, 10, tzinfo=timezone.utc))
        self.assertEqual(ready["status"], "ready")

        time_sensitive = {**reviewed, "task_type": "time_sensitive"}
        stale = validate_reference_case(
            time_sensitive,
            now=datetime(2026, 8, 15, tzinfo=timezone.utc),
        )
        self.assertEqual(stale["status"], "stale")
        self.assertIn("stale_time_sensitive_reference", stale["issues"])

    def test_human_review_is_scored_with_a_blind_validated_rubric(self):
        scores = {key: 4 for key in ("fact_correctness", "evidence_support", "citation_accuracy", "completeness", "instruction_adherence", "risk_handling", "usability")}
        review_a = create_review(question_id="Q-REVIEW", reviewer_id="reviewer-1", blind_label="A", scores=scores, variant="baseline")
        review_b = create_review(question_id="Q-REVIEW", reviewer_id="reviewer-1", blind_label="B", scores={**scores, "usability": 5}, variant="candidate")
        aggregate = aggregate_reviews([review_a, review_b])
        paired = paired_blind_summary([review_a, review_b])
        self.assertEqual(aggregate["groups"]["baseline"]["review_count"], 1)
        self.assertEqual(paired["b_wins"], 1)

    def test_blind_packet_keeps_variant_key_out_of_reviewer_payload(self):
        baseline = {
            "experiment_id": "exp-a",
            "results": [{
                "case": {"question_id": "Q-BLIND", "question": "Which answer is supported?"},
                "trace": {"final_answer": "A answer", "citations": [{"url": "https://example.com/a"}], "manifest": {"status": "completed"}},
            }],
        }
        candidate = {
            "experiment_id": "exp-b",
            "results": [{
                "case": {"question_id": "Q-BLIND", "question": "Which answer is supported?"},
                "trace": {"final_answer": "B answer", "citations": [{"url": "https://example.com/b"}], "manifest": {"status": "completed"}},
            }],
        }
        packet, key = build_blind_packet(baseline, candidate, seed=3)
        self.assertEqual(packet["sample_count"], 1)
        self.assertEqual(set(packet["cases"][0]) , {"question_id", "question", "A", "B"})
        self.assertNotIn("baseline", json.dumps(packet["cases"], ensure_ascii=False).casefold())
        self.assertEqual(set(key["cases"][0]), {"question_id", "A", "B"})

    def test_model_judge_parser_requires_all_scores_and_reason(self):
        valid = {
            "winner": "A",
            "scores": {label: {rubric: 4 for rubric in ("fact_correctness", "evidence_support", "citation_accuracy", "completeness", "instruction_adherence", "risk_handling", "usability")} for label in ("A", "B")},
            "reason": "A cites the supporting source more directly.",
        }
        parsed = parse_judge_output(json.dumps(valid))
        self.assertTrue(parsed["valid"])
        self.assertIn("Question:", build_judge_prompt("Question", {"answer": "A"}, {"answer": "B"}))
        self.assertFalse(parse_judge_output('{"winner":"A"}') ["valid"])

    def test_operational_metrics_are_secret_free_and_trace_derived(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                append_task_event("METRICS_TRACE", "user_input", content="question")
                append_task_event("METRICS_TRACE", "final", status="completed", content="answer")
                finalize_manifest("METRICS_TRACE", status="completed")
                metrics = collect_operational_metrics(
                    [{"task_id": "METRICS_TRACE"}],
                    output_directory=output,
                )
            self.assertEqual(metrics["tasks"]["total"], 1)
            self.assertEqual(metrics["tasks"]["by_status"]["completed"], 1)
            self.assertEqual(metrics["trace_integrity"]["invalid_count"], 0)
            rendered = prometheus_text(metrics)
            self.assertIn("rwkv_ecra_tasks_total", rendered)
            self.assertNotIn("question", rendered)

    def test_preflight_redacts_model_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.dict("config.DATA_PIPELINE", {
                    "input_directory": str(root / "input"),
                    "output_directory": str(root / "output"),
                    "checkpoint_directory": str(root / "checkpoints"),
                }, clear=False),
                patch("scripts.preflight.probe_model_service", return_value={"available": True, "model_match": True}),
                patch("scripts.preflight.config.get_llm_api_key", return_value="test-local-key"),
            ):
                result = run_preflight()
            serialized = json.dumps(result, ensure_ascii=False)
            self.assertNotIn("test-local-key", serialized)
            self.assertNotIn('"api_key"', serialized)
            self.assertTrue(result["contract"]["model"]["api_key_configured"])

    def test_stage_metrics_compare_without_inventing_missing_references(self):
        case = {
            "question_id": "Q1",
            "domain": "software_development",
            "persona": "software_engineer",
            "task_type": "single_fact",
            "difficulty": "L1",
            "dataset_version": "eval-test",
            "sources": [{"url": "https://example.com/source"}],
            "reference_answer": "The answer is 42.",
            "reference_citations": [{"url": "https://example.com/source"}],
            "key_facts": ["The answer is 42."],
        }
        trace = {
            "manifest": {"run_id": "RUN1", "status": "completed", "experiment": {}},
            "events": [{"type": "tool_call"}, {"type": "tool_result"}],
            "search_queries": ["answer"],
            "search_results": [{"title": "Primary", "url": "https://example.com/source", "content": "The answer is 42."}],
            "evidence": [{"content": "The answer is 42."}],
            "citations": [{"ref_id": "S1", "url": "https://example.com/source", "content": "The answer is 42."}],
            "final_answer": "The answer is 42 [S1].",
        }
        score = score_trace(trace, case)
        self.assertEqual(score["retrieval"]["recall_at_5"], 1.0)
        self.assertEqual(score["evidence"]["key_fact_recall"], 100.0)
        self.assertEqual(score["answer"]["fact_accuracy"], 100.0)
        self.assertEqual(score["answer"]["evidence_support_rate"], 100.0)
        self.assertEqual(score["answer"]["reference_citation_recall"], 100.0)
        self.assertEqual(score["answer"]["citation_completeness"], 100.0)
        self.assertEqual(score["answer"]["citation_locator_coverage"], 100.0)
        comparison = compare_scores([score], [score])
        self.assertEqual(comparison["groups"]["all"]["metrics"]["retrieval.recall_at_5"]["absolute_change"], 0.0)
        self.assertEqual(comparison["paired_statistics"]["retrieval.recall_at_5"]["n"], 1)
        self.assertIsNone(comparison["paired_statistics"]["retrieval.recall_at_5"]["ci95"])

    def test_stage_timing_reads_nested_round_duration(self):
        trace = {
            "manifest": {"run_id": "RUN-TIMING", "status": "completed", "experiment": {}},
            "events": [
                {"type": "candidate_merge", "data": {"duration_ms": 12.5}},
                {"type": "page_extract", "duration_ms": 3.0},
                {"type": "synthesis", "duration_ms": 8.0},
            ],
            "search_results": [],
            "evidence": [],
            "citations": [],
            "final_answer": "",
        }
        timings = score_trace(trace)["timings"]
        self.assertEqual(timings["retrieval_ms"], 12.5)
        self.assertEqual(timings["content_extraction_ms"], 3.0)
        self.assertEqual(timings["answer_generation_ms"], 8.0)

    def test_citation_validator_rejects_search_result_pages(self):
        result = validate_citations(
            [{"ref_id": "S1", "url": "https://www.google.com/search?q=rwkv", "content": "result"}],
            answer="[S1]",
        )
        self.assertEqual(result["invalid"], 1)
        self.assertIn("search_result_page", result["rows"][0]["issues"])

    def test_citation_validator_joins_provider_ref_to_captured_result(self):
        result = validate_citations(
            [{"ref_id": "S1", "url": "https://example.com/source"}],
            answer="The answer is 42 [S1].",
            evidence=[
                {
                    "url": "https://example.com/source/",
                    "page_excerpt": "The answer is 42.",
                }
            ],
        )
        row = result["rows"][0]
        self.assertEqual(row["evidence_source"], "retrieved_result")
        self.assertEqual(result["supported"], 1)
        self.assertEqual(result["invalid"], 0)

    def test_retry_events_are_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            token = current_task_id.set("RETRY_1")
            calls = {"count": 0}

            @retry_with_fallback(max_retries=2, delay=0, backoff=1)
            def flaky():
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RuntimeError("temporary")
                return "ok"

            try:
                with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                    self.assertEqual(flaky(), "ok")
                    events = reconstruct_run("RETRY_1", output)["events"]
            finally:
                current_task_id.reset(token)
            self.assertEqual([event["type"] for event in events], ["retry"])

    def test_retry_policy_does_not_repeat_authentication_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            token = current_task_id.set("AUTH_1")
            calls = {"count": 0}

            @retry_with_fallback(max_retries=3, delay=0, backoff=1)
            def unauthorized():
                calls["count"] += 1
                raise RuntimeError("401 unauthorized")

            try:
                with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                    with self.assertRaises(RuntimeError):
                        unauthorized()
                    events = reconstruct_run("AUTH_1", output)["events"]
            finally:
                current_task_id.reset(token)
            self.assertEqual(calls["count"], 1)
            self.assertEqual([event["type"] for event in events], ["error"])

    def test_retry_policy_does_not_repeat_exhausted_quota_failures(self):
        calls = {"count": 0}

        @retry_with_fallback(max_retries=3, delay=0, backoff=1)
        def exhausted():
            calls["count"] += 1
            raise RuntimeError("HTTP 432: request exceeds the plan usage limit")

        with self.assertRaises(RuntimeError):
            exhausted()
        self.assertEqual(calls["count"], 1)

    def test_retry_policy_can_fail_fast_on_timeouts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            token = current_task_id.set("TIMEOUT_1")
            calls = {"count": 0}

            @retry_with_fallback(max_retries=3, delay=0, backoff=1, retry_timeout_errors=False)
            def timed_out():
                calls["count"] += 1
                raise TimeoutError("model read timeout")

            try:
                with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                    with self.assertRaises(TimeoutError):
                        timed_out()
                    events = reconstruct_run("TIMEOUT_1", output)["events"]
            finally:
                current_task_id.reset(token)
            self.assertEqual(calls["count"], 1)
            self.assertEqual([event["type"] for event in events], ["error"])

    def test_retry_policy_never_repeats_exhausted_task_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            token = current_task_id.set("TASK_BUDGET_1")
            calls = {"count": 0}

            @retry_with_fallback(max_retries=3, delay=0, backoff=1)
            def expired_task():
                calls["count"] += 1
                raise TaskTimeoutError("analysis task expired")

            try:
                with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                    with self.assertRaises(TaskTimeoutError):
                        expired_task()
                    events = reconstruct_run("TASK_BUDGET_1", output)["events"]
            finally:
                current_task_id.reset(token)
            self.assertEqual(calls["count"], 1)
            self.assertEqual([event["type"] for event in events], ["error"])

    def test_agentic_loop_bounds_duplicate_path_and_synthesizes(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task"
            task_dir.mkdir()
            orchestrator = Orchestrator()
            orchestrator.state.task_id = "REPEAT_STEP_TRACE"
            orchestrator.state.task_output_dir = str(task_dir)
            orchestrator.state.user_query = "find stations"
            orchestrator.state.run_metadata = {"max_tool_steps": 3}
            plan = {
                "contract": "rwkv.ecra.runtime.task-plan",
                "goal": "find stations",
                "records": [
                    {
                        "id": "P1",
                        "task": "find stations",
                        "objective": "find station list",
                        "evidence_needed": ["station list"],
                        "acceptance_criteria": ["list is directly supported"],
                        "output_format": "list",
                        "status": "pending",
                    }
                ],
                "completion_rule": "supported",
            }
            decisions = iter(
                [
                    {"action": "web_search", "args": {"query": "stations", "max_results": 2}},
                    {"action": "web_search", "args": {"query": "stations", "max_results": 2}},
                    {"action": "web_search", "args": {"query": "stations", "max_results": 2}},
                    {"action": "finish_task", "args": {}},
                ]
            )
            executed = []
            synthesis_calls = []

            def fake_execute(action, args, context, phase=None):
                executed.append((action, args, phase))
                if action == "web_search":
                    return json.dumps(
                        {
                            "status": "ok",
                            "retrieval_role": "discovery",
                            "results": [
                                {"title": "one", "url": "https://example.com/one"},
                                {"title": "two", "url": "https://example.com/two"},
                            ],
                        }
                    )
                return json.dumps(
                    {
                        "status": "error",
                        "retrieval_role": "evidence",
                        "message": "temporary page failure",
                        "results": [],
                    }
                )

            def fake_synthesis(*args, **kwargs):
                synthesis_calls.append(kwargs)
                return {
                    "content": "A bounded summary of the attempted retrieval.",
                    "mode": "test_final",
                    "evidence_count": 0,
                    "citation_refs": [],
                    "prompt": "final prompt",
                    "model_output": "A bounded summary of the attempted retrieval.",
                    "context_text": kwargs.get("execution_context", ""),
                    "selected_evidence": [],
                    "context_stats": {},
                }

            with (
                patch("agent.orchestrator.append_task_event"),
                patch("tools.registry.ToolRegistry.execute", side_effect=fake_execute),
                patch("agent.orchestrator.synthesize_retrieval_answer", side_effect=fake_synthesis),
            ):
                orchestrator.planner.create_task_plan = lambda *_args: plan
                orchestrator.planner.begin_task = lambda *_args: None
                orchestrator.planner.plan_next_action = lambda *_args: next(decisions)
                orchestrator.planner.observe_tool_result = lambda *_args: None
                orchestrator.planner.rebuild_session_after_review = (
                    lambda *_args, **_kwargs: None
                )
                orchestrator._review_evidence = lambda *_args, **_kwargs: {
                    "decision": "replan",
                    "missing_point_id": "P1",
                    "evidence_needed": "station list",
                }
                with self.assertRaises(EvidenceReviewDecisionError):
                    orchestrator._run_model_tool_loop("find stations", {})

            # The controller reports exact duplicates to RWKV and never
            # creates a replacement query of its own.
            self.assertEqual([item[0] for item in executed], ["web_search"])
            self.assertEqual(synthesis_calls, [])

    def test_runtime_gate_is_persisted_and_released(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                with analysis_slot("GATE_TRACE"):
                    pass
            trace = reconstruct_run("GATE_TRACE", output)
            statuses = [event.get("status") for event in trace["events"] if event["type"] == "runtime_gate"]
            self.assertEqual(statuses, ["waiting", "acquired", "released"])

    def test_runtime_gate_records_workspace_contention(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            entered = threading.Event()
            finished = threading.Event()
            with (
                patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False),
                patch.dict("config.EXPERIMENT_CONFIG", {"max_parallel_cases": 1}, clear=False),
            ):
                def waiting_task():
                    with analysis_slot("GATE_WAITING"):
                        entered.set()
                    finished.set()

                with analysis_slot("GATE_HOLDER"):
                    worker = threading.Thread(target=waiting_task)
                    worker.start()
                    time.sleep(0.15)
                    self.assertFalse(finished.is_set())
                worker.join(timeout=2)
                self.assertTrue(entered.is_set())
                self.assertTrue(finished.is_set())
            waiting_trace = reconstruct_run("GATE_WAITING", output)
            acquired = [
                event for event in waiting_trace["events"]
                if event["type"] == "runtime_gate" and event.get("status") == "acquired"
            ]
            self.assertEqual(len(acquired), 1)
            self.assertGreaterEqual(acquired[0].get("wait_ms", 0), 100)

    def test_task_time_budget_records_timeout_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                with self.assertRaises(TaskTimeoutError):
                    with task_time_budget("BUDGET_TRACE", timeout_seconds=0.01):
                        time.sleep(0.03)
            trace = reconstruct_run("BUDGET_TRACE", output)
            statuses = [event.get("status") for event in trace["events"] if event["type"] == "runtime_budget"]
            self.assertEqual(statuses, ["started", "network_error"])

    def test_model_call_trace_keeps_visible_io_and_drops_hidden_thought(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            token = current_task_id.set("MODEL_TRACE")
            try:
                with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                    append_task_event("MODEL_TRACE", "user_input", content="test")
                    record_model_event(
                        "MODEL_TRACE",
                        status="completed",
                        operation="query_rewrite",
                        prompt="User: test",
                        output="<think>private</think>visible query",
                    )
                    append_task_event("MODEL_TRACE", "final", status="completed", content="done")
                    trace = reconstruct_run("MODEL_TRACE", output)
            finally:
                current_task_id.reset(token)
            model_output = next(item for item in trace["model_outputs"] if item["phase"] == "query_rewrite")
            self.assertEqual(model_output["output"], "visible query")
            self.assertNotIn("private", json.dumps(trace))

    def test_acceptance_trace_keeps_request_level_sampling_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            token = current_task_id.set("SAMPLING_TRACE")
            try:
                with patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False):
                    record_model_event(
                        "SAMPLING_TRACE",
                        status="completed",
                        operation="chat_completion",
                        prompt="User: replan",
                        output='{"name":"web_search"}',
                        request_stage="planner_replan",
                        sampling_policy_reason="repeated_strategy_failure",
                        temperature=0.35,
                        seed=None,
                        sampling_parameters={
                            "temperature": 0.35,
                            "top_k": 50,
                            "top_p": 0.35,
                        },
                        request_max_tokens=640,
                        finish_reason="stop",
                        stop=["\nUser:", "\nAssistant:"],
                    )
                    trace = _trace_summary("SAMPLING_TRACE")
            finally:
                current_task_id.reset(token)
            call = trace["model_calls"][0]
            self.assertEqual(call["request_stage"], "planner_replan")
            self.assertEqual(call["sampling_policy_reason"], "repeated_strategy_failure")
            self.assertEqual(call["temperature"], 0.35)
            self.assertEqual(call["request_max_tokens"], 640)
            self.assertEqual(call["finish_reason"], "stop")
            self.assertEqual(call["stop"], ["\nUser:", "\nAssistant:"])
            self.assertEqual(
                call["sampling_parameters"],
                {"temperature": 0.35, "top_k": 50, "top_p": 0.35},
            )

    def test_offline_risk_policy_does_not_enter_or_rewrite_runtime_answer(self):
        policy = {"domain": "medicine_literacy", "risk_checks": ["professional confirmation"]}
        self.assertFalse(validate_risk_answer("Take this as a diagnosis.", policy)["valid"])
        self.assertTrue(validate_risk_answer("仅供参考，请咨询专业人员。", policy)["valid"])

        class FakeLLM:
            def text_completion(self, prompt, max_tokens=384, stop=None):
                self.prompt = prompt
                self.stop = stop
                return type("Response", (), {"content": "仅供参考，请咨询专业人员。"})()

        llm = FakeLLM()
        result = synthesize_retrieval_answer(
            "请解释医学信息",
            {"results": [], "citation_refs": []},
            llm=llm,
            constraints=policy,
        )
        self.assertNotIn("high-risk medical information", llm.prompt)
        self.assertNotIn("qualified professional", llm.prompt)
        self.assertEqual(result["content"], llm.prompt and result["model_output"])
        self.assertEqual(result["answer_quality"], {})


if __name__ == "__main__":
    unittest.main()
