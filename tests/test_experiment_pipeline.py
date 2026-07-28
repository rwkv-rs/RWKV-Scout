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
from utils.retry import retry_with_fallback
from utils.runtime_gate import analysis_slot
from utils.time_budget import TaskTimeoutError, task_time_budget
from utils.model_events import record_model_event
from utils.risk_policy import validate_risk_answer
from utils.reference_validation import validate_dataset_references, validate_reference_case
from agent.retrieval_synthesis import synthesize_retrieval_answer
from utils.task_events import append_task_event
from utils.token_tracker import current_task_id
from utils.trace_validation import validate_replay_trace
from config import get_experiment_model_config, validate_experiment_model_contract
from agent.orchestrator import Orchestrator


class ExperimentPipelineTests(unittest.TestCase):
    def test_experiment_model_contract_is_the_active_rwkv_13b_profile(self):
        profile = get_experiment_model_config()
        self.assertEqual(profile["model"], "rwkv7-g1i_preview4922-13.3b-20260720-ctx12288")
        self.assertEqual(profile["endpoint"], "http://172.21.122.93:29613/v1")
        self.assertEqual(profile["context_length"], 12288)
        self.assertEqual(validate_experiment_model_contract()["api_key"], "rwkv-skills")

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
            ):
                result = run_preflight()
            serialized = json.dumps(result, ensure_ascii=False)
            self.assertNotIn("rwkv-skills", serialized)
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

    def test_controlled_orchestrator_run_writes_full_trace_without_model_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            trace_log = root / "logs"
            retrieved = {
                "status": "ok",
                "real_network": False,
                "retrieved_at": "2026-07-27T00:00:00+00:00",
                "results": [{"title": "Fixture source", "url": "https://example.com/fixture", "page_excerpt": "The fixture fact is 42."}],
                "sources": ["https://example.com/fixture"],
                "citation_refs": [{"ref_id": "S1", "title": "Fixture source", "url": "https://example.com/fixture", "content": "The fixture fact is 42."}],
                "provider_errors": [],
            }
            with (
                patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False),
                patch.dict("config.TRACKING", {"log_dir": str(trace_log), "enable": False}, clear=False),
                patch("agent.orchestrator.generate_query_candidates", return_value={"queries": ["fixture fact"], "source": "test"}),
                patch("agent.orchestrator.execute_parallel_candidates", return_value=[("fixture fact", retrieved)]),
                patch(
                    "agent.orchestrator.synthesize_retrieval_answer",
                    return_value={
                        "content": "The fixture fact is 42 [S1].",
                        "mode": "test_model",
                        "evidence_count": 1,
                        "citation_refs": retrieved["citation_refs"],
                        "prompt": "test prompt",
                        "model_output": "The fixture fact is 42 [S1].",
                        "context_text": "[S1] Fixture source\nFacts: The fixture fact is 42.",
                        "selected_evidence": [
                            {
                                "ref_id": "S1",
                                "url": "https://example.com/fixture",
                                "source_chars": 25,
                                "selected_chars": 25,
                                "chunk_count": 1,
                                "truncated": False,
                            }
                        ],
                        "context_stats": {"context_tokens": 12, "chunk_count": 1, "truncated_count": 0},
                    },
                ),
            ):
                orchestrator = Orchestrator()
                def fail_if_planner_runs(*_args):
                    raise AssertionError("controlled retrieval must bypass the planner")

                orchestrator.planner.plan_next_action = fail_if_planner_runs
                result = orchestrator.run(
                    "search fixture fact",
                    task_id="CONTROLLED_TRACE",
                    run_metadata={
                        "experiment_id": "test",
                        "variant": "baseline",
                        "dataset_version": "eval-test",
                        "search_action": "search_web_keyless",
                    },
                )
                trace = reconstruct_run("CONTROLLED_TRACE", output)

            self.assertIn("42", result)
            event_types = {event["type"] for event in trace["events"]}
            self.assertTrue(
                {
                    "query_candidates",
                    "tool_result",
                    "content_extract",
                    "ranking",
                    "context_build",
                    "citation_validation",
                    "synthesis",
                    "final",
                }.issubset(event_types)
            )
            self.assertEqual(trace["manifest"]["experiment"]["dataset_version"], "eval-test")
            self.assertEqual(trace["model_outputs"][0]["prompt"], "test prompt")
            self.assertEqual(trace["context_trace"][0]["data"]["context_stats"]["chunk_count"], 1)
            self.assertEqual(trace["ranking_trace"][0]["data"]["method"], "candidate_support_then_rank.v1")

    def test_agentic_loop_allows_repeat_and_forces_summary_at_step_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task"
            task_dir.mkdir()
            orchestrator = Orchestrator()
            orchestrator.state.task_id = "REPEAT_STEP_TRACE"
            orchestrator.state.task_output_dir = str(task_dir)
            orchestrator.state.user_query = "find stations"
            orchestrator.state.run_metadata = {"max_tool_steps": 3}
            plan = {
                "schema_version": "task_plan.v1",
                "goal": "find stations",
                "atomic_points": [
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
                    {"action": "search_mediawiki", "args": {"query": "stations", "max_results": 2}},
                    {"action": "fetch_mediawiki_page", "args": {"url": "https://example.com/one"}},
                    {"action": "fetch_mediawiki_page", "args": {"url": "https://example.com/one"}},
                ]
            )
            executed = []
            synthesis_calls = []

            def fake_execute(action, args, context, phase=None):
                executed.append((action, args, phase))
                if action == "search_mediawiki":
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
                patch("agent.orchestrator.ToolRegistry.execute", side_effect=fake_execute),
                patch("agent.orchestrator.synthesize_retrieval_answer", side_effect=fake_synthesis),
            ):
                orchestrator.planner.create_task_plan = lambda *_args: plan
                orchestrator.planner.begin_task = lambda *_args: None
                orchestrator.planner.plan_next_action = lambda *_args: next(decisions)
                orchestrator.planner.observe_tool_result = lambda *_args: None
                result = orchestrator._run_model_tool_loop("find stations", {})

            self.assertIn("bounded summary", result)
            self.assertEqual([item[0] for item in executed], ["search_mediawiki", "fetch_mediawiki_page", "fetch_mediawiki_page"])
            self.assertEqual(len(synthesis_calls), 1)
            self.assertEqual(synthesis_calls[0]["termination_reason"], "max_steps_reached")

    def test_invalid_citation_triggers_one_bounded_recovery_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            trace_log = root / "logs"
            bad = {
                "status": "ok",
                "real_network": False,
                "results": [{"title": "Search page", "url": "https://www.google.com/search?q=bad", "page_excerpt": "bad"}],
                "sources": ["https://www.google.com/search?q=bad"],
                "citation_refs": [{"ref_id": "BAD", "url": "https://www.google.com/search?q=bad"}],
                "provider_errors": [],
            }
            good = {
                "status": "ok",
                "real_network": False,
                "results": [{"title": "Primary", "url": "https://example.com/primary", "page_excerpt": "The reviewed fact is 7."}],
                "sources": ["https://example.com/primary"],
                "citation_refs": [{"ref_id": "GOOD", "url": "https://example.com/primary"}],
                "provider_errors": [],
            }
            syntheses = [
                {
                    "content": "Unsupported [BAD].",
                    "mode": "test_model",
                    "evidence_count": 1,
                    "citation_refs": bad["citation_refs"],
                    "prompt": "bad prompt",
                    "model_output": "Unsupported [BAD].",
                },
                {
                    "content": "The reviewed fact is 7 [GOOD].",
                    "mode": "test_model",
                    "evidence_count": 1,
                    "citation_refs": good["citation_refs"],
                    "prompt": "good prompt",
                    "model_output": "The reviewed fact is 7 [GOOD].",
                },
            ]
            with (
                patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False),
                patch.dict("config.TRACKING", {"log_dir": str(trace_log), "enable": False}, clear=False),
                patch(
                    "agent.orchestrator.generate_query_candidates",
                    side_effect=[{"queries": ["bad"], "source": "test"}, {"queries": ["good"], "source": "test"}],
                ),
                patch("agent.orchestrator.execute_parallel_candidates", side_effect=[[('bad', bad)], [('good', good)]]),
                patch("agent.orchestrator.synthesize_retrieval_answer", side_effect=syntheses),
            ):
                orchestrator = Orchestrator()
                orchestrator.planner.create_task_plan = lambda *_args: {
                    "schema_version": "task_plan.v1",
                    "goal": "recover citation",
                    "atomic_points": [
                        {
                            "id": "P1",
                            "task": "verify the reviewed fact",
                            "objective": "recover the reviewed fact",
                            "evidence_needed": ["reviewed fact"],
                            "acceptance_criteria": ["the reviewed fact is directly supported"],
                            "output_format": "prose",
                            "status": "pending",
                        }
                    ],
                    "completion_rule": "The reviewed fact is supported.",
                }
                orchestrator.planner.plan_next_action = lambda *_args: {
                    "action": "search_web_keyless",
                    "args": {"max_results": 1, "fetch_pages": 1},
                    "router": "test",
                }
                orchestrator.run("recover citation", task_id="CITATION_RECOVERY")
                trace = reconstruct_run("CITATION_RECOVERY", output)

            recovery = [event for event in trace["events"] if event["type"] == "citation_recovery"]
            final = [event for event in trace["events"] if event["type"] == "final"][-1]
            self.assertEqual([event["status"] for event in recovery], ["scheduled", "completed"])
            self.assertEqual(final["status"], "completed")
            self.assertIn("reviewed fact", trace["final_answer"])

    def test_retrieval_only_mode_skips_synthesis_without_model_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            trace_log = root / "logs"
            retrieved = {
                "status": "ok",
                "real_network": False,
                "results": [
                    {
                        "title": "Fixture source",
                        "url": "https://example.com/fixture",
                        "page_excerpt": "The fixture fact is 42.",
                    }
                ],
                "sources": ["https://example.com/fixture"],
                "citation_refs": [
                    {
                        "ref_id": "S1",
                        "title": "Fixture source",
                        "url": "https://example.com/fixture",
                        "content": "The fixture fact is 42.",
                    }
                ],
            }
            with (
                patch.dict("config.DATA_PIPELINE", {"output_directory": str(output)}, clear=False),
                patch.dict("config.TRACKING", {"log_dir": str(trace_log), "enable": False}, clear=False),
                patch("agent.orchestrator.generate_query_candidates", return_value={"queries": ["fixture fact"], "source": "test"}),
                patch("agent.orchestrator.execute_parallel_candidates", return_value=[("fixture fact", retrieved)]),
                patch(
                    "agent.orchestrator.synthesize_retrieval_answer",
                    side_effect=AssertionError("retrieval-only mode must not call synthesis"),
                ),
            ):
                orchestrator = Orchestrator()
                orchestrator.planner.plan_next_action = lambda *_args: (_ for _ in ()).throw(
                    AssertionError("retrieval-only mode must bypass planning")
                )
                result = orchestrator.run(
                    "search fixture fact",
                    task_id="RETRIEVAL_ONLY_TRACE",
                    run_metadata={
                        "search_action": "search_web_keyless",
                        "retrieval_only": True,
                    },
                )
                trace = reconstruct_run("RETRIEVAL_ONLY_TRACE", output)

            self.assertEqual(result, "")
            self.assertEqual(trace["final_answer"], "")
            self.assertEqual(trace["events"][-1]["mode"], "retrieval_only")
            self.assertFalse([event for event in trace["events"] if event["type"] == "model_call"])
            self.assertEqual(trace["model_outputs"][0]["mode"], "retrieval_only")
            validation = validate_replay_trace(trace)
            self.assertTrue(validation["valid"], validation)

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
            self.assertEqual(statuses, ["started", "timed_out"])

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

    def test_high_risk_policy_is_in_prompt_and_answer_gate(self):
        policy = {"domain": "medicine_literacy", "risk_checks": ["professional confirmation"]}
        self.assertFalse(validate_risk_answer("Take this as a diagnosis.", policy)["valid"])
        self.assertTrue(validate_risk_answer("仅供参考，请咨询专业人员。", policy)["valid"])

        class FakeLLM:
            def text_completion(self, prompt, max_tokens=384):
                self.prompt = prompt
                return type("Response", (), {"content": "仅供参考，请咨询专业人员。"})()

        llm = FakeLLM()
        result = synthesize_retrieval_answer(
            "请解释医学信息",
            {"results": [], "citation_refs": []},
            llm=llm,
            constraints=policy,
        )
        self.assertIn("high-risk medical information", llm.prompt)
        self.assertIn("qualified professional", llm.prompt)
        self.assertTrue(result["content"])


if __name__ == "__main__":
    unittest.main()
