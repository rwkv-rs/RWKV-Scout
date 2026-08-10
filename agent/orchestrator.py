"""RWKV-owned retrieval orchestration.

The controller executes model decisions and records evidence.  It does not
rewrite the task plan, choose replacement routes, judge answer quality, repair
model prose, or assign success/partial/error labels to an answer.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from agent.planner import Planner
from agent.retrieval_loop import merge_retrieval_results
from agent.retrieval_synthesis import build_evidence_context, synthesize_retrieval_answer
from agent.slm_scheduler import GLOBAL_SLM_INPUT_SCHEDULER
from agent.state import AgentState
from agent.unified_research import run_unified_research_loop
from config import (
    DATA_PIPELINE,
    TRACKING,
    get_llm_base_url,
    get_llm_context_length,
    get_llm_model,
    get_llm_provider,
)
from tools.builtin import load_builtin_tools
from clients.llm_client import LLMClient
from utils.experiment_strategies import normalize_strategy
from utils.freshness import build_freshness_policy
from utils.chunker import get_token_count
from utils.task_events import append_task_event
from utils.task_manager import update_task_progress


def planner_environment_context(now: datetime | None = None) -> str:
    """Expose observable runtime time to RWKV before current/latest planning."""

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    return json.dumps(
        {
            "current_utc_datetime": current.isoformat(timespec="seconds"),
            "current_utc_date": current.date().isoformat(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
from utils.time_budget import check_time_budget
from utils.tracker import EventTracker


DEFAULT_MAX_TOOL_STEPS = 50

_RUNTIME_METADATA_KEYS = frozenset(
    {
        "context_source_count",
        "generic_web_search_only",
        "max_tool_steps",
        "prompt_variant",
        "ranking_strategy",
        "strategy_config",
    }
)
_AUDIT_METADATA_KEYS = frozenset(
    {
        "baseline_run_id",
        "dataset_version",
        "experiment_id",
        "prompt_version",
        "variant",
    }
)


def runtime_metadata_only(value: dict | None) -> dict:
    """Keep runtime controls while excluding benchmark/reference material."""

    raw = value if isinstance(value, dict) else {}
    return {key: raw[key] for key in _RUNTIME_METADATA_KEYS if key in raw}


def _audit_metadata_only(value: dict | None) -> dict:
    raw = value if isinstance(value, dict) else {}
    return {key: raw[key] for key in _AUDIT_METADATA_KEYS if key in raw}


class Orchestrator:
    def __init__(self):
        load_builtin_tools()
        self.tracker = EventTracker(
            log_dir=TRACKING.get("log_dir", "./logs"),
            enable=TRACKING.get("enable", True),
        )
        self.state = AgentState()
        self.state.working_memory["__category_tree__"] = {}
        self.llm = LLMClient()
        self.planner = Planner()
        self._task_plan: dict[str, Any] = {}
        self._retrieval_ledger = self.state.retrieval.progress
        self._calculation_results: list[dict[str, Any]] = []
        self._arithmetic_results: list[dict[str, Any]] = []
        self._time_results: list[dict[str, Any]] = []
        self._model_protocol_failure = False

    def _deterministic_results(self) -> list[dict[str, Any]]:
        return [*self._calculation_results, *self._time_results, *self._arithmetic_results]

    def _cross_validate_research(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        *,
        step: int,
    ) -> dict[str, Any]:
        """Let RWKV decide whether to finish research or replan."""

        claim_snapshot = self.state.retrieval.claims.snapshot()
        context = build_evidence_context(
            {
                "results": self.state.retrieval.source_records(),
                "calculation_results": self._deterministic_results(),
                "claim_ledger": claim_snapshot,
            },
            constraints=self.state.run_metadata,
            query=user_query,
            max_sources=6,
            token_budget=3200,
            max_chunks_per_source=2,
            source_locator_char_limit=800,
        )
        routing_snapshot = self.state.retrieval.planner_routing_snapshot()
        review_context = (
            "CURRENT RUNTIME:\n"
            + str(self.state.run_metadata.get("runtime_environment") or planner_environment_context())
            + "\n\n"
            + context["text"]
            + "\n\nSHARED RETRIEVAL STATE:\n"
            + json.dumps(
                routing_snapshot,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        validation_context_stats = {
            **context["context_stats"],
            "review_context_tokens": get_token_count(review_context),
            "routing_snapshot_schema": str(
                routing_snapshot.get("schema_version") or ""
            ),
        }
        available_evidence_refs = [
            str(item.get("ref_id") or "").strip()
            for item in context.get("citation_refs") or []
            if isinstance(item, dict) and str(item.get("ref_id") or "").strip()
        ]
        if self._deterministic_results():
            available_evidence_refs.append("TOOL_RESULTS")
        evidence_text_by_ref = {
            str(item.get("ref_id") or ""): "\n".join(
                value
                for value in (
                    str(item.get("title") or ""),
                    str(item.get("url") or ""),
                    str(item.get("evidence_text") or ""),
                )
                if value
            )
            for item in context.get("selected_evidence") or []
            if isinstance(item, dict) and str(item.get("ref_id") or "").strip()
        }
        if self._deterministic_results():
            evidence_text_by_ref["TOOL_RESULTS"] = json.dumps(
                self._deterministic_results(), ensure_ascii=False
            )
        review = self.planner.cross_validate_research(
            user_query,
            task_plan,
            review_context,
            evidence_refs=available_evidence_refs,
            evidence_text_by_ref=evidence_text_by_ref,
        )
        evidence_ref_map = {
            str(item.get("ref_id") or ""): {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
            }
            for item in context.get("citation_refs") or []
            if isinstance(item, dict) and str(item.get("ref_id") or "").strip()
        }
        validated_source_urls: list[str] = []
        point_status = review.get("task_point_status") or {}
        if isinstance(point_status, dict):
            for value in point_status.values():
                if not isinstance(value, dict):
                    continue
                if str(value.get("status") or "").casefold() not in {
                    "answered",
                    "complete",
                    "completed",
                    "supported",
                    "verified",
                }:
                    continue
                for ref in value.get("evidence_refs") or []:
                    url = str((evidence_ref_map.get(str(ref)) or {}).get("url") or "").strip()
                    if url and url not in validated_source_urls:
                        validated_source_urls.append(url)
        review["evidence_ref_map"] = evidence_ref_map
        review["validated_source_urls"] = validated_source_urls
        evidence_revision = int(self.state.retrieval.evidence_revision or 0)
        review["evidence_revision"] = evidence_revision
        point_status_projection: dict[str, dict[str, Any]] = {}
        if isinstance(point_status, dict):
            for point_id, value in point_status.items():
                if not isinstance(value, dict):
                    continue
                review_refs = [
                    str(ref)
                    for ref in value.get("evidence_refs") or []
                    if str(ref).strip()
                ][:8]
                point_status_projection[str(point_id)] = {
                    "status": str(value.get("status") or "")[:120],
                    # [S#] identifiers are local to the validation context and
                    # may be renumbered by final evidence packing. Persist the
                    # bound URLs instead so the final writer cannot confuse a
                    # review-local S1 with a different final S1.
                    "evidence_urls": [
                        str((evidence_ref_map.get(ref) or {}).get("url") or "")[:1000]
                        for ref in review_refs
                        if str((evidence_ref_map.get(ref) or {}).get("url") or "").strip()
                    ],
                    "supported_facts": [
                        {
                            "field": str(fact.get("field") or "")[:160],
                            "value": str(fact.get("value") or "")[:500],
                            "evidence_url": str(
                                (
                                    evidence_ref_map.get(
                                        str(fact.get("evidence_ref") or "")
                                    )
                                    or {}
                                ).get("url")
                                or ""
                            )[:1000],
                            "quote": str(fact.get("quote") or "")[:1000],
                        }
                        for fact in value.get("supported_facts") or []
                        if isinstance(fact, dict)
                    ][:12],
                    "missing_fields": [
                        str(field)[:300]
                        for field in value.get("missing_fields") or []
                        if str(field).strip()
                    ][:12],
                }
        self.state.run_metadata["last_cross_validation"] = {
            "schema_version": str(
                review.get("schema_version") or "rwkv-cross-validation.v1"
            ),
            "decision": str(review.get("decision") or "")[:120],
            "missing_points": [
                str(value)[:120]
                for value in review.get("missing_points") or []
                if str(value).strip()
            ][:16],
            "conflicts": [
                str(value)[:500]
                for value in review.get("conflicts") or []
                if str(value).strip()
            ][:8],
            "task_point_status": point_status_projection,
            "next_focus": str(review.get("next_focus") or "")[:1000],
            "reason": str(review.get("reason") or "")[:1000],
            "evidence_revision": evidence_revision,
        }
        if str(review.get("decision") or "").casefold() in {"finish", "replan"}:
            self.state.run_metadata["last_cross_validation_evidence_revision"] = (
                evidence_revision
            )
            self.state.run_metadata["validated_source_urls"] = validated_source_urls[:24]
        append_task_event(
            self.state.task_id,
            "cross_validation",
            step=step,
            phase="VALIDATION",
            decision=review.get("decision", ""),
            missing_points=review.get("missing_points") or [],
            conflicts=review.get("conflicts") or [],
            next_focus=review.get("next_focus", ""),
            reason=review.get("reason", ""),
            error_class=review.get("error_class", ""),
            message=review.get("message", ""),
            sampling_temperature=review.get("sampling_temperature"),
            sampling_seed=review.get("sampling_seed"),
            task_point_status=point_status_projection,
            raw_model_output=review.get("raw_model_output", ""),
            prompt=review.get("prompt", ""),
            context_text=review_context,
            context_stats=validation_context_stats,
            evidence_revision=evidence_revision,
            claim_ledger=claim_snapshot,
            decision_owner="rwkv",
        )
        return review

    def _cross_validate_if_evidence_changed(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        *,
        step: int,
    ) -> dict[str, Any] | None:
        """Review one material evidence revision at most once before synthesis."""

        current_revision = int(self.state.retrieval.evidence_revision or 0)
        reviewed_revision_value = self.state.run_metadata.get(
            "last_cross_validation_evidence_revision"
        )
        reviewed_revision = (
            int(reviewed_revision_value)
            if reviewed_revision_value is not None
            else -1
        )
        if current_revision <= reviewed_revision:
            return None
        return self._cross_validate_research(
            user_query,
            task_plan,
            step=step,
        )

    def _retrieval_context(self) -> dict[str, Any]:
        return {
            "original_goal": self.state.user_query,
            "path_to_id": self.state.path_to_id,
            "id_to_path": self.state.id_to_path,
            "working_memory": self.state.working_memory,
            "tracker": self.tracker,
            "agent_state": self.state,
            "task_id": self.state.task_id,
            "task_plan": self._task_plan,
            "slm_scheduler": GLOBAL_SLM_INPUT_SCHEDULER,
        }

    def _agentic_tool_context(self) -> dict[str, Any]:
        context = self._retrieval_context()
        context["agentic_tool_loop"] = True
        context["retrieval_ledger"] = self._retrieval_ledger.observation()
        return context

    def _ranking_strategy(self) -> str:
        metadata = self.state.run_metadata
        configured = metadata.get("strategy_config")
        if not isinstance(configured, dict):
            configured = {
                key: metadata[key]
                for key in ("ranking_strategy", "context_source_count", "prompt_variant")
                if key in metadata
            }
        return normalize_strategy(configured).get("ranking_strategy", "evidence_quality.v1")

    def _record_retrieval_progress(
        self,
        result: dict[str, Any],
        *,
        query: str,
        step: int,
        action: str,
        phase: str,
        branch_id: str = "",
        task_point_id: str = "",
    ) -> dict[str, Any]:
        """Record retrieval state without changing the model-selected route."""

        delta = self._retrieval_ledger.record(
            query,
            result,
            step=step,
            branch_id=branch_id,
            task_point_id=task_point_id,
            action=action,
            phase=phase,
        )
        observation = self._retrieval_ledger.observation(
            branch_id=branch_id,
            task_point_id=task_point_id,
        )
        enriched = {**result, "retrieval_delta": delta, "retrieval_ledger": observation}
        append_task_event(
            self.state.task_id,
            "retrieval_ledger",
            step=step,
            phase=phase,
            branch_id=branch_id,
            task_point_id=task_point_id,
            action=action,
            query=query,
            data=delta,
            snapshot=observation,
        )
        return enriched

    def _record_final_ranking(self, action: str, data: dict[str, Any], step: int) -> None:
        append_task_event(
            self.state.task_id,
            "ranking",
            step=step,
            phase="CONTEXT",
            action=action,
            data={
                "method": data.get("ranking_strategy", self._ranking_strategy()),
                "output_count": len(data.get("results") or []),
                "results": [
                    {
                        "rank": item.get("retrieval_rank"),
                        "url": item.get("url", ""),
                        "candidate_queries": item.get("candidate_queries") or [],
                    }
                    for item in data.get("results") or []
                    if isinstance(item, dict)
                ],
            },
        )

    def _write_agentic_report(
        self,
        user_query: str,
        action: str,
        answer: str,
        *,
        data: dict[str, Any] | None = None,
        mode: str = "rwkv_final",
    ) -> None:
        report_path = os.path.join(self.state.task_output_dir, "retrieval_report.jsonl")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "record_type": "retrieval_result",
                        "task_id": self.state.task_id,
                        "query": user_query,
                        "action": action,
                        "answer": answer,
                        "answer_mode": mode,
                        "data": data or {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _complete_model_tool_loop(
        self,
        user_query: str,
        action: str,
        rounds: list[tuple[str, dict[str, Any]]],
        step: int,
        *,
        termination_reason: str = "model_requested_finish",
    ) -> str:
        """Give collected material to one RWKV final writer and publish it."""

        if termination_reason == "max_steps_reached":
            append_task_event(
                self.state.task_id,
                "step_limit_reached",
                step=step,
                phase="SYNTHESIS",
                action=action,
                max_steps=step,
                evidence_rounds=len(rounds),
            )

        if rounds:
            merged = merge_retrieval_results(
                user_query,
                action,
                rounds,
                ranking_strategy=self._ranking_strategy(),
            )
            if not merged.get("results"):
                merged["results"] = self.state.retrieval.source_records()
        else:
            merged = {
                "query": user_query,
                "status": "no_retrieval",
                "results": self.state.retrieval.source_records(),
                "sources": [],
                "citation_refs": [],
                "round_count": 0,
                "real_network": True,
            }

        merged["calculation_results"] = self._deterministic_results()
        claim_snapshot = self.state.retrieval.claims.snapshot()
        merged["claim_ledger"] = claim_snapshot
        self.state.run_metadata["claim_ledger"] = claim_snapshot
        self._record_final_ranking(action, merged, step)

        synthesis = synthesize_retrieval_answer(
            user_query,
            merged,
            llm=self.llm,
            constraints=self.state.run_metadata,
            termination_reason=termination_reason,
        )
        answer = str(synthesis.get("content") or "")

        append_task_event(
            self.state.task_id,
            "context_build",
            step=step,
            phase="CONTEXT",
            data={
                "context_text": synthesis.get("context_text", ""),
                "selected_evidence": synthesis.get("selected_evidence") or [],
                "context_stats": synthesis.get("context_stats") or {},
                "claim_ledger": claim_snapshot,
            },
        )
        append_task_event(
            self.state.task_id,
            "synthesis",
            step=step,
            phase="SYNTHESIS",
            action=action,
            content=answer,
            mode="rwkv_final",
            evidence_count=synthesis.get("evidence_count", 0),
            citation_refs=synthesis.get("citation_refs") or [],
            prompt=synthesis.get("prompt", ""),
            model_output=synthesis.get("model_output", ""),
            generation_attempts=synthesis.get("generation_attempts") or [],
            context_text=synthesis.get("context_text", ""),
            selected_evidence=synthesis.get("selected_evidence") or [],
            context_stats=synthesis.get("context_stats") or {},
            claim_ledger=claim_snapshot,
            termination_reason=termination_reason,
        )

        self.state.final_result = answer
        self.state.is_finished = True
        append_task_event(
            self.state.task_id,
            "final",
            content=answer,
            action=action,
            mode="rwkv_final",
            round_count=merged.get("round_count", len(rounds)),
            citation_refs=synthesis.get("citation_refs") or [],
            termination_reason=termination_reason,
            model_output_available=True,
            claim_ledger=claim_snapshot,
        )
        self._write_agentic_report(
            user_query,
            action,
            answer,
            data=merged,
            mode="rwkv_final",
        )
        return answer

    def _prepare_task_plan(self, user_query: str, phase: str) -> dict[str, Any]:
        """Use the RWKV plan directly; no deterministic Intake rewrite."""

        self.planner.reset()
        environment_context = planner_environment_context()
        self.state.run_metadata["runtime_environment"] = environment_context
        task_plan = self.planner.create_task_plan(user_query, environment_context)
        append_task_event(
            self.state.task_id,
            "task_plan",
            step=0,
            phase="ROUTING",
            data=task_plan,
            plan_owner="rwkv",
            controller_rewritten=False,
        )
        self._task_plan = dict(task_plan)
        self.state.run_metadata["task_plan"] = self._task_plan
        if task_plan.get("status") == "error":
            return task_plan

        freshness_policy = build_freshness_policy(user_query, task_plan)
        self.state.run_metadata["freshness_policy"] = freshness_policy
        self.state.retrieval.claims.initialize(task_plan, user_query)
        self.state.run_metadata["claim_ledger"] = self.state.retrieval.claims.snapshot()
        self.planner.begin_task(user_query, environment_context, task_plan, phase)
        return task_plan

    def _run_model_tool_loop(self, user_query: str, model_profile: dict[str, Any]) -> str:
        max_steps = max(
            1,
            int(self.state.run_metadata.get("max_tool_steps", DEFAULT_MAX_TOOL_STEPS) or DEFAULT_MAX_TOOL_STEPS),
        )
        phase = "DISCOVERY"
        task_plan = self._prepare_task_plan(user_query, phase)
        if task_plan.get("status") == "error":
            append_task_event(
                self.state.task_id,
                "planning_error",
                step=0,
                phase="ROUTING",
                error=task_plan.get("message", ""),
                raw_model_output=task_plan.get("raw_model_output", ""),
            )
            return self._complete_model_tool_loop(
                user_query,
                "task_plan",
                self.state.retrieval.rounds,
                0,
                termination_reason="planner_protocol_error",
            )

        append_task_event(
            self.state.task_id,
            "retrieval_strategy_selected",
            step=0,
            phase="ROUTING",
            strategy="single_rwkv_loop",
            source="rwkv",
            controller_override=False,
        )
        self.state.run_metadata["retrieval_strategy"] = "single_rwkv_loop"
        self.state.run_metadata["retrieval_strategy_decision"] = {
            "owner": "rwkv",
            "point_count": len(task_plan.get("atomic_points") or []),
            "controller_override": False,
        }
        return self._run_single_loop(user_query, model_profile, task_plan, max_steps)

    def run(
        self,
        user_query: str,
        task_id: str | None = None,
        run_metadata: dict | None = None,
    ) -> str:
        check_time_budget(minimum_seconds=0.2)
        self.state.task_id = task_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        supplied_metadata = dict(run_metadata or {})
        self.state.run_metadata = runtime_metadata_only(supplied_metadata)
        self.state.retrieval.reset()
        self.state.is_finished = False
        self.state.final_result = ""
        self._retrieval_ledger = self.state.retrieval.progress
        self._calculation_results = []
        self._time_results = []
        self._arithmetic_results = []
        self._model_protocol_failure = False
        self.state.task_output_dir = os.path.join(
            DATA_PIPELINE.get("output_directory", "./data/output"),
            self.state.task_id,
        )
        os.makedirs(self.state.task_output_dir, exist_ok=True)

        self.tracker.track("User_Input", input_data=user_query, output_data=None)
        self.state.user_query = user_query
        append_task_event(self.state.task_id, "user_input", content=user_query)
        model_profile = {
            "model": get_llm_model(),
            "endpoint": get_llm_base_url(),
            "provider": get_llm_provider(),
            "context_length": get_llm_context_length(),
        }
        append_task_event(
            self.state.task_id,
            "run_started",
            phase="ROUTING",
            experiment=_audit_metadata_only(supplied_metadata),
            config={"model": model_profile, "decision_owner": "rwkv"},
            prompt_version=supplied_metadata.get("prompt_version", "unversioned"),
            run_metadata=self.state.run_metadata,
        )
        update_task_progress(self.state.task_id, "RWKV is planning and retrieving sources.")
        return self._run_model_tool_loop(user_query, model_profile)

    def _run_single_loop(
        self,
        user_query: str,
        model_profile: dict[str, Any],
        task_plan: dict[str, Any],
        max_steps: int,
    ) -> str:
        return run_unified_research_loop(
            self,
            user_query,
            model_profile,
            task_plan,
            max_steps,
        )


__all__ = ["Orchestrator", "runtime_metadata_only"]
