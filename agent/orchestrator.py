"""RWKV-owned retrieval orchestration.

The controller executes model decisions and records evidence.  It does not
rewrite the task plan, choose replacement routes, judge answer quality, repair
model prose, or assign success/partial/error labels to an answer.
"""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Any

from agent.evidence_resolution import (
    EVIDENCE_RESOLUTION_CONTRACT,
    attach_evidence_resolution_to_context,
    build_evidence_record_set,
    evidence_resolution_signature,
    resolve_evidence,
)
from agent.runtime_contracts import EVIDENCE_REVIEW_CONTRACT
from agent.planner import Planner
from agent.task_plan_contract import task_records
from agent.retrieval_loop import merge_retrieval_results
from agent.retrieval_synthesis import (
    build_evidence_context,
    synthesize_retrieval_answer,
)
from agent.slm_scheduler import GLOBAL_SLM_INPUT_SCHEDULER
from agent.state import AgentState
from agent.unified_research import (
    EvidenceReviewDecisionError,
    run_unified_research_loop,
)
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
        self._writer_context_cache_signature = ""
        self._writer_context_cache: dict[str, Any] | None = None
        self._evidence_resolution_cache_signature = ""
        self._evidence_resolution_cache: dict[str, Any] | None = None

    def _deterministic_results(self) -> list[dict[str, Any]]:
        return [*self._calculation_results, *self._time_results, *self._arithmetic_results]

    def _evidence_review_state_signature(self) -> str:
        """Review identity is the immutable evidence revision, never route churn."""

        return f"evidence:{int(self.state.retrieval.evidence_revision or 0)}"

    def _evidence_review_binding(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind a model decision to the exact immutable evidence packet."""

        context_stats = dict(context.get("context_stats") or {})
        evidence_lane_digest = str(
            context_stats.get("evidence_lane_digest") or ""
        ).strip()
        if not evidence_lane_digest:
            raise RuntimeError("immutable evidence lane has no digest")
        return {
            "evidence_revision": int(self.state.retrieval.evidence_revision or 0),
            "validation_state_signature": self._evidence_review_state_signature(),
            "evidence_lane_digest": evidence_lane_digest,
        }

    def _evidence_review_authorizes_writer(
        self,
        review: dict[str, Any] | None,
        user_query: str,
        task_plan: dict[str, Any],
        *,
        step: int,
        trigger: str,
    ) -> bool:
        """Return true only for a finish bound to the current evidence digest."""

        if not isinstance(review, dict):
            return False
        if str(review.get("decision") or "").casefold() != "finish":
            return False
        context = self._resolved_writer_context(
            user_query,
            task_plan,
            step=step,
            trigger=trigger,
        )
        binding = self._evidence_review_binding(context)
        return all(review.get(key) == value for key, value in binding.items())

    def _current_retrieval_data(
        self,
        user_query: str,
        action: str,
        rounds: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Build the one evidence record consumed by Evidence Review and the Writer.

        Round 19 let evidence-review inspect a smaller, differently ordered
        projection than final synthesis.  A binary review cannot be meaningful
        when the two RWKV calls see different source records.  This helper is
        intentionally mechanical: it merges retrieval rounds and attaches
        state, but it never decides whether any fact is true or sufficient.
        """

        active_rounds = list(
            self.state.retrieval.rounds if rounds is None else rounds
        )
        if active_rounds:
            data = merge_retrieval_results(
                user_query,
                action,
                active_rounds,
                ranking_strategy=self._ranking_strategy(),
            )
            if not data.get("results"):
                data["results"] = self.state.retrieval.source_records()
        else:
            data = {
                "query": user_query,
                "status": "no_retrieval",
                "results": self.state.retrieval.source_records(),
                "sources": [],
                "citation_refs": [],
                "round_count": 0,
                "real_network": True,
            }
        data["calculation_results"] = self._deterministic_results()
        data["evidence_ledger"] = self.state.retrieval.evidence_ledger.snapshot()
        return data

    def _review_evidence(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        *,
        step: int,
        trigger: str = "planner_finish",
        terminal: bool = False,
    ) -> dict[str, Any]:
        """Let RWKV make the binary continue-or-write Evidence Review decision."""

        evidence_ledger_snapshot = self.state.retrieval.evidence_ledger.snapshot()
        freshness_policy = self.state.run_metadata.get("freshness_policy")
        if not isinstance(freshness_policy, dict) or not freshness_policy:
            freshness_policy = build_freshness_policy(user_query, task_plan)
        context = self._resolved_writer_context(
            user_query,
            task_plan,
            step=step,
            trigger=trigger,
        )
        evidence_resolution_view = str(
            context.get("evidence_resolution_view") or ""
        ).strip()
        exact_evidence_text = str(
            context.get("evidence_text") or context.get("text") or ""
        )
        review_binding = self._evidence_review_binding(context)
        routing_snapshot = self.state.retrieval.planner_routing_snapshot(
            max_sources=0,
            max_queries=12,
            max_frozen_paths=12,
        )
        routing_context = json.dumps(
            routing_snapshot,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        allowed_evidence_record_ids = list(
            dict.fromkeys(
                str(item.get("evidence_record_id") or "").strip()
                for item in context.get("selected_evidence") or []
                if isinstance(item, dict)
                and str(item.get("evidence_record_id") or "").strip()
            )
        )
        allowed_route_ids = list(
            dict.fromkeys(
                str(row.get("route_id") or row.get("request_key") or "").strip()
                for key in ("queries", "frozen_paths")
                for row in routing_snapshot.get(key) or []
                if isinstance(row, dict)
                and str(row.get("route_id") or row.get("request_key") or "").strip()
            )
        )
        validation_context_stats = {
            **context["context_stats"],
            "exact_evidence_tokens": get_token_count(exact_evidence_text),
            "resolution_advisory_tokens": get_token_count(
                evidence_resolution_view
            ),
            "routing_context_tokens": get_token_count(routing_context),
        }
        review = self.planner.review_evidence(
            user_query,
            task_plan,
            exact_evidence_text,
            routing_context,
            resolution_advisory=evidence_resolution_view,
            allowed_evidence_record_ids=allowed_evidence_record_ids,
            allowed_route_ids=allowed_route_ids,
            terminal=terminal,
        )
        review.update(review_binding)
        review["trigger"] = str(trigger or "planner_finish")[:120]
        review["review_mode"] = "terminal" if terminal else "binary"

        # The review is audit/routing state only.  It never supplies facts,
        # validates sources, reorders evidence, or enters the final Writer
        # context.
        self.state.run_metadata.pop("validated_source_urls", None)
        self.state.run_metadata["last_evidence_review"] = {
            "contract": str(
                review.get("contract") or EVIDENCE_REVIEW_CONTRACT
            ),
            "decision": str(review.get("decision") or "")[:120],
            "trigger": str(trigger or "planner_finish")[:120],
            **review_binding,
        }
        append_task_event(
            self.state.task_id,
            "evidence_review",
            step=step,
            phase="VALIDATION",
            trigger=str(trigger or "planner_finish"),
            decision=review.get("decision", ""),
            error_class=review.get("error_class", ""),
            message=review.get("message", ""),
            sampling_temperature=review.get("sampling_temperature"),
            sampling_seed=review.get("sampling_seed"),
            review_attempts=review.get("review_attempts"),
            review_mode=review.get("review_mode", "binary"),
            raw_model_output=review.get("raw_model_output", ""),
            prompt=review.get("prompt", ""),
            exact_evidence_text=exact_evidence_text,
            resolution_advisory=evidence_resolution_view,
            routing_context=routing_context,
            context_stats=validation_context_stats,
            **review_binding,
            allowed_evidence_record_ids=allowed_evidence_record_ids,
            allowed_route_ids=allowed_route_ids,
            evidence_ledger=evidence_ledger_snapshot,
            decision_owner="rwkv",
        )
        return review

    def _resolved_writer_context(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        *,
        step: int,
        trigger: str,
    ) -> dict[str, Any]:
        """Build one immutable evidence packet shared by Evidence Review and Writer."""

        freshness_policy = self.state.run_metadata.get("freshness_policy")
        if not isinstance(freshness_policy, dict) or not freshness_policy:
            freshness_policy = build_freshness_policy(user_query, task_plan)
        writer_constraints = {
            **runtime_metadata_only(self.state.run_metadata),
            "freshness_policy": freshness_policy,
            "task_plan": task_plan,
        }
        constraint_digest = hashlib.sha256(
            json.dumps(
                writer_constraints,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
        signature = (
            f"{self._evidence_review_state_signature()}:"
            f"writer_constraints:{constraint_digest}"
        )
        if (
            signature == self._writer_context_cache_signature
            and isinstance(self._writer_context_cache, dict)
        ):
            return self._writer_context_cache

        base_context = build_evidence_context(
            self._current_retrieval_data(user_query, "writer_context"),
            constraints=writer_constraints,
            query=user_query,
        )
        resolution: dict[str, Any]
        if bool(DATA_PIPELINE.get("evidence_resolution_enabled", True)):
            evidence_records = build_evidence_record_set(
                base_context.get("selected_evidence") or [],
                max_records=int(
                    DATA_PIPELINE.get("evidence_resolution_max_records", 24)
                    or 24
                ),
                max_chars_per_record=int(
                    DATA_PIPELINE.get(
                        "evidence_resolution_max_chars_per_record", 12000
                    )
                    or 12000
                ),
            )
            resolution_cache_signature = evidence_resolution_signature(
                user_query,
                task_plan,
                evidence_records,
            )
            if (
                resolution_cache_signature
                == self._evidence_resolution_cache_signature
                and isinstance(self._evidence_resolution_cache, dict)
            ):
                resolution = self._evidence_resolution_cache
            else:
                resolution = resolve_evidence(
                    user_query,
                    task_plan,
                    evidence_records,
                    self.llm,
                )
                self._evidence_resolution_cache_signature = resolution_cache_signature
                self._evidence_resolution_cache = resolution
                append_task_event(
                    self.state.task_id,
                    "evidence_resolution",
                    step=step,
                    phase="VALIDATION",
                    trigger=str(trigger or "finalize")[:120],
                    status=resolution.get("status", ""),
                    evidence_record_count=resolution.get("evidence_record_count", 0),
                    decisions=resolution.get("decisions") or [],
                    input_digest=resolution.get("input_digest", ""),
                    attempts=resolution.get("attempts", 0),
                    error=resolution.get("error", ""),
                    prompt=resolution.get("prompt", ""),
                    raw_model_output=resolution.get("raw_model_output", ""),
                    stage_calls=resolution.get("stage_calls") or [],
                    sampling_temperature=resolution.get("sampling_temperature"),
                    sampling_parameters=resolution.get("sampling_parameters") or {},
                    decision_owner="rwkv_evidence_resolution",
                )
        else:
            resolution = {
                "contract": EVIDENCE_RESOLUTION_CONTRACT,
                "status": "disabled",
                "evidence_record_count": len(base_context.get("selected_evidence") or []),
                "decisions": [],
                "attempts": 0,
            }
        resolved_context = attach_evidence_resolution_to_context(
            base_context,
            resolution,
            task_plan,
        )
        self._writer_context_cache_signature = signature
        self._writer_context_cache = resolved_context
        append_task_event(
            self.state.task_id,
            "writer_context",
            step=step,
            phase="VALIDATION",
            trigger=str(trigger or "finalize")[:120],
            context_text=resolved_context.get("text", ""),
            selected_evidence=resolved_context.get("selected_evidence") or [],
            context_stats=resolved_context.get("context_stats") or {},
            evidence_resolution=resolution,
            decision_owner="rwkv_evidence_review_and_writer",
        )
        return resolved_context

    def _review_evidence_if_changed(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        *,
        step: int,
        trigger: str = "planner_finish",
        terminal: bool = False,
    ) -> dict[str, Any] | None:
        """Review one revision once, plus an explicit terminal handoff if needed."""

        context = self._resolved_writer_context(
            user_query,
            task_plan,
            step=step,
            trigger=trigger,
        )
        current_binding = self._evidence_review_binding(context)
        current_revision = int(current_binding["evidence_revision"])
        current_signature = str(current_binding["validation_state_signature"])
        reviewed_signature = str(
            self.state.run_metadata.get("last_evidence_review_state_signature")
            or ""
        )
        if not terminal and current_signature == reviewed_signature:
            return None
        reviewed_revision_value = self.state.run_metadata.get(
            "last_evidence_review_revision"
        )
        if not terminal and reviewed_revision_value is not None and current_revision <= int(
            reviewed_revision_value
        ):
            return None
        review = self._review_evidence(
            user_query,
            task_plan,
            step=step,
            trigger=trigger,
            terminal=terminal,
        )
        # Binding is controller transport metadata, never model-authored
        # semantics.  Overwrite rather than trust any similarly named field.
        review.update(current_binding)
        # Only a valid RWKV decision closes this immutable evidence revision.
        # A protocol error remains fully auditable in ``last_evidence_review``
        # but must not suppress the next terminal review attempt for the same
        # evidence.  Otherwise one malformed continuation bypasses the only
        # model-owned finish/replan gate and the Writer runs without a
        # successful review.
        if str(review.get("decision") or "").casefold() in {"finish", "replan"}:
            self.state.run_metadata["last_evidence_review_revision"] = current_revision
            self.state.run_metadata["last_evidence_review_state_signature"] = current_signature
            self.state.run_metadata["last_evidence_review"] = {
                "contract": str(review.get("contract") or EVIDENCE_REVIEW_CONTRACT),
                "decision": str(review.get("decision") or "")[:120],
                "selected_action": str(review.get("selected_action") or "")[:120],
                "review_mode": "terminal" if terminal else "binary",
                "trigger": str(trigger or "planner_finish")[:120],
                **current_binding,
            }
        return review

    def _current_evidence_review_decision(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        *,
        step: int,
        trigger: str,
    ) -> dict[str, Any] | None:
        """Return a cached valid decision only when its exact binding is current."""

        context = self._resolved_writer_context(
            user_query,
            task_plan,
            step=step,
            trigger=trigger,
        )
        current_binding = self._evidence_review_binding(context)
        cached = self.state.run_metadata.get("last_evidence_review")
        if not isinstance(cached, dict):
            return None
        if str(cached.get("decision") or "").casefold() not in {
            "finish",
            "replan",
        }:
            return None
        if not all(cached.get(key) == value for key, value in current_binding.items()):
            return None
        return {**cached, "cached_review": True}

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
        arguments: dict[str, Any] | None = None,
        phase: str,
        branch_id: str = "",
        task_record_id: str = "",
    ) -> dict[str, Any]:
        """Record retrieval state without changing the model-selected route."""

        delta = self._retrieval_ledger.record(
            query,
            result,
            step=step,
            branch_id=branch_id,
            task_record_id=task_record_id,
            action=action,
            arguments=arguments,
            phase=phase,
        )
        observation = self._retrieval_ledger.observation(
            branch_id=branch_id,
            task_record_id=task_record_id,
        )
        enriched = {**result, "retrieval_delta": delta, "retrieval_ledger": observation}
        append_task_event(
            self.state.task_id,
            "retrieval_ledger",
            step=step,
            phase=phase,
            branch_id=branch_id,
            task_record_id=task_record_id,
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

        merged = self._current_retrieval_data(user_query, action, rounds)
        evidence_ledger_snapshot = self.state.retrieval.evidence_ledger.snapshot()
        self.state.run_metadata["evidence_ledger"] = evidence_ledger_snapshot
        self._record_final_ranking(action, merged, step)

        synthesis = synthesize_retrieval_answer(
            user_query,
            merged,
            llm=self.llm,
            constraints=self.state.run_metadata,
            termination_reason=termination_reason,
            prebuilt_context=self._resolved_writer_context(
                user_query,
                self._task_plan,
                step=step,
                trigger="final_writer",
            ),
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
                "evidence_resolution": synthesis.get(
                    "evidence_resolution"
                )
                or {},
                "evidence_ledger": evidence_ledger_snapshot,
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
            evidence_resolution=synthesis.get(
                "evidence_resolution"
            )
            or {},
            evidence_ledger=evidence_ledger_snapshot,
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
            evidence_ledger=evidence_ledger_snapshot,
        )
        report_data = {
            **merged,
            "citation_refs": synthesis.get("citation_refs") or [],
            "selected_evidence": synthesis.get("selected_evidence") or [],
            "context_stats": synthesis.get("context_stats") or {},
            "context_text": synthesis.get("context_text", ""),
            "evidence_resolution": synthesis.get(
                "evidence_resolution"
            )
            or {},
        }
        self._write_agentic_report(
            user_query,
            action,
            answer,
            data=report_data,
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
        self.state.retrieval.evidence_ledger.initialize(task_plan, user_query)
        self.state.run_metadata["evidence_ledger"] = self.state.retrieval.evidence_ledger.snapshot()
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
            raise EvidenceReviewDecisionError(
                "Writer cannot run because Task Plan failed and Evidence Review "
                "was not executed: "
                + str(task_plan.get("message") or "task plan protocol error")[:1000]
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
            "record_count": len(task_records(task_plan)),
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
        self._writer_context_cache_signature = ""
        self._writer_context_cache = None
        self._evidence_resolution_cache_signature = ""
        self._evidence_resolution_cache = None
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
