# RWKV-ECRA/agent/orchestrator.py
import json
import os
from datetime import datetime

from agent.analyzer import Analyzer
from agent.planner import Planner
from agent.state import AgentState
from agent.slm_scheduler import GLOBAL_SLM_INPUT_SCHEDULER
from agent.retrieval_synthesis import synthesize_retrieval_answer
from agent.retrieval_loop import (
    merge_retrieval_results,
)
from utils.tracker import EventTracker
from config import (
    TRACKING,
    DATA_PIPELINE,
    get_citation_remote_validation,
    get_llm_base_url,
    get_llm_context_length,
    get_llm_model,
    get_llm_provider,
)
from utils.task_manager import update_task_progress
from utils.task_events import append_task_event
from utils.citation_validator import validate_citations
from utils.evidence_quality import substantive_evidence_items
from utils.risk_policy import validate_risk_answer
from utils.experiment_strategies import normalize_strategy
from utils.time_budget import check_time_budget
from tools.builtin import load_builtin_tools
from utils.retrieval_ledger import RetrievalLedger
from utils.answer_fact_check import check_answer_facts
from utils.freshness import build_freshness_policy
from agent.unified_research import run_unified_research_loop


DEFAULT_MAX_TOOL_STEPS = 50


class Orchestrator:
    def __init__(self):
        load_builtin_tools()
        self.tracker = EventTracker(log_dir=TRACKING.get("log_dir", "./logs"), enable=TRACKING.get("enable", True))
        self.state = AgentState()
        self.state.working_memory["__category_tree__"] = {} 
        self.analyzer = Analyzer()
        self.planner = Planner()
        self._task_plan: dict = {}
        self._retrieval_ledger = RetrievalLedger()
        self._calculation_results: list[dict] = []
        self._arithmetic_results: list[dict] = []
        self._time_results: list[dict] = []
        self._model_protocol_failure = False

    def _deterministic_results(self) -> list[dict]:
        """Return deterministic observations in execution order by type."""

        return [*self._calculation_results, *self._time_results, *self._arithmetic_results]

    def _retrieval_context(self) -> dict:
        return {
            "original_goal": self.state.user_query,
            "path_to_id": self.state.path_to_id,
            "id_to_path": self.state.id_to_path,
            "working_memory": self.state.working_memory,
            "tracker": self.tracker,
            # The task state is the single source of truth shared by the
            # planner, web tool, validators and final synthesizer.
            "agent_state": self.state,
            "task_id": self.state.task_id,
            # The task plan is system-owned routing context.  It lets the
            # generic web tool enforce source policy without exposing provider
            # selection to RWKV or allowing model arguments to rewrite it.
            "task_plan": self._task_plan,
            "slm_scheduler": GLOBAL_SLM_INPUT_SCHEDULER,
        }

    def _agentic_tool_context(self) -> dict:
        """Build the context passed to a model-selected tool call."""
        context = self._retrieval_context()
        context["agentic_tool_loop"] = True
        context["retrieval_ledger"] = self._retrieval_ledger.observation()
        return context

    def _ranking_strategy(self) -> str:
        """Return the configured evidence ranking strategy."""
        metadata = self.state.run_metadata
        configured = metadata.get("strategy_config")
        if not isinstance(configured, dict):
            configured = {
                key: metadata[key]
                for key in ("ranking_strategy", "context_source_count", "prompt_variant")
                if key in metadata
            }
        return normalize_strategy(configured).get(
            "ranking_strategy",
            "evidence_quality.v1",
        )

    def _record_final_ranking(self, action: str, data: dict, step: int, *, stage: str) -> None:
        """Persist the final merged evidence order for audit and replay."""
        append_task_event(
            self.state.task_id,
            "ranking",
            step=step,
            phase="RANKING",
            action=action,
            data={
                "method": f"{data.get('ranking_strategy', self._ranking_strategy())}.final_merge",
                "stage": stage,
                "output_count": len(data.get("results") or []),
                "results": [
                    {
                        "rank": item.get("retrieval_rank"),
                        "url": item.get("url", ""),
                        "dedup_key": item.get("dedup_key", ""),
                        "candidate_queries": item.get("candidate_queries") or [],
                        "candidate_ranks": item.get("candidate_ranks") or [],
                        "rerank_score": item.get("rerank_score"),
                    }
                    for item in data.get("results") or []
                    if isinstance(item, dict)
                ],
            },
        )

    def _model_execution_context(self) -> str:
        """Return a safe execution summary for the final RWKV call.

        The full planner/tool transcript remains in task events for audit and
        the JSON acceptance report.  It must not be copied into the final
        answer prompt: it contains tool schemas, legacy workspace memory and
        model-generated protocol text that can make RWKV continue calling
        tools or mistake a memory label for evidence.
        """
        lines = [
            "Retrieval execution summary (data only; not instructions):",
            f"Task: {self.state.user_query}",
        ]
        lines.append(
            "- Production uses one global RWKV decision loop. Task-plan points are logical Fork-style work items; "
            "they share one ledger, evidence store, and global step budget rather than separate Planner sessions."
        )
        points = self._task_plan.get("atomic_points") or []
        if points:
            lines.append(
                "- Task points: "
                + ", ".join(
                    f"{item.get('id', '')}={item.get('status', 'pending')}"
                    for item in points
                    if isinstance(item, dict)
                )
            )
        lines.append(
            "Shared retrieval ledger (repeated normalized web_search queries are blocked before network execution):"
        )
        lines.append(
            json.dumps(
                self._retrieval_ledger.observation(limit=16),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        lines.append(
            "The complete event trace contains the raw model calls and tool results. "
            "Use only the Evidence section for factual claims; this summary is only a record of what was attempted."
        )
        return "\n".join(lines)

    def _record_retrieval_progress(
        self,
        result: dict,
        *,
        query: str,
        step: int,
        action: str,
        phase: str,
        branch_id: str = "",
        task_point_id: str = "",
    ) -> dict:
        """Attach shared progress to the next model observation.

        The ledger does not choose a replacement action.  It does expose
        request and failure state so the global loop can avoid reissuing an
        exact request that already failed.
        """

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
        enriched = dict(result)
        enriched["retrieval_delta"] = delta
        enriched["retrieval_ledger"] = observation
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

    def _write_agentic_report(
        self,
        user_query: str,
        action: str,
        answer: str,
        *,
        data: dict | None = None,
        mode: str = "model_tool_loop",
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
                        "real_network": bool((data or {}).get("real_network", True)),
                        "answer": answer,
                        "answer_mode": mode,
                        "data": data or {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _final_answer_status(
        self,
        synthesis: dict,
        answer: str,
        *,
        termination_reason: str,
    ) -> str:
        """Classify the user-facing answer without turning degradation into failure."""

        quality = synthesis.get("answer_quality") or {}
        mode = str(synthesis.get("mode") or "")
        if not str(answer or "").strip():
            # This is a controller invariant violation, but the public contract
            # still remains answer-shaped for the acceptance runner.
            return "completed_refusal"
        if mode == "controller_refusal" or quality.get("fallback_kind") == "refusal":
            return "completed_refusal"
        if (
            mode == "controller_fallback"
            or quality.get("fallback_used")
            or termination_reason in {
                "max_steps_reached",
                "model_tool_decision_parse_error",
                "replan_limit_reached",
                "replan_no_alternative",
                "research_budget_reached",
            }
            or self._model_protocol_failure
            or mode in {"local_rwkv_error", "local_rwkv_empty", "rwkv_unavailable"}
        ):
            return "completed_partial"
        return "completed"


    def _complete_without_evidence(
        self,
        user_query: str,
        action: str,
        step: int,
        *,
        termination_reason: str,
    ) -> str:
        """Force a visible final model answer even when no page was usable."""
        execution_context = self._model_execution_context()
        data = {
            "query": user_query,
            "status": "no_evidence",
            "results": [],
            "citation_refs": [],
            "sources": [],
            "calculation_results": self._deterministic_results(),
        }
        synthesis = synthesize_retrieval_answer(
            user_query,
            data,
            llm=self.analyzer.llm,
            constraints=self.state.run_metadata,
            execution_context=execution_context,
            termination_reason=termination_reason,
        )
        answer = str(synthesis.get("content") or "RWKV did not return a final summary.")
        answer_fact_check = check_answer_facts(
            answer,
            evidence=[],
            calculation_results=self._deterministic_results(),
            freshness_policy=self.state.run_metadata.get("freshness_policy") or {},
        )
        append_task_event(
            self.state.task_id,
            "answer_fact_validation",
            step=step,
            phase="VALIDATION",
            data=answer_fact_check,
        )
        append_task_event(
            self.state.task_id,
            "context_build",
            step=step,
            phase="CONTEXT",
            data={
                "context_text": synthesis.get("context_text", ""),
                "selected_evidence": synthesis.get("selected_evidence") or [],
                "context_stats": synthesis.get("context_stats") or {},
                "execution_context": execution_context,
            },
        )
        append_task_event(
            self.state.task_id,
            "synthesis",
            step=step,
            phase="SYNTHESIS",
            action=action,
            content=answer,
            mode=synthesis.get("mode") or "local_rwkv_final",
                evidence_count=0,
                citation_refs=synthesis.get("citation_refs") or [],
                validation=synthesis.get("validation") or {},
                answer_alignment=synthesis.get("answer_alignment") or {},
                answer_quality=synthesis.get("answer_quality") or {},
                answer_fact_check=answer_fact_check,
                prompt=synthesis.get("prompt", ""),
            model_output=synthesis.get("model_output", ""),
            context_text=synthesis.get("context_text", ""),
            selected_evidence=synthesis.get("selected_evidence") or [],
            context_stats=synthesis.get("context_stats") or {},
            termination_reason=termination_reason,
        )
        self.state.final_result = answer
        self.state.is_finished = True
        model_output_available = bool(str(synthesis.get("model_output") or "").strip())
        final_status = self._final_answer_status(
            synthesis,
            answer,
            termination_reason=termination_reason,
        )
        append_task_event(
            self.state.task_id,
            "final",
            status=final_status,
            content=answer,
            action=action,
            mode=synthesis.get("mode") or "local_rwkv_final",
            termination_reason=termination_reason,
            model_output_available=model_output_available,
            model_protocol_failure=self._model_protocol_failure,
            answer_quality=synthesis.get("answer_quality") or {},
            answer_fact_check=answer_fact_check,
        )
        self._write_agentic_report(
            user_query,
            action,
            answer,
            data={**data, "answer_fact_check": answer_fact_check},
            mode=synthesis.get("mode") or "local_rwkv_final",
        )
        return answer

    def _complete_model_tool_loop(
        self,
        user_query: str,
        action: str,
        rounds: list[tuple[str, dict]],
        step: int,
        *,
        termination_reason: str = "model_requested_finish",
    ) -> str:
        """Generate the final answer for a finish decision or step limit."""
        if termination_reason == "max_steps_reached":
            append_task_event(
                self.state.task_id,
                "step_limit_reached",
                step=step,
                phase="SYNTHESIS",
                action=action,
                max_steps=step,
                evidence_rounds=len(rounds),
                message="The retrieval loop reached max_tool_steps; force a final RWKV summary.",
            )
        if not rounds:
            return self._complete_without_evidence(
                user_query,
                action,
                step,
                termination_reason=termination_reason,
            )
        execution_context = self._model_execution_context()
        if rounds:
            merged = merge_retrieval_results(
                user_query,
                action,
                rounds,
                ranking_strategy=self._ranking_strategy(),
            )
            if self._calculation_results or self._time_results:
                merged = dict(merged)
                merged["calculation_results"] = self._deterministic_results()
            self._record_final_ranking(action, merged, step, stage="model_tool_loop_merge")
            synthesis = synthesize_retrieval_answer(
                user_query,
                merged,
                llm=self.analyzer.llm,
                constraints=self.state.run_metadata,
                execution_context=execution_context,
                termination_reason=termination_reason,
            )
            answer = synthesis.get("content") or ""
            citation_validation = validate_citations(
                synthesis.get("citation_refs") or [],
                answer=answer,
                evidence=substantive_evidence_items(merged.get("results") or []),
                check_remote=get_citation_remote_validation(),
            )
            answer_fact_check = check_answer_facts(
                answer,
                evidence=substantive_evidence_items(merged.get("results") or []),
                calculation_results=merged.get("calculation_results") or self._deterministic_results(),
                freshness_policy=self.state.run_metadata.get("freshness_policy") or merged.get("freshness_policy") or {},
            )
            risk_validation = validate_risk_answer(answer, self.state.run_metadata)
            append_task_event(
                self.state.task_id,
                "context_build",
                step=step,
                phase="CONTEXT",
                data={
                    "context_text": synthesis.get("context_text", ""),
                    "selected_evidence": synthesis.get("selected_evidence") or [],
                    "context_stats": synthesis.get("context_stats") or {},
                },
            )
            append_task_event(
                self.state.task_id,
                "evidence_validation",
                step=step,
                phase="VALIDATION",
                data={
                    "validation": synthesis.get("validation") or {},
                    "answer_alignment": synthesis.get("answer_alignment") or {},
                },
            )
            append_task_event(
                self.state.task_id,
                "citation_validation",
                step=step,
                phase="VALIDATION",
                data=citation_validation,
            )
            append_task_event(
                self.state.task_id,
                "risk_validation",
                step=step,
                phase="VALIDATION",
                data=risk_validation,
            )
            append_task_event(
                self.state.task_id,
                "answer_fact_validation",
                step=step,
                phase="VALIDATION",
                data=answer_fact_check,
            )
            append_task_event(
                self.state.task_id,
                "synthesis",
                step=step,
                phase="SYNTHESIS",
                action=action,
                content=answer,
                mode=synthesis.get("mode"),
                evidence_count=synthesis.get("evidence_count", 0),
                citation_refs=synthesis.get("citation_refs") or [],
                citation_validation=citation_validation,
                risk_validation=risk_validation,
                answer_fact_check=answer_fact_check,
                validation=synthesis.get("validation") or {},
                answer_alignment=synthesis.get("answer_alignment") or {},
                answer_quality=synthesis.get("answer_quality") or {},
                prompt=synthesis.get("prompt", ""),
                model_output=synthesis.get("model_output", ""),
                repair_prompt=synthesis.get("repair_prompt", ""),
                repair_output=synthesis.get("repair_output", ""),
                context_text=synthesis.get("context_text", ""),
                selected_evidence=synthesis.get("selected_evidence") or [],
                context_stats=synthesis.get("context_stats") or {},
            )
            self.state.final_result = answer
            self.state.is_finished = True
            model_output_available = bool(str(synthesis.get("model_output") or "").strip())
            final_status = self._final_answer_status(
                synthesis,
                answer,
                termination_reason=termination_reason,
            )
            append_task_event(
                self.state.task_id,
                "final",
                status=final_status,
                content=answer,
                action=action,
                mode=synthesis.get("mode"),
                round_count=merged.get("round_count", len(rounds)),
                citation_refs=synthesis.get("citation_refs") or [],
                citation_validation=citation_validation,
                risk_validation=risk_validation,
                answer_fact_check=answer_fact_check,
                validation=synthesis.get("validation") or {},
                answer_alignment=synthesis.get("answer_alignment") or {},
                answer_quality=synthesis.get("answer_quality") or {},
                termination_reason=termination_reason,
                model_output_available=model_output_available,
                model_protocol_failure=self._model_protocol_failure,
            )
            self._write_agentic_report(
                user_query,
                action,
                answer,
                data={**merged, "answer_fact_check": answer_fact_check},
                mode=synthesis.get("mode", "model_tool_loop"),
            )
            return answer

    def _prepare_task_plan(self, user_query: str, phase: str) -> dict:
        """Create the model plan once and initialize the shared transcript."""
        self.planner.reset()
        # Planning starts from the user goal only. Workspace contents remain
        # available to later tool turns, but cannot bias decomposition into a
        # local-file plan.
        plan_context = ""
        task_plan = self.planner.create_task_plan(user_query, plan_context)
        if task_plan.get("status") == "error":
            append_task_event(
                self.state.task_id,
                "task_plan",
                step=0,
                phase="ROUTING",
                data=task_plan,
            )
            return task_plan

        freshness_policy = build_freshness_policy(user_query, task_plan)
        task_plan = {**task_plan, "freshness_policy": freshness_policy}
        append_task_event(
            self.state.task_id,
            "task_plan",
            step=0,
            phase="ROUTING",
            data=task_plan,
        )
        self._task_plan = task_plan
        self.state.run_metadata["task_plan"] = task_plan
        self.state.run_metadata["freshness_policy"] = freshness_policy
        generic_web_mode = bool(self.state.run_metadata.get("generic_web_search_only"))
        # Keep the legacy call shape for existing planner test doubles. The
        # generic web mode is the only path that needs an explicit phase.
        if generic_web_mode:
            self.planner.begin_task(user_query, plan_context, task_plan, phase)
        else:
            self.planner.begin_task(user_query, plan_context, task_plan)
        return task_plan

    def _fail_task_plan(self, user_query: str, task_plan: dict) -> str:
        answer = (
            "I could not establish a reliable retrieval plan for this question, "
            "so I cannot confirm an evidence-based answer."
        )
        self.state.final_result = answer
        self.state.is_finished = True
        append_task_event(
            self.state.task_id,
            "final",
            status="completed_refusal",
            content=answer,
            action="task_plan",
            mode="controller_refusal",
            error_class=task_plan.get("error_class", "task_plan_invalid"),
            planner_error=task_plan.get("message", ""),
            answer_quality={
                "fallback_used": True,
                "fallback_kind": "refusal",
                "fallback_reason": "task_plan_failed",
                "model_error_recorded": bool(task_plan.get("message")),
            },
        )
        self._write_agentic_report(user_query, "task_plan", answer, mode="controller_refusal")
        return answer

    def _run_model_tool_loop(self, user_query: str, model_profile: dict) -> str:
        """Run the one global model-owned retrieval loop.

        The task plan is a shared checklist (the Fork idea), not a request to
        create independent planner conversations.  All task points, queries,
        evidence, validation, and the step budget stay in one state object.
        """
        generic_web_mode = bool(self.state.run_metadata.get("generic_web_search_only"))
        max_steps = max(
            1,
            int(
                self.state.run_metadata.get(
                    "max_tool_steps",
                    DEFAULT_MAX_TOOL_STEPS,
                )
                or DEFAULT_MAX_TOOL_STEPS
            ),
        )
        phase = "GENERIC_WEB" if generic_web_mode else "DISCOVERY"
        task_plan = self._prepare_task_plan(user_query, phase)
        if task_plan.get("status") == "error":
            return self._fail_task_plan(user_query, task_plan)

        point_ids = [
            str(point.get("id") or "")
            for point in task_plan.get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        decision = {
            "strategy": "single_loop",
            "point_count": len(point_ids),
            "point_ids": point_ids,
            "source": "global_shared_state",
            "reason": (
                "task points are logical Fork-style work items inside one "
                "shared planner, retrieval ledger, evidence store, and step budget"
            ),
        }
        self.state.run_metadata["retrieval_strategy"] = "single_loop"
        self.state.run_metadata["retrieval_strategy_decision"] = decision
        append_task_event(
            self.state.task_id,
            "retrieval_strategy_selected",
            step=0,
            phase="ROUTING",
            **decision,
            override=False,
        )
        return self._run_single_loop(
            user_query,
            model_profile,
            task_plan,
            max_steps,
        )

    def run(
        self,
        user_query: str,
        task_id: str = None,
        run_metadata: dict | None = None,
    ) -> str:
        check_time_budget(minimum_seconds=0.2)
        self.state.task_id = task_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state.run_metadata = dict(run_metadata or {})
        self.state.retrieval.reset()
        self._retrieval_ledger = RetrievalLedger()
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
        check_time_budget(minimum_seconds=0.2)
        try:
            model_profile = {
                "model": get_llm_model(),
                "endpoint": get_llm_base_url(),
                "provider": get_llm_provider(),
                "context_length": get_llm_context_length(),
            }
        except AttributeError:
            model_profile = {}
        append_task_event(
            self.state.task_id,
            "run_started",
            phase="ROUTING",
            experiment=run_metadata or {},
            config={
                "model": model_profile,
                "retrieval": {"strategy": "global_shared_state"},
            },
            prompt_version=(run_metadata or {}).get("prompt_version", "unversioned"),
            run_metadata=run_metadata or {},
        )
        debug_dir = DATA_PIPELINE.get("debug_directory", "./data/debug_slm")
        os.makedirs(debug_dir, exist_ok=True)
        session_id = self.state.task_id
        trace_file = os.path.join(debug_dir, f"DeepResearch_Trace_{session_id}.md")
        with open(trace_file, "w", encoding="utf-8") as handle:
            handle.write(
                f"# Deep Research 执行追踪日志\n\n"
                f"**启动时间**: {session_id}\n"
                f"**用户指令**: {user_query}\n\n---\n\n"
            )

        step_count = 0
        progress_log: list[str] = []

        def push_progress(message: str) -> None:
            progress_log.append(message)
            update_task_progress(self.state.task_id, "\n".join(progress_log))
            append_task_event(
                self.state.task_id,
                "progress",
                step=step_count,
                message=message,
            )

        push_progress("🚀 已启用联网检索，等待 RWKV 自主选择工具...")
        return self._run_model_tool_loop(user_query, model_profile)

    def _run_single_loop(
        self,
        user_query: str,
        model_profile: dict,
        task_plan: dict,
        max_steps: int,
    ) -> str:
        """Run one continuous RWKV retrieval context with shared state.

        Task-plan points remain a semantic checklist, not independent
        conversations.  Retrieval backends and chunk extraction may still
        use bounded concurrency; the model-facing decision state is global.
        """
        # Compatibility entry point for older callers. The runtime now uses
        # one unified research loop; the old phase/recovery implementation is
        # intentionally bypassed so it cannot create a second evidence path.
        return run_unified_research_loop(
            self,
            user_query,
            model_profile,
            task_plan,
            max_steps,
        )
