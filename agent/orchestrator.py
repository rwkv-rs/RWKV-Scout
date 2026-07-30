# RWKV-ECRA/agent/orchestrator.py
import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from agent.analyzer import Analyzer
from agent.planner import Planner
from agent.state import AgentState
from agent.slm_scheduler import GLOBAL_SLM_INPUT_SCHEDULER
from agent.retrieval_synthesis import build_evidence_context, synthesize_retrieval_answer
from agent.evidence_verifier import verify_evidence
from agent.page_evidence import extract_single_page_evidence
from agent.retrieval_loop import (
    merge_retrieval_results,
)
from utils.tracker import EventTracker
from config import (
    TRACKING,
    DATA_PIPELINE,
    get_citation_remote_validation,
    get_llm_base_url,
    get_llm_concurrency,
    get_llm_context_length,
    get_llm_model,
    get_llm_provider,
)
from tools.registry import ToolRegistry
from utils.task_manager import is_task_stopped, update_task_progress
from utils.task_events import append_task_event
from utils.citation_validator import validate_citations
from utils.evidence_quality import MIN_PAGE_BODY_CHARS, has_substantive_evidence, substantive_evidence_items
from utils.risk_policy import risk_context, validate_risk_answer
from utils.experiment_strategies import normalize_strategy
from utils.time_budget import check_time_budget
from tools.builtin import load_builtin_tools
from retrieval_plugins import is_error, plugin_environment_snapshot
from utils.retrieval_ledger import RetrievalLedger
from agent.execution_strategy import StrategyDecision, select_strategy
from agent.retrieval_runners import build_runner
from agent.controlled_retrieval import ControlledRetrievalMixin


DEFAULT_MAX_TOOL_STEPS = 100
DEFAULT_MAX_REPLAN_ATTEMPTS = 3


class Orchestrator(ControlledRetrievalMixin):
    def __init__(self):
        load_builtin_tools()
        self.tracker = EventTracker(log_dir=TRACKING.get("log_dir", "./logs"), enable=TRACKING.get("enable", True))
        self.state = AgentState()
        self.state.working_memory["__category_tree__"] = {} 
        self.analyzer = Analyzer()
        self.planner = Planner()
        self._task_plan: dict = {}
        self._fork_transcripts: list[dict] = []
        self._retrieval_ledger = RetrievalLedger()
        self._last_evidence_verification: dict = {}

    def _retrieval_context(self) -> dict:
        return {
            "original_goal": self.state.user_query,
            "path_to_id": self.state.path_to_id,
            "id_to_path": self.state.id_to_path,
            "working_memory": self.state.working_memory,
            "tracker": self.tracker,
            "agent_state": None,
            "task_id": self.state.task_id,
            "slm_scheduler": GLOBAL_SLM_INPUT_SCHEDULER,
        }

    def _agentic_tool_context(self) -> dict:
        """Build the context passed to a model-selected tool call."""
        context = self._retrieval_context()
        context["agentic_tool_loop"] = True
        context["retrieval_ledger"] = self._retrieval_ledger.observation()
        return context

    def _validation_architecture(self) -> str:
        """Return the explicitly selected evidence-validation architecture."""
        value = str(self.state.run_metadata.get("validation_architecture") or "rwkv_verifier").strip().casefold()
        if value in {"engineering", "engineering_validator", "rules", "deterministic"}:
            return "engineering_validator"
        if value in {"rwkv", "rwkv_verifier", "model", "model_verifier"}:
            return "rwkv_verifier"
        return "rwkv_verifier"

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
        lines.append("- One global RWKV decision loop owns all task points and retrieval state.")
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

    def _process_single_page_result(
        self,
        evidence_query: str,
        data: dict,
        step: int,
        evidence_action: str = "fetch_web_url",
    ) -> dict:
        """Map one model-selected page into compact chunk candidates.

        ``data`` may contain the full fetched page because the fetch tool must
        remain auditable.  That raw body is deliberately not returned to the
        planner.  Only the per-chunk candidates and their URL provenance are
        allowed back into the next model turn.
        """

        pages = [item for item in data.get("results") or [] if isinstance(item, dict)]
        if len(pages) != 1:
            compact = dict(data)
            compact["results"] = []
            compact["citation_refs"] = []
            compact["page_evidence"] = {
                "status": "error",
                "message": "the evidence tool must return exactly one page",
            }
            return compact

        page = pages[0]
        url = str(page.get("url") or "")

        def record_chunk(*, chunk, prompt, candidate, task_id):
            append_task_event(
                self.state.task_id,
                "page_chunk",
                step=step,
                phase="EXTRACTION",
                action=evidence_action,
                url=url,
                chunk_id=chunk.get("chunk_id", ""),
                chunk_index=chunk.get("index", 0),
                chunk_chars=len(str(chunk.get("text") or "")),
                chunk_tokens=chunk.get("token_count", 0),
                chunk_text=str(chunk.get("text") or ""),
                evidence_query=evidence_query,
            )
            append_task_event(
                self.state.task_id,
                "page_chunk_candidate",
                step=step,
                phase="EXTRACTION",
                action=evidence_action,
                url=url,
                chunk_id=chunk.get("chunk_id", ""),
                prompt=prompt,
                prompt_chars=len(prompt),
                evidence_query=evidence_query,
                model_output=candidate.get("raw_output", ""),
                model_output_chars=len(str(candidate.get("raw_output") or "")),
                finish_reason=candidate.get("finish_reason", ""),
                retry_count=candidate.get("retry_count", 0),
                candidate=candidate,
            )

        try:
            evidence = extract_single_page_evidence(
                query=evidence_query,
                page=page,
                llm=self.analyzer.llm,
                task_id=self.state.task_id,
                on_chunk=record_chunk,
            )
        except Exception as exc:
            evidence = {
                "status": "error",
                "url": url,
                "title": str(page.get("title") or url),
                "page_chars": len(str(page.get("page_excerpt") or page.get("content") or "")),
                "chunk_count": 0,
                "candidates": [],
                "compact_facts": "",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }

        # Keep source text, model locators and request metadata in separate
        # fields.  Previously these were concatenated into ``content``;
        # titles, API metadata and model-generated facts could then pass as
        # page evidence during final synthesis.
        source_body = str(page.get("page_excerpt") or page.get("content") or "").strip()
        source_excerpt = source_body[:14000]
        model_extracted_facts = str(evidence.get("compact_facts") or "").strip()[:14000]
        source_metadata = {
            key: page.get(key)
            for key in ("api_url", "request_params", "source", "project", "language")
            if page.get(key)
        }
        body_verified = len(source_excerpt) >= MIN_PAGE_BODY_CHARS
        chunk_candidates = evidence.get("candidates") or []
        merged_page = {
            "title": page.get("title") or url,
            "url": url,
            "snippet": str(page.get("snippet") or source_excerpt[:600]),
            "page_excerpt": source_excerpt,
            "source_excerpt": source_excerpt,
            "content": source_excerpt,
            "model_extracted_facts": model_extracted_facts,
            "structured_metadata": source_metadata,
            "source": page.get("source") or "explicit model-selected URL",
            "content_type": page.get("content_type", ""),
            "untrusted_content": True,
            "evidence_origin": "fetched_page_body",
            "evidence_boundary": "page_body_only",
            "body_verified": body_verified,
            "content_sha256": hashlib.sha256(source_excerpt.encode("utf-8")).hexdigest() if source_excerpt else "",
            "source_locator": {
                "type": "page_excerpt",
                "char_start": 0,
                "char_end": len(source_excerpt),
            },
            "chunk_count": evidence.get("chunk_count", 0),
            "chunk_candidates": chunk_candidates,
            "evidence_status": "ok" if body_verified else "no_evidence",
            "model_extraction_status": "ok" if model_extracted_facts else "no_evidence",
        }
        compact = dict(data)
        compact["results"] = [merged_page] if has_substantive_evidence(merged_page) else []
        compact["sources"] = [url] if url else []
        source_refs = [item for item in data.get("citation_refs") or [] if isinstance(item, dict)]
        compact["citation_refs"] = [
            {
                "ref_id": (source_refs[0].get("ref_id") if source_refs else "") or f"WEB_FETCH_{step}",
                "title": merged_page["title"],
                "url": url,
                "source": merged_page["source"],
                "evidence_text": source_excerpt,
                "evidence_origin": "fetched_page_body",
                "evidence_boundary": "page_body_only",
                "source_locator": merged_page["source_locator"],
            }
        ] if has_substantive_evidence(merged_page) else []
        compact["page_evidence"] = {
            "status": evidence.get("status", "no_evidence"),
            "url": url,
            "title": merged_page["title"],
            "page_chars": evidence.get("page_chars", 0),
            "chunk_count": evidence.get("chunk_count", 0),
            "chunk_window_tokens": evidence.get("chunk_window_tokens", 0),
            "candidate_count": len(evidence.get("candidates") or []),
            "parallel_candidate": evidence.get("parallel_candidate") or {},
            "errors": evidence.get("errors") or [],
        }
        append_task_event(
            self.state.task_id,
            "page_candidate_merge",
            step=step,
            phase="EXTRACTION",
            action=evidence_action,
            url=url,
            data=compact["page_evidence"],
            candidates=evidence.get("candidates") or [],
            source_excerpt=source_excerpt,
            model_extracted_facts=model_extracted_facts,
            structured_metadata=source_metadata,
        )
        return compact

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

    def _replan_after_retrieval_failure(
        self,
        user_query: str,
        observation: dict,
        step: int,
        *,
        attempt: int = 0,
        max_attempts: int = 0,
        validation: dict | None = None,
    ) -> bool:
        """Ask RWKV to split the remaining work after retrieval is insufficient.

        The controller reports execution facts and, when available, the
        independent verifier's missing-point report. It never creates a
        replacement query or URL itself.
        """
        if not self._task_plan:
            return False
        if validation:
            retrieval_observation = {
                "schema_version": "evidence_verification.v1",
                "status": "error",
                "error_class": "evidence_verification_incomplete",
                "message": "The independent evidence verifier found missing or conflicting task-point evidence.",
                "missing_point_ids": validation.get("missing_point_ids") or [],
                "conflict_point_ids": validation.get("conflict_point_ids") or [],
                "next_queries": validation.get("next_queries") or [],
                "verification_points": validation.get("points") or [],
            }
        else:
            retrieval_observation = {
                "schema_version": "retrieval.v1",
                "status": "error",
                "error_class": str(observation.get("error_class") or "no_usable_evidence"),
                "message": "The selected retrieval observation did not yield usable evidence.",
            }
        followup_plan = self.planner.replan_task(
            user_query,
            self._task_plan,
            retrieval_observation,
            json.dumps(observation, ensure_ascii=False)[:6000],
        )
        append_task_event(
            self.state.task_id,
            "task_replan",
            step=step,
            phase="RECOVERY",
            data=followup_plan,
            previous_observation={
                "status": observation.get("status"),
                "error_class": observation.get("error_class"),
                "page_evidence": observation.get("page_evidence"),
                "evidence_validation": retrieval_observation if validation else {},
            },
            replan_attempt=attempt,
            max_replan_attempts=max_attempts,
            counts_toward_global_steps=False,
        )
        if followup_plan.get("status") == "error":
            self.planner.observe_tool_result(followup_plan)
            return False
        self._task_plan = followup_plan
        self.state.run_metadata["task_plan"] = followup_plan
        self.planner.update_task_plan(followup_plan)
        return True

    def _verify_retrieval_completion(
        self,
        user_query: str,
        action: str,
        rounds: list[tuple[str, dict]],
        step: int,
    ) -> tuple[dict, dict]:
        """Audit merged evidence with a fresh RWKV call before synthesis."""
        merged = merge_retrieval_results(
            user_query,
            action,
            rounds,
            ranking_strategy=self._strategy()["ranking_strategy"],
        )
        context = build_evidence_context(
            merged,
            constraints=self.state.run_metadata,
            query=user_query,
        )
        verification = verify_evidence(
            self.analyzer.llm,
            query=user_query,
            task_plan=self._task_plan,
            evidence_context=context,
        )
        self._last_evidence_verification = {
            key: verification.get(key)
            for key in (
                "schema_version",
                "status",
                "completion_ready",
                "requires_replan",
                "points",
                "missing_point_ids",
                "conflict_point_ids",
                "next_queries",
                "is_truth_judgement",
            )
        }
        append_task_event(
            self.state.task_id,
            "evidence_verification",
            step=step,
            phase="VALIDATION",
            action="evidence_verifier",
            status=verification.get("status"),
            completion_ready=verification.get("completion_ready"),
            requires_replan=verification.get("requires_replan"),
            missing_point_ids=verification.get("missing_point_ids") or [],
            conflict_point_ids=verification.get("conflict_point_ids") or [],
            next_queries=verification.get("next_queries") or [],
            prompt=verification.get("prompt", ""),
            model_output=verification.get("model_output", ""),
            error=verification.get("error", ""),
            data=verification,
        )
        return merged, verification

    def _evidence_query_for_point(self, task_point_id: str, fallback: str) -> str:
        """Project the model-selected atomic point into the evidence prompt."""
        point_id = str(task_point_id or "").strip()
        for point in self._task_plan.get("atomic_points") or []:
            if isinstance(point, dict) and str(point.get("id") or "").strip() == point_id:
                objective = str(point.get("objective") or "").strip()
                task = str(point.get("task") or "").strip()
                needed = point.get("evidence_needed") or []
                acceptance = point.get("acceptance_criteria") or []
                output_format = str(point.get("output_format") or "prose").strip()
                if objective:
                    needed_text = "; ".join(str(item).strip() for item in needed if str(item).strip())
                    criteria_text = "; ".join(str(item).strip() for item in acceptance if str(item).strip())
                    return (
                        f"Task: {task or objective}\n"
                        f"Atomic objective: {objective}\n"
                        f"Output format: {output_format}\n"
                        f"Evidence needed: {needed_text}\n"
                        f"Acceptance criteria: {criteria_text}"
                    )
        # A missing point id is a model protocol omission, not permission to
        # inject the entire checklist into every page chunk.  The checklist
        # remains available to planning and judging; page extraction should
        # receive only the original question so it reads the selected page.
        return fallback

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
                evidence_verification=synthesis.get("evidence_verification") or {},
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
        final_status = (
            "failed"
            if termination_reason == "max_steps_reached" or not model_output_available
            else "completed"
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
        )
        self._write_agentic_report(
            user_query,
            action,
            answer,
            data=data,
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
        # Every terminal path must pass through the selected evidence
        # architecture before synthesis.  The ordinary ``finish_task`` path
        # performs verification earlier so it can replan; duplicate-query,
        # parse-error, and max-step paths can arrive here without that earlier
        # hook.  Keep this as a one-call fallback rather than allowing those
        # paths to silently skip the model verifier.
        if (
            self._validation_architecture() == "rwkv_verifier"
            and not self._last_evidence_verification
        ):
            self._verify_retrieval_completion(
                user_query,
                action,
                rounds,
                step,
            )
        execution_context = self._model_execution_context()
        if rounds:
            merged = merge_retrieval_results(
                user_query,
                action,
                rounds,
                ranking_strategy=self._strategy()["ranking_strategy"],
            )
            self._record_final_ranking(action, merged, step, stage="model_tool_loop_merge")
            synthesis = synthesize_retrieval_answer(
                user_query,
                merged,
                llm=self.analyzer.llm,
                constraints=self.state.run_metadata,
                execution_context=execution_context,
                termination_reason=termination_reason,
                verification=self._last_evidence_verification,
            )
            answer = synthesis.get("content") or ""
            citation_validation = validate_citations(
                synthesis.get("citation_refs") or [],
                answer=answer,
                evidence=substantive_evidence_items(merged.get("results") or []),
                check_remote=get_citation_remote_validation(),
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
                validation=synthesis.get("validation") or {},
                answer_alignment=synthesis.get("answer_alignment") or {},
                evidence_verification=synthesis.get("evidence_verification") or {},
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
            force_output = termination_reason == "max_steps_reached"
            final_status = (
                "failed"
                if force_output
                or not model_output_available
                or synthesis.get("mode") in {"local_rwkv_error", "local_rwkv_empty", "rwkv_unavailable"}
                else "completed"
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
                validation=synthesis.get("validation") or {},
                answer_alignment=synthesis.get("answer_alignment") or {},
                termination_reason=termination_reason,
                model_output_available=model_output_available,
            )
            self._write_agentic_report(user_query, action, answer, data=merged, mode=synthesis.get("mode", "model_tool_loop"))
            return answer

    def _run_fork_branch_step(
        self,
        user_query: str,
        branch: dict,
        model_profile: dict,
        step: int,
        branch_phase: str,
        generic_web_mode: bool,
    ) -> tuple[int, tuple[str, dict] | None]:
        """Execute one model-owned step for one fork.

        A branch owns its planner and mutable counters.  The outer Fork loop
        may therefore submit several of these steps concurrently while the
        shared ledger and event stream remain the only synchronized state.
        """
        branch_id = str(branch["branch_id"])
        point = branch["point"]
        branch_planner = branch["planner"]
        branch_step = int(branch.get("branch_step", 0)) + 1
        branch["branch_step"] = branch_step
        task_point_id = str(point.get("id") or "")
        plan = branch_planner.plan_next_action(user_query, {}, "", branch_phase)
        action = str(plan.get("action") or "").strip()
        args = dict(plan.get("args") or {}) if isinstance(plan.get("args"), dict) else {}
        branch["tools"].append(action or "(empty)")
        append_task_event(
            self.state.task_id,
            "model_tool_decision",
            step=step,
            phase="FORK",
            branch_id=branch_id,
            branch_step=branch_step,
            action=action,
            args=args,
            task_point_id=task_point_id,
            router=plan.get("router", "model_tool_decision"),
            raw_model_output=plan.get("raw_model_output", ""),
            planner_error=plan.get("planner_error", ""),
            model=model_profile,
        )
        if plan.get("planner_error"):
            branch["stop_reason"] = "planner_error"
            branch["active"] = False
            branch_planner.observe_tool_result(
                {
                    "schema_version": "retrieval.v1",
                    "status": "error",
                    "error_class": "model_tool_decision",
                    "message": plan.get("planner_error", ""),
                    "results": [],
                }
            )
            return step, None

        if action in {"answer_user", "finish_task"}:
            branch["stop_reason"] = "model_requested_finish"
            branch["active"] = False
            return step, None

        if not ToolRegistry.can_execute(action, branch_phase):
            observation = {
                "schema_version": "retrieval.v1",
                "status": "error",
                "error_class": "tool_not_allowed",
                "message": f"tool '{action or '(empty)'}' is not available in retrieval episode",
                "allowed_tools": ToolRegistry.model_visible_names(branch_phase),
                "results": [],
            }
            branch_planner.observe_tool_result(observation)
            append_task_event(
                self.state.task_id,
                "tool_result",
                step=step,
                phase="FORK",
                retrieval_phase=branch_phase,
                branch_id=branch_id,
                branch_step=branch_step,
                action=action,
                result=json.dumps(observation, ensure_ascii=False),
                execution_status="error",
                decision_source="model",
            )
            return step, None

        append_task_event(
            self.state.task_id,
            "tool_call",
            step=step,
            phase="FORK",
            branch_id=branch_id,
            branch_step=branch_step,
            action=action,
            args=args,
            decision_source="model",
        )
        raw_result: Any = ""
        try:
            raw_result = ToolRegistry.execute(
                action,
                args,
                self._agentic_tool_context(),
                phase=branch_phase,
            )
            structured_result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
        except (TypeError, json.JSONDecodeError):
            structured_result = {"status": "ok", "raw": str(raw_result)}
        except Exception as exc:
            structured_result = {
                "schema_version": "retrieval.v1",
                "status": "error",
                "error_class": "tool_execution",
                "message": f"{type(exc).__name__}: {exc}",
                "results": [],
            }
        if not isinstance(structured_result, dict):
            structured_result = {"status": "ok", "raw": structured_result}

        tool_meta = ToolRegistry.metadata(action)
        retrieval_role = str(tool_meta.get("retrieval_role") or "")
        observed_result = structured_result
        round_item: tuple[str, dict] | None = None
        if generic_web_mode and action == "web_search" and not is_error(structured_result):
            branch["discovery_results"] = [
                item for item in (structured_result.get("results") or [])
                if isinstance(item, dict)
            ]
            if structured_result.get("evidence_ready") and branch["discovery_results"]:
                branch["evidence_count"] += len(branch["discovery_results"])
                round_item = (str(args.get("query") or user_query), structured_result)
        elif retrieval_role == "evidence" and not is_error(structured_result):
            evidence_query = self._evidence_query_for_point(task_point_id, user_query)
            observed_result = self._process_single_page_result(
                evidence_query,
                structured_result,
                step,
                evidence_action=action,
            )
            if observed_result.get("results"):
                branch["evidence_count"] += len(observed_result.get("results") or [])
                round_item = (str(args.get("url") or user_query), observed_result)
        elif retrieval_role == "discovery":
            branch["discovery_results"] = [
                item for item in (structured_result.get("results") or [])
                if isinstance(item, dict)
            ]

        if retrieval_role in {"discovery", "evidence"} or action == "web_search":
            observed_result = self._record_retrieval_progress(
                observed_result,
                query=str(args.get("query") or args.get("url") or user_query),
                step=step,
                action=action,
                phase=branch_phase,
                branch_id=branch_id,
                task_point_id=task_point_id,
            )
        branch_planner.observe_tool_result(observed_result)
        append_task_event(
            self.state.task_id,
            "tool_result",
            step=step,
            phase="FORK",
            branch_id=branch_id,
            branch_step=branch_step,
            action=action,
            result=raw_result if isinstance(raw_result, str) else json.dumps(raw_result, ensure_ascii=False),
            real_network=bool(structured_result.get("real_network", True)),
            decision_source="model",
            retrieval_role=retrieval_role,
            evidence_count=len(observed_result.get("results") or []),
            retrieval_phase=branch_phase,
        )
        return step, round_item

    def _run_forked_retrieval(
        self,
        user_query: str,
        task_plan: dict,
        model_profile: dict,
        max_steps: int,
    ) -> str:
        """Run model-generated task points as independent retrieval branches.

        The task planner supplies the semantic branches. Each forked Planner
        receives the same narrow public retrieval contract and independently
        chooses whether to search and what query to issue. Provider selection,
        page fetching and evidence processing remain backend responsibilities.
        The controller records the branch transcript, enforces the global step
        budget, and sends all page evidence to the final RWKV synthesis.
        """
        generic_web_mode = bool(self.state.run_metadata.get("generic_web_search_only"))
        branch_phase = "GENERIC_WEB" if generic_web_mode else "ALL"
        points = [
            point for point in (task_plan.get("atomic_points") or [])
            if isinstance(point, dict)
        ]
        configured_width = self.state.run_metadata.get("retrieval_branch_width")
        if configured_width is None:
            branch_width = len(points)
        else:
            branch_width = max(1, min(int(configured_width or 1), len(points) or 1))
        points = points[:branch_width] or [
            {
                "id": "ROOT",
                "task": user_query,
                "objective": user_query,
                "evidence_needed": [user_query],
                "acceptance_criteria": ["Answer the user's request directly."],
                "output_format": "mixed",
                "status": "pending",
            }
        ]
        self._fork_transcripts = []
        append_task_event(
            self.state.task_id,
            "retrieval_fork_started",
            step=0,
            phase="FORK",
            retrieval_phase=branch_phase,
            generic_web_search_only=generic_web_mode,
            branch_width=len(points),
            max_tool_steps=max_steps,
            branches=[
                {
                    "branch_id": f"B{index + 1}",
                    "task_point_id": str(point.get("id") or ""),
                    "objective": str(point.get("objective") or point.get("task") or ""),
                }
                for index, point in enumerate(points)
            ],
        )

        rounds: list[tuple[str, dict]] = []
        branch_states: list[dict] = []
        total_steps = 0
        for index, point in enumerate(points, start=1):
            branch_id = f"B{index}"
            if generic_web_mode:
                branch_planner = self.planner.fork_for_point(
                    branch_id,
                    point,
                    phase=branch_phase,
                )
            else:
                # Preserve the legacy test-double call shape for the
                # provider-specific Fork path.
                branch_planner = self.planner.fork_for_point(branch_id, point)
            branch_states.append(
                {
                    "branch_id": branch_id,
                    "point": point,
                    "planner": branch_planner,
                    "tools": [],
                    "evidence_count": 0,
                    "discovery_results": [],
                    "branch_step": 0,
                    "active": True,
                    "stop_reason": "model_step_budget",
                }
            )

        # Execute one step per active branch concurrently.  The global step
        # budget remains unchanged; only independent branches in the same
        # round are submitted together.  RWKV/vLLM performs continuous
        # batching, while this bound prevents an unbounded request storm.
        configured_concurrency = self.state.run_metadata.get("retrieval_fork_concurrency")
        fork_concurrency = get_llm_concurrency()
        if configured_concurrency is not None:
            fork_concurrency = max(1, int(configured_concurrency or 1))
        fork_concurrency = max(1, min(fork_concurrency, len(branch_states) or 1))
        append_task_event(
            self.state.task_id,
            "retrieval_fork_concurrency",
            step=0,
            phase="FORK",
            configured=fork_concurrency,
            source="run_metadata" if configured_concurrency is not None else "llm_concurrency",
        )
        while total_steps < max_steps and any(item["active"] for item in branch_states):
            active = [item for item in branch_states if item["active"]]
            batch = active[: min(fork_concurrency, max_steps - total_steps)]
            if not batch:
                break
            assignments = {
                id(branch): total_steps + index + 1
                for index, branch in enumerate(batch)
            }
            with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="rwkv-fork") as pool:
                futures = {
                    pool.submit(
                        self._run_fork_branch_step,
                        user_query,
                        branch,
                        model_profile,
                        assignments[id(branch)],
                        branch_phase,
                        generic_web_mode,
                    ): branch
                    for branch in batch
                }
                completed_steps = []
                for future in as_completed(futures):
                    completed_steps.append(future.result())
            total_steps += len(batch)
            for _, round_item in sorted(completed_steps, key=lambda item: item[0]):
                if round_item is not None:
                    rounds.append(round_item)

        if total_steps >= max_steps:
            for branch in branch_states:
                if branch["active"]:
                    branch["active"] = False
                    branch["stop_reason"] = "global_step_limit"
        branch_records = []
        for branch in branch_states:
            branch_record = {
                "branch_id": branch["branch_id"],
                "task_point_id": str(branch["point"].get("id") or ""),
                "objective": str(branch["point"].get("objective") or branch["point"].get("task") or ""),
                "tools": branch["tools"],
                "evidence_count": branch["evidence_count"],
                "discovery_count": len(branch["discovery_results"]),
                "stop_reason": branch["stop_reason"],
                "retrieval_ledger": self._retrieval_ledger.observation(
                    branch_id=branch["branch_id"],
                    task_point_id=str(branch["point"].get("id") or ""),
                ),
                "transcript": branch["planner"].execution_transcript(),
            }
            branch_records.append(branch_record)
            self._fork_transcripts.append(branch_record)

        append_task_event(
            self.state.task_id,
            "retrieval_fork_completed",
            step=total_steps,
            phase="FORK",
            retrieval_phase=branch_phase,
            generic_web_search_only=generic_web_mode,
            branch_count=len(branch_records),
            evidence_rounds=len(rounds),
            total_tool_steps=total_steps,
            retrieval_ledger=self._retrieval_ledger.snapshot(),
            branches=[
                dict(item)
                for item in branch_records
            ],
        )
        termination_reason = "max_steps_reached" if total_steps >= max_steps else "model_requested_finish"
        return self._complete_model_tool_loop(
            user_query,
            "retrieval_fork",
            rounds,
            max(total_steps, 1),
            termination_reason=termination_reason,
        )

    def _prepare_task_plan(self, user_query: str, phase: str) -> dict:
        """Create the model plan once and initialize the shared transcript."""
        self.planner.reset()
        # Planning starts from the user goal only. Workspace contents remain
        # available to later tool turns, but cannot bias decomposition into a
        # local-file plan.
        plan_context = ""
        task_plan = self.planner.create_task_plan(user_query, plan_context)
        append_task_event(
            self.state.task_id,
            "task_plan",
            step=0,
            phase="ROUTING",
            data=task_plan,
        )
        if task_plan.get("status") == "error":
            return task_plan

        self._task_plan = task_plan
        self.state.run_metadata["task_plan"] = task_plan
        generic_web_mode = bool(self.state.run_metadata.get("generic_web_search_only"))
        # Keep the legacy call shape for existing planner test doubles. The
        # generic web mode is the only path that needs an explicit phase.
        if generic_web_mode:
            self.planner.begin_task(user_query, plan_context, task_plan, phase)
        else:
            self.planner.begin_task(user_query, plan_context, task_plan)
        self._fork_transcripts = []
        return task_plan

    def _fail_task_plan(self, user_query: str, task_plan: dict) -> str:
        answer = "Task planning failed; retrieval did not start."
        self.state.final_result = answer
        self.state.is_finished = True
        append_task_event(
            self.state.task_id,
            "final",
            status="failed",
            content=answer,
            action="task_plan",
            mode="model_task_plan_failed",
            planner_error=task_plan.get("message", ""),
        )
        self._write_agentic_report(user_query, "task_plan", answer, mode="model_task_plan_failed")
        return answer

    def _select_retrieval_strategy(self, task_plan: dict) -> StrategyDecision:
        """Persist and trace the policy chosen from the atomic task points."""
        decision = select_strategy(task_plan, self.state.run_metadata)
        selected = decision.as_dict()
        self.state.run_metadata["retrieval_strategy"] = decision.strategy
        self.state.run_metadata["retrieval_strategy_decision"] = selected
        append_task_event(
            self.state.task_id,
            "retrieval_strategy_selected",
            step=0,
            phase="ROUTING",
            strategy=decision.strategy,
            point_count=decision.point_count,
            point_ids=list(decision.point_ids),
            source=decision.source,
            reason=decision.reason,
            override=decision.source.endswith("override"),
        )
        return decision

    def _run_model_tool_loop(self, user_query: str, model_profile: dict) -> str:
        """Prepare the task once, select a runner, and execute the episode."""
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
        phase = "GENERIC_WEB" if generic_web_mode else "ALL"
        task_plan = self._prepare_task_plan(user_query, phase)
        if task_plan.get("status") == "error":
            return self._fail_task_plan(user_query, task_plan)

        decision = self._select_retrieval_strategy(task_plan)
        runner = build_runner(decision, self)
        return runner.run(user_query, task_plan, model_profile, max_steps)

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
        generic_web_mode = bool(self.state.run_metadata.get("generic_web_search_only"))
        rounds: list[tuple[str, dict]] = []
        phase = "GENERIC_WEB" if generic_web_mode else "ALL"
        last_action = "model_tool_loop"
        retrieval_attempted = False
        last_discovery_results: list[dict] = []
        empty_evidence_attempts = 0
        replan_attempts = 0
        max_replan_attempts = max(
            0,
            int(
                self.state.run_metadata.get(
                    "max_replan_attempts",
                    DEFAULT_MAX_REPLAN_ATTEMPTS,
                )
                or DEFAULT_MAX_REPLAN_ATTEMPTS
            ),
        )
        append_task_event(
            self.state.task_id,
            "retrieval_global_started",
            step=0,
            phase="GLOBAL",
            retrieval_phase=phase,
            max_tool_steps=max_steps,
            max_replan_attempts=max_replan_attempts,
            task_point_count=len(task_plan.get("atomic_points") or []),
        )
        # Replanning and evidence verification run inside the current logical
        # step. Their model calls do not consume the global tool-step budget;
        # the next actual planner/tool turn advances the loop normally.
        for step in range(1, max_steps + 1):
            check_time_budget(minimum_seconds=0.2)
            if is_task_stopped(self.state.task_id):
                self.state.final_result = "执行中止: 任务已被手动停止。"
                self.state.is_finished = True
                append_task_event(self.state.task_id, "final", status="stopped", content=self.state.final_result)
                return self.state.final_result

            context_text = self.state.to_markdown_context()
            plan = self.planner.plan_next_action(user_query, {}, context_text, phase)
            action = str(plan.get("action") or "").strip()
            args = dict(plan.get("args") or {}) if isinstance(plan.get("args"), dict) else {}
            task_point_id = str(plan.get("task_point_id") or "").strip()
            last_action = action or last_action
            append_task_event(
                self.state.task_id,
                "model_tool_decision",
                step=step,
                phase=phase,
                action=action,
                args=args,
                task_point_id=task_point_id,
                router=plan.get("router", "model_tool_decision"),
                raw_model_output=plan.get("raw_model_output", ""),
                planner_error=plan.get("planner_error", ""),
                model=model_profile,
            )

            if plan.get("planner_error"):
                # Tool turns require JSON, but the final turn is ordinary
                # RWKV continuation.  A malformed tool turn must therefore
                # transition to synthesis instead of replacing the model's
                # answer with a controller-authored error string.  Existing
                # evidence is still passed through unchanged; with no
                # evidence the final model is explicitly told that retrieval
                # returned nothing.
                append_task_event(
                    self.state.task_id,
                    "model_tool_decision_error",
                    step=step,
                    phase=phase,
                    action="model_tool_decision",
                    planner_error=plan.get("planner_error", ""),
                    raw_model_output=plan.get("raw_model_output", ""),
                    transition="final_synthesis",
                )
                termination_reason = (
                    "max_steps_reached" if step >= max_steps else "model_tool_decision_parse_error"
                )
                return self._complete_model_tool_loop(
                    user_query,
                    last_action,
                    rounds,
                    step,
                    termination_reason=termination_reason,
                )

            # Keep compatibility with the existing controlled test harness;
            # real planner responses use router=model_tool_decision.
            if step == 1 and plan.get("router") == "test":
                data, _, _ = self._run_rwkv_search_round(
                    user_query,
                    action,
                    args,
                    step=1,
                    round_name="initial",
                )
                return self._finish_rwkv_retrieval(user_query, action, data, 2)

            if action in {"answer_user", "finish_task"}:
                plan_requires_evidence = any(
                    isinstance(point, dict) and bool(point.get("evidence_needed"))
                    for point in (self._task_plan.get("atomic_points") or [])
                )
                if not rounds and plan_requires_evidence and not retrieval_attempted:
                    answer_rejection = {
                        "schema_version": "retrieval.v1",
                        "status": "error",
                        "error_class": "answer_without_evidence",
                        "message": "the task plan declares evidence, but no retrieval evidence has been collected; choose a tool before answering",
                        "results": [],
                    }
                    self.planner.observe_tool_result(answer_rejection)
                    append_task_event(
                        self.state.task_id,
                        "error",
                        step=step,
                        phase="VALIDATION",
                        action=action,
                        error=answer_rejection["message"],
                        error_class=answer_rejection["error_class"],
                    )
                    phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                    continue
                if rounds:
                    if self._validation_architecture() == "rwkv_verifier":
                        merged, verification = self._verify_retrieval_completion(
                            user_query,
                            last_action,
                            rounds,
                            step,
                        )
                        if verification.get("requires_replan") and replan_attempts < max_replan_attempts:
                            replan_attempts += 1
                            append_task_event(
                                self.state.task_id,
                                "task_replan_attempt",
                                step=step,
                                phase="RECOVERY",
                                attempt=replan_attempts,
                                max_attempts=max_replan_attempts,
                                reason="evidence_verifier_found_missing_or_conflicting_points",
                                missing_point_ids=verification.get("missing_point_ids") or [],
                                conflict_point_ids=verification.get("conflict_point_ids") or [],
                                next_queries=verification.get("next_queries") or [],
                                counts_toward_global_steps=False,
                            )
                            replanned = self._replan_after_retrieval_failure(
                                user_query,
                                merged,
                                step,
                                attempt=replan_attempts,
                                max_attempts=max_replan_attempts,
                                validation=verification,
                            )
                            if replanned:
                                retrieval_attempted = True
                                phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                                continue
                        elif verification.get("requires_replan"):
                            append_task_event(
                                self.state.task_id,
                                "task_replan_limit_reached",
                                step=step,
                                phase="RECOVERY",
                                attempts=replan_attempts,
                                max_attempts=max_replan_attempts,
                                reason="evidence_verifier_still_found_missing_or_conflicting_points",
                                missing_point_ids=verification.get("missing_point_ids") or [],
                                conflict_point_ids=verification.get("conflict_point_ids") or [],
                                transition="final_synthesis",
                            )
                    else:
                        merged = merge_retrieval_results(
                            user_query,
                            last_action,
                            rounds,
                            ranking_strategy=self._strategy()["ranking_strategy"],
                        )
                        self._last_evidence_verification = {}
                    answer = self._complete_model_tool_loop(
                        user_query,
                        last_action,
                        rounds,
                        step,
                        termination_reason=("max_steps_reached" if step >= max_steps else "model_requested_finish"),
                    )
                    return answer
                if retrieval_attempted:
                    # Once the model has entered a web-retrieval episode, do
                    # not let answer_user bypass the evidence contract by
                    # launching a second free-form answer call.  This is a
                    # fail-closed boundary, not a routing or repetition
                    # penalty: the model still owns every tool and URL choice.
                    return self._complete_model_tool_loop(
                        user_query,
                        last_action,
                        [],
                        step,
                        termination_reason=("max_steps_reached" if step >= max_steps else "model_requested_finish"),
                    )
                result = ToolRegistry.execute(
                    "answer_user",
                    {"original_goal": user_query},
                    self._agentic_tool_context(),
                )
                answer = str(result or "")
                self.state.final_result = answer
                self.state.is_finished = True
                append_task_event(
                    self.state.task_id,
                    "tool_result",
                    step=step,
                    phase="SYNTHESIS",
                    action="answer_user",
                    result=answer,
                    agent_decision=True,
                )
                append_task_event(
                    self.state.task_id,
                    "final",
                    status="completed",
                    content=answer,
                    action="answer_user",
                    mode="model_tool_loop_direct_answer",
                )
                self._write_agentic_report(user_query, "answer_user", answer)
                return answer

            if not ToolRegistry.can_execute(action, phase):
                allowed_tools = ToolRegistry.model_visible_names(phase)
                if ToolRegistry.has(action):
                    error = f"模型选择的工具在当前阶段不可用: {action or '(empty)'} (phase={phase})"
                else:
                    error = f"模型选择了未注册工具: {action or '(empty)'}"
                self.state.last_feedback = error
                self.planner.observe_tool_result(
                    {
                        "schema_version": "retrieval.v1",
                        "status": "error",
                        "error_class": "tool_not_allowed",
                        "message": f"tool '{action or '(empty)'}' is not available in phase '{phase}'",
                        "allowed_tools": allowed_tools,
                        "results": [],
                    }
                )
                append_task_event(self.state.task_id, "error", step=step, phase="ROUTING", error=error, error_class="model_tool_decision")
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                continue

            duplicate_query = None
            if action == "web_search":
                duplicate_query = self._retrieval_ledger.query_status(args.get("query"))
            if duplicate_query and duplicate_query.get("attempted"):
                duplicate_block = self._retrieval_ledger.record_duplicate_block(
                    args.get("query"),
                    step=step,
                    task_point_id=task_point_id,
                )
                blocked = {
                    "schema_version": "retrieval.v1",
                    "status": "error",
                    "error_class": "duplicate_query",
                    "message": (
                        "This web_search query is an exact or equivalent query already executed in the current "
                        "retrieval episode; no network request was made. Choose a materially different query or "
                        "finish."
                    ),
                    "query_status": {**duplicate_query, **duplicate_block},
                    "retrieval_ledger": self._retrieval_ledger.observation(
                        task_point_id=task_point_id,
                    ),
                    "results": [],
                }
                self.state.last_feedback = blocked["message"]
                self.planner.observe_tool_result(blocked)
                append_task_event(
                    self.state.task_id,
                    "retrieval_duplicate_blocked",
                    step=step,
                    phase="GLOBAL",
                    action=action,
                    args=args,
                    task_point_id=task_point_id,
                    query_status=duplicate_query,
                    execution_status="skipped",
                )
                append_task_event(
                    self.state.task_id,
                    "tool_result",
                    step=step,
                    phase=phase,
                    action=action,
                    result=json.dumps(blocked, ensure_ascii=False),
                    execution_status="blocked_duplicate",
                    decision_source="model",
                    retrieval_environment=plugin_environment_snapshot(),
                )
                append_task_event(
                    self.state.task_id,
                    "retrieval_duplicate_terminated",
                    step=step,
                    phase="SYNTHESIS",
                    action=action,
                    query=str(args.get("query") or ""),
                    termination_reason="duplicate_query_blocked",
                    evidence_rounds=len(rounds),
                )
                if rounds and self._validation_architecture() == "rwkv_verifier":
                    merged, verification = self._verify_retrieval_completion(
                        user_query,
                        action,
                        rounds,
                        step,
                    )
                    if verification.get("requires_replan") and replan_attempts < max_replan_attempts:
                        replan_attempts += 1
                        append_task_event(
                            self.state.task_id,
                            "task_replan_attempt",
                            step=step,
                            phase="RECOVERY",
                            attempt=replan_attempts,
                            max_attempts=max_replan_attempts,
                            reason="evidence_verifier_found_missing_or_conflicting_points",
                            missing_point_ids=verification.get("missing_point_ids") or [],
                            conflict_point_ids=verification.get("conflict_point_ids") or [],
                            next_queries=verification.get("next_queries") or [],
                            counts_toward_global_steps=False,
                        )
                        replanned = self._replan_after_retrieval_failure(
                            user_query,
                            merged,
                            step,
                            attempt=replan_attempts,
                            max_attempts=max_replan_attempts,
                            validation=verification,
                        )
                        if replanned:
                            retrieval_attempted = True
                            phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                            continue
                    elif verification.get("requires_replan"):
                        append_task_event(
                            self.state.task_id,
                            "task_replan_limit_reached",
                            step=step,
                            phase="RECOVERY",
                            attempts=replan_attempts,
                            max_attempts=max_replan_attempts,
                            reason="evidence_verifier_still_found_missing_or_conflicting_points",
                            missing_point_ids=verification.get("missing_point_ids") or [],
                            conflict_point_ids=verification.get("conflict_point_ids") or [],
                            transition="final_synthesis",
                        )
                # A duplicate is an execution error, not a new observation
                # that should re-enter the same model loop. Greedy RWKV can
                # reproduce the same JSON forever.  Verification may grant a
                # bounded replan first; after that, transition to synthesis
                # and let the final model answer from existing evidence or
                # state that evidence is insufficient.
                return self._complete_model_tool_loop(
                    user_query,
                    action,
                    rounds,
                    step,
                    termination_reason="duplicate_query_blocked",
                )

            previous_request = self._retrieval_ledger.request_status(action, args)
            if previous_request and previous_request.get("failed"):
                blocked = {
                    "schema_version": "retrieval.v1",
                    "status": "error",
                    "error_class": "previous_attempt_failed",
                    "message": (
                        "This exact tool request already failed in the current retrieval episode and was not "
                        "re-executed. Choose a different URL/query or finish."
                    ),
                    "previous_request": previous_request,
                    "results": [],
                }
                blocked["retrieval_ledger"] = self._retrieval_ledger.observation(
                    task_point_id=task_point_id,
                )
                append_task_event(
                    self.state.task_id,
                    "retrieval_request_reused",
                    step=step,
                    phase="GLOBAL",
                    action=action,
                    args=args,
                    task_point_id=task_point_id,
                    previous_request=previous_request,
                    execution_status="skipped",
                )
                self.planner.observe_tool_result(blocked)
                append_task_event(
                    self.state.task_id,
                    "tool_result",
                    step=step,
                    phase=phase,
                    action=action,
                    result=json.dumps(blocked, ensure_ascii=False),
                    execution_status="skipped",
                    decision_source="model",
                    retrieval_environment=plugin_environment_snapshot(),
                )
                return self._complete_model_tool_loop(
                    user_query,
                    last_action,
                    rounds,
                    step,
                    termination_reason="repeated_failed_request",
                )

            append_task_event(
                self.state.task_id,
                "tool_call",
                step=step,
                phase=phase,
                action=action,
                args=args,
                decision_source="model",
            )
            try:
                raw_result = ToolRegistry.execute(
                    action,
                    args,
                    self._agentic_tool_context(),
                    phase=phase,
                )
                if isinstance(raw_result, str):
                    try:
                        structured_result = json.loads(raw_result)
                    except json.JSONDecodeError:
                        # Not every registered tool is a retrieval provider.
                        # Plain text is a successful observation, not a tool
                        # execution failure; only structured status=error is
                        # routed through generic recovery.
                        structured_result = {"status": "ok", "raw": raw_result}
                else:
                    structured_result = raw_result
                if not isinstance(structured_result, dict):
                    structured_result = {"status": "ok", "raw": raw_result}
            except Exception as exc:
                structured_result = {
                    "status": "error",
                    "provider_errors": [f"{type(exc).__name__}: {exc}"],
                    "results": [],
                }
                raw_result = json.dumps(structured_result, ensure_ascii=False)

            request_status = self._retrieval_ledger.record_request(
                action,
                args,
                structured_result,
                step=step,
                task_point_id=task_point_id,
            )
            structured_result["request_status"] = request_status

            # A search result is metadata only.  A fetched page is processed
            # as one document and one chunk at a time before the planner sees
            # any observation.  Never feed the raw page body back into the
            # model-owned routing transcript.
            observed_result = structured_result
            tool_meta = ToolRegistry.metadata(action)
            retrieval_role = str(tool_meta.get("retrieval_role") or "")
            if retrieval_role in {"discovery", "evidence"}:
                retrieval_attempted = True
            if action == "web_search":
                # ``web_search`` is a complete bounded retrieval transaction
                # in every production mode: its result already contains
                # admitted URLs, fetched pages, Markdown chunks and merged
                # evidence. Keep usable records for final synthesis instead
                # of silently dropping them outside legacy GENERIC_WEB mode.
                last_discovery_results = [
                    item for item in (structured_result.get("results") or []) if isinstance(item, dict)
                ]
                if structured_result.get("evidence_ready") and last_discovery_results:
                    rounds.append((str(args.get("query") or user_query), structured_result))
            if is_error(structured_result):
                if retrieval_role == "evidence" and last_discovery_results:
                    selected_url = str(args.get("url") or "").strip()
                    alternative_urls = [
                        str(item.get("url") or "").strip()
                        for item in last_discovery_results
                        if isinstance(item, dict)
                        and str(item.get("url") or "").strip()
                        and str(item.get("url") or "").strip() != selected_url
                    ][:8]
                    if alternative_urls:
                        structured_result = dict(structured_result)
                        structured_result["alternative_urls"] = alternative_urls
                        structured_result["recovery_instruction"] = (
                            "The selected page failed. Choose one different URL from alternative_urls "
                            "with the matching evidence tool before starting another search."
                        )
                self.state.last_feedback = (
                    f"[{action}] execution failed; plugin is unavailable for this turn:\n"
                    f"{json.dumps(structured_result, ensure_ascii=False)[:6000]}"
                )
                if retrieval_role in {"discovery", "evidence"} or action == "web_search":
                    structured_result = self._record_retrieval_progress(
                        structured_result,
                        query=str(args.get("query") or args.get("url") or user_query),
                        step=step,
                        action=action,
                        phase=phase,
                        task_point_id=task_point_id,
                    )
                self.planner.observe_tool_result(structured_result)
                append_task_event(
                    self.state.task_id,
                    "tool_result",
                    step=step,
                    phase=phase,
                    action=action,
                    result=raw_result,
                    real_network=bool(structured_result.get("real_network", True)),
                    decision_source="model",
                    execution_status="error",
                    retrieval_environment=plugin_environment_snapshot(),
                )
                append_task_event(
                    self.state.task_id,
                    "provider_error",
                    step=step,
                    phase="RECOVERY",
                    action=action,
                    error=str(structured_result.get("message") or (structured_result.get("provider_errors") or ["tool execution failed"])[0]),
                    error_class=str(structured_result.get("error_class") or "provider_error"),
                    provider=structured_result.get("provider", ""),
                )
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                if retrieval_role == "evidence" and structured_result.get("alternative_urls"):
                    phase = "ALL"
                continue
            if retrieval_role == "evidence":
                evidence_query = self._evidence_query_for_point(task_point_id, user_query)
                observed_result = self._process_single_page_result(
                    evidence_query,
                    structured_result,
                    step,
                    evidence_action=action,
                )
                page_evidence = observed_result.get("page_evidence") or {}
                page_evidence_status = str(page_evidence.get("status") or "").casefold()
                usable_page_evidence = page_evidence_status in {
                    "ok",
                    "supported",
                    "completed",
                }
                # A fetch can return HTTP success while page extraction yields
                # no supported chunk evidence.  For the global request ledger
                # that is still a failed request: otherwise RWKV can keep
                # selecting the same empty page forever.
                if not usable_page_evidence:
                    observed_result.update(
                        {
                            "status": "error",
                            "error_class": "no_evidence",
                            "message": "the selected page produced no usable chunk evidence",
                        }
                    )
                    request_status = self._retrieval_ledger.record_request(
                        action,
                        args,
                        observed_result,
                        step=step,
                        task_point_id=task_point_id,
                    )
                    observed_result["request_status"] = request_status
                if not usable_page_evidence and last_discovery_results:
                    selected_url = str(args.get("url") or "").strip()
                    alternative_urls = [
                        str(item.get("url") or "").strip()
                        for item in last_discovery_results
                        if isinstance(item, dict)
                        and str(item.get("url") or "").strip()
                        and str(item.get("url") or "").strip() != selected_url
                    ][:8]
                    if alternative_urls:
                        observed_result = dict(observed_result)
                        observed_result["alternative_urls"] = alternative_urls
                        observed_result["recovery_instruction"] = (
                            "The selected page produced no evidence. Choose one different URL from alternative_urls "
                            "with the matching evidence tool before refining the search."
                        )
            if retrieval_role in {"discovery", "evidence"} or action == "web_search":
                observed_result = self._record_retrieval_progress(
                    observed_result,
                    query=str(args.get("query") or args.get("url") or user_query),
                    step=step,
                    action=action,
                    phase=phase,
                    task_point_id=task_point_id,
                )
            self.state.last_feedback = (
                f"[{action}] model-selected tool result:\n"
                f"{json.dumps(observed_result, ensure_ascii=False)[:6000]}"
            )
            self.planner.observe_tool_result(observed_result)
            append_task_event(
                self.state.task_id,
                "tool_result",
                step=step,
                phase=phase,
                action=action,
                result=raw_result,
                real_network=bool(structured_result.get("real_network", True)),
                decision_source="model",
                retrieval_environment=plugin_environment_snapshot(),
            )
            if retrieval_role == "evidence":
                # Only chunk-derived facts enter the final retrieval rounds.
                # A fetch with no supported chunk remains a discovery result,
                # allowing the model to select another URL itself.
                if usable_page_evidence:
                    empty_evidence_attempts = 0
                    rounds.append((str(args.get("url") or user_query), observed_result))
                    # Evidence for one point does not finish a multi-point
                    # task.  Return to the shared recovery phase so the same
                    # Planner can select another point, refine the query, or
                    # explicitly finish after inspecting the global ledger.
                    phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                else:
                    empty_evidence_attempts += 1
                    if empty_evidence_attempts >= 2 and empty_evidence_attempts % 2 == 0:
                        if replan_attempts < max_replan_attempts:
                            replan_attempts += 1
                            append_task_event(
                                self.state.task_id,
                                "task_replan_attempt",
                                step=step,
                                phase="RECOVERY",
                                attempt=replan_attempts,
                                max_attempts=max_replan_attempts,
                                counts_toward_global_steps=False,
                            )
                            self._replan_after_retrieval_failure(
                                user_query,
                                observed_result,
                                step,
                                attempt=replan_attempts,
                                max_attempts=max_replan_attempts,
                            )
                        else:
                            append_task_event(
                                self.state.task_id,
                                "task_replan_limit_reached",
                                step=step,
                                phase="RECOVERY",
                                attempts=replan_attempts,
                                max_attempts=max_replan_attempts,
                                transition="final_synthesis",
                            )
                            return self._complete_model_tool_loop(
                                user_query,
                                action,
                                rounds,
                                step,
                                termination_reason="replan_limit_reached",
                            )
                    if observed_result.get("alternative_urls"):
                        phase = "ALL"
                    else:
                        phase = "ALL"
            elif retrieval_role == "discovery":
                # Search results expose candidate URLs to the planner but are
                # not final evidence and are never merged into the answer.
                last_discovery_results = [
                    item for item in (structured_result.get("results") or []) if isinstance(item, dict)
                ]
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"
            else:
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"

        return self._complete_model_tool_loop(
            user_query,
            last_action,
            rounds,
            max_steps,
            termination_reason="max_steps_reached",
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
        self._retrieval_ledger = RetrievalLedger()
        self._fork_transcripts = []
        self._last_evidence_verification = {}
        self.state.task_output_dir = os.path.join(DATA_PIPELINE.get("output_directory", "./data/output"), self.state.task_id)
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
                "retrieval": {"search_action": (run_metadata or {}).get("search_action", "")},
            },
            prompt_version=(run_metadata or {}).get("prompt_version", "unversioned"),
            run_metadata=run_metadata or {},
        )
        append_task_event(
            self.state.task_id,
            "validation_architecture_selected",
            phase="ROUTING",
            architecture=self._validation_architecture(),
            verifier_message_contract=(
                "control-only: task-point status, evidence refs, missing fields, conflicts, and next queries; "
                "never an answer or new factual claim"
            ),
        )
        
        debug_dir = DATA_PIPELINE.get("debug_directory", "./data/debug_slm")
        os.makedirs(debug_dir, exist_ok=True)
        session_id = self.state.task_id
        trace_file = os.path.join(debug_dir, f"DeepResearch_Trace_{session_id}.md")
        
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(f"# Deep Research 执行追踪日志\n\n**启动时间**: {session_id}\n**用户指令**: {user_query}\n\n---\n\n")

        # Controlled experiments may pin one registered retrieval plugin, but
        # normal routing never depends on a provider name. Keep this legacy
        # path metadata-driven so adding a plugin does not add an orchestrator
        # branch.
        discovery_actions = [
            name
            for name in ToolRegistry.names("DISCOVERY")
            if ToolRegistry.metadata(name).get("retrieval_role") == "discovery"
        ]
        if not discovery_actions:
            discovery_actions = [
                name
                for name in ToolRegistry.names()
                if ToolRegistry.metadata(name).get("retrieval_role") == "discovery"
            ]

        def is_registered_discovery(action_name: str) -> bool:
            return ToolRegistry.metadata(action_name).get("retrieval_role") == "discovery"

        def discovery_args(action_name: str, scope: str = "") -> dict:
            allowed = set(ToolRegistry.metadata(action_name).get("allowed_args") or ())
            values = {"scope": scope or "paper", "max_results": 8}
            if action_name != "search_papers":
                values = {"max_results": 6, "fetch_pages": 3}
            return {key: value for key, value in values.items() if key in allowed}

        step_count = 0
        progress_log = []
        def push_progress(msg: str):
            progress_log.append(msg)
            update_task_progress(self.state.task_id, "\n".join(progress_log))
            append_task_event(
                self.state.task_id,
                "progress",
                step=step_count,
                message=msg,
            )

        push_progress("🚀 已启用联网检索，等待 RWKV 自主选择工具...")

        # Do not inject a local-workspace inventory into every ordinary web
        # conversation.  It adds unrelated state, creates the misleading
        # “感知到 N 份文件” progress line, and biases the planner toward the
        # legacy file-research path.  Local files remain available through the
        # registered local-file tool when RWKV explicitly chooses it.  A
        # controlled run may opt into the old inventory with metadata.
        if (run_metadata or {}).get("include_workspace"):
            try:
                initial_files_json = ToolRegistry.execute("search_local_file", {"keyword": ""}, {})
                initial_files = json.loads(initial_files_json)
                for i, p in enumerate(initial_files):
                    fid = f"DOC_{i+1}"
                    self.state.id_to_path[fid] = p
                    self.state.path_to_id[p] = fid
                self.state.last_feedback = f"本地工作区已按任务要求挂载，共 {len(initial_files)} 份文件。"
                push_progress(f"本地工作区已挂载：{len(initial_files)} 份文件。\n")
            except Exception:
                self.state.last_feedback = "按任务要求挂载本地工作区失败。"
                push_progress("本地工作区挂载失败，将继续使用联网检索。\n")
            
        # Scholarly and live-web requests have a deterministic, auditable
        # retrieval path.  Keep the local RWKV analysis as a trace signal, but
        # do not make a network retrieval wait on the legacy file-research
        # loop or on a second free-form planner generation.
        forced_action = str((run_metadata or {}).get("search_action") or "").strip()
        if not forced_action and not self.state.run_metadata.get("retrieval_only"):
            # Production/API requests are model-owned.  A search_action is
            # reserved for controlled experiments where the provider is the
            # explicitly varied variable; it must not silently redefine the
            # normal routing contract.
            return self._run_model_tool_loop(user_query, model_profile)
        if not forced_action and self.state.run_metadata.get("retrieval_only"):
            # Diagnostic retrieval-only runs still need a concrete provider,
            # but they must not spend a model routing call before measuring it.
            forced_action = discovery_actions[0] if discovery_actions else ""

        if is_registered_discovery(forced_action):
            forced_args = discovery_args(forced_action)
            direct_plan = {
                "action": forced_action,
                "args": forced_args,
                "router": "experiment_single_variable_override",
            }
            append_task_event(
                self.state.task_id,
                "experiment_control",
                phase="ROUTING",
                changed_variable="search_action",
                baseline_action=(run_metadata or {}).get("baseline_search_action", ""),
                candidate_action=forced_action,
                invariant_model=model_profile,
            )
        else:
            # A controlled experiment must not spend an untracked planner call
            # before the explicitly selected retrieval action. This also lets
            # retrieval-only diagnostics run when the RWKV endpoint is offline.
            direct_plan = self.planner.plan_next_action(user_query, {}, "", "DISCOVERY")
        if direct_plan.get("action") == "multi_hop_research":
            first_action = str(
                (direct_plan.get("args") or {}).get("first_action")
                or (discovery_actions[0] if discovery_actions else "")
            )
            if not is_registered_discovery(first_action):
                first_action = discovery_actions[0] if discovery_actions else ""
            scope = str((direct_plan.get("args") or {}).get("scope") or "paper")
            append_task_event(
                self.state.task_id,
                "plan",
                step=1,
                phase="DISCOVERY",
                action="multi_hop_research",
                args={"first_action": first_action, "scope": scope},
                router=direct_plan.get("router", "static_multi_hop_cue"),
            )
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=1,
                phase="DISCOVERY",
                query=user_query,
                router_hint=direct_plan.get("router", "static_multi_hop_cue"),
            )
            first_args = discovery_args(first_action, scope)
            first_data, first_queries, _ = self._run_rwkv_search_round(
                user_query,
                first_action,
                first_args,
                step=1,
                round_name="initial",
            )
            followup_action = first_action
            followup_args = discovery_args(followup_action, scope)
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=2,
                phase="DISCOVERY",
                query=user_query,
                router_hint="rwkv_followup_from_initial_evidence",
            )
            second_data, second_queries, _ = self._run_rwkv_search_round(
                user_query,
                followup_action,
                followup_args,
                step=2,
                round_name="followup",
                previous_query=first_queries[0] if first_queries else user_query,
                observation=first_data,
            )
            merged = merge_retrieval_results(
                user_query,
                followup_action,
                [
                    (first_queries[0] if first_queries else user_query, first_data),
                    (second_queries[0] if second_queries else user_query, second_data),
                ],
                scope=scope,
                ranking_strategy=self._strategy()["ranking_strategy"],
            )
            merged["round_count"] = 2
            merged["candidate_queries"] = [*first_queries, *second_queries]
            self._record_final_ranking(followup_action, merged, 2, stage="multi_hop_merge")
            append_task_event(
                self.state.task_id,
                "multi_hop_merge",
                step=2,
                phase="DISCOVERY",
                data={
                    "round_count": 2,
                    "first_queries": first_queries,
                    "second_queries": second_queries,
                    "first_result_count": len(first_data.get("results") or []),
                    "second_result_count": len(second_data.get("results") or []),
                    "final_result_count": len(merged.get("results") or []),
                    "result_count": merged.get("count", 0),
                },
            )
            return self._finish_rwkv_retrieval(user_query, followup_action, merged, 3)

        if is_registered_discovery(str(direct_plan.get("action") or "")):
            action = direct_plan["action"]
            args = dict(direct_plan.get("args") or {})
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=1,
                phase="DISCOVERY",
                query=user_query,
                router_hint=direct_plan.get("router", "local_rwkv_candidate_search"),
            )
            append_task_event(
                self.state.task_id,
                "plan",
                step=1,
                phase="DISCOVERY",
                action=action,
                args=args,
                router=direct_plan.get("router", "local_rwkv_candidate_search"),
            )
            data, _, _ = self._run_rwkv_search_round(
                user_query,
                action,
                args,
                step=1,
                round_name="initial",
            )
            return self._finish_rwkv_retrieval(user_query, action, data, 2)

        # Closed-world prompts should not enter the legacy analyzer/planner
        # loop. That loop can spend minutes retrying a local-model request and
        # may hallucinate a different tool (for example weather for a
        # translation request). Keep the step visible in the trace while
        # executing exactly one deterministic/local action.
        if direct_plan.get("action") in {"answer_user", "get_current_weather"}:
            step_count = 1
            action = direct_plan["action"]
            args = dict(direct_plan.get("args") or {})
            context_text = self.state.to_markdown_context()
            append_task_event(
                self.state.task_id,
                "analysis",
                step=step_count,
                phase="DIRECT",
                context_snapshot=context_text,
                data={
                    "execution_route": "direct_closed_world_action",
                    "intent_mode": "NO_SEARCH",
                    "refined_query": user_query,
                    "next_phase": "DIRECT",
                },
            )
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=step_count,
                phase="DIRECT",
                query=user_query,
                router_hint=direct_plan.get("router", "deterministic_no_search"),
            )
            append_task_event(
                self.state.task_id,
                "plan",
                step=step_count,
                phase="DIRECT",
                action=action,
                args=args,
                router=direct_plan.get("router", "deterministic_no_search"),
            )
            append_task_event(
                self.state.task_id,
                "tool_call",
                step=step_count,
                phase="DIRECT",
                action=action,
                args=args,
            )
            env_context = {
                "original_goal": user_query,
                "path_to_id": self.state.path_to_id,
                "id_to_path": self.state.id_to_path,
                "working_memory": self.state.working_memory,
                "tracker": self.tracker,
                "agent_state": self.state,
                "task_id": self.state.task_id,
                "slm_scheduler": GLOBAL_SLM_INPUT_SCHEDULER,
            }
            started_at = time.perf_counter()
            result = ToolRegistry.execute(action, args=args, context=env_context)
            duration_ms = round((time.perf_counter() - started_at) * 1000, 1)
            append_task_event(
                self.state.task_id,
                "tool_result",
                step=step_count,
                phase="DIRECT",
                action=action,
                result=result,
                real_network=action == "get_current_weather",
            )
            self.state.is_finished = True
            self.state.final_result = self.state.final_result or result
            append_task_event(
                self.state.task_id,
                "final",
                status="completed",
                content=self.state.final_result,
                action=action,
                duration_ms=duration_ms,
            )
            return self.state.final_result

        # The former Analyzer -> Planner -> retry loop is intentionally no
        # longer part of the runtime. Unknown work is stopped safely instead
        # of allowing a small model to invent a tool or loop over local files.
        self.state.is_finished = True
        self.state.final_result = "当前请求未命中受支持的静态路由，未执行旧式循环，也未编造答案。"
        append_task_event(
            self.state.task_id,
            "analysis",
            step=1,
            phase="ROUTING",
            data={
                "execution_route": "unsupported_safe_stop",
                "intent_mode": "UNSUPPORTED",
                "refined_query": user_query,
                "next_phase": "STOP",
            },
        )
        append_task_event(
            self.state.task_id,
            "final",
            status="completed",
            content=self.state.final_result,
            action="safe_stop",
        )
        return self.state.final_result
