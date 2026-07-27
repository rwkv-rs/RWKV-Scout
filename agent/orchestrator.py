# RWKV-ECRA/agent/orchestrator.py
import os
import json
import time
from datetime import datetime

from agent.analyzer import Analyzer
from agent.planner import Planner
from agent.state import AgentState
from agent.slm_scheduler import GLOBAL_SLM_INPUT_SCHEDULER
from agent.retrieval_synthesis import synthesize_retrieval_answer
from agent.page_evidence import extract_single_page_evidence
from agent.retrieval_loop import (
    execute_parallel_candidates,
    generate_query_candidates,
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
from tools.registry import ToolRegistry
from utils.task_manager import is_task_stopped, update_task_progress
from utils.task_events import append_task_event
from utils.citation_validator import validate_citations
from utils.risk_policy import risk_context, validate_risk_answer
from utils.experiment_strategies import normalize_strategy
from utils.time_budget import check_time_budget
from tools.builtin import load_builtin_tools
from retrieval_plugins import is_error, plugin_environment_snapshot


class Orchestrator:
    def __init__(self):
        load_builtin_tools()
        self.tracker = EventTracker(log_dir=TRACKING.get("log_dir", "./logs"), enable=TRACKING.get("enable", True))
        self.state = AgentState()
        self.state.working_memory["__category_tree__"] = {} 
        self.analyzer = Analyzer()
        self.planner = Planner()
        self._task_plan: dict = {}
        self._last_completion_judgement: dict = {}
        self._last_completion_context = ""

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
        return context

    def _process_single_page_result(self, evidence_query: str, data: dict, step: int) -> dict:
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
                "message": "fetch_web_url must return exactly one page",
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
                action="fetch_web_url",
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
                action="fetch_web_url",
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

        compact_facts = str(evidence.get("compact_facts") or "")[:6000]
        merged_page = {
            "title": page.get("title") or url,
            "url": url,
            "snippet": compact_facts[:1000],
            "page_excerpt": compact_facts,
            "content": compact_facts,
            "source": page.get("source") or "explicit model-selected URL",
            "untrusted_content": True,
            "chunk_count": evidence.get("chunk_count", 0),
            "chunk_candidates": evidence.get("candidates") or [],
            "evidence_status": evidence.get("status", "no_evidence"),
        }
        compact = dict(data)
        compact["results"] = [merged_page] if compact_facts else []
        compact["sources"] = [url] if url else []
        source_refs = [item for item in data.get("citation_refs") or [] if isinstance(item, dict)]
        compact["citation_refs"] = [
            {
                "ref_id": (source_refs[0].get("ref_id") if source_refs else "") or f"WEB_FETCH_{step}",
                "title": merged_page["title"],
                "url": url,
                "source": merged_page["source"],
                "evidence_text": compact_facts,
            }
        ] if compact_facts else []
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
            action="fetch_web_url",
            url=url,
            data=compact["page_evidence"],
            candidates=evidence.get("candidates") or [],
            compact_facts=compact_facts,
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
    ) -> bool:
        """Ask RWKV to split the remaining work after unusable evidence.

        The controller reports only the execution fact (the selected page did
        not yield evidence).  It does not infer which semantic field is
        missing or create a replacement query/URL.
        """
        if not self._task_plan:
            return False
        judgement = {
            "schema_version": "completion_judgement.v1",
            "status": "incomplete",
            "reason": "The selected retrieval observation did not yield usable evidence.",
            "missing_point_ids": [],
            "next_focus": [],
        }
        followup_plan = self.planner.replan_task(
            user_query,
            self._task_plan,
            judgement,
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
            },
        )
        if followup_plan.get("status") == "error":
            self.planner.observe_tool_result(followup_plan)
            return False
        self._task_plan = followup_plan
        self.state.run_metadata["task_plan"] = followup_plan
        self.planner.update_task_plan(followup_plan)
        return True

    def _evidence_query_for_point(self, task_point_id: str, fallback: str) -> str:
        """Project the model-selected atomic point into the evidence prompt."""
        point_id = str(task_point_id or "").strip()
        for point in self._task_plan.get("atomic_points") or []:
            if isinstance(point, dict) and str(point.get("id") or "").strip() == point_id:
                objective = str(point.get("objective") or "").strip()
                needed = point.get("evidence_needed") or []
                if objective:
                    needed_text = "; ".join(str(item).strip() for item in needed if str(item).strip())
                    return f"Atomic objective: {objective}\nEvidence needed: {needed_text}"
        points = [point for point in self._task_plan.get("atomic_points") or [] if isinstance(point, dict)]
        if points:
            objectives = []
            for point in points:
                objective = str(point.get("objective") or "").strip()
                needed = "; ".join(str(item).strip() for item in (point.get("evidence_needed") or []) if str(item).strip())
                if objective:
                    objectives.append(f"- {point.get('id', '')}: {objective}; evidence: {needed}")
            if objectives:
                return "Task plan atomic objectives:\n" + "\n".join(objectives)
        return fallback

    def _replan_after_incomplete_judgement(self, user_query: str, step: int) -> bool:
        """Turn an incomplete model judgment into the next model plan."""
        judgement = self._last_completion_judgement
        if judgement.get("status") != "incomplete" or self.state.is_finished:
            return False
        followup_plan = self.planner.replan_task(
            user_query,
            self._task_plan,
            judgement,
            self._last_completion_context,
        )
        append_task_event(
            self.state.task_id,
            "task_replan",
            step=step,
            phase="RECOVERY",
            data=followup_plan,
            previous_judgement=judgement,
        )
        if followup_plan.get("status") == "error":
            self.planner.observe_tool_result(followup_plan)
            return False
        self._task_plan = followup_plan
        self.state.run_metadata["task_plan"] = followup_plan
        self.planner.update_task_plan(followup_plan)
        return True

    def _complete_model_tool_loop(
        self,
        user_query: str,
        action: str,
        rounds: list[tuple[str, dict]],
        step: int,
    ) -> str:
        """Generate the final answer only after the model chose to finish."""
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
            )
            answer = synthesis.get("content") or ""
            citation_validation = validate_citations(
                synthesis.get("citation_refs") or [],
                answer=answer,
                evidence=merged.get("results") or [],
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
                prompt=synthesis.get("prompt", ""),
                model_output=synthesis.get("model_output", ""),
                repair_prompt=synthesis.get("repair_prompt", ""),
                repair_output=synthesis.get("repair_output", ""),
                context_text=synthesis.get("context_text", ""),
                selected_evidence=synthesis.get("selected_evidence") or [],
                context_stats=synthesis.get("context_stats") or {},
            )
            task_plan = self._task_plan or self.state.run_metadata.get("task_plan") or {}
            judgement = self.planner.judge_completion(
                user_query,
                task_plan,
                answer,
                synthesis.get("context_text", ""),
            )
            self._last_completion_judgement = judgement
            self._last_completion_context = synthesis.get("context_text", "")
            append_task_event(
                self.state.task_id,
                "completion_judgement",
                step=step,
                phase="VALIDATION",
                data=judgement,
            )
            max_steps = max(1, min(int(self.state.run_metadata.get("max_tool_steps", 8) or 8), 16))
            if judgement.get("status") == "incomplete" and step < max_steps:
                # Put the independent judge's result back into the planner
                # transcript before asking it to split the remaining work.
                # Without this observation the next model turn only sees the
                # draft answer and can repeatedly choose answer_user.
                self.planner.observe_tool_result(
                    {
                        "schema_version": "retrieval.v1",
                        "status": "incomplete",
                        "error_class": "completion_incomplete",
                        "message": "completion judge found unsupported or missing atomic points; continue retrieval",
                        "completion_judgement": judgement,
                        "missing_point_ids": judgement.get("missing_point_ids") or [],
                        "next_focus": judgement.get("next_focus") or [],
                        "results": [],
                    }
                )
                self.state.final_result = answer
                self.state.is_finished = False
                append_task_event(
                    self.state.task_id,
                    "completion_pending",
                    step=step,
                    phase="RECOVERY",
                    data=judgement,
                )
                return answer
            if judgement.get("status") == "error":
                answer = "当前无法完成可靠的完成性判断，因此不能确认检索结果已经完成。"
                synthesis["mode"] = "model_completion_judgement_failed"
                risk_validation = validate_risk_answer(answer, self.state.run_metadata)
            elif judgement.get("status") == "incomplete":
                missing = "、".join(judgement.get("missing_point_ids") or []) or "未说明的任务点"
                answer = f"检索未完成，完成性判断仍缺少任务点：{missing}。"
                synthesis["mode"] = "model_completion_incomplete"
            self.state.final_result = answer
            self.state.is_finished = True
            merged["task_plan"] = task_plan
            merged["completion_judgement"] = judgement
            final_is_complete = judgement.get("status") == "complete"
            append_task_event(
                self.state.task_id,
                "final",
                status=(
                    "failed"
                    if not final_is_complete
                    else (
                        "completed"
                        if risk_validation.get("valid", True)
                        else "completed_with_warnings"
                    )
                ),
                content=answer,
                action=action,
                mode=synthesis.get("mode"),
                round_count=merged.get("round_count", len(rounds)),
                citation_refs=synthesis.get("citation_refs") or [],
                citation_validation=citation_validation,
                risk_validation=risk_validation,
                completion_judgement=judgement,
            )
            self._write_agentic_report(user_query, action, answer, data=merged, mode=synthesis.get("mode", "model_tool_loop"))
            return answer

        answer = "未完成检索，无法生成有来源支持的答案。"
        self.state.final_result = answer
        self.state.is_finished = True
        append_task_event(
            self.state.task_id,
            "final",
            status="failed",
            content=answer,
            action=action,
            mode="model_tool_loop_no_evidence",
        )
        self._write_agentic_report(user_query, action, answer, mode="model_tool_loop_no_evidence")
        return answer

    def _run_model_tool_loop(self, user_query: str, model_profile: dict) -> str:
        """Let RWKV choose tools, URLs and arguments until it chooses to answer."""
        max_steps = max(1, min(int(self.state.run_metadata.get("max_tool_steps", 8) or 8), 16))
        rounds: list[tuple[str, dict]] = []
        seen_tool_calls: dict[str, int] = {}
        phase = "DISCOVERY"
        last_action = "model_tool_loop"
        retrieval_attempted = False
        last_discovery_results: list[dict] = []
        empty_evidence_attempts = 0
        repeat_recovery_attempts = 0
        self.planner.reset()
        self._last_completion_judgement = {}
        self._last_completion_context = ""
        # Planning starts from the user goal only.  Workspace contents remain
        # available to the later tool-decision turn, but cannot bias the
        # semantic decomposition into a local-file plan.
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
            answer = "任务规划失败，无法安全开始检索。"
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
        self._task_plan = task_plan
        self.state.run_metadata["task_plan"] = task_plan
        self.planner.begin_task(user_query, plan_context, task_plan)

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
                # A malformed model call is an episode failure.  Never invent
                # a provider or query here: the model owns the next action.
                answer = "模型未返回可执行的 RWKV JSON 工具调用，无法安全继续检索。"
                self.state.final_result = answer
                self.state.is_finished = True
                append_task_event(
                    self.state.task_id,
                    "final",
                    status="failed",
                    content=answer,
                    action="model_tool_decision",
                    mode="model_rwkv_json_parse_error",
                    planner_error=plan.get("planner_error", ""),
                )
                self._write_agentic_report(
                    user_query,
                    "model_tool_decision",
                    answer,
                    mode="model_rwkv_json_parse_error",
                )
                return answer

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
                    phase = "DISCOVERY"
                    continue
                if rounds:
                    answer = self._complete_model_tool_loop(user_query, last_action, rounds, step)
                    if self._replan_after_incomplete_judgement(user_query, step):
                        phase = "DISCOVERY"
                        continue
                    return answer
                if retrieval_attempted:
                    # Once the model has entered a web-retrieval episode, do
                    # not let answer_user bypass the evidence contract by
                    # launching a second free-form answer call.  This is a
                    # fail-closed boundary, not a routing or repetition
                    # penalty: the model still owns every tool and URL choice.
                    return self._complete_model_tool_loop(user_query, last_action, [], step)
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

            call_key = f"{action}:{json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
            repeat_count = seen_tool_calls.get(call_key, 0)
            seen_tool_calls[call_key] = repeat_count + 1
            if repeat_count:
                repeated_result = {
                    "schema_version": "retrieval.v1",
                    "status": "error",
                    "error_class": "repeated_tool_call",
                    "message": (
                        "the exact tool and arguments were already executed in this retrieval episode; "
                        "do not repeat them. Choose another listed tool or make a materially different query."
                    ),
                    "action": action,
                    "args": args,
                    "repeat_count": repeat_count,
                    "allowed_tools": ToolRegistry.names(phase),
                    "results": [],
                }
                self.state.last_feedback = json.dumps(repeated_result, ensure_ascii=False)
                self.planner.observe_tool_result(repeated_result)
                append_task_event(
                    self.state.task_id,
                    "error",
                    step=step,
                    phase="ROUTING",
                    action=action,
                    args=args,
                    error=repeated_result["message"],
                    error_class="repeated_tool_call",
                    repeat_count=repeat_count,
                )
                if repeat_count >= 2:
                    if repeat_recovery_attempts < 1:
                        repeat_recovery_attempts += 1
                        if self._replan_after_retrieval_failure(user_query, repeated_result, step):
                            phase = "DISCOVERY"
                            continue
                    answer = "模型在检索过程中重复执行相同工具调用，无法安全继续。"
                    self.state.final_result = answer
                    self.state.is_finished = True
                    append_task_event(
                        self.state.task_id,
                        "final",
                        status="failed",
                        content=answer,
                        action=action,
                        mode="model_repeated_tool_call",
                        repeat_count=repeat_count,
                    )
                    self._write_agentic_report(
                        user_query,
                        action,
                        answer,
                        mode="model_repeated_tool_call",
                    )
                    return answer
                phase = "RECOVERY"
                continue

            if not ToolRegistry.can_execute(action, phase):
                allowed_tools = ToolRegistry.names(phase)
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
                phase = "RECOVERY"
                continue

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

            # A search result is metadata only.  A fetched page is processed
            # as one document and one chunk at a time before the planner sees
            # any observation.  Never feed the raw page body back into the
            # model-owned routing transcript.
            observed_result = structured_result
            tool_meta = ToolRegistry.metadata(action)
            retrieval_role = str(tool_meta.get("retrieval_role") or "")
            if retrieval_role in {"discovery", "evidence"}:
                retrieval_attempted = True
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
                            "with fetch_web_url before starting another search."
                        )
                self.state.last_feedback = (
                    f"[{action}] execution failed; plugin is unavailable for this turn:\n"
                    f"{json.dumps(structured_result, ensure_ascii=False)[:6000]}"
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
                phase = "RECOVERY"
                if retrieval_role == "evidence" and structured_result.get("alternative_urls"):
                    phase = "EXTRACTION"
                continue
            if retrieval_role == "evidence":
                evidence_query = self._evidence_query_for_point(task_point_id, user_query)
                observed_result = self._process_single_page_result(evidence_query, structured_result, step)
                if not observed_result.get("results") and last_discovery_results:
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
                            "with fetch_web_url before refining the search."
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
                if observed_result.get("results"):
                    empty_evidence_attempts = 0
                    rounds.append((str(args.get("url") or user_query), observed_result))
                    # One usable page is enough to trigger the model-owned
                    # synthesis and independent completion judgment.  If the
                    # judgment is incomplete, the model receives a follow-up
                    # atomic plan before another discovery turn.
                    answer = self._complete_model_tool_loop(user_query, action, rounds, step)
                    if self._replan_after_incomplete_judgement(user_query, step):
                        phase = "DISCOVERY"
                        continue
                    return answer
                else:
                    empty_evidence_attempts += 1
                    if empty_evidence_attempts >= 2 and empty_evidence_attempts % 2 == 0:
                        self._replan_after_retrieval_failure(user_query, observed_result, step)
                    if observed_result.get("alternative_urls"):
                        phase = "EXTRACTION"
                    else:
                        phase = "DISCOVERY"
            elif retrieval_role == "discovery":
                # Search results expose candidate URLs to the planner but are
                # not final evidence and are never merged into the answer.
                last_discovery_results = [
                    item for item in (structured_result.get("results") or []) if isinstance(item, dict)
                ]
                phase = "EXTRACTION"
            else:
                phase = "DISCOVERY"

        if rounds:
            return self._complete_model_tool_loop(user_query, last_action, rounds, max_steps)
        return self._complete_model_tool_loop(user_query, last_action, [], max_steps)

    def _strategy(self) -> dict:
        return normalize_strategy(self.state.run_metadata.get("strategy_config"))

    def _record_final_ranking(self, action: str, data: dict, step: int, *, stage: str) -> None:
        """Persist the post-recovery/post-multi-hop ranking snapshot."""
        append_task_event(
            self.state.task_id,
            "ranking",
            step=step,
            phase="RANKING",
            action=action,
            data={
                "method": f"{data.get('ranking_strategy', self._strategy()['ranking_strategy'])}.final_merge",
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

    def _run_rwkv_search_round(
        self,
        user_query: str,
        action: str,
        args: dict,
        *,
        step: int,
        round_name: str,
        previous_query: str = "",
        observation: dict | None = None,
    ) -> tuple[dict, list[str], dict]:
        check_time_budget(minimum_seconds=0.2)
        round_started = time.perf_counter()
        query_started = time.perf_counter()
        plan = generate_query_candidates(
            self.analyzer.llm,
            user_query,
            action=action,
            scope=str(args.get("scope") or ""),
            observation=observation,
            previous_query=previous_query,
            max_candidates=3 if not previous_query else 2,
        )
        query_duration_ms = round((time.perf_counter() - query_started) * 1000, 1)
        candidates = list(plan.get("queries") or [user_query])
        append_task_event(
            self.state.task_id,
            "query_candidates",
            step=step,
            phase="DISCOVERY",
            round=round_name,
            action=action,
            source=plan.get("source"),
            queries=candidates,
            raw_model_output=plan.get("raw_model_output", ""),
            planner_error=plan.get("error", ""),
            duration_ms=query_duration_ms,
        )
        for candidate in candidates:
            append_task_event(
                self.state.task_id,
                "tool_call",
                step=step,
                phase="DISCOVERY",
                action=action,
                args={**args, "query": candidate},
                round=round_name,
                query_source=plan.get("source"),
            )
        rows = execute_parallel_candidates(action, candidates, args, self._retrieval_context())
        round_duration_ms = round((time.perf_counter() - round_started) * 1000, 1)
        for candidate, value in rows:
            append_task_event(
                self.state.task_id,
                "tool_result",
                step=step,
                phase="DISCOVERY",
                action=action,
                result=json.dumps(value, ensure_ascii=False, indent=2),
                real_network=bool(value.get("real_network", True)),
                round=round_name,
                query=candidate,
                duration_ms=round_duration_ms,
            )
            for result_index, item in enumerate(value.get("results") or [], start=1):
                if not isinstance(item, dict):
                    continue
                extracted = item.get("page_excerpt") or item.get("content") or item.get("abstract") or item.get("snippet") or ""
                append_task_event(
                    self.state.task_id,
                    "content_extract",
                    step=step,
                    phase="EXTRACTION",
                    action=action,
                    query=candidate,
                    result_index=result_index,
                    url=item.get("url", ""),
                    extraction_method="page_excerpt|content|abstract|snippet",
                    extracted_chars=len(str(extracted)),
                    captured_at=item.get("captured_at") or value.get("retrieved_at", ""),
                )
        merged = merge_retrieval_results(
            user_query,
            action,
            rows,
            scope=str(args.get("scope") or ""),
            ranking_strategy=self._strategy()["ranking_strategy"],
        )
        append_task_event(
            self.state.task_id,
            "ranking",
            step=step,
            phase="RANKING",
            action=action,
            data={
                "method": merged.get("ranking_strategy", self._strategy()["ranking_strategy"]),
                "input_count": sum(len(value.get("results") or []) for _, value in rows),
                "output_count": len(merged.get("results") or []),
                "results": [
                    {
                        "rank": item.get("retrieval_rank"),
                        "url": item.get("url", ""),
                        "dedup_key": item.get("dedup_key", ""),
                        "candidate_queries": item.get("candidate_queries") or [],
                        "candidate_ranks": item.get("candidate_ranks") or [],
                        "rerank_score": item.get("rerank_score"),
                    }
                    for item in merged.get("results") or []
                ],
            },
        )
        append_task_event(
            self.state.task_id,
            "candidate_merge",
            step=step,
            phase="DISCOVERY",
            round=round_name,
            action=action,
            data={
                "query_count": len(candidates),
                "queries": candidates,
                "result_count": merged.get("count", 0),
                "round_count": merged.get("round_count", 1),
                "duration_ms": round_duration_ms,
            },
        )
        return merged, candidates, plan

    @staticmethod
    def _citation_recovery_needed(validation: dict, data: dict | None = None) -> bool:
        """Return true only for source-integrity failures worth re-querying."""
        if data and data.get("results") and not data.get("citation_refs"):
            return True
        if data and data.get("evidence_missing_count"):
            return True
        recoverable = {
            "invalid_url",
            "search_result_page",
            "missing_evidence",
            "empty_page",
        }
        return any(
            issue in recoverable
            for row in validation.get("rows") or []
            for issue in row.get("issues") or []
        ) or any(
            str(issue).startswith("inaccessible:")
            for row in validation.get("rows") or []
            for issue in row.get("issues") or []
        )

    def _finish_retrieval_only(self, user_query: str, action: str, data: dict, step: int) -> str:
        """Close a retrieval trace without requiring a model answer.

        This mode is intentionally diagnostic: it measures search, extraction,
        ranking and citation integrity while the RWKV endpoint is unavailable.
        It never fabricates an answer or a reference answer.
        """
        working_data = dict(data)
        citation_refs: list[dict] = []
        citation_validation: dict = {}
        recovery_attempted = False
        risk_policy = risk_context(self.state.run_metadata)
        risk_validation = {
            "validator_version": "risk-policy.v1",
            "high_risk": risk_policy["high_risk"],
            "domain": risk_policy["domain"],
            "valid": True,
            "skipped": True,
            "warning_present": None,
            "issues": ["answer_not_generated_retrieval_only"],
        }
        for attempt in range(2):
            check_time_budget(minimum_seconds=0.2)
            citation_refs = working_data.get("citation_refs") or []
            append_task_event(
                self.state.task_id,
                "synthesis_start",
                step=step + attempt,
                phase="SYNTHESIS",
                action=action,
                attempt=attempt + 1,
                status="skipped",
                reason="retrieval_only",
                evidence_count=len(working_data.get("results") or []),
                round_count=working_data.get("round_count", 1),
            )
            citation_validation = validate_citations(
                citation_refs,
                answer="",
                evidence=working_data.get("results") or [],
                check_remote=get_citation_remote_validation(),
            )
            append_task_event(
                self.state.task_id,
                "citation_validation",
                step=step + attempt,
                phase="VALIDATION",
                attempt=attempt + 1,
                data=citation_validation,
            )
            if attempt == 0 and self._citation_recovery_needed(citation_validation, working_data):
                recovery_attempted = True
                append_task_event(
                    self.state.task_id,
                    "citation_recovery",
                    step=step,
                    phase="RECOVERY",
                    status="scheduled",
                    invalid_count=citation_validation.get("invalid", 0),
                    reason="source_integrity_failure",
                )
                allowed = set(ToolRegistry.metadata(action).get("allowed_args") or ())
                recovery_args = {
                    key: value
                    for key, value in {
                        "scope": "paper",
                        "max_results": 8,
                        "fetch_pages": 3,
                    }.items()
                    if key in allowed
                }
                try:
                    recovery_data, recovery_queries, _ = self._run_rwkv_search_round(
                        user_query,
                        action,
                        recovery_args,
                        step=step + 1,
                        round_name="citation_recovery",
                        previous_query=str(working_data.get("query") or user_query),
                        observation=working_data,
                    )
                    recovery_query = recovery_queries[0] if recovery_queries else user_query
                    working_data = merge_retrieval_results(
                        user_query,
                        action,
                        [
                            (str(working_data.get("query") or user_query), working_data),
                            (recovery_query, recovery_data),
                        ],
                        scope=str(working_data.get("scope") or ""),
                        ranking_strategy=self._strategy()["ranking_strategy"],
                    )
                    working_data["round_count"] = int(data.get("round_count", 1) or 1) + 1
                    self._record_final_ranking(action, working_data, step + attempt, stage="citation_recovery")
                    append_task_event(
                        self.state.task_id,
                        "citation_recovery",
                        step=step,
                        phase="RECOVERY",
                        status="completed",
                        query=recovery_query,
                        result_count=len(recovery_data.get("results") or []),
                    )
                    continue
                except Exception as exc:
                    append_task_event(
                        self.state.task_id,
                        "citation_recovery",
                        step=step,
                        phase="RECOVERY",
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}"[:1000],
                    )
            break
        synthesis = {
            "content": "",
            "mode": "retrieval_only",
            "evidence_count": len(working_data.get("results") or []),
            "citation_refs": citation_refs,
            "context_text": "",
            "selected_evidence": [],
            "context_stats": {"retrieval_only": True, "source_count": len(working_data.get("results") or [])},
        }
        append_task_event(
            self.state.task_id,
            "context_build",
            step=step,
            phase="CONTEXT",
            attempt=2 if recovery_attempted else 1,
            data={
                "context_text": "",
                "selected_evidence": [],
                "context_stats": synthesis["context_stats"],
            },
        )
        append_task_event(
            self.state.task_id,
            "risk_validation",
            step=step,
            phase="VALIDATION",
            attempt=2 if recovery_attempted else 1,
            data=risk_validation,
        )
        self.state.is_finished = True
        self.state.final_result = ""
        append_task_event(
            self.state.task_id,
            "synthesis",
            step=step,
            phase="SYNTHESIS",
            attempt=2 if recovery_attempted else 1,
            content="",
            mode="retrieval_only",
            evidence_count=synthesis["evidence_count"],
            citation_refs=citation_refs,
            citation_validation=citation_validation,
            risk_validation=risk_validation,
            context_text="",
            selected_evidence=[],
            context_stats=synthesis["context_stats"],
            duration_ms=0.0,
        )
        append_task_event(
            self.state.task_id,
            "final",
            status="completed",
            content="",
            action=action,
            mode="retrieval_only",
            round_count=working_data.get("round_count", 1),
            citation_refs=citation_refs,
            citation_validation=citation_validation,
            risk_validation=risk_validation,
            citation_recovery_attempted=recovery_attempted,
            duration_ms=0.0,
        )
        report_path = os.path.join(self.state.task_output_dir, "retrieval_report.jsonl")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "record_type": "retrieval_result",
                "task_id": self.state.task_id,
                "query": user_query,
                "action": action,
                "real_network": bool(working_data.get("real_network", True)),
                "answer": "",
                "answer_mode": "retrieval_only",
                "citation_validation": citation_validation,
                "risk_validation": risk_validation,
                "citation_recovery_attempted": recovery_attempted,
                "retrieval_only": True,
                "data": working_data,
            }, ensure_ascii=False) + "\n")
        return ""

    def _finish_rwkv_retrieval(self, user_query: str, action: str, data: dict, step: int) -> str:
        if self.state.run_metadata.get("retrieval_only") is True:
            return self._finish_retrieval_only(user_query, action, data, step)
        working_data = dict(data)
        synthesis: dict = {}
        citation_validation: dict = {}
        synthesis_duration_ms = 0.0
        recovery_attempted = False

        for attempt in range(2):
            check_time_budget(minimum_seconds=0.2)
            append_task_event(
                self.state.task_id,
                "synthesis_start",
                step=step + attempt,
                phase="SYNTHESIS",
                action=action,
                attempt=attempt + 1,
                evidence_count=len(working_data.get("results") or []),
                round_count=working_data.get("round_count", 1),
            )
            synthesis_started = time.perf_counter()
            synthesis = synthesize_retrieval_answer(
                user_query,
                working_data,
                llm=self.analyzer.llm,
                constraints=self.state.run_metadata,
            )
            synthesis_duration_ms = round((time.perf_counter() - synthesis_started) * 1000, 1)
            answer = synthesis.get("content") or ""
            append_task_event(
                self.state.task_id,
                "context_build",
                step=step + attempt,
                phase="CONTEXT",
                attempt=attempt + 1,
                data={
                    "context_text": synthesis.get("context_text", ""),
                    "selected_evidence": synthesis.get("selected_evidence") or [],
                    "context_stats": synthesis.get("context_stats") or {},
                },
            )
            citation_validation = validate_citations(
                synthesis.get("citation_refs") or working_data.get("citation_refs") or [],
                answer=answer,
                evidence=working_data.get("results") or [],
                check_remote=get_citation_remote_validation(),
            )
            append_task_event(
                self.state.task_id,
                "citation_validation",
                step=step + attempt,
                phase="VALIDATION",
                attempt=attempt + 1,
                data=citation_validation,
            )
            risk_validation = validate_risk_answer(answer, self.state.run_metadata)
            append_task_event(
                self.state.task_id,
                "risk_validation",
                step=step + attempt,
                phase="VALIDATION",
                attempt=attempt + 1,
                data=risk_validation,
            )

            if attempt == 0 and self._citation_recovery_needed(citation_validation, working_data):
                recovery_attempted = True
                append_task_event(
                    self.state.task_id,
                    "citation_recovery",
                    step=step + attempt,
                    phase="RECOVERY",
                    status="scheduled",
                    invalid_count=citation_validation.get("invalid", 0),
                    reason="source_integrity_failure",
                )
                allowed = set(ToolRegistry.metadata(action).get("allowed_args") or ())
                recovery_args = {
                    key: value
                    for key, value in {
                        "scope": "paper",
                        "max_results": 8,
                        "fetch_pages": 3,
                    }.items()
                    if key in allowed
                }
                try:
                    recovery_data, recovery_queries, _ = self._run_rwkv_search_round(
                        user_query,
                        action,
                        recovery_args,
                        step=step + 1,
                        round_name="citation_recovery",
                        previous_query=str(working_data.get("query") or user_query),
                        observation=working_data,
                    )
                    recovery_query = recovery_queries[0] if recovery_queries else user_query
                    working_data = merge_retrieval_results(
                        user_query,
                        action,
                        [
                            (str(working_data.get("query") or user_query), working_data),
                            (recovery_query, recovery_data),
                        ],
                        scope=str(working_data.get("scope") or ""),
                        ranking_strategy=self._strategy()["ranking_strategy"],
                    )
                    working_data["round_count"] = int(data.get("round_count", 1) or 1) + 1
                    self._record_final_ranking(action, working_data, step + attempt, stage="citation_recovery")
                    append_task_event(
                        self.state.task_id,
                        "citation_recovery",
                        step=step + attempt,
                        phase="RECOVERY",
                        status="completed",
                        query=recovery_query,
                        result_count=len(recovery_data.get("results") or []),
                    )
                    continue
                except Exception as exc:
                    append_task_event(
                        self.state.task_id,
                        "citation_recovery",
                        step=step + attempt,
                        phase="RECOVERY",
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}"[:1000],
                    )
            break

        self.state.is_finished = True
        self.state.final_result = synthesis.get("content") or ""
        risk_validation = validate_risk_answer(self.state.final_result, self.state.run_metadata)
        if citation_validation.get("invalid"):
            final_status = "completed_with_citation_warnings"
        elif not risk_validation.get("valid", True):
            final_status = "completed_with_risk_warnings"
        else:
            final_status = "completed"
        append_task_event(
            self.state.task_id,
            "synthesis",
            step=step,
            phase="SYNTHESIS",
            attempt=2 if recovery_attempted else 1,
            content=self.state.final_result,
            mode=synthesis.get("mode"),
            evidence_count=synthesis.get("evidence_count", 0),
            citation_refs=synthesis.get("citation_refs") or [],
            citation_validation=citation_validation,
            risk_validation=risk_validation,
            prompt=synthesis.get("prompt", ""),
            model_output=synthesis.get("model_output", ""),
            repair_prompt=synthesis.get("repair_prompt", ""),
            repair_output=synthesis.get("repair_output", ""),
            context_text=synthesis.get("context_text", ""),
            selected_evidence=synthesis.get("selected_evidence") or [],
            context_stats=synthesis.get("context_stats") or {},
            duration_ms=synthesis_duration_ms,
        )
        append_task_event(
            self.state.task_id,
            "final",
            status=final_status,
            content=self.state.final_result,
            action=action,
            mode=synthesis.get("mode"),
            round_count=working_data.get("round_count", 1),
            citation_refs=synthesis.get("citation_refs") or [],
            citation_validation=citation_validation,
            risk_validation=risk_validation,
            citation_recovery_attempted=recovery_attempted,
            duration_ms=synthesis_duration_ms,
        )
        report_path = os.path.join(self.state.task_output_dir, "retrieval_report.jsonl")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "record_type": "retrieval_result",
                "task_id": self.state.task_id,
                "query": user_query,
                "action": action,
                "real_network": bool(working_data.get("real_network", True)),
                "answer": self.state.final_result,
                "answer_mode": synthesis.get("mode"),
                "citation_validation": citation_validation,
                "risk_validation": risk_validation,
                "citation_recovery_attempted": recovery_attempted,
                "data": working_data,
            }, ensure_ascii=False) + "\n")
        return self.state.final_result

    def run(
        self,
        user_query: str,
        task_id: str = None,
        run_metadata: dict | None = None,
    ) -> str:
        check_time_budget(minimum_seconds=0.2)
        self.state.task_id = task_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state.run_metadata = dict(run_metadata or {})
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
        
        debug_dir = DATA_PIPELINE.get("debug_directory", "./data/debug_slm")
        os.makedirs(debug_dir, exist_ok=True)
        session_id = self.state.task_id
        trace_file = os.path.join(debug_dir, f"DeepResearch_Trace_{session_id}.md")
        
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(f"# Deep Research 执行追踪日志\n\n**启动时间**: {session_id}\n**用户指令**: {user_query}\n\n---\n\n")

        PHASE_MAP = {
            "DISCOVERY": "探测与发现",
            "EXTRACTION": "深度提取",
            "SYNTHESIS": "聚合适成"
        }
        ACTION_MAP = {
            "search_local_file": "检索本地工作区文件",
            "preview_document_content": "试读文档摘要",
            "delegate_to_small_models": "调度小模型提炼全文",
            "query_checkpoint_via_slm": "执行记忆区细节捞针",
            "batch_process_individual_reports": "归档单篇独立报告",
            "compress_working_memory": "执行工作记忆压缩",
            "generate_final_aggregate_reports": "排版聚合最终研报",
            "execute_web_search": "执行互联网检索",
            "search_web_wigolo": "通过本地 wigolo 检索互联网并保留证据",
            "search_papers": "检索论文数据源",
            "finish_task": "任务逻辑闭环退出",
            "none": "思考下一步方向"
        }

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

        push_progress("🚀 正在初始化环境，构建工作区内存与检索本地文件...")

        try:
            initial_files_json = ToolRegistry.execute("search_local_file", {"keyword": ""}, {})
            initial_files = json.loads(initial_files_json)
            for i, p in enumerate(initial_files):
                fid = f"DOC_{i+1}"
                self.state.id_to_path[fid] = p
                self.state.path_to_id[p] = fid
            self.state.last_feedback = f"系统就绪，目录中发现 {len(initial_files)} 份可用文件。"
            push_progress(f"环境就绪：感知到 {len(initial_files)} 份文件。\n")
        except Exception:
            self.state.last_feedback = "目录为空。"
            push_progress(f"环境就绪：本地工作区目录为空。\n")
            
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

        # Removed legacy one-shot retrieval branch.  Search actions are handled
        # only by the RWKV candidate/multi-hop path above.
        if direct_plan.get("action") == "__legacy_removed__":
            step_count = 1
            context_text = self.state.to_markdown_context()
            try:
                analysis = self.analyzer.analyze_intent_and_phase(user_query, context_text)
            except Exception as exc:  # The deterministic retrieval route remains usable.
                analysis = {
                    "intent_mode": "DEEP_RESEARCH",
                    "entity_audit": {},
                    "refined_query": user_query,
                    "missing_information": f"local RWKV analysis unavailable: {exc}",
                    "next_phase": "DISCOVERY",
                }
            analysis["execution_route"] = "direct_keyless_retrieval"
            analysis["local_model"] = "rwkv7-g1h-1.5b-20260710-ctx10240"
            append_task_event(
                self.state.task_id,
                "analysis",
                step=step_count,
                phase="DISCOVERY",
                context_snapshot=context_text,
                data=analysis,
            )
            self.state.refined_query = user_query

            action = direct_plan["action"]
            args = dict(direct_plan.get("args") or {})
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=step_count,
                phase="DISCOVERY",
                query=user_query,
                router_hint=direct_plan.get("router", "deterministic_retrieval"),
            )
            append_task_event(
                self.state.task_id,
                "plan",
                step=step_count,
                phase="DISCOVERY",
                action=action,
                args=args,
                router=direct_plan.get("router", "deterministic_retrieval"),
            )
            append_task_event(
                self.state.task_id,
                "tool_call",
                step=step_count,
                phase="DISCOVERY",
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
            try:
                structured_result = json.loads(result)
            except (TypeError, json.JSONDecodeError):
                structured_result = {"raw": result}
            self.state.last_feedback = f"[{action}] retrieval result:\n{result}"
            append_task_event(
                self.state.task_id,
                "tool_result",
                step=step_count,
                phase="DISCOVERY",
                action=action,
                result=result,
                real_network=bool(structured_result.get("real_network", True)),
            )

            append_task_event(
                self.state.task_id,
                "synthesis_start",
                step=step_count + 1,
                phase="SYNTHESIS",
                action=action,
                evidence_count=len(structured_result.get("results") or []),
            )
            synthesis = synthesize_retrieval_answer(
                user_query,
                structured_result,
                llm=self.analyzer.llm,
            )
            duration_ms = round((time.perf_counter() - started_at) * 1000, 1)
            append_task_event(
                self.state.task_id,
                "synthesis",
                step=step_count + 1,
                phase="SYNTHESIS",
                content=synthesis.get("content", ""),
                mode=synthesis.get("mode"),
                evidence_count=synthesis.get("evidence_count", 0),
                duration_ms=duration_ms,
                citation_refs=synthesis.get("citation_refs") or [],
            )
            self.state.is_finished = True
            self.state.final_result = synthesis.get("content") or result
            append_task_event(
                self.state.task_id,
                "final",
                status="completed",
                content=self.state.final_result,
                action=action,
                duration_ms=duration_ms,
            )
            report_path = os.path.join(self.state.task_output_dir, "retrieval_report.jsonl")
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "record_type": "retrieval_result",
                            "task_id": self.state.task_id,
                            "query": user_query,
                            "action": action,
                            "router": direct_plan.get("router", "deterministic_retrieval"),
                            "real_network": bool(structured_result.get("real_network", True)),
                            "answer": self.state.final_result,
                            "answer_mode": synthesis.get("mode"),
                            "duration_ms": duration_ms,
                            "data": structured_result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return self.state.final_result

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
