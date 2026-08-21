"""One RWKV-owned research loop with deterministic protocol boundaries only.

RWKV owns task decomposition, tool choice, query, source target, replanning and
the final answer. The controller executes calls, keeps shared state, blocks an
already executed request and enforces resource limits. Before synthesis, one
revision-scoped RWKV binary review may either accept the retained evidence or
request another planner session; deterministic code never makes that semantic
decision and never edits the Writer's answer.
"""

from __future__ import annotations

import json
from typing import Any

from retrieval_plugins import is_error, plugin_environment_snapshot
from agent.retrieval_object_contract import attach_result_object_contract
from tools.registry import ToolRegistry
from utils.task_events import append_task_event
from utils.task_manager import is_task_stopped
from utils.time_budget import check_time_budget


class EvidenceReviewDecisionError(RuntimeError):
    """The current evidence revision has no valid RWKV finish/replan decision."""


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "error_class": "tool_protocol",
                "message": value[:1000],
                "results": [],
            }
        if isinstance(parsed, dict):
            return parsed
    return {
        "status": "error",
        "error_class": "tool_protocol",
        "message": "tool returned a non-object result",
        "results": [],
    }


def _is_retrieval_tool(action: str) -> bool:
    metadata = ToolRegistry.metadata(action)
    return str(metadata.get("retrieval_role") or "").casefold() in {
        "discovery",
        "evidence",
        "retrieval",
    } or action in {"web_search", "connector_lookup"}


def _record_deterministic_result(
    owner: Any,
    action: str,
    result: dict[str, Any],
    *,
    step: int,
    task_record_id: str = "",
) -> None:
    if str(result.get("status") or "").casefold() != "ok":
        return
    if action == "calculator":
        owner._arithmetic_results.append(dict(result))
    elif action == "current_time":
        owner._time_results.append(dict(result))
    elif action == "date_diff":
        owner._calculation_results.append(dict(result))
    if action in {"calculator", "current_time", "date_diff"}:
        owner.state.retrieval.record_deterministic_result(
            action,
            result,
            step=step,
            task_record_id=task_record_id,
        )
    append_task_event(
        owner.state.task_id,
        "calculation_result",
        step=step,
        phase="CALCULATION",
        action=action,
        data=dict(result),
        decision_owner="rwkv",
    )


def _request_recovery_turn(
    owner: Any,
    feedback: dict[str, Any],
    user_query: str,
    phase: str,
    *,
    step: int,
) -> None:
    """Freeze the stalled transcript and ask RWKV in a rebuilt session."""

    request_recovery = getattr(owner.planner, "request_recovery_turn", None)
    if callable(request_recovery):
        replan_count = owner.state.retrieval.record_replan()
        request_recovery(
            feedback,
            user_query=user_query,
            env_context=owner.state.to_retrieval_context(),
            phase=phase,
        )
        append_task_event(
            owner.state.task_id,
            "planner_session_rebuilt",
            step=step,
            phase="REPLAN",
            reason=str(feedback.get("error_class") or "retrieval_no_progress"),
            replan_count=replan_count,
            shared_state=owner.state.retrieval.routing_snapshot(),
            decision_owner="rwkv",
            controller_query_generated=False,
        )


def run_unified_research_loop(
    owner: Any,
    user_query: str,
    model_profile: dict[str, Any],
    task_plan: dict[str, Any],
    max_steps: int,
) -> str:
    """Execute one bounded RWKV-owned tool loop.

    The Planner already sees retained evidence and owns the first finish
    decision. A revision-scoped RWKV Evidence Review sees the same context as
    the Writer and makes only a binary continue-or-write decision. A continue
    decision rebuilds the Planner from authoritative state. Duplicate and
    no-progress recovery also rebuild the Planner before a resource boundary.
    """

    state = owner.state
    if not state.retrieval.evidence_ledger.task_record_ids():
        # Kept as provenance metadata for compatibility.  It never decides
        # whether retrieval may finish or which sources enter the final answer.
        state.retrieval.evidence_ledger.initialize(task_plan, user_query)
    rounds = state.retrieval.rounds
    phase = "DISCOVERY"
    consecutive_no_progress = 0
    duplicate_recovery_count = 0
    evidence_review_replan_count = 0
    planner_protocol_recovery_count = 0
    max_consecutive_no_progress = 4
    max_duplicate_recoveries = 2
    max_evidence_review_replans = 2
    max_planner_protocol_recoveries = 2
    step_limit = max(1, int(max_steps))
    append_task_event(
        state.task_id,
        "research_loop_started",
        step=0,
        phase=phase,
        max_steps=max_steps,
        shared_state=True,
        decision_owner="rwkv",
        model=model_profile,
        completion_policy="current_evidence_digest_requires_rwkv_finish",
    )

    last_action = "rwkv_research"

    def apply_review(
        review: dict[str, Any] | None,
        *,
        step: int,
        action: str,
        trigger: str,
    ) -> tuple[str, str]:
        """Apply an existing review without reviewing the revision again."""

        nonlocal evidence_review_replan_count
        nonlocal consecutive_no_progress, duplicate_recovery_count
        nonlocal planner_protocol_recovery_count, step_limit

        decision = (
            str(review.get("decision") or "").casefold()
            if isinstance(review, dict)
            else ""
        )
        if decision == "replan":
            evidence_review_replan_count += 1
            feedback = {
                "status": "evidence_review_replan",
                "error_class": "rwkv_structured_evidence_gap",
                "evidence_review": review,
                "retrieval_ledger": owner._retrieval_ledger.observation(limit=8),
                "results": [],
            }
            _request_recovery_turn(
                owner, feedback, user_query, phase, step=step
            )
            consecutive_no_progress = 0
            duplicate_recovery_count = 0
            planner_protocol_recovery_count = 0
            # Resource allowance only; the gap and next route stay RWKV-owned.
            step_limit = max(step_limit, step + 3)
            return "continue", ""
        if decision == "finish":
            authorize_writer = getattr(
                owner,
                "_evidence_review_authorizes_writer",
                None,
            )
            if not callable(authorize_writer) or not authorize_writer(
                review,
                user_query,
                task_plan,
                step=step,
                trigger=trigger,
            ):
                return "invalid", ""
            return "finished", owner._complete_model_tool_loop(
                user_query,
                action,
                rounds,
                step,
                termination_reason="rwkv_evidence_review_finish",
            )
        return "none", ""

    def finalize_or_replan(
        *,
        step: int,
        trigger: str,
        termination_reason: str,
        action: str,
    ) -> tuple[bool, str]:
        """Route every synthesis attempt through the revision review."""

        terminal_review = evidence_review_replan_count >= max_evidence_review_replans
        if terminal_review:
            # Retrieval is resource-bounded, but entering the Writer is still
            # an explicit RWKV action.  A fresh one-action terminal review is
            # intentionally distinct from reusing the cached ``replan`` for
            # the same evidence digest.
            review = owner._review_evidence_if_changed(
                user_query,
                task_plan,
                step=step,
                trigger=f"{trigger}:terminal_resource_boundary",
                terminal=True,
            )
            if (
                not isinstance(review, dict)
                or str(review.get("decision") or "").casefold() != "finish"
            ):
                disposition = "terminal_review_invalid"
                answer = ""
            else:
                disposition, answer = apply_review(
                    review,
                    step=step,
                    action=action,
                    trigger=trigger,
                )
        else:
            review = owner._review_evidence_if_changed(
                user_query,
                task_plan,
                step=step,
                trigger=trigger,
            )
            if review is None:
                current_decision = getattr(
                    owner,
                    "_current_evidence_review_decision",
                    None,
                )
                if callable(current_decision):
                    review = current_decision(
                        user_query,
                        task_plan,
                        step=step,
                        trigger=trigger,
                    )
            disposition, answer = apply_review(
                review,
                step=step,
                action=action,
                trigger=trigger,
            )
        if disposition == "continue":
            return True, ""
        if disposition == "finished":
            return False, answer
        decision = (
            str(review.get("decision") or "").casefold()
            if isinstance(review, dict)
            else ""
        )
        error_class = (
            str(review.get("error_class") or "")
            if isinstance(review, dict)
            else ""
        )
        append_task_event(
            state.task_id,
            "evidence_review_gate_failed",
            step=step,
            phase="VALIDATION",
            trigger=trigger,
            requested_termination_reason=termination_reason,
            disposition=disposition,
            decision=decision,
            error_class=error_class or "evidence_review_decision",
            evidence_review_replan_count=evidence_review_replan_count,
            max_evidence_review_replans=max_evidence_review_replans,
            decision_owner="rwkv",
        )
        raise EvidenceReviewDecisionError(
            "Writer requires a valid RWKV finish decision bound to the current "
            f"evidence digest (trigger={trigger}, disposition={disposition}, "
            f"decision={decision or 'missing'}, error_class={error_class or 'none'})"
        )

    step = 0
    while True:
        if step >= step_limit:
            should_continue, final_answer = finalize_or_replan(
                step=step,
                trigger="max_steps_resource_boundary",
                termination_reason="resource_max_steps",
                action=last_action,
            )
            if should_continue:
                continue
            return final_answer
        step += 1
        check_time_budget(minimum_seconds=0.2)
        if is_task_stopped(state.task_id):
            state.is_finished = True
            state.final_result = ""
            append_task_event(state.task_id, "stopped", step=step)
            return ""

        plan = owner.planner.plan_next_action(
            user_query,
            {},
            state.to_retrieval_context(),
            phase,
        )
        action = str(plan.get("action") or "").strip()
        args = dict(plan.get("args") or {}) if isinstance(plan.get("args"), dict) else {}
        task_record_id = str(plan.get("task_record_id") or "").strip()
        last_action = action or last_action

        append_task_event(
            state.task_id,
            "model_tool_decision",
            step=step,
            phase=phase,
            action=action,
            args=args,
            task_record_id=task_record_id,
            call_id=plan.get("call_id", ""),
            raw_model_output=plan.get("raw_model_output", ""),
            planner_error=plan.get("planner_error", ""),
            protocol_input_format=plan.get("protocol_input_format", "invalid"),
            protocol_normalized=bool(plan.get("protocol_normalized", False)),
            sampling_stage=plan.get("sampling_stage", "planner"),
            sampling_temperature=plan.get("sampling_temperature"),
            sampling_seed=plan.get("sampling_seed"),
            task_record_binding_method=plan.get("task_record_binding_method", "not_required"),
            task_record_binding_raw_model_output=plan.get(
                "task_record_binding_raw_model_output", ""
            ),
            task_record_binding_error=plan.get("task_record_binding_error", ""),
            task_record_binding_temperature=plan.get(
                "task_record_binding_temperature"
            ),
            decision_owner="rwkv",
            gateway_override=False,
            model=model_profile,
        )

        if plan.get("planner_error"):
            owner._model_protocol_failure = True
            planner_protocol_recovery_count += 1
            if planner_protocol_recovery_count < max_planner_protocol_recoveries:
                feedback = {
                    "status": "protocol_error",
                    "error_class": "planner_protocol_error",
                    "message": str(plan.get("planner_error") or "")[:1000],
                    "raw_model_output": str(plan.get("raw_model_output") or "")[:1000],
                    "allowed_tools": ToolRegistry.model_visible_names(),
                    "results": [],
                }
                state.last_feedback = json.dumps(feedback, ensure_ascii=False)
                _request_recovery_turn(
                    owner,
                    feedback,
                    user_query,
                    phase,
                    step=step,
                )
                continue
            should_continue, final_answer = finalize_or_replan(
                step=step,
                trigger="planner_protocol_resource_boundary",
                termination_reason="planner_protocol_resource_stop",
                action=last_action,
            )
            if should_continue:
                continue
            return final_answer

        if action == "finish_task":
            should_continue, final_answer = finalize_or_replan(
                step=step,
                trigger="planner_finish",
                termination_reason="rwkv_planner_finish",
                action=action,
            )
            if should_continue:
                continue
            return final_answer

        if not ToolRegistry.has(action):
            feedback = {
                "status": "error",
                "error_class": "unknown_tool",
                "message": f"tool is not registered: {action}",
                "allowed_tools": ToolRegistry.model_visible_names(),
                "results": [],
            }
            owner._model_protocol_failure = True
            owner.planner.observe_tool_result(feedback)
            state.last_feedback = json.dumps(feedback, ensure_ascii=False)
            append_task_event(
                state.task_id,
                "tool_result",
                step=step,
                phase=phase,
                action=action,
                result=feedback,
                execution_status="protocol_error",
                decision_owner="rwkv",
            )
            consecutive_no_progress += 1
            if consecutive_no_progress >= max_consecutive_no_progress:
                should_continue, final_answer = finalize_or_replan(
                    step=step,
                    trigger="unknown_tool_resource_boundary",
                    termination_reason="resource_no_progress",
                    action=action or last_action,
                )
                if should_continue:
                    continue
                return final_answer
            continue

        duplicate = owner._retrieval_ledger.request_status(
            action,
            args,
            task_record_id=task_record_id,
        )
        if _is_retrieval_tool(action) and duplicate:
            duplicate_query = str(args.get("query") or args.get("url") or user_query)
            duplicate_kind = "exact_duplicate_request"
            duplicate_record = owner._retrieval_ledger.record_duplicate_block(
                duplicate_query,
                step=step,
                task_record_id=task_record_id,
            )
            frozen_path = state.retrieval.freeze_path(
                duplicate_query,
                action=action,
                arguments=args,
                task_record_id=task_record_id,
                step=step,
                reason=duplicate_kind,
            )
            feedback = {
                "status": "no_new_evidence",
                "error_class": duplicate_kind,
                "message": (
                    "This retrieval route already ran for this task record. "
                    "Choose a materially different query/source, use another tool, "
                    "or finish from the retained sources."
                ),
                "request": {"action": action, "arguments": args},
                "repeat_count": duplicate_record.get("blocked_count", 1),
                "frozen_path": frozen_path,
                "retrieval_ledger": owner._retrieval_ledger.observation(limit=8),
                "results": [],
            }
            state.last_feedback = json.dumps(feedback, ensure_ascii=False)
            owner.planner.observe_tool_result(feedback)
            append_task_event(
                state.task_id,
                "tool_result",
                step=step,
                phase=phase,
                action=action,
                result=feedback,
                execution_status=duplicate_kind,
                decision_owner="rwkv",
            )
            duplicate_recovery_count += 1
            consecutive_no_progress += 1
            if (
                duplicate_recovery_count >= max_duplicate_recoveries
                or consecutive_no_progress >= max_consecutive_no_progress
            ):
                should_continue, final_answer = finalize_or_replan(
                    step=step,
                    trigger="duplicate_resource_boundary",
                    termination_reason="resource_duplicate_limit",
                    action=action,
                )
                if should_continue:
                    continue
                return final_answer
            _request_recovery_turn(
                owner,
                feedback,
                user_query,
                phase,
                step=step,
            )
            continue

        tool_context = owner._agentic_tool_context()
        if _is_retrieval_tool(action) and task_record_id:
            # Optional model-authored trace metadata only.  The backend may use
            # it as a soft focus hint but absence never triggers another model call.
            tool_context = {**tool_context, "task_record_id": task_record_id}
        raw_result = ToolRegistry.execute(action, args, tool_context, phase=None)
        result = _as_dict(raw_result)
        result = attach_result_object_contract(
            result,
            action=action,
            arguments=args,
            task_record_id=task_record_id,
            task_plan=task_plan,
        )
        owner._retrieval_ledger.record_request(
            action,
            args,
            result,
            step=step,
            task_record_id=task_record_id,
        )

        material_changed = False
        if _is_retrieval_tool(action):
            query = str(args.get("query") or args.get("url") or user_query)
            state_delta = state.retrieval.record_query(
                query,
                result,
                step=step,
                task_record_id=task_record_id,
                strategy="rwkv_selected",
                action=action,
                arguments=args,
            )
            result = owner._record_retrieval_progress(
                result,
                query=query,
                step=step,
                action=action,
                arguments=args,
                phase=phase,
                task_record_id=task_record_id,
            )
            ledger_delta = result.get("retrieval_delta") or {}
            material_changed = bool(
                state_delta.get("material_changed")
                or ledger_delta.get("material_changed")
            )
            state.run_metadata["evidence_ledger"] = state.retrieval.evidence_ledger.snapshot()
        else:
            _record_deterministic_result(
                owner,
                action,
                result,
                step=step,
                task_record_id=task_record_id,
            )
            material_changed = str(result.get("status") or "").casefold() == "ok"

        if is_error(result):
            owner._model_protocol_failure = owner._model_protocol_failure or str(
                result.get("error_class") or ""
            ).casefold() in {"tool_protocol", "unknown_tool"}

        result["shared_research_state"] = state.retrieval.routing_snapshot()
        state.last_feedback = json.dumps(
            {
                "status": result.get("status", ""),
                "tool": action,
                "query": str(args.get("query") or args.get("url") or ""),
                "error_class": result.get("error_class", ""),
                "message": result.get("message", ""),
                "result_count": len(result.get("results") or []),
                "material_changed": material_changed,
                "real_network": bool(
                    result.get("real_network", _is_retrieval_tool(action))
                ),
            },
            ensure_ascii=False,
        )
        owner.planner.observe_tool_result(result)
        append_task_event(
            state.task_id,
            "tool_result",
            step=step,
            phase=phase,
            action=action,
            result=raw_result,
            retrieval_request=result.get("retrieval_request") or {},
            source_objects=[
                row.get("source_object") or {}
                for row in result.get("results") or []
                if isinstance(row, dict)
            ],
            real_network=bool(
                result.get("real_network", _is_retrieval_tool(action))
            ),
            decision_owner="rwkv",
            retrieval_environment=plugin_environment_snapshot(),
            shared_state=state.retrieval.routing_snapshot(),
        )

        if material_changed:
            consecutive_no_progress = 0
            duplicate_recovery_count = 0
            planner_protocol_recovery_count = 0
            owner.planner.mark_replan_progress(task_record_id)
            if (
                _is_retrieval_tool(action)
                and evidence_review_replan_count < max_evidence_review_replans
            ):
                revision_review = owner._review_evidence_if_changed(
                    user_query,
                    task_plan,
                    step=step,
                    trigger="evidence_revision",
                )
                disposition, answer = apply_review(
                    revision_review,
                    step=step,
                    action=action,
                    trigger="evidence_revision",
                )
                if disposition == "continue":
                    continue
                if disposition == "finished":
                    return answer
        else:
            consecutive_no_progress += 1
            if consecutive_no_progress >= max_consecutive_no_progress:
                should_continue, final_answer = finalize_or_replan(
                    step=step,
                    trigger="no_progress_resource_boundary",
                    termination_reason="resource_no_progress",
                    action=action,
                )
                if should_continue:
                    continue
                return final_answer
            # One empty or unchanged retrieval is ordinary evidence for the
            # next independent Planner decision. Rebuild only after sustained
            # no progress; rebuilding every empty search multiplied model
            # calls and discarded useful routing state.
            if _is_retrieval_tool(action) and consecutive_no_progress == 2:
                recovery_feedback = {
                    **result,
                    "error_class": str(
                        result.get("error_class") or "retrieval_no_progress"
                    ),
                    "consecutive_no_progress": consecutive_no_progress,
                }
                _request_recovery_turn(
                    owner,
                    recovery_feedback,
                    user_query,
                    phase,
                    step=step,
                )

__all__ = ["run_unified_research_loop"]
