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
    task_point_id: str = "",
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
            task_point_id=task_point_id,
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
    decision. A minimal two-action RWKV review sees the same evidence context
    as the Writer and may request one new planner session. Duplicate and
    no-progress recovery also rebuild the Planner before a resource boundary.
    """

    state = owner.state
    if not state.retrieval.claims.claim_ids():
        # Kept as provenance metadata for compatibility.  It never decides
        # whether retrieval may finish or which sources enter the final answer.
        state.retrieval.claims.initialize(task_plan, user_query)
    rounds = state.retrieval.rounds
    phase = "DISCOVERY"
    consecutive_no_progress = 0
    duplicate_recovery_count = 0
    planner_protocol_recovery_count = 0
    max_consecutive_no_progress = 4
    max_duplicate_recoveries = 2
    max_planner_protocol_recoveries = 2
    append_task_event(
        state.task_id,
        "research_loop_started",
        step=0,
        phase=phase,
        max_steps=max_steps,
        shared_state=True,
        decision_owner="rwkv",
        model=model_profile,
        completion_policy="rwkv_finish_or_resource_boundary",
    )

    last_action = "rwkv_research"

    def rwkv_requests_more_retrieval(*, step: int, trigger: str) -> bool:
        """Run one revision-scoped RWKV binary review before synthesis."""

        review = owner._cross_validate_if_evidence_changed(
            user_query,
            task_plan,
            step=step,
            trigger=trigger,
        )
        if not isinstance(review, dict) or str(review.get("decision") or "") != "replan":
            return False
        feedback = {
            "status": "cross_validation_replan",
            "error_class": "rwkv_requested_more_retrieval",
            "evidence_review": review,
            "retrieval_ledger": owner._retrieval_ledger.observation(limit=8),
            "results": [],
        }
        _request_recovery_turn(
            owner,
            feedback,
            user_query,
            phase,
            step=step,
        )
        return True

    def review_new_evidence_revision(*, step: int) -> str:
        """Let RWKV close or replan immediately after material evidence.

        Waiting until the Planner happened to emit finish_task allowed a
        successful search to be followed by several copied duplicate queries.
        This review is still only the established RWKV binary decision: the
        controller neither judges evidence sufficiency nor authors a query.
        """

        review = owner._cross_validate_if_evidence_changed(
            user_query,
            task_plan,
            step=step,
            trigger="evidence_revision",
        )
        if not isinstance(review, dict):
            return ""
        decision = str(review.get("decision") or "").casefold()
        if decision != "replan":
            return decision if decision == "finish" else ""
        feedback = {
            "status": "cross_validation_replan",
            "error_class": "rwkv_requested_more_retrieval",
            "evidence_review": review,
            "retrieval_ledger": owner._retrieval_ledger.observation(limit=8),
            "results": [],
        }
        _request_recovery_turn(
            owner,
            feedback,
            user_query,
            phase,
            step=step,
        )
        return "replan"

    for step in range(1, max(1, int(max_steps)) + 1):
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
        task_point_id = str(plan.get("task_point_id") or "").strip()
        last_action = action or last_action

        append_task_event(
            state.task_id,
            "model_tool_decision",
            step=step,
            phase=phase,
            action=action,
            args=args,
            task_point_id=task_point_id,
            call_id=plan.get("call_id", ""),
            raw_model_output=plan.get("raw_model_output", ""),
            planner_error=plan.get("planner_error", ""),
            sampling_stage=plan.get("sampling_stage", "planner"),
            sampling_temperature=plan.get("sampling_temperature"),
            sampling_seed=plan.get("sampling_seed"),
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
            return owner._complete_model_tool_loop(
                user_query,
                last_action,
                rounds,
                step,
                termination_reason="planner_protocol_resource_stop",
            )

        if action == "finish_task":
            if rwkv_requests_more_retrieval(step=step, trigger="planner_finish"):
                consecutive_no_progress = 0
                duplicate_recovery_count = 0
                continue
            return owner._complete_model_tool_loop(
                user_query,
                action,
                rounds,
                step,
                termination_reason="rwkv_planner_finish",
            )

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
                return owner._complete_model_tool_loop(
                    user_query,
                    action or last_action,
                    rounds,
                    step,
                    termination_reason="resource_no_progress",
                )
            continue

        exact_duplicate = owner._retrieval_ledger.request_status(
            action,
            args,
            task_point_id=task_point_id,
        )
        if action == "web_search" and str(args.get("query") or "").strip():
            exact_query = owner._retrieval_ledger.query_status(
                args.get("query"),
                task_point_id=task_point_id,
                threshold=1.0,
            )
            if exact_query.get("exact_match"):
                exact_duplicate = exact_duplicate or exact_query
        duplicate = exact_duplicate
        if _is_retrieval_tool(action) and duplicate:
            duplicate_query = str(args.get("query") or args.get("url") or user_query)
            duplicate_kind = "exact_duplicate_request"
            duplicate_record = owner._retrieval_ledger.record_duplicate_block(
                duplicate_query,
                step=step,
                task_point_id=task_point_id,
            )
            frozen_path = state.retrieval.freeze_path(
                duplicate_query,
                action=action,
                arguments=args,
                task_point_id=task_point_id,
                step=step,
                reason=duplicate_kind,
            )
            feedback = {
                "status": "no_new_evidence",
                "error_class": duplicate_kind,
                "message": (
                    "This exact retrieval request already ran for this task record. "
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
                if rwkv_requests_more_retrieval(
                    step=step,
                    trigger="duplicate_resource_boundary",
                ):
                    consecutive_no_progress = 0
                    duplicate_recovery_count = 0
                    continue
                return owner._complete_model_tool_loop(
                    user_query,
                    action,
                    rounds,
                    step,
                    termination_reason="resource_duplicate_limit",
                )
            _request_recovery_turn(
                owner,
                feedback,
                user_query,
                phase,
                step=step,
            )
            continue

        tool_context = owner._agentic_tool_context()
        if _is_retrieval_tool(action) and task_point_id:
            # Optional model-authored trace metadata only.  The backend may use
            # it as a soft focus hint but absence never triggers another model call.
            tool_context = {**tool_context, "task_point_id": task_point_id}
        raw_result = ToolRegistry.execute(action, args, tool_context, phase=None)
        result = _as_dict(raw_result)
        result = attach_result_object_contract(
            result,
            action=action,
            arguments=args,
            task_record_id=task_point_id,
            task_plan=task_plan,
        )
        owner._retrieval_ledger.record_request(
            action,
            args,
            result,
            step=step,
            task_point_id=task_point_id,
        )

        material_changed = False
        if _is_retrieval_tool(action):
            query = str(args.get("query") or args.get("url") or user_query)
            state_delta = state.retrieval.record_query(
                query,
                result,
                step=step,
                task_point_id=task_point_id,
                strategy="rwkv_selected",
            )
            result = owner._record_retrieval_progress(
                result,
                query=query,
                step=step,
                action=action,
                phase=phase,
                task_point_id=task_point_id,
            )
            ledger_delta = result.get("retrieval_delta") or {}
            material_changed = bool(
                state_delta.get("material_changed")
                or ledger_delta.get("material_changed")
            )
            state.run_metadata["claim_ledger"] = state.retrieval.claims.snapshot()
        else:
            _record_deterministic_result(
                owner,
                action,
                result,
                step=step,
                task_point_id=task_point_id,
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
            owner.planner.mark_replan_progress(task_point_id)
            if _is_retrieval_tool(action):
                evidence_decision = review_new_evidence_revision(step=step)
                if evidence_decision == "finish":
                    return owner._complete_model_tool_loop(
                        user_query,
                        action,
                        rounds,
                        step,
                        termination_reason="rwkv_cross_validation_finish",
                    )
                if evidence_decision == "replan":
                    continue
        else:
            consecutive_no_progress += 1
            if consecutive_no_progress >= max_consecutive_no_progress:
                if rwkv_requests_more_retrieval(
                    step=step,
                    trigger="no_progress_resource_boundary",
                ):
                    consecutive_no_progress = 0
                    duplicate_recovery_count = 0
                    continue
                return owner._complete_model_tool_loop(
                    user_query,
                    action,
                    rounds,
                    step,
                    termination_reason="resource_no_progress",
                )
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

    return owner._complete_model_tool_loop(
        user_query,
        last_action,
        rounds,
        max_steps,
        termination_reason="resource_max_steps",
    )


__all__ = ["run_unified_research_loop"]
