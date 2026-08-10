"""One RWKV-owned research loop with no controller-authored retrieval route.

RWKV chooses every tool, query, source target, and finish point.  The
controller only validates the tool protocol, executes the exact call, stores
results, and returns a bounded observation to RWKV.
"""

from __future__ import annotations

import json
from typing import Any

from retrieval_plugins import is_error, plugin_environment_snapshot
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


def _review_missing_point_ids(
    review: dict[str, Any] | None,
    task_plan: dict[str, Any],
) -> set[str]:
    """Return exact model-plan IDs named missing by the RWKV reviewer."""

    planned_ids = {
        str(point.get("id") or "").strip()
        for point in task_plan.get("atomic_points") or []
        if isinstance(point, dict) and str(point.get("id") or "").strip()
    }
    return {
        str(value).strip()
        for value in (review or {}).get("missing_points") or []
        if str(value).strip() in planned_ids
    }


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


def run_unified_research_loop(
    owner: Any,
    user_query: str,
    model_profile: dict[str, Any],
    task_plan: dict[str, Any],
    max_steps: int,
) -> str:
    """Execute RWKV decisions without a gateway or controller completion gate."""

    state = owner.state
    if not state.retrieval.claims.claim_ids():
        state.retrieval.claims.initialize(task_plan, user_query)
    rounds = state.retrieval.rounds
    phase = "DISCOVERY"
    # A reviewer-requested replan stays valid until the task obtains new
    # material. Repeating another frozen request is fed back to that rebuilt
    # planner; it must not rebuild the same session again and again.
    pending_replan_review: dict[str, Any] | None = None
    stalled_actions_after_replan = 0
    # One repeated frozen path is enough to prove that the active strategy
    # batch stalled.  Waiting for the same model choice three times only
    # amplifies RWKV repetition and delays the already-requested replan.
    max_stalled_actions_after_replan = 1
    replan_rebuilds_without_progress = 0
    max_replan_rebuilds_without_progress = 2
    cross_validation_protocol_failures_after_duplicate = 0
    max_cross_validation_protocol_failures_after_duplicate = 2

    append_task_event(
        state.task_id,
        "research_loop_started",
        step=0,
        phase=phase,
        max_steps=max_steps,
        shared_state=True,
        decision_owner="rwkv",
        model=model_profile,
    )

    last_action = "rwkv_research"
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
            task_point_binding_method=plan.get("task_point_binding_method", ""),
            task_point_binding_raw_model_output=plan.get(
                "task_point_binding_raw_model_output", ""
            ),
            task_point_binding_error=plan.get("task_point_binding_error", ""),
            task_point_binding_temperature=plan.get(
                "task_point_binding_temperature"
            ),
            sampling_temperature=plan.get("sampling_temperature"),
            sampling_seed=plan.get("sampling_seed"),
            decision_owner="rwkv",
            gateway_override=False,
            model=model_profile,
        )

        if plan.get("planner_error"):
            owner._model_protocol_failure = True
            return owner._complete_model_tool_loop(
                user_query,
                last_action,
                rounds,
                step,
                termination_reason="planner_protocol_error",
            )

        if action == "finish_task":
            review = owner._cross_validate_research(
                user_query,
                task_plan,
                step=step,
            )
            review_decision = str(review.get("decision") or "").casefold()
            if review_decision == "finish":
                return owner._complete_model_tool_loop(
                    user_query,
                    action,
                    rounds,
                    step,
                    termination_reason="rwkv_cross_validation_finish",
                )

            compact_review = {
                key: value
                for key, value in review.items()
                if key not in {"prompt", "raw_model_output"}
            }
            if review_decision == "replan":
                pending_replan_review = dict(review)
                replan_rebuilds_without_progress = 1
                replan_count = state.retrieval.record_replan()
                feedback = {
                    "status": "cross_validation_replan",
                    "evidence_review": compact_review,
                    "replan_count": replan_count,
                    "message": (
                        "RWKV cross-validation requested another planning round. "
                        "The next tool, query, and source remain RWKV decisions."
                    ),
                }
                state.last_feedback = json.dumps(feedback, ensure_ascii=False)
                replan_action = owner.planner.rebuild_session_after_review(
                    user_query,
                    state.to_retrieval_context(),
                    review,
                    phase,
                )
                append_task_event(
                    state.task_id,
                    "planner_session_rebuilt",
                    step=step,
                    phase="REPLAN",
                    review=compact_review,
                    replan_count=replan_count,
                    shared_state=state.retrieval.routing_snapshot(),
                    replan_action=replan_action,
                    decision_owner="rwkv",
                    controller_query_generated=False,
                )
                continue

            feedback = {
                "status": "cross_validation_protocol_error",
                "evidence_review": compact_review,
                "message": (
                    "The RWKV cross-validation response was not an executable finish/replan "
                    "decision. Decide the next action yourself."
                ),
            }
            state.last_feedback = json.dumps(feedback, ensure_ascii=False)
            owner.planner.observe_tool_result(feedback)
            replan_action = owner.planner.rebuild_session_after_review(
                user_query,
                state.to_retrieval_context(),
                review,
                "REPLAN",
                routing_observation=feedback,
            )
            append_task_event(
                state.task_id,
                "planner_session_rebuilt",
                step=step,
                phase="REPLAN",
                reason="rwkv_cross_validation_protocol_replan",
                replan_action=replan_action,
                decision_owner="rwkv",
                controller_query_generated=False,
            )
            continue

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
            continue

        # An exact retrieval transaction is global to the run because its
        # observed result can be shared across claims. Conservative equivalent
        # queries remain point-scoped so a related but distinct claim is not
        # suppressed. The controller never authors an alternate query or route.
        exact_duplicate = owner._retrieval_ledger.request_status(
            action,
            args,
        )
        equivalent_duplicate: dict[str, Any] | None = None
        if (
            action == "web_search"
            and task_point_id
            and str(args.get("query") or "").strip()
        ):
            query_status = owner._retrieval_ledger.query_status(
                args.get("query"),
                task_point_id=task_point_id,
                threshold=0.88,
            )
            if query_status.get("attempted"):
                equivalent_duplicate = query_status
        duplicate = exact_duplicate or equivalent_duplicate
        if _is_retrieval_tool(action) and duplicate:
            duplicate_query = str(args.get("query") or args.get("url") or user_query)
            duplicate_kind = (
                "exact_duplicate_request"
                if exact_duplicate
                else "equivalent_duplicate_query"
            )
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
            frozen_paths = list(
                state.retrieval.routing_snapshot().get("frozen_paths") or []
            )[-8:]
            feedback = {
                "status": "no_new_evidence",
                "error_class": duplicate_kind,
                "message": (
                    "This retrieval path already ran for the same task point and is frozen. "
                    "Choose a materially different path or finish from the available evidence."
                ),
                "request": {"action": action, "arguments": args},
                "previous_request_status": duplicate,
                "repeat_count": duplicate_record.get("blocked_count", 1),
                "frozen_path": frozen_path,
                "frozen_paths": frozen_paths,
                "retrieval_ledger": owner._retrieval_ledger.observation(limit=16),
                "results": [],
            }
            state.last_feedback = json.dumps(feedback, ensure_ascii=False)
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

            if pending_replan_review is not None:
                stalled_actions_after_replan += 1
                pending_feedback = {
                    **feedback,
                    "pending_replan": {
                        key: value
                        for key, value in pending_replan_review.items()
                        if key not in {"prompt", "raw_model_output"}
                    },
                    "stalled_actions_after_replan": stalled_actions_after_replan,
                    "max_stalled_actions_after_replan": max_stalled_actions_after_replan,
                }
                state.last_feedback = json.dumps(pending_feedback, ensure_ascii=False)
                owner.planner.observe_tool_result(pending_feedback)
                append_task_event(
                    state.task_id,
                    "replan_path_stalled",
                    step=step,
                    phase="REPLAN",
                    action=action,
                    duplicate_kind=duplicate_kind,
                    stalled_actions_after_replan=stalled_actions_after_replan,
                    replan_count=state.retrieval.replan_count,
                    decision_owner="rwkv",
                )
                if stalled_actions_after_replan >= max_stalled_actions_after_replan:
                    if (
                        replan_rebuilds_without_progress
                        < max_replan_rebuilds_without_progress
                    ):
                        replan_rebuilds_without_progress += 1
                        replan_count = state.retrieval.record_replan()
                        escalation_feedback = {
                            **pending_feedback,
                            "status": "replan_stall_escalation",
                            "replan_count": replan_count,
                            "replan_rebuilds_without_progress": replan_rebuilds_without_progress,
                            "message": (
                                "The prior RWKV replan repeated frozen paths without new evidence. "
                                "Rebuild once at the next request-level replan temperature; RWKV must "
                                "choose the next task point, tool, query, and source itself."
                            ),
                        }
                        state.last_feedback = json.dumps(
                            escalation_feedback,
                            ensure_ascii=False,
                        )
                        replan_action = owner.planner.rebuild_session_after_review(
                            user_query,
                            state.to_retrieval_context(),
                            pending_replan_review,
                            "REPLAN",
                            routing_observation=escalation_feedback,
                        )
                        append_task_event(
                            state.task_id,
                            "planner_session_rebuilt",
                            step=step,
                            phase="REPLAN",
                            reason="rwkv_replan_stall_escalation",
                            replan_count=replan_count,
                            replan_rebuilds_without_progress=replan_rebuilds_without_progress,
                            frozen_path=frozen_path,
                            shared_state=state.retrieval.routing_snapshot(),
                            replan_action=replan_action,
                            decision_owner="rwkv",
                            controller_query_generated=False,
                        )
                        stalled_actions_after_replan = 0
                        continue
                    review_latest = getattr(
                        owner,
                        "_cross_validate_if_evidence_changed",
                        None,
                    )
                    latest_review = (
                        review_latest(user_query, task_plan, step=step)
                        if callable(review_latest)
                        else None
                    )
                    if latest_review is not None:
                        latest_decision = str(
                            latest_review.get("decision") or ""
                        ).casefold()
                        if latest_decision == "finish":
                            return owner._complete_model_tool_loop(
                                user_query,
                                action,
                                rounds,
                                step,
                                termination_reason="rwkv_cross_validation_finish_after_stall",
                            )
                        if latest_decision == "replan" and step < max_steps:
                            pending_replan_review = dict(latest_review)
                            replan_rebuilds_without_progress = 1
                            stalled_actions_after_replan = 0
                            replan_count = state.retrieval.record_replan()
                            replan_action = owner.planner.rebuild_session_after_review(
                                user_query,
                                state.to_retrieval_context(),
                                latest_review,
                                "REPLAN",
                            )
                            append_task_event(
                                state.task_id,
                                "planner_session_rebuilt",
                                step=step,
                                phase="REPLAN",
                                reason="new_evidence_cross_validation_replan_after_stall",
                                review={
                                    key: value
                                    for key, value in latest_review.items()
                                    if key not in {"prompt", "raw_model_output"}
                                },
                                replan_count=replan_count,
                                replan_action=replan_action,
                                shared_state=state.retrieval.routing_snapshot(),
                                decision_owner="rwkv",
                                controller_query_generated=False,
                            )
                            continue
                    return owner._complete_model_tool_loop(
                        user_query,
                        action,
                        rounds,
                        step,
                        termination_reason="rwkv_replan_stalled",
                    )
                continue

            review = owner._cross_validate_research(
                user_query,
                task_plan,
                step=step,
            )
            review_decision = str(review.get("decision") or "").casefold()
            compact_review = {
                key: value
                for key, value in review.items()
                if key not in {"prompt", "raw_model_output"}
            }

            if review_decision == "finish":
                return owner._complete_model_tool_loop(
                    user_query,
                    action,
                    rounds,
                    step,
                    termination_reason="rwkv_cross_validation_finish_after_duplicate",
                )

            if review_decision == "replan":
                pending_replan_review = dict(review)
                replan_rebuilds_without_progress = 1
                replan_count = state.retrieval.record_replan()
                replan_feedback = {
                    **feedback,
                    "status": "cross_validation_replan",
                    "evidence_review": compact_review,
                    "replan_count": replan_count,
                    "message": (
                        "The latest RWKV cross-validation found a material evidence gap, "
                        "and the repeated path is frozen. The next tool, query, and source remain "
                        "RWKV planner decisions."
                    ),
                }
                state.last_feedback = json.dumps(replan_feedback, ensure_ascii=False)
                replan_action = owner.planner.rebuild_session_after_review(
                    user_query,
                    state.to_retrieval_context(),
                    review,
                    "REPLAN",
                    routing_observation=replan_feedback,
                )
                append_task_event(
                    state.task_id,
                    "planner_session_rebuilt",
                    step=step,
                    phase="REPLAN",
                    reason="rwkv_cross_validation_replan_after_duplicate",
                    review=compact_review,
                    cross_validation_reused=False,
                    frozen_path=frozen_path,
                    repeat_count=duplicate_record.get("blocked_count", 1),
                    replan_count=replan_count,
                    replan_action=replan_action,
                    shared_state=state.retrieval.routing_snapshot(),
                    decision_owner="rwkv",
                    controller_query_generated=False,
                )
                continue

            protocol_feedback = {
                **feedback,
                "status": "cross_validation_protocol_error",
                "evidence_review": compact_review,
                "message": (
                    "The RWKV cross-validation response was not an executable finish/replan "
                    "decision. Decide the next action from the retained evidence and frozen path."
                ),
            }
            cross_validation_protocol_failures_after_duplicate += 1
            protocol_feedback["protocol_failure_count"] = (
                cross_validation_protocol_failures_after_duplicate
            )
            protocol_feedback["max_protocol_failures"] = (
                max_cross_validation_protocol_failures_after_duplicate
            )
            state.last_feedback = json.dumps(protocol_feedback, ensure_ascii=False)
            owner.planner.observe_tool_result(protocol_feedback)
            append_task_event(
                state.task_id,
                "cross_validation_protocol_error",
                step=step,
                phase="REPLAN",
                reason="cross_validation_protocol_error_after_duplicate",
                review=compact_review,
                frozen_path=frozen_path,
                repeat_count=duplicate_record.get("blocked_count", 1),
                replan_count=state.retrieval.replan_count,
                shared_state=state.retrieval.routing_snapshot(),
                decision_owner="rwkv",
                controller_query_generated=False,
            )
            if (
                cross_validation_protocol_failures_after_duplicate
                >= max_cross_validation_protocol_failures_after_duplicate
            ):
                return owner._complete_model_tool_loop(
                    user_query,
                    action,
                    rounds,
                    step,
                    termination_reason="rwkv_cross_validation_protocol_stalled",
                )
            replan_action = owner.planner.rebuild_session_after_review(
                user_query,
                state.to_retrieval_context(),
                review,
                "REPLAN",
                routing_observation=protocol_feedback,
            )
            append_task_event(
                state.task_id,
                "planner_session_rebuilt",
                step=step,
                phase="REPLAN",
                reason="rwkv_cross_validation_protocol_replan_after_duplicate",
                replan_action=replan_action,
                decision_owner="rwkv",
                controller_query_generated=False,
            )
            continue

        tool_context = owner._agentic_tool_context()
        if _is_retrieval_tool(action) and task_point_id:
            # ``task_point_id`` is the exact point selected by RWKV.  Passing
            # it as system-owned context lets chunk selection and evidence
            # binding focus on that point without exposing a new model
            # argument or changing the model's tool/query choice.
            tool_context = {**tool_context, "task_point_id": task_point_id}
        raw_result = ToolRegistry.execute(
            action,
            args,
            tool_context,
            phase=None,
        )
        result = _as_dict(raw_result)
        owner._retrieval_ledger.record_request(
            action,
            args,
            result,
            step=step,
            task_point_id=task_point_id,
        )

        replan_material_changed = False
        if _is_retrieval_tool(action):
            query = str(args.get("query") or args.get("url") or user_query)
            retrieval_delta = state.retrieval.record_query(
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
            state.run_metadata["claim_ledger"] = state.retrieval.claims.snapshot()
            replan_material_changed = bool(
                pending_replan_review is not None
                and (
                    not _review_missing_point_ids(pending_replan_review, task_plan)
                    or task_point_id
                    in _review_missing_point_ids(pending_replan_review, task_plan)
                )
                and bool(retrieval_delta.get("material_changed"))
            )
            if replan_material_changed:
                stalled_actions_after_replan = 0
            cross_validation_protocol_failures_after_duplicate = 0
        else:
            _record_deterministic_result(
                owner,
                action,
                result,
                step=step,
                task_point_id=task_point_id,
            )
            replan_material_changed = bool(
                pending_replan_review is not None
                and action in {"calculator", "current_time", "date_diff"}
                and str(result.get("status") or "").casefold() == "ok"
                and (
                    not _review_missing_point_ids(pending_replan_review, task_plan)
                    or task_point_id
                    in _review_missing_point_ids(pending_replan_review, task_plan)
                )
            )
            if replan_material_changed:
                stalled_actions_after_replan = 0

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
                "real_network": bool(result.get("real_network", _is_retrieval_tool(action))),
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
            real_network=bool(result.get("real_network", _is_retrieval_tool(action))),
            decision_owner="rwkv",
            retrieval_environment=plugin_environment_snapshot(),
            shared_state=state.retrieval.routing_snapshot(),
        )

        if replan_material_changed:
            # A new URL or tool result is only observable progress; it is not
            # proof that the missing fact was resolved.  Return the revised
            # evidence to RWKV immediately.  Only its cross-validation may
            # finish the task or define the next missing obligation.
            review_latest = getattr(
                owner,
                "_cross_validate_if_evidence_changed",
                None,
            )
            latest_review = (
                review_latest(user_query, task_plan, step=step)
                if callable(review_latest)
                else owner._cross_validate_research(
                    user_query,
                    task_plan,
                    step=step,
                )
            )
            if latest_review is not None:
                latest_decision = str(
                    latest_review.get("decision") or ""
                ).casefold()
                if latest_decision == "finish":
                    mark_replan_progress = getattr(
                        owner.planner,
                        "mark_replan_progress",
                        None,
                    )
                    if callable(mark_replan_progress):
                        mark_replan_progress(task_point_id)
                    return owner._complete_model_tool_loop(
                        user_query,
                        action,
                        rounds,
                        step,
                        termination_reason="rwkv_cross_validation_finish_after_replan_progress",
                    )
                if latest_decision == "replan":
                    pending_replan_review = dict(latest_review)
                    stalled_actions_after_replan = 0
                    replan_rebuilds_without_progress = 1
                    replan_count = state.retrieval.record_replan()
                    replan_action = owner.planner.rebuild_session_after_review(
                        user_query,
                        state.to_retrieval_context(),
                        latest_review,
                        "REPLAN",
                    )
                    append_task_event(
                        state.task_id,
                        "planner_session_rebuilt",
                        step=step,
                        phase="REPLAN",
                        reason="new_evidence_cross_validation_replan",
                        review={
                            key: value
                            for key, value in latest_review.items()
                            if key not in {"prompt", "raw_model_output"}
                        },
                        replan_count=replan_count,
                        replan_action=replan_action,
                        shared_state=state.retrieval.routing_snapshot(),
                        decision_owner="rwkv",
                        controller_query_generated=False,
                    )
                    continue

    review_latest = getattr(owner, "_cross_validate_if_evidence_changed", None)
    if callable(review_latest):
        review_latest(
            user_query,
            task_plan,
            step=max_steps,
        )
    return owner._complete_model_tool_loop(
        user_query,
        last_action,
        rounds,
        max_steps,
        termination_reason="max_steps_reached",
    )


__all__ = ["run_unified_research_loop"]
