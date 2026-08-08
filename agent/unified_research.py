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
    # material.  Repeating another frozen request must rebuild the planner,
    # not invoke the same reviewer on an unchanged evidence snapshot.
    pending_replan_review: dict[str, Any] | None = None

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
                owner.planner.rebuild_session_after_review(
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

        # Exact duplicates freeze only the already-executed path.  RWKV then
        # reviews the shared evidence and decides whether research is complete
        # or a new planner session is needed.  The controller never authors an
        # alternate query or source route.
        duplicate = owner._retrieval_ledger.request_status(action, args)
        if _is_retrieval_tool(action) and duplicate:
            duplicate_query = str(args.get("query") or args.get("url") or user_query)
            duplicate_record = owner._retrieval_ledger.record_duplicate_block(
                duplicate_query,
                step=step,
                task_point_id=task_point_id,
            )
            frozen_path = state.retrieval.freeze_path(
                duplicate_query,
                action=action,
                arguments=args,
                step=step,
                reason="exact_duplicate_request",
            )
            feedback = {
                "status": "no_new_evidence",
                "error_class": "exact_duplicate_request",
                "message": (
                    "This exact retrieval request already ran and its path is frozen. "
                    "RWKV cross-validation will decide whether to finish or rebuild planning."
                ),
                "request": {"action": action, "arguments": args},
                "previous_request_status": duplicate,
                "repeat_count": duplicate_record.get("blocked_count", 1),
                "frozen_path": frozen_path,
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
                execution_status="exact_duplicate",
                decision_owner="rwkv",
            )

            review_reused = pending_replan_review is not None
            review = (
                dict(pending_replan_review)
                if pending_replan_review is not None
                else owner._cross_validate_research(
                    user_query,
                    task_plan,
                    step=step,
                )
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
                owner.planner.rebuild_session_after_review(
                    user_query,
                    state.to_retrieval_context(),
                    review,
                    "REPLAN",
                )
                append_task_event(
                    state.task_id,
                    "planner_session_rebuilt",
                    step=step,
                    phase="REPLAN",
                    reason=(
                        "rwkv_pending_replan_after_duplicate"
                        if review_reused
                        else "rwkv_cross_validation_replan_after_duplicate"
                    ),
                    review=compact_review,
                    cross_validation_reused=review_reused,
                    frozen_path=frozen_path,
                    repeat_count=duplicate_record.get("blocked_count", 1),
                    replan_count=replan_count,
                    shared_state=state.retrieval.routing_snapshot(),
                    decision_owner="rwkv",
                    controller_query_generated=False,
                )
                continue

            replan_count = state.retrieval.record_replan()
            protocol_feedback = {
                **feedback,
                "status": "cross_validation_protocol_error",
                "evidence_review": compact_review,
                "replan_count": replan_count,
                "message": (
                    "The RWKV cross-validation response was not an executable finish/replan "
                    "decision. Rebuild planning while retaining the shared evidence and frozen path."
                ),
            }
            state.last_feedback = json.dumps(protocol_feedback, ensure_ascii=False)
            owner.planner.rebuild_session(
                user_query,
                state.to_retrieval_context(),
                protocol_feedback,
                "REPLAN",
            )
            append_task_event(
                state.task_id,
                "planner_session_rebuilt",
                step=step,
                phase="REPLAN",
                reason="cross_validation_protocol_error_after_duplicate",
                review=compact_review,
                frozen_path=frozen_path,
                repeat_count=duplicate_record.get("blocked_count", 1),
                replan_count=replan_count,
                shared_state=state.retrieval.routing_snapshot(),
                decision_owner="rwkv",
                controller_query_generated=False,
            )
            continue

        raw_result = ToolRegistry.execute(
            action,
            args,
            owner._agentic_tool_context(),
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

        if _is_retrieval_tool(action):
            query = str(args.get("query") or args.get("url") or user_query)
            state.retrieval.record_query(
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
            if (
                pending_replan_review is not None
                and str(result.get("status") or "").casefold() == "ok"
                and bool(result.get("results"))
            ):
                pending_replan_review = None
        else:
            _record_deterministic_result(
                owner,
                action,
                result,
                step=step,
                task_point_id=task_point_id,
            )
            if (
                pending_replan_review is not None
                and action in {"calculator", "current_time", "date_diff"}
                and str(result.get("status") or "").casefold() == "ok"
            ):
                pending_replan_review = None

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

    return owner._complete_model_tool_loop(
        user_query,
        last_action,
        rounds,
        max_steps,
        termination_reason="max_steps_reached",
    )


__all__ = ["run_unified_research_loop"]
