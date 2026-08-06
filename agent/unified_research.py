"""Single shared research loop for the RWKV web workflow.

The model may enter another research round whenever coverage is incomplete,
but every round uses the same state, evidence store and web-search backend.
There is intentionally no model-visible recovery/page-fetch branch here.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agent.retrieval_synthesis import build_evidence_context
from retrieval_plugins import is_error, plugin_environment_snapshot
from tools.registry import ToolRegistry
from utils.task_events import append_task_event
from utils.task_manager import is_task_stopped
from utils.time_budget import check_time_budget
from utils.concurrency import submit_with_context


def _replan_query_candidates(
    owner: Any,
    user_query: str,
    task_plan: dict[str, Any],
    report: dict[str, Any],
    blocked_query: str,
    *,
    step: int,
    phase: str,
) -> list[str]:
    """Rebuild the planner session and return safe alternate query seeds."""

    missing = report.get("missing") or []
    focus_items = [
        str(item.get("task") or item.get("point_id") or "").strip()
        for item in missing
        if isinstance(item, dict)
    ]
    if not focus_items:
        focus_items = [
            str(point.get("task") or point.get("objective") or "").strip()
            for point in task_plan.get("atomic_points") or []
            if isinstance(point, dict)
        ]
    focus_items = [item[:320] for item in focus_items if item][:4]
    if not focus_items:
        return []

    ledger = owner._retrieval_ledger.observation(limit=16)
    feedback = {
        "schema_version": "retrieval_replan.v1",
        "status": "replan_required",
        "error_class": "duplicate_query",
        "message": (
            "The previous retrieval path is frozen. Rebuild the planner "
            "session and select a materially different focus; the blocked "
            "query cannot be executed again."
        ),
        "blocked_query": str(blocked_query or "")[:500],
        "missing": missing[:12],
        "next_focus": focus_items,
        "retrieval_ledger": ledger,
        "shared_research_state": owner.state.retrieval.routing_snapshot(),
    }
    owner.state.retrieval.freeze_path(
        blocked_query,
        step=step,
        reason="duplicate_query_requires_micro_replan",
    )
    replan_count = owner.state.retrieval.record_replan()
    append_task_event(
        owner.state.task_id,
        "retrieval_path_frozen",
        step=step,
        phase="RECOVERY",
        action="web_search",
        query=str(blocked_query or ""),
        reason="duplicate_query",
        replan_count=replan_count,
        missing=missing[:12],
    )
    owner.planner.begin_replan(
        user_query,
        owner.state.to_retrieval_context(),
        task_plan,
        feedback,
        phase=phase,
    )

    # Let the fresh planner session nominate one direction.  The controller
    # adds bounded deterministic variants below, so a malformed or repeated
    # model choice cannot prevent recovery or create a hallucinated route.
    nominated: list[str] = []
    plan = owner.planner.plan_next_action(
        user_query,
        {},
        owner.state.to_retrieval_context(),
        phase,
    )
    if str(plan.get("action") or "").strip() == "web_search":
        query = str((plan.get("args") or {}).get("query") or "").strip()
        if query:
            nominated.append(query)

    source_policy = str(task_plan.get("source_policy") or "").casefold()
    variants: list[str] = []
    for focus in focus_items:
        variants.append(focus)
        variants.append(f"{focus} primary source")
        if source_policy == "official_required":
            variants.append(f"{focus} official source")
        else:
            variants.append(f"{focus} original paper documentation")
    if not variants and blocked_query:
        variants = [f"{blocked_query} primary source"]

    candidates: list[str] = []
    seen: set[str] = set()
    width = max(1, min(int(owner.state.run_metadata.get("replan_query_width", 3) or 3), 4))
    for query in [*nominated, *variants]:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen or owner._retrieval_ledger.query_status(normalized).get("attempted"):
            continue
        seen.add(key)
        candidates.append(normalized)
        if len(candidates) >= width:
            break

    append_task_event(
        owner.state.task_id,
        "task_replan_attempt",
        step=step,
        phase="RECOVERY",
        attempt=replan_count,
        queries=candidates,
        missing=missing[:12],
        counts_toward_global_steps=False,
    )
    return candidates


def _run_replan_search_batch(
    owner: Any,
    queries: list[str],
    *,
    step: int,
    phase: str,
) -> tuple[list[dict[str, Any]], int]:
    """Execute alternate queries concurrently, then merge them in order."""

    if not queries:
        return [], 0
    context = owner._agentic_tool_context()
    max_workers = max(1, min(len(queries), int(owner.state.run_metadata.get("replan_workers", len(queries)) or len(queries))))

    def execute(query: str) -> tuple[str, dict[str, Any], Any]:
        raw = ToolRegistry.execute(
            "web_search",
            {"query": query, "max_results": 8},
            context,
            phase=phase,
        )
        return query, _as_dict(raw), raw

    completed: dict[str, tuple[dict[str, Any], Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="replan-search") as pool:
        futures = {
            submit_with_context(pool, execute, query): query
            for query in queries
        }
        for future in as_completed(futures):
            query = futures[future]
            try:
                _, result, raw = future.result()
            except Exception as exc:
                result = {
                    "status": "error",
                    "error_class": "replan_search_exception",
                    "message": f"{type(exc).__name__}: {exc}",
                    "results": [],
                }
                raw = result
            completed[query] = (result, raw)

    enriched_results: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    for query in queries:
        result, raw = completed.get(
            query,
            ({"status": "error", "error_class": "replan_search_missing", "results": []}, {}),
        )
        owner.state.retrieval.record_query(query, result, step=step)
        result["shared_research_state"] = owner.state.retrieval.routing_snapshot()
        result = owner._record_retrieval_progress(
            result,
            query=query,
            step=step,
            action="web_search",
            phase=phase,
        )
        enriched_results.append(result)
        compact_rows.extend(
            item for item in (result.get("results") or [])[:8]
            if isinstance(item, dict)
        )
        append_task_event(
            owner.state.task_id,
            "tool_result",
            step=step,
            phase=phase,
            action="web_search",
            query=query,
            result=raw,
            real_network=bool(result.get("real_network", True)),
            decision_source="controller_replan",
            shared_state=owner.state.retrieval.routing_snapshot(),
        )

    aggregate = {
        "schema_version": "retrieval_batch.v1",
        "status": "ok" if any(str(item.get("status") or "") == "ok" for item in enriched_results) else "no_evidence",
        "tool": "web_search",
        "retrieval_role": "discovery",
        "replan_batch": True,
        "queries": queries,
        "results": compact_rows[:16],
        "retrieval_ledger": owner._retrieval_ledger.observation(limit=16),
        "shared_research_state": owner.state.retrieval.routing_snapshot(),
    }
    owner.state.last_feedback = json.dumps(aggregate, ensure_ascii=False)[:7000]
    owner.planner.observe_tool_result(aggregate)
    return enriched_results, len(enriched_results)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"status": "error", "message": value[:1000], "results": []}
        return parsed if isinstance(parsed, dict) else {"status": "error", "results": []}
    return {"status": "error", "results": []}


def _constraints(owner: Any, task_plan: dict[str, Any]) -> dict[str, Any]:
    value = dict(owner.state.run_metadata or {})
    value["task_plan"] = task_plan
    return value


def _coverage(owner: Any, user_query: str, task_plan: dict[str, Any]) -> dict[str, Any]:
    records = owner.state.retrieval.source_records()
    context = build_evidence_context(
        {"query": user_query, "results": records},
        constraints=_constraints(owner, task_plan),
        query=user_query,
    )
    validation = context.get("validation") or {}
    rows = [row for row in validation.get("subquestion_coverage") or [] if isinstance(row, dict)]
    missing = [
        {
            "point_id": row.get("point_id", ""),
            "task": row.get("task", ""),
            "status": row.get("status", "missing"),
        }
        for row in rows
        if not row.get("answerable")
    ]
    report = {
        "status": "complete"
        if records and (not task_plan.get("atomic_points") or not missing)
        else "insufficient_evidence",
        "source_count": len(records),
        "usable_evidence_count": int(context.get("usable_evidence_count") or 0),
        "missing": missing,
        "conflicts": (validation.get("cross_source") or {}).get("candidate_conflicts") or [],
        "validation": validation,
    }
    owner.state.retrieval.coverage = {
        str(item.get("point_id") or index): item
        for index, item in enumerate(rows, start=1)
    }
    return report


def _requires_research(task_plan: dict[str, Any], owner: Any) -> bool:
    if owner._calculation_results or owner._time_results or owner._arithmetic_results:
        return False
    points = [point for point in task_plan.get("atomic_points") or [] if isinstance(point, dict)]
    if any(bool(point.get("evidence_needed")) for point in points):
        return True
    if owner.state.run_metadata.get("generic_web_search_only"):
        return True
    return False


def _research_feedback(
    report: dict[str, Any],
    *,
    round_count: int,
    max_rounds: int,
) -> dict[str, Any]:
    missing = report.get("missing") or []
    focus = [str(item.get("task") or item.get("point_id") or "")[:300] for item in missing]
    return {
        "schema_version": "research_loop.v1",
        "status": "insufficient_evidence",
        "message": "The finish request was checked against the shared evidence store. Continue the same research loop for the missing points.",
        "round_count": round_count,
        "max_rounds": max_rounds,
        "missing": missing[:12],
        "next_focus": [item for item in focus if item][:8],
        "conflicts": (report.get("conflicts") or [])[:8],
        "source_count": report.get("source_count", 0),
        "usable_evidence_count": report.get("usable_evidence_count", 0),
    }


def run_unified_research_loop(
    owner: Any,
    user_query: str,
    model_profile: dict[str, Any],
    task_plan: dict[str, Any],
    max_steps: int,
) -> str:
    """Run one model decision loop with repeatable, stateful research rounds."""

    state = owner.state
    rounds = state.retrieval.rounds
    max_rounds = max(
        1,
        int(state.run_metadata.get("max_network_searches", 6) or 6),
    )
    search_rounds = 0
    last_action = "model_tool_loop"
    phase = "DISCOVERY"

    append_task_event(
        state.task_id,
        "research_loop_started",
        step=0,
        phase=phase,
        max_steps=max_steps,
        max_rounds=max_rounds,
        shared_state=True,
        model=model_profile,
    )

    for step in range(1, max(1, max_steps) + 1):
        check_time_budget(minimum_seconds=0.2)
        if is_task_stopped(state.task_id):
            state.final_result = "任务已停止。"
            state.is_finished = True
            append_task_event(state.task_id, "final", status="stopped", content=state.final_result)
            return state.final_result

        plan = owner.planner.plan_next_action(
            user_query,
            {},
            state.to_retrieval_context(),
            phase,
        )
        action = str(plan.get("action") or "").strip()
        args = dict(plan.get("args") or {}) if isinstance(plan.get("args"), dict) else {}
        last_action = action or last_action
        append_task_event(
            state.task_id,
            "model_tool_decision",
            step=step,
            phase=phase,
            action=action,
            args=args,
            call_id=plan.get("call_id", ""),
            raw_model_output=plan.get("raw_model_output", ""),
            planner_error=plan.get("planner_error", ""),
            model=model_profile,
        )

        if plan.get("planner_error"):
            owner._model_protocol_failure = True
            return owner._complete_model_tool_loop(
                user_query,
                last_action,
                rounds,
                step,
                termination_reason="model_tool_decision_parse_error",
            )

        if action in {"finish_task", "answer_user"}:
            if _requires_research(task_plan, owner):
                report = _coverage(owner, user_query, task_plan)
                if report["status"] != "complete" and search_rounds < max_rounds:
                    feedback = _research_feedback(
                        report,
                        round_count=search_rounds,
                        max_rounds=max_rounds,
                    )
                    state.last_feedback = json.dumps(feedback, ensure_ascii=False, separators=(",", ":"))
                    owner.planner.observe_tool_result(feedback)
                    append_task_event(
                        state.task_id,
                        "research_gap",
                        step=step,
                        phase="VALIDATION",
                        data=feedback,
                    )
                    continue
            return owner._complete_model_tool_loop(
                user_query,
                last_action,
                rounds,
                step,
                termination_reason="model_requested_finish",
            )

        deterministic_action = action in {"calculator", "current_time", "date_diff"}
        if not ToolRegistry.can_execute(action, phase) and not (
            deterministic_action and ToolRegistry.can_execute(action, "ALL")
        ):
            error = {
                "status": "error",
                "error_class": "tool_not_allowed",
                "message": f"tool '{action}' is not available in the unified research loop",
                "allowed_tools": ToolRegistry.model_visible_names(phase),
                "results": [],
            }
            owner.planner.observe_tool_result(error)
            state.last_feedback = json.dumps(error, ensure_ascii=False)
            owner._model_protocol_failure = True
            continue

        if action == "web_search" and search_rounds >= max_rounds:
            feedback = {
                "status": "research_budget_reached",
                "message": "The research round budget is reached. Finish with the evidence already stored or state the remaining uncertainty.",
                "source_count": len(state.retrieval.sources),
                "results": [],
            }
            owner.planner.observe_tool_result(feedback)
            state.last_feedback = json.dumps(feedback, ensure_ascii=False)
            append_task_event(
                state.task_id,
                "research_budget_reached",
                step=step,
                phase="RECOVERY",
                action=action,
                data=feedback,
                transition="final_synthesis",
            )
            return owner._complete_model_tool_loop(
                user_query,
                action,
                rounds,
                step,
                termination_reason="research_budget_reached",
            )

        if action == "web_search":
            duplicate = owner._retrieval_ledger.query_status(args.get("query"))
            if duplicate and duplicate.get("attempted"):
                blocked_query = str(args.get("query") or "")
                owner._retrieval_ledger.record_duplicate_block(
                    blocked_query,
                    step=step,
                )
                feedback = {
                    "status": "no_new_evidence",
                    "error_class": "duplicate_query",
                    "message": "This query was already executed in the shared research episode. The current path is frozen; rebuild the planner session and choose a materially different focus.",
                    "query": blocked_query,
                    "retrieval_ledger": owner._retrieval_ledger.observation(limit=16),
                    "results": [],
                }
                state.last_feedback = json.dumps(feedback, ensure_ascii=False)
                owner.planner.observe_tool_result(feedback)
                append_task_event(
                    state.task_id,
                    "retrieval_duplicate_blocked",
                    step=step,
                    phase=phase,
                    action=action,
                    query=str(args.get("query") or ""),
                )
                append_task_event(
                    state.task_id,
                    "tool_result",
                    step=step,
                    phase=phase,
                    action=action,
                    result=feedback,
                    execution_status="blocked_duplicate",
                    decision_source="controller",
                )

                # A duplicate is a controller transition, not another normal
                # model turn.  Preserve the shared evidence store, freeze the
                # dead route, and rebuild the planner session around the
                # currently missing points.  The alternate queries are then
                # executed as one bounded concurrent batch.
                report = _coverage(owner, user_query, task_plan)
                if report.get("status") == "complete":
                    return owner._complete_model_tool_loop(
                        user_query,
                        action,
                        rounds,
                        step,
                        termination_reason="controller_evidence_complete",
                    )
                configured_replans = state.run_metadata.get("max_replan_attempts")
                max_replans = (
                    3
                    if configured_replans is None
                    else max(0, int(configured_replans or 0))
                )
                if state.retrieval.replan_count >= max_replans:
                    append_task_event(
                        state.task_id,
                        "task_replan_limit_reached",
                        step=step,
                        phase="RECOVERY",
                        attempts=state.retrieval.replan_count,
                        max_attempts=max_replans,
                        transition="final_synthesis",
                    )
                    return owner._complete_model_tool_loop(
                        user_query,
                        action,
                        rounds,
                        step,
                        termination_reason="replan_limit_reached",
                    )
                queries = _replan_query_candidates(
                    owner,
                    user_query,
                    task_plan,
                    report,
                    blocked_query,
                    step=step,
                    phase=phase,
                )
                remaining_rounds = max(0, max_rounds - search_rounds)
                queries = queries[:remaining_rounds]
                if queries:
                    _, executed = _run_replan_search_batch(
                        owner,
                        queries,
                        step=step,
                        phase=phase,
                    )
                    search_rounds += executed
                else:
                    append_task_event(
                        state.task_id,
                        "task_replan_empty",
                        step=step,
                        phase="RECOVERY",
                        transition="final_synthesis",
                    )
                    return owner._complete_model_tool_loop(
                        user_query,
                        action,
                        rounds,
                        step,
                        termination_reason="replan_no_alternative",
                    )
                continue

        execution_phase = "ALL" if action in {"calculator", "current_time", "date_diff"} else phase
        raw_result = ToolRegistry.execute(
            action,
            args,
            owner._agentic_tool_context(),
            phase=execution_phase,
        )
        result = _as_dict(raw_result)
        if action == "web_search":
            search_rounds += 1
            delta = state.retrieval.record_query(
                str(args.get("query") or user_query),
                result,
                step=step,
            )
            result["shared_state_delta"] = delta
            result["shared_research_state"] = state.retrieval.routing_snapshot()
            result = owner._record_retrieval_progress(
                result,
                query=str(args.get("query") or user_query),
                step=step,
                action=action,
                phase=phase,
            )
            if is_error(result):
                owner._model_protocol_failure = owner._model_protocol_failure or str(
                    result.get("error_class") or ""
                ).casefold() in {"tool_protocol", "unknown_tool"}
        elif action == "connector_lookup" and not is_error(result):
            if any(
                isinstance(item, dict) and str(item.get("evidence_origin") or "").strip()
                for item in result.get("results") or []
            ):
                state.retrieval.record_query(
                    str(args.get("query") or user_query),
                    result,
                    step=step,
                )
        elif action == "date_diff" and str(result.get("status") or "") == "ok":
            calculation = {
                "status": "ok",
                "tool": "date_diff",
                "date_a": result.get("date_a", ""),
                "date_b": result.get("date_b", ""),
                "days": result.get("days"),
                "signed_days": result.get("signed_days"),
                "formula": result.get("formula", ""),
                "source_refs": list(result.get("source_refs") or []),
            }
            owner._calculation_results.append(calculation)
            append_task_event(
                state.task_id,
                "calculation_result",
                step=step,
                phase="CALCULATION",
                action=action,
                data=calculation,
                decision_source="model",
            )
        elif action == "calculator" and str(result.get("status") or "") == "ok":
            arithmetic = {
                "status": "ok",
                "tool": "calculator",
                "expression": result.get("expression", ""),
                "result": result.get("result"),
                "formatted_result": result.get("formatted_result", ""),
                "deterministic": True,
                "source_refs": [],
            }
            owner._arithmetic_results.append(arithmetic)
            append_task_event(
                state.task_id,
                "calculation_result",
                step=step,
                phase="CALCULATION",
                action=action,
                data=arithmetic,
                decision_source="model",
            )
        elif action == "current_time" and str(result.get("status") or "") == "ok":
            clock = {
                "status": "ok",
                "tool": "current_time",
                "timezone": result.get("timezone", ""),
                "iso": result.get("iso", ""),
                "date": result.get("date", ""),
                "utc_offset": result.get("utc_offset", ""),
                "observed_at_utc": result.get("observed_at_utc", ""),
                "deterministic": True,
            }
            owner._time_results.append(clock)
            append_task_event(
                state.task_id,
                "deterministic_tool_result",
                step=step,
                phase="CALCULATION",
                action=action,
                data=clock,
                decision_source="model",
            )
        if is_error(result):
            owner._model_protocol_failure = owner._model_protocol_failure or str(
                result.get("error_class") or ""
            ).casefold() in {"tool_protocol", "unknown_tool"}

        result["shared_research_state"] = state.retrieval.routing_snapshot()
        state.last_feedback = f"[{action}] {json.dumps(result, ensure_ascii=False)[:7000]}"
        owner.planner.observe_tool_result(result)
        append_task_event(
            state.task_id,
            "tool_result",
            step=step,
            phase=phase,
            action=action,
            result=raw_result,
            real_network=bool(result.get("real_network", action == "web_search")),
            decision_source="model",
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
