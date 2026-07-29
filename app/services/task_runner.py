"""Background task lifecycle for the HTTP application."""

from __future__ import annotations

import config

from app.models import AnalyzeRequest
from utils.experiment_manifest import finalize_manifest
from utils.error_policy import classify_error
from utils.runtime_gate import analysis_slot
from utils.task_events import append_task_event, get_task_events
from utils.task_manager import is_task_stopped, record_task
from utils.time_budget import TaskTimeoutError, task_time_budget
from utils.token_tracker import current_task_id


def run_background_analysis(
    task_id: str,
    request: AnalyzeRequest,
    task_output_dir: str,
) -> None:
    task_token = current_task_id.set(task_id)
    context_tokens: list[tuple[object, object]] = []

    def override(context, value) -> None:
        if value is not None:
            context_tokens.append((context, context.set(value)))

    override(config.override_llm_key, request.llm_api_key)
    override(config.override_llm_url, request.llm_base_url)
    override(config.override_llm_provider, request.llm_provider)
    override(config.override_slm_endpoint, request.slm_endpoint)
    override(config.override_slm_password, request.slm_password)
    override(config.override_slm_async_enabled, request.slm_async_enabled)

    try:
        if request.model_key:
            profile = config.get_model_profile(request.model_key)
            override(config.override_llm_provider, request.model_key)
            override(config.override_model_backend, profile.get("runtime_backend"))
            override(config.override_direct_rwkv_config, profile.get("direct_runtime"))
            base_url = str(profile.get("base_url") or "")
            override(config.override_llm_url, base_url)
            if base_url:
                override(config.override_slm_endpoint, base_url.rstrip("/") + "/chat/completions")

        from agent.orchestrator import Orchestrator

        run_metadata = {
            "experiment_id": request.experiment_id or task_id,
            "variant": request.variant,
            "baseline_run_id": request.baseline_run_id or "",
            "dataset_version": request.dataset_version or "",
            "persona": request.persona or "",
            "domain": request.domain or "",
            "task_type": request.task_type or "",
            "difficulty": request.difficulty or "",
            "hypothesis": request.hypothesis or "",
            "changed_variable": request.changed_variable or "",
            "expected_metrics": request.expected_metrics or [],
            "side_effects": request.side_effects or [],
            "acceptance_criteria": request.acceptance_criteria or [],
            "rejection_criteria": request.rejection_criteria or [],
            "risk_checks": request.risk_checks or [],
            "prompt_version": request.prompt_version or "",
            "search_action": request.search_action or "",
            "baseline_search_action": request.baseline_search_action or "",
            "retrieval_only": request.retrieval_only,
            "retrieval_strategy": request.retrieval_strategy or "",
            "retrieval_fork": request.retrieval_fork,
            "strategy_config": request.strategy_config,
        }
        with task_time_budget(task_id), analysis_slot(task_id):
            Orchestrator().run(user_query=request.query, task_id=task_id, run_metadata=run_metadata)
            if not is_task_stopped(task_id):
                final_events = [event for event in get_task_events(task_id) if event.get("type") == "final"]
                terminal_status = str(final_events[-1].get("status") or "completed") if final_events else "completed"
                record_task(
                    task_id,
                    request.query,
                    "completed",
                    task_output_dir,
                    acceptance_case_id=request.acceptance_case_id,
                )
                finalize_manifest(task_id, status=terminal_status)
    except TaskTimeoutError as exc:
        append_task_event(
            task_id,
            "error",
            phase="RUNTIME",
            error=f"{type(exc).__name__}: {exc}"[:1000],
            error_class="timeout",
        )
        append_task_event(task_id, "final", status="timed_out", content="", error_class="timeout")
        if not is_task_stopped(task_id):
            record_task(
                task_id,
                request.query,
                "timed_out",
                task_output_dir,
                str(exc),
                acceptance_case_id=request.acceptance_case_id,
            )
            finalize_manifest(task_id, status="timed_out", error=str(exc))
    except Exception as exc:
        append_task_event(
            task_id,
            "error",
            phase="RUNTIME",
            error=f"{type(exc).__name__}: {exc}"[:1000],
            error_class=classify_error(exc),
        )
        append_task_event(task_id, "final", status="failed", content="", error_class=classify_error(exc))
        if not is_task_stopped(task_id):
            record_task(
                task_id,
                request.query,
                "failed",
                task_output_dir,
                str(exc),
                acceptance_case_id=request.acceptance_case_id,
            )
            finalize_manifest(task_id, status="failed", error=str(exc))
    finally:
        for context, token in reversed(context_tokens):
            context.reset(token)
        current_task_id.reset(task_token)
