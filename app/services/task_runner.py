"""Background task lifecycle for the HTTP application."""

from __future__ import annotations

import config

from app.models import AnalyzeRequest
from utils.experiment_manifest import finalize_manifest
from utils.error_policy import classify_error
from utils.runtime_gate import analysis_slot
from utils.task_events import append_task_event, get_task_events
from utils.task_manager import is_task_stopped, record_task
from utils.time_budget import task_time_budget
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

        run_metadata = {"strategy_config": request.strategy_config}
        with task_time_budget(task_id), analysis_slot(task_id):
            answer = Orchestrator().run(
                user_query=request.query,
                task_id=task_id,
                run_metadata=run_metadata,
            )
            if is_task_stopped(task_id):
                return
            final_events = [
                event for event in get_task_events(task_id) if event.get("type") == "final"
            ]
            if not final_events and str(answer or ""):
                append_task_event(
                    task_id,
                    "final",
                    content=str(answer),
                    action="task_runner",
                    mode="rwkv_final",
                )
            elif not final_events:
                raise ConnectionError("RWKV returned no final output")
            record_task(task_id, request.query, "ready", task_output_dir)
            finalize_manifest(task_id, status="ready")
    except Exception as exc:
        failure_kind = classify_error(exc)
        if failure_kind not in {
            "timeout",
            "network",
            "auth",
            "quota",
            "rate_limit",
            "provider",
        }:
            append_task_event(
                task_id,
                "error",
                phase="RUNTIME",
                error=f"{type(exc).__name__}: {exc}"[:1000],
                error_class="engineering_error",
            )
            raise
        append_task_event(
            task_id,
            "error",
            phase="RUNTIME",
            error=f"{type(exc).__name__}: {exc}"[:1000],
            error_class="network_error",
        )
        append_task_event(
            task_id,
            "final",
            status="network_error",
            content="",
            message="The model or retrieval network did not return a final answer.",
        )
        if not is_task_stopped(task_id):
            record_task(
                task_id,
                request.query,
                "network_error",
                task_output_dir,
                str(exc),
            )
            finalize_manifest(task_id, status="network_error", error=str(exc))
    finally:
        for context, token in reversed(context_tokens):
            context.reset(token)
        current_task_id.reset(task_token)
