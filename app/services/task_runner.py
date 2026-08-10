"""Background task lifecycle for the HTTP application."""

from __future__ import annotations

import os

import config

from app.models import AnalyzeRequest
from utils.experiment_manifest import finalize_manifest
from utils.error_policy import classify_error
from utils.runtime_gate import analysis_slot
from utils.task_events import append_task_event, get_task_events
from utils.task_manager import is_task_stopped, record_task
from utils.time_budget import task_time_budget
from utils.token_tracker import current_task_id


def _first_configured(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def resolve_model_connection(request: AnalyzeRequest, profile: dict | None = None) -> dict[str, str]:
    """Resolve request-local model settings without exposing credentials."""

    selected_provider = _first_configured(
        request.llm_provider,
        request.model_key,
        config.DEFAULT_LLM_PROVIDER,
    )
    selected_profile = profile or {}
    configured_profile = config.LLM_ENDPOINTS.get(selected_provider, {})
    if not isinstance(configured_profile, dict):
        configured_profile = {}

    base_url = _first_configured(
        request.llm_base_url,
        os.environ.get("RWKV_ECRA_LLM_BASE_URL"),
        selected_profile.get("base_url"),
        configured_profile.get("base_url"),
    )
    api_key = _first_configured(
        request.llm_api_key,
        os.environ.get("RWKV_ECRA_LLM_API_KEY"),
        selected_profile.get("api_key"),
        config.API_KEYS.get(selected_provider),
    )
    return {
        "provider": selected_provider,
        "base_url": base_url,
        "api_key": api_key,
    }


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

    try:
        profile: dict = {}
        if request.model_key:
            profile = config.get_model_profile(request.model_key)
            override(config.override_model_backend, profile.get("runtime_backend"))
            override(config.override_direct_rwkv_config, profile.get("direct_runtime"))

        connection = resolve_model_connection(request, profile)
        override(config.override_llm_provider, connection["provider"])
        override(config.override_llm_url, connection["base_url"])
        override(config.override_llm_key, connection["api_key"])
        slm_endpoint = _first_configured(
            request.slm_endpoint,
            connection["base_url"].rstrip("/") + "/chat/completions"
            if connection["base_url"]
            else "",
        )
        override(config.override_slm_endpoint, slm_endpoint)
        override(config.override_slm_password, request.slm_password)
        override(config.override_slm_async_enabled, request.slm_async_enabled)

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
