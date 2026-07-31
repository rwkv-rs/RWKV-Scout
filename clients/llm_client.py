# RWKV-ECRA/clients/llm_client.py
"""Model client facade.

The workflow layer depends on this small facade, while local model execution
is delegated to ``runtime.ModelBackend``.  Cloud providers remain supported as
explicit adapters, but the local path no longer depends on an OpenAI SDK or
an OpenAI-compatible HTTP service.
"""

import time

from config import (
    LLM_ENDPOINTS,
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
    get_llm_provider,
    is_local_provider,
)
from runtime import get_model_backend
from runtime.transcript import render_rwkv_transcript
from utils.model_events import record_model_event, visible_model_text
from utils.retry import retry_with_fallback
from utils.token_tracker import current_task_id, global_token_tracker


def _event_messages(messages: list | None) -> list[dict]:
    """Persist only the text contract of a request, never SDK objects/secrets."""
    output = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        output.append(
            {
                "role": str(message.get("role") or ""),
                "content": visible_model_text(message.get("content") or ""),
            }
        )
    return output


def _usage_values(response) -> tuple[int, int, int]:
    usage = getattr(response, "usage", {}) or {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "__dict__") and not isinstance(usage, dict):
        usage = vars(usage)
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    return (
        int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        int(details.get("reasoning_tokens") or 0) if isinstance(details, dict) else 0,
    )


class LLMClient:
    """Stable workflow-facing client backed by a project-owned runtime."""

    @property
    def provider(self):
        return get_llm_provider()

    @property
    def model(self):
        return get_llm_model()

    @property
    def client(self):
        """Return the cloud-provider SDK only for explicit cloud providers."""
        if is_local_provider(self.provider):
            raise RuntimeError(
                "local providers use runtime.ModelBackend; direct SDK access is not available"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "the optional OpenAI SDK is required only when a cloud provider is selected"
            ) from exc
        return OpenAI(
            api_key=get_llm_api_key(),
            base_url=get_llm_base_url(),
            timeout=60.0,
        )

    def _record_backend_response(
        self,
        response,
        *,
        operation: str,
        started: float,
        messages: list | None = None,
        prompt: str | None = None,
    ):
        prompt_tokens, completion_tokens, reasoning_tokens = _usage_values(response)
        global_token_tracker.add_llm(prompt_tokens, completion_tokens, reasoning_tokens)
        payload = {
            "task_id": current_task_id.get(),
            "status": "completed",
            "operation": operation,
            "provider": self.provider,
            "backend": getattr(get_model_backend(), "backend_name", "unknown"),
            "model": self.model,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "output": visible_model_text(getattr(response, "content", "")),
        }
        if messages is not None:
            payload["input_messages"] = _event_messages(messages)
        if prompt is not None:
            payload["prompt"] = prompt
        record_model_event(payload.pop("task_id"), **payload)
        return response

    @retry_with_fallback(max_retries=3, delay=3)
    def chat_completion(
        self,
        messages: list,
        tools: list = None,
        enable_native_search: bool = False,
        max_tokens: int | None = None,
    ):
        started = time.perf_counter()
        task_id = current_task_id.get()

        if is_local_provider(self.provider):
            try:
                backend = get_model_backend()
                if not tools and not enable_native_search:
                    response = backend.text_completion(
                        render_rwkv_transcript(messages),
                        max_tokens=max_tokens or 768,
                    )
                else:
                    response = backend.chat_completion(
                        messages,
                        tools=tools,
                        enable_native_search=enable_native_search,
                        max_tokens=max_tokens,
                    )
                return self._record_backend_response(
                    response,
                    operation="chat_completion",
                    started=started,
                    messages=messages,
                )
            except Exception as exc:
                record_model_event(
                    task_id,
                    status="failed",
                    operation="chat_completion",
                    provider=self.provider,
                    model=self.model,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                    input_messages=_event_messages(messages),
                    error=f"{type(exc).__name__}: {exc}"[:1000],
                )
                raise

        provider_config = LLM_ENDPOINTS.get(self.provider, {})
        kwargs = {"model": self.model, "messages": messages, "stream": False}
        if self.provider == "baidu":
            kwargs["max_completion_tokens"] = provider_config.get("max_completion_tokens", 65536)
            if enable_native_search and provider_config.get("enable_web_search", False):
                kwargs["extra_body"] = {
                    "web_search": {
                        "enable": True,
                        "enable_citation": True,
                        "enable_trace": True,
                    }
                }
        elif self.provider == "volcengine":
            reasoning = provider_config.get("reasoning_effort")
            if reasoning:
                kwargs["extra_body"] = {"reasoning_effort": reasoning}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "function", "function": {"name": "system_router"}}

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            record_model_event(
                task_id,
                status="failed",
                operation="chat_completion",
                provider=self.provider,
                model=self.model,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                input_messages=_event_messages(messages),
                error=f"{type(exc).__name__}: {exc}"[:1000],
            )
            raise

        in_tok, out_tok, reasoning_tok = _usage_values(resp)
        global_token_tracker.add_llm(in_tok, out_tok, reasoning_tok)
        msg = resp.choices[0].message
        raw_dict = resp.model_dump()
        msg.search_results = raw_dict.get("search_results", [])
        usage_dict = raw_dict.get("usage") or {}
        record_model_event(
            task_id,
            status="completed",
            operation="chat_completion",
            provider=self.provider,
            model=self.model,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            input_messages=_event_messages(messages),
            prompt_tokens=usage_dict.get("prompt_tokens", 0) or usage_dict.get("input_tokens", 0) or 0,
            completion_tokens=usage_dict.get("completion_tokens", 0) or usage_dict.get("output_tokens", 0) or 0,
            output=visible_model_text(getattr(msg, "content", "")),
        )
        return msg

    @retry_with_fallback(max_retries=3, delay=3)
    def text_completion(
        self,
        prompt: str,
        max_tokens: int = 768,
        stop: list[str] | tuple[str, ...] | None = None,
    ):
        """Run a raw completion through the configured local runtime."""
        if not is_local_provider(self.provider):
            raise RuntimeError("text_completion is only supported by the local provider")
        started = time.perf_counter()
        task_id = current_task_id.get()
        try:
            response = get_model_backend().text_completion(
                prompt,
                max_tokens=max_tokens,
                stop=stop,
            )
            return self._record_backend_response(
                response,
                operation="text_completion",
                started=started,
                prompt=prompt,
            )
        except Exception as exc:
            record_model_event(
                task_id,
                status="failed",
                operation="text_completion",
                provider=self.provider,
                model=self.model,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                prompt=prompt,
                error=f"{type(exc).__name__}: {exc}"[:1000],
            )
            raise
