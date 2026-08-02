"""OpenAI-compatible HTTP backend kept as a replaceable compatibility layer."""

from __future__ import annotations

import concurrent.futures
import contextvars
import json
import threading
from typing import Any, Sequence

import requests

from config import (
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
    get_model_connect_timeout_seconds,
    get_model_read_timeout_seconds,
    get_slm_concurrency,
)
from runtime.backend import BackendResponse
from utils.runtime_gate import model_request_slot
from utils.text_encoding import repair_mojibake
from utils.token_tracker import current_task_id
from utils.time_budget import bounded_timeout


class OpenAICompatBackend:
    """Talk to a local model server without exposing its wire format upstream."""

    backend_name = "openai_compat"

    def __init__(self):
        # Requests sessions are not guaranteed to be thread-safe. Page
        # chunk extraction can issue several model calls in one process, so
        # keep connection pooling while giving each worker thread its own
        # session and cookie/header state.
        self._session = requests.Session()
        self._session.trust_env = False
        self._sessions = threading.local()

    def _request_session(self) -> requests.Session:
        session = getattr(self._sessions, "session", None)
        if session is None:
            if threading.current_thread() is threading.main_thread():
                session = self._session
            else:
                session = requests.Session()
                session.trust_env = False
            self._sessions.session = session
        return session

    @property
    def model_name(self) -> str:
        return get_llm_model()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = get_llm_base_url().rstrip("/") + path
        task_id = current_task_id.get() or "model-request"
        with model_request_slot(task_id):
            response = self._request_session().post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {get_llm_api_key() or 'local'}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(
                    bounded_timeout(get_model_connect_timeout_seconds()),
                    bounded_timeout(get_model_read_timeout_seconds()),
                ),
            )
        if response.status_code >= 400:
            detail = response.text[:1000].replace("\n", " ")
            raise RuntimeError(f"HTTP {response.status_code}: {detail}")
        # Some compatible servers omit ``charset`` on application/json.  In
        # that case Requests may decode non-ASCII response text as Latin-1,
        # turning Chinese UTF-8 into mojibake before the JSON layer sees it.
        # Decode the wire bytes explicitly because the API contract is UTF-8.
        try:
            data = json.loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("model backend returned invalid UTF-8 JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("model backend returned a non-object JSON response")
        return data

    @staticmethod
    def _response(data: dict[str, Any], *, text_key: str = "content") -> BackendResponse:
        choices = data.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        if text_key == "text":
            content = choice.get("text", "") or ""
        else:
            content = message.get("content", "") or choice.get("text", "") or ""
        content = repair_mojibake(content)
        usage = data.get("usage") or {}
        return BackendResponse(
            role=message.get("role", "assistant"),
            content=str(content),
            tool_calls=message.get("tool_calls"),
            search_results=data.get("search_results") or [],
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
                "completion_tokens": int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
            },
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    def chat_completion(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_native_search: bool = False,
        max_tokens: int | None = None,
    ) -> BackendResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "stream": False,
            "max_tokens": max_tokens or 768,
            "temperature": 0.0,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = {"type": "function", "function": {"name": "system_router"}}
        return self._response(self._post("/chat/completions", payload))

    def text_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 768,
        stop: Sequence[str] | None = None,
    ) -> BackendResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": 0.0,
            "stream": False,
        }
        if stop:
            payload["stop"] = list(stop)
        return self._response(self._post("/completions", payload), text_key="text")

    def batch_text_completion(self, prompts: Sequence[str], *, max_tokens: int = 768) -> list[str]:
        def generate(prompt: str) -> str:
            return self.text_completion(prompt, max_tokens=max_tokens).content

        if not prompts:
            return []
        worker_count = min(max(1, get_slm_concurrency()), len(prompts))
        results = [""] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(contextvars.copy_context().run, generate, prompt): index
                for index, prompt in enumerate(prompts)
            }
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def health(self) -> dict[str, Any]:
        try:
            endpoint = get_llm_base_url().rstrip("/") + "/models"
            response = self._request_session().get(
                endpoint,
                headers={"Authorization": f"Bearer {get_llm_api_key() or 'local'}"},
                timeout=(
                    bounded_timeout(get_model_connect_timeout_seconds()),
                    bounded_timeout(get_model_read_timeout_seconds()),
                ),
            )
            response.raise_for_status()
            data = response.json() if response.content else {}
            model_ids = [
                str(item.get("id"))
                for item in data.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
            return {"available": True, "backend": self.backend_name, "models": model_ids[:8]}
        except Exception as exc:
            return {"available": False, "backend": self.backend_name, "reason": f"{type(exc).__name__}: {exc}"[:300]}
