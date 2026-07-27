"""Internal model backend contract used by the application layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class BackendResponse:
    """Provider-neutral response returned to agents and workflows.

    The shape intentionally mirrors only the small subset of the old SDK
    response that the project consumes.  It keeps SDK objects and wire
    formats out of the domain code.
    """

    role: str = "assistant"
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    search_results: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


class ModelBackend(Protocol):
    """The model capability required by RWKV-ECRA."""

    @property
    def backend_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    def chat_completion(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_native_search: bool = False,
        max_tokens: int | None = None,
    ) -> BackendResponse:
        ...

    def text_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 768,
        stop: Sequence[str] | None = None,
    ) -> BackendResponse:
        ...

    def batch_text_completion(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int = 768,
    ) -> list[str]:
        ...

    def health(self) -> dict[str, Any]:
        ...
