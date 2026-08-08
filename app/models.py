"""Transport models shared by the HTTP API and background task runner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=1)
    strategy_config: dict[str, Any] = Field(default_factory=dict)
    model_key: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_provider: str | None = None
    slm_endpoint: str | None = None
    slm_password: str | None = None
    queued_at: str | None = None
    slm_async_enabled: bool | None = None


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(min_length=1)
    model_key: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=4096)
