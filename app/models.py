"""Transport models shared by the HTTP API and background task runner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=1)
    acceptance_case_id: str | None = None
    experiment_id: str | None = None
    variant: str = "candidate"
    baseline_run_id: str | None = None
    dataset_version: str | None = None
    persona: str | None = None
    domain: str | None = None
    task_type: str | None = None
    difficulty: str | None = None
    hypothesis: str | None = None
    changed_variable: str | None = None
    expected_metrics: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    rejection_criteria: list[str] = Field(default_factory=list)
    risk_checks: list[str] = Field(default_factory=list)
    prompt_version: str | None = None
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
