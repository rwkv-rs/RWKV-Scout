from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.models import AnalyzeRequest
from app.public_access import public_frontend_request_allowed
from app.services.task_runner import resolve_model_connection


def test_public_frontend_api_uses_an_explicit_allowlist():
    task_id = "TASK_20260810_120000_abcdef"
    allowed = {
        ("GET", "/frontend-api/config"),
        ("POST", "/frontend-api/analyze"),
        ("POST", f"/frontend-api/analyze/{task_id}/stop"),
        ("GET", f"/frontend-api/history/{task_id}/events"),
        ("GET", f"/frontend-api/history/{task_id}/report"),
    }
    denied = {
        ("GET", "/frontend-api/history"),
        ("POST", "/frontend-api/upload"),
        ("GET", "/frontend-api/files"),
        ("GET", "/frontend-api/files/content"),
        ("DELETE", f"/frontend-api/history/{task_id}"),
        ("POST", "/frontend-api/chat"),
        ("GET", f"/frontend-api/history/{task_id}/trace"),
        ("GET", "/frontend-api/metrics/operational"),
        ("GET", "/frontend-api/metrics/tokens"),
    }

    assert all(public_frontend_request_allowed(method, path) for method, path in allowed)
    assert not any(public_frontend_request_allowed(method, path) for method, path in denied)


def test_public_api_guard_blocks_management_and_returns_sanitized_config(monkeypatch):
    monkeypatch.setenv("RWKV_ECRA_PUBLIC_MODE", "1")
    from api import app

    with TestClient(app) as client:
        assert client.get("/frontend-api/history").status_code == 403
        assert client.post("/frontend-api/chat", json={"messages": []}).status_code == 403
        response = client.get("/frontend-api/config")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["public_mode"] is True
    assert all("base_url" not in profile for profile in data["models"])
    assert "api_key" not in json.dumps(data).casefold()


def test_model_connection_request_overrides_environment(monkeypatch):
    monkeypatch.setenv("RWKV_ECRA_LLM_BASE_URL", "http://env.example/v1")
    monkeypatch.setenv("RWKV_ECRA_LLM_API_KEY", "env-key")
    request = AnalyzeRequest(
        query="test",
        model_key="local_13b",
        llm_base_url="http://request.example/v1",
        llm_api_key="request-key",
    )

    resolved = resolve_model_connection(
        request,
        {"base_url": "http://profile.example/v1", "api_key": "profile-key"},
    )

    assert resolved == {
        "provider": "local_13b",
        "base_url": "http://request.example/v1",
        "api_key": "request-key",
    }


def test_model_connection_environment_overrides_profile(monkeypatch):
    monkeypatch.setenv("RWKV_ECRA_LLM_BASE_URL", "http://env.example/v1")
    monkeypatch.setenv("RWKV_ECRA_LLM_API_KEY", "env-key")
    request = AnalyzeRequest(query="test", model_key="local_13b")

    resolved = resolve_model_connection(
        request,
        {"base_url": "http://profile.example/v1", "api_key": "profile-key"},
    )

    assert resolved["base_url"] == "http://env.example/v1"
    assert resolved["api_key"] == "env-key"


def test_model_connection_profile_precedes_provider_defaults(monkeypatch):
    monkeypatch.delenv("RWKV_ECRA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("RWKV_ECRA_LLM_API_KEY", raising=False)
    request = AnalyzeRequest(query="test", model_key="local_13b")

    resolved = resolve_model_connection(
        request,
        {"base_url": "http://profile.example/v1", "api_key": "profile-key"},
    )

    assert resolved["base_url"] == "http://profile.example/v1"
    assert resolved["api_key"] == "profile-key"
