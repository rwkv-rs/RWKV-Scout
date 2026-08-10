from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.models import AnalyzeRequest
from app.public_access import public_frontend_request_allowed
from app.services.task_runner import resolve_model_connection


def test_public_frontend_api_uses_an_explicit_allowlist():
    task_id = "TASK_20260810_120000_abcdef"
    allowed = {
        ("GET", "/frontend-api/config"),
        ("GET", "/frontend-api/history"),
        ("GET", "/frontend-api/metrics/operational"),
        ("GET", "/frontend-api/metrics/tokens"),
        ("GET", f"/frontend-api/metrics/tokens/{task_id}"),
        ("POST", "/frontend-api/analyze"),
        ("POST", "/frontend-api/chat"),
        ("POST", f"/frontend-api/analyze/{task_id}/stop"),
        ("POST", f"/frontend-api/history/{task_id}/stop"),
        ("GET", f"/frontend-api/history/{task_id}/events"),
        ("GET", f"/frontend-api/history/{task_id}/report"),
        ("GET", f"/frontend-api/history/{task_id}/trace"),
    }
    denied = {
        ("POST", "/frontend-api/upload"),
        ("GET", "/frontend-api/files"),
        ("GET", "/frontend-api/files/content"),
        ("DELETE", "/frontend-api/files"),
        ("DELETE", f"/frontend-api/history/{task_id}"),
    }

    assert all(public_frontend_request_allowed(method, path) for method, path in allowed)
    assert not any(public_frontend_request_allowed(method, path) for method, path in denied)


def test_public_api_guard_only_blocks_files_and_destructive_deletion(monkeypatch):
    monkeypatch.setenv("RWKV_ECRA_PUBLIC_MODE", "1")
    from api import app

    with TestClient(app) as client:
        assert client.get("/frontend-api/history").status_code == 200
        assert client.get("/frontend-api/metrics/operational").status_code == 200
        # Validation reaches the chat route (422), so the public-mode guard did not hide it.
        assert client.post("/frontend-api/chat", json={"messages": []}).status_code == 422
        assert client.get("/frontend-api/files").status_code == 403
        assert client.delete("/frontend-api/history/TASK_20260810_120000_abcdef").status_code == 403
        response = client.get("/frontend-api/config")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["public_mode"] is True
    assert all("base_url" not in profile for profile in data["models"])
    assert "api_key" not in json.dumps(data).casefold()


def test_public_mode_keeps_direct_rwkv_chat_available(monkeypatch):
    monkeypatch.setenv("RWKV_ECRA_PUBLIC_MODE", "1")
    monkeypatch.setattr(
        "clients.llm_client.LLMClient.chat_completion",
        lambda self, messages, max_tokens: SimpleNamespace(
            role="assistant",
            content=f"direct reply to: {messages[-1]['content']}",
        ),
    )
    from api import app

    with TestClient(app) as client:
        response = client.post(
            "/frontend-api/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert response.json()["data"]["message"] == {
        "role": "assistant",
        "content": "direct reply to: hello",
    }


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
