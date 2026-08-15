import json
from unittest.mock import patch

from tools.connectors import connector_lookup
from tools.github_rest import _content_text


def test_github_release_record_is_rendered_as_structured_evidence():
    payload = {
        "name": "Version 2.4.1",
        "tag_name": "v2.4.1",
        "published_at": "2026-08-10T09:00:00Z",
        "created_at": "2026-08-09T20:00:00Z",
        "target_commitish": "main",
        "draft": False,
        "prerelease": False,
        "html_url": "https://github.com/example/project/releases/tag/v2.4.1",
        "author": {"login": "maintainer"},
        "body": "Stable release notes.",
    }

    text, content_type, url = _content_text(payload, 20_000)

    assert content_type == "release"
    assert url == payload["html_url"]
    assert "Release: Version 2.4.1" in text
    assert "Tag: v2.4.1" in text
    assert "Published: 2026-08-10T09:00:00Z" in text
    assert "Prerelease: False" in text
    assert "Stable release notes." in text


@patch("tools.connectors.fetch_github_rest")
@patch("tools.connectors.search_github_rest")
def test_github_latest_release_scope_rejects_ambiguous_repository_text(
    search_github_rest,
    fetch_github_rest,
):
    result = json.loads(
        connector_lookup(
            "github",
            "example project",
            scope="latest_release",
            original_goal="What is the latest release?",
        )
    )

    search_github_rest.assert_not_called()
    fetch_github_rest.assert_not_called()
    assert result["status"] == "error"
    assert result["error_class"] == "object_type_mismatch"
    assert result["identity_error_class"] == "invalid_repository_identifier"
    assert result["accepted_object_types"] == ["github_repository"]
    assert result["connector_runtime"]["status"] == "available"
    assert result["real_network"] is False
    assert result["results"] == []


@patch("tools.connectors.fetch_github_rest")
def test_github_403_reports_current_run_rate_limit_without_selecting_fallback(
    fetch_github_rest,
):
    fetch_github_rest.return_value = json.dumps(
        {
            "status": "error",
            "provider": "github.rest",
            "error_class": "provider_error",
            "provider_errors": [
                "NetworkFetchError: HTTP request failed: 403 rate limit exceeded"
            ],
            "results": [],
        }
    )

    result = json.loads(
        connector_lookup(
            "github_release",
            "owner/repository",
            original_goal="What is the latest release?",
        )
    )

    fetch_github_rest.assert_called_once()
    assert result["status"] == "error"
    assert result["error_class"] == "rate_limited"
    assert result["connector_runtime"] == {
        "provider": "connector.github",
        "operation": "github_release",
        "status": "rate_limited",
        "available": False,
        "cooldown_seconds": 300,
        "error_class": "provider_error",
        "message": "NetworkFetchError: HTTP request failed: 403 rate limit exceeded",
    }
    assert "web_search" not in result


@patch("tools.connectors.fetch_github_rest")
def test_github_latest_release_scope_accepts_direct_repo(fetch_github_rest):
    fetch_github_rest.return_value = json.dumps(
        {
            "status": "ok",
            "results": [
                {
                    "title": "owner/repo — v1",
                    "url": "https://github.com/owner/repo/releases/tag/v1",
                    "content": "Release: v1",
                }
            ],
        }
    )

    result = json.loads(
        connector_lookup("github", "owner/repo", scope="latest-release")
    )

    assert fetch_github_rest.call_args.args[0] == (
        "https://api.github.com/repos/owner/repo/releases/latest"
    )
    assert result["status"] == "ok"
