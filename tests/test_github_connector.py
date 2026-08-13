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
def test_github_latest_release_scope_discovers_repo_then_fetches_release(
    search_github_rest,
    fetch_github_rest,
):
    search_github_rest.return_value = json.dumps(
        {
            "status": "ok",
            "results": [
                {
                    "full_name": "example/project",
                    "url": "https://github.com/example/project",
                }
            ],
        }
    )
    fetch_github_rest.return_value = json.dumps(
        {
            "status": "ok",
            "results": [
                {
                    "title": "example/project — v2.4.1",
                    "url": "https://github.com/example/project/releases/tag/v2.4.1",
                    "content": "Release: v2.4.1\nPublished: 2026-08-10",
                    "content_type": "release",
                    "published": "2026-08-10",
                    "evidence_origin": "structured_api_record",
                }
            ],
        }
    )

    result = json.loads(
        connector_lookup(
            "github",
            "example project",
            scope="latest_release",
            original_goal="What is the latest release?",
        )
    )

    search_github_rest.assert_called_once()
    assert search_github_rest.call_args.kwargs["scope"] == "repositories"
    assert fetch_github_rest.call_args.args[0] == (
        "https://api.github.com/repos/example/project/releases/latest"
    )
    assert result["status"] == "ok"
    assert result["results"][0]["content_type"] == "release"
    assert result["results"][0]["structured_evidence_text"].startswith("Release:")


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
