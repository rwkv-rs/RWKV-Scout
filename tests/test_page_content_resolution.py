import json
from unittest.mock import patch

from tools.registry import ToolRegistry
from tools.web_search_generic import _resolve_failed_page_fetches


def _candidate(url: str, rank: int) -> dict:
    return {"url": url, "title": url, "candidate_rank": rank}


def _page(url: str, text: str, *, resolver: str = "direct_http") -> dict:
    return {
        "url": url,
        "title": url,
        "page_excerpt": text,
        "content": text,
        "body_verified": True,
        "body_cleaned": True,
        "body_quality": {"body_eligible": True, "clean_chars": len(text)},
        "content_resolver": resolver,
    }


def test_direct_body_does_not_call_optional_content_extractor():
    url = "https://example.com/article"
    text = "A substantive direct page body with enough factual text for downstream extraction."
    fetched = [(_candidate(url, 1), {"status": "ok", "results": [_page(url, text)]})]

    with (
        patch.object(
            ToolRegistry,
            "capability_names",
            return_value=["extract_mock"],
        ),
        patch.object(ToolRegistry, "execute") as execute,
    ):
        resolved = _resolve_failed_page_fetches("article facts", fetched, "TEST")

    assert resolved == fetched
    execute.assert_not_called()


def test_failed_direct_fetch_is_recovered_by_registered_exact_url_extractor():
    url = "https://example.com/dynamic"
    text = "Dynamic page source text containing the requested release date and version details."
    fetched = [
        (
            _candidate(url, 1),
            {"status": "no_evidence", "message": "SPA shell", "results": []},
        )
    ]
    adapter_result = json.dumps(
        {
            "status": "ok",
            "provider": "Mock Extractor",
            "results": [_page(url, text, resolver="mock.extract")],
        }
    )

    with (
        patch.object(
            ToolRegistry,
            "capability_names",
            return_value=["extract_mock"],
        ),
        patch.object(ToolRegistry, "execute", return_value=adapter_result) as execute,
    ):
        resolved = _resolve_failed_page_fetches("release date", fetched, "TEST")

    result = resolved[0][1]
    assert result["status"] == "ok"
    assert result["results"][0]["url"] == url
    assert result["results"][0]["content"] == text
    assert result["primary_fetch"]["status"] == "no_evidence"
    execute.assert_called_once_with(
        "extract_mock",
        {"urls": [url], "query": "release date"},
        {"task_id": "TEST"},
        phase="ALL",
    )


def test_extractor_cannot_attach_content_to_a_different_candidate_url():
    first = "https://example.com/first"
    second = "https://example.com/second"
    fetched = [
        (_candidate(first, 1), {"status": "no_evidence", "results": []}),
        (_candidate(second, 2), {"status": "no_evidence", "results": []}),
    ]
    adapter_result = json.dumps(
        {
            "status": "ok",
            "provider": "Mock Extractor",
            "results": [_page(second, "Substantive evidence for only the second exact URL.")],
        }
    )

    with (
        patch.object(
            ToolRegistry,
            "capability_names",
            return_value=["extract_mock"],
        ),
        patch.object(ToolRegistry, "execute", return_value=adapter_result),
    ):
        resolved = _resolve_failed_page_fetches("facts", fetched, "TEST")

    assert resolved[0][1]["status"] == "no_evidence"
    assert resolved[0][1]["results"] == []
    assert resolved[1][1]["status"] == "ok"
    assert resolved[1][1]["results"][0]["url"] == second


def test_unavailable_extractor_preserves_primary_fetch_failure():
    url = "https://example.com/dynamic"
    fetched = [
        (
            _candidate(url, 1),
            {"status": "no_evidence", "message": "SPA shell", "results": []},
        )
    ]

    with (
        patch.object(
            ToolRegistry,
            "capability_names",
            return_value=["extract_mock"],
        ),
        patch.object(
            ToolRegistry,
            "execute",
            return_value=json.dumps(
                {
                    "status": "error",
                    "provider": "Mock Extractor",
                    "results": [],
                    "provider_errors": ["not configured"],
                }
            ),
        ),
    ):
        resolved = _resolve_failed_page_fetches("facts", fetched, "TEST")

    assert resolved[0][1]["status"] == "no_evidence"
    assert resolved[0][1]["message"] == "SPA shell"
    assert resolved[0][1]["content_resolver_attempts"][0]["status"] == "error"


def test_tavily_extractor_is_internal_and_discoverable_by_capability():
    assert "extract_web_urls_tavily" in ToolRegistry.capability_names(
        "page_content_extract",
        phase="ALL",
    )
    assert "extract_web_urls_tavily" not in ToolRegistry.model_visible_names("ALL")
