import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from tools.web_search_tavily import (
    _reset_provider_health_for_tests,
    extract_web_urls_tavily,
    search_web_tavily,
)


class _ProviderError(RuntimeError):
    def __init__(self, message: str, response) -> None:
        super().__init__(message)
        self.response = response


class _Response:
    def __init__(self, status_code: int, text: str, payload=None) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _ProviderError(f"HTTP {self.status_code}", self)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, response: _Response, *, pause: float = 0.0) -> None:
        self.response = response
        self.pause = pause
        self.calls = 0
        self.requests = []
        self.lock = threading.Lock()

    def post(self, *args, **kwargs):
        with self.lock:
            self.calls += 1
            self.requests.append((args, kwargs))
        if self.pause:
            time.sleep(self.pause)
        return self.response


class TavilyProviderHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_provider_health_for_tests()

    def tearDown(self) -> None:
        _reset_provider_health_for_tests()

    def test_permanent_quota_failures_are_probed_only_once_per_key(self):
        session = _Session(
            _Response(432, "This request exceeds your plan's set usage limit")
        )
        with (
            patch("tools.web_search_tavily.get_search_api_keys", return_value=["key-a", "key-b"]),
            patch("tools.web_search_tavily.create_network_session", return_value=session),
            patch("tools.web_search_tavily.retire_search_api_key", return_value=True) as retire,
        ):
            first = json.loads(search_web_tavily("test query"))
            second = json.loads(search_web_tavily("another query"))

        self.assertEqual(session.calls, 2)
        self.assertEqual(first["provider_attempts"], 2)
        self.assertTrue(first["provider_disabled"])
        self.assertEqual(first["removed_credentials"], 2)
        self.assertEqual(second["provider_attempts"], 0)
        self.assertTrue(second["provider_disabled"])
        self.assertEqual(retire.call_count, 2)

    def test_concurrent_cases_share_the_per_key_probe(self):
        session = _Session(
            _Response(432, "quota exhausted"),
            pause=0.03,
        )
        with (
            patch("tools.web_search_tavily.get_search_api_keys", return_value=["only-key"]),
            patch("tools.web_search_tavily.create_network_session", return_value=session),
            patch("tools.web_search_tavily.retire_search_api_key", return_value=True) as retire,
        ):
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(search_web_tavily, ["q1", "q2", "q3", "q4"]))

        self.assertEqual(session.calls, 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual(retire.call_count, 1)

    def test_transient_failure_is_not_permanently_disabled(self):
        session = _Session(_Response(500, "temporary upstream failure"))
        with (
            patch("tools.web_search_tavily.get_search_api_keys", return_value=["key-a"]),
            patch("tools.web_search_tavily.create_network_session", return_value=session),
            patch("tools.web_search_tavily.retire_search_api_key") as retire,
        ):
            first = json.loads(search_web_tavily("q1"))
            second = json.loads(search_web_tavily("q2"))

        self.assertEqual(session.calls, 2)
        self.assertFalse(first["provider_disabled"])
        self.assertFalse(second["provider_disabled"])
        retire.assert_not_called()

    def test_extract_returns_cleaned_body_for_the_exact_requested_url(self):
        url = "https://example.com/release"
        body = (
            "# Release notes\n\n"
            "Version 4.2 was released on 2026-08-09. "
            "This page describes the supported features and upgrade procedure. "
            "The text is long enough to be treated as substantive source evidence."
        )
        session = _Session(
            _Response(
                200,
                "",
                {"results": [{"url": url, "raw_content": body}], "failed_results": []},
            )
        )
        with (
            patch("tools.web_search_tavily.get_search_api_keys", return_value=["key-a"]),
            patch("tools.web_search_tavily.create_network_session", return_value=session),
        ):
            result = json.loads(
                extract_web_urls_tavily([url], query="Version 4.2 release date")
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["results"][0]["url"], url)
        self.assertTrue(result["results"][0]["body_verified"])
        self.assertEqual(result["results"][0]["content_resolver"], "web.tavily")
        self.assertNotIn("raw_content", result["results"][0])
        request_args, request_kwargs = session.requests[0]
        self.assertEqual(request_args[0], "https://api.tavily.com/extract")
        self.assertEqual(request_kwargs["json"]["urls"], [url])
        self.assertEqual(request_kwargs["json"]["query"], "Version 4.2 release date")
        self.assertEqual(request_kwargs["json"]["chunks_per_source"], 5)
        self.assertEqual(request_kwargs["json"]["format"], "markdown")

    def test_extract_is_optional_when_tavily_is_not_configured(self):
        with patch("tools.web_search_tavily.get_search_api_keys", return_value=[]):
            result = json.loads(
                extract_web_urls_tavily(
                    ["https://example.com/release"],
                    query="release date",
                )
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_class"], "provider_not_configured")
        self.assertEqual(result["results"], [])


if __name__ == "__main__":
    unittest.main()
