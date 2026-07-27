from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.web_search_wigolo import search_web_wigolo
from utils.wigolo_client import WigoloClient, WigoloError
from utils.experiment_manifest import reconstruct_run


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class WigoloClientTests(unittest.TestCase):
    def test_post_decodes_json_without_external_dependency(self):
        with patch(
            "utils.wigolo_client.urlopen",
            return_value=_FakeResponse({"results": [{"title": "demo"}]}),
        ) as mocked:
            value = WigoloClient(base_url="http://127.0.0.1:3333").search("demo")

        self.assertEqual(value["results"][0]["title"], "demo")
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:3333/v1/search")


class WigoloToolTests(unittest.TestCase):
    def test_unavailable_wigolo_falls_back_to_existing_keyless_provider(self):
        fallback = {
            "status": "ok",
            "real_network": True,
            "provider": "keyless",
            "count": 1,
            "results": [{"title": "fallback", "url": "https://example.com"}],
            "sources": ["https://example.com"],
            "citation_refs": [{"ref_id": "WEB_REF_KEYLESS_1", "url": "https://example.com"}],
            "provider_errors": [],
        }

        class BrokenClient:
            def search(self, *_args, **_kwargs):
                raise WigoloError("daemon is not running")

        with (
            patch("tools.web_search_wigolo.WigoloClient", return_value=BrokenClient()),
            patch("tools.web_search_wigolo.get_wigolo_mode", return_value="auto"),
            patch(
                "tools.web_search_wigolo.search_web_keyless",
                return_value=json.dumps(fallback),
            ),
        ):
            result = json.loads(search_web_wigolo("demo"))

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["requested_provider"], "wigolo")
        self.assertEqual(result["count"], 1)
        self.assertIn("wigolo unavailable", result["provider_errors"][0])

    def test_wigolo_result_keeps_citation_and_evidence_metadata(self):
        class FakeClient:
            def search(self, *_args, **_kwargs):
                return {
                    "results": [
                        {
                            "title": "official result",
                            "url": "https://example.com/docs",
                            "excerpt": "supported fact",
                            "citation_id": "src-1",
                            "source_span": {"start": 1, "end": 20},
                            "evidence_score": {"final": 0.9},
                        }
                    ]
                }

            def fetch(self, *_args, **_kwargs):
                return {"content": "full page evidence"}

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("config.DATA_PIPELINE", {"output_directory": str(Path(directory) / "output")}, clear=False):
                with patch("tools.web_search_wigolo.WigoloClient", return_value=FakeClient()):
                    result = json.loads(search_web_wigolo("demo", fetch_pages=1, task_id="WIGOLO_TRACE"))

                item = result["results"][0]
                self.assertEqual(item["page_excerpt"], "full page evidence")
                self.assertEqual(item["wigolo_citation_id"], "src-1")
                self.assertEqual(result["citation_refs"][0]["source_span"]["start"], 1)
                self.assertTrue(item["untrusted_content"])
                event_types = [event["type"] for event in reconstruct_run("WIGOLO_TRACE")["events"]]
                self.assertEqual(event_types, ["page_fetch", "page_extract"])


if __name__ == "__main__":
    unittest.main()
