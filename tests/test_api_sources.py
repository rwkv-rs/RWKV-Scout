from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

from tools.builtin import load_builtin_tools
from tools.registry import ToolRegistry


class ApiSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_builtin_tools()

    def test_crossref_discovery_and_doi_evidence(self):
        payload = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/example",
                        "title": ["Example paper"],
                        "author": [{"given": "A", "family": "Author"}],
                        "published": {"date-parts": [[2026, 1, 2]]},
                        "URL": "https://publisher.example/paper",
                        "abstract": "<jats:p>Supported abstract.</jats:p>",
                    }
                ]
            }
        }
        with patch("tools.crossref.fetch_json", return_value=payload):
            result = json.loads(ToolRegistry.execute("search_crossref", {"query": "Example paper"}, {}, phase="DISCOVERY"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["provider"], "crossref.rest")
        self.assertEqual(result["results"][0]["url"], "https://doi.org/10.1234/example")

        with patch("tools.crossref.fetch_json", return_value={"message": payload["message"]["items"][0]}):
            evidence = json.loads(
                ToolRegistry.execute(
                    "fetch_crossref_record",
                    {"url": "https://doi.org/10.1234/example"},
                    {},
                    phase="EXTRACTION",
                )
            )
        self.assertEqual(evidence["status"], "ok")
        self.assertIn("Supported abstract", evidence["results"][0]["page_excerpt"])

    def test_github_rest_repository_and_file_evidence(self):
        search_payload = {
            "total_count": 1,
            "items": [
                {
                    "full_name": "example/project",
                    "html_url": "https://github.com/example/project",
                    "url": "https://api.github.com/repos/example/project",
                    "description": "A project",
                    "default_branch": "main",
                }
            ],
        }
        with patch("tools.github_rest.fetch_json", return_value=search_payload), patch("tools.github_rest._github_token", return_value=""):
            result = json.loads(
                ToolRegistry.execute(
                    "search_github_rest",
                    {"query": "example project", "scope": "repositories"},
                    {},
                    phase="DISCOVERY",
                )
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["results"][0]["url"], "https://github.com/example/project")

        encoded = base64.b64encode("RWKV evidence".encode()).decode()
        with patch("tools.github_rest.fetch_json", return_value={
            "name": "README.md",
            "path": "README.md",
            "content": encoded,
            "encoding": "base64",
            "html_url": "https://github.com/example/project/blob/main/README.md",
        }):
            evidence = json.loads(
                ToolRegistry.execute(
                    "fetch_github_rest",
                    {"url": "https://github.com/example/project/blob/main/README.md"},
                    {},
                    phase="EXTRACTION",
                )
            )
        self.assertEqual(evidence["status"], "ok")
        self.assertIn("RWKV evidence", evidence["results"][0]["page_excerpt"])

    def test_mediawiki_search_and_page_evidence(self):
        search_payload = {
            "query": {
                "search": [
                    {"pageid": 1, "title": "深圳地铁", "snippet": "城市轨道交通"}
                ]
            }
        }
        with patch("tools.mediawiki.fetch_json", return_value=search_payload):
            result = json.loads(
                ToolRegistry.execute(
                    "search_mediawiki",
                    {"query": "深圳地铁", "project": "wikipedia", "language": "zh"},
                    {},
                    phase="DISCOVERY",
                )
            )
        self.assertEqual(result["status"], "ok")
        self.assertIn("wikipedia.org/wiki", result["results"][0]["url"])

        with patch("tools.mediawiki.fetch_json", return_value={
            "query": {
                "pages": [
                    {
                        "pageid": 1,
                        "title": "深圳地铁",
                        "extract": "深圳地铁是城市轨道交通系统。",
                        "fullurl": "https://zh.wikipedia.org/wiki/深圳地铁",
                    }
                ]
            }
        }):
            evidence = json.loads(
                ToolRegistry.execute(
                    "fetch_mediawiki_page",
                    {"url": "https://zh.wikipedia.org/wiki/深圳地铁"},
                    {},
                    phase="EXTRACTION",
                )
            )
        self.assertEqual(evidence["status"], "ok")
        self.assertIn("城市轨道交通", evidence["results"][0]["page_excerpt"])


if __name__ == "__main__":
    unittest.main()
