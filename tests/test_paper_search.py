from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from agent.retrieval_loop import merge_retrieval_results
from tools.paper_search import search_papers


class PaperSearchTests(unittest.TestCase):
    def test_agentic_paper_search_is_discovery_only(self):
        record = {
            "title": "RWKV retrieval",
            "authors": ["Author"],
            "published": "2026-01-01",
            "abstract": "An abstract",
            "doi": "10.1000/example",
            "url": "https://doi.org/10.1000/example",
            "citations": 1,
            "source": "OpenAlex",
        }
        hydrate_limits = []

        def observe_hydration(rows, task_id, limit):
            hydrate_limits.append(limit)
            return rows

        with (
            patch("tools.paper_search._openalex", return_value=[record]),
            patch("tools.paper_search._arxiv", return_value=[]),
            patch("tools.paper_search._crossref", return_value=[]),
            patch("tools.paper_search._hydrate_evidence", side_effect=observe_hydration),
        ):
            payload = json.loads(search_papers("RWKV retrieval", agentic_tool_loop=True))

        self.assertEqual(hydrate_limits, [0])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["results"][0]["url"], "https://doi.org/10.1000/example")

    def test_paper_results_only_create_citations_when_evidence_exists(self):
        record = {
            "title": "RWKV retrieval",
            "authors": ["Author"],
            "published": "2026-01-01",
            "abstract": "A compact abstract with the reviewed fact.",
            "doi": "10.1000/example",
            "url": "https://doi.org/10.1000/example",
            "citations": 1,
            "source": "OpenAlex",
        }
        without_evidence = {
            **record,
            "title": "Metadata only",
            "abstract": "",
            "url": "https://doi.org/10.1000/metadata",
            "source": "Crossref",
        }
        with (
            patch("tools.paper_search._openalex", return_value=[record]),
            patch("tools.paper_search._arxiv", return_value=[]),
            patch("tools.paper_search._crossref", return_value=[without_evidence]),
            patch("tools.paper_search._hydrate_evidence", side_effect=lambda rows, task_id, limit: rows),
        ):
            payload = json.loads(search_papers("RWKV retrieval", max_results=4))

        self.assertEqual(payload["count"], 2)
        self.assertEqual(len(payload["citation_refs"]), 1)
        self.assertEqual(payload["evidence_missing_count"], 1)
        self.assertEqual(payload["citation_refs"][0]["url"], "https://doi.org/10.1000/example")

    def test_merge_deduplicates_citations_by_url_across_rounds(self):
        first = {
            "results": [{"title": "Source", "url": "https://example.com/source", "content": "fact"}],
            "sources": ["https://example.com/source"],
            "citation_refs": [{"ref_id": "S1", "url": "https://example.com/source", "evidence_text": "fact"}],
            "real_network": False,
        }
        second = {
            "results": [{"title": "Source", "url": "https://example.com/source", "content": "fact"}],
            "sources": ["https://example.com/source"],
            "citation_refs": [{"ref_id": "S2", "url": "https://example.com/source", "evidence_text": "fact"}],
            "real_network": False,
        }
        merged = merge_retrieval_results("source", "search_papers", [("source", first), ("source", second)])
        self.assertEqual(len(merged["results"]), 1)
        self.assertEqual(len(merged["citation_refs"]), 1)

    def test_merge_canonicalizes_doi_and_landing_page_variants(self):
        first = {
            "results": [{"title": "Source", "doi": "10.1000/example", "url": "https://doi.org/10.1000/example"}],
            "citation_refs": [{"ref_id": "S1", "doi": "10.1000/example", "url": "https://doi.org/10.1000/example"}],
            "real_network": True,
        }
        second = {
            "results": [{"title": "Source", "doi": "10.1000/example", "url": "https://publisher.example/paper?id=1"}],
            "citation_refs": [{"ref_id": "S2", "doi": "10.1000/example", "url": "https://publisher.example/paper?id=1"}],
            "real_network": True,
        }
        merged = merge_retrieval_results("source", "search_papers", [("source", first), ("source", second)])
        self.assertEqual(len(merged["results"]), 1)
        self.assertEqual(len(merged["citation_refs"]), 1)


if __name__ == "__main__":
    unittest.main()
