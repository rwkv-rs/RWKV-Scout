from __future__ import annotations

import json
from unittest.mock import patch

from tools.connectors import connector_lookup


@patch("tools.connectors.search_papers")
def test_paper_connector_serializes_structured_version_and_dates(search_papers):
    search_papers.return_value = json.dumps(
        {
            "status": "ok",
            "results": [
                {
                    "title": "Example Paper",
                    "source": "arXiv",
                    "url": "https://arxiv.org/abs/2501.00001v2",
                    "version": "v2",
                    "published": "2025-01-01",
                    "updated": "2025-02-03",
                    "abstract": "The provider abstract.",
                }
            ],
        }
    )

    payload = json.loads(
        connector_lookup(
            "papers",
            "Example Paper",
            scope="paper",
            original_goal="Find the latest arXiv version and revision date.",
        )
    )

    evidence = payload["results"][0]["structured_evidence_text"]
    assert "arXiv version: v2" in evidence
    assert "Published: 2025-01-01" in evidence
    assert "Updated: 2025-02-03" in evidence
    assert "Abstract: The provider abstract." in evidence
