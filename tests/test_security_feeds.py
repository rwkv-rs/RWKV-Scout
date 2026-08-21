from __future__ import annotations

import json
from unittest.mock import patch

from tools.security_feeds import security_advisories_payload
from tools.web_search_generic import _admit_cached_and_novel_candidates


def test_unsupported_vendor_returns_typed_error_without_substitution():
    payload = security_advisories_payload("Cisco IOS XE advisories")
    assert payload["status"] == "error"
    assert payload["error_class"] == "object_type_mismatch"
    assert payload["results"] == []
    assert any("cisa_kev" in row for row in payload["supported_sources"])


def test_cisa_kev_rows_are_newest_first_verbatim_entries():
    feed = {
        "catalogVersion": "2026.08.19",
        "dateReleased": "2026-08-19T00:00:00Z",
        "count": 3,
        "vulnerabilities": [
            {"cveID": "CVE-2026-1", "vendorProject": "A", "product": "P1", "dateAdded": "2026-08-03"},
            {"cveID": "CVE-2026-2", "vendorProject": "B", "product": "P2", "dateAdded": "2026-08-18"},
            {"cveID": "CVE-2026-3", "vendorProject": "C", "product": "P3", "dateAdded": "2026-07-21"},
        ],
    }
    with patch("tools.security_feeds.fetch_json", return_value=feed):
        payload = security_advisories_payload("CISA KEV 最近新增", max_results=2)
    assert payload["status"] == "ok"
    assert payload["provider"] == "security.cisa_kev"
    rows = payload["results"]
    assert len(rows) == 2
    assert "CVE-2026-2" in rows[0]["title"] and rows[0]["published"] == "2026-08-18"
    assert "CVE-2026-1" in rows[1]["title"]
    evidence = json.loads(rows[0]["structured_evidence_text"])
    assert evidence["cveID"] == "CVE-2026-2"


def test_msrc_rows_sorted_by_release_date():
    feed = {
        "value": [
            {"ID": "2026-Jul", "DocumentTitle": "July 2026", "CurrentReleaseDate": "2026-07-08T00:00:00Z", "CvrfUrl": "https://api.msrc.microsoft.com/cvrf/v3.0/document/2026-Jul"},
            {"ID": "2026-Aug", "DocumentTitle": "August 2026", "CurrentReleaseDate": "2026-08-11T00:00:00Z", "CvrfUrl": "https://api.msrc.microsoft.com/cvrf/v3.0/document/2026-Aug"},
        ]
    }
    with patch("tools.security_feeds.fetch_json", return_value=feed):
        payload = security_advisories_payload("Microsoft Patch Tuesday", max_results=1)
    assert payload["status"] == "ok"
    assert "August 2026" in payload["results"][0]["title"]


def test_mozilla_index_parses_mfsa_links_in_document_order():
    body = (
        '<h2>August 2026</h2>'
        '<a href="/en-US/security/advisories/mfsa2026-70/">Security Vulnerabilities fixed in Firefox 145</a>'
        '<a href="/en-US/security/advisories/mfsa2026-68/">Security Vulnerabilities fixed in Firefox 144</a>'
    )
    with patch("tools.security_feeds.fetch_text", return_value=body):
        payload = security_advisories_payload("Firefox 最新 MFSA")
    rows = payload["results"]
    assert payload["status"] == "ok"
    assert rows[0]["title"].startswith("MFSA 2026-70")
    assert rows[0]["url"] == "https://www.mozilla.org/en-US/security/advisories/mfsa2026-70/"
    assert rows[1]["title"].startswith("MFSA 2026-68")


def test_priority_hosts_reserve_fetch_slots_without_expanding_window():
    candidates = [
        {"url": f"https://third-{index}.example/page", "title": f"t{index}"}
        for index in range(8)
    ]
    candidates.append(
        {"url": "https://releases.ubuntu.com/26.04/", "title": "official"}
    )
    cached, novel = _admit_cached_and_novel_candidates(
        candidates,
        set(),
        limit=8,
        per_domain_limit=3,
        priority_hosts=["releases.ubuntu.com"],
    )
    assert cached == []
    assert len(novel) == 8
    reserved = [row for row in novel if row.get("priority_host_reserved")]
    assert len(reserved) == 1
    assert reserved[0]["url"].startswith("https://releases.ubuntu.com/")


def test_priority_hosts_do_not_duplicate_already_selected_candidates():
    candidates = [
        {"url": "https://releases.ubuntu.com/26.04/", "title": "official"},
        {"url": "https://third-a.example/page", "title": "a"},
        {"url": "https://third-b.example/page", "title": "b"},
    ]
    cached, novel = _admit_cached_and_novel_candidates(
        candidates,
        set(),
        limit=3,
        per_domain_limit=3,
        priority_hosts=["releases.ubuntu.com"],
    )
    assert len(novel) == 3
    assert sum(1 for row in novel if "ubuntu.com" in row["url"]) == 1
    assert not any(row.get("priority_host_reserved") for row in novel)
