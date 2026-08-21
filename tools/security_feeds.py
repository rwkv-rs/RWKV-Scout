"""Structured security-advisory feed lookups for the connector surface.

Each source below is the vendor's own machine-readable advisory publication
(a JSON feed, a REST endpoint, or the official advisory index page). The
module is pure transport: it resolves the model-supplied vendor identity to
one feed, fetches it, and returns the newest entries verbatim as typed source
rows. It never decides which advisory answers a question — that remains the
model's job downstream.

Every row carries a ``source_record_id`` so downstream URL-identity merging
(which strips query strings and fragments) can never collapse distinct feed
entries that share one catalog URL.
"""

from __future__ import annotations

import json
import re
from typing import Any

from utils.network_fetch import NetworkFetchError, fetch_json, fetch_text

_USER_AGENT = {"User-Agent": "RWKV-ECRA/0.1 structured advisory lookup"}
_MAX_ROWS = 8
_EVIDENCE_CHARS = 4800


def _clip(value: Any, limit: int = _EVIDENCE_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def _row(
    *,
    title: str,
    url: str,
    published: str,
    source: str,
    evidence: Any,
    record_id: str,
) -> dict[str, Any]:
    text = (
        json.dumps(evidence, ensure_ascii=False)
        if isinstance(evidence, (dict, list))
        else str(evidence or "")
    )
    return {
        "title": _clip(title, 300),
        "url": str(url or ""),
        "published": _clip(published, 40),
        "published_at": _clip(published, 40),
        "snippet": _clip(text, 600),
        "structured_evidence_text": _clip(text, _EVIDENCE_CHARS),
        "source": source,
        "source_record_id": _clip(record_id, 200),
    }


def _fetch_cisa_kev(query: str, max_results: int) -> dict[str, Any]:
    payload = fetch_json(
        "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        timeout=30,
        headers=_USER_AGENT,
    )
    entries = [
        item
        for item in payload.get("vulnerabilities") or []
        if isinstance(item, dict)
    ]
    entries.sort(key=lambda item: str(item.get("dateAdded") or ""), reverse=True)
    catalog_meta = {
        "catalogVersion": payload.get("catalogVersion"),
        "dateReleased": payload.get("dateReleased"),
        "count": payload.get("count"),
    }
    rows = [
        _row(
            title=(
                f"{item.get('cveID')} {item.get('vendorProject')} "
                f"{item.get('product')} (added {item.get('dateAdded')})"
            ),
            url="https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
            published=str(item.get("dateAdded") or ""),
            source="CISA Known Exploited Vulnerabilities Catalog (official JSON feed)",
            evidence={**item, "catalog": catalog_meta},
            record_id=f"cisa-kev:{item.get('cveID')}",
        )
        for item in entries[:max_results]
    ]
    return {"status": "ok" if rows else "no_results", "results": rows}


def _fetch_kubernetes_cve_feed(query: str, max_results: int) -> dict[str, Any]:
    payload = fetch_json(
        "https://kubernetes.io/docs/reference/issues-security/official-cve-feed/index.json",
        timeout=30,
        headers=_USER_AGENT,
    )
    items = [item for item in payload.get("items") or [] if isinstance(item, dict)]
    items.sort(key=lambda item: str(item.get("date_published") or ""), reverse=True)
    rows = [
        _row(
            title=f"{item.get('id')} {item.get('summary') or item.get('title') or ''}",
            url=str(item.get("external_url") or item.get("url") or ""),
            published=str(item.get("date_published") or ""),
            source="Kubernetes Official CVE Feed (kubernetes.io JSON feed)",
            evidence=item,
            record_id=f"k8s-cve:{item.get('id')}",
        )
        for item in items[:max_results]
    ]
    return {"status": "ok" if rows else "no_results", "results": rows}


def _threat_values(vulnerability: dict[str, Any]) -> list[str]:
    return [
        str((threat.get("Description") or {}).get("Value") or "")
        for threat in vulnerability.get("Threats") or []
        if isinstance(threat, dict)
    ]


def _fetch_msrc_updates(query: str, max_results: int) -> dict[str, Any]:
    listing = fetch_json(
        "https://api.msrc.microsoft.com/cvrf/v3.0/updates",
        timeout=30,
        headers={**_USER_AGENT, "Accept": "application/json"},
    )
    items = [item for item in listing.get("value") or [] if isinstance(item, dict)]
    items.sort(
        key=lambda item: str(item.get("InitialReleaseDate") or item.get("CurrentReleaseDate") or ""),
        reverse=True,
    )
    if not items:
        return {"status": "no_results", "results": []}
    newest = items[0]
    cvrf_url = str(newest.get("CvrfUrl") or "")
    document = fetch_json(
        cvrf_url,
        timeout=60,
        headers={**_USER_AGENT, "Accept": "application/json"},
    )
    tracking = document.get("DocumentTracking") or {}
    initial = str(tracking.get("InitialReleaseDate") or newest.get("InitialReleaseDate") or "")
    current = str(tracking.get("CurrentReleaseDate") or newest.get("CurrentReleaseDate") or "")
    vulnerabilities = [
        item for item in document.get("Vulnerability") or [] if isinstance(item, dict)
    ]
    exploited = [
        item
        for item in vulnerabilities
        if any("Exploited:Yes" in value for value in _threat_values(item))
    ]
    critical = [
        item
        for item in vulnerabilities
        if item not in exploited
        and any(value.strip() == "Critical" for value in _threat_values(item))
    ]
    doc_title = str(
        (document.get("DocumentTitle") or {}).get("Value")
        or newest.get("DocumentTitle")
        or newest.get("ID")
        or ""
    )
    rows = [
        _row(
            title=(
                f"MSRC {doc_title}: released {initial}, "
                f"{len(vulnerabilities)} vulnerabilities, "
                f"{len(exploited)} exploited in the wild, {len(critical)} additional Critical"
            ),
            url=cvrf_url,
            published=initial,
            source="Microsoft Security Response Center CVRF API (official)",
            evidence={
                "DocumentTitle": doc_title,
                "InitialReleaseDate": initial,
                "CurrentReleaseDate_revision": current,
                "vulnerability_count": len(vulnerabilities),
                "exploited_in_wild": [item.get("CVE") for item in exploited],
                "critical_sample": [item.get("CVE") for item in critical[:12]],
            },
            record_id=f"msrc:{newest.get('ID')}",
        )
    ]
    for item in [*exploited, *critical][: max(0, max_results - 1)]:
        values = _threat_values(item)
        flags = {
            "exploited_in_wild": any("Exploited:Yes" in value for value in values),
            "severity": next(
                (value for value in values if value.strip() in {"Critical", "Important", "Moderate", "Low"}),
                "",
            ),
        }
        rows.append(
            _row(
                title=(
                    f"{item.get('CVE')} {(item.get('Title') or {}).get('Value')}"
                    f"{' [Exploited in the wild]' if flags['exploited_in_wild'] else ''}"
                ),
                url=cvrf_url,
                published=initial,
                source="Microsoft Security Response Center CVRF API (official)",
                evidence={
                    "CVE": item.get("CVE"),
                    "Title": (item.get("Title") or {}).get("Value"),
                    "InitialReleaseDate": initial,
                    **flags,
                    "threat_descriptions": sorted(set(values))[:8],
                },
                record_id=f"msrc:{item.get('CVE')}",
            )
        )
    return {"status": "ok", "results": rows}


def _fetch_github_advisories(query: str, max_results: int) -> dict[str, Any]:
    body = fetch_text(
        "https://api.github.com/advisories",
        params={"per_page": max(1, min(max_results, 10))},
        timeout=30,
        headers={**_USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        items = json.loads(body)
    except json.JSONDecodeError as exc:
        raise NetworkFetchError(f"Invalid JSON from GitHub advisories: {body[:200]}") from exc
    rows = [
        _row(
            title=(
                f"{item.get('ghsa_id')} [{item.get('severity')}] "
                f"{item.get('summary')} (published {item.get('published_at')})"
            ),
            url=str(item.get("html_url") or ""),
            published=str(item.get("published_at") or ""),
            source="GitHub Security Advisories REST API (official; package/ecosystem advisories, not GitHub Enterprise Server release notes)",
            evidence={
                key: item.get(key)
                for key in (
                    "ghsa_id",
                    "cve_id",
                    "summary",
                    "severity",
                    "published_at",
                    "updated_at",
                    "vulnerabilities",
                    "html_url",
                )
            },
            record_id=f"ghsa:{item.get('ghsa_id')}",
        )
        for item in items
        if isinstance(item, dict)
    ][:max_results]
    return {"status": "ok" if rows else "no_results", "results": rows}


_TAG_RE = re.compile(r"<[^>]+>")
_MFSA_LINK_RE = re.compile(
    r"<a[^>]+href=\"(?P<href>/[^\"]*security/advisories/mfsa(?P<id>\d{4}-\d+)/?)\"[^>]*>(?P<text>.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_TEXT_DATE_RE = re.compile(
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)


def _product_token_mismatch(query: str, row_text: str) -> bool:
    """Mechanical product-token boundary for feeds mixing sibling products."""

    wants = str(query or "").casefold()
    text = str(row_text or "").casefold()
    if "firefox" in wants and "thunderbird" not in wants:
        return "thunderbird" in text and "firefox" not in text
    if "thunderbird" in wants and "firefox" not in wants:
        return "firefox" in text and "thunderbird" not in text
    return False


def _fetch_mozilla_advisories(query: str, max_results: int) -> dict[str, Any]:
    body = fetch_text(
        "https://www.mozilla.org/en-US/security/advisories/",
        timeout=30,
        headers=_USER_AGENT,
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _MFSA_LINK_RE.finditer(body):
        mfsa_id = match.group("id")
        if mfsa_id in seen:
            continue
        text = _clip(_TAG_RE.sub(" ", match.group("text")), 300)
        if _product_token_mismatch(query, text):
            continue
        seen.add(mfsa_id)
        # The index lists advisories newest-first; keep a window of page text
        # around the link so impact levels stay verbatim, and take the nearest
        # preceding date heading (one heading covers a batch of advisories) as
        # the announcement date.
        before = _TAG_RE.sub(" ", body[max(0, match.start() - 2400) : match.start()])
        dates = _TEXT_DATE_RE.findall(before)
        context = _clip(
            _TAG_RE.sub(" ", body[max(0, match.start() - 600) : match.end() + 200]),
            700,
        )
        label = re.sub(r"^\s*MFSA\s+\d{4}-\d+\s*", "", text)
        rows.append(
            _row(
                title=f"MFSA {mfsa_id}: {label}",
                url=f"https://www.mozilla.org{match.group('href')}",
                published=dates[-1] if dates else "",
                source="Mozilla Foundation Security Advisories index (official)",
                evidence=context,
                record_id=f"mfsa:{mfsa_id}",
            )
        )
        if len(rows) >= max_results:
            break
    return {"status": "ok" if rows else "no_results", "results": rows}


_OPENSSL_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}")


def _fetch_openssl_vulnerabilities(query: str, max_results: int) -> dict[str, Any]:
    body = fetch_text(
        "https://openssl-library.org/news/vulnerabilities/",
        timeout=30,
        headers=_USER_AGENT,
    )
    text = _TAG_RE.sub(" ", body)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    # The official page lists entries newest-first; keep each CVE's verbatim
    # surrounding text so severity/fixed-version/date stay quotable.
    for match in _OPENSSL_CVE_RE.finditer(text):
        cve = match.group(0)
        if cve in seen:
            continue
        seen.add(cve)
        context = _clip(text[max(0, match.start() - 200) : match.end() + 900], 1000)
        dates = _TEXT_DATE_RE.findall(context)
        rows.append(
            _row(
                title=f"OpenSSL vulnerability {cve}",
                url="https://openssl-library.org/news/vulnerabilities/",
                published=dates[0] if dates else "",
                source="OpenSSL vulnerabilities page (official)",
                evidence=context,
                record_id=f"openssl:{cve}",
            )
        )
        if len(rows) >= max_results:
            break
    return {"status": "ok" if rows else "no_results", "results": rows}


_FEED_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "slug": "cisa_kev",
        "label": "CISA Known Exploited Vulnerabilities catalog",
        "keywords": ("cisa", "kev", "known exploited"),
        "fetch": _fetch_cisa_kev,
    },
    {
        "slug": "mozilla_mfsa",
        "label": "Mozilla/Firefox security advisories (MFSA)",
        "keywords": ("mozilla", "firefox", "mfsa", "thunderbird"),
        "fetch": _fetch_mozilla_advisories,
    },
    {
        "slug": "microsoft_msrc",
        "label": "Microsoft security updates (MSRC)",
        "keywords": ("microsoft", "msrc", "patch tuesday", "windows", "微软"),
        "fetch": _fetch_msrc_updates,
    },
    {
        "slug": "kubernetes_cve",
        "label": "Kubernetes official CVE feed",
        "keywords": ("kubernetes", "k8s"),
        "fetch": _fetch_kubernetes_cve_feed,
    },
    {
        "slug": "openssl",
        "label": "OpenSSL vulnerabilities page",
        "keywords": ("openssl",),
        "fetch": _fetch_openssl_vulnerabilities,
    },
    {
        "slug": "github_advisories",
        "label": "GitHub Security Advisories (GHSA; package/ecosystem advisories only)",
        "keywords": ("ghsa", "github advisory", "github advisories"),
        "fetch": _fetch_github_advisories,
    },
)


def supported_advisory_sources() -> list[str]:
    return [f"{item['slug']} ({item['label']})" for item in _FEED_CATALOG]


def _resolve_feed(query: str) -> dict[str, Any] | None:
    """Resolve the model-supplied vendor identity to one catalogued feed.

    Mechanical substring identity resolution only (same nature as owner/repo
    or package-name parsing); no relevance judgement.
    """

    text = " ".join(str(query or "").split()).casefold()
    if not text:
        return None
    for entry in _FEED_CATALOG:
        if entry["slug"] in text.replace("-", "_"):
            return entry
    for entry in _FEED_CATALOG:
        if any(keyword in text for keyword in entry["keywords"]):
            return entry
    return None


def security_advisories_payload(query: str, max_results: int = 8) -> dict[str, Any]:
    """Fetch the newest entries of one vendor's official advisory feed."""

    limit = max(1, min(int(max_results or _MAX_ROWS), 10))
    entry = _resolve_feed(query)
    if entry is None:
        return {
            "status": "error",
            "error_class": "object_type_mismatch",
            "identity_error_class": "unsupported_advisory_source",
            "query": query,
            "results": [],
            "supported_sources": supported_advisory_sources(),
            "provider_errors": [
                "security_advisories requires one supported vendor identity "
                "(CISA KEV / Mozilla MFSA / Microsoft MSRC / Kubernetes / OpenSSL / "
                "GitHub package advisories); Cisco, Android, Chrome, GitLab and "
                "GitHub Enterprise Server advisories remain available through web_search"
            ],
        }
    try:
        payload = entry["fetch"](query, limit)
    except NetworkFetchError as exc:
        return {
            "status": "error",
            "error_class": "network_error",
            "query": query,
            "results": [],
            "provider_errors": [f"{type(exc).__name__}: {exc}"],
        }
    payload.setdefault("query", query)
    payload["provider"] = f"security.{entry['slug']}"
    payload["advisory_source"] = entry["label"]
    payload["sources"] = [
        row.get("url") for row in payload.get("results") or [] if row.get("url")
    ]
    return payload


__all__ = ["security_advisories_payload", "supported_advisory_sources"]
