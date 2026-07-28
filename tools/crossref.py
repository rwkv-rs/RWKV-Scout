"""Crossref REST discovery and evidence tools.

Crossref is deliberately split into discovery and evidence operations.  The
search call returns DOI candidates and short metadata; the model must select a
candidate and call ``fetch_crossref_record`` before the result can enter the
shared chunk/parallel-candidate pipeline.
"""

from __future__ import annotations

import json
import re
from html import unescape
from typing import Any
from urllib.parse import quote, unquote, urlparse

from tools.registry import ToolRegistry
from utils.network_fetch import NetworkFetchError, fetch_json


_CROSSREF_ENDPOINT = "https://api.crossref.org/works"
_CROSSREF_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "RWKV-ECRA/0.1 (research retrieval; Crossref REST)",
}


def _clean_abstract(value: Any) -> str:
    text = unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return " ".join(text.split())


def _work_authors(work: dict[str, Any]) -> list[str]:
    names = []
    for author in work.get("author") or []:
        if not isinstance(author, dict):
            continue
        name = " ".join(
            part.strip()
            for part in (str(author.get("given") or ""), str(author.get("family") or ""))
            if part.strip()
        )
        if name:
            names.append(name)
    return names


def _published(work: dict[str, Any]) -> str:
    date_parts = ((work.get("published") or {}).get("date-parts") or [[]])[0]
    return "-".join(str(part) for part in date_parts if part is not None)


def _record_from_work(work: dict[str, Any]) -> dict[str, Any]:
    doi = str(work.get("DOI") or "").strip()
    title = str((work.get("title") or [""])[0] or "").strip()
    landing_url = str(work.get("URL") or "").strip()
    doi_url = f"https://doi.org/{doi}" if doi else landing_url
    api_url = f"{_CROSSREF_ENDPOINT}/{quote(doi, safe='')}" if doi else ""
    abstract = _clean_abstract(work.get("abstract"))
    authors = _work_authors(work)
    snippet_parts = [title]
    if authors:
        snippet_parts.append("作者: " + ", ".join(authors[:6]))
    if _published(work):
        snippet_parts.append("年份: " + _published(work))
    if abstract:
        snippet_parts.append(abstract[:500])
    return {
        "title": title or doi_url,
        "url": doi_url,
        "api_url": api_url,
        "doi": doi,
        "authors": authors,
        "published": _published(work),
        "abstract": abstract,
        "publisher": str(work.get("publisher") or ""),
        "container_title": str((work.get("container-title") or [""])[0] or ""),
        "snippet": " ".join(snippet_parts)[:900],
        "source": "Crossref REST API",
        "untrusted_content": True,
    }


def query_crossref(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Return normalized Crossref work candidates for reuse by paper search."""

    payload = fetch_json(
        _CROSSREF_ENDPOINT,
        {
            "query.title": " ".join(str(query or "").split())[:180],
            "rows": max(1, min(int(limit or 8), 25)),
            "select": "DOI,title,author,published,URL,abstract,publisher,container-title",
        },
        timeout=20,
        headers=_CROSSREF_HEADERS,
    )
    records = [_record_from_work(item) for item in ((payload.get("message") or {}).get("items") or []) if isinstance(item, dict)]
    query_terms = {
        term.casefold()
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}|[\u3400-\u9fff]{2,}", query)
    }
    if query_terms:
        ranked = []
        for index, record in enumerate(records):
            title_terms = {
                term.casefold()
                for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}|[\u3400-\u9fff]{2,}", record.get("title", ""))
            }
            overlap = len(query_terms & title_terms)
            ranked.append((overlap, -index, record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        positive = [record for overlap, _, record in ranked if overlap]
        if positive:
            # Exact-title lookups should not flood the model with works that
            # happen to share only one generic word such as “era”.  Broad
            # topic/author lookups retain the ranked positive candidates.
            max_overlap = ranked[0][0]
            if len(query_terms) >= 4 and max_overlap >= 3:
                records = [record for overlap, _, record in ranked if overlap == max_overlap]
            else:
                records = positive
    return records[: max(1, min(int(limit or 8), 25))]


def _error(query: str, message: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "real_network": True,
            "provider": "crossref.rest",
            "query": query,
            "results": [],
            "provider_errors": [message[:500]],
        },
        ensure_ascii=False,
    )


@ToolRegistry.register(
    name="search_crossref",
    phase="ALL",
    plugin="crossref.rest",
    capabilities=("url_discovery", "scholarly_search", "crossref_rest"),
    retrieval_role="discovery",
    signature="""[Tool] search_crossref
- 功能: 使用无需 API Key 的 Crossref REST API 检索论文、作者、DOI 和出版信息。
- 参数: query (论文标题、作者或主题), max_results (最多 12)。
- 规则: 返回的是 DOI 候选；必须再调用 fetch_crossref_record 获取 API 正文证据。""",
)
def search_crossref(
    query: str,
    max_results: int = 8,
    working_memory: dict | None = None,
    agent_state=None,
    **kwargs: Any,
) -> str:
    del agent_state, kwargs
    query = " ".join(str(query or "").split()).strip()
    if not query:
        return _error(query, "query is empty")
    try:
        rows = query_crossref(query, max_results)
    except (NetworkFetchError, ValueError, TypeError) as exc:
        return _error(query, f"{type(exc).__name__}: {exc}")
    result = {
        "status": "ok" if rows else "no_results",
        "real_network": True,
        "provider": "crossref.rest",
        "query": query,
        "count": len(rows),
        "results": rows[: max(1, min(int(max_results or 8), 12))],
        "sources": [row["url"] for row in rows if row.get("url")],
        "citation_refs": [],
        "provider_errors": [],
        "evidence_policy": "Crossref 搜索结果是候选元数据，必须经过 fetch_crossref_record 和 chunk 证据流程。",
    }
    if working_memory is not None:
        working_memory[f"Crossref_{query[:60]}"] = json.dumps(result, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _doi_from_url(value: str) -> str:
    candidate = str(value or "").strip()
    parsed = urlparse(candidate)
    if parsed.netloc.casefold() == "api.crossref.org":
        prefix = "/works/"
        if parsed.path.startswith(prefix):
            return unquote(parsed.path[len(prefix) :]).strip().strip("/")
    if parsed.netloc.casefold() == "doi.org" or parsed.netloc.casefold().endswith(".doi.org"):
        return unquote(parsed.path.strip("/"))
    if candidate.lower().startswith("doi:"):
        return candidate[4:].strip()
    return ""


def _crossref_evidence_text(message: dict[str, Any], max_chars: int) -> str:
    record = _record_from_work(message)
    lines = [
        f"Title: {record.get('title', '')}",
        f"DOI: {record.get('doi', '')}",
        f"Authors: {', '.join(record.get('authors') or [])}",
        f"Published: {record.get('published', '')}",
        f"Publisher: {record.get('publisher', '')}",
        f"Container: {record.get('container_title', '')}",
        f"URL: {record.get('url', '')}",
        f"Abstract: {_clean_abstract(message.get('abstract'))}",
    ]
    return "\n".join(line for line in lines if line.split(": ", 1)[-1].strip())[:max_chars]


@ToolRegistry.register(
    name="fetch_crossref_record",
    phase="ALL",
    plugin="crossref.rest",
    capabilities=("page_fetch", "page_evidence", "crossref_rest"),
    retrieval_role="evidence",
    signature="""[Tool] fetch_crossref_record
- 功能: 读取模型从 Crossref 搜索结果中选择的单篇 DOI API 记录，返回可分块的正文证据。
- 参数: url (Crossref 返回的 DOI/API URL), max_chars (正文上限，默认 20000)。""",
)
def fetch_crossref_record(
    url: str,
    max_chars: int = 20000,
    working_memory: dict | None = None,
    agent_state=None,
    **kwargs: Any,
) -> str:
    del working_memory, agent_state, kwargs
    doi = _doi_from_url(url)
    if not doi:
        return _error(str(url or ""), "url must be a Crossref DOI URL or api.crossref.org works URL")
    api_url = f"{_CROSSREF_ENDPOINT}/{quote(doi, safe='')}"
    try:
        payload = fetch_json(api_url, timeout=20, headers=_CROSSREF_HEADERS)
        message = payload.get("message") or {}
        if not isinstance(message, dict):
            raise ValueError("Crossref response has no work record")
        record = _record_from_work(message)
        human_url = record.get("url") or f"https://doi.org/{doi}"
        page_text = _crossref_evidence_text(message, max(1000, min(int(max_chars or 20000), 30000)))
        result = {
            "status": "ok" if page_text else "no_results",
            "real_network": True,
            "provider": "crossref.rest",
            "query": doi,
            "results": [
                {
                    "title": record.get("title") or doi,
                    "url": human_url,
                    "api_url": api_url,
                    "snippet": page_text[:800],
                    "page_excerpt": page_text,
                    "content": page_text,
                    "source": "Crossref REST API",
                    "untrusted_content": True,
                }
            ],
            "sources": [human_url],
            "citation_refs": [
                {
                    "ref_id": f"CROSSREF_{re.sub(r'[^A-Za-z0-9]+', '_', doi)[:60]}",
                    "title": record.get("title") or doi,
                    "url": human_url,
                    "source": "Crossref REST API",
                }
            ],
            "provider_errors": [],
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (NetworkFetchError, ValueError, TypeError) as exc:
        return _error(doi, f"{type(exc).__name__}: {exc}")


__all__ = ["query_crossref", "search_crossref", "fetch_crossref_record"]
