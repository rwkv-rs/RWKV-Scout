"""Real, keyless scholarly retrieval for paper and series queries."""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from html import unescape

from tools.registry import ToolRegistry
from utils.network_fetch import NetworkFetchError, fetch_json, fetch_text
from utils.retrieval_events import record_retrieval_event
from utils.web_retrieval import attach_page


ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _abstract_from_inverted_index(index: dict) -> str:
    words: list[tuple[int, str]] = []
    for word, positions in (index or {}).items():
        for position in positions or []:
            words.append((int(position), word))
    return " ".join(word for _, word in sorted(words))


def _authors(authorships: list[dict]) -> list[str]:
    names = []
    for author in authorships or []:
        name = ((author.get("author") or {}).get("display_name") or "").strip()
        if name:
            names.append(name)
    return names


def _clean_abstract(value: str) -> str:
    text = unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return " ".join(text.split())


def _openalex(query: str, limit: int) -> list[dict]:
    payload = fetch_json(
        "https://api.openalex.org/works",
        {
            "search": query,
            "per-page": min(limit, 25),
            "select": "id,doi,title,publication_date,authorships,primary_location,cited_by_count,open_access,abstract_inverted_index",
        },
    )
    records = []
    for work in payload.get("results") or []:
        location = work.get("primary_location") or {}
        landing = location.get("landing_page_url") or ""
        doi = work.get("doi") or ""
        records.append(
            {
                "title": (work.get("title") or "").strip(),
                "authors": _authors(work.get("authorships") or []),
                "published": work.get("publication_date") or "",
                "abstract": _abstract_from_inverted_index(work.get("abstract_inverted_index") or {}),
                "doi": doi,
                "url": landing or doi or work.get("id") or "",
                "citations": work.get("cited_by_count", 0),
                "source": "OpenAlex",
            }
        )
    return records


def _arxiv(query: str, limit: int) -> list[dict]:
    body = fetch_text(
        "https://export.arxiv.org/api/query",
        {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": min(limit, 25),
            "sortBy": "relevance",
        },
    )
    root = ET.fromstring(body)
    records = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        link = ""
        for item in entry.findall(f"{ATOM_NS}link"):
            if item.attrib.get("rel") == "alternate":
                link = item.attrib.get("href", "")
                break
        authors = [
            (node.findtext(f"{ATOM_NS}name") or "").strip()
            for node in entry.findall(f"{ATOM_NS}author")
        ]
        records.append(
            {
                "title": " ".join((entry.findtext(f"{ATOM_NS}title") or "").split()),
                "authors": [name for name in authors if name],
                "published": (entry.findtext(f"{ATOM_NS}published") or "")[:10],
                "abstract": " ".join((entry.findtext(f"{ATOM_NS}summary") or "").split()),
                "doi": "",
                "url": link or (entry.findtext(f"{ATOM_NS}id") or ""),
                "citations": None,
                "source": "arXiv",
            }
        )
    return records


def _crossref(query: str, limit: int) -> list[dict]:
    payload = fetch_json(
        "https://api.crossref.org/works",
        {
            "query.title": query,
            "rows": min(limit, 25),
            "select": "DOI,title,author,published,URL,abstract",
        },
    )
    records = []
    for work in ((payload.get("message") or {}).get("items") or []):
        date_parts = ((work.get("published") or {}).get("date-parts") or [[]])[0]
        records.append(
            {
                "title": ((work.get("title") or [""])[0]).strip(),
                "authors": [
                    " ".join(part for part in [a.get("given", ""), a.get("family", "")] if part).strip()
                    for a in work.get("author") or []
                ],
                "published": "-".join(str(part) for part in date_parts),
                "abstract": _clean_abstract(work.get("abstract", "")),
                "doi": work.get("DOI", ""),
                "url": work.get("URL", ""),
                "citations": None,
                "source": "Crossref",
            }
        )
    return records


def _provider_query(query: str) -> str:
    """Normalize model-generated text without semantic benchmark shortcuts."""
    raw = " ".join((query or "").split())
    cleaned = re.sub(
        r"(?:请)?检索并回答|(?:请)?检索|给出关键证据|来源链接|难度下的限制|领域的任务\s*\d+|任务\s*\d+",
        " ",
        raw,
    )
    cleaned = re.sub(
        r"(?i)\b(search|find|look up|list|latest|papers?|authors?|abstracts?|links?|related|series)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"[，。！？、：；,.!?]", " ", cleaned)
    return " ".join(cleaned.split())[:180] or raw[:180]


def _deduplicate(records: list[dict], limit: int) -> list[dict]:
    seen: dict[str, int] = {}
    unique = []
    for record in records:
        url = str(record.get("url") or "").lower()
        arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", url)
        if arxiv_match:
            key = f"arxiv:{arxiv_match.group(1)}"
        else:
            key = re.sub(r"\W+", " ", str(record.get("title") or "").casefold()).strip()
            key = f"title:{key or (record.get('doi') or url).casefold()}"
        if not key or key in seen:
            if key in seen:
                current = unique[seen[key]]
                priority = {"arXiv": 3, "OpenAlex": 2, "Crossref": 1}
                if priority.get(record.get("source"), 0) > priority.get(current.get("source"), 0):
                    for field in ("title", "authors", "published", "abstract", "url", "source"):
                        if record.get(field):
                            current[field] = record[field]
            continue
        seen[key] = len(unique)
        unique.append(record)
    return unique[:limit]


def _short_provider_error(name: str, exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    detail = lines[-1] if lines else str(exc)
    return f"{name}: {detail[-300:]}"


def _hydrate_evidence(records: list[dict], task_id: str, limit: int = 4) -> list[dict]:
    """Attach provider abstracts or bounded page text to paper records."""
    hydrated = []
    page_budget = max(0, int(limit))
    for record in records:
        value = dict(record)
        abstract = _clean_abstract(value.get("abstract", ""))
        if abstract:
            value["abstract"] = abstract
            value["content"] = abstract
            value["page_excerpt"] = abstract
            value["evidence_type"] = "provider_abstract"
            value["evidence_source"] = value.get("source", "")
            value["body_available"] = True
            value["body_chars"] = len(abstract)
            hydrated.append(value)
            continue
        if page_budget and value.get("url"):
            page_budget -= 1
            started = time.perf_counter()
            try:
                value = attach_page(value, timeout=12)
                body = str(value.get("page_excerpt") or value.get("content") or "").strip()
                if body:
                    value["evidence_type"] = "page_text"
                    value["evidence_source"] = value.get("url", "")
                record_retrieval_event(
                    task_id,
                    "page_fetch",
                    action="search_papers",
                    url=value.get("url", ""),
                    status="completed" if body else "empty",
                    body_chars=len(body),
                    captured_at=value.get("captured_at", ""),
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
                record_retrieval_event(
                    task_id,
                    "page_extract",
                    action="search_papers",
                    url=value.get("url", ""),
                    status="completed" if body else "empty",
                    excerpt_chars=len(body),
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            except (NetworkFetchError, ValueError, OSError) as exc:
                value["body_available"] = False
                value["body_error"] = str(exc)[-300:]
                record_retrieval_event(
                    task_id,
                    "page_fetch",
                    action="search_papers",
                    url=value.get("url", ""),
                    status="failed",
                    error=str(exc)[:500],
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
        hydrated.append(value)
    return hydrated


@ToolRegistry.register(
    name="search_papers",
    phase="ALL",
    plugin="scholarly.multi_source",
    capabilities=("url_discovery", "scholarly_search"),
    retrieval_role="discovery",
    signature="""[Tool] search_papers
- 功能: 真实调用 OpenAlex、arXiv 和 Crossref 的公开接口，检索论文或一个论文/模型系列；不需要 API Key。
- 参数: query (论文名、作者、主题或系列关键词), scope (paper 或 series), max_results (最多 12)""",
)
def search_papers(
    query: str,
    scope: str = "paper",
    max_results: int = 8,
    working_memory: dict | None = None,
    agent_state=None,
    **kwargs,
) -> str:
    query = (query or "").strip()
    if not query:
        return json.dumps({"status": "error", "message": "query is empty"}, ensure_ascii=False)

    limit = max(1, min(int(max_results or 8), 12))
    provider_query = _provider_query(query)
    errors = []
    records: list[dict] = []
    for name, fetcher in (("OpenAlex", _openalex), ("arXiv", _arxiv), ("Crossref", _crossref)):
        try:
            records.extend(fetcher(provider_query, limit))
        except (NetworkFetchError, ET.ParseError, ValueError) as exc:
            errors.append(_short_provider_error(name, exc))

    records = _deduplicate(records, limit)
    task_id = str(kwargs.get("task_id") or "")
    # In the model-owned route this tool is discovery only.  The model must
    # select one returned paper URL; the shared fetch/chunk/candidate pipeline
    # is the only path that can turn it into final evidence.  Legacy direct
    # experiments may still request hydrated paper metadata explicitly.
    agentic_tool_loop = bool(kwargs.get("agentic_tool_loop"))
    records = _hydrate_evidence(
        records,
        task_id,
        limit=0 if agentic_tool_loop else min(4, len(records)),
    )
    now = datetime.now().isoformat(timespec="seconds")
    safe = re.sub(r"[^\w\-]+", "_", query, flags=re.UNICODE)[:40] or "query"
    sources = [record.get("url") for record in records if record.get("url")]
    citation_refs = [
        {
            "ref_id": f"PAPER_REF_{safe}_{index}",
            "title": record.get("title", ""),
            "url": record.get("url", ""),
            "source": record.get("source", ""),
            "evidence_text": str(record.get("page_excerpt") or record.get("content") or record.get("abstract") or "")[:6000],
            "evidence_type": record.get("evidence_type", ""),
        }
        for index, record in enumerate(records, start=1)
        if str(record.get("page_excerpt") or record.get("content") or record.get("abstract") or "").strip()
    ]
    result = {
        "status": "ok" if records else "no_results",
        "real_network": True,
        "scope": scope if scope in {"paper", "series"} else "paper",
        "query": query,
        "provider_query": provider_query,
        "retrieved_at": now,
        "count": len(records),
        "results": records,
        "sources": sources,
        "citation_refs": citation_refs,
        "evidence_missing_count": sum(
            not bool(str(record.get("page_excerpt") or record.get("content") or record.get("abstract") or "").strip())
            for record in records
        ),
        "provider_errors": errors,
    }
    result_text = json.dumps(result, ensure_ascii=False, indent=2)

    if working_memory is not None:
        memory_key = f"WebFact_Papers_{safe}"
        working_memory[memory_key] = result_text
        structured = working_memory.setdefault("__web_structured_facts__", [])
        for index, record in enumerate(records, start=1):
            structured.append(
                {
                    "ref_id": f"PAPER_REF_{safe}_{index}",
                    "title": record.get("title", ""),
                    "url": record.get("url", ""),
                    "content": f"{record.get('source', '')} real retrieval",
                }
            )

    if agent_state is not None and not kwargs.get("agentic_tool_loop"):
        agent_state.is_finished = True
        if records:
            preview = "\n".join(
                f"{i}. {record.get('title')} ({record.get('published')}) — {record.get('url')}"
                for i, record in enumerate(records[:5], start=1)
            )
            agent_state.final_result = (
                f"已通过 OpenAlex/arXiv/Crossref 真实检索“{query}”，找到 {len(records)} 条结果。\n{preview}"
            )
        else:
            agent_state.final_result = f"已真实调用论文源，但没有找到“{query}”的结果。"

    return result_text
