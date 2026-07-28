"""MediaWiki/Wikimedia API discovery and evidence tools."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, unquote, urlparse

from tools.registry import ToolRegistry
from utils.html_markdown import html_to_markdown, markdown_table
from utils.network_fetch import NetworkFetchError, fetch_json


_PROJECTS = {
    "wikipedia",
    "wiktionary",
    "wikibooks",
    "wikinews",
    "wikiquote",
    "wikisource",
    "wikiversity",
    "wikivoyage",
    "wikidata",
    "commons",
}
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "RWKV-ECRA/0.1 (MediaWiki API retrieval)",
}


def _error(query: str, message: str, *, error_class: str = "provider_error") -> str:
    return json.dumps(
        {
            "status": "error",
            "real_network": True,
            "provider": "mediawiki.api",
            "query": query,
            "error_class": error_class,
            "results": [],
            "provider_errors": [message[:500]],
        },
        ensure_ascii=False,
    )


def _site(project: str, language: str) -> tuple[str, str]:
    project = str(project or "wikipedia").strip().casefold()
    language = str(language or "zh").strip().casefold()
    # Small models occasionally place the language code in ``project`` after
    # reading the bilingual catalog.  Normalize that value to the explicit
    # Wikipedia default without accepting arbitrary unknown project names.
    if project not in _PROJECTS and re.fullmatch(r"[a-z0-9-]{2,12}", project):
        language, project = project, "wikipedia"
    if project not in _PROJECTS:
        raise ValueError(f"project must be one of: {', '.join(sorted(_PROJECTS))}")
    if not re.fullmatch(r"[a-z0-9-]{2,12}", language):
        raise ValueError("language must be a short MediaWiki language code")
    if project == "commons":
        host = "commons.wikimedia.org"
    elif project == "wikidata":
        host = "www.wikidata.org"
    else:
        host = f"{language}.{project}.org"
    return f"https://{host}", f"https://{host}/w/api.php"


def _site_from_url(url: str, project: str, language: str) -> tuple[str, str]:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").casefold()
    if host.endswith(".wikipedia.org") or host.endswith(".wiktionary.org") or host.endswith(".wikibooks.org") or host.endswith(".wikinews.org") or host.endswith(".wikiquote.org") or host.endswith(".wikisource.org") or host.endswith(".wikiversity.org") or host.endswith(".wikivoyage.org") or host.endswith(".wikimedia.org") or host == "www.wikidata.org":
        return f"{parsed.scheme or 'https'}://{parsed.netloc}", f"{parsed.scheme or 'https'}://{parsed.netloc}/w/api.php"
    return _site(project, language)


def _page_url(base: str, title: str) -> str:
    normalized = str(title or "").strip().replace(" ", "_")
    return f"{base}/wiki/{quote(normalized, safe='()/:_')}"


def _strip_html(value: Any) -> str:
    return html_to_markdown(value)


def _revision_content(payload: dict[str, Any]) -> str:
    pages = ((payload.get("query") or {}).get("pages") or [])
    page = pages[0] if pages and isinstance(pages[0], dict) else {}
    revisions = page.get("revisions") or []
    revision = revisions[0] if revisions and isinstance(revisions[0], dict) else {}
    slots = revision.get("slots") or {}
    main = slots.get("main") or {}
    return str(main.get("content") or revision.get("content") or "")


def _clean_wikitext(value: str) -> str:
    """Convert readable wikitext fields and simple tables to Markdown."""

    text = str(value or "")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"[\2](\1)", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://(\S+)\s+([^\]]+)\]", r"[\2](https://\1)", text)
    text = re.sub(r"'{2,}", "", text)
    lines: list[str] = []
    table_rows: list[list[str]] = []
    in_table = False

    def flush_table() -> None:
        nonlocal table_rows
        if table_rows:
            lines.extend(markdown_table(table_rows))
            lines.append("")
            table_rows = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("{|"):
            flush_table()
            in_table = True
            continue
        if in_table and line.startswith("|}"):
            flush_table()
            in_table = False
            continue
        if in_table and line.startswith("|-"):
            continue
        if in_table and (line.startswith("!") or line.startswith("|")):
            marker = line[1:].strip()
            delimiter = "!!" if line.startswith("!") else "||"
            cells = [re.sub(r"\s+", " ", cell).strip() for cell in marker.split(delimiter)]
            if cells and any(cells):
                table_rows.append(cells)
            continue
        if not line:
            continue
        if line.startswith(("|", "!")):
            line = line[1:].strip()
        line = re.sub(r"^=+|=+$", "", line).strip()
        line = re.sub(r"\{\{[^{}]*\}\}", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    if in_table:
        flush_table()
    return "\n".join(lines)


def _fetch_wikitext(endpoint: str, title: str) -> str:
    payload = fetch_json(
        endpoint,
        {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "formatversion": 2,
            "utf8": 1,
            "origin": "*",
        },
        timeout=20,
        headers=_HEADERS,
    )
    return _revision_content(payload)


def _fetch_parsed_text(endpoint: str, title: str) -> str:
    payload = fetch_json(
        endpoint,
        {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "formatversion": 2,
            "utf8": 1,
            "origin": "*",
        },
        timeout=20,
        headers=_HEADERS,
    )
    return _strip_html(str(((payload.get("parse") or {}).get("text")) or ""))


@ToolRegistry.register(
    name="search_mediawiki",
    phase="ALL",
    plugin="mediawiki.api",
    capabilities=("url_discovery", "mediawiki_search", "wikimedia_search"),
    retrieval_role="discovery",
    signature="""[Tool] search_mediawiki
- 功能: 使用 MediaWiki/Wikimedia API 检索 Wikipedia、Wikidata、Commons 等站点的页面。
- 参数: query (页面主题), project (wikipedia|wikidata|commons 等), language (语言代码，默认 zh), max_results (最多 10)。
- 规则: 返回候选页面 URL；必须再调用 fetch_mediawiki_page 获取 API 正文证据。""",
)
def search_mediawiki(
    query: str,
    project: str = "wikipedia",
    language: str = "zh",
    max_results: int = 8,
    working_memory: dict | None = None,
    agent_state=None,
    **kwargs: Any,
) -> str:
    del working_memory, agent_state, kwargs
    query = " ".join(str(query or "").split()).strip()
    if not query:
        return _error(query, "query is empty", error_class="invalid_query")
    limit = max(1, min(int(max_results or 8), 10))
    try:
        base, endpoint = _site(project, language)
        payload = fetch_json(
            endpoint,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srprop": "snippet|titlesnippet|timestamp",
                "format": "json",
                "formatversion": 2,
                "utf8": 1,
                "origin": "*",
            },
            timeout=20,
            headers=_HEADERS,
        )
    except (NetworkFetchError, ValueError, TypeError) as exc:
        return _error(query, f"{type(exc).__name__}: {exc}")
    rows = []
    for item in ((payload.get("query") or {}).get("search") or [])[:limit]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        url = _page_url(base, title)
        rows.append(
            {
                "title": title,
                "url": url,
                "api_url": endpoint,
                "snippet": _strip_html(item.get("snippet"))[:900],
                "source": "MediaWiki API",
                "pageid": item.get("pageid"),
                "timestamp": item.get("timestamp", ""),
                "project": project,
                "language": language,
                "untrusted_content": True,
            }
        )
    result = {
        "status": "ok" if rows else "no_results",
        "real_network": True,
        "provider": "mediawiki.api",
        "query": query,
        "project": project,
        "language": language,
        "count": len(rows),
        "results": rows,
        "sources": [row["url"] for row in rows],
        "citation_refs": [],
        "provider_errors": [],
        "evidence_policy": "MediaWiki 搜索结果是候选页面，必须经过 fetch_mediawiki_page 和 chunk 证据流程。",
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _title_from_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    marker = "/wiki/"
    if marker in parsed.path:
        return unquote(parsed.path.split(marker, 1)[1]).replace("_", " ").strip()
    return ""


@ToolRegistry.register(
    name="fetch_mediawiki_page",
    phase="ALL",
    plugin="mediawiki.api",
    capabilities=("page_fetch", "page_evidence", "mediawiki_api", "wikimedia_api"),
    retrieval_role="evidence",
    signature="""[Tool] fetch_mediawiki_page
- 功能: 读取模型选择的 MediaWiki/Wikimedia 页面 API 正文，返回可分块的纯文本证据。
- 参数: url (搜索结果中的页面 URL), project (可选项目), language (可选语言), title (可选页面标题), max_chars (正文上限，默认 20000)。""",
)
def fetch_mediawiki_page(
    url: str,
    project: str = "wikipedia",
    language: str = "zh",
    title: str = "",
    max_chars: int = 20000,
    working_memory: dict | None = None,
    agent_state=None,
    **kwargs: Any,
) -> str:
    del working_memory, agent_state, kwargs
    if not str(url or "").strip() and not str(title or "").strip():
        return _error(str(url or ""), "url or title is required", error_class="invalid_url")
    try:
        base, endpoint = _site_from_url(url, project, language)
        page_title = str(title or _title_from_url(url)).strip()
        if not page_title:
            raise ValueError("could not determine a MediaWiki page title")
        limit = max(1000, min(int(max_chars or 20000), 30000))
        payload = fetch_json(
            endpoint,
            {
                "action": "query",
                "titles": page_title,
                "redirects": 1,
                "prop": "extracts|info",
                "explaintext": 1,
                "exchars": limit,
                "inprop": "url",
                "format": "json",
                "formatversion": 2,
                "utf8": 1,
                "origin": "*",
            },
            timeout=20,
            headers=_HEADERS,
        )
        pages = ((payload.get("query") or {}).get("pages") or [])
        page = pages[0] if pages and isinstance(pages[0], dict) else {}
        extract = " ".join(str(page.get("extract") or "").split())[:limit]
        raw_wikitext = ""
        try:
            raw_wikitext = _fetch_wikitext(endpoint, page_title)
        except (NetworkFetchError, ValueError, TypeError):
            # The short extracts response remains a valid fallback when a
            # project disallows revision content through the current network.
            raw_wikitext = ""
        readable_wikitext = _clean_wikitext(raw_wikitext)

        # Station/entity lists are commonly transcluded from a template and
        # therefore absent from the page extract.  Resolve only templates
        # whose names clearly indicate list/station data; this keeps the
        # evidence request bounded for ordinary encyclopedic pages.
        template_evidence: list[str] = []
        template_names = re.findall(r"\{\{\s*([^{}\n|]+)", raw_wikitext)
        for template_name in dict.fromkeys(name.strip() for name in template_names):
            lowered = template_name.casefold()
            if not ("list" in lowered or "station" in lowered or "车站" in template_name or "列表" in template_name):
                continue
            if template_name.casefold().startswith(("cite", "wayback", "note", "infobox", "reflist")):
                continue
            try:
                template_title = f"Template:{template_name}"
                template_text = _fetch_parsed_text(endpoint, template_title)
                if not template_text:
                    template_text = _clean_wikitext(_fetch_wikitext(endpoint, template_title))
            except (NetworkFetchError, ValueError, TypeError):
                continue
            if template_text:
                template_evidence.append(f"Template: {template_name}\n{template_text}")
            if len(template_evidence) >= 3:
                break

        human_url = str(page.get("fullurl") or _page_url(base, str(page.get("title") or page_title)))
        final_title = str(page.get("title") or page_title)
        evidence_parts = [
            f"MediaWiki API endpoint: {endpoint}",
            f"MediaWiki API response status: ok",
            f"Page URL: {human_url}",
            f"Page title: {final_title}",
        ]
        if template_evidence:
            evidence_parts.append("Template evidence:\n" + "\n\n".join(template_evidence))
        if extract:
            evidence_parts.append(f"Page extract:\n{extract}")
        if readable_wikitext:
            evidence_parts.append(f"Page wikitext fields:\n{readable_wikitext}")
        evidence_text = "\n\n".join(evidence_parts)[:limit]
        if not evidence_text:
            raise ValueError("MediaWiki page has no extractable text")
        request_params: dict[str, Any] = {
            "page_extract": {
                "action": "query",
                "titles": page_title,
                "prop": "extracts|info",
                "explaintext": 1,
                "exchars": limit,
                "format": "json",
                "formatversion": 2,
            },
            "page_wikitext": {
                "action": "query",
                "titles": page_title,
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "format": "json",
                "formatversion": 2,
            },
        }
        if template_evidence:
            request_params["template_render"] = {
                "action": "parse",
                "prop": "text",
                "format": "json",
                "formatversion": 2,
                "resolved_templates": [
                    item.split("\n", 1)[0].removeprefix("Template: ")
                    for item in template_evidence
                ],
            }
        result = {
            "status": "ok",
            "real_network": True,
            "provider": "mediawiki.api",
            "query": page_title,
            "api_url": endpoint,
            "results": [
                {
                    "title": final_title,
                    "url": human_url,
                    "api_url": endpoint,
                    "request_params": request_params,
                    "snippet": evidence_text[:800],
                    "page_excerpt": evidence_text,
                    "content": evidence_text,
                    "source": "MediaWiki API",
                    "pageid": page.get("pageid"),
                    "project": project,
                    "language": language,
                    "content_type": "mediawiki-wikitext",
                    "untrusted_content": True,
                }
            ],
            "sources": [human_url],
            "citation_refs": [
                {
                    "ref_id": f"MEDIAWIKI_{re.sub(r'[^A-Za-z0-9]+', '_', final_title)[:60]}",
                    "title": final_title,
                    "url": human_url,
                    "source": "MediaWiki API",
                }
            ],
            "provider_errors": [],
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (NetworkFetchError, ValueError, TypeError) as exc:
        return _error(str(url or title), f"{type(exc).__name__}: {exc}")


__all__ = ["search_mediawiki", "fetch_mediawiki_page"]
