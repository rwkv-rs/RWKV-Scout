"""Explicit page operations layered on the existing web fetcher.

Search remains the preferred discovery operation.  These tools let the model
open a selected URL and locate text on that page without changing the
existing chunk pipeline or making the page itself an instruction source.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from retrieval_plugins import normalize_result
from tools.registry import ToolRegistry
from tools.web_search_keyless import _is_search_result_url, fetch_web_url
from utils.freshness import annotate_result_freshness, build_freshness_policy


def _load_page(url: str, max_chars: int, task_id: str) -> dict[str, Any]:
    raw = fetch_web_url(url, max_chars=max_chars, task_id=task_id, agentic_tool_loop=True)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "error", "message": "page fetch returned invalid JSON", "results": []}
    return normalize_result(payload, provider="web.page", query=url, role="evidence", real_network=True)


@ToolRegistry.register(
    name="open_page",
    phase="ALL",
    plugin="web.page",
    capabilities=("page_fetch", "page_evidence", "url_open"),
    retrieval_role="evidence",
    model_visible=True,
    category="retrieval",
    signature="""[Tool] open_page
- Function: open one URL selected from search results and return its page body as evidence.
- Parameters: url (complete http/https URL), max_chars (optional, 1000-20000).
- Search-result pages are rejected. The returned page still goes through the normal chunk/evidence pipeline.""",
)
def open_page(url: str, max_chars: int = 20000, **kwargs: Any) -> str:
    value = str(url or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return json.dumps({"status": "error", "tool": "open_page", "error_class": "invalid_url", "results": []}, ensure_ascii=False)
    if _is_search_result_url(value):
        return json.dumps({"status": "error", "tool": "open_page", "error_class": "search_result_page", "message": "search-result pages cannot be used as evidence", "results": []}, ensure_ascii=False)
    result = _load_page(value, max(1000, min(int(max_chars or 20000), 20000)), str(kwargs.get("task_id") or ""))
    policy = build_freshness_policy(kwargs.get("original_goal") or value, kwargs.get("task_plan"))
    result = annotate_result_freshness(result, policy)
    result["tool"] = "open_page"
    return json.dumps(result, ensure_ascii=False, indent=2)


@ToolRegistry.register(
    name="find_in_page",
    phase="ALL",
    plugin="web.page",
    capabilities=("page_fetch", "page_find", "page_locator"),
    retrieval_role="evidence",
    model_visible=True,
    category="retrieval",
    signature="""[Tool] find_in_page
- Function: open one selected URL and return matching line excerpts plus the full page evidence record.
- Parameters: url (complete http/https URL), pattern (short text or regular expression), max_chars (optional).
- It only locates text; it does not decide what the matched text means.""",
)
def find_in_page(url: str, pattern: str, max_chars: int = 20000, **kwargs: Any) -> str:
    value = str(url or "").strip()
    needle = str(pattern or "").strip()
    if not needle or len(needle) > 300:
        return json.dumps({"status": "error", "tool": "find_in_page", "error_class": "invalid_pattern", "results": []}, ensure_ascii=False)
    opened = json.loads(open_page(value, max_chars=max_chars, **kwargs))
    if str(opened.get("status") or "").casefold() != "ok":
        opened["tool"] = "find_in_page"
        return json.dumps(opened, ensure_ascii=False, indent=2)
    page = (opened.get("results") or [{}])[0]
    body = str(page.get("page_excerpt") or page.get("content") or "")
    try:
        matcher = re.compile(needle, re.IGNORECASE)
    except re.error:
        matcher = re.compile(re.escape(needle), re.IGNORECASE)
    lines = body.splitlines()
    matches: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if matcher.search(line):
            matches.append({"line": index, "text": line.strip()[:800]})
            if len(matches) >= 20:
                break
    opened["tool"] = "find_in_page"
    opened["pattern"] = needle
    opened["matches"] = matches
    opened["match_count"] = len(matches)
    return json.dumps(opened, ensure_ascii=False, indent=2)


__all__ = ["find_in_page", "open_page"]
