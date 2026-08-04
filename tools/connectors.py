"""Stable model-facing wrappers for the repository's domain connectors.

The existing provider adapters remain internal implementations.  This small
surface gives RWKV one explicit connector contract while preserving the
provider-specific code and its source metadata.
"""

from __future__ import annotations

import json
from typing import Any

from tools.github_rest import fetch_github_rest, search_github_rest
from tools.paper_search import search_papers
from tools.registry import ToolRegistry
from tools.weather import get_current_weather
from utils.freshness import annotate_result_freshness, build_freshness_policy


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError:
        parsed = None
    return dict(parsed) if isinstance(parsed, dict) else {"status": "error", "results": [], "message": "connector returned invalid JSON"}


def _structured_rows(payload: dict[str, Any], connector: str) -> dict[str, Any]:
    result = dict(payload)
    rows = []
    for raw in result.get("results") or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        text = str(
            row.get("structured_evidence_text")
            or row.get("page_excerpt")
            or row.get("content")
            or row.get("abstract")
            or row.get("snippet")
            or ""
        ).strip()
        if text:
            row["structured_evidence_text"] = text[:14000]
            row.setdefault("evidence_origin", "structured_api_record")
            row.setdefault("evidence_kind", "structured_record")
            row.setdefault("evidence_boundary", "structured_api_record_only")
            row.setdefault("body_verified", True)
        row["connector"] = connector
        rows.append(row)
    result["results"] = rows
    result["connector"] = connector
    result["retrieval_role"] = "evidence"
    result["evidence_ready"] = bool(rows)
    if not result.get("citation_refs"):
        result["citation_refs"] = [
            {
                "ref_id": f"{connector.upper()}_CONNECTOR_{index}",
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "source": row.get("source", connector),
                "evidence_text": row.get("structured_evidence_text", ""),
                "evidence_origin": row.get("evidence_origin", "structured_api_record"),
                "evidence_boundary": row.get("evidence_boundary", "structured_api_record_only"),
            }
            for index, row in enumerate(rows, start=1)
            if row.get("url") or row.get("structured_evidence_text")
        ]
    return result


@ToolRegistry.register(
    name="connector_lookup",
    phase="ALL",
    plugin="connectors.domain",
    capabilities=("weather", "github", "papers", "structured_api"),
    retrieval_role="discovery",
    model_visible=True,
    category="connector",
    signature="""[Tool] connector_lookup
- Function: use one curated structured connector rather than general web search.
- Parameters: connector (weather|github|papers), query, scope (optional), max_results (optional).
- The result is structured evidence with provider/source metadata; it is not a final answer.
- Use weather for current conditions, github for repositories/files, and papers for scholarly records.""",
)
def connector_lookup(
    connector: str,
    query: str,
    scope: str = "",
    max_results: int = 8,
    **kwargs: Any,
) -> str:
    name = str(connector or "").strip().casefold()
    aliases = {"weather": "weather", "天气": "weather", "github": "github", "代码": "github", "papers": "papers", "paper": "papers", "论文": "papers"}
    name = aliases.get(name, name)
    text = " ".join(str(query or "").split()).strip()
    if name not in {"weather", "github", "papers"}:
        return json.dumps({"status": "error", "tool": "connector_lookup", "error_class": "unsupported_connector", "connector": name, "results": []}, ensure_ascii=False)
    if not text:
        return json.dumps({"status": "error", "tool": "connector_lookup", "error_class": "empty_query", "connector": name, "results": []}, ensure_ascii=False)

    context = {
        "task_id": str(kwargs.get("task_id") or ""),
        "agentic_tool_loop": True,
        "original_goal": kwargs.get("original_goal", ""),
        "task_plan": kwargs.get("task_plan") if isinstance(kwargs.get("task_plan"), dict) else {},
    }
    try:
        if name == "weather":
            payload = _payload(get_current_weather(text, **context))
            payload = _structured_rows(
                {
                    "status": payload.get("status") or "ok",
                    "provider": "open-meteo",
                    "query": text,
                    "sources": payload.get("sources") or [],
                    "results": [
                        {
                            "title": f"Current weather: {payload.get('location') or text}",
                            "url": (payload.get("sources") or [""])[-1] if payload.get("sources") else "",
                            "structured_evidence_text": json.dumps(payload, ensure_ascii=False),
                            "source": "Open-Meteo",
                        }
                    ],
                },
                name,
            )
        elif name == "github":
            if text.startswith(("http://", "https://")) or "/" in text and " " not in text:
                payload = _payload(fetch_github_rest(text, path=scope, max_chars=20000, **context))
            else:
                discovered = _payload(search_github_rest(text, scope=scope or "repositories", max_results=max_results, **context))
                first = next((item for item in discovered.get("results") or [] if isinstance(item, dict) and item.get("url")), None)
                payload = _payload(fetch_github_rest(str(first.get("url")), max_chars=20000, **context)) if first else discovered
            payload = _structured_rows(payload, name)
        else:
            payload = _payload(search_papers(text, scope=scope or "paper", max_results=max_results, **context))
            payload = _structured_rows(payload, name)
    except Exception as exc:
        payload = {"status": "error", "provider": f"connector.{name}", "query": text, "results": [], "provider_errors": [f"{type(exc).__name__}: {exc}"], "error_class": "connector_execution"}

    policy = build_freshness_policy(kwargs.get("original_goal") or text, context["task_plan"])
    payload = annotate_result_freshness(payload, policy)
    payload.update({"tool": "connector_lookup", "connector": name, "real_network": True})
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = ["connector_lookup"]
