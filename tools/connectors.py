"""Stable model-facing wrappers for the repository's domain connectors.

The existing provider adapters remain internal implementations.  This small
surface gives RWKV one explicit connector contract while preserving the
provider-specific code and its source metadata.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tools.github_rest import fetch_github_rest, search_github_rest
from tools.paper_search import search_papers
from tools.registry import ToolRegistry
from tools.weather import get_current_weather
from tools.weather_alerts import get_current_weather_alerts
from utils.freshness import annotate_result_freshness, build_freshness_policy
from utils.source_authority import annotate_source


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
    provider_status = str(result.get("status") or "").casefold()
    source_rows = result.get("results") or [] if provider_status == "ok" else []
    for raw in source_rows:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        if connector == "papers":
            paper_lines = []
            for label, value in (
                ("Title", row.get("title")),
                ("Provider", row.get("source")),
                ("arXiv version", row.get("version")),
                ("Published", row.get("published")),
                ("Updated", row.get("updated")),
                ("DOI", row.get("doi")),
                ("URL", row.get("url")),
                (
                    "Abstract",
                    row.get("page_excerpt")
                    or row.get("content")
                    or row.get("abstract"),
                ),
            ):
                rendered = str(value or "").strip()
                if rendered:
                    paper_lines.append(f"{label}: {rendered}")
            text = "\n".join(paper_lines)
        else:
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
    result["evidence_ready"] = bool(rows) and provider_status == "ok"
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
    capabilities=("weather", "weather_alerts", "github", "papers", "structured_api"),
    retrieval_role="discovery",
    model_visible=True,
    category="connector",
    description="Structured lookup for weather/alerts, a specific GitHub repository/code/release, or scholarly arXiv/DOI paper records; product status pages and ordinary websites belong to web_search.",
    argument_schema={
        "type": "object",
        "properties": {
            "connector": {
                "type": "string",
                "enum": ["weather", "weather_alerts", "github", "papers"],
            },
            "query": {"type": "string"},
            "scope": {
                "type": "string",
                "enum": ["current", "repositories", "code", "latest_release", "paper"],
            },
            "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
        },
        "required": ["connector", "query"],
        "additionalProperties": False,
    },
    signature="""[Tool] connector_lookup
- Function: use one curated structured connector rather than general web search.
- Parameters: connector (weather|weather_alerts|github|papers), query, scope (optional; current for weather/alerts; GitHub supports repositories|code|latest_release), max_results (optional).
- The result is structured evidence with provider/source metadata; it is not a final answer.
- Use weather for current conditions, github for repositories/files/releases, and papers for scholarly records.""",
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
    if name not in {"weather", "weather_alerts", "github", "papers"}:
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
        if name == "weather_alerts":
            payload = _structured_rows(
                get_current_weather_alerts(text, max_results=max_results, **context),
                name,
            )
        elif name == "weather":
            payload = _payload(get_current_weather(text, **context))
            weather_status = str(payload.get("status") or "error").casefold()
            payload = _structured_rows(
                {
                    **payload,
                    "status": weather_status,
                    "provider": "open-meteo",
                    "query": text,
                    "sources": payload.get("sources") or [],
                    "results": (
                        [
                            {
                                "title": f"Current weather: {payload.get('location') or text}",
                                "url": (payload.get("sources") or [""])[-1] if payload.get("sources") else "",
                                "structured_evidence_text": json.dumps(payload, ensure_ascii=False),
                                "source": "Open-Meteo",
                            }
                        ]
                        if weather_status == "ok" and isinstance(payload.get("current"), dict)
                        else []
                    ),
                },
                name,
            )
        elif name == "github":
            github_scope = str(scope or "repositories").strip().casefold().replace("-", "_")
            release_scope = github_scope in {"release", "releases", "latest_release"}
            direct_repository = text.startswith(("http://", "https://")) or "/" in text and " " not in text
            if release_scope:
                if direct_repository:
                    candidate = text
                else:
                    discovered = _payload(
                        search_github_rest(
                            text,
                            scope="repositories",
                            max_results=max_results,
                            **context,
                        )
                    )
                    first = next(
                        (
                            item
                            for item in discovered.get("results") or []
                            if isinstance(item, dict)
                            and (item.get("full_name") or item.get("url"))
                        ),
                        None,
                    )
                    candidate = str(
                        (first or {}).get("full_name")
                        or (first or {}).get("url")
                        or ""
                    )
                if candidate and not candidate.startswith(("http://", "https://")):
                    candidate = f"https://github.com/{candidate.strip('/')}"
                repository_match = re.search(
                    r"(?:api\.github\.com/repos/|github\.com/)?([^/\s]+/[^/\s]+)",
                    candidate.removesuffix("/").split("/releases", 1)[0],
                    flags=re.IGNORECASE,
                )
                owner_repo = repository_match.group(1) if repository_match else ""
                release_url = f"https://api.github.com/repos/{owner_repo}/releases/latest" if "/" in owner_repo else ""
                payload = (
                    _payload(fetch_github_rest(release_url, max_chars=20000, **context))
                    if release_url
                    else {"status": "no_results", "results": []}
                )
            elif direct_repository:
                payload = _payload(fetch_github_rest(text, path=scope, max_chars=20000, **context))
            else:
                discovered = _payload(search_github_rest(text, scope=github_scope, max_results=max_results, **context))
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
    payload["results"] = [
        annotate_source(
            row,
            str(kwargs.get("original_goal") or text),
            {"task_plan": context["task_plan"]},
        )
        for row in payload.get("results") or []
        if isinstance(row, dict)
    ]
    payload.update({"tool": "connector_lookup", "connector": name, "real_network": True})
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = ["connector_lookup"]
