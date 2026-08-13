"""Stable model-facing wrappers for the repository's domain connectors.

The existing provider adapters remain internal implementations.  This small
surface gives RWKV one explicit connector contract while preserving the
provider-specific code and its source metadata.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, unquote, urlparse

from tools.github_rest import fetch_github_rest, search_github_rest
from tools.paper_search import search_papers
from tools.registry import ToolRegistry
from tools.weather import get_current_weather
from tools.weather_alerts import get_current_weather_alerts
from agent.retrieval_object_contract import (
    github_repository_target,
    source_object_contract,
)
from utils.freshness import annotate_result_freshness, build_freshness_policy
from utils.source_authority import annotate_source
from utils.network_fetch import fetch_json


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
        if connector == "paper":
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
        row["source_object"] = source_object_contract(row, connector=connector)
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


def _package_identifier(value: str, registry: str) -> str:
    """Read one explicit package identifier from the model-authored request."""

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    host = parsed.netloc.casefold().removeprefix("www.")
    if registry == "crates" and host == "crates.io" and len(parts) >= 2 and parts[0] == "crates":
        return parts[1]
    if registry == "pypi" and host == "pypi.org" and len(parts) >= 2 and parts[0] in {"project", "pypi"}:
        return parts[1]
    if registry == "npm" and host == "npmjs.com" and len(parts) >= 2 and parts[0] == "package":
        return "/".join(parts[1:3] if parts[1].startswith("@") else parts[1:2])
    patterns = {
        "crates": r"\bcrates\.io(?:\s+上)?\s+([A-Za-z0-9_.-]+)",
        "pypi": r"\bPyPI(?:\s+上)?\s+([A-Za-z0-9_.-]+)",
        "npm": r"\bnpm(?:\s+上)?\s+(@?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)",
    }
    match = re.search(patterns[registry], text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return text if re.fullmatch(r"@?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?", text) else ""


def _package_release_payload(registry: str, query: str) -> dict[str, Any]:
    """Fetch one registry-declared current release as typed source evidence."""

    package = _package_identifier(query, registry)
    if not package:
        return {
            "status": "error",
            "error_class": "invalid_package_identifier",
            "query": query,
            "results": [],
            "provider_errors": [
                f"{registry} release lookup requires one exact package identifier or registry URL"
            ],
        }
    encoded = quote(package, safe="@")
    if registry == "crates":
        payload = fetch_json(
            f"https://crates.io/api/v1/crates/{encoded}",
            timeout=20,
            headers={"User-Agent": "RWKV-ECRA/0.1 structured registry lookup"},
        )
        crate = payload.get("crate") if isinstance(payload.get("crate"), dict) else {}
        version = str(crate.get("max_stable_version") or "").strip()
        record = next(
            (
                row
                for row in payload.get("versions") or []
                if isinstance(row, dict) and str(row.get("num") or "") == version
            ),
            {},
        )
        public_url = f"https://crates.io/crates/{package}"
        fields = {
            "Package": package,
            "Version": version,
            "Published": record.get("created_at") or "",
            "Yanked": record.get("yanked"),
            "Repository": crate.get("repository") or "",
            "Registry URL": public_url,
        }
        provider = "crates.io API"
    elif registry == "pypi":
        payload = fetch_json(f"https://pypi.org/pypi/{encoded}/json", timeout=20)
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        version = str(info.get("version") or "").strip()
        releases = payload.get("releases") if isinstance(payload.get("releases"), dict) else {}
        files = [row for row in releases.get(version, []) if isinstance(row, dict)]
        published = next(
            (
                str(row.get("upload_time_iso_8601") or row.get("upload_time") or "")
                for row in files
                if row.get("upload_time_iso_8601") or row.get("upload_time")
            ),
            "",
        )
        public_url = f"https://pypi.org/project/{package}/"
        fields = {
            "Package": package,
            "Version": version,
            "Published": published,
            "Project URL": info.get("project_url") or public_url,
            "Registry URL": public_url,
        }
        provider = "PyPI JSON API"
    else:
        payload = fetch_json(f"https://registry.npmjs.org/{encoded}", timeout=20)
        tags = payload.get("dist-tags") if isinstance(payload.get("dist-tags"), dict) else {}
        version = str(tags.get("latest") or "").strip()
        versions = payload.get("versions") if isinstance(payload.get("versions"), dict) else {}
        record = versions.get(version) if isinstance(versions.get(version), dict) else {}
        times = payload.get("time") if isinstance(payload.get("time"), dict) else {}
        public_url = f"https://www.npmjs.com/package/{package}"
        fields = {
            "Package": package,
            "Version": version,
            "Published": times.get(version) or "",
            "Repository": record.get("repository") or "",
            "Registry URL": public_url,
        }
        provider = "npm registry API"
    evidence = "\n".join(
        f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}"
        for key, value in fields.items()
        if value not in (None, "", [], {})
    )
    return {
        "status": "ok" if version else "no_results",
        "provider": provider,
        "query": query,
        "results": [
            {
                "title": f"{package} {version}".strip(),
                "url": public_url,
                "version": version,
                "published": fields.get("Published") or "",
                "structured_evidence_text": evidence,
                "source": provider,
            }
        ] if version else [],
        "provider_errors": [],
    }


@ToolRegistry.register(
    name="connector_lookup",
    phase="ALL",
    plugin="connectors.domain",
    capabilities=("weather", "weather_alerts", "github", "paper", "package_registry", "structured_api"),
    retrieval_role="discovery",
    model_visible=True,
    category="connector",
    description="Structured lookup with one unambiguous operation for weather, GitHub, package-registry, or scholarly records; product status pages and ordinary websites belong to web_search.",
    argument_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "weather_current",
                    "weather_alerts",
                    "github_repository",
                    "github_code",
                    "github_release",
                    "paper",
                    "paper_series",
                    "crates_release",
                    "pypi_release",
                    "npm_release",
                ],
            },
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
        },
        "required": ["operation", "query"],
        "additionalProperties": False,
    },
    signature="""[Tool] connector_lookup
- Function: use one curated structured connector rather than general web search.
- Parameters: operation (weather_current|weather_alerts|github_repository|github_code|github_release|paper|paper_series|crates_release|pypi_release|npm_release), query, max_results (optional).
- The result is structured evidence with provider/source metadata; it is not a final answer.
- Use weather for current conditions, github for repositories/files/releases, package operations for one exact registry package identifier, and paper for scholarly records.""",
)
def connector_lookup(
    operation: str,
    query: str,
    max_results: int = 8,
    **kwargs: Any,
) -> str:
    selected_operation = str(operation or "").strip().casefold().replace("-", "_")
    legacy_scope = str(kwargs.get("scope") or "").strip().casefold().replace("-", "_")
    if selected_operation == "github" and legacy_scope:
        selected_operation = {
            "repositories": "github_repository",
            "repository": "github_repository",
            "code": "github_code",
            "release": "github_release",
            "releases": "github_release",
            "latest_release": "github_release",
        }.get(legacy_scope, selected_operation)
    elif selected_operation in {"paper", "papers"} and legacy_scope == "series":
        selected_operation = "paper_series"
    aliases = {
        "weather": "weather_current",
        "天气": "weather_current",
        "github": "github_repository",
        "代码": "github_code",
        "papers": "paper",
        "论文": "paper",
    }
    selected_operation = aliases.get(selected_operation, selected_operation)
    operation_contract = {
        "weather_current": ("weather", "current"),
        "weather_alerts": ("weather_alerts", "alerts"),
        "github_repository": ("github", "repository"),
        "github_code": ("github", "code"),
        "github_release": ("github", "release"),
        "paper": ("paper", "paper"),
        "paper_series": ("paper", "series"),
        "crates_release": ("crates", "release"),
        "pypi_release": ("pypi", "release"),
        "npm_release": ("npm", "release"),
    }
    name, scope = operation_contract.get(selected_operation, ("", ""))
    text = " ".join(str(query or "").split()).strip()
    if not name:
        return json.dumps({"status": "error", "tool": "connector_lookup", "error_class": "unsupported_operation", "operation": selected_operation, "results": []}, ensure_ascii=False)
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
            github_scope = str(scope or "repository").strip().casefold().replace("-", "_")
            github_scope = {
                "repositories": "repository",
                "latest_release": "release",
                "releases": "release",
            }.get(github_scope, github_scope)
            release_scope = github_scope == "release"
            explicit_repository = github_repository_target(text)
            direct_repository = bool(explicit_repository)
            if release_scope:
                if direct_repository:
                    candidate = explicit_repository
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
                owner_repo = github_repository_target(candidate)
                release_url = f"https://api.github.com/repos/{owner_repo}/releases/latest" if "/" in owner_repo else ""
                payload = (
                    _payload(fetch_github_rest(release_url, max_chars=20000, **context))
                    if release_url
                    else {"status": "no_results", "results": []}
                )
            elif direct_repository:
                payload = _payload(
                    fetch_github_rest(
                        f"https://github.com/{explicit_repository}",
                        max_chars=20000,
                        **context,
                    )
                )
            else:
                search_scope = "code" if github_scope == "code" else "repositories"
                discovered = _payload(search_github_rest(text, scope=search_scope, max_results=max_results, **context))
                first = next((item for item in discovered.get("results") or [] if isinstance(item, dict) and item.get("url")), None)
                payload = _payload(fetch_github_rest(str(first.get("url")), max_chars=20000, **context)) if first else discovered
            payload = _structured_rows(payload, name)
        elif name in {"crates", "pypi", "npm"}:
            payload = _structured_rows(
                _package_release_payload(name, text),
                name,
            )
        else:
            paper_scope = str(scope or "paper").strip().casefold()
            paper_scope = "series" if paper_scope == "series" else "paper"
            payload = _payload(search_papers(text, scope=paper_scope, max_results=max_results, **context))
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
    payload.update({"tool": "connector_lookup", "connector": name, "operation": selected_operation, "real_network": True})
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = ["connector_lookup"]
