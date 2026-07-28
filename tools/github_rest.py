"""GitHub REST discovery and evidence tools.

The generic web fetcher is intentionally not used for GitHub candidates.  A
repository or file selected by the model is converted to the GitHub REST API,
then the returned JSON or decoded file body is passed through the same
single-document chunk/parallel-candidate evidence path as an ordinary page.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Any
from urllib.parse import quote, unquote, urlparse

from tools.registry import ToolRegistry
from utils.network_fetch import NetworkFetchError, fetch_json


_API_ROOT = "https://api.github.com"
_API_HEADERS_BASE = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "RWKV-ECRA/0.1 (GitHub REST retrieval)",
}


@lru_cache(maxsize=1)
def _github_token() -> str:
    """Use an explicit token first, then the local gh credential store."""

    explicit = os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()
    if explicit:
        return explicit
    executable = shutil.which("gh") or "/usr/bin/gh"
    if not os.path.exists(executable):
        return ""
    try:
        result = subprocess.run(
            [executable, "auth", "token"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _github_headers() -> dict[str, str]:
    headers = dict(_API_HEADERS_BASE)
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _error(query: str, message: str, *, error_class: str = "provider_error") -> str:
    return json.dumps(
        {
            "status": "error",
            "real_network": True,
            "provider": "github.rest",
            "query": query,
            "error_class": error_class,
            "results": [],
            "provider_errors": [message[:500]],
        },
        ensure_ascii=False,
    )


def _repo_url(full_name: str) -> str:
    return f"https://github.com/{full_name.strip('/') }" if full_name else ""


def _repository_row(item: dict[str, Any]) -> dict[str, Any]:
    full_name = str(item.get("full_name") or "").strip()
    html_url = str(item.get("html_url") or _repo_url(full_name)).strip()
    api_url = str(item.get("url") or f"{_API_ROOT}/repos/{full_name}").strip()
    description = " ".join(str(item.get("description") or "").split())
    return {
        "title": full_name or str(item.get("name") or html_url),
        "url": html_url,
        "api_url": api_url,
        "snippet": description[:900],
        "source": "GitHub REST API",
        "full_name": full_name,
        "default_branch": str(item.get("default_branch") or ""),
        "language": str(item.get("language") or ""),
        "stars": item.get("stargazers_count"),
        "untrusted_content": True,
    }


def _code_row(item: dict[str, Any]) -> dict[str, Any]:
    repository = item.get("repository") if isinstance(item.get("repository"), dict) else {}
    full_name = str(repository.get("full_name") or "").strip()
    html_url = str(item.get("html_url") or "").strip()
    api_url = str(item.get("url") or "").strip()
    path = str(item.get("path") or item.get("name") or "").strip()
    return {
        "title": f"{full_name}/{path}".strip("/") or html_url,
        "url": html_url or _repo_url(full_name),
        "api_url": api_url,
        "snippet": f"GitHub code result: {path}"[:900],
        "source": "GitHub REST API",
        "full_name": full_name,
        "path": path,
        "untrusted_content": True,
    }


@ToolRegistry.register(
    name="search_github_rest",
    phase="ALL",
    plugin="github.rest",
    capabilities=("url_discovery", "github_rest_search", "repository_search"),
    retrieval_role="discovery",
    signature="""[Tool] search_github_rest
- 功能: 使用 GitHub REST API 检索公开仓库或代码文件，不读取 GitHub HTML 搜索页。
- 参数: query (仓库名、组织、代码或项目关键词), scope (repositories 或 code), max_results (最多 10)。
- 规则: 返回候选 URL；必须再调用 fetch_github_rest 获取仓库/文件 API 正文证据。""",
)
def search_github_rest(
    query: str,
    scope: str = "repositories",
    max_results: int = 8,
    working_memory: dict | None = None,
    agent_state=None,
    **kwargs: Any,
) -> str:
    del working_memory, agent_state, kwargs
    query = " ".join(str(query or "").split()).strip()
    scope = str(scope or "repositories").strip().casefold()
    if not query:
        return _error(query, "query is empty", error_class="invalid_query")
    if scope not in {"repositories", "code"}:
        return _error(query, "scope must be repositories or code", error_class="invalid_scope")
    limit = max(1, min(int(max_results or 8), 10))
    endpoint = f"{_API_ROOT}/search/{scope}"
    params = {"q": query, "per_page": limit}
    if scope == "repositories":
        params["sort"] = "stars"
        params["order"] = "desc"
    try:
        payload = fetch_json(endpoint, params, timeout=20, headers=_github_headers())
    except (NetworkFetchError, ValueError, TypeError) as exc:
        return _error(query, f"{type(exc).__name__}: {exc}")
    items = [item for item in (payload.get("items") or []) if isinstance(item, dict)]
    rows = [(_repository_row(item) if scope == "repositories" else _code_row(item)) for item in items[:limit]]
    result = {
        "status": "ok" if rows else "no_results",
        "real_network": True,
        "provider": "github.rest",
        "query": query,
        "scope": scope,
        "count": len(rows),
        "total_count": payload.get("total_count"),
        "results": rows,
        "sources": [row.get("url") for row in rows if row.get("url")],
        "citation_refs": [],
        "provider_errors": [],
        "evidence_policy": "GitHub REST 搜索结果是候选元数据，必须经过 fetch_github_rest 和 chunk 证据流程。",
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _repo_parts(value: str) -> tuple[str, str, list[str]]:
    parsed = urlparse(str(value or "").strip())
    path = [unquote(part) for part in parsed.path.split("/") if part]
    if parsed.netloc.casefold() in {"api.github.com", "www.api.github.com"}:
        if len(path) >= 3 and path[0] == "repos":
            return path[1], path[2], path[3:]
        return "", "", []
    if parsed.netloc.casefold().removeprefix("www.") == "github.com" and len(path) >= 2:
        return path[0], path[1].removesuffix(".git"), path[2:]
    return "", "", []


def _github_api_url(url: str, path: str = "", ref: str = "") -> tuple[str, str]:
    candidate = str(url or "").strip()
    parsed = urlparse(candidate)
    owner, repo, trailing = _repo_parts(candidate)
    if not owner or not repo:
        return "", ""
    if parsed.netloc.casefold() == "api.github.com":
        api_url = candidate.split("?", 1)[0].rstrip("/")
        if not api_url.startswith(f"{_API_ROOT}/repos/{owner}/{repo}"):
            return "", ""
        title_path = "/".join(trailing)
        return api_url, title_path

    selected_path = "/".join(trailing)
    selected_ref = str(ref or "").strip()
    if trailing and trailing[0] in {"blob", "tree"} and len(trailing) >= 2:
        selected_ref = selected_ref or trailing[1]
        selected_path = "/".join(trailing[2:])
    selected_path = str(path or selected_path).strip("/")
    if selected_path:
        api_url = f"{_API_ROOT}/repos/{owner}/{repo}/contents/{quote(selected_path, safe='/')}"
        if selected_ref:
            api_url += f"?ref={quote(selected_ref, safe='')}"
    else:
        api_url = f"{_API_ROOT}/repos/{owner}/{repo}"
    return api_url, selected_path


def _repository_text(payload: dict[str, Any], max_chars: int) -> str:
    owner = payload.get("owner") if isinstance(payload.get("owner"), dict) else {}
    lines = [
        f"Repository: {payload.get('full_name', '')}",
        f"Description: {payload.get('description', '')}",
        f"Owner: {owner.get('login', '')}",
        f"URL: {payload.get('html_url', '')}",
        f"Default branch: {payload.get('default_branch', '')}",
        f"License: {(payload.get('license') or {}).get('name', '') if isinstance(payload.get('license'), dict) else ''}",
        f"Language: {payload.get('language', '')}",
        f"Topics: {', '.join(payload.get('topics') or [])}",
        f"Stars: {payload.get('stargazers_count', '')}",
        f"Created: {payload.get('created_at', '')}",
        f"Updated: {payload.get('updated_at', '')}",
    ]
    return "\n".join(line for line in lines if line.split(": ", 1)[-1].strip())[:max_chars]


def _content_text(payload: Any, max_chars: int) -> tuple[str, str, str]:
    if isinstance(payload, list):
        lines = []
        for item in payload:
            if isinstance(item, dict):
                lines.append(f"{item.get('type', '')}: {item.get('path', '')} — {item.get('html_url', '')}")
        return "\n".join(lines)[:max_chars], "directory", ""
    if not isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False)[:max_chars], "json", ""
    encoded = str(payload.get("content") or "").replace("\n", "")
    if encoded and str(payload.get("encoding") or "").casefold() == "base64":
        try:
            content = base64.b64decode(encoded).decode("utf-8", errors="replace")
        except (ValueError, UnicodeError):
            content = ""
        return content[:max_chars], "file", str(payload.get("html_url") or "")
    if "tree" in payload:
        lines = [
            f"{item.get('type', '')}: {item.get('path', '')} — {item.get('url', '')}"
            for item in payload.get("tree") or []
            if isinstance(item, dict)
        ]
        return "\n".join(lines)[:max_chars], "tree", str(payload.get("url") or "")
    return _repository_text(payload, max_chars), "repository", str(payload.get("html_url") or "")


@ToolRegistry.register(
    name="fetch_github_rest",
    phase="ALL",
    plugin="github.rest",
    capabilities=("page_fetch", "page_evidence", "github_rest"),
    retrieval_role="evidence",
    signature="""[Tool] fetch_github_rest
- 功能: 读取模型选择的 GitHub 仓库、代码文件或目录的 REST API 正文，支持仓库 URL、blob/tree URL 或 api.github.com URL。
- 参数: url (搜索结果中的 GitHub URL), path (可选文件/目录路径), ref (可选分支或提交), max_chars (正文上限，默认 20000)。""",
)
def fetch_github_rest(
    url: str,
    path: str = "",
    ref: str = "",
    max_chars: int = 20000,
    working_memory: dict | None = None,
    agent_state=None,
    **kwargs: Any,
) -> str:
    del working_memory, agent_state, kwargs
    api_url, selected_path = _github_api_url(url, path, ref)
    if not api_url:
        return _error(str(url or ""), "url must identify a GitHub repository, blob, tree, or REST API URL", error_class="invalid_url")
    limit = max(1000, min(int(max_chars or 20000), 30000))
    try:
        payload = fetch_json(api_url, timeout=20, headers=_github_headers())
        page_text, content_type, returned_url = _content_text(payload, limit)
        if not page_text:
            raise ValueError("GitHub REST response contains no readable content")
        owner, repo, _ = _repo_parts(url)
        human_url = returned_url or str(url).split("?", 1)[0]
        if content_type == "file" and selected_path and "/blob/" not in human_url:
            default_ref = str(ref or "main")
            human_url = f"https://github.com/{owner}/{repo}/blob/{default_ref}/{selected_path}"
        title = f"{owner}/{repo}" + (f"/{selected_path}" if selected_path else "")
        result = {
            "status": "ok",
            "real_network": True,
            "provider": "github.rest",
            "query": str(url),
            "api_url": api_url,
            "results": [
                {
                    "title": title,
                    "url": human_url,
                    "api_url": api_url,
                    "snippet": page_text[:800],
                    "page_excerpt": page_text,
                    "content": page_text,
                    "source": "GitHub REST API",
                    "content_type": content_type,
                    "untrusted_content": True,
                }
            ],
            "sources": [human_url],
            "citation_refs": [
                {
                    "ref_id": f"GITHUB_REST_{re.sub(r'[^A-Za-z0-9]+', '_', title)[:60]}",
                    "title": title,
                    "url": human_url,
                    "source": "GitHub REST API",
                }
            ],
            "provider_errors": [],
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (NetworkFetchError, ValueError, TypeError) as exc:
        return _error(str(url), f"{type(exc).__name__}: {exc}")


__all__ = ["search_github_rest", "fetch_github_rest"]
