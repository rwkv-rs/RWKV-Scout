"""Versioned experiment metadata and replay helpers.

The task event stream is the source of truth for a run.  This module adds a
small, stable manifest beside that stream so a run can be compared, replayed,
or diagnosed without relying on process memory or the current configuration.
Secrets are deliberately redacted from the persisted model snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    API_KEYS,
    AGENT_CONFIG,
    DATA_PIPELINE,
    DEFAULT_LLM_PROVIDER,
    LLM_ENDPOINTS,
    SEARCH_CONFIG,
    LLM_CONFIG,
    RUNTIME_CONFIG,
    SLM_CONFIG,
    TRACKING,
    WIGOLO_CONFIG,
    get_direct_rwkv_config,
    get_llm_base_url,
    get_llm_api_key,
    get_llm_model,
    get_llm_provider,
    get_model_backend_name,
)
from utils.file_lock import atomic_file_lease


_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()
_TASK_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_git_revision_cache: str | None = None
_workspace_snapshot_cache: dict[str, Any] | None = None


def _git_args(repo_root: Path, *args: str) -> list[str]:
    safe_directory = repo_root.as_posix()
    return ["git", "-c", f"safe.directory={safe_directory}", *args]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _lock_for(task_id: str) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(task_id, threading.RLock())


def _output_root(output_directory: str | os.PathLike[str] | None = None) -> Path:
    root = Path(output_directory or DATA_PIPELINE.get("output_directory", "./data/output"))
    return root.expanduser().resolve()


def task_directory(task_id: str, output_directory: str | os.PathLike[str] | None = None) -> Path:
    task_id = str(task_id or "").strip()
    if not task_id or not _TASK_ID.fullmatch(task_id):
        raise ValueError("invalid task id")
    root = _output_root(output_directory)
    candidate = (root / task_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("task directory escapes output directory") from exc
    return candidate


def manifest_path(task_id: str, output_directory: str | os.PathLike[str] | None = None) -> Path:
    return task_directory(task_id, output_directory) / "run_manifest.json"


def _git_revision() -> str:
    global _git_revision_cache
    if _git_revision_cache is not None:
        return _git_revision_cache
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            _git_args(repo_root, "rev-parse", "HEAD"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        _git_revision_cache = os.environ.get("RWKV_ECRA_CODE_REVISION", "unknown")
        return _git_revision_cache
    revision = result.stdout.strip()
    _git_revision_cache = revision if result.returncode == 0 and revision else os.environ.get("RWKV_ECRA_CODE_REVISION", "unknown")
    return _git_revision_cache


def _workspace_snapshot() -> dict[str, Any]:
    """Capture the commit plus a hash of current source files.

    A commit alone is insufficient while experiments are run from an
    uncommitted worktree.  The hash is an audit fingerprint, not a substitute
    for committing the experiment code.
    """
    global _workspace_snapshot_cache
    if _workspace_snapshot_cache is not None:
        return dict(_workspace_snapshot_cache)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        status = subprocess.run(
            _git_args(repo_root, "status", "--porcelain=v1"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout
        tracked = subprocess.run(
            _git_args(repo_root, "ls-files", "-m", "--others", "--exclude-standard"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        _workspace_snapshot_cache = {"dirty": True, "hash": "unavailable"}
        return dict(_workspace_snapshot_cache)

    digest = hashlib.sha256()
    included = 0
    excluded_parts = {".git", ".venv", "data/output", "logs", "__pycache__"}
    for relative in sorted(set(item.strip() for item in tracked if item.strip())):
        normalized = relative.replace("\\", "/")
        if any(normalized == part or normalized.startswith(part + "/") for part in excluded_parts):
            continue
        path = repo_root / relative
        if not path.is_file():
            continue
        try:
            digest.update(normalized.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
            included += 1
        except OSError:
            continue
    _workspace_snapshot_cache = {"dirty": bool(status.strip()), "hash": digest.hexdigest()[:16], "file_count": included}
    return dict(_workspace_snapshot_cache)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _model_snapshot() -> dict[str, Any]:
    provider = os.environ.get("RWKV_ECRA_MODEL_PROVIDER", "") or get_llm_provider() or DEFAULT_LLM_PROVIDER
    profile = LLM_ENDPOINTS.get(provider) if isinstance(LLM_ENDPOINTS, dict) else None
    profile = profile if isinstance(profile, dict) else {}
    backend = get_model_backend_name()
    direct_runtime = get_direct_rwkv_config()
    runtime_model = get_llm_model() or profile.get("model", "")
    runtime_label = profile.get("label", provider)
    runtime_endpoint = get_llm_base_url() or profile.get("base_url", "")
    api_key_configured = bool(get_llm_api_key() or API_KEYS.get(provider, ""))
    if backend in {"direct_rwkv", "auto"}:
        raw_model_path = str(direct_runtime.get("model_path") or "").strip()
        if raw_model_path:
            model_name = Path(raw_model_path).name
            runtime_model = model_name[:-4] if model_name.endswith(".pth") else model_name
            runtime_label = str(direct_runtime.get("model_name") or runtime_model)
        runtime_endpoint = ""
        api_key_configured = False
    return {
        "label": runtime_label,
        "provider": provider,
        "backend": backend,
        "model": runtime_model,
        "endpoint": runtime_endpoint,
        "api_key_configured": api_key_configured,
        "context_length": profile.get("context_length", 10240),
        "direct_runtime": {
            key: value
            for key, value in direct_runtime.items()
            if key not in {"api_key", "password"}
        },
    }


def build_manifest(task_id: str, query: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(metadata or {})
    workspace = _workspace_snapshot()
    experiment = {
        "experiment_id": metadata.get("experiment_id") or task_id,
        "variant": metadata.get("variant") or "candidate",
        "baseline_run_id": metadata.get("baseline_run_id") or "",
        "hypothesis": metadata.get("hypothesis") or "",
        "changed_variable": metadata.get("changed_variable") or "",
        "expected_metrics": metadata.get("expected_metrics") or [],
        "side_effects": metadata.get("side_effects") or [],
        "dataset_version": metadata.get("dataset_version") or os.environ.get("RWKV_ECRA_DATASET_VERSION", "unversioned"),
        "persona": metadata.get("persona") or "",
        "domain": metadata.get("domain") or "",
        "task_type": metadata.get("task_type") or "",
        "difficulty": metadata.get("difficulty") or "",
    }
    config_snapshot = {
        "model": _model_snapshot(),
        "retrieval": copy.deepcopy(SEARCH_CONFIG),
        "pipeline": {
            key: copy.deepcopy(value)
            for key, value in DATA_PIPELINE.items()
            if key not in {"asset_directory"}
        },
        "agent": copy.deepcopy(AGENT_CONFIG),
        "runtime": {
            "llm": copy.deepcopy(LLM_CONFIG),
            "slm": {key: copy.deepcopy(value) for key, value in SLM_CONFIG.items() if key not in {"password"}},
            "budgets": copy.deepcopy(RUNTIME_CONFIG),
            "model_backend": get_model_backend_name(),
        },
        "wigolo": {
            "mode": os.environ.get("RWKV_ECRA_WIGOLO_MODE", WIGOLO_CONFIG.get("mode", "auto")),
            "base_url": os.environ.get("WIGOLO_BASE_URL", WIGOLO_CONFIG.get("base_url", "")),
            "timeout_seconds": os.environ.get("WIGOLO_TIMEOUT_SECONDS", WIGOLO_CONFIG.get("timeout_seconds", 20)),
        },
        "tracking": {"enabled": bool(TRACKING.get("enable", True))},
    }
    return {
        "schema_version": "rwkv-ecra.run.v1",
        "run_id": task_id,
        "task_id": task_id,
        "status": "running",
        "created_at": utc_now(),
        "started_at": utc_now(),
        "ended_at": "",
        "query": query,
        "experiment": experiment,
        "code_revision": _git_revision(),
        "workspace_dirty": workspace["dirty"],
        "workspace_hash": workspace["hash"],
        "config_version": _json_hash(config_snapshot),
        "config": config_snapshot,
        "prompt_version": metadata.get("prompt_version") or os.environ.get("RWKV_ECRA_PROMPT_VERSION", "unversioned"),
        "search_sources": metadata.get("search_sources") or ["wigolo", "keyless_fallback", "openalex", "arxiv", "crossref"],
        "artifacts": {
            "events": "events.jsonl",
            "retrieval_report": "retrieval_report.jsonl",
        },
        "summary": {},
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_file_lease(path):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        temporary.replace(path)


def ensure_manifest(
    task_id: str,
    *,
    query: str = "",
    metadata: dict[str, Any] | None = None,
    output_directory: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    path = manifest_path(task_id, output_directory)
    lock = _lock_for(task_id)
    with lock:
        if path.exists():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = build_manifest(task_id, query, metadata)
        else:
            manifest = build_manifest(task_id, query, metadata)
        if query and not manifest.get("query"):
            manifest["query"] = query
        if metadata:
            experiment = manifest.setdefault("experiment", {})
            for key, value in metadata.items():
                if value not in (None, "", [], {}):
                    experiment[key] = value
        _write_json(path, manifest)
        return manifest


def update_manifest(task_id: str, **updates: Any) -> dict[str, Any]:
    path = manifest_path(task_id)
    lock = _lock_for(task_id)
    with lock:
        manifest = ensure_manifest(task_id)
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(manifest.get(key), dict):
                manifest[key].update(value)
            else:
                manifest[key] = value
        _write_json(path, manifest)
        return manifest


def finalize_manifest(
    task_id: str,
    *,
    status: str,
    error: str = "",
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "status": status,
        "ended_at": utc_now(),
        "error": error[:1000],
    }
    # A caller may only know the terminal status. Preserve a summary already
    # emitted by the `final` event unless an explicit replacement was given.
    if summary is not None:
        updates["summary"] = dict(summary)
    manifest = update_manifest(task_id, **updates)
    started = manifest.get("started_at")
    ended = manifest.get("ended_at")
    try:
        duration = (
            datetime.fromisoformat(ended).timestamp() - datetime.fromisoformat(started).timestamp()
        ) * 1000
        manifest = update_manifest(task_id, duration_ms=round(max(0.0, duration), 1))
    except (TypeError, ValueError, OSError):
        pass
    return manifest


def _read_events(task_id: str, output_directory: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    path = task_directory(task_id, output_directory) / "events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _parse_result(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("result")
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _norm_url(value: str) -> str:
    return value.strip().casefold().rstrip("/")


def reconstruct_run(task_id: str, output_directory: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Rebuild the normalized run view from the persisted manifest and events."""
    root = task_directory(task_id, output_directory)
    manifest_file = root / "run_manifest.json"
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = build_manifest(task_id)
    events = _read_events(task_id, output_directory)
    search_queries: list[str] = []
    search_results: list[dict[str, Any]] = []
    navigation_trace: list[dict[str, Any]] = []
    ranking_trace: list[dict[str, Any]] = []
    context_trace: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    model_outputs: list[dict[str, Any]] = []
    final_answer = ""
    citations: list[dict[str, Any]] = []
    retrieval_citations: list[dict[str, Any]] = []
    risk_validation: dict[str, Any] = {}
    for event in events:
        event_type = event.get("type")
        if event_type == "user_input" and not manifest.get("query"):
            manifest["query"] = event.get("content", "")
        if event_type == "query_candidates":
            search_queries.extend(str(item) for item in event.get("queries") or [])
        if event_type == "tool_result":
            result = _parse_result(event)
            for item in result.get("results") or []:
                if isinstance(item, dict):
                    search_results.append(item)
                    if item.get("page_excerpt") or item.get("abstract") or item.get("content"):
                        evidence.append(item)
            for ref in result.get("citation_refs") or []:
                if isinstance(ref, dict):
                    retrieval_citations.append(ref)
            for url in result.get("sources") or []:
                sources.append({"url": url, "action": event.get("action"), "query": event.get("query", "")})
        if event_type in {"navigation", "page_fetch", "page_extract", "content_extract", "one_hop"}:
            navigation_trace.append(event)
        if event_type == "ranking":
            ranking_trace.append(event)
        if event_type == "context_build":
            context_trace.append(event)
        if event_type == "risk_validation" and isinstance(event.get("data"), dict):
            risk_validation = event["data"]
        if event_type in {"error", "retry", "provider_error"} or (
            event_type == "model_call" and event.get("status") == "failed"
        ):
            errors.append(event)
        if event_type == "model_call":
            model_outputs.append({
                "phase": event.get("operation") or "model_call",
                "status": event.get("status", ""),
                "prompt": event.get("prompt") or event.get("input_messages") or "",
                "output": event.get("output") or "",
                "model": event.get("model", ""),
                "duration_ms": event.get("duration_ms"),
                "prompt_tokens": event.get("prompt_tokens", 0),
                "completion_tokens": event.get("completion_tokens", 0),
                "request_max_tokens": event.get("request_max_tokens"),
                "finish_reason": event.get("finish_reason", ""),
                "stop": event.get("stop") or [],
            })
        if event_type in {"synthesis", "final"}:
            if event.get("content"):
                final_answer = str(event.get("content"))
            if event_type == "synthesis":
                model_outputs.append({
                    "phase": "synthesis",
                    "prompt": event.get("prompt", ""),
                    "output": event.get("model_output", event.get("content", "")),
                    "repair_prompt": event.get("repair_prompt", ""),
                    "repair_output": event.get("repair_output", ""),
                    "mode": event.get("mode", ""),
                    "context_text": event.get("context_text", ""),
                    "selected_evidence": event.get("selected_evidence") or [],
                    "context_stats": event.get("context_stats") or {},
                })
                citations.extend(item for item in event.get("citation_refs") or [] if isinstance(item, dict))
    unique_sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in sources:
        url = str(item.get("url") or "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(item)
    def _unique_citations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            key = _norm_url(str(item.get("url") or "")) or str(item.get("ref_id") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(item)
        return output

    unique_citations = _unique_citations(citations)
    unique_retrieval_citations = _unique_citations(retrieval_citations)
    return {
        "manifest": manifest,
        "events": events,
        "query": manifest.get("query", ""),
        "search_queries": list(dict.fromkeys(search_queries)),
        "search_results": search_results,
        "navigation_trace": navigation_trace,
        "ranking_trace": ranking_trace,
        "context_trace": context_trace,
        "sources": unique_sources,
        "evidence": evidence,
        "model_outputs": model_outputs,
        "final_answer": final_answer,
        "citations": unique_citations,
        "retrieval_citations": unique_retrieval_citations,
        "risk_validation": risk_validation,
        "errors": errors,
    }
