"""Safe workspace file and report operations.

The HTTP layer should not construct filesystem paths from request strings.
This module owns path containment, upload limits, file listing and report
loading so the same rules are used by every endpoint.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO


class WorkspacePathError(ValueError):
    """Raised when a user-controlled path escapes its configured workspace."""


class WorkspaceFileError(ValueError):
    """Raised when a workspace file cannot be read or written safely."""


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class WorkspaceFile:
    path: Path
    relative_path: str
    is_image: bool
    text: str | None = None


def resolve_workspace_path(
    root: str | os.PathLike[str],
    relative_path: str,
    *,
    allow_root: bool = False,
) -> tuple[Path, str]:
    """Resolve a relative request path and prove it remains under root."""
    raw = str(relative_path or "").strip().replace("\\", "/")
    if not raw:
        raise WorkspacePathError("path is empty")

    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    parts = tuple(part for part in posix.parts if part not in {"", "."})
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise WorkspacePathError("absolute paths are not allowed")
    if not parts or any(part == ".." for part in parts):
        raise WorkspacePathError("parent traversal is not allowed")

    root_path = Path(root).expanduser().resolve()
    candidate = root_path.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise WorkspacePathError("path escapes the workspace") from exc
    if not allow_root and candidate == root_path:
        raise WorkspacePathError("workspace root is not a file")
    return candidate, "/".join(parts)


def _ensure_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_file(
    root: str | os.PathLike[str],
    relative_path: str,
    source: BinaryIO,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> str:
    root_path = _ensure_root(root)
    target, normalized = resolve_workspace_path(root_path, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with target.open("wb") as destination:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise WorkspaceFileError(f"file exceeds the {max_bytes} byte upload limit")
                destination.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return normalized


def list_workspace_files(root: str | os.PathLike[str]) -> list[str]:
    root_path = _ensure_root(root)
    files: list[str] = []
    for current_root, directories, filenames in os.walk(root_path, followlinks=False):
        directories[:] = [name for name in directories if not name.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            full_path = Path(current_root) / filename
            try:
                _, relative = resolve_workspace_path(root_path, str(full_path.relative_to(root_path)))
            except WorkspacePathError:
                continue
            files.append(relative)
    return sorted(files)


def delete_workspace_file(root: str | os.PathLike[str], relative_path: str) -> bool:
    root_path = _ensure_root(root)
    target, _ = resolve_workspace_path(root_path, relative_path)
    if not target.exists():
        return False
    if not target.is_file() or target.is_symlink():
        raise WorkspaceFileError("only regular files can be deleted")
    target.unlink()
    parent = target.parent
    while parent != root_path and parent.exists():
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return True


def read_workspace_file(root: str | os.PathLike[str], relative_path: str) -> WorkspaceFile | None:
    root_path = _ensure_root(root)
    target, normalized = resolve_workspace_path(root_path, relative_path)
    if not target.exists():
        return None
    if not target.is_file() or target.is_symlink():
        raise WorkspaceFileError("only regular files can be read")
    if target.suffix.casefold() in IMAGE_EXTENSIONS:
        return WorkspaceFile(target, normalized, True)
    return WorkspaceFile(
        target,
        normalized,
        False,
        target.read_text(encoding="utf-8", errors="replace"),
    )


def cleanup_directories(directories: list[str | os.PathLike[str]]) -> list[str]:
    """Clear configured workspace directories and return non-fatal errors."""
    errors: list[str] = []
    for raw_directory in directories:
        directory = Path(raw_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            try:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except OSError as exc:
                errors.append(f"{child}: {exc}")
    return errors


def read_task_report(output_root: str | os.PathLike[str], task_id: str) -> list[dict[str, Any]] | None:
    task_directory, _ = resolve_workspace_path(output_root, task_id)
    if not task_directory.exists() or not task_directory.is_dir():
        return None

    jsonl_candidates = sorted(
        (path for path in task_directory.rglob("*.jsonl") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if jsonl_candidates:
        records: list[dict[str, Any]] = []
        for line in jsonl_candidates[0].read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    md_candidates = sorted(
        (path for path in task_directory.rglob("*.md") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if md_candidates:
        return [{
            "record_type": "final_beautified_markdown",
            "content": md_candidates[0].read_text(encoding="utf-8", errors="replace"),
        }]
    return []
