"""Keyless network fetching with a WSL curl fallback for this Windows setup."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

import requests


class NetworkFetchError(RuntimeError):
    pass


def _fetch_via_wsl_curl(url: str, params: dict[str, Any] | None, timeout: int) -> str:
    args = [
        "wsl.exe",
        "-d",
        os.environ.get("RWKV_WSL_DISTRO", "UbuntuRecovered"),
        "--",
        "curl",
        "-sS",
        "--compressed",
        "--max-time",
        str(timeout),
        "-G",
        url,
    ]
    for key, value in (params or {}).items():
        args.extend(["--data-urlencode", f"{key}={value}"])

    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout + 5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NetworkFetchError(f"WSL curl failed: {exc}") from exc
    if proc.returncode != 0:
        error = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise NetworkFetchError(f"WSL curl exited with {proc.returncode}: {error}")
    return proc.stdout.decode("utf-8", errors="replace")


def fetch_text(url: str, params: dict[str, Any] | None = None, timeout: int = 20) -> str:
    """Fetch a public URL without requiring a paid API key."""
    if os.name == "nt" and shutil.which("wsl.exe"):
        return _fetch_via_wsl_curl(url, params, timeout)

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        raise NetworkFetchError(f"HTTP request failed: {exc}") from exc


def fetch_json(url: str, params: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    body = fetch_text(url, params=params, timeout=timeout)
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise NetworkFetchError(f"Invalid JSON from {url}: {body[:200]}") from exc
    if not isinstance(value, dict):
        raise NetworkFetchError(f"Expected an object from {url}")
    return value
