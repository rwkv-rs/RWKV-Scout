"""Keyless network fetching with a WSL curl fallback for this Windows setup."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any

import requests

from config import get_network_timeout_seconds
from utils.time_budget import bounded_timeout


class NetworkFetchError(RuntimeError):
    pass


_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)
_META_CHARSET_RE = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)
_MOJIBAKE_MARKERS = ("Ã", "Â", "â", "æ", "å", "ç", "è", "é", "ï¿½", "�")


def _encoding_candidates(
    payload: bytes,
    *,
    declared_encoding: str = "",
    apparent_encoding: str = "",
) -> list[str]:
    candidates: list[str] = []

    def add(value: str) -> None:
        normalized = str(value or "").strip().strip("\"'").lower()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add(declared_encoding)
    match = _CHARSET_RE.search(str(declared_encoding or ""))
    if match:
        add(match.group(1))
    meta = _META_CHARSET_RE.search(payload[:65536])
    if meta:
        add(meta.group(1).decode("ascii", errors="ignore"))
    add("utf-8")
    add(apparent_encoding)
    add("gb18030")
    add("gbk")
    add("latin-1")
    return candidates


def _text_quality(value: str, order: int) -> tuple[int, int, int, int]:
    """Rank decoded text and penalize classic UTF-8-as-Latin-1 mojibake."""
    replacement_count = value.count("\ufffd") + value.count("�")
    mojibake_count = sum(value.count(marker) for marker in _MOJIBAKE_MARKERS)
    control_count = sum(1 for char in value if ord(char) < 32 and char not in "\r\n\t")
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", value))
    return (replacement_count * 1000 + mojibake_count * 20 + control_count * 10, -cjk_count, order, -len(value))


def decode_http_body(
    payload: bytes,
    *,
    declared_encoding: str = "",
    apparent_encoding: str = "",
) -> str:
    """Decode a page body without trusting a broken ISO-8859-1 declaration.

    Several Chinese sites label UTF-8 pages as ISO-8859-1 or omit charset
    metadata. Decode the raw bytes with a small, deterministic candidate set
    and choose the text with the fewest replacement/mojibake markers.
    """
    if not payload:
        return ""
    decoded: list[tuple[int, str]] = []
    for order, encoding in enumerate(
        _encoding_candidates(
            payload,
            declared_encoding=declared_encoding,
            apparent_encoding=apparent_encoding,
        )
    ):
        try:
            value = payload.decode(encoding, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
        decoded.append((order, value))
    if not decoded:
        return payload.decode("utf-8", errors="replace")
    return min(decoded, key=lambda item: _text_quality(item[1], item[0]))[1]


def _decode_process_output(value: bytes) -> str:
    """Decode WSL/Windows subprocess output without mojibake diagnostics."""
    if b"\x00" in value[:80]:
        try:
            return value.decode("utf-16-le", errors="replace")
        except UnicodeDecodeError:
            pass
    return value.decode("utf-8", errors="replace")


def _fetch_via_wsl_curl(url: str, params: dict[str, Any] | None, timeout: float) -> bytes:
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
        error = _decode_process_output(proc.stderr)[-500:]
        raise NetworkFetchError(f"WSL curl exited with {proc.returncode}: {error}")
    return proc.stdout


def fetch_text(url: str, params: dict[str, Any] | None = None, timeout: float = 20) -> str:
    """Fetch a public URL without requiring a paid API key."""
    effective_timeout = bounded_timeout(min(float(timeout), get_network_timeout_seconds()))
    if os.name == "nt" and shutil.which("wsl.exe"):
        try:
            return decode_http_body(_fetch_via_wsl_curl(url, params, effective_timeout))
        except NetworkFetchError as wsl_error:
            # WSL networking can be unavailable even while the Windows
            # process has a working direct route. Keep fetching provider
            # agnostic and try the native transport before giving up.
            try:
                session = requests.Session()
                session.trust_env = False
                response = session.get(url, params=params, timeout=effective_timeout)
                response.raise_for_status()
                return decode_http_body(
                    response.content,
                    declared_encoding=response.headers.get("content-type", "") or response.encoding or "",
                    apparent_encoding=response.apparent_encoding or "",
                )
            except requests.RequestException as native_error:
                raise NetworkFetchError(
                    f"WSL transport failed ({wsl_error}); native transport failed ({native_error})"
                ) from native_error

    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(url, params=params, timeout=effective_timeout)
        response.raise_for_status()
        return decode_http_body(
            response.content,
            declared_encoding=response.headers.get("content-type", "") or response.encoding or "",
            apparent_encoding=response.apparent_encoding or "",
        )
    except requests.RequestException as exc:
        raise NetworkFetchError(f"HTTP request failed: {exc}") from exc


def fetch_json(url: str, params: dict[str, Any] | None = None, timeout: float = 20) -> dict[str, Any]:
    body = fetch_text(url, params=params, timeout=timeout)
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise NetworkFetchError(f"Invalid JSON from {url}: {body[:200]}") from exc
    if not isinstance(value, dict):
        raise NetworkFetchError(f"Expected an object from {url}")
    return value
