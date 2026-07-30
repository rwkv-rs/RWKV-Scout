"""Keyless network fetching with a WSL curl fallback for this Windows setup."""

from __future__ import annotations

import json
import ipaddress
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from utils.text_encoding import repair_mojibake
from config import get_network_timeout_seconds
from utils.time_budget import bounded_timeout


class NetworkFetchError(RuntimeError):
    pass


_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)
_META_CHARSET_RE = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)
_MOJIBAKE_MARKERS = ("Ã", "Â", "â", "æ", "å", "ç", "è", "é", "ï¿½", "�")
_DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}


def _is_pdf_url(value: str) -> bool:
    """Return whether a URL explicitly points at a PDF document."""

    path = urlparse(str(value or "")).path.casefold()
    return path.endswith(".pdf") or path.endswith(".pdf/")


def _reject_binary_document(payload: bytes, url: str) -> None:
    """Keep binary documents out of the HTML/Markdown text pipeline."""

    if _is_pdf_url(url) or bytes(payload or b"").lstrip().startswith(b"%PDF-"):
        raise NetworkFetchError(f"PDF document is not supported by the HTML text pipeline: {url}")


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
    value = min(decoded, key=lambda item: _text_quality(item[1], item[0]))[1]
    return repair_mojibake(value)


def _decode_process_output(value: bytes) -> str:
    """Decode WSL/Windows subprocess output without mojibake diagnostics."""
    if b"\x00" in value[:80]:
        try:
            return value.decode("utf-16-le", errors="replace")
        except UnicodeDecodeError:
            pass
    return value.decode("utf-8", errors="replace")


def _wsl_default_gateway() -> str:
    """Return the Windows-side gateway for a WSL2 network namespace."""

    try:
        for line in Path("/proc/net/route").read_text(encoding="ascii").splitlines()[1:]:
            fields = line.split()
            if len(fields) < 3 or fields[1] != "00000000":
                continue
            raw = bytes.fromhex(fields[2])
            return str(ipaddress.ip_address(bytes(reversed(raw))))
    except (OSError, ValueError):
        pass
    return ""


def _replace_loopback_proxy_host(value: str) -> str:
    """Make a Windows loopback proxy reachable from WSL2."""

    candidate = str(value or "").strip()
    if not candidate:
        return ""
    parsed = urlparse(candidate if "://" in candidate else f"http://{candidate}")
    host = (parsed.hostname or "").casefold()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return parsed.geturl()
    gateway = _wsl_default_gateway()
    if not gateway:
        return parsed.geturl()
    try:
        port = parsed.port
    except ValueError:
        port = None
    username = parsed.username or ""
    password = parsed.password or ""
    auth = ""
    if username:
        auth = username
        if password:
            auth += f":{password}"
        auth += "@"
    netloc = f"{auth}{gateway}"
    if port:
        netloc += f":{port}"
    return parsed._replace(netloc=netloc).geturl()


def _windows_proxy_settings() -> dict[str, str]:
    """Read the Windows Internet Settings proxy when running inside WSL."""

    if os.name == "nt":
        return {}
    if "microsoft" not in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8", errors="ignore").casefold():
        return {}
    executable = next(
        (
            candidate
            for candidate in (shutil.which("reg.exe"), "/mnt/c/Windows/System32/reg.exe")
            if candidate and os.path.exists(candidate)
        ),
        "",
    )
    if not executable:
        return {}
    try:
        process = subprocess.run(
            [
                executable,
                "query",
                r"HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            ],
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    output = _decode_process_output(process.stdout + process.stderr)
    enabled = re.search(r"ProxyEnable\s+REG_DWORD\s+0x([0-9a-f]+)", output, re.IGNORECASE)
    if not enabled or int(enabled.group(1), 16) == 0:
        return {}
    match = re.search(r"ProxyServer\s+REG_SZ\s+(.+)", output, re.IGNORECASE)
    if not match:
        return {}
    raw = match.group(1).strip()
    values: dict[str, str] = {}
    for entry in raw.split(";"):
        if "=" in entry:
            scheme, proxy = entry.split("=", 1)
            values[scheme.strip().casefold()] = proxy.strip()
        else:
            values["default"] = entry.strip()
    default = values.get("default") or values.get("http") or values.get("https") or ""
    http_proxy = _replace_loopback_proxy_host(values.get("http") or default)
    https_proxy = _replace_loopback_proxy_host(values.get("https") or default)
    return {key: value for key, value in {"http": http_proxy, "https": https_proxy}.items() if value}


def get_network_proxies() -> dict[str, str]:
    """Resolve explicit, environment, then Windows-system proxy settings.

    Requests sessions in this project deliberately disable ``trust_env`` so a
    broken inherited proxy cannot silently change the route.  This function is
    the single explicit opt-in route for proxies and also translates a Windows
    loopback proxy to the WSL2 gateway address.
    """

    explicit = os.environ.get("RWKV_ECRA_HTTP_PROXY", "").strip()
    explicit_https = os.environ.get("RWKV_ECRA_HTTPS_PROXY", "").strip() or explicit
    if explicit or explicit_https:
        return {
            key: _replace_loopback_proxy_host(value)
            for key, value in {"http": explicit or explicit_https, "https": explicit_https or explicit}.items()
            if value
        }

    environment = {
        "http": os.environ.get("HTTP_PROXY", "").strip() or os.environ.get("http_proxy", "").strip(),
        "https": os.environ.get("HTTPS_PROXY", "").strip() or os.environ.get("https_proxy", "").strip(),
    }
    if any(environment.values()):
        return {key: _replace_loopback_proxy_host(value) for key, value in environment.items() if value}

    auto = os.environ.get("RWKV_ECRA_AUTO_WINDOWS_PROXY", "1").strip().casefold()
    if auto not in {"0", "false", "no", "off"}:
        return _windows_proxy_settings()
    return {}


def create_network_session(headers: dict[str, str] | None = None) -> requests.Session:
    """Create the one HTTP transport used by all provider adapters."""

    session = requests.Session()
    session.trust_env = False
    proxies = get_network_proxies()
    if proxies:
        session.proxies.update(proxies)
    session.headers.update(_DEFAULT_HTTP_HEADERS)
    if headers:
        session.headers.update({str(key): str(value) for key, value in headers.items()})
    return session


def _fetch_via_wsl_curl(
    url: str,
    params: dict[str, Any] | None,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> bytes:
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
    for key, value in (headers or {}).items():
        args.extend(["-H", f"{key}: {value}"])
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


def fetch_text(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 20,
    headers: dict[str, str] | None = None,
) -> str:
    """Fetch a public URL without requiring a paid API key."""
    if _is_pdf_url(url):
        raise NetworkFetchError(f"PDF document is not supported by the HTML text pipeline: {url}")
    effective_timeout = bounded_timeout(min(float(timeout), get_network_timeout_seconds()))
    effective_headers = dict(_DEFAULT_HTTP_HEADERS)
    if headers:
        effective_headers.update({str(key): str(value) for key, value in headers.items()})
    if os.name == "nt" and shutil.which("wsl.exe"):
        try:
            payload = _fetch_via_wsl_curl(url, params, effective_timeout, effective_headers)
            _reject_binary_document(payload, url)
            return decode_http_body(payload)
        except NetworkFetchError as wsl_error:
            # WSL networking can be unavailable even while the Windows
            # process has a working direct route. Keep fetching provider
            # agnostic and try the native transport before giving up.
            try:
                session = create_network_session(effective_headers)
                response = session.get(url, params=params, timeout=effective_timeout)
                response.raise_for_status()
                if "application/pdf" in str(response.headers.get("content-type") or "").casefold():
                    raise NetworkFetchError(f"PDF document is not supported by the HTML text pipeline: {url}")
                _reject_binary_document(response.content, url)
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
        session = create_network_session(effective_headers)
        response = session.get(url, params=params, timeout=effective_timeout)
        response.raise_for_status()
        if "application/pdf" in str(response.headers.get("content-type") or "").casefold():
            raise NetworkFetchError(f"PDF document is not supported by the HTML text pipeline: {url}")
        _reject_binary_document(response.content, url)
        return decode_http_body(
            response.content,
            declared_encoding=response.headers.get("content-type", "") or response.encoding or "",
            apparent_encoding=response.apparent_encoding or "",
        )
    except requests.RequestException as exc:
        raise NetworkFetchError(f"HTTP request failed: {exc}") from exc


def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 20,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = fetch_text(url, params=params, timeout=timeout, headers=headers)
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise NetworkFetchError(f"Invalid JSON from {url}: {body[:200]}") from exc
    if not isinstance(value, dict):
        raise NetworkFetchError(f"Expected an object from {url}")
    return value
