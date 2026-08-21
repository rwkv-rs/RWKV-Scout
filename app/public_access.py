"""Public-tunnel access policy shared by the API and frontend proxy."""

from __future__ import annotations

import os
import re
import urllib.parse


_PUBLIC_TASK_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
_PUBLIC_TASK_GET = re.compile(
    rf"^/frontend-api/history/{_PUBLIC_TASK_ID}/(?:events|report|trace)$"
)
_PUBLIC_TASK_STOP = re.compile(
    rf"^/frontend-api/(?:analyze|history)/{_PUBLIC_TASK_ID}/stop$"
)
_PUBLIC_TASK_METRICS = re.compile(
    rf"^/frontend-api/metrics/tokens/{_PUBLIC_TASK_ID}$"
)

_PUBLIC_GET_ENDPOINTS = {
    "/frontend-api/config",
    "/frontend-api/history",
    "/frontend-api/metrics/operational",
    "/frontend-api/metrics/tokens",
    "/frontend-api/tokens",
}

_PUBLIC_POST_ENDPOINTS = {
    "/frontend-api/analyze",
    "/frontend-api/chat",
}


def public_mode_enabled() -> bool:
    value = str(os.environ.get("RWKV_ECRA_PUBLIC_MODE", "") or "").strip().casefold()
    return value in {"1", "true", "yes", "on"}


def public_frontend_request_allowed(method: str, path: str) -> bool:
    """Allow internal-company UI features except files and destructive deletion."""

    normalized_method = str(method or "").upper()
    normalized_path = urllib.parse.unquote(str(path or "")).rstrip("/") or "/"
    if normalized_method == "GET":
        return (
            normalized_path in _PUBLIC_GET_ENDPOINTS
            or bool(_PUBLIC_TASK_GET.fullmatch(normalized_path))
            or bool(_PUBLIC_TASK_METRICS.fullmatch(normalized_path))
        )
    if normalized_method == "POST":
        return normalized_path in _PUBLIC_POST_ENDPOINTS or bool(
            _PUBLIC_TASK_STOP.fullmatch(normalized_path)
        )
    # File deletion and task deletion remain disabled in company-public mode.
    return False
