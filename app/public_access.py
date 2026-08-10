"""Public-tunnel access policy shared by the API and frontend proxy."""

from __future__ import annotations

import os
import re
import urllib.parse


_PUBLIC_TASK_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
_PUBLIC_TASK_GET = re.compile(
    rf"^/frontend-api/history/{_PUBLIC_TASK_ID}/(?:events|report)$"
)
_PUBLIC_TASK_STOP = re.compile(
    rf"^/frontend-api/analyze/{_PUBLIC_TASK_ID}/stop$"
)


def public_mode_enabled() -> bool:
    value = str(os.environ.get("RWKV_ECRA_PUBLIC_MODE", "") or "").strip().casefold()
    return value in {"1", "true", "yes", "on"}


def public_frontend_request_allowed(method: str, path: str) -> bool:
    """Return whether one ``/frontend-api`` request is safe for a public tunnel."""

    normalized_method = str(method or "").upper()
    normalized_path = urllib.parse.unquote(str(path or "")).rstrip("/") or "/"
    if normalized_method == "GET":
        return normalized_path == "/frontend-api/config" or bool(
            _PUBLIC_TASK_GET.fullmatch(normalized_path)
        )
    if normalized_method == "POST":
        return normalized_path == "/frontend-api/analyze" or bool(
            _PUBLIC_TASK_STOP.fullmatch(normalized_path)
        )
    return False
