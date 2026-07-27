"""Small stdlib-only client for an optional local wigolo REST daemon.

The project deliberately does not require a wigolo package or an API key at
install time.  When a daemon is available, this client talks to its local
``/v1/{tool}`` endpoints; callers can decide whether to fall back to the
existing public HTML search when it is not available.
"""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import (
    get_wigolo_api_token,
    get_wigolo_base_url,
    get_wigolo_timeout,
)


class WigoloError(RuntimeError):
    """Raised when the local wigolo endpoint cannot return a JSON response."""


class WigoloClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        api_token: str | None = None,
    ) -> None:
        self.base_url = (base_url or get_wigolo_base_url()).rstrip("/")
        self.timeout = timeout if timeout is not None else get_wigolo_timeout()
        self.api_token = api_token if api_token is not None else get_wigolo_api_token()

    def _post(self, tool: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            raise WigoloError("WIGOLO_BASE_URL is empty")

        request = Request(
            f"{self.base_url}/v1/{tool.lstrip('/')}",
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "RWKV-ECRA/wigolo-adapter",
            },
            method="POST",
        )
        if self.api_token:
            request.add_header("Authorization", f"Bearer {self.api_token}")

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise WigoloError(f"HTTP {exc.code} from wigolo {tool}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise WigoloError(f"cannot reach wigolo at {self.base_url}: {exc}") from exc

        try:
            value = json.loads(body)
        except json.JSONDecodeError as exc:
            raise WigoloError(f"wigolo returned non-JSON data for {tool}") from exc
        if not isinstance(value, dict):
            raise WigoloError(f"wigolo returned an unexpected response for {tool}")
        if value.get("error") and not value.get("results"):
            raise WigoloError(str(value["error"])[:500])
        return value

    def search(self, query: str, *, max_results: int = 6, search_depth: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
        }
        if search_depth:
            payload["search_depth"] = search_depth
        return self._post("search", payload)

    def fetch(self, url: str) -> dict[str, Any]:
        return self._post("fetch", {"url": url})


def first_text(value: Any, *keys: str) -> str:
    """Extract the first useful text field from a wigolo response."""
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""
