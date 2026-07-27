"""Small, stable error taxonomy used by runtime logs and metrics."""

from __future__ import annotations


def classify_error(error: BaseException | str) -> str:
    text = str(error).casefold()
    name = type(error).__name__.casefold() if isinstance(error, BaseException) else ""
    combined = f"{name} {text}"
    if "timeout" in combined or "timed out" in combined:
        return "timeout"
    if any(term in combined for term in ("401", "403", "unauthorized", "forbidden", "api key", "authentication")):
        return "auth"
    if "429" in combined or "rate limit" in combined or "too many" in combined:
        return "rate_limit"
    if any(term in combined for term in ("json", "decode", "schema", "parse")):
        return "parse"
    if any(term in combined for term in ("http", "url", "network", "connection", "dns", "socket")):
        return "network"
    if any(term in combined for term in ("file", "directory", "permission", "no such")):
        return "filesystem"
    if any(term in combined for term in ("provider", "wigolo", "model", "llm")):
        return "provider"
    if isinstance(error, (ValueError, TypeError)):
        return "validation"
    return "unknown"
