"""Protocol adapters at the RWKV-Scout tool boundary.

The RWKV engine keeps its existing transcript and JSON contract.  This module
only converts transport envelopes at the harness boundary so the same agent
loop can accept the flat rwkv-skills shape and common function-call shapes.
It intentionally does not select tools, rewrite arguments, or infer missing
values.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


CALL_SCHEMA_VERSION = "tool_call.v1"
RESULT_SCHEMA_VERSION = "tool_result.v1"


def _arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("tool call arguments must be a JSON object")
    return dict(value)


def canonicalize_tool_call(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a supported transport envelope into one internal call shape."""

    value = dict(payload or {})
    call_id = str(value.get("call_id") or value.get("tool_call_id") or value.get("id") or "").strip()
    native_calls = value.get("tool_calls")
    if isinstance(native_calls, list) and native_calls:
        first = native_calls[0] if isinstance(native_calls[0], Mapping) else {}
        function = first.get("function") if isinstance(first, Mapping) else {}
        function = function if isinstance(function, Mapping) else {}
        name = str(function.get("name") or first.get("name") or "").strip()
        arguments = _arguments(function.get("arguments", first.get("arguments", {})))
        call_id = call_id or str(first.get("id") or "").strip()
    else:
        function_value = value.get("function")
        function = function_value if isinstance(function_value, Mapping) else {}
        function_name = (
            str(function_value).strip()
            if isinstance(function_value, str)
            else str(function.get("name") or "").strip()
        )
        name = str(
            value.get("name")
            or value.get("tool_name")
            or value.get("action")
            or value.get("tool")
            or function_name
            or ""
        ).strip()
        arguments = _arguments(
            value.get("arguments")
            if value.get("arguments") is not None
            else value.get("args")
            if value.get("args") is not None
            else value.get("parameters")
            if value.get("parameters") is not None
            else function.get("arguments", {})
        )

    if not name:
        raise ValueError("tool call name is empty")
    result: dict[str, Any] = {
        "schema_version": CALL_SCHEMA_VERSION,
        "name": name,
        "arguments": arguments,
    }
    if call_id:
        result["call_id"] = call_id
    for key in ("task_point_id", "point_id"):
        if value.get(key):
            result["task_point_id"] = value[key]
            break
    return result


def normalize_tool_result(
    result: Any,
    *,
    tool_name: str,
    call_id: str = "",
) -> dict[str, Any]:
    """Attach stable call metadata without changing the tool payload itself."""

    if isinstance(result, Mapping):
        payload = dict(result)
    elif isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            parsed = None
        payload = dict(parsed) if isinstance(parsed, Mapping) else {"status": "ok", "raw": result}
    else:
        payload = {"status": "ok", "raw": result}

    payload.setdefault("protocol_version", RESULT_SCHEMA_VERSION)
    payload.setdefault("tool", str(tool_name or ""))
    if call_id:
        payload.setdefault("tool_call_id", str(call_id))
    return payload


__all__ = [
    "CALL_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "canonicalize_tool_call",
    "normalize_tool_result",
]
