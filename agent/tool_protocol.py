"""Protocol adapters at the RWKV-Scout tool boundary.

The RWKV engine keeps its existing transcript and JSON contract.  This module
only converts transport envelopes at the harness boundary so the same agent
loop can accept the flat rwkv-skills shape and common function-call shapes.
It intentionally does not select tools, rewrite arguments, or infer missing
values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from agent.runtime_contracts import TOOL_CALL_CONTRACT


@dataclass(frozen=True)
class ToolCallFormatInspection:
    """Observable accounting for one representation-only conversion.

    ``consumed_fields`` names every envelope field understood by the adapter.
    ``unconsumed_fields`` names authored envelope fields that normalization
    would otherwise discard.  Argument-object members are values of the call,
    not envelope syntax, so they are consumed as a unit.
    """

    canonical: dict[str, Any]
    consumed_fields: tuple[str, ...]
    unconsumed_fields: tuple[str, ...]


def _arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("tool call arguments must be a JSON object")
    return dict(value)


def _text_aliases(*values: Any) -> list[str]:
    return [str(value).strip() for value in values if str(value or "").strip()]


def _single_text_alias(values: list[str], *, label: str) -> str:
    if len(set(values)) > 1:
        raise ValueError(f"tool call contains conflicting {label} aliases")
    return values[0] if values else ""


def _call_parts(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Collect every explicitly authored representation of one tool call."""

    value = dict(payload or {})
    # Some RWKV continuations use the function name as the sole object key:
    # ``{"web_search": {"query": "..."}}``.  With exactly one key and one
    # object value this is a lossless call envelope, not an inferred tool or a
    # repaired argument.  Unknown names are still rejected by the registry.
    named_wrapper_name = ""
    named_wrapper_arguments: Mapping[str, Any] | None = None
    standard_envelope_keys = {
        "name",
        "tool_name",
        "action",
        "tool",
        "arguments",
        "args",
        "parameters",
        "function",
        "function_call",
        "tool_calls",
        "task_record_id",
        "task_point_id",
        "point_id",
        "call_id",
        "tool_call_id",
        "id",
        "contract",
        "schema_version",
        "type",
    }
    if len(value) == 1:
        wrapper_name, wrapper_value = next(iter(value.items()))
        if wrapper_name not in standard_envelope_keys and isinstance(wrapper_value, Mapping):
            named_wrapper_name = str(wrapper_name or "").strip()
            named_wrapper_arguments = wrapper_value

    native_calls = value.get("tool_calls")
    first: Mapping[str, Any] = {}
    if native_calls is not None:
        if not isinstance(native_calls, list):
            raise ValueError("tool_calls must be a JSON array")
        if native_calls:
            if len(native_calls) != 1:
                raise ValueError("tool call payload must contain exactly one call")
            if not isinstance(native_calls[0], Mapping):
                raise ValueError("tool call entry must be a JSON object")
            first = native_calls[0]

    top_function_value = value.get("function")
    top_function = (
        top_function_value if isinstance(top_function_value, Mapping) else {}
    )
    native_function_value = first.get("function") if first else None
    native_function = (
        native_function_value
        if isinstance(native_function_value, Mapping)
        else {}
    )
    function_call_value = value.get("function_call")
    if function_call_value is not None and not isinstance(function_call_value, Mapping):
        raise ValueError("function_call must be a JSON object")
    function_call = (
        function_call_value if isinstance(function_call_value, Mapping) else {}
    )

    names = _text_aliases(
        value.get("name"),
        value.get("tool_name"),
        value.get("action"),
        value.get("tool"),
        top_function_value if isinstance(top_function_value, str) else None,
        top_function.get("name"),
        first.get("name") if first else None,
        first.get("tool_name") if first else None,
        first.get("action") if first else None,
        first.get("tool") if first else None,
        native_function_value if isinstance(native_function_value, str) else None,
        native_function.get("name"),
        function_call.get("name"),
        named_wrapper_name,
    )
    name = _single_text_alias(names, label="name")

    argument_values = [
        item
        for item in (
            value.get("arguments"),
            value.get("args"),
            value.get("parameters"),
            top_function.get("arguments") if top_function else None,
            top_function.get("args") if top_function else None,
            top_function.get("parameters") if top_function else None,
            first.get("arguments") if first else None,
            first.get("args") if first else None,
            first.get("parameters") if first else None,
            native_function.get("arguments") if native_function else None,
            native_function.get("args") if native_function else None,
            native_function.get("parameters") if native_function else None,
            function_call.get("arguments") if function_call else None,
            function_call.get("args") if function_call else None,
            function_call.get("parameters") if function_call else None,
            named_wrapper_arguments,
        )
        if item is not None
    ]
    arguments = [_arguments(item) for item in argument_values]
    if arguments and any(item != arguments[0] for item in arguments[1:]):
        raise ValueError("tool call contains conflicting argument aliases")

    task_record_ids = _text_aliases(
        value.get("task_record_id"),
        value.get("task_point_id"),
        value.get("point_id"),
        top_function.get("task_record_id") if top_function else None,
        top_function.get("task_point_id") if top_function else None,
        top_function.get("point_id") if top_function else None,
        first.get("task_record_id") if first else None,
        first.get("task_point_id") if first else None,
        first.get("point_id") if first else None,
        native_function.get("task_record_id") if native_function else None,
        native_function.get("task_point_id") if native_function else None,
        native_function.get("point_id") if native_function else None,
        function_call.get("task_record_id") if function_call else None,
        function_call.get("task_point_id") if function_call else None,
        function_call.get("point_id") if function_call else None,
    )
    call_ids = _text_aliases(
        value.get("call_id"),
        value.get("tool_call_id"),
        value.get("id"),
        top_function.get("call_id") if top_function else None,
        top_function.get("tool_call_id") if top_function else None,
        top_function.get("id") if top_function else None,
        first.get("call_id") if first else None,
        first.get("tool_call_id") if first else None,
        first.get("id") if first else None,
        native_function.get("call_id") if native_function else None,
        native_function.get("tool_call_id") if native_function else None,
        native_function.get("id") if native_function else None,
        function_call.get("call_id") if function_call else None,
        function_call.get("tool_call_id") if function_call else None,
        function_call.get("id") if function_call else None,
    )
    return {
        "name": name,
        "arguments": arguments[0] if arguments else {},
        "task_record_id": _single_text_alias(
            task_record_ids,
            label="task-record",
        ),
        "call_id": _single_text_alias(call_ids, label="call-id"),
    }


_TOP_LEVEL_ENVELOPE_KEYS = frozenset(
    {
        "name",
        "tool_name",
        "action",
        "tool",
        "arguments",
        "args",
        "parameters",
        "function",
        "function_call",
        "tool_calls",
        "task_record_id",
        "task_point_id",
        "point_id",
        "call_id",
        "tool_call_id",
        "id",
        "contract",
        "schema_version",
        "type",
    }
)
_FUNCTION_ENVELOPE_KEYS = frozenset(
    {
        "name",
        "arguments",
        "args",
        "parameters",
        "task_record_id",
        "task_point_id",
        "point_id",
        "call_id",
        "tool_call_id",
        "id",
    }
)
_NATIVE_CALL_ENVELOPE_KEYS = frozenset(
    {
        "name",
        "tool_name",
        "action",
        "tool",
        "arguments",
        "args",
        "parameters",
        "function",
        "task_record_id",
        "task_point_id",
        "point_id",
        "call_id",
        "tool_call_id",
        "id",
        "type",
    }
)


def _field_accounting(payload: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return consumed and unconsumed envelope paths without changing values."""

    value = dict(payload or {})
    if len(value) == 1:
        wrapper_name, wrapper_value = next(iter(value.items()))
        if (
            wrapper_name not in _TOP_LEVEL_ENVELOPE_KEYS
            and isinstance(wrapper_value, Mapping)
        ):
            return (str(wrapper_name),), ()

    consumed: set[str] = set()
    unconsumed: set[str] = set()
    for key in value:
        path = str(key)
        if key in _TOP_LEVEL_ENVELOPE_KEYS:
            consumed.add(path)
        else:
            unconsumed.add(path)

    type_value = value.get("type")
    if type_value is not None and str(type_value).casefold() != "function":
        consumed.discard("type")
        unconsumed.add("type")

    for wrapper_key in ("function", "function_call"):
        wrapper = value.get(wrapper_key)
        if not isinstance(wrapper, Mapping):
            continue
        for key in wrapper:
            path = f"{wrapper_key}.{key}"
            if key in _FUNCTION_ENVELOPE_KEYS:
                consumed.add(path)
            else:
                unconsumed.add(path)

    native_calls = value.get("tool_calls")
    if isinstance(native_calls, list):
        for index, call in enumerate(native_calls):
            if not isinstance(call, Mapping):
                continue
            prefix = f"tool_calls[{index}]"
            for key in call:
                path = f"{prefix}.{key}"
                if key in _NATIVE_CALL_ENVELOPE_KEYS:
                    consumed.add(path)
                else:
                    unconsumed.add(path)
            native_type = call.get("type")
            if (
                native_type is not None
                and str(native_type).casefold() != "function"
            ):
                consumed.discard(f"{prefix}.type")
                unconsumed.add(f"{prefix}.type")
            native_function = call.get("function")
            if isinstance(native_function, Mapping):
                for key in native_function:
                    path = f"{prefix}.function.{key}"
                    if key in _FUNCTION_ENVELOPE_KEYS:
                        consumed.add(path)
                    else:
                        unconsumed.add(path)
    return tuple(sorted(consumed)), tuple(sorted(unconsumed))


def _normalize_parts(parts: Mapping[str, Any]) -> dict[str, Any]:
    name = str(parts.get("name") or "")
    if not name:
        raise ValueError("tool call name is empty")
    result: dict[str, Any] = {
        "contract": TOOL_CALL_CONTRACT,
        "name": name,
        "arguments": dict(parts.get("arguments") or {}),
    }
    call_id = str(parts.get("call_id") or "")
    if call_id:
        result["call_id"] = call_id
    task_record_id = str(parts.get("task_record_id") or "")
    if task_record_id:
        result["task_record_id"] = task_record_id
    return result


def inspect_tool_call_format(payload: Mapping[str, Any]) -> ToolCallFormatInspection:
    """Inspect one common call envelope and expose any would-be data loss."""

    parts = _call_parts(payload)
    consumed, unconsumed = _field_accounting(payload)
    return ToolCallFormatInspection(
        canonical=_normalize_parts(parts),
        consumed_fields=consumed,
        unconsumed_fields=unconsumed,
    )


def _validate_convertible_tool_call_format(payload: Mapping[str, Any]) -> None:
    """Reject shapes that cannot be converted to one call without data loss.

    This is protocol validation, not part of format conversion.  In
    particular, it prevents the converter from silently choosing one of two
    model-authored calls or two conflicting aliases.
    """

    inspection = inspect_tool_call_format(payload)
    if inspection.unconsumed_fields:
        raise ValueError(
            "tool call contains unconsumed fields: "
            + ", ".join(inspection.unconsumed_fields)
        )


def normalize_tool_call_format(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Map a validated common representation to the sole internal format.

    This function only changes representation.  It does not choose a tool,
    add or rewrite arguments, judge evidence, trigger retries, or handle final
    answer text.
    """

    inspection = inspect_tool_call_format(payload)
    if inspection.unconsumed_fields:
        raise ValueError(
            "tool call contains unconsumed fields: "
            + ", ".join(inspection.unconsumed_fields)
        )
    return inspection.canonical


def canonicalize_tool_call(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Protocol entry: validate lossless conversion, then normalize format."""

    _validate_convertible_tool_call_format(payload)
    result = normalize_tool_call_format(payload)
    if not result["name"]:
        raise ValueError("tool call name is empty")
    return result


__all__ = [
    "TOOL_CALL_CONTRACT",
    "ToolCallFormatInspection",
    "canonicalize_tool_call",
    "inspect_tool_call_format",
    "normalize_tool_call_format",
]
