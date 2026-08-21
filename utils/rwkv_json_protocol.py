"""Small, lossless normalizer for RWKV structured-output envelopes.

This module only removes transport framing around one JSON object.  It never
repairs JSON content, chooses a tool, renames fields, inserts arguments, or
touches final-answer text.  Ambiguous output remains a protocol error so RWKV
can receive the existing same-temperature correction request.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

from utils.model_events import visible_model_text


_ASSISTANT_PREFIX = re.compile(r"^Assistant:\s*", flags=re.IGNORECASE)
_FENCE_PREFIX = re.compile(r"^```(?:json)?(?:\s*\r?\n|\s+)", flags=re.IGNORECASE)
_FENCE_SUFFIX = re.compile(r"\s*```\s*$")


@dataclass(frozen=True)
class NormalizedJSONObject:
    """One decoded object plus auditable transport-format metadata."""

    payload: dict[str, Any]
    input_format: str
    normalized: bool


def _decode_exact_json(text: str) -> Any:
    """Decode one JSON value and reject any second value or prose."""

    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(text)
    if text[end:].strip():
        raise ValueError("structured output contains trailing text or another value")
    return value


def normalize_json_object_envelope(
    value: Any,
    *,
    allow_singleton_array: bool = False,
) -> NormalizedJSONObject:
    """Decode one object from a small set of unambiguous RWKV envelopes.

    Accepted shapes are deliberately limited to:

    * a complete JSON object;
    * the object tail produced when ``{`` was prefilled by the request;
    * either shape inside one optional ``Assistant:`` / JSON-fence wrapper;
    * optionally, a one-element array containing exactly one object.

    No prose scanning or content repair is performed.
    """

    text = visible_model_text(value).strip()
    if not text:
        raise ValueError("structured output is empty")

    labels: list[str] = []
    assistant_match = _ASSISTANT_PREFIX.match(text)
    if assistant_match:
        labels.append("assistant_prefix")
        text = text[assistant_match.end() :].lstrip()

    fence_match = _FENCE_PREFIX.match(text)
    if fence_match:
        labels.append("json_fence")
        text = text[fence_match.end() :].strip()
        text = _FENCE_SUFFIX.sub("", text).strip()
    elif _FENCE_SUFFIX.search(text):
        # The opening fence can be part of the request prefill while the model
        # still emits a closing fence.  Removing that one suffix is transport
        # normalization, not semantic recovery.
        labels.append("closing_fence")
        text = _FENCE_SUFFIX.sub("", text).strip()

    prefilled = False
    try:
        decoded = _decode_exact_json(text)
    except (TypeError, json.JSONDecodeError, ValueError) as first_error:
        if not text.startswith('"'):
            raise ValueError(f"structured output is not one complete JSON value: {first_error}") from first_error
        try:
            decoded = _decode_exact_json("{" + text)
            prefilled = True
            labels.append("prefilled_object_tail")
        except (TypeError, json.JSONDecodeError, ValueError) as second_error:
            raise ValueError(
                f"structured output is not one complete JSON object: {second_error}"
            ) from second_error

    if isinstance(decoded, list):
        if not allow_singleton_array or len(decoded) != 1 or not isinstance(decoded[0], Mapping):
            raise ValueError("structured output must be one JSON object")
        decoded = decoded[0]
        labels.append("singleton_object_array")

    if not isinstance(decoded, Mapping):
        raise ValueError("structured output must be one JSON object")

    input_format = "+".join(labels) if labels else "json_object"
    return NormalizedJSONObject(
        payload=dict(decoded),
        input_format=input_format,
        normalized=bool(labels or prefilled),
    )


__all__ = ["NormalizedJSONObject", "normalize_json_object_envelope"]
