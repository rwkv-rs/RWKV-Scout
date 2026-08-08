"""Prompt and transcript contracts for the RWKV completion runtime.

This module renders model input only.  It intentionally contains no helper
that cleans, truncates, repairs, or otherwise transforms a final answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


USER_HEADER = "### User"
ASSISTANT_HEADER = "### Assistant"
SYSTEM_HEADER = USER_HEADER
TOOL_CALL_HEADER = "**Tool Call:**"
TOOL_OUTPUT_HEADER = "### Tool Output"

# Compatibility names used by archived tool-call traces.  They are not
# emitted by the production renderer.
FLOWER_USER_HEADER = "User✿"
FLOWER_ASSISTANT_HEADER = "Bot✿"
FLOWER_DELIMITER = "✿"

# Tool calls are a JSON protocol, not a public answer.  Stops are retained for
# that protocol so a call cannot consume a controller-owned Tool Output turn.
JSON_CALL_STOP_SUFFIXES = (
    "\n### Tool Output",
    "### Tool Output",
    "\n### User",
    "### User",
    "\n### Assistant",
    "### Assistant",
    "\n**Tool Call:**",
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
    "\nUser✿",
    "User✿",
    "\nBot✿",
    "Bot✿",
)


def assistant_prose_prefix(*, enable_think: bool = False) -> str:
    """Return the native assistant continuation marker."""

    if enable_think:
        return f"{ASSISTANT_HEADER}\n<think></think"
    return ASSISTANT_HEADER


def assistant_json_prefix(*, enable_think: bool = False, prefill_object: bool = True) -> str:
    """Return the generic JSON continuation used by planning calls."""

    if enable_think:
        prefix = f"{ASSISTANT_HEADER}\n<think></think\n"
    else:
        prefix = f"{ASSISTANT_HEADER}\n```json\n"
    return prefix + ("{" if prefill_object else "")


def tool_call_prefix() -> str:
    """Return the native G1i tool-call continuation marker."""

    return f"{ASSISTANT_HEADER}\n{TOOL_CALL_HEADER}\n"


def render_final_continuation_prompt(user_prompt: str, *, enable_think: bool = False) -> str:
    """Render one user input followed by an assistant prose continuation."""

    body = _sanitize_embedded_role_headers(str(user_prompt or "").strip())
    return f"{USER_HEADER}\n{body}\n{assistant_prose_prefix(enable_think=enable_think)}"


def build_final_continuation_prompt(user_prompt: str) -> str:
    """Build the final-answer prompt with G1i's empty-think prefill.

    G1i completes the missing final ``>`` before emitting user-facing prose.
    """

    body = _sanitize_embedded_role_headers(str(user_prompt or "").strip())
    return f"User: {body}\nAssistant: <think></think"


def consume_final_prefill_boundary(value: Any) -> str:
    """Consume only the token that completes the empty-think prefill.

    The raw model event retains the untouched continuation.  This decoder
    removes no model prose: the first ``>`` belongs to the prompt delimiter,
    and at most one immediately following line break is framing.
    """

    text = "" if value is None else str(value)
    if not text.startswith(">"):
        return text
    text = text[1:]
    if text.startswith("\r\n"):
        return text[2:]
    if text.startswith("\n"):
        return text[1:]
    return text


def render_tool_transcript(
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    json_output: bool = True,
) -> str:
    """Render a model-visible transcript for structured tool decisions."""

    parts: list[str] = []
    for message in messages or ():
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "user").strip().casefold()
        raw_content = message.get("content")
        content = raw_content if isinstance(raw_content, Mapping) else str(raw_content or "")

        if role in {"system", "user"}:
            parts.append(
                f"{USER_HEADER}\n{_sanitize_embedded_role_headers(str(content))}".rstrip()
            )
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                for tool_call in tool_calls:
                    payload = _tool_call_payload(tool_call)
                    if payload is not None:
                        parts.append(_render_tool_call(payload))
                if str(content).strip():
                    parts.append(
                        f"{ASSISTANT_HEADER}\n{_sanitize_embedded_role_headers(str(content))}".rstrip()
                    )
                continue
            payload = _tool_call_payload(content)
            if payload is not None:
                parts.append(_render_tool_call(payload))
            else:
                parts.append(
                    f"{ASSISTANT_HEADER}\n{_sanitize_embedded_role_headers(str(content))}".rstrip()
                )
            continue

        if role in {"tool", "function", "observation"}:
            parts.append(_render_tool_output(content))
            continue

        parts.append(
            f"{USER_HEADER}\n{_sanitize_embedded_role_headers(str(content))}".rstrip()
        )

    parts.append(
        tool_call_prefix()
        if json_output
        else assistant_prose_prefix(enable_think=False)
    )
    return "\n\n".join(parts)


def _tool_call_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        if isinstance(value, str) and _looks_like_json_text(value):
            value = _json_content(value)
        else:
            return None
    if not isinstance(value, Mapping):
        return None
    name = value.get("name") or value.get("tool_name") or value.get("action") or value.get("tool")
    arguments = value.get("arguments")
    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(arguments, (Mapping, str, type(None)))
    ):
        return None
    if isinstance(arguments, str):
        arguments = _json_content(arguments)
    if not isinstance(arguments, Mapping):
        arguments = {}
    return {"name": name.strip(), "arguments": dict(arguments)}


def _render_tool_call(payload: Mapping[str, Any]) -> str:
    return (
        f"{ASSISTANT_HEADER}\n{TOOL_CALL_HEADER}\n```json\n"
        f"{json.dumps(dict(payload), ensure_ascii=False, indent=2)}\n```"
    )


def _render_tool_output(content: Any) -> str:
    if isinstance(content, Mapping):
        rendered = json.dumps(dict(content), ensure_ascii=False, indent=2)
    elif isinstance(content, str):
        parsed = _json_content(content)
        rendered = (
            json.dumps(parsed, ensure_ascii=False, indent=2)
            if parsed is not None
            else content
        )
    else:
        rendered = json.dumps(content, ensure_ascii=False, indent=2)
    return f"{TOOL_OUTPUT_HEADER}\n```json\n{rendered}\n```"


def _json_content(value: str) -> dict[str, Any] | None:
    cleaned = str(value or "").strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _looks_like_json_text(value: str) -> bool:
    return str(value or "").lstrip().startswith(("{", "```json", "```"))


def _sanitize_embedded_role_headers(value: str) -> str:
    text = str(value or "")
    return re.sub(
        r"(?im)^\s*(###\s+(?:User|System|Assistant|Tool Output)|User:|System:|Assistant:|User✿|Bot✿)",
        lambda match: f"[embedded {match.group(1)}]",
        text,
    )
