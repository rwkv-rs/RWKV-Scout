"""Canonical prompt and stop contracts for the RWKV completion runtime.

The local server is an OpenAI-compatible transport, but the checkpoint is
trained as a plain transcript continuation model.  Keeping these strings in
one module prevents the planner, final synthesizer, and probes from drifting
apart.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


USER_HEADER = "User:"
ASSISTANT_HEADER = "Assistant:"
SYSTEM_HEADER = "System:"
FLOWER_USER_HEADER = "User✿"
FLOWER_ASSISTANT_HEADER = "Bot✿"
FLOWER_DELIMITER = "✿"

# This is the stop contract used by rwkv-skills for JSON calls.  Do not stop
# on ``<think>``: the checkpoint may emit a short reasoning block before the
# actual JSON object, and the caller strips it before parsing.
JSON_CALL_STOP_SUFFIXES = (
    "\n```",
    "```",
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
    "\nUser✿",
    "User✿",
    "\nBot✿",
    "Bot✿",
    "✿",
)

# Final synthesis is prose continuation, not a JSON/tool protocol.  Markdown
# fences are deliberately absent: a valid answer may contain a fenced table
# or code block.  Role boundaries are the cutoff markers that otherwise cause
# the model to continue writing a second transcript turn.
FINAL_CONTINUATION_STOP_SUFFIXES = (
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
    "\nUser✿",
    "User✿",
    "\nBot✿",
    "Bot✿",
)


def assistant_json_prefix(*, enable_think: bool = True, prefill_object: bool = False) -> str:
    prefix = "Assistant:"
    if enable_think:
        prefix += " <think></think>"
    prefix += "\n```json\n"
    return prefix + ("{" if prefill_object else "")


def render_final_continuation_prompt(user_prompt: str, *, enable_think: bool = True) -> str:
    """Render the exact official User/Assistant prose continuation prefix."""

    body = _sanitize_embedded_role_headers(str(user_prompt or "").strip())
    think = " <think></think>" if enable_think else ""
    return f"User: {body}\nAssistant:{think}\n"


def build_final_continuation_prompt(user_prompt: str) -> str:
    """Compatibility alias used by probes and the final answer path."""

    return render_final_continuation_prompt(user_prompt, enable_think=True)


def render_tool_transcript(messages: Sequence[Mapping[str, Any]] | None) -> str:
    """Render the official JSON-call transcript used by the planner."""

    parts: list[str] = []
    for message in messages or ():
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "user").strip().casefold()
        raw_content = message.get("content")
        content = raw_content if isinstance(raw_content, Mapping) else str(raw_content or "")
        if role == "system":
            parts.append(f"{SYSTEM_HEADER} {content}".rstrip())
        elif role == "assistant":
            payload = content if isinstance(content, Mapping) else _json_content(content)
            parts.append(
                f"{ASSISTANT_HEADER} <think></think>\n```json\n"
                f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n```"
            )
        else:
            parts.append(f"{USER_HEADER} {_sanitize_embedded_role_headers(str(content))}".rstrip())
    parts.append(assistant_json_prefix(enable_think=True))
    return "\n\n".join(parts)


def clean_final_continuation(text: str) -> str:
    """Remove transcript artifacts while preserving the model's prose.

    This is protocol cleanup only.  It does not add facts, citations, refusal
    text, or a controller-written answer.
    """

    value = str(text or "").replace("\r\n", "\n").strip()
    value = re.sub(r"^\s*(?:Assistant:|Bot✿)\s*", "", value, count=1, flags=re.IGNORECASE)
    if re.match(r"^\s*<think>", value, flags=re.IGNORECASE) and not re.search(
        r"</think>", value, flags=re.IGNORECASE
    ):
        return ""
    value = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", value, count=1, flags=re.IGNORECASE)
    value = value.replace("</think>", "").strip()
    boundary = re.search(
        r"(?im)^\s*(?:User:|System:|Assistant:|User✿|Bot✿)\s*",
        value,
    )
    if boundary:
        value = value[: boundary.start()].rstrip()
    # A model that ignored the prose contract can still emit a tool object.
    # Remove only a leading tool-call wrapper; ordinary JSON in an answer is
    # otherwise left untouched because the final model owns the wording.
    if re.match(r"^\s*```json\s*\{\s*[\"'](?:name|tool_name|action)[\"']\s*:", value, re.I):
        value = re.sub(r"^\s*```json\s*|\s*```\s*$", "", value, flags=re.I).strip()
    return value


def _json_content(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"name": "", "arguments": {}}
    return parsed if isinstance(parsed, dict) else {"name": "", "arguments": {}}


def _sanitize_embedded_role_headers(value: str) -> str:
    text = str(value or "")
    # Evidence and execution records are data.  Role-looking lines inside
    # them must not become a new turn in the RWKV transcript.
    return re.sub(
        r"(?im)^\s*(User:|System:|Assistant:|User✿|Bot✿)",
        lambda match: f"[embedded {match.group(1)}]",
        text,
    )
