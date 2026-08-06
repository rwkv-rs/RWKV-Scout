"""Prompt and transcript contracts for the RWKV completion runtime.

The local G1i checkpoints are most stable when the tool boundary mirrors the
RWKV-native transcript used by ``rwkv-skills``::

    ### User
    <instructions, task, or observation>
    ### Assistant
    **Tool Call:**
    ```json
    {"name":"tool_name","arguments":{...}}
    ```

The controller owns ``### Tool Output``.  A model completion is never allowed
to turn its own continuation into evidence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


USER_HEADER = "### User"
ASSISTANT_HEADER = "### Assistant"
# Kept as a compatibility name for callers that imported the old constant.
# It deliberately renders as a user block; the local contract has no separate
# ``System:`` turn.
SYSTEM_HEADER = USER_HEADER
TOOL_CALL_HEADER = "**Tool Call:**"
TOOL_OUTPUT_HEADER = "### Tool Output"

# Legacy flower-delimiter names remain import-compatible for old traces.  They
# are not emitted by the current production renderer.
FLOWER_USER_HEADER = "User✿"
FLOWER_ASSISTANT_HEADER = "Bot✿"
FLOWER_DELIMITER = "✿"

# Stop at transcript boundaries, especially before a model-written Tool Output.
# Do not include a bare ````` `` stop: with the native prefix the completion
# begins with `````json`` and that would terminate the response immediately.
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

FINAL_CONTINUATION_STOP_SUFFIXES = (
    "\n### User",
    "### User",
    "\n### Assistant",
    "### Assistant",
    "\n### Tool Output",
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
    "\nUser✿",
    "User✿",
    "\nBot✿",
    "Bot✿",
)


def assistant_prose_prefix(*, enable_think: bool = False) -> str:
    """Return the native assistant continuation marker.

    ``enable_think`` is retained for compatibility with archived probes.  The
    production paths pass ``False`` so tool and final-answer generations do
    not enter an incomplete ``<think>`` continuation.
    """

    if enable_think:
        return f"{ASSISTANT_HEADER}\n<think></think"
    return ASSISTANT_HEADER


def assistant_json_prefix(*, enable_think: bool = False, prefill_object: bool = True) -> str:
    """Return the generic JSON continuation used by task-plan calls."""

    if enable_think:
        prefix = f"{ASSISTANT_HEADER}\n<think></think\n"
    else:
        prefix = f"{ASSISTANT_HEADER}\n```json\n"
    return prefix + ("{" if prefill_object else "")


def tool_call_prefix() -> str:
    """Return the exact G1i native tool-call continuation marker."""

    return f"{ASSISTANT_HEADER}\n{TOOL_CALL_HEADER}\n"


def render_final_continuation_prompt(user_prompt: str, *, enable_think: bool = False) -> str:
    """Render a single-user, no-CoT prose continuation."""

    body = _sanitize_embedded_role_headers(str(user_prompt or "").strip())
    return f"{USER_HEADER}\n{body}\n{assistant_prose_prefix(enable_think=enable_think)}"


def build_final_continuation_prompt(user_prompt: str) -> str:
    """Build the production final-answer continuation prompt."""

    return render_final_continuation_prompt(user_prompt, enable_think=False)


def render_tool_transcript(
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    json_output: bool = True,
) -> str:
    """Render the model-visible transcript using the native G1i boundaries.

    System instructions are folded into ``### User`` as required by the local
    contract.  Tool results are emitted only from controller-owned ``tool``
    messages as ``### Tool Output``; an assistant continuation cannot create a
    trusted tool-result turn.
    """

    parts: list[str] = []
    for message in messages or ():
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "user").strip().casefold()
        raw_content = message.get("content")
        content = raw_content if isinstance(raw_content, Mapping) else str(raw_content or "")

        if role in {"system", "user"}:
            parts.append(f"{USER_HEADER}\n{_sanitize_embedded_role_headers(str(content))}".rstrip())
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                for tool_call in tool_calls:
                    payload = _tool_call_payload(tool_call)
                    if payload is not None:
                        parts.append(_render_tool_call(payload))
                if str(content).strip():
                    parts.append(f"{ASSISTANT_HEADER}\n{_sanitize_embedded_role_headers(str(content))}".rstrip())
                continue
            payload = _tool_call_payload(content)
            if payload is not None:
                parts.append(_render_tool_call(payload))
            else:
                parts.append(f"{ASSISTANT_HEADER}\n{_sanitize_embedded_role_headers(str(content))}".rstrip())
            continue

        if role in {"tool", "function", "observation"}:
            parts.append(_render_tool_output(content))
            continue

        parts.append(f"{USER_HEADER}\n{_sanitize_embedded_role_headers(str(content))}".rstrip())

    parts.append(tool_call_prefix() if json_output else assistant_prose_prefix(enable_think=False))
    return "\n\n".join(parts)


def clean_final_continuation(text: str) -> str:
    """Remove transcript artifacts without adding or changing factual text."""

    value = str(text or "").replace("\r\n", "\n").strip()
    value = re.sub(
        r"^\s*(?:###\s*Assistant:|###\s*Assistant|Assistant:|Bot✿)\s*",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    )
    # Compatibility with archived probes that ended at ``</think``.
    value = re.sub(r"^\s*>\s*", "", value, count=1)
    if re.match(r"^\s*<think>", value, flags=re.IGNORECASE) and not re.search(
        r"</think>", value, flags=re.IGNORECASE
    ):
        return ""
    value = re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", value, count=1, flags=re.IGNORECASE)
    value = value.replace("</think>", "").strip()
    boundary = re.search(
        r"(?im)^\s*(?:###\s*(?:User|System|Assistant|Tool Output)|User:|System:|Assistant:|User✿|Bot✿)\s*",
        value,
    )
    if boundary:
        value = value[: boundary.start()].rstrip()
    return value


def _tool_call_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        if isinstance(value, str) and _looks_like_json_text(value):
            value = _json_content(value)
        else:
            return None
    name = value.get("name") or value.get("tool_name") or value.get("action") or value.get("tool")
    arguments = value.get("arguments")
    if not isinstance(name, str) or not name.strip() or not isinstance(arguments, (Mapping, str, type(None))):
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
        rendered = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None else content
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
