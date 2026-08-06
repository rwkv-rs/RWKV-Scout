"""RWKV prompt rendering kept outside the HTTP and agent layers."""

from __future__ import annotations

import json
from typing import Any, Sequence

from utils.rwkv_prompt import render_tool_transcript


def render_rwkv_transcript(
    messages: Sequence[dict[str, Any]] | None,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Render the native G1i User/Assistant and Tool Call contract.

    Tool metadata is folded into a user block.  The renderer appends the
    explicit ``**Tool Call:**`` continuation marker and never emits a separate
    ``System:`` turn.
    """

    rendered_messages = [dict(message) for message in (messages or []) if isinstance(message, dict)]
    if tools:
        tool_content = (
            "Select a tool only when the request requires it. Continue after "
            "**Tool Call:** with one fenced JSON object containing name and arguments. "
            "Do not write Tool Output; the controller supplies it. Available tools:\n"
            f"{json.dumps(tools, ensure_ascii=False, separators=(',', ':'))}"
        )
        existing_system = next(
            (
                message
                for message in rendered_messages
                if str(message.get("role") or "").strip().casefold() == "system"
            ),
            None,
        )
        if existing_system is not None:
            existing_system["content"] = (
                f"{str(existing_system.get('content') or '').strip()}\n\n{tool_content}"
            ).strip()
        else:
            rendered_messages.insert(
                0,
                {"role": "system", "content": tool_content},
            )
    return render_tool_transcript(rendered_messages, json_output=bool(tools))
