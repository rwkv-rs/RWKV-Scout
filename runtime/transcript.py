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
    """Render the online G1i System/User/function-output contract.

    Tool metadata lives in the System turn.  The renderer appends the exact
    ``Assistant: ```json`` continuation used by the online G1i service.
    """

    rendered_messages = [dict(message) for message in (messages or []) if isinstance(message, dict)]
    if tools:
        tool_content = (
            "Tools: "
            f"{json.dumps(tools, ensure_ascii=False, separators=(',', ':'))}\n"
            "Return only a JSON function call."
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
            existing_content = str(existing_system.get("content") or "").strip()
            existing_system["content"] = (
                f"{tool_content}\n\n{existing_content}"
            ).strip()
        else:
            rendered_messages.insert(
                0,
                {"role": "system", "content": tool_content},
            )
    return render_tool_transcript(rendered_messages, json_output=bool(tools))
