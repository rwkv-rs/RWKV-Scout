"""RWKV prompt rendering kept outside the HTTP and agent layers."""

from __future__ import annotations

import json
from typing import Any, Sequence

from utils.model_events import visible_model_text


def render_rwkv_transcript(
    messages: Sequence[dict[str, Any]] | None,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Render the plain User/Assistant contract used by RWKV checkpoints."""

    rows: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").casefold()
        label = {
            "system": "System",
            "assistant": "Assistant",
            "tool": "Tool",
        }.get(role, "User")
        rows.append(f"{label}:\n{visible_model_text(message.get('content') or '')}")

    transcript = "\n\n".join(rows)
    if tools:
        transcript += (
            "\n\nSystem:\n"
            "Select a tool only when the request requires it. Return one JSON object "
            "with keys name and arguments, or answer directly. Available tools:\n"
            f"{json.dumps(tools, ensure_ascii=False, separators=(',', ':'))}"
        )
    return transcript + "\n\nAssistant:"
