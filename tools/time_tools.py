"""Deterministic clock and timezone tools."""

from __future__ import annotations

import json
from datetime import datetime, timezone as dt_timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tools.registry import ToolRegistry


@ToolRegistry.register(
    name="current_time",
    phase="ALL",
    model_visible=True,
    category="computation",
    description="Return the current clock reading for an IANA timezone; use only for current time and never as a historical fact source.",
    signature="""[Tool] current_time
- Function: return the current clock reading for an IANA timezone.
- Parameters: timezone (for example UTC, Asia/Shanghai, America/New_York).
- It returns the system observation time; it does not answer historical date questions.""",
)
def current_time(timezone: str = "UTC", **_: Any) -> str:
    requested = str(timezone or "UTC").strip() or "UTC"
    try:
        zone = ZoneInfo(requested)
    except (ZoneInfoNotFoundError, ValueError):
        return json.dumps(
            {"status": "error", "tool": "current_time", "error_class": "invalid_timezone", "timezone": requested},
            ensure_ascii=False,
        )
    observed = datetime.now(dt_timezone.utc).astimezone(zone)
    return json.dumps(
        {
            "status": "ok",
            "tool": "current_time",
            "timezone": requested,
            "iso": observed.isoformat(timespec="seconds"),
            "date": observed.date().isoformat(),
            "utc_offset": observed.strftime("%z"),
            "observed_at_utc": datetime.now(dt_timezone.utc).isoformat(timespec="seconds"),
            "deterministic": True,
        },
        ensure_ascii=False,
    )


__all__ = ["current_time"]
