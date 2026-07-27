"""Provider plugins and runtime environment state for retrieval.

This module deliberately performs no startup network checks and never assumes
that a provider needs an API key. A plugin becomes ``ready`` or ``unavailable``
only after the application observes an actual execution result.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievalPlugin:
    name: str
    label: str = ""
    capabilities: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    requires_credentials: bool = False


@dataclass
class RetrievalState:
    plugin: str
    status: str = "unknown"
    last_error: str = ""
    last_action: str = ""
    last_checked: str = ""
    executions: int = 0
    failures: int = 0


class PluginRegistry:
    """In-process registry for provider capabilities and observed health."""

    _plugins: dict[str, RetrievalPlugin] = {}
    _states: dict[str, RetrievalState] = {}
    _lock = threading.RLock()

    @classmethod
    def register(
        cls,
        name: str,
        *,
        label: str = "",
        capabilities: Iterable[str] = (),
        tools: Iterable[str] = (),
        requires_credentials: bool = False,
    ) -> RetrievalPlugin:
        plugin = RetrievalPlugin(
            name=str(name),
            label=str(label or name),
            capabilities=tuple(str(item) for item in capabilities),
            tools=tuple(str(item) for item in tools),
            requires_credentials=bool(requires_credentials),
        )
        with cls._lock:
            previous = cls._plugins.get(plugin.name)
            if previous is not None:
                merged_label = previous.label if plugin.label == plugin.name and previous.label else plugin.label
                plugin = RetrievalPlugin(
                    name=plugin.name,
                    label=merged_label or previous.label,
                    capabilities=tuple(dict.fromkeys((*previous.capabilities, *plugin.capabilities))),
                    tools=tuple(dict.fromkeys((*previous.tools, *plugin.tools))),
                    requires_credentials=previous.requires_credentials or plugin.requires_credentials,
                )
            cls._plugins[plugin.name] = plugin
            cls._states.setdefault(plugin.name, RetrievalState(plugin=plugin.name))
        return plugin

    @classmethod
    def get(cls, name: str) -> RetrievalPlugin | None:
        with cls._lock:
            return cls._plugins.get(str(name or ""))

    @classmethod
    def observe(cls, plugin: str, *, action: str, result: Any) -> None:
        """Update health from a tool result; no provider-specific assumptions."""

        plugin_name = str(plugin or "").strip()
        if not plugin_name:
            return
        status = "ok"
        error = ""
        if isinstance(result, dict):
            status = str(result.get("status") or "ok").casefold()
            errors = result.get("provider_errors") or result.get("errors") or []
            if isinstance(errors, list) and errors:
                error = str(errors[0])[:500]
            elif errors:
                error = str(errors)[:500]
        elif isinstance(result, str) and result.strip():
            status = "ok"

        now = datetime.now().isoformat(timespec="seconds")
        with cls._lock:
            state = cls._states.setdefault(plugin_name, RetrievalState(plugin=plugin_name))
            state.last_action = str(action or "")
            state.last_checked = now
            state.executions += 1
            if status in {"error", "failed", "unavailable", "unauthorized"}:
                state.status = "unavailable"
                state.last_error = error or status
                state.failures += 1
            elif status in {"no_results", "empty"}:
                state.status = "degraded"
                state.last_error = ""
            else:
                state.status = "ready"
                state.last_error = ""

    @classmethod
    def snapshot(cls) -> list[dict[str, Any]]:
        with cls._lock:
            rows = []
            for name, plugin in cls._plugins.items():
                state = cls._states.setdefault(name, RetrievalState(plugin=name))
                rows.append(
                    {
                        "plugin": name,
                        "label": plugin.label,
                        "capabilities": list(plugin.capabilities),
                        "status": state.status,
                        "last_error": state.last_error,
                        "last_action": state.last_action,
                        "last_checked": state.last_checked,
                        "executions": state.executions,
                        "failures": state.failures,
                    }
                )
            return rows

    @classmethod
    def catalog(cls) -> list[dict[str, Any]]:
        with cls._lock:
            return [
                {
                    "plugin": plugin.name,
                    "label": plugin.label,
                    "capabilities": list(plugin.capabilities),
                    "tools": list(plugin.tools),
                    "requires_credentials": plugin.requires_credentials,
                }
                for plugin in cls._plugins.values()
            ]


def plugin_environment_snapshot() -> dict[str, Any]:
    """Return a safe, serializable environment view for planner observations."""

    return {"plugins": PluginRegistry.snapshot()}
