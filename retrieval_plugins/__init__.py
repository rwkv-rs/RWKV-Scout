"""Extensible retrieval-provider runtime.

Provider modules register capabilities here without requiring the agent
planner or orchestrator to know provider-specific names.
"""

from .runtime import (
    PluginRegistry,
    RetrievalPlugin,
    RetrievalState,
    plugin_environment_snapshot,
)
from .formats import (
    DISCOVERY_ROLE,
    EVIDENCE_ROLE,
    SCHEMA_VERSION,
    error_result,
    is_error,
    normalize_result,
)

__all__ = [
    "PluginRegistry",
    "RetrievalPlugin",
    "RetrievalState",
    "plugin_environment_snapshot",
    "SCHEMA_VERSION",
    "DISCOVERY_ROLE",
    "EVIDENCE_ROLE",
    "normalize_result",
    "error_result",
    "is_error",
]
