"""Stable identities for every object exchanged by the RWKV-ECRA runtime.

Contract names describe domain meaning, not an experiment round or an
implementation revision.  Producers emit ``contract`` and consumers compare
against this registry.  Historical ``schema_version`` values belong only in
explicit migration/admission adapters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RUNTIME_CONTRACT_NAMESPACE = "rwkv.ecra.runtime"

TASK_PLAN_CONTRACT = f"{RUNTIME_CONTRACT_NAMESPACE}.task-plan"
TOOL_CALL_CONTRACT = f"{RUNTIME_CONTRACT_NAMESPACE}.tool-call"
RETRIEVAL_OBJECT_CONTRACT = f"{RUNTIME_CONTRACT_NAMESPACE}.retrieval-object"
RETRIEVAL_QUERY_PLAN_CONTRACT = (
    f"{RUNTIME_CONTRACT_NAMESPACE}.retrieval-query-plan"
)
EVIDENCE_LEDGER_CONTRACT = f"{RUNTIME_CONTRACT_NAMESPACE}.evidence-ledger"
EVIDENCE_RECORD_SET_CONTRACT = (
    f"{RUNTIME_CONTRACT_NAMESPACE}.evidence-record-set"
)
EVIDENCE_RESOLUTION_CONTRACT = (
    f"{RUNTIME_CONTRACT_NAMESPACE}.evidence-resolution"
)
EVIDENCE_REVIEW_CONTRACT = f"{RUNTIME_CONTRACT_NAMESPACE}.evidence-review"
PLANNER_EVIDENCE_CONTRACT = f"{RUNTIME_CONTRACT_NAMESPACE}.planner-evidence"
RETRIEVAL_INFRASTRUCTURE_CONTRACT = (
    f"{RUNTIME_CONTRACT_NAMESPACE}.retrieval-infrastructure"
)
RETRIEVAL_ROUTING_CONTRACT = (
    f"{RUNTIME_CONTRACT_NAMESPACE}.retrieval-routing-state"
)
MODEL_EXTRACTION_DIAGNOSTICS_CONTRACT = (
    f"{RUNTIME_CONTRACT_NAMESPACE}.model-extraction-diagnostics"
)
RETRIEVAL_EVENT_LEDGER_CONTRACT = (
    f"{RUNTIME_CONTRACT_NAMESPACE}.retrieval-event-ledger"
)


def runtime_contract(payload: Mapping[str, Any] | None) -> str:
    """Return only the formal runtime contract identity from ``payload``."""

    row = payload if isinstance(payload, Mapping) else {}
    return str(row.get("contract") or "").strip()


def require_runtime_contract(
    payload: Mapping[str, Any],
    expected: str,
) -> None:
    """Reject ambiguous or incorrectly typed runtime payloads.

    ``schema_version`` is never silently treated as a current contract because
    that would let a historical object cross a live stage boundary without
    going through its migration adapter.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("runtime contract payload must be a JSON object")
    if "schema_version" in payload:
        raise ValueError("legacy schema_version is not valid at a runtime boundary")
    supplied = runtime_contract(payload)
    if supplied != expected:
        raise ValueError(f"expected runtime contract {expected!r}, got {supplied!r}")


__all__ = [
    "EVIDENCE_LEDGER_CONTRACT",
    "EVIDENCE_RECORD_SET_CONTRACT",
    "EVIDENCE_RESOLUTION_CONTRACT",
    "EVIDENCE_REVIEW_CONTRACT",
    "MODEL_EXTRACTION_DIAGNOSTICS_CONTRACT",
    "PLANNER_EVIDENCE_CONTRACT",
    "RETRIEVAL_EVENT_LEDGER_CONTRACT",
    "RETRIEVAL_INFRASTRUCTURE_CONTRACT",
    "RETRIEVAL_OBJECT_CONTRACT",
    "RETRIEVAL_QUERY_PLAN_CONTRACT",
    "RETRIEVAL_ROUTING_CONTRACT",
    "RUNTIME_CONTRACT_NAMESPACE",
    "TASK_PLAN_CONTRACT",
    "TOOL_CALL_CONTRACT",
    "require_runtime_contract",
    "runtime_contract",
]
