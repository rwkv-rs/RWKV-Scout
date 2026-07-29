"""Selection of the retrieval execution policy.

The first RWKV call decomposes a goal into atomic points.  This module keeps
the policy decision deterministic and separate from tool selection: RWKV does
not need to emit a second ``strategy`` field, and the retrieval runners do not
need to know how the policy was selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


SINGLE_LOOP = "single_loop"
FORK = "fork"


@dataclass(frozen=True)
class StrategyDecision:
    """The execution policy selected for one retrieval episode."""

    strategy: str
    point_count: int
    point_ids: tuple[str, ...]
    source: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["point_ids"] = list(self.point_ids)
        return value


def atomic_points(task_plan: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return valid atomic task points without trusting model metadata."""

    if not isinstance(task_plan, Mapping):
        return []
    points = task_plan.get("atomic_points")
    if not isinstance(points, list):
        return []
    return [
        point
        for point in points
        if isinstance(point, Mapping)
        and str(point.get("id") or "").strip()
        and str(point.get("task") or point.get("objective") or "").strip()
    ]


def _normalize_override(value: Any) -> str | None:
    normalized = str(value or "").strip().casefold().replace("-", "_")
    if normalized in {"single", "single_loop", "singleloop"}:
        return SINGLE_LOOP
    if normalized in {"fork", "forked", "fork_loop", "parallel"}:
        return FORK
    return None


def select_strategy(
    task_plan: Mapping[str, Any] | None,
    run_metadata: Mapping[str, Any] | None = None,
) -> StrategyDecision:
    """Select Single-loop or Fork from the model-generated point list.

    Production routing is based on the number of independently verifiable
    atomic points: one or two use one continuous context, three or more use
    workflow-level Fork.  Explicit overrides remain available for controlled
    architecture comparisons and backwards-compatible ``retrieval_fork``
    cases; normal requests do not need to provide either field.
    """

    metadata = run_metadata if isinstance(run_metadata, Mapping) else {}
    points = atomic_points(task_plan)
    point_ids = tuple(str(point.get("id") or "") for point in points)

    explicit = _normalize_override(metadata.get("retrieval_strategy"))
    if explicit:
        return StrategyDecision(
            strategy=explicit,
            point_count=len(points),
            point_ids=point_ids,
            source="retrieval_strategy_override",
            reason=f"explicit retrieval_strategy={explicit}",
        )

    # Keep the old experiment input working, but only when it is explicitly
    # present.  The old default of True would make every normal request Fork.
    if "retrieval_fork" in metadata and metadata.get("retrieval_fork") is not None:
        strategy = FORK if bool(metadata.get("retrieval_fork")) else SINGLE_LOOP
        return StrategyDecision(
            strategy=strategy,
            point_count=len(points),
            point_ids=point_ids,
            source="legacy_retrieval_fork_override",
            reason=f"explicit retrieval_fork={bool(metadata.get('retrieval_fork'))}",
        )

    if len(points) >= 3:
        return StrategyDecision(
            strategy=FORK,
            point_count=len(points),
            point_ids=point_ids,
            source="atomic_point_count",
            reason="three or more independently verifiable task points",
        )

    return StrategyDecision(
        strategy=SINGLE_LOOP,
        point_count=len(points),
        point_ids=point_ids,
        source="atomic_point_count",
        reason="one or two task points, or a conservative fallback for an invalid plan",
    )


__all__ = ["FORK", "SINGLE_LOOP", "StrategyDecision", "atomic_points", "select_strategy"]
