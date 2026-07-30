"""Execution-policy boundary for retrieval episodes.

The runner objects deliberately contain no tool policy.  They delegate to the
controller's already-tested execution implementations while keeping the
selection point independent from the large orchestration lifecycle.  The
implementations can be moved out of ``Orchestrator`` later without changing
the public runner contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.execution_strategy import FORK, SINGLE_LOOP, StrategyDecision


class RetrievalController(Protocol):
    def _run_single_loop(
        self,
        user_query: str,
        model_profile: dict[str, Any],
        task_plan: dict[str, Any],
        max_steps: int,
    ) -> str: ...

    def _run_forked_retrieval(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        model_profile: dict[str, Any],
        max_steps: int,
    ) -> str: ...


class RetrievalRunner(Protocol):
    def run(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        model_profile: dict[str, Any],
        max_steps: int,
    ) -> str: ...


@dataclass
class SingleLoopRunner:
    controller: RetrievalController

    def run(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        model_profile: dict[str, Any],
        max_steps: int,
    ) -> str:
        return self.controller._run_single_loop(
            user_query,
            model_profile,
            task_plan,
            max_steps,
        )


@dataclass
class LegacyForkRunner:
    """Compatibility runner for explicit pre-global architecture experiments."""

    controller: RetrievalController

    def run(
        self,
        user_query: str,
        task_plan: dict[str, Any],
        model_profile: dict[str, Any],
        max_steps: int,
    ) -> str:
        return self.controller._run_forked_retrieval(
            user_query,
            task_plan,
            model_profile,
            max_steps,
        )


def build_runner(decision: StrategyDecision, controller: RetrievalController) -> RetrievalRunner:
    if decision.strategy == FORK:
        return LegacyForkRunner(controller)
    if decision.strategy == SINGLE_LOOP:
        return SingleLoopRunner(controller)
    raise ValueError(f"unsupported retrieval strategy: {decision.strategy}")


# ``ForkRunner`` remains an import-compatible alias for archived comparison
# scripts; production routing never selects it implicitly.
ForkRunner = LegacyForkRunner

__all__ = ["ForkRunner", "LegacyForkRunner", "RetrievalRunner", "SingleLoopRunner", "build_runner"]
