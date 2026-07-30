"""Validated single-variable strategy overrides for controlled experiments."""

from __future__ import annotations

from typing import Any


DEFAULT_STRATEGY = {
    "ranking_strategy": "evidence_quality.v1",
    "context_source_count": None,
    "prompt_variant": "default.v1",
}

RANKING_STRATEGIES = {
    "candidate_support_then_rank.v1",
    "best_rank.v1",
    "dedup_order.v1",
    "evidence_quality.v1",
}
PROMPT_VARIANTS = {"default.v1", "citation_first.v1", "compact_evidence.v1"}


def normalize_strategy(value: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate and normalize a strategy override before runtime execution."""

    raw = dict(value or {})
    unknown = sorted(set(raw) - set(DEFAULT_STRATEGY))
    if unknown:
        raise ValueError(f"unknown strategy keys: {', '.join(unknown)}")
    output = dict(DEFAULT_STRATEGY)
    output.update({key: raw[key] for key in raw})
    if output["ranking_strategy"] not in RANKING_STRATEGIES:
        raise ValueError(f"unsupported ranking_strategy: {output['ranking_strategy']}")
    if output["prompt_variant"] not in PROMPT_VARIANTS:
        raise ValueError(f"unsupported prompt_variant: {output['prompt_variant']}")
    count = output["context_source_count"]
    if count is not None:
        try:
            count = int(count)
        except (TypeError, ValueError) as exc:
            raise ValueError("context_source_count must be an integer or null") from exc
        if not 1 <= count <= 4:
            raise ValueError("context_source_count must be between 1 and 4")
        output["context_source_count"] = count
    return output


def load_strategy_file(path: str) -> dict[str, Any]:
    """Load a JSON strategy file and apply the same runtime validation."""

    import json
    from pathlib import Path

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strategy config: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("strategy config must be a JSON object")
    return normalize_strategy(value)
