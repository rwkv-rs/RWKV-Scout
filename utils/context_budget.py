"""Context budgets shared by routing, evidence packing, and synthesis.

The configured model context is one global resource.  Keeping the derived
budgets here prevents each stage from inventing a different character cap and
silently discarding a different part of the same retrieval episode.
"""

from __future__ import annotations


def context_limit(configured: int) -> int:
    return max(1024, int(configured or 1024))


def routing_observation_tokens(configured: int) -> int:
    """Reserve a bounded routing view while leaving room for history/output."""
    limit = context_limit(configured)
    return max(1024, min(3000, limit // 4))


def evidence_tokens(configured: int) -> int:
    """Return the evidence budget after prompt and completion headroom."""
    limit = context_limit(configured)
    # RWKV can handle a larger bounded evidence packet than the old fixed
    # 7.5K ceiling. Keep one shared packer, but scale the packet with the
    # actual model window instead of silently throwing away relevant chunks.
    # Reserve roughly 5K output tokens plus system/instruction margin for the
    # final writer.  Evidence is packed claim-first inside the remaining room.
    return max(2048, min(6000, limit - 7000))


def planner_prompt_tokens(configured: int) -> int:
    """Keep planner history below the endpoint's input-plus-output limit."""
    return max(2048, context_limit(configured) - 768)


def observation_chars(configured: int) -> int:
    # Routing observations are UTF-8 text, so this is deliberately a loose
    # character envelope; the final token budget is enforced by the tokenizer.
    return routing_observation_tokens(configured) * 3
