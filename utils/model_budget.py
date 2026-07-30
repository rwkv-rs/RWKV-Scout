"""Context-safe completion budgets for OpenAI-compatible RWKV endpoints."""

from __future__ import annotations

from utils.chunker import get_token_count


def bounded_completion_budget(
    prompt: str,
    *,
    context_limit: int,
    requested_max: int,
    safety_margin: int = 256,
) -> int:
    """Return a completion budget that never exceeds the remaining context.

    The endpoint counts input and requested output together.  A fixed minimum
    output budget can therefore turn an otherwise valid prompt into a 400
    request when only a few hundred tokens remain.  The caller is still
    responsible for trimming prompts whose input alone exceeds the model
    limit; this helper prevents the more common input-plus-output overflow.
    """

    limit = max(1, int(context_limit))
    requested = max(1, int(requested_max))
    margin = max(0, int(safety_margin))
    remaining = limit - get_token_count(prompt) - margin
    return max(1, min(requested, remaining))
