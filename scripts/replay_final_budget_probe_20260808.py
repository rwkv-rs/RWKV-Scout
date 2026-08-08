"""Replay the exact saved final prompt that previously exceeded 16K context."""

from __future__ import annotations

import json
from pathlib import Path

from clients.llm_client import LLMClient
from config import DATA_PIPELINE, get_llm_context_length
from utils.chunker import get_token_count
from utils.model_budget import bounded_completion_budget


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "data/evaluation/full_272_think_prefill_20260808/original_random_40"
    / "original_random_40/part_03.json.result.json"
)
OUTPUT = ROOT / "data/evaluation/final_budget_exact_replay_20260808.json"


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    prompt = str(report["cases"][0]["trace"]["model_calls"][-1]["prompt"])
    requested = int(DATA_PIPELINE.get("final_answer_max_tokens", 5120) or 5120)
    effective = bounded_completion_budget(
        prompt,
        context_limit=get_llm_context_length(),
        requested_max=requested,
        safety_margin=256,
    )
    response = LLMClient().text_completion(prompt, max_tokens=effective)
    result = {
        "source": str(SOURCE),
        "prompt_tokens_local": get_token_count(prompt),
        "requested_output_tokens": requested,
        "effective_output_tokens": effective,
        "context_limit": get_llm_context_length(),
        "finish_reason": getattr(response, "finish_reason", ""),
        "usage": getattr(response, "usage", {}),
        "raw_output": str(response.content or ""),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
