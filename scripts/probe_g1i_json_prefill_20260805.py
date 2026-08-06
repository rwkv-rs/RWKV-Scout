from __future__ import annotations

import json
from pathlib import Path

from clients.llm_client import LLMClient


def main() -> None:
    prompt = "User: 回复OK。\nAssistant: <think></think"
    response = LLMClient().text_completion(
        prompt,
        max_tokens=128,
        stop=["\nUser:", "\nAssistant:"],
    )
    payload = {"prompt": prompt, "raw_output": str(response.content or "")}
    destination = Path("data/evaluation/g1i_json_prefill_probe_20260805.json")
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
