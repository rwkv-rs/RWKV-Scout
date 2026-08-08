"""Probe the active RWKV endpoint with the prompt formats used by the workflow.

The output is JSON so Chinese text and raw continuation boundaries can be
inspected without depending on a terminal's locale.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from clients.llm_client import LLMClient
from utils.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    build_final_continuation_prompt,
)


def main() -> None:
    llm = LLMClient()
    cases = [
        {
            "id": "official_final_with_think_prefill",
            "prompt": build_final_continuation_prompt(
                "问题：2024年诺贝尔物理学奖授予了谁？\n\n证据：\n[S1] Nobel Prize 页面说明相关获奖者。"
            ),
            "stop": [],
        },
        {
            "id": "official_final_no_evidence",
            "prompt": build_final_continuation_prompt(
                "问题：请确认一个没有提供来源的冷门事实。\n\n证据状态：没有检索到可用页面正文或证据。\n"
                "请只根据证据作答；如果无法确认，请明确说无法确认。"
            ),
            "stop": [],
        },
        {
            "id": "official_tool_call",
            "prompt": (
                "System: Tools:\n"
                '[{"name":"web_search","description":"Search the public web.","arguments":{"query":{"type":"string"}}}]\n\n'
                "User: 查找 2024 Nobel Physics winners\n\n"
                "Assistant: <think></think>\n```json\n"
            ),
            "stop": list(JSON_CALL_STOP_SUFFIXES),
        },
    ]
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": llm.model,
        "provider": llm.provider,
        "cases": [],
    }
    for case in cases:
        row = dict(case)
        try:
            response = llm.text_completion(
                case["prompt"],
                max_tokens=512,
                stop=case["stop"],
            )
            row.update(
                {
                    "raw_output": str(response.content or ""),
                    "finish_reason": getattr(response, "finish_reason", ""),
                    "usage": getattr(response, "usage", {}),
                }
            )
        except Exception as exc:  # pragma: no cover - endpoint-dependent probe
            row.update({"error": f"{type(exc).__name__}: {exc}"})
        output["cases"].append(row)
    destination = Path("data/evaluation/rwkv_prompt_format_probe_20260729.json")
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
