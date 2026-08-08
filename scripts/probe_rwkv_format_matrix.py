"""Explore RWKV transcript and stop-string variants against the live endpoint.

This is intentionally not an ECRA retrieval run: no tools or pages are
executed.  It only measures the model's raw continuation behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from clients.llm_client import LLMClient
from utils.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    build_final_continuation_prompt,
)


OUTPUT = Path("data/evaluation/rwkv_format_matrix_20260729.json")


def _metrics(raw: str, *, kind: str) -> dict[str, object]:
    value = str(raw or "")
    lowered = value.casefold()
    return {
        "chars": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest(),
        "has_user_boundary": bool(re.search(r"(?im)^\s*User:", value)),
        "has_assistant_boundary": bool(re.search(r"(?im)^\s*Assistant:", value)),
        "has_system_boundary": bool(re.search(r"(?im)^\s*System:", value)),
        "has_think_tag": "<think>" in lowered or "</think>" in lowered,
        "has_tool_fence": "```json" in lowered or "<tool_call>" in lowered,
        "looks_flat_tool_call": bool(re.search(r'\{\s*"name"\s*:', value)),
        "looks_native_tool_envelope": '"tool_calls"' in lowered,
        "published_chars": len(value) if kind == "final" else None,
        "published_has_role_boundary": bool(
            re.search(r"(?im)^\s*(?:User:|Assistant:|System:)", value)
        ) if kind == "final" else None,
    }


def _cases() -> list[dict[str, object]]:
    final_body = (
        "回答这个问题，只输出用户可见的最终答案：2024年诺贝尔物理学奖授予了谁？"
        "\n证据：[S1] Nobel Prize 官方页面说明获奖者为 John J. Hopfield 和 Geoffrey E. Hinton。"
    )
    no_evidence_body = (
        "回答这个问题，只输出用户可见的最终答案：确认一个没有任何来源支持的虚构事实。"
        "\n证据状态：NO_USABLE_EVIDENCE，没有检索到页面正文、记录或 chunk。"
        "\n如果不能确认，请明确说无法确认，不要根据记忆猜测。"
    )
    final_role_stops: list[str] = []
    role_only_stops = ["\nUser:", "\nSystem:", "\nAssistant:"]
    tool_prompt = (
        "System: Tools:\n"
        '[{"name":"web_search","description":"Search the public web.","arguments":{"query":{"type":"string"}}}]\n\n'
        "User: Search for the 2024 Nobel Prize in Physics winners. Return one tool call.\n\n"
        "Assistant: <think></think>\n```json\n"
    )
    return [
        {
            "id": "final_official_think_role_stops",
            "kind": "final",
            "prompt": build_final_continuation_prompt(final_body),
            "stop": final_role_stops,
        },
        {
            "id": "final_official_no_think_role_stops",
            "kind": "final",
            "prompt": "User: " + final_body + "\nAssistant:\n",
            "stop": final_role_stops,
        },
        {
            "id": "final_official_think_roles_only",
            "kind": "final",
            "prompt": build_final_continuation_prompt(final_body),
            "stop": role_only_stops,
        },
        {
            "id": "final_official_think_no_evidence",
            "kind": "final",
            "prompt": build_final_continuation_prompt(no_evidence_body),
            "stop": final_role_stops,
        },
        {
            "id": "tool_official_think_canonical_stops",
            "kind": "tool",
            "prompt": tool_prompt,
            "stop": list(JSON_CALL_STOP_SUFFIXES),
        },
        {
            "id": "tool_official_think_role_only_stops",
            "kind": "tool",
            "prompt": tool_prompt,
            "stop": role_only_stops,
        },
    ]


def main() -> None:
    llm = LLMClient()
    cases = _cases()
    report: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": llm.model,
        "provider": llm.provider,
        "repetitions": 3,
        "scope": "format_only_no_retrieval",
        "cases": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        for repetition in range(1, 4):
            row = {
                "case_id": case["id"],
                "kind": case["kind"],
                "repetition": repetition,
                "stop": case["stop"],
                "prompt": case["prompt"],
            }
            try:
                response = llm.text_completion(
                    str(case["prompt"]),
                    max_tokens=512,
                    stop=case["stop"],
                )
                raw = str(response.content or "")
                row.update(
                    {
                        "raw_output": raw,
                        "finish_reason": getattr(response, "finish_reason", ""),
                        "usage": getattr(response, "usage", {}),
                        "metrics": _metrics(raw, kind=str(case["kind"])),
                    }
                )
            except Exception as exc:  # pragma: no cover - endpoint dependent
                row["error"] = f"{type(exc).__name__}: {exc}"
            report["cases"].append(row)
            OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
