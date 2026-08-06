"""Probe the rwkv-skills JSON tool protocol over multiple model turns."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clients.llm_client import LLMClient


OUTPUT = Path("data/evaluation/rwkv_skills_multiturn_20260805.json")
REPETITIONS = 8
WORKERS = 4
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the public web for current information.",
        "arguments": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {"name": "finish_task", "description": "Finish when the gathered evidence is sufficient.", "arguments": {"type": "object"}},
    {"name": "final_answer", "description": "Submit the user-facing final answer.", "arguments": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}},
]
TOOLS_TEXT = json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":"))
STOP = ("\n```", "```", "\nUser:", "\nSystem:", "\nAssistant:")


def _tool_prefix(*, prefill: bool = True) -> str:
    return "Assistant: ```json\n" + ("{" if prefill else "")


def _base_prompt(instruction: str, *, history: str = "", prefill: bool = True) -> str:
    system = (
        "System: Tools:\n"
        f"{TOOLS_TEXT}\n"
        "Return only a JSON function call.\n"
        'The JSON shape is {"name":"tool_name","arguments":{...}}.\n'
        "Use only listed tool names."
    )
    return f"{system}\n\nUser: {instruction}{history}\n\n{_tool_prefix(prefill=prefill)}"


def _decode(raw: str, *, prefilled: bool = True) -> dict[str, Any] | None:
    value = str(raw or "").strip()
    if prefilled and value.startswith('"'):
        value = "{" + value
    try:
        parsed, _ = json.JSONDecoder().raw_decode(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _call(prompt: str, *, max_tokens: int = 256) -> tuple[str, dict[str, Any] | None, float, str]:
    started = time.perf_counter()
    try:
        response = LLMClient().text_completion(prompt, max_tokens=max_tokens, stop=STOP)
        raw = str(response.content or "")
        return raw, _decode(raw), round((time.perf_counter() - started) * 1000, 1), ""
    except Exception as exc:
        return "", None, round((time.perf_counter() - started) * 1000, 1), f"{type(exc).__name__}: {exc}"


def _run_one(scenario: str, repetition: int) -> dict[str, Any]:
    row: dict[str, Any] = {"scenario": scenario, "repetition": repetition, "steps": []}
    first_prompt = _base_prompt(
        "Find the current official title of the RWKV project. Start by using web_search.",
    )
    first_raw, first_payload, first_ms, first_error = _call(
        first_prompt,
        max_tokens=6 if scenario == "json_truncated" else 256,
    )
    row["steps"].append(
        {
            "prompt": first_prompt,
            "raw_output": first_raw,
            "payload": first_payload,
            "duration_ms": first_ms,
            "error": first_error,
        }
    )
    if scenario == "json_truncated":
        row["result"] = {"truncated_json_parse_failed": first_payload is None}
        return row

    assistant_block = first_raw.strip()
    if first_payload is not None:
        assistant_block = json.dumps(first_payload, ensure_ascii=False, separators=(",", ":"))
    if scenario == "finish_task":
        feedback = {
            "status": "ok",
            "tool": "web_search",
            "evidence_ready": True,
            "results": [{"url": "https://example.com/rwkv", "title": "RWKV", "content": "RWKV is an RNN language model."}],
        }
        instruction = "The evidence is sufficient. Call finish_task now."
        expected = "finish_task"
    elif scenario == "final_answer":
        feedback = {
            "status": "ok",
            "tool": "web_search",
            "evidence_ready": True,
            "results": [{"url": "https://example.com/rwkv", "title": "RWKV", "content": "RWKV is an RNN language model."}],
        }
        instruction = "The evidence is sufficient. Call final_answer now with a concise answer."
        expected = "final_answer"
    elif scenario == "tool_failure":
        feedback = {
            "status": "error",
            "error_class": "provider_error",
            "message": "The previous search failed. Decide the next action yourself; do not assume facts.",
            "results": [],
        }
        instruction = "The previous tool failed. Decide the next listed tool yourself."
        expected = "any_valid_tool"
    elif scenario == "empty_result":
        feedback = {
            "status": "ok",
            "tool": "web_search",
            "evidence_ready": False,
            "results": [],
            "message": "No usable evidence was returned. Decide whether another action is needed.",
        }
        instruction = "The tool returned no usable evidence. Decide the next listed tool yourself."
        expected = "any_valid_tool"
    else:
        raise ValueError(scenario)
    history = (
        f"\n\nAssistant: ```json\n{assistant_block}\n```"
        f"\n\nUser: Function output:\n{json.dumps(feedback, ensure_ascii=False, separators=(',', ':'))}"
        f"\n\nUser: {instruction}"
    )
    second_prompt = _base_prompt("Continue the task.", history=history)
    second_raw, second_payload, second_ms, second_error = _call(second_prompt)
    row["steps"].append(
        {
            "prompt": second_prompt,
            "raw_output": second_raw,
            "payload": second_payload,
            "duration_ms": second_ms,
            "error": second_error,
        }
    )
    second_name = str((second_payload or {}).get("name") or "")
    row["result"] = {
        "first_json_valid": first_payload is not None,
        "second_json_valid": second_payload is not None,
        "second_tool_name": second_name,
        "expected": expected,
        "second_expected_name": second_name == expected if expected != "any_valid_tool" else bool(second_name),
    }
    return row


def main() -> None:
    scenarios = ("finish_task", "final_answer", "tool_failure", "empty_result", "json_truncated")
    jobs = [(scenario, repetition) for scenario in scenarios for repetition in range(1, REPETITIONS + 1)]
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "rwkv_skills_official_json_multiturn_no_retrieval",
        "model": LLMClient().model,
        "provider": LLMClient().provider,
        "workers": WORKERS,
        "repetitions": REPETITIONS,
        "cases": [],
    }
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(_run_one, *job) for job in jobs]
        for future in as_completed(futures):
            report["cases"].append(future.result())
    report["cases"].sort(key=lambda row: (row["scenario"], row["repetition"]))
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
