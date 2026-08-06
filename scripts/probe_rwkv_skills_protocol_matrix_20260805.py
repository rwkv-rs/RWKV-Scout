"""Batch-probe RWKV tool/final transcript contracts against the local model.

This is a format probe only.  It does not run retrieval, execute tools, or
score benchmark answers.  The official tool shapes mirror the local
rwkv-skills function-calling implementation; the other shapes are controls.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from clients.llm_client import LLMClient


OUTPUT = Path("data/evaluation/rwkv_skills_protocol_matrix_20260805.json")
REPETITIONS = 8
WORKERS = 4

TOOLS = (
    '{"name":"web_search","description":"Search the public web.",'
    '"parameters":{"type":"object","properties":{"query":{"type":"string"},'
    '"time_range":{"type":"string"}},"required":["query"]}}'
)
TOOL_BODY = (
    "Find the current official title of the RWKV project. Choose exactly one listed tool. "
    'Return only a JSON function call with shape {"name":"tool_name","arguments":{...}}. '
    f"Tools: [{TOOLS}]"
)
FINAL_BODY = (
    "Answer from the evidence below in concise user-facing prose only. Do not expose reasoning. "
    "Question: Who won the 2024 Nobel Prize in Physics? "
    "Evidence: The official Nobel Prize page names John J. Hopfield and Geoffrey E. Hinton."
)


def _prompt(style: str, kind: str) -> tuple[str, tuple[str, ...]]:
    body = TOOL_BODY if kind == "tool" else FINAL_BODY
    if style == "rwkv_official_prefill":
        prompt = (
            "System: Tools:\n"
            f"[{TOOLS}]\n"
            "Return only a JSON function call.\n"
            'The JSON shape is {"name":"tool_name","arguments":{...}}.\n'
            "Use only listed tool names.\n"
            f"User: {body}\n\nAssistant: ```json\n{{"
            if kind == "tool"
            else f"User: {body}\n\nAssistant:"
        )
        stop = ("\n```", "```", "\nUser:", "\nSystem:", "\nAssistant:")
        return prompt, stop
    if style == "rwkv_official_no_prefill":
        prompt = (
            "System: Tools:\n"
            f"[{TOOLS}]\n"
            "Return only a JSON function call.\n"
            'The JSON shape is {"name":"tool_name","arguments":{...}}.\n'
            "Use only listed tool names.\n"
            f"User: {body}\n\nAssistant: ```json\n"
            if kind == "tool"
            else f"User: {body}\n\nAssistant:"
        )
        stop = ("\n```", "```", "\nUser:", "\nSystem:", "\nAssistant:")
        return prompt, stop
    if style == "rwkv_official_empty_think":
        prompt = (
            "System: Tools:\n"
            f"[{TOOLS}]\n"
            "Return only a JSON function call.\n"
            'The JSON shape is {"name":"tool_name","arguments":{...}}.\n'
            f"User: {body}\n\nAssistant: <think></think>\n"
            + ("```json\n" if kind == "tool" else "")
        )
        stop = ("\n```", "```", "\nUser:", "\nSystem:", "\nAssistant:")
        return prompt, stop
    if style == "single_user_no_cot":
        prompt = f"User: {body}\n\nAssistant: "
        if kind == "tool":
            prompt += "```json\n"
        stop = ("\n```", "```", "\nUser:", "\nAssistant:")
        return prompt, stop
    if style == "current_incomplete_think":
        prompt = f"User: {body}\n\nAssistant: <think></think"
        stop = ("\n```", "```", "\nUser:", "\nAssistant:")
        return prompt, stop
    if style == "current_incomplete_think_json":
        prompt = f"User: {body}\n\nAssistant: <think></think\n```json\n"
        stop = ("\n```", "```", "\nUser:", "\nAssistant:")
        return prompt, stop
    raise ValueError(style)


def _first_json(raw: str, *, prefilled: bool) -> dict | None:
    value = str(raw or "").strip()
    value = re.sub(r"^\s*(?:Assistant:)\s*", "", value, flags=re.I)
    value = re.sub(r"^\s*```json\s*", "", value, flags=re.I)
    if prefilled and value.startswith('"'):
        value = "{" + value
    try:
        parsed, _ = json.JSONDecoder().raw_decode(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _run_one(style: str, kind: str, repetition: int) -> dict:
    prompt, stop = _prompt(style, kind)
    row = {
        "style": style,
        "kind": kind,
        "repetition": repetition,
        "prompt": prompt,
        "stop": list(stop),
    }
    started = time.perf_counter()
    try:
        response = LLMClient().text_completion(prompt, max_tokens=256, stop=stop)
        raw = str(response.content or "")
        prefilled = style == "rwkv_official_prefill" and kind == "tool"
        parsed = _first_json(raw, prefilled=prefilled) if kind == "tool" else None
        row.update(
            {
                "raw_output": raw,
                "finish_reason": getattr(response, "finish_reason", ""),
                "usage": getattr(response, "usage", {}),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "metrics": {
                    "json_valid": parsed is not None,
                    "json_function_shape": bool(
                        isinstance(parsed, dict)
                        and isinstance(parsed.get("name"), str)
                        and isinstance(parsed.get("arguments"), dict)
                    )
                    if kind == "tool"
                    else None,
                    "has_think_content": bool(re.search(r"(?is)<think>\s*[^<]+?</think>", raw)),
                    "has_role_leak": bool(re.search(r"(?im)^\s*(?:User|Assistant|System):", raw)),
                    "has_meta_leak": any(
                        marker in raw.casefold()
                        for marker in ("the user", "i should", "as an ai", "routing")
                    ),
                    "has_answer_signal": (
                        "hopfield" in raw.casefold() and "hinton" in raw.casefold()
                        if kind == "final"
                        else None
                    ),
                },
            }
        )
    except Exception as exc:
        row.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
    return row


def main() -> None:
    styles = (
        "rwkv_official_prefill",
        "rwkv_official_no_prefill",
        "rwkv_official_empty_think",
        "single_user_no_cot",
        "current_incomplete_think",
        "current_incomplete_think_json",
    )
    jobs = [
        (style, kind, repetition)
        for style in styles
        for kind in ("tool", "final")
        for repetition in range(1, REPETITIONS + 1)
    ]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "format_only_no_retrieval",
        "repetitions": REPETITIONS,
        "workers": WORKERS,
        "model": LLMClient().model,
        "provider": LLMClient().provider,
        "cases": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(_run_one, *job) for job in jobs]
        for future in as_completed(futures):
            report["cases"].append(future.result())
    report["cases"].sort(key=lambda row: (row["style"], row["kind"], row["repetition"]))
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
