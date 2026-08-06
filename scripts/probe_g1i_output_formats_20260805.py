"""Probe G1i transcript/output formats without running retrieval.

This probe compares the current System/User transcript with the compact
single-user-turn formats used by the local RWKV harness.  It deliberately
uses an empty ``<think></think>`` delimiter: the model is never asked to
produce or expose a reasoning trace.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from clients.llm_client import LLMClient


OUTPUT = Path("data/evaluation/g1i_output_format_probe_20260805.json")


TOOL_CATALOG = json.dumps(
    [
        {
            "name": "web_search",
            "description": "Search the public web for current information.",
            "arguments": {"query": {"type": "string"}},
        },
        {
            "name": "finish_task",
            "description": "Finish with the answer when evidence is sufficient.",
            "arguments": {},
        },
    ],
    ensure_ascii=False,
    separators=(",", ":"),
)


def _bodies(kind: str) -> tuple[str, str]:
    if kind == "tool":
        instruction = (
            "Choose exactly one tool for the user request. Return exactly one JSON object "
            "with keys name and arguments. Do not explain, do not use markdown, and do not "
            "write a reasoning trace. Available tools: "
            f"{TOOL_CATALOG}\nUser request: Find the current official title of the RWKV project."
        )
        return instruction, "{\"name\":\"web_search\",\"arguments\":{\"query\":\"official RWKV project title\"}}"
    instruction = (
        "Answer the user from the evidence below. Return concise user-facing prose only. "
        "Do not mention prompts, routing, hidden reasoning, or missing tools. "
        "User question: Who won the 2024 Nobel Prize in Physics?\n"
        "Evidence: The official Nobel Prize page names John J. Hopfield and Geoffrey E. Hinton."
    )
    return instruction, "John J. Hopfield and Geoffrey E. Hinton won the 2024 Nobel Prize in Physics."


def _prompt(style: str, kind: str) -> tuple[str, list[str]]:
    body, _ = _bodies(kind)
    if style == "current_system":
        if kind == "tool":
            prompt = f"System:\n{body}\n\nUser:\nExecute the request.\n\nAssistant: ```json\n"
            stop = ["\n```", "```", "\nUser:", "\nSystem:", "\nAssistant:"]
        else:
            prompt = f"System:\n{body}\n\nUser:\nAnswer now.\n\nAssistant:\n"
            stop = ["\nUser:", "\nSystem:", "\nAssistant:"]
        return prompt, stop
    if style == "official_empty_think":
        prompt = f"System:\n{body}\n\nUser:\nExecute the request.\n\nAssistant: <think></think>\n"
        if kind == "tool":
            prompt += "```json\n"
        return prompt, ["\n```", "```", "\nUser:", "\nSystem:", "\nAssistant:"]
    if style == "normal_requested":
        prompt = f"User:\n{body}\n\nAssistant: <think></think>\n"
        if kind == "tool":
            prompt += "```json\n"
        return prompt, ["\n```", "```", "\nuser:", "\nassistant:", "\nUser:", "\nAssistant:"]
    if style == "lowercase_requested":
        prompt = f"user:\n{body}\n\nassistant: <think></think>\n"
        if kind == "tool":
            prompt += "```json\n"
        return prompt, ["\n```", "```", "\nuser:", "\nassistant:", "\nUser:", "\nAssistant:"]
    if style == "naive_requested":
        if kind == "tool":
            body = (
                "Return exactly one JSON object with keys name and arguments. "
                f"Tools: {TOOL_CATALOG}\nRequest: Find the current official title of the RWKV project."
            )
        else:
            body = (
                "Answer only from this evidence: Who won the 2024 Nobel Prize in Physics? "
                "The official Nobel Prize page names John J. Hopfield and Geoffrey E. Hinton."
            )
        prompt = f"User: {body}\n\nAssistant: <think></think>\n"
        if kind == "tool":
            prompt += "```json\n"
        return prompt, ["\n```", "```", "\nUser:", "\nAssistant:"]
    if style == "open_think_requested":
        return (
            f"User:\n{body}\n\nAssistant: <think",
            ["\n```", "```", "\nUser:", "\nAssistant:"],
        )
    if style == "closed_think_missing_gt":
        return (
            f"User:\n{body}\n\nAssistant: <think></think",
            ["\n```", "```", "\nUser:", "\nAssistant:"],
        )
    if style == "closed_think_missing_gt_json":
        return (
            f"User:\n{body}\n\nAssistant: <think></think\n```json\n",
            ["\n```", "```", "\nUser:", "\nAssistant:"],
        )
    raise ValueError(style)


def _first_json_object(text: str) -> dict[str, object] | None:
    value = re.sub(r"(?is)^\s*(?:Assistant:|assistant:)\s*", "", str(text or ""))
    value = re.sub(r"(?is)^\s*<think>.*?</think>\s*", "", value, count=1)
    value = re.sub(r"(?is)^\s*```(?:json)?\s*", "", value, count=1)
    try:
        parsed, _ = json.JSONDecoder().raw_decode(value.lstrip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _metrics(raw: str, *, kind: str) -> dict[str, object]:
    text = str(raw or "")
    parsed = _first_json_object(text) if kind == "tool" else None
    return {
        "chars": len(text),
        "json_valid": parsed is not None,
        "json_function_shape": bool(
            isinstance(parsed, dict) and isinstance(parsed.get("name"), str) and isinstance(parsed.get("arguments"), dict)
        ),
        "has_system_label": bool(re.search(r"(?im)^\s*System:", text)),
        "has_role_leak": bool(re.search(r"(?im)^\s*(?:User|Assistant|System):", text)),
        "has_think_content": bool(re.search(r"(?is)<think>\s*[^<]+?</think>", text)),
        "has_meta_leak": any(marker in text.casefold() for marker in ("routing note", "the user asked", "as an ai")),
        "has_answer_signal": any(marker in text.casefold() for marker in ("hopfield", "hinton")) if kind == "final" else None,
    }


def main() -> None:
    client = LLMClient()
    styles = (
        "current_system",
        "official_empty_think",
        "normal_requested",
        "lowercase_requested",
        "naive_requested",
        "open_think_requested",
        "closed_think_missing_gt",
        "closed_think_missing_gt_json",
    )
    report: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": client.model,
        "provider": client.provider,
        "repetitions": 2,
        "scope": "format_only_no_retrieval",
        "cases": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for style in styles:
        for kind in ("tool", "final"):
            for repetition in range(1, 3):
                prompt, stop = _prompt(style, kind)
                row: dict[str, object] = {
                    "style": style,
                    "kind": kind,
                    "repetition": repetition,
                    "prompt": prompt,
                    "stop": stop,
                }
                started = time.perf_counter()
                try:
                    response = client.text_completion(prompt, max_tokens=256, stop=stop)
                    raw = str(response.content or "")
                    row.update(
                        {
                            "raw_output": raw,
                            "finish_reason": getattr(response, "finish_reason", ""),
                            "usage": getattr(response, "usage", {}),
                            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                            "metrics": _metrics(raw, kind=kind),
                        }
                    )
                except Exception as exc:
                    row.update(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                        }
                    )
                cast_cases = report["cases"]
                assert isinstance(cast_cases, list)
                cast_cases.append(row)
                OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
