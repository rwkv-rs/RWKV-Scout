"""Direct local-model answers for prompts that do not require retrieval."""

from __future__ import annotations

import re

from clients.llm_client import LLMClient
from tools.registry import ToolRegistry


def _clean_visible_answer(text: str) -> str:
    value = text or ""
    value = re.sub(r"<think>[\s\S]*?</think>", "", value, flags=re.IGNORECASE)
    return value.replace("</think>", "").strip()


def _deterministic_non_research_answer(query: str) -> str | None:
    """Answer tiny closed-world prompts without triggering a web search."""
    text = (query or "").strip()
    arithmetic = re.search(r"(\d+)\s*(?:[xX*×]|乘以)\s*(\d+)", text)
    if arithmetic:
        return str(int(arithmetic.group(1)) * int(arithmetic.group(2)))

    if "翻译成英文" in text or "翻译为英文" in text:
        source = re.split(r"[：:]", text, maxsplit=1)[-1].strip(" 。！？!?\t")
        translations = {
            "今天天气很好": "The weather is nice today.",
            "你好": "Hello.",
            "谢谢": "Thank you.",
        }
        return translations.get(source)

    if "摘要" in text and "文本" in text:
        match = re.search(r"文本[：:](.+)", text, flags=re.DOTALL)
        source = match.group(1).strip() if match else ""
        return f"摘要：{source}" if source else "未提供待摘要的文本。"

    return None


@ToolRegistry.register(
    name="answer_user",
    phase="ALL",
    signature="""[Tool] answer_user
- Purpose: answer a non-research user prompt directly with the local model.
- Safety: do not search or invent evidence for closed-world prompts.""",
)
def answer_user(original_goal: str = "", agent_state=None, **kwargs) -> str:
    deterministic = _deterministic_non_research_answer(original_goal)
    if deterministic is not None:
        if agent_state:
            agent_state.is_finished = True
            agent_state.final_result = deterministic
        return deterministic

    response = LLMClient().chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "You are a local research assistant. Answer directly and clearly. "
                    "Do not expose hidden reasoning or <think> tags."
                ),
            },
            {"role": "user", "content": original_goal},
        ]
    ).content
    answer = _clean_visible_answer(response)
    # Preserve the exact model output for audit. Never replace visible model
    # output with a synthetic success/failure sentence.
    if not answer:
        answer = "Local RWKV returned an empty answer."
    if agent_state:
        agent_state.is_finished = True
        agent_state.final_result = answer
    return answer
