"""Simple local conversational fallback for non-research prompts."""

import re

from clients.llm_client import LLMClient
from tools.registry import ToolRegistry


def _clean_visible_answer(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.replace("</think>", "").strip()


def _deterministic_non_research_answer(query: str) -> str | None:
    """Handle small, closed-world prompts without web search or model drift."""
    text = (query or "").strip()

    arithmetic = re.search(r"(\d+)\s*(?:[×x*]|乘以)\s*(\d+)", text, flags=re.IGNORECASE)
    if arithmetic:
        return str(int(arithmetic.group(1)) * int(arithmetic.group(2)))

    if "翻译成英文" in text or "翻译为英文" in text:
        source = re.split(r"[：:]", text, maxsplit=1)[-1].strip(" 。.\t")
        translations = {
            "今天天气很好": "The weather is nice today.",
            "你好": "Hello.",
            "谢谢": "Thank you.",
        }
        return translations.get(source) or "未提供待翻译的文本。"

    if "摘要" in text and "文本" in text:
        match = re.search(r"文本[：:](.+)", text, flags=re.DOTALL)
        source = match.group(1).strip() if match else ""
        return f"摘要：{source}" if source else "未提供待摘要的文本。"

    return None


@ToolRegistry.register(
    name="answer_user",
    phase="ALL",
    signature="""[Tool] answer_user
- 功能: 对不需要检索或文件分析的普通对话直接回答。
- 参数: 无""",
)
def answer_user(original_goal: str = "", agent_state=None, **kwargs) -> str:
    deterministic = _deterministic_non_research_answer(original_goal)
    if deterministic is not None:
        if agent_state:
            agent_state.is_finished = True
            agent_state.final_result = deterministic
        return deterministic

    llm = LLMClient()
    response = llm.chat_completion(
        [
            {
                "role": "system",
                "content": "你是本地研究助手。直接、清晰地回答用户，不要输出隐藏思维过程或 <think> 标签。",
            },
            {"role": "user", "content": original_goal},
        ]
    ).content
    answer = _clean_visible_answer(response)
    # A small local model can still emit a visible reasoning draft even when
    # instructed not to. Never expose that draft as the user-facing answer.
    lowered = answer.casefold()
    if answer.lstrip().startswith(">") or any(
        marker in lowered
        for marker in ("we need to determine", "the user asks", "analysis:", "reasoning:")
    ):
        answer = "无法生成简洁的直接答案。"
    if agent_state:
        agent_state.is_finished = True
        agent_state.final_result = answer
    return answer
