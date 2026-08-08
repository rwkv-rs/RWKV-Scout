"""Prompt used by the optional local-file preview tool."""

import re


def _wash_slm_input(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\n{2,}", "\n", text).strip()


def build_slm_preview_prompt(chunk_str: str) -> str:
    clean_chunk = _wash_slm_input(chunk_str)
    return (
        "User: 概括文本核心主题与类型。\n"
        f"文本：\n{clean_chunk}\n\n"
        "Assistant: <think>\n</think>"
    )


__all__ = ["build_slm_preview_prompt"]
