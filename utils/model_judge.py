"""Traceable, supplementary model-assisted pairwise judging.

Model judging is intentionally separate from the promotion gate.  The raw
prompt, visible output, parsed scores and parse errors are persisted so a
human can audit or discard the judge result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from config import get_model_stage_temperature, model_sampling_parameters
from utils.human_review import RUBRIC
from utils.model_events import visible_model_text


JUDGE_VERSION = "model-judge.v1"


def build_judge_prompt(question: str, answer_a: dict[str, Any], answer_b: dict[str, Any]) -> str:
    rubric = ", ".join(RUBRIC)
    return (
        "You are a supplementary evaluator for a retrieval-augmented answer experiment.\n"
        "Evaluate A and B only from the question and the visible answers/citations. "
        "Do not infer which system produced either answer. Do not reward verbosity.\n"
        f"Score each answer from 1 to 5 on: {rubric}.\n"
        "Return JSON only with this shape: "
        '{"winner":"A|B|tie","scores":{"A":{"fact_correctness":1,...},'
        '"B":{"fact_correctness":1,...}},"reason":"short evidence-based reason"}.\n\n'
        f"Question:\n{question}\n\n"
        f"Answer A:\n{json.dumps(answer_a, ensure_ascii=False, indent=2)}\n\n"
        f"Answer B:\n{json.dumps(answer_b, ensure_ascii=False, indent=2)}"
    )


def _json_object(raw: str) -> dict[str, Any] | None:
    text = visible_model_text(raw).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    if not fenced:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        value = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_judge_output(raw: str) -> dict[str, Any]:
    """Validate a judge response without silently filling missing scores."""
    value = _json_object(raw)
    if value is None:
        return {"valid": False, "issues": ["invalid_json"], "raw_output": visible_model_text(raw)}
    issues: list[str] = []
    winner = str(value.get("winner") or "").strip().lower()
    if winner not in {"a", "b", "tie"}:
        issues.append("invalid_winner")
    scores = value.get("scores")
    normalized_scores: dict[str, dict[str, int]] = {}
    if not isinstance(scores, dict):
        issues.append("missing_scores")
    else:
        for label in ("A", "B"):
            row = scores.get(label) or scores.get(label.lower())
            if not isinstance(row, dict):
                issues.append(f"missing_scores:{label}")
                continue
            normalized_scores[label] = {}
            for rubric_key in RUBRIC:
                score = row.get(rubric_key)
                if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
                    issues.append(f"invalid_score:{label}.{rubric_key}")
                else:
                    normalized_scores[label][rubric_key] = score
    reason = str(value.get("reason") or "").strip()
    if not reason:
        issues.append("missing_reason")
    return {
        "valid": not issues,
        "issues": issues,
        "winner": winner.upper() if winner in {"a", "b"} else winner,
        "scores": normalized_scores,
        "reason": reason,
        "raw_output": visible_model_text(raw),
    }


def judge_pair(
    question: str,
    answer_a: dict[str, Any],
    answer_b: dict[str, Any],
    *,
    llm: Any,
) -> dict[str, Any]:
    prompt = build_judge_prompt(question, answer_a, answer_b)
    try:
        temperature = get_model_stage_temperature("model_judge")
        with model_sampling_parameters(
            temperature,
            stage="model_judge",
            policy_reason="structured_supplementary_quality_judgement",
        ):
            response = llm.chat_completion(
                [
                    {"role": "system", "content": "Return only the requested JSON evaluation."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
            )
        raw = str(getattr(response, "content", "") or "")
        parsed = parse_judge_output(raw)
        return {
            "judge_version": JUDGE_VERSION,
            "status": "completed" if parsed.get("valid") else "invalid_output",
            "prompt": prompt,
            "output": parsed.get("raw_output", ""),
            "parsed": parsed,
        }
    except Exception as exc:
        return {
            "judge_version": JUDGE_VERSION,
            "status": "failed",
            "prompt": prompt,
            "output": "",
            "parsed": {"valid": False, "issues": [f"judge_error:{type(exc).__name__}"]},
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }
