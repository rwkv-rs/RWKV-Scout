"""Validated human-review records for blind paired experiment evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


RUBRIC = (
    "fact_correctness",
    "evidence_support",
    "citation_accuracy",
    "completeness",
    "instruction_adherence",
    "risk_handling",
    "usability",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def blind_order(question_id: str, seed: int = 20260726) -> tuple[str, str]:
    """Return a deterministic A/B order without exposing the variant label."""
    labels = ["baseline", "candidate"]
    random.Random(f"{seed}:{question_id}").shuffle(labels)
    return labels[0], labels[1]


def _artifact_rows(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in artifact.get("results") or []:
        if not isinstance(item, dict):
            continue
        case = item.get("case") or {}
        trace = item.get("trace") or {}
        question_id = str(case.get("question_id") or item.get("question_id") or "").strip()
        if question_id:
            rows[question_id] = {"case": case, "trace": trace, "item": item}
    return rows


def build_blind_packet(
    baseline_artifact: dict[str, Any],
    candidate_artifact: dict[str, Any],
    *,
    seed: int = 20260726,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a reviewer-facing A/B packet and a separate unblinding key.

    The packet intentionally contains no variant labels.  The key is returned
    separately so it can be held by the experiment operator and joined only
    after the human reviews are complete.
    """
    baseline = _artifact_rows(baseline_artifact)
    candidate = _artifact_rows(candidate_artifact)
    packet_cases: list[dict[str, Any]] = []
    key_cases: list[dict[str, Any]] = []
    for question_id in sorted(set(baseline) & set(candidate)):
        base = baseline[question_id]
        cand = candidate[question_id]
        first, second = blind_order(question_id, seed)
        variants = {"baseline": base, "candidate": cand}

        def answer_payload(row: dict[str, Any]) -> dict[str, Any]:
            trace = row["trace"]
            return {
                "answer": str(trace.get("final_answer") or ""),
                "citations": [
                    {
                        key: value
                        for key, value in dict(citation).items()
                        if key in {"ref_id", "title", "url", "evidence_text", "content", "source_span", "evidence_locator"}
                    }
                    for citation in trace.get("citations") or []
                    if isinstance(citation, dict)
                ],
                "status": str((trace.get("manifest") or {}).get("status") or "unknown"),
            }

        packet_cases.append(
            {
                "question_id": question_id,
                "question": str((base["case"] or {}).get("question") or ""),
                "A": answer_payload(variants[first]),
                "B": answer_payload(variants[second]),
            }
        )
        key_cases.append({"question_id": question_id, "A": first, "B": second})
    packet = {
        "packet_version": "human-review-packet.v1",
        "created_at": utc_now(),
        "seed": seed,
        "baseline_experiment_id": baseline_artifact.get("experiment_id", ""),
        "candidate_experiment_id": candidate_artifact.get("experiment_id", ""),
        "sample_count": len(packet_cases),
        "rubric": list(RUBRIC),
        "cases": packet_cases,
    }
    key = {
        "key_version": "human-review-key.v1",
        "created_at": packet["created_at"],
        "seed": seed,
        "cases": key_cases,
    }
    return packet, key


def validate_review(review: dict[str, Any]) -> None:
    required = ("review_id", "question_id", "reviewer_id", "blind_label", "scores", "created_at")
    missing = [key for key in required if not review.get(key)]
    if missing:
        raise ValueError(f"human review missing fields: {', '.join(missing)}")
    if str(review["reviewer_id"]).casefold() in {"model", "auto", "self"}:
        raise ValueError("reviewer_id must identify a human reviewer")
    if review["blind_label"] not in {"A", "B"}:
        raise ValueError("blind_label must be A or B")
    scores = review["scores"]
    if not isinstance(scores, dict):
        raise ValueError("scores must be an object")
    missing_scores = [key for key in RUBRIC if key not in scores]
    if missing_scores:
        raise ValueError(f"human review missing rubric scores: {', '.join(missing_scores)}")
    for key in RUBRIC:
        value = scores[key]
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise ValueError(f"score {key} must be an integer from 1 to 5")


def create_review(
    *,
    question_id: str,
    reviewer_id: str,
    blind_label: str,
    scores: dict[str, int],
    notes: str = "",
    run_id: str = "",
    experiment_id: str = "",
    variant: str = "",
    evidence_notes: list[str] | None = None,
) -> dict[str, Any]:
    seed_material = f"{question_id}|{reviewer_id}|{blind_label}|{utc_now()}"
    review = {
        "review_id": "review-" + hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:12],
        "question_id": question_id,
        "reviewer_id": reviewer_id,
        "blind_label": blind_label,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "variant": variant,
        "rubric_version": "human-review.v1",
        "scores": dict(scores),
        "notes": str(notes or ""),
        "evidence_notes": list(evidence_notes or []),
        "created_at": utc_now(),
    }
    validate_review(review)
    return review


def append_review(path: str | Path, review: dict[str, Any]) -> None:
    validate_review(review)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(review, ensure_ascii=False) + "\n")


def load_reviews(path: str | Path) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        review = json.loads(line)
        validate_review(review)
        reviews.append(review)
    return reviews


def aggregate_reviews(reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        validate_review(review)
        groups[str(review.get("variant") or review.get("blind_label"))].append(review)
    output: dict[str, Any] = {}
    for group, rows in groups.items():
        output[group] = {
            "review_count": len(rows),
            "question_count": len({row["question_id"] for row in rows}),
            "scores": {
                key: round(mean(row["scores"][key] for row in rows), 4)
                for key in RUBRIC
            },
        }
    return {"rubric_version": "human-review.v1", "groups": output}


def paired_blind_summary(reviews: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize A/B wins while keeping the raw reviewer identity intact."""
    rows = list(reviews)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        validate_review(row)
        by_question[row["question_id"]].append(row)
    comparisons = []
    for question_id, pair in sorted(by_question.items()):
        if len(pair) != 2 or {row["blind_label"] for row in pair} != {"A", "B"}:
            continue
        a, b = sorted(pair, key=lambda row: row["blind_label"])
        a_total = sum(a["scores"].values())
        b_total = sum(b["scores"].values())
        comparisons.append(
            {
                "question_id": question_id,
                "a_total": a_total,
                "b_total": b_total,
                "winner": "A" if a_total > b_total else "B" if b_total > a_total else "tie",
            }
        )
    return {
        "paired_question_count": len(comparisons),
        "a_wins": sum(row["winner"] == "A" for row in comparisons),
        "b_wins": sum(row["winner"] == "B" for row in comparisons),
        "ties": sum(row["winner"] == "tie" for row in comparisons),
        "comparisons": comparisons,
    }
