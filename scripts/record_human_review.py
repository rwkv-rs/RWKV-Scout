"""Record one blind human review using the shared rubric."""

from __future__ import annotations

import argparse
import json

from utils.human_review import RUBRIC, append_review, create_review


def _scores(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        key, separator, raw_score = value.partition("=")
        if not separator or key not in RUBRIC:
            raise ValueError(f"score must use RUBRIC_KEY=1..5: {value}")
        parsed[key] = int(raw_score)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("question_id")
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--blind-label", choices=["A", "B"], required=True)
    parser.add_argument("--score", action="append", default=[], help="rubric_key=1..5; repeat for all rubric keys")
    parser.add_argument("--notes", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--variant", default="")
    args = parser.parse_args()
    review = create_review(
        question_id=args.question_id,
        reviewer_id=args.reviewer_id,
        blind_label=args.blind_label,
        scores=_scores(args.score),
        notes=args.notes,
        run_id=args.run_id,
        experiment_id=args.experiment_id,
        variant=args.variant,
    )
    append_review(args.output, review)
    print(json.dumps(review, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
