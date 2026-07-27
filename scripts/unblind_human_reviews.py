"""Join blinded human reviews with the operator-held A/B key."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from utils.human_review import load_reviews, validate_review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reviews", type=Path)
    parser.add_argument("key", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key_payload = json.loads(args.key.read_text(encoding="utf-8"))
    mapping = {
        (str(row.get("question_id")), str(label)): variant
        for row in key_payload.get("cases") or []
        for label, variant in (("A", row.get("A")), ("B", row.get("B")))
        if variant
    }
    output = []
    for review in load_reviews(args.reviews):
        validate_review(review)
        variant = mapping.get((review["question_id"], review["blind_label"]))
        if not variant:
            raise ValueError(f"review has no matching blind key: {review['question_id']} {review['blind_label']}")
        output.append(
            {
                **review,
                "variant": variant,
                "unblinded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "unblinding_key_version": key_payload.get("key_version", ""),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "review_count": len(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
