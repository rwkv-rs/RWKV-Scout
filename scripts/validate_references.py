"""Validate human reference readiness before an experiment may be promoted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.evaluation_dataset import load_dataset
from utils.reference_validation import validate_dataset_references


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--stale-after-days", type=int, default=30)
    parser.add_argument("--check-remote", action="store_true", help="also fetch each reference URL")
    args = parser.parse_args()
    result = validate_dataset_references(
        load_dataset(args.dataset),
        default_stale_after_days=max(1, args.stale_after_days),
        check_remote=args.check_remote,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
