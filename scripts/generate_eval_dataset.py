"""Generate a versioned evaluation JSONL dataset for RWKV-ECRA."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.evaluation_dataset import append_dataset, generate_cases, save_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", default="data/evaluation/dynamic.jsonl")
    parser.add_argument("--append", action="store_true", help="append new deterministic cases and preserve existing references")
    args = parser.parse_args()
    target = Path(args.output)
    if args.append:
        version = append_dataset(target, args.count, seed=args.seed)
        sample_count = sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        rows = generate_cases(args.count, seed=args.seed)
        version = save_dataset(rows, target)
        sample_count = len(rows)
    print(f"dataset_version={version} samples={sample_count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
