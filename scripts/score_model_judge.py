"""Run a supplementary model-assisted blind pairwise evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from clients.llm_client import LLMClient
from utils.human_review import build_blind_packet
from utils.model_judge import judge_pair


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--dry-run", action="store_true", help="persist prompts without calling the model")
    args = parser.parse_args()

    packet, key = build_blind_packet(_read(args.baseline), _read(args.candidate), seed=args.seed)
    cases = packet["cases"][: args.limit] if args.limit > 0 else packet["cases"]
    llm = None if args.dry_run else LLMClient()
    results = []
    for case in cases:
        if args.dry_run:
            result = {
                "judge_version": "model-judge.v1",
                "status": "not_run",
                "prompt": "",
                "parsed": {"valid": False, "issues": ["dry_run"]},
            }
        else:
            result = judge_pair(case["question"], case["A"], case["B"], llm=llm)
        results.append({"question_id": case["question_id"], **result})

    payload = {
        "artifact_version": "model-judge-artifact.v1",
        "baseline_experiment_id": packet.get("baseline_experiment_id", ""),
        "candidate_experiment_id": packet.get("candidate_experiment_id", ""),
        "seed": args.seed,
        "sample_count": len(results),
        "dry_run": args.dry_run,
        "model": {
            **{key: value for key, value in config.get_experiment_model_config().items() if key != "api_key"},
            "api_key_configured": bool(config.get_experiment_model_config().get("api_key")),
        },
        "prompt_version": config.get_prompt_version(),
        "rubric": packet["rubric"],
        "unblinding_key": key,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_count": len(results), "dry_run": args.dry_run}, ensure_ascii=False))
    return 0 if args.dry_run or all(item.get("status") == "completed" for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
