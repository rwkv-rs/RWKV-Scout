"""Run repeated baseline/candidate evaluations with order swapping."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils.experiment_strategies import load_strategy_file, normalize_strategy


def _run(command: list[str]) -> int:
    print("COMMAND", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-action", required=True)
    parser.add_argument("--candidate-action", required=True)
    parser.add_argument("--baseline-strategy-config", type=Path, default=None)
    parser.add_argument("--candidate-strategy-config", type=Path, default=None)
    parser.add_argument("--changed-variable", default="")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--side-effect", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output/experiments"))
    parser.add_argument("--human-reviews", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    baseline_strategy = load_strategy_file(str(args.baseline_strategy_config)) if args.baseline_strategy_config else normalize_strategy()
    candidate_strategy = load_strategy_file(str(args.candidate_strategy_config)) if args.candidate_strategy_config else normalize_strategy()
    strategy_differences = sorted(
        key for key in baseline_strategy
        if baseline_strategy.get(key) != candidate_strategy.get(key)
    )
    if args.baseline_action != args.candidate_action and strategy_differences:
        parser.error("controlled experiment changes search_action and strategy_config simultaneously")
    if len(strategy_differences) > 1:
        parser.error("strategy_config must change only one strategy variable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    comparison_paths: list[Path] = []
    exit_code = 0
    for repeat in range(1, args.repeats + 1):
        execution_order = "baseline-first" if repeat % 2 else "candidate-first"
        paths: dict[str, Path] = {}
        for variant, action, baseline_action in (
            ("baseline", args.baseline_action, ""),
            ("candidate", args.candidate_action, args.baseline_action),
        ):
            output = args.output_dir / f"{args.experiment_id}-r{repeat}-{variant}.json"
            paths[variant] = output
            command = [
                sys.executable,
                "-m",
                "scripts.run_dynamic_eval",
                str(args.dataset),
                "--experiment-id",
                args.experiment_id,
                "--variant",
                variant,
                "--repeat-id",
                str(repeat),
                "--execution-order",
                execution_order,
                "--search-action",
                action,
                "--output",
                str(output),
            ]
            if baseline_action:
                command.extend(["--baseline-search-action", baseline_action])
            if args.limit > 0:
                command.extend(["--limit", str(args.limit)])
            if args.changed_variable:
                command.extend(["--changed-variable", args.changed_variable])
            if args.hypothesis:
                command.extend(["--hypothesis", args.hypothesis])
            for side_effect in args.side_effect:
                command.extend(["--side-effect", side_effect])
            if args.retrieval_only:
                command.append("--retrieval-only")
            strategy_path = args.baseline_strategy_config if variant == "baseline" else args.candidate_strategy_config
            if strategy_path:
                command.extend(["--strategy-config", str(strategy_path)])
            code = _run(command)
            runs.append({"repeat_id": repeat, "variant": variant, "execution_order": execution_order, "artifact": str(output), "exit_code": code})
            exit_code = max(exit_code, code)

        comparison = args.output_dir / f"{args.experiment_id}-r{repeat}-comparison.json"
        compare_command = [
            sys.executable,
            "-m",
            "scripts.compare_experiment_runs",
            str(paths["baseline"]),
            str(paths["candidate"]),
            "--output",
            str(comparison),
        ]
        if args.human_reviews:
            compare_command.extend(["--human-reviews", str(args.human_reviews)])
        compare_code = _run(compare_command)
        comparison_paths.append(comparison)
        runs.append({"repeat_id": repeat, "comparison": str(comparison), "exit_code": compare_code})
        exit_code = max(exit_code, compare_code)

    stability: dict[str, dict[str, object]] = {}
    for comparison in comparison_paths:
        try:
            payload = json.loads(comparison.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for metric, values in ((payload.get("comparison") or {}).get("paired_statistics") or {}).items():
            if not isinstance(values, dict) or not isinstance(values.get("mean_delta"), (int, float)):
                stability.setdefault(metric, {"repeat_deltas": [], "direction": "unknown", "consistent": False})
                continue
            stability.setdefault(metric, {"repeat_deltas": [], "direction": "unknown", "consistent": False})["repeat_deltas"].append(values["mean_delta"])
    for metric, summary in stability.items():
        deltas = [float(value) for value in summary["repeat_deltas"]]
        signs = {1 if value > 0 else -1 if value < 0 else 0 for value in deltas}
        if not deltas:
            summary["direction"] = "unknown"
        elif signs == {1}:
            summary["direction"] = "positive"
        elif signs == {-1}:
            summary["direction"] = "negative"
        elif signs == {0}:
            summary["direction"] = "flat"
        else:
            summary["direction"] = "mixed"
        summary["consistent"] = len(deltas) >= 2 and len(signs) == 1 and 0 not in signs

    manifest = {
        "artifact_version": "paired-eval-manifest.v1",
        "experiment_id": args.experiment_id,
        "dataset": str(args.dataset),
        "baseline_action": args.baseline_action,
        "candidate_action": args.candidate_action,
        "baseline_strategy": baseline_strategy,
        "candidate_strategy": candidate_strategy,
        "strategy_differences": strategy_differences,
        "changed_variable": args.changed_variable,
        "repeats": args.repeats,
        "retrieval_only": args.retrieval_only,
        "runs": runs,
        "stability": stability,
        "exit_code": exit_code,
    }
    manifest_path = args.output_dir / f"{args.experiment_id}-paired-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "exit_code": exit_code}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
