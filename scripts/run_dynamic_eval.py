"""Execute a versioned evaluation dataset through the real RWKV-Scout flow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import config
from agent.orchestrator import Orchestrator
from utils.evaluation_dataset import load_dataset
from utils.experiment_manifest import finalize_manifest, reconstruct_run
from utils.experiment_metrics import score_trace
from utils.reference_validation import validate_reference_case
from utils.runtime_gate import analysis_slot
from utils.task_events import append_task_event, get_task_events
from utils.trace_validation import validate_replay_trace
from utils.task_manager import record_task
from utils.time_budget import TaskTimeoutError, task_time_budget
from utils.token_tracker import current_task_id
from utils.experiment_strategies import load_strategy_file, normalize_strategy


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or ""))[:80] or "run"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--experiment-id", default="dynamic-eval")
    parser.add_argument("--variant", choices=["baseline", "candidate"], default="candidate")
    parser.add_argument("--baseline-run-id", default="")
    parser.add_argument(
        "--search-action",
        choices=["", "search_web_keyless", "search_web_wigolo", "search_papers"],
        default="",
    )
    parser.add_argument("--baseline-search-action", default="")
    parser.add_argument("--changed-variable", default="", help="single controlled variable being evaluated")
    parser.add_argument("--hypothesis", default="", help="testable improvement hypothesis")
    parser.add_argument("--side-effect", action="append", default=[], help="metric or risk to monitor; repeatable")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip RWKV synthesis and record retrieval/evidence diagnostics only",
    )
    parser.add_argument("--repeat-id", default="1", help="repeat label for stability experiments")
    parser.add_argument("--execution-order", choices=["baseline-first", "candidate-first"], default="baseline-first")
    parser.add_argument("--strategy-config", type=Path, default=None, help="JSON strategy override for this variant")
    parser.add_argument("--output", type=Path, default=Path("data/output/experiments/dynamic-eval.json"))
    args = parser.parse_args()

    config.validate_experiment_model_contract()
    strategy_config = load_strategy_file(str(args.strategy_config)) if args.strategy_config else normalize_strategy()

    rows = load_dataset(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]
    results = []
    changed_variable = args.changed_variable or ("search_action" if args.search_action else "none")
    hypothesis = args.hypothesis or "Only change the retrieval strategy and measure evidence quality."
    side_effects = args.side_effect or ["latency", "failure_rate", "source_coverage"]
    for index, case in enumerate(rows, start=1):
        task_id = _safe(f"{args.experiment_id}_{args.repeat_id}_{args.variant}_{case['question_id']}")
        task_dir = Path(config.DATA_PIPELINE["output_directory"]) / task_id
        record_task(task_id, case["question"], "running", str(task_dir))
        metadata = {
            "experiment_id": args.experiment_id,
            "variant": args.variant,
            "baseline_run_id": args.baseline_run_id,
            "dataset_version": case["dataset_version"],
            "persona": case["persona"],
            "domain": case["domain"],
            "task_type": case["task_type"],
            "difficulty": case["difficulty"],
            "hypothesis": hypothesis,
            "changed_variable": changed_variable,
            "expected_metrics": case.get("recommended_metrics") or [],
            "side_effects": side_effects,
            "acceptance_criteria": case.get("acceptance_criteria") or [],
            "rejection_criteria": case.get("rejection_criteria") or [],
            "risk_checks": case.get("risk_checks") or [],
            "prompt_version": config.get_prompt_version(),
            "search_action": args.search_action,
            "baseline_search_action": args.baseline_search_action,
            "repeat_id": args.repeat_id,
            "execution_order": args.execution_order,
            "retrieval_only": args.retrieval_only,
            "strategy_config": strategy_config,
        }
        task_token = current_task_id.set(task_id)
        try:
            try:
                with task_time_budget(task_id), analysis_slot(task_id):
                    answer = Orchestrator().run(case["question"], task_id=task_id, run_metadata=metadata)
                run_trace = reconstruct_run(task_id)
                final_events = [event for event in run_trace.get("events") or [] if event.get("type") == "final"]
                if not final_events:
                    append_task_event(
                        task_id,
                        "final",
                        status="failed",
                        error_type="missing_final_event",
                        content=(
                            str(answer or "").strip()
                            or "The task ended without a final result."
                        ),
                        action="dynamic_eval_runner",
                        mode="runtime_contract_error",
                    )
                    run_trace = reconstruct_run(task_id)
                    final_events = [
                        event
                        for event in run_trace.get("events") or []
                        if event.get("type") == "final"
                    ]
                terminal_status = str(final_events[-1].get("status") or "failed")
                record_task(task_id, case["question"], terminal_status, str(task_dir))
                finalize_manifest(task_id, status=terminal_status)
            except TaskTimeoutError as exc:
                if not any(event.get("type") == "final" for event in get_task_events(task_id)):
                    append_task_event(task_id, "error", phase="RUNTIME", error=str(exc)[:1000], error_class="timeout")
                    append_task_event(
                        task_id,
                        "final",
                        status="timed_out",
                        error_type="timeout",
                        content="The task timed out before a reliable answer could be completed.",
                        error_class="timeout",
                    )
                record_task(task_id, case["question"], "timed_out", str(task_dir), str(exc))
                finalize_manifest(task_id, status="timed_out", error=str(exc))
            except Exception as exc:
                if not any(event.get("type") == "final" for event in get_task_events(task_id)):
                    append_task_event(
                        task_id,
                        "error",
                        phase="RUNTIME",
                        error=f"{type(exc).__name__}: {exc}"[:1000],
                        error_class="runtime",
                    )
                    append_task_event(
                        task_id,
                        "final",
                        status="failed",
                        error_type="runtime_error",
                        content="The task encountered a runtime failure before a reliable answer could be completed.",
                        error_class="runtime",
                    )
                record_task(task_id, case["question"], "failed", str(task_dir), str(exc))
                finalize_manifest(task_id, status="failed", error=str(exc))
        finally:
            current_task_id.reset(task_token)
        trace = reconstruct_run(task_id)
        trace_validation = validate_replay_trace(trace)
        reference_validation = validate_reference_case(case)
        results.append(
            {
                "case": case,
                "task_id": task_id,
                "trace": trace,
                "trace_validation": trace_validation,
                "reference_validation": reference_validation,
                "score": score_trace(trace, case),
            }
        )
        print(
            json.dumps(
                {"progress": f"{index}/{len(rows)}", "task_id": task_id, "status": trace["manifest"].get("status")},
                ensure_ascii=False,
            ),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": args.experiment_id,
        "variant": args.variant,
        "repeat_id": args.repeat_id,
        "execution_order": args.execution_order,
        "dataset": str(args.dataset),
        "dataset_version": rows[0]["dataset_version"] if rows else "",
        "model": config.get_experiment_model_config(),
        "prompt_version": config.get_prompt_version(),
        "changed_variable": changed_variable,
        "hypothesis": hypothesis,
        "side_effects": side_effects,
        "retrieval_only": args.retrieval_only,
        "strategy_config": strategy_config,
        "sample_count": len(results),
        "trace_validation": {
            "valid_count": sum(item["trace_validation"].get("valid") is True for item in results),
            "invalid_count": sum(item["trace_validation"].get("valid") is not True for item in results),
            "issues": sorted(
                {
                    issue
                    for item in results
                    for issue in item["trace_validation"].get("issues") or []
                }
            ),
        },
        "risk_validation": {
            "high_risk_count": sum(bool(item["trace"].get("risk_validation", {}).get("high_risk")) for item in results),
            "invalid_count": sum(
                bool(item["trace"].get("risk_validation", {}).get("high_risk"))
                and item["trace"].get("risk_validation", {}).get("valid") is False
                for item in results
            ),
            "issues": sorted({
                issue
                for item in results
                for issue in item["trace"].get("risk_validation", {}).get("issues") or []
            }),
        },
        "reference_validation": {
            "sample_count": len(results),
            "ready_count": sum(item["reference_validation"].get("ready_for_scoring") is True for item in results),
            "pending_count": sum(item["reference_validation"].get("status") == "pending" for item in results),
            "stale_count": sum(item["reference_validation"].get("status") == "stale" for item in results),
            "invalid_count": sum(item["reference_validation"].get("status") == "invalid" for item in results),
            "all_ready": bool(results) and all(item["reference_validation"].get("ready_for_scoring") is True for item in results),
            "issues": sorted({
                issue
                for item in results
                for issue in item["reference_validation"].get("issues") or []
            }),
        },
        "results": results,
    }
    payload["model"]["api_key_configured"] = bool(config.get_llm_api_key())
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"ARTIFACT {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
