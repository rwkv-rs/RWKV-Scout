"""Run the deterministic prompt-injection harness and audit recorded answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.adversarial_eval import ADVERSARIAL_CASES, audit_fixture, summarize_adversarial_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, default=[], help="recorded experiment artifact; repeatable")
    parser.add_argument("--require-answers", action="store_true", help="fail unless all I01-I04 have recorded final answers")
    parser.add_argument("--output", type=Path, default=Path("data/output/experiments/adversarial-eval.json"))
    args = parser.parse_args()

    fixtures = [audit_fixture(item["variant"]) for item in ADVERSARIAL_CASES]
    artifacts = []
    load_errors = []
    for path in args.artifact:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("artifact must be a JSON object")
            artifacts.append(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            load_errors.append({"path": str(path), "error": str(exc)})

    answers = summarize_adversarial_artifact(artifacts)
    expected = len(ADVERSARIAL_CASES)
    output = {
        "artifact_version": "adversarial-eval.v1",
        "scope": "fixture-boundary-and-recorded-answer-audit",
        "quality_gate": "recorded answers only; no synthetic answer is scored",
        "fixture_count": len(fixtures),
        "fixture_fail_count": sum(item["status"] != "pass" for item in fixtures),
        "fixtures": fixtures,
        "answer_audit": answers,
        "artifacts": [str(path) for path in args.artifact],
        "load_errors": load_errors,
        "status": "pass" if not load_errors and not any(item["status"] != "pass" for item in fixtures) else "fail",
    }
    if args.require_answers:
        complete = answers["sample_count"] >= expected and answers["fail_count"] == 0 and answers["pending_count"] == 0
        output["status"] = "pass" if output["status"] == "pass" and complete else "pending"
        output["require_answers_satisfied"] = complete
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": output["status"], "answer_audit": answers}, ensure_ascii=False))
    return 0 if output["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
