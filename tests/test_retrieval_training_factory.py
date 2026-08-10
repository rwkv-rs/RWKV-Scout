from __future__ import annotations

import json
from pathlib import Path

from scripts.build_retrieval_rst_bootstrap import build_bootstrap
from scripts.build_retrieval_rst_pilot import build_pilot
from utils.retrieval_training_factory import (
    contamination_scan,
    export_training_records,
    select_diverse_tasks,
    validate_bundle,
    validate_pool,
)


def _build(tmp_path: Path) -> Path:
    root = tmp_path / "pilot"
    manifest = build_pilot(root, replace=False)
    assert manifest["task_count"] == 5
    return root


def test_pilot_pool_is_oracle_and_contract_valid(tmp_path: Path) -> None:
    root = _build(tmp_path)

    manifest = validate_pool(root / "tasks", report_dir=root / "validation")

    assert manifest["task_count"] == 5
    assert manifest["accepted_count"] == 5
    assert manifest["rejected_count"] == 0
    for task in manifest["tasks"]:
        assert task["provenance"]["bundle_sha256"]
        assert task["metrics"]["claim_count"] >= 1
        assert task["metrics"]["oracle_cross_validation_count"] >= 1


def test_exports_keep_tool_outputs_unmasked_and_failures_out_of_sft(
    tmp_path: Path,
) -> None:
    root = _build(tmp_path)
    manifest = validate_pool(root / "tasks")

    exported = export_training_records(manifest, root / "exports")

    assert exported["trajectory_sft"] == 5
    assert exported["stage_sft"] == 22
    assert exported["preference"] == 3
    assert exported["verifier_rl"] == 5
    trajectories = [
        json.loads(line)
        for line in (root / "exports/trajectory_sft.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(
        message["loss_mask"] is False
        for row in trajectories
        for message in row["messages"]
        if message["role"] in {"system", "user", "tool"}
    )
    assert all(
        "failure_class" not in json.dumps(row, ensure_ascii=False)
        for row in trajectories
    )


def test_contamination_scan_rejects_exact_thirteen_token_overlap() -> None:
    prompt = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"

    result = contamination_scan(
        "prefix one two three four five six seven eight nine ten eleven twelve thirteen suffix",
        [{"id": "held-out", "path": "eval.json", "text": prompt}],
    )

    assert result["exact_matches"]
    assert result["exact_matches"][0]["id"] == "held-out"


def test_validator_rejects_ungrounded_private_quote(tmp_path: Path) -> None:
    root = _build(tmp_path)
    bundle = root / "tasks/rrst_seed_lumen_current_001"
    contract_path = bundle / "private/contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["claims"][0]["quotes"][0]["text"] = "a sentence absent from every source"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = validate_bundle(bundle)

    assert report.accepted is False
    assert "claim_quote_not_grounded" in {issue.code for issue in report.issues}


def test_validator_rejects_answer_leak_in_public_instruction(tmp_path: Path) -> None:
    root = _build(tmp_path)
    bundle = root / "tasks/rrst_seed_lumen_current_001"
    instruction_path = bundle / "instruction.md"
    instruction_path.write_text(
        instruction_path.read_text(encoding="utf-8") + " The expected answer is Orbit After Rain.\n",
        encoding="utf-8",
    )

    report = validate_bundle(bundle)

    assert report.accepted is False
    assert "answer_leaked_in_instruction" in {issue.code for issue in report.issues}


def test_validator_rejects_impossible_source_timeline(tmp_path: Path) -> None:
    root = _build(tmp_path)
    bundle = root / "tasks/rrst_seed_lumen_current_001"
    sources_path = bundle / "environment/sources.jsonl"
    sources = [
        json.loads(line)
        for line in sources_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources[0]["retrieved_at"] = "2030-01-01T00:00:00Z"
    sources_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sources),
        encoding="utf-8",
    )

    report = validate_bundle(bundle)

    assert report.accepted is False
    assert "source_published_after_retrieval" in {
        issue.code for issue in report.issues
    }


def test_diversity_selection_applies_domain_caps(tmp_path: Path) -> None:
    root = _build(tmp_path)
    manifest = validate_pool(root / "tasks")

    selected = select_diverse_tasks(manifest, limit=10, max_per_domain=1)

    domains = [row["domain"] for row in selected]
    assert len(selected) == 4
    assert len(domains) == len(set(domains))


def test_fictional_bootstrap_pool_validates_without_benchmark_answers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bootstrap"
    build_manifest = build_bootstrap(root, replace=False)

    manifest = validate_pool(
        root / "tasks",
        evaluation_paths=[
            Path("data/evaluation/retrieval_required_100_20260808.json")
        ],
    )

    assert build_manifest["task_count"] == 20
    assert build_manifest["fictional_data_only"] is True
    assert build_manifest["benchmark_answers_used"] is False
    assert build_manifest["model_service_used"] is False
    assert manifest["accepted_count"] == 20
    assert manifest["rejected_count"] == 0
    assert all(
        not task["metrics"]["contamination"]["exact_matches"]
        for task in manifest["tasks"]
    )
    source_files = sorted((root / "tasks").glob("*/environment/sources.jsonl"))
    assert len(source_files) == 20
    assert all(".invalid" in path.read_text(encoding="utf-8") for path in source_files)
