#!/usr/bin/env python3
"""Rewrite structured historical state to stable RWKV-ECRA contracts.

The migration is deliberately limited to machine-readable JSON/JSONL objects.
Raw model prompt/output strings are immutable audit material and are not
rewritten to pretend an older model emitted the new protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.runtime_contracts import (
    EVIDENCE_LEDGER_CONTRACT,
    EVIDENCE_RECORD_SET_CONTRACT,
    EVIDENCE_RESOLUTION_CONTRACT,
    EVIDENCE_REVIEW_CONTRACT,
    MODEL_EXTRACTION_DIAGNOSTICS_CONTRACT,
    PLANNER_EVIDENCE_CONTRACT,
    RETRIEVAL_EVENT_LEDGER_CONTRACT,
    RETRIEVAL_INFRASTRUCTURE_CONTRACT,
    RETRIEVAL_OBJECT_CONTRACT,
    RETRIEVAL_QUERY_PLAN_CONTRACT,
    RETRIEVAL_ROUTING_CONTRACT,
    TASK_PLAN_CONTRACT,
    TOOL_CALL_CONTRACT,
)
from agent.task_plan_contract import (
    LEGACY_TASK_PLAN_SCHEMA_VERSIONS,
    normalize_task_plan,
    record_fields,
)


LEGACY_RUNTIME_CONTRACTS = {
    **{value: TASK_PLAN_CONTRACT for value in LEGACY_TASK_PLAN_SCHEMA_VERSIONS},
    "tool_call.v1": TOOL_CALL_CONTRACT,
    "retrieval-object.v1": RETRIEVAL_OBJECT_CONTRACT,
    "retrieval-object.v2": RETRIEVAL_OBJECT_CONTRACT,
    "task-query-fanout.v1": RETRIEVAL_QUERY_PLAN_CONTRACT,
    "claim-ledger.v1": EVIDENCE_LEDGER_CONTRACT,
    "evidence-ledger.v2": EVIDENCE_LEDGER_CONTRACT,
    "planner-records.v1": EVIDENCE_RECORD_SET_CONTRACT,
    "candidate-record-resolution.v2": EVIDENCE_RESOLUTION_CONTRACT,
    "rwkv-cross-validation.v1": EVIDENCE_REVIEW_CONTRACT,
    "rwkv-cross-validation.v2": EVIDENCE_REVIEW_CONTRACT,
    "rwkv-cross-validation.v3": EVIDENCE_REVIEW_CONTRACT,
    "rwkv-cross-validation.v4": EVIDENCE_REVIEW_CONTRACT,
    "planner-evidence.v1": PLANNER_EVIDENCE_CONTRACT,
    "retrieval-infrastructure.v1": RETRIEVAL_INFRASTRUCTURE_CONTRACT,
    "planner-routing.v1": RETRIEVAL_ROUTING_CONTRACT,
    "planner-routing.v2": RETRIEVAL_ROUTING_CONTRACT,
    "model-extraction-diagnostics.v1": MODEL_EXTRACTION_DIAGNOSTICS_CONTRACT,
    "retrieval_ledger.v1": RETRIEVAL_EVENT_LEDGER_CONTRACT,
}
SINGLE_RECORD_ID_KEYS = {
    "claim_id",
    "missing_point_id",
    "point_id",
    "task_point_id",
    "task_record_id",
}
MULTI_RECORD_ID_KEYS = {
    "affected_claim_ids",
    "claim_ids",
    "conflict_point_ids",
    "covered_point_ids",
    "missing_point_ids",
    "point_ids",
    "task_point_ids",
    "task_record_ids",
}
CANONICAL_RUNTIME_KEYS = {
    "claim_ledger": "evidence_ledger",
    "claims": "task_records",
    "claim_id": "task_record_id",
    "point_id": "task_record_id",
    "task_point_id": "task_record_id",
    "claim_ids": "task_record_ids",
    "point_ids": "task_record_ids",
    "task_point_ids": "task_record_ids",
    "affected_claim_ids": "affected_task_record_ids",
    "covered_point_ids": "covered_task_record_ids",
    "missing_point_id": "missing_task_record_id",
    "missing_point_ids": "missing_task_record_ids",
    "conflict_point_ids": "conflict_task_record_ids",
    "task_point_status": "task_record_status",
    "claim_count": "task_record_count",
    "candidate_record_count": "evidence_record_count",
}


def _looks_like_task_plan(value: Mapping[str, Any]) -> bool:
    schema = str(value.get("schema_version") or "")
    contract = str(value.get("contract") or "")
    return bool(
        isinstance(value.get("atomic_points"), list)
        or (
            (schema in LEGACY_TASK_PLAN_SCHEMA_VERSIONS or contract == TASK_PLAN_CONTRACT)
            and isinstance(value.get("records"), list)
        )
    )


def _fallback_goal(value: Mapping[str, Any]) -> str:
    goal = str(value.get("goal") or "").strip()
    if goal:
        return goal
    rows = value.get("atomic_points") or value.get("records") or []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in ("question", "task", "objective"):
            text = str(row.get(key) or "").strip()
            if text:
                return text
    return "Historical Task Plan"


def _collect_identity_maps(
    value: Any,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    record_pairs: list[tuple[str, str]] = []
    field_pairs: dict[str, list[tuple[str, str]]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if _looks_like_task_plan(node):
                raw_rows = [
                    row
                    for row in (node.get("atomic_points") or node.get("records") or [])
                    if isinstance(row, Mapping)
                ]
                canonical = normalize_task_plan(
                    node,
                    fallback_goal=_fallback_goal(node),
                    preserve_runtime_ids=str(node.get("contract") or "")
                    == TASK_PLAN_CONTRACT,
                )
                for index, (raw, record) in enumerate(
                    zip(raw_rows, canonical["records"]),
                    start=1,
                ):
                    if not isinstance(raw, Mapping):
                        continue
                    legacy_id = str(
                        raw.get("record_id") or raw.get("id") or f"P{index}"
                    ).strip()
                    canonical_id = record["record_id"]
                    if legacy_id:
                        record_pairs.append((legacy_id, canonical_id))
                    raw_fields = raw.get("fields") or raw.get("requested_fields") or []
                    if not raw_fields:
                        raw_fields = raw.get("evidence_needed") or []
                    raw_names = record_fields(raw)
                    for field_index, name in enumerate(raw_names, start=1):
                        canonical_field_id = f"{canonical_id}:F{field_index}"
                        field_pairs.setdefault(canonical_id, []).append(
                            (str(name), canonical_field_id)
                        )
                        if field_index <= len(raw_fields) and isinstance(
                            raw_fields[field_index - 1], Mapping
                        ):
                            legacy_field_id = str(
                                raw_fields[field_index - 1].get("field_id") or ""
                            ).strip()
                            if legacy_field_id:
                                field_pairs[canonical_id].append(
                                    (legacy_field_id, canonical_field_id)
                                )
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    record_counts = Counter(key for key, _ in record_pairs)
    record_map = {
        key: target
        for key, target in record_pairs
        if record_counts[key] == 1 or key == target
    }
    field_maps: dict[str, dict[str, str]] = {}
    for record_key, pairs in field_pairs.items():
        counts = Counter(key.casefold() for key, _ in pairs)
        field_maps[record_key] = {
            key.casefold(): target
            for key, target in pairs
            if counts[key.casefold()] == 1 or key == target
        }
    return record_map, field_maps


def migrate_document(
    value: Any,
    *,
    oracle_training: bool = False,
) -> tuple[Any, dict[str, int]]:
    """Return migrated data and deterministic migration counters."""

    record_map, field_maps = _collect_identity_maps(value)
    stats = {
        "task_plans": 0,
        "runtime_contracts": 0,
        "event_names": 0,
        "record_references": 0,
        "field_references": 0,
        "key_collisions": 0,
    }

    def map_record_id(raw: Any) -> Any:
        if not isinstance(raw, str):
            return raw
        mapped = record_map.get(raw, raw)
        if mapped != raw:
            stats["record_references"] += 1
        return mapped

    def visit(node: Any, *, active_record_id: str = "") -> Any:
        if isinstance(node, Mapping):
            if _looks_like_task_plan(node):
                stats["task_plans"] += 1
                return normalize_task_plan(
                    node,
                    fallback_goal=_fallback_goal(node),
                    preserve_runtime_ids=str(node.get("contract") or "")
                    == TASK_PLAN_CONTRACT,
                )

            record_context = active_record_id
            for key in ("task_record_id", "task_point_id", "claim_id", "point_id"):
                if isinstance(node.get(key), str):
                    record_context = str(record_map.get(node[key], node[key]))
                    break

            output: dict[str, Any] = {}
            canonical_input_keys = {
                str(key)
                for key in node
                if str(key) not in CANONICAL_RUNTIME_KEYS
            }

            def assign(target: str, migrated: Any, *, source: str) -> None:
                """Write one canonical key without iteration-order data loss."""

                if target not in output:
                    output[target] = migrated
                    return
                if output[target] == migrated:
                    return
                stats["key_collisions"] += 1
                # An explicitly canonical key always wins over a legacy alias,
                # regardless of which one appeared first in the input mapping.
                if source == target:
                    output[target] = migrated
                    return
                if target in canonical_input_keys:
                    return
                # Two legacy set-like aliases may safely retain their complete
                # transport identity.  Scalars and mappings keep the first
                # value rather than being silently overwritten by dict order.
                if isinstance(output[target], list) and isinstance(migrated, list):
                    output[target] = list(
                        dict.fromkeys([*output[target], *migrated])
                    )

            for key, child in node.items():
                target_key = str(key)
                if target_key in {"schema_version", "contract"}:
                    legacy_name = str(child or "")
                    formal_name = LEGACY_RUNTIME_CONTRACTS.get(legacy_name)
                    if formal_name:
                        assign("contract", formal_name, source=target_key)
                        if target_key != "contract" or legacy_name != formal_name:
                            stats["runtime_contracts"] += 1
                        continue
                if target_key == "claim_ledger":
                    assign(
                        CANONICAL_RUNTIME_KEYS[target_key],
                        visit(child, active_record_id=record_context),
                        source=target_key,
                    )
                    stats["runtime_contracts"] += 1
                    continue
                assessment_content = node.get("content")
                assessment_chosen = node.get("chosen")
                is_oracle_assessment = oracle_training or bool(
                    isinstance(assessment_content, Mapping)
                    and isinstance(
                        assessment_content.get("task_record_status")
                        or assessment_content.get("task_point_status"),
                        Mapping,
                    )
                ) or bool(
                    isinstance(assessment_chosen, Mapping)
                    and isinstance(
                        assessment_chosen.get("task_record_status")
                        or assessment_chosen.get("task_point_status"),
                        Mapping,
                    )
                )
                if target_key in {"type", "stage"} and child in {
                    "cross_validation",
                    "evidence_review",
                }:
                    assign(
                        target_key,
                        (
                            "oracle_evidence_assessment"
                            if is_oracle_assessment
                            else "evidence_review"
                        ),
                        source=target_key,
                    )
                    stats["event_names"] += 1
                    continue
                if target_key in {
                    "oracle_cross_validation_count",
                    "oracle_evidence_review_count",
                }:
                    assign(
                        "oracle_evidence_assessment_count",
                        visit(child, active_record_id=record_context),
                        source=target_key,
                    )
                    stats["event_names"] += 1
                    continue
                if (
                    target_key == "operator"
                    and child
                    in {
                        "add_cross_validation_protocol_repair",
                        "add_evidence_review_protocol_repair",
                    }
                ):
                    assign(
                        target_key,
                        "add_oracle_evidence_assessment_protocol_repair",
                        source=target_key,
                    )
                    stats["event_names"] += 1
                    continue
                if target_key == "extension" and isinstance(child, str):
                    renamed = child.replace(
                        "cross-validation",
                        "Oracle Evidence Assessment",
                    ).replace(
                        "Evidence Review",
                        "Oracle Evidence Assessment",
                    )
                    if renamed != child:
                        assign(target_key, renamed, source=target_key)
                        stats["event_names"] += 1
                        continue
                if target_key in SINGLE_RECORD_ID_KEYS:
                    canonical_key = CANONICAL_RUNTIME_KEYS.get(target_key, target_key)
                    assign(
                        canonical_key,
                        map_record_id(child),
                        source=target_key,
                    )
                    continue
                if target_key in MULTI_RECORD_ID_KEYS and isinstance(child, list):
                    canonical_key = CANONICAL_RUNTIME_KEYS.get(target_key, target_key)
                    assign(
                        canonical_key,
                        [map_record_id(item) for item in child],
                        source=target_key,
                    )
                    continue
                if target_key in {"field_keys", "field_ids"} and isinstance(child, list):
                    mapping = field_maps.get(record_context, {})
                    converted: list[Any] = []
                    for item in child:
                        mapped = mapping.get(str(item).casefold(), item)
                        if mapped != item:
                            stats["field_references"] += 1
                        converted.append(mapped)
                    assign("field_ids", converted, source=target_key)
                    continue
                canonical_key = CANONICAL_RUNTIME_KEYS.get(target_key, target_key)
                assign(
                    canonical_key,
                    visit(child, active_record_id=record_context),
                    source=target_key,
                )
            return output
        if isinstance(node, list):
            return [visit(child, active_record_id=active_record_id) for child in node]
        return node

    return visit(value), stats


def _load_path(path: Path) -> tuple[Any, str]:
    if path.suffix.casefold() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows, "jsonl"
    return json.loads(path.read_text(encoding="utf-8")), "json"


def _render(value: Any, kind: str, *, pretty: bool = True) -> str:
    if kind == "jsonl":
        return "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in value
        )
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _iter_paths(raw_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.casefold() in {".json", ".jsonl"}
            )
        elif path.is_file():
            files.append(path)
    return sorted(dict.fromkeys(files))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="JSON/JSONL files or directories")
    parser.add_argument(
        "--write",
        action="store_true",
        help="atomically replace files; default is a read-only audit",
    )
    args = parser.parse_args()

    totals = {
        "files": 0,
        "changed_files": 0,
        "task_plans": 0,
        "runtime_contracts": 0,
        "event_names": 0,
        "record_references": 0,
        "field_references": 0,
        "key_collisions": 0,
    }
    errors: list[str] = []
    for path in _iter_paths(args.paths):
        totals["files"] += 1
        try:
            original = path.read_text(encoding="utf-8")
            value, kind = _load_path(path)
            migrated, stats = migrate_document(
                value,
                oracle_training="retrieval_rst" in path.parts,
            )
            rendered = _render(
                migrated,
                kind,
                pretty=kind == "json" and "\n  " in original,
            )
            changed = rendered != original
            if changed and any(stats.values()):
                totals["changed_files"] += 1
                if args.write:
                    _write_atomic(path, rendered)
            for key in (
                "task_plans",
                "runtime_contracts",
                "event_names",
                "record_references",
                "field_references",
                "key_collisions",
            ):
                totals[key] += stats[key]
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    print(json.dumps({"write": args.write, **totals, "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
