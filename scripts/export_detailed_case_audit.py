"""Export lossless, per-case audit bundles from retrieval evaluation results.

The acceptance runner already persists every task event.  This exporter keeps
those event payloads verbatim while grouping them into reviewable stages, so a
reviewer can inspect retrieval decisions, fetched page/chunk evidence, Claim
Ledger admission, final context, and every model prompt/visible output without
loading one very large suite file.  Runtime-internal hidden state is not
available; literal ``<think>`` text returned in the model's visible completion
is preserved verbatim like every other output token.

Gold/reference fields are copied only into the post-run review section.  They
are never used to alter an answer or an execution trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "detailed-case-audit.v1"

_PLANNING_EVENTS = {
    "user_input",
    "run_started",
    "research_intake",
    "task_plan",
    "task_replan",
    "retrieval_strategy_selected",
    "research_loop_started",
    "planner_session_rebuilt",
    "task_replan_attempt",
    "task_replan_limit_reached",
}
_RETRIEVAL_EVENTS = {
    "model_tool_decision",
    "tool_call",
    "tool_result",
    "web_search_stage",
    "web_search_chunk",
    "page_fetch",
    "page_extract",
    "page_chunk",
    "page_chunk_candidate",
    "page_candidate_merge",
    "web_candidate_pool_shadow",
    "retrieval_ledger",
    "retrieval_fork_started",
    "retrieval_fork_completed",
}
_EVIDENCE_EVENTS = {
    "completion_gate",
    "completion_judgement",
    "ranking",
    "context_build",
    "synthesis",
    "evidence_validation",
    "citation_validation",
    "risk_validation",
    "answer_fact_validation",
}
_OUTPUT_EVENTS = {
    "final",
    "step_limit_reached",
    "case_timeout",
    "error",
    "provider_error",
}
_RUNTIME_EVENTS = {
    "runtime_gate",
    "model_request_gate",
    "progress",
}


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "case"))
    return text.strip("._") or "case"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _events(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace = case.get("trace") if isinstance(case.get("trace"), Mapping) else {}
    values = trace.get("events") if isinstance(trace, Mapping) else []
    return [dict(event) for event in values or [] if isinstance(event, Mapping)]


def _event_group(event_type: str) -> str:
    if event_type == "model_call":
        return "model_io"
    if event_type in _PLANNING_EVENTS:
        return "planning"
    if event_type in _RETRIEVAL_EVENTS:
        return "retrieval"
    if event_type in _EVIDENCE_EVENTS:
        return "evidence_and_synthesis"
    if event_type in _OUTPUT_EVENTS:
        return "validation_and_output"
    if event_type in _RUNTIME_EVENTS:
        return "runtime_and_concurrency"
    return "other"


def _group_events(events: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    groups = {
        "planning": [],
        "retrieval": [],
        "model_io": [],
        "evidence_and_synthesis": [],
        "validation_and_output": [],
        "runtime_and_concurrency": [],
        "other": [],
    }
    timeline: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        event_type = str(event.get("type") or "unknown")
        group = _event_group(event_type)
        groups[group].append(event)
        timeline.append(
            {
                "event_index": event_index,
                "seq": event.get("seq"),
                "timestamp": event.get("timestamp"),
                "type": event_type,
                "group": group,
            }
        )
    return groups, timeline


def _last_event(events: Iterable[Mapping[str, Any]], event_type: str) -> dict[str, Any]:
    for event in reversed(list(events)):
        if str(event.get("type") or "") == event_type:
            return dict(event)
    return {}


def _final_answer(case: Mapping[str, Any], final_event: Mapping[str, Any]) -> str:
    for key in ("answer", "final_output"):
        value = case.get(key)
        if value is not None:
            return str(value)
    return str(final_event.get("content") or "")


def _retrieval_queries(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending: dict[tuple[str, str], list[int]] = {}

    def key_for(query: Any, action: Any) -> tuple[str, str]:
        return (
            re.sub(r"\s+", " ", str(query or "")).strip().casefold(),
            str(action or "").strip().casefold(),
        )

    for event in events:
        event_type = str(event.get("type") or "")
        if event_type == "model_tool_decision":
            args = event.get("args") if isinstance(event.get("args"), Mapping) else {}
            query = args.get("query") or args.get("q") or ""
            if query:
                row_index = len(rows)
                rows.append(
                    {
                        "seq": event.get("seq"),
                        "source": "model_tool_decision",
                        "phase": event.get("phase"),
                        "query": query,
                        "action": event.get("action") or "",
                        "task_point_id": event.get("task_point_id") or "",
                    }
                )
                pending.setdefault(key_for(query, event.get("action")), []).append(row_index)
        elif event_type == "web_search_stage" and str(event.get("stage") or "") == "start":
            query = event.get("query") or ""
            action = event.get("action") or ""
            pending_rows = pending.get(key_for(query, action)) or []
            if pending_rows:
                row = rows[pending_rows.pop(0)]
                row.update(
                    {
                        "decision_seq": row.get("seq"),
                        "execution_seq": event.get("seq"),
                        "seq": event.get("seq"),
                        "source": "model_tool_decision+web_search_stage",
                        "phase": event.get("phase") or row.get("phase"),
                        "budget": event.get("budget") or {},
                    }
                )
                continue
            rows.append(
                {
                    "seq": event.get("seq"),
                    "execution_seq": event.get("seq"),
                    "source": "web_search_stage",
                    "phase": event.get("phase"),
                    "query": query,
                    "action": action,
                    "budget": event.get("budget") or {},
                }
            )
    return rows


def _latest_claim_ledger(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    for event in reversed(list(events)):
        ledger = event.get("claim_ledger")
        if isinstance(ledger, Mapping):
            return dict(ledger)
    return {}


def _latest_synthesis(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return _last_event(events, "synthesis")


def _nonempty_payload(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple)):
        return bool(value)
    return value is not None


def _normalized_quote(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _chunk_candidate_metrics(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for event in events:
        if str(event.get("type") or "") not in {
            "web_search_chunk",
            "page_chunk",
            "page_chunk_candidate",
        }:
            continue
        candidate = event.get("candidate")
        if isinstance(candidate, Mapping):
            rows.append(event)

    supported = [
        event
        for event in rows
        if (event.get("candidate") or {}).get("supported") is True
    ]
    quoted = [
        event
        for event in supported
        if _nonempty_payload((event.get("candidate") or {}).get("quote"))
    ]
    exact_quote_count = 0
    for event in quoted:
        candidate = event.get("candidate") or {}
        chunk = event.get("chunk") if isinstance(event.get("chunk"), Mapping) else {}
        quote = _normalized_quote(candidate.get("quote"))
        body = _normalized_quote(chunk.get("text") or event.get("text"))
        if quote and body and quote in body:
            exact_quote_count += 1

    empty_raw_output_seqs = [
        event.get("seq")
        for event in rows
        if "raw_output" in (event.get("candidate") or {})
        and not _nonempty_payload((event.get("candidate") or {}).get("raw_output"))
    ]
    return {
        "candidate_count": len(rows),
        "supported_count": len(supported),
        "unsupported_count": len(rows) - len(supported),
        "nonempty_quote_count": len(quoted),
        "exact_quote_count": exact_quote_count,
        "nonexact_quote_count": len(quoted) - exact_quote_count,
        "empty_raw_output_count": len(empty_raw_output_seqs),
        "empty_raw_output_seqs": empty_raw_output_seqs,
    }


def _post_gate_candidate_metrics(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the source-grounding boundary after raw RWKV extraction."""

    page_rows: list[dict[str, Any]] = []
    all_candidates: list[Mapping[str, Any]] = []
    merged_count = 0
    for event in events:
        if str(event.get("type") or "") != "page_candidate_merge":
            continue
        candidates = [
            candidate
            for candidate in event.get("chunk_candidates") or []
            if isinstance(candidate, Mapping)
        ]
        merged = [
            candidate
            for candidate in event.get("candidates") or []
            if isinstance(candidate, Mapping)
        ]
        all_candidates.extend(candidates)
        merged_count += len(merged)
        page_rows.append(
            {
                "seq": event.get("seq"),
                "url": event.get("url") or "",
                "chunk_candidate_count": len(candidates),
                "grounded_supported_count": sum(
                    candidate.get("supported") is True
                    and candidate.get("source_grounded") is True
                    for candidate in candidates
                ),
                "rejected_ungrounded_count": sum(
                    str(candidate.get("rejection_reason") or "")
                    in {"model_quote_not_grounded", "deterministic_quote_not_grounded"}
                    for candidate in candidates
                ),
                "merged_candidate_count": len(merged),
            }
        )

    reasons = Counter(
        str(candidate.get("rejection_reason") or "")
        for candidate in all_candidates
        if str(candidate.get("rejection_reason") or "")
    )
    return {
        "page_count": len(page_rows),
        "chunk_candidate_count": len(all_candidates),
        "supported_count": sum(
            candidate.get("supported") is True for candidate in all_candidates
        ),
        "grounded_supported_count": sum(
            candidate.get("supported") is True
            and candidate.get("source_grounded") is True
            for candidate in all_candidates
        ),
        "rejected_ungrounded_count": sum(
            str(candidate.get("rejection_reason") or "")
            in {"model_quote_not_grounded", "deterministic_quote_not_grounded"}
            for candidate in all_candidates
        ),
        "merged_candidate_count": merged_count,
        "rejection_reasons": dict(reasons),
        "pages": page_rows,
    }


def _manual_review_template() -> dict[str, Any]:
    return {
        "verdict": "unreviewed",
        "strict_pass": None,
        "semantic_pass": None,
        "scores": {
            "answer_correctness_0_to_4": None,
            "completeness_0_to_4": None,
            "evidence_grounding_0_to_4": None,
            "retrieval_relevance_0_to_4": None,
            "source_authority_0_to_4": None,
            "citation_correctness_0_to_4": None,
            "language_quality_0_to_4": None,
        },
        "checks": {
            "question_fully_understood": None,
            "retrieval_queries_appropriate": None,
            "selected_pages_relevant": None,
            "selected_chunks_support_claims": None,
            "claim_ledger_complete_and_correct": None,
            "final_context_preserves_required_evidence": None,
            "model_output_adds_no_unsupported_facts": None,
            "citations_bind_to_correct_spans": None,
            "answer_directly_addresses_question": None,
        },
        "failure_classes": [],
        "unsupported_claims": [],
        "missing_claims": [],
        "notes": "",
    }


def build_case_audit(
    case: Mapping[str, Any],
    *,
    source_path: str = "",
    source_case_index: int = 0,
) -> dict[str, Any]:
    """Build one lossless, stage-grouped audit record."""

    events = _events(case)
    groups, timeline = _group_events(events)
    final_event = _last_event(events, "final")
    synthesis = _latest_synthesis(events)
    claim_ledger = _latest_claim_ledger(events)
    answer = _final_answer(case, final_event)
    selected_evidence = synthesis.get("selected_evidence") or []
    context_text = synthesis.get("context_text") or ""
    reference_answer = case.get("reference_answer") or case.get("final_answer") or ""
    event_counts = Counter(str(event.get("type") or "unknown") for event in events)
    case_id = str(case.get("case_id") or case.get("id") or f"case_{source_case_index + 1:03d}")
    chunk_candidate_metrics = _chunk_candidate_metrics(groups["retrieval"])
    post_gate_candidate_metrics = _post_gate_candidate_metrics(groups["retrieval"])
    empty_model_output_seqs = [
        event.get("seq")
        for event in groups["model_io"]
        if "output" in event and not _nonempty_payload(event.get("output"))
    ]
    missing_model_output_seqs = [
        event.get("seq")
        for event in groups["model_io"]
        if "output" not in event
    ]
    failed_model_calls = [
        event
        for event in groups["model_io"]
        if str(event.get("status") or "").casefold() == "failed"
    ]
    model_lane_counts = Counter(
        str(event.get("model_lane") or "unknown") for event in groups["model_io"]
    )
    model_operation_counts = Counter(
        str(event.get("operation") or "unknown") for event in groups["model_io"]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "result_path": source_path,
            "case_index": source_case_index,
            "case_sha256": _canonical_hash(case),
        },
        "case": {
            "case_id": case_id,
            "benchmark_id": case.get("benchmark_id") or case.get("id") or case_id,
            "task_id": case.get("task_id") or "",
            "query": case.get("query") or case.get("question") or "",
            "architecture": case.get("architecture") or "",
            "status": case.get("status") or "",
            "final_status": case.get("final_status") or final_event.get("status") or "",
            "error": case.get("error") or "",
            "error_type": case.get("error_type") or final_event.get("error_type") or "",
            "failure_reason": case.get("failure_reason") or "",
        },
        "post_run_reference": {
            "reference_answer": reference_answer,
            "gold": case.get("gold") if isinstance(case.get("gold"), Mapping) else None,
            "usage_boundary": "Post-run review only; never supplied to retrieval or model execution.",
        },
        "answer": {
            "content": answer,
            "chars": len(answer),
            "blank": not bool(answer.strip()),
            "final_event": final_event,
        },
        "retrieval_summary": {
            "queries": _retrieval_queries(events),
            "page_fetch_count": event_counts.get("page_fetch", 0),
            "page_extract_count": event_counts.get("page_extract", 0),
            "page_chunk_count": event_counts.get("web_search_chunk", 0)
            + event_counts.get("page_chunk", 0),
            "chunk_candidates": chunk_candidate_metrics,
            "post_gate_candidates": post_gate_candidate_metrics,
        },
        "evidence_chain": {
            "claim_ledger": claim_ledger,
            "selected_evidence": selected_evidence,
            "final_context_text": context_text,
            "final_context_stats": synthesis.get("context_stats") or {},
            "synthesis_generation_attempts": synthesis.get("generation_attempts") or [],
            "synthesis_answer_quality": synthesis.get("answer_quality") or {},
        },
        # Every original event appears exactly once in these groups.  Prompt,
        # output, page body, chunk text, candidate JSON, and validation payloads
        # are intentionally not shortened or redacted.
        "events": groups,
        "timeline": timeline,
        "integrity": {
            "event_count": len(events),
            "grouped_event_count": sum(len(values) for values in groups.values()),
            "all_events_accounted_for": len(events)
            == sum(len(values) for values in groups.values()),
            "event_type_counts": dict(event_counts),
            "model_call_count": event_counts.get("model_call", 0),
            "model_calls_with_prompt": sum(
                "prompt" in event or "input_messages" in event
                for event in groups["model_io"]
            ),
            "model_calls_with_output": sum("output" in event for event in groups["model_io"]),
            "model_calls_with_nonempty_prompt": sum(
                _nonempty_payload(event.get("prompt"))
                or _nonempty_payload(event.get("input_messages"))
                for event in groups["model_io"]
            ),
            "model_calls_with_nonempty_output": sum(
                _nonempty_payload(event.get("output"))
                for event in groups["model_io"]
            ),
            "model_calls_with_empty_output": len(empty_model_output_seqs),
            "model_call_empty_output_seqs": empty_model_output_seqs,
            "model_calls_without_output": len(missing_model_output_seqs),
            "model_call_without_output_seqs": missing_model_output_seqs,
            "model_failed_call_count": len(failed_model_calls),
            "model_failed_call_seqs": [event.get("seq") for event in failed_model_calls],
            "model_failed_call_errors": [
                str(event.get("error") or event.get("message") or "")
                for event in failed_model_calls
            ],
            "model_lane_counts": dict(model_lane_counts),
            "model_operation_counts": dict(model_operation_counts),
            "model_calls_with_generation_settings": sum(
                "request_max_tokens" in event or "stop" in event
                for event in groups["model_io"]
            ),
            "selected_evidence_count": len(selected_evidence),
            "claim_count": int(claim_ledger.get("claim_count") or 0),
        },
        "manual_review": _manual_review_template(),
    }


def _payload_cases(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        values = payload.get("cases")
        if not isinstance(values, list):
            values = payload.get("results")
        if isinstance(values, list):
            return [dict(value) for value in values if isinstance(value, Mapping)]
        if isinstance(payload.get("trace"), Mapping):
            return [dict(payload)]
    if isinstance(payload, list):
        return [dict(value) for value in payload if isinstance(value, Mapping)]
    raise ValueError(f"result file contains no cases: {path}")


def discover_result_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        path = value.resolve()
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*.result.json")))
        else:
            raise FileNotFoundError(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ValueError("no result JSON files found")
    return unique


def export_audit_bundle(result_paths: Iterable[Path], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    used_names: Counter[str] = Counter()

    for result_path in result_paths:
        resolved = result_path.resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        cases = _payload_cases(payload, resolved)
        source_rows.append(
            {
                "path": str(resolved),
                "sha256": _file_hash(resolved),
                "case_count": len(cases),
            }
        )
        for case_index, case in enumerate(cases):
            audit = build_case_audit(
                case,
                source_path=str(resolved),
                source_case_index=case_index,
            )
            base = _safe_name(
                f"{resolved.stem}__{case_index + 1:04d}__{audit['case']['case_id']}"
            )
            used_names[base] += 1
            suffix = f"__{used_names[base]}" if used_names[base] > 1 else ""
            audit_name = f"{base}{suffix}.audit.json"
            audit_path = cases_dir / audit_name
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            relative_path = audit_path.relative_to(output_dir).as_posix()
            row = {
                "case_id": audit["case"]["case_id"],
                "benchmark_id": audit["case"]["benchmark_id"],
                "query": audit["case"]["query"],
                "status": audit["case"]["status"],
                "final_status": audit["case"]["final_status"],
                "answer_chars": audit["answer"]["chars"],
                "blank_answer": audit["answer"]["blank"],
                "model_call_count": audit["integrity"]["model_call_count"],
                "model_nonempty_output_count": audit["integrity"][
                    "model_calls_with_nonempty_output"
                ],
                "model_empty_output_count": audit["integrity"][
                    "model_calls_with_empty_output"
                ],
                "model_no_output_count": audit["integrity"][
                    "model_calls_without_output"
                ],
                "model_failed_call_count": audit["integrity"][
                    "model_failed_call_count"
                ],
                "retrieval_query_count": len(audit["retrieval_summary"]["queries"]),
                "chunk_candidate_count": audit["retrieval_summary"]["chunk_candidates"][
                    "candidate_count"
                ],
                "chunk_supported_count": audit["retrieval_summary"]["chunk_candidates"][
                    "supported_count"
                ],
                "chunk_exact_quote_count": audit["retrieval_summary"]["chunk_candidates"][
                    "exact_quote_count"
                ],
                "post_gate_grounded_count": audit["retrieval_summary"]["post_gate_candidates"][
                    "grounded_supported_count"
                ],
                "post_gate_rejected_ungrounded_count": audit["retrieval_summary"]["post_gate_candidates"][
                    "rejected_ungrounded_count"
                ],
                "selected_evidence_count": audit["integrity"]["selected_evidence_count"],
                "claim_count": audit["integrity"]["claim_count"],
                "audit_file": relative_path,
                "audit_sha256": _file_hash(audit_path),
                "review_verdict": "unreviewed",
            }
            manifest_rows.append(row)
            review_rows.append(
                {
                    "case_id": row["case_id"],
                    "benchmark_id": row["benchmark_id"],
                    "query": row["query"],
                    "audit_file": relative_path,
                    **_manual_review_template(),
                }
            )

    status_counts = Counter(str(row.get("status") or "unknown") for row in manifest_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_files": source_rows,
        "case_count": len(manifest_rows),
        "status_counts": dict(status_counts),
        "blank_answer_count": sum(bool(row["blank_answer"]) for row in manifest_rows),
        "unreviewed_count": len(manifest_rows),
        "cases": manifest_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "manual_review.jsonl").open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Detailed retrieval audit",
        "",
        f"Cases: {len(manifest_rows)}",
        "",
        "Automated execution status is not a correctness verdict. Strict and semantic pass remain unreviewed until the evidence chain and answer are checked.",
        "",
        "| Case | Status | Answer chars | Model outputs | Empty outputs | No output | Failed calls | Queries | Raw candidates | Raw supported | Raw exact | Grounded | Rejected ungrounded | Evidence | Review | Audit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in manifest_rows:
        label = str(row["case_id"]).replace("|", "\\|")
        lines.append(
            f"| {label} | {row['status']} | {row['answer_chars']} | "
            f"{row['model_nonempty_output_count']}/{row['model_call_count']} | "
            f"{row['model_empty_output_count']} | {row['model_no_output_count']} | "
            f"{row['model_failed_call_count']} | {row['retrieval_query_count']} | "
            f"{row['chunk_candidate_count']} | {row['chunk_supported_count']} | "
            f"{row['chunk_exact_quote_count']} | {row['post_gate_grounded_count']} | "
            f"{row['post_gate_rejected_ungrounded_count']} | {row['selected_evidence_count']} | unreviewed | "
            f"[{row['audit_file']}]({row['audit_file']}) |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Result JSON files or run directories")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = discover_result_paths(args.inputs)
    manifest = export_audit_bundle(paths, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "source_file_count": len(paths),
                "case_count": manifest["case_count"],
                "blank_answer_count": manifest["blank_answer_count"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
