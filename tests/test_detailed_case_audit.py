import json
from pathlib import Path

from scripts.export_detailed_case_audit import build_case_audit, export_audit_bundle


def _case() -> dict:
    prompt = "System exact\nUser: 检索这个事实"
    output = "<think>保留原始过程</think>\n精确输出"
    chunk_text = "CUDA 11.8\npip3 install torch --index-url https://example/cu118"
    ledger = {
        "schema_version": "claim-ledger.v1",
        "claim_count": 1,
        "claims": [{"claim_id": "P1", "status": "supported"}],
    }
    return {
        "case_id": "case/exact",
        "query": "应该使用哪个命令？",
        "status": "completed",
        "answer": "使用证据中的命令。",
        "reference_answer": "仅用于运行后审核",
        "trace": {
            "events": [
                {
                    "seq": 1,
                    "type": "model_call",
                    "operation": "page_evidence",
                    "model_lane": "chunk",
                    "request_max_tokens": 768,
                    "stop": ["### User"],
                    "finish_reason": "stop",
                    "prompt": prompt,
                    "output": output,
                },
                {
                    "seq": 2,
                    "type": "web_search_chunk",
                    "url": "https://example.test/install",
                    "prompt": "chunk prompt exact",
                    "chunk": {"chunk_id": "chunk-1", "text": chunk_text},
                    "candidate": {"raw_output": output},
                },
                {
                    "seq": 3,
                    "type": "synthesis",
                    "claim_ledger": ledger,
                    "context_text": chunk_text,
                    "selected_evidence": [
                        {
                            "url": "https://example.test/install",
                            "chunks": [{"chunk_id": "chunk-1", "text": chunk_text}],
                        }
                    ],
                    "generation_attempts": [
                        {"stage": "writer_initial", "output": output, "accepted": True}
                    ],
                },
                {"seq": 4, "type": "final", "status": "completed", "content": "使用证据中的命令。"},
            ]
        },
    }


def test_case_audit_preserves_exact_model_and_chunk_payloads():
    case = _case()
    audit = build_case_audit(case, source_path="input.result.json", source_case_index=0)

    model_event = audit["events"]["model_io"][0]
    retrieval_event = audit["events"]["retrieval"][0]
    assert model_event["prompt"] == case["trace"]["events"][0]["prompt"]
    assert model_event["output"] == case["trace"]["events"][0]["output"]
    assert retrieval_event["chunk"]["text"] == case["trace"]["events"][1]["chunk"]["text"]
    assert audit["evidence_chain"]["final_context_text"] == case["trace"]["events"][2]["context_text"]
    assert audit["evidence_chain"]["claim_ledger"] == case["trace"]["events"][2]["claim_ledger"]
    assert audit["integrity"]["all_events_accounted_for"] is True
    assert audit["integrity"]["grouped_event_count"] == 4
    assert audit["integrity"]["model_calls_with_nonempty_output"] == 1
    assert audit["integrity"]["model_lane_counts"] == {"chunk": 1}
    assert audit["integrity"]["model_operation_counts"] == {"page_evidence": 1}
    assert audit["integrity"]["model_calls_with_generation_settings"] == 1


def test_case_audit_counts_one_logical_query_for_decision_and_execution_events():
    case = _case()
    case["trace"]["events"] = [
        {
            "seq": 10,
            "type": "model_tool_decision",
            "phase": "DISCOVERY",
            "action": "web_search",
            "args": {"query": "site:nodejs.org fetch stable"},
            "task_point_id": "P1",
        },
        {
            "seq": 11,
            "type": "web_search_stage",
            "phase": "DISCOVERY",
            "stage": "start",
            "action": "web_search",
            "query": "site:nodejs.org fetch stable",
            "budget": {"max_pages": 8},
        },
        *case["trace"]["events"],
    ]

    audit = build_case_audit(case)

    assert len(audit["retrieval_summary"]["queries"]) == 1
    query = audit["retrieval_summary"]["queries"][0]
    assert query["decision_seq"] == 10
    assert query["execution_seq"] == 11
    assert query["source"] == "model_tool_decision+web_search_stage"


def test_case_audit_distinguishes_empty_model_outputs_and_exact_chunk_quotes():
    case = _case()
    case["trace"]["events"].insert(
        1,
        {
            "seq": 5,
            "type": "model_call",
            "operation": "page_evidence",
            "prompt": "non-empty prompt",
            "output": "",
        },
    )
    chunk_event = case["trace"]["events"][2]
    chunk_event["candidate"] = {
        "supported": True,
        "quote": "pip3 install torch --index-url https://example/cu118",
        "raw_output": "",
    }
    case["trace"]["events"].insert(
        3,
        {
            "seq": 6,
            "type": "page_candidate_merge",
            "url": "https://example.test/install",
            "candidates": [
                {
                    "chunk_id": "chunk-1",
                    "supported": True,
                    "source_grounded": True,
                    "quote": "pip3 install torch --index-url https://example/cu118",
                }
            ],
            "chunk_candidates": [
                {
                    "chunk_id": "chunk-1",
                    "supported": True,
                    "source_grounded": True,
                    "quote": "pip3 install torch --index-url https://example/cu118",
                },
                {
                    "chunk_id": "chunk-2",
                    "supported": False,
                    "source_grounded": False,
                    "rejection_reason": "model_quote_not_grounded",
                    "model_quote": "invented command",
                },
            ],
        },
    )

    audit = build_case_audit(case)

    assert audit["integrity"]["model_call_count"] == 2
    assert audit["integrity"]["model_calls_with_output"] == 2
    assert audit["integrity"]["model_calls_with_nonempty_output"] == 1
    assert audit["integrity"]["model_calls_with_empty_output"] == 1
    assert audit["integrity"]["model_call_empty_output_seqs"] == [5]
    assert audit["integrity"]["model_calls_without_output"] == 0
    assert audit["integrity"]["model_failed_call_count"] == 0
    assert audit["retrieval_summary"]["chunk_candidates"] == {
        "candidate_count": 1,
        "supported_count": 1,
        "unsupported_count": 0,
        "nonempty_quote_count": 1,
        "exact_quote_count": 1,
        "nonexact_quote_count": 0,
        "empty_raw_output_count": 1,
        "empty_raw_output_seqs": [2],
    }
    assert audit["retrieval_summary"]["post_gate_candidates"] == {
        "page_count": 1,
        "chunk_candidate_count": 2,
        "supported_count": 1,
        "grounded_supported_count": 1,
        "rejected_ungrounded_count": 1,
        "merged_candidate_count": 1,
        "rejection_reasons": {"model_quote_not_grounded": 1},
        "pages": [
            {
                "seq": 6,
                "url": "https://example.test/install",
                "chunk_candidate_count": 2,
                "grounded_supported_count": 1,
                "rejected_ungrounded_count": 1,
                "merged_candidate_count": 1,
            }
        ],
    }


def test_case_audit_reports_failed_model_calls_without_conflating_empty_output():
    case = _case()
    case["trace"]["events"].insert(
        1,
        {
            "seq": 7,
            "type": "model_call",
            "operation": "text_completion",
            "model_lane": "chunk",
            "status": "failed",
            "prompt": "exact failed prompt",
            "error": "ReadTimeout: request timed out",
        },
    )

    audit = build_case_audit(case)

    assert audit["integrity"]["model_call_count"] == 2
    assert audit["integrity"]["model_calls_with_empty_output"] == 0
    assert audit["integrity"]["model_calls_without_output"] == 1
    assert audit["integrity"]["model_call_without_output_seqs"] == [7]
    assert audit["integrity"]["model_failed_call_count"] == 1
    assert audit["integrity"]["model_failed_call_seqs"] == [7]
    assert audit["integrity"]["model_failed_call_errors"] == [
        "ReadTimeout: request timed out"
    ]


def test_bundle_writes_split_audits_manifest_and_review_template(tmp_path: Path):
    result_path = tmp_path / "sample.result.json"
    result_path.write_text(
        json.dumps({"cases": [_case()]}, ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"

    manifest = export_audit_bundle([result_path], output_dir)

    assert manifest["case_count"] == 1
    assert manifest["blank_answer_count"] == 0
    audit_path = output_dir / manifest["cases"][0]["audit_file"]
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    assert persisted["events"]["model_io"][0]["prompt"] == _case()["trace"]["events"][0]["prompt"]
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.md").is_file()
    review = json.loads((output_dir / "manual_review.jsonl").read_text(encoding="utf-8"))
    assert review["verdict"] == "unreviewed"
    assert review["strict_pass"] is None
