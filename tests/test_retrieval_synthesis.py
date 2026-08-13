from types import SimpleNamespace

import pytest
import config

from agent.retrieval_synthesis import (
    _clean_answer,
    _writer_prompt,
    build_evidence_context,
    synthesize_retrieval_answer,
)
from utils.chunker import get_token_count


class FakeRWKV:
    provider = "local_test"

    def __init__(self, output):
        self.output = output
        self.calls = []

    def text_completion(self, prompt, max_tokens=0, stop=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "stop": stop})
        return SimpleNamespace(content=self.output)


def _source(index=1):
    return {
        "title": f"Source {index}",
        "url": f"https://example.com/{index}",
        "content": f"Body {index}",
        "source_chunks": [
            {"chunk_id": f"s{index}-1", "index": 0, "text": f"First chunk from source {index}."},
            {"chunk_id": f"s{index}-2", "index": 1, "text": f"Second chunk from source {index}."},
        ],
    }


def test_context_packer_prioritizes_direct_user_url_without_dropping_sources():
    direct = {
        "title": "Requested page",
        "url": "https://docs.example.test/exact",
        "content": "Exact requested page body.",
    }
    earlier = {
        "title": "Earlier candidate",
        "url": "https://other.example.test/page",
        "content": "Earlier candidate body.",
        "evidence_origin": "structured_api_record",
    }

    context = build_evidence_context(
        {"results": [earlier, direct]},
        query="Read https://docs.example.test/exact and answer.",
        max_sources=2,
    )

    selected = context["selected_evidence"]
    assert [row["url"] for row in selected] == [direct["url"], earlier["url"]]
    assert selected[0]["context_selection"]["direct_user_url"] is True
    assert len(selected) == 2


def test_context_packer_routes_structured_and_recent_records_first_for_latest_query():
    ordinary = {
        "title": "Old web page",
        "url": "https://example.test/old",
        "content": "Old general-web material.",
        "published": "2024-01-01",
    }
    older_release = {
        "title": "Release v2",
        "url": "https://github.com/example/project/releases/tag/v2",
        "content": "Release: v2",
        "published": "2025-01-01",
        "evidence_origin": "structured_api_record",
        "content_type": "release",
    }
    latest_release = {
        "title": "Release v3",
        "url": "https://github.com/example/project/releases/tag/v3",
        "content": "Release: v3",
        "published": "2026-08-10",
        "evidence_origin": "structured_api_record",
        "content_type": "release",
    }

    context = build_evidence_context(
        {"results": [ordinary, older_release, latest_release]},
        query="What is the current latest release?",
        max_sources=3,
    )

    selected = context["selected_evidence"]
    assert [row["title"] for row in selected] == [
        "Release v3",
        "Release v2",
        "Old web page",
    ]
    assert selected[0]["context_selection"]["structured_record"] is True
    assert selected[0]["context_selection"]["date_priority_active"] is True
    assert selected[0]["context_selection"]["source_date"] == "2026-08-10"


def test_rwkv_output_is_returned_without_semantic_rewrite():
    output = "**P1** – model wording\nDo not replace this answer. [S1]"
    model = FakeRWKV(output)
    result = synthesize_retrieval_answer(
        "question",
        {"results": [_source()]},
        llm=model,
    )
    assert result["content"] == output
    assert result["model_output"] == output
    assert len(model.calls) == 1
    assert "future-dated or explicitly historical record" in model.calls[0]["prompt"]
    assert "exact owner/repository" in model.calls[0]["prompt"]
    assert "every requested field" in model.calls[0]["prompt"]
    assert "If the answer starts to loop, stop immediately" in model.calls[0]["prompt"]
    assert "judge it directly" in model.calls[0]["prompt"]
    assert "state the time conflict or uncertainty instead" in model.calls[0]["prompt"]
    assert "repair" not in result


def test_empty_think_prefill_boundary_is_not_published_as_answer_text():
    model = FakeRWKV(">\n  Answer starts here.\n")
    result = synthesize_retrieval_answer(
        "question",
        {"results": [_source()]},
        llm=model,
    )
    assert result["content"] == "  Answer starts here.\n"
    assert result["model_output"] == "  Answer starts here.\n"
    assert result["raw_model_output"] == ">\n  Answer starts here.\n"


def test_clean_answer_preserves_every_model_character():
    output = "  Assistant: <think>visible model text</think>\nKeep **all** wording.  \n"
    assert _clean_answer(output) == output


def test_final_writer_does_not_install_output_stop_sequences():
    model = FakeRWKV("answer")
    synthesize_retrieval_answer("question", {"results": []}, llm=model)
    assert len(model.calls) == 1
    assert all(call["stop"] is None for call in model.calls)


def test_final_writer_bounds_completion_against_the_complete_prompt(monkeypatch):
    model = FakeRWKV("answer")
    monkeypatch.setattr(
        "agent.retrieval_synthesis.get_llm_context_length",
        lambda: 3072,
    )
    monkeypatch.setitem(
        __import__("agent.retrieval_synthesis", fromlist=["DATA_PIPELINE"]).DATA_PIPELINE,
        "final_answer_max_tokens",
        2048,
    )
    source = _source()
    source["source_chunks"] = [
        {
            "chunk_id": f"large-{index}",
            "index": index,
            "text": ("retrieved evidence " * 220).strip(),
        }
        for index in range(3)
    ]

    result = synthesize_retrieval_answer("question", {"results": [source]}, llm=model)

    call = model.calls[0]
    assert call["max_tokens"] <= 2048
    assert get_token_count(call["prompt"]) + call["max_tokens"] + 256 <= 3072
    assert result["context_stats"]["requested_output_tokens"] == 2048
    assert result["context_stats"]["effective_output_tokens"] == call["max_tokens"]


def test_context_keeps_source_chunks_and_source_identity():
    context = build_evidence_context({"results": [_source(1), _source(2)]})
    assert "<s1-1>" in context["text"]
    assert "<s2-1>" in context["text"]
    assert "https://example.com/1" in context["text"]
    assert [row["ref_id"] for row in context["citation_refs"]] == ["S1", "S2"]


def test_context_prioritizes_selected_chunks_then_backfills_original_chunks():
    source = _source(1)
    source["selected_source_chunks"] = [source["source_chunks"][1]]

    context = build_evidence_context(
        {"results": [source]},
        token_budget=128,
        max_chunks_per_source=1,
    )

    assert "<s1-2>" in context["text"]
    assert "Second chunk from source 1." in context["text"]
    assert "<s1-1>" not in context["text"]


def test_query_focused_source_span_precedes_unrelated_grounded_quotes():
    source = {
        "title": "Kubernetes Network Policies",
        "url": "https://kubernetes.io/docs/concepts/services-networking/network-policies/",
        "content": (
            "Default deny ingress and egress:\n"
            "spec:\n  podSelector: {}\n  policyTypes:\n"
            "  - Ingress\n  - Egress"
        ),
        "source_chunks": [
            {
                "chunk_id": "page-intro",
                "index": 0,
                "text": "A NetworkPolicy controls traffic between pods.",
            },
            {
                "chunk_id": "default-deny",
                "index": 1,
                "text": (
                    "Default deny ingress and egress:\n"
                    "spec:\n  podSelector: {}\n  policyTypes:\n"
                    "  - Ingress\n  - Egress"
                ),
            },
        ],
        "selected_source_chunks": [
            {
                "chunk_id": "default-deny",
                "index": 1,
                "text": (
                    "Default deny ingress and egress:\n"
                    "spec:\n  podSelector: {}\n  policyTypes:\n"
                    "  - Ingress\n  - Egress"
                ),
                "attention_rank": 1,
                "attention_score": 96,
                "attention_reasons": ["query_focused_source_span"],
            }
        ],
        "chunk_candidates": [
            {
                "supported": True,
                "source_grounded": True,
                "chunk_id": f"noise-{index}",
                "chunk_index": index + 2,
                "quote": f"Related but incomplete NetworkPolicy locator {index}.",
            }
            for index in range(3)
        ],
    }

    context = build_evidence_context(
        {"results": [source]},
        token_budget=512,
        max_chunks_per_source=4,
    )

    packed = context["selected_evidence"][0]["packed_chunks"]
    assert packed[0]["chunk_id"] == "default-deny"
    assert "podSelector: {}" in context["text"]
    assert context["text"].index("podSelector: {}") < context["text"].index(
        "Related but incomplete NetworkPolicy locator 0."
    )


def test_contained_locator_is_merged_into_selected_span_once():
    selected_text = (
        "Release record context. Version 4.2.1 was released on 2026-08-05. "
        "This paragraph explains the changes."
    )
    source = {
        "title": "Release",
        "url": "https://example.com/release",
        "content": selected_text,
        "selected_source_chunks": [
            {
                "chunk_id": "release-section",
                "index": 2,
                "text": selected_text,
                "attention_rank": 1,
            }
        ],
        "source_chunks": [
            {"chunk_id": "release-section", "index": 2, "text": selected_text}
        ],
        "chunk_candidates": [
            {
                "supported": True,
                "source_grounded": True,
                "chunk_id": "release-section",
                "chunk_index": 2,
                "claim_ids": ["P1"],
                "quote": "Version 4.2.1 was released on 2026-08-05.",
            }
        ],
    }

    context = build_evidence_context({"results": [source]}, token_budget=512)

    assert context["text"].count("Version 4.2.1 was released on 2026-08-05.") == 1
    packed = context["selected_evidence"][0]["packed_chunks"]
    assert packed[0]["chunk_id"] == "release-section"
    assert packed[0]["claim_ids"] == ["P1"]
    assert packed[0]["contained_locator_ids"] == ["locator-release-section"]


def test_temporal_attention_metadata_reaches_final_writer_context():
    source = {
        "title": "Release history",
        "url": "https://example.com/releases",
        "content": "Version 4.2.1 was released on 2026-08-05.",
        "selected_source_chunks": [
            {
                "chunk_id": "release-section",
                "index": 2,
                "text": "Version 4.2.1 was released on 2026-08-05.",
                "record_date": "2026-08-05",
                "record_date_precision": "day",
                "record_temporal_role": "newest_dated_window_in_page",
                "temporal_attention_score": 96,
                "rank_scores": {
                    "requirement": 54,
                    "lexical": 1.0,
                    "temporal": 96,
                    "final": 220,
                },
                "attention_rank": 1,
                "attention_reasons": ["query_focused_source_span"],
            }
        ],
        "source_chunks": [
            {
                "chunk_id": "release-section",
                "index": 2,
                "text": "Version 4.2.1 was released on 2026-08-05.",
            }
        ],
        "chunk_candidates": [
            {
                "supported": True,
                "source_grounded": True,
                "chunk_id": "release-section",
                "chunk_index": 2,
                "claim_ids": ["P1"],
                "quote": "Version 4.2.1 was released on 2026-08-05.",
            }
        ],
    }

    context = build_evidence_context(
        {"results": [source]},
        constraints={
            "task_plan": {
                "schema_version": "task_plan.v2",
                "atomic_points": [
                    {
                        "id": "P1",
                        "question": "What is the latest release and date?",
                        "fields": ["release_version", "release_date"],
                        "time_scope": "current",
                    }
                ],
            }
        },
        token_budget=512,
    )

    packed = context["selected_evidence"][0]["packed_chunks"][0]
    assert packed["record_date"] == "2026-08-05"
    assert packed["record_date_precision"] == "day"
    assert packed["record_temporal_role"] == "newest_dated_window_in_page"
    assert "rank_scores" not in packed
    assert "attention_rank" not in packed
    assert (
        '"record_date":"2026-08-05","date_precision":"day",'
        '"role":"newest_dated_window_in_page"'
    ) in context["text"]


def test_non_temporal_task_keeps_temporal_state_out_of_writer_text():
    source = {
        "title": "Official command reference",
        "url": "https://example.test/command",
        "content": "Use --jobs N. The command opens N+1 connections.",
        "selected_source_chunks": [
            {
                "chunk_id": "chunk-1",
                "index": 0,
                "text": "Use --jobs N. The command opens N+1 connections.",
                "record_date": "2026-07-16",
                "record_date_precision": "day",
                "record_temporal_role": "newest_dated_window_in_page",
            }
        ],
    }
    task_plan = {
        "schema_version": "task_plan.v2",
        "atomic_points": [
            {
                "id": "P1",
                "question": "How many connections does --jobs use?",
                "fields": ["extra_connections"],
                "time_scope": "current",
            }
        ],
    }

    context = build_evidence_context(
        {"results": [source]},
        constraints={"task_plan": task_plan},
        query="Read the current documentation and explain --jobs.",
        token_budget=512,
    )

    packed = context["selected_evidence"][0]["packed_chunks"][0]
    assert packed["record_date"] == "2026-07-16"
    assert "Same-page temporal routing metadata" not in context["text"]


def test_broad_source_neighbour_cannot_replace_selected_exact_window():
    selected = "Exact option record: jobs + 1 connections."
    broad = "Long unrelated preface.\n" + selected + "\nLong unrelated appendix."
    source = {
        "title": "Command reference",
        "url": "https://example.com/reference",
        "content": broad,
        "selected_source_chunks": [
            {
                "chunk_id": "full-record",
                "index": 0,
                "text": selected,
                "attention_rank": 1,
                "attention_score": 90,
            }
        ],
        "source_chunks": [
            {"chunk_id": "full-record", "index": 0, "text": broad}
        ],
    }

    context = build_evidence_context(
        {"results": [source]},
        token_budget=512,
        max_chunks_per_source=1,
    )

    packed = context["selected_evidence"][0]["packed_chunks"]
    assert packed[0]["text"] == selected
    assert packed[0]["packing_role"] == "selected_attention_window"
    assert "Long unrelated preface" not in context["text"]


def test_context_exposes_observed_record_order_without_selecting_truth():
    source = {
        "title": "Release history",
        "url": "https://example.com/history",
        "content": "Version 4.1 on 2026-07-01. Version 4.2 on 2026-08-05.",
        "source_chunks": [
            {
                "chunk_id": "history",
                "index": 0,
                "text": "Version 4.1 on 2026-07-01. Version 4.2 on 2026-08-05.",
            }
        ],
    }

    context = build_evidence_context({"results": [source]})

    assert '"dates":["2026-07-01","2026-08-05"]' in context["text"]
    assert '"versions":["Version 4.1","Version 4.2"]' in context["text"]
    assert "no truth/currentness judgment" in context["text"]


def test_context_backfills_unselected_original_chunks_when_budget_allows():
    source = _source(1)
    source["selected_source_chunks"] = [source["source_chunks"][1]]

    context = build_evidence_context({"results": [source]}, token_budget=512)

    assert context["text"].index("<s1-2>") < context["text"].index("<s1-1>")
    assert context["chunk_count"] == 2


def test_context_routes_selected_span_with_only_adjacent_original_chunks():
    source = _source(1)
    source["source_chunks"] = [
        {"chunk_id": f"row-{index}", "index": index, "text": f"source row {index} unique text"}
        for index in range(5)
    ]
    source["selected_source_chunks"] = [source["source_chunks"][2]]

    context = build_evidence_context({"results": [source]}, token_budget=1024)

    assert "<row-1>" in context["text"]
    assert "<row-2>" in context["text"]
    assert "<row-3>" in context["text"]
    assert "<row-0>" not in context["text"]
    assert "<row-4>" not in context["text"]


def test_claim_grounded_spans_become_compact_citable_sources():
    ledger = {
        "claims": [
            {
                "claim_id": "P1",
                "task": "identify the current release",
                "evidence_needed": ["exact version and date"],
                "retrieval_state": "retrieved",
                "sources": [
                    {
                        "title": "Official downloads",
                        "url": "https://example.com/releases",
                        "provider": "official adapter",
                        "freshness": {"state": "dated", "source_date": "2026-08-05"},
                        "grounded_spans": [
                            {
                                "chunk_id": "release-row",
                                "index": 4,
                                "text": "Version 4.2.1 was released on 2026-08-05.",
                                "source_locator": {
                                    "type": "source_quote_span",
                                    "char_start": 100,
                                    "char_end": 145,
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }

    context = build_evidence_context({"results": [], "claim_ledger": ledger})

    assert "[S1] Source label (routing metadata only): Official downloads" in context["text"]
    assert "https://example.com/releases" in context["text"]
    assert context["text"].count("Version 4.2.1 was released on 2026-08-05.") == 1
    assert '"source_locator"' not in context["text"]
    assert "RWKV task-point bindings: P1" in context["text"]
    citation = context["citation_refs"][0]
    assert citation["ref_id"] == "S1"
    assert citation["title"] == "Official downloads"
    assert citation["url"] == "https://example.com/releases"
    assert citation["quote"] == "Version 4.2.1 was released on 2026-08-05."
    assert citation["chunk_ids"] == ["locator-release-row"]
    assert citation["task_point_ids"] == ["P1"]


def test_evidence_records_on_same_url_keep_version_identity_separate():
    ledger = {
        "claims": [
            {
                "claim_id": "P1",
                "question": "current version, title and date",
                "fields": ["version", "title", "date"],
                "evidence_records": [
                    {
                        "evidence_record_id": "E-current",
                        "task_record_id": "P1",
                        "subject_key": "Example Game",
                        "record_key": "Version 4.4",
                        "field_keys": ["version", "title"],
                        "title": "Release history",
                        "url": "https://example.com/releases",
                        "chunk_id": "current",
                        "quote": "Version 4.4 is titled New Dawn.",
                        "support_state": "rwkv_supported_exact_span",
                    },
                    {
                        "evidence_record_id": "E-old",
                        "task_record_id": "P1",
                        "subject_key": "Example Game",
                        "record_key": "Version 4.0",
                        "field_keys": ["date"],
                        "title": "Release history",
                        "url": "https://example.com/releases",
                        "chunk_id": "old",
                        "quote": "Version 4.0 was released on 2025-01-01.",
                        "support_state": "rwkv_supported_exact_span",
                    },
                ],
                "sources": [],
            }
        ]
    }

    context = build_evidence_context(
        {"results": [], "claim_ledger": ledger},
        constraints={"context_source_count": 4},
    )

    assert context["context_stats"]["bound_evidence_record_count"] == 0
    assert context["context_stats"]["candidate_evidence_record_count"] == 2
    assert [row["evidence_record_id"] for row in context["selected_evidence"]] == [
        "E-current",
        "E-old",
    ]
    assert context["text"].count("RWKV-CANDIDATE SOURCE RECORD") == 2
    assert '"record_key":"Version 4.4"' in context["text"]
    assert '"record_key":"Version 4.0"' in context["text"]


def test_current_record_context_orders_grounded_literal_dates_without_merging_rows():
    ledger = {
        "claims": [
            {
                "claim_id": "P1",
                "question": "latest version and date",
                "fields": ["version", "date"],
                "evidence_records": [
                    {
                        "evidence_record_id": "E-old",
                        "task_record_id": "P1",
                        "record_key": "4.0",
                        "field_keys": ["version", "date"],
                        "title": "Release notes",
                        "url": "https://example.com/releases",
                        "chunk_id": "old",
                        "quote": "4.0\n2026-07-01",
                        "support_state": "rwkv_candidate_other_record",
                    },
                    {
                        "evidence_record_id": "E-new",
                        "task_record_id": "P1",
                        "record_key": "5.0",
                        "field_keys": ["version", "date"],
                        "title": "Release notes",
                        "url": "https://example.com/releases",
                        "chunk_id": "new",
                        "quote": "5.0\n2026-08-10",
                        "support_state": "rwkv_candidate_other_record",
                    },
                ],
                "sources": [],
            }
        ]
    }
    plan = {
        "goal": "latest version and date",
        "atomic_points": [
            {
                "id": "P1",
                "question": "latest version and date",
                "fields": ["version", "date"],
                "time_scope": "current",
            }
        ],
    }

    context = build_evidence_context(
        {"results": [], "claim_ledger": ledger},
        constraints={"task_plan": plan, "context_source_count": 4},
        query="What is the latest version and date?",
    )

    assert [row["evidence_record_id"] for row in context["selected_evidence"]] == [
        "E-new",
        "E-old",
    ]
    assert "5.0\n2026-08-10" in context["citation_refs"][0]["quote"]


def test_grounded_source_slots_are_round_robin_across_claims():
    def grounded_source(name: str, fact: str) -> dict:
        return {
            "title": name,
            "url": f"https://example.com/{name}",
            "grounded_spans": [
                {"chunk_id": name, "index": 0, "text": fact}
            ],
        }

    ledger = {
        "claims": [
            {
                "claim_id": "P1",
                "task": "first fact",
                "sources": [
                    grounded_source("p1-a", "first claim primary evidence"),
                    grounded_source("p1-b", "first claim secondary evidence"),
                    grounded_source("p1-c", "first claim tertiary evidence"),
                ],
            },
            {
                "claim_id": "P2",
                "task": "second fact",
                "sources": [grounded_source("p2-a", "second claim evidence")],
            },
        ]
    }

    context = build_evidence_context(
        {"results": [], "claim_ledger": ledger},
        constraints={"context_source_count": 2},
    )

    assert "https://example.com/p1-a" in context["text"]
    assert "https://example.com/p2-a" in context["text"]
    assert [row["url"] for row in context["citation_refs"]] == [
        "https://example.com/p1-a",
        "https://example.com/p2-a",
    ]


def test_grounded_quote_does_not_repeat_as_verbatim_locator_block():
    source = _source(1)
    source["chunk_candidates"] = [
        {
            "supported": True,
            "source_grounded": True,
            "chunk_id": "s1-1",
            "chunk_index": 0,
            "quote": "Exact model-grounded fact.",
        }
    ]
    source["model_locator_facts"] = "[s1-1] Exact model-grounded fact."

    context = build_evidence_context({"results": [source]}, token_budget=512)

    assert context["text"].count("Exact model-grounded fact.") == 1
    assert "VERBATIM SOURCE LOCATORS" not in context["text"]


def test_final_context_caps_grounded_spans_per_source(monkeypatch):
    source = _source(1)
    source["source_chunks"] = []
    source["chunk_candidates"] = [
        {
            "supported": True,
            "source_grounded": True,
            "chunk_id": f"fact-{index}",
            "chunk_index": index,
            "quote": f"grounded fact number {index}",
        }
        for index in range(7)
    ]
    monkeypatch.setitem(
        __import__("agent.retrieval_synthesis", fromlist=["DATA_PIPELINE"]).DATA_PIPELINE,
        "final_context_max_grounded_spans_per_source",
        3,
    )

    context = build_evidence_context({"results": [source]}, token_budget=1024)

    assert "grounded fact number 0" in context["text"]
    assert "grounded fact number 2" in context["text"]
    assert "grounded fact number 3" not in context["text"]
    assert context["context_stats"]["max_grounded_spans_per_source"] == 3


def test_cross_validation_review_does_not_enter_writer_context():
    review = {
        "schema_version": "rwkv-cross-validation.v1",
        "decision": "replan",
        "missing_points": ["P2"],
        "conflicts": [],
        "task_point_status": {
            "P1": {"status": "supported", "evidence_refs": ["S1"]},
            "P2": {"status": "missing", "evidence_refs": []},
        },
        "next_focus": "the exact release date is still absent",
        "reason": "one requested field is missing",
    }

    context = build_evidence_context(
        {"results": [_source(1)]},
        constraints={"last_cross_validation": review},
    )

    assert "LATEST RWKV CROSS-VALIDATION REVIEW" not in context["text"]
    assert "the exact release date is still absent" not in context["text"]
    assert context["context_stats"]["cross_validation_review_included"] is False


def test_context_keeps_reference_ids_contiguous_when_a_source_does_not_fit():
    first = _source(1)
    first["source_chunks"] = [
        {"chunk_id": "first", "index": 0, "text": "small first evidence"}
    ]
    oversized = _source(2)
    oversized["source_chunks"] = [
        {
            "chunk_id": "oversized",
            "index": 0,
            "text": ("oversized evidence " * 600).strip(),
        }
    ]
    third = _source(3)
    third["source_chunks"] = [
        {"chunk_id": "third", "index": 0, "text": "small third evidence"}
    ]

    context = build_evidence_context(
        {"results": [first, oversized, third]},
        token_budget=128,
    )

    assert "[S1] UNBOUND SOURCE EXCERPT: Source 1" in context["text"]
    assert "[S2] UNBOUND SOURCE EXCERPT: Source 3" in context["text"]
    assert "[S3]" not in context["text"]
    assert [
        (row["ref_id"], row["title"], row["url"])
        for row in context["citation_refs"]
    ] == [
        ("S1", "Source 1", "https://example.com/1"),
        ("S2", "Source 3", "https://example.com/3"),
    ]


def test_cross_validation_cannot_reorder_writer_sources():
    context = build_evidence_context(
        {"results": [_source(1), _source(2)]},
        constraints={
            "context_source_count": 1,
            "validated_source_urls": ["https://example.com/2"],
        },
    )

    assert "https://example.com/1" in context["text"]
    assert "https://example.com/2" not in context["text"]


def test_validation_context_can_bound_sources_chunks_and_locator_spans():
    sources = []
    for source_index in range(1, 4):
        source = _source(source_index)
        source["source_chunks"] = [
            {
                "chunk_id": f"s{source_index}-{chunk_index}",
                "index": chunk_index,
                "text": (f"source {source_index} chunk {chunk_index} " * 20).strip(),
            }
            for chunk_index in range(1, 5)
        ]
        source["model_locator_facts"] = (
            f"LOCATOR_{source_index}_START "
            + ("verbatim locator material " * 200)
            + f" LOCATOR_{source_index}_END"
        )
        sources.append(source)

    context = build_evidence_context(
        {"results": sources},
        max_sources=2,
        token_budget=512,
        max_chunks_per_source=1,
        source_locator_char_limit=120,
    )

    assert "https://example.com/1" in context["text"]
    assert "https://example.com/2" in context["text"]
    assert "https://example.com/3" not in context["text"]
    assert "<s1-1>" in context["text"]
    assert "<s1-2>" not in context["text"]
    assert "LOCATOR_1_START" in context["text"]
    assert "LOCATOR_1_END" not in context["text"]
    assert context["context_truncated"] is True
    assert context["context_stats"]["evidence_budget_tokens"] == 512
    assert context["context_stats"]["configured_source_limit"] == 2
    assert context["context_stats"]["max_chunks_per_source"] == 1
    assert context["context_stats"]["source_locator_char_limit"] == 120


def test_raw_page_level_claim_ids_do_not_reserve_evidence_slots():
    p1_first = _source(1)
    p1_first["claim_ids"] = ["P1"]
    p1_second = _source(2)
    p1_second["claim_ids"] = ["P1"]
    p2 = _source(3)
    p2["claim_ids"] = ["P2"]
    ledger = {
        "claims": [
            {"claim_id": "P1", "task": "historical date", "sources": []},
            {"claim_id": "P2", "task": "current theme", "sources": []},
        ]
    }

    context = build_evidence_context(
        {"results": [p1_first, p1_second, p2], "claim_ledger": ledger},
        constraints={"context_source_count": 2},
    )

    assert "https://example.com/1" in context["text"]
    assert "https://example.com/2" in context["text"]
    assert "https://example.com/3" not in context["text"]


def test_raw_page_level_bindings_are_removed_from_fallback_context():
    p1_first = _source(1)
    p1_first["claim_ids"] = ["P1"]
    p1_second = _source(2)
    p1_second["claim_ids"] = ["P1"]
    p1_and_p2 = _source(3)
    p1_and_p2["claim_ids"] = ["P1", "P2"]
    p3 = _source(4)
    p3["claim_ids"] = ["P3"]
    plan = {
        "schema_version": "task_plan.v2",
        "goal": "answer three records",
        "atomic_points": [
            {"id": "P1", "question": "first", "fields": [], "time_scope": "unspecified"},
            {"id": "P2", "question": "second", "fields": [], "time_scope": "unspecified"},
            {"id": "P3", "question": "third", "fields": [], "time_scope": "unspecified"},
        ],
    }

    context = build_evidence_context(
        {"results": [p1_first, p1_second, p1_and_p2, p3]},
        constraints={"context_source_count": 3, "task_plan": plan},
    )

    urls = [row["url"] for row in context["selected_evidence"]]
    assert urls == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]
    assert context["context_stats"]["factual_points_with_selected_sources"] == 0
    assert all(not row.get("claim_ids") for row in context["selected_evidence"])


def test_context_exposes_source_dates_and_question_freshness_policy():
    source = _source(1)
    source.update(
        {
            "claim_ids": ["P2"],
            "published": "2026-07-01",
            "updated": "2026-07-20",
            "provider": "official-feed",
            "freshness": {"state": "within_cutoff"},
        }
    )

    context = build_evidence_context(
        {"results": [source]},
        constraints={"freshness_policy": {"as_of": "2026-08-09", "mode": "latest"}},
    )

    assert "RWKV task-point bindings: P2" not in context["text"]
    assert "UNBOUND SOURCE EXCERPT" in context["text"]
    assert "Published: 2026-07-01" in context["text"]
    assert "Updated: 2026-07-20" in context["text"]
    assert "official-feed" in context["text"]
    assert "QUESTION TIME/FRESHNESS POLICY" in context["text"]
    assert '"as_of": "2026-08-09"' in context["text"]


def test_writer_is_told_not_to_expand_into_adjacent_unrequested_material():
    prompt = _writer_prompt(
        "Which output format is required?",
        {"text": "<chunk-1>Directory format. Nearby unrelated limitation."},
    )

    assert "Answer only the fields the user requested" in prompt
    assert "Do not append adjacent limitations" in prompt


def test_writer_includes_the_verbatim_user_question_once():
    question = "Which exact version, title, and release date are requested?"
    evidence = "<chunk-1>The exact record is preserved here.</chunk-1>"

    prompt = _writer_prompt(question, {"text": evidence})

    question_block = f"USER QUESTION:\n{question}"
    assert prompt.count(question_block) == 1
    assert prompt.index(question_block) < prompt.index(evidence)
    assert prompt.endswith("Write the final answer now.")


def test_claim_ledger_is_advisory_context_not_a_gate():
    model = FakeRWKV("RWKV answers even with no retrieved source.")
    result = synthesize_retrieval_answer(
        "question",
        {
            "results": [],
            "claim_ledger": {
                "claims": [
                    {
                        "claim_id": "P1",
                        "task": "missing task",
                        "retrieval_state": "not_retrieved",
                        "sources": [],
                    }
                ]
            },
        },
        llm=model,
    )
    assert result["content"] == "RWKV answers even with no retrieved source."
    assert "No source text was retrieved" in model.calls[0]["prompt"]


def test_empty_rwkv_output_is_the_only_writer_failure():
    with pytest.raises(ConnectionError):
        synthesize_retrieval_answer("question", {"results": []}, llm=FakeRWKV(""))


def test_missing_rwkv_client_has_no_algorithmic_fallback():
    with pytest.raises(ConnectionError):
        synthesize_retrieval_answer("question", {"results": [_source()]}, llm=None)


def test_final_writer_uses_request_level_stage_temperature(monkeypatch):
    seen = []

    class SamplingAwareRWKV:
        provider = "local_test"

        def text_completion(self, prompt, max_tokens=0, stop=None):
            del prompt, max_tokens, stop
            seen.append(config.get_llm_temperature())
            return SimpleNamespace(content="answer")

    monkeypatch.setattr(
        "agent.retrieval_synthesis.get_model_stage_temperature",
        lambda stage: 0.37 if stage == "final_writer" else 0.0,
    )
    result = synthesize_retrieval_answer(
        "question",
        {"results": [_source()]},
        llm=SamplingAwareRWKV(),
    )

    assert seen == [0.37]
    assert result["generation_attempts"][0]["sampling_temperature"] == 0.37
    assert result["context_stats"]["sampling_temperature"] == 0.37
