from types import SimpleNamespace

import pytest
import config

from agent.retrieval_synthesis import (
    _clean_answer,
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
    assert "marked missing by the latest RWKV cross-validation remains unsupported" in model.calls[0]["prompt"]
    assert "an explicitly historical record cannot resolve a missing current value" in model.calls[0]["prompt"]
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

    assert "[S1] Official downloads" in context["text"]
    assert "https://example.com/releases" in context["text"]
    assert context["text"].count("Version 4.2.1 was released on 2026-08-05.") == 1
    assert '"source_locator"' not in context["text"]
    assert '"grounded_source_count": 1' in context["text"]
    assert context["citation_refs"] == [
        {
            "ref_id": "S1",
            "title": "Official downloads",
            "url": "https://example.com/releases",
        }
    ]


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


def test_latest_cross_validation_review_is_forwarded_as_advisory_context():
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

    assert "LATEST RWKV CROSS-VALIDATION REVIEW" in context["text"]
    assert "the exact release date is still absent" in context["text"]
    assert context["context_stats"]["cross_validation_review_included"] is True


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

    assert "[S1] Source 1" in context["text"]
    assert "[S2] Source 3" in context["text"]
    assert "[S3]" not in context["text"]
    assert context["citation_refs"] == [
        {"ref_id": "S1", "title": "Source 1", "url": "https://example.com/1"},
        {"ref_id": "S2", "title": "Source 3", "url": "https://example.com/3"},
    ]


def test_context_prioritizes_sources_bound_by_rwkv_cross_validation():
    context = build_evidence_context(
        {"results": [_source(1), _source(2)]},
        constraints={
            "context_source_count": 1,
            "validated_source_urls": ["https://example.com/2"],
        },
    )

    assert "https://example.com/2" in context["text"]
    assert "https://example.com/1" not in context["text"]


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


def test_context_reserves_a_source_slot_for_each_bound_claim():
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
    assert "https://example.com/3" in context["text"]
    assert "https://example.com/2" not in context["text"]


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

    assert "RWKV task-point bindings: P2" in context["text"]
    assert "Published: 2026-07-01" in context["text"]
    assert "Updated: 2026-07-20" in context["text"]
    assert "official-feed" in context["text"]
    assert "QUESTION TIME/FRESHNESS POLICY" in context["text"]
    assert '"as_of": "2026-08-09"' in context["text"]


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
