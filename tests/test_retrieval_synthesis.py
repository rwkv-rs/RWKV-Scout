from types import SimpleNamespace

import pytest

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
    assert model.calls == [
        {
            "prompt": model.calls[0]["prompt"],
            "max_tokens": model.calls[0]["max_tokens"],
            "stop": None,
        }
    ]


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
    assert call["max_tokens"] < 2048
    assert get_token_count(call["prompt"]) + call["max_tokens"] + 256 <= 3072
    assert result["context_stats"]["requested_output_tokens"] == 2048
    assert result["context_stats"]["effective_output_tokens"] == call["max_tokens"]


def test_context_keeps_source_chunks_and_source_identity():
    context = build_evidence_context({"results": [_source(1), _source(2)]})
    assert "<s1-1>" in context["text"]
    assert "<s2-1>" in context["text"]
    assert "https://example.com/1" in context["text"]
    assert [row["ref_id"] for row in context["citation_refs"]] == ["S1", "S2"]


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
