from types import SimpleNamespace

from agent.page_evidence import extract_single_page_evidence
from agent.retrieval_loop import merge_retrieval_results
from agent.retrieval_synthesis import _clean_answer, build_evidence_context, synthesize_retrieval_answer
from utils.evidence_quality import evidence_kind, evidence_text, has_substantive_evidence


class _Model:
    def __init__(self, output="Supported fact [S1]."):
        self.output = output
        self.calls = []

    def text_completion(self, prompt, max_tokens=0, **kwargs):
        self.calls.append((prompt, max_tokens, kwargs))
        return SimpleNamespace(content=self.output)


def _body(label):
    return f"{label} is directly supported by the fetched page body and contains enough source text."


def test_discovery_metadata_never_enters_final_evidence_context():
    context = build_evidence_context(
        {
            "query": "release date anniversary latest version theme",
            "results": [
                {
                    "title": "Search result title",
                    "url": "https://example.com/search",
                    "snippet": "2024-01-01 and an unsupported summary",
                    "published": "2026-07-30",
                }
            ],
        }
    )
    assert context["selected_evidence"] == []
    assert context["usable_evidence_count"] == 0
    assert context["text"] == "(no retrieved evidence)"


def test_date_query_keeps_multiple_substantive_sources_and_separates_page_date():
    context = build_evidence_context(
        {
            "query": "\u53d1\u5e03\u65e5\u671f\u3001\u5468\u5e74\u5e86\u65e5\u671f\u3001\u6700\u65b0\u7248\u672c\u4e3b\u9898",
            "results": [
                {"title": "A", "url": "https://example.com/a", "page_excerpt": _body("Release date 2024-01-01")},
                {"title": "B", "url": "https://example.com/b", "page_excerpt": _body("Latest version theme Cloud Wings")},
                {"title": "C", "url": "https://example.com/c", "page_excerpt": _body("Anniversary date 2025-01-01")},
            ],
        }
    )
    assert len(context["selected_evidence"]) == 3
    assert "Published:" not in context["text"]
    assert "unsupported summary" not in context["text"].casefold()
    assert "2026-07-30" not in context["text"]
    assert all(item["evidence_boundary"].endswith("only") for item in context["selected_evidence"])


def test_merge_discards_title_only_results_before_ranking():
    merged = merge_retrieval_results(
        "query",
        "web_search",
        [
            (
                "query",
                {
                    "results": [
                        {"title": "Only a title", "url": "https://example.com/title", "snippet": "summary"},
                        {"title": "Body", "url": "https://example.com/body", "content": _body("The answer")},
                    ]
                },
            )
        ],
    )
    assert [item["url"] for item in merged["results"]] == ["https://example.com/body"]
    assert merged["evidence_missing_count"] == 1


def test_model_extraction_is_not_the_final_evidence_body():
    context = build_evidence_context(
        {
            "query": "verify the fact",
            "results": [
                {
                    "url": "https://example.com/body",
                    "page_excerpt": _body("The original source body"),
                    "chunk_candidates": [
                        {"supported": True, "facts": ["A model-only unsupported claim"], "quote": ""}
                    ],
                }
            ],
        }
    )
    assert "The original source body" in context["text"]
    assert "model-only unsupported claim" not in context["text"]


def test_discovery_snippet_cannot_be_promoted_by_model_facts():
    item = {
        "title": "Search result title",
        "url": "https://example.com/result",
        "snippet": "The answer is 2024-01-01.",
        "model_extracted_facts": "The answer is 2024-01-01.",
        "chunk_candidates": [
            {"supported": True, "facts": ["The answer is 2024-01-01."]}
        ],
        "evidence_origin": "discovery",
    }
    assert evidence_kind(item) == "discovery"
    assert evidence_text(item) == ""
    assert not has_substantive_evidence(item)


def test_fetched_body_keeps_model_locator_out_of_canonical_text():
    item = {
        "title": "Fetched page",
        "url": "https://example.com/page",
        "snippet": "Search summary with a wrong date 2024-01-01.",
        "source_excerpt": "The fetched page body states the release date is 2025-05-20.",
        "model_extracted_facts": "The model guessed 2024-01-01.",
        "evidence_origin": "fetched_page_body",
    }
    assert evidence_kind(item) == "page_body"
    assert evidence_text(item) == item["source_excerpt"]
    assert "2024-01-01" not in evidence_text(item)
    assert has_substantive_evidence(item)


def test_short_page_is_rejected_without_model_extraction_call():
    model = _Model()
    result = extract_single_page_evidence(
        query="find the fact",
        page={"url": "https://example.com/short", "title": "Short", "page_excerpt": "Welcome."},
        llm=model,
    )
    assert result["status"] == "no_evidence"
    assert model.calls == []


def test_final_cleanup_removes_protocol_and_exact_repeats():
    cleaned = _clean_answer(
        "Assistant: <think>hidden</think>\n"
        "The answer is supported.\n\nThe answer is supported.\n"
        "Function output: {\"name\":\"web_search\"}"
    )
    assert cleaned == "The answer is supported."


def test_final_no_evidence_prompt_excludes_discovery_facts():
    model = _Model("No reliable source body was retrieved.")
    result = synthesize_retrieval_answer(
        "What is the release date?",
        {
            "query": "What is the release date?",
            "results": [{"title": "Search title", "url": "https://example.com", "snippet": "2024-01-01"}],
            "citation_refs": [],
        },
        llm=model,
    )
    assert "NO_USABLE_EVIDENCE" in model.calls[0][0]
    assert "2024-01-01" not in model.calls[0][0]
    assert result["citation_refs"] == []
