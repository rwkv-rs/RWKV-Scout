from types import SimpleNamespace

from agent.page_evidence import extract_single_page_evidence
from agent.retrieval_loop import merge_retrieval_results
from agent.retrieval_synthesis import _clean_answer, build_evidence_context, synthesize_retrieval_answer
from agent.state import AgentState
from utils.evidence_quality import clean_page_body, evidence_kind, evidence_text, has_substantive_evidence


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
    assert context["text"] == "RETRIEVED SOURCES:\nNo source text was retrieved."


def test_unbound_date_pages_never_enter_writer_context():
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
    assert context["selected_evidence"] == []
    assert "unbound_fallback_source_limit" not in context["context_stats"]
    assert "Published:" not in context["text"]
    assert "unsupported summary" not in context["text"].casefold()
    assert "2026-07-30" not in context["text"]


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


def test_merge_retains_new_and_prior_selected_chunks_for_same_source():
    source_chunks = [
        {"chunk_id": "old", "index": 0, "text": _body("old fact")},
        {"chunk_id": "new", "index": 1, "text": _body("new fact")},
    ]
    merged = merge_retrieval_results(
        "two-part question",
        "web_search",
        [
            (
                "first point",
                {
                    "results": [
                        {
                            "url": "https://example.com/shared",
                            "content": _body("shared page"),
                            "source_chunks": source_chunks,
                            "selected_source_chunks": [source_chunks[0]],
                        }
                    ]
                },
            ),
            (
                "missing second point",
                {
                    "results": [
                        {
                            "url": "https://example.com/shared",
                            "content": _body("shared page"),
                            "source_chunks": source_chunks,
                            "selected_source_chunks": [source_chunks[1]],
                        }
                    ]
                },
            ),
        ],
    )

    selected = merged["results"][0]["selected_source_chunks"]
    assert [row["chunk_id"] for row in selected] == ["new", "old"]


def test_shared_state_retains_replan_selected_chunks_for_same_source():
    state = AgentState()
    source_chunks = [
        {"chunk_id": "old", "index": 0, "text": _body("old fact")},
        {"chunk_id": "new", "index": 1, "text": _body("new fact")},
    ]
    for query, selected in (
        ("first point", source_chunks[0]),
        ("missing second point", source_chunks[1]),
    ):
        state.retrieval.record_query(
            query,
            {
                "status": "ok",
                "results": [
                    {
                        "url": "https://example.com/shared",
                        "content": _body("shared page"),
                        "source_chunks": source_chunks,
                        "selected_source_chunks": [selected],
                    }
                ],
            },
            task_record_id="",
        )

    selected = state.retrieval.source_records()[0]["selected_source_chunks"]
    assert [row["chunk_id"] for row in selected] == ["new", "old"]


def test_model_extraction_is_not_the_final_evidence_body():
    context = build_evidence_context(
        {
            "query": "verify the fact",
            "results": [
                {
                    "url": "https://example.com/body",
                    "page_excerpt": _body("The original source body"),
                    "chunk_candidates": [
                        {"supported": True, "facts": ["A model-only unsupported task_record"], "quote": ""}
                    ],
                }
            ],
        }
    )
    assert "The original source body" not in context["text"]
    assert "model-only unsupported task_record" not in context["text"]


def test_final_context_repeats_only_grounded_locator_spans():
    source = "Install the tool, then verify it with `tool --version`."
    context = build_evidence_context(
        {
            "results": [
                {
                    "url": "https://example.com/install",
                    "source_excerpt": source,
                    "model_locator_facts": "[chunk-1] verify it with `tool --version`.",
                    "model_extracted_facts": "The model invented `other --version`.",
                }
            ]
        }
    )

    assert "VERBATIM SOURCE LOCATORS" not in context["text"]
    assert "tool --version" not in context["text"]
    assert "other --version" not in context["text"]


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
        "source_excerpt": "The fetched page body states the release date is 2025-05-20. The source also includes the publication context.",
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


def test_short_technical_configuration_is_substantive_evidence():
    body = 'proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";'
    item = {
        "content": body,
        "evidence_origin": "fetched_page_body",
        "body_verified": True,
    }
    assert has_substantive_evidence(item)
    assert clean_page_body(body)["body_eligible"] is True


def test_short_factual_page_survives_navigation_cleanup():
    body = (
        "Skip to content\n登录\n[Image/Complex Table Filtered]\n"
        "百度百科：Python 是一种广泛使用的高级编程语言，强调代码可读性和开发效率。"
        "它支持面向对象、函数式和过程式编程，常用于数据分析、自动化和人工智能开发。"
    )
    quality = clean_page_body(body)
    assert quality["body_eligible"] is True
    assert quality["clean_chars"] >= 70
    assert "Skip to" not in quality["text"]
    assert "登录" not in quality["text"]
    assert "Image/Complex Table Filtered" not in quality["text"]
    assert "Python" in quality["text"]


def test_navigation_only_body_is_not_substantive_evidence():
    body = "\n".join(
        ["Skip to content", "Log in", "Sign up", "[Image/Complex Table Filtered]", "[Home](https://example.com)"]
        * 20
    )
    quality = clean_page_body(body)
    assert quality["body_eligible"] is False
    assert quality["text"] == ""


def test_long_markdown_navigation_line_keeps_facts_but_drops_targets():
    body = (
        "[Skip](javascript:void((function(){open_menu()}))) "
        "[TypeScript 6.0 RC: Temporal API and Breaking Changes](https://example.com/share) "
        "March 11, 2026. "
        "The release notes describe the Temporal API changes and the compatibility impact for applications."
    )
    quality = clean_page_body(body)
    assert quality["body_eligible"] is True
    assert "javascript:" not in quality["text"]
    assert "https://example.com/share" in quality["text"]
    assert "TypeScript 6.0 RC" in quality["text"]
    assert "March 11, 2026" in quality["text"]


def test_final_answer_adapter_preserves_protocol_and_exact_repeats():
    output = (
        "Assistant: <think>hidden</think>\n"
        "The answer is supported.\n\nThe answer is supported.\n"
        "Function output: {\"name\":\"web_search\"}"
    )
    assert _clean_answer(output) == output


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
    assert "No source text was retrieved." in model.calls[0][0]
    assert "do not substitute a plausible value or invent an example" in model.calls[0][0]
    assert "2024-01-01" not in model.calls[0][0]
    assert result["citation_refs"] == []
