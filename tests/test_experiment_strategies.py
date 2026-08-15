from types import SimpleNamespace

from agent.retrieval_loop import merge_retrieval_results
from agent.retrieval_synthesis import build_evidence_context, synthesize_retrieval_answer
from utils.experiment_strategies import normalize_strategy


class FakeModel:
    provider = "local_test"

    def __init__(self, output="RWKV answer"):
        self.output = output
        self.prompt = ""

    def text_completion(self, prompt, max_tokens=0, stop=None):
        self.prompt = prompt
        return SimpleNamespace(content=self.output)


def _round(query, url, text):
    return (
        query,
        {
            "status": "ok",
            "results": [
                {
                    "title": url,
                    "url": url,
                    "content": text,
                    "evidence_origin": "fetched_page_body",
                }
            ],
        },
    )


def test_strategy_normalization_remains_retrieval_only():
    strategy = normalize_strategy({"ranking_strategy": "best_rank.v1"})
    assert strategy["ranking_strategy"] == "best_rank.v1"


def test_merge_deduplicates_same_source_before_writer_context():
    merged = merge_retrieval_results(
        "topic",
        "web_search",
        [
            _round("first", "https://example.com/a", "A substantive fetched source body with the requested topic and enough factual context. " * 3),
            _round("second", "https://example.com/a", "A richer substantive source body with more direct details and enough factual context. " * 3),
        ],
    )
    assert len(merged["results"]) == 1
    assert merged["results"][0]["candidate_queries"] == ["first", "second"]


def test_context_source_count_cannot_admit_unbound_raw_pages():
    context = build_evidence_context(
        {
            "results": [
                {"title": "A", "url": "https://a.example", "content": "body A"},
                {"title": "B", "url": "https://b.example", "content": "body B"},
            ]
        },
        {"strategy_config": {"context_source_count": 1}},
    )
    assert context["selected_evidence"] == []
    assert "unbound_fallback_source_limit" not in context["context_stats"]


def test_prompt_variant_cannot_rewrite_rwkv_answer():
    output = "RWKV's exact wording, without an imposed citation."
    model = FakeModel(output)
    result = synthesize_retrieval_answer(
        "question",
        {"results": [{"title": "A", "url": "https://a.example", "content": "body A"}]},
        llm=model,
        constraints={"strategy_config": {"prompt_variant": "citation_first.v1"}},
    )
    assert result["content"] == output
