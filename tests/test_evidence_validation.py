from utils.evidence_validation import assess_answer_alignment, build_evidence_validation, source_quality
from agent.retrieval_synthesis import build_evidence_context


def _page(url: str, body: str) -> dict:
    return {
        "url": url,
        "title": "test page",
        "content": body,
        "evidence_origin": "fetched_page_body",
        "body_verified": True,
        "source": "generic_web_search",
    }


def test_source_quality_does_not_treat_discovery_metadata_as_body_evidence():
    discovery = {"url": "https://example.org/item", "title": "release date", "snippet": "release date"}
    body = _page("https://example.org/item", "release date: 2024-07-04. The official page states this directly.")

    assert source_quality(discovery)["kind"] == "discovery"
    assert source_quality(body)["kind"] == "page_body"
    assert source_quality(body)["score"] > source_quality(discovery)["score"]


def test_validation_reports_coverage_and_candidate_date_conflict_without_calling_truth():
    results = [
        _page("https://official.example/release", "release date: 2024-07-04. Official release notice."),
        _page("https://archive.example/release", "release date: 2024-07-05. Archived release notice."),
    ]
    report = build_evidence_validation(
        {"results": results},
        query="release date",
        constraints={"task_plan": {"atomic_points": [{"id": "P1", "task": "release date"}]}},
        selected=results,
    )

    assert report["is_truth_judgement"] is False
    assert report["subquestion_coverage"][0]["status"] == "covered"
    assert report["cross_source"]["multi_source_points"] == 1
    assert report["cross_source"]["candidate_conflicts"]


def test_answer_alignment_marks_supported_and_unmatched_claim_lines():
    sources = [_page("https://official.example/release", "release date: 2024-07-04. Official release notice.")]
    alignment = assess_answer_alignment(
        "Release date is 2024-07-04 [S1].\nThe unrelated founder is Ada Lovelace.",
        sources,
    )

    assert alignment["claim_line_count"] == 2
    assert alignment["aligned_line_count"] == 1
    assert alignment["unsupported_line_count"] == 1


def test_context_ranking_prefers_verified_structured_record_before_relevance_tiebreak():
    page = _page(
        "https://blog.example/release",
        "release date: 2024-07-04. This is a long enough fetched body record for the ranking test.",
    )
    structured = {
        "url": "https://api.example/release",
        "title": "release record",
        "structured_evidence_text": "release date: 2024-07-04. Structured API record with direct fields.",
        "evidence_origin": "structured_api_record",
        "source": "api",
    }
    context = build_evidence_context(
        {"query": "release date", "results": [page, structured]},
        constraints={"strategy_config": {"context_source_count": 1}},
    )

    assert context["selected_evidence"][0]["url"] == structured["url"]
    assert context["selected_evidence"][0]["source_quality"]["kind"] == "structured_record"


def test_context_projection_keeps_evidence_available_to_alignment():
    page = _page(
        "https://official.example/release",
        "release date: 2024-07-04. Official release notice with enough source text.",
    )
    context = build_evidence_context(
        {"query": "release date", "results": [page]},
        constraints={"strategy_config": {"context_source_count": 1}},
    )

    alignment = assess_answer_alignment(
        "Release date is 2024-07-04 [S1:C1].",
        context["selected_evidence"],
    )

    assert context["selected_evidence"][0]["evidence_text"]
    assert alignment["aligned_line_count"] == 1
    assert alignment["unsupported_line_count"] == 0
    assert "[S1:C1]" in context["text"]
