from utils.evidence_validation import assess_answer_alignment, build_evidence_validation, source_quality
from agent.retrieval_synthesis import build_evidence_context, _verification_prompt
from agent.evidence_verifier import verify_evidence


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


class _VerifierResponse:
    def __init__(self, content: str):
        self.content = content


class _VerifierModel:
    def __init__(self, content: str):
        self.content = content
        self.prompts = []

    def text_completion(self, prompt, max_tokens=0, stop=None):
        self.prompts.append(prompt)
        return _VerifierResponse(self.content)


def test_independent_verifier_returns_structured_point_decisions():
    model = _VerifierModel(
        '{"schema_version":"evidence_verification.v1","status":"supported",'
        '"completion_ready":true,"points":[{"id":"P1","status":"supported",'
        '"evidence":["S1"],"reason":"direct body statement"}]}'
    )
    result = verify_evidence(
        model,
        query="release date",
        task_plan={"atomic_points": [{"id": "P1", "task": "release date"}]},
        evidence_context={
            "text": "BEGIN EVIDENCE SOURCE S1\nEVIDENCE BODY\nrelease date: 2024-07-04\nEND EVIDENCE SOURCE S1",
            "selected_evidence": [{"url": "https://example.org/release"}],
            "validation": {
                "subquestion_coverage": [{"point_id": "P1", "status": "covered"}],
                "cross_source": {"missing_points": 0},
            },
        },
    )

    assert result["completion_ready"] is True
    assert result["points"][0]["evidence"] == ["S1"]
    assert "Do not answer the user" in model.prompts[0]


def test_mechanical_missing_boundary_overrides_verifier_claim_of_completion():
    model = _VerifierModel(
        '{"status":"supported","completion_ready":true,"points":'
        '[{"id":"P1","status":"supported","evidence":["S1"]}]}'
    )
    result = verify_evidence(
        model,
        query="anniversary date",
        task_plan={"atomic_points": [{"id": "P1", "task": "anniversary date"}]},
        evidence_context={
            "text": "BEGIN EVIDENCE SOURCE S1\nEVIDENCE BODY\nUnrelated page body\nEND EVIDENCE SOURCE S1",
            "selected_evidence": [{"url": "https://example.org/page"}],
            "validation": {
                "subquestion_coverage": [{"point_id": "P1", "status": "missing"}],
                "cross_source": {"missing_points": 1},
            },
        },
    )

    assert result["completion_ready"] is False
    assert result["requires_replan"] is True
    assert result["missing_point_ids"] == ["P1"]


def test_final_model_receives_only_verifier_control_fields():
    prompt = _verification_prompt(
        {
            "status": "needs_more_evidence",
            "completion_ready": False,
            "requires_replan": True,
            "points": [
                {
                    "id": "P1",
                    "status": "missing",
                    "evidence": [],
                    "missing": ["the verifier's free-form factual detail"],
                    "next_queries": ["a verifier-generated routing query"],
                }
            ],
            "missing_point_ids": ["P1"],
            "conflict_point_ids": [],
            "next_queries": ["another verifier-generated routing query"],
        }
    )

    assert "P1" in prompt
    assert "status=missing" in prompt
    assert "the verifier's free-form factual detail" not in prompt
    assert "verifier-generated routing query" not in prompt
