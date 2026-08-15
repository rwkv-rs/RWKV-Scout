from utils.evidence_validation import assess_answer_alignment, build_evidence_validation, source_quality
from utils.retrieval_ranking import rrf_fuse
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


def _grounded_data(results):
    records = []
    for index, item in enumerate(results, start=1):
        records.append(
            {
                "evidence_record_id": f"E-{index}",
                "task_record_id": "P1",
                "title": item.get("title"),
                "url": item.get("url"),
                "chunk_id": f"chunk-{index}",
                "quote": item.get("structured_evidence_text") or item.get("content"),
                "evidence_origin": item.get("evidence_origin"),
                "source": item.get("source"),
            }
        )
    return {
        "query": "release date",
        "results": results,
        "evidence_ledger": {"task_records": [{"task_record_id": "P1", "evidence_records": records}]},
    }


def test_source_quality_does_not_treat_discovery_metadata_as_body_evidence():
    discovery = {"url": "https://example.org/item", "title": "release date", "snippet": "release date"}
    body = _page("https://example.org/item", "release date: 2024-07-04. The official page states this directly.")

    assert source_quality(discovery)["kind"] == "discovery"
    assert source_quality(body)["kind"] == "page_body"
    assert source_quality(body)["score"] > source_quality(discovery)["score"]


def test_independent_query_lanes_do_not_masquerade_as_independent_providers():
    fused = rrf_fuse(
        [
            {
                "provider": "search",
                "ranking_stream": "search::Q1",
                "results": [{"url": "https://example.test/a", "title": "A"}],
            },
            {
                "provider": "search",
                "ranking_stream": "search::Q2",
                "results": [{"url": "https://example.test/a", "title": "A"}],
            },
        ]
    )

    assert fused[0]["ranking_streams"] == ["search::Q1", "search::Q2"]
    assert fused[0]["discovery_providers"] == ["search"]
    assert "multi_provider_discovery" not in source_quality(fused[0])["signals"]


def test_validation_reports_coverage_and_candidate_date_conflict_without_calling_truth():
    results = [
        _page("https://official.example/release", "release date: 2024-07-04. Official release notice."),
        _page("https://archive.example/release", "release date: 2024-07-05. Archived release notice."),
    ]
    report = build_evidence_validation(
        {"results": results},
        query="release date",
        constraints={"task_plan": {"records": [{"id": "P1", "task": "release date"}]}},
        selected=results,
    )

    assert report["is_truth_judgement"] is False
    assert report["task_record_coverage"][0]["status"] == "covered"
    assert report["cross_source"]["multi_source_task_records"] == 1
    assert report["cross_source"]["candidate_conflicts"]


def test_validation_recognizes_a_cjk_ordered_list_without_planner_word_overlap():
    body = _page(
        "https://metro.example/line-1",
        "深圳地铁1号线站点：罗湖、国贸、老街、大剧院、科学馆，按线路顺序排列。",
    )
    report = build_evidence_validation(
        {"results": [body]},
        query="深圳地铁一号线有哪些站点？请列出完整站点，并保持线路顺序。",
        constraints={
            "task_plan": {
                "records": [
                    {
                        "id": "P1",
                        "task": "验证站点列表的完整性和顺序",
                        "objective": "确认列表中的站点数量与线路信息一致",
                        "output_format": "prose",
                    }
                ]
            }
        },
        selected=[body],
    )

    row = report["task_record_coverage"][0]
    assert row["status"] == "covered"
    assert row["sources"]


def test_validation_does_not_cover_unstated_subpoints_from_shared_topic_words():
    body = _page(
        "https://example.org/gpu",
        "Docker Compose can request GPU resources for a container.",
    )
    report = build_evidence_validation(
        {"results": [body]},
        query="docker compose 怎么给容器用GPU",
        constraints={
            "task_plan": {
                "records": [
                    {"id": "P1", "task": "查找 Docker Compose GPU 性能表现"}
                ]
            }
        },
        selected=[body],
    )

    assert report["task_record_coverage"][0]["status"] == "missing"


def test_validation_uses_bilingual_topic_anchors_for_english_official_body():
    body = _page(
        "https://docs.python.org/3/howto/free-threading-python.html",
        (
            "Python support for free threading. Starting with Python 3.13, the official "
            "build can disable the GIL. Build from source with --disable-gil; PYTHON_GIL "
            "and -Xgil control runtime behavior. The threading module and Lock are used "
            "for synchronization. "
        )
        * 4,
    )
    report = build_evidence_validation(
        {"results": [body]},
        query="python 3.14 free threading到底怎么开，查 docs.python.org 官方文档",
        constraints={
            "task_plan": {
                "contract": "rwkv.ecra.runtime.task-plan",
                "goal": "查 docs.python.org 官方文档回答 free threading 问题",
                "records": [
                    {
                        "id": "P1",
                        "question": "查找 Python 3.14 官方文档中关于 free threading 的说明",
                        "fields": ["启用方法"],
                        "time_scope": "timeless",
                    },
                    {
                        "id": "P2",
                        "question": "查找 threading 模块的使用说明",
                        "fields": ["与 free threading 的关系"],
                        "time_scope": "timeless",
                    },
                ],
            }
        },
        selected=[body],
    )

    rows = report["task_record_coverage"]
    assert all(row["status"] == "covered" for row in rows)
    assert all(
        any(source["coverage_basis"] == "bilingual_topic_anchors" for source in row["sources"])
        for row in rows
    )


def test_answer_alignment_marks_supported_and_unmatched_claim_lines():
    sources = [_page("https://official.example/release", "release date: 2024-07-04. Official release notice.")]
    alignment = assess_answer_alignment(
        "Release date is 2024-07-04 [S1].\nThe unrelated founder is Ada Lovelace.",
        sources,
    )

    assert alignment["claim_line_count"] == 2
    assert alignment["aligned_line_count"] == 1
    assert alignment["unsupported_line_count"] == 1


def test_context_builder_routes_structured_evidence_before_ordinary_web_when_limited():
    page = _page(
        "https://blog.example/release",
        "release date: 2024-07-04. This is a long enough fetched body record for the ranking test and source context.",
    )
    structured = {
        "url": "https://api.example/release",
        "title": "release record",
        "structured_evidence_text": "release date: 2024-07-04. Structured API record with direct fields and authoritative publication context.",
        "evidence_origin": "structured_api_record",
        "source": "api",
    }
    context = build_evidence_context(
        _grounded_data([page, structured]),
        constraints={"strategy_config": {"context_source_count": 1}},
    )

    assert context["selected_evidence"][0]["url"] == structured["url"]
    assert context["selected_evidence"][0]["context_selection"]["structured_record"] is True


def test_context_projection_keeps_evidence_available_to_alignment():
    page = _page(
        "https://official.example/release",
        "release date: 2024-07-04. Official release notice with enough source text and publication context.",
    )
    context = build_evidence_context(
        _grounded_data([page]),
        constraints={"strategy_config": {"context_source_count": 1}},
    )

    alignment = assess_answer_alignment(
        "Release date is 2024-07-04 [S1:C1].",
        context["selected_evidence"],
    )

    assert context["selected_evidence"][0]["evidence_text"]
    assert alignment["aligned_line_count"] == 1
    assert alignment["unsupported_line_count"] == 0
    assert "[S1]" in context["text"]
    assert "<locator-chunk-1>" in context["text"]


def test_cross_language_alignment_is_not_reported_as_unsupported():
    selected = [
        {
            "ref_id": "S1",
            "url": "https://nodejs.org/api/globals.html",
            "evidence_text": "Node.js fetch: v21.0.0 no longer experimental.",
            "evidence_origin": "fetched_page_body",
            "evidence_boundary": "fetched_page_or_structured_record_only",
        }
    ]

    alignment = assess_answer_alignment(
        (
            "Node.js \u5185\u7f6e fetch \u4ece v21.0.0 \u8d77"
            "\u4e0d\u518d\u662f\u5b9e\u9a8c\u6027\u529f\u80fd\u3002[S1:C1]"
        ),
        selected,
    )

    assert alignment["unsupported_line_count"] == 0
    assert alignment["non_applicable_line_count"] == 1
    assert alignment["rows"][0]["applicable"] is False
