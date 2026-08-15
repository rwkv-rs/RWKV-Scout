from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from agent.retrieval_query_plan import (
    generate_retrieval_query_plan,
    parse_retrieval_query_plan_output,
    select_retrieval_query_records,
)
from agent.state import RetrievalEpisodeState
from agent.task_plan_contract import normalize_task_plan
from tools.web_search_generic import (
    _admit_cached_and_novel_candidates,
    _merge_candidates,
    web_search,
)
from utils.retrieval_ranking import rrf_fuse
from utils.hard_literals import hard_literal_keys


def _plan() -> dict:
    return normalize_task_plan({
        "goal": "Compare the current releases of Alpha and Beta",
        "records": [
            {
                "id": "P1",
                "question": "What is the current Alpha release and release date?",
                "subject": "Alpha",
                "relation": "current release",
                "fields": ["version", "release date"],
                "time_scope": "current",
            },
            {
                "id": "P2",
                "question": "What is the current Beta release and release date?",
                "subject": "Beta",
                "relation": "current release",
                "fields": ["version", "release date"],
                "time_scope": "current",
            },
        ],
    })


class _FakeLLM:
    provider = "local_13b"

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def text_completion(self, prompt: str, **_kwargs):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.outputs.pop(0))


def test_select_retrieval_query_records_prefers_active_then_uncovered_records():
    snapshot = {
        "task_records": [
            {"task_record_id": "P1", "retrieval_state": "evidence_recorded"},
            {"task_record_id": "P2", "retrieval_state": "not_recorded"},
        ]
    }
    selected = select_retrieval_query_records(
        _plan(), query="current releases", evidence_ledger_snapshot=snapshot
    )
    assert [row["record_id"] for row in selected] == ["P2"]

    active = select_retrieval_query_records(
        _plan(),
        query="Alpha current release",
        task_record_id="P1",
        evidence_ledger_snapshot=snapshot,
    )
    assert [row["record_id"] for row in active] == ["P1"]


def test_parse_retrieval_query_plan_keeps_primary_assigns_ids_and_deduplicates():
    selected = select_retrieval_query_records(
        _plan(), query="Alpha release", task_record_id="P1"
    )
    parsed = parse_retrieval_query_plan_output(
        json.dumps(
            {
                "contract": "rwkv.ecra.runtime.retrieval-query-plan",
                "queries": [
                    {
                        "task_record_id": "P1",
                        "intent": "official_primary",
                        "query": "Alpha release",
                    },
                    {
                        "task_record_id": "P1",
                        "intent": "temporal_version",
                        "query": "Alpha latest release notes release date",
                    },
                    {
                        "task_record_id": "P1",
                        "intent": "verification_counterevidence",
                        "query": "Alpha independent release verification",
                    },
                ],
            }
        ),
        primary_query="Alpha release",
        selected_records=selected,
        task_record_id="P1",
        max_queries=4,
        recent_queries=["Alpha independent release verification"],
    )
    assert parsed["expanded"] is True
    assert parsed["queries"] == [
        {
            "query_id": "Q1",
            "task_record_id": "P1",
            "intent": "planner_primary",
            "query": "Alpha release",
            "origin": "planner",
        },
        {
            "query_id": "Q2",
            "task_record_id": "P1",
            "intent": "temporal_version",
            "query": "Alpha latest release notes release date",
            "origin": "rwkv_retrieval_query_plan",
        },
    ]


def test_generate_retrieval_query_plan_retries_protocol_and_preserves_primary():
    llm = _FakeLLM(
        [
            "not json",
            json.dumps(
                {
                    "contract": "rwkv.ecra.runtime.retrieval-query-plan",
                    "queries": [
                        {
                            "task_record_id": "P2",
                            "intent": "official_primary",
                            "query": "Beta official release notes current version",
                        }
                    ],
                }
            ),
        ]
    )
    with patch.dict(
        "agent.retrieval_query_plan.DATA_PIPELINE",
        {
            "retrieval_query_plan_protocol_retries": 1,
            "retrieval_query_plan_max_tokens": 640,
        },
        clear=False,
    ):
        result = generate_retrieval_query_plan(
            "Beta current release",
            "Compare Alpha and Beta current releases",
            _plan(),
            llm,
            task_record_id="P2",
        )
    assert result["status"] == "ok"
    assert result["attempts"] == 2
    assert result["queries"][0]["query"] == "Beta current release"
    assert result["queries"][1]["task_record_id"] == "P2"
    assert "PROTOCOL CORRECTION" in llm.prompts[1]


def test_generate_retrieval_query_plan_falls_back_without_rewriting_primary():
    llm = _FakeLLM(["still not json"])
    with patch.dict(
        "agent.retrieval_query_plan.DATA_PIPELINE",
        {"retrieval_query_plan_protocol_retries": 0},
        clear=False,
    ):
        result = generate_retrieval_query_plan(
            "literal Planner query 2026",
            "goal",
            _plan(),
            llm,
            task_record_id="P1",
        )
    assert result["status"] == "unavailable"
    assert result["expanded"] is False
    assert [row["query"] for row in result["queries"]] == [
        "literal Planner query 2026"
    ]


def test_query_plan_rejects_only_rows_that_introduce_untrusted_hard_literals():
    selected = select_retrieval_query_records(_plan(), query="Alpha release", task_record_id="P1")
    parsed = parse_retrieval_query_plan_output(
        json.dumps({
            "contract": "rwkv.ecra.runtime.retrieval-query-plan",
            "queries": [
                {
                    "task_record_id": "P1",
                    "intent": "temporal_version",
                    "query": "Alpha 2025 release notes v4.2",
                },
                {
                    "task_record_id": "P1",
                    "intent": "official_primary",
                    "query": "Alpha official current release notes",
                },
            ],
        }),
        primary_query="Alpha release",
        selected_records=selected,
        task_record_id="P1",
        allowed_hard_literal_keys=set(),
    )

    assert [row["query"] for row in parsed["queries"]] == [
        "Alpha release",
        "Alpha official current release notes",
    ]
    assert parsed["rejected_query_count"] == 1
    assert {row["kind"] for row in parsed["rejected_queries"][0]["hard_literals"]} == {
        "year",
        "version",
    }


def test_query_plan_accepts_current_date_and_task_record_anchors():
    plan = normalize_task_plan(
        {
            "goal": "Alpha v4.2 current release",
            "records": [
                {
                    "question": "What changed in Alpha v4.2?",
                    "fields": ["release date"],
                    "time_scope": "current",
                }
            ],
        }
    )
    selected = plan["records"]
    allowed = hard_literal_keys(
        [
            plan["goal"],
            json.dumps(selected),
            "current_utc_date=2026-08-14",
        ]
    )
    parsed = parse_retrieval_query_plan_output(
        json.dumps({
            "contract": "rwkv.ecra.runtime.retrieval-query-plan",
            "queries": [
                {
                    "task_record_id": "P1",
                    "intent": "temporal_version",
                    "query": "Alpha v4.2 official release 2026",
                }
            ]
        }),
        primary_query="Alpha v4.2 current release",
        selected_records=selected,
        task_record_id="P1",
        allowed_hard_literal_keys=allowed,
    )

    assert parsed["rejected_query_count"] == 0
    assert parsed["queries"][1]["query"] == "Alpha v4.2 official release 2026"


def test_rrf_treats_each_query_lane_as_an_independent_ranking_stream():
    fused = rrf_fuse(
        [
            {
                "provider": "search",
                "ranking_stream": "search::Q1",
                "retrieval_query_id": "Q1",
                "retrieval_query_intent": "planner_primary",
                "retrieval_query_text": "Alpha release",
                "results": [
                    {"url": "https://example.com/shared", "title": "Shared"},
                    {"url": "https://example.com/a", "title": "A"},
                ],
            },
            {
                "provider": "search",
                "ranking_stream": "search::Q2",
                "retrieval_query_id": "Q2",
                "retrieval_query_intent": "official_primary",
                "retrieval_query_text": "Alpha official release notes",
                "results": [
                    {"url": "https://example.com/shared", "title": "Shared"},
                    {"url": "https://example.com/b", "title": "B"},
                ],
            },
        ],
        pool_limit=8,
    )
    assert fused[0]["url"] == "https://example.com/shared"
    assert fused[0]["provider_ranks"] == {
        "search::Q1": 1,
        "search::Q2": 1,
    }
    assert fused[0]["discovery_providers"] == ["search"]
    assert fused[0]["ranking_streams"] == ["search::Q1", "search::Q2"]
    assert [row["query_id"] for row in fused[0]["discovery_queries"]] == [
        "Q1",
        "Q2",
    ]


def test_cached_top_slice_does_not_hide_novel_candidates_below_it():
    candidates = [
        {"url": f"https://source{index}.example/item", "candidate_rank": index}
        for index in range(1, 7)
    ]
    seen = {
        "https://source1.example/item",
        "https://source2.example/item",
    }

    cached, novel = _admit_cached_and_novel_candidates(
        candidates,
        seen,
        limit=2,
        per_domain_limit=1,
    )

    assert [row["candidate_rank"] for row in cached] == [1, 2]
    assert [row["candidate_rank"] for row in novel] == [3, 4]


def test_candidate_relevance_uses_the_best_matching_query_plan_lane():
    candidates = _merge_candidates(
        "Python GIL status",
        [
            {
                "provider": "search",
                "results": [
                    {
                        "url": "https://docs.python.org/free-threading.html",
                        "title": "Python free threading official documentation",
                        "snippet": "How to enable free threading in Python",
                        "discovery_queries": [
                            {
                                "query_id": "Q2",
                                "task_record_id": "P1",
                                "intent": "alias_cross_language",
                                "query": "Python free threading official docs",
                            }
                        ],
                    }
                ],
            }
        ],
        constraint_query="Python GIL status and free threading",
    )
    assert candidates[0]["query_relevance"]["matched_query"]["query_id"] == "Q2"
    assert candidates[0]["query_relevance"]["evaluated_query_count"] == 2


def test_web_search_runs_query_lanes_concurrently_and_fetches_canonical_url_once():
    query_plan = {
        "contract": "rwkv.ecra.runtime.retrieval-query-plan",
        "status": "ok",
        "expanded": True,
        "attempts": 1,
        "queries": [
            {
                "query_id": "Q1",
                "task_record_id": "P1",
                "intent": "planner_primary",
                "query": "Alpha current release",
                "origin": "planner",
            },
            {
                "query_id": "Q2",
                "task_record_id": "P1",
                "intent": "official_primary",
                "query": "Alpha official release notes current version",
                "origin": "rwkv_retrieval_query_plan",
            },
        ],
    }
    barrier = threading.Barrier(4)
    seen_queries: list[str] = []
    seen_lock = threading.Lock()

    def provider(query: str, **_kwargs):
        with seen_lock:
            seen_queries.append(query)
        barrier.wait(timeout=2)
        return {
            "status": "ok",
            "provider": "test search",
            "count": 1,
            "results": [
                {
                    "title": "Alpha official release notes",
                    "url": "https://alpha.example/releases/current",
                    "snippet": "Alpha current version and release date",
                    "source": "test search",
                }
            ],
        }

    body = "Alpha official release notes contain the current version and release date. " * 8
    record = {
        "title": "Alpha official release notes",
        "url": "https://alpha.example/releases/current",
        "content": body,
        "source_excerpt": body,
        "body_verified": True,
        "evidence_origin": "fetched_page_body",
    }
    with (
        patch(
            "tools.web_search_generic.generate_retrieval_query_plan",
            return_value=query_plan,
        ),
        patch.dict(
            "tools.web_search_generic.DATA_PIPELINE",
            {"retrieval_query_plan_enabled": True},
        ),
        patch("tools.web_search_generic.search_web_keyless", side_effect=provider),
        patch("tools.web_search_generic.search_web_tavily", side_effect=provider),
        patch(
            "tools.web_search_generic._fetch_candidate",
            return_value={"status": "ok", "results": []},
        ) as fetch,
        patch(
            "tools.web_search_generic._compact_page",
            return_value=(record, {"status": "ok"}),
        ),
        patch("tools.web_search_generic.append_task_event"),
    ):
        result = json.loads(
            web_search(
                "Alpha current release",
                task_plan=normalize_task_plan({
                    "goal": "Alpha current release",
                    "records": [_plan()["records"][0]],
                }),
                task_record_id="P1",
                original_goal="Alpha current release",
                task_id="QUERY_FANOUT_TEST",
                agentic_tool_loop=True,
            )
        )

    assert sorted(seen_queries) == sorted(
        ["Alpha current release", "Alpha current release", "Alpha official release notes current version", "Alpha official release notes current version"]
    )
    assert result["retrieval_query_plan"]["expanded"] is True
    assert result["candidate_count"] == 1
    assert len(result["candidate_urls"][0]["discovery_queries"]) == 2
    fetch.assert_called_once()


def test_retrieval_state_projects_executed_query_plan_queries_to_planner():
    state = RetrievalEpisodeState()
    state.record_query(
        "Alpha release",
        {
            "status": "no_results",
            "results": [],
            "retrieval_query_plan": {
                "status": "ok",
                "queries": [
                    {
                        "query_id": "Q1",
                        "task_record_id": "P1",
                        "intent": "planner_primary",
                        "query": "Alpha release",
                    },
                    {
                        "query_id": "Q2",
                        "task_record_id": "P1",
                        "intent": "official_primary",
                        "query": "Alpha official release notes",
                    },
                ],
            },
        },
        action="web_search",
        arguments={"query": "Alpha release"},
        task_record_id="P1",
    )
    snapshot = state.planner_routing_snapshot(max_queries=1)
    assert snapshot["queries"][0]["executed_queries"][1]["query"] == (
        "Alpha official release notes"
    )


def test_next_query_plan_receives_every_executed_query_lane():
    retrieval = RetrievalEpisodeState()
    retrieval.record_query(
        "Alpha release",
        {
            "status": "no_results",
            "results": [],
            "retrieval_query_plan": {
                "status": "ok",
                "queries": [
                    {"query_id": "Q1", "query": "Alpha release"},
                    {
                        "query_id": "Q2",
                        "query": "Alpha official release notes",
                    },
                ],
            },
        },
        action="web_search",
        arguments={"query": "Alpha release"},
        task_record_id="P1",
    )
    state = SimpleNamespace(retrieval=retrieval)
    empty = {"status": "no_results", "provider": "test", "results": []}
    next_plan = {
        "contract": "rwkv.ecra.runtime.retrieval-query-plan",
        "status": "ok",
        "expanded": False,
        "attempts": 1,
        "queries": [
            {
                "query_id": "Q1",
                "task_record_id": "P1",
                "intent": "planner_primary",
                "query": "Alpha verification",
                "origin": "planner",
            }
        ],
    }
    with (
        patch(
            "tools.web_search_generic.generate_retrieval_query_plan",
            return_value=next_plan,
        ) as query_plan,
        patch.dict(
            "tools.web_search_generic.DATA_PIPELINE",
            {"retrieval_query_plan_enabled": True},
        ),
        patch("tools.web_search_generic.search_web_keyless", return_value=empty),
        patch("tools.web_search_generic.search_web_tavily", return_value=empty),
        patch("tools.web_search_generic.append_task_event"),
    ):
        web_search(
            "Alpha verification",
            task_plan=_plan(),
            task_record_id="P1",
            original_goal="Compare Alpha and Beta current releases",
            task_id="FANOUT_HISTORY_TEST",
            agentic_tool_loop=True,
            agent_state=state,
        )

    assert query_plan.call_args.kwargs["recent_queries"] == [
        "Alpha release",
        "Alpha official release notes",
    ]
