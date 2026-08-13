import json
from unittest.mock import patch

from agent.page_evidence import (
    _apply_deterministic_candidate_gates,
    build_chunk_candidate_prompt,
    parse_chunk_candidate,
)
from agent.claim_ledger import ClaimLedger
from agent.retrieval_object_contract import (
    attach_result_object_contract,
    explicit_object_targets,
    github_repository_target,
    object_alignment,
    retrieval_request_contract,
    rwkv_subject_alignment,
    source_object_contract,
    task_record_contract,
)
from agent.retrieval_synthesis import build_evidence_context
from agent.retrieval_loop import merge_retrieval_results
from agent.state import RetrievalEpisodeState
from tools.connectors import connector_lookup
from tools.registry import ToolRegistry
from tools.web_search_generic import _merge_candidates
from utils.evidence_quality import clean_page_body
from utils.html_markdown import html_to_markdown


def test_github_target_parser_does_not_consume_natural_language_suffix():
    assert (
        github_repository_target("pydantic/pydantic latest release and date")
        == "pydantic/pydantic"
    )
    assert (
        github_repository_target(
            "Look up https://github.com/oven-sh/setup-bun releases/latest please"
        )
        == "oven-sh/setup-bun"
    )


def test_source_object_keeps_related_github_repositories_distinct():
    target = source_object_contract(
        {"url": "https://github.com/oven-sh/bun/releases/tag/bun-v1.2.0"}
    )
    setup = source_object_contract(
        {"url": "https://github.com/oven-sh/setup-bun/releases/tag/v2.0.2"}
    )
    assert target["source_object_id"] == "github:oven-sh/bun"
    assert setup["source_object_id"] == "github:oven-sh/setup-bun"
    assert target["source_object_id"] != setup["source_object_id"]


def test_result_contract_carries_request_and_source_identity_without_fact_judgment():
    result = attach_result_object_contract(
        {
            "status": "ok",
            "results": [
                {
                    "title": "setup-bun",
                    "url": "https://github.com/oven-sh/setup-bun",
                    "content": "Repository metadata",
                }
            ],
        },
        action="connector_lookup",
        arguments={
            "operation": "github_repository",
            "query": "oven-sh/setup-bun",
        },
        task_record_id="P1",
        task_plan={
            "goal": "find bun release",
            "atomic_points": [
                {
                    "id": "P1",
                    "question": "find bun release",
                    "subject": "oven-sh/bun",
                    "relation": "latest release",
                    "fields": ["tag"],
                    "time_scope": "current",
                }
            ],
        },
    )
    row = result["results"][0]
    assert row["retrieval_request"]["requested_object"]["requested_subject"] == "oven-sh/bun"
    assert row["source_object"]["source_object_id"] == "github:oven-sh/setup-bun"
    assert row["object_alignment"]["relation"] == "conflict"
    assert "correct" not in json.dumps(row).casefold()


def test_explicit_object_targets_cover_repository_registry_and_public_ids():
    values = explicit_object_targets(
        "Compare oven-sh/bun, https://crates.io/crates/serde, "
        "arXiv:2312.00752v2 and CVE-2026-12345"
    )
    assert {row["object_id"] for row in values} == {
        "github:oven-sh/bun",
        "crates:serde",
        "arxiv:2312.00752",
        "cve:cve-2026-12345",
    }


def test_task_record_uses_registry_plus_rwkv_subject_as_typed_scope():
    contract = task_record_contract(
        {
            "goal": "crates.io 上 serde 的最新版本是什么？",
            "atomic_points": [
                {
                    "id": "P1",
                    "question": "crates.io 上 serde 的最新版本是什么？",
                    "subject": "serde",
                    "relation": "latest_version",
                    "fields": ["version"],
                }
            ],
        },
        "P1",
    )
    assert contract["requested_object_targets"] == [
        {
            "object_id": "crates:serde",
            "object_type": "rust_crate",
            "literal": "serde",
            "basis": "explicit_registry_plus_rwkv_subject",
        }
    ]


def test_task_record_does_not_copy_multiple_goal_objects_into_every_point():
    plan = {
        "goal": "Compare owner/repo-a with owner/repo-b",
        "atomic_points": [
            {
                "id": "P1",
                "question": "Find the latest release of owner/repo-a",
                "subject": "owner/repo-a",
                "fields": ["tag"],
            },
            {
                "id": "P2",
                "question": "Find the latest release of owner/repo-b",
                "subject": "owner/repo-b",
                "fields": ["tag"],
            },
        ],
    }
    first = task_record_contract(plan, "P1")
    second = task_record_contract(plan, "P2")
    assert [row["object_id"] for row in first["requested_object_targets"]] == [
        "github:owner/repo-a"
    ]
    assert [row["object_id"] for row in second["requested_object_targets"]] == [
        "github:owner/repo-b"
    ]


def test_task_record_uses_single_goal_object_as_transport_fallback():
    contract = task_record_contract(
        {
            "goal": "Find the latest release of owner/project",
            "atomic_points": [
                {
                    "id": "P1",
                    "question": "Find the release date",
                    "subject": "project",
                    "fields": ["date"],
                }
            ],
        },
        "P1",
    )
    assert [row["object_id"] for row in contract["requested_object_targets"]] == [
        "github:owner/project"
    ]


def test_object_alignment_is_identifier_transport_not_truth_judgment():
    requested = [{"object_id": "github:oven-sh/bun"}]
    assert object_alignment(
        requested, {"source_object_id": "github:oven-sh/bun"}
    )["relation"] == "exact"
    mismatch = object_alignment(
        requested, {"source_object_id": "github:oven-sh/setup-bun"}
    )
    assert mismatch["relation"] == "conflict"
    assert mismatch["transport_only"] is True
    assert rwkv_subject_alignment("serde", "serde-json")["relation"] == "conflict"


def test_candidate_ranking_prioritizes_exact_typed_object_without_dropping_conflict():
    plan = {
        "goal": "oven-sh/bun latest release",
        "atomic_points": [
            {
                "id": "P1",
                "question": "oven-sh/bun latest release",
                "subject": "oven-sh/bun",
                "relation": "latest_release",
                "fields": ["tag"],
            }
        ],
    }
    candidates = _merge_candidates(
        "oven-sh/bun latest release",
        [
            {
                "provider": "test",
                "results": [
                    {
                        "title": "Releases · oven-sh/setup-bun",
                        "url": "https://github.com/oven-sh/setup-bun/releases",
                        "snippet": "bun release",
                    },
                    {
                        "title": "Releases · oven-sh/bun",
                        "url": "https://github.com/oven-sh/bun/releases",
                        "snippet": "bun release",
                    },
                ],
            }
        ],
        limit=2,
        task_plan=plan,
        constraint_query=plan["goal"],
        policy_query=plan["goal"],
    )
    assert candidates[0]["source_object"]["source_object_id"] == "github:oven-sh/bun"
    assert candidates[0]["object_alignment"]["relation"] == "exact"
    assert candidates[1]["object_alignment"]["relation"] == "conflict"


def test_request_identity_is_stable_across_json_argument_order():
    first = retrieval_request_contract(
        "web_search",
        {"query": "example", "max_results": 8},
        task_record_id="P1",
    )
    second = retrieval_request_contract(
        "web_search",
        {"max_results": 8, "query": "example"},
        task_record_id="P1",
    )
    assert first["request_id"] == second["request_id"]


def test_singleton_task_keeps_object_scope_when_optional_point_id_is_omitted():
    request = retrieval_request_contract(
        "web_search",
        {"query": "oven-sh/bun latest release"},
        task_plan={
            "goal": "oven-sh/bun latest release",
            "atomic_points": [
                {
                    "id": "P1",
                    "question": "oven-sh/bun latest release",
                    "subject": "oven-sh/bun",
                    "fields": ["tag"],
                }
            ],
        },
    )
    assert request["task_record_id"] == ""
    assert request["requested_object"]["evidence_binding"] is False
    assert request["requested_object"]["object_scope_basis"] == "singleton_task_plan"
    assert request["requested_object"]["requested_object_targets"][0]["object_id"] == (
        "github:oven-sh/bun"
    )


def test_empty_task_point_does_not_erase_existing_provider_object_alignment():
    result = attach_result_object_contract(
        {
            "status": "ok",
            "results": [
                {
                    "url": "https://github.com/owner/repo-a/releases",
                    "source_object": {
                        "source_object_id": "github:owner/repo-a",
                    },
                    "object_alignment": {
                        "relation": "exact",
                        "requested_object_ids": ["github:owner/repo-a"],
                        "source_object_id": "github:owner/repo-a",
                        "transport_only": True,
                    },
                }
            ],
        },
        action="web_search",
        arguments={"query": "compare repositories"},
        task_plan={
            "goal": "Compare owner/repo-a with owner/repo-b",
            "atomic_points": [
                {"id": "P1", "question": "owner/repo-a", "subject": "owner/repo-a"},
                {"id": "P2", "question": "owner/repo-b", "subject": "owner/repo-b"},
            ],
        },
    )
    assert result["results"][0]["object_alignment"]["relation"] == "exact"
    assert result["retrieval_request"]["task_record_id"] == ""


def test_source_object_uses_explicit_record_id_not_publication_date():
    source = source_object_contract(
        {
            "url": "https://example.test/releases",
            "published": "2026-08-13",
        }
    )
    assert source["source_record_id"] == ""

    arxiv = source_object_contract(
        {"url": "https://arxiv.org/html/2608.05466v1"}
    )
    assert arxiv["source_object_id"] == "arxiv:2608.05466v1"


def test_connector_catalog_has_one_unambiguous_operation_axis():
    catalog = json.loads(
        ToolRegistry.get_json_catalog("ALL", model_visible_only=True)
    )
    schema = next(
        row["arguments"] for row in catalog if row["name"] == "connector_lookup"
    )
    assert schema["required"] == ["operation", "query"]
    assert "scope" not in schema["properties"]
    assert schema["properties"]["operation"]["enum"] == [
        "weather_current",
        "weather_alerts",
        "github_repository",
        "github_code",
        "github_release",
        "paper",
        "paper_series",
        "crates_release",
        "pypi_release",
        "npm_release",
    ]


@patch("tools.connectors.fetch_json")
def test_crates_release_connector_returns_registry_typed_record(fetch_json):
    fetch_json.return_value = {
        "crate": {
            "id": "serde",
            "max_stable_version": "1.0.229",
            "repository": "https://github.com/serde-rs/serde",
        },
        "versions": [
            {
                "num": "1.0.229",
                "created_at": "2026-07-20T12:34:56Z",
                "yanked": False,
            }
        ],
    }
    result = json.loads(connector_lookup("crates_release", "serde"))
    assert result["status"] == "ok"
    assert result["operation"] == "crates_release"
    row = result["results"][0]
    assert row["source_object"]["source_object_id"] == "crates:serde"
    assert "Version: 1.0.229" in row["structured_evidence_text"]
    assert "Published: 2026-07-20T12:34:56Z" in row["structured_evidence_text"]


@patch("tools.connectors.search_papers")
def test_singular_paper_operation_reaches_paper_backend(search_papers):
    search_papers.return_value = json.dumps(
        {
            "status": "ok",
            "results": [
                {
                    "title": "Mamba",
                    "url": "https://arxiv.org/abs/2312.00752",
                    "arxiv_id": "2312.00752",
                    "abstract": "A selective state space model.",
                }
            ],
        }
    )
    result = json.loads(connector_lookup("paper", "Mamba", max_results=1))
    assert result["status"] == "ok"
    assert result["operation"] == "paper"
    assert result["results"][0]["source_object"]["source_object_id"] == "arxiv:2312.00752"


def test_page_extractor_contract_never_declares_chunk_local_currentness():
    prompt = build_chunk_candidate_prompt(
        "What is the latest version?",
        "https://example.test/releases",
        "Release history",
        {"chunk_id": "chunk-1", "index": 0, "text": "Version 1.0 - 2025\nVersion 2.0 - 2026"},
        1,
        task_points=[
            {
                "id": "P1",
                "question": "latest version",
                "subject": "Example",
                "relation": "latest release",
                "fields": ["version"],
                "time_scope": "current",
            }
        ],
    )
    assert "exact_requested_record" not in prompt
    assert "禁止判断该记录是否为全局最新" in prompt
    parsed = parse_chunk_candidate(
        json.dumps(
            {
                "supported": True,
                "record_match": "exact_requested_record",
                "task_record_ids": ["P1"],
                "field_keys": ["version"],
                "source_subject": "Example",
                "source_record_key": "Version 1.0",
                "quote": "Version 1.0 - 2025",
            }
        ),
        {"chunk_id": "chunk-1", "index": 0, "text": "Version 1.0 - 2025"},
        {"P1"},
        {"P1": {"version"}},
    )
    assert parsed["record_match"] == "candidate_record"
    assert parsed["extractor_declared_record_match"] == "exact_requested_record"


def test_record_key_can_locate_verbatim_markdown_table_line_without_rewriting_it():
    source = (
        "| Release | Released | Latest |\n"
        "| 1 | 07 Sep 2023 | [1.3.14](https://github.com/oven-sh/bun/releases/tag/bun-v1.3.14) (12 May 2026) |"
    )
    candidate = {
        "chunk_id": "chunk-1",
        "chunk_index": 0,
        "supported": True,
        "task_record_ids": ["P1"],
        "claim_ids": ["P1"],
        "field_keys": ["release_tag", "release_date"],
        "subject_key": "oven-sh/bun",
        "record_key": "1.3.14 (12 May 2026)",
        "quote": "Released | 1.3.14 (12 May 2026)",
    }
    rows = _apply_deterministic_candidate_gates(
        "oven-sh/bun latest release",
        [{"chunk_id": "chunk-1", "index": 0, "text": source}],
        [candidate],
        {"atomic_points": [{"id": "P1", "fields": ["release_tag", "release_date"]}]},
    )
    assert rows[0]["source_grounded"] is True
    assert rows[0]["quote"] == source.splitlines()[1]
    assert "https://github.com/oven-sh/bun" in rows[0]["quote"]


def test_pre_code_yaml_and_exact_punctuation_survive_cleaning():
    markdown = html_to_markdown(
        "<h2>Compose</h2><pre><code>services:\n  web:\n    image: app:1\n"
        "    command: pip install --require-hashes -r req.txt\n"
        "    # --hash=sha256:abc\n</code></pre>"
    )
    cleaned = clean_page_body(markdown)["text"]
    assert "```" in cleaned
    assert "services:\n  web:\n    image: app:1" in cleaned
    assert "--hash=sha256:abc" in cleaned


def test_context_groups_multiple_spans_from_one_observable_source_record():
    source_object = {
        "schema_version": "retrieval-object.v1",
        "source_object_id": "github:owner/project",
        "source_object_type": "github_repository",
        "source_record_id": "v2.0.0",
        "source_url": "https://github.com/owner/project/releases/tag/v2.0.0",
        "provider": "github",
    }
    records = [
        {
            "evidence_record_id": "E-version",
            "task_record_id": "P1",
            "record_key": "v2.0.0",
            "field_keys": ["version"],
            "quote": "Version v2.0.0",
            "chunk_id": "chunk-1",
            "url": source_object["source_url"],
            "source_object": source_object,
        },
        {
            "evidence_record_id": "E-date",
            "task_record_id": "P1",
            "record_key": "v2.0.0",
            "field_keys": ["date"],
            "quote": "Published 2026-08-13",
            "chunk_id": "chunk-2",
            "url": source_object["source_url"],
            "source_object": source_object,
        },
    ]
    context = build_evidence_context(
        {
            "results": [],
            "claim_ledger": {
                "claims": [
                    {
                        "claim_id": "P1",
                        "question": "latest release",
                        "fields": ["version", "date"],
                        "evidence_records": records,
                    }
                ]
            },
        },
        constraints={"context_source_count": 8},
        query="latest release",
    )
    assert context["context_stats"]["source_count"] == 1
    assert "Version v2.0.0" in context["text"]
    assert "Published 2026-08-13" in context["text"]
    metadata = context["selected_evidence"][0]["record_metadata"]
    assert metadata["field_keys"] == ["version", "date"]


def test_related_repository_objects_stay_distinct_from_tool_result_to_writer_context():
    plan = {
        "goal": "find the latest oven-sh/bun release",
        "atomic_points": [
            {
                "id": "P1",
                "question": "latest oven-sh/bun release",
                "subject": "oven-sh/bun",
                "relation": "latest release",
                "fields": ["tag"],
                "time_scope": "current",
            }
        ],
    }
    rows = []
    for full_name, tag in (
        ("oven-sh/bun", "bun-v1.2.0"),
        ("oven-sh/setup-bun", "v2.0.2"),
    ):
        quote = f"{full_name} release {tag}"
        rows.append(
            {
                "title": full_name,
                "full_name": full_name,
                "url": f"https://github.com/{full_name}/releases/tag/{tag}",
                "tag_name": tag,
                "content": quote,
                "chunk_candidates": [
                    {
                        "chunk_id": "structured-record",
                        "chunk_index": 0,
                        "supported": True,
                        "source_grounded": True,
                        "task_record_ids": ["P1"],
                        "field_keys": ["tag"],
                        "source_subject": full_name,
                        "source_record_key": tag,
                        "record_key": tag,
                        "quote": quote,
                    }
                ],
            }
        )
    result = attach_result_object_contract(
        {"status": "ok", "connector": "github", "results": rows},
        action="connector_lookup",
        arguments={"operation": "github_release", "query": "oven-sh/bun"},
        task_record_id="P1",
        task_plan=plan,
    )
    ledger = ClaimLedger()
    ledger.initialize(plan, plan["goal"])
    ledger.ingest(
        "oven-sh/bun",
        result,
        task_point_id="P1",
        strategy="rwkv_selected",
        step=1,
    )
    context = build_evidence_context(
        {"results": result["results"], "claim_ledger": ledger.snapshot()},
        constraints={"task_plan": plan, "context_source_count": 8},
        query=plan["goal"],
    )
    source_ids = {
        row["source_object"]["source_object_id"]
        for row in context["selected_evidence"]
    }
    assert source_ids == {"github:oven-sh/bun", "github:oven-sh/setup-bun"}
    assert context["context_stats"]["source_count"] == 2
    assert (
        context["selected_evidence"][0]["source_object"]["source_object_id"]
        == "github:oven-sh/bun"
    )


def test_omitted_optional_point_id_keeps_exact_object_first_end_to_end():
    plan = {
        "goal": "find the latest oven-sh/bun release",
        "atomic_points": [
            {
                "id": "P1",
                "question": "latest oven-sh/bun release",
                "subject": "oven-sh/bun",
                "relation": "latest release",
                "fields": ["tag"],
                "time_scope": "current",
            }
        ],
    }
    rows = []
    for full_name, tag in (
        ("oven-sh/setup-bun", "v2.2.0"),
        ("oven-sh/bun", "bun-v1.3.14"),
    ):
        quote = f"{full_name} release {tag}"
        rows.append(
            {
                "title": full_name,
                "full_name": full_name,
                "url": f"https://github.com/{full_name}/releases/tag/{tag}",
                "content": quote,
                "chunk_candidates": [
                    {
                        "chunk_id": "chunk-1",
                        "supported": True,
                        "source_grounded": True,
                        "task_record_ids": ["P1"],
                        "field_keys": ["tag"],
                        "source_subject": full_name,
                        "source_record_key": tag,
                        "record_key": tag,
                        "quote": quote,
                        "object_alignment": object_alignment(
                            [{"object_id": "github:oven-sh/bun"}],
                            {"source_object_id": f"github:{full_name}"},
                        ),
                    }
                ],
            }
        )
    result = attach_result_object_contract(
        {"status": "ok", "results": rows},
        action="web_search",
        arguments={"query": "oven-sh/bun latest release"},
        task_plan=plan,
    )
    assert result["retrieval_request"]["task_record_id"] == ""
    assert [row["object_alignment"]["relation"] for row in result["results"]] == [
        "conflict",
        "exact",
    ]
    ledger = ClaimLedger()
    ledger.initialize(plan, plan["goal"])
    ledger.ingest(plan["goal"], result, task_point_id="", step=1)
    context = build_evidence_context(
        {"results": result["results"], "claim_ledger": ledger.snapshot()},
        constraints={"task_plan": plan, "context_source_count": 8},
        query=plan["goal"],
    )
    assert (
        context["selected_evidence"][0]["source_object"]["source_object_id"]
        == "github:oven-sh/bun"
    )


def test_unkeyed_spans_from_one_source_object_share_one_unresolved_block():
    source_object = {
        "source_object_id": "url:https://docs.example.test/guide",
        "source_object_type": "page_body",
        "source_record_id": "",
        "source_url": "https://docs.example.test/guide",
    }
    records = [
        {
            "evidence_record_id": "E-1",
            "task_record_id": "P1",
            "record_key": "",
            "field_keys": ["permission"],
            "quote": "id-token: write",
            "chunk_id": "chunk-1",
            "url": source_object["source_url"],
            "source_object": source_object,
        },
        {
            "evidence_record_id": "E-2",
            "task_record_id": "P1",
            "record_key": "",
            "field_keys": ["example"],
            "quote": "contents: read",
            "chunk_id": "chunk-2",
            "url": source_object["source_url"],
            "source_object": source_object,
        },
    ]
    context = build_evidence_context(
        {
            "claim_ledger": {
                "claims": [
                    {
                        "claim_id": "P1",
                        "question": "minimal permissions",
                        "fields": ["permission", "example"],
                        "evidence_records": records,
                    }
                ]
            }
        },
        query="minimal permissions",
    )
    assert context["context_stats"]["source_count"] == 1
    assert "record identity unresolved" in context["text"]
    assert "id-token: write" in context["text"]
    assert "contents: read" in context["text"]


def test_shared_source_merge_preserves_task_scoped_candidates_and_bindings():
    state = RetrievalEpisodeState()
    state.claims.initialize(
        {
            "goal": "Read two records from one release page",
            "atomic_points": [
                {"id": "P1", "question": "version", "fields": ["version"]},
                {"id": "P2", "question": "date", "fields": ["date"]},
            ],
        },
        "Read two records from one release page",
    )

    def result(point_id: str, quote: str, field: str) -> dict:
        return {
            "status": "ok",
            "results": [
                {
                    "url": "https://example.test/releases",
                    "title": "Releases",
                    "content": quote,
                    "claim_ids": [point_id],
                    "chunk_candidates": [
                        {
                            "supported": True,
                            "source_grounded": True,
                            "chunk_id": f"chunk-{point_id}",
                            "task_record_ids": [point_id],
                            "field_keys": [field],
                            "quote": quote,
                        }
                    ],
                    "retrieval_request": {
                        "request_id": f"R-{point_id}",
                        "task_record_id": point_id,
                    },
                    "object_alignment": {
                        "relation": "not_explicitly_scoped",
                    },
                }
            ],
        }

    state.record_query("version query", result("P1", "Version 2.0", "version"), task_point_id="P1")
    state.record_query("date query", result("P2", "Released 2026-08-13", "date"), task_point_id="P2")

    source = state.source_records()[0]
    assert {row["quote"] for row in source["chunk_candidates"]} == {
        "Version 2.0",
        "Released 2026-08-13",
    }
    assert {row["task_record_id"] for row in source["retrieval_bindings"]} == {
        "P1",
        "P2",
    }


def test_shared_source_merge_preserves_later_exact_object_alignment():
    state = RetrievalEpisodeState()

    def payload(relation: str, request_id: str) -> dict:
        return {
            "status": "ok",
            "results": [
                {
                    "url": "https://github.com/owner/project/releases",
                    "title": "owner/project releases",
                    "content": "Version 2.0 released 2026-08-13.",
                    "retrieval_request": {"request_id": request_id},
                    "object_alignment": {"relation": relation},
                }
            ],
        }

    first = state.record_query("project release", payload("unresolved", "R-1"), step=1)
    second = state.record_query("owner/project release", payload("exact", "R-2"), step=2)

    source = state.source_records()[0]
    assert first["evidence_revision"] == 1
    assert second["evidence_revision"] == 2
    assert {row["relation"] for row in source["object_alignments"]} == {
        "unresolved",
        "exact",
    }


def test_shared_source_refreshes_every_bound_task_point_with_canonical_source():
    state = RetrievalEpisodeState()

    def payload(point_id: str, quote: str, request_id: str) -> dict:
        return {
            "status": "ok",
            "results": [
                {
                    "url": "https://example.test/releases",
                    "title": "Release table",
                    "content": "Version 2.0\nReleased 2026-08-13",
                    "claim_ids": [point_id],
                    "retrieval_request": {
                        "request_id": request_id,
                        "task_record_id": point_id,
                    },
                    "object_alignment": {"relation": "exact"},
                    "chunk_candidates": [
                        {
                            "supported": True,
                            "source_grounded": True,
                            "chunk_id": "table-record-1",
                            "quote": quote,
                            "task_record_ids": [point_id],
                        }
                    ],
                }
            ],
        }

    state.record_query("version", payload("P1", "Version 2.0", "R-1"), task_point_id="P1")
    state.record_query(
        "release date",
        payload("P2", "Released 2026-08-13", "R-2"),
        task_point_id="P2",
    )

    p1_source = next(iter(state.sources_by_claim["P1"].values()))
    p2_source = next(iter(state.sources_by_claim["P2"].values()))
    for source in (p1_source, p2_source):
        assert set(source["claim_ids"]) == {"P1", "P2"}
        assert {row["request_id"] for row in source["retrieval_requests"]} == {
            "R-1",
            "R-2",
        }
        assert {row["task_record_id"] for row in source["retrieval_bindings"]} == {
            "P1",
            "P2",
        }


def test_round_merge_unions_task_bindings_for_the_same_grounded_span():
    url = "https://example.test/releases"

    def result(point_id: str) -> dict:
        return {
            "status": "ok",
            "results": [
                {
                    "url": url,
                    "title": "Releases",
                    "content": (
                        "Version 2.0 released 2026-08-13 with the documented "
                        "project changes and compatibility notes for supported users. "
                        "This official release record preserves the complete source row."
                    ),
                    "evidence_origin": "fetched_page_body",
                    "body_verified": True,
                    "chunk_candidates": [
                        {
                            "supported": True,
                            "source_grounded": True,
                            "chunk_id": "table-record-1",
                            "quote": "Version 2.0 released 2026-08-13.",
                            "task_record_ids": [point_id],
                            "claim_ids": [point_id],
                        }
                    ],
                }
            ],
        }

    merged = merge_retrieval_results(
        "version and date",
        "web_search",
        [("version", result("P1")), ("date", result("P2"))],
    )

    candidates = merged["results"][0]["chunk_candidates"]
    assert len(candidates) == 1
    assert set(candidates[0]["task_record_ids"]) == {"P1", "P2"}
    assert set(candidates[0]["claim_ids"]) == {"P1", "P2"}


def test_object_identity_survives_round_merge_state_ledger_and_writer_context():
    plan = {
        "schema_version": "task_plan.v2",
        "goal": "Find the current release of owner/project.",
        "atomic_points": [
            {
                "id": "P1",
                "question": "What is the current owner/project release?",
                "subject": "owner/project",
                "fields": ["version", "date"],
                "time_scope": "current",
            }
        ],
    }
    url = "https://github.com/owner/project/releases"
    body = (
        "owner/project release v2.0 was published on 2026-08-13. "
        "The official release record includes compatibility notes, artifacts, "
        "and the complete changelog for the current project version."
    )

    def result(relation: str, request_id: str) -> dict:
        return {
            "status": "ok",
            "results": [
                {
                    "url": url,
                    "title": "owner/project releases",
                    "content": body,
                    "source_excerpt": body,
                    "evidence_origin": "fetched_page_body",
                    "body_verified": True,
                    "claim_ids": ["P1"],
                    "source_object": {
                        "source_object_id": "github:owner/project",
                        "source_object_type": "github_repository",
                    },
                    "object_alignment": {"relation": relation},
                    "retrieval_request": {
                        "request_id": request_id,
                        "task_record_id": "P1",
                    },
                    "chunk_candidates": [
                        {
                            "supported": True,
                            "source_grounded": True,
                            "chunk_id": "release-record-v2",
                            "quote": "owner/project release v2.0 was published on 2026-08-13.",
                            "task_record_ids": ["P1"],
                            "claim_ids": ["P1"],
                            "record_key": "v2.0",
                            "field_keys": ["version", "date"],
                        }
                    ],
                }
            ],
        }

    merged = merge_retrieval_results(
        plan["goal"],
        "web_search",
        [
            ("project release", result("unresolved", "R-1")),
            ("owner/project exact release", result("exact", "R-2")),
        ],
    )
    state = RetrievalEpisodeState()
    state.claims.initialize(plan, plan["goal"])
    state.record_query(
        "owner/project release",
        merged,
        task_point_id="P1",
        step=1,
    )

    source = state.source_records()[0]
    claim_snapshot = state.claims.snapshot()
    record = claim_snapshot["claims"][0]["evidence_records"][0]
    planner_source = state.planner_evidence_snapshot()["sources"][0]
    planner_record = state.planner_record_snapshot()["records"][0]
    context = build_evidence_context(
        {"results": state.source_records(), "claim_ledger": claim_snapshot},
        constraints={"task_plan": plan, "context_source_count": 2},
        query=plan["goal"],
    )

    for row in (source, record, planner_source, planner_record):
        assert {item["relation"] for item in row["object_alignments"]} == {
            "unresolved",
            "exact",
        }
    assert (
        context["selected_evidence"][0]["context_selection"][
            "object_alignment_relation"
        ]
        == "exact"
    )
    assert "owner/project release v2.0" in context["text"]


def test_round_merge_preserves_later_candidates_and_all_object_alignments():
    url = "https://github.com/owner/project/releases"
    merged = merge_retrieval_results(
        "owner/project releases",
        "web_search",
        [
            (
                "first query",
                {
                    "status": "ok",
                    "results": [
                        {
                            "url": url,
                            "title": "Releases",
                            "content": "Official project release record. Version 1.0 is available for users.",
                            "object_alignment": {"relation": "unresolved"},
                            "retrieval_request": {"request_id": "R-1"},
                            "chunk_candidates": [
                                {
                                    "supported": True,
                                    "chunk_id": "chunk-1",
                                    "task_record_ids": ["P1"],
                                    "quote": "Version 1.0",
                                }
                            ],
                        }
                    ],
                },
            ),
            (
                "second query",
                {
                    "status": "ok",
                    "results": [
                        {
                            "url": url,
                            "title": "Releases",
                            "content": (
                                "Official project release record. Version 1.0 is available for users.\n"
                                "Released 2026-08-13 with the documented project changes."
                            ),
                            "source_object": {
                                "source_object_id": "github:owner/project"
                            },
                            "object_alignment": {"relation": "exact"},
                            "retrieval_request": {"request_id": "R-2"},
                            "chunk_candidates": [
                                {
                                    "supported": True,
                                    "chunk_id": "chunk-2",
                                    "task_record_ids": ["P2"],
                                    "quote": "Released 2026-08-13",
                                }
                            ],
                        }
                    ],
                },
            ),
        ],
    )
    row = merged["results"][0]
    assert {item["quote"] for item in row["chunk_candidates"]} == {
        "Version 1.0",
        "Released 2026-08-13",
    }
    assert {item["relation"] for item in row["object_alignments"]} == {
        "unresolved",
        "exact",
    }
    assert {item["request_id"] for item in row["retrieval_requests"]} == {
        "R-1",
        "R-2",
    }
    context = build_evidence_context(
        {"results": [row]},
        constraints={"context_source_count": 1},
        query="owner/project releases",
    )
    assert (
        context["selected_evidence"][0]["context_selection"][
            "object_alignment_relation"
        ]
        == "exact"
    )
