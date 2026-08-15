from agent.evidence_ledger import EvidenceLedger, locate_grounded_quote_span, source_quote_view
from agent.evidence_records import assemble_grounded_candidates
from agent.task_plan_contract import normalize_task_plan


def _plan():
    return normalize_task_plan(
        {
            "goal": "answer two facts",
            "records": [
                {"question": "first fact", "fields": ["date"]},
                {"question": "second fact", "fields": ["version"]},
            ],
        }
    )


def _result(url="https://example.com/a", *, task_record_id="", record_key=""):
    candidates = []
    if task_record_id:
        candidates.append(
            {
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "supported": True,
                "source_grounded": True,
                "task_record_ids": [task_record_id],
                "task_record_ids": [task_record_id],
                "field_ids": [f"{task_record_id}:F1"],
                "subject_key": "Source A",
                "record_key": record_key,
                "quote": "Original fetched source text.",
                "source_locator": {"char_start": 0, "char_end": 29},
                "grounding_basis": "exact",
            }
        )
    return {
        "status": "ok",
        "results": [
            {
                "title": "Source A",
                "url": url,
                "content": "Original fetched source text.",
                "source_chunks": [
                    {"chunk_id": "chunk-1", "index": 0, "text": "Original fetched source text."}
                ],
                "chunk_candidates": candidates,
            }
        ],
    }


def test_ledger_preserves_rwkv_task_records_without_semantic_rewrite():
    ledger = EvidenceLedger()
    ledger.initialize(_plan(), "question")
    snapshot = ledger.snapshot()
    assert [row["task_record_id"] for row in snapshot["task_records"]] == ["P1", "P2"]
    assert [row["question"] for row in snapshot["task_records"]] == ["first fact", "second fact"]
    assert snapshot["contract"] == "rwkv.ecra.runtime.evidence-ledger"
    assert snapshot["task_records"][0]["fields"] == [
        {"field_id": "P1:F1", "name": "date"}
    ]
    assert snapshot["advisory_only"] is True


def test_route_scope_does_not_bind_a_whole_page_to_selected_point():
    ledger = EvidenceLedger()
    ledger.initialize(_plan(), "question")
    delta = ledger.ingest("query", _result(), task_record_id="P2", strategy="rwkv_selected", step=1)
    snapshot = ledger.snapshot()
    by_id = {row["task_record_id"]: row for row in snapshot["task_records"]}
    assert delta["touched_task_record_ids"] == []
    assert by_id["P1"]["retrieval_state"] == "not_recorded"
    assert by_id["P2"]["retrieval_state"] == "not_recorded"
    assert by_id["P2"]["attempt_count"] == 1
    assert snapshot["unassigned_source_count"] == 1


def test_rwkv_grounded_span_binds_one_evidence_record():
    ledger = EvidenceLedger()
    ledger.initialize(_plan(), "question")

    delta = ledger.ingest(
        "query",
        _result(task_record_id="P2", record_key="v2"),
        task_record_id="P2",
    )
    point = ledger.snapshot()["task_records"][1]

    assert delta["added_evidence_records"] == 1
    assert point["retrieval_state"] == "evidence_recorded"
    assert point["evidence_record_count"] == 1
    record = point["evidence_records"][0]
    assert record["task_record_id"] == "P2"
    assert record["record_key"] == "v2"
    assert record["quote"] == "Original fetched source text."
    assert record["record_span_id"].startswith("SPAN-")
    assert record["parent_candidate_id"].startswith("C-")
    assert record["assembly_basis"] == "single_grounded_span"


def test_record_assembler_keeps_two_unkeyed_locators_atomic():
    item = _result(task_record_id="P1")["results"][0]
    first = dict(item["chunk_candidates"][0])
    first["record_key"] = ""
    first["quote"] = "First exact source row."
    first["source_locator"] = {
        "chunk_id": "chunk-1",
        "char_start": 0,
        "char_end": 23,
    }
    second = dict(first)
    second["quote"] = "Second exact source row."
    second["source_locator"] = {
        "chunk_id": "chunk-1",
        "char_start": 24,
        "char_end": 48,
    }
    item["chunk_candidates"] = [first, second]

    records = assemble_grounded_candidates(item)

    assert len(records) == 2
    assert records[0]["quote"] == "First exact source row."
    assert records[1]["quote"] == "Second exact source row."
    assert records[0]["record_span_id"] != records[1]["record_span_id"]
    assert all(row["assembly_basis"] == "single_grounded_span" for row in records)


def test_ledger_primary_key_keeps_identical_text_at_distinct_spans_atomic():
    ledger = EvidenceLedger()
    ledger.initialize(_plan(), "question")
    item = _result(task_record_id="P1")["results"][0]
    first = dict(item["chunk_candidates"][0])
    first["quote"] = "Repeated source text."
    first["source_locator"] = {
        "chunk_id": "chunk-1",
        "char_start": 0,
        "char_end": 21,
    }
    second = dict(first)
    second["source_locator"] = {
        "chunk_id": "chunk-1",
        "char_start": 50,
        "char_end": 71,
    }
    item["chunk_candidates"] = [first, second]

    delta = ledger.ingest(
        "query",
        {"status": "ok", "results": [item]},
        task_record_id="P1",
    )
    records = ledger.snapshot()["task_records"][0]["evidence_records"]

    assert delta["added_evidence_records"] == 2
    assert len(records) == 2
    assert len({row["record_span_id"] for row in records}) == 2
    assert len({row["evidence_record_id"] for row in records}) == 2
    assert {row["source_locator"]["char_start"] for row in records} == {0, 50}


def test_record_assembler_deduplicates_exact_span_without_losing_rwkv_bindings():
    item = _result(task_record_id="P1")["results"][0]
    first = dict(item["chunk_candidates"][0])
    first["task_record_ids"] = ["P1"]
    first["task_record_ids"] = ["P1"]
    first["field_ids"] = ["P1:F1"]
    duplicate_view = dict(first)
    duplicate_view["task_record_ids"] = ["P2"]
    duplicate_view["task_record_ids"] = ["P2"]
    duplicate_view["field_ids"] = ["P2:F1"]
    item["chunk_candidates"] = [first]
    item["candidates"] = [duplicate_view]

    records = assemble_grounded_candidates(item)

    assert len(records) == 1
    assert records[0]["task_record_ids"] == ["P1", "P2"]
    assert records[0]["task_record_ids"] == ["P1", "P2"]
    assert records[0]["field_ids"] == ["P1:F1", "P2:F1"]


def test_candidate_record_is_stored_but_not_counted_as_exact_binding():
    ledger = EvidenceLedger()
    ledger.initialize(_plan(), "question")
    result = _result(task_record_id="P1", record_key="old-v1")
    result["results"][0]["chunk_candidates"][0]["record_match"] = (
        "same_subject_other_record"
    )

    ledger.ingest("query", result, task_record_id="P1")
    point = ledger.snapshot()["task_records"][0]

    assert point["evidence_record_count"] == 1
    assert point["exact_record_count"] == 0
    assert (
        point["evidence_records"][0]["support_state"]
        == "rwkv_evidence_record_candidate"
    )


def test_structured_route_fallback_builds_stable_span_identity():
    ledger = EvidenceLedger()
    ledger.initialize(_plan(), "question")
    result = {
        "status": "ok",
        "results": [
            {
                "title": "Structured release record",
                "url": "https://example.com/releases/latest",
                "content": "Version 2.0 was released on 2026-08-14.",
                "evidence_kind": "structured_record",
                "source_locator": {
                    "chunk_id": "structured-record",
                    "char_start": 0,
                    "char_end": 40,
                },
            }
        ],
    }

    delta = ledger.ingest(
        "latest release",
        result,
        task_record_id="P1",
        strategy="rwkv_selected",
        step=1,
    )
    record = ledger.snapshot()["task_records"][0]["evidence_records"][0]

    assert delta["added_evidence_records"] == 1
    assert record["binding_origin"] == "rwkv_structured_tool_route"
    assert record["source_evidence_kind"] == "structured_record"
    assert record["record_span_id"] == ""
    assert record["evidence_record_id"].startswith("E-")


def test_ledger_does_not_claim_semantic_support_or_answer_completion():
    ledger = EvidenceLedger()
    ledger.initialize(_plan(), "question")
    ledger.ingest("query", _result(), task_record_id="P1")
    snapshot = ledger.snapshot()
    assert "complete" not in snapshot
    assert "status" not in snapshot["task_records"][0]
    assert not hasattr(ledger, "answer_requirement_report")


def test_quote_locator_only_maps_text_back_to_source():
    source = "Alpha   beta\nGamma delta."
    span = locate_grounded_quote_span({"quote": "alpha beta Gamma"}, source)
    assert span is not None
    assert span["grounding_basis"] in {"casefold", "normalized_whitespace"}
    assert source[span["char_start"] : span["char_end"]] == span["text"]


def test_quote_locator_maps_unicode_casefold_expansion_to_raw_offsets():
    source = "Prefix Straße release notes. Suffix"
    span = locate_grounded_quote_span(
        {"quote": "STRASSE RELEASE NOTES."}, source
    )

    assert span is not None
    assert span["grounding_basis"] == "casefold"
    assert span["text"] == "Straße release notes."
    assert source[span["char_start"] : span["char_end"]] == span["text"]


def test_quote_locator_maps_combining_casefold_expansion_to_raw_offsets():
    source = "Prefix İstanbul release. Suffix"
    span = locate_grounded_quote_span(
        {"quote": "i̇stanbul release."}, source
    )

    assert span is not None
    assert span["text"] == "İstanbul release."


def test_quote_locator_rejects_text_not_present_in_source():
    assert locate_grounded_quote_span({"quote": "invented fact"}, "source text") is None


def test_quote_view_removes_only_common_markdown_presentation_syntax():
    source = (
        "### Release notes\n"
        "Use [the official guide](https://example.test/guide) for **CVE-2026-1234**.\n"
        "Run `cargo --version` with `foo_bar`."
    )

    view = source_quote_view(source)

    assert view == (
        "Release notes\n"
        "Use the official guide for CVE-2026-1234.\n"
        "Run cargo --version with foo_bar."
    )


def test_markdown_autolink_remains_visible_and_groundable():
    source = "Visit <https://example.test/guide> for the release guide."
    quote = "Visit https://example.test/guide for the release guide."

    assert source_quote_view(source) == quote
    span = locate_grounded_quote_span({"quote": quote}, source)
    assert span is not None
    assert span["grounding_basis"] == "markdown_visible_exact"
    assert span["text"] == source


def test_quote_locator_maps_exact_visible_markdown_text_to_raw_source_span():
    source = "OpenSSL released [patches](https://example.test) for **CVE-2026-1234**."
    quote = "OpenSSL released patches for CVE-2026-1234."

    span = locate_grounded_quote_span({"quote": quote}, source)

    assert span is not None
    assert span["grounding_basis"] == "markdown_visible_exact"
    assert span["text"] == source


def test_quote_locator_does_not_fuzzily_accept_a_changed_fact():
    source = "Affected areas include Hubei north and west."
    quote = "Affected areas include Hubei northwest."

    assert locate_grounded_quote_span({"quote": quote}, source) is None


def test_quote_locator_rejects_model_quote_that_omits_source_content():
    source = (
        "### Cargo: the Rust build tool and package manager\n"
        "Source-only formatting line.\n"
        "When you install Rustup you also get Cargo.\n"
        "To test that Rust and Cargo are installed, run:\n"
        "`cargo --version`\n"
        "[Read the cargo book](https://doc.rust-lang.org/cargo/)"
    )
    quote = (
        "Cargo: the Rust build tool and package manager\n\n"
        "When you install Rustup you also get Cargo.\n\n"
        "To test that Rust and Cargo are installed, run:\n"
        "`cargo --version`"
    )

    span = locate_grounded_quote_span({"quote": quote}, source)

    assert span is None


def test_quote_locator_maps_unique_unicode_whitespace_variant_to_raw_span():
    source = (
        "Prefix. 原\u3000神\n官\t方\u00a0公 告：鸣潮版本更新内容保持不变。 Suffix."
    )
    quote = "原神官方公告：鸣潮版本更新内容保持不变。"

    span = locate_grounded_quote_span({"quote": quote}, source)

    assert span is not None
    assert span["grounding_basis"] == "unique_without_whitespace"
    assert span["text"] == "原\u3000神\n官\t方\u00a0公 告：鸣潮版本更新内容保持不变。"
    assert source[span["char_start"] : span["char_end"]] == span["text"]


def test_quote_locator_rejects_non_unique_without_whitespace_match():
    source = (
        "守望\n先锋官方版本更新公告内容保持不变。 Middle. "
        "守望\u3000先锋官方版本更新公告内容保持不变。"
    )
    quote = "守望先锋官方版本更新公告内容保持不变。"

    assert locate_grounded_quote_span({"quote": quote}, source) is None


def test_quote_locator_whitespace_fallback_rejects_changed_nonspace_character():
    source = "原\u3000神官方公告：版本更新内容保持不变。"
    quote = "原神官方公告：版本更新日期保持不变。"

    assert locate_grounded_quote_span({"quote": quote}, source) is None


def test_multi_claim_source_without_a_valid_point_stays_unassigned():
    ledger = EvidenceLedger()
    ledger.initialize(_plan(), "mixed historical and current question")

    delta = ledger.ingest(
        "ambiguous result",
        _result(),
        task_record_id="UNKNOWN",
    )

    snapshot = ledger.snapshot()
    assert delta["added_source_bindings"] == 0
    assert delta["added_unassigned_sources"] == 1
    assert [row["retrieval_state"] for row in snapshot["task_records"]] == [
        "not_recorded",
        "not_recorded",
    ]
    assert snapshot["unassigned_source_count"] == 1
    assert snapshot["unassigned_sources"][0]["url"] == "https://example.com/a"


def test_single_claim_does_not_auto_bind_an_unassigned_page():
    ledger = EvidenceLedger()
    ledger.initialize(
        {"records": [{"id": "P1", "task": "one fact"}]},
        "one fact",
    )

    delta = ledger.ingest("one fact", _result())

    assert delta["added_source_bindings"] == 0
    assert delta["added_unassigned_sources"] == 1
    assert ledger.snapshot()["task_records"][0]["retrieval_state"] == "not_recorded"


def test_claim_source_preserves_observable_freshness_metadata():
    ledger = EvidenceLedger()
    ledger.initialize(
        {"records": [{"id": "P1", "task": "current version"}]},
        "current version",
    )
    result = _result("https://example.com/current", task_record_id="P1")
    result["results"][0].update(
        {
            "published": "2026-07-01",
            "updated": "2026-07-20",
            "provider": "example-provider",
            "freshness": {"state": "within_cutoff"},
        }
    )

    ledger.ingest("current version", result, task_record_id="P1")

    source = ledger.snapshot()["task_records"][0]["sources"][0]
    assert source["published"] == "2026-07-01"
    assert source["updated"] == "2026-07-20"
    assert source["provider"] == "example-provider"
    assert source["freshness"] == {"state": "within_cutoff"}


def test_duplicate_source_merges_later_exact_grounded_span():
    ledger = EvidenceLedger()
    ledger.initialize(
        {"records": [{"id": "P1", "task": "current release"}]},
        "current release",
    )
    ledger.ingest("discovery", _result(), task_record_id="P1", step=1)

    improved = _result(task_record_id="P1")
    improved["results"][0]["chunk_candidates"] = [
        {
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "supported": True,
            "source_grounded": True,
            "quote": "Original fetched source text.",
            "source_locator": {"char_start": 0, "char_end": 29},
            "grounding_basis": "exact",
            "task_record_ids": ["P1"],
            "task_record_ids": ["P1"],
        }
    ]
    ledger.ingest("focused follow-up", improved, task_record_id="P1", step=2)

    source = ledger.snapshot()["task_records"][0]["sources"][0]
    assert len(source["grounded_spans"]) == 1
    assert source["grounded_spans"][0]["text"] == "Original fetched source text."
    assert source["grounded_spans"][0]["grounding_basis"] == "exact"


def test_duplicate_evidence_record_preserves_later_object_and_route_observations():
    ledger = EvidenceLedger()
    ledger.initialize(
        {"records": [{"id": "P1", "task": "current release"}]},
        "current release",
    )

    first = _result(task_record_id="P1", record_key="v2")
    first["results"][0].update(
        {
            "object_alignment": {"relation": "unresolved"},
            "retrieval_request": {"request_id": "R-1"},
            "retrieval_bindings": [{"request_id": "R-1", "task_record_id": "P1"}],
        }
    )
    ledger.ingest("discovery", first, task_record_id="P1", step=1)

    exact = _result(task_record_id="P1", record_key="v2")
    exact["results"][0].update(
        {
            "object_alignment": {"relation": "exact"},
            "object_alignments": [
                {"relation": "unresolved"},
                {"relation": "exact"},
            ],
            "retrieval_request": {"request_id": "R-2"},
            "retrieval_bindings": [{"request_id": "R-2", "task_record_id": "P1"}],
        }
    )
    delta = ledger.ingest("focused exact route", exact, task_record_id="P1", step=2)

    point = ledger.snapshot()["task_records"][0]
    record = point["evidence_records"][0]
    source = point["sources"][0]
    assert delta["added_evidence_records"] == 0
    assert delta["updated_evidence_records"] == 1
    assert {row["relation"] for row in record["object_alignments"]} == {
        "unresolved",
        "exact",
    }
    assert {row["request_id"] for row in record["retrieval_requests"]} == {
        "R-1",
        "R-2",
    }
    assert {row["request_id"] for row in source["retrieval_bindings"]} == {
        "R-1",
        "R-2",
    }
    assert {row["relation"] for row in source["object_alignments"]} == {
        "unresolved",
        "exact",
    }
