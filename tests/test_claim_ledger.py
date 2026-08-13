from agent.claim_ledger import ClaimLedger, locate_grounded_quote_span


def _plan():
    return {
        "schema_version": "task_plan.v2",
        "goal": "answer two facts",
        "atomic_points": [
            {"id": "P1", "question": "first fact", "fields": [], "time_scope": "unspecified"},
            {"id": "P2", "question": "second fact", "fields": [], "time_scope": "unspecified"},
        ],
    }


def _result(url="https://example.com/a", *, claim_id="", record_key=""):
    candidates = []
    if claim_id:
        candidates.append(
            {
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "supported": True,
                "source_grounded": True,
                "task_record_ids": [claim_id],
                "claim_ids": [claim_id],
                "field_keys": ["date"],
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


def test_ledger_preserves_rwkv_task_points_without_semantic_rewrite():
    ledger = ClaimLedger()
    ledger.initialize(_plan(), "question")
    snapshot = ledger.snapshot()
    assert [row["claim_id"] for row in snapshot["claims"]] == ["P1", "P2"]
    assert [row["question"] for row in snapshot["claims"]] == ["first fact", "second fact"]
    assert snapshot["schema_version"] == "evidence-ledger.v1"
    assert snapshot["advisory_only"] is True


def test_route_scope_does_not_bind_a_whole_page_to_selected_point():
    ledger = ClaimLedger()
    ledger.initialize(_plan(), "question")
    delta = ledger.ingest("query", _result(), task_point_id="P2", strategy="rwkv_selected", step=1)
    snapshot = ledger.snapshot()
    by_id = {row["claim_id"]: row for row in snapshot["claims"]}
    assert delta["touched_claim_ids"] == []
    assert by_id["P1"]["retrieval_state"] == "not_recorded"
    assert by_id["P2"]["retrieval_state"] == "not_recorded"
    assert by_id["P2"]["attempt_count"] == 1
    assert snapshot["unassigned_source_count"] == 1


def test_rwkv_grounded_span_binds_one_evidence_record():
    ledger = ClaimLedger()
    ledger.initialize(_plan(), "question")

    delta = ledger.ingest(
        "query",
        _result(claim_id="P2", record_key="v2"),
        task_point_id="P2",
    )
    point = ledger.snapshot()["claims"][1]

    assert delta["added_evidence_records"] == 1
    assert point["retrieval_state"] == "evidence_recorded"
    assert point["evidence_record_count"] == 1
    record = point["evidence_records"][0]
    assert record["task_record_id"] == "P2"
    assert record["record_key"] == "v2"
    assert record["quote"] == "Original fetched source text."


def test_candidate_record_is_stored_but_not_counted_as_exact_binding():
    ledger = ClaimLedger()
    ledger.initialize(_plan(), "question")
    result = _result(claim_id="P1", record_key="old-v1")
    result["results"][0]["chunk_candidates"][0]["record_match"] = (
        "same_subject_other_record"
    )

    ledger.ingest("query", result, task_point_id="P1")
    point = ledger.snapshot()["claims"][0]

    assert point["evidence_record_count"] == 1
    assert point["exact_record_count"] == 0
    assert point["candidate_record_count"] == 1
    assert (
        point["evidence_records"][0]["support_state"]
        == "rwkv_candidate_record"
    )


def test_ledger_does_not_claim_semantic_support_or_answer_completion():
    ledger = ClaimLedger()
    ledger.initialize(_plan(), "question")
    ledger.ingest("query", _result(), task_point_id="P1")
    snapshot = ledger.snapshot()
    assert "complete" not in snapshot
    assert "status" not in snapshot["claims"][0]
    assert not hasattr(ledger, "answer_requirement_report")


def test_quote_locator_only_maps_text_back_to_source():
    source = "Alpha   beta\nGamma delta."
    span = locate_grounded_quote_span({"quote": "alpha beta Gamma"}, source)
    assert span is not None
    assert span["grounding_basis"] in {"casefold", "normalized_whitespace"}
    assert source[span["char_start"] : span["char_end"]] == span["text"]


def test_quote_locator_rejects_text_not_present_in_source():
    assert locate_grounded_quote_span({"quote": "invented fact"}, "source text") is None


def test_quote_locator_maps_markdown_normalized_lines_to_original_span():
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

    assert span is not None
    assert span["grounding_basis"] == "ordered_source_segments"
    assert span["grounded_segment_count"] == 4
    assert span["text"].startswith("Cargo: the Rust build tool")
    assert span["text"].endswith("`cargo --version`")


def test_multi_claim_source_without_a_valid_point_stays_unassigned():
    ledger = ClaimLedger()
    ledger.initialize(_plan(), "mixed historical and current question")

    delta = ledger.ingest(
        "ambiguous result",
        _result(),
        task_point_id="UNKNOWN",
    )

    snapshot = ledger.snapshot()
    assert delta["added_source_bindings"] == 0
    assert delta["added_unassigned_sources"] == 1
    assert [row["retrieval_state"] for row in snapshot["claims"]] == [
        "not_recorded",
        "not_recorded",
    ]
    assert snapshot["unassigned_source_count"] == 1
    assert snapshot["unassigned_sources"][0]["url"] == "https://example.com/a"


def test_single_claim_does_not_auto_bind_an_unassigned_page():
    ledger = ClaimLedger()
    ledger.initialize(
        {"atomic_points": [{"id": "P1", "task": "one fact"}]},
        "one fact",
    )

    delta = ledger.ingest("one fact", _result())

    assert delta["added_source_bindings"] == 0
    assert delta["added_unassigned_sources"] == 1
    assert ledger.snapshot()["claims"][0]["retrieval_state"] == "not_recorded"


def test_claim_source_preserves_observable_freshness_metadata():
    ledger = ClaimLedger()
    ledger.initialize(
        {"atomic_points": [{"id": "P1", "task": "current version"}]},
        "current version",
    )
    result = _result("https://example.com/current", claim_id="P1")
    result["results"][0].update(
        {
            "published": "2026-07-01",
            "updated": "2026-07-20",
            "provider": "example-provider",
            "freshness": {"state": "within_cutoff"},
        }
    )

    ledger.ingest("current version", result, task_point_id="P1")

    source = ledger.snapshot()["claims"][0]["sources"][0]
    assert source["published"] == "2026-07-01"
    assert source["updated"] == "2026-07-20"
    assert source["provider"] == "example-provider"
    assert source["freshness"] == {"state": "within_cutoff"}


def test_duplicate_source_merges_later_exact_grounded_span():
    ledger = ClaimLedger()
    ledger.initialize(
        {"atomic_points": [{"id": "P1", "task": "current release"}]},
        "current release",
    )
    ledger.ingest("discovery", _result(), task_point_id="P1", step=1)

    improved = _result(claim_id="P1")
    improved["results"][0]["chunk_candidates"] = [
        {
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "supported": True,
            "source_grounded": True,
            "quote": "Original fetched source text.",
            "source_locator": {"char_start": 0, "char_end": 29},
            "grounding_basis": "exact",
            "claim_ids": ["P1"],
            "task_record_ids": ["P1"],
        }
    ]
    ledger.ingest("focused follow-up", improved, task_point_id="P1", step=2)

    source = ledger.snapshot()["claims"][0]["sources"][0]
    assert len(source["grounded_spans"]) == 1
    assert source["grounded_spans"][0]["text"] == "Original fetched source text."
    assert source["grounded_spans"][0]["grounding_basis"] == "exact"


def test_duplicate_evidence_record_preserves_later_object_and_route_observations():
    ledger = ClaimLedger()
    ledger.initialize(
        {"atomic_points": [{"id": "P1", "task": "current release"}]},
        "current release",
    )

    first = _result(claim_id="P1", record_key="v2")
    first["results"][0].update(
        {
            "object_alignment": {"relation": "unresolved"},
            "retrieval_request": {"request_id": "R-1"},
            "retrieval_bindings": [{"request_id": "R-1", "task_record_id": "P1"}],
        }
    )
    ledger.ingest("discovery", first, task_point_id="P1", step=1)

    exact = _result(claim_id="P1", record_key="v2")
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
    delta = ledger.ingest("focused exact route", exact, task_point_id="P1", step=2)

    point = ledger.snapshot()["claims"][0]
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
