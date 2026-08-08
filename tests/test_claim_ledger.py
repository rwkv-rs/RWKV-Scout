from agent.claim_ledger import ClaimLedger, locate_grounded_quote_span


def _plan():
    return {
        "source_policy": "primary_preferred",
        "atomic_points": [
            {"id": "P1", "task": "first fact", "objective": "find first fact"},
            {"id": "P2", "task": "second fact", "objective": "find second fact"},
        ],
    }


def _result(url="https://example.com/a"):
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
            }
        ],
    }


def test_ledger_preserves_rwkv_task_points_without_semantic_rewrite():
    ledger = ClaimLedger()
    ledger.initialize(_plan(), "question")
    snapshot = ledger.snapshot()
    assert [row["claim_id"] for row in snapshot["claims"]] == ["P1", "P2"]
    assert [row["task"] for row in snapshot["claims"]] == ["first fact", "second fact"]
    assert snapshot["advisory_only"] is True


def test_ledger_records_retrieved_source_for_selected_point():
    ledger = ClaimLedger()
    ledger.initialize(_plan(), "question")
    delta = ledger.ingest("query", _result(), task_point_id="P2", strategy="rwkv_selected", step=1)
    snapshot = ledger.snapshot()
    by_id = {row["claim_id"]: row for row in snapshot["claims"]}
    assert delta["touched_claim_ids"] == ["P2"]
    assert by_id["P1"]["retrieval_state"] == "not_retrieved"
    assert by_id["P2"]["retrieval_state"] == "retrieved"
    assert by_id["P2"]["sources"][0]["url"] == "https://example.com/a"


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
