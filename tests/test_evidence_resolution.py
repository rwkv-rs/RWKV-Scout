import json
from types import SimpleNamespace

import pytest

from agent.evidence_resolution import (
    attach_evidence_resolution_to_context,
    build_evidence_object_groups,
    build_evidence_record_set,
    build_evidence_resolution_prompt,
    evidence_resolution_completion_token_budget,
    parse_evidence_resolution_output,
    evidence_resolution_signature,
    resolve_evidence,
)
from agent.task_plan_contract import normalize_task_plan
from utils.rwkv_prompt import JSON_CALL_STOP_SUFFIXES


TASK_PLAN = normalize_task_plan(
    {
        "goal": "current version and release date",
        "records": [
            {
                "question": "current version and release date",
                "fields": ["version", "date"],
                "time_scope": "current",
            },
            {"question": "maintainer name", "fields": ["maintainer"]},
        ],
    }
)


def _selected_record(
    ref_id="S1",
    evidence_record_id="E-release-44",
    text="Version 4.4 was released on 2026-08-01.",
):
    return {
        "ref_id": ref_id,
        "evidence_record_id": evidence_record_id,
        "title": "Official release",
        "url": "https://example.test/releases/4.4",
        "context_role": "candidate_evidence_record",
        "task_record_ids": ["P1"],
        "record_metadata": {
            "record_key": "4.4",
            "field_ids": ["P1:F1", "P1:F2", "invented-field"],
            "source_object": {
                "source_object_id": "repo:owner/project",
                "source_object_type": "release",
                "source_record_id": "4.4",
            },
        },
        "packed_chunks": [{"chunk_id": "release-4.4", "text": text}],
        "evidence_text": text,
    }


class FakeRWKV:
    provider = "local_test"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def text_completion(self, prompt, max_tokens=0, stop=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "stop": stop})
        return SimpleNamespace(content=self.outputs.pop(0))


def _valid_decisions():
    object_group_id = build_evidence_record_set([_selected_record()])[0][
        "object_group_id"
    ]
    return [
        {
            "task_record_id": "P1",
            "status": "resolved",
            "selected_object_group_id": object_group_id,
            "conflicting_object_group_ids": [],
            "selected_evidence_record_ids": ["E-release-44"],
            "conflicting_evidence_record_ids": [],
            "field_evidence_record_ids": {
                "P1:F1": ["E-release-44"],
                "P1:F2": ["E-release-44"],
            },
            "missing_field_ids": [],
            "conflict_field_ids": [],
            "needs_more_evidence": False,
        },
        {
            "task_record_id": "P2",
            "status": "missing",
            "selected_object_group_id": "",
            "conflicting_object_group_ids": [],
            "selected_evidence_record_ids": [],
            "conflicting_evidence_record_ids": [],
            "field_evidence_record_ids": {},
            "missing_field_ids": ["P2:F1"],
            "conflict_field_ids": [],
            "needs_more_evidence": True,
        },
    ]


def test_evidence_records_use_internal_evidence_ids_and_report_full_coverage():
    evidence_records = build_evidence_record_set([_selected_record()])

    assert evidence_records[0]["evidence_record_id"] == "E-release-44"
    assert evidence_records[0]["packet_alias"] == "S1"
    assert evidence_records[0]["coverage_complete"] is True
    assert evidence_records[0]["object_group_id"].startswith("O-")
    assert evidence_records[0]["object_group_identity"] == {
        "identity_method": "source_object_record",
        "source_object_id": "repo:owner/project",
        "source_object_type": "release",
        "source_record_id": "4.4",
    }
    assert evidence_records[0]["declared_task_record_ids"] == ["P1"]
    assert evidence_records[0]["exact_spans"] == [
        {
            "span_id": "release-4.4",
            "text": "Version 4.4 was released on 2026-08-01.",
        }
    ]
    assert "invented-field" not in json.dumps(evidence_records, ensure_ascii=False)


def test_record_key_does_not_merge_different_subjects_without_stable_object_id():
    first = _selected_record(evidence_record_id="E-game-a")
    second = _selected_record(ref_id="S2", evidence_record_id="E-game-b")
    for selected, subject in ((first, "Game A"), (second, "Game B")):
        selected["record_metadata"]["subject_key"] = subject
        selected["record_metadata"]["source_object"].pop("source_object_id")
    evidence_records = build_evidence_record_set([first, second])

    assert evidence_records[0]["record_key"] == evidence_records[1]["record_key"]
    assert evidence_records[0]["object_group_id"] != evidence_records[1]["object_group_id"]
    assert len(build_evidence_object_groups(evidence_records)) == 2


def test_resolution_parser_enforces_complete_field_closure():
    evidence_records = build_evidence_record_set([_selected_record()])
    result = parse_evidence_resolution_output(
        json.dumps(
            {
                "contract": "rwkv.ecra.runtime.evidence-resolution",
                "decisions": _valid_decisions(),
            }
        ),
        query=TASK_PLAN["goal"],
        task_plan=TASK_PLAN,
        evidence_records=evidence_records,
    )

    assert result["coverage_complete"] is True
    assert result["decisions"] == _valid_decisions()
    assert result["packet_aliases"] == {"E-release-44": "S1"}


def test_unknown_ids_are_protocol_errors_not_silently_dropped():
    decisions = _valid_decisions()
    decisions[0]["selected_evidence_record_ids"] = ["S1"]
    with pytest.raises(ValueError, match="unknown controller-owned ID"):
        parse_evidence_resolution_output(
            json.dumps({
                "contract": "rwkv.ecra.runtime.evidence-resolution",
                "decisions": decisions,
            }),
            query=TASK_PLAN["goal"],
            task_plan=TASK_PLAN,
            evidence_records=build_evidence_record_set([_selected_record()]),
        )


def test_resolved_is_forbidden_when_any_span_was_not_viewed():
    evidence_records = build_evidence_record_set(
        [_selected_record(text="x" * 500)], max_chars_per_record=160
    )
    assert evidence_records[0]["coverage_complete"] is False
    assert evidence_records[0]["unviewed_span_ids"] == ["release-4.4"]

    with pytest.raises(ValueError, match="unviewed evidence spans"):
        parse_evidence_resolution_output(
            json.dumps({
                "contract": "rwkv.ecra.runtime.evidence-resolution",
                "decisions": _valid_decisions(),
            }),
            query=TASK_PLAN["goal"],
            task_plan=TASK_PLAN,
            evidence_records=evidence_records,
        )


def test_conflict_requires_two_internal_evidence_records():
    evidence_records = build_evidence_record_set([_selected_record()])
    decisions = _valid_decisions()
    decisions[0] = {
        "task_record_id": "P1",
        "status": "conflict",
        "selected_object_group_id": evidence_records[0]["object_group_id"],
        "conflicting_object_group_ids": [evidence_records[0]["object_group_id"]],
        "selected_evidence_record_ids": ["E-release-44"],
        "conflicting_evidence_record_ids": ["E-release-44"],
        "field_evidence_record_ids": {"P1:F2": ["E-release-44"]},
        "missing_field_ids": [],
        "conflict_field_ids": ["P1:F1"],
        "needs_more_evidence": True,
    }
    with pytest.raises(ValueError, match="at least two"):
        parse_evidence_resolution_output(
            json.dumps({
                "contract": "rwkv.ecra.runtime.evidence-resolution",
                "decisions": decisions,
            }),
            query=TASK_PLAN["goal"],
            task_plan=TASK_PLAN,
            evidence_records=evidence_records,
        )


def test_missing_status_cannot_smuggle_conflict_state():
    evidence_records = build_evidence_record_set([_selected_record()])
    decisions = _valid_decisions()
    decisions[0] = {
        "task_record_id": "P1",
        "status": "missing",
        "selected_object_group_id": evidence_records[0]["object_group_id"],
        "conflicting_object_group_ids": [],
        "selected_evidence_record_ids": ["E-release-44"],
        "conflicting_evidence_record_ids": ["E-release-44"],
        "field_evidence_record_ids": {},
        "missing_field_ids": ["P1:F2"],
        "conflict_field_ids": ["P1:F1"],
        "needs_more_evidence": True,
    }

    with pytest.raises(ValueError, match="missing status cannot carry conflict"):
        parse_evidence_resolution_output(
            json.dumps({
                "contract": "rwkv.ecra.runtime.evidence-resolution",
                "decisions": decisions,
            }),
            query=TASK_PLAN["goal"],
            task_plan=TASK_PLAN,
            evidence_records=evidence_records,
        )


def test_resolved_fields_cannot_stitch_two_object_groups():
    first = _selected_record()
    second = _selected_record(
        ref_id="S2",
        evidence_record_id="E-release-45",
        text="Version 4.5 was released on 2026-08-08.",
    )
    second["record_metadata"]["record_key"] = "4.5"
    second["record_metadata"]["source_object"]["source_record_id"] = "4.5"
    evidence_records = build_evidence_record_set([first, second])
    groups = build_evidence_object_groups(evidence_records)
    assert len(groups) == 2

    decisions = _valid_decisions()
    decisions[0].update(
        {
            "selected_object_group_id": evidence_records[0]["object_group_id"],
            "selected_evidence_record_ids": ["E-release-44", "E-release-45"],
            "field_evidence_record_ids": {
                "P1:F1": ["E-release-44"],
                "P1:F2": ["E-release-45"],
            },
        }
    )

    with pytest.raises(ValueError, match="selected object group"):
        parse_evidence_resolution_output(
            json.dumps(
                {
                    "contract": "rwkv.ecra.runtime.evidence-resolution",
                    "decisions": decisions,
                }
            ),
            query=TASK_PLAN["goal"],
            task_plan=TASK_PLAN,
            evidence_records=evidence_records,
        )


def test_multi_object_conflict_requires_and_preserves_competing_groups():
    first = _selected_record()
    second = _selected_record(
        ref_id="S2",
        evidence_record_id="E-release-45",
        text="Version 4.5 was released on 2026-08-08.",
    )
    second["record_metadata"]["record_key"] = "4.5"
    second["record_metadata"]["source_object"]["source_record_id"] = "4.5"
    evidence_records = build_evidence_record_set([first, second])
    first_group = evidence_records[0]["object_group_id"]
    second_group = evidence_records[1]["object_group_id"]
    decisions = _valid_decisions()
    decisions[0] = {
        "task_record_id": "P1",
        "status": "conflict",
        "selected_object_group_id": first_group,
        "conflicting_object_group_ids": [first_group, second_group],
        "selected_evidence_record_ids": ["E-release-44", "E-release-45"],
        "conflicting_evidence_record_ids": ["E-release-44", "E-release-45"],
        "field_evidence_record_ids": {"P1:F2": ["E-release-44"]},
        "missing_field_ids": [],
        "conflict_field_ids": ["P1:F1"],
        "needs_more_evidence": True,
    }

    result = parse_evidence_resolution_output(
        json.dumps(
            {
                "contract": "rwkv.ecra.runtime.evidence-resolution",
                "decisions": decisions,
            }
        ),
        query=TASK_PLAN["goal"],
        task_plan=TASK_PLAN,
        evidence_records=evidence_records,
    )

    assert result["decisions"][0]["status"] == "conflict"
    assert result["decisions"][0]["conflicting_object_group_ids"] == [
        first_group,
        second_group,
    ]
    assert len(result["object_groups"]) == 2


def test_resolution_prompt_uses_current_ids_without_concrete_call_example():
    evidence_records = build_evidence_record_set([_selected_record()])
    prompt, _ = build_evidence_resolution_prompt(
        TASK_PLAN["goal"], TASK_PLAN, evidence_records
    )

    assert "OBJECT GROUPS" in prompt
    assert evidence_records[0]["object_group_id"] in prompt
    assert '"selected_object_group_id"' in prompt
    assert '{"task_record_id":"P1","status"' not in prompt
    assert prompt.endswith("Assistant: ```json\n")


def test_resolution_completion_budget_scales_with_record_and_field_count():
    small = normalize_task_plan(
        {"goal": "one fact", "records": [{"question": "one", "fields": ["value"]}]}
    )
    large = normalize_task_plan(
        {
            "goal": "many facts",
            "records": [
                {
                    "question": f"record {index}",
                    "fields": [f"field {field}" for field in range(8)],
                }
                for index in range(4)
            ],
        }
    )

    small_budget = evidence_resolution_completion_token_budget("one fact", small)
    large_budget = evidence_resolution_completion_token_budget("many facts", large)

    assert 384 <= small_budget < large_budget
    assert large_budget <= 4096


def test_local_resolution_uses_record_scoped_object_then_field_calls(monkeypatch):
    evidence_records = build_evidence_record_set([_selected_record()])
    object_group_id = evidence_records[0]["object_group_id"]
    model = FakeRWKV(
        [
            json.dumps(
                {
                    "contract": "rwkv.ecra.runtime.evidence-resolution",
                    "task_record_id": "P1",
                    "selection": "selected",
                    "object_group_ids": [object_group_id],
                }
            ),
            json.dumps(
                {
                    "contract": "rwkv.ecra.runtime.evidence-resolution",
                    "task_record_id": "P1",
                    "fields": [
                        {
                            "field_id": "P1:F1",
                            "state": "supported",
                            "evidence_record_ids": ["E-release-44"],
                        },
                        {
                            "field_id": "P1:F2",
                            "state": "supported",
                            "evidence_record_ids": ["E-release-44"],
                        },
                    ],
                }
            ),
            json.dumps(
                {
                    "contract": "rwkv.ecra.runtime.evidence-resolution",
                    "task_record_id": "P2",
                    "selection": "missing",
                    "object_group_ids": [],
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "agent.evidence_resolution.get_llm_context_length", lambda: 32768
    )

    result = resolve_evidence(
        TASK_PLAN["goal"],
        TASK_PLAN,
        evidence_records,
        model,
    )

    assert result["status"] == "ok"
    assert result["attempts"] == 3
    assert result["decisions"] == _valid_decisions()
    assert [call["stage"] for call in result["stage_calls"]] == [
        "object_selection",
        "field_binding",
        "object_selection",
    ]
    assert len(model.calls) == 3
    assert all(call["stop"] == JSON_CALL_STOP_SUFFIXES for call in model.calls)
    assert "EVIDENCE RECORDS" in model.calls[0]["prompt"]
    assert "SELECTED EVIDENCE RECORDS" in model.calls[1]["prompt"]
    assert all(
        call["prompt"].endswith("Assistant: ```json\n") for call in model.calls
    )


def test_resolution_admission_reserves_the_full_requested_completion(monkeypatch):
    model = FakeRWKV([])
    monkeypatch.setattr(
        "agent.evidence_resolution.get_token_count", lambda _value: 1000
    )
    monkeypatch.setattr(
        "agent.evidence_resolution.get_llm_context_length", lambda: 1500
    )

    result = resolve_evidence(
        TASK_PLAN["goal"],
        TASK_PLAN,
        build_evidence_record_set([_selected_record()]),
        model,
    )

    assert result["status"] == "unavailable"
    assert result["attempts"] == 0
    assert "exceeds model context" in result["error"]
    assert model.calls == []


def test_malformed_resolution_degrades_without_mutating_evidence(monkeypatch):
    model = FakeRWKV(["not json", "still not json"])
    monkeypatch.setattr(
        "agent.evidence_resolution.get_llm_context_length", lambda: 32768
    )
    result = resolve_evidence(
        TASK_PLAN["goal"],
        TASK_PLAN,
        build_evidence_record_set([_selected_record()]),
        model,
    )
    assert result["status"] == "unavailable"
    assert result["attempts"] == 3
    assert [call["task_record_id"] for call in result["stage_calls"]] == ["P1", "P2"]

    context = {
        "text": "ORIGINAL EVIDENCE",
        "selected_evidence": [_selected_record()],
        "citation_refs": [{"ref_id": "S1"}],
        "context_stats": {"context_tokens": 2},
    }
    attached = attach_evidence_resolution_to_context(context, result)
    assert attached["text"] == "ORIGINAL EVIDENCE"
    assert attached["evidence_text"] == "ORIGINAL EVIDENCE"


def test_partial_resolution_is_advisory_and_keeps_writer_evidence_immutable(monkeypatch):
    selected = _selected_record()
    evidence_records = build_evidence_record_set([selected])
    object_group_id = evidence_records[0]["object_group_id"]
    model = FakeRWKV(
        [
            json.dumps(
                {
                    "contract": "rwkv.ecra.runtime.evidence-resolution",
                    "task_record_id": "P1",
                    "selection": "selected",
                    "object_group_ids": [object_group_id],
                }
            ),
            "invalid field binding",
            "still invalid",
            json.dumps(
                {
                    "contract": "rwkv.ecra.runtime.evidence-resolution",
                    "task_record_id": "P2",
                    "selection": "missing",
                    "object_group_ids": [],
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "agent.evidence_resolution.get_llm_context_length", lambda: 32768
    )

    resolution = resolve_evidence(TASK_PLAN["goal"], TASK_PLAN, evidence_records, model)
    assert resolution["status"] == "partial"
    assert resolution["decisions"][0]["resolver_fallback"] == "field_binding_unavailable"

    output = attach_evidence_resolution_to_context(
        {
            "text": "FLAT FALLBACK",
            "selected_evidence": [selected],
            "citation_refs": [{"ref_id": "S1", "url": selected["url"]}],
            "context_stats": {},
        },
        resolution,
        TASK_PLAN,
    )

    assert output["text"] == "FLAT FALLBACK"
    assert output["evidence_text"] == "FLAT FALLBACK"
    assert output["selected_evidence"] == [selected]
    assert output["citation_refs"] == [{"ref_id": "S1", "url": selected["url"]}]
    assert output["context_stats"]["evidence_resolution_advisory_only"] is True
    assert output["context_stats"]["evidence_lane_preserved"] is True


def test_resolution_control_lane_is_separate_from_evidence_lane():
    context = {
        "text": "RETRIEVED SOURCES:\n[S1]\n<release-4.4>\nVersion 4.4",
        "evidence_text": "RETRIEVED SOURCES:\n[S1]\nVersion 4.4",
        "selected_evidence": [_selected_record()],
        "citation_refs": [{"ref_id": "S1"}],
        "context_stats": {"context_tokens": 10},
    }
    resolution = {
        "contract": "rwkv.ecra.runtime.evidence-resolution",
        "status": "ok",
        "evidence_record_count": 1,
        "coverage_complete": True,
        "decisions": _valid_decisions(),
    }

    output = attach_evidence_resolution_to_context(context, resolution)

    assert output["text"] == context["text"]
    assert output["evidence_text"] == context["evidence_text"]
    assert output["evidence_resolution_view"].startswith("EVIDENCE RESOLUTION CONTROL MAP")
    assert "E-release-44" in output["evidence_resolution_view"]
    assert output["selected_evidence"] == context["selected_evidence"]
    assert output["citation_refs"] == context["citation_refs"]
    assert output["context_stats"]["evidence_resolution_advisory_only"] is True
    assert output["context_stats"]["evidence_lane_preserved"] is True
    assert len(output["context_stats"]["evidence_lane_digest"]) == 16


def test_resolution_status_cannot_change_the_evidence_lane_digest():
    context = {
        "text": "EXACT EVIDENCE LANE",
        "evidence_text": "EXACT EVIDENCE LANE",
        "selected_evidence": [_selected_record()],
        "citation_refs": [{"ref_id": "S1"}],
        "context_stats": {},
    }
    resolved = attach_evidence_resolution_to_context(
        context,
        {
            "contract": "rwkv.ecra.runtime.evidence-resolution",
            "status": "ok",
            "evidence_record_count": 1,
            "decisions": _valid_decisions(),
        },
    )
    missing = attach_evidence_resolution_to_context(
        context,
        {
            "contract": "rwkv.ecra.runtime.evidence-resolution",
            "status": "partial",
            "evidence_record_count": 1,
            "decisions": [],
        },
    )

    for key in ("text", "evidence_text", "selected_evidence", "citation_refs"):
        assert resolved[key] == context[key]
        assert missing[key] == context[key]
    assert (
        resolved["context_stats"]["evidence_lane_digest"]
        == missing["context_stats"]["evidence_lane_digest"]
    )


def test_valid_resolution_cannot_replace_or_filter_writer_evidence():
    selected = _selected_record()
    evidence_records = build_evidence_record_set([selected])
    resolution = parse_evidence_resolution_output(
        json.dumps(
            {
                "contract": "rwkv.ecra.runtime.evidence-resolution",
                "decisions": _valid_decisions(),
            }
        ),
        query=TASK_PLAN["goal"],
        task_plan=TASK_PLAN,
        evidence_records=evidence_records,
    )
    context = {
        "text": "LEGACY SOURCE ORDER",
        "evidence_text": "LEGACY SOURCE ORDER",
        "selected_evidence": [selected],
        "citation_refs": [{"ref_id": "S1", "url": selected["url"]}],
        "calculation_results": [],
        "context_stats": {"context_tokens": 3},
    }

    output = attach_evidence_resolution_to_context(context, resolution, TASK_PLAN)

    assert output["text"] == context["text"]
    assert output["evidence_text"] == context["evidence_text"]
    assert output["selected_evidence"] == [selected]
    assert output["citation_refs"] == [{"ref_id": "S1", "url": selected["url"]}]
    assert output["context_stats"]["evidence_lane_preserved"] is True
    assert output.get("usable_evidence_count") == context.get("usable_evidence_count")
    assert output.get("chunk_count") == context.get("chunk_count")
    assert output["context_tokens"] > 0


def test_conflict_resolution_cannot_reorder_or_delete_competing_sources():
    first = _selected_record()
    second = _selected_record(
        ref_id="S2",
        evidence_record_id="E-release-45",
        text="Version 4.5 was released on 2026-08-08.",
    )
    second["record_metadata"]["record_key"] = "4.5"
    second["record_metadata"]["source_object"]["source_record_id"] = "4.5"
    evidence_records = build_evidence_record_set([first, second])
    first_group = evidence_records[0]["object_group_id"]
    second_group = evidence_records[1]["object_group_id"]
    decisions = _valid_decisions()
    decisions[0] = {
        "task_record_id": "P1",
        "status": "conflict",
        "selected_object_group_id": first_group,
        "conflicting_object_group_ids": [first_group, second_group],
        "selected_evidence_record_ids": ["E-release-44", "E-release-45"],
        "conflicting_evidence_record_ids": ["E-release-44", "E-release-45"],
        "field_evidence_record_ids": {"P1:F2": ["E-release-44"]},
        "missing_field_ids": [],
        "conflict_field_ids": ["P1:F1"],
        "needs_more_evidence": True,
    }
    resolution = parse_evidence_resolution_output(
        json.dumps(
            {
                "contract": "rwkv.ecra.runtime.evidence-resolution",
                "decisions": decisions,
            }
        ),
        query=TASK_PLAN["goal"],
        task_plan=TASK_PLAN,
        evidence_records=evidence_records,
    )
    context = {
        "text": "BASE",
        "selected_evidence": [first, second],
        "citation_refs": [{"ref_id": "S1"}, {"ref_id": "S2"}],
        "calculation_results": [],
        "context_stats": {},
    }

    output = attach_evidence_resolution_to_context(context, resolution, TASK_PLAN)

    assert output["text"] == context["text"]
    assert output["selected_evidence"] == [first, second]
    assert output["citation_refs"] == [{"ref_id": "S1"}, {"ref_id": "S2"}]
    assert output["context_stats"]["evidence_lane_preserved"] is True
    assert first_group in output["evidence_resolution_view"]
    assert second_group in output["evidence_resolution_view"]


def test_no_evidence_records_skip_model_and_signature_tracks_exact_span():
    model = FakeRWKV([])
    result = resolve_evidence(TASK_PLAN["goal"], TASK_PLAN, [], model)
    assert result["status"] == "no_evidence_records"
    assert model.calls == []

    first = build_evidence_record_set([_selected_record(text="Version 4.4")])
    second = build_evidence_record_set([_selected_record(text="Version 4.5")])
    assert evidence_resolution_signature(TASK_PLAN["goal"], TASK_PLAN, first) != evidence_resolution_signature(
        TASK_PLAN["goal"], TASK_PLAN, second
    )
