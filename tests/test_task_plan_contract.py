import json
from unittest.mock import Mock

from agent.planner import Planner
from agent.task_plan_contract import (
    compact_task_plan,
    field_ids,
    normalize_task_plan,
    record_fields,
    record_id,
    task_records,
)


def test_historical_v2_is_rewritten_once_to_the_canonical_runtime_shape():
    plan = normalize_task_plan(
        {
            "schema_version": "task_plan.v2",
            "goal": "Compare a historical release date with the current theme",
            "source_policy": "official_required",
            "required_domains": ["invented.example"],
            "atomic_points": [
                {
                    "id": "legacy-a",
                    "question": "What was the release date?",
                    "fields": ["release date"],
                    "time_scope": "historical",
                },
                {
                    "id": "legacy-b",
                    "question": "What is the current theme?",
                    "fields": ["version", "theme"],
                    "time_scope": "current",
                },
            ],
        }
    )

    assert set(plan) == {"contract", "goal", "records"}
    assert plan["contract"] == "rwkv.ecra.runtime.task-plan"
    assert [record_id(row) for row in plan["records"]] == ["P1", "P2"]
    assert record_fields(plan["records"][1]) == ["version", "theme"]
    assert field_ids(plan["records"][1]) == ["P2:F1", "P2:F2"]
    assert "atomic_points" not in plan
    assert "source_policy" not in plan


def test_transport_ids_are_controller_owned_and_deterministic():
    payload = {
        "contract": "rwkv.ecra.runtime.task-plan",
        "goal": "current release",
        "records": [
            {
                "record_id": "MODEL-INVENTED",
                "question": "current version and date",
                "fields": [
                    {"field_id": "WRONG", "name": "version"},
                    {"field_id": "ALSO-WRONG", "name": "date"},
                ],
            }
        ],
    }

    first = normalize_task_plan(payload)
    second = normalize_task_plan(first)

    assert first == second
    assert first["records"][0]["record_id"] == "P1"
    assert first["records"][0]["fields"] == [
        {"field_id": "P1:F1", "name": "version"},
        {"field_id": "P1:F2", "name": "date"},
    ]


def test_admitted_runtime_ids_survive_reordering_and_recovery_projection():
    admitted = normalize_task_plan(
        {
            "goal": "compare",
            "records": [
                {"question": "alpha", "fields": ["version"]},
                {"question": "beta", "fields": ["date"]},
            ],
        }
    )
    recovered = {
        **admitted,
        "records": list(reversed(admitted["records"])),
    }

    assert [row["record_id"] for row in task_records(recovered)] == ["P2", "P1"]
    assert task_records(recovered)[0]["fields"] == [
        {"field_id": "P2:F1", "name": "date"}
    ]


def test_equal_wording_records_are_not_merged_at_the_contract_boundary():
    plan = normalize_task_plan(
        {
            "goal": "one release record",
            "records": [
                {"question": "current release", "fields": ["version"]},
                {"question": " current   release ", "fields": ["date"]},
            ],
        }
    )

    assert len(plan["records"]) == 2
    assert record_fields(plan["records"][0]) == ["version"]
    assert record_fields(plan["records"][1]) == ["date"]
    assert field_ids(plan["records"][0]) == ["P1:F1"]
    assert field_ids(plan["records"][1]) == ["P2:F1"]


def test_historical_record_count_is_never_capped_or_silently_emptied():
    plan = normalize_task_plan(
        {
            "goal": "preserve every historical record",
            "atomic_points": [
                {"id": f"legacy-{index}", "task": f"fact {index}"}
                for index in range(1, 9)
            ],
        }
    )

    assert [record["record_id"] for record in plan["records"]] == [
        f"P{index}" for index in range(1, 9)
    ]


def test_v1_task_objective_and_evidence_needed_survive_migration():
    plan = normalize_task_plan(
        {
            "goal": "historical goal",
            "atomic_points": [
                {
                    "task": "find the release",
                    "objective": "return its exact date",
                    "evidence_needed": ["release identifier", "release date"],
                }
            ],
        }
    )

    record = plan["records"][0]
    assert record["question"] == "find the release — return its exact date"
    assert record_fields(record) == ["release identifier", "release date"]


def test_compact_plan_never_reintroduces_legacy_or_policy_fields():
    compact = compact_task_plan(
        {
            "contract": "rwkv.ecra.runtime.task-plan",
            "goal": "answer",
            "source_policy": "official_required",
            "records": [
                {
                    "record_id": "P1",
                    "question": "requested fact",
                    "fields": [{"field_id": "P1:F1", "name": "date"}],
                    "acceptance_criteria": ["controller gate"],
                }
            ],
        }
    )

    assert set(compact) == {"contract", "goal", "records"}
    assert set(compact["records"][0]) == {
        "record_id",
        "question",
        "subject",
        "relation",
        "fields",
        "time_scope",
        "set_semantics",
        "premise_requires_verification",
    }


def test_task_records_returns_canonical_records_at_a_historical_boundary():
    records = task_records(
        {
            "goal": "historical date",
            "requested_fields": ["date"],
            "atomic_points": [{"id": "old", "task": "historical date"}],
        }
    )

    assert records[0]["record_id"] == "P1"
    assert records[0]["fields"] == [{"field_id": "P1:F1", "name": "date"}]


def test_invalid_model_plan_falls_back_to_exact_goal():
    planner = Planner()
    planner.llm = Mock()
    planner.llm.text_completion.return_value = Mock(content="not json")

    result = planner.create_task_plan("Exact user question", "current UTC date")

    assert result["plan_fallback"] is True
    assert result["contract"] == "rwkv.ecra.runtime.task-plan"
    assert result["goal"] == "Exact user question"
    assert result["records"][0]["question"] == "Exact user question"
    assert result["records"][0]["fields"] == []
    assert planner.llm.text_completion.call_count == 2


def test_planner_requests_canonical_contract_and_returns_no_legacy_shape():
    planner = Planner()
    prompts = []

    class FixedRWKV:
        provider = "local_13b"

        def text_completion(self, prompt, max_tokens=0, stop=None):
            del max_tokens, stop
            prompts.append(prompt)
            return Mock(
                content=json.dumps(
                    {
                        "contract": "rwkv.ecra.runtime.task-plan",
                        "goal": "answer",
                        "records": [
                            {
                                "record_id": "P1",
                                "question": "requested fact",
                                "fields": [
                                    {"field_id": "P1:F1", "name": "date"}
                                ],
                                "time_scope": "historical",
                            }
                        ],
                    }
                )
            )

    planner.llm = FixedRWKV()
    result = planner.create_task_plan("requested fact", "current UTC date")

    assert len(prompts) == 1
    assert "rwkv.ecra.runtime.task-plan" in prompts[0]
    assert "atomic_points" not in prompts[0]
    assert result["contract"] == "rwkv.ecra.runtime.task-plan"
    assert "atomic_points" not in result
