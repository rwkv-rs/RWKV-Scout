import json
from unittest.mock import Mock

from agent.planner import Planner
from agent.task_plan_contract import compact_task_plan, normalize_task_plan, task_points


def test_v2_plan_keeps_only_factual_records_and_fields():
    plan = normalize_task_plan(
        {
            "schema_version": "task_plan.v2",
            "goal": "Compare a historical release date with the current theme",
            "source_policy": "official_required",
            "required_domains": ["invented.example"],
            "atomic_points": [
                {
                    "id": "P1",
                    "question": "What was the release date?",
                    "fields": ["release date"],
                    "time_scope": "historical",
                },
                {
                    "id": "P2",
                    "question": "What is the current theme?",
                    "fields": ["version", "theme"],
                    "time_scope": "current",
                },
            ],
        }
    )

    assert plan["schema_version"] == "task_plan.v2"
    assert [row["id"] for row in plan["atomic_points"]] == ["P1", "P2"]
    assert plan["atomic_points"][1]["fields"] == ["version", "theme"]
    assert "source_policy" not in plan
    assert "required_domains" not in plan


def test_legacy_trace_is_normalized_once_at_the_contract_boundary():
    points = task_points(
        {
            "requested_fields": ["date"],
            "atomic_points": [
                {"id": "P1", "task": "historical date", "objective": "find it"}
            ],
        }
    )

    assert points == [
        {
            "id": "P1",
            "question": "historical date",
            "subject": "",
            "relation": "",
            "fields": ["date"],
            "time_scope": "unspecified",
            "set_semantics": "single",
            "premise_requires_verification": False,
            "source_ids": ["P1"],
        }
    ]


def test_compact_plan_does_not_reintroduce_workflow_or_policy_fields():
    compact = compact_task_plan(
        {
            "goal": "answer",
            "source_policy": "official_required",
            "atomic_points": [
                {
                    "id": "P1",
                    "question": "requested fact",
                    "fields": ["date"],
                    "time_scope": "historical",
                    "acceptance_criteria": ["controller gate"],
                }
            ],
        }
    )

    assert set(compact) == {"schema_version", "goal", "atomic_points"}
    assert set(compact["atomic_points"][0]) == {
        "id",
        "question",
        "subject",
        "relation",
        "fields",
        "time_scope",
        "set_semantics",
        "premise_requires_verification",
    }


def test_record_identity_and_set_semantics_survive_the_shared_contract():
    plan = normalize_task_plan(
        {
            "goal": "latest version fields",
            "atomic_points": [
                {
                    "id": "P1",
                    "question": "current version, title and date",
                    "subject": "Example Game",
                    "relation": "current release",
                    "fields": ["version", "title", "date"],
                    "time_scope": "current",
                    "set_semantics": "single",
                    "premise_requires_verification": True,
                }
            ],
        }
    )

    point = plan["atomic_points"][0]
    assert point["subject"] == "Example Game"
    assert point["relation"] == "current release"
    assert point["set_semantics"] == "single"
    assert point["premise_requires_verification"] is True


def test_model_records_key_normalizes_to_runtime_atomic_points():
    plan = normalize_task_plan(
        {
            "schema_version": "task_plan.v3",
            "goal": "current release record",
            "records": [
                {
                    "id": "P1",
                    "question": "current release identifier and date",
                    "subject": "Example Product",
                    "relation": "current release",
                    "fields": ["identifier", "date"],
                    "time_scope": "current",
                    "set_semantics": "single",
                }
            ],
        }
    )

    assert len(plan["atomic_points"]) == 1
    assert plan["atomic_points"][0]["fields"] == ["identifier", "date"]


def test_invalid_model_plan_falls_back_to_exact_user_goal_without_answer_logic():
    planner = Planner()
    planner.llm = Mock()
    planner.llm.text_completion.return_value = Mock(content="not json")

    result = planner.create_task_plan("Exact user question", "current UTC date")

    assert result["plan_fallback"] is True
    assert result["goal"] == "Exact user question"
    assert result["atomic_points"][0]["question"] == "Exact user question"
    assert result["atomic_points"][0]["fields"] == []
    assert planner.llm.text_completion.call_count == 2


def test_planner_request_contains_only_the_record_factual_contract():
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
                        "schema_version": "task_plan.v2",
                        "goal": "answer",
                        "atomic_points": [
                            {
                                "id": "P1",
                                "question": "requested fact",
                                "fields": ["date"],
                                "time_scope": "historical",
                            }
                        ],
                    }
                )
            )

    planner.llm = FixedRWKV()
    planner.create_task_plan("requested fact", "current UTC date")

    assert len(prompts) == 1
    assert "task_plan.v3" in prompts[0]
    assert "task_plan.v1" not in prompts[0]
    assert "source_policy" not in prompts[0]
    assert "acceptance_criteria" not in prompts[0]
