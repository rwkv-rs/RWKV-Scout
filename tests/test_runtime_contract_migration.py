from scripts.migrate_runtime_contracts import migrate_document


def test_migration_rewrites_plan_and_linked_record_and_field_references():
    source = {
        "task_plan": {
            "schema_version": "task_plan.v2",
            "goal": "compare releases",
            "atomic_points": [
                {
                    "id": "legacy-alpha",
                    "task": "find Alpha release",
                    "fields": [
                        {"field_id": "old-version", "name": "version"},
                        {"field_id": "old-date", "name": "date"},
                    ],
                }
            ],
        },
        "evidence": {
            "claim_id": "legacy-alpha",
            "claim_ids": ["legacy-alpha"],
            "field_keys": ["old-version", "date"],
        },
    }

    migrated, stats = migrate_document(source)

    assert migrated["task_plan"] == {
        "contract": "rwkv.ecra.runtime.task-plan",
        "goal": "compare releases",
        "records": [
            {
                "record_id": "P1",
                "question": "find Alpha release",
                "subject": "",
                "relation": "",
                "fields": [
                    {"field_id": "P1:F1", "name": "version"},
                    {"field_id": "P1:F2", "name": "date"},
                ],
                "time_scope": "unspecified",
                "set_semantics": "single",
                "premise_requires_verification": False,
            }
        ],
    }
    assert migrated["evidence"] == {
        "task_record_id": "P1",
        "task_record_ids": ["P1"],
        "field_ids": ["P1:F1", "P1:F2"],
    }
    assert stats == {
        "task_plans": 1,
        "runtime_contracts": 0,
        "event_names": 0,
        "record_references": 2,
        "field_references": 2,
        "key_collisions": 0,
    }


def test_migration_preserves_raw_model_strings_as_audit_material():
    raw = '{"schema_version":"task_plan.v1","goal":"old","atomic_points":[]}'

    migrated, stats = migrate_document({"raw_model_output": raw})

    assert migrated["raw_model_output"] == raw
    assert stats["task_plans"] == 0


def test_migration_replaces_structured_runtime_names_but_not_audit_strings():
    legacy_prompt = '{"schema_version":"rwkv-cross-validation.v4"}'
    migrated, stats = migrate_document(
        {
            "event": {
                "type": "cross_validation",
                "content": {
                    "schema_version": "rwkv-cross-validation.v4",
                    "decision": "finish",
                },
                "claim_ledger": {"claims": []},
            },
            "prompt": legacy_prompt,
        }
    )

    assert migrated["event"] == {
        "type": "evidence_review",
        "content": {
            "contract": "rwkv.ecra.runtime.evidence-review",
            "decision": "finish",
        },
        "evidence_ledger": {"task_records": []},
    }
    assert migrated["prompt"] == legacy_prompt
    assert stats["runtime_contracts"] == 2
    assert stats["event_names"] == 1


def test_migration_canonical_key_wins_alias_collision_in_both_orders():
    first, first_stats = migrate_document(
        {
            "claim_ids": ["legacy"],
            "task_record_ids": ["P1"],
            "field_keys": ["old"],
            "field_ids": ["P1:F1"],
        }
    )
    second, second_stats = migrate_document(
        {
            "task_record_ids": ["P1"],
            "claim_ids": ["legacy"],
            "field_ids": ["P1:F1"],
            "field_keys": ["old"],
        }
    )

    assert first == second == {
        "task_record_ids": ["P1"],
        "field_ids": ["P1:F1"],
    }
    assert first_stats["key_collisions"] == 2
    assert second_stats["key_collisions"] == 2


def test_migration_preserves_already_canonical_runtime_ids():
    source = {
        "contract": "rwkv.ecra.runtime.task-plan",
        "goal": "recover",
        "records": [
            {
                "record_id": "P7",
                "question": "recover date",
                "subject": "",
                "relation": "",
                "fields": [{"field_id": "P7:F3", "name": "date"}],
                "time_scope": "historical",
                "set_semantics": "single",
                "premise_requires_verification": False,
            }
        ],
    }

    migrated, _ = migrate_document(source)

    assert migrated["records"][0]["record_id"] == "P7"
    assert migrated["records"][0]["fields"][0]["field_id"] == "P7:F3"
