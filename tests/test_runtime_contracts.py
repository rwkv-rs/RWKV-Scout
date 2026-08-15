import pytest

from agent import runtime_contracts


def test_runtime_contract_registry_uses_stable_semantic_names():
    contracts = {
        name: value
        for name, value in vars(runtime_contracts).items()
        if name.endswith("_CONTRACT") and isinstance(value, str)
    }

    assert contracts
    assert len(set(contracts.values())) == len(contracts)
    assert all(
        value.startswith("rwkv.ecra.runtime.") and ".v" not in value
        for value in contracts.values()
    )


def test_runtime_boundary_rejects_legacy_schema_discriminator():
    with pytest.raises(ValueError, match="legacy schema_version"):
        runtime_contracts.require_runtime_contract(
            {
                "schema_version": "task-query-fanout.v1",
                "contract": runtime_contracts.RETRIEVAL_QUERY_PLAN_CONTRACT,
            },
            runtime_contracts.RETRIEVAL_QUERY_PLAN_CONTRACT,
        )


def test_model_boundary_may_neither_omit_nor_change_expected_contract():
    with pytest.raises(ValueError, match="expected runtime contract"):
        runtime_contracts.require_runtime_contract(
            {},
            runtime_contracts.EVIDENCE_REVIEW_CONTRACT,
        )
    with pytest.raises(ValueError, match="expected runtime contract"):
        runtime_contracts.require_runtime_contract(
            {"contract": runtime_contracts.TASK_PLAN_CONTRACT},
            runtime_contracts.EVIDENCE_REVIEW_CONTRACT,
        )
