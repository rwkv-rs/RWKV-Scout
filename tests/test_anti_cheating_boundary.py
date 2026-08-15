from __future__ import annotations

import ast
from pathlib import Path

from agent.orchestrator import runtime_metadata_only
from scripts.run_json_acceptance import _runtime_metadata_for_case


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = [ROOT / "api.py"] + [
    path
    for folder in ("agent", "tools", "runtime", "app")
    for path in (ROOT / folder).rglob("*.py")
]


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _query_variable_names(node: ast.AST) -> set[str]:
    names = {
        child.id.casefold()
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    }
    return {
        name
        for name in names
        if name in {"query", "user_query", "original_goal", "prompt", "user_prompt"}
        or name in {"query_folded", "query_lower", "query_text"}
        or name.endswith("_query")
    }


def test_production_does_not_import_offline_evaluation_or_fixture_modules():
    forbidden = {
        "utils.acceptance_metrics",
        "utils.adversarial_eval",
        "utils.harness_fixtures",
    }
    violations: list[str] = []
    for path in PRODUCTION_PATHS:
        for node in ast.walk(_tree(path)):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module in forbidden:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{module}")
    assert violations == []


def test_production_has_no_plain_task_literal_condition_on_user_query():
    """A product/topic phrase must not activate a hidden runtime answer path."""

    violations: list[str] = []
    for path in PRODUCTION_PATHS:
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Compare) or not _query_variable_names(node):
                continue
            if not any(
                isinstance(operator, (ast.In, ast.NotIn, ast.Eq, ast.NotEq))
                for operator in node.ops
            ):
                continue
            literals = [
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and child.value.strip()
            ]
            if literals:
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}:{ast.unparse(node)}"
                )
    assert violations == []


def test_runtime_metadata_strips_reference_and_external_plan_fields():
    raw = {
        "max_tool_steps": 50,
        "retrieval_branch_width": 4,
        "source_policy": "official_required",
        "required_domains": ["example.org"],
        "gold": {"answer": "secret"},
        "reference_answer": "secret",
        "acceptance_case_id": "Q01",
        "acceptance_criteria": ["must equal secret"],
        "task_plan": {"goal": "externally supplied"},
        "evidence_ledger": {"task_records": [{"answer": "secret"}]},
    }
    projected = runtime_metadata_only(raw)
    assert projected == {"max_tool_steps": 50}


def test_suite_runner_never_projects_gold_or_reference_into_runtime():
    case = {
        "query": "What changed today?",
        "gold": {"answer": "hidden"},
        "reference_answer": "hidden",
        "acceptance_case_id": "Q01",
        "task_plan": {"goal": "hidden"},
        "generic_web_search_only": True,
        "max_tool_steps": 12,
    }
    projected = _runtime_metadata_for_case(case, max_tool_steps_override=50)
    assert projected == {"max_tool_steps": 50}


def test_public_api_and_frontend_do_not_expose_evaluation_controls():
    sources = [
        ROOT / "api.py",
        ROOT / "app" / "models.py",
        ROOT / "frontend" / "src" / "App.jsx",
        ROOT / "frontend" / "src" / "api.js",
    ]
    forbidden = (
        "acceptance_case_id",
        "harness-fixtures",
        "getAcceptanceMetrics",
    )
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{marker} leaked into {path.relative_to(ROOT)}"
