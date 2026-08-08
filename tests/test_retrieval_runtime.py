import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.orchestrator import Orchestrator, runtime_metadata_only
from agent.planner import Planner
from agent.state import AgentState
from agent.unified_research import run_unified_research_loop
from tools.registry import ToolRegistry
from app.services.workspace_files import read_task_report
from utils.task_events import get_task_events


class FakePlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.observations = []
        self.rebuilt_sessions = []

    def plan_next_action(self, *_args):
        return self.decisions.pop(0)

    def observe_tool_result(self, result):
        self.observations.append(result)

    def rebuild_session_after_review(self, user_query, env_context, review, phase):
        self.rebuilt_sessions.append(
            {
                "user_query": user_query,
                "env_context": env_context,
                "review": review,
                "phase": phase,
            }
        )

    def rebuild_session(self, user_query, env_context, observation, phase):
        self.observations.append(observation)
        self.rebuilt_sessions.append(
            {
                "user_query": user_query,
                "env_context": env_context,
                "observation": observation,
                "phase": phase,
            }
        )


class FakeOwner:
    def __init__(self, decisions, reviews=None):
        self.state = AgentState(task_id="TEST", user_query="question")
        self.state.run_metadata = {}
        self.planner = FakePlanner(decisions)
        self._retrieval_ledger = self.state.retrieval.progress
        self._model_protocol_failure = False
        self._calculation_results = []
        self._time_results = []
        self._arithmetic_results = []
        self.finished = []
        self.reviews = list(reviews or [{"decision": "finish"}])
        self.review_calls = []

    def _agentic_tool_context(self):
        return {"agent_state": self.state, "task_id": self.state.task_id}

    def _record_retrieval_progress(self, result, **_kwargs):
        return result

    def _cross_validate_research(self, *args, **kwargs):
        self.review_calls.append({"args": args, "kwargs": kwargs})
        return self.reviews.pop(0)

    def _complete_model_tool_loop(self, query, action, rounds, step, *, termination_reason):
        self.finished.append(
            {
                "query": query,
                "action": action,
                "rounds": list(rounds),
                "step": step,
                "termination_reason": termination_reason,
            }
        )
        return "rwkv final"


def _plan():
    return {
        "atomic_points": [
            {"id": "P1", "task": "question", "objective": "answer question"}
        ]
    }


def test_loop_executes_exact_rwkv_tool_and_query(monkeypatch):
    calls = []
    owner = FakeOwner(
        [
            {
                "action": "web_search",
                "args": {"query": "exact RWKV query"},
                "task_point_id": "P1",
                "raw_model_output": "model call",
            },
            {"action": "finish_task", "args": {}, "task_point_id": "P1"},
        ]
    )
    monkeypatch.setattr(ToolRegistry, "has", classmethod(lambda cls, name: name == "web_search"))
    monkeypatch.setattr(
        ToolRegistry,
        "metadata",
        classmethod(lambda cls, name: {"retrieval_role": "discovery"}),
    )

    def execute(_cls, action, args, context, phase=None):
        calls.append((action, dict(args), phase))
        return json.dumps(
            {
                "status": "ok",
                "results": [
                    {
                        "title": "source",
                        "url": "https://example.com",
                        "content": "retrieved body",
                    }
                ],
            }
        )

    monkeypatch.setattr(ToolRegistry, "execute", classmethod(execute))
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 5)
    assert answer == "rwkv final"
    assert calls == [("web_search", {"query": "exact RWKV query"}, None)]
    assert owner.finished[0]["termination_reason"] == "rwkv_cross_validation_finish"


def test_finish_request_is_decided_by_rwkv_review_not_claim_rules():
    owner = FakeOwner([{"action": "finish_task", "args": {}, "task_point_id": "P1"}])
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 5)
    assert answer == "rwkv final"
    assert owner.finished[0]["rounds"] == []
    assert owner.finished[0]["termination_reason"] == "rwkv_cross_validation_finish"


def test_exact_duplicate_is_reviewed_by_rwkv_without_controller_recovery_route(monkeypatch):
    calls = []
    decision = {
        "action": "web_search",
        "args": {"query": "same query"},
        "task_point_id": "P1",
    }
    owner = FakeOwner([decision, decision])
    monkeypatch.setattr(ToolRegistry, "has", classmethod(lambda cls, name: True))
    monkeypatch.setattr(
        ToolRegistry,
        "metadata",
        classmethod(lambda cls, name: {"retrieval_role": "discovery"}),
    )

    def execute(_cls, action, args, context, phase=None):
        calls.append((action, dict(args)))
        return json.dumps({"status": "ok", "results": []})

    monkeypatch.setattr(ToolRegistry, "execute", classmethod(execute))
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 5)
    assert answer == "rwkv final"
    assert calls == [("web_search", {"query": "same query"})]
    assert len(owner.review_calls) == 1
    assert owner.state.retrieval.routing_snapshot()["frozen_path_count"] == 1
    assert owner.state.retrieval.replan_count == 0
    assert owner.planner.rebuilt_sessions == []
    assert owner.finished[0]["termination_reason"] == "rwkv_cross_validation_finish_after_duplicate"
    assert not hasattr(owner.planner, "begin_replan")


def test_duplicate_review_replan_runs_only_the_next_rwkv_selected_query(monkeypatch):
    first = {
        "action": "web_search",
        "args": {"query": "same query"},
        "task_point_id": "P1",
    }
    follow_up = {
        "action": "web_search",
        "args": {"query": "RWKV selected missing evidence query"},
        "task_point_id": "P1",
    }
    owner = FakeOwner(
        [first, first, follow_up, {"action": "finish_task", "args": {}}],
        reviews=[
            {
                "decision": "replan",
                "missing_points": ["P1"],
                "conflicts": [],
                "next_focus": "an independent official confirmation",
            },
            {"decision": "finish", "missing_points": [], "conflicts": []},
        ],
    )
    calls = []
    monkeypatch.setattr(ToolRegistry, "has", classmethod(lambda cls, name: True))
    monkeypatch.setattr(
        ToolRegistry,
        "metadata",
        classmethod(lambda cls, name: {"retrieval_role": "discovery"}),
    )

    def execute(_cls, action, args, context, phase=None):
        calls.append((action, dict(args)))
        return json.dumps({"status": "ok", "results": []})

    monkeypatch.setattr(ToolRegistry, "execute", classmethod(execute))
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 6)

    assert answer == "rwkv final"
    assert calls == [
        ("web_search", {"query": "same query"}),
        ("web_search", {"query": "RWKV selected missing evidence query"}),
    ]
    assert len(owner.review_calls) == 2
    assert len(owner.planner.rebuilt_sessions) == 1
    assert owner.planner.rebuilt_sessions[0]["review"]["decision"] == "replan"
    assert owner.state.retrieval.replan_count == 1


def test_repeated_frozen_path_reuses_pending_rwkv_replan_without_reviewer_loop(monkeypatch):
    repeated = {
        "action": "web_search",
        "args": {"query": "same frozen query"},
        "task_point_id": "P1",
    }
    alternative = {
        "action": "web_search",
        "args": {"query": "RWKV independently selected alternative"},
        "task_point_id": "P1",
    }
    owner = FakeOwner(
        [repeated, repeated, repeated, alternative, {"action": "finish_task", "args": {}}],
        reviews=[
            {
                "decision": "replan",
                "missing_points": ["P1"],
                "conflicts": [],
                "next_focus": "another source route",
            },
            {"decision": "finish", "missing_points": [], "conflicts": []},
        ],
    )
    calls = []
    monkeypatch.setattr(ToolRegistry, "has", classmethod(lambda cls, name: True))
    monkeypatch.setattr(
        ToolRegistry,
        "metadata",
        classmethod(lambda cls, name: {"retrieval_role": "discovery"}),
    )

    def execute(_cls, action, args, context, phase=None):
        calls.append((action, dict(args)))
        results = (
            [{"title": "new source", "url": "https://example.com/new", "content": "new evidence"}]
            if args.get("query") == "RWKV independently selected alternative"
            else []
        )
        return json.dumps({"status": "ok", "results": results})

    monkeypatch.setattr(ToolRegistry, "execute", classmethod(execute))
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 7)

    assert answer == "rwkv final"
    assert calls == [
        ("web_search", {"query": "same frozen query"}),
        ("web_search", {"query": "RWKV independently selected alternative"}),
    ]
    assert len(owner.review_calls) == 2
    assert len(owner.planner.rebuilt_sessions) == 2
    assert owner.state.retrieval.replan_count == 2


def test_rwkv_cross_validation_can_rebuild_planner_and_run_another_round(monkeypatch):
    owner = FakeOwner(
        [
            {"action": "finish_task", "args": {}, "task_point_id": "P1"},
            {
                "action": "web_search",
                "args": {"query": "model selected follow-up"},
                "task_point_id": "P1",
            },
            {"action": "finish_task", "args": {}, "task_point_id": "P1"},
        ],
        reviews=[
            {
                "decision": "replan",
                "missing_points": ["P1"],
                "conflicts": [],
                "next_focus": "missing official confirmation",
            },
            {"decision": "finish", "missing_points": [], "conflicts": []},
        ],
    )
    calls = []
    monkeypatch.setattr(ToolRegistry, "has", classmethod(lambda cls, name: True))
    monkeypatch.setattr(
        ToolRegistry,
        "metadata",
        classmethod(lambda cls, name: {"retrieval_role": "discovery"}),
    )

    def execute(_cls, action, args, context, phase=None):
        calls.append((action, dict(args), phase))
        return json.dumps(
            {
                "status": "ok",
                "results": [
                    {
                        "title": "official",
                        "url": "https://example.com/official",
                        "content": "original confirmation",
                    }
                ],
            }
        )

    monkeypatch.setattr(ToolRegistry, "execute", classmethod(execute))
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 6)

    assert answer == "rwkv final"
    assert calls == [("web_search", {"query": "model selected follow-up"}, None)]
    assert len(owner.planner.rebuilt_sessions) == 1
    assert owner.planner.rebuilt_sessions[0]["review"]["decision"] == "replan"
    assert owner.state.retrieval.replan_count == 1


def test_planner_context_keeps_original_source_spans_after_feedback_changes():
    state = AgentState(task_id="PERSISTENT_EVIDENCE", user_query="question")
    state.retrieval.claims.initialize(_plan(), "question")
    state.retrieval.record_query(
        "first query",
        {
            "status": "ok",
            "results": [
                {
                    "title": "source",
                    "url": "https://example.com/source",
                    "source_chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "index": 0,
                            "text": "ORIGINAL_PERSISTENT_SPAN exact source text",
                        }
                    ],
                    "model_extracted_facts": "GENERATED_PARAPHRASE_MUST_NOT_ENTER",
                }
            ],
        },
        task_point_id="P1",
        step=1,
    )
    state.last_feedback = json.dumps(
        {"status": "no_new_evidence", "error_class": "exact_duplicate_request"}
    )

    context = state.to_retrieval_context()
    assert "ORIGINAL_PERSISTENT_SPAN exact source text" in context
    assert "GENERATED_PARAPHRASE_MUST_NOT_ENTER" not in context

    planner = Planner()
    planner._task_plan = _plan()
    planner._latest_routing_observation = '{"status":"latest"}'
    decision_body = planner._build_isolated_decision_body(
        "question",
        context,
        "DISCOVERY",
    )
    assert "ORIGINAL_PERSISTENT_SPAN exact source text" in decision_body


def test_runtime_has_no_answer_quality_status_classifier():
    orchestrator = Orchestrator()
    assert not hasattr(orchestrator, "_final_answer_status")
    assert not hasattr(orchestrator, "_final_answer_error_type")


def test_runtime_metadata_cannot_receive_gold_or_external_plan():
    projected = runtime_metadata_only(
        {
            "max_tool_steps": 50,
            "gold": {"answer": "secret"},
            "reference_answer": "secret",
            "task_plan": {"goal": "injected"},
        }
    )
    assert projected == {"max_tool_steps": 50}


def test_rwkv_answer_is_identical_in_return_event_and_report():
    output = "  Assistant: <think>model-owned text</think>\nRepeated.\nRepeated.  \n"

    class ExactRWKV:
        provider = "local_13b"

        def text_completion(self, _prompt, max_tokens=0):
            assert max_tokens > 0
            return SimpleNamespace(content=output)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.dict("config.DATA_PIPELINE", {"output_directory": str(root)}, clear=False):
            orchestrator = Orchestrator()
            orchestrator.llm = ExactRWKV()
            orchestrator.state.task_id = "EXACT_OUTPUT_CHAIN"
            orchestrator.state.user_query = "question"
            orchestrator.state.task_output_dir = str(root / "EXACT_OUTPUT_CHAIN")
            Path(orchestrator.state.task_output_dir).mkdir(parents=True, exist_ok=True)

            returned = orchestrator._complete_model_tool_loop(
                "question",
                "finish_task",
                [],
                1,
            )
            final = [
                event
                for event in get_task_events("EXACT_OUTPUT_CHAIN")
                if event.get("type") == "final"
            ][-1]
            report = read_task_report(root, "EXACT_OUTPUT_CHAIN")

    assert returned == output
    assert final["content"] == output
    assert "status" not in final
    assert report[0]["answer"] == output
