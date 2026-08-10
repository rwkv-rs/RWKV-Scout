import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.orchestrator import Orchestrator, planner_environment_context, runtime_metadata_only
from agent.planner import Planner
from agent.state import AgentState
from agent.unified_research import run_unified_research_loop
from tools.registry import ToolRegistry
from app.services.workspace_files import read_task_report
from utils.chunker import get_token_count
from utils.task_events import get_task_events


class FakePlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.observations = []
        self.rebuilt_sessions = []
        self.replan_progress = []

    def plan_next_action(self, *_args):
        return self.decisions.pop(0)

    def observe_tool_result(self, result):
        self.observations.append(result)

    def rebuild_session_after_review(
        self,
        user_query,
        env_context,
        review,
        phase,
        routing_observation=None,
    ):
        self.rebuilt_sessions.append(
            {
                "user_query": user_query,
                "env_context": env_context,
                "review": review,
                "phase": phase,
                "routing_observation": routing_observation,
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

    def mark_replan_progress(self, task_point_id):
        self.replan_progress.append(task_point_id)


def test_planner_environment_context_exposes_observable_utc_time():
    context = json.loads(
        planner_environment_context(
            datetime(2026, 8, 9, 10, 11, 12, tzinfo=timezone.utc)
        )
    )

    assert context == {
        "current_utc_datetime": "2026-08-09T10:11:12+00:00",
        "current_utc_date": "2026-08-09",
    }

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

    def _record_retrieval_progress(self, result, **kwargs):
        self._retrieval_ledger.record(
            kwargs.get("query", ""),
            result,
            step=kwargs.get("step", 0),
            task_point_id=kwargs.get("task_point_id", ""),
            action=kwargs.get("action", ""),
            phase=kwargs.get("phase", ""),
        )
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
        calls.append((action, dict(args), dict(context), phase))
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
    assert calls == [
        (
            "web_search",
            {"query": "exact RWKV query"},
            {
                "agent_state": owner.state,
                "task_id": owner.state.task_id,
                "task_point_id": "P1",
            },
            None,
        )
    ]
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
    assert owner.planner.rebuilt_sessions[0]["routing_observation"]["frozen_path"][
        "query"
    ] == "same query"
    assert owner.state.retrieval.replan_count == 1


def test_new_url_without_claim_binding_does_not_clear_pending_replan(monkeypatch):
    first = {
        "action": "web_search",
        "args": {"query": "same query"},
        "task_point_id": "P1",
    }
    alternative = {
        "action": "web_search",
        "args": {"query": "different route with an unsupported page"},
        "task_point_id": "P1",
    }
    owner = FakeOwner(
        [first, first, alternative, {"action": "finish_task", "args": {}}],
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
    monkeypatch.setattr(ToolRegistry, "has", classmethod(lambda cls, name: True))
    monkeypatch.setattr(
        ToolRegistry,
        "metadata",
        classmethod(lambda cls, name: {"retrieval_role": "discovery"}),
    )

    def execute(_cls, action, args, context, phase=None):
        results = (
            [
                {
                    "title": "unsupported page",
                    "url": "https://example.com/unsupported",
                    "content": "unrelated body",
                }
            ]
            if args.get("query") == alternative["args"]["query"]
            else []
        )
        return json.dumps({"status": "ok", "results": results})

    monkeypatch.setattr(ToolRegistry, "execute", classmethod(execute))
    monkeypatch.setattr(
        owner.state.retrieval,
        "record_query",
        lambda *_args, **_kwargs: {
            "new_source_count": 1,
            "claim_delta": {"added_source_bindings": 0},
        },
    )

    answer = run_unified_research_loop(owner, "question", {}, _plan(), 6)

    assert answer == "rwkv final"
    assert owner.planner.replan_progress == []


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
    assert any(
        observation.get("status") == "no_new_evidence"
        for observation in owner.planner.observations
    )


def test_pending_replan_stall_is_bounded_without_rebuild_loop(monkeypatch):
    repeated = {
        "action": "web_search",
        "args": {"query": "same frozen query"},
        "task_point_id": "P1",
    }
    owner = FakeOwner(
        [repeated, repeated, repeated, repeated, repeated, repeated, repeated, repeated],
        reviews=[
            {
                "decision": "replan",
                "missing_points": ["P1"],
                "conflicts": [],
                "next_focus": "another source route",
            }
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
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 20)

    assert answer == "rwkv final"
    assert calls == [("web_search", {"query": "same frozen query"})]
    assert len(owner.review_calls) == 1
    assert len(owner.planner.rebuilt_sessions) == 2
    assert owner.state.retrieval.replan_count == 2
    assert owner.finished[0]["termination_reason"] == "rwkv_replan_stalled"


def test_equivalent_query_is_frozen_only_within_the_same_task_point(monkeypatch):
    first = {
        "action": "web_search",
        "args": {"query": "alpha beta gamma delta epsilon official"},
        "task_point_id": "P1",
    }
    equivalent = {
        "action": "web_search",
        "args": {"query": "official epsilon delta gamma beta alpha"},
        "task_point_id": "P1",
    }
    owner = FakeOwner([first, equivalent], reviews=[{"decision": "finish"}])
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
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 5)

    assert answer == "rwkv final"
    assert calls == [
        ("web_search", {"query": "alpha beta gamma delta epsilon official"})
    ]
    assert owner.finished[0]["termination_reason"] == "rwkv_cross_validation_finish_after_duplicate"


def test_exact_retrieval_request_is_not_reexecuted_for_another_task_point(monkeypatch):
    first = {
        "action": "web_search",
        "args": {"query": "one exact shared query"},
        "task_point_id": "P1",
    }
    repeated_for_another_point = {
        "action": "web_search",
        "args": {"query": "one exact shared query"},
        "task_point_id": "P2",
    }
    owner = FakeOwner(
        [first, repeated_for_another_point],
        reviews=[{"decision": "finish"}],
    )
    calls = []
    plan = {
        "atomic_points": [
            {"id": "P1", "task": "first fact"},
            {"id": "P2", "task": "second fact"},
        ]
    }
    monkeypatch.setattr(ToolRegistry, "has", classmethod(lambda cls, name: True))
    monkeypatch.setattr(
        ToolRegistry,
        "metadata",
        classmethod(lambda cls, name: {"retrieval_role": "discovery"}),
    )

    def execute(_cls, action, args, context, phase=None):
        del context, phase
        calls.append((action, dict(args)))
        return json.dumps({"status": "ok", "results": []})

    monkeypatch.setattr(ToolRegistry, "execute", classmethod(execute))
    answer = run_unified_research_loop(owner, "question", {}, plan, 5)

    assert answer == "rwkv final"
    assert calls == [("web_search", {"query": "one exact shared query"})]
    assert owner.finished[0]["termination_reason"] == (
        "rwkv_cross_validation_finish_after_duplicate"
    )


def test_duplicate_cross_validation_protocol_errors_are_bounded(monkeypatch):
    repeated = {
        "action": "web_search",
        "args": {"query": "same query"},
        "task_point_id": "P1",
    }
    owner = FakeOwner(
        [repeated, repeated, repeated],
        reviews=[
            {"decision": "protocol_error", "message": "invalid review one"},
            {"decision": "protocol_error", "message": "invalid review two"},
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
        del context, phase
        calls.append((action, dict(args)))
        return json.dumps({"status": "ok", "results": []})

    monkeypatch.setattr(ToolRegistry, "execute", classmethod(execute))
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 10)

    assert answer == "rwkv final"
    assert calls == [("web_search", {"query": "same query"})]
    assert len(owner.review_calls) == 2
    assert owner.finished[0]["termination_reason"] == (
        "rwkv_cross_validation_protocol_stalled"
    )


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
    assert owner.planner.replan_progress == ["P1"]
    assert owner.state.retrieval.replan_count == 1


def test_unrelated_new_source_does_not_clear_model_requested_missing_point(monkeypatch):
    owner = FakeOwner(
        [
            {"action": "finish_task", "args": {}, "task_point_id": "P1"},
            {
                "action": "web_search",
                "args": {"query": "another P1 source"},
                "task_point_id": "P1",
            },
            {
                "action": "web_search",
                "args": {"query": "another P1 source"},
                "task_point_id": "P1",
            },
            {"action": "finish_task", "args": {}, "task_point_id": "P1"},
        ],
        reviews=[
            {
                "decision": "replan",
                "missing_points": ["P2"],
                "conflicts": [],
                "next_focus": "P2 evidence",
            },
            {"decision": "finish", "missing_points": [], "conflicts": []},
        ],
    )
    calls = []
    plan = {
        "atomic_points": [
            {"id": "P1", "task": "historical fact"},
            {"id": "P2", "task": "current fact"},
        ]
    }
    monkeypatch.setattr(ToolRegistry, "has", classmethod(lambda cls, name: True))
    monkeypatch.setattr(
        ToolRegistry,
        "metadata",
        classmethod(lambda cls, name: {"retrieval_role": "discovery"}),
    )

    def execute(_cls, action, args, context, phase=None):
        del context, phase
        calls.append((action, dict(args)))
        return json.dumps(
            {
                "status": "ok",
                "results": [
                    {
                        "title": "new P1 source",
                        "url": "https://example.com/p1-new",
                        "content": "P1 material",
                    }
                ],
            }
        )

    monkeypatch.setattr(ToolRegistry, "execute", classmethod(execute))
    answer = run_unified_research_loop(owner, "mixed question", {}, plan, 6)

    assert answer == "rwkv final"
    assert calls == [("web_search", {"query": "another P1 source"})]
    assert owner.planner.replan_progress == []
    assert any("pending_replan" in observation for observation in owner.planner.observations)


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


def test_planner_context_prefers_selected_original_chunk_over_page_preamble():
    state = AgentState(task_id="SELECTED_ROUTING_SPAN", user_query="current release")
    state.retrieval.claims.initialize(_plan(), "current release")
    state.retrieval.record_query(
        "current release",
        {
            "status": "ok",
            "results": [
                {
                    "title": "long release page",
                    "url": "https://example.com/releases",
                    "source_chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "index": 0,
                            "text": "IRRELEVANT_PAGE_PREAMBLE " + ("navigation " * 200),
                        },
                        {
                            "chunk_id": "chunk-9",
                            "index": 8,
                            "text": "CURRENT_RELEASE_GROUNDED_SECTION",
                        },
                    ],
                    "selected_source_chunks": [
                        {
                            "chunk_id": "chunk-9",
                            "index": 8,
                            "text": "CURRENT_RELEASE_GROUNDED_SECTION",
                        }
                    ],
                }
            ],
        },
        task_point_id="P1",
        step=1,
    )

    context = state.to_retrieval_context()

    assert "CURRENT_RELEASE_GROUNDED_SECTION" in context
    assert "IRRELEVANT_PAGE_PREAMBLE" not in context


def test_planner_context_keeps_adjacent_original_context_around_selected_chunk():
    state = AgentState(task_id="SELECTED_ROUTING_NEIGHBOUR", user_query="current release")
    state.retrieval.claims.initialize(_plan(), "current release")
    source_chunks = [
        {"chunk_id": "chunk-7", "index": 7, "text": "PRECEDING_VERSION_CONTEXT"},
        {"chunk_id": "chunk-8", "index": 8, "text": "CURRENT_RELEASE_GROUNDED_SECTION"},
        {"chunk_id": "chunk-9", "index": 9, "text": "FOLLOWING_DATE_CONTEXT"},
        {"chunk_id": "chunk-1", "index": 0, "text": "UNRELATED_PAGE_PREAMBLE"},
    ]
    state.retrieval.record_query(
        "current release",
        {
            "status": "ok",
            "results": [
                {
                    "title": "long release page",
                    "url": "https://example.com/releases",
                    "source_chunks": source_chunks,
                    "selected_source_chunks": [source_chunks[1]],
                }
            ],
        },
        task_point_id="P1",
        step=1,
    )

    context = state.to_retrieval_context()

    assert "CURRENT_RELEASE_GROUNDED_SECTION" in context
    assert "PRECEDING_VERSION_CONTEXT" in context
    assert "FOLLOWING_DATE_CONTEXT" in context
    assert "UNRELATED_PAGE_PREAMBLE" not in context


def test_planner_context_compacts_large_feedback_and_source_bodies():
    state = AgentState(task_id="BOUNDED_ROUTING", user_query="mixed question")
    plan = {
        "atomic_points": [
            {"id": "P1", "task": "historical fact"},
            {"id": "P2", "task": "current fact"},
        ]
    }
    state.retrieval.claims.initialize(plan, "mixed question")
    state.retrieval.record_query(
        "historical query",
        {
            "status": "ok",
            "results": [
                {
                    "title": "large source",
                    "url": "https://example.com/large",
                    "source_chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "text": "ROUTING_SOURCE_MARKER " + ("large source text " * 5000),
                        }
                    ],
                }
            ],
        },
        task_point_id="P1",
        step=1,
    )
    state.last_feedback = json.dumps(
        {
            "status": "cross_validation_replan",
            "message": "choose a different path",
            "previous_request_status": {
                "attempted": True,
                "match_type": "exact",
                "matched_query": "historical query",
                "urls": [f"https://noise.example/{index}" for index in range(500)],
            },
            "evidence_review": {
                "decision": "replan",
                "missing_points": ["P2"],
                "next_focus": "current fact",
                "task_point_status": {
                    "P1": {"status": "completed", "large": "x" * 10000},
                    "P2": {"status": "not_retrieved", "large": "x" * 10000},
                },
            },
        }
    )

    context = state.to_retrieval_context()

    assert get_token_count(context) <= 3000
    assert "ROUTING_SOURCE_MARKER" in context
    assert '"missing_points":["P2"]' in context
    assert '"P2":{"status":"not_retrieved"}' in context
    assert "https://noise.example/499" not in context


def test_cross_validation_uses_bounded_evidence_and_compact_routing(monkeypatch):
    orchestrator = Orchestrator()
    orchestrator.state.task_id = "COMPACT_CROSS_VALIDATION"
    orchestrator.state.user_query = "mixed historical and current question"
    plan = {
        "atomic_points": [
            {"id": "P1", "task": "historical fact"},
            {"id": "P2", "task": "current fact"},
        ]
    }
    orchestrator.state.retrieval.claims.initialize(plan, orchestrator.state.user_query)
    orchestrator.state.retrieval.record_query(
        "historical query",
        {
            "status": "ok",
            "results": [
                {
                    "title": "large fetched page",
                    "url": "https://example.com/large",
                    "claim_ids": ["P1"],
                    "content": "retrieved page body",
                    "model_locator_facts": "LOCATOR " + ("exact source span " * 1000),
                    "source_chunks": [
                        {
                            "chunk_id": f"chunk-{index}",
                            "index": index,
                            "text": f"ORIGINAL_CHUNK_{index} "
                            + ("retrieved page body " * 900),
                        }
                        for index in range(6)
                    ],
                }
            ],
        },
        task_point_id="P1",
        step=1,
    )

    def reject_full_routing(*_args, **_kwargs):
        raise AssertionError("full routing snapshot must not enter cross-validation")

    compact_routing = {
        "schema_version": "planner-routing.v1",
        "marker": "COMPACT_ROUTING_MARKER",
        "claims": [
            {"claim_id": "P1", "source_count": 1},
            {"claim_id": "P2", "source_count": 0},
        ],
    }
    monkeypatch.setattr(
        orchestrator.state.retrieval,
        "routing_snapshot",
        reject_full_routing,
    )
    monkeypatch.setattr(
        orchestrator.state.retrieval,
        "planner_routing_snapshot",
        lambda: compact_routing,
    )
    captured = {}

    def review(
        _query,
        _plan,
        evidence_context,
        *,
        evidence_refs=None,
        evidence_text_by_ref=None,
    ):
        captured["context"] = evidence_context
        captured["evidence_refs"] = list(evidence_refs or [])
        captured["evidence_text_by_ref"] = dict(evidence_text_by_ref or {})
        return {"decision": "replan", "missing_points": ["P2"]}

    orchestrator.planner.cross_validate_research = review
    monkeypatch.setattr("agent.orchestrator.append_task_event", lambda *_args, **_kwargs: None)

    result = orchestrator._cross_validate_research(
        orchestrator.state.user_query,
        plan,
        step=2,
    )

    assert result["decision"] == "replan"
    assert "COMPACT_ROUTING_MARKER" in captured["context"]
    assert "ORIGINAL_CHUNK_0" in captured["context"]
    assert "ORIGINAL_CHUNK_2" not in captured["context"]
    assert captured["evidence_refs"] == ["S1"]
    assert get_token_count(captured["context"]) <= 5000


def test_cross_validation_persists_rwkv_supported_source_bindings(monkeypatch):
    orchestrator = Orchestrator()
    orchestrator.state.task_id = "CROSS_VALIDATED_SOURCE_BINDING"
    orchestrator.state.user_query = "current release"
    plan = {
        "atomic_points": [
            {"id": "P1", "task": "current release", "evidence_needed": ["version"]}
        ]
    }
    orchestrator.state.retrieval.claims.initialize(plan, orchestrator.state.user_query)
    orchestrator.state.retrieval.record_query(
        "official current release",
        {
            "status": "ok",
            "results": [
                {
                    "title": "Official release",
                    "url": "https://example.com/current",
                    "content": "The current release and version are explicitly listed here.",
                    "source_chunks": [
                        {
                            "chunk_id": "current",
                            "index": 0,
                            "text": "The current release and version are explicitly listed here.",
                        }
                    ],
                }
            ],
        },
        task_point_id="P1",
        step=1,
    )
    orchestrator.planner.cross_validate_research = lambda *_args, **_kwargs: {
        "decision": "finish",
        "missing_points": [],
        "conflicts": [],
        "task_point_status": {
            "P1": {
                "status": "supported",
                "evidence_refs": ["S1"],
            }
        },
    }
    emitted_events = []
    monkeypatch.setattr(
        "agent.orchestrator.append_task_event",
        lambda *args, **kwargs: emitted_events.append((args, kwargs)),
    )

    review = orchestrator._cross_validate_research(
        orchestrator.state.user_query,
        plan,
        step=2,
    )

    assert review["validated_source_urls"] == ["https://example.com/current"]
    assert review["evidence_ref_map"]["S1"]["url"] == "https://example.com/current"
    assert emitted_events[-1][0][1] == "cross_validation"
    assert emitted_events[-1][1]["task_point_status"]["P1"] == {
        "status": "supported",
        "evidence_urls": ["https://example.com/current"],
        "supported_facts": [],
        "missing_fields": [],
    }
    assert orchestrator.state.run_metadata["validated_source_urls"] == [
        "https://example.com/current"
    ]
    assert orchestrator.state.run_metadata["last_cross_validation"] == {
        "schema_version": "rwkv-cross-validation.v1",
        "decision": "finish",
        "missing_points": [],
        "conflicts": [],
        "task_point_status": {
                "P1": {
                    "status": "supported",
                    "evidence_urls": ["https://example.com/current"],
                    "supported_facts": [],
                    "missing_fields": [],
                }
        },
        "next_focus": "",
        "reason": "",
        "evidence_revision": 1,
    }


def test_evidence_revision_changes_only_for_materially_new_evidence():
    state = AgentState(task_id="EVIDENCE_REVISION", user_query="current release")
    state.retrieval.claims.initialize(
        {"atomic_points": [{"id": "P1", "task": "current release"}]},
        state.user_query,
    )
    base_result = {
        "status": "ok",
        "results": [
            {
                "title": "Release",
                "url": "https://example.com/release",
                "content": "Version 2 is current.",
                "source_chunks": [
                    {"chunk_id": "c1", "index": 0, "text": "Version 2 is current."}
                ],
            }
        ],
    }

    first = state.retrieval.record_query(
        "current release", base_result, task_point_id="P1", step=1
    )
    duplicate = state.retrieval.record_query(
        "current release", base_result, task_point_id="P1", step=2
    )
    improved_result = json.loads(json.dumps(base_result))
    improved_result["results"][0]["chunk_candidates"] = [
        {
            "chunk_id": "c1",
            "chunk_index": 0,
            "supported": True,
            "source_grounded": True,
            "quote": "Version 2 is current.",
        }
    ]
    improved = state.retrieval.record_query(
        "official release", improved_result, task_point_id="P1", step=3
    )

    assert first["evidence_revision"] == 1
    assert first["material_changed"] is True
    assert duplicate["evidence_revision"] == 1
    assert duplicate["material_changed"] is False
    assert improved["evidence_revision"] == 2
    assert improved["material_changed"] is True


def test_latest_evidence_revision_is_cross_validated_once(monkeypatch):
    orchestrator = Orchestrator()
    orchestrator.state.task_id = "LATEST_REVISION_REVIEW"
    orchestrator.state.user_query = "current release"
    plan = {"atomic_points": [{"id": "P1", "task": "current release"}]}
    orchestrator.state.retrieval.claims.initialize(plan, orchestrator.state.user_query)
    orchestrator.state.retrieval.record_query(
        "current release",
        {
            "status": "ok",
            "results": [
                {
                    "title": "Release",
                    "url": "https://example.com/release",
                    "content": "Version 2 is current.",
                    "source_chunks": [
                        {"chunk_id": "c1", "index": 0, "text": "Version 2 is current."}
                    ],
                }
            ],
        },
        task_point_id="P1",
        step=1,
    )
    calls = []

    def review(*_args, **_kwargs):
        calls.append(orchestrator.state.retrieval.evidence_revision)
        return {
            "decision": "finish",
            "missing_points": [],
            "conflicts": [],
            "task_point_status": {
                "P1": {"status": "supported", "evidence_refs": ["S1"]}
            },
        }

    orchestrator.planner.cross_validate_research = review
    monkeypatch.setattr("agent.orchestrator.append_task_event", lambda *_args, **_kwargs: None)

    first = orchestrator._cross_validate_if_evidence_changed(
        orchestrator.state.user_query, plan, step=2
    )
    repeated = orchestrator._cross_validate_if_evidence_changed(
        orchestrator.state.user_query, plan, step=3
    )

    assert first["decision"] == "finish"
    assert repeated is None
    assert calls == [1]


def test_empty_evidence_revision_is_cross_validated_once(monkeypatch):
    orchestrator = Orchestrator()
    orchestrator.state.task_id = "EMPTY_REVISION_REVIEW"
    orchestrator.state.user_query = "current release"
    plan = {"atomic_points": [{"id": "P1", "task": "current release"}]}
    orchestrator.state.retrieval.claims.initialize(plan, orchestrator.state.user_query)
    calls = []

    def review(*_args, **_kwargs):
        calls.append(orchestrator.state.retrieval.evidence_revision)
        return {
            "decision": "replan",
            "missing_points": ["P1"],
            "conflicts": [],
            "task_point_status": {
                "P1": {"status": "missing", "evidence_refs": []}
            },
        }

    orchestrator.planner.cross_validate_research = review
    monkeypatch.setattr("agent.orchestrator.append_task_event", lambda *_args, **_kwargs: None)

    first = orchestrator._cross_validate_if_evidence_changed(
        orchestrator.state.user_query, plan, step=1
    )
    repeated = orchestrator._cross_validate_if_evidence_changed(
        orchestrator.state.user_query, plan, step=2
    )

    assert first["decision"] == "replan"
    assert repeated is None
    assert calls == [0]


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
