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
        self.recovery_turns = []
        self.replan_progress = []
        self.active_missing_point = ""

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
        review = observation.get("evidence_review") if isinstance(observation, dict) else {}
        self.active_missing_point = str(
            (review or {}).get("missing_point_id") or ""
        )
        self.rebuilt_sessions.append(
            {
                "user_query": user_query,
                "env_context": env_context,
                "observation": observation,
                "phase": phase,
            }
        )

    def request_recovery_turn(
        self,
        observation,
        *,
        user_query="",
        env_context="",
        phase="DISCOVERY",
    ):
        self.recovery_turns.append(
            {
                "observation": observation,
                "user_query": user_query,
                "env_context": env_context,
                "phase": phase,
            }
        )
        self.rebuild_session(user_query, env_context, observation, phase)

    def mark_replan_progress(self, task_point_id):
        point_id = str(task_point_id or "")
        if not self.active_missing_point or point_id != self.active_missing_point:
            return
        self.replan_progress.append(point_id)
        self.active_missing_point = ""


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


def test_planner_routing_snapshot_exposes_zero_bound_factual_points():
    state = AgentState(task_id="POINT_PROGRESS", user_query="answer both facts")
    plan = {
        "schema_version": "task_plan.v2",
        "goal": "answer both facts",
        "atomic_points": [
            {"id": "P1", "question": "first fact", "fields": ["date"]},
            {"id": "P2", "question": "second fact", "fields": ["name"]},
        ],
    }
    state.retrieval.claims.initialize(plan, state.user_query)
    state.retrieval.record_query(
        "first fact query",
        {
            "status": "ok",
            "results": [
                {
                    "title": "First source",
                    "url": "https://example.test/first",
                    "content": "first fact source body",
                    "chunk_candidates": [
                        {
                            "chunk_id": "chunk-1",
                            "chunk_index": 0,
                            "supported": True,
                            "source_grounded": True,
                            "task_record_ids": ["P1"],
                            "claim_ids": ["P1"],
                            "field_keys": ["date"],
                            "quote": "first fact source body",
                            "source_locator": {"char_start": 0, "char_end": 22},
                            "grounding_basis": "exact",
                        }
                    ],
                }
            ],
        },
        step=1,
        task_point_id="P1",
        strategy="rwkv_selected",
    )

    snapshot = state.retrieval.planner_routing_snapshot()

    assert snapshot["factual_point_progress"] == [
            {
                "id": "P1",
                "task_bound_candidate_record_count": 1,
                "candidate_evidence_record_count": 1,
                "total_evidence_record_count": 1,
                "attempt_count": 1,
                "retrieval_state": "evidence_recorded",
            },
            {
                "id": "P2",
                "task_bound_candidate_record_count": 0,
                "candidate_evidence_record_count": 0,
                "total_evidence_record_count": 0,
                "attempt_count": 0,
                "retrieval_state": "not_recorded",
            },
    ]


def test_planner_prompt_projection_preserves_factual_point_progress():
    state = AgentState(task_id="POINT_PROMPT", user_query="answer both facts")
    plan = {
        "schema_version": "task_plan.v2",
        "goal": "answer both facts",
        "atomic_points": [
            {"id": "P1", "question": "first fact", "fields": ["date"]},
            {"id": "P2", "question": "second fact", "fields": ["name"]},
        ],
    }
    state.retrieval.claims.initialize(plan, state.user_query)

    projected = json.loads(
        Planner._replan_environment_projection(state.to_retrieval_context())
    )

    assert projected["retrieval_ledger"]["factual_point_progress"] == [
            {
                "id": "P1",
                "task_bound_candidate_record_count": 0,
                "candidate_evidence_record_count": 0,
                "total_evidence_record_count": 0,
                "attempt_count": 0,
                "retrieval_state": "not_recorded",
            },
            {
                "id": "P2",
                "task_bound_candidate_record_count": 0,
                "candidate_evidence_record_count": 0,
                "total_evidence_record_count": 0,
                "attempt_count": 0,
                "retrieval_state": "not_recorded",
            },
    ]
    assert projected["retrieval_ledger"]["unassigned_source_count"] == 0


def test_planner_projection_carries_exact_records_not_whole_page_bindings():
    state = AgentState(task_id="RECORD_PROMPT", user_query="current version and date")
    plan = {
        "schema_version": "task_plan.v2",
        "goal": "current version and date",
        "atomic_points": [
            {
                "id": "P1",
                "question": "current version and date",
                "fields": ["version", "date"],
            }
        ],
    }
    state.retrieval.claims.initialize(plan, state.user_query)
    state.retrieval.record_query(
        "current release",
        {
            "status": "ok",
            "results": [
                {
                    "title": "Official release",
                    "url": "https://example.test/releases",
                    "content": "Version 4.4 was released on 2026-08-01. Old navigation text.",
                    "chunk_candidates": [
                        {
                            "chunk_id": "release-row",
                            "chunk_index": 0,
                            "supported": True,
                            "source_grounded": True,
                            "task_record_ids": ["P1"],
                            "field_keys": ["version", "date"],
                            "subject_key": "",
                            "record_key": "Version 4.4",
                            "quote": "Version 4.4 was released on 2026-08-01.",
                        }
                    ],
                }
            ],
        },
        task_point_id="P1",
    )

    projected = json.loads(
        Planner._replan_environment_projection(state.to_retrieval_context())
    )
    records = projected["evidence_records"]["records"]

    assert len(records) == 1
    assert records[0]["task_record_id"] == "P1"
    assert records[0]["record_key"] == "Version 4.4"
    assert records[0]["quote"] == "Version 4.4 was released on 2026-08-01."
    assert "Old navigation text" not in json.dumps(records)

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

    def _cross_validate_if_evidence_changed(self, *args, **kwargs):
        revision = int(self.state.retrieval.evidence_revision or 0)
        if self.state.run_metadata.get("last_cross_validation_evidence_revision") == revision:
            return None
        review = self._cross_validate_research(*args, **kwargs)
        if str(review.get("decision") or "") in {"finish", "replan"}:
            self.state.run_metadata["last_cross_validation_evidence_revision"] = revision
        return review

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
        "schema_version": "task_plan.v2",
        "goal": "question",
        "atomic_points": [
            {"id": "P1", "question": "question", "fields": [], "time_scope": "unspecified"}
        ]
    }


def _duplicate_recovery_turns(owner):
    return [
        row
        for row in owner.planner.recovery_turns
        if str((row.get("observation") or {}).get("error_class") or "")
        in {"exact_duplicate_request", "equivalent_duplicate_query"}
    ]


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


def test_finish_request_receives_one_rwkv_binary_review_before_writer():
    owner = FakeOwner([{"action": "finish_task", "args": {}, "task_point_id": "P1"}])
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 5)
    assert answer == "rwkv final"
    assert owner.finished[0]["rounds"] == []
    assert owner.finished[0]["termination_reason"] == "rwkv_planner_finish"
    assert len(owner.review_calls) == 1


def test_exact_duplicate_is_frozen_and_replanned_without_network_reexecution(monkeypatch):
    calls = []
    decision = {
        "action": "web_search",
        "args": {"query": "same query"},
        "task_point_id": "P1",
    }
    owner = FakeOwner([decision, decision, {"action": "finish_task", "args": {}}])
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
    assert owner.state.retrieval.replan_count == 1
    assert len(owner.planner.rebuilt_sessions) == 1
    assert len(_duplicate_recovery_turns(owner)) == 1
    assert _duplicate_recovery_turns(owner)[0]["observation"]["frozen_path"]["query"] == "same query"
    assert owner.finished[0]["termination_reason"] == "rwkv_planner_finish"
    assert not hasattr(owner.planner, "begin_replan")


def test_duplicate_recovery_runs_only_the_next_rwkv_selected_query(monkeypatch):
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
    owner = FakeOwner([first, first, follow_up, {"action": "finish_task", "args": {}}])
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
    assert len(owner.review_calls) == 1
    assert len(_duplicate_recovery_turns(owner)) == 1
    assert _duplicate_recovery_turns(owner)[0]["observation"]["frozen_path"]["query"] == "same query"
    assert owner.state.retrieval.replan_count == 1


def test_distinct_model_selected_url_is_not_blocked_by_claim_binding(monkeypatch):
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
    owner = FakeOwner([first, first, alternative, {"action": "finish_task", "args": {}}])
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
    assert calls == [
        ("web_search", {"query": "same query"}),
        ("web_search", {"query": "different route with an unsupported page"}),
    ]
    assert owner.planner.replan_progress == []


def test_second_duplicate_hits_resource_boundary_after_one_binary_review(monkeypatch):
    repeated = {
        "action": "web_search",
        "args": {"query": "same frozen query"},
        "task_point_id": "P1",
    }
    owner = FakeOwner(
        [repeated, repeated, repeated, {"action": "finish_task", "args": {}}],
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
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 7)

    assert answer == "rwkv final"
    assert calls == [("web_search", {"query": "same frozen query"})]
    assert len(owner.review_calls) == 1
    assert len(_duplicate_recovery_turns(owner)) == 1
    assert owner.state.retrieval.replan_count == 1
    assert owner.finished[0]["termination_reason"] == "resource_duplicate_limit"
    assert any(
        observation.get("status") == "no_new_evidence"
        for observation in owner.planner.observations
    )


def test_duplicate_stall_allows_only_one_rebuilt_planner_then_stops(monkeypatch):
    repeated = {
        "action": "web_search",
        "args": {"query": "same frozen query"},
        "task_point_id": "P1",
    }
    owner = FakeOwner(
        [repeated, repeated, repeated, repeated, repeated, repeated, repeated, repeated],
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
    assert len(_duplicate_recovery_turns(owner)) == 1
    assert len(owner.planner.rebuilt_sessions) == 1
    assert owner.state.retrieval.replan_count == 1
    assert owner.finished[0]["termination_reason"] == "resource_duplicate_limit"


def test_semantically_similar_but_distinct_rwkv_query_is_allowed(monkeypatch):
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
    owner = FakeOwner(
        [first, equivalent, {"action": "finish_task", "args": {}}],
        reviews=[{"decision": "finish"}],
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
    answer = run_unified_research_loop(owner, "question", {}, _plan(), 5)

    assert answer == "rwkv final"
    assert calls == [
        ("web_search", {"query": "alpha beta gamma delta epsilon official"}),
        ("web_search", {"query": "official epsilon delta gamma beta alpha"}),
    ]
    assert len(_duplicate_recovery_turns(owner)) == 0
    assert owner.finished[0]["termination_reason"] == "rwkv_planner_finish"


def test_exact_retrieval_request_can_be_reextracted_for_another_task_point(monkeypatch):
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
        [first, repeated_for_another_point, {"action": "finish_task", "args": {}}],
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
    assert calls == [
        ("web_search", {"query": "one exact shared query"}),
        ("web_search", {"query": "one exact shared query"}),
    ]
    assert len(_duplicate_recovery_turns(owner)) == 0
    assert owner.finished[0]["termination_reason"] == "rwkv_planner_finish"


def test_cross_validation_protocol_error_cannot_block_writer(monkeypatch):
    repeated = {
        "action": "web_search",
        "args": {"query": "same query"},
        "task_point_id": "P1",
    }
    owner = FakeOwner(
        [repeated, repeated, {"action": "finish_task", "args": {}}],
        reviews=[{"decision": "protocol_error", "message": "invalid review"}],
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
    assert len(owner.review_calls) == 1
    assert len(_duplicate_recovery_turns(owner)) == 1
    assert owner.finished[0]["termination_reason"] == "rwkv_planner_finish"


def test_planner_finish_replans_only_when_rwkv_binary_review_requests_it(monkeypatch):
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
            {"decision": "replan"},
            {"decision": "finish"},
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
    assert len(owner.review_calls) == 2
    assert len(owner.planner.rebuilt_sessions) == 1
    assert owner.state.retrieval.replan_count == 1
    assert owner.finished[0]["termination_reason"] == "rwkv_cross_validation_finish"


def test_binary_review_never_supplies_or_overrides_replan_task_point(monkeypatch):
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
            {"decision": "replan"},
            {"decision": "finish"},
        ],
    )
    calls = []
    plan = {
        "schema_version": "task_plan.v2",
        "goal": "mixed question",
        "atomic_points": [
            {"id": "P1", "question": "historical fact", "fields": [], "time_scope": "historical"},
            {"id": "P2", "question": "current fact", "fields": [], "time_scope": "current"},
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
    assert len(owner.review_calls) == 2
    assert owner.planner.active_missing_point == ""
    assert len(owner.planner.rebuilt_sessions) == 1
    assert owner.finished[0]["termination_reason"] == "rwkv_cross_validation_finish"


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
                "missing_point_id": "P2",
                "evidence_needed": "current fact",
            },
        }
    )

    context = state.to_retrieval_context()

    assert get_token_count(context) <= 3000
    assert "ROUTING_SOURCE_MARKER" in context
    assert '"missing_point_id":"P2"' in context
    assert '"evidence_needed":"current fact"' in context
    assert "https://noise.example/499" not in context


def test_cross_validation_uses_only_bounded_source_evidence(monkeypatch):
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
    orchestrator.state.run_metadata["freshness_policy"] = {
        "as_of": None,
        "now": "2026-08-12T00:00:00+00:00",
        "mode": "retrieval_time_only",
        "unknown_date_policy": "preserve_unknown",
    }
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

    monkeypatch.setattr(
        orchestrator.state.retrieval,
        "routing_snapshot",
        reject_full_routing,
    )
    captured = {}

    def review(_query, _plan, evidence_context, routing_context=""):
        captured["context"] = evidence_context
        captured["routing_context"] = routing_context
        return {"decision": "replan"}

    orchestrator.planner.cross_validate_research = review
    orchestrator.planner.select_replan_focus = lambda *_args, **_kwargs: {
        "schema_version": "rwkv-replan-focus.v1",
        "task_point_id": "P2",
        "missing_field": "current fact",
        "focus_owner": "rwkv",
    }
    monkeypatch.setattr("agent.orchestrator.append_task_event", lambda *_args, **_kwargs: None)

    result = orchestrator._cross_validate_research(
        orchestrator.state.user_query,
        plan,
        step=2,
    )

    assert result["decision"] == "replan"
    assert "planner-routing.v1" not in captured["context"]
    assert captured["routing_context"] == ""
    assert "2026-08-12T00:00:00+00:00" in captured["context"]
    assert "ORIGINAL_CHUNK_0" in captured["context"]
    assert "ORIGINAL_CHUNK_2" in captured["context"]
    assert get_token_count(captured["context"]) <= 14000


def test_cross_validation_persists_only_binary_routing_state(monkeypatch):
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
        "schema_version": "rwkv-cross-validation.v3",
        "decision": "finish",
        "selected_action": "write_answer",
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

    assert "validated_source_urls" not in review
    assert "evidence_ref_map" not in review
    assert emitted_events[-1][0][1] == "cross_validation"
    assert "task_point_status" not in emitted_events[-1][1]
    assert "validated_source_urls" not in orchestrator.state.run_metadata
    assert orchestrator.state.run_metadata["last_cross_validation"] == {
        "schema_version": "rwkv-cross-validation.v3",
        "decision": "finish",
        "trigger": "planner_finish",
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


def test_evidence_revision_ignores_candidate_order_only_changes():
    state = AgentState(task_id="REVISION_ORDER", user_query="current release")
    candidates = [
        {
            "chunk_id": "c1",
            "chunk_index": 0,
            "supported": True,
            "source_grounded": True,
            "quote": "Version 2.0",
        },
        {
            "chunk_id": "c2",
            "chunk_index": 1,
            "supported": True,
            "source_grounded": True,
            "quote": "Released 2026-08-13",
        },
    ]

    def payload(rows):
        return {
            "status": "ok",
            "results": [
                {
                    "title": "Release",
                    "url": "https://example.test/release",
                    "content": "Version 2.0\nReleased 2026-08-13",
                    "source_excerpt": "Version 2.0\nReleased 2026-08-13",
                    "chunk_candidates": rows,
                }
            ],
        }

    first = state.retrieval.record_query("release", payload(candidates), step=1)
    reordered = state.retrieval.record_query(
        "release details", payload(list(reversed(candidates))), step=2
    )

    assert first["evidence_revision"] == 1
    assert reordered["evidence_revision"] == 1
    assert reordered["material_changed"] is False


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
            "schema_version": "rwkv-cross-validation.v3",
            "decision": "finish",
            "missing_point_id": "",
            "evidence_needed": "",
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
            "schema_version": "rwkv-cross-validation.v3",
            "decision": "replan",
            "missing_point_id": "P1",
            "evidence_needed": "official current release",
        }

    orchestrator.planner.cross_validate_research = review
    orchestrator.planner.select_replan_focus = lambda *_args, **_kwargs: {
        "schema_version": "rwkv-replan-focus.v1",
        "task_point_id": "P1",
        "missing_field": "current release",
        "focus_owner": "rwkv",
    }
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


def test_new_unique_frozen_route_reopens_cross_validation_once(monkeypatch):
    orchestrator = Orchestrator()
    orchestrator.state.task_id = "ROUTING_REVISION_REVIEW"
    orchestrator.state.user_query = "current release"
    plan = {"atomic_points": [{"id": "P1", "task": "current release"}]}
    orchestrator.state.retrieval.claims.initialize(plan, orchestrator.state.user_query)
    calls: list[str] = []

    def review(_query, _plan, _evidence_context, routing_context=""):
        calls.append(routing_context)
        return {
            "schema_version": "rwkv-cross-validation.v3",
            "decision": "finish",
        }

    orchestrator.planner.cross_validate_research = review
    monkeypatch.setattr("agent.orchestrator.append_task_event", lambda *_args, **_kwargs: None)

    initial = orchestrator._cross_validate_if_evidence_changed(
        orchestrator.state.user_query, plan, step=1
    )
    orchestrator.state.retrieval.freeze_path(
        "current release official",
        action="web_search",
        arguments={"query": "current release official"},
        task_point_id="P1",
        step=2,
        reason="exact_duplicate_request",
    )
    route_changed = orchestrator._cross_validate_if_evidence_changed(
        orchestrator.state.user_query,
        plan,
        step=3,
        trigger="duplicate_resource_boundary",
    )
    repeated = orchestrator._cross_validate_if_evidence_changed(
        orchestrator.state.user_query,
        plan,
        step=4,
        trigger="duplicate_resource_boundary",
    )

    assert initial["decision"] == "finish"
    assert route_changed["decision"] == "finish"
    assert repeated is None
    assert calls[0] == ""
    assert '"frozen_paths":[' in calls[1]


def test_freeze_path_counts_repeats_without_creating_false_route_revisions():
    state = AgentState(task_id="FROZEN_PATH_IDENTITY", user_query="question")
    first = state.retrieval.freeze_path(
        "same query",
        action="web_search",
        arguments={"query": "same query"},
        task_point_id="P1",
        step=2,
        reason="exact_duplicate_request",
    )
    repeated = state.retrieval.freeze_path(
        "same query",
        action="web_search",
        arguments={"query": "same query"},
        task_point_id="P1",
        step=5,
        reason="exact_duplicate_request",
    )

    assert first["route_id"] == repeated["route_id"]
    assert len(state.retrieval.frozen_paths) == 1
    assert state.retrieval.frozen_paths[0]["first_step"] == 2
    assert state.retrieval.frozen_paths[0]["last_step"] == 5
    assert state.retrieval.frozen_paths[0]["blocked_count"] == 2


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
