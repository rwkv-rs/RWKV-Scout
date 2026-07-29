"""Controlled retrieval path used by explicit experiments.

The normal production path is model-owned and enters the strategy-selected
Single-loop/Fork runners. This mixin keeps legacy provider-pinned and
retrieval-only experiments isolated without changing their trace contract.
"""

from __future__ import annotations

import json
import os
import time

from agent.retrieval_loop import (
    execute_parallel_candidates,
    generate_query_candidates,
    merge_retrieval_results,
)
from agent.retrieval_synthesis import synthesize_retrieval_answer
from config import get_citation_remote_validation
from tools.registry import ToolRegistry
from utils.citation_validator import validate_citations
from utils.experiment_strategies import normalize_strategy
from utils.risk_policy import risk_context, validate_risk_answer
from utils.task_events import append_task_event
from utils.time_budget import check_time_budget


class ControlledRetrievalMixin:
    """Methods for provider-pinned and retrieval-only experiment runs."""


    def _strategy(self) -> dict:
        return normalize_strategy(self.state.run_metadata.get("strategy_config"))

    def _record_final_ranking(self, action: str, data: dict, step: int, *, stage: str) -> None:
        """Persist the post-recovery/post-multi-hop ranking snapshot."""
        append_task_event(
            self.state.task_id,
            "ranking",
            step=step,
            phase="RANKING",
            action=action,
            data={
                "method": f"{data.get('ranking_strategy', self._strategy()['ranking_strategy'])}.final_merge",
                "stage": stage,
                "output_count": len(data.get("results") or []),
                "results": [
                    {
                        "rank": item.get("retrieval_rank"),
                        "url": item.get("url", ""),
                        "dedup_key": item.get("dedup_key", ""),
                        "candidate_queries": item.get("candidate_queries") or [],
                        "candidate_ranks": item.get("candidate_ranks") or [],
                        "rerank_score": item.get("rerank_score"),
                    }
                    for item in data.get("results") or []
                    if isinstance(item, dict)
                ],
            },
        )

    def _run_rwkv_search_round(
        self,
        user_query: str,
        action: str,
        args: dict,
        *,
        step: int,
        round_name: str,
        previous_query: str = "",
        observation: dict | None = None,
    ) -> tuple[dict, list[str], dict]:
        check_time_budget(minimum_seconds=0.2)
        round_started = time.perf_counter()
        query_started = time.perf_counter()
        plan = generate_query_candidates(
            self.analyzer.llm,
            user_query,
            action=action,
            scope=str(args.get("scope") or ""),
            observation=observation,
            previous_query=previous_query,
            max_candidates=3 if not previous_query else 2,
        )
        query_duration_ms = round((time.perf_counter() - query_started) * 1000, 1)
        candidates = list(plan.get("queries") or [user_query])
        append_task_event(
            self.state.task_id,
            "query_candidates",
            step=step,
            phase="DISCOVERY",
            round=round_name,
            action=action,
            source=plan.get("source"),
            queries=candidates,
            raw_model_output=plan.get("raw_model_output", ""),
            planner_error=plan.get("error", ""),
            duration_ms=query_duration_ms,
        )
        for candidate in candidates:
            append_task_event(
                self.state.task_id,
                "tool_call",
                step=step,
                phase="DISCOVERY",
                action=action,
                args={**args, "query": candidate},
                round=round_name,
                query_source=plan.get("source"),
            )
        rows = execute_parallel_candidates(action, candidates, args, self._retrieval_context())
        round_duration_ms = round((time.perf_counter() - round_started) * 1000, 1)
        for candidate, value in rows:
            append_task_event(
                self.state.task_id,
                "tool_result",
                step=step,
                phase="DISCOVERY",
                action=action,
                result=json.dumps(value, ensure_ascii=False, indent=2),
                real_network=bool(value.get("real_network", True)),
                round=round_name,
                query=candidate,
                duration_ms=round_duration_ms,
            )
            for result_index, item in enumerate(value.get("results") or [], start=1):
                if not isinstance(item, dict):
                    continue
                extracted = item.get("page_excerpt") or item.get("content") or item.get("abstract") or item.get("snippet") or ""
                append_task_event(
                    self.state.task_id,
                    "content_extract",
                    step=step,
                    phase="EXTRACTION",
                    action=action,
                    query=candidate,
                    result_index=result_index,
                    url=item.get("url", ""),
                    extraction_method="page_excerpt|content|abstract|snippet",
                    extracted_chars=len(str(extracted)),
                    captured_at=item.get("captured_at") or value.get("retrieved_at", ""),
                )
        merged = merge_retrieval_results(
            user_query,
            action,
            rows,
            scope=str(args.get("scope") or ""),
            ranking_strategy=self._strategy()["ranking_strategy"],
        )
        append_task_event(
            self.state.task_id,
            "ranking",
            step=step,
            phase="RANKING",
            action=action,
            data={
                "method": merged.get("ranking_strategy", self._strategy()["ranking_strategy"]),
                "input_count": sum(len(value.get("results") or []) for _, value in rows),
                "output_count": len(merged.get("results") or []),
                "results": [
                    {
                        "rank": item.get("retrieval_rank"),
                        "url": item.get("url", ""),
                        "dedup_key": item.get("dedup_key", ""),
                        "candidate_queries": item.get("candidate_queries") or [],
                        "candidate_ranks": item.get("candidate_ranks") or [],
                        "rerank_score": item.get("rerank_score"),
                    }
                    for item in merged.get("results") or []
                ],
            },
        )
        append_task_event(
            self.state.task_id,
            "candidate_merge",
            step=step,
            phase="DISCOVERY",
            round=round_name,
            action=action,
            data={
                "query_count": len(candidates),
                "queries": candidates,
                "result_count": merged.get("count", 0),
                "round_count": merged.get("round_count", 1),
                "duration_ms": round_duration_ms,
            },
        )
        return merged, candidates, plan

    @staticmethod
    def _citation_recovery_needed(validation: dict, data: dict | None = None) -> bool:
        """Return true only for source-integrity failures worth re-querying."""
        if data and data.get("results") and not data.get("citation_refs"):
            return True
        if data and data.get("evidence_missing_count"):
            return True
        recoverable = {
            "invalid_url",
            "search_result_page",
            "missing_evidence",
            "empty_page",
        }
        return any(
            issue in recoverable
            for row in validation.get("rows") or []
            for issue in row.get("issues") or []
        ) or any(
            str(issue).startswith("inaccessible:")
            for row in validation.get("rows") or []
            for issue in row.get("issues") or []
        )

    def _finish_retrieval_only(self, user_query: str, action: str, data: dict, step: int) -> str:
        """Close a retrieval trace without requiring a model answer.

        This mode is intentionally diagnostic: it measures search, extraction,
        ranking and citation integrity while the RWKV endpoint is unavailable.
        It never fabricates an answer or a reference answer.
        """
        working_data = dict(data)
        citation_refs: list[dict] = []
        citation_validation: dict = {}
        recovery_attempted = False
        risk_policy = risk_context(self.state.run_metadata)
        risk_validation = {
            "validator_version": "risk-policy.v1",
            "high_risk": risk_policy["high_risk"],
            "domain": risk_policy["domain"],
            "valid": True,
            "skipped": True,
            "warning_present": None,
            "issues": ["answer_not_generated_retrieval_only"],
        }
        for attempt in range(2):
            check_time_budget(minimum_seconds=0.2)
            citation_refs = working_data.get("citation_refs") or []
            append_task_event(
                self.state.task_id,
                "synthesis_start",
                step=step + attempt,
                phase="SYNTHESIS",
                action=action,
                attempt=attempt + 1,
                status="skipped",
                reason="retrieval_only",
                evidence_count=len(working_data.get("results") or []),
                round_count=working_data.get("round_count", 1),
            )
            citation_validation = validate_citations(
                citation_refs,
                answer="",
                evidence=working_data.get("results") or [],
                check_remote=get_citation_remote_validation(),
            )
            append_task_event(
                self.state.task_id,
                "citation_validation",
                step=step + attempt,
                phase="VALIDATION",
                attempt=attempt + 1,
                data=citation_validation,
            )
            if attempt == 0 and self._citation_recovery_needed(citation_validation, working_data):
                recovery_attempted = True
                append_task_event(
                    self.state.task_id,
                    "citation_recovery",
                    step=step,
                    phase="RECOVERY",
                    status="scheduled",
                    invalid_count=citation_validation.get("invalid", 0),
                    reason="source_integrity_failure",
                )
                allowed = set(ToolRegistry.metadata(action).get("allowed_args") or ())
                recovery_args = {
                    key: value
                    for key, value in {
                        "scope": "paper",
                        "max_results": 8,
                        "fetch_pages": 3,
                    }.items()
                    if key in allowed
                }
                try:
                    recovery_data, recovery_queries, _ = self._run_rwkv_search_round(
                        user_query,
                        action,
                        recovery_args,
                        step=step + 1,
                        round_name="citation_recovery",
                        previous_query=str(working_data.get("query") or user_query),
                        observation=working_data,
                    )
                    recovery_query = recovery_queries[0] if recovery_queries else user_query
                    working_data = merge_retrieval_results(
                        user_query,
                        action,
                        [
                            (str(working_data.get("query") or user_query), working_data),
                            (recovery_query, recovery_data),
                        ],
                        scope=str(working_data.get("scope") or ""),
                        ranking_strategy=self._strategy()["ranking_strategy"],
                    )
                    working_data["round_count"] = int(data.get("round_count", 1) or 1) + 1
                    self._record_final_ranking(action, working_data, step + attempt, stage="citation_recovery")
                    append_task_event(
                        self.state.task_id,
                        "citation_recovery",
                        step=step,
                        phase="RECOVERY",
                        status="completed",
                        query=recovery_query,
                        result_count=len(recovery_data.get("results") or []),
                    )
                    continue
                except Exception as exc:
                    append_task_event(
                        self.state.task_id,
                        "citation_recovery",
                        step=step,
                        phase="RECOVERY",
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}"[:1000],
                    )
            break
        synthesis = {
            "content": "",
            "mode": "retrieval_only",
            "evidence_count": len(working_data.get("results") or []),
            "citation_refs": citation_refs,
            "context_text": "",
            "selected_evidence": [],
            "context_stats": {"retrieval_only": True, "source_count": len(working_data.get("results") or [])},
        }
        append_task_event(
            self.state.task_id,
            "context_build",
            step=step,
            phase="CONTEXT",
            attempt=2 if recovery_attempted else 1,
            data={
                "context_text": "",
                "selected_evidence": [],
                "context_stats": synthesis["context_stats"],
            },
        )
        append_task_event(
            self.state.task_id,
            "risk_validation",
            step=step,
            phase="VALIDATION",
            attempt=2 if recovery_attempted else 1,
            data=risk_validation,
        )
        self.state.is_finished = True
        self.state.final_result = ""
        append_task_event(
            self.state.task_id,
            "synthesis",
            step=step,
            phase="SYNTHESIS",
            attempt=2 if recovery_attempted else 1,
            content="",
            mode="retrieval_only",
            evidence_count=synthesis["evidence_count"],
            citation_refs=citation_refs,
            citation_validation=citation_validation,
            risk_validation=risk_validation,
            context_text="",
            selected_evidence=[],
            context_stats=synthesis["context_stats"],
            duration_ms=0.0,
        )
        append_task_event(
            self.state.task_id,
            "final",
            status="completed",
            content="",
            action=action,
            mode="retrieval_only",
            round_count=working_data.get("round_count", 1),
            citation_refs=citation_refs,
            citation_validation=citation_validation,
            risk_validation=risk_validation,
            citation_recovery_attempted=recovery_attempted,
            duration_ms=0.0,
        )
        report_path = os.path.join(self.state.task_output_dir, "retrieval_report.jsonl")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "record_type": "retrieval_result",
                "task_id": self.state.task_id,
                "query": user_query,
                "action": action,
                "real_network": bool(working_data.get("real_network", True)),
                "answer": "",
                "answer_mode": "retrieval_only",
                "citation_validation": citation_validation,
                "risk_validation": risk_validation,
                "citation_recovery_attempted": recovery_attempted,
                "retrieval_only": True,
                "data": working_data,
            }, ensure_ascii=False) + "\n")
        return ""

    def _finish_rwkv_retrieval(self, user_query: str, action: str, data: dict, step: int) -> str:
        if self.state.run_metadata.get("retrieval_only") is True:
            return self._finish_retrieval_only(user_query, action, data, step)
        working_data = dict(data)
        synthesis: dict = {}
        citation_validation: dict = {}
        synthesis_duration_ms = 0.0
        recovery_attempted = False

        for attempt in range(2):
            check_time_budget(minimum_seconds=0.2)
            append_task_event(
                self.state.task_id,
                "synthesis_start",
                step=step + attempt,
                phase="SYNTHESIS",
                action=action,
                attempt=attempt + 1,
                evidence_count=len(working_data.get("results") or []),
                round_count=working_data.get("round_count", 1),
            )
            synthesis_started = time.perf_counter()
            synthesis = synthesize_retrieval_answer(
                user_query,
                working_data,
                llm=self.analyzer.llm,
                constraints=self.state.run_metadata,
            )
            synthesis_duration_ms = round((time.perf_counter() - synthesis_started) * 1000, 1)
            answer = synthesis.get("content") or ""
            append_task_event(
                self.state.task_id,
                "context_build",
                step=step + attempt,
                phase="CONTEXT",
                attempt=attempt + 1,
                data={
                    "context_text": synthesis.get("context_text", ""),
                    "selected_evidence": synthesis.get("selected_evidence") or [],
                    "context_stats": synthesis.get("context_stats") or {},
                },
            )
            citation_validation = validate_citations(
                synthesis.get("citation_refs") or working_data.get("citation_refs") or [],
                answer=answer,
                evidence=working_data.get("results") or [],
                check_remote=get_citation_remote_validation(),
            )
            append_task_event(
                self.state.task_id,
                "citation_validation",
                step=step + attempt,
                phase="VALIDATION",
                attempt=attempt + 1,
                data=citation_validation,
            )
            risk_validation = validate_risk_answer(answer, self.state.run_metadata)
            append_task_event(
                self.state.task_id,
                "risk_validation",
                step=step + attempt,
                phase="VALIDATION",
                attempt=attempt + 1,
                data=risk_validation,
            )

            if attempt == 0 and self._citation_recovery_needed(citation_validation, working_data):
                recovery_attempted = True
                append_task_event(
                    self.state.task_id,
                    "citation_recovery",
                    step=step + attempt,
                    phase="RECOVERY",
                    status="scheduled",
                    invalid_count=citation_validation.get("invalid", 0),
                    reason="source_integrity_failure",
                )
                allowed = set(ToolRegistry.metadata(action).get("allowed_args") or ())
                recovery_args = {
                    key: value
                    for key, value in {
                        "scope": "paper",
                        "max_results": 8,
                        "fetch_pages": 3,
                    }.items()
                    if key in allowed
                }
                try:
                    recovery_data, recovery_queries, _ = self._run_rwkv_search_round(
                        user_query,
                        action,
                        recovery_args,
                        step=step + 1,
                        round_name="citation_recovery",
                        previous_query=str(working_data.get("query") or user_query),
                        observation=working_data,
                    )
                    recovery_query = recovery_queries[0] if recovery_queries else user_query
                    working_data = merge_retrieval_results(
                        user_query,
                        action,
                        [
                            (str(working_data.get("query") or user_query), working_data),
                            (recovery_query, recovery_data),
                        ],
                        scope=str(working_data.get("scope") or ""),
                        ranking_strategy=self._strategy()["ranking_strategy"],
                    )
                    working_data["round_count"] = int(data.get("round_count", 1) or 1) + 1
                    self._record_final_ranking(action, working_data, step + attempt, stage="citation_recovery")
                    append_task_event(
                        self.state.task_id,
                        "citation_recovery",
                        step=step + attempt,
                        phase="RECOVERY",
                        status="completed",
                        query=recovery_query,
                        result_count=len(recovery_data.get("results") or []),
                    )
                    continue
                except Exception as exc:
                    append_task_event(
                        self.state.task_id,
                        "citation_recovery",
                        step=step + attempt,
                        phase="RECOVERY",
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}"[:1000],
                    )
            break

        self.state.is_finished = True
        self.state.final_result = synthesis.get("content") or ""
        risk_validation = validate_risk_answer(self.state.final_result, self.state.run_metadata)
        if citation_validation.get("invalid"):
            final_status = "completed_with_citation_warnings"
        elif not risk_validation.get("valid", True):
            final_status = "completed_with_risk_warnings"
        else:
            final_status = "completed"
        append_task_event(
            self.state.task_id,
            "synthesis",
            step=step,
            phase="SYNTHESIS",
            attempt=2 if recovery_attempted else 1,
            content=self.state.final_result,
            mode=synthesis.get("mode"),
            evidence_count=synthesis.get("evidence_count", 0),
            citation_refs=synthesis.get("citation_refs") or [],
            citation_validation=citation_validation,
            risk_validation=risk_validation,
            prompt=synthesis.get("prompt", ""),
            model_output=synthesis.get("model_output", ""),
            repair_prompt=synthesis.get("repair_prompt", ""),
            repair_output=synthesis.get("repair_output", ""),
            context_text=synthesis.get("context_text", ""),
            selected_evidence=synthesis.get("selected_evidence") or [],
            context_stats=synthesis.get("context_stats") or {},
            duration_ms=synthesis_duration_ms,
        )
        append_task_event(
            self.state.task_id,
            "final",
            status=final_status,
            content=self.state.final_result,
            action=action,
            mode=synthesis.get("mode"),
            round_count=working_data.get("round_count", 1),
            citation_refs=synthesis.get("citation_refs") or [],
            citation_validation=citation_validation,
            risk_validation=risk_validation,
            citation_recovery_attempted=recovery_attempted,
            duration_ms=synthesis_duration_ms,
        )
        report_path = os.path.join(self.state.task_output_dir, "retrieval_report.jsonl")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "record_type": "retrieval_result",
                "task_id": self.state.task_id,
                "query": user_query,
                "action": action,
                "real_network": bool(working_data.get("real_network", True)),
                "answer": self.state.final_result,
                "answer_mode": synthesis.get("mode"),
                "citation_validation": citation_validation,
                "risk_validation": risk_validation,
                "citation_recovery_attempted": recovery_attempted,
                "data": working_data,
            }, ensure_ascii=False) + "\n")
        return self.state.final_result
