# RWKV-ECRA/agent/orchestrator.py
import hashlib
import json
import os
import time
from datetime import datetime

from agent.analyzer import Analyzer
from agent.planner import Planner
from agent.state import AgentState
from agent.slm_scheduler import GLOBAL_SLM_INPUT_SCHEDULER
from agent.retrieval_synthesis import build_evidence_context, synthesize_retrieval_answer
from agent.page_evidence import extract_single_page_evidence
from agent.retrieval_loop import (
    merge_retrieval_results,
)
from utils.tracker import EventTracker
from config import (
    TRACKING,
    DATA_PIPELINE,
    get_citation_remote_validation,
    get_llm_base_url,
    get_llm_context_length,
    get_llm_model,
    get_llm_provider,
)
from tools.registry import ToolRegistry
from tools.date_calculator import extract_date_candidates
from utils.task_manager import is_task_stopped, update_task_progress
from utils.task_events import append_task_event
from utils.citation_validator import validate_citations
from utils.evidence_quality import MIN_PAGE_BODY_CHARS, has_substantive_evidence, substantive_evidence_items
from utils.risk_policy import risk_context, validate_risk_answer
from utils.experiment_strategies import normalize_strategy
from utils.time_budget import check_time_budget
from tools.builtin import load_builtin_tools
from retrieval_plugins import is_error, plugin_environment_snapshot
from utils.retrieval_ledger import RetrievalLedger
from utils.answer_fact_check import check_answer_facts
from utils.freshness import build_freshness_policy
from agent.tool_protocol import normalize_tool_result


DEFAULT_MAX_TOOL_STEPS = 12
DEFAULT_MAX_REPLAN_ATTEMPTS = 3
DEFAULT_MAX_NETWORK_SEARCHES = 3
DEFAULT_MAX_DUPLICATE_RETRIES = 1
DEFAULT_MAX_NO_PROGRESS_STEPS = 2


class Orchestrator:
    def __init__(self):
        load_builtin_tools()
        self.tracker = EventTracker(log_dir=TRACKING.get("log_dir", "./logs"), enable=TRACKING.get("enable", True))
        self.state = AgentState()
        self.state.working_memory["__category_tree__"] = {} 
        self.analyzer = Analyzer()
        self.planner = Planner()
        self._task_plan: dict = {}
        self._retrieval_ledger = RetrievalLedger()
        self._calculation_results: list[dict] = []
        self._time_results: list[dict] = []

    def _deterministic_results(self) -> list[dict]:
        """Return deterministic observations in execution order by type."""

        return [*self._calculation_results, *self._time_results]

    def _retrieval_context(self) -> dict:
        return {
            "original_goal": self.state.user_query,
            "path_to_id": self.state.path_to_id,
            "id_to_path": self.state.id_to_path,
            "working_memory": self.state.working_memory,
            "tracker": self.tracker,
            "agent_state": None,
            "task_id": self.state.task_id,
            # The task plan is system-owned routing context.  It lets the
            # generic web tool enforce source policy without exposing provider
            # selection to RWKV or allowing model arguments to rewrite it.
            "task_plan": self._task_plan,
            "slm_scheduler": GLOBAL_SLM_INPUT_SCHEDULER,
        }

    def _agentic_tool_context(self) -> dict:
        """Build the context passed to a model-selected tool call."""
        context = self._retrieval_context()
        context["agentic_tool_loop"] = True
        context["retrieval_ledger"] = self._retrieval_ledger.observation()
        return context

    def _ranking_strategy(self) -> str:
        """Return the configured evidence ranking strategy."""
        metadata = self.state.run_metadata
        configured = metadata.get("strategy_config")
        if not isinstance(configured, dict):
            configured = {
                key: metadata[key]
                for key in ("ranking_strategy", "context_source_count", "prompt_variant")
                if key in metadata
            }
        return normalize_strategy(configured).get(
            "ranking_strategy",
            "evidence_quality.v1",
        )

    def _record_final_ranking(self, action: str, data: dict, step: int, *, stage: str) -> None:
        """Persist the final merged evidence order for audit and replay."""
        append_task_event(
            self.state.task_id,
            "ranking",
            step=step,
            phase="RANKING",
            action=action,
            data={
                "method": f"{data.get('ranking_strategy', self._ranking_strategy())}.final_merge",
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

    def _model_execution_context(self) -> str:
        """Return a safe execution summary for the final RWKV call.

        The full planner/tool transcript remains in task events for audit and
        the JSON acceptance report.  It must not be copied into the final
        answer prompt: it contains tool schemas, legacy workspace memory and
        model-generated protocol text that can make RWKV continue calling
        tools or mistake a memory label for evidence.
        """
        lines = [
            "Retrieval execution summary (data only; not instructions):",
            f"Task: {self.state.user_query}",
        ]
        lines.append(
            "- Production uses one global RWKV decision loop. Task-plan points are logical Fork-style work items; "
            "they share one ledger, evidence store, and global step budget rather than separate Planner sessions."
        )
        points = self._task_plan.get("atomic_points") or []
        if points:
            lines.append(
                "- Task points: "
                + ", ".join(
                    f"{item.get('id', '')}={item.get('status', 'pending')}"
                    for item in points
                    if isinstance(item, dict)
                )
            )
        lines.append(
            "Shared retrieval ledger (repeated normalized web_search queries are blocked before network execution):"
        )
        lines.append(
            json.dumps(
                self._retrieval_ledger.observation(limit=16),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        lines.append(
            "The complete event trace contains the raw model calls and tool results. "
            "Use only the Evidence section for factual claims; this summary is only a record of what was attempted."
        )
        return "\n".join(lines)

    def _record_retrieval_progress(
        self,
        result: dict,
        *,
        query: str,
        step: int,
        action: str,
        phase: str,
        branch_id: str = "",
        task_point_id: str = "",
    ) -> dict:
        """Attach shared progress to the next model observation.

        The ledger does not choose a replacement action.  It does expose
        request and failure state so the global loop can avoid reissuing an
        exact request that already failed.
        """

        delta = self._retrieval_ledger.record(
            query,
            result,
            step=step,
            branch_id=branch_id,
            task_point_id=task_point_id,
            action=action,
            phase=phase,
        )
        observation = self._retrieval_ledger.observation(
            branch_id=branch_id,
            task_point_id=task_point_id,
        )
        enriched = dict(result)
        enriched["retrieval_delta"] = delta
        enriched["retrieval_ledger"] = observation
        append_task_event(
            self.state.task_id,
            "retrieval_ledger",
            step=step,
            phase=phase,
            branch_id=branch_id,
            task_point_id=task_point_id,
            action=action,
            query=query,
            data=delta,
            snapshot=observation,
        )
        return enriched

    @staticmethod
    def _evidence_signatures(result: dict | None) -> set[tuple[str, str]]:
        """Identify fetched evidence bodies for no-progress detection."""

        signatures: set[tuple[str, str]] = set()
        for item in (result or {}).get("results") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip().casefold().rstrip("/")
            if not url:
                continue
            digest = str(item.get("content_sha256") or "").strip()
            if not digest:
                body = str(item.get("source_excerpt") or item.get("content") or "")
                digest = hashlib.sha256(body[:14000].encode("utf-8")).hexdigest()
            signatures.add((url, digest))
        return signatures

    @staticmethod
    def _attach_date_candidates(result: dict) -> dict:
        """Expose deterministic date candidates without choosing their meaning."""

        enriched = dict(result or {})
        rows = []
        for item in enriched.get("results") or []:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            body = "\n".join(
                str(row.get(key) or "")
                for key in ("source_excerpt", "page_excerpt", "content", "structured_evidence_text")
            )
            candidates = extract_date_candidates(body)
            if candidates:
                row["date_candidates"] = candidates[:32]
            rows.append(row)
        if rows:
            enriched["results"] = rows
        return enriched

    def _process_single_page_result(
        self,
        evidence_query: str,
        data: dict,
        step: int,
        evidence_action: str = "fetch_web_url",
    ) -> dict:
        """Map one model-selected page into compact chunk candidates.

        ``data`` may contain the full fetched page because the fetch tool must
        remain auditable.  That raw body is deliberately not returned to the
        planner.  Only the per-chunk candidates and their URL provenance are
        allowed back into the next model turn.
        """

        pages = [item for item in data.get("results") or [] if isinstance(item, dict)]
        if len(pages) != 1:
            compact = dict(data)
            compact["results"] = []
            compact["citation_refs"] = []
            compact["page_evidence"] = {
                "status": "error",
                "message": "the evidence tool must return exactly one page",
            }
            return compact

        page = pages[0]
        url = str(page.get("url") or "")

        def record_chunk(*, chunk, prompt, candidate, task_id):
            append_task_event(
                self.state.task_id,
                "page_chunk",
                step=step,
                phase="EXTRACTION",
                action=evidence_action,
                url=url,
                chunk_id=chunk.get("chunk_id", ""),
                chunk_index=chunk.get("index", 0),
                chunk_chars=len(str(chunk.get("text") or "")),
                chunk_tokens=chunk.get("token_count", 0),
                chunk_text=str(chunk.get("text") or ""),
                evidence_query=evidence_query,
            )
            append_task_event(
                self.state.task_id,
                "page_chunk_candidate",
                step=step,
                phase="EXTRACTION",
                action=evidence_action,
                url=url,
                chunk_id=chunk.get("chunk_id", ""),
                prompt=prompt,
                prompt_chars=len(prompt),
                evidence_query=evidence_query,
                model_output=candidate.get("raw_output", ""),
                model_output_chars=len(str(candidate.get("raw_output") or "")),
                finish_reason=candidate.get("finish_reason", ""),
                retry_count=candidate.get("retry_count", 0),
                candidate=candidate,
            )

        try:
            evidence = extract_single_page_evidence(
                query=evidence_query,
                page=page,
                llm=self.analyzer.llm,
                task_id=self.state.task_id,
                on_chunk=record_chunk,
                task_plan=self._task_plan,
            )
        except Exception as exc:
            evidence = {
                "status": "error",
                "url": url,
                "title": str(page.get("title") or url),
                "page_chars": len(str(page.get("page_excerpt") or page.get("content") or "")),
                "chunk_count": 0,
                "candidates": [],
                "compact_facts": "",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }

        # Keep source text, model locators and request metadata in separate
        # fields.  Previously these were concatenated into ``content``;
        # titles, API metadata and model-generated facts could then pass as
        # page evidence during final synthesis.
        source_body = str(page.get("page_excerpt") or page.get("content") or "").strip()
        source_excerpt = source_body[:14000]
        model_extracted_facts = str(evidence.get("compact_facts") or "").strip()[:14000]
        source_metadata = {
            key: page.get(key)
            for key in ("api_url", "request_params", "source", "project", "language")
            if page.get(key)
        }
        body_verified = len(source_excerpt) >= MIN_PAGE_BODY_CHARS
        chunk_candidates = evidence.get("candidates") or []
        merged_page = {
            "title": page.get("title") or url,
            "url": url,
            "snippet": str(page.get("snippet") or source_excerpt[:600]),
            "page_excerpt": source_excerpt,
            "source_excerpt": source_excerpt,
            "content": source_excerpt,
            "model_extracted_facts": model_extracted_facts,
            "structured_metadata": source_metadata,
            "source": page.get("source") or "explicit model-selected URL",
            "content_type": page.get("content_type", ""),
            "untrusted_content": True,
            "evidence_origin": "fetched_page_body",
            "evidence_boundary": "page_body_only",
            "body_verified": body_verified,
            "content_sha256": hashlib.sha256(source_excerpt.encode("utf-8")).hexdigest() if source_excerpt else "",
            "source_locator": {
                "type": "page_excerpt",
                "char_start": 0,
                "char_end": len(source_excerpt),
            },
            "chunk_count": evidence.get("chunk_count", 0),
            "chunk_window_tokens": evidence.get("chunk_window_tokens", 0),
            "chunk_candidates": chunk_candidates,
            "source_chunks": evidence.get("source_chunks") or [],
            "evidence_status": "ok" if body_verified else "no_evidence",
            "model_extraction_status": "ok" if model_extracted_facts else "no_evidence",
        }
        compact = dict(data)
        compact["results"] = [merged_page] if has_substantive_evidence(merged_page) else []
        compact["sources"] = [url] if url else []
        source_refs = [item for item in data.get("citation_refs") or [] if isinstance(item, dict)]
        compact["citation_refs"] = [
            {
                "ref_id": (source_refs[0].get("ref_id") if source_refs else "") or f"WEB_FETCH_{step}",
                "title": merged_page["title"],
                "url": url,
                "source": merged_page["source"],
                "evidence_text": source_excerpt,
                "evidence_origin": "fetched_page_body",
                "evidence_boundary": "page_body_only",
                "source_locator": merged_page["source_locator"],
            }
        ] if has_substantive_evidence(merged_page) else []
        compact["page_evidence"] = {
            "status": evidence.get("status", "no_evidence"),
            "url": url,
            "title": merged_page["title"],
            "page_chars": evidence.get("page_chars", 0),
            "chunk_count": evidence.get("chunk_count", 0),
            "chunk_window_tokens": evidence.get("chunk_window_tokens", 0),
            "candidate_count": len(evidence.get("candidates") or []),
            "parallel_candidate": evidence.get("parallel_candidate") or {},
            "errors": evidence.get("errors") or [],
        }
        append_task_event(
            self.state.task_id,
            "page_candidate_merge",
            step=step,
            phase="EXTRACTION",
            action=evidence_action,
            url=url,
            data=compact["page_evidence"],
            candidates=evidence.get("candidates") or [],
            source_excerpt=source_excerpt,
            model_extracted_facts=model_extracted_facts,
            structured_metadata=source_metadata,
        )
        return compact

    def _write_agentic_report(
        self,
        user_query: str,
        action: str,
        answer: str,
        *,
        data: dict | None = None,
        mode: str = "model_tool_loop",
    ) -> None:
        report_path = os.path.join(self.state.task_output_dir, "retrieval_report.jsonl")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "record_type": "retrieval_result",
                        "task_id": self.state.task_id,
                        "query": user_query,
                        "action": action,
                        "real_network": bool((data or {}).get("real_network", True)),
                        "answer": answer,
                        "answer_mode": mode,
                        "data": data or {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _replan_after_retrieval_failure(
        self,
        user_query: str,
        observation: dict,
        step: int,
        *,
        attempt: int = 0,
        max_attempts: int = 0,
    ) -> bool:
        """Ask RWKV to split the remaining work after retrieval is insufficient.

        The controller reports execution facts. It never creates a replacement
        query or URL itself; the planner remains responsible for deciding how
        to recover.
        """
        if not self._task_plan:
            return False
        retrieval_observation = {
            "schema_version": "retrieval.v1",
            "status": "error",
            "error_class": str(observation.get("error_class") or "no_usable_evidence"),
            "message": "The selected retrieval observation did not yield usable evidence.",
        }
        followup_plan = self.planner.replan_task(
            user_query,
            self._task_plan,
            retrieval_observation,
            json.dumps(observation, ensure_ascii=False)[:6000],
        )
        append_task_event(
            self.state.task_id,
            "task_replan",
            step=step,
            phase="RECOVERY",
            data=followup_plan,
            previous_observation={
                "status": observation.get("status"),
                "error_class": observation.get("error_class"),
                "page_evidence": observation.get("page_evidence"),
                "evidence_validation": retrieval_observation,
            },
            replan_attempt=attempt,
            max_replan_attempts=max_attempts,
            counts_toward_global_steps=False,
        )
        if followup_plan.get("status") == "error":
            self.planner.observe_tool_result(followup_plan)
            return False
        self._task_plan = followup_plan
        self.state.run_metadata["task_plan"] = followup_plan
        self.planner.update_task_plan(followup_plan)
        return True

    def _build_engineering_evidence_review(
        self,
        user_query: str,
        rounds: list[tuple[str, dict]],
        *,
        observation: dict | None = None,
    ) -> tuple[dict, dict]:
        """Build the control record that RWKV uses for the next decision.

        The engineering validator measures coverage and provenance; it does
        not decide whether the task is complete.  The returned review keeps
        every source URL and visible chunk locator together so the planner can
        choose a new retrieval direction without confusing a search result
        with page evidence.
        """
        if rounds:
            merged = merge_retrieval_results(
                user_query,
                str((observation or {}).get("action") or "retrieval_review"),
                rounds,
                ranking_strategy=self._ranking_strategy(),
            )
        else:
            merged = {
                "query": user_query,
                "status": "no_evidence",
                "results": [],
                "citation_refs": [],
                "sources": [],
                "round_count": 0,
            }
        context = build_evidence_context(
            merged,
            constraints=self.state.run_metadata,
            query=user_query,
        )
        validation = context.get("validation") or {}
        coverage = [
            {
                "point_id": str(row.get("point_id") or ""),
                "status": str(row.get("status") or "missing"),
                "source_count": int(row.get("source_count") or 0),
                "independent_host_count": int(row.get("independent_host_count") or 0),
                "agreement": str(row.get("agreement") or "no_source"),
                "observed_dates": list(row.get("observed_dates") or [])[:16],
                "sources": [
                    {
                        "ref_id": str(source.get("ref_id") or ""),
                        "host": str(source.get("host") or ""),
                        "authority_label": str((source.get("authority") or {}).get("label") or ""),
                        "authority_satisfied": bool((source.get("authority") or {}).get("satisfied")),
                    }
                    for source in list(row.get("sources") or [])[:8]
                    if isinstance(source, dict)
                ],
            }
            for row in list(validation.get("subquestion_coverage") or [])
            if isinstance(row, dict)
        ]
        missing_point_ids = [
            row["point_id"]
            for row in coverage
            if row["status"] != "covered" and row["point_id"]
        ]
        conflict_rows = list((validation.get("cross_source") or {}).get("candidate_conflicts") or [])
        conflict_point_ids = [
            str(row.get("point_id") or "")
            for row in conflict_rows
            if isinstance(row, dict) and str(row.get("point_id") or "")
        ]
        source_bindings: list[dict] = []
        for item in list(context.get("selected_evidence") or [])[:8]:
            if not isinstance(item, dict):
                continue
            ref_id = str(item.get("ref_id") or "")
            spans = []
            for chunk in list(item.get("chunks") or [])[:8]:
                if not isinstance(chunk, dict):
                    continue
                index = int(chunk.get("index", 0) or 0)
                spans.append(
                    {
                        "span_id": f"{ref_id}:C{index + 1}",
                        "chunk_id": str(chunk.get("chunk_id") or f"chunk-{index + 1}"),
                        "index": index,
                        "token_count": int(chunk.get("token_count") or 0),
                    }
                )
            source_bindings.append(
                {
                    "ref_id": ref_id,
                    "url": str(item.get("url") or ""),
                    "title": str(item.get("title") or ""),
                    "evidence_origin": str(item.get("evidence_origin") or ""),
                    "evidence_boundary": str(item.get("evidence_boundary") or ""),
                    "authority": item.get("authority") or {},
                    "spans": spans,
                }
            )
        if not context.get("usable_evidence_count"):
            evidence_state = "none"
        elif missing_point_ids or conflict_point_ids:
            evidence_state = "partial_or_conflicted"
        else:
            evidence_state = "all_points_have_candidate_coverage"
        review = {
            "schema_version": "evidence_review.v1",
            "validator": "engineering_evidence_validation.v1",
            "decision_owner": "rwkv",
            "status": "ok",
            "evidence_state": evidence_state,
            "round_count": len(rounds),
            "usable_evidence_count": int(context.get("usable_evidence_count") or 0),
            "task_point_count": len(self._task_plan.get("atomic_points") or []),
            "covered_point_ids": [
                row["point_id"] for row in coverage if row["status"] == "covered" and row["point_id"]
            ],
            "missing_point_ids": missing_point_ids,
            "conflict_point_ids": conflict_point_ids,
            "coverage": coverage,
            "cross_source": {
                "multi_source_points": int((validation.get("cross_source") or {}).get("multi_source_points") or 0),
                "single_source_points": int((validation.get("cross_source") or {}).get("single_source_points") or 0),
                "missing_points": int((validation.get("cross_source") or {}).get("missing_points") or 0),
                "repeated_fact_signatures": list((validation.get("cross_source") or {}).get("repeated_fact_signatures") or [])[:16],
                "candidate_conflicts": conflict_rows[:8],
            },
            "source_bindings": source_bindings,
            "next_decision": {
                "must_be_made_by": "rwkv",
                "continue_when": "a requested point is missing, a fact/date conflicts, or another independent source is needed",
                "finish_when": "rwkv judges the requested points sufficiently supported by the visible evidence bodies",
                "duplicate_query": "network request is blocked; keep evidence and choose a different direction or finish",
            },
            "last_observation": {
                "status": str((observation or {}).get("status") or ""),
                "error_class": str((observation or {}).get("error_class") or ""),
            },
        }
        return review, validation

    def _attach_engineering_evidence_review(
        self,
        user_query: str,
        rounds: list[tuple[str, dict]],
        observation: dict,
        *,
        step: int,
        action: str,
        phase: str,
        branch_id: str = "",
        task_point_id: str = "",
    ) -> dict:
        """Attach a bounded validator review to the next RWKV observation."""
        review, validation = self._build_engineering_evidence_review(
            user_query,
            rounds,
            observation={**observation, "action": action},
        )
        enriched = dict(observation)
        enriched["evidence_review"] = review
        append_task_event(
            self.state.task_id,
            "evidence_review",
            step=step,
            phase="VALIDATION",
            action=action,
            branch_id=branch_id,
            task_point_id=task_point_id,
            data=review,
            validation=validation,
            evidence_rounds=len(rounds),
        )
        return enriched

    def _evidence_query_for_point(self, task_point_id: str, fallback: str) -> str:
        """Project the model-selected atomic point into the evidence prompt."""
        point_id = str(task_point_id or "").strip()
        for point in self._task_plan.get("atomic_points") or []:
            if isinstance(point, dict) and str(point.get("id") or "").strip() == point_id:
                objective = str(point.get("objective") or "").strip()
                task = str(point.get("task") or "").strip()
                needed = point.get("evidence_needed") or []
                acceptance = point.get("acceptance_criteria") or []
                output_format = str(point.get("output_format") or "prose").strip()
                if objective:
                    needed_text = "; ".join(str(item).strip() for item in needed if str(item).strip())
                    criteria_text = "; ".join(str(item).strip() for item in acceptance if str(item).strip())
                    return (
                        f"Task: {task or objective}\n"
                        f"Atomic objective: {objective}\n"
                        f"Output format: {output_format}\n"
                        f"Evidence needed: {needed_text}\n"
                        f"Acceptance criteria: {criteria_text}"
                    )
        # A missing point id is a model protocol omission, not permission to
        # inject the entire checklist into every page chunk.  The checklist
        # remains available to planning and judging; page extraction should
        # receive only the original question so it reads the selected page.
        return fallback

    def _complete_without_evidence(
        self,
        user_query: str,
        action: str,
        step: int,
        *,
        termination_reason: str,
    ) -> str:
        """Force a visible final model answer even when no page was usable."""
        execution_context = self._model_execution_context()
        data = {
            "query": user_query,
            "status": "no_evidence",
            "results": [],
            "citation_refs": [],
            "sources": [],
            "calculation_results": self._deterministic_results(),
        }
        synthesis = synthesize_retrieval_answer(
            user_query,
            data,
            llm=self.analyzer.llm,
            constraints=self.state.run_metadata,
            execution_context=execution_context,
            termination_reason=termination_reason,
        )
        answer = str(synthesis.get("content") or "RWKV did not return a final summary.")
        answer_fact_check = check_answer_facts(
            answer,
            evidence=[],
            calculation_results=self._deterministic_results(),
            freshness_policy=self.state.run_metadata.get("freshness_policy") or {},
        )
        append_task_event(
            self.state.task_id,
            "answer_fact_validation",
            step=step,
            phase="VALIDATION",
            data=answer_fact_check,
        )
        append_task_event(
            self.state.task_id,
            "context_build",
            step=step,
            phase="CONTEXT",
            data={
                "context_text": synthesis.get("context_text", ""),
                "selected_evidence": synthesis.get("selected_evidence") or [],
                "context_stats": synthesis.get("context_stats") or {},
                "execution_context": execution_context,
            },
        )
        append_task_event(
            self.state.task_id,
            "synthesis",
            step=step,
            phase="SYNTHESIS",
            action=action,
            content=answer,
            mode=synthesis.get("mode") or "local_rwkv_final",
                evidence_count=0,
                citation_refs=synthesis.get("citation_refs") or [],
                validation=synthesis.get("validation") or {},
                answer_alignment=synthesis.get("answer_alignment") or {},
                answer_quality=synthesis.get("answer_quality") or {},
                answer_fact_check=answer_fact_check,
                prompt=synthesis.get("prompt", ""),
            model_output=synthesis.get("model_output", ""),
            context_text=synthesis.get("context_text", ""),
            selected_evidence=synthesis.get("selected_evidence") or [],
            context_stats=synthesis.get("context_stats") or {},
            termination_reason=termination_reason,
        )
        self.state.final_result = answer
        self.state.is_finished = True
        model_output_available = bool(str(synthesis.get("model_output") or "").strip())
        final_status = (
            "failed"
            if termination_reason == "max_steps_reached" or not model_output_available
            else "completed"
        )
        append_task_event(
            self.state.task_id,
            "final",
            status=final_status,
            content=answer,
            action=action,
            mode=synthesis.get("mode") or "local_rwkv_final",
            termination_reason=termination_reason,
            model_output_available=model_output_available,
            answer_quality=synthesis.get("answer_quality") or {},
            answer_fact_check=answer_fact_check,
        )
        self._write_agentic_report(
            user_query,
            action,
            answer,
            data={**data, "answer_fact_check": answer_fact_check},
            mode=synthesis.get("mode") or "local_rwkv_final",
        )
        return answer

    def _complete_model_tool_loop(
        self,
        user_query: str,
        action: str,
        rounds: list[tuple[str, dict]],
        step: int,
        *,
        termination_reason: str = "model_requested_finish",
    ) -> str:
        """Generate the final answer for a finish decision or step limit."""
        if termination_reason == "max_steps_reached":
            append_task_event(
                self.state.task_id,
                "step_limit_reached",
                step=step,
                phase="SYNTHESIS",
                action=action,
                max_steps=step,
                evidence_rounds=len(rounds),
                message="The retrieval loop reached max_tool_steps; force a final RWKV summary.",
            )
        if not rounds:
            return self._complete_without_evidence(
                user_query,
                action,
                step,
                termination_reason=termination_reason,
            )
        execution_context = self._model_execution_context()
        if rounds:
            merged = merge_retrieval_results(
                user_query,
                action,
                rounds,
                ranking_strategy=self._ranking_strategy(),
            )
            if self._calculation_results or self._time_results:
                merged = dict(merged)
                merged["calculation_results"] = self._deterministic_results()
            self._record_final_ranking(action, merged, step, stage="model_tool_loop_merge")
            synthesis = synthesize_retrieval_answer(
                user_query,
                merged,
                llm=self.analyzer.llm,
                constraints=self.state.run_metadata,
                execution_context=execution_context,
                termination_reason=termination_reason,
            )
            answer = synthesis.get("content") or ""
            citation_validation = validate_citations(
                synthesis.get("citation_refs") or [],
                answer=answer,
                evidence=substantive_evidence_items(merged.get("results") or []),
                check_remote=get_citation_remote_validation(),
            )
            answer_fact_check = check_answer_facts(
                answer,
                evidence=substantive_evidence_items(merged.get("results") or []),
                calculation_results=merged.get("calculation_results") or self._deterministic_results(),
                freshness_policy=self.state.run_metadata.get("freshness_policy") or merged.get("freshness_policy") or {},
            )
            risk_validation = validate_risk_answer(answer, self.state.run_metadata)
            append_task_event(
                self.state.task_id,
                "context_build",
                step=step,
                phase="CONTEXT",
                data={
                    "context_text": synthesis.get("context_text", ""),
                    "selected_evidence": synthesis.get("selected_evidence") or [],
                    "context_stats": synthesis.get("context_stats") or {},
                },
            )
            append_task_event(
                self.state.task_id,
                "evidence_validation",
                step=step,
                phase="VALIDATION",
                data={
                    "validation": synthesis.get("validation") or {},
                    "answer_alignment": synthesis.get("answer_alignment") or {},
                },
            )
            append_task_event(
                self.state.task_id,
                "citation_validation",
                step=step,
                phase="VALIDATION",
                data=citation_validation,
            )
            append_task_event(
                self.state.task_id,
                "risk_validation",
                step=step,
                phase="VALIDATION",
                data=risk_validation,
            )
            append_task_event(
                self.state.task_id,
                "answer_fact_validation",
                step=step,
                phase="VALIDATION",
                data=answer_fact_check,
            )
            append_task_event(
                self.state.task_id,
                "synthesis",
                step=step,
                phase="SYNTHESIS",
                action=action,
                content=answer,
                mode=synthesis.get("mode"),
                evidence_count=synthesis.get("evidence_count", 0),
                citation_refs=synthesis.get("citation_refs") or [],
                citation_validation=citation_validation,
                risk_validation=risk_validation,
                answer_fact_check=answer_fact_check,
                validation=synthesis.get("validation") or {},
                answer_alignment=synthesis.get("answer_alignment") or {},
                answer_quality=synthesis.get("answer_quality") or {},
                prompt=synthesis.get("prompt", ""),
                model_output=synthesis.get("model_output", ""),
                repair_prompt=synthesis.get("repair_prompt", ""),
                repair_output=synthesis.get("repair_output", ""),
                context_text=synthesis.get("context_text", ""),
                selected_evidence=synthesis.get("selected_evidence") or [],
                context_stats=synthesis.get("context_stats") or {},
            )
            self.state.final_result = answer
            self.state.is_finished = True
            model_output_available = bool(str(synthesis.get("model_output") or "").strip())
            force_output = termination_reason == "max_steps_reached"
            final_status = (
                "failed"
                if force_output
                or not model_output_available
                or synthesis.get("mode") in {"local_rwkv_error", "local_rwkv_empty", "rwkv_unavailable"}
                else "completed"
            )
            append_task_event(
                self.state.task_id,
                "final",
                status=final_status,
                content=answer,
                action=action,
                mode=synthesis.get("mode"),
                round_count=merged.get("round_count", len(rounds)),
                citation_refs=synthesis.get("citation_refs") or [],
                citation_validation=citation_validation,
                risk_validation=risk_validation,
                answer_fact_check=answer_fact_check,
                validation=synthesis.get("validation") or {},
                answer_alignment=synthesis.get("answer_alignment") or {},
                answer_quality=synthesis.get("answer_quality") or {},
                termination_reason=termination_reason,
                model_output_available=model_output_available,
            )
            self._write_agentic_report(
                user_query,
                action,
                answer,
                data={**merged, "answer_fact_check": answer_fact_check},
                mode=synthesis.get("mode", "model_tool_loop"),
            )
            return answer

    def _prepare_task_plan(self, user_query: str, phase: str) -> dict:
        """Create the model plan once and initialize the shared transcript."""
        self.planner.reset()
        # Planning starts from the user goal only. Workspace contents remain
        # available to later tool turns, but cannot bias decomposition into a
        # local-file plan.
        plan_context = ""
        task_plan = self.planner.create_task_plan(user_query, plan_context)
        if task_plan.get("status") == "error":
            append_task_event(
                self.state.task_id,
                "task_plan",
                step=0,
                phase="ROUTING",
                data=task_plan,
            )
            return task_plan

        freshness_policy = build_freshness_policy(user_query, task_plan)
        task_plan = {**task_plan, "freshness_policy": freshness_policy}
        append_task_event(
            self.state.task_id,
            "task_plan",
            step=0,
            phase="ROUTING",
            data=task_plan,
        )
        self._task_plan = task_plan
        self.state.run_metadata["task_plan"] = task_plan
        self.state.run_metadata["freshness_policy"] = freshness_policy
        generic_web_mode = bool(self.state.run_metadata.get("generic_web_search_only"))
        # Keep the legacy call shape for existing planner test doubles. The
        # generic web mode is the only path that needs an explicit phase.
        if generic_web_mode:
            self.planner.begin_task(user_query, plan_context, task_plan, phase)
        else:
            self.planner.begin_task(user_query, plan_context, task_plan)
        return task_plan

    def _fail_task_plan(self, user_query: str, task_plan: dict) -> str:
        answer = "Task planning failed; retrieval did not start."
        self.state.final_result = answer
        self.state.is_finished = True
        append_task_event(
            self.state.task_id,
            "final",
            status="failed",
            content=answer,
            action="task_plan",
            mode="model_task_plan_failed",
            error_class=task_plan.get("error_class", "task_plan_invalid"),
            planner_error=task_plan.get("message", ""),
        )
        self._write_agentic_report(user_query, "task_plan", answer, mode="model_task_plan_failed")
        return answer

    def _run_model_tool_loop(self, user_query: str, model_profile: dict) -> str:
        """Run the one global model-owned retrieval loop.

        The task plan is a shared checklist (the Fork idea), not a request to
        create independent planner conversations.  All task points, queries,
        evidence, validation, and the step budget stay in one state object.
        """
        generic_web_mode = bool(self.state.run_metadata.get("generic_web_search_only"))
        max_steps = max(
            1,
            int(
                self.state.run_metadata.get(
                    "max_tool_steps",
                    DEFAULT_MAX_TOOL_STEPS,
                )
                or DEFAULT_MAX_TOOL_STEPS
            ),
        )
        phase = "GENERIC_WEB" if generic_web_mode else "ALL"
        task_plan = self._prepare_task_plan(user_query, phase)
        if task_plan.get("status") == "error":
            return self._fail_task_plan(user_query, task_plan)

        point_ids = [
            str(point.get("id") or "")
            for point in task_plan.get("atomic_points") or []
            if isinstance(point, dict) and str(point.get("id") or "").strip()
        ]
        decision = {
            "strategy": "single_loop",
            "point_count": len(point_ids),
            "point_ids": point_ids,
            "source": "global_shared_state",
            "reason": (
                "task points are logical Fork-style work items inside one "
                "shared planner, retrieval ledger, evidence store, and step budget"
            ),
        }
        self.state.run_metadata["retrieval_strategy"] = "single_loop"
        self.state.run_metadata["retrieval_strategy_decision"] = decision
        append_task_event(
            self.state.task_id,
            "retrieval_strategy_selected",
            step=0,
            phase="ROUTING",
            **decision,
            override=False,
        )
        return self._run_single_loop(user_query, model_profile, task_plan, max_steps)

    def _run_single_loop(
        self,
        user_query: str,
        model_profile: dict,
        task_plan: dict,
        max_steps: int,
    ) -> str:
        """Run one continuous RWKV retrieval context with shared state.

        Task-plan points remain a semantic checklist, not independent
        conversations.  Retrieval backends and chunk extraction may still
        use bounded concurrency; the model-facing decision state is global.
        """
        generic_web_mode = bool(self.state.run_metadata.get("generic_web_search_only"))
        rounds: list[tuple[str, dict]] = []
        phase = "GENERIC_WEB" if generic_web_mode else "ALL"
        last_action = "model_tool_loop"
        retrieval_attempted = False
        last_discovery_results: list[dict] = []
        last_retrieval_failure: dict | None = None
        empty_evidence_attempts = 0
        replan_attempts = 0
        duplicate_recovery_turns = 0
        network_searches = 0
        no_progress_steps = 0
        max_network_searches = max(
            1,
            int(
                self.state.run_metadata.get(
                    "max_network_searches",
                    DEFAULT_MAX_NETWORK_SEARCHES,
                )
                or DEFAULT_MAX_NETWORK_SEARCHES
            ),
        )
        max_duplicate_retries = max(
            0,
            int(
                self.state.run_metadata.get(
                    "max_duplicate_retries",
                    DEFAULT_MAX_DUPLICATE_RETRIES,
                )
                or DEFAULT_MAX_DUPLICATE_RETRIES
            ),
        )
        max_no_progress_steps = max(
            1,
            int(
                self.state.run_metadata.get(
                    "max_no_progress_steps",
                    DEFAULT_MAX_NO_PROGRESS_STEPS,
                )
                or DEFAULT_MAX_NO_PROGRESS_STEPS
            ),
        )
        max_replan_attempts = max(
            0,
            int(
                self.state.run_metadata.get(
                    "max_replan_attempts",
                    DEFAULT_MAX_REPLAN_ATTEMPTS,
                )
                or DEFAULT_MAX_REPLAN_ATTEMPTS
            ),
        )
        append_task_event(
            self.state.task_id,
            "retrieval_global_started",
            step=0,
            phase="GLOBAL",
            retrieval_phase=phase,
            max_tool_steps=max_steps,
            max_network_searches=max_network_searches,
            max_duplicate_retries=max_duplicate_retries,
            max_no_progress_steps=max_no_progress_steps,
            max_replan_attempts=max_replan_attempts,
            task_point_count=len(task_plan.get("atomic_points") or []),
        )
        # Replanning and evidence verification run inside the current logical
        # step. Their model calls do not consume the global tool-step budget;
        # the next actual planner/tool turn advances the loop normally.
        for step in range(1, max_steps + 1):
            check_time_budget(minimum_seconds=0.2)
            if is_task_stopped(self.state.task_id):
                self.state.final_result = "执行中止: 任务已被手动停止。"
                self.state.is_finished = True
                append_task_event(self.state.task_id, "final", status="stopped", content=self.state.final_result)
                return self.state.final_result

            if no_progress_steps >= max_no_progress_steps:
                append_task_event(
                    self.state.task_id,
                    "retrieval_no_progress_terminated",
                    step=step,
                    phase="RECOVERY",
                    action=last_action,
                    no_progress_steps=no_progress_steps,
                    max_no_progress_steps=max_no_progress_steps,
                    network_searches=network_searches,
                    message="retrieval produced no new evidence or coverage; force a bounded final summary",
                )
                return self._complete_model_tool_loop(
                    user_query,
                    last_action,
                    rounds,
                    step,
                    termination_reason="no_progress",
                )

            context_text = self.state.to_retrieval_context()
            plan = self.planner.plan_next_action(user_query, {}, context_text, phase)
            action = str(plan.get("action") or "").strip()
            args = dict(plan.get("args") or {}) if isinstance(plan.get("args"), dict) else {}
            task_point_id = str(plan.get("task_point_id") or "").strip()
            call_id = str(plan.get("call_id") or "").strip()
            last_action = action or last_action
            append_task_event(
                self.state.task_id,
                "model_tool_decision",
                step=step,
                phase=phase,
                action=action,
                args=args,
                call_id=call_id,
                task_point_id=task_point_id,
                router=plan.get("router", "model_tool_decision"),
                raw_model_output=plan.get("raw_model_output", ""),
                planner_error=plan.get("planner_error", ""),
                model=model_profile,
            )

            if plan.get("planner_error"):
                # Tool turns require JSON, but the final turn is ordinary
                # RWKV continuation.  A malformed tool turn must therefore
                # transition to synthesis instead of replacing the model's
                # answer with a controller-authored error string.  Existing
                # evidence is still passed through unchanged; with no
                # evidence the final model is explicitly told that retrieval
                # returned nothing.
                append_task_event(
                    self.state.task_id,
                    "model_tool_decision_error",
                    step=step,
                    phase=phase,
                    action="model_tool_decision",
                    planner_error=plan.get("planner_error", ""),
                    raw_model_output=plan.get("raw_model_output", ""),
                    transition="final_synthesis",
                )
                termination_reason = (
                    "max_steps_reached" if step >= max_steps else "model_tool_decision_parse_error"
                )
                return self._complete_model_tool_loop(
                    user_query,
                    last_action,
                    rounds,
                    step,
                    termination_reason=termination_reason,
                )

            if action in {"answer_user", "finish_task"}:
                plan_requires_evidence = any(
                    isinstance(point, dict) and bool(point.get("evidence_needed"))
                    for point in (self._task_plan.get("atomic_points") or [])
                )
                if (
                    not rounds
                    and plan_requires_evidence
                    and not retrieval_attempted
                    and not (self._calculation_results or self._time_results)
                ):
                    answer_rejection = {
                        "schema_version": "retrieval.v1",
                        "status": "error",
                        "error_class": "answer_without_evidence",
                        "message": "the task plan declares evidence, but no retrieval evidence has been collected; choose a tool before answering",
                        "results": [],
                    }
                    self.planner.observe_tool_result(answer_rejection)
                    append_task_event(
                        self.state.task_id,
                        "error",
                        step=step,
                        phase="VALIDATION",
                        action=action,
                        error=answer_rejection["message"],
                        error_class=answer_rejection["error_class"],
                    )
                    no_progress_steps += 1
                    phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                    continue
                if rounds:
                    answer = self._complete_model_tool_loop(
                        user_query,
                        last_action,
                        rounds,
                        step,
                        termination_reason=("max_steps_reached" if step >= max_steps else "model_requested_finish"),
                    )
                    return answer
                if self._calculation_results or self._time_results:
                    # Deterministic tool observations (arithmetic or clock)
                    # are agent observations, not direct-answer shortcuts.
                    # Route them through final synthesis so RWKV receives the
                    # exact tool result while preserving the model/tool
                    # boundary and without pretending the result is web
                    # evidence.
                    return self._complete_model_tool_loop(
                        user_query,
                        last_action,
                        [],
                        step,
                        termination_reason=("max_steps_reached" if step >= max_steps else "model_requested_finish"),
                    )
                if retrieval_attempted and last_retrieval_failure and replan_attempts < max_replan_attempts:
                    # A discovery/provider failure is not evidence.  Give the
                    # model-owned planner the configured recovery rounds even
                    # when RWKV emits finish_task immediately after the error.
                    # These replan calls are deliberately outside the global
                    # tool-step counter; the next real tool decision advances
                    # that counter as usual.
                    replan_attempts += 1
                    append_task_event(
                        self.state.task_id,
                        "task_replan_attempt",
                        step=step,
                        phase="RECOVERY",
                        attempt=replan_attempts,
                        max_attempts=max_replan_attempts,
                        trigger="no_usable_retrieval_evidence",
                        counts_toward_global_steps=False,
                    )
                    replanned = self._replan_after_retrieval_failure(
                        user_query,
                        last_retrieval_failure,
                        step,
                        attempt=replan_attempts,
                        max_attempts=max_replan_attempts,
                    )
                    if replanned:
                        no_progress_steps = 0
                        phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                        continue
                if retrieval_attempted:
                    # Once the model has entered a web-retrieval episode, do
                    # not let answer_user bypass the evidence contract by
                    # launching a second free-form answer call.  This is a
                    # fail-closed boundary, not a routing or repetition
                    # penalty: the model still owns every tool and URL choice.
                    return self._complete_model_tool_loop(
                        user_query,
                        last_action,
                        [],
                        step,
                        termination_reason=("max_steps_reached" if step >= max_steps else "model_requested_finish"),
                    )
                result = ToolRegistry.execute(
                    "answer_user",
                    {"original_goal": user_query},
                    self._agentic_tool_context(),
                )
                answer = str(result or "")
                self.state.final_result = answer
                self.state.is_finished = True
                append_task_event(
                    self.state.task_id,
                    "tool_result",
                    step=step,
                    phase="SYNTHESIS",
                    action="answer_user",
                    result=answer,
                    agent_decision=True,
                )
                append_task_event(
                    self.state.task_id,
                    "final",
                    status="completed",
                    content=answer,
                    action="answer_user",
                    mode="model_tool_loop_direct_answer",
                )
                self._write_agentic_report(user_query, "answer_user", answer)
                return answer

            if not ToolRegistry.can_execute(action, phase):
                allowed_tools = ToolRegistry.model_visible_names(phase)
                if ToolRegistry.has(action):
                    error = f"模型选择的工具在当前阶段不可用: {action or '(empty)'} (phase={phase})"
                else:
                    error = f"模型选择了未注册工具: {action or '(empty)'}"
                self.state.last_feedback = error
                self.planner.observe_tool_result(
                    {
                        "schema_version": "retrieval.v1",
                        "status": "error",
                        "error_class": "tool_not_allowed",
                        "message": f"tool '{action or '(empty)'}' is not available in phase '{phase}'",
                        "allowed_tools": allowed_tools,
                        "results": [],
                    }
                )
                append_task_event(self.state.task_id, "error", step=step, phase="ROUTING", error=error, error_class="model_tool_decision")
                no_progress_steps += 1
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                continue

            duplicate_query = None
            if action == "web_search":
                duplicate_query = self._retrieval_ledger.query_status(args.get("query"))
                if not duplicate_query or not duplicate_query.get("attempted"):
                    if network_searches >= max_network_searches:
                        append_task_event(
                            self.state.task_id,
                            "retrieval_network_budget_reached",
                            step=step,
                            phase="RECOVERY",
                            action=action,
                            network_searches=network_searches,
                            max_network_searches=max_network_searches,
                        )
                        return self._complete_model_tool_loop(
                            user_query,
                            last_action,
                            rounds,
                            step,
                            termination_reason="network_budget_reached",
                        )
            if duplicate_query and duplicate_query.get("attempted"):
                duplicate_block = self._retrieval_ledger.record_duplicate_block(
                    args.get("query"),
                    step=step,
                    task_point_id=task_point_id,
                )
                blocked = {
                    "schema_version": "retrieval.v1",
                    "status": "error",
                    "error_class": "duplicate_query",
                    "message": (
                        "This web_search query is an exact or equivalent query already executed in the current "
                        "retrieval episode; no network request was made. Choose a materially different query or "
                        "finish."
                    ),
                    "query_status": {**duplicate_query, **duplicate_block},
                    "retrieval_ledger": self._retrieval_ledger.observation(
                        task_point_id=task_point_id,
                    ),
                    "results": [],
                }
                blocked = self._attach_engineering_evidence_review(
                    user_query,
                    rounds,
                    blocked,
                    step=step,
                    action=action,
                    phase=phase,
                    task_point_id=task_point_id,
                )
                self.state.last_feedback = blocked["message"]
                self.planner.observe_tool_result(blocked)
                duplicate_recovery_turns += 1
                # A blocked duplicate is routing feedback, not a new failed
                # retrieval attempt. Leave a recovery turn so RWKV can choose
                # a materially different query or direction before the
                # no-progress guard forces synthesis.
                no_progress_steps = max(0, no_progress_steps - 1)
                append_task_event(
                    self.state.task_id,
                    "retrieval_duplicate_blocked",
                    step=step,
                    phase="GLOBAL",
                    action=action,
                    args=args,
                    task_point_id=task_point_id,
                    query_status=duplicate_query,
                    execution_status="skipped",
                )
                append_task_event(
                    self.state.task_id,
                    "tool_result",
                    step=step,
                    phase=phase,
                    action=action,
                    result=json.dumps(blocked, ensure_ascii=False),
                    execution_status="blocked_duplicate",
                    decision_source="model",
                    retrieval_environment=plugin_environment_snapshot(),
                )
                append_task_event(
                    self.state.task_id,
                    "retrieval_duplicate_routed",
                    step=step,
                    phase="RECOVERY",
                    action=action,
                    query=str(args.get("query") or ""),
                    evidence_rounds=len(rounds),
                    next_decision="model",
                    network_request_made=False,
                )
                if duplicate_recovery_turns > max_duplicate_retries:
                    append_task_event(
                        self.state.task_id,
                        "retrieval_no_progress_terminated",
                        step=step,
                        phase="RECOVERY",
                        action=action,
                        duplicate_recovery_turns=duplicate_recovery_turns,
                        max_duplicate_retries=max_duplicate_retries,
                        message="duplicate retrieval request made no progress; force final synthesis",
                    )
                    return self._complete_model_tool_loop(
                        user_query,
                        action,
                        rounds,
                        step,
                        termination_reason="duplicate_no_progress",
                    )
                # A duplicate stops only the network request.  The shared
                # ledger and evidence review are now observations for the
                # same RWKV loop, which may cross-check existing evidence,
                # choose a materially different direction, or finish.  The
                # global step limit remains the only loop guard.
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                continue

            previous_request = self._retrieval_ledger.request_status(action, args)
            if previous_request and previous_request.get("failed"):
                blocked = {
                    "schema_version": "retrieval.v1",
                    "status": "error",
                    "error_class": "previous_attempt_failed",
                    "message": (
                        "This exact tool request already failed in the current retrieval episode and was not "
                        "re-executed. Choose a different URL/query or finish."
                    ),
                    "previous_request": previous_request,
                    "results": [],
                }
                blocked["retrieval_ledger"] = self._retrieval_ledger.observation(
                    task_point_id=task_point_id,
                )
                append_task_event(
                    self.state.task_id,
                    "retrieval_request_reused",
                    step=step,
                    phase="GLOBAL",
                    action=action,
                    args=args,
                    task_point_id=task_point_id,
                    previous_request=previous_request,
                    execution_status="skipped",
                )
                blocked = self._attach_engineering_evidence_review(
                    user_query,
                    rounds,
                    blocked,
                    step=step,
                    action=action,
                    phase=phase,
                    task_point_id=task_point_id,
                )
                self.planner.observe_tool_result(blocked)
                no_progress_steps += 1
                append_task_event(
                    self.state.task_id,
                    "tool_result",
                    step=step,
                    phase=phase,
                    action=action,
                    result=json.dumps(blocked, ensure_ascii=False),
                    execution_status="skipped",
                    decision_source="model",
                    retrieval_environment=plugin_environment_snapshot(),
                )
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                continue

            append_task_event(
                self.state.task_id,
                "tool_call",
                step=step,
                phase=phase,
                action=action,
                args=args,
                call_id=call_id,
                decision_source="model",
            )
            try:
                raw_result = ToolRegistry.execute(
                    action,
                    args,
                    self._agentic_tool_context(),
                    phase=phase,
                )
                if isinstance(raw_result, str):
                    try:
                        structured_result = json.loads(raw_result)
                    except json.JSONDecodeError:
                        # Not every registered tool is a retrieval provider.
                        # Plain text is a successful observation, not a tool
                        # execution failure; only structured status=error is
                        # routed through generic recovery.
                        structured_result = {"status": "ok", "raw": raw_result}
                else:
                    structured_result = raw_result
                if not isinstance(structured_result, dict):
                    structured_result = {"status": "ok", "raw": raw_result}
                structured_result = normalize_tool_result(
                    structured_result,
                    tool_name=action,
                    call_id=call_id,
                )
                if action == "web_search":
                    structured_result = self._attach_date_candidates(structured_result)
            except Exception as exc:
                structured_result = {
                    "status": "error",
                    "provider_errors": [f"{type(exc).__name__}: {exc}"],
                    "results": [],
                }
                raw_result = json.dumps(structured_result, ensure_ascii=False)

            if action == "web_search":
                network_searches += 1

            request_status = self._retrieval_ledger.record_request(
                action,
                args,
                structured_result,
                step=step,
                task_point_id=task_point_id,
            )
            structured_result["request_status"] = request_status

            if (
                action == "connector_lookup"
                and not is_error(structured_result)
                and any(
                    isinstance(item, dict) and has_substantive_evidence(item)
                    for item in structured_result.get("results") or []
                )
            ):
                # Structured connector records are already evidence. They do
                # not enter the page chunker, so preserve them as a retrieval
                # round for final synthesis without changing the chunk path.
                rounds.append((str(args.get("query") or user_query), structured_result))
                retrieval_progressed = True
                no_progress_steps = 0
                last_retrieval_failure = None

            if action == "date_diff" and str(structured_result.get("status") or "") == "ok":
                calculation = {
                    "status": "ok",
                    "tool": "date_diff",
                    "date_a": structured_result.get("date_a", ""),
                    "date_b": structured_result.get("date_b", ""),
                    "days": structured_result.get("days"),
                    "signed_days": structured_result.get("signed_days"),
                    "formula": structured_result.get("formula", ""),
                    "source_refs": list(structured_result.get("source_refs") or []),
                }
                self._calculation_results.append(calculation)
                append_task_event(
                    self.state.task_id,
                    "calculation_result",
                    step=step,
                    phase="CALCULATION",
                    action=action,
                    data=calculation,
                    decision_source="model",
                )

            if action == "current_time" and str(structured_result.get("status") or "") == "ok":
                clock = {
                    "status": "ok",
                    "tool": "current_time",
                    "timezone": structured_result.get("timezone", ""),
                    "iso": structured_result.get("iso", ""),
                    "date": structured_result.get("date", ""),
                    "utc_offset": structured_result.get("utc_offset", ""),
                    "observed_at_utc": structured_result.get("observed_at_utc", ""),
                    "deterministic": True,
                }
                self._time_results.append(clock)
                append_task_event(
                    self.state.task_id,
                    "deterministic_tool_result",
                    step=step,
                    phase="CALCULATION",
                    action=action,
                    data=clock,
                    decision_source="model",
                )

            # A search result is metadata only.  A fetched page is processed
            # as one document and one chunk at a time before the planner sees
            # any observation.  Never feed the raw page body back into the
            # model-owned routing transcript.
            observed_result = structured_result
            retrieval_progressed = False
            progress_counted = False
            tool_meta = ToolRegistry.metadata(action)
            retrieval_role = str(tool_meta.get("retrieval_role") or "")
            if retrieval_role in {"discovery", "evidence"}:
                retrieval_attempted = True
            if action == "web_search":
                # ``web_search`` is a complete bounded retrieval transaction
                # in every production mode: its result already contains
                # admitted URLs, fetched pages, Markdown chunks and merged
                # evidence. Keep usable records for final synthesis instead
                # of silently dropping them outside legacy GENERIC_WEB mode.
                retrieval_attempted = True
                last_discovery_results = [
                    item for item in (structured_result.get("results") or []) if isinstance(item, dict)
                ]
                current_signatures = self._evidence_signatures(structured_result)
                previous_signatures = set().union(
                    *(self._evidence_signatures(round_result) for _, round_result in rounds)
                ) if rounds else set()
                retrieval_progressed = bool(current_signatures - previous_signatures)
                if structured_result.get("evidence_ready") and last_discovery_results and retrieval_progressed:
                    rounds.append((str(args.get("query") or user_query), structured_result))
                    last_retrieval_failure = None
                    duplicate_recovery_turns = 0
                    no_progress_steps = 0
                elif structured_result.get("evidence_ready") and last_discovery_results:
                    no_progress_steps += 1
                    progress_counted = True
                    last_retrieval_failure = {
                        **dict(structured_result),
                        "error_class": "no_new_evidence",
                        "message": "web_search returned only evidence bodies already seen in this episode",
                    }
                else:
                    no_progress_steps += 1
                    progress_counted = True
                    last_retrieval_failure = dict(structured_result)
                    last_retrieval_failure.setdefault("error_class", "no_evidence")
                    last_retrieval_failure.setdefault(
                        "message",
                        "web_search returned no usable page evidence",
                    )
            if is_error(structured_result):
                if not progress_counted:
                    no_progress_steps += 1
                    progress_counted = True
                if retrieval_role == "evidence" and last_discovery_results:
                    selected_url = str(args.get("url") or "").strip()
                    alternative_urls = [
                        str(item.get("url") or "").strip()
                        for item in last_discovery_results
                        if isinstance(item, dict)
                        and str(item.get("url") or "").strip()
                        and str(item.get("url") or "").strip() != selected_url
                    ][:8]
                    if alternative_urls:
                        structured_result = dict(structured_result)
                        structured_result["alternative_urls"] = alternative_urls
                        structured_result["recovery_instruction"] = (
                            "The selected page failed. Choose one different URL from alternative_urls "
                            "with the matching evidence tool before starting another search."
                        )
                self.state.last_feedback = (
                    f"[{action}] execution failed; plugin is unavailable for this turn:\n"
                    f"{json.dumps(structured_result, ensure_ascii=False)[:6000]}"
                )
                if retrieval_role in {"discovery", "evidence"} or action == "web_search":
                    structured_result = self._record_retrieval_progress(
                        structured_result,
                        query=str(args.get("query") or args.get("url") or user_query),
                        step=step,
                        action=action,
                        phase=phase,
                        task_point_id=task_point_id,
                    )
                if retrieval_role == "discovery" or action == "web_search":
                    last_retrieval_failure = dict(structured_result)
                structured_result = self._attach_engineering_evidence_review(
                    user_query,
                    rounds,
                    structured_result,
                    step=step,
                    action=action,
                    phase=phase,
                    task_point_id=task_point_id,
                )
                self.planner.observe_tool_result(structured_result)
                append_task_event(
                    self.state.task_id,
                    "tool_result",
                    step=step,
                    phase=phase,
                    action=action,
                    result=raw_result,
                    real_network=bool(structured_result.get("real_network", True)),
                    decision_source="model",
                    execution_status="error",
                    retrieval_environment=plugin_environment_snapshot(),
                )
                append_task_event(
                    self.state.task_id,
                    "provider_error",
                    step=step,
                    phase="RECOVERY",
                    action=action,
                    error=str(structured_result.get("message") or (structured_result.get("provider_errors") or ["tool execution failed"])[0]),
                    error_class=str(structured_result.get("error_class") or "provider_error"),
                    provider=structured_result.get("provider", ""),
                )
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                if retrieval_role == "evidence" and structured_result.get("alternative_urls"):
                    phase = "ALL"
                continue
            if retrieval_role == "evidence":
                evidence_query = self._evidence_query_for_point(task_point_id, user_query)
                observed_result = self._process_single_page_result(
                    evidence_query,
                    structured_result,
                    step,
                    evidence_action=action,
                )
                page_evidence = observed_result.get("page_evidence") or {}
                page_evidence_status = str(page_evidence.get("status") or "").casefold()
                usable_page_evidence = page_evidence_status in {
                    "ok",
                    "supported",
                    "completed",
                }
                # A fetch can return HTTP success while page extraction yields
                # no supported chunk evidence.  For the global request ledger
                # that is still a failed request: otherwise RWKV can keep
                # selecting the same empty page forever.
                if not usable_page_evidence:
                    observed_result.update(
                        {
                            "status": "error",
                            "error_class": "no_evidence",
                            "message": "the selected page produced no usable chunk evidence",
                        }
                    )
                    last_retrieval_failure = dict(observed_result)
                    request_status = self._retrieval_ledger.record_request(
                        action,
                        args,
                        observed_result,
                        step=step,
                        task_point_id=task_point_id,
                    )
                    observed_result["request_status"] = request_status
                if not usable_page_evidence and last_discovery_results:
                    selected_url = str(args.get("url") or "").strip()
                    alternative_urls = [
                        str(item.get("url") or "").strip()
                        for item in last_discovery_results
                        if isinstance(item, dict)
                        and str(item.get("url") or "").strip()
                        and str(item.get("url") or "").strip() != selected_url
                    ][:8]
                    if alternative_urls:
                        observed_result = dict(observed_result)
                        observed_result["alternative_urls"] = alternative_urls
                        observed_result["recovery_instruction"] = (
                            "The selected page produced no evidence. Choose one different URL from alternative_urls "
                            "with the matching evidence tool before refining the search."
                        )
            if retrieval_role in {"discovery", "evidence"} or action == "web_search":
                observed_result = self._record_retrieval_progress(
                    observed_result,
                    query=str(args.get("query") or args.get("url") or user_query),
                    step=step,
                    action=action,
                    phase=phase,
                    task_point_id=task_point_id,
                )
                if retrieval_role == "evidence" and usable_page_evidence:
                    # The current page must be visible to the validator and
                    # to RWKV's next decision in the same turn.  Previously
                    # this append happened after evidence_review was built,
                    # so the review lagged one successful fetch behind.
                    current_signatures = self._evidence_signatures(observed_result)
                    previous_signatures = set().union(
                        *(self._evidence_signatures(round_result) for _, round_result in rounds)
                    ) if rounds else set()
                    retrieval_progressed = bool(current_signatures - previous_signatures)
                    if retrieval_progressed:
                        rounds.append((str(args.get("url") or user_query), observed_result))
                observed_result = self._attach_engineering_evidence_review(
                    user_query,
                    rounds,
                    observed_result,
                    step=step,
                    action=action,
                    phase=phase,
                    task_point_id=task_point_id,
                )
            self.state.last_feedback = (
                f"[{action}] model-selected tool result:\n"
                f"{json.dumps(observed_result, ensure_ascii=False)[:6000]}"
            )
            self.planner.observe_tool_result(observed_result)
            append_task_event(
                self.state.task_id,
                "tool_result",
                step=step,
                phase=phase,
                action=action,
                result=raw_result,
                real_network=bool(structured_result.get("real_network", True)),
                decision_source="model",
                retrieval_environment=plugin_environment_snapshot(),
            )
            if retrieval_role == "evidence":
                # Only chunk-derived facts enter the final retrieval rounds.
                # A fetch with no supported chunk remains a discovery result,
                # allowing the model to select another URL itself.
                if usable_page_evidence:
                    empty_evidence_attempts = 0
                    if retrieval_progressed:
                        no_progress_steps = 0
                        duplicate_recovery_turns = 0
                    elif not progress_counted:
                        no_progress_steps += 1
                        progress_counted = True
                    # Evidence for one point does not finish a multi-point
                    # task.  Return to the shared recovery phase so the same
                    # Planner can select another point, refine the query, or
                    # explicitly finish after inspecting the global ledger.
                    phase = "GENERIC_WEB" if generic_web_mode else "ALL"
                else:
                    empty_evidence_attempts += 1
                    if empty_evidence_attempts >= 2 and empty_evidence_attempts % 2 == 0:
                        if replan_attempts < max_replan_attempts:
                            replan_attempts += 1
                            append_task_event(
                                self.state.task_id,
                                "task_replan_attempt",
                                step=step,
                                phase="RECOVERY",
                                attempt=replan_attempts,
                                max_attempts=max_replan_attempts,
                                counts_toward_global_steps=False,
                            )
                            replanned = self._replan_after_retrieval_failure(
                                user_query,
                                observed_result,
                                step,
                                attempt=replan_attempts,
                                max_attempts=max_replan_attempts,
                            )
                            if replanned:
                                no_progress_steps = 0
                                empty_evidence_attempts = 0
                        else:
                            append_task_event(
                                self.state.task_id,
                                "task_replan_limit_reached",
                                step=step,
                                phase="RECOVERY",
                                attempts=replan_attempts,
                                max_attempts=max_replan_attempts,
                                transition="final_synthesis",
                            )
                            return self._complete_model_tool_loop(
                                user_query,
                                action,
                                rounds,
                                step,
                                termination_reason="replan_limit_reached",
                            )
                    if observed_result.get("alternative_urls"):
                        phase = "ALL"
                    else:
                        phase = "ALL"
            elif retrieval_role == "discovery":
                # Search results expose candidate URLs to the planner but are
                # not final evidence and are never merged into the answer.
                last_discovery_results = [
                    item for item in (structured_result.get("results") or []) if isinstance(item, dict)
                ]
                if last_discovery_results:
                    # A genuinely new candidate set resolves the previous
                    # discovery failure; the next empty/failed retrieval may
                    # start a fresh bounded replan sequence.
                    last_retrieval_failure = None
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"
            else:
                phase = "GENERIC_WEB" if generic_web_mode else "ALL"

        return self._complete_model_tool_loop(
            user_query,
            last_action,
            rounds,
            max_steps,
            termination_reason="max_steps_reached",
        )


    def run(
        self,
        user_query: str,
        task_id: str = None,
        run_metadata: dict | None = None,
    ) -> str:
        check_time_budget(minimum_seconds=0.2)
        self.state.task_id = task_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state.run_metadata = dict(run_metadata or {})
        self._retrieval_ledger = RetrievalLedger()
        self._calculation_results = []
        self._time_results = []
        self.state.task_output_dir = os.path.join(DATA_PIPELINE.get("output_directory", "./data/output"), self.state.task_id)
        os.makedirs(self.state.task_output_dir, exist_ok=True)
        
        self.tracker.track("User_Input", input_data=user_query, output_data=None)
        self.state.user_query = user_query
        append_task_event(self.state.task_id, "user_input", content=user_query)
        check_time_budget(minimum_seconds=0.2)
        try:
            model_profile = {
                "model": get_llm_model(),
                "endpoint": get_llm_base_url(),
                "provider": get_llm_provider(),
                "context_length": get_llm_context_length(),
            }
        except AttributeError:
            model_profile = {}
        append_task_event(
            self.state.task_id,
            "run_started",
            phase="ROUTING",
            experiment=run_metadata or {},
            config={
                "model": model_profile,
                "retrieval": {"strategy": "global_shared_state"},
            },
            prompt_version=(run_metadata or {}).get("prompt_version", "unversioned"),
            run_metadata=run_metadata or {},
        )
        debug_dir = DATA_PIPELINE.get("debug_directory", "./data/debug_slm")
        os.makedirs(debug_dir, exist_ok=True)
        session_id = self.state.task_id
        trace_file = os.path.join(debug_dir, f"DeepResearch_Trace_{session_id}.md")
        
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(f"# Deep Research 执行追踪日志\n\n**启动时间**: {session_id}\n**用户指令**: {user_query}\n\n---\n\n")

        step_count = 0
        progress_log = []
        def push_progress(msg: str):
            progress_log.append(msg)
            update_task_progress(self.state.task_id, "\n".join(progress_log))
            append_task_event(
                self.state.task_id,
                "progress",
                step=step_count,
                message=msg,
            )

        push_progress("🚀 已启用联网检索，等待 RWKV 自主选择工具...")

        return self._run_model_tool_loop(user_query, model_profile)
