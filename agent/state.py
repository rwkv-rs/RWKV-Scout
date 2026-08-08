"""State and context projection for one research task."""

from __future__ import annotations

import os
import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Set

from agent.claim_ledger import ClaimLedger
from utils.chunker import get_token_count
from utils.retrieval_ledger import RetrievalLedger, canonical_url


_SOURCE_BODY_FIELDS = (
    "source_excerpt",
    "page_excerpt",
    "structured_evidence_text",
    "content",
    "abstract",
)


def _original_source_chunks(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Return fetched source text without extractor paraphrases or snippets."""

    rows = item.get("source_chunks") or item.get("selected_source_chunks") or []
    chunks = [
        {
            "chunk_id": str(row.get("chunk_id") or f"chunk-{index + 1}"),
            "text": str(row.get("text") or "").replace("\x00", ""),
        }
        for index, row in enumerate(rows)
        if isinstance(row, dict) and str(row.get("text") or "")
    ]
    if chunks:
        return chunks

    for field_name in _SOURCE_BODY_FIELDS:
        body = str(item.get(field_name) or "").replace("\x00", "")
        if body:
            return [{"chunk_id": "body-1", "text": body}]
    return []


@dataclass
class RetrievalEpisodeState:
    """Single shared evidence state for one model-owned retrieval episode.

    The planner, orchestrator, validator and final synthesizer must observe
    the same evidence ledger.  Keeping these collections on the task state
    prevents a replan or phase transition from creating a fresh local view
    and losing the already collected page/point bindings.
    """

    rounds: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    rounds_by_point: Dict[str, list[tuple[str, dict[str, Any]]]] = field(default_factory=dict)
    point_state: Dict[str, dict[str, Any]] = field(default_factory=dict)
    last_discovery_results: list[dict[str, Any]] = field(default_factory=list)
    last_retrieval_failure: dict[str, Any] | None = None
    # The evidence store is task-scoped and shared by the planner, tools and
    # final synthesizer.  It is deliberately external to the model
    # transcript: a follow-up search adds to this store instead of creating a
    # second recovery conversation.
    sources: Dict[str, dict[str, Any]] = field(default_factory=dict)
    sources_by_claim: Dict[str, Dict[str, dict[str, Any]]] = field(default_factory=dict)
    deterministic_results: list[dict[str, Any]] = field(default_factory=list)
    attempted_urls: Set[str] = field(default_factory=set)
    url_attempt_counts: Dict[str, int] = field(default_factory=dict)
    query_history: list[dict[str, Any]] = field(default_factory=list)
    coverage: Dict[str, dict[str, Any]] = field(default_factory=dict)
    frozen_paths: list[dict[str, Any]] = field(default_factory=list)
    infrastructure_events: list[dict[str, Any]] = field(default_factory=list)
    replan_count: int = 0
    progress: RetrievalLedger = field(default_factory=RetrievalLedger)
    claims: ClaimLedger = field(default_factory=ClaimLedger)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @staticmethod
    def source_key(item: dict[str, Any]) -> str:
        url = str(item.get("url") or "").strip().casefold().rstrip("/")
        if url:
            return url
        digest = str(item.get("content_sha256") or "").strip()
        if digest:
            return f"sha256:{digest}"
        body = str(item.get("source_excerpt") or item.get("content") or "")
        return f"sha256:{hashlib.sha256(body.encode('utf-8')).hexdigest()}"

    def record_query(
        self,
        query: str,
        result: dict[str, Any],
        *,
        step: int = 0,
        task_point_id: str = "",
        strategy: str = "",
    ) -> dict[str, Any]:
        """Merge one research round into the shared evidence store."""

        query_text = " ".join(str(query or "").split()).strip()
        added: list[str] = []
        with self._lock:
            self.query_history.append(
                {
                    "query": query_text,
                    "step": int(step or 0),
                    "status": str(result.get("status") or ""),
                    "new_source_count": 0,
                    "task_point_id": str(task_point_id or ""),
                    "strategy": str(strategy or ""),
                }
            )
            extraction = result.get("model_extraction") or {}
            if isinstance(extraction, dict):
                unresolved_chunks = int(extraction.get("unresolved_chunk_count") or 0)
                degraded_pages = int(extraction.get("degraded_page_count") or 0)
                recovered_chunks = int(extraction.get("recovered_chunk_count") or 0)
                self.query_history[-1]["model_extraction_unresolved_chunks"] = unresolved_chunks
                self.query_history[-1]["model_extraction_degraded_pages"] = degraded_pages
                if unresolved_chunks or degraded_pages:
                    self.infrastructure_events.append(
                        {
                            "kind": "model_extraction_incomplete",
                            "query": query_text,
                            "step": int(step or 0),
                            "task_point_id": str(task_point_id or ""),
                            "unresolved_chunk_count": unresolved_chunks,
                            "degraded_page_count": degraded_pages,
                            "transport_error_count": int(extraction.get("transport_error_count") or 0),
                            "recovered_chunk_count": recovered_chunks,
                            "pages": [
                                dict(value)
                                for value in (extraction.get("pages") or [])[:16]
                                if isinstance(value, dict)
                            ],
                        }
                    )
            attempted_this_round: set[str] = set()
            for key in ("results", "candidate_urls", "page_evidence"):
                rows = result.get(key) or []
                if isinstance(rows, dict):
                    rows = [rows]
                for row in rows if isinstance(rows, list) else []:
                    if not isinstance(row, dict):
                        continue
                    url = canonical_url(row.get("url") or row.get("page_url") or row.get("source_url"))
                    if url:
                        self.attempted_urls.add(url)
                        attempted_this_round.add(url)
            for url in attempted_this_round:
                self.url_attempt_counts[url] = self.url_attempt_counts.get(url, 0) + 1
            for item in result.get("results") or []:
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                if task_point_id:
                    claim_ids = [str(value) for value in item.get("claim_ids") or [] if str(value).strip()]
                    if task_point_id not in claim_ids:
                        claim_ids.append(task_point_id)
                    item["claim_ids"] = claim_ids
                item.setdefault("retrieval_query", query_text)
                item.setdefault("retrieval_strategy", str(strategy or ""))
                key = self.source_key(item)
                if key not in self.sources:
                    self.sources[key] = dict(item)
                    added.append(key)
                else:
                    # Keep the richest representation when the same URL is
                    # encountered by a focused follow-up search.
                    current = self.sources[key]
                    if len(str(item.get("content") or "")) > len(str(current.get("content") or "")):
                        self.sources[key] = {**current, **item}
                    else:
                        current["claim_ids"] = list(dict.fromkeys([
                            *[
                                str(value)
                                for value in current.get("claim_ids") or []
                                if str(value).strip()
                            ],
                            *[
                                str(value)
                                for value in item.get("claim_ids") or []
                                if str(value).strip()
                            ],
                        ]))
                if task_point_id:
                    point_sources = self.sources_by_claim.setdefault(task_point_id, {})
                    current_point_source = point_sources.get(key)
                    if (
                        current_point_source is None
                        or len(str(item.get("content") or ""))
                        >= len(str(current_point_source.get("content") or ""))
                    ):
                        point_sources[key] = dict(item)
            self.query_history[-1]["new_source_count"] = len(added)
            self.rounds.append((query_text, result))
            self.last_discovery_results[:] = [
                item for item in result.get("results") or [] if isinstance(item, dict)
            ]
        source_resolution = result.get("source_resolution") or {}
        if isinstance(source_resolution, dict):
            self.claims.update_required_domains(source_resolution.get("required_domains") or [])
        claim_delta = self.claims.ingest(
            query_text,
            result,
            task_point_id=task_point_id,
            strategy=strategy,
            step=step,
        )
        return {
            "new_source_keys": added,
            "new_source_count": len(added),
            "total_sources": len(self.sources),
            "claim_delta": claim_delta,
        }

    def source_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self.sources.values()]

    def record_deterministic_result(
        self,
        action: str,
        result: dict[str, Any],
        *,
        step: int = 0,
        task_point_id: str = "",
    ) -> None:
        """Persist an exact successful calculator/time result for replanning."""

        if str(result.get("status") or "").casefold() != "ok":
            return
        record = {
            "tool": str(action or ""),
            "step": int(step or 0),
            "task_point_id": str(task_point_id or ""),
            "result": dict(result),
        }
        with self._lock:
            self.deterministic_results.append(record)
            if len(self.deterministic_results) > 32:
                del self.deterministic_results[:-32]

    def planner_evidence_snapshot(
        self,
        *,
        max_sources: int = 6,
        max_chars_per_source: int = 3000,
        max_total_chars: int = 9000,
    ) -> dict[str, Any]:
        """Project bounded original source spans into every planner decision.

        This is task memory, not an answerability judgement.  It deliberately
        ignores extractor ``supported`` flags and never includes generated
        fact paraphrases.  Character caps only protect the RWKV context window.
        """

        max_sources = max(1, int(max_sources or 1))
        max_chars_per_source = max(256, int(max_chars_per_source or 256))
        max_total_chars = max(512, int(max_total_chars or 512))
        with self._lock:
            all_sources = [dict(item) for item in self.sources.values()]

        if len(all_sources) <= max_sources:
            selected = all_sources
        else:
            first_count = max_sources // 2
            selected = [
                *all_sources[:first_count],
                *all_sources[-(max_sources - first_count) :],
            ]

        projected: list[dict[str, Any]] = []
        remaining = max_total_chars
        for source_index, item in enumerate(selected, start=1):
            sources_left = len(selected) - source_index + 1
            source_budget = min(
                max_chars_per_source,
                max(0, remaining // max(1, sources_left)),
            )
            if source_budget <= 0:
                break

            spans: list[dict[str, Any]] = []
            source_chars = 0
            included_chars = 0
            for chunk in _original_source_chunks(item):
                text = str(chunk.get("text") or "")
                source_chars += len(text)
                if included_chars >= source_budget:
                    continue
                visible = text[: source_budget - included_chars]
                if not visible:
                    continue
                spans.append(
                    {
                        "chunk_id": str(chunk.get("chunk_id") or ""),
                        "start_char": 0,
                        "text": visible,
                        "truncated": len(visible) < len(text),
                    }
                )
                included_chars += len(visible)

            if not spans:
                continue
            remaining -= included_chars
            projected.append(
                {
                    "source_id": f"R{source_index}",
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "claim_ids": [
                        str(value)
                        for value in item.get("claim_ids") or []
                        if str(value).strip()
                    ],
                    "spans": spans,
                    "source_chars": source_chars,
                    "visible_chars": included_chars,
                    "truncated": included_chars < source_chars,
                }
            )

        return {
            "schema_version": "planner-evidence.v1",
            "source_count": len(all_sources),
            "visible_source_count": len(projected),
            "visible_chars": sum(item["visible_chars"] for item in projected),
            "truncated": len(projected) < len(all_sources)
            or any(bool(item.get("truncated")) for item in projected),
            "sources": projected,
        }

    def infrastructure_report(self, *, claim_ids: list[str] | None = None) -> dict[str, Any]:
        """Return unresolved retrieval/model failures, optionally by Claim."""

        requested = {str(value) for value in (claim_ids or []) if str(value).strip()}
        with self._lock:
            events = [
                dict(item)
                for item in self.infrastructure_events
                if not requested
                or not str(item.get("task_point_id") or "")
                or str(item.get("task_point_id") or "") in requested
            ]
        affected_claim_ids = sorted(
            {
                str(item.get("task_point_id") or "")
                for item in events
                if str(item.get("task_point_id") or "")
            }
        )
        return {
            "schema_version": "retrieval-infrastructure.v1",
            "complete": not events,
            "event_count": len(events),
            "affected_claim_ids": affected_claim_ids,
            "unresolved_chunk_count": sum(int(item.get("unresolved_chunk_count") or 0) for item in events),
            "degraded_page_count": sum(int(item.get("degraded_page_count") or 0) for item in events),
            "transport_error_count": sum(int(item.get("transport_error_count") or 0) for item in events),
            "recovered_chunk_count": sum(int(item.get("recovered_chunk_count") or 0) for item in events),
            "events": events[:24],
        }

    def routing_snapshot(self, *, max_sources: int = 8, max_queries: int = 8) -> dict[str, Any]:
        with self._lock:
            return {
                "round_count": len(self.query_history),
                "source_count": len(self.sources),
                "queries": [
                    {
                        "query": item.get("query", ""),
                        "status": item.get("status", ""),
                        "new_source_count": item.get("new_source_count", 0),
                    }
                    for item in self.query_history[-max_queries:]
                ],
                "sources": [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "chunk_count": item.get("chunk_count", 0),
                    }
                    for item in list(self.sources.values())[-max_sources:]
                ],
                "coverage": dict(self.coverage),
                "claim_ledger": self.claims.snapshot(max_spans_per_claim=0),
                "attempted_url_count": len(self.attempted_urls),
                "claim_source_counts": {
                    point_id: len(values)
                    for point_id, values in self.sources_by_claim.items()
                },
                "frozen_path_count": len(self.frozen_paths),
                "frozen_paths": [dict(item) for item in self.frozen_paths[-8:]],
                "replan_count": self.replan_count,
                "deterministic_result_count": len(self.deterministic_results),
                "retrieval_infrastructure": self.infrastructure_report(),
            }

    def freeze_path(
        self,
        query: str,
        *,
        action: str = "",
        arguments: dict[str, Any] | None = None,
        step: int = 0,
        reason: str = "",
    ) -> dict[str, Any]:
        """Freeze a retrieval route without discarding its evidence."""

        record = {
            "query": " ".join(str(query or "").split()).strip(),
            "action": str(action or ""),
            "arguments": dict(arguments or {}),
            "step": int(step or 0),
            "reason": str(reason or "")[:500],
        }
        with self._lock:
            self.frozen_paths.append(record)
        return dict(record)

    def record_replan(self) -> int:
        """Increment and return the task-scoped recovery count."""

        with self._lock:
            self.replan_count += 1
            return self.replan_count

    def reset(self) -> None:
        self.rounds.clear()
        self.rounds_by_point.clear()
        self.point_state.clear()
        self.last_discovery_results.clear()
        self.last_retrieval_failure = None
        self.sources.clear()
        self.sources_by_claim.clear()
        self.deterministic_results.clear()
        self.attempted_urls.clear()
        self.url_attempt_counts.clear()
        self.query_history.clear()
        self.coverage.clear()
        self.frozen_paths.clear()
        self.infrastructure_events.clear()
        self.replan_count = 0
        self.progress.reset()
        self.claims.reset()


@dataclass
class AgentState:
    task_id: str = ""
    task_output_dir: str = ""
    user_query: str = ""
    refined_query: str = ""
    id_to_path: Dict[str, str] = field(default_factory=dict)
    path_to_id: Dict[str, str] = field(default_factory=dict)
    working_memory: Dict[str, str] = field(default_factory=dict)
    memory_catalog: Dict[str, str] = field(default_factory=dict)
    last_feedback: str = ""
    entity_audit: Dict[str, str] = field(default_factory=dict)
    abandoned_file_ids: Set[str] = field(default_factory=set)
    is_finished: bool = False
    final_result: str = ""
    run_metadata: dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalEpisodeState = field(default_factory=RetrievalEpisodeState)

    def _mount_global_env(self) -> str:
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_query = self.refined_query or self.user_query
        env_lines = [
            "【挂载模块: 任务环境】",
            f"- 系统时间: {current_time_str}",
            f"- 当前执行目标: {active_query}",
        ]
        if self.entity_audit:
            env_lines.append("- 🎯 当前实体校验状态 (Entity Audit):")
            for entity, status in self.entity_audit.items():
                env_lines.append(f"  * {entity}: {status}")
        return "\n".join(env_lines)

    def _mount_memory_catalog(self) -> str:
        filtered_memory = {}
        for key, value in self.working_memory.items():
            if any(file_id in key for file_id in self.abandoned_file_ids):
                continue
            if not key.startswith(("__", "AbsPath_", "Path_", "Category_")):
                filtered_memory[key] = value

        if not filtered_memory:
            return "【挂载模块: 情报目录大纲】\n*(记忆区当前为空)*"

        memory_total_tokens = sum(get_token_count(str(value)) for value in filtered_memory.values())
        memory_details = [
            f"- `{key}`: [{self.memory_catalog.get(key, '已存储有效结构化数据')}]"
            for key in filtered_memory
        ]
        lines = [
            "【挂载模块: 情报目录大纲】",
            f"[系统状态] 当前可用知识库缓存已挂载（体积估算: {memory_total_tokens} Tokens）。",
            "【工作指引】: 请继续检查并收集其他缺漏情报；如果所有核心事实均已齐备，请立即调用 generate_final_aggregate_reports 进入最终聚合。",
            "\n以下是已获取的可用情报，请据此决定下一步：",
            *memory_details,
        ]
        return "\n".join(lines)

    def _mount_local_workspace(self) -> str:
        pending_preview = []
        pending_extract = []
        for file_id, path in self.id_to_path.items():
            if file_id in self.abandoned_file_ids or f"Summary_{file_id}" in self.memory_catalog:
                continue
            if f"Preview_{file_id}" in self.memory_catalog:
                pending_extract.append(
                    f"- {file_id}: {os.path.basename(path)} [已试读判定为相关，等待进行全文深度提炼]"
                )
            else:
                pending_preview.append(
                    f"- {file_id}: {os.path.basename(path)} [未读，可试读或直接全文提取]"
                )

        pending_items = pending_preview + pending_extract
        if not pending_items:
            return ""

        lines = [
            "【挂载模块: 本地工作区文件 (Local Workspace)】",
            "核心防幻觉红线：本地工作区中的文件可能是【完全独立、毫无关联】的，不要在无原文依据时捏造它们的合作关系！",
            f"📊 静态代码审计提醒：总共 {len(self.id_to_path)} 份文件中，仍有 {len(pending_items)} 份未被处理（系统防漏缺扫描）！",
            "💡 快捷通配符：如果缺口是需要处理大量未读文件，在调用工具的 file_ids 时可直接传入 [\"ALL\"]，底层引擎会自动将所有剩余未处理文件安全映射展开！",
            f"\n剩余清单 (共 {len(pending_items)} 项)：",
            *pending_items,
        ]
        return "\n".join(lines)

    def _mount_feedback(self) -> str:
        if not self.last_feedback:
            return ""
        return f"【挂载模块: 最新执行反馈】\n{self.last_feedback}"

    def to_markdown_context(self) -> str:
        modules = [
            self._mount_global_env(),
            self._mount_memory_catalog(),
            self._mount_local_workspace(),
            self._mount_feedback(),
        ]
        return "\n\n".join(module for module in modules if module)

    def to_retrieval_context(self) -> str:
        """Return persistent task state for the next RWKV tool decision.

        The legacy Markdown context describes the local workspace and its
        file-processing workflow.  Sending that context to an ordinary web
        retrieval turn creates a false "environment ready" signal and spends
        model tokens on an unrelated task.  File research still uses
        ``to_markdown_context``; the model-owned web loop gets this narrow
        routing view instead.
        """
        lines = ["Retrieval task state:", f"Task: {self.refined_query or self.user_query}"]
        if self.last_feedback:
            lines.append(f"Latest controller feedback: {self.last_feedback}")
        lines.append("Shared retrieval ledger (progress metadata; not a finish gate):")
        lines.append(json.dumps(self.retrieval.routing_snapshot(), ensure_ascii=False, separators=(",", ":")))
        lines.append(
            "Persistent original source spans (shared evidence for RWKV planning; no extractor paraphrases):"
        )
        lines.append(
            json.dumps(
                self.retrieval.planner_evidence_snapshot(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        if self.retrieval.deterministic_results:
            lines.append("Persistent deterministic tool results:")
            lines.append(
                json.dumps(
                    self.retrieval.deterministic_results[-16:],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        return "\n".join(lines)
