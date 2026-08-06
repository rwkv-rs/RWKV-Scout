"""State and context projection for one research task."""

from __future__ import annotations

import os
import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Set

from utils.chunker import get_token_count


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
    query_history: list[dict[str, Any]] = field(default_factory=list)
    coverage: Dict[str, dict[str, Any]] = field(default_factory=dict)
    frozen_paths: list[dict[str, Any]] = field(default_factory=list)
    replan_count: int = 0
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

    def record_query(self, query: str, result: dict[str, Any], *, step: int = 0) -> dict[str, Any]:
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
                }
            )
            for item in result.get("results") or []:
                if not isinstance(item, dict):
                    continue
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
            self.query_history[-1]["new_source_count"] = len(added)
            self.rounds.append((query_text, result))
            self.last_discovery_results[:] = [
                item for item in result.get("results") or [] if isinstance(item, dict)
            ]
        return {"new_source_keys": added, "new_source_count": len(added), "total_sources": len(self.sources)}

    def source_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self.sources.values()]

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
                "frozen_path_count": len(self.frozen_paths),
                "replan_count": self.replan_count,
            }

    def freeze_path(self, query: str, *, step: int = 0, reason: str = "") -> dict[str, Any]:
        """Freeze a retrieval route without discarding its evidence."""

        record = {
            "query": " ".join(str(query or "").split()).strip(),
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
        self.query_history.clear()
        self.coverage.clear()
        self.frozen_paths.clear()
        self.replan_count = 0


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
        """Return only state that can affect the next web-tool decision.

        The legacy Markdown context describes the local workspace and its
        file-processing workflow.  Sending that context to an ordinary web
        retrieval turn creates a false "environment ready" signal and spends
        model tokens on an unrelated task.  File research still uses
        ``to_markdown_context``; the model-owned web loop gets this narrow
        routing view instead.
        """
        lines = [
            "Retrieval task state (routing metadata only; not evidence):",
            f"Task: {self.refined_query or self.user_query}",
        ]
        if self.last_feedback:
            lines.append(f"Latest controller feedback: {self.last_feedback}")
        lines.append(
            "Shared research state (routing metadata only; source bodies remain in the evidence store):"
        )
        lines.append(json.dumps(self.retrieval.routing_snapshot(), ensure_ascii=False, separators=(",", ":")))
        return "\n".join(lines)
