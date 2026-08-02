"""State and context projection for one research task."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Set

from utils.chunker import get_token_count


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
        return "\n".join(lines)
