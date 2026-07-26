# RWKV-ECRA/agent/orchestrator.py
import os
import json
import traceback
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Set

from agent.analyzer import Analyzer
from agent.planner import Planner
from agent.retrieval_synthesis import synthesize_retrieval_answer
from agent.retrieval_loop import (
    compact_observation,
    execute_parallel_candidates,
    generate_query_candidates,
    merge_retrieval_results,
)
from utils.tracker import EventTracker
from clients.slm_client import SLMClient
from config import TRACKING, DATA_PIPELINE, get_slm_async_batch_wait_ms, get_slm_async_enabled, get_slm_async_parallelism, get_slm_concurrency
from tools.registry import ToolRegistry
from utils.chunker import get_token_count
from utils.token_tracker import global_token_tracker
from utils.task_manager import is_task_stopped, update_task_progress
from utils.task_events import append_task_event
from workflows.report_flow import generate_final_aggregate_reports
import tools.static_ops 
import tools.web_search
import tools.weather
import tools.chat
import tools.paper_search
import tools.web_search_keyless
import workflows.map_reduce_flow
import workflows.memory_query_flow
import workflows.report_flow


class _SLMInputQueueItem:
    def __init__(self, request_id: str, task_id: str, index: int, content: str, tracker, endpoint: str, password: str):
        self.request_id = request_id
        self.task_id = task_id
        self.index = index
        self.content = content
        self.tracker = tracker
        self.endpoint = endpoint
        self.password = password
        self.result = ""
        self.error = None
        self.done = threading.Event()

    @property
    def backend_key(self):
        return (self.endpoint, self.password)


class SLMInputScheduler:
    """上层 SLM 输入队列：只调度原始 prompt 列表，调用方继续按 index 解析结果。"""

    def __init__(self):
        self._queue = deque()
        self._condition = threading.Condition()
        self._worker_started = False
        self._active_batches = 0

    def submit(self, contents: list[str], tracker=None, task_id: str = "") -> list[str]:
        if not contents:
            return []

        if not get_slm_async_enabled():
            return SLMClient().batch_generate(contents, tracker=tracker, task_id=task_id)

        request_id = uuid.uuid4().hex
        client = SLMClient()
        items = [
            _SLMInputQueueItem(request_id, task_id or "UNKNOWN_TASK", idx, content, tracker, client.endpoint, client.password)
            for idx, content in enumerate(contents)
        ]

        with self._condition:
            self._ensure_worker_locked()
            self._queue.extend(items)
            self._condition.notify()

        for item in items:
            item.done.wait()
            if item.error:
                raise item.error

        return [item.result for item in items]

    def _ensure_worker_locked(self):
        if self._worker_started:
            return
        worker = threading.Thread(target=self._run, name="SLMInputScheduler", daemon=True)
        worker.start()
        self._worker_started = True

    def _run(self):
        while True:
            batch = self._take_batch()
            worker = threading.Thread(target=self._process_batch, args=(batch,), name="SLMInputBatch", daemon=True)
            worker.start()

    def _take_batch(self):
        with self._condition:
            while not self._queue or self._active_batches >= get_slm_async_parallelism():
                self._condition.wait()

            max_batch = get_slm_concurrency()
            first = self._queue.popleft()
            backend_key = first.backend_key
            batch = [first]

            wait_until = time.monotonic() + (get_slm_async_batch_wait_ms() / 1000.0)
            while len(batch) < max_batch:
                scan_idx = 0
                matched = False
                while len(batch) < max_batch and scan_idx < len(self._queue):
                    candidate = self._queue[scan_idx]
                    if candidate.backend_key == backend_key:
                        batch.append(candidate)
                        del self._queue[scan_idx]
                        matched = True
                    else:
                        scan_idx += 1

                if len(batch) >= max_batch:
                    break

                remaining = wait_until - time.monotonic()
                if remaining <= 0:
                    break

                if not matched:
                    self._condition.wait(timeout=remaining)
                    if self._active_batches >= get_slm_async_parallelism():
                        break

            self._active_batches += 1
            return batch

    def _process_batch(self, batch):
        try:
            print(f"[SLM 输入队列] 发射 {len(batch)} 个片段 | 首任务: {batch[0].task_id}")
            
            # 1. 🚀 发送前计费：统计批次中每个片段的 Input Token
            for item in batch:
                in_toks = get_token_count(item.content)
                global_token_tracker.add_slm(in_toks, 0, task_id=item.task_id)

            # 调用 direct 底层方法绕开默认的总额计费，避免重复
            client = SLMClient(endpoint_override=batch[0].endpoint, password_override=batch[0].password)
            results = client._batch_generate_direct([item.content for item in batch])
            
            for item, result in zip(batch, results):
                item.result = result
                
                # 2. 📥 返回后计费：统计真实生成的 Output Token
                out_toks = get_token_count(result)
                global_token_tracker.add_slm(0, out_toks, task_id=item.task_id)

                if item.tracker:
                    item.tracker.track_slm(input_prompt=item.content, output_text=result, task_id=item.task_id)
        except Exception as exc:
            for item in batch:
                item.error = exc
        finally:
            for item in batch:
                item.done.set()
            with self._condition:
                self._active_batches = max(0, self._active_batches - 1)
                self._condition.notify_all()


GLOBAL_SLM_INPUT_SCHEDULER = SLMInputScheduler()

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

    def _mount_global_env(self) -> str:
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_query = self.refined_query if self.refined_query else self.user_query
        
        env_lines = [
            "【挂载模块: 任务环境】",
            f"- 系统时间: {current_time_str}", 
            f"- 当前执行目标: {active_query}"
        ]
        
        if self.entity_audit:
            env_lines.append("- 🎯 当前实体校验状态 (Entity Audit):")
            for ent, status in self.entity_audit.items():
                env_lines.append(f"  * {ent}: {status}")
                
        return "\n".join(env_lines)

    def _mount_memory_catalog(self) -> str:
        filtered_mem = {}
        for k, v in self.working_memory.items():
            if any(fid in k for fid in self.abandoned_file_ids):
                continue
            if not k.startswith("__") and not k.startswith("AbsPath_") and not k.startswith("Path_") and not k.startswith("Category_"):
                filtered_mem[k] = v

        if not filtered_mem:
            return "【挂载模块: 情报目录大纲】\n*(记忆区当前为空)*"
            
        memory_total_tokens = sum(get_token_count(str(v)) for v in filtered_mem.values())
        memory_details = []
        
        for k in filtered_mem.keys():
            desc = self.memory_catalog.get(k, "已存储有效结构化数据")
            memory_details.append(f"- `{k}`: [{desc}]")
                
        lines = ["【挂载模块: 情报目录大纲】"]
        lines.append(f"[系统状态] 当前可用知识库缓存已挂载（体积估算: {memory_total_tokens} Tokens）。")
        lines.append("【工作指引】: 请继续检查并收集其他缺漏情报；如果所有核心事实均已齐备，请立即调用 generate_final_aggregate_reports 进入最终聚合。")
        lines.append("\n以下是已获取的可用情报，请据此决定下一步：")
        lines.extend(memory_details)
        return "\n".join(lines)

    def _mount_local_workspace(self) -> str:
        pending_preview = []
        pending_extract = []
        for fid, path in self.id_to_path.items():
            if fid in self.abandoned_file_ids:
                continue 
            if f"Summary_{fid}" in self.memory_catalog:
                continue 
            
            if f"Preview_{fid}" in self.memory_catalog:
                pending_extract.append(f"- {fid}: {os.path.basename(path)} [已试读判定为相关，等待进行全文深度提炼]")
            else:
                pending_preview.append(f"- {fid}: {os.path.basename(path)} [未读，可试读或直接全文提取]")
                
        pending_items = pending_preview + pending_extract
        if not pending_items:
            return ""
            
        lines = [
            "【挂载模块: 本地工作区文件 (Local Workspace)】", 
            "核心防幻觉红线：本地工作区中的文件可能是【完全独立、毫无关联】的，不要在无原文依据时捏造它们的合作关系！"
        ]
        
        lines.append(f"📊 静态代码审计提醒：总共 {len(self.id_to_path)} 份文件中，仍有 {len(pending_items)} 份未被处理（系统防漏缺扫描）！")
        lines.append("💡 快捷通配符：如果缺口是需要处理大量未读文件，在调用工具的 file_ids 时可直接传入 [\"ALL\"]，底层引擎会自动将所有剩余未处理文件安全映射展开！")
        
        lines.append(f"\n剩余清单 (共 {len(pending_items)} 项)：")
        lines.extend(pending_items)
            
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
            self._mount_feedback()
        ]
        return "\n\n".join(m for m in modules if m)


class Orchestrator:
    def __init__(self):
        self.tracker = EventTracker(log_dir=TRACKING.get("log_dir", "./logs"), enable=TRACKING.get("enable", True))
        self.state = AgentState()
        self.state.working_memory["__category_tree__"] = {} 
        self.analyzer = Analyzer()
        self.planner = Planner()

    def _retrieval_context(self) -> dict:
        return {
            "original_goal": self.state.user_query,
            "path_to_id": self.state.path_to_id,
            "id_to_path": self.state.id_to_path,
            "working_memory": self.state.working_memory,
            "tracker": self.tracker,
            "agent_state": None,
            "task_id": self.state.task_id,
            "slm_scheduler": GLOBAL_SLM_INPUT_SCHEDULER,
        }

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
        plan = generate_query_candidates(
            self.analyzer.llm,
            user_query,
            action=action,
            scope=str(args.get("scope") or ""),
            observation=observation,
            previous_query=previous_query,
            max_candidates=3 if not previous_query else 2,
        )
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
            )
        merged = merge_retrieval_results(
            user_query,
            action,
            rows,
            scope=str(args.get("scope") or ""),
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
            },
        )
        return merged, candidates, plan

    def _finish_rwkv_retrieval(self, user_query: str, action: str, data: dict, step: int) -> str:
        append_task_event(
            self.state.task_id,
            "synthesis_start",
            step=step,
            phase="SYNTHESIS",
            action=action,
            evidence_count=len(data.get("results") or []),
            round_count=data.get("round_count", 1),
        )
        synthesis = synthesize_retrieval_answer(user_query, data, llm=self.analyzer.llm)
        self.state.is_finished = True
        self.state.final_result = synthesis.get("content") or ""
        append_task_event(
            self.state.task_id,
            "synthesis",
            step=step,
            phase="SYNTHESIS",
            content=self.state.final_result,
            mode=synthesis.get("mode"),
            evidence_count=synthesis.get("evidence_count", 0),
            citation_refs=synthesis.get("citation_refs") or [],
        )
        append_task_event(
            self.state.task_id,
            "final",
            status="completed",
            content=self.state.final_result,
            action=action,
            mode=synthesis.get("mode"),
            round_count=data.get("round_count", 1),
        )
        report_path = os.path.join(self.state.task_output_dir, "retrieval_report.jsonl")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "record_type": "retrieval_result",
                "task_id": self.state.task_id,
                "query": user_query,
                "action": action,
                "real_network": bool(data.get("real_network", True)),
                "answer": self.state.final_result,
                "answer_mode": synthesis.get("mode"),
                "data": data,
            }, ensure_ascii=False) + "\n")
        return self.state.final_result

    def run(self, user_query: str, task_id: str = None) -> str:
        self.state.task_id = task_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state.task_output_dir = os.path.join(DATA_PIPELINE.get("output_directory", "./data/output"), self.state.task_id)
        os.makedirs(self.state.task_output_dir, exist_ok=True)
        
        self.tracker.track("User_Input", input_data=user_query, output_data=None)
        self.state.user_query = user_query
        append_task_event(self.state.task_id, "user_input", content=user_query)
        
        debug_dir = DATA_PIPELINE.get("debug_directory", "./data/debug_slm")
        os.makedirs(debug_dir, exist_ok=True)
        session_id = self.state.task_id
        trace_file = os.path.join(debug_dir, f"DeepResearch_Trace_{session_id}.md")
        
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(f"# Deep Research 执行追踪日志\n\n**启动时间**: {session_id}\n**用户指令**: {user_query}\n\n---\n\n")

        PHASE_MAP = {
            "DISCOVERY": "探测与发现",
            "EXTRACTION": "深度提取",
            "SYNTHESIS": "聚合适成"
        }
        ACTION_MAP = {
            "search_local_file": "检索本地工作区文件",
            "preview_document_content": "试读文档摘要",
            "delegate_to_small_models": "调度小模型提炼全文",
            "query_checkpoint_via_slm": "执行记忆区细节捞针",
            "batch_process_individual_reports": "归档单篇独立报告",
            "compress_working_memory": "执行工作记忆压缩",
            "generate_final_aggregate_reports": "排版聚合最终研报",
            "execute_web_search": "执行互联网检索",
            "search_papers": "检索论文数据源",
            "finish_task": "任务逻辑闭环退出",
            "none": "思考下一步方向"
        }

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

        push_progress("🚀 正在初始化环境，构建工作区内存与检索本地文件...")

        try:
            initial_files_json = ToolRegistry.execute("search_local_file", {"keyword": ""}, {})
            initial_files = json.loads(initial_files_json)
            for i, p in enumerate(initial_files):
                fid = f"DOC_{i+1}"
                self.state.id_to_path[fid] = p
                self.state.path_to_id[p] = fid
            self.state.last_feedback = f"系统就绪，目录中发现 {len(initial_files)} 份可用文件。"
            push_progress(f"环境就绪：感知到 {len(initial_files)} 份文件。\n")
        except Exception:
            self.state.last_feedback = "目录为空。"
            push_progress(f"环境就绪：本地工作区目录为空。\n")
            
        # Scholarly and live-web requests have a deterministic, auditable
        # retrieval path.  Keep the local RWKV analysis as a trace signal, but
        # do not make a network retrieval wait on the legacy file-research
        # loop or on a second free-form planner generation.
        direct_plan = self.planner.plan_next_action(user_query, {}, "", "DISCOVERY")
        if direct_plan.get("action") == "multi_hop_research":
            first_action = str((direct_plan.get("args") or {}).get("first_action") or "search_web_keyless")
            scope = str((direct_plan.get("args") or {}).get("scope") or "paper")
            append_task_event(
                self.state.task_id,
                "plan",
                step=1,
                phase="DISCOVERY",
                action="multi_hop_research",
                args={"first_action": first_action, "scope": scope},
                router=direct_plan.get("router", "static_multi_hop_cue"),
            )
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=1,
                phase="DISCOVERY",
                query=user_query,
                router_hint=direct_plan.get("router", "static_multi_hop_cue"),
            )
            first_args = {
                "scope": scope,
                "max_results": 8,
            } if first_action == "search_papers" else {
                "max_results": 6,
                "fetch_pages": 3,
            }
            first_data, first_queries, _ = self._run_rwkv_search_round(
                user_query,
                first_action,
                first_args,
                step=1,
                round_name="initial",
            )
            followup_action = (
                "search_web_keyless"
                if any(term in user_query.lower() for term in ["\u673a\u6784", "institution", "\u5b98\u65b9\u6570\u636e", "\u4ea4\u96c6"])
                else first_action
            )
            followup_args = {
                "scope": scope,
                "max_results": 8,
            } if followup_action == "search_papers" else {
                "max_results": 6,
                "fetch_pages": 3,
            }
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=2,
                phase="DISCOVERY",
                query=user_query,
                router_hint="rwkv_followup_from_initial_evidence",
            )
            second_data, second_queries, _ = self._run_rwkv_search_round(
                user_query,
                followup_action,
                followup_args,
                step=2,
                round_name="followup",
                previous_query=first_queries[0] if first_queries else user_query,
                observation=first_data,
            )
            merged = merge_retrieval_results(
                user_query,
                followup_action,
                [
                    (first_queries[0] if first_queries else user_query, first_data),
                    (second_queries[0] if second_queries else user_query, second_data),
                ],
                scope=scope,
            )
            merged["round_count"] = 2
            merged["candidate_queries"] = [*first_queries, *second_queries]
            append_task_event(
                self.state.task_id,
                "multi_hop_merge",
                step=2,
                phase="DISCOVERY",
                data={
                    "round_count": 2,
                    "first_queries": first_queries,
                    "second_queries": second_queries,
                    "result_count": merged.get("count", 0),
                },
            )
            return self._finish_rwkv_retrieval(user_query, followup_action, merged, 3)

        if direct_plan.get("action") in {"search_papers", "search_web_keyless"}:
            action = direct_plan["action"]
            args = dict(direct_plan.get("args") or {})
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=1,
                phase="DISCOVERY",
                query=user_query,
                router_hint=direct_plan.get("router", "local_rwkv_candidate_search"),
            )
            append_task_event(
                self.state.task_id,
                "plan",
                step=1,
                phase="DISCOVERY",
                action=action,
                args=args,
                router=direct_plan.get("router", "local_rwkv_candidate_search"),
            )
            data, _, _ = self._run_rwkv_search_round(
                user_query,
                action,
                args,
                step=1,
                round_name="initial",
            )
            return self._finish_rwkv_retrieval(user_query, action, data, 2)

        # Removed legacy one-shot retrieval branch.  Search actions are handled
        # only by the RWKV candidate/multi-hop path above.
        if direct_plan.get("action") == "__legacy_removed__":
            step_count = 1
            context_text = self.state.to_markdown_context()
            try:
                analysis = self.analyzer.analyze_intent_and_phase(user_query, context_text)
            except Exception as exc:  # The deterministic retrieval route remains usable.
                analysis = {
                    "intent_mode": "DEEP_RESEARCH",
                    "entity_audit": {},
                    "refined_query": user_query,
                    "missing_information": f"local RWKV analysis unavailable: {exc}",
                    "next_phase": "DISCOVERY",
                }
            analysis["execution_route"] = "direct_keyless_retrieval"
            analysis["local_model"] = "rwkv7-g1h-1.5b-20260710-ctx10240"
            append_task_event(
                self.state.task_id,
                "analysis",
                step=step_count,
                phase="DISCOVERY",
                context_snapshot=context_text,
                data=analysis,
            )
            self.state.refined_query = user_query

            action = direct_plan["action"]
            args = dict(direct_plan.get("args") or {})
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=step_count,
                phase="DISCOVERY",
                query=user_query,
                router_hint=direct_plan.get("router", "deterministic_retrieval"),
            )
            append_task_event(
                self.state.task_id,
                "plan",
                step=step_count,
                phase="DISCOVERY",
                action=action,
                args=args,
                router=direct_plan.get("router", "deterministic_retrieval"),
            )
            append_task_event(
                self.state.task_id,
                "tool_call",
                step=step_count,
                phase="DISCOVERY",
                action=action,
                args=args,
            )
            env_context = {
                "original_goal": user_query,
                "path_to_id": self.state.path_to_id,
                "id_to_path": self.state.id_to_path,
                "working_memory": self.state.working_memory,
                "tracker": self.tracker,
                "agent_state": self.state,
                "task_id": self.state.task_id,
                "slm_scheduler": GLOBAL_SLM_INPUT_SCHEDULER,
            }
            started_at = time.perf_counter()
            result = ToolRegistry.execute(action, args=args, context=env_context)
            try:
                structured_result = json.loads(result)
            except (TypeError, json.JSONDecodeError):
                structured_result = {"raw": result}
            self.state.last_feedback = f"[{action}] retrieval result:\n{result}"
            append_task_event(
                self.state.task_id,
                "tool_result",
                step=step_count,
                phase="DISCOVERY",
                action=action,
                result=result,
                real_network=bool(structured_result.get("real_network", True)),
            )

            append_task_event(
                self.state.task_id,
                "synthesis_start",
                step=step_count + 1,
                phase="SYNTHESIS",
                action=action,
                evidence_count=len(structured_result.get("results") or []),
            )
            synthesis = synthesize_retrieval_answer(
                user_query,
                structured_result,
                llm=self.analyzer.llm,
            )
            duration_ms = round((time.perf_counter() - started_at) * 1000, 1)
            append_task_event(
                self.state.task_id,
                "synthesis",
                step=step_count + 1,
                phase="SYNTHESIS",
                content=synthesis.get("content", ""),
                mode=synthesis.get("mode"),
                evidence_count=synthesis.get("evidence_count", 0),
                duration_ms=duration_ms,
                citation_refs=synthesis.get("citation_refs") or [],
            )
            self.state.is_finished = True
            self.state.final_result = synthesis.get("content") or result
            append_task_event(
                self.state.task_id,
                "final",
                status="completed",
                content=self.state.final_result,
                action=action,
                duration_ms=duration_ms,
            )
            report_path = os.path.join(self.state.task_output_dir, "retrieval_report.jsonl")
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "record_type": "retrieval_result",
                            "task_id": self.state.task_id,
                            "query": user_query,
                            "action": action,
                            "router": direct_plan.get("router", "deterministic_retrieval"),
                            "real_network": bool(structured_result.get("real_network", True)),
                            "answer": self.state.final_result,
                            "answer_mode": synthesis.get("mode"),
                            "duration_ms": duration_ms,
                            "data": structured_result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return self.state.final_result

        # Closed-world prompts should not enter the legacy analyzer/planner
        # loop. That loop can spend minutes retrying a local-model request and
        # may hallucinate a different tool (for example weather for a
        # translation request). Keep the step visible in the trace while
        # executing exactly one deterministic/local action.
        if direct_plan.get("action") in {"answer_user", "get_current_weather"}:
            step_count = 1
            action = direct_plan["action"]
            args = dict(direct_plan.get("args") or {})
            context_text = self.state.to_markdown_context()
            append_task_event(
                self.state.task_id,
                "analysis",
                step=step_count,
                phase="DIRECT",
                context_snapshot=context_text,
                data={
                    "execution_route": "direct_closed_world_action",
                    "intent_mode": "NO_SEARCH",
                    "refined_query": user_query,
                    "next_phase": "DIRECT",
                },
            )
            append_task_event(
                self.state.task_id,
                "planner_start",
                step=step_count,
                phase="DIRECT",
                query=user_query,
                router_hint=direct_plan.get("router", "deterministic_no_search"),
            )
            append_task_event(
                self.state.task_id,
                "plan",
                step=step_count,
                phase="DIRECT",
                action=action,
                args=args,
                router=direct_plan.get("router", "deterministic_no_search"),
            )
            append_task_event(
                self.state.task_id,
                "tool_call",
                step=step_count,
                phase="DIRECT",
                action=action,
                args=args,
            )
            env_context = {
                "original_goal": user_query,
                "path_to_id": self.state.path_to_id,
                "id_to_path": self.state.id_to_path,
                "working_memory": self.state.working_memory,
                "tracker": self.tracker,
                "agent_state": self.state,
                "task_id": self.state.task_id,
                "slm_scheduler": GLOBAL_SLM_INPUT_SCHEDULER,
            }
            started_at = time.perf_counter()
            result = ToolRegistry.execute(action, args=args, context=env_context)
            duration_ms = round((time.perf_counter() - started_at) * 1000, 1)
            append_task_event(
                self.state.task_id,
                "tool_result",
                step=step_count,
                phase="DIRECT",
                action=action,
                result=result,
                real_network=action == "get_current_weather",
            )
            self.state.is_finished = True
            self.state.final_result = self.state.final_result or result
            append_task_event(
                self.state.task_id,
                "final",
                status="completed",
                content=self.state.final_result,
                action=action,
                duration_ms=duration_ms,
            )
            return self.state.final_result

        # The former Analyzer -> Planner -> retry loop is intentionally no
        # longer part of the runtime. Unknown work is stopped safely instead
        # of allowing a small model to invent a tool or loop over local files.
        self.state.is_finished = True
        self.state.final_result = "当前请求未命中受支持的静态路由，未执行旧式循环，也未编造答案。"
        append_task_event(
            self.state.task_id,
            "analysis",
            step=1,
            phase="ROUTING",
            data={
                "execution_route": "unsupported_safe_stop",
                "intent_mode": "UNSUPPORTED",
                "refined_query": user_query,
                "next_phase": "STOP",
            },
        )
        append_task_event(
            self.state.task_id,
            "final",
            status="completed",
            content=self.state.final_result,
            action="safe_stop",
        )
        return self.state.final_result
