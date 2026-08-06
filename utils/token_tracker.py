# RWKV-ECRA/utils/token_tracker.py
import os
import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from config import DATA_PIPELINE

current_task_id: ContextVar[str] = ContextVar("current_task_id", default="UNKNOWN_TASK")
current_model_lane: ContextVar[str] = ContextVar("current_model_lane", default="control")


@contextmanager
def model_lane(lane: str):
    """Attach a scheduling lane to model calls made in this context."""
    token = current_model_lane.set(str(lane or "control").casefold())
    try:
        yield
    finally:
        current_model_lane.reset(token)

class GlobalTokenTracker:
    def __init__(self):
        self.lock = threading.Lock()
        output_dir = DATA_PIPELINE.get("output_directory", "./data/output")
        os.makedirs(output_dir, exist_ok=True)
        self.file_path = os.path.join(output_dir, "global_token_usage.json")

        # 极简的数据结构
        self.stats = {
            "global": {
                "rwkv_slm": {"input_tokens": 0, "output_tokens": 0, "execution_time_sec": 0.0},
                "cloud_llm": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "execution_time_sec": 0.0}
            },
            "tasks": {}
        }

        # 极简的“发送/接收”计时器
        self._active_states = {
            "global": {"slm_count": 0, "slm_start": 0.0, "llm_count": 0, "llm_start": 0.0},
            "tasks": {}
        }
        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "global" in data: self.stats = data
            except Exception: pass

    def _save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception: pass

    def _init_task_locked(self, task_id: str):
        if task_id not in self.stats["tasks"]:
            self.stats["tasks"][task_id] = {
                "rwkv_slm": {"input_tokens": 0, "output_tokens": 0, "execution_time_sec": 0.0},
                "cloud_llm": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "execution_time_sec": 0.0}
            }
        if task_id not in self._active_states["tasks"]:
            self._active_states["tasks"][task_id] = {"slm_count": 0, "slm_start": 0.0, "llm_count": 0, "llm_start": 0.0}

    # 1. 指令发送时调用
    def start_timer(self, model_type: str, task_id: str = None):
        tid = task_id or current_task_id.get()
        now = time.time()
        with self.lock:
            # 全局计时：如果是第一个并发请求，记录起跑时间
            g_state = self._active_states["global"]
            if g_state[f"{model_type}_count"] == 0: g_state[f"{model_type}_start"] = now
            g_state[f"{model_type}_count"] += 1

            # 独立任务计时
            if tid and tid != "UNKNOWN_TASK":
                self._init_task_locked(tid)
                t_state = self._active_states["tasks"][tid]
                if t_state[f"{model_type}_count"] == 0: t_state[f"{model_type}_start"] = now
                t_state[f"{model_type}_count"] += 1

    # 2. 接收回复时调用
    def stop_timer(self, model_type: str, task_id: str = None):
        tid = task_id or current_task_id.get()
        now = time.time()
        key = "cloud_llm" if model_type == "llm" else "rwkv_slm"
        with self.lock:
            g_state = self._active_states["global"]
            if g_state[f"{model_type}_count"] > 0:
                g_state[f"{model_type}_count"] -= 1
                # 如果最后一个并发请求回来了，结算总时间
                if g_state[f"{model_type}_count"] == 0:
                    self.stats["global"][key]["execution_time_sec"] += (now - g_state[f"{model_type}_start"])

            if tid and tid != "UNKNOWN_TASK":
                t_state = self._active_states["tasks"].get(tid)
                if t_state and t_state[f"{model_type}_count"] > 0:
                    t_state[f"{model_type}_count"] -= 1
                    if t_state[f"{model_type}_count"] == 0:
                        self.stats["tasks"][tid][key]["execution_time_sec"] += (now - t_state[f"{model_type}_start"])
            self._save()

    def add_slm(self, input_tokens: int, output_tokens: int, task_id: str = None):
        tid = task_id or current_task_id.get()
        with self.lock:
            self.stats["global"]["rwkv_slm"]["input_tokens"] += input_tokens
            self.stats["global"]["rwkv_slm"]["output_tokens"] += output_tokens
            if tid and tid != "UNKNOWN_TASK":
                self._init_task_locked(tid)
                self.stats["tasks"][tid]["rwkv_slm"]["input_tokens"] += input_tokens
                self.stats["tasks"][tid]["rwkv_slm"]["output_tokens"] += output_tokens
            self._save()

    def add_llm(self, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0, task_id: str = None):
        tid = task_id or current_task_id.get()
        with self.lock:
            self.stats["global"]["cloud_llm"]["input_tokens"] += input_tokens
            self.stats["global"]["cloud_llm"]["output_tokens"] += output_tokens
            self.stats["global"]["cloud_llm"]["reasoning_tokens"] += reasoning_tokens
            if tid and tid != "UNKNOWN_TASK":
                self._init_task_locked(tid)
                self.stats["tasks"][tid]["cloud_llm"]["input_tokens"] += input_tokens
                self.stats["tasks"][tid]["cloud_llm"]["output_tokens"] += output_tokens
                self.stats["tasks"][tid]["cloud_llm"]["reasoning_tokens"] += reasoning_tokens
            self._save()

    def get_stats(self, task_id: str = None):
        with self.lock: return self.stats["tasks"].get(task_id, {}) if task_id else self.stats

    def reset(self):
        with self.lock:
            self.__init__()

global_token_tracker = GlobalTokenTracker()
