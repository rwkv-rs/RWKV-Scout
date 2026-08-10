# RWKV-Scout/utils/slm_metrics.py
import json
import os
import threading
import time
from collections import deque
from datetime import datetime

from config import TRACKING
from utils.chunker import get_token_count

class SLMThroughputMeter:
    def __init__(self):
        self.window_seconds = 15 * 60
        self._events = deque()
        self._lock = threading.Lock()

        log_dir = TRACKING.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        self.events_file = os.path.join(log_dir, "slm_throughput_events.jsonl")
        self.history_file = os.path.join(log_dir, "slm_throughput_15min.jsonl")
        self.latest_snapshot_file = os.path.join(log_dir, "slm_throughput_15min_latest.json")
        self.snapshot_file = self.latest_snapshot_file

    def record(self, input_prompt: str, output_text: str, task_id: str = ""):
        now = time.time()
        input_tokens = get_token_count(input_prompt or "")
        output_tokens = get_token_count(output_text or "")

        event = {
            "timestamp": datetime.fromtimestamp(now).isoformat(),
            "epoch": now,
            "task_id": task_id or "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        with self._lock:
            self._events.append(event)
            self._prune_locked(now)
            snapshot = self._build_snapshot_locked(now)
            self._append_jsonl(self.events_file, event)
            self._append_jsonl(self.history_file, snapshot)
            self._write_json(self.snapshot_file, snapshot)

    def _prune_locked(self, now: float):
        cutoff = now - self.window_seconds
        while self._events and self._events[0]["epoch"] < cutoff:
            self._events.popleft()

    def _build_snapshot_locked(self, now: float) -> dict:
        input_total = sum(item["input_tokens"] for item in self._events)
        output_total = sum(item["output_tokens"] for item in self._events)
        event_count = len(self._events)
        elapsed_seconds = self.window_seconds
        if event_count:
            elapsed_seconds = max(1.0, min(self.window_seconds, now - self._events[0]["epoch"]))

        return {
            "updated_at": datetime.fromtimestamp(now).isoformat(),
            "window_seconds": self.window_seconds,
            "event_count": event_count,
            "input_tokens_15min": input_total,
            "output_tokens_15min": output_total,
            "input_tokens_per_minute": input_total / (elapsed_seconds / 60.0),
            "output_tokens_per_minute": output_total / (elapsed_seconds / 60.0),
            "events_file": self.events_file,
            "history_file": self.history_file,
        }

    def _append_jsonl(self, path: str, payload: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_json(self, path: str, payload: dict):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                
            # Windows 环境下，os.replace 极易因杀软或前端读取产生文件锁
            # 加入短时重试机制，避开毫秒级的共享冲突
            import time
            for _ in range(5):
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    time.sleep(0.05)
        except Exception as e:

            print(f"[Metrics Warning] 吞吐量统计文件写入失败，已被安全忽略: {e}")


slm_throughput_meter = SLMThroughputMeter()