# RWKV-ECRA/utils/task_manager.py
import os
import json
import threading
from datetime import datetime
from config import DATA_PIPELINE

TASK_LOG_FILE = os.path.join(DATA_PIPELINE.get("output_directory", "./data/output"), "tasks.jsonl")

class TaskStore:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._task_index = {}
        self._ordered_keys = []
        self._load()
        self._sync_with_filesystem()

    def _load(self):
        if not os.path.exists(self.filepath): return
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    tid = record.get("task_id")
                    if tid:
                        if tid not in self._task_index:
                            self._ordered_keys.append(tid)
                            self._task_index[tid] = record
                        else:
                            self._task_index[tid].update(record)
                        
                        if self._task_index[tid].get('status') == 'deleted':
                            if tid in self._ordered_keys: self._ordered_keys.remove(tid)
                            del self._task_index[tid]
                except: continue

    def _sync_with_filesystem(self):
        output_dir = DATA_PIPELINE.get("output_directory", "./data/output")
        if not os.path.exists(output_dir): return

        changed = False
        with self._lock:
            # 1. 剔除死链
            dead_tasks = []
            for tid in self._ordered_keys:
                expected_dir = os.path.join(output_dir, tid)
                record_dir = self._task_index[tid].get("result_dir", "")
                if not os.path.exists(expected_dir) and not os.path.exists(record_dir):
                    dead_tasks.append(tid)
            for tid in dead_tasks:
                self._ordered_keys.remove(tid)
                del self._task_index[tid]
                changed = True

            # 2. 补录遗漏
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if not os.path.isdir(item_path) or item in self._task_index: continue
                
                has_valid_file = False
                target_file = None
                
                # 递归遍历找文件 (免疫特殊字符路径)
                for root_dir, _, files in os.walk(item_path):
                    for f in files:
                        if f.endswith(".jsonl") or f.endswith(".md"):
                            has_valid_file = True
                            target_file = os.path.join(root_dir, f)
                            break
                    if has_valid_file: break
                
                if has_valid_file:
                    try:
                        mtime = os.path.getmtime(target_file)
                        time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    self._ordered_keys.append(item)
                    self._task_index[item] = {
                        "task_id": item,
                        "timestamp": time_str,
                        "query": item, 
                        "status": "completed",
                        "result_dir": item_path,
                        "error": "",
                        "queued_at": time_str
                    }
                    changed = True
                    
            if changed:
                self._ordered_keys.sort(key=lambda k: self._task_index[k].get("timestamp", ""))
                os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
                with open(self.filepath, "w", encoding="utf-8") as f:
                    for k in self._ordered_keys:
                        f.write(json.dumps(self._task_index[k], ensure_ascii=False) + "\n")

    def record_task(self, task_id: str, query: str, status: str, result_dir: str = "", error: str = "", queued_at: str = None, acceptance_case_id: str = None):
        with self._lock:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            record = {
                "task_id": task_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "status": status,
                "result_dir": result_dir,
                "error": error
            }
            if queued_at: record["queued_at"] = queued_at
            if acceptance_case_id: record["acceptance_case_id"] = acceptance_case_id
                
            if task_id not in self._task_index:
                self._ordered_keys.append(task_id)
                self._task_index[task_id] = record
            else:
                if "queued_at" not in record and "queued_at" in self._task_index[task_id]:
                    record["queued_at"] = self._task_index[task_id]["queued_at"]
                if "acceptance_case_id" not in record and "acceptance_case_id" in self._task_index[task_id]:
                    record["acceptance_case_id"] = self._task_index[task_id]["acceptance_case_id"]
                self._task_index[task_id].update(record)
            
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._task_index[task_id], ensure_ascii=False) + "\n")

    def update_task_progress(self, task_id: str, progress: str):
        with self._lock:
            if task_id in self._task_index:
                self._task_index[task_id]['progress'] = progress
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self._task_index[task_id], ensure_ascii=False) + "\n")

    def request_stop(self, task_id: str):
        with self._lock:
            if task_id in self._task_index and self._task_index[task_id]['status'] == 'running':
                self._task_index[task_id]['status'] = 'stopped'
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self._task_index[task_id], ensure_ascii=False) + "\n")

    def delete_task(self, task_id: str):
        with self._lock:
            if task_id in self._task_index:
                self._task_index[task_id]['status'] = 'deleted'
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self._task_index[task_id], ensure_ascii=False) + "\n")
                
                task_dir = self._task_index[task_id].get("result_dir")
                if task_dir and os.path.exists(task_dir):
                    import shutil
                    try: shutil.rmtree(task_dir)
                    except: pass
                
                if task_id in self._ordered_keys: self._ordered_keys.remove(task_id)
                del self._task_index[task_id]

    def is_task_stopped(self, task_id: str) -> bool:
        with self._lock:
            task = self._task_index.get(task_id)
            if not task: return True
            return task.get('status') in ('stopped', 'deleted')

    def get_all_tasks(self) -> list:
        with self._lock: return [self._task_index[k] for k in reversed(self._ordered_keys)]

_store = None
def _get_store() -> TaskStore:
    global _store
    if _store is None: _store = TaskStore(TASK_LOG_FILE)
    return _store

def init_task_file(): _get_store()
def record_task(task_id: str, query: str, status: str, result_dir: str = "", error: str = "", queued_at: str = None, acceptance_case_id: str = None): _get_store().record_task(task_id, query, status, result_dir, error, queued_at, acceptance_case_id)
def update_task_progress(task_id: str, progress: str): _get_store().update_task_progress(task_id, progress)
def request_stop(task_id: str): _get_store().request_stop(task_id)
def delete_task(task_id: str): _get_store().delete_task(task_id)
def is_task_stopped(task_id: str) -> bool: return _get_store().is_task_stopped(task_id)
def get_all_tasks() -> list: return _get_store().get_all_tasks()
