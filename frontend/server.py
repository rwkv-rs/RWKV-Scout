# RWKV-ECRA/frontend/server.py
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FRONTEND_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "output"
API_BASE_URL = os.environ.get("RWKV_ECRA_API_BASE", "http://127.0.0.1:8787")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

def file_time(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "-"

def find_report_files(base: Path) -> tuple[Path | None, Path | None]:
    jsonl_candidates = sorted(
        [p for p in base.glob("*.jsonl") if "结构化" in p.name or "_03_" in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    md_candidates = sorted(
        [p for p in base.glob("*.md") if "深度排版" in p.name or "_02_" in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not md_candidates:
        md_candidates = sorted(base.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return (jsonl_candidates[0] if jsonl_candidates else None, md_candidates[0] if md_candidates else None)

def collect_history() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items: dict[str, dict[str, Any]] = {}

    task_records = read_jsonl(OUTPUT_DIR / "tasks.jsonl")

    # ✨ 核心修复：先按 task_id 将所有增量日志（Deltas）进行深度合并还原全量状态
    merged_records = {}
    for record in task_records:
        # 🛡️ 防护 1：过滤掉非字典的脏数据（防止 record.get 报错）
        if not isinstance(record, dict):
            continue

        task_id = str(record.get("id") or record.get("task_id") or "").strip()
        if not task_id:
            continue

        if task_id not in merged_records:
            merged_records[task_id] = {}
        merged_records[task_id].update(record)

    for task_id, record in merged_records.items():
        try:
            # 🛡️ 防护 2：安全处理路径组合，捕获非法路径字符导致的 OSError
            result_dir_val = record.get("result_dir")
            result_dir = Path(result_dir_val) if result_dir_val else (OUTPUT_DIR / task_id)

            if not result_dir.is_absolute():
                result_dir = PROJECT_DIR / result_dir

            try:
                dir_exists = result_dir.exists()
            except OSError:
                dir_exists = False

            jsonl_path, md_path = find_report_files(result_dir) if dir_exists else (None, None)
        except Exception as e:
            print(f"[警告] 解析任务 {task_id} 的路径时出错: {e}")
            jsonl_path, md_path = None, None

        # ✨ 提取动态生成的 step_X 字典键为有序数组
        steps = []
        for k, v in record.items():
            if k.startswith("step_") and k[5:].isdigit():
                steps.append((int(k[5:]), v))
        steps.sort(key=lambda x: x[0])
        steps_list = [s[1] for s in steps]

        items[task_id] = {
            "id": task_id,
            "title": task_id,
            "query": record.get("query") or "",
            "status": record.get("status") or "ready",
            "progress": record.get("progress") or "",
            "steps": steps_list,
            "updated_at": record.get("timestamp") or file_time(jsonl_path or md_path or OUTPUT_DIR),
            "queued_at": record.get("queued_at") or "",
            "start_time": record.get("start_time") or "",
            "end_time": record.get("end_time") or "",
            "path": str(result_dir) if 'result_dir' in locals() else "",
            "has_structured_report": bool(jsonl_path),
        }

    for child in sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        jsonl_path, md_path = find_report_files(child)
        if not jsonl_path and not md_path:
            continue
        items.setdefault(
            child.name,
            {
                "id": child.name,
                "title": child.name,
                "query": "",
                "status": "ready",
                "progress": "",
                "steps": [],
                "updated_at": file_time(jsonl_path or md_path or child),
                "queued_at": "",
                "path": str(child),
                "has_structured_report": bool(jsonl_path),
            },
        )

    return sorted(items.values(), key=lambda item: item.get("updated_at") or "", reverse=True)

def resolve_report_paths(report_id: str) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    history = collect_history()
    match = next((item for item in history if item["id"] == report_id), None)
    if not match:
        return None, None, None
    base = Path(match["path"])
    jsonl_path, md_path = find_report_files(base)
    return jsonl_path, md_path, match

def structured_report(report_id: str) -> dict[str, Any]:
    jsonl_path, md_path, meta = resolve_report_paths(report_id)
    if not meta:
        raise FileNotFoundError(report_id)

    sources: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    markdown = ""

    if jsonl_path:
        for record in read_jsonl(jsonl_path):
            record_type = record.get("record_type")
            if record_type == "global_citation_map":
                sources = record.get("data") or []
            elif record_type == "report_node":
                nodes.append(
                    {
                        "id": record.get("node_id") or f"node_{len(nodes) + 1}",
                        "title": record.get("title") or f"章节 {len(nodes) + 1}",
                        "content": record.get("content") or "",
                        "sources": record.get("sources") or [],
                    }
                )
            elif record_type == "final_beautified_markdown":
                markdown = record.get("content") or markdown

    if not nodes and md_path:
        markdown = md_path.read_text(encoding="utf-8", errors="replace")

    return {
        "id": report_id,
        "title": meta.get("title") or report_id,
        "status": meta.get("status") or "ready",
        "updated_at": meta.get("updated_at") or "-",
        "kind": "structured" if nodes else "markdown",
        "sources": sources,
        "nodes": nodes,
        "markdown": markdown,
    }

def proxy_api(path: str, query: str, method: str, body: bytes | None = None) -> tuple[int, bytes, str]:
    full_path = f"{path}?{query}" if query else path
    target = urllib.parse.urljoin(API_BASE_URL, full_path)
    request = urllib.request.Request(
        target,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get_content_type()
    except OSError as exc:
        payload = json.dumps({"code": 502, "message": f"API proxy failed: {exc}"}, ensure_ascii=False).encode("utf-8")
        return 502, payload, "application/json"

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[%s] %s\n" % (datetime.now().strftime("%H:%M:%S"), fmt % args))

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/frontend-api/history":
            self.send_json({"code": 200, "data": collect_history()})
            return

        if path == "/frontend-api/tokens":
            token_file = OUTPUT_DIR / "global_token_usage.json"
            if token_file.exists():
                try:
                    data = json.loads(token_file.read_text(encoding="utf-8"))
                    self.send_json({"code": 200, "data": data})
                except Exception as e:
                    self.send_json({"code": 500, "message": str(e)}, status=500)
            else:
                self.send_json({"code": 200, "data": {"tasks": {}}})
            return

        if path.startswith("/frontend-api/report/"):
            report_id = urllib.parse.unquote(path.rsplit("/", 1)[-1])
            try:
                self.send_json({"code": 200, "data": structured_report(report_id)})
            except FileNotFoundError:
                self.send_json({"code": 404, "message": "report not found"}, status=404)
            return

        if path.startswith("/frontend-api/"):
            status, payload, content_type = proxy_api(parsed.path, parsed.query, "GET", None)
            self.send_response(status)
            self.send_header("Content-Type", content_type or "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.serve_static(path)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/frontend-api/"):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b"{}"

            content_type = self.headers.get("Content-Type", "application/json")
            full_path = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
            target = urllib.parse.urljoin(API_BASE_URL, full_path)

            request = urllib.request.Request(target, data=body, method="POST", headers={"Content-Type": content_type})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    status, payload, ctype = response.status, response.read(), response.headers.get_content_type()
            except urllib.error.HTTPError as exc:
                status, payload, ctype = exc.code, exc.read(), exc.headers.get_content_type()
            except OSError as exc:
                payload = json.dumps({"code": 502, "message": f"API proxy failed: {exc}"}, ensure_ascii=False).encode("utf-8")
                status, ctype = 502, "application/json"

            self.send_response(status)
            self.send_header("Content-Type", ctype or "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_json({"code": 404, "message": "not found"}, status=404)

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/frontend-api/"):
            status, payload, content_type = proxy_api(parsed.path, parsed.query, "DELETE", None)
            self.send_response(status)
            self.send_header("Content-Type", content_type or "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_json({"code": 404, "message": "not found"}, status=404)

    def serve_static(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"
        relative = Path(urllib.parse.unquote(path.lstrip("/")))
        target = (FRONTEND_DIR / relative).resolve()
        if FRONTEND_DIR not in target.parents and target != FRONTEND_DIR:
            self.send_error(403)
            return
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        raw = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

def main() -> None:
    parser = argparse.ArgumentParser(description="Serve RWKV-ECRA frontend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5177, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RWKV-ECRA frontend: http://{args.host}:{args.port}")
    print(f"Project data output: {OUTPUT_DIR}")
    print(f"API proxy target: {API_BASE_URL}")
    server.serve_forever()

if __name__ == "__main__":
    main()