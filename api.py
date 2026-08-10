# RWKV-Scout/api.py
import os
import sys
import uvicorn
import uuid
import requests
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, HTTPException, Request
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
import config
from app.public_access import public_frontend_request_allowed, public_mode_enabled
from runtime import get_model_backend
from app.models import AnalyzeRequest, ChatRequest
from app.services.task_runner import run_background_analysis
from app.services.workspace_files import (
    WorkspaceFileError,
    WorkspacePathError,
    cleanup_directories,
    delete_workspace_file,
    list_workspace_files,
    read_task_report,
    read_workspace_file,
    save_uploaded_file,
)
from utils.task_manager import record_task, get_all_tasks, request_stop, delete_task
from utils.token_tracker import global_token_tracker
from utils.task_events import get_task_events
from utils.experiment_manifest import reconstruct_run
from utils.operational_metrics import collect_operational_metrics, prometheus_text
from main import setup_env


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

setup_env()

app = FastAPI(title="RWKV-Scout Agent API", description="支持前端隔离请求、文件上传与历史回溯")


@app.middleware("http")
async def restrict_public_frontend_api(request: Request, call_next):
    path = request.url.path
    if (
        public_mode_enabled()
        and path.startswith("/frontend-api/")
        and not public_frontend_request_allowed(request.method, path)
    ):
        return JSONResponse(
            status_code=403,
            content={"code": 403, "message": "This endpoint is disabled in public mode."},
        )
    return await call_next(request)

# =====================================
@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "rwkv-ecra", "trace_schema": "rwkv-ecra.run.v1"}


def _probe_model_service() -> dict:
    """Probe the configured local model runtime without exposing credentials."""
    if not config.is_local_provider():
        return {"available": False, "reason": "configured_provider_is_not_local"}
    if config.get_model_backend_name() in {"direct_rwkv", "auto"}:
        backend = get_model_backend()
        if backend.backend_name == "direct_rwkv":
            health = dict(backend.health())
            configured_model = config.get_llm_model()
            health["configured_model"] = configured_model
            health["model_match"] = bool(health.get("model_match", True)) and (not configured_model or health.get("model") == configured_model)
            return health
    endpoint = config.get_llm_base_url().rstrip("/") + "/models"
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(
            endpoint,
            headers={"Authorization": f"Bearer {config.get_llm_api_key()}"},
            timeout=(min(config.get_model_connect_timeout_seconds(), 2.0), 2.0),
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        model_ids = [str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)]
        expected = config.get_llm_model()
        return {
            "available": True,
            "status_code": response.status_code,
            "model_match": not model_ids or expected in model_ids,
            "configured_model": expected,
            "served_models": model_ids[:8],
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}"[:300],
            "configured_model": config.get_llm_model(),
        }


@app.get("/readyz")
def readyz():
    required_paths = {
        "input_directory": config.DATA_PIPELINE.get("input_directory"),
        "output_directory": config.DATA_PIPELINE.get("output_directory"),
    }
    writable = all(path and os.path.isdir(path) and os.access(path, os.W_OK) for path in required_paths.values())
    model_service = _probe_model_service()
    payload = {
        "status": "ready" if writable and model_service.get("available") and model_service.get("model_match", True) else "not_ready",
        "model": config.get_experiment_model_config(),
        "model_service": model_service,
        "paths": {name: bool(value and os.path.isdir(value)) for name, value in required_paths.items()},
        "wigolo_mode": config.get_wigolo_mode(),
        "runtime": {
            "max_parallel_cases": config.get_experiment_max_parallel_cases(),
            "max_model_inflight_requests": config.get_model_request_concurrency(),
            "analysis_timeout_seconds": config.get_analysis_timeout_seconds(),
        },
    }
    payload["model"].pop("api_key", None)
    return payload


# 2. 基础系统接口 (上传与清理)
# =====================================
@app.post("/api/v1/upload")
@app.post("/frontend-api/upload")
async def upload_files(files: list[UploadFile] = File(...), paths: list[str] | None = Form(None)):
    input_dir = config.DATA_PIPELINE["input_directory"]
    saved_files = []

    for i, file in enumerate(files):
        relative_path = paths[i] if paths and i < len(paths) else (file.filename or "")
        try:
            saved_files.append(save_uploaded_file(input_dir, relative_path, file.file))
        except (WorkspacePathError, WorkspaceFileError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"code": 200, "message": "上传成功", "data": {"saved": saved_files}}

@app.post("/api/v1/cleanup")
@app.post("/frontend-api/cleanup")
def cleanup_environment():
    dirs_to_clean = [
        config.DATA_PIPELINE["input_directory"],
        config.DATA_PIPELINE["checkpoint_directory"],
        config.DATA_PIPELINE.get("debug_directory", "./data/debug_slm"),
        config.DATA_PIPELINE.get("asset_directory", "./data/knowledge_assets")
    ]
    errors = cleanup_directories(dirs_to_clean)
    global_token_tracker.reset()
    response = {"code": 200, "message": "运行环境及缓存已重置"}
    if errors:
        response["warnings"] = errors
    return response

# =====================================
# ✨ 新增：Token 账本双重查询接口
# =====================================
@app.get("/api/v1/metrics/tokens")
@app.get("/frontend-api/metrics/tokens")
def get_global_token_metrics():
    """实时获取系统 SLM 和 LLM 的全局 Tokens 总开销和所有历史任务数据"""
    return {
        "code": 200,
        "message": "success",
        "data": global_token_tracker.get_stats()
    }

@app.get("/api/v1/metrics/tokens/{task_id}")
@app.get("/frontend-api/metrics/tokens/{task_id}")
def get_task_token_metrics(task_id: str):
    """精准获取某一个历史任务的 Token 开销"""
    return {
        "code": 200,
        "message": "success",
        "data": global_token_tracker.get_stats(task_id)
    }

# =====================================
# 3. 文件夹感知的文件管理 API 
# =====================================
@app.get("/api/v1/files")
@app.get("/frontend-api/files")
def list_input_files():
    return {
        "code": 200,
        "data": list_workspace_files(config.DATA_PIPELINE["input_directory"]),
    }

@app.delete("/api/v1/files")
@app.delete("/frontend-api/files")
def delete_input_file(path: str):
    input_dir = config.DATA_PIPELINE["input_directory"]
    try:
        deleted = delete_workspace_file(input_dir, path)
    except WorkspacePathError:
        return {"code": 403, "message": "非法路径"}
    except WorkspaceFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if deleted:
        return {"code": 200, "message": f"{path} 已删除"}
    return {"code": 404, "message": "文件不存在"}

@app.get("/api/v1/files/content")
@app.get("/frontend-api/files/content")
def get_input_file(path: str):
    input_dir = config.DATA_PIPELINE["input_directory"]
    try:
        workspace_file = read_workspace_file(input_dir, path)
    except WorkspacePathError:
        return PlainTextResponse("Invalid path", status_code=403)
    except WorkspaceFileError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    if workspace_file is not None:
        if workspace_file.is_image:
            return FileResponse(workspace_file.path)
        return PlainTextResponse(workspace_file.text or "")
    return PlainTextResponse("File not found", status_code=404)

@app.get("/frontend-api/config")
def get_frontend_config():
    public_mode = public_mode_enabled()
    model_profiles = config.get_model_profiles()
    if public_mode:
        model_profiles = [
            {key: value for key, value in profile.items() if key != "base_url"}
            for profile in model_profiles
        ]
    return {
        "code": 200,
        "data": {
            "default_model": config.DEFAULT_LLM_PROVIDER,
            "models": model_profiles,
            "public_mode": public_mode,
            "slm_async_enabled": config.get_slm_async_enabled(),
            "slm_concurrency": config.get_slm_concurrency(),
            "slm_async_parallelism": config.get_slm_async_parallelism(),
            "llm_concurrency": config.get_llm_concurrency(),
            "max_parallel_cases": config.get_experiment_max_parallel_cases(),
            "max_model_inflight_requests": config.get_model_request_concurrency(),
            "analysis_timeout_seconds": config.get_analysis_timeout_seconds(),
        }
    }

@app.post("/frontend-api/chat")
def chat_endpoint(req: ChatRequest):
    """Direct local-model chat for interactive frontend testing."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    model_key = req.model_key or config.DEFAULT_LLM_PROVIDER
    provider_token = config.override_llm_provider.set(model_key)
    backend_token = None
    direct_token = None
    url_token = None
    slm_token = None
    try:
        profile = config.get_model_profile(model_key)
        backend_token = config.override_model_backend.set(profile.get("runtime_backend")) if profile.get("runtime_backend") else None
        direct_token = config.override_direct_rwkv_config.set(profile.get("direct_runtime")) if profile.get("direct_runtime") is not None else None
        base_url = str(profile.get("base_url") or "")
        url_token = config.override_llm_url.set(base_url)
        slm_token = config.override_slm_endpoint.set(base_url.rstrip("/") + "/chat/completions") if base_url else None
        from clients.llm_client import LLMClient

        message = LLMClient().chat_completion(
            messages=req.messages,
            max_tokens=max(1, min(int(req.max_tokens), 8192)),
        )
        return {
            "code": 200,
            "data": {
                "model_key": model_key,
                "model": profile["model"],
                "message": {
                    "role": getattr(message, "role", "assistant"),
                    "content": getattr(message, "content", "") or "",
                },
            },
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"local model request failed: {exc}") from exc
    finally:
        if slm_token is not None:
            config.override_slm_endpoint.reset(slm_token)
        if url_token is not None:
            config.override_llm_url.reset(url_token)
        if direct_token is not None:
            config.override_direct_rwkv_config.reset(direct_token)
        if backend_token is not None:
            config.override_model_backend.reset(backend_token)
        config.override_llm_provider.reset(provider_token)


@app.get("/api/v1/metrics/operational")
@app.get("/frontend-api/metrics/operational")
def get_operational_metrics():
    """Return aggregate metrics derived from persisted traces only."""
    return {"code": 200, "data": collect_operational_metrics(get_all_tasks())}


@app.get("/metrics")
def get_prometheus_metrics():
    """Expose a secret-free Prometheus-compatible view for local monitoring."""
    return PlainTextResponse(
        prometheus_text(collect_operational_metrics(get_all_tasks())),
        media_type="text/plain; version=0.0.4",
    )


@app.post("/api/v1/analyze")
@app.post("/frontend-api/analyze")
def analyze_endpoint(req: AnalyzeRequest, bg_tasks: BackgroundTasks):
    if public_mode_enabled():
        # Public browsers may submit work, but concurrency/sampling execution
        # and model-service connection details remain operator-owned policies.
        req = req.model_copy(
            update={
                "llm_api_key": None,
                "llm_base_url": None,
                "llm_provider": None,
                "slm_endpoint": None,
                "slm_password": None,
                "slm_async_enabled": None,
            }
        )
    task_id = f"TASK_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    task_output_dir = os.path.join(config.DATA_PIPELINE["output_directory"], task_id)
    
    record_task(task_id, req.query, "running", task_output_dir, queued_at=req.queued_at)
    bg_tasks.add_task(run_background_analysis, task_id, req, task_output_dir)
    
    return {
        "code": 200,
        "status": "success",
        "task_id": task_id,
        "message": "分析任务已提交后台执行"
    }

# =====================================
# 5. 数据回溯及运维控制接口
# =====================================
@app.get("/frontend-api/history")
def get_task_history():
    enriched_tasks = []
    for item in get_all_tasks():
        enriched = dict(item)
        task_id = str(enriched.get("task_id") or enriched.get("id") or "")
        raw_query = str(enriched.get("query") or enriched.get("title") or "").strip()
        # Most live tasks already carry their query and status in the task
        # index. Only filesystem-discovered/acceptance tasks need the event
        # stream scan to recover the real input and final status. This keeps
        # the history endpoint cheap while preserving the detailed trace.
        needs_trace_metadata = not raw_query or raw_query == task_id or raw_query.startswith("JSON_ACCEPTANCE_")
        events = get_task_events(task_id) if task_id and needs_trace_metadata else []
        user_inputs = [
            event.get("content")
            for event in events
            if event.get("type") == "user_input" and str(event.get("content") or "").strip()
        ]
        if user_inputs:
            # Acceptance runs and filesystem-discovered tasks may have been
            # indexed with their internal id as the query. Prefer the actual
            # user input persisted in the auditable event stream.
            enriched["query"] = str(user_inputs[0]).strip()
            enriched["title"] = str(user_inputs[0]).strip()
        enriched["event_count"] = len(events)
        enriched_tasks.append(enriched)
    return {"code": 200, "data": enriched_tasks}

@app.get("/frontend-api/history/{task_id}/events")
def get_task_events_endpoint(task_id: str, after: int = 0):
    events = get_task_events(task_id, max(0, after))
    public_events = []
    for event in events:
        public_events.append(event)
    next_seq = public_events[-1]["seq"] if public_events else max(0, after)
    return {"code": 200, "data": {"events": public_events, "next_seq": next_seq}}

@app.post("/api/v1/analyze/{task_id}/stop")
@app.post("/frontend-api/analyze/{task_id}/stop")
@app.post("/frontend-api/history/{task_id}/stop")
def stop_task_endpoint(task_id: str):
    if not task_id or task_id == "undefined":
        return {"code": 400, "message": "无效的任务 ID"}
    request_stop(task_id)
    return {"code": 200, "message": f"任务 {task_id} 已发出停止指令"}

@app.delete("/api/v1/history/{task_id}")
@app.delete("/frontend-api/history/{task_id}")
def delete_task_endpoint(task_id: str):
    if not task_id or task_id == "undefined":
        return {"code": 400, "message": "无效的任务 ID"}
    delete_task(task_id)
    return {"code": 200, "message": f"任务 {task_id} 及其数据已被物理删除"}



@app.get("/frontend-api/history/{task_id}/report")
def get_task_report(task_id: str):
    if not task_id or task_id == "undefined":
        return {"code": 404, "message": "无有效的任务 ID 供查询"}
    try:
        report_data = read_task_report(config.DATA_PIPELINE["output_directory"], task_id)
    except WorkspacePathError:
        return {"code": 403, "message": "非法任务路径"}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"报告读取失败: {exc}") from exc
    if report_data is None:
        return {"code": 404, "message": "报告文件夹在物理系统上已丢失或被移除"}
    if report_data:
        return {"code": 200, "data": report_data}
    return {"code": 404, "message": "系统已扫描文件夹，但未发现任何 JSONL 或 MD 格式的报告"}

@app.get("/frontend-api/history/{task_id}/trace")
def get_task_trace(task_id: str):
    """Return the replayable manifest and normalized execution trace."""
    if not task_id or task_id == "undefined":
        return {"code": 404, "message": "无效的任务 ID"}
    try:
        trace = reconstruct_run(task_id, config.DATA_PIPELINE["output_directory"])
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    manifest_file = os.path.join(config.DATA_PIPELINE["output_directory"], task_id, "run_manifest.json")
    if not trace.get("events") and not os.path.exists(manifest_file):
        return {"code": 404, "message": "任务轨迹不存在"}
    return {"code": 200, "data": trace}


def main():
    uvicorn.run(
        "api:app",
        host=config.get_api_host(),
        port=config.get_api_port(),
        workers=config.get_api_workers(),
    )


if __name__ == "__main__":
    host = config.get_api_host()
    port = config.get_api_port()
    workers = config.get_api_workers()
    print(f"[系统] API 服务启动中... 监听: http://{host}:{port} workers={workers}")
    uvicorn.run("api:app", host=host, port=port, workers=workers)
