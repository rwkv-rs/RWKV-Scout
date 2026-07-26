# RWKV-ECRA/config.py
import os
import json
from contextvars import ContextVar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

# ==========================================
# 从 config.json 加载配置
# ==========================================
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    _cfg = json.load(f)

DEFAULT_LLM_PROVIDER = _cfg.get("LLM_PROVIDER", "volcengine")
API_KEYS = _cfg.get("API_KEYS", {})
LLM_ENDPOINTS = _cfg.get("LLM_ENDPOINTS", {})
SEARCH_CONFIG = _cfg.get("SEARCH_CONFIG", {})

DATA_PIPELINE = _cfg.get("DATA_PIPELINE", {})
# 将 json 中的相对路径 "./" 转换为基于当前项目路径的绝对路径
for key in ["input_directory", "output_directory", "checkpoint_directory", "debug_directory", "asset_directory"]:
    if key in DATA_PIPELINE and isinstance(DATA_PIPELINE[key], str) and DATA_PIPELINE[key].startswith("./"):
        DATA_PIPELINE[key] = os.path.join(BASE_DIR, DATA_PIPELINE[key][2:])

if "asset_directory" not in DATA_PIPELINE:
    DATA_PIPELINE["asset_directory"] = os.path.join(BASE_DIR, "data", "knowledge_assets")

AGENT_CONFIG = _cfg.get("AGENT_CONFIG", {})
LLM_CONFIG = _cfg.get("LLM_CONFIG", {})
SLM_CONFIG = _cfg.get("SLM_CONFIG", {})

TRACKING = _cfg.get("TRACKING", {})
if TRACKING.get("log_dir", "").startswith("./"):
    TRACKING["log_dir"] = os.path.join(BASE_DIR, TRACKING["log_dir"][2:])

# ==========================================
# 接口化临时配置覆盖区 (线程/协程安全)
# 允许 API 接口单次请求动态覆盖 JSON 中的默认值
# ==========================================
override_llm_key: ContextVar[str] = ContextVar("override_llm_key", default=None)
override_llm_url: ContextVar[str] = ContextVar("override_llm_url", default=None)
override_llm_provider: ContextVar[str] = ContextVar("override_llm_provider", default=None)
override_slm_endpoint: ContextVar[str] = ContextVar("override_slm_endpoint", default=None)
override_slm_password: ContextVar[str] = ContextVar("override_slm_password", default=None)
override_slm_async_enabled: ContextVar[bool] = ContextVar("override_slm_async_enabled", default=None)

def get_llm_provider() -> str:
    return override_llm_provider.get() or DEFAULT_LLM_PROVIDER

def get_llm_api_key() -> str:
    provider = get_llm_provider()
    return override_llm_key.get() or API_KEYS.get(provider, "")

def get_llm_base_url() -> str:
    provider = get_llm_provider()
    return override_llm_url.get() or LLM_ENDPOINTS.get(provider, {}).get("base_url", "")

def get_llm_model() -> str:
    provider = get_llm_provider()
    return LLM_ENDPOINTS.get(provider, {}).get("model", "")

def is_local_provider(provider: str | None = None) -> bool:
    """Return whether a provider points at one of the local RWKV servers."""
    return (provider or get_llm_provider()).startswith("local")

def get_model_profiles() -> list[dict]:
    """Expose only the local model choices that the frontend may select."""
    profiles = []
    for key, value in LLM_ENDPOINTS.items():
        if not isinstance(value, dict) or not value.get("ui_visible"):
            continue
        profiles.append({
            "key": key,
            "label": value.get("label", key),
            "base_url": value.get("base_url", ""),
            "model": value.get("model", ""),
            "context_length": value.get("context_length", 10240),
        })
    return profiles

def get_model_profile(key: str | None = None) -> dict:
    selected = key or get_llm_provider()
    profile = LLM_ENDPOINTS.get(selected)
    if not isinstance(profile, dict) or not profile.get("ui_visible"):
        raise ValueError(f"Unknown local model profile: {selected}")
    return profile

def get_slm_endpoint() -> str:
    return override_slm_endpoint.get() or SLM_CONFIG.get("endpoint", "")

def get_slm_protocol() -> str:
    return str(SLM_CONFIG.get("protocol", "rwkv_lightning"))

def get_slm_password() -> str:
    return override_slm_password.get() or SLM_CONFIG.get("password", "")

def get_slm_concurrency() -> int:
    return max(1, int(SLM_CONFIG.get("concurrency", 16)))

def get_slm_async_parallelism() -> int:
    return max(1, int(SLM_CONFIG.get("async_parallelism", 1)))

def get_slm_async_batch_wait_ms() -> int:
    return max(0, int(SLM_CONFIG.get("async_batch_wait_ms", 20)))

def get_slm_async_enabled() -> bool:
    override_value = override_slm_async_enabled.get()
    if override_value is not None:
        return bool(override_value)
    return bool(SLM_CONFIG.get("enable_async_parallel", False))

def get_llm_concurrency() -> int:
    return max(1, int(LLM_CONFIG.get("concurrency", 6)))
