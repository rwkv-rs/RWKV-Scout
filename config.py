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
WIGOLO_CONFIG = _cfg.get("WIGOLO", {})

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
EXPERIMENT_CONFIG = _cfg.get("EXPERIMENT", {})
RUNTIME_CONFIG = _cfg.get("RUNTIME", {})
MODEL_RUNTIME_CONFIG = _cfg.get("MODEL_RUNTIME", {})
SERVER_CONFIG = _cfg.get("SERVER", {})

MODEL_CONTRACTS = {
    "local_7b": {
        "label": "RWKV 7.2B",
        "model": "rwkv7-g1h-7.2b-20260710-ctx10240",
        "endpoint": "http://127.0.0.1:29572/v1",
        "api_key": "rwkv-skills",
        "context_length": 10240,
        "provider": "local_7b",
    },
    "local_13b": {
        "label": "RWKV 13.3B",
        "model": "rwkv7-g1i_preview4922-13.3b-20260720-ctx12288",
        "endpoint": "http://172.21.122.93:29613/v1",
        "api_key": "rwkv-skills",
        "context_length": 12288,
        "provider": "local_13b",
    },
    "local_direct_1p5b": {
        "label": "RWKV 1.5B (Direct)",
        "model": "rwkv7-g1h-1.5b-20260710-ctx10240",
        "endpoint": "",
        "api_key": "",
        "context_length": 10240,
        "provider": "local_direct_1p5b",
    },
}

# The active contract follows EXPERIMENT.model_profile.  The 7.2B contract is
# retained as an explicit historical/baseline profile, but it must not be mixed
# with 13.3B results in a single promotion comparison.
EXPERIMENT_MODEL_CONTRACT = MODEL_CONTRACTS.get(
    EXPERIMENT_CONFIG.get("model_profile", "local_13b"),
    MODEL_CONTRACTS["local_13b"],
)

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
override_model_backend: ContextVar[str] = ContextVar("override_model_backend", default=None)
override_direct_rwkv_config: ContextVar[dict | None] = ContextVar("override_direct_rwkv_config", default=None)
override_slm_endpoint: ContextVar[str] = ContextVar("override_slm_endpoint", default=None)
override_slm_password: ContextVar[str] = ContextVar("override_slm_password", default=None)
override_slm_async_enabled: ContextVar[bool] = ContextVar("override_slm_async_enabled", default=None)

def get_llm_provider() -> str:
    return override_llm_provider.get() or DEFAULT_LLM_PROVIDER


def _normalize_model_backend(value: object) -> str:
    value = str(value or "openai_compat").strip().lower()
    if value not in {"direct_rwkv", "openai_compat", "auto"}:
        raise ValueError(f"Unknown model backend: {value}")
    return value


def get_model_backend_name() -> str:
    """Return the model execution backend, independent of provider protocol."""
    value = override_model_backend.get()
    if value is None:
        value = os.environ.get(
            "RWKV_ECRA_MODEL_BACKEND",
            MODEL_RUNTIME_CONFIG.get("backend", "openai_compat"),
        )
    return _normalize_model_backend(value)


def get_direct_rwkv_config() -> dict:
    """Return direct-runtime settings with environment overrides applied."""
    configured = MODEL_RUNTIME_CONFIG.get("direct_rwkv", {})
    settings = dict(configured) if isinstance(configured, dict) else {}
    request_override = override_direct_rwkv_config.get()
    if isinstance(request_override, dict):
        settings.update(request_override)
    env_names = {
        "engine_root": "RWKV_ECRA_RWKV_ENGINE_ROOT",
        "model_path": "RWKV_ECRA_RWKV_MODEL_PATH",
        "vocab_path": "RWKV_ECRA_RWKV_VOCAB_PATH",
        "device": "RWKV_ECRA_RWKV_DEVICE",
    }
    for key, env_name in env_names.items():
        if os.environ.get(env_name) is not None:
            settings[key] = os.environ[env_name]

    for key in ("engine_root", "model_path", "vocab_path"):
        value = settings.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value) and value.startswith((".", "..")):
            settings[key] = os.path.abspath(os.path.join(BASE_DIR, value))
    return settings

def get_llm_api_key() -> str:
    provider = get_llm_provider()
    return override_llm_key.get() or API_KEYS.get(provider, "")


def get_search_api_key(service: str) -> str:
    """Read a search-service key from the process environment first."""
    name = str(service or "").strip().upper()
    if not name:
        return ""
    return os.environ.get(f"{name}_API_KEY", "") or API_KEYS.get(service, "")

def get_llm_base_url() -> str:
    provider = get_llm_provider()
    return override_llm_url.get() or LLM_ENDPOINTS.get(provider, {}).get("base_url", "")

def get_llm_model() -> str:
    provider = get_llm_provider()
    return LLM_ENDPOINTS.get(provider, {}).get("model", "")


def get_llm_context_length() -> int:
    provider = get_llm_provider()
    try:
        return max(1, int(LLM_ENDPOINTS.get(provider, {}).get("context_length", 10240)))
    except (TypeError, ValueError):
        return 10240


def get_experiment_model_config(profile_key: str | None = None) -> dict:
    """Return the single model contract used by reproducible experiments."""
    selected = profile_key or EXPERIMENT_CONFIG.get("model_profile") or DEFAULT_LLM_PROVIDER
    profile = LLM_ENDPOINTS.get(selected, {})
    if not isinstance(profile, dict):
        raise ValueError(f"Unknown experiment model profile: {selected}")
    return {
        "label": profile.get("label", selected),
        "model": profile.get("model", ""),
        "endpoint": profile.get("base_url", ""),
        "api_key": API_KEYS.get(selected, ""),
        "context_length": profile.get("context_length", 10240),
        "provider": selected,
    }


def validate_experiment_model_contract(profile_key: str | None = None) -> dict:
    """Fail closed unless the selected model profile matches its pinned contract."""
    selected = profile_key or EXPERIMENT_CONFIG.get("model_profile") or DEFAULT_LLM_PROVIDER
    actual = get_experiment_model_config(selected)
    expected_contract = MODEL_CONTRACTS.get(selected, EXPERIMENT_MODEL_CONTRACT)
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected_contract.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"experiment model contract mismatch: {mismatches}")
    return actual


def get_prompt_version() -> str:
    return str(os.environ.get("RWKV_ECRA_PROMPT_VERSION", EXPERIMENT_CONFIG.get("prompt_version", "unversioned")))


def get_citation_remote_validation() -> bool:
    value = os.environ.get("RWKV_ECRA_CITATION_REMOTE_VALIDATION")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(EXPERIMENT_CONFIG.get("citation_remote_validation", False))

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
            "runtime_backend": value.get("runtime_backend", ""),
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


def get_experiment_max_parallel_cases() -> int:
    """Maximum long-running analysis tasks per API process."""
    return max(1, int(EXPERIMENT_CONFIG.get("max_parallel_cases", 1)))


def _bounded_seconds(value: object, default: float, *, minimum: float = 0.1, maximum: float = 3600.0) -> float:
    try:
        return max(minimum, min(float(value), maximum))
    except (TypeError, ValueError):
        return default


def get_analysis_timeout_seconds() -> float:
    """Maximum wall-clock budget for one long-running analysis task."""
    return _bounded_seconds(RUNTIME_CONFIG.get("analysis_timeout_seconds", 600), 600.0, maximum=3600.0)


def get_model_connect_timeout_seconds() -> float:
    return _bounded_seconds(RUNTIME_CONFIG.get("model_connect_timeout_seconds", 10), 10.0, maximum=60.0)


def get_model_read_timeout_seconds() -> float:
    return _bounded_seconds(RUNTIME_CONFIG.get("model_read_timeout_seconds", 180), 180.0, maximum=900.0)


def get_network_timeout_seconds() -> float:
    return _bounded_seconds(RUNTIME_CONFIG.get("network_timeout_seconds", 20), 20.0, maximum=300.0)


def get_api_host() -> str:
    return str(os.environ.get("RWKV_ECRA_API_HOST", SERVER_CONFIG.get("host", "0.0.0.0")))


def get_api_port() -> int:
    try:
        value = int(os.environ.get("RWKV_ECRA_API_PORT", SERVER_CONFIG.get("port", 8787)))
    except (TypeError, ValueError):
        value = 8787
    return max(1, min(65535, value))


def get_api_workers() -> int:
    try:
        value = int(os.environ.get("RWKV_ECRA_API_WORKERS", SERVER_CONFIG.get("workers", 1)))
    except (TypeError, ValueError):
        value = 1
    return max(1, min(32, value))


def get_wigolo_mode() -> str:
    """Select how the optional local wigolo web provider is used."""
    value = os.environ.get("RWKV_ECRA_WIGOLO_MODE", WIGOLO_CONFIG.get("mode", "auto"))
    value = str(value or "auto").strip().lower()
    return value if value in {"auto", "off", "only"} else "auto"


def get_wigolo_base_url() -> str:
    value = os.environ.get("WIGOLO_BASE_URL", WIGOLO_CONFIG.get("base_url", "http://127.0.0.1:3333"))
    return str(value or "").strip().rstrip("/")


def get_wigolo_api_token() -> str:
    return os.environ.get("WIGOLO_API_TOKEN", str(WIGOLO_CONFIG.get("api_token", "") or ""))


def get_wigolo_timeout() -> float:
    value = os.environ.get("WIGOLO_TIMEOUT_SECONDS", WIGOLO_CONFIG.get("timeout_seconds", 20))
    try:
        return max(1.0, min(float(value), 120.0))
    except (TypeError, ValueError):
        return 20.0
