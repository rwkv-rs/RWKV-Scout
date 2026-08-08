# RWKV-ECRA/config.py
import os
import json
import tempfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
LOCAL_ENV_FILE = os.path.join(BASE_DIR, ".env.local")
_LOCAL_SECRET_LOCK = threading.Lock()


def _load_local_env_file(path: str) -> None:
    """Load ignored, deployment-local secrets without overriding real env vars."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _parse_env_key_pool(value: object) -> list[str]:
    """Parse a JSON or comma-separated credential pool without logging it."""

    text = str(value or "").strip()
    if not text:
        return []
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = text.replace("\n", ",").split(",")
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    values: list[str] = []
    for item in parsed:
        candidate = str(item or "").strip().strip("'\"")
        if candidate and candidate not in values:
            values.append(candidate)
    return values


def _remove_key_from_local_env(path: str, env_name: str, api_key: str) -> bool:
    """Atomically remove one credential from the ignored local env file."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return False

    pool_name = f"{env_name}_API_KEYS"
    primary_name = f"{env_name}_API_KEY"
    changed = False
    rewritten: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            rewritten.append(raw_line)
            continue
        key, separator, raw_value = stripped.partition("=")
        key = key.strip()
        if not separator or key not in {pool_name, primary_name}:
            rewritten.append(raw_line)
            continue

        if key == primary_name:
            current = str(raw_value or "").strip().strip("'\"")
            if current != api_key:
                rewritten.append(raw_line)
                continue
            changed = True
            continue

        current_pool = _parse_env_key_pool(raw_value)
        filtered_pool = [value for value in current_pool if value != api_key]
        if len(filtered_pool) == len(current_pool):
            rewritten.append(raw_line)
            continue
        changed = True
        if filtered_pool:
            encoded = json.dumps(filtered_pool, ensure_ascii=False, separators=(",", ":"))
            rewritten.append(f"{pool_name}={encoded}\n")

    if not changed:
        return False

    directory = os.path.dirname(os.path.abspath(path))
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(rewritten)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
    return True


def retire_search_api_key(
    service: str,
    api_key: str,
    *,
    local_env_file: str | None = None,
) -> bool:
    """Remove a permanently unusable search key from runtime and local storage.

    Environment variables inherited from a parent process cannot be rewritten in
    that parent, but the current process is updated immediately. Deployment-local
    keys loaded from ``.env.local`` are also removed atomically so a restart does
    not reintroduce a rejected or permanently exhausted credential.
    """

    name = str(service or "").strip()
    target = str(api_key or "").strip()
    if not name or not target:
        return False
    env_name = name.upper()
    pool_name = f"{env_name}_API_KEYS"
    primary_name = f"{env_name}_API_KEY"
    changed = False

    with _LOCAL_SECRET_LOCK:
        pool = _parse_env_key_pool(os.environ.get(pool_name, ""))
        filtered_pool = [value for value in pool if value != target]
        if len(filtered_pool) != len(pool):
            changed = True
            if filtered_pool:
                os.environ[pool_name] = json.dumps(
                    filtered_pool,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            else:
                os.environ.pop(pool_name, None)

        if str(os.environ.get(primary_name, "")).strip() == target:
            os.environ.pop(primary_name, None)
            changed = True

        configured_primary = str(API_KEYS.get(name, "") or "").strip()
        if configured_primary == target:
            API_KEYS[name] = ""
            changed = True
        configured_pool = API_KEYS.get(f"{name}_pool", [])
        if isinstance(configured_pool, list):
            filtered_config = [
                value for value in configured_pool if str(value or "").strip() != target
            ]
            if len(filtered_config) != len(configured_pool):
                API_KEYS[f"{name}_pool"] = filtered_config
                changed = True

        path = local_env_file if local_env_file is not None else LOCAL_ENV_FILE
        if _remove_key_from_local_env(path, env_name, target):
            changed = True
    return changed


_load_local_env_file(LOCAL_ENV_FILE)

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
        "context_length": 10240,
        "provider": "local_7b",
    },
    "local_13b": {
        "label": "RWKV 13.3B",
        "model": "rwkv7-g1i-13.3b-20260805-ctx16384",
        "context_length": 16384,
        "provider": "local_13b",
    },
    "local_direct_1p5b": {
        "label": "RWKV 1.5B (Direct)",
        "model": "rwkv7-g1h-1.5b-20260710-ctx10240",
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
override_llm_temperature: ContextVar[float | None] = ContextVar(
    "override_llm_temperature",
    default=None,
)
override_llm_seed: ContextVar[int | None] = ContextVar(
    "override_llm_seed",
    default=None,
)
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
    return (
        override_llm_key.get()
        or os.environ.get("RWKV_ECRA_LLM_API_KEY", "")
        or API_KEYS.get(provider, "")
    )


def get_search_api_key(service: str) -> str:
    """Read a search-service key from the process environment first."""
    name = str(service or "").strip().upper()
    if not name:
        return ""
    return os.environ.get(f"{name}_API_KEY", "") or API_KEYS.get(service, "")


def get_search_api_keys(service: str) -> list[str]:
    """Return configured search keys in deterministic failover order.

    A service may define its primary key as ``API_KEYS[service]`` and optional
    fallbacks as ``API_KEYS[f"{service}_pool"]``.  Environment overrides are
    supported for local deployment, but configuration remains the normal
    project-level path.  Duplicate values are removed without exposing keys to
    callers or logs.
    """

    name = str(service or "").strip()
    env_name = name.upper()
    candidates: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)
            return
        text = str(value or "").strip()
        if text and text not in candidates:
            candidates.append(text)

    configured_env_pool = os.environ.get(f"{env_name}_API_KEYS", "").strip()
    if configured_env_pool:
        try:
            parsed = json.loads(configured_env_pool)
        except json.JSONDecodeError:
            parsed = configured_env_pool.replace("\n", ",").split(",")
        add(parsed)
    add(os.environ.get(f"{env_name}_API_KEY", ""))
    add(API_KEYS.get(name, ""))
    add(API_KEYS.get(f"{name}_pool", []))
    return candidates

def get_llm_base_url() -> str:
    provider = get_llm_provider()
    return (
        override_llm_url.get()
        or os.environ.get("RWKV_ECRA_LLM_BASE_URL", "")
        or LLM_ENDPOINTS.get(provider, {}).get("base_url", "")
    )

def get_llm_model() -> str:
    provider = get_llm_provider()
    return os.environ.get(
        "RWKV_ECRA_LLM_MODEL",
        LLM_ENDPOINTS.get(provider, {}).get("model", ""),
    )


def get_llm_temperature() -> float:
    """Return the configured sampling temperature for model requests."""
    provider = get_llm_provider()
    raw_value = override_llm_temperature.get()
    if raw_value is None:
        raw_value = os.environ.get(
            "RWKV_ECRA_LLM_TEMPERATURE",
            LLM_ENDPOINTS.get(provider, {}).get("temperature", 0.0),
        )
    try:
        return max(0.0, min(float(raw_value), 2.0))
    except (TypeError, ValueError):
        return 0.0


def get_model_stage_temperature(stage: str) -> float:
    """Return an explicit role temperature without changing the global model profile."""

    stage_name = str(stage or "").strip().casefold()
    environment_name = "RWKV_ECRA_" + "".join(
        character if character.isalnum() else "_" for character in stage_name.upper()
    ) + "_TEMPERATURE"
    sampling = MODEL_RUNTIME_CONFIG.get("sampling", {})
    configured = sampling.get(stage_name) if isinstance(sampling, dict) else None
    raw_value = os.environ.get(environment_name, configured)
    if raw_value is None:
        return get_llm_temperature()
    try:
        return max(0.0, min(float(raw_value), 2.0))
    except (TypeError, ValueError):
        return get_llm_temperature()


def get_model_replan_temperature(generation: int) -> float:
    """Return the temperature for one planner rebuild generation.

    Only the request-local temperature changes.  Sampling parameters other
    than ``temperature`` and the explicit replan ``seed`` remain untouched.
    """

    sampling = MODEL_RUNTIME_CONFIG.get("sampling", {})
    values = sampling if isinstance(sampling, dict) else {}
    base = get_model_stage_temperature("planner_replan")

    def configured_float(name: str, fallback: float) -> float:
        environment_name = "RWKV_ECRA_" + name.upper() + "_TEMPERATURE"
        raw_value = os.environ.get(environment_name, values.get(name, fallback))
        try:
            return max(0.0, min(float(raw_value), 2.0))
        except (TypeError, ValueError):
            return fallback

    increment = configured_float("planner_replan_increment", 0.1)
    maximum = configured_float("planner_replan_max", 0.55)
    return min(maximum, base + increment * max(0, int(generation or 1) - 1))


def get_llm_seed() -> int | None:
    """Return an optional request-local seed; absence keeps the fast sampler path."""

    raw_value = override_llm_seed.get()
    if raw_value is None:
        raw_value = os.environ.get("RWKV_ECRA_LLM_SEED")
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return max(-(2**63), min(value, 2**63 - 1))


@contextmanager
def model_sampling_parameters(temperature: float, *, seed: int | None = None):
    """Apply task-local temperature/seed without leaking across concurrent tasks."""

    temperature_token = override_llm_temperature.set(
        max(0.0, min(float(temperature), 2.0))
    )
    seed_token = override_llm_seed.set(seed)
    try:
        yield
    finally:
        override_llm_seed.reset(seed_token)
        override_llm_temperature.reset(temperature_token)


@contextmanager
def model_sampling_temperature(temperature: float):
    """Backward-compatible temperature-only request scope."""

    with model_sampling_parameters(temperature):
        yield


def get_llm_context_length() -> int:
    provider = get_llm_provider()
    try:
        return max(
            1,
            int(
                os.environ.get(
                    "RWKV_ECRA_LLM_CONTEXT_LENGTH",
                    LLM_ENDPOINTS.get(provider, {}).get("context_length", 10240),
                )
            ),
        )
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
        "context_length": profile.get("context_length", 10240),
        "provider": selected,
    }


def validate_experiment_model_contract(profile_key: str | None = None) -> dict:
    """Validate model identity without pinning a deployment-specific endpoint.

    The endpoint is deliberately read from ``config.json`` at runtime: the
    same model may be reached through a local loopback, an SSH forward, or a
    server address.  Treating that address as model identity made preflight
    fail after a valid deployment switch.
    """
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
    return (
        override_slm_password.get()
        or os.environ.get("RWKV_ECRA_SLM_PASSWORD", "")
        or get_llm_api_key()
        or SLM_CONFIG.get("password", "")
    )

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


def get_model_request_concurrency() -> int:
    """Maximum concurrent requests sent to the model service workspace-wide."""
    return max(1, int(MODEL_RUNTIME_CONFIG.get("max_inflight_requests", 8)))


def get_model_reserved_control_slots() -> int:
    """Model slots reserved for planner, validation, replanning and synthesis."""
    limit = get_model_request_concurrency()
    try:
        configured = int(MODEL_RUNTIME_CONFIG.get("reserved_control_slots", 2))
    except (TypeError, ValueError):
        configured = 2
    return max(0, min(max(0, limit - 1), configured))


def get_model_chunk_requests_per_task() -> int:
    """Maximum simultaneous chunk-evidence model requests for one task."""
    try:
        configured = int(MODEL_RUNTIME_CONFIG.get("max_chunk_requests_per_task", 2))
    except (TypeError, ValueError):
        configured = 2
    return max(1, min(configured, get_model_request_concurrency()))


def _bounded_seconds(value: object, default: float, *, minimum: float = 0.1, maximum: float = 3600.0) -> float:
    try:
        return max(minimum, min(float(value), maximum))
    except (TypeError, ValueError):
        return default


def get_analysis_timeout_seconds() -> float | None:
    """Maximum wall-clock budget, or ``None`` when single-task timeout is disabled."""
    configured = RUNTIME_CONFIG.get("analysis_timeout_seconds", 600)
    if configured is None:
        return None
    try:
        value = float(configured)
    except (TypeError, ValueError):
        return 600.0
    if value <= 0:
        return None
    return max(0.1, min(value, 3600.0))


def get_model_connect_timeout_seconds() -> float:
    return _bounded_seconds(RUNTIME_CONFIG.get("model_connect_timeout_seconds", 10), 10.0, maximum=60.0)


def get_model_read_timeout_seconds() -> float:
    return _bounded_seconds(RUNTIME_CONFIG.get("model_read_timeout_seconds", 180), 180.0, maximum=900.0)


def get_model_retry_attempts() -> int:
    """Maximum attempts for one model request, including the first attempt."""
    try:
        value = int(MODEL_RUNTIME_CONFIG.get("max_retry_attempts", 2))
    except (TypeError, ValueError):
        value = 2
    return max(1, min(value, 4))


def get_model_retry_delay_seconds() -> float:
    return _bounded_seconds(MODEL_RUNTIME_CONFIG.get("retry_delay_seconds", 1), 1.0, maximum=30.0)


def get_model_retry_timeout_errors() -> bool:
    value = MODEL_RUNTIME_CONFIG.get("retry_timeout_errors", False)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


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
