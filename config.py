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
override_llm_top_p: ContextVar[float | None] = ContextVar(
    "override_llm_top_p",
    default=None,
)
override_llm_top_k: ContextVar[int | None] = ContextVar(
    "override_llm_top_k",
    default=None,
)
override_llm_presence_penalty: ContextVar[float | None] = ContextVar(
    "override_llm_presence_penalty",
    default=None,
)
override_llm_frequency_penalty: ContextVar[float | None] = ContextVar(
    "override_llm_frequency_penalty",
    default=None,
)
override_llm_penalty_decay: ContextVar[float | None] = ContextVar(
    "override_llm_penalty_decay",
    default=None,
)
override_llm_no_penalty_token_ids: ContextVar[tuple[int, ...] | None] = ContextVar(
    "override_llm_no_penalty_token_ids",
    default=None,
)
override_model_request_stage: ContextVar[str] = ContextVar(
    "override_model_request_stage",
    default="",
)
override_sampling_policy_reason: ContextVar[str] = ContextVar(
    "override_sampling_policy_reason",
    default="",
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
    if isinstance(configured, dict):
        configured = configured.get("temperature")
    raw_value = os.environ.get(environment_name, configured)
    if raw_value is None:
        return get_llm_temperature()
    try:
        return max(0.0, min(float(raw_value), 2.0))
    except (TypeError, ValueError):
        return get_llm_temperature()


def get_model_replan_temperature(generation: int, reason: str = "") -> float:
    """Return a failure-aware request temperature for one RWKV replan.

    The complete request-local ``planner_replan`` sampling profile is applied
    by the caller; this function selects only its failure-aware temperature.
    No request seed is injected by the controller.

    Replanning is exploratory, but a protocol repair, a source conflict and a
    repeated frozen path are not the same kind of request.  The old policy
    raised temperature only by generation count, even when the prompt was
    asking for strict JSON, and reached 0.9 without changing the missing tool
    contract.  This policy uses the model-visible failure reason first and only
    permits a small extra increase for a repeated strategy stall.  The base
    profile mirrors the empirically stable Math-500 NoCoT anti-repetition
    sampler, which preserved JSON while escaping recurrent query trajectories.
    """

    sampling = MODEL_RUNTIME_CONFIG.get("sampling", {})
    values = sampling if isinstance(sampling, dict) else {}
    base = get_model_stage_temperature("planner_replan")

    def configured_float(name: str, fallback: float) -> float:
        environment_name = "RWKV_ECRA_" + name.upper() + "_TEMPERATURE"
        configured_value = values.get(name, fallback)
        if isinstance(configured_value, dict):
            configured_value = configured_value.get("temperature", fallback)
        raw_value = os.environ.get(environment_name, configured_value)
        try:
            return max(0.0, min(float(raw_value), 2.0))
        except (TypeError, ValueError):
            return fallback

    reason_text = str(reason or "").casefold()
    if "protocol" in reason_text or "json" in reason_text:
        return configured_float("planner_replan_protocol", 0.1)
    if "conflict" in reason_text or "contradict" in reason_text:
        return configured_float("planner_replan_conflict", 0.5)
    if any(
        marker in reason_text
        for marker in (
            "duplicate",
            "frozen",
            "stall",
            "no_new_evidence",
            "no new evidence",
            "repeated",
        )
    ):
        value = configured_float("planner_replan_stall", 0.8)
        if int(generation or 1) > 1:
            value = configured_float("planner_replan_stall_escalated", 0.9)
        return min(configured_float("planner_replan_max", 0.9), value)
    return min(configured_float("planner_replan_max", 0.9), base)


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


def get_model_stage_sampling(stage: str) -> dict[str, object]:
    """Return a normalized request-level decoding profile for one model role."""

    stage_name = str(stage or "").strip().casefold()
    sampling = MODEL_RUNTIME_CONFIG.get("sampling", {})
    configured = sampling.get(stage_name, {}) if isinstance(sampling, dict) else {}
    profile = dict(configured) if isinstance(configured, dict) else {}
    profile["temperature"] = get_model_stage_temperature(stage_name)

    def bounded_float(name: str, minimum: float, maximum: float) -> float | None:
        value = profile.get(name)
        if value is None:
            return None
        try:
            return max(minimum, min(float(value), maximum))
        except (TypeError, ValueError):
            return None

    normalized: dict[str, object] = {"temperature": profile["temperature"]}
    for name, minimum, maximum in (
        ("top_p", 0.00001, 1.0),
        ("presence_penalty", -2.0, 2.0),
        ("frequency_penalty", -2.0, 2.0),
        ("penalty_decay", 0.0, 1.0),
    ):
        value = bounded_float(name, minimum, maximum)
        if value is not None:
            normalized[name] = value
    if profile.get("top_k") is not None:
        try:
            normalized["top_k"] = max(0, int(profile["top_k"]))
        except (TypeError, ValueError):
            pass
    token_ids = profile.get("no_penalty_token_ids")
    if isinstance(token_ids, (list, tuple)):
        normalized["no_penalty_token_ids"] = tuple(
            dict.fromkeys(max(0, int(value)) for value in token_ids)
        )
    return normalized


def get_llm_sampling_parameters() -> dict[str, object]:
    """Return the active request-local wire sampling values for audit/runtime."""

    output: dict[str, object] = {"temperature": get_llm_temperature()}
    for name, context in (
        ("top_p", override_llm_top_p),
        ("top_k", override_llm_top_k),
        ("presence_penalty", override_llm_presence_penalty),
        ("frequency_penalty", override_llm_frequency_penalty),
        ("penalty_decay", override_llm_penalty_decay),
        ("no_penalty_token_ids", override_llm_no_penalty_token_ids),
    ):
        value = context.get()
        if value is not None:
            output[name] = value
    return output


def get_model_request_stage() -> str:
    """Return the current request role for audit logging."""

    return str(override_model_request_stage.get() or "")


def get_sampling_policy_reason() -> str:
    """Return the auditable reason for the current request temperature."""

    return str(override_sampling_policy_reason.get() or "")


@contextmanager
def model_sampling_parameters(
    temperature: float,
    *,
    seed: int | None = None,
    stage: str = "",
    policy_reason: str = "",
    top_p: float | None = None,
    top_k: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    penalty_decay: float | None = None,
    no_penalty_token_ids: tuple[int, ...] | list[int] | None = None,
):
    """Apply one complete request-local decoding policy without leakage."""

    stage_profile = get_model_stage_sampling(stage) if stage else {}
    top_p = stage_profile.get("top_p") if top_p is None else top_p
    top_k = stage_profile.get("top_k") if top_k is None else top_k
    presence_penalty = (
        stage_profile.get("presence_penalty")
        if presence_penalty is None
        else presence_penalty
    )
    frequency_penalty = (
        stage_profile.get("frequency_penalty")
        if frequency_penalty is None
        else frequency_penalty
    )
    penalty_decay = (
        stage_profile.get("penalty_decay")
        if penalty_decay is None
        else penalty_decay
    )
    no_penalty_token_ids = (
        stage_profile.get("no_penalty_token_ids")
        if no_penalty_token_ids is None
        else no_penalty_token_ids
    )

    temperature_token = override_llm_temperature.set(
        max(0.0, min(float(temperature), 2.0))
    )
    seed_token = override_llm_seed.set(seed)
    top_p_token = override_llm_top_p.set(
        None if top_p is None else max(0.00001, min(float(top_p), 1.0))
    )
    top_k_token = override_llm_top_k.set(
        None if top_k is None else max(0, int(top_k))
    )
    presence_token = override_llm_presence_penalty.set(
        None
        if presence_penalty is None
        else max(-2.0, min(float(presence_penalty), 2.0))
    )
    frequency_token = override_llm_frequency_penalty.set(
        None
        if frequency_penalty is None
        else max(-2.0, min(float(frequency_penalty), 2.0))
    )
    decay_token = override_llm_penalty_decay.set(
        None if penalty_decay is None else max(0.0, min(float(penalty_decay), 1.0))
    )
    no_penalty_token = override_llm_no_penalty_token_ids.set(
        None
        if no_penalty_token_ids is None
        else tuple(dict.fromkeys(max(0, int(value)) for value in no_penalty_token_ids))
    )
    stage_token = override_model_request_stage.set(str(stage or ""))
    reason_token = override_sampling_policy_reason.set(str(policy_reason or ""))
    try:
        yield
    finally:
        override_sampling_policy_reason.reset(reason_token)
        override_model_request_stage.reset(stage_token)
        override_llm_no_penalty_token_ids.reset(no_penalty_token)
        override_llm_penalty_decay.reset(decay_token)
        override_llm_frequency_penalty.reset(frequency_token)
        override_llm_presence_penalty.reset(presence_token)
        override_llm_top_k.reset(top_k_token)
        override_llm_top_p.reset(top_p_token)
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
