"""Create the configured project-owned model backend."""

from __future__ import annotations

import json
import threading

from config import (
    get_direct_rwkv_config,
    get_llm_base_url,
    get_llm_model,
    get_llm_provider,
    get_model_backend_name,
)
from runtime.backend import ModelBackend
from runtime.compat import OpenAICompatBackend
from runtime.direct_rwkv import DirectRWKVBackend


_LOCK = threading.RLock()
_BACKENDS: dict[tuple[str, str], ModelBackend] = {}


def _effective_backend_name() -> str:
    selected = get_model_backend_name()
    if selected == "auto":
        settings = get_direct_rwkv_config()
        has_direct_paths = bool(
            str(settings.get("engine_root") or "").strip()
            and str(settings.get("model_path") or "").strip()
        )
        return "direct_rwkv" if has_direct_paths else "openai_compat"
    return selected


def _backend_key(selected: str) -> tuple[str, str]:
    if selected == "direct_rwkv":
        identity = json.dumps(get_direct_rwkv_config(), sort_keys=True, default=str)
    else:
        identity = json.dumps(
            {
                "provider": get_llm_provider(),
                "base_url": get_llm_base_url(),
                "model": get_llm_model(),
            },
            sort_keys=True,
            default=str,
        )
    return selected, identity


def get_model_backend() -> ModelBackend:
    selected = _effective_backend_name()
    key = _backend_key(selected)
    with _LOCK:
        backend = _BACKENDS.get(key)
        if backend is not None:
            return backend
        if selected == "direct_rwkv":
            backend: ModelBackend = DirectRWKVBackend()
        elif selected == "openai_compat":
            backend = OpenAICompatBackend()
        else:
            raise ValueError(f"unknown model backend: {selected}")
        _BACKENDS[key] = backend
        return backend


def reset_model_backend() -> None:
    with _LOCK:
        _BACKENDS.clear()
