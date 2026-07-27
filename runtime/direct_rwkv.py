"""Direct RWKV-7 inference backend.

This backend deliberately loads the checkpoint in the project process rather
than calling a model API.  The Albatross engine is loaded lazily because its
CUDA extension compilation is expensive and is not needed by unit tests.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from config import get_direct_rwkv_config, get_llm_model
from runtime.backend import BackendResponse
from runtime.transcript import render_rwkv_transcript
from utils.model_events import visible_model_text


class DirectRWKVConfigurationError(RuntimeError):
    """Raised when direct inference is selected but cannot be configured."""


class DirectRWKVBackend:
    """Single-model, state-isolated RWKV runtime owned by this project."""

    backend_name = "direct_rwkv"

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = dict(settings or get_direct_rwkv_config())
        self._load_lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._loaded = False
        configured_name = str(self.settings.get("model_name") or "").strip()
        model_path = str(self.settings.get("model_path") or "").strip()
        inferred_name = self._model_label(model_path)
        self._configured_model_name = configured_name
        self._checkpoint_model_name = inferred_name
        self._model_name = configured_name or inferred_name or get_llm_model() or "rwkv-direct"

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _model_label(model_path: str) -> str:
        if not model_path:
            return ""
        name = Path(model_path).name
        return name[:-4] if name.endswith(".pth") else name

    @staticmethod
    def _visible_output(text: str) -> str:
        """Hide a reasoning preamble even when a short run ends mid-block."""
        raw = str(text or "")
        lowered = raw.casefold()
        start = lowered.find("<think>")
        if start >= 0:
            end = lowered.find("</think>", start + len("<think>"))
            if end < 0:
                return raw[:start].strip()
        return visible_model_text(raw)

    @staticmethod
    def _path_value(value: Any) -> Path | None:
        raw = str(value or "").strip()
        return Path(raw).expanduser() if raw else None

    def _model_base_path(self) -> Path:
        raw = self._path_value(self.settings.get("model_path"))
        if raw is None:
            raise DirectRWKVConfigurationError(
                "direct_rwkv requires MODEL_RUNTIME.direct_rwkv.model_path "
                "or RWKV_ECRA_RWKV_MODEL_PATH"
            )
        if raw.is_file() and raw.suffix == ".pth":
            return raw.with_suffix("")
        if raw.with_suffix(raw.suffix + ".pth").is_file():
            return raw
        if raw.suffix == ".pth":
            return raw.with_suffix("")
        if raw.is_file():
            return raw
        raise DirectRWKVConfigurationError(f"RWKV checkpoint not found: {raw} (or {raw}.pth)")

    def _engine_root(self) -> Path:
        raw = self._path_value(self.settings.get("engine_root"))
        if raw is None or not raw.is_dir():
            raise DirectRWKVConfigurationError(
                "direct_rwkv requires MODEL_RUNTIME.direct_rwkv.engine_root "
                "pointing to an Albatross engine checkout"
            )
        return raw

    def _vocab_path(self, engine_root: Path) -> Path:
        configured = self._path_value(self.settings.get("vocab_path"))
        candidates = [
            configured,
            engine_root / "reference" / "rwkv_vocab_v20230424.txt",
            Path(__file__).resolve().parents[1] / "rwkv_vocab_v20230424.txt",
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return candidate
        raise DirectRWKVConfigurationError("RWKV tokenizer vocabulary file was not found")

    def _load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            engine_root = self._engine_root()
            model_base = self._model_base_path()
            if str(engine_root) not in sys.path:
                sys.path.insert(0, str(engine_root))

            try:
                import torch

                if not torch.cuda.is_available():
                    raise DirectRWKVConfigurationError(
                        "direct_rwkv currently requires a CUDA-enabled PyTorch runtime"
                    )
                rwkv_module = importlib.import_module("reference.rwkv7")
                tokenizer_module = importlib.import_module("reference.utils")
                model_args = SimpleNamespace(
                    vocab_size=int(self.settings.get("vocab_size", 65536)),
                    head_size=int(self.settings.get("head_size", 64)),
                    MODEL_NAME=str(model_base),
                )
                model = rwkv_module.RWKV_x070(model_args)
                tokenizer = tokenizer_module.TRIE_TOKENIZER(str(self._vocab_path(engine_root)))
            except DirectRWKVConfigurationError:
                raise
            except Exception as exc:
                raise DirectRWKVConfigurationError(
                    f"failed to load direct RWKV engine: {type(exc).__name__}: {exc}"
                ) from exc

            self._torch = torch
            self._model = model
            self._tokenizer = tokenizer
            self._loaded = True

    @staticmethod
    def _sample(logits: Any, *, temperature: float, top_p: float, top_k: int) -> int:
        values = logits.float()
        if temperature <= 0:
            return int(values.argmax().item())
        values = values / max(float(temperature), 1e-5)
        if top_k > 0 and top_k < values.numel():
            threshold = values.topk(int(top_k)).values[-1]
            values = values.masked_fill(values < threshold, float("-inf"))
        probabilities = values.softmax(dim=-1)
        if 0 < top_p < 1:
            sorted_probs, sorted_ids = probabilities.sort(descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            remove = cumulative > float(top_p)
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            probabilities[sorted_ids[remove]] = 0
            probabilities = probabilities / probabilities.sum()
        return int(probabilities.multinomial(1).item())

    def _generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: Sequence[str] | None = None,
    ) -> BackendResponse:
        self._load()
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        torch = self._torch
        token_ids = self._tokenizer.encode(str(prompt)) or [0]
        max_tokens = max(1, int(max_tokens))
        temperature = float(self.settings.get("temperature", 0.0) or 0.0)
        top_p = float(self.settings.get("top_p", 1.0) or 1.0)
        top_k = max(0, int(self.settings.get("top_k", 0) or 0))
        stop_strings = tuple(str(item) for item in (stop or ()) if str(item))
        stop_tokens = {int(item) for item in (self.settings.get("stop_tokens") or [0])}

        with self._load_lock, torch.inference_mode():
            state = self._model.generate_zero_state(1)
            logits = self._model.forward_batch([token_ids], state)[0]
            generated: list[int] = []
            output = ""
            finish_reason = "length"
            for _ in range(max_tokens):
                token = self._sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                if token in stop_tokens:
                    finish_reason = "stop"
                    break
                generated.append(token)
                output = self._tokenizer.decode(generated, utf8_errors="ignore")
                if any(marker in output for marker in stop_strings):
                    for marker in stop_strings:
                        if marker in output:
                            output = output.split(marker, 1)[0]
                    finish_reason = "stop"
                    break
                logits = self._model.forward_batch([[token]], state)[0]

        return BackendResponse(
            content=self._visible_output(output),
            usage={"prompt_tokens": len(token_ids), "completion_tokens": len(generated)},
            finish_reason=finish_reason,
        )

    def chat_completion(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_native_search: bool = False,
        max_tokens: int | None = None,
    ) -> BackendResponse:
        del enable_native_search
        prompt = render_rwkv_transcript(messages, tools=tools)
        return self._generate(
            prompt,
            max_tokens=max_tokens or int(self.settings.get("max_tokens", 768)),
        )

    def text_completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 768,
        stop: Sequence[str] | None = None,
    ) -> BackendResponse:
        return self._generate(prompt, max_tokens=max_tokens, stop=stop)

    def batch_text_completion(self, prompts: Sequence[str], *, max_tokens: int = 768) -> list[str]:
        if not prompts:
            return []
        # The model state is request-local.  Keeping the initial implementation
        # serialized is intentional; a scheduler can add safe same-length
        # batching later without exposing mutable state to callers.
        return [self.text_completion(prompt, max_tokens=max_tokens).content for prompt in prompts]

    def health(self) -> dict[str, Any]:
        model_path = self._path_value(self.settings.get("model_path"))
        engine_root = self._path_value(self.settings.get("engine_root"))
        checkpoint_exists = bool(
            model_path
            and (
                model_path.is_file()
                or Path(str(model_path) + ".pth").is_file()
            )
        )
        cuda_available: bool | None = None
        if checkpoint_exists and engine_root and engine_root.is_dir():
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            except Exception:
                cuda_available = False
        device = str(self.settings.get("device", "cuda") or "cuda").casefold()
        device_ready = device != "cuda" or cuda_available is True
        model_match = not self._configured_model_name or not self._checkpoint_model_name or self._configured_model_name == self._checkpoint_model_name
        return {
            "available": bool(checkpoint_exists and engine_root and engine_root.is_dir() and device_ready and model_match),
            "backend": self.backend_name,
            "loaded": self._loaded,
            "model": self.model_name,
            "checkpoint_model": self._checkpoint_model_name,
            "model_match": model_match,
            "checkpoint_configured": bool(model_path),
            "checkpoint_exists": checkpoint_exists,
            "engine_root_exists": bool(engine_root and engine_root.is_dir()),
            "device": device,
            "cuda_available": cuda_available,
        }
