"""Probe and optionally exercise the configured project model backend."""

from __future__ import annotations

import argparse
import json

import config
from runtime import get_model_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="User:\n1+1=?\n\nAssistant:")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--model-key", default=None, help="probe a configured local model profile")
    args = parser.parse_args()

    context_tokens: list[tuple[object, object]] = []
    try:
        if args.model_key:
            profile = config.get_model_profile(args.model_key)
            context_tokens.append((config.override_llm_provider, config.override_llm_provider.set(args.model_key)))
            context_tokens.append((config.override_llm_url, config.override_llm_url.set(profile.get("base_url", ""))))
            if profile.get("runtime_backend"):
                context_tokens.append((config.override_model_backend, config.override_model_backend.set(profile["runtime_backend"])))
            if profile.get("direct_runtime") is not None:
                context_tokens.append((config.override_direct_rwkv_config, config.override_direct_rwkv_config.set(profile["direct_runtime"])))

        backend = get_model_backend()
        health = backend.health()
        result = {
            "schema_version": "rwkv-ecra.model-backend-smoke.v1",
            "model_key": args.model_key or "",
            "backend": backend.backend_name,
            "model": backend.model_name,
            "health": health,
        }
        if not args.probe_only and health.get("available"):
            response = backend.text_completion(args.prompt, max_tokens=args.max_tokens)
            result["health_after"] = backend.health()
            result["response"] = {
                "content": response.content,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if health.get("available") else 2
    finally:
        for context, token in reversed(context_tokens):
            context.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())
