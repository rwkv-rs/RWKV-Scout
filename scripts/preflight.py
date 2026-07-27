"""Production preflight for the exact local RWKV-ECRA deployment contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

import config
from runtime import get_model_backend
from utils.evaluation_dataset import load_dataset
from utils.reference_validation import validate_dataset_references


def probe_model_service() -> dict[str, Any]:
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
            timeout=(min(config.get_model_connect_timeout_seconds(), 2.0), 3.0),
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        model_ids = [
            str(item.get("id"))
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
        expected = config.get_llm_model()
        return {
            "available": True,
            "status_code": response.status_code,
            "configured_model": expected,
            "served_models": model_ids[:8],
            "model_match": not model_ids or expected in model_ids,
        }
    except Exception as exc:
        return {
            "available": False,
            "configured_model": config.get_llm_model(),
            "reason": f"{type(exc).__name__}: {exc}"[:300],
        }


def _directory_checks() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name in ("input_directory", "output_directory", "checkpoint_directory"):
        raw = config.DATA_PIPELINE.get(name)
        path = Path(str(raw)).expanduser() if raw else Path("")
        try:
            path.mkdir(parents=True, exist_ok=True)
            result[name] = path.is_dir() and os.access(path, os.W_OK)
        except OSError:
            result[name] = False
    return result


def run_preflight(
    dataset: Path | None = None,
    *,
    require_references: bool = False,
    check_remote: bool = False,
) -> dict[str, Any]:
    try:
        contract = config.validate_experiment_model_contract()
        contract = dict(contract)
        contract["api_key_configured"] = bool(contract.pop("api_key", ""))
        contract_ok = True
        contract_error = ""
    except Exception as exc:
        contract = {}
        contract_ok = False
        contract_error = f"{type(exc).__name__}: {exc}"

    directories = _directory_checks()
    model = probe_model_service()
    reference_validation: dict[str, Any] | None = None
    dataset_error = ""
    if dataset:
        try:
            rows = load_dataset(dataset)
            reference_validation = validate_dataset_references(rows, check_remote=check_remote)
        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"[:300]
    references_ok = not require_references or bool(reference_validation and reference_validation.get("all_ready"))
    ready = (
        contract_ok
        and all(directories.values())
        and model.get("available") is True
        and model.get("model_match", True) is True
        and not dataset_error
        and references_ok
    )
    return {
        "schema_version": "rwkv-ecra.preflight.v1",
        "status": "ready" if ready else "not_ready",
        "python": sys.version.split()[0],
        "contract": {
            "valid": contract_ok,
            "model": contract,
            "error": contract_error,
        },
        "directories": directories,
        "model_service": model,
        "dataset": str(dataset) if dataset else "",
        "check_remote": check_remote,
        "dataset_error": dataset_error,
        "reference_validation": reference_validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--require-references", action="store_true")
    parser.add_argument("--check-remote", action="store_true", help="also fetch reference URLs when a dataset is supplied")
    parser.add_argument("--allow-model-down", action="store_true", help="report not_ready but return success for offline CI")
    args = parser.parse_args()
    result = run_preflight(
        args.dataset,
        require_references=args.require_references,
        check_remote=args.check_remote,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready" or args.allow_model_down else 2


if __name__ == "__main__":
    raise SystemExit(main())
