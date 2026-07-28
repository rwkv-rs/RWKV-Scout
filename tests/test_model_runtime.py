from __future__ import annotations

import os
import json
import unittest
from unittest.mock import Mock, patch

import config
from clients.llm_client import LLMClient
from clients.slm_client import SLMClient
from runtime.backend import BackendResponse
from runtime.compat import OpenAICompatBackend
from runtime.direct_rwkv import DirectRWKVBackend
from runtime.factory import get_model_backend, reset_model_backend
from runtime.transcript import render_rwkv_transcript
from scripts.preflight import probe_model_service


class ModelRuntimeTests(unittest.TestCase):
    def tearDown(self):
        reset_model_backend()

    def test_transcript_is_owned_by_runtime_layer(self):
        prompt = render_rwkv_transcript(
            [
                {"role": "system", "content": "Use evidence."},
                {"role": "user", "content": "Question"},
            ]
        )
        self.assertEqual(prompt, "System:\nUse evidence.\n\nUser:\nQuestion\n\nAssistant:")

    def test_compat_response_is_normalized_without_sdk_objects(self):
        response = OpenAICompatBackend._response(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        )
        self.assertIsInstance(response, BackendResponse)
        self.assertEqual(response.content, "answer")
        self.assertEqual(response.usage["prompt_tokens"], 4)

    def test_compat_backend_decodes_utf8_json_when_server_omits_charset(self):
        payload = json.dumps(
            {"choices": [{"text": "深圳地铁"}]},
            ensure_ascii=False,
        ).encode("utf-8")
        http_response = Mock(status_code=200, content=payload)
        backend = OpenAICompatBackend()
        backend._session.post = Mock(return_value=http_response)
        with (
            patch("runtime.compat.get_llm_base_url", return_value="http://model/v1"),
            patch("runtime.compat.get_llm_api_key", return_value="rwkv-skills"),
        ):
            result = backend._post("/completions", {})
        self.assertEqual(result["choices"][0]["text"], "深圳地铁")

    def test_direct_backend_is_lazy_and_reports_missing_local_contract(self):
        backend = DirectRWKVBackend({"engine_root": "", "model_path": ""})
        self.assertFalse(backend.health()["available"])
        self.assertFalse(backend.health()["loaded"])

    def test_factory_can_select_direct_backend_without_loading_cuda(self):
        with patch.dict(os.environ, {"RWKV_ECRA_MODEL_BACKEND": "direct_rwkv"}, clear=False):
            backend = get_model_backend()
        self.assertEqual(backend.backend_name, "direct_rwkv")
        self.assertFalse(backend.health()["loaded"])

    def test_model_profile_can_select_project_owned_direct_runtime(self):
        profile = config.get_model_profile("local_direct_1p5b")
        self.assertEqual(profile["runtime_backend"], "direct_rwkv")
        token = config.override_model_backend.set(profile["runtime_backend"])
        direct_token = config.override_direct_rwkv_config.set(profile["direct_runtime"])
        try:
            self.assertEqual(config.get_model_backend_name(), "direct_rwkv")
            self.assertEqual(config.get_direct_rwkv_config()["model_name"], profile["model"])
        finally:
            config.override_direct_rwkv_config.reset(direct_token)
            config.override_model_backend.reset(token)

    def test_llm_client_routes_local_completion_through_backend(self):
        fake = BackendResponse(content="local answer", usage={"prompt_tokens": 3, "completion_tokens": 2})
        backend = Mock()
        backend.backend_name = "direct_rwkv"
        backend.text_completion.return_value = fake
        with patch("clients.llm_client.get_model_backend", return_value=backend):
            response = LLMClient().text_completion("User:\nhello\n\nAssistant:", max_tokens=8)
        self.assertEqual(response.content, "local answer")
        backend.text_completion.assert_called_once()

    def test_slm_client_routes_batch_generation_to_direct_backend(self):
        backend = Mock()
        backend.batch_text_completion.return_value = ["a", "b"]
        with (
            patch.dict(os.environ, {"RWKV_ECRA_MODEL_BACKEND": "direct_rwkv"}, clear=False),
            patch("clients.slm_client.get_model_backend", return_value=backend),
        ):
            result = SLMClient()._batch_generate_direct(["one", "two"])
        self.assertEqual(result, ["a", "b"])
        backend.batch_text_completion.assert_called_once()

    def test_direct_preflight_rejects_checkpoint_that_does_not_match_contract(self):
        backend = Mock()
        backend.model_name = "rwkv7-g1h-1.5b"
        backend.health.return_value = {
            "available": True,
            "backend": "direct_rwkv",
            "model": backend.model_name,
        }
        with (
            patch.dict(os.environ, {"RWKV_ECRA_MODEL_BACKEND": "direct_rwkv"}, clear=False),
            patch("scripts.preflight.get_model_backend", return_value=backend),
            patch("scripts.preflight.config.get_llm_model", return_value="rwkv7-g1i-13.3b"),
        ):
            result = probe_model_service()
        self.assertTrue(result["available"])
        self.assertFalse(result["model_match"])

    def test_direct_health_rejects_configured_checkpoint_name_mismatch(self):
        backend = DirectRWKVBackend(
            {
                "engine_root": "/tmp/engine",
                "model_path": "/tmp/rwkv7-g1h-1.5b-20260710-ctx10240.pth",
                "model_name": "rwkv7-g1i-13.3b",
            }
        )
        health = backend.health()
        self.assertFalse(health["available"])
        self.assertFalse(health["model_match"])


if __name__ == "__main__":
    unittest.main()
