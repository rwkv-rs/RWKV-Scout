from __future__ import annotations

import os
import json
import unittest
from contextlib import nullcontext
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
from utils.token_tracker import current_task_id, model_lane


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
        self.assertEqual(prompt, "### User\nUse evidence.\n\n### User\nQuestion\n\n### Assistant")

    def test_tool_transcript_matches_rwkv_skills_json_protocol(self):
        prompt = render_rwkv_transcript(
            [
                {"role": "system", "content": "Use the available tools."},
                {"role": "user", "content": "Find the answer."},
            ],
            tools=[{"name": "web_search"}],
        )
        self.assertEqual(prompt.count("System:"), 0)
        self.assertIn("Use the available tools.", prompt)
        self.assertIn('"name":"web_search"', prompt)
        self.assertIn("**Tool Call:**", prompt)
        self.assertTrue(prompt.endswith("### Assistant\n**Tool Call:**\n"))

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

    def test_compat_response_preserves_every_model_character(self):
        output = "  Assistant: <think>model text</think>\nanswer  \n"
        response = OpenAICompatBackend._response(
            {"choices": [{"text": output, "finish_reason": "stop"}]},
            text_key="text",
        )
        self.assertEqual(response.content, output)

    def test_compat_request_scopes_only_temperature_and_optional_seed(self):
        backend = OpenAICompatBackend()
        backend._post = Mock(
            return_value={"choices": [{"text": "ok", "finish_reason": "stop"}]}
        )

        with config.model_sampling_parameters(0.25, seed=17):
            response = backend.text_completion("prompt", max_tokens=12)

        self.assertEqual(response.content, "ok")
        path, payload = backend._post.call_args.args
        self.assertEqual(path, "/completions")
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["seed"], 17)
        for name in ("top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty"):
            self.assertNotIn(name, payload)
        self.assertIsNone(config.get_llm_seed())

    def test_direct_backend_preserves_every_decoded_model_character(self):
        output = "  Assistant: <think>model text</think>\nanswer  \n"
        backend = DirectRWKVBackend(
            {"temperature": 0, "top_p": 1, "top_k": 0, "stop_tokens": [0]}
        )
        backend._load = lambda: None
        backend._torch = type("Torch", (), {"inference_mode": staticmethod(nullcontext)})()
        backend._model = Mock()
        backend._model.generate_zero_state.return_value = object()
        backend._model.forward_batch.return_value = [object()]
        backend._tokenizer = Mock()
        backend._tokenizer.encode.return_value = [1]
        backend._tokenizer.decode.return_value = output
        tokens = iter([2, 0])
        backend._sample = lambda *_args, **_kwargs: next(tokens)

        response = backend._generate("prompt", max_tokens=2)
        self.assertEqual(response.content, output)

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
            patch("runtime.compat.get_llm_api_key", return_value="test-local-key"),
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

    def test_model_event_records_lane_generation_budget_stop_and_finish_reason(self):
        fake = BackendResponse(
            content="bounded answer",
            usage={"prompt_tokens": 3, "completion_tokens": 2},
            finish_reason="length",
        )
        backend = Mock()
        backend.backend_name = "openai_compat"
        backend.text_completion.return_value = fake
        token = current_task_id.set("MODEL_AUDIT_METADATA")
        try:
            with (
                model_lane("writer"),
                patch("clients.llm_client.get_model_backend", return_value=backend),
                patch("clients.llm_client.record_model_event") as record,
            ):
                LLMClient().text_completion(
                    "exact prompt",
                    max_tokens=9,
                    stop=["### User"],
                )
        finally:
            current_task_id.reset(token)

        self.assertEqual(record.call_args.args[0], "MODEL_AUDIT_METADATA")
        payload = record.call_args.kwargs
        self.assertEqual(payload["model_lane"], "writer")
        self.assertEqual(payload["request_max_tokens"], 9)
        self.assertEqual(payload["stop"], ["### User"])
        self.assertEqual(payload["finish_reason"], "length")
        self.assertEqual(payload["prompt"], "exact prompt")
        self.assertEqual(payload["output"], "bounded answer")

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

    def test_compat_batch_completion_preserves_task_context_in_workers(self):
        backend = OpenAICompatBackend()
        seen = []

        def fake_text_completion(prompt, *, max_tokens=768, stop=None):
            del max_tokens, stop
            seen.append(current_task_id.get())
            return BackendResponse(content=prompt)

        backend.text_completion = fake_text_completion
        token = current_task_id.set("CONTEXT_BATCH_1")
        try:
            result = backend.batch_text_completion(["a", "b"], max_tokens=8)
        finally:
            current_task_id.reset(token)
        self.assertEqual(result, ["a", "b"])
        self.assertEqual(seen, ["CONTEXT_BATCH_1", "CONTEXT_BATCH_1"])

    def test_direct_preflight_rejects_checkpoint_that_does_not_match_contract(self):
        backend = Mock()
        backend.backend_name = "direct_rwkv"
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
