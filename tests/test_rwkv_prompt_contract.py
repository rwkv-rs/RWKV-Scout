import unittest
from types import SimpleNamespace

import utils.rwkv_prompt as prompt_contract
from agent.planner import _canonicalize_tool_payload, _extract_json_object
from agent.retrieval_synthesis import synthesize_retrieval_answer
from utils.rwkv_prompt import (
    assistant_json_prefix,
    build_final_continuation_prompt,
    consume_final_prefill_boundary,
    tool_call_prefix,
)
from utils.rwkv_json_protocol import normalize_json_object_envelope


class _PromptFakeLLM:
    provider = "local_13b"

    def __init__(self, content="RWKV raw answer"):
        self.content = content
        self.prompt = ""
        self.stop = None

    def text_completion(self, prompt, max_tokens=0, stop=None):
        self.prompt = prompt
        self.stop = stop
        return SimpleNamespace(content=self.content)


class RWKVPromptContractTests(unittest.TestCase):
    def test_final_prompt_uses_raw_official_continuation_prefix(self):
        prompt = build_final_continuation_prompt("回答问题")
        self.assertTrue(prompt.endswith("Assistant: <think></think"))
        self.assertTrue(prompt.startswith("User: "))

    def test_json_prefix_matches_online_g1i_tool_continuation(self):
        self.assertEqual(
            assistant_json_prefix(enable_think=True, prefill_object=True),
            "Assistant: <think></think\n```json\n{",
        )
        self.assertEqual(
            assistant_json_prefix(enable_think=False, prefill_object=True),
            "Assistant: ```json\n{",
        )
        self.assertEqual(tool_call_prefix(), "Assistant: ```json\n")

    def test_stop_suffixes_match_ecra_role_boundaries(self):
        self.assertEqual(
            prompt_contract.JSON_CALL_STOP_SUFFIXES,
            (
                "\n```",
                "\nUser:",
                "\nSystem:",
                "\nAssistant:",
            ),
        )
        self.assertNotIn("```", prompt_contract.JSON_CALL_STOP_SUFFIXES)
        self.assertNotIn("User:", prompt_contract.JSON_CALL_STOP_SUFFIXES)
        self.assertNotIn("System:", prompt_contract.JSON_CALL_STOP_SUFFIXES)

    def test_online_g1i_tool_role_blocks_are_exact(self):
        rendered = prompt_contract.render_tool_transcript(
            [
                {
                    "role": "system",
                    "content": 'Tools: [{"name":"read_file"}]\nReturn only a JSON function call.',
                },
                {"role": "user", "content": "Read the requested file."},
                {
                    "role": "assistant",
                    "content": {
                        "name": "read_file",
                        "arguments": {"path": "notes.txt"},
                    },
                },
                {
                    "role": "tool",
                    "content": {"status": "ok", "content": "hello"},
                },
            ]
        )
        self.assertEqual(
            rendered,
            'System: Tools: [{"name":"read_file"}]\n'
            'Return only a JSON function call.\n\n'
            'User: Read the requested file.\n\n'
            'Assistant: ```json\n'
            '{"name":"read_file","arguments":{"path":"notes.txt"}}\n\n'
            'User: Function output: {\n  "status": "ok",\n  "content": "hello"\n}\n\n'
            'Assistant: ```json\n',
        )
        self.assertNotIn("###", rendered)
        self.assertNotIn("**Tool Call:**", rendered)
        self.assertNotIn("### Tool Output", rendered)

    def test_online_g1i_followup_request_can_start_from_function_output(self):
        rendered = prompt_contract.render_tool_transcript(
            [
                {
                    "role": "system",
                    "content": 'Tools: [{"name":"submit"}]\nReturn only a JSON function call.',
                },
                {
                    "role": "tool",
                    "content": '{"status":"ok","current_goal":"finish the task"}',
                },
            ]
        )
        self.assertEqual(
            rendered,
            'System: Tools: [{"name":"submit"}]\n'
            'Return only a JSON function call.\n\n'
            'User: Function output: {\n'
            '  "status": "ok",\n'
            '  "current_goal": "finish the task"\n'
            '}\n\n'
            'Assistant: ```json\n',
        )
        self.assertEqual(rendered.count("Assistant: ```json"), 1)

    def test_final_prefill_decoder_consumes_only_protocol_boundary(self):
        self.assertEqual(
            consume_final_prefill_boundary(">\n  RWKV answer.\n"),
            "  RWKV answer.\n",
        )
        self.assertEqual(
            consume_final_prefill_boundary("RWKV answer without boundary"),
            "RWKV answer without boundary",
        )

    def test_final_output_cleanup_and_stop_contracts_do_not_exist(self):
        self.assertFalse(hasattr(prompt_contract, "clean_final_continuation"))
        self.assertFalse(hasattr(prompt_contract, "FINAL_CONTINUATION_STOP_SUFFIXES"))

    def test_native_tool_envelope_is_adapted_without_changing_model_choice(self):
        payload = _canonicalize_tool_payload(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"test query"}',
                        },
                    }
                ]
            }
        )
        self.assertEqual(payload["name"], "web_search")
        self.assertEqual(payload["arguments"], {"query": "test query"})
        self.assertEqual(payload["call_id"], "call_1")

    def test_json_parser_accepts_only_the_prefilled_object_tail(self):
        payload = _extract_json_object('"name":"web_search","arguments":{"query":"rwkv"}}')
        self.assertEqual(payload, {"name": "web_search", "arguments": {"query": "rwkv"}})
        with self.assertRaises(ValueError):
            _extract_json_object("the model did not return a function call")

    def test_json_envelope_normalizer_accepts_only_lossless_common_formats(self):
        samples = {
            '{"name":"read_file","arguments":{}}': "json_object",
            '"name":"read_file","arguments":{}}': "prefilled_object_tail",
            '```json\n{"name":"read_file","arguments":{}}\n```': "json_fence",
            'Assistant: ```json\n{"name":"read_file","arguments":{}}': (
                "assistant_prefix+json_fence"
            ),
        }
        for raw, expected_format in samples.items():
            with self.subTest(raw=raw):
                normalized = normalize_json_object_envelope(raw)
                self.assertEqual(normalized.payload["name"], "read_file")
                self.assertEqual(normalized.input_format, expected_format)

    def test_json_envelope_normalizer_rejects_ambiguous_or_repaired_content(self):
        invalid = (
            'explanation {"name":"read_file","arguments":{}}',
            '{"name":"read_file","arguments":{}} {"name":"submit","arguments":{}}',
            "{'name':'read_file','arguments':{}}",
            '{"name":"read_file","arguments":{"path":"unterminated}}',
            '[{"name":"read_file","arguments":{}},{"name":"submit","arguments":{}}]',
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_json_object_envelope(raw, allow_singleton_array=True)

    def test_no_evidence_is_context_for_rwkv_not_a_controller_refusal(self):
        llm = _PromptFakeLLM("  RWKV decides what to say.\n")
        result = synthesize_retrieval_answer(
            "确认一个事实",
            {"query": "确认一个事实", "results": [], "citation_refs": []},
            llm=llm,
        )
        self.assertIn("No source text was retrieved.", llm.prompt)
        self.assertEqual(
            llm.stop,
            (
                "\nUser:",
                "\nSystem:",
                "\nAssistant:",
            ),
        )
        self.assertEqual(result["content"], llm.content)
        self.assertEqual(result["mode"], "rwkv_final")
        self.assertEqual(result["answer_quality"], {})


if __name__ == "__main__":
    unittest.main()
