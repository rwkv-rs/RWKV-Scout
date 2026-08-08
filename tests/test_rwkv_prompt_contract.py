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

    def test_json_prefix_matches_rwkv_skills_tool_continuation(self):
        self.assertEqual(
            assistant_json_prefix(enable_think=True, prefill_object=True),
            "### Assistant\n<think></think\n{",
        )
        self.assertEqual(
            assistant_json_prefix(enable_think=False, prefill_object=True),
            "### Assistant\n```json\n{",
        )
        self.assertEqual(tool_call_prefix(), "### Assistant\n**Tool Call:**\n")

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

    def test_no_evidence_is_context_for_rwkv_not_a_controller_refusal(self):
        llm = _PromptFakeLLM("  RWKV decides what to say.\n")
        result = synthesize_retrieval_answer(
            "确认一个事实",
            {"query": "确认一个事实", "results": [], "citation_refs": []},
            llm=llm,
        )
        self.assertIn("No source text was retrieved.", llm.prompt)
        self.assertIsNone(llm.stop)
        self.assertEqual(result["content"], llm.content)
        self.assertEqual(result["mode"], "rwkv_final")
        self.assertEqual(result["answer_quality"], {})


if __name__ == "__main__":
    unittest.main()
