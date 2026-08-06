import unittest
from types import SimpleNamespace

from agent.planner import _canonicalize_tool_payload, _extract_json_object
from agent.retrieval_synthesis import synthesize_retrieval_answer
from utils.rwkv_prompt import (
    FINAL_CONTINUATION_STOP_SUFFIXES,
    assistant_json_prefix,
    build_final_continuation_prompt,
    clean_final_continuation,
    tool_call_prefix,
)


class _PromptFakeLLM:
    provider = "local_13b"

    def __init__(self, content="无法确认"):
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
        self.assertTrue(prompt.endswith("### Assistant"))
        self.assertTrue(prompt.startswith("### User\n"))

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

    def test_json_parser_accepts_only_the_prefilled_object_tail(self):
        payload = _extract_json_object('"name":"web_search","arguments":{"query":"rwkv"}}')
        self.assertEqual(payload, {"name": "web_search", "arguments": {"query": "rwkv"}})
        with self.assertRaises(ValueError):
            _extract_json_object("the model did not return a function call")

    def test_final_cleanup_removes_generated_transcript_boundary_only(self):
        self.assertEqual(
            clean_final_continuation("答案\n### Assistant\n### User\n后续污染"),
            "答案",
        )
        self.assertEqual(clean_final_continuation("### Assistant\n<think></think>答案"), "答案")

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

    def test_no_evidence_is_explicitly_marked_for_model(self):
        llm = _PromptFakeLLM()
        result = synthesize_retrieval_answer(
            "确认一个事实",
            {"query": "确认一个事实", "results": [], "citation_refs": []},
            llm=llm,
        )
        self.assertIn("NO_USABLE_EVIDENCE", llm.prompt)
        self.assertEqual(tuple(llm.stop), FINAL_CONTINUATION_STOP_SUFFIXES)
        self.assertEqual(result["content"], "无法确认")


if __name__ == "__main__":
    unittest.main()
