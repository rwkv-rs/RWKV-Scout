import unittest
from types import SimpleNamespace

from agent.planner import _canonicalize_tool_payload
from agent.retrieval_synthesis import synthesize_retrieval_answer
from utils.rwkv_prompt import (
    FINAL_CONTINUATION_STOP_SUFFIXES,
    build_final_continuation_prompt,
    clean_final_continuation,
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
        self.assertTrue(prompt.endswith("Assistant: <think></think>\n"))
        self.assertTrue(prompt.startswith("User: "))

    def test_final_cleanup_removes_generated_transcript_boundary_only(self):
        self.assertEqual(
            clean_final_continuation("答案\nAssistant: User: 后续污染"),
            "答案",
        )
        self.assertEqual(clean_final_continuation("Assistant: <think></think>答案"), "答案")

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
