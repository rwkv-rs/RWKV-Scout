import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.page_evidence import (
    build_page_chunks,
    extract_single_page_evidence,
    parse_chunk_candidate,
)
from tools.web_search_keyless import search_web_keyless


class _FakeLLM:
    def __init__(self):
        self.prompts = []

    def text_completion(self, prompt, max_tokens=None):
        self.prompts.append((prompt, max_tokens))
        return SimpleNamespace(
            content=json.dumps(
                {
                    "supported": True,
                    "facts": ["深圳地铁一号线经过罗湖、老街和机场东。"],
                    "quote": "深圳地铁一号线经过罗湖、老街和机场东。",
                },
                ensure_ascii=False,
            )
        )


class PageEvidenceTests(unittest.TestCase):
    def test_page_is_split_into_independent_chunks(self):
        page = "\n".join(f"第{i}段：深圳地铁一号线站点信息。" for i in range(80))
        chunks = build_page_chunks(page, max_tokens=120, overlap_ratio=0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([item["index"] for item in chunks], list(range(len(chunks))))
        self.assertTrue(all(item["text"] for item in chunks))

    def test_oversized_single_sentence_respects_chunk_window(self):
        page = "深圳地铁一号线站点信息：" + "、".join(f"站点{i}" for i in range(1200))
        chunks = build_page_chunks(page, max_tokens=128, overlap_ratio=0)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(item["token_count"] <= 128 for item in chunks))

    def test_each_chunk_gets_one_parallel_candidate_prompt(self):
        page = {
            "title": "线路资料",
            "url": "https://example.com/line1",
            "page_excerpt": "\n".join(f"第{i}段：深圳地铁一号线站点信息。" for i in range(80)),
        }
        llm = _FakeLLM()
        evidence = extract_single_page_evidence(
            query="深圳地铁一号线有哪些站点",
            page=page,
            llm=llm,
            max_chunk_tokens=120,
        )
        self.assertEqual(len(llm.prompts), evidence["chunk_count"])
        self.assertTrue(all("网页正文片段" in prompt for prompt, _ in llm.prompts))
        self.assertTrue(all(max_tokens == 8192 for _, max_tokens in llm.prompts))
        self.assertGreaterEqual(len(evidence["candidates"]), 1)
        self.assertEqual(evidence["parallel_candidate"]["strategy"], "one-RWKV-call-per-chunk")
        self.assertEqual(evidence["parallel_candidate"]["completed_calls"], evidence["chunk_count"])
        self.assertNotIn("第79段", evidence["compact_facts"])

    def test_plain_candidate_is_not_allowed_to_be_a_tool_call(self):
        candidate = parse_chunk_candidate(
            '{"supported":true,"facts":["事实"],"quote":"原文"}',
            {"chunk_id": "chunk-1", "index": 0, "text": "事实", "token_count": 2},
        )
        self.assertTrue(candidate["supported"])
        self.assertEqual(candidate["facts"], ["事实"])
        self.assertNotIn("name", candidate)

    def test_agentic_search_never_fetches_multiple_pages(self):
        html = """
        <html><body>
          <h2><a href="https://example.com/a">深圳地铁一号线官方页面</a></h2>
          <p>站点摘要</p>
        </body></html>
        """
        with (
            patch("tools.web_search_keyless.fetch_text", return_value=html),
            patch("tools.web_search_keyless._page_excerpt", side_effect=AssertionError("search must not fetch pages")),
        ):
            result = json.loads(
                search_web_keyless(
                    "深圳地铁一号线官方页面",
                    max_results=1,
                    fetch_pages=3,
                    agentic_tool_loop=True,
                )
            )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["page_excerpt"], "")
        self.assertEqual(result["results"][0]["url"], "https://example.com/a")


if __name__ == "__main__":
    unittest.main()
