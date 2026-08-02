import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.planner import Planner
from agent.orchestrator import Orchestrator
from agent.retrieval_synthesis import (
    _clean_answer,
    _enforce_latest_list_shape,
    build_evidence_context,
    synthesize_retrieval_answer,
)
from tools.builtin import load_builtin_tools
from tools.registry import ToolRegistry
from tools.web_search_generic import web_search
from utils.evidence_validation import build_evidence_validation
from utils.source_authority import authority_for_url


class SourceAuthorityTests(unittest.TestCase):
    class _FinalLLM:
        provider = "local_13b"

        def __init__(self, content="MIIT reported the statistic [S1]"):
            self.calls = []
            self.content = content

        def text_completion(self, prompt, max_tokens=0, **kwargs):
            self.calls.append((prompt, max_tokens, kwargs))
            return SimpleNamespace(content=self.content)

    def test_required_domain_is_not_satisfied_by_third_party_summary(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["miit.gov.cn"],
            "atomic_points": [
                {
                    "id": "P1",
                    "task": "MIIT official telecom statistics",
                    "objective": "find the official statistics",
                    "evidence_needed": ["official source"],
                    "acceptance_criteria": ["official domain"],
                }
            ],
        }
        report = build_evidence_validation(
            {
                "results": [
                    {
                        "url": "https://www.researchandmarkets.com/report",
                        "content": "MIIT telecom statistics 2025",
                        "evidence_origin": "fetched_page_body",
                        "body_verified": True,
                    }
                ]
            },
            query="latest MIIT telecommunications industry statistics",
            constraints={"task_plan": plan},
        )
        row = report["subquestion_coverage"][0]
        self.assertEqual(row["status"], "authority_missing")
        self.assertFalse(row["answerable"])

    def test_official_domain_satisfies_policy(self):
        authority = authority_for_url(
            "https://www.miit.gov.cn/report",
            "latest MIIT telecommunications industry statistics",
            {"task_plan": {"source_policy": "official_required", "required_domains": ["miit.gov.cn"]}},
        )
        self.assertTrue(authority["satisfied"])
        self.assertEqual(authority["label"], "official_required")

    def test_plan_keeps_task_scope_fields(self):
        plan = Planner._validate_task_plan(
            {
                "goal": "latest notices",
                "task_mode": "latest_list",
                "source_policy": "official_required",
                "required_domains": ["ubuntu.com"],
                "requested_fields": ["notice_id", "title"],
                "max_items": 5,
                "atomic_points": [
                    {
                        "id": "P1",
                        "task": "find notices",
                        "objective": "find latest notices",
                        "evidence_needed": ["notice rows"],
                        "acceptance_criteria": ["date ordered"],
                    }
                ],
            }
        )
        self.assertEqual(plan["task_mode"], "latest_list")
        self.assertEqual(plan["source_policy"], "official_required")
        self.assertEqual(plan["required_domains"], ["ubuntu.com"])
        self.assertEqual(plan["max_items"], 5)

    def test_model_cannot_override_runtime_context(self):
        load_builtin_tools()
        result = json.loads(
            ToolRegistry.execute(
                "web_search",
                {"query": "example", "task_id": "spoofed"},
                {"task_id": "real-task"},
                phase="ALL",
            )
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_class"], "tool_protocol")
        self.assertEqual(result["unknown_arguments"], ["task_id"])

    def test_latest_list_context_uses_bounded_candidate_facts(self):
        plan = {
            "task_mode": "latest_list",
            "max_items": 2,
            "requested_fields": ["id", "title"],
            "atomic_points": [
                {
                    "id": "P1",
                    "task": "latest notices",
                    "objective": "return the latest notice rows",
                    "evidence_needed": ["notice rows"],
                    "acceptance_criteria": ["at most two rows"],
                }
            ],
        }
        context = build_evidence_context(
            {
                "query": "latest notices",
                "results": [
                    {
                        "url": "https://ubuntu.com/security/notices",
                        "title": "Ubuntu security notices",
                        "content": "Item A\nItem B\nItem C\nNavigation and unrelated archive text " * 8,
                        "evidence_origin": "fetched_page_body",
                        "source_chunks": [
                            {"chunk_id": "chunk-1", "index": 0, "text": "Item A\nItem B\nItem C"},
                            {"chunk_id": "chunk-2", "index": 1, "text": "Navigation and unrelated archive text " * 8},
                        ],
                        "chunk_candidates": [
                            {"chunk_id": "chunk-1", "supported": True, "facts": ["Item A", "Item B", "Item C"]}
                        ],
                    }
                ],
            },
            constraints={"task_plan": plan},
        )
        self.assertIn("Item A", context["text"])
        self.assertIn("Item B", context["text"])
        self.assertNotIn("Item C", context["text"])
        self.assertNotIn("Navigation and unrelated archive text", context["text"])

    def test_clean_answer_removes_bold_planner_labels(self):
        cleaned = _clean_answer("**P1** – internal routing\n**Answer:**\nUsable answer [S1]")
        self.assertEqual(cleaned, "Usable answer [S1]")

    def test_latest_list_shape_caps_numbered_rows(self):
        answer = "1. A [S1]\n2. B [S1]\n3. C [S1]\n4. D [S1]"
        bounded = _enforce_latest_list_shape(
            answer,
            {"task_mode": "latest_list", "max_items": 2},
        )
        self.assertIn("1. A", bounded)
        self.assertIn("2. B", bounded)
        self.assertNotIn("3. C", bounded)
        self.assertNotIn("4. D", bounded)

    def test_source_copy_guard_does_not_publish_full_page_body(self):
        body = "The requested fact is explicitly stated in the captured page body. " * 80
        llm = self._FinalLLM(content=body)
        result = synthesize_retrieval_answer(
            "requested fact",
            {
                "query": "requested fact",
                "results": [
                    {
                        "url": "https://example.com/body",
                        "content": body,
                        "evidence_origin": "fetched_page_body",
                    }
                ],
            },
            llm=llm,
        )
        self.assertTrue(result["answer_quality"]["source_copy_guard_triggered"])
        self.assertIn("正文过长", result["content"])

    def test_official_missing_closes_final_answer_evidence_context(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["miit.gov.cn"],
            "atomic_points": [
                {
                    "id": "P1",
                    "task": "MIIT official statistic",
                    "objective": "confirm the official statistic",
                    "evidence_needed": ["official source"],
                    "acceptance_criteria": ["official domain"],
                }
            ],
        }
        llm = self._FinalLLM()
        result = synthesize_retrieval_answer(
            "latest MIIT official statistic",
            {
                "query": "latest MIIT official statistic",
                "results": [
                    {
                        "url": "https://www.researchandmarkets.com/report",
                        "title": "Third-party MIIT report",
                        "content": "MIIT reported the statistic as 42. This third-party page is not the official source. " * 4,
                        "evidence_origin": "fetched_page_body",
                    }
                ],
            },
            llm=llm,
            constraints={"task_plan": plan},
        )
        self.assertTrue(result["context_stats"]["final_evidence_suppressed"])
        self.assertEqual(result["selected_evidence"], [])
        self.assertNotIn("Third-party MIIT report", llm_prompt_text(llm))
        self.assertTrue(result["answer_quality"]["closed_world_boundary_enforced"])

    def test_official_body_is_not_suppressed_by_bilingual_coverage_miss(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["nginx.org", "nginx.com"],
            "atomic_points": [
                {
                    "id": "P1",
                    "task": "查找 nginx 官方文档中关于 WebSocket 反向代理的配置说明",
                    "objective": "获取官方文档中的配置示例和关键指令说明",
                },
                {
                    "id": "P2",
                    "task": "验证 nginx 官方文档中 WebSocket 反向代理配置的正确性",
                    "objective": "验证官方配置的正确性",
                },
            ],
        }
        llm = self._FinalLLM(content="Use the official WebSocket proxying example [S1]")
        result = synthesize_retrieval_answer(
            "nginx websocket 反向代理 官方文档",
            {
                "query": "nginx websocket 反向代理 官方文档",
                "results": [
                    {
                        "url": "https://nginx.org/en/docs/http/websocket.html",
                        "title": "WebSocket proxying",
                        "content": (
                            "## WebSocket proxying\n"
                            "The Upgrade and Connection headers have to be passed explicitly.\n"
                            "location /chat/ { proxy_pass http://backend; "
                            "proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; "
                            "proxy_set_header Connection \"upgrade\"; }\n"
                        )
                        * 4,
                        "evidence_origin": "fetched_page_body",
                        "body_verified": True,
                    }
                ],
            },
            llm=llm,
            constraints={"task_plan": plan},
        )

        self.assertFalse(result["context_stats"]["final_evidence_suppressed"])
        self.assertTrue(result["context_stats"]["official_body_available"])
        self.assertTrue(result["selected_evidence"])
        self.assertIn("WebSocket proxying", llm_prompt_text(llm))

    def test_review_reports_authority_missing_as_unfinished(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["miit.gov.cn"],
            "atomic_points": [
                {
                    "id": "P1",
                    "task": "official statistic",
                    "objective": "confirm the official statistic",
                    "evidence_needed": ["official source"],
                    "acceptance_criteria": ["official domain"],
                }
            ],
        }
        orchestrator = Orchestrator()
        orchestrator.state.task_id = "SOURCE_AUTHORITY_REVIEW_TEST"
        orchestrator.state.run_metadata = {"task_plan": plan}
        orchestrator._task_plan = plan
        with patch("agent.orchestrator.append_task_event"):
            review, _validation = orchestrator._build_engineering_evidence_review(
                "latest MIIT official statistic",
                [
                    (
                        "latest MIIT official statistic",
                        {
                            "status": "ok",
                            "results": [
                                {
                                    "url": "https://www.researchandmarkets.com/report",
                                    "content": "MIIT official statistic summary " * 8,
                                    "evidence_origin": "fetched_page_body",
                                }
                            ],
                        },
                    )
                ],
            )
        self.assertIn("P1", review["missing_point_ids"])
        self.assertEqual(review["evidence_state"], "partial_or_conflicted")

    def test_official_gate_skips_third_party_page_fetches(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["miit.gov.cn"],
        }
        provider_result = {
            "status": "ok",
            "results": [
                {
                    "title": "Third-party MIIT report",
                    "url": "https://www.researchandmarkets.com/report",
                    "snippet": "MIIT statistic",
                }
            ],
        }
        with (
            patch("tools.web_search_generic.search_web_keyless", return_value=provider_result),
            patch("tools.web_search_generic.search_web_tavily", return_value=provider_result),
            patch("tools.web_search_generic._fetch_candidate") as fetch_candidate,
            patch("tools.web_search_generic.append_task_event"),
        ):
            result = json.loads(
                web_search(
                    "latest MIIT official statistic",
                    task_plan=plan,
                    task_id="SOURCE_AUTHORITY_GATE_TEST",
                )
            )
        self.assertTrue(result["authority_missing"])
        self.assertFalse(result["evidence_ready"])
        fetch_candidate.assert_not_called()


def llm_prompt_text(llm):
    # Keep the helper outside the test class so the assertion remains readable
    # while avoiding a dependency on a particular completion call count.
    return "\n".join(str(prompt) for prompt, _tokens, _kwargs in getattr(llm, "calls", []))


if __name__ == "__main__":
    unittest.main()
