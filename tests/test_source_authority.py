import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.planner import Planner
from agent.retrieval_synthesis import (
    _clean_answer,
    build_evidence_context,
    synthesize_retrieval_answer,
)
from tools.builtin import load_builtin_tools
from tools.registry import ToolRegistry
from tools.web_search_generic import _merge_candidates, web_search
from utils.evidence_validation import build_evidence_validation
from utils.source_authority import (
    authority_for_url,
    infer_candidate_authority_domains,
    resolve_source_policy,
)


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

    def test_cross_script_official_sitemap_admits_one_strong_topic_for_fetch(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["nginx.org"],
        }
        url = "https://nginx.org/en/docs/http/websocket.html"
        candidates = _merge_candidates(
            "site:nginx.org nginx websocket \u53cd\u4ee3\u914d\u7f6e\u5b98\u65b9\u5199\u6cd5",
            [
                {
                    "provider": "official site adapter",
                    "results": [
                        {
                            "title": "en docs http websocket",
                            "url": url,
                            "snippet": "Official URL discovered from the site's sitemap.",
                            "source": "official sitemap adapter",
                        }
                    ],
                }
            ],
            task_plan=plan,
            constraint_query="nginx websocket \u53cd\u4ee3\u914d\u7f6e\u5b98\u65b9\u5199\u6cd5",
        )

        self.assertEqual(candidates[0]["url"], url)
        relevance = candidates[0]["query_relevance"]
        self.assertTrue(relevance["related"])
        self.assertEqual(relevance["required_term_hits"], 1)
        self.assertEqual(
            relevance["admission_basis"],
            "official_sitemap_cross_script_topic",
        )

    def test_official_domain_satisfies_policy(self):
        authority = authority_for_url(
            "https://www.miit.gov.cn/report",
            "latest MIIT telecommunications industry statistics",
            {"task_plan": {"source_policy": "official_required", "required_domains": ["miit.gov.cn"]}},
        )
        self.assertTrue(authority["satisfied"])
        self.assertEqual(authority["label"], "official_required")

    def test_global_academic_domain_conventions_are_institutional(self):
        for url in (
            "https://ocean.example.ac.uk/research",
            "https://lab.example.edu.cn/report",
            "https://agency.example.go.jp/bulletin",
        ):
            with self.subTest(url=url):
                authority = authority_for_url(url, "research report", {})
                self.assertEqual(authority["label"], "institutional")
                self.assertTrue(authority["satisfied"])

    def test_explicit_domain_works_without_a_known_product_profile(self):
        policy = resolve_source_policy(
            "Use the official page at docs.djangoproject.com for this date",
            {"task_plan": {"source_policy": "official_required"}},
        )
        self.assertEqual(policy["required_domains"], ["docs.djangoproject.com"])
        self.assertEqual(policy["domain_source"], "explicit_query")
        self.assertTrue(policy["required"])

    def test_unseen_official_domain_can_be_bootstrapped_by_strong_signals(self):
        inferred = infer_candidate_authority_domains(
            "Django 5.2 official release date",
            [
                {
                    "provider": "provider-a",
                    "results": [
                        {
                            "url": "https://docs.djangoproject.com/en/5.2/releases/5.2/",
                            "title": "Django documentation | Django 5.2 release notes",
                            "snippet": "Official documentation for the Django project.",
                        },
                        {
                            "url": "https://example-news.com/django-release",
                            "title": "Django release explained",
                            "snippet": "A third-party summary.",
                        },
                    ],
                },
                {
                    "provider": "provider-b",
                    "results": [
                        {
                            "url": "https://docs.djangoproject.com/en/5.2/",
                            "title": "Django documentation",
                            "snippet": "Project documentation.",
                        }
                    ],
                },
            ],
        )
        self.assertEqual(inferred[0]["domain"], "docs.djangoproject.com")
        self.assertNotIn("example-news.com", [row["domain"] for row in inferred])

    def test_unverified_hostname_alias_does_not_satisfy_policy(self):
        authority = authority_for_url(
            "https://docs.example-project.test/release",
            "official project documentation",
            {
                "task_plan": {
                    "source_policy": "official_required",
                    "required_domains": ["legacy.example-project.test"],
                }
            },
        )
        self.assertFalse(authority["satisfied"])
        self.assertEqual(authority["label"], "third_party")

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
                        "required_domains": ["security.ubuntu.com"],
                    }
                ],
            }
        )
        self.assertEqual(plan["task_mode"], "latest_list")
        self.assertEqual(plan["source_policy"], "official_required")
        self.assertEqual(plan["required_domains"], ["ubuntu.com"])
        self.assertEqual(
            plan["atomic_points"][0]["required_domains"],
            ["security.ubuntu.com"],
        )
        self.assertEqual(plan["max_items"], 5)

    def test_web_search_scopes_official_adapter_and_hard_query_to_active_claim(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["docs.djangoproject.com", "python.org"],
            "answer_requirements": [{"type": "date", "label": "all dates"}],
            "atomic_points": [
                {
                    "id": "P1",
                    "task": "Django 5.2 official release date",
                    "answer_requirements": [{"type": "date", "label": "Django date"}],
                },
                {
                    "id": "P2",
                    "task": "Python 3.13 official release date",
                    "answer_requirements": [{"type": "date", "label": "Python date"}],
                },
            ],
        }
        empty = {"status": "no_results", "provider": "test", "results": []}
        with (
            patch("tools.web_search_generic.search_web_keyless", return_value=empty),
            patch("tools.web_search_generic.search_web_tavily", return_value=empty),
            patch(
                "tools.web_search_generic.discover_official_urls",
                return_value={"status": "no_results", "provider": "official site adapter", "results": []},
            ) as discover,
            patch("tools.web_search_generic.append_task_event"),
        ):
            web_search(
                "Python 3.13 release date",
                task_plan=plan,
                task_point_id="P2",
                original_goal="Compare Django 5.2 and Python 3.13 release dates",
                task_id="CLAIM_DOMAIN_SCOPE_TEST",
            )

        args, kwargs = discover.call_args
        self.assertEqual(args[1], ["python.org"])
        self.assertEqual(kwargs["constraint_query"], "Python 3.13 official release date")
        self.assertEqual(
            kwargs["answer_requirements"],
            [{"type": "date", "label": "Python date"}],
        )

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

    def test_context_keeps_retrieved_chunks_without_answer_shape_rules(self):
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
                            {
                                "chunk_id": "chunk-1",
                                "supported": True,
                                "source_grounded": True,
                                "facts": ["MODEL ITEM MUST NOT ENTER CONTEXT"],
                                "quote": "Item A\nItem B\nItem C",
                            }
                        ],
                    }
                ],
            },
            constraints={"task_plan": plan},
        )
        self.assertIn("Item A", context["text"])
        self.assertIn("Item B", context["text"])
        self.assertIn("Item C", context["text"])
        self.assertNotIn("MODEL ITEM MUST NOT ENTER CONTEXT", context["text"])
        self.assertIn("Navigation and unrelated archive text", context["text"])

    def test_clean_answer_does_not_rewrite_rwkv_output(self):
        answer = "**P1** – internal routing\n**Answer:**\nUsable answer [S1]"
        self.assertEqual(_clean_answer(answer), answer)

    def test_latest_list_candidate_ranking_prefers_newest_dated_official_url(self):
        plan = {
            "task_mode": "latest_list",
            "source_policy": "official_required",
            "required_domains": ["cisa.gov"],
        }
        candidates = _merge_candidates(
            "latest CISA KEV addition",
            [
                {
                    "provider": "official site adapter",
                    "results": [
                        {
                            "title": "CISA adds three vulnerabilities",
                            "url": "https://www.cisa.gov/news-events/alerts/2026/08/04/older",
                            "snippet": "CISA KEV catalog addition",
                        },
                        {
                            "title": "CISA adds one vulnerability",
                            "url": "https://www.cisa.gov/news-events/alerts/2026/08/05/newer",
                            "snippet": "CISA KEV catalog addition",
                        },
                    ],
                }
            ],
            task_plan=plan,
        )
        self.assertIn("/2026/08/05/", candidates[0]["url"])

    def test_community_policy_fetches_threads_instead_of_board_navigation(self):
        plan = {
            "source_policy": "community_required",
            "required_domains": ["v2ex.com"],
        }
        candidates = _merge_candidates(
            "V2EX macOS 升级体验",
            [
                {
                    "provider": "keyless",
                    "results": [
                        {
                            "title": "macOS board",
                            "url": "https://www.v2ex.com/go/macos",
                            "snippet": "首页 注册 登录 RSS JSON Feed",
                        },
                        {
                            "title": "升级之后的真实体验",
                            "url": "https://www.v2ex.com/t/1229637",
                            "snippet": "用户讨论升级后的兼容性和续航。",
                        },
                    ],
                }
            ],
            task_plan=plan,
        )
        self.assertEqual([row["url"] for row in candidates], ["https://www.v2ex.com/t/1229637"])

    def test_official_adapter_relevance_survives_provider_candidate_merge(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["fastapi.tiangolo.com"],
        }
        candidates = _merge_candidates(
            "site:fastapi.tiangolo.com FastAPI latest stable version 2026",
            [
                {
                    "provider": "official site adapter",
                    "results": [
                        {
                            "title": "release notes",
                            "url": "https://fastapi.tiangolo.com/release-notes/",
                            "snippet": "Official URL discovered from the site's sitemap.",
                            "discovery_score": 9.5,
                        },
                        {
                            "title": "https://fastapi.tiangolo.com/advanced/graphql/",
                            "url": "https://fastapi.tiangolo.com/advanced/graphql/",
                            "snippet": "Official-site link.",
                            "discovery_score": 3.0,
                        },
                    ],
                }
            ],
            task_plan=plan,
        )
        self.assertEqual(candidates[0]["url"], "https://fastapi.tiangolo.com/release-notes/")

    def test_provider_private_score_cannot_displace_raw_topic_match(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["postgresql.org"],
        }
        candidates = _merge_candidates(
            "site:postgresql.org PostgreSQL jsonb index example official documentation",
            [
                {
                    "provider": "official site adapter",
                    "results": [
                        {
                            "title": "docs current app-reindexdb",
                            "url": "https://www.postgresql.org/docs/current/app-reindexdb.html",
                            "snippet": "Official URL discovered from the site's sitemap.",
                            "discovery_score": 99.0,
                        }
                    ],
                },
                {
                    "provider": "Tavily API",
                    "results": [
                        {
                            "title": "PostgreSQL Documentation: JSON Types",
                            "url": "https://www.postgresql.org/docs/current/datatype-json.html",
                            "snippet": "The default GIN operator class for jsonb supports containment queries.",
                        },
                        {
                            "title": "PostgreSQL Documentation: Partial Indexes",
                            "url": "https://www.postgresql.org/docs/current/indexes-partial.html",
                            "snippet": "An official index example and documentation reference.",
                        },
                        {
                            "title": "PostgreSQL Documentation 9.5: JSON Functions",
                            "url": "https://www.postgresql.org/docs/9.5/functions-json.html",
                            "snippet": "An older jsonb index example and documentation reference.",
                        }
                    ],
                },
            ],
            task_plan=plan,
            constraint_query="PostgreSQL jsonb索引怎么写，官网例子",
        )

        self.assertEqual(
            candidates[0]["url"],
            "https://www.postgresql.org/docs/current/datatype-json.html",
        )
        self.assertTrue(candidates[0]["query_relevance"]["raw_threshold_satisfied"])
        self.assertEqual(candidates[0]["query_relevance"]["constraint_topic_hits"], ["jsonb"])
        self.assertEqual(candidates[0]["query_relevance"]["canonical_path_rank"], 2)

    def test_explicit_api_documentation_request_prefers_api_path_over_blog(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["nodejs.org"],
        }
        candidates = _merge_candidates(
            "site:nodejs.org nodejs fetch API stable official documentation",
            [
                {
                    "provider": "search provider",
                    "results": [
                        {
                            "title": "Migrate from Axios to WHATWG Fetch",
                            "url": "https://nodejs.org/en/blog/migrations/axios-to-fetch",
                            "snippet": "Node.js Fetch API stable status and documentation.",
                        },
                        {
                            "title": "Node.js globals: fetch",
                            "url": "https://nodejs.org/api/globals.html",
                            "snippet": "Node.js Fetch API status in the official documentation.",
                        },
                    ],
                }
            ],
            task_plan=plan,
            constraint_query="node自带fetch现在稳定了吗 看API文档",
        )

        self.assertEqual(candidates[0]["url"], "https://nodejs.org/api/globals.html")
        by_url = {item["url"]: item for item in candidates}
        self.assertEqual(
            by_url["https://nodejs.org/api/globals.html"]["query_relevance"]["canonical_path_rank"],
            3,
        )
        self.assertEqual(
            by_url["https://nodejs.org/en/blog/migrations/axios-to-fetch"]["query_relevance"]["canonical_path_rank"],
            0,
        )

    def test_duplicate_url_recomputes_relevance_from_the_merged_rich_snippet(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["postgresql.org"],
        }
        current_url = "https://www.postgresql.org/docs/current/datatype-json.html"
        candidates = _merge_candidates(
            "site:postgresql.org PostgreSQL jsonb index example official documentation",
            [
                {
                    "provider": "official site adapter",
                    "results": [
                        {
                            "title": "PostgreSQL jsonb documentation",
                            "url": current_url,
                            "snippet": "Official URL discovered from the sitemap.",
                            "discovery_score": 1.0,
                        }
                    ],
                },
                {
                    "provider": "search provider",
                    "results": [
                        {
                            "title": "PostgreSQL Documentation: JSON Types",
                            "url": current_url,
                            "snippet": "Official jsonb index example using CREATE INDEX and GIN.",
                        },
                        {
                            "title": "PostgreSQL 9.4 JSONB indexing wiki",
                            "url": "https://wiki.postgresql.org/wiki/JSONB_indexing",
                            "snippet": "A jsonb index example for PostgreSQL 9.4.",
                        },
                    ],
                },
            ],
            task_plan=plan,
            constraint_query="PostgreSQL jsonb索引怎么写，官网例子",
        )

        self.assertEqual(candidates[0]["url"], current_url)
        relevance = candidates[0]["query_relevance"]
        self.assertTrue(relevance["raw_threshold_satisfied"])
        self.assertIn("index", relevance["term_hits"])
        self.assertIn("jsonb", relevance["term_hits"])

    def test_runtime_does_not_rewrite_source_copying_model_output(self):
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
        self.assertEqual(result["content"], body)
        self.assertEqual(result["model_output"], body)
        self.assertEqual(result["answer_quality"], {})

    def test_official_policy_does_not_suppress_material_from_final_rwkv(self):
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
        self.assertEqual(len(result["selected_evidence"]), 1)
        self.assertIn("Third-party MIIT report", llm_prompt_text(llm))
        self.assertEqual(result["content"], llm.content)
        self.assertEqual(result["answer_quality"], {})

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

        self.assertTrue(result["selected_evidence"])
        self.assertIn("WebSocket proxying", llm_prompt_text(llm))

    def test_user_explicit_official_domain_gate_skips_third_party_page_fetches(self):
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
            patch(
                "tools.web_search_generic.discover_official_urls",
                return_value={"status": "no_results", "provider": "official site adapter", "results": []},
            ),
            patch("tools.web_search_generic._fetch_candidate") as fetch_candidate,
            patch("tools.web_search_generic.append_task_event"),
        ):
            result = json.loads(
                web_search(
                    "latest MIIT official statistic site:miit.gov.cn",
                    task_plan=plan,
                    task_id="SOURCE_AUTHORITY_GATE_TEST",
                )
            )
        self.assertTrue(result["authority_missing"])
        self.assertFalse(result["evidence_ready"])
        fetch_candidate.assert_not_called()

    def test_model_planned_domain_is_a_discovery_hypothesis_not_a_hard_domain(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["miit.gov.cn"],
        }
        empty = {
            "status": "no_results",
            "provider": "test",
            "results": [],
        }
        with (
            patch("tools.web_search_generic.search_web_keyless", return_value=empty),
            patch("tools.web_search_generic.search_web_tavily", return_value=empty),
            patch(
                "tools.web_search_generic.discover_official_urls",
                return_value={
                    "status": "no_results",
                    "provider": "official site adapter",
                    "results": [],
                },
            ) as discover,
            patch("tools.web_search_generic._fetch_candidate") as fetch_candidate,
            patch("tools.web_search_generic.append_task_event"),
        ):
            result = json.loads(
                web_search(
                    "latest MIIT official statistic",
                    task_plan=plan,
                    task_id="SOURCE_AUTHORITY_HYPOTHESIS_TEST",
                )
            )

        self.assertNotIn("required_domains", result)
        self.assertEqual(
            result["source_resolution"]["domain_source"],
            "model_hypothesis_unverified",
        )
        self.assertEqual(result["source_resolution"]["preferred_domains"], ["miit.gov.cn"])
        self.assertEqual(discover.call_args.args[1], ["miit.gov.cn"])
        fetch_candidate.assert_not_called()

    def test_unseen_official_task_bootstraps_domain_before_page_fetch(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": [],
        }
        discovered = {
            "status": "ok",
            "provider": "provider",
            "results": [
                {
                    "title": "ExampleProject official documentation",
                    "url": "https://docs.exampleproject.org/releases/current/",
                    "snippet": "Official release documentation for ExampleProject.",
                }
            ],
        }
        official = {
            "status": "ok",
            "provider": "official site adapter",
            "results": discovered["results"],
        }
        body = "ExampleProject official release documentation states the requested fact. " * 4
        record = {
            "title": "ExampleProject official documentation",
            "url": "https://docs.exampleproject.org/releases/current/",
            "content": body,
            "source_excerpt": body,
            "evidence_origin": "fetched_page_body",
            "body_verified": True,
        }
        with (
            patch("tools.web_search_generic.search_web_keyless", return_value=discovered),
            patch("tools.web_search_generic.search_web_tavily", return_value=discovered),
            patch("tools.web_search_generic.discover_official_urls", return_value=official) as discover,
            patch(
                "tools.web_search_generic._fetch_candidate",
                return_value={"status": "ok", "results": []},
            ),
            patch(
                "tools.web_search_generic._compact_page",
                return_value=(record, {"status": "ok"}),
            ),
            patch("tools.web_search_generic.append_task_event"),
        ):
            result = json.loads(
                web_search(
                    "ExampleProject official release date",
                    task_plan=plan,
                    task_id="UNSEEN_AUTHORITY_BOOTSTRAP_TEST",
                )
            )
        self.assertEqual(plan["required_domains"], [])
        self.assertEqual(result["source_resolution"]["domain_source"], "provider_bootstrap")
        self.assertTrue(result["evidence_ready"])
        discover.assert_called_once()

    def test_verified_sitemap_alias_reaches_fetch_and_authority_gate(self):
        plan = {
            "source_policy": "official_required",
            "required_domains": ["project.example"],
        }
        official = {
            "status": "ok",
            "provider": "official site adapter",
            "resolved_domains": ["project.example", "project-docs.example"],
            "domain_aliases": [
                {
                    "requested_domain": "project.example",
                    "canonical_domain": "project-docs.example",
                    "verification": "dominant_https_sitemap_host",
                }
            ],
            "results": [
                {
                    "title": "Project 5.2 released",
                    "url": "https://project-docs.example/news/project-52-released/",
                    "snippet": "Official Project 5.2 release announcement.",
                    "source": "official sitemap adapter",
                }
            ],
        }
        body = "Project 5.2 was officially released on April 2, 2025. " * 3
        record = {
            "title": "Project 5.2 released",
            "url": "https://project-docs.example/news/project-52-released/",
            "content": body,
            "source_excerpt": body,
            "evidence_origin": "fetched_page_body",
            "body_verified": True,
        }
        empty = {"status": "no_results", "provider": "test", "results": []}
        with (
            patch("tools.web_search_generic.search_web_keyless", return_value=empty),
            patch("tools.web_search_generic.search_web_tavily", return_value=empty),
            patch("tools.web_search_generic.discover_official_urls", return_value=official),
            patch("tools.web_search_generic._fetch_candidate", return_value={"status": "ok", "results": []}) as fetch,
            patch("tools.web_search_generic._compact_page", return_value=(record, {"status": "ok"})),
            patch("tools.web_search_generic.append_task_event"),
        ):
            result = json.loads(
                web_search(
                    "Project 5.2 official release date",
                    task_plan=plan,
                    task_id="SITEMAP_ALIAS_TEST",
                )
            )

        self.assertTrue(result["evidence_ready"])
        self.assertEqual(
            result["source_resolution"]["domain_source"],
            "official_sitemap_canonicalization",
        )
        self.assertEqual(plan["required_domains"], ["project.example"])
        fetch.assert_called_once()

    def test_cached_page_is_reextracted_for_a_different_claim_without_refetch(self):
        url = "https://example.org/releases"
        cached = {
            "url": url,
            "title": "Python release history",
            "snippet": "Python 3.13 release date and release history",
            "source": "test",
            "content": "Python 3.13 release evidence. " * 20,
            "source_excerpt": "Python 3.13 release evidence. " * 20,
            "source_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "index": 0,
                    "text": "Python 3.13 was released on October 7, 2024. " * 8,
                }
            ],
            "evidence_origin": "fetched_page_body",
            "body_verified": True,
            "locator_claim_id": "P1",
        }
        state = SimpleNamespace(
            retrieval=SimpleNamespace(
                sources={url: cached},
                sources_by_claim={"P1": {url: cached}},
                attempted_urls={url},
                url_attempt_counts={url: 1},
            )
        )
        provider = {
            "status": "ok",
            "provider": "test",
            "results": [
                {
                    "title": cached["title"],
                    "url": url,
                    "snippet": cached["snippet"],
                }
            ],
        }
        rebound = {
            **cached,
            "locator_claim_id": "P2",
            "claim_ids": ["P2"],
        }
        plan = {
            "source_policy": "open_web",
            "required_domains": [],
            "atomic_points": [
                {"id": "P1", "task": "Django release date"},
                {"id": "P2", "task": "Python 3.13 release date"},
            ],
        }
        with (
            patch("tools.web_search_generic.search_web_keyless", return_value=provider),
            patch("tools.web_search_generic.search_web_tavily", return_value=provider),
            patch("tools.web_search_generic._fetch_candidate") as fetch,
            patch(
                "tools.web_search_generic._compact_page",
                return_value=(rebound, {"status": "ok"}),
            ) as compact,
            patch("tools.web_search_generic.append_task_event"),
        ):
            result = json.loads(
                web_search(
                    "Python 3.13 release date",
                    task_plan=plan,
                    task_point_id="P2",
                    original_goal="Compare Django and Python release dates",
                    task_id="CLAIM_CACHE_REEXTRACT_TEST",
                    agent_state=state,
                )
            )

        fetch.assert_not_called()
        compact.assert_called_once()
        self.assertTrue(result["reextracted_cached_sources"])
        self.assertEqual(result["results"][0]["locator_claim_id"], "P2")


def llm_prompt_text(llm):
    # Keep the helper outside the test class so the assertion remains readable
    # while avoiding a dependency on a particular completion call count.
    return "\n".join(str(prompt) for prompt, _tokens, _kwargs in getattr(llm, "calls", []))


if __name__ == "__main__":
    unittest.main()
