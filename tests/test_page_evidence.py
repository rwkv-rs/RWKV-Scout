import concurrent.futures
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.page_evidence import (
    _apply_deterministic_candidate_gates,
    _build_candidate_retry_prompt,
    _entity_status_record_candidates,
    _procedure_command_candidates,
    build_page_chunks,
    extract_single_page_evidence,
    parse_chunk_candidate,
    select_grounded_source_chunks,
)


def test_candidate_retry_instruction_stays_in_user_turn():
    prompt = "### User\nExtract one fact.\n\n### Assistant\n```json\n"
    repaired = _build_candidate_retry_prompt(prompt)

    user_turn, assistant_turn = repaired.rsplit("\n\n### Assistant\n", 1)
    assert "supported=false" in user_turn
    assert "no more than 6 facts" in user_turn
    assert assistant_turn == "```json\n"


def test_exact_query_focused_source_span_outranks_model_candidate_order():
    chunks = [
        {
            "chunk_id": "intro",
            "index": 0,
            "text": "NetworkPolicy is a Kubernetes resource for controlling traffic.",
        },
        {
            "chunk_id": "unrelated-example",
            "index": 1,
            "text": "This example selects frontend pods and allows traffic from monitoring.",
        },
        {
            "chunk_id": "default-deny",
            "index": 2,
            "text": (
                "Default deny ingress and egress policy:\n"
                "spec:\n  podSelector: {}\n  policyTypes:\n"
                "  - Ingress\n  - Egress"
            ),
        },
    ]
    candidates = [
        {
            "supported": True,
            "chunk_id": "unrelated-example",
            "quote": "allows traffic from monitoring",
        },
        {
            "supported": True,
            "chunk_id": "intro",
            "quote": "controlling traffic",
        },
    ]

    selected = select_grounded_source_chunks(
        "Kubernetes default deny ingress and egress NetworkPolicy YAML podSelector",
        chunks,
        candidates,
        max_chunks=3,
        task_plan={
            "answer_requirements": [
                {"type": "procedure", "requested_fields": ["exact YAML"]}
            ]
        },
        preferred_chunks=[chunks[2]],
    )

    assert selected[0]["chunk_id"] == "default-deny"
    assert "podSelector: {}" in selected[0]["text"]
    assert selected[0]["attention_rank"] == 1
    assert "query_focused_source_span" in selected[0]["attention_reasons"]
from tools.web_search_keyless import (
    _YahooResultParser,
    _looks_related,
    _provider_query,
    _unwrap_ddg_url,
    search_web_keyless,
)
from tools.web_search_generic import (
    _extract_direct_url,
    _extract_single_goal_url,
    _fetch_candidate,
    _merge_candidates,
    _model_extraction_diagnostics,
)
from utils.network_fetch import NetworkFetchError, fetch_text


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


class _GroundedFakeLLM:
    def __init__(self):
        self.prompts = []

    def text_completion(self, prompt, max_tokens=None):
        self.prompts.append((prompt, max_tokens))
        source_line = prompt.rsplit("\n\n### Assistant", 1)[0].splitlines()[-1]
        return SimpleNamespace(
            content=json.dumps(
                {
                    "supported": True,
                    "facts": [source_line],
                    "quote": source_line,
                },
                ensure_ascii=False,
            )
        )


def test_locator_quote_finishes_sentence_instead_of_cutting_a_word():
    quote = ("x " * 395) + "The two most common ions are chloride and sodium. Trailing material is unrelated."
    candidate = parse_chunk_candidate(
        json.dumps({"supported": True, "facts": ["salt fact"], "quote": quote}),
        {"chunk_id": "chunk-1", "index": 0, "text": quote, "token_count": 100},
    )

    assert candidate["quote_truncated"] is True
    assert candidate["quote"].endswith("sodium.")
    assert "chloride and sodium" in candidate["quote"]
    assert len(candidate["quote"]) <= 1040


def test_multiline_model_locator_keeps_structure_until_source_grounding():
    source = (
        "### Package manager\n"
        "Formatting-only source line.\n"
        "The installer also provides the package manager.\n"
        "To verify the installation, run:\n"
        "`package-manager --version`\n"
    )
    model_quote = (
        "Package manager\n\n"
        "The installer also provides the package manager.\n\n"
        "To verify the installation, run:\n"
        "`package-manager --version`"
    )
    chunk = {"chunk_id": "chunk-1", "index": 0, "text": source, "token_count": 40}
    parsed = parse_chunk_candidate(
        json.dumps(
            {
                "supported": True,
                "facts": ["The package-manager --version command verifies installation."],
                "quote": model_quote,
            }
        ),
        chunk,
    )

    assert "\n" in parsed["quote"]
    gated = _apply_deterministic_candidate_gates(
        "How do I install and verify this tool?",
        [chunk],
        [parsed],
        {},
    )[0]
    assert gated["supported"] is True
    assert gated["source_grounded"] is True
    assert gated["grounding_basis"] == "ordered_source_segments"
    assert gated["grounded_segment_count"] == 4
    assert gated["quote"].endswith("`package-manager --version`")


def test_current_plan_schema_activates_markdown_version_command_locator():
    source = (
        "## Install\n"
        "Install the tool by running:\n"
        "`uv tool install example`\n"
        "After installation, verify the installed toolchain with:\n"
        "`uv --version`\n"
        "For normal use you can also run:\n"
        "`uv run example`\n"
    )
    candidates = _procedure_command_candidates(
        "https://docs.example.org/get-started",
        [{"chunk_id": "chunk-1", "index": 0, "text": source, "token_count": 50}],
        {
            "requested_fields": ["installation_method", "version_check_command"],
            "atomic_points": [
                {
                    "task": "extract the installation and verification commands",
                    "objective": "report the exact commands from the source",
                    "evidence_needed": ["terminal command"],
                }
            ],
        },
    )

    assert candidates
    assert "uv --version" in candidates[0]["record"]["command"]
    assert "version_check_command" in candidates[0]["selection_reasons"]
    assert candidates[0]["quote"] in source


def test_entity_status_locator_accepts_glued_numeric_stability_badge():
    query = "Is package fetch stable?"
    candidates = _entity_status_record_candidates(
        query,
        [
            {
                "chunk_id": "chunk-1",
                "index": 0,
                "text": "# Fetch\nHistory\nIntroduced in: v1.0.0\nStability: 2Stable",
                "token_count": 20,
            }
        ],
        {"answer_requirements": [{"type": "status"}]},
    )

    assert len(candidates) == 1
    assert candidates[0]["deterministic_locator"] == "entity_status_record"
    assert "Stability: 2Stable" in candidates[0]["quote"]


class _NegativeLLM:
    def __init__(self):
        self.prompts = []

    def text_completion(self, prompt, max_tokens=None, stop=None):
        self.prompts.append((prompt, max_tokens))
        return SimpleNamespace(
            content='{"supported":false,"facts":[],"quote":""}',
            finish_reason="stop",
        )


class _PartiallyFailingLLM:
    def text_completion(self, prompt, max_tokens=None, stop=None):
        if "FAIL_CHUNK" in prompt:
            raise ConnectionError("temporary model connection loss")
        return SimpleNamespace(
            content=json.dumps(
                {
                    "supported": True,
                    "facts": ["SUPPORTED_CHUNK contains the requested fact."],
                    "quote": "SUPPORTED_CHUNK contains the requested fact.",
                }
            ),
            finish_reason="stop",
        )


class _RecoveringNegativeLLM:
    def __init__(self):
        self.failed_once = False

    def text_completion(self, prompt, max_tokens=None, stop=None):
        if "FAIL_ONCE" in prompt and not self.failed_once:
            self.failed_once = True
            raise ConnectionError("temporary model connection loss")
        return SimpleNamespace(
            content='{"supported":false,"facts":[],"quote":""}',
            finish_reason="stop",
        )


class _FixedCandidateLLM:
    def __init__(self, *, facts, quote):
        self.facts = list(facts)
        self.quote = quote

    def text_completion(self, prompt, max_tokens=None, stop=None):
        return SimpleNamespace(
            content=json.dumps(
                {"supported": True, "facts": self.facts, "quote": self.quote},
                ensure_ascii=False,
            ),
            finish_reason="stop",
        )


class PageEvidenceTests(unittest.TestCase):
    @staticmethod
    def _node_page() -> dict:
        return {
            "title": "Node.js v18 release",
            "url": "https://nodejs.org/en/blog/announcements/v18-release-announce",
            "page_excerpt": (
                "# Node.js v18 release\n"
                "Node.js v18.0.0 introduced a global\nfetch API as experimental.\n"
                + "This official release note documents runtime changes and compatibility details. " * 24
            ),
            "body_cleaned": True,
        }

    def test_ungrounded_positive_model_quote_is_rejected_before_merge(self):
        evidence = extract_single_page_evidence(
            query="Node.js v18 fetch API status",
            page=self._node_page(),
            llm=_FixedCandidateLLM(
                facts=["Node.js 自带 fetch 现在已经稳定"],
                quote="Node.js 自带 fetch 现在已经稳定",
            ),
            task_plan={"answer_requirements": [{"type": "status"}]},
        )

        candidate = evidence["chunk_candidates"][0]
        self.assertFalse(candidate["supported"])
        self.assertFalse(candidate["source_grounded"])
        self.assertEqual(candidate["rejection_reason"], "model_quote_not_grounded")
        self.assertEqual(candidate["model_quote"], "Node.js 自带 fetch 现在已经稳定")
        self.assertEqual(len(evidence["candidates"]), 1)
        self.assertEqual(
            evidence["candidates"][0]["deterministic_locator"],
            "entity_status_record",
        )
        self.assertIn("fetch API as experimental", evidence["compact_facts"])
        self.assertNotIn("现在已经稳定", evidence["compact_facts"])

    def test_grounded_model_quote_is_rewritten_to_exact_source_and_facts_stay_out_of_routing(self):
        model_quote = "Node.js v18.0.0 introduced a global fetch API as experimental."
        evidence = extract_single_page_evidence(
            query="Node.js v18 fetch API status",
            page=self._node_page(),
            llm=_FixedCandidateLLM(
                facts=["invented adjacent fact that is not in the page"],
                quote=model_quote,
            ),
            task_plan={"answer_requirements": [{"type": "status"}]},
        )

        candidate = evidence["chunk_candidates"][0]
        self.assertTrue(candidate["supported"])
        self.assertTrue(candidate["source_grounded"])
        self.assertEqual(candidate["model_quote"], model_quote)
        self.assertIn("\n", candidate["quote"])
        self.assertIn(candidate["quote"], self._node_page()["page_excerpt"])
        self.assertEqual(candidate["grounding_basis"], "normalized_whitespace")
        self.assertIn("fetch API as experimental", evidence["compact_facts"])
        self.assertNotIn("invented adjacent fact", evidence["compact_facts"])

    def test_post_gate_rejection_is_distinct_from_raw_model_negative(self):
        evidence = extract_single_page_evidence(
            query="Target product current release theme",
            page={
                "title": "Different product release notes",
                "url": "https://example.com/different-product",
                "page_excerpt": (
                    "Different Product version 4.4 introduces an unrelated festival. " * 24
                ),
                "body_cleaned": True,
            },
            llm=_FixedCandidateLLM(
                facts=["Target product version 4.4 has the unrelated festival theme."],
                quote="Target product version 4.4 has the unrelated festival theme.",
            ),
            task_plan={"answer_requirements": [{"type": "latest_version"}]},
        )

        self.assertEqual(evidence["valid_contract_count"], 1)
        self.assertEqual(evidence["negative_response_count"], 0)
        self.assertFalse(evidence["all_selected_chunks_valid_negative"])
        self.assertTrue(evidence["all_selected_chunks_semantically_rejected"])
        self.assertFalse(evidence["chunk_candidates"][0]["supported"])
        self.assertEqual(evidence["candidates"], [])

    def test_entity_status_history_is_located_when_model_translates_the_quote(self):
        body = (
            "# Global APIs\n"
            "### `fetch` #\n"
            "Added in: v17.5.0, v16.15.0\n"
            "| Version | Changes |\n"
            "| --- | --- |\n"
            "| v21.0.0 | No longer experimental. |\n"
            "| v18.0.0 | No longer behind `--experimental-fetch` CLI flag. |\n"
            "A browser-compatible implementation of the fetch function.\n"
            "### Other API\n"
            + ("Unrelated reference material for another global API. " * 18)
        )
        evidence = extract_single_page_evidence(
            query="node自带fetch现在稳定了吗 看API文档",
            page={
                "title": "Node.js globals",
                "url": "https://nodejs.org/api/globals.html",
                "page_excerpt": body,
                "body_cleaned": True,
            },
            llm=_FixedCandidateLLM(
                facts=["fetch 在 v21.0.0 中不再是实验性的"],
                quote="fetch 在 v21.0.0 中不再是实验性的",
            ),
            task_plan={"answer_requirements": [{"type": "status"}]},
        )

        deterministic = [
            row
            for row in evidence["chunk_candidates"]
            if row.get("deterministic_locator") == "entity_status_record"
        ]
        self.assertEqual(len(deterministic), 1)
        self.assertTrue(deterministic[0]["supported"])
        self.assertTrue(deterministic[0]["source_grounded"])
        self.assertIn("### `fetch` #", deterministic[0]["quote"])
        self.assertIn("v21.0.0 | No longer experimental", deterministic[0]["quote"])
        self.assertNotIn("Other API", deterministic[0]["quote"])

    def test_selector_rule_is_retained_separately_from_dynamic_command(self):
        command = (
            "pip3 install torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cu118"
        )
        cpu_branch = (
            "If you do not have a CUDA-capable system or do not require CUDA, "
            "choose Compute Platform: CPU."
        )
        body = (
            "## Start Locally\n"
            "Select your preferences and run the install command.\n"
            "PyTorch Build\nYour OS\nPackage\nLanguage\nCompute Platform\n"
            "CUDA 11.8\nCUDA 12.6\nCUDA 12.8\nCPU\n"
            f"{command}\n"
            "## Installing on Linux\n"
            f"{cpu_branch}\n"
            "To install PyTorch via pip on a CUDA-capable system, choose OS: Linux, "
            "Package: Pip, Language: Python and the CUDA version suited to your machine. "
            "Then run the command presented by the selector.\n"
            + ("Official installation prerequisites and verification notes. " * 16)
        )
        evidence = extract_single_page_evidence(
            query="PyTorch装CUDA版本该选哪个命令",
            page={
                "title": "Install PyTorch",
                "url": "https://pytorch.org/get-started/locally",
                "page_excerpt": body,
                "body_cleaned": True,
            },
            llm=_FixedCandidateLLM(facts=[cpu_branch], quote=cpu_branch),
            task_plan={
                "answer_requirements": [
                    {"type": "command"},
                    {"type": "selection"},
                ]
            },
        )

        selectors = [
            row
            for row in evidence["chunk_candidates"]
            if row.get("deterministic_locator") == "selection_record"
        ]
        commands = [
            row
            for row in evidence["chunk_candidates"]
            if row.get("deterministic_locator") == "procedure_command"
        ]
        self.assertTrue(selectors)
        overview = next(
            row for row in selectors if "Select your preferences" in row["quote"]
        )
        self.assertIn("CUDA 11.8", overview["quote"])
        self.assertNotIn(command, overview["quote"])
        cpu_rule = next(row for row in selectors if row["quote"].startswith("If you do not"))
        self.assertNotIn("do have", cpu_rule["quote"])
        self.assertTrue(
            any("CUDA-capable system, choose OS: Linux" in row["quote"] for row in selectors)
        )
        self.assertTrue(commands)
        self.assertEqual(commands[0]["record"]["command"], command)

    def test_complete_url_query_is_marked_for_direct_fetch(self):
        self.assertEqual(
            _extract_direct_url("https://docs.python.org/3/whatsnew/3.13.html"),
            "https://docs.python.org/3/whatsnew/3.13.html",
        )
        self.assertEqual(_extract_direct_url("summarize https://example.com/page"), "")
        self.assertEqual(
            _extract_single_goal_url("Summarize this page: https://example.com/page"),
            "https://example.com/page",
        )

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

    def test_long_page_bounds_model_prompts_but_keeps_every_source_chunk(self):
        page = {
            "title": "线路资料",
            "url": "https://example.com/line1",
            "page_excerpt": "\n".join(f"第{i}段：深圳地铁一号线站点信息。" for i in range(80)),
        }
        llm = _GroundedFakeLLM()
        evidence = extract_single_page_evidence(
            query="深圳地铁一号线有哪些站点",
            page=page,
            llm=llm,
            max_chunk_tokens=120,
        )
        self.assertEqual(len(llm.prompts), evidence["inspected_chunk_count"])
        self.assertEqual(evidence["inspected_chunk_count"], 4)
        self.assertGreater(evidence["chunk_count"], evidence["inspected_chunk_count"])
        self.assertTrue(all("网页正文片段" in prompt for prompt, _ in llm.prompts))
        self.assertTrue(all(max_tokens >= 384 for _, max_tokens in llm.prompts))
        self.assertGreaterEqual(len(evidence["candidates"]), 1)
        self.assertEqual(evidence["parallel_candidate"]["strategy"], "one-RWKV-call-per-chunk")
        self.assertEqual(evidence["parallel_candidate"]["completed_calls"], evidence["inspected_chunk_count"])
        self.assertEqual(evidence["parallel_candidate"]["source_chunk_count"], evidence["chunk_count"])
        self.assertEqual(len(evidence["source_chunks"]), evidence["chunk_count"])
        self.assertIn("第79段", evidence["source_chunks"][-1]["text"])

    def test_procedure_chunk_focus_and_deterministic_command_locator(self):
        chunks = [
            {
                "chunk_id": f"chunk-{index + 1}",
                "index": index,
                "text": (f"Unrelated PostgreSQL section {index}. " * 180),
                "token_count": 900,
            }
            for index in range(6)
        ]
        chunks[-1]["text"] = (
            "General jsonb background and document design. " * 120
            + "The default GIN operator class supports jsonb queries.\n"
            + "An example of creating an index with this operator class is:\n"
            + "CREATE INDEX idxgin ON api USING GIN (jdoc);\n"
            + "The technical difference between jsonb_ops and jsonb_path_ops concerns index items. " * 120
        )
        chunks[-1]["token_count"] = 1800
        page = {
            "title": "PostgreSQL JSON Types",
            "url": "https://www.postgresql.org/docs/current/datatype-json.html",
            "page_excerpt": "Substantive source body. " * 80,
        }
        llm = _NegativeLLM()
        with patch("agent.page_evidence.build_page_chunks", return_value=chunks):
            evidence = extract_single_page_evidence(
                query="PostgreSQL jsonb index 如何创建",
                page=page,
                llm=llm,
                task_plan={"answer_requirements": [{"type": "procedure"}]},
            )

        self.assertEqual(evidence["chunk_count"], 6)
        self.assertEqual(evidence["inspected_chunk_count"], 4)
        self.assertTrue(any("CREATE INDEX idxgin" in prompt for prompt, _ in llm.prompts))
        locator = next(
            row for row in evidence["chunk_candidates"]
            if row.get("deterministic_locator") == "procedure_command"
        )
        self.assertEqual(locator["chunk_id"], "chunk-6")
        self.assertIn("An example of creating an index", locator["quote"])
        self.assertTrue(locator["quote"].endswith("CREATE INDEX idxgin ON api USING GIN (jdoc);"))
        self.assertEqual(locator["record"]["command"], "CREATE INDEX idxgin ON api USING GIN (jdoc);")
        self.assertEqual(evidence["status"], "ok")

    def test_command_locator_rejects_selector_navigation_and_keeps_pip3_command(self):
        selector = (
            "PyTorch Build\nYour OS\nPackage\nLanguage\nCompute Platform\n"
            "Run this Command:\nStable\nLinux\nWindows\nPip\nLibTorch\nPython\n"
            "CUDA 11.8\nCUDA 12.6\nCPU\n"
            "pip3 install torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cu118\n"
        )
        chunks = [
            {"chunk_id": "chunk-1", "index": 0, "text": selector, "token_count": 120}
        ]
        page = {
            "title": "Start Locally | PyTorch",
            "url": "https://pytorch.org/get-started/locally/",
            "page_excerpt": selector,
        }
        with patch("agent.page_evidence.build_page_chunks", return_value=chunks):
            evidence = extract_single_page_evidence(
                query="Which PyTorch CUDA install command should I choose?",
                page=page,
                llm=_NegativeLLM(),
                task_plan={"answer_requirements": [{"type": "command"}, {"type": "selection"}]},
            )

        locators = [
            row
            for row in evidence["chunk_candidates"]
            if row.get("deterministic_locator") == "procedure_command"
        ]
        commands = [row["record"]["command"] for row in locators]
        self.assertIn(
            "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            commands,
        )
        self.assertNotIn("Pip\nLibTorch", commands)
        command_locator = next(
            row for row in locators
            if row["record"]["command"].startswith("pip3 install torch")
        )
        self.assertEqual(command_locator["quote"], command_locator["record"]["command"])
        self.assertNotIn("CPU", command_locator["quote"])

    def test_command_locator_does_not_promote_make_sure_prose(self):
        chunks = [
            {
                "chunk_id": "chunk-1",
                "index": 0,
                "text": (
                    "Make sure to select the correct CUDA version when generating the "
                    "installation command."
                ),
                "token_count": 18,
            }
        ]

        candidates = _procedure_command_candidates(
            "Which CUDA install command should I choose?",
            chunks,
            {"answer_requirements": [{"type": "command"}, {"type": "selection"}]},
        )

        self.assertEqual(candidates, [])

    def test_partial_chunk_transport_failure_is_not_a_semantic_negative(self):
        chunks = [
            {
                "chunk_id": "chunk-1",
                "index": 0,
                "text": "SUPPORTED_CHUNK contains the requested fact.",
                "token_count": 12,
            },
            {
                "chunk_id": "chunk-2",
                "index": 1,
                "text": "FAIL_CHUNK may contain another requested fact.",
                "token_count": 12,
            },
        ]
        page = {
            "title": "Model outage fixture",
            "url": "https://example.com/outage",
            "page_excerpt": "Substantive source body. " * 40,
        }
        with patch("agent.page_evidence.build_page_chunks", return_value=chunks):
            evidence = extract_single_page_evidence(
                query="requested fact",
                page=page,
                llm=_PartiallyFailingLLM(),
            )

        self.assertEqual(evidence["status"], "ok")
        self.assertTrue(evidence["extraction_degraded"])
        self.assertEqual(evidence["valid_contract_count"], 1)
        self.assertEqual(evidence["invalid_response_count"], 1)
        self.assertEqual(evidence["unresolved_chunk_indexes"], [1])
        self.assertGreaterEqual(evidence["transport_error_count"], 2)
        diagnostics = _model_extraction_diagnostics(
            [
                {
                    "url": page["url"],
                    "status": "partial",
                    "extraction_degraded": evidence["extraction_degraded"],
                    "invalid_response_count": evidence["invalid_response_count"],
                    "transport_error_count": evidence["transport_error_count"],
                    "parallel_candidate": evidence["parallel_candidate"],
                    "errors": evidence["errors"],
                }
            ]
        )
        self.assertFalse(diagnostics["complete"])
        self.assertEqual(diagnostics["unresolved_chunk_count"], 1)

    def test_recovered_chunk_failure_keeps_final_valid_negative(self):
        chunks = [
            {"chunk_id": "chunk-1", "index": 0, "text": "NORMAL_CHUNK", "token_count": 4},
            {"chunk_id": "chunk-2", "index": 1, "text": "FAIL_ONCE", "token_count": 4},
        ]
        page = {
            "title": "Recovered outage fixture",
            "url": "https://example.com/recovered",
            "page_excerpt": "Substantive source body. " * 40,
        }
        candidate_events = []
        with patch("agent.page_evidence.build_page_chunks", return_value=chunks):
            evidence = extract_single_page_evidence(
                query="requested fact",
                page=page,
                llm=_RecoveringNegativeLLM(),
                on_chunk=lambda **payload: candidate_events.append(payload),
            )

        self.assertEqual(evidence["status"], "no_evidence")
        self.assertTrue(evidence["all_chunks_valid_negative"])
        self.assertFalse(evidence["extraction_degraded"])
        self.assertEqual(evidence["unresolved_chunk_indexes"], [])
        self.assertEqual(evidence["recovered_chunk_indexes"], [1])
        self.assertEqual(len(candidate_events), 3)
        self.assertEqual(
            [
                (event["chunk"]["chunk_id"], event["candidate"]["retry_count"])
                for event in candidate_events
            ],
            [("chunk-1", 0), ("chunk-2", 0), ("chunk-2", 1)],
        )

    def test_latest_release_record_is_selected_deterministically_with_cutoff(self):
        page = {
            "title": "Release Notes",
            "url": "https://example.org/release-notes/",
            "page_excerpt": (
                "# Release Notes\n"
                "## 0.142.0 (2026-08-01)\nFuture release after the requested cutoff. "
                + "Future details. " * 20
                + "\n## 0.141.1 (2026-07-29)\nStable release at the requested cutoff. "
                + "Release details. " * 20
                + "\n## 0.141.0 (2026-07-29)\nEarlier release on the same day. "
                + "Earlier details. " * 20
            ),
        }
        evidence = extract_single_page_evidence(
            query="latest stable version as of 2026-07-29",
            page=page,
            llm=_NegativeLLM(),
            task_plan={
                "answer_requirements": [{"type": "version"}, {"type": "date"}],
                "freshness_policy": {"as_of": "2026-07-29"},
            },
        )
        selected = [
            row for row in evidence["chunk_candidates"]
            if row.get("deterministic_locator") == "latest_release_record"
        ]
        self.assertEqual(selected[0]["record"], {"version": "0.141.1", "date": "2026-07-29"})
        self.assertIn("0.141.1 (2026-07-29)", evidence["compact_facts"])
        self.assertNotIn("0.142.0 (2026-08-01)", evidence["compact_facts"])

    def test_latest_release_record_accepts_ordered_product_rows_and_rejects_prerelease(self):
        page = {
            "title": "Widget downloads",
            "url": "https://example.org/downloads/",
            "page_excerpt": (
                "# Widget downloads\n"
                "1. Widget 4.7.2 Aug. 5, 2026 Download\n"
                "2. Widget 5.0 pre-release Oct. 1, 2026 Download\n"
                + "Supported release details. " * 30
            ),
        }
        evidence = extract_single_page_evidence(
            query="Widget latest stable version and release date",
            page=page,
            llm=_NegativeLLM(),
            task_plan={
                "answer_requirements": [{"type": "version"}, {"type": "date"}],
                "freshness_policy": {
                    "as_of": None,
                    "now": "2026-08-09T12:00:00+00:00",
                },
            },
        )
        selected = [
            row for row in evidence["chunk_candidates"]
            if row.get("deterministic_locator") == "latest_release_record"
        ]
        self.assertEqual(selected[0]["record"], {"version": "4.7.2", "date": "2026-08-05"})
        self.assertIn("Widget 4.7.2 Aug. 5, 2026", evidence["compact_facts"])
        self.assertNotIn("Widget 5.0 pre-release", evidence["compact_facts"])

    def test_explicit_version_anchor_rejects_adjacent_release_page(self):
        page = {
            "title": "Kubernetes 1.14",
            "url": "https://example.org/kubernetes-1-14",
            "page_excerpt": (
                "# Kubernetes 1.14\nReleased Monday, March 25, 2019.\n"
                + "Official release details for Kubernetes 1.14. " * 20
            ),
        }
        evidence = extract_single_page_evidence(
            query="Kubernetes 1.30 official release date",
            page=page,
            llm=_GroundedFakeLLM(),
            task_plan={"answer_requirements": [{"type": "date"}]},
        )
        self.assertEqual(evidence["status"], "no_evidence")
        self.assertFalse(any(row.get("supported") for row in evidence["chunk_candidates"]))

    def test_explicit_version_anchor_recovers_source_span_when_model_misses_it(self):
        page = {
            "title": "Kubernetes v1.30",
            "url": "https://example.org/kubernetes-v1-30",
            "page_excerpt": (
                "# Kubernetes v1.30: Example\n"
                "By the release team | Wednesday, April 17, 2024\n"
                + "Official release details for Kubernetes v1.30. " * 20
            ),
        }
        evidence = extract_single_page_evidence(
            query="Kubernetes 1.30 official release date",
            page=page,
            llm=_NegativeLLM(),
            task_plan={"answer_requirements": [{"type": "date"}]},
        )
        self.assertEqual(evidence["status"], "ok")
        self.assertFalse(evidence["all_chunks_valid_negative"])
        self.assertTrue(
            any(row.get("deterministic_locator") == "explicit_identity_anchor" for row in evidence["chunk_candidates"])
        )
        locator = next(
            row for row in evidence["chunk_candidates"]
            if row.get("deterministic_locator") == "explicit_identity_anchor"
        )
        self.assertEqual(locator["record"], {"date": "2024-04-17"})

    def test_explicit_version_anchor_does_not_promote_an_unrelated_procedure(self):
        page = {
            "title": "What is new in Python 3.14",
            "url": "https://docs.python.org/3.14/whatsnew/3.14.html",
            "page_excerpt": (
                "# Python 3.14\n"
                "The concurrent.interpreters module provides multiple interpreters. " * 30
            ),
        }
        evidence = extract_single_page_evidence(
            query="python 3.14 free threading到底怎么开，查官方文档",
            page=page,
            llm=_NegativeLLM(),
            task_plan={"answer_requirements": [{"type": "procedure"}]},
        )

        self.assertEqual(evidence["status"], "no_evidence")
        self.assertFalse(
            any(
                row.get("deterministic_locator") == "explicit_identity_anchor"
                for row in evidence["chunk_candidates"]
            )
        )

    def test_generic_fetch_uses_the_configured_long_page_limit(self):
        payload = json.dumps({"status": "ok", "results": []})
        with (
            patch.dict("tools.web_search_generic.DATA_PIPELINE", {"web_page_max_chars": 120000}),
            patch("tools.web_search_keyless.fetch_web_url", return_value=payload) as fetch,
        ):
            _fetch_candidate({"url": "https://docs.python.org/3.14/whatsnew/3.14.html"}, "task-1")

        self.assertEqual(fetch.call_args.kwargs["max_chars"], 120000)

    def test_same_host_fetches_are_bounded_without_serializing_other_hosts(self):
        active = 0
        peak = 0
        lock = threading.Lock()

        def fake_fetch(_url, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return json.dumps({"status": "ok", "results": []})

        candidates = [
            {"url": f"https://docs.example.org/page-{index}"}
            for index in range(6)
        ]
        with (
            patch.dict(
                "tools.web_search_generic.DATA_PIPELINE",
                {
                    "web_page_fetch_per_host_concurrency": 2,
                    "web_page_fetch_attempts": 1,
                },
            ),
            patch("tools.web_search_keyless.fetch_web_url", side_effect=fake_fetch),
            concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool,
        ):
            list(pool.map(lambda candidate: _fetch_candidate(candidate, "task-1"), candidates))

        self.assertEqual(peak, 2)

    def test_transient_page_fetch_is_retried_inside_the_host_gate(self):
        responses = [
            json.dumps({"status": "error", "message": "SSL unexpected EOF", "results": []}),
            json.dumps({"status": "ok", "results": []}),
        ]
        with (
            patch.dict(
                "tools.web_search_generic.DATA_PIPELINE",
                {"web_page_fetch_per_host_concurrency": 2, "web_page_fetch_attempts": 2},
            ),
            patch("tools.web_search_keyless.fetch_web_url", side_effect=responses) as fetch,
        ):
            result = _fetch_candidate({"url": "https://docs.example.org/page"}, "task-1")

        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["fetch_attempt_count"], 2)
        self.assertEqual(result["fetch_retry_errors"], ["SSL unexpected EOF"])

    def test_subthreshold_cleaned_page_stays_single_pass(self):
        page = "\n".join(f"事实{i}: 深圳地铁一号线站点信息。" for i in range(120))
        from utils.chunker import get_token_count

        self.assertLessEqual(get_token_count(page), 2400)
        chunks = build_page_chunks(page)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["token_count"], get_token_count(page))

    def test_long_cleaned_page_uses_parallel_chunks(self):
        page = "\n".join(f"事实{i}: 深圳地铁一号线站点信息。" for i in range(2600))
        from utils.chunker import get_token_count

        self.assertGreater(get_token_count(page), 2400)
        chunks = build_page_chunks(page)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(item["token_count"] <= 2400 for item in chunks))

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

    def test_site_query_discards_provider_results_from_other_domains(self):
        html = """
        <html><body>
          <h2><a href="https://www.python.org/">Python home</a></h2><p>Python</p>
          <h2><a href="https://docs.python.org/3/howto/free-threading-python.html">Free-threading HOWTO</a></h2><p>Python free threading</p>
        </body></html>
        """
        with patch("tools.web_search_keyless.fetch_text", return_value=html):
            result = json.loads(
                search_web_keyless(
                    "site:docs.python.org Python free threading",
                    max_results=4,
                    agentic_tool_loop=True,
                )
            )
        self.assertEqual(result["count"], 1)
        self.assertEqual(
            result["results"][0]["url"],
            "https://docs.python.org/3/howto/free-threading-python.html",
        )

    def test_yahoo_parser_extracts_title_snippet_and_unwrapped_url(self):
        redirect = (
            "https://r.search.yahoo.com/_ylt=x/RU="
            "https%3A%2F%2Fwww.earthdata.nasa.gov%2Ftopics%2Focean%2Fsalinity"
            "/RK=2/RS=x"
        )
        html = (
            '<div class="dd algo algo-sr">'
            f'<div><a data-matarget="algo" href="{redirect}">'
            '<div><span>Earthdata</span>https://www.earthdata.nasa.gov</div>'
            '<h3 class="title"><span>Salinity - NASA Earthdata</span></h3>'
            '</a></div>'
            '<div class="compText"><p>Ocean salinity is influenced by river runoff and evaporation.</p></div>'
            '</div>'
        )
        parser = _YahooResultParser()
        parser.feed(html)
        self.assertEqual(len(parser.rows), 1)
        self.assertEqual(parser.rows[0]["title"], "Salinity - NASA Earthdata")
        self.assertEqual(
            parser.rows[0]["url"],
            "https://www.earthdata.nasa.gov/topics/ocean/salinity",
        )
        self.assertIn("river runoff", parser.rows[0]["snippet"])
        self.assertEqual(_unwrap_ddg_url(redirect), parser.rows[0]["url"])

    def test_keyless_search_ranks_across_providers_after_all_complete(self):
        redirect = (
            "https://r.search.yahoo.com/_ylt=x/RU="
            "https%3A%2F%2Fscience.example%2Focean-salinity-balance"
            "/RK=2/RS=x"
        )
        yahoo = (
            '<div class="dd algo"><div>'
            f'<a data-matarget="algo" href="{redirect}">'
            '<h3 class="title">Ocean salinity inputs and long-term balance</h3></a>'
            '</div><div class="compText"><p>Salt sources, sinks, and long-term ocean balance.</p></div></div>'
        )

        def fetch(endpoint, *_args, **_kwargs):
            return yahoo if "yahoo.com" in endpoint else "<html></html>"

        with patch("tools.web_search_keyless.fetch_text", side_effect=fetch):
            result = json.loads(
                search_web_keyless(
                    "ocean salinity salt sources long-term balance",
                    max_results=4,
                    agentic_tool_loop=True,
                )
            )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["source"], "Yahoo HTML (keyless)")

    def test_pdf_urls_are_not_sent_to_html_evidence_pipeline(self):
        with self.assertRaises(NetworkFetchError):
            fetch_text("https://example.com/archive/trustees.pdf")

    def test_generic_search_prefers_html_over_pdf_candidates(self):
        candidates = _merge_candidates(
            "VLDB Endowment Board of Directors 2022",
            [
                {
                    "provider": "test",
                    "results": [
                        {"title": "old trustees PDF", "url": "https://vldb.org/old.pdf", "snippet": "trustees"},
                        {"title": "current trustees", "url": "https://vldb.org/trustees.html", "snippet": "Board of Directors"},
                    ],
                }
            ],
            limit=2,
        )
        self.assertEqual(candidates[0]["url"], "https://vldb.org/trustees.html")
        self.assertEqual(len(candidates), 1)

    def test_generic_serp_gate_rejects_an_interrogative_dictionary_page(self):
        query = "why is seawater salty salt sources ocean salinity balance"
        dictionary = {
            "title": "Why - English Grammar Today",
            "url": "https://dictionary.example/grammar/why",
            "snippet": "Why is a question word used to ask for reasons.",
        }
        relevant = {
            "title": "Why is the ocean salty?",
            "url": "https://science.example/ocean-salinity",
            "snippet": "Sources of salt and the long-term ocean salinity balance.",
        }
        self.assertFalse(_looks_related(dictionary, query))
        self.assertTrue(_looks_related(relevant, query))

    def test_public_search_transport_removes_interrogative_noise_but_keeps_constraints(self):
        self.assertEqual(
            _provider_query("why is seawater salty salt sources ocean salinity balance"),
            "seawater salty salt ocean salinity balance",
        )
        normalized = _provider_query("site:example.org Project 5.2 official release date")
        self.assertTrue(normalized.startswith("site:example.org "))
        self.assertIn("project", normalized)
        self.assertIn("5.2", normalized)
        self.assertEqual(
            _provider_query("请帮我搜索 Django 5.2 的官方发布日期"),
            "Django 5.2 的官方发布日期",
        )

    def test_provider_noise_is_rejected_again_at_the_shared_candidate_gate(self):
        candidates = _merge_candidates(
            "why is seawater salty salt sources ocean salinity balance",
            [
                {
                    "provider": "test",
                    "results": [
                        {
                            "title": "Why - English Grammar Today",
                            "url": "https://dictionary.example/grammar/why",
                            "snippet": "Why is a question word.",
                        },
                        {
                            "title": "Ocean salinity",
                            "url": "https://science.example/ocean-salinity",
                            "snippet": "Seawater salt sources and salinity balance.",
                        },
                    ],
                }
            ],
            limit=8,
        )
        self.assertEqual([row["url"] for row in candidates], ["https://science.example/ocean-salinity"])


if __name__ == "__main__":
    unittest.main()
