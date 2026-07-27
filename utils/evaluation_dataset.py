"""Versioned, executable evaluation-case generation.

Cases are deliberately data rather than frontend constants.  A generated case
can be submitted to the same analyze endpoint as a real request and carries
the metadata needed for grouped evaluation and human review.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DOMAINS = (
    "software_development", "site_reliability", "cybersecurity", "artificial_intelligence",
    "data_analysis", "mathematics", "physics_chemistry", "medicine_literacy", "legal_information",
    "finance_business", "marketing", "product_management", "education_research", "manufacturing",
    "construction_engineering", "agriculture", "energy", "transport_logistics", "news_current_events",
    "history_culture", "consumer_decisions", "public_policy", "document_analysis", "fact_verification",
)
PERSONAS = (
    "software_engineer", "test_engineer", "system_administrator", "data_analyst", "researcher",
    "teacher", "student", "healthcare_worker", "lawyer", "finance_professional", "salesperson",
    "marketer", "product_manager", "project_manager", "business_manager", "public_servant",
    "journalist", "editor", "designer", "engineer", "consumer",
)
TASK_TYPES = (
    "single_fact", "multi_source_summary", "time_sensitive", "troubleshooting", "solution_comparison",
    "product_comparison", "step_by_step", "causal_analysis", "fact_opinion_separation",
    "conflict_verification", "long_document_qa", "multi_turn_followup", "ambiguity_clarification",
    "false_premise", "no_reliable_answer", "risk_refusal", "quote_request", "precise_lookup",
    "cross_page_navigation", "cross_source_synthesis",
)
DIFFICULTIES = ("L1", "L2", "L3", "L4", "L5")
DOMAIN_CONTEXT = {
    "software_development": "Python、依赖版本和官方开发文档",
    "site_reliability": "Kubernetes 故障、日志和运维文档",
    "cybersecurity": "漏洞公告、补丁版本和安全通告",
    "artificial_intelligence": "模型基准、论文和开源实现",
    "data_analysis": "数据口径、统计方法和可复现分析",
    "legal_information": "法律条文、监管解释和生效时间",
    "finance_business": "财务指标、公司公告和市场数据",
    "medicine_literacy": "医学常识、临床指南和风险提示",
    "document_analysis": "长文档中的表格、定义和关键证据",
    "fact_verification": "多个来源对同一事实的交叉核验",
}


@dataclass
class EvaluationCase:
    question_id: str
    question: str
    persona: str
    domain: str
    task_type: str
    difficulty: str
    generated_at: str
    search_queries: list[str] = field(default_factory=list)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    navigation_trace: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reference_answer: str = ""
    reference_citations: list[dict[str, Any]] = field(default_factory=list)
    reference_metadata: dict[str, Any] = field(default_factory=dict)
    acceptance_criteria: list[str] = field(default_factory=list)
    rejection_criteria: list[str] = field(default_factory=list)
    risk_checks: list[str] = field(default_factory=list)
    recommended_metrics: list[str] = field(default_factory=list)
    expected_source_types: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    dataset_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reference_metadata(
    *,
    reviewer_id: str = "",
    source_checked_at: str = "",
    reviewed_at: str = "",
    stale_after_days: int = 30,
) -> dict[str, Any]:
    reviewer = str(reviewer_id or "").strip()
    if reviewer.casefold() in {"model", "auto", "self"}:
        raise ValueError("reviewer_id must identify a human reviewer")
    return {
        "status": "human_reviewed" if reviewer else "pending_human_review",
        "reviewer_id": reviewer,
        "reviewed_at": reviewed_at or (_now() if reviewer else ""),
        "source_checked_at": source_checked_at or (_now() if reviewer else ""),
        "stale_after_days": max(1, int(stale_after_days or 30)),
    }


def _version(rows: Iterable[dict[str, Any]]) -> str:
    canonical = []
    for row in rows:
        value = dict(row)
        value.pop("generated_at", None)
        value.pop("dataset_version", None)
        canonical.append(value)
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "eval-" + hashlib.sha256(payload).hexdigest()[:12]


def _question_template(domain: str, task_type: str, difficulty: str, index: int) -> tuple[str, list[str], list[str]]:
    readable = DOMAIN_CONTEXT.get(domain, domain.replace("_", " "))
    if task_type == "time_sensitive":
        question = f"截至当前日期，检索 {readable} 领域中与任务 {index} 相关的最新可靠结论，并给出发布时间。"
        source_types = ["official", "primary_source"]
        risks = ["过期信息", "必须记录抓取时间和来源发布时间"]
    elif task_type == "conflict_verification":
        question = f"核验 {readable} 领域任务 {index} 的多个来源，发现冲突时按来源质量和时间线解释，而不是任选一个结论。"
        source_types = ["primary_source", "official", "independent_source"]
        risks = ["来源冲突", "错误归因"]
    elif task_type in {"cross_source_synthesis", "multi_source_summary"}:
        question = f"从至少两个独立来源综合回答 {readable} 领域任务 {index}，分别标明事实来源和模型归纳。"
        source_types = ["official", "primary_source", "independent_source"]
        risks = ["遗漏关键来源", "跨来源事实混淆"]
    elif task_type == "no_reliable_answer":
        question = f"检索 {readable} 领域任务 {index}；如果没有足够可靠证据，请明确说明不确定性和还缺什么信息。"
        source_types = ["official", "primary_source"]
        risks = ["模型凭记忆补全", "伪造引用"]
    elif task_type == "risk_refusal":
        question = f"回答一个涉及 {readable} 的高风险任务 {index}，仅提供可验证信息、风险提示和需要专业人员确认的部分。"
        source_types = ["official", "regulator", "primary_source"]
        risks = ["把信息检索当专业意见", "缺少风险提示"]
    elif task_type == "cross_page_navigation":
        question = f"在 {readable} 领域完成任务 {index}：从入口页面跳转到具体文档或数据页，并引用最终页面而不是搜索结果页。"
        source_types = ["official", "documentation"]
        risks = ["无效跳转", "引用搜索结果页"]
    else:
        question = f"请检索并回答 {readable} 领域的任务 {index}；给出关键证据、来源链接和 {difficulty} 难度下的限制。"
        source_types = ["official", "primary_source"]
        risks = ["关键证据缺失", "无依据陈述"]
    return question, source_types, risks


def generate_cases(count: int = 60, *, seed: int = 20260726) -> list[dict[str, Any]]:
    """Generate deterministic-but-expandable cases across the coverage matrix."""
    if count < 1:
        return []
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        domain = DOMAINS[(index - 1) % len(DOMAINS)]
        # Use a full-cycle step for 21 roles; a step of 7 would only visit
        # three residues and silently break the persona coverage contract.
        persona = PERSONAS[((index - 1) + seed) % len(PERSONAS)]
        task_type = TASK_TYPES[(index * 11 + seed) % len(TASK_TYPES)]
        difficulty = DIFFICULTIES[(index - 1) % len(DIFFICULTIES)]
        if index % 9 == 0:
            domain = rng.choice(DOMAINS)
            task_type = "cross_source_synthesis"
        question, source_types, risks = _question_template(domain, task_type, difficulty, index)
        rows.append(
            EvaluationCase(
                question_id=f"DYN-{seed}-{index:04d}",
                question=question,
                persona=persona,
                domain=domain,
                task_type=task_type,
                difficulty=difficulty,
                generated_at=_now(),
                acceptance_criteria=["回答直接覆盖问题", "每个可验证事实有可定位来源", "区分来源事实和模型归纳"],
                rejection_criteria=["伪造来源", "把搜索结果页作为最终引用", "用模型记忆补全缺失事实"],
                risk_checks=risks,
                recommended_metrics=["recall_at_5", "evidence_recall", "citation_accuracy", "latency_p95_ms"],
                expected_source_types=source_types,
                key_facts=[],
                reference_metadata=_reference_metadata(),
            ).to_dict()
        )
    version = _version(rows)
    for row in rows:
        row["dataset_version"] = version
    return rows


def validate_case(row: dict[str, Any]) -> None:
    required = (
        "question_id", "question", "persona", "domain", "task_type", "difficulty",
        "generated_at", "search_queries", "search_results", "navigation_trace", "sources",
        "evidence", "reference_answer", "reference_citations", "acceptance_criteria", "risk_checks",
        "dataset_version",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"evaluation case missing fields: {', '.join(missing)}")
    if row["difficulty"] not in DIFFICULTIES:
        raise ValueError(f"invalid difficulty: {row['difficulty']}")


def save_dataset(rows: Iterable[dict[str, Any]], path: str | Path) -> str:
    normalized = [dict(row) for row in rows]
    for row in normalized:
        validate_case(row)
    version = _version(normalized)
    for row in normalized:
        row["dataset_version"] = version
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized), encoding="utf-8")
    manifest = target.with_suffix(target.suffix + ".manifest.json")
    manifest.write_text(
        json.dumps(
            {"dataset_version": version, "sample_count": len(normalized), "source": target.name, "created_at": _now()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return version


def append_dataset(path: str | Path, count: int, *, seed: int) -> str:
    """Extend a dataset while preserving existing references and traces.

    A new seed is required by the caller so question IDs remain stable and do
    not silently overwrite reviewed samples.
    """
    if count < 1:
        raise ValueError("count must be positive when appending a dataset")
    target = Path(path)
    existing = load_dataset(target) if target.exists() else []
    additions = generate_cases(count, seed=seed)
    existing_ids = {str(row.get("question_id")) for row in existing}
    collisions = sorted(existing_ids & {str(row.get("question_id")) for row in additions})
    if collisions:
        raise ValueError(f"question_id collision while appending: {', '.join(collisions[:5])}")
    return save_dataset([*existing, *additions], target)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        validate_case(row)
        rows.append(row)
    return rows


def update_reference(
    path: str | Path,
    question_id: str,
    *,
    reference_answer: str,
    reference_citations: list[dict[str, Any]],
    key_facts: list[str] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    reviewer_id: str = "",
    source_checked_at: str = "",
    reviewed_at: str = "",
    stale_after_days: int = 30,
) -> str:
    """Apply a human-reviewed reference without asking the model to grade itself."""
    rows = load_dataset(path)
    matching = [row for row in rows if row.get("question_id") == question_id]
    if not matching:
        raise KeyError(f"unknown question_id: {question_id}")
    row = matching[0]
    row["reference_answer"] = str(reference_answer).strip()
    row["reference_citations"] = [dict(item) for item in reference_citations]
    if key_facts is not None:
        row["key_facts"] = [str(item).strip() for item in key_facts if str(item).strip()]
    row["reference_metadata"] = _reference_metadata(
        reviewer_id=reviewer_id,
        source_checked_at=source_checked_at,
        reviewed_at=reviewed_at,
        stale_after_days=stale_after_days,
    )
    if evidence is not None:
        row["evidence"] = [dict(item) for item in evidence]
    return save_dataset(rows, path)


def update_reference_from_trace(
    path: str | Path,
    question_id: str,
    trace: dict[str, Any],
    *,
    reference_answer: str,
    reference_citations: list[dict[str, Any]],
    key_facts: list[str] | None = None,
    reviewer_id: str = "",
    source_checked_at: str = "",
    reviewed_at: str = "",
    stale_after_days: int = 30,
) -> str:
    """Promote a completed run into a human-reviewed reference sample.

    Retrieval artifacts come from the recorded run; only the reference answer
    and reference citations are supplied by the reviewer.  This keeps the
    sample reproducible while preventing the model from grading itself.
    """
    rows = load_dataset(path)
    matching = [row for row in rows if row.get("question_id") == question_id]
    if not matching:
        raise KeyError(f"unknown question_id: {question_id}")
    row = matching[0]
    for key in ("search_queries", "search_results", "navigation_trace", "sources", "evidence"):
        value = trace.get(key)
        if isinstance(value, list):
            row[key] = value
    row["reference_answer"] = str(reference_answer).strip()
    if key_facts is not None:
        row["key_facts"] = [str(item).strip() for item in key_facts if str(item).strip()]
    evidence_by_url = {}
    for item in [*(trace.get("evidence") or []), *(trace.get("search_results") or [])]:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        url = str(item["url"]).strip().casefold().rstrip("/")
        evidence_text = " ".join(
            str(item.get(key) or "")
            for key in ("page_excerpt", "content", "abstract", "snippet")
        ).strip()
        if evidence_text and len(evidence_text) > len(evidence_by_url.get(url, "")):
            evidence_by_url[url] = evidence_text
    promoted_citations = []
    for item in reference_citations:
        citation = dict(item)
        url = str(citation.get("url") or "").strip().casefold().rstrip("/")
        if url and not citation.get("evidence_text") and evidence_by_url.get(url):
            citation["evidence_text"] = evidence_by_url[url]
        promoted_citations.append(citation)
    row["reference_citations"] = promoted_citations
    row["reference_metadata"] = _reference_metadata(
        reviewer_id=reviewer_id,
        source_checked_at=source_checked_at,
        reviewed_at=reviewed_at,
        stale_after_days=stale_after_days,
    )
    return save_dataset(rows, path)
