"""Build an evidence-grounded comparison report for completed retrieval runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "evaluation"
RWKV_SEARCH_ROOT = ROOT.parent / "rwkv-search"

RUNS = [
    {
        "label": "Latest 100-question set",
        "result": EVAL / "rwkv_search_100_fixed_20260729.json",
        "input": RWKV_SEARCH_ROOT / "bench" / "realtime_web_retrieval_dev_v2.jsonl",
    },
    {
        "label": "Local 50-question set",
        "result": EVAL / "rwkv_search_50_fixed_20260729.json",
        "input": EVAL / "rwkv_search_20260729_questions.jsonl",
    },
    {
        "label": "Date regression set",
        "result": EVAL / "date_retrieval_fixed_20260729.json",
        "input": EVAL / "date_retrieval_tasks_20260729.jsonl",
    },
]

OLD_RUNS = [
    EVAL / "rwkv_search_20260729_50_global_results_v2.json",
    EVAL / "rwkv_search_20260729_50_global_results.json",
    EVAL / "rwkv_search_100_adaptive7000_results_20260729.json",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def text(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    value = re.sub(r"\s+", " ", str(value)).strip()
    if limit and len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def md(value: Any, limit: int | None = None) -> str:
    return text(value, limit).replace("`", "\\`").replace("|", "\\|")


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def case_input_map(path: Path) -> dict[str, dict[str, Any]]:
    return {text(item.get("query")): item for item in read_jsonl(path)}


def stats_for(case: dict[str, Any]) -> dict[str, int]:
    stats = case.get("trace", {}).get("stats", {}) or {}
    errors = stats.get("error_class_counts", {}) or {}
    actions = stats.get("action_counts", {}) or {}
    context_tokens = stats.get("context_tokens", 0)
    if isinstance(context_tokens, dict):
        context_tokens = context_tokens.get("max", 0)
    return {
        "model_calls": int(stats.get("model_call_count", 0) or 0),
        "searches": int(actions.get("web_search", 0) or 0),
        "duplicate_blocks": int(errors.get("duplicate_query", 0) or 0),
        "chunks": int(stats.get("chunk_count", 0) or 0),
        "context_tokens": int(context_tokens or 0),
    }


def citation_data(case: dict[str, Any]) -> tuple[list[str], list[str]]:
    trace = case.get("trace", {}) or {}
    final = trace.get("final", {}) or {}
    refs = final.get("citation_refs", []) or []
    urls: list[str] = []
    lines: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        url = text(ref.get("url"))
        if url:
            urls.append(url)
        label = text(ref.get("title")) or url or "citation"
        evidence = text(ref.get("evidence_text"), 500)
        lines.append(f"{label}: {evidence}" if evidence else label)

    for item in trace.get("web_search_chunks", []) or []:
        candidate = item.get("candidate", {}) if isinstance(item, dict) else {}
        if not isinstance(candidate, dict):
            continue
        facts = candidate.get("facts", []) or []
        quote = text(candidate.get("quote"), 300)
        if facts or quote:
            support = "supported" if candidate.get("supported") else "not-supported"
            fact_text = "; ".join(text(f, 180) for f in facts[:6])
            detail = f"facts={fact_text}"
            if quote:
                detail += f"; quote={quote}"
            lines.append(f"[{support}] {detail}")
    return unique(urls), unique(lines)[:12]


def usable_evidence(case: dict[str, Any]) -> int:
    contexts = case.get("trace", {}).get("contexts", []) or []
    values = [
        int(context.get("usable_evidence_count", 0) or 0)
        for context in contexts
        if isinstance(context, dict)
    ]
    if values:
        return max(values)
    statuses = case.get("trace", {}).get("stats", {}).get("page_evidence_statuses", {}) or {}
    return int(statuses.get("ok", 0) or 0)


def answer_for(case: dict[str, Any]) -> str:
    trace_final = case.get("trace", {}).get("final", {}) or {}
    return text(
        case.get("answer") or case.get("final_output") or trace_final.get("content"),
        5000,
    )


def diagnosis(case: dict[str, Any], urls: list[str], evidence: list[str], usable: int) -> tuple[str, str]:
    status = text(case.get("status"))
    answer = answer_for(case)
    if status != "completed":
        error = text(case.get("error")) or answer or "task planning did not complete"
        return "task-planning failure", f"未进入稳定检索/汇总链路：{error}"
    if usable == 0 or not urls:
        # A refusal normally declares the lack of evidence in its opening
        # sentence.  Long, structured answers that begin with "完成了/以下是
        # 完整回答" are unsupported completions instead, even if they later
        # contain a disclaimer or the word "无法".
        opening = answer[:220]
        refusal_opening = bool(re.search(r"无法|未能|没有|不足|不可|查不到|未找到", opening))
        fabricated_opening = bool(re.search(r"完成了|完整回答|以下是.*(?:答案|报告|分析)", opening))
        if refusal_opening and not fabricated_opening:
            return "evidence-free refusal", "最终答案是拒答或保守说明；当前没有可用正文证据，不能据此判断事实错误。"
        return "unsupported completion", "模型给出了事实性答案，但当前结果没有可用正文证据；这是证据链断裂，不能视为已验证。"
    thin = all(len(line) < 100 for line in evidence) if evidence else True
    if thin:
        return "thin evidence", "有引用/证据记录，但证据主要是标题、导航词或极短片段；事实是否正确仍需打开原文人工核验。"
    if any(token in answer for token in ("User:", "Assistant:", "<think>", "```json")):
        return "format contamination", "检索证据存在，但最终续写混入协议/角色标记，属于输出格式错误。"
    return "evidence available", "已有可用证据，但输入数据没有统一标准答案；本报告不把有证据直接等同于事实正确。"


def run_summary(data: dict[str, Any]) -> dict[str, Any]:
    cases = data.get("cases", []) or []
    completed = [case for case in cases if case.get("status") == "completed"]
    aggregate = Counter()
    diagnoses = Counter()
    for case in cases:
        urls, evidence = citation_data(case)
        kind, _ = diagnosis(case, urls, evidence, usable_evidence(case))
        diagnoses[kind] += 1
        for key, value in stats_for(case).items():
            aggregate[key] += value
    return {
        "total": len(cases),
        "completed": len(completed),
        "noncompleted": len(cases) - len(completed),
        "aggregate": dict(aggregate),
        "diagnoses": dict(diagnoses),
        "status": data.get("status", ""),
    }


def old_summary(path: Path) -> str:
    if not path.exists():
        return f"{path.name}: missing"
    try:
        data = read_json(path)
    except Exception as exc:  # pragma: no cover - report helper
        return f"{path.name}: parse error ({exc})"
    cases = data.get("cases", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    completed = sum(1 for case in cases if case.get("status") == "completed")
    status = data.get("status", "") if isinstance(data, dict) else "n/a"
    return f"{path.name}: {completed}/{len(cases)} completed; top-level status={status}"


def build_report() -> str:
    sections: list[str] = [
        "# RWKV-Scout 三批测试逐题答案与证据对照",
        "",
        "> 生成时间：2026-07-30。此报告严格区分“模型最终答案”“检索器返回的正文/抽取事实”和“事实正确性”。原始题库没有为所有题目提供标准答案，因此机器结论不能冒充准确率；需要事实核验的行会明确标记。",
        "",
        "## 总体结果",
        "",
        "| 批次 | 总题数 | 完成 | 未完成 | 模型调用 | web_search | 重复拦截 | 机器诊断 |\n|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    all_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for run in RUNS:
        data = read_json(run["result"])
        summary = run_summary(data)
        aggregate = summary["aggregate"]
        diagnoses = ", ".join(f"{key}={value}" for key, value in summary["diagnoses"].items())
        sections.append(
            f"| {run['label']} | {summary['total']} | {summary['completed']} | {summary['noncompleted']} | "
            f"{aggregate.get('model_calls', 0)} | {aggregate.get('searches', 0)} | {aggregate.get('duplicate_blocks', 0)} | {diagnoses or '—'} |"
        )
        input_map = case_input_map(run["input"])
        for case in data.get("cases", []) or []:
            all_rows.append((run["label"], case, input_map.get(text(case.get("query")), {})))

    sections.extend(
        [
            "",
            "### 旧结果可比性",
            "",
            "- " + "\n- ".join(old_summary(path) for path in OLD_RUNS),
            "- 旧 50 题 v2 完成数与新固定版相同，但旧文件没有统一记录 `action_counts.web_search` 和重复拦截字段，不能把旧统计中的 0 当成没有检索。",
            "- 新固定版 50 题模型调用 218 次，旧 50 题 v2 为 267 次，减少 49 次（约 18.4%）；这只能说明执行开销下降，不能单独证明答案准确率提升。",
            "",
            "## 逐题对照",
            "",
        ]
    )

    for index, (label, case, expected) in enumerate(all_rows, start=1):
        urls, evidence = citation_data(case)
        usable = usable_evidence(case)
        kind, why = diagnosis(case, urls, evidence, usable)
        stats = stats_for(case)
        answer = answer_for(case) or "（无最终答案）"
        queries = unique(
            [
                text(call.get("args", {}).get("query"))
                for call in (case.get("trace", {}).get("tool_calls", []) or [])
                if isinstance(call, dict) and call.get("action") == "web_search"
            ]
        )
        constraints = []
        for key in ("source_policy", "expected_domains_any", "target_url_patterns_any", "notes"):
            if expected.get(key):
                constraints.append(f"{key}={text(expected[key], 500)}")
        sections.extend(
            [
                f"### {index}. {label} / {text(case.get('case_id') or case.get('task_id') or index)}",
                "",
                f"- **题目**：{md(case.get('query'), 1000)}",
                f"- **状态**：{md(case.get('status'))}；**机器诊断**：{kind}",
                f"- **最终答案**：{md(answer, 5000)}",
                f"- **模型实际检索查询**：{md('；'.join(queries) or '未执行 web_search', 1200)}",
                f"- **检索证据摘要**：{md('；'.join(evidence) or '（没有可用的正文证据/抽取事实）', 1800)}",
                f"- **引用 URL**：{'；'.join(f'[{url}]({url})' for url in urls) if urls else '（无）'}",
                f"- **执行统计**：model_calls={stats['model_calls']}，searches={stats['searches']}，duplicate_blocks={stats['duplicate_blocks']}，chunks={stats['chunks']}，usable_evidence={usable}",
                f"- **错误分析**：{md(why, 1200)}",
            ]
        )
        if constraints:
            sections.append(f"- **题库约束（不是标准答案）**：{md('；'.join(constraints), 1200)}")
        sections.append("")

    sections.extend(
        [
            "## 如何阅读这份对照",
            "",
            "1. `evidence available` 只表示流水线提取到了正文片段，不表示事实已经被证明；必须检查片段是否真正回答问题。",
            "2. `thin evidence` 表示引用只剩标题、导航项或重复短文本，最容易出现“模型看到了相关词就自行补全”的幻觉。",
            "3. `unsupported completion` 表示模型说了事实，但当前检索结果没有支撑；这是最严重的工程问题。",
            "4. `evidence-free refusal` 是无证据时的保守拒答，不应当算成事实答错；它说明系统没有足够证据完成任务。",
            "5. `task-planning failure` 表示模型在检索开始前规划失败，不能归因于搜索源或网页清洗。",
        ]
    )
    return "\n".join(sections) + "\n"


def main() -> None:
    output = EVAL / "fixed_batches_answer_comparison_20260730.md"
    output.write_text(build_report(), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
