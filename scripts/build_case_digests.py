#!/usr/bin/env python3
"""Build compact per-case causal digests from raw_cases_gz for manual audit.

Read-only over run artifacts. The digest is an analysis-side extraction: it
clips long text for reading efficiency but never rewrites content. Every
digest points back to the raw gz case for full-fidelity inspection.

Usage:
  uv run python scripts/build_case_digests.py --audit-dir outputs/round53_full_chain_manual_review_20260815
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

WRITER_STAGES = {"final_writer"}
CONTROL_STAGES = {
    "evidence_review", "evidence_resolution", "planner_replan", "planner_recovery",
    "record_resolution", "cross_validation", "retrieval_query_plan",
}


def clip(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def stage_of(call: dict) -> str:
    return str(call.get("request_stage") or call.get("phase") or "unknown")


def load_prior_audit(audit_dir: Path) -> dict[str, dict]:
    prior: dict[str, dict] = {}
    for p in audit_dir.glob("ROUND*_MANUAL_CASE_AUDIT.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prior[row["case_id"]] = row
    if not prior:
        # A fresh round has no manual audit yet; surface the frozen reference
        # so scorers can compare without opening a second file.
        frozen = audit_dir.parent / "round50_full_chain_manual_review_20260815/frozen_reference_round12.json"
        if frozen.exists():
            for row in json.loads(frozen.read_text(encoding="utf-8")):
                prior[row["case_id"]] = {
                    "score": None,
                    "layer": "",
                    "grounding": "",
                    "drift": "",
                    "finding": "(no prior audit for this round; reference from frozen round12 file)",
                    "frozen_reference_answer": row.get("frozen_reference_answer", ""),
                }
    for p in audit_dir.glob("round41_manual_semantic_review.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prior[row["case_id"]] = {
                "score": row.get("round41_frozen_reference_similarity_score"),
                "layer": "",
                "grounding": "",
                "drift": "",
                "finding": f"(R40 score={row.get('round40_score')}) termination={row.get('termination_reason','')}",
                "frozen_reference_answer": row.get("frozen_standard_answer", ""),
            }
    return prior


def writer_excerpt(prompt: str, head: int = 1000, evidence: int = 5200) -> str:
    anchor = prompt.find("[S1]")
    if anchor < 0:
        for marker in ("EXACT EVIDENCE", "Source S1", "<span-ref", "<chunk-id"):
            anchor = prompt.find(marker)
            if anchor >= 0:
                break
    if anchor < 0:
        return clip(prompt, head + evidence)
    head_text = prompt[:head]
    evidence_text = prompt[anchor: anchor + evidence]
    tail_note = "" if anchor + evidence >= len(prompt) else f"\n…[+{len(prompt) - anchor - evidence} more chars in raw case]"
    return head_text + "\n…[instructions clipped]…\n" + evidence_text + tail_note


def build_digest(case: dict, prior_row: dict | None) -> str:
    tr = case.get("trace") or {}
    calls = tr.get("model_calls") or []
    final = tr.get("final") or {}
    answer = str(case.get("answer") or case.get("final_output") or "")
    lines: list[str] = []
    cid = case.get("case_id") or case.get("benchmark_id")
    lines.append(f"# {cid}")
    lines.append(f"- question: {clip(case.get('query') or '', 500)}")
    if prior_row:
        lines.append(
            f"- prior_audit: score={prior_row.get('score')} layer={prior_row.get('layer') or prior_row.get('earliest_observed_layer')} "
            f"grounding={prior_row.get('grounding','')} drift={prior_row.get('drift','')}")
        lines.append(f"- prior_finding: {clip(prior_row.get('finding') or '', 400)}")
        ref = prior_row.get("frozen_reference_answer") or prior_row.get("reference") or ""
        lines.append(f"- frozen_reference: {clip(ref, 900)}")
    lines.append(f"- termination: {final.get('termination_reason') or ''} status={final.get('status')}")
    stage_counts = Counter(stage_of(m) for m in calls)
    lines.append(f"- model_calls_by_stage: {json.dumps(dict(stage_counts), ensure_ascii=False)}")
    lines.append("")
    lines.append("## FINAL ANSWER (verbatim, clipped)")
    lines.append(clip(answer, 1700))
    lines.append("")

    plans = tr.get("plans") or []
    if plans:
        p0 = plans[0]
        lines.append("## TASK PLAN")
        lines.append(
            f"- status={p0.get('status')} error_class={p0.get('error_class','')} records={p0.get('record_ids')}")
        tp = next((m for m in calls if stage_of(m) == "task_plan"), None)
        if tp:
            lines.append(f"- task_plan_output: {clip(tp.get('output') or '', 700)}")
        lines.append("")

    lines.append("## DECISION / TOOL TIMELINE")
    tool_results = {t.get("step"): t for t in (tr.get("tool_results") or [])}
    for d in (tr.get("decisions") or [])[:40]:
        t = tool_results.get(d.get("step")) or {}
        res = t.get("result") or {}
        err = ""
        if isinstance(res, dict):
            err = res.get("error_class") or ""
        lines.append(
            f"- step {d.get('step')} [{d.get('phase')}] {d.get('action')} args={clip(d.get('args'), 150)} "
            f"exec={t.get('execution_status') or 'ok'}{(' err=' + err) if err else ''}"
            f"{(' planner_error=' + str(d.get('planner_error'))) if d.get('planner_error') else ''}")
    stages = tr.get("web_search_stages") or []
    if stages:
        lines.append("### web_search stages")
        for s in stages[:20]:
            lines.append(
                f"- q={clip(s.get('query') or '', 110)} candidates={s.get('candidate_count')} "
                f"fetched={s.get('fetched_count')} evidence={s.get('evidence_count')} status={s.get('status')}")
    lines.append("")

    pe = tr.get("page_evidence") or []
    if pe:
        lines.append("## PAGE EVIDENCE (per URL)")
        grounded_total = rejected_total = 0
        for row in pe[:24]:
            data = row.get("data") or {}
            g = data.get("grounded_candidate_count") or 0
            r = data.get("rejected_ungrounded_count") or 0
            grounded_total += g
            rejected_total += r
            lines.append(
                f"- {clip(row.get('url') or data.get('url') or '', 110)} chunks={data.get('chunk_count')} "
                f"grounded={g} rejected={r}")
        lines.append(f"- TOTAL grounded={grounded_total} rejected={rejected_total} urls={len(pe)}")
        lines.append("")

    control = [m for m in calls if stage_of(m) in CONTROL_STAGES]
    if control:
        lines.append("## CONTROL-STAGE RWKV OUTPUTS (resolution / review / replan)")
        shown = control[:4] + control[-8:] if len(control) > 12 else control
        seen = set()
        for m in shown:
            key = id(m)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- [{stage_of(m)} in={m.get('prompt_tokens')}tok] out: {clip(m.get('output') or '', 320)}")
        lines.append("")

    writer = [m for m in calls if stage_of(m) in WRITER_STAGES]
    if writer:
        w = writer[-1]
        ctxs = tr.get("contexts") or []
        if ctxs:
            c0 = ctxs[0]
            keys = ("source_count", "context_tokens", "bound_evidence_record_count",
                    "candidate_evidence_record_count", "evidence_resolution_status",
                    "factual_task_records_with_selected_sources")
            lines.append("## WRITER CONTEXT STATS")
            lines.append(json.dumps({k: c0.get(k) for k in keys if k in c0}, ensure_ascii=False))
        lines.append("")
        lines.append(f"## WRITER INPUT (head + evidence section; prompt={len(w.get('prompt') or '')} chars, in={w.get('prompt_tokens')}tok)")
        lines.append(writer_excerpt(str(w.get("prompt") or "")))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--out-subdir", default="causal_digests")
    args = ap.parse_args()
    audit_dir = Path(args.audit_dir)
    raw_dir = audit_dir / "raw_cases_gz"
    out_dir = audit_dir / args.out_subdir
    out_dir.mkdir(exist_ok=True)
    prior = load_prior_audit(audit_dir)
    total = 0
    sizes = []
    for gz in sorted(raw_dir.glob("*.json.gz")):
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            case = json.load(f)
        cid = case.get("case_id") or case.get("benchmark_id")
        digest = build_digest(case, prior.get(cid))
        (out_dir / f"{gz.stem.replace('.json','')}.md").write_text(digest, encoding="utf-8")
        sizes.append(len(digest))
        total += 1
    print(json.dumps({
        "audit_dir": str(audit_dir), "digests": total,
        "avg_chars": int(sum(sizes) / max(1, len(sizes))), "max_chars": max(sizes or [0]),
    }))


if __name__ == "__main__":
    main()
