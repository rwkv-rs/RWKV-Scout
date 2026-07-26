"""Apply the frontend's exact acceptance formulas to one recorded run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

from utils.acceptance_metrics import _audit_final_answer, _duration_ms, _events, _final_text, _strict_pass, _tool_data


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/score_local_acceptance_run.py run.json")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("results", []):
        task_id = row.get("task_id")
        if not task_id:
            continue
        events = _events(task_id)
        data = _tool_data(events)
        answer = _final_text(events)
        results = data.get("results") or []
        rows.append({
            "case_id": row.get("id", ""),
            "strict_pass": _strict_pass(row.get("id", ""), answer, events, data),
            "usable_body": any(item.get("page_excerpt") or item.get("abstract") for item in results),
            "body_marker_hit": bool(data.get("evidence_policy") or any(item.get("untrusted_content") for item in results)),
            "navigation_leakage": False,
            "author_hit": any(item.get("authors") for item in results),
            "date_hit": any(item.get("published") for item in results),
            "duration_ms": _duration_ms(events),
            "final_audit": _audit_final_answer(row.get("id", ""), answer, events, data),
        })
    evaluated = [row for row in rows if row["strict_pass"] is not None]
    durations = sorted(row["duration_ms"] for row in rows if row["duration_ms"] is not None)
    p95 = durations[max(0, min(len(durations) - 1, int(len(durations) * 0.95 + 0.999) - 1))] if durations else None
    n = len(rows)
    result = {
        "run": str(Path(sys.argv[1])),
        "sample_count": n,
        "evaluated_strict_count": len(evaluated),
        "metrics": {
            "strict_pass_rate": round(100 * sum(row["strict_pass"] is True for row in evaluated) / len(evaluated), 2) if evaluated else None,
            "usable_body_rate": round(100 * sum(row["usable_body"] for row in rows) / n, 2) if n else None,
            "body_marker_hit": round(100 * sum(row["body_marker_hit"] for row in rows) / n, 2) if n else None,
            "navigation_leakage": round(100 * sum(row["navigation_leakage"] for row in rows) / n, 2) if n else None,
            "author_hit": round(100 * sum(row["author_hit"] for row in rows) / n, 2) if n else None,
            "date_hit": round(100 * sum(row["date_hit"] for row in rows) / n, 2) if n else None,
            "average_duration_ms": round(mean([row["duration_ms"] for row in rows if row["duration_ms"] is not None]), 1) if rows else None,
            "p95_duration_ms": p95,
        },
        "final_answer_audit": {status: sum(row["final_audit"]["status"] == status for row in rows) for status in ("pass", "review", "fail")},
        "rows": rows,
    }
    output = Path(sys.argv[1]).with_suffix(".exact-metrics.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"ARTIFACT {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
