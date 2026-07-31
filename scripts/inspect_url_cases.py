import json
from pathlib import Path

data = json.loads(Path("data/evaluation/url_summary_results_direct_20260731.json").read_text(encoding="utf-8"))
for row in data.get("cases") or []:
    if row.get("case_id") not in {"url_004", "url_005", "url_007", "url_008"}:
        continue
    trace = row.get("trace") or {}
    print("CASE", row.get("case_id"))
    print("DECISIONS", json.dumps(trace.get("decisions") or [], ensure_ascii=True))
    print("ANSWER", str(row.get("answer") or "")[:1200].replace("\n", " "))
