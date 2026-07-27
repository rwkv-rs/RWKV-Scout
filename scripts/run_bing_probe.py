"""Run the project's keyless Bing search directly against a UTF-8 JSON suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.web_search_keyless import search_web_keyless


def run(input_path: Path, output_path: Path) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("input JSON must contain a non-empty 'cases' list")

    rows = []
    for case in cases:
        query = str(case.get("query") or "").strip()
        if not query:
            raise ValueError("each case must contain a non-empty query")
        result = json.loads(search_web_keyless(query, max_results=8, fetch_pages=0))
        rows.append(
            {
                "case_id": case.get("case_id", ""),
                "query": query,
                "provider": result.get("provider", ""),
                "status": result.get("status", ""),
                "count": result.get("count", 0),
                "results": [
                    {
                        "rank": index,
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                    }
                    for index, item in enumerate(result.get("results") or [], start=1)
                    if isinstance(item, dict)
                ],
                "provider_errors": result.get("provider_errors") or [],
            }
        )

    report = {
        "suite": "direct-bing-probe",
        "input": str(input_path),
        "cases": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/evaluation/manual_real_queries.json")
    parser.add_argument("--output", default="data/evaluation/manual_bing_results.json")
    args = parser.parse_args()
    report = run(Path(args.input), Path(args.output))
    print(json.dumps({"output": args.output, "case_count": len(report["cases"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
