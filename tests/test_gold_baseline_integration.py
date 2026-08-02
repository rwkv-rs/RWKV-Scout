import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_gold_runs import load_gold, score
from scripts.run_concurrent_json_suite_20260731 import load_cases
from scripts.run_json_acceptance import _load_cases


class GoldBaselineIntegrationTests(unittest.TestCase):
    def test_gold_question_and_id_aliases_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "WR-TEST",
                        "question": "标准答案题目",
                        "final_answer": "参考答案",
                        "gold": {"required_facts": ["事实"]},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            concurrent_case = load_cases(path)[0]
            acceptance_case = _load_cases(path)[0]

        for case in (concurrent_case, acceptance_case):
            self.assertEqual(case["case_id"], "WR-TEST")
            self.assertEqual(case["query"], "标准答案题目")
            self.assertEqual(case["final_answer"], "参考答案")
            self.assertEqual(case["gold"]["required_facts"], ["事实"])

    def test_current_cases_shape_is_scored_against_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            gold_path = Path(directory) / "gold.jsonl"
            run_path = Path(directory) / "run.json"
            gold_path.write_text(
                json.dumps(
                    {
                        "id": "WR-TEST",
                        "question": "题目",
                        "final_answer": "标准答案",
                        "gold": {"required_facts": ["事实"], "forbidden_facts": []},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            run_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "WR-TEST",
                                "final_output": "模型回答包含事实",
                                "status": "completed",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = score("current", run_path, load_gold(gold_path))

        self.assertEqual(report["n"], 1)
        self.assertEqual(report["strict_pass"], 1)
        self.assertEqual(report["details"][0]["reference_answer"], "标准答案")


if __name__ == "__main__":
    unittest.main()
