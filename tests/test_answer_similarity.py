from __future__ import annotations

import unittest

from utils.answer_similarity import extract_canonical_dates, score_answer_similarity


class AnswerSimilarityTests(unittest.TestCase):
    def test_dates_are_normalized_across_presentation_styles(self):
        self.assertEqual(
            extract_canonical_dates("2026 年 6 月 10 日 and June 10, 2026"),
            {"2026-06-10"},
        )

    def test_semantically_equivalent_date_answer_can_pass_without_exact_format(self):
        result = score_answer_similarity(
            "Python 3.14.6，发布日期为 2026 年 6 月 10 日。",
            "Python 3.14.6 was released on June 10, 2026. [S1](https://python.org)",
            required_facts=["Python 3.14.6", "2026-06-10"],
        )
        self.assertGreaterEqual(result["score"], 0.90)
        self.assertEqual(result["forbidden_fact_matches"], [])

    def test_forbidden_fact_prevents_similarity_pass(self):
        result = score_answer_similarity(
            "Go 1.27 尚未正式发布，预计 2026 年 8 月发布。",
            "Go 1.27 已经正式发布。",
            required_facts=["尚未正式发布", "2026-08"],
            forbidden_facts=["已经正式发布"],
        )
        self.assertLess(result["score"], 0.90)
        self.assertEqual(result["forbidden_fact_matches"], ["已经正式发布"])

    def test_dataset_specific_aliases_are_supported(self):
        result = score_answer_similarity(
            "第 72 届世界卫生大会于 2019 年通过。",
            "It was adopted by the 72nd World Health Assembly in 2019.",
            required_facts=["第 72 届世界卫生大会", "2019"],
            fact_aliases={"第 72 届世界卫生大会": ["72nd World Health Assembly"]},
        )
        self.assertGreaterEqual(result["score"], 0.90)


if __name__ == "__main__":
    unittest.main()
