import unittest

from utils.calculation_check import build_calculation_check


class CalculationCheckTests(unittest.TestCase):
    def test_percentage_check_uses_question_operands_only_when_body_confirms_them(self):
        hint = build_calculation_check(
            "500 个实例约占完整 2,294 个实例的百分比是多少？",
            "The page states 500 instances and 2,294 total instances.",
        )
        self.assertIn("500 / 2294", hint)
        self.assertIn("21.8%", hint)

    def test_missing_body_operand_does_not_create_arithmetic(self):
        self.assertEqual(
            build_calculation_check(
                "500 个实例约占完整 2,294 个实例的百分比是多少？",
                "The page states 500 instances but not the total.",
            ),
            "",
        )

    def test_difference_check_is_absolute_and_unit_neutral(self):
        hint = build_calculation_check(
            "完整集合 2,294 比 Verified 500 多多少个实例？",
            "The source lists 2,294 total instances and 500 verified instances.",
        )
        self.assertIn("|2294 - 500| = 1794", hint)


if __name__ == "__main__":
    unittest.main()
