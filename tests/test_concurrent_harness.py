import unittest
from pathlib import Path

from scripts.run_concurrent_json_suite_20260731 import build_part_command


class ConcurrentHarnessTests(unittest.TestCase):
    def test_part_command_passes_a_bounded_case_timeout(self):
        command = build_part_command(
            Path("part.json"),
            Path("result.json"),
            case_timeout_seconds=600.0,
        )
        self.assertEqual(command[-2:], ["--case-timeout-seconds", "600.0"])


if __name__ == "__main__":
    unittest.main()
