import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_concurrent_json_suite_20260731 import build_part_command, merge_parts


class ConcurrentHarnessTests(unittest.TestCase):
    def test_merged_suite_is_written_as_one_complete_json_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = []
            cases = []
            for index in range(2):
                case = {
                    "case_id": f"case_{index + 1}",
                    "delivery": "answer",
                    "answer": f"answer {index + 1}",
                }
                cases.append(case)
                part = root / f"part_{index + 1:02d}.result.json"
                part.write_text(
                    json.dumps(
                        {
                            "state": "ready",
                            "started_at": "2026-08-13T00:00:00",
                            "case_timeout_seconds": 600,
                            "max_tool_steps_override": 50,
                            "cases": [case],
                        }
                    ),
                    encoding="utf-8",
                )
                parts.append(part)
            output = root / "merged.json"

            report = merge_parts("atomic", parts, output, root / "input.json", cases)

            self.assertEqual(report["state"], "ready")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["processed_cases"], 2)
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])

    def test_part_command_passes_a_bounded_case_timeout(self):
        command = build_part_command(
            Path("part.json"),
            Path("result.json"),
            case_timeout_seconds=600.0,
        )
        self.assertEqual(command[-2:], ["--case-timeout-seconds", "600.0"])

    def test_part_command_passes_suite_wide_step_override(self):
        command = build_part_command(
            Path("part.json"),
            Path("result.json"),
            case_timeout_seconds=600.0,
            max_tool_steps=50,
        )
        self.assertEqual(command[-4:], ["--case-timeout-seconds", "600.0", "--max-tool-steps", "50"])


if __name__ == "__main__":
    unittest.main()
