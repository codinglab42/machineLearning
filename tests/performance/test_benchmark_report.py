import importlib.util
import unittest
from pathlib import Path


script = Path(__file__).resolve().parents[2] / "scripts" / "generate_all_benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark_report", script)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkReportTests(unittest.TestCase):
    def test_missing_executable_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            benchmark.run_executable("/nonexistent/benchmark")

    def test_parser_rejects_missing_sequence_section(self):
        with self.assertRaisesRegex(ValueError, "Missing sequence section"):
            benchmark.parse_sequence_output("Benchmark completato!\n")

    def test_incomplete_or_unbalanced_measurements_are_rejected(self):
        incomplete = {section: [] for section in benchmark.EXPECTED["neural"]}
        with self.assertRaisesRegex(ValueError, "expected 5 rows"):
            benchmark.validate_result("neural", incomplete)
        row = {"samples": 100, "time_ms": 1.0, "accuracy": 1.0, "baseline": 1.0}
        data = {section: [row] * count for section, count in benchmark.EXPECTED["neural"].items()}
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            benchmark.validate_result("neural", data)

    def test_baseline_comparison_checks_case_and_threshold(self):
        baseline = {"linear": {"sample_scaling": [{"samples": 100, "time_ms": 10.0}]}}
        current = {"linear": {"sample_scaling": [{"samples": 100, "time_ms": 13.0}]}}
        self.assertEqual(len(benchmark.compare_baseline(current, baseline, 20)), 1)
        self.assertEqual(benchmark.compare_baseline(current, baseline, 30), [])
        current["linear"]["sample_scaling"][0]["samples"] = 200
        with self.assertRaisesRegex(ValueError, "Baseline case differs"):
            benchmark.compare_baseline(current, baseline, 20)


if __name__ == "__main__":
    unittest.main()
