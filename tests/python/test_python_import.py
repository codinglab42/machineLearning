import unittest
from pathlib import Path

import ml_core as ml


class TestPythonModuleImport(unittest.TestCase):
    def test_module_reports_version_and_smoke_check(self):
        self.assertEqual(ml.__version__, "3.0.0")
        self.assertIn("working correctly", ml.test_library())
        build_root = Path(__file__).resolve().parents[2] / "build"
        self.assertIn(build_root.resolve(), Path(ml.__file__).resolve().parents)