import unittest
from pathlib import Path

import ml_core as ml


class TestPythonModuleImport(unittest.TestCase):
    def test_module_reports_version_and_smoke_check(self):
        self.assertEqual(ml.__version__, "4.0.0")
        self.assertIn("v4.0.0 is working correctly", ml.test_library())
        self.assertTrue(ml.check_version().startswith("v4.0.0"))
        build_root = Path(__file__).resolve().parents[2] / "build"
        self.assertIn(build_root.resolve(), Path(ml.__file__).resolve().parents)