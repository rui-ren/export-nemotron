"""Dependency-free checks; these do not validate CUDA or model numerics."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from shared.export import parse_args


ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_custom_kernel_build_sources_are_centralized(self):
        kernels = ROOT.parent / "src" / "lightning"
        self.assertTrue((kernels / "lightning_ops.cu").is_file())
        self.assertTrue((kernels / "README.md").is_file())
        cmake = (kernels / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("add_library(lightning_ops SHARED lightning_ops.cu)", cmake)
        self.assertFalse((ROOT / "02-custom" / "kernels" / "lightning_ops.cu").exists())

    def test_version_entrypoints_work_from_another_directory(self):
        for folder in ("01-standard", "02-custom", "03-int4", "04-mtp"):
            with self.subTest(folder=folder), tempfile.TemporaryDirectory() as working:
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / folder / "export.py"), "--help"],
                    cwd=working, capture_output=True, text=True, check=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.assertIn("--model-dir", result.stdout)
                self.assertNotIn("--variant", result.stdout)

    def test_each_entrypoint_selects_its_own_variant(self):
        for variant in ("bf16", "custom", "int4", "mtp"):
            with self.subTest(variant=variant):
                args = parse_args(["--model-dir", "unused", "--out", "unused"], variant)
                self.assertEqual(args.variant, variant)

    def test_legacy_cli_remains_available(self):
        args = parse_args(["--model-dir", "unused", "--out", "unused", "--variant", "custom"])
        self.assertEqual(args.variant, "custom")

    def test_invalid_export_options_fail_before_importing_dependencies(self):
        for extra in (["--max-length", "0"], ["--block-size", "17"], ["--dt-min", "nan"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--model-dir", "unused", "--out", "unused", *extra], "custom")


if __name__ == "__main__":
    unittest.main()
