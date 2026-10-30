from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark
import build


class ArchiveTest(unittest.TestCase):
    def test_include_relocation_preserves_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = build.prepare_sources(Path(directory))
            manifest = json.loads((build.ROOT / "provenance.json").read_text())["source_files"]
            for name, entry in manifest.items():
                self.assertEqual(build.digest(build.ROOT / "sources" / name), entry["sha256"])
                for match in build.INCLUDE.finditer((stage / name).read_text()):
                    path = Path(match[2])
                    if not path.is_absolute():
                        path = (stage / name).parent / path
                    if match[2] != "onnxruntime_cxx_api.h":
                        self.assertTrue(path.is_file(), str(path))

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sources").mkdir()
            source = root / "sources/kernel.cu"
            source.write_text("original")
            entry = {"original": "/historical/kernel.cu", "sha256": build.digest(source)}
            (root / "provenance.json").write_text(
                json.dumps({"source_files": {"kernel.cu": entry}})
            )
            source.write_text("changed")
            with (
                mock.patch.object(build, "ROOT", root),
                self.assertRaisesRegex(ValueError, "Archived source changed"),
            ):
                build.prepare_sources(root / "stage")

    def test_rebuild_opt_in_does_not_allow_graph_or_unchanged_library_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.onnx", "genai_config.json", "fixed.so", "rebuilt.so"):
                (root / name).write_bytes(name.encode())
            provenance = {
                "graph_sha256": build.digest(root / "model.onnx"),
                "config_sha256": build.digest(root / "genai_config.json"),
                "required_assets": {},
                "measured_library_sha256": {
                    name: build.digest(root / name) for name in ("fixed.so", "rebuilt.so")
                },
                "required_unchanged_libraries": ["fixed.so"],
            }
            (root / "provenance.json").write_text(json.dumps(provenance))
            (root / "rebuilt.so").write_bytes(b"rebuilt")
            with mock.patch.object(benchmark, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "rebuilt.so"):
                    benchmark.verify_package(root, allow_rebuilt=False)
                benchmark.verify_package(root, allow_rebuilt=True)
                for name in ("model.onnx", "fixed.so"):
                    original = (root / name).read_bytes()
                    (root / name).write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, name):
                        benchmark.verify_package(root, allow_rebuilt=True)
                    (root / name).write_bytes(original)


if __name__ == "__main__":
    unittest.main()
