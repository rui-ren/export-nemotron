from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark
import build
from rebuild import module_cleanup


class ArchiveTest(unittest.TestCase):
    def test_build_stage_cannot_overwrite_archive_even_through_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sources").mkdir()
            alias = root / "alias"
            alias.symlink_to(root, target_is_directory=True)
            with mock.patch.object(build, "ROOT", root):
                for destination in (root, alias):
                    with self.assertRaisesRegex(ValueError, "must not overlap"):
                        build.prepare_sources(destination)
            self.assertEqual(list((root / "sources").iterdir()), [])

    def test_opt_in_cleanup_changes_only_generated_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = build.prepare_sources(Path(directory))
            original = stage / "mamba/source/ssd_binary_source.cuh"
            before = original.read_text()
            module_cleanup.apply(stage)
            after = original.read_text()
            self.assertNotEqual(before, after)
            self.assertIn("std::unique_ptr<SourceSsdBinary>", after)
            provenance = json.loads((build.ROOT / "provenance.json").read_text())
            for name, entry in provenance["source_files"].items():
                self.assertEqual(build.digest(build.ROOT / "sources" / name), entry["sha256"])

    def test_mapped_runtime_rejects_mismatched_copies_and_missing_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good/libonnxruntime.so.1"
            bad = root / "other/libonnxruntime.so.1"
            for path in (good, bad):
                path.parent.mkdir()
            good.write_bytes(b"qualified")
            bad.write_bytes(b"wrong")
            maps = root / "maps"
            expected = {good.name: build.digest(good)}
            maps.write_text(f"1-2 r-xp 0 00:00 1 {good}\n")
            self.assertEqual(
                benchmark.verify_mapped_runtime(expected, maps)[str(good)]["sha256"],
                expected[good.name],
            )
            maps.write_text(maps.read_text() + f"2-3 r-xp 0 00:00 2 {bad}\n")
            with self.assertRaisesRegex(ValueError, "Mapped runtime mismatch"):
                benchmark.verify_mapped_runtime(expected, maps)
            maps.write_text("")
            with self.assertRaisesRegex(ValueError, "not mapped"):
                benchmark.verify_mapped_runtime(expected, maps)
            maps.write_text(f"1-2 r-xp 0 00:00 1 {good} (deleted)\n")
            with self.assertRaisesRegex(ValueError, "was deleted"):
                benchmark.verify_mapped_runtime(expected, maps)

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
