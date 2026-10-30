from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_runtime", ROOT / "native_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class NativeRuntimeTest(unittest.TestCase):
    def setUp(self):
        runtime._PROVIDER_REGISTRATIONS.clear()
        self.maps = patch.object(runtime, "_mapped_cuda_providers", return_value=set())
        self.maps.start()
        self.addCleanup(self.maps.stop)

    def test_graph_overlay_and_provider_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("genai_config.json", *runtime._LIBRARIES, "cuda_provider.so"):
                (root / name).touch()
            config = Mock()
            og = SimpleNamespace(
                Config=Mock(return_value=config),
                Model=Mock(),
                register_execution_provider_library=Mock(),
            )
            with patch.dict(sys.modules, {"onnxruntime_genai": og}):
                result = runtime.create_model(
                    root, cuda_provider_library=root / "cuda_provider.so"
                )
            og.Config.assert_called_once_with(str(root.resolve()))
            og.register_execution_provider_library.assert_called_once_with(
                "CUDAExecutionProvider", str(root / "cuda_provider.so")
            )
            overlay = json.loads(config.overlay.call_args.args[0])
            self.assertEqual(
                overlay["model"]["decoder"]["session_options"]["custom_ops_libraries"],
                list(runtime._LIBRARIES),
            )
            self.assertEqual(overlay["model"]["decoder"]["filename"], "model.onnx")
            config.add_model_data.assert_called_once_with("model.onnx", runtime._GRAPH_BYTES)
            self.assertIs(result, og.Model.return_value)

    def test_registration_is_once_per_runtime_and_rejects_conflicting_paths(self):
        og = SimpleNamespace(register_execution_provider_library=Mock())
        provider = Path("/first/libonnxruntime_providers_cuda.so")
        runtime._register_cuda_provider(og, provider)
        runtime._register_cuda_provider(og, provider)
        og.register_execution_provider_library.assert_called_once()
        with self.assertRaisesRegex(ValueError, "already registered"):
            runtime._register_cuda_provider(
                og, Path("/second/libonnxruntime_providers_cuda.so")
            )
        other = SimpleNamespace(register_execution_provider_library=Mock())
        runtime._register_cuda_provider(other, provider)
        other.register_execution_provider_library.assert_called_once()

    def test_external_registration_must_have_a_verified_matching_mapping(self):
        provider = Path("/first/libonnxruntime_providers_cuda.so")
        og = SimpleNamespace(
            register_execution_provider_library=Mock(
                side_effect=RuntimeError(
                    "EP library is already registered under CUDAExecutionProvider"
                )
            )
        )
        with self.assertRaisesRegex(RuntimeError, "cannot verify"):
            runtime._register_cuda_provider(og, provider)
        with patch.object(runtime, "_mapped_cuda_providers", return_value={provider}):
            runtime._register_cuda_provider(og, provider)
        runtime._PROVIDER_REGISTRATIONS.clear()
        with (
            patch.object(
                runtime, "_mapped_cuda_providers", return_value={Path("/other/provider")}
            ),
            self.assertRaisesRegex(ValueError, "Conflicting mapped"),
        ):
            runtime._register_cuda_provider(og, provider)

    def test_registration_errors_are_not_suppressed(self):
        og = SimpleNamespace(
            register_execution_provider_library=Mock(
                side_effect=RuntimeError("invalid library")
            )
        )
        with self.assertRaisesRegex(RuntimeError, "invalid library"):
            runtime._register_cuda_provider(og, Path("/provider"))

    def test_missing_library_fails_before_model_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "genai_config.json").touch()
            og = SimpleNamespace(Config=Mock())
            with (
                patch.dict(sys.modules, {"onnxruntime_genai": og}),
                self.assertRaises(FileNotFoundError),
            ):
                runtime.create_model(root)
            og.Config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
