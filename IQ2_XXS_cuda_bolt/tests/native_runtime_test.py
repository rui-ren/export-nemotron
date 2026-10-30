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
            config.add_model_data.assert_called_once_with("model.onnx", runtime._GRAPH_BYTES)
            self.assertIs(result, og.Model.return_value)

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
