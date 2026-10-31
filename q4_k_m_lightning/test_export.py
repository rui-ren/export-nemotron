"""Audit failure cases using tiny GGUF files, without model weights or CUDA."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import gguf
import numpy as np

spec = importlib.util.spec_from_file_location("q4km_export", Path(__file__).with_name("export.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, tensors, preset=gguf.LlamaFileType.MOSTLY_Q4_K_M, quantized=False):
        path = self.root / name
        writer = gguf.GGUFWriter(path, "nemotron_h_moe")
        writer.add_file_type(preset)
        for key, value in tensors.items():
            if quantized:
                writer.add_tensor(key, gguf.quantize(value, gguf.GGMLQuantizationType.Q8_0),
                                  raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            else:
                writer.add_tensor(key, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path

    def test_preserves_unquantized_tensor_bytes(self):
        values = np.arange(64, dtype=np.float32).reshape(2, 32)
        before = self.write("before.gguf", {"weight": values})
        after = self.write("after.gguf", {"weight": values.copy()})
        report = module.audit(before, after, self.root / "audit.json")
        self.assertTrue(report["tensors"][0]["unchanged_bytes"])
        self.assertEqual(report["tensors"][0]["relative_weight_l2"], 0)

    def test_audits_fallback_quantized_blocks_against_original(self):
        values = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
        before = self.write("before.gguf", {"weight": values})
        after = self.write("after.gguf", {"weight": values}, quantized=True)
        report = module.audit(before, after, self.root / "audit.json")
        tensor = report["tensors"][0]
        self.assertFalse(tensor["unchanged_bytes"])
        self.assertEqual(tensor["type"], "Q8_0")
        self.assertGreater(tensor["relative_weight_l2"], 0)
        self.assertLess(tensor["relative_weight_l2"], 0.02)

    def test_rejects_inventory_and_shape_changes(self):
        before = self.write("before.gguf", {"weight": np.zeros((2, 32), np.float32)})
        for index, tensors in enumerate((
                {"renamed": np.zeros((2, 32), np.float32)},
                {"weight": np.zeros((4, 16), np.float32)})):
            after = self.write(f"after{index}.gguf", tensors)
            with self.assertRaisesRegex(ValueError, "inventory|Shape changed"):
                module.audit(before, after, self.root / "audit.json")

    def test_rejects_nonfinite_or_modified_unquantized_values(self):
        before = self.write("before.gguf", {"weight": np.ones((2, 32), np.float32)})
        for index, value in enumerate((np.nan, 2.0)):
            after = self.write(f"after{index}.gguf",
                               {"weight": np.full((2, 32), value, np.float32)})
            with self.assertRaisesRegex(ValueError, "Nonfinite|Unquantized tensor changed"):
                module.audit(before, after, self.root / "audit.json")

    def test_rejects_wrong_preset(self):
        tensors = {"weight": np.zeros((2, 32), np.float32)}
        before = self.write("before.gguf", tensors)
        after = self.write("after.gguf", tensors, gguf.LlamaFileType.MOSTLY_Q4_K_S)
        with self.assertRaisesRegex(ValueError, "mixed Q4_K_M"):
            module.audit(before, after, self.root / "audit.json")

    def test_rejects_changed_signed_zero_bytes(self):
        before = self.write("before.gguf", {"weight": np.zeros((2, 32), np.float32)})
        after = self.write("after.gguf", {"weight": -np.zeros((2, 32), np.float32)})
        with self.assertRaisesRegex(ValueError, "Unquantized tensor changed"):
            module.audit(before, after, self.root / "audit.json")


if __name__ == "__main__":
    unittest.main()
