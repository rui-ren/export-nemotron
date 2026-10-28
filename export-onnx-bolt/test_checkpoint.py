"""Test safetensors checkpoint loading without a real model or GPU."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HAS_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "safetensors")
)


@unittest.skipUnless(HAS_DEPENDENCIES, "Loader tests require torch and safetensors")
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file
        from checkpoint import SafetensorsCheckpoint

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "config.json").write_text(json.dumps({"llm_config": {"hidden_size": 2}}))
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"weight": "shard-1.safetensors", "bias": "shard-1.safetensors",
                           "other": "shard-2.safetensors"},
        }))
        save_file({"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.5, 0.75])},
                  self.root / "shard-1.safetensors")
        save_file({"other": torch.tensor([3.0, 4.0])}, self.root / "shard-2.safetensors")
        self.checkpoint = SafetensorsCheckpoint(self.root)

    def test_configuration_and_index(self):
        self.assertEqual(self.checkpoint.config["hidden_size"], 2)
        self.assertTrue(self.checkpoint.has("weight"))
        self.assertFalse(self.checkpoint.has("unknown"))
        self.assertEqual(self.checkpoint._open_shards, {})

    def test_reads_weights_and_stacks_across_shards(self):
        import torch

        torch.testing.assert_close(self.checkpoint.get("weight"), torch.tensor([1.0, 2.0]))
        actual = self.checkpoint.get_many(["weight", "other"], "cpu")
        torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_reuses_open_shards_without_quantization_imports(self):
        from safetensors import safe_open
        from checkpoint import SafetensorsCheckpoint

        with patch.dict(sys.modules, {"gguf": None, "quantization": None}):
            with patch("safetensors.safe_open", wraps=safe_open) as open_shard:
                reader = SafetensorsCheckpoint(self.root)
                reader.get("weight")
                reader.get("bias")
                reader.get("weight")
                self.assertEqual(open_shard.call_count, 1)

    def test_unknown_weight_raises(self):
        with self.assertRaises(KeyError):
            self.checkpoint.get("unknown")

    def test_missing_shard_raises(self):
        self.checkpoint.weight_map["missing"] = "missing.safetensors"
        with self.assertRaises(FileNotFoundError):
            self.checkpoint.get("missing")


if __name__ == "__main__":
    unittest.main()
