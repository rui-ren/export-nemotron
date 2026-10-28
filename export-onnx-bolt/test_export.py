"""CPU-only tests. Full graph tests need PyTorch, but never a real checkpoint."""
from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from export_bolt import parse_args
from graph import GraphBuilder
from tensor_store import ExternalTensorStore

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_GGUF = importlib.util.find_spec("gguf") is not None


class ExternalDataTests(unittest.TestCase):
    def test_external_weight_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays = [np.ones((2, 3), np.float32), np.zeros(3, np.float32)]
            with ExternalTensorStore(Path(temporary) / "model.onnx.data") as store:
                for index, array in enumerate(arrays):
                    store.add(f"weight.{index}", array)
            for tensor, expected in zip(store.initializers, arrays):
                self.assertEqual(tensor.data_location, TensorProto.EXTERNAL)
                self.assertFalse(tensor.raw_data)
                fields = {item.key: item.value for item in tensor.external_data}
                self.assertEqual(fields["location"], "model.onnx.data")
                self.assertEqual(int(fields["offset"]) % 64, 0)
                self.assertEqual(int(fields["length"]), expected.nbytes)
                actual = numpy_helper.to_array(tensor, base_dir=temporary)
                np.testing.assert_array_equal(actual, expected)

    def test_scalar_and_noncontiguous_array(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "weights.data"
            arrays = [np.array(2.0, np.float32), np.arange(12, dtype=np.int32).reshape(3, 4).T]
            with ExternalTensorStore(data) as store:
                for index, array in enumerate(arrays):
                    store.add(str(index), array)
            for tensor, expected in zip(store.initializers, arrays):
                actual = numpy_helper.to_array(tensor, base_dir=temporary)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape, expected.shape)

    def test_duplicate_and_closed_store_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            with ExternalTensorStore(Path(temporary) / "weights.data") as store:
                store.add("w", np.ones(2, np.float32))
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    store.add("w", np.ones(2, np.float32))
            with self.assertRaisesRegex(ValueError, "closed"):
                store.add("another", np.ones(2, np.float32))

    def test_closes_file_on_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "expected"):
                with ExternalTensorStore(Path(temporary) / "weights.data") as store:
                    raise RuntimeError("expected")
            self.assertTrue(store.file.closed)

    def test_constant_shapes_and_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            with ExternalTensorStore(Path(temporary) / "weights.data") as store:
                graph = GraphBuilder(store)
                scalar = graph.const(3)
                vector = graph.const([3])
                self.assertNotEqual(scalar, vector)
                self.assertEqual(list(graph.nodes[0].attribute[0].t.dims), [])
                self.assertEqual(list(graph.nodes[1].attribute[0].t.dims), [1])


class CliTests(unittest.TestCase):
    def options(self, *extra):
        return parse_args(["--model-dir", "checkpoint", "--out", "output", *extra])

    def test_defaults(self):
        args = self.options()
        self.assertEqual(args.q8, "bf16")
        self.assertFalse(args.fused)
        self.assertEqual(args.num_layers, 0)

    def test_verify_implies_decode_and_fused(self):
        args = self.options("--verify-rows", "4", "--iq2", "native", "--experts", "native")
        self.assertTrue(args.decode)
        self.assertTrue(args.fused)

    def test_safetensors_is_the_only_source(self):
        args = self.options()
        self.assertFalse(hasattr(args, "weights"))
        self.assertFalse(hasattr(args, "gguf_path"))

    def test_unsupported_options_are_rejected(self):
        cases = [
            ["--fused"],
            ["--verify-rows", "9"],
            ["--verify-rows", "-1"],
            ["--num-layers", "-1"],
            ["--max-len", "0"],
            ["--dt-min", "nan"],
            ["--dt-min", "-1"],
            ["--iq2", "lut4"],
            ["--experts", "native"],
            ["--decode", "--fused", "--experts", "int8"],
            ["--decode", "--fused", "--experts", "lut4"],
            ["--verify-rows", "1", "--iq2", "lut4"],
            ["--weights", "gguf", "--gguf-path", ""],
            ["--weights", "hybrid", "--gguf-path", ""],
            ["--weights", "hf"],
            ["--gguf-path", "weights.gguf"],
        ]
        for case in cases:
            with self.subTest(case=case), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.options(*case)


@unittest.skipUnless(HAS_TORCH and HAS_GGUF, "Full tests require torch and gguf")
class FullExporterTests(unittest.TestCase):
    def test_cli_writes_both_files_and_refuses_overwrite(self):
        from unittest.mock import patch
        from export_bolt import main

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "export"
            args = ["--model-dir", "unused", "--out", str(output), "--device", "cpu"]
            with patch("checkpoint.SafetensorsCheckpoint", return_value=SyntheticCheckpoint()):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(args)
                self.assertTrue((output / "model.onnx").is_file())
                self.assertTrue((output / "model.onnx.data").is_file())
                model = onnx.load_model(output / "model.onnx")
                self.assertEqual(model.graph.output[0].name, "logits")
                with self.assertRaises(FileExistsError):
                    main(args)

    def test_graphs_match_original_exporter(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from layers import LayerBuilder
        from model import build_model

        root = Path(__file__).resolve().parent.parent

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        original_weights = load_module("original_bolt_weights", root / "bolt-weights.py")
        with patch.dict(sys.modules, {"bolt_weights": original_weights}), patch.object(sys, "path", sys.path.copy()):
            original = load_module("original_bolt_export", root / "export_bolt.py")

        def canonical_nodes(model):
            names = {item.name: item.name for item in model.graph.input}
            names.update({item.name: item.name for item in model.graph.initializer})
            names[""] = ""
            nodes = []
            for index, node in enumerate(model.graph.node):
                copy = onnx.NodeProto()
                copy.CopyFrom(node)
                copy.name = ""
                copy.input[:] = [names[name] for name in node.input]
                for output_index, name in enumerate(node.output):
                    names[name] = f"node{index}.output{output_index}"
                copy.output[:] = [names[name] for name in node.output]
                for attribute in copy.attribute:
                    if attribute.HasField("t"):
                        attribute.t.name = ""
                nodes.append(copy.SerializeToString(deterministic=True))
            return nodes

        modes = [
            [],
            ["--q8", "int8", "--iq2", "int4", "--experts", "int8"],
            ["--decode", "--fused", "--q8", "int8", "--iq2", "int4"],
            ["--verify-rows", "2", "--q8", "int8", "--iq2", "lut4", "--experts", "lut4"],
            ["--verify-rows", "2", "--q8", "int8", "--iq2", "native", "--experts", "native"],
        ]
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = parse_args(["--model-dir", "unused", "--out", str(root / "old"),
                                   "--device", "cpu", *mode])
                checkpoint = SyntheticCheckpoint()
                with patch.object(original, "Checkpoint", return_value=checkpoint):
                    legacy = original.Exporter(SimpleNamespace(**vars(args), weights="hf"))
                legacy.dev = "cpu"
                with contextlib.redirect_stdout(io.StringIO()):
                    legacy.build()
                before = onnx.load_model(root / "old" / "model.onnx")

                with ExternalTensorStore(root / "new.data") as store:
                    layers = LayerBuilder(args, checkpoint, store)
                    with contextlib.redirect_stdout(io.StringIO()):
                        after = build_model(layers)
                onnx.save_model(after, root / "new.onnx")
                after = onnx.load_model(root / "new.onnx")
                self.assertEqual(canonical_nodes(before), canonical_nodes(after))
                self.assertEqual(list(before.graph.input), list(after.graph.input))
                self.assertEqual(list(before.graph.output), list(after.graph.output))
                before_weights = {tensor.name: tensor for tensor in before.graph.initializer}
                after_weights = {tensor.name: tensor for tensor in after.graph.initializer}
                self.assertEqual(before_weights.keys(), after_weights.keys())
                for name, tensor in before_weights.items():
                    other = after_weights[name]
                    self.assertEqual(tensor.data_type, other.data_type, name)
                    self.assertEqual(list(tensor.dims), list(other.dims), name)
                    self.assertEqual(tensor.raw_data, other.raw_data, name)

    def test_quantized_formats(self):
        import torch
        import quantization as quant
        from gguf.quants import IQ2_XXS

        q8 = torch.zeros(2, 1, 34, dtype=torch.uint8)
        q8[..., :2] = torch.tensor([0.5], dtype=torch.float16).view(torch.uint8)
        signed = torch.arange(-16, 16, dtype=torch.int8).expand(2, 32)
        q8[..., 2:] = signed.view(torch.uint8).unsqueeze(1)
        torch.testing.assert_close(quant.dequantize_q8_0(q8), signed.float() * 0.5)
        values, scales = quant.q8_0_split(q8)
        torch.testing.assert_close((values.float() - 128) * scales.float().repeat_interleave(32, -1),
                                   quant.dequantize_q8_0(q8))

        generator = torch.Generator().manual_seed(7)
        iq2 = torch.randint(0, 256, (2, 1, 66), dtype=torch.uint8, generator=generator)
        iq2[..., :2] = torch.tensor([0.125], dtype=torch.float16).view(torch.uint8)
        decoded = quant.dequantize_iq2xxs(iq2)
        reference = torch.from_numpy(IQ2_XXS.dequantize(iq2.numpy().reshape(2, 66)))
        torch.testing.assert_close(decoded, reference)
        values, scales = quant.iq2xxs_to_int8(iq2)
        torch.testing.assert_close((values.float() - 128) * scales.repeat_interleave(32, -1), decoded)
        packed, scales = quant.iq2xxs_to_lut4(iq2)
        codes = torch.stack([packed & 15, packed >> 4], -1).reshape(2, 256).long()
        magnitudes = torch.tensor([8, 25, 43])[codes % 4]
        exact = magnitudes * torch.where(codes >= 4, -1, 1) * scales.repeat_interleave(32, -1)
        torch.testing.assert_close(exact, decoded)
        records, raw_scales = quant.iq2xxs_to_native(iq2)
        reconstructed = torch.cat([
            raw_scales.view(torch.int16).view(torch.uint8).reshape(2, 1, 2),
            records.reshape(2, 1, 64),
        ], -1)
        self.assertTrue(torch.equal(iq2, reconstructed))
        packed_int4, scales_int4 = quant.iq2xxs_to_int4(iq2)
        self.assertEqual(tuple(packed_int4.shape), (2, 128))
        self.assertEqual(tuple(scales_int4.shape), (2, 8))

        packed_fp4 = torch.tensor([[0x21] * 8], dtype=torch.uint8)
        actual = quant.dequantize_nvfp4(packed_fp4, torch.tensor([[2.0]]), torch.tensor(3.0))
        torch.testing.assert_close(actual, torch.tensor([[3.0, 6.0] * 8]))

    def test_checkpoint_reader(self):
        import json
        import torch
        from safetensors.torch import save_file
        from checkpoint import SafetensorsCheckpoint

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"llm_config": {"hidden_size": 2}}))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "shard.safetensors"}}))
            save_file({"weight": torch.tensor([1.0, 2.0])}, root / "shard.safetensors")
            checkpoint = SafetensorsCheckpoint(temporary)
            self.assertTrue(checkpoint.has("weight"))
            self.assertFalse(checkpoint.has("missing"))
            self.assertEqual(checkpoint.config["hidden_size"], 2)
            torch.testing.assert_close(checkpoint.get("weight"), torch.tensor([1.0, 2.0]))
            self.assertEqual(tuple(checkpoint.get_many(["weight", "weight"], "cpu").shape), (2, 2))
            with self.assertRaises(KeyError):
                checkpoint.get("missing")

    def test_synthetic_layer_stacks(self):
        from layers import LayerBuilder
        from model import build_model

        modes = [
            [],
            ["--q8", "int8", "--iq2", "int4", "--experts", "int8"],
            ["--decode", "--fused", "--q8", "int8", "--iq2", "int4"],
            ["--verify-rows", "2", "--q8", "int8", "--iq2", "lut4", "--experts", "lut4"],
            ["--verify-rows", "2", "--q8", "int8", "--iq2", "native", "--experts", "native"],
        ]
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                args = parse_args(["--model-dir", "unused", "--out", temporary, "--device", "cpu", *mode])
                checkpoint = SyntheticCheckpoint()
                with ExternalTensorStore(Path(temporary) / "model.onnx.data") as store:
                    layers = LayerBuilder(args, checkpoint, store)
                    with contextlib.redirect_stdout(io.StringIO()):
                        model = build_model(layers)
                path = Path(temporary) / "model.onnx"
                onnx.save_model(model, path)
                loaded = onnx.load_model(path)
                self.assertEqual(loaded.graph.output[0].name, "logits")
                operations = {node.op_type for node in loaded.graph.node}
                self.assertIn("GroupQueryAttention", operations)
                self.assertIn("MambaVerify" if args.verify_rows else "MambaStep" if args.fused
                              else "LinearAttention", operations)
                self.assertIn("MoeVerify" if args.verify_rows else "MoeDecode" if args.fused else "QMoE",
                              operations)
                imports = {item.domain for item in loaded.opset_import}
                self.assertEqual("bolt" in imports, args.fused)
                self.assertEqual("pred_ids" in {output.name for output in loaded.graph.output},
                                 bool(args.verify_rows))
                available = {item.name for item in loaded.graph.input}
                available.update(item.name for item in loaded.graph.initializer)
                for node in loaded.graph.node:
                    self.assertTrue(set(node.input) - {""} <= available, node.op_type)
                    available.update(name for name in node.output if name)
                self.assertTrue({item.name for item in loaded.graph.output} <= available)


class SyntheticCheckpoint:
    """Small, deterministic tensors in the real checkpoint layouts."""

    def __init__(self):
        self.config = {
            "layers_block_type": ["mamba", "attention", "moe"],
            "hidden_size": 256, "vocab_size": 32, "layer_norm_epsilon": 1e-5,
            "mamba_num_heads": 4, "mamba_head_dim": 64, "ssm_state_size": 8, "n_groups": 2,
            "conv_kernel": 4, "rms_norm_eps": 1e-5,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
            "moe_latent_size": 256, "moe_shared_expert_intermediate_size": 256,
            "routed_scaling_factor": 5.0,
            "block_configs": [{}, {}, {"n_routed_experts": 2, "num_experts_per_tok": 1}],
        }
        self.cfg = self.config

    def get(self, name, device="cpu"):
        import torch

        if name.endswith("embeddings.weight"):
            tensor = torch.zeros(32, 256, dtype=torch.bfloat16)
        elif name.endswith(("norm.weight", "norm_f.weight")):
            tensor = torch.ones(256, dtype=torch.bfloat16)
        elif name.endswith("conv1d.weight"):
            tensor = torch.zeros(288, 1, 4, dtype=torch.bfloat16)
        elif name.endswith("conv1d.bias"):
            tensor = torch.zeros(288, dtype=torch.bfloat16)
        elif name.endswith(("dt_bias", "A_log", ".D")):
            tensor = torch.zeros(4)
        elif name.endswith("gate.weight"):
            tensor = torch.zeros(2, 256, dtype=torch.bfloat16)
        elif name.endswith("gate.e_score_correction_bias"):
            tensor = torch.zeros(2)
        elif ".in_proj." in name or ".out_proj." in name:
            rows, columns = (548, 256) if ".in_proj." in name else (256, 256)
            if name.endswith(".weight_scale_2"):
                tensor = torch.tensor(1.0)
            elif name.endswith(".weight_scale"):
                tensor = torch.ones(rows, columns // 16).to(torch.float8_e4m3fn)
            else:
                tensor = torch.zeros(rows, columns // 2, dtype=torch.uint8)
        elif "experts." in name:
            tensor = torch.zeros(256, 1, 66, dtype=torch.uint8)
            tensor[..., :2] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
        else:
            rows = 32 if "lm_head" in name else 128 if any(
                part in name for part in ("k_proj", "v_proj")) else 256
            tensor = torch.zeros(rows, 8, 34, dtype=torch.uint8)
            tensor[..., :2] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
        return tensor.to(device)

    def get_many(self, names, device="cpu"):
        import torch

        return torch.stack([self.get(name) for name in names]).to(device)


if __name__ == "__main__":
    unittest.main()
