from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx.reference import ReferenceEvaluator

from shared.checkpoint import Checkpoint, validate_config
from shared.export import main, parse_args
from fixtures import create_checkpoint
from shared.graph import TensorStore
from shared.quantization import dequantize_int4, quantize_int4
from shared.reference import TorchDecoder
from reference_ops import OPERATORS


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "source"
        self.config = create_checkpoint(self.model)

    def export(self, variant, attention="standard", extra=()):
        destination = self.root / f"{variant}-{attention}-{'-'.join(extra)}"
        args = ["--model-dir", str(self.model), "--out", str(destination),
                "--variant", variant, "--attention", attention,
                "--dtype", "float32", "--max-length", "8", *extra]
        options = parse_args(args)
        with contextlib.redirect_stdout(io.StringIO()):
            main(args)
        return destination, options

    def test_all_variants_match_eager_multistep(self):
        cases = [("bf16", "standard", ()), ("bf16", "gqa", ()),
                 ("custom", "gqa", ()), ("int4", "gqa", ()),
                 ("mtp", "standard", ()), ("mtp", "gqa", ("--mtp-backend", "custom"))]
        for variant, attention, extra in cases:
            with self.subTest(variant=variant, attention=attention, extra=extra):
                destination, options = self.export(variant, attention, extra)
                manifest = json.loads((destination / "manifest.json").read_text())
                model = onnx.load(destination / "model.onnx")
                evaluator = ReferenceEvaluator(model, new_ops=OPERATORS)
                reference = TorchDecoder(Checkpoint(self.model), options)
                states = {item["input"]: np.zeros(item["shape"], np.float32) for item in manifest["states"]}
                generator = np.random.default_rng(77)
                for position, token in enumerate((1, 3, 5, 2)):
                    feeds = {"input_ids": np.array([[token]], np.int64),
                             "seqlens_k": np.array([position], np.int32),
                             "total_sequence_length": np.array(position + 1, np.int32), **states}
                    hidden = None
                    if variant == "mtp":
                        hidden = torch.from_numpy(generator.normal(size=(1, 1, 8)).astype(np.float32))
                        feeds["target_hidden_states"] = hidden.numpy()
                    actual = dict(zip([value.name for value in model.graph.output], evaluator.run(None, feeds)))
                    expected = reference.step(token, hidden)
                    for name, value in expected.items():
                        np.testing.assert_allclose(actual[name], value.numpy(), atol=2e-6, rtol=2e-5,
                                                   err_msg=f"{variant}, step {position}, {name}")
                    states = {state["input"]: actual[state["output"]] for state in manifest["states"]}

    def test_standard_graph_runs_in_real_ort_cpu(self):
        destination, options = self.export("bf16")
        session = ort.InferenceSession(str(destination / "model.onnx"), providers=["CPUExecutionProvider"])
        manifest = json.loads((destination / "manifest.json").read_text())
        states = {value["input"]: np.zeros(value["shape"], np.float32) for value in manifest["states"]}
        reference = TorchDecoder(Checkpoint(self.model), options)
        for index, token in enumerate((1, 2, 4)):
            feeds = {"input_ids": np.array([[token]], np.int64), "seqlens_k": np.array([index], np.int32),
                     "total_sequence_length": np.array(index + 1, np.int32), **states}
            actual = dict(zip([item.name for item in session.get_outputs()], session.run(None, feeds)))
            expected = reference.step(token)
            for name, value in expected.items():
                np.testing.assert_allclose(actual[name], value.numpy(), atol=3e-6, rtol=3e-5, err_msg=name)
            states = {state["input"]: actual[state["output"]] for state in manifest["states"]}
        del session

    def test_bf16_metadata_and_weight_separation(self):
        output = self.root / "bf16-native"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--model-dir", str(self.model), "--out", str(output), "--variant", "bf16"])
        model = onnx.load(output / "model.onnx", load_external_data=False)
        for tensor in model.graph.initializer:
            self.assertFalse(tensor.raw_data)
            fields = {item.key: item.value for item in tensor.external_data}
            self.assertEqual(fields["location"], "model.onnx.data")
            self.assertEqual(int(fields["offset"]) % 64, 0)
        self.assertIn(onnx.TensorProto.BFLOAT16, {t.data_type for t in model.graph.initializer})
        self.assertNotIn("lightning", {node.domain for node in model.graph.node})

    def test_custom_graph_has_bolt_style_fusion_and_state_contract(self):
        for variant, decode_op in (("custom", "MoeDecodeBF16"), ("int4", "MoeDecode")):
            with self.subTest(variant=variant):
                destination, _ = self.export(variant)
                model = onnx.load(destination / "model.onnx", load_external_data=False)
                custom = [node for node in model.graph.node if node.domain == "lightning"]
                self.assertEqual([node.op_type for node in custom], ["MambaStep", "MoeRoute", decode_op])
                self.assertFalse(any(node.op_type in ("TopK", "Sigmoid") for node in model.graph.node))
                for node in custom:
                    expected = 5 if node.op_type == "MambaStep" else 3
                    self.assertEqual(len(node.output), expected)
                manifest = json.loads((destination / "manifest.json").read_text())
                state = next(item for item in manifest["states"] if item["input"] == "past_ssm.0")
                self.assertEqual(state["shape"], [1, 2, 3, 4])
                self.assertEqual(manifest["kernel_contract"], "bolt_style_lightning_v1")

    def test_custom_router_ties_and_unbiased_weights(self):
        from reference_ops import MoeRoute
        x = np.zeros((1, 8), np.float32)
        gate = np.zeros((4, 8), np.float32)
        bias = np.asarray([0, 1, 1, -1], np.float32)
        indices, weights, scores = MoeRoute._run(None, x, gate, bias, topk=2, scale=2.5)
        np.testing.assert_array_equal(indices, [[1, 2]])
        np.testing.assert_array_equal(scores, np.full((1, 4), 0.5, np.float32))
        np.testing.assert_allclose(weights, [[1.25, 1.25]])

    def test_invalid_configuration_and_overwrite_fail(self):
        validate_config(self.config)
        for key, value in (("moe_latent_size", 4), ("n_group", 2), ("attention_bias", True),
                           ("n_groups", 0), ("conv_kernel", 1), ("layer_norm_epsilon", float("nan"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**self.config, key: value})
        destination, _ = self.export("bf16")
        with self.assertRaises(FileExistsError):
            main(["--model-dir", str(self.model), "--out", str(destination), "--variant", "bf16"])

    def test_rtn_padding_zero_and_error_bound(self):
        value = torch.tensor([[0.0] * 19, [float(i - 9) for i in range(19)]])
        packed, scales = quantize_int4(value, 16, torch.float32)
        restored = dequantize_int4(packed, scales, 19)
        self.assertEqual(tuple(packed.shape), (2, 2, 8))
        self.assertTrue(torch.equal(restored[0], value[0]))
        error = (restored - value).abs()
        self.assertTrue((error <= scales.repeat_interleave(16, -1)[:, :19] / 2 + 1e-6).all())
        self.assertTrue(torch.all((packed[0] & 15) == 8))
        with self.assertRaises(ValueError):
            quantize_int4(torch.tensor([[float("nan")]]), 32, torch.float32)

    def test_streamed_tensor_length_is_checked(self):
        with TensorStore(self.root / "bad.data") as store:
            with self.assertRaises(ValueError):
                store.add_chunks("bad", (3,), torch.float32, [torch.ones(2)])


if __name__ == "__main__":
    unittest.main()
