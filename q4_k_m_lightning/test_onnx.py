"""ONNX packing contracts and the actual host/device block decoder against gguf."""
import ctypes
import os
from pathlib import Path
import unittest

import gguf
import numpy as np
import onnx
from onnx import helper

from onnx_export import original_qk_rows, replace_nodes
from validate_model import logit_metrics


def packed_fixture(format, n=17, k=512):
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(format)]
    data = np.random.default_rng(42 + format).integers(
        0, 256, (n, k // block, size), dtype=np.uint8)
    data[:, :, :2] = np.array([0.013], np.float16).view(np.uint8)
    if format == 12:
        data[:, :, 2:4] = np.array([0.007], np.float16).view(np.uint8)
    return data.reshape(n, -1)


class OnnxPackingTests(unittest.TestCase):
    def test_inverse_qk_permutation_restores_original_rows(self):
        original = np.arange(32 * 128 * 22, dtype=np.int64).reshape(4096, 22)
        converted = original.reshape(32, 2, 64, 22).swapaxes(1, 2).reshape(original.shape)
        np.testing.assert_array_equal(original_qk_rows(converted, 32), original)
        with self.assertRaises(ValueError):
            original_qk_rows(converted[:4095], 32)

    def test_graph_replacement_keeps_outputs_and_routes(self):
        nodes = [
            helper.make_node("Gather", ["embedding", "ids"], ["x"]),
            helper.make_node("RowLinear", ["x", "linear"], ["projected"], domain="lightning"),
            helper.make_node("MoeDeviceBF16", ["projected", "idx", "prob", "up", "down"],
                             ["out", "act", "parts"], domain="lightning", topk=6, source_order=1),
        ]
        model = helper.make_model(helper.make_graph(nodes, "test", [], []))
        matrix = {"packed_name": "packed", "format": 6, "n": 1856, "k": 2688,
                  "shape": [128, 1856, 1848]}
        counts = replace_nodes(model, {
            "embedding": {**matrix, "packed_name": "embedding.q"},
            "linear": {**matrix, "packed_name": "linear.q"},
            "up": {**matrix, "packed_name": "up.q"},
            "down": {**matrix, "packed_name": "down.q", "n": 2688, "k": 1856,
                     "shape": [128, 2688, 1276]},
        })
        self.assertEqual(counts, {"Embedding": 1, "Linear": 1, "MoE": 1})
        self.assertEqual(list(model.graph.node[-1].input),
                         ["projected", "idx", "prob", "up.q", "down.q"])
        self.assertEqual(list(model.graph.node[-1].output), ["out", "act", "parts"])
        self.assertTrue(all(node.domain == "lightning_q4" for node in model.graph.node))

    def test_rejects_expert_route_semantics_change(self):
        for topk, source_order in ((4, 1), (6, 0)):
            node = helper.make_node("MoeDeviceBF16", ["x", "idx", "prob", "up", "down"],
                                    ["out", "act", "parts"], domain="lightning",
                                    topk=topk, source_order=source_order)
            model = helper.make_model(helper.make_graph([node], "bad_routes", [], []))
            with self.assertRaisesRegex(ValueError, "six source-ordered"):
                replace_nodes(model, {"up": {}, "down": {}})


class BlockDecoderTests(unittest.TestCase):
    def test_compiled_host_device_decoder_matches_all_three_reference_formats(self):
        path = os.environ.get("Q4_BLOCK_TEST_LIBRARY")
        if not path:
            self.fail("Build cuda/test_blocks.cpp and set Q4_BLOCK_TEST_LIBRARY; not a validated skip")
        library = ctypes.CDLL(str(Path(path).resolve()))
        decode = library.decode_blocks
        decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        decode.restype = None
        for format in (6, 8, 12):
            with self.subTest(format=format):
                packed = packed_fixture(format)
                expected = gguf.dequantize(packed, gguf.GGMLQuantizationType(format))
                actual = np.empty_like(expected)
                decode(packed.ctypes.data, actual.ctypes.data, 17, 512, format)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


class LogitMetricTests(unittest.TestCase):
    def test_detects_distribution_drift_even_when_top1_is_unchanged(self):
        a = np.log(np.array([0.7, 0.2, 0.1], np.float32))
        b = np.log(np.array([0.6, 0.3, 0.1], np.float32))
        metrics = logit_metrics(a, b)
        expected = 0.7 * np.log(0.7 / 0.6) + 0.2 * np.log(0.2 / 0.3)
        self.assertTrue(metrics["top1_equal"])
        self.assertFalse(metrics["bitwise_equal"])
        self.assertAlmostEqual(metrics["kl_reference_to_candidate"], expected, places=7)
        self.assertGreater(metrics["relative_logit_l2"], 0)

    def test_rejects_nonfinite_native_logits(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            logit_metrics(np.array([0, 1], np.float32), np.array([0, np.nan], np.float32))


if __name__ == "__main__":
    unittest.main()
