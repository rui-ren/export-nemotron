"""ORT CUDA integration checks; requires BOLT_PLUGIN and built native libraries."""

from __future__ import annotations

import ctypes
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx_ir as ir
import torch
from gguf import GGMLQuantizationType, quants

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.float16: ir.DataType.FLOAT16,
    torch.float32: ir.DataType.FLOAT,
    torch.uint8: ir.DataType.UINT8,
    torch.int32: ir.DataType.INT32,
}


class TestTensor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("rank", ctypes.c_size_t),
        ("dtype", ctypes.c_int),
    ]


def descriptors(tensors, prefix):
    shapes = [(ctypes.c_int64 * t.ndim)(*t.shape) for t in tensors]
    descriptors = (TestTensor * len(tensors))(
        *[
            TestTensor(
                f"{prefix}_{i}".encode(),
                t.data_ptr(),
                t.numel() * t.element_size(),
                shape,
                t.ndim,
                int(DTYPES[t.dtype]),
            )
            for i, (t, shape) in enumerate(zip(tensors, shapes))
        ]
    )
    return descriptors, shapes


def run(domain, operator, inputs, outputs, **attributes):
    values = [
        ir.Value(
            name=f"input_{i}", shape=ir.Shape(t.shape), type=ir.TensorType(DTYPES[t.dtype])
        )
        for i, t in enumerate(inputs)
    ]
    results = [
        ir.Value(name=f"output_{i}", type=ir.TensorType(DTYPES[dtype]))
        for i, (dtype, _) in enumerate(outputs)
    ]
    node = ir.Node(
        domain,
        operator,
        values,
        outputs=results,
        attributes=[ir.AttrInt64(name, value) for name, value in attributes.items()],
    )
    graph = ir.Graph(values, results, nodes=[node], opset_imports={"": 21, domain: 1})
    model = ir.Model(graph, ir_version=10)
    library = (
        ROOT
        / "build/lib"
        / ("libbolt_q8_fp16.so" if domain == "bolt_fast_q8" else "libbolt_iq2_source.so")
    )
    runner = ctypes.CDLL(str(ROOT / "build/lib/libort_cuda_runner.so"))
    runner.RunTestModel.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(TestTensor),
        ctypes.c_size_t,
        ctypes.POINTER(TestTensor),
        ctypes.c_size_t,
    ]
    runner.RunTestModel.restype = ctypes.c_char_p
    buffers = [t.cuda().contiguous() for t in inputs]
    output_buffers = [
        torch.empty(shape, dtype=dtype, device="cuda") for dtype, shape in outputs
    ]
    in_desc, _in_shapes = descriptors(buffers, "input")
    out_desc, _out_shapes = descriptors(output_buffers, "output")
    torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.onnx"
        ir.save(model, path)
        error = runner.RunTestModel(
            str(path).encode(),
            os.environ["BOLT_PLUGIN"].encode(),
            str(library).encode(),
            in_desc,
            len(inputs),
            out_desc,
            len(outputs),
        )
    if error:
        raise RuntimeError(error.decode())
    torch.cuda.synchronize()
    return [value.cpu() for value in output_buffers]


def iq2_weights(*lead, k):
    # Real GGML 66-byte superblocks: FP16 scale followed by eight packed records.
    blocks = torch.randint(0, 256, (*lead, k // 256, 66), dtype=torch.uint8)
    scales = torch.linspace(0.001003, 0.007019, int(np.prod(lead)) * (k // 256))
    scales = scales.reshape(*lead, k // 256).half()
    blocks[..., :2] = scales.contiguous().view(torch.uint8).reshape(*lead, k // 256, 2)
    packed = blocks[..., 2:].reshape(*lead, k // 32, 8).contiguous()
    # The published ABI transports FP16 scale bits in a BF16-typed tensor.
    transported = scales.view(torch.bfloat16)
    dequantized = quants.dequantize(
        blocks.reshape(-1, (k // 256) * 66).numpy(), GGMLQuantizationType.IQ2_XXS
    )
    reference = torch.from_numpy(dequantized.copy()).reshape(*lead, k).bfloat16().float()
    return packed, transported, reference


class OperatorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.backends.cuda.matmul.allow_tf32 = False

    def assert_projection(self, actual, expected, tolerance=0.006):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        self.assertLess(float(error), tolerance)

    def test_q8_fp16_scales_single_row_and_higher_rank(self):
        torch.manual_seed(31)
        n, k = 48, 512
        packed = torch.randint(0, 256, (n, k // 32, 32), dtype=torch.uint8)
        scales = torch.linspace(0.001003, 0.027019, n * (k // 32)).reshape(n, k // 32).half()
        weights = ((packed.float() - 128) * scales.float()[..., None]).reshape(n, k)
        weights = weights.bfloat16().float()
        for shape in ((1, k), (2, 3, k)):
            with self.subTest(shape=shape):
                x = torch.randn(shape).bfloat16()
                (actual,) = run(
                    "bolt_fast_q8",
                    "Q8GemvFp16Scales",
                    [x, packed, scales],
                    [(torch.bfloat16, (*shape[:-1], n))],
                )
                self.assert_projection(actual, x.float() @ weights.T)

    def test_iq2_shared_single_and_multiple_rows(self):
        torch.manual_seed(32)
        packed, scales, weights = iq2_weights(48, k=512)
        for rows in (1, 6):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 512).bfloat16()
                (actual,) = run(
                    "bolt_iq2_source",
                    "Q4Gemv",
                    [x, packed, scales],
                    [(torch.bfloat16, (rows, 48))],
                    iq2=2,
                )
                self.assert_projection(actual, x.float() @ weights.T)

    def test_iq2_routed_single_and_multiple_rows(self):
        torch.manual_seed(33)
        experts, inter, topk = 4, 256, 3
        up, up_scale, up_ref = iq2_weights(experts, inter, k=1024)
        down, down_scale, down_ref = iq2_weights(experts, 1024, k=inter)
        for rows in (1, 4):
            with self.subTest(rows=rows):
                x = (torch.randn(rows, 1024) * 0.25).bfloat16()
                ids = torch.stack([torch.randperm(experts)[:topk] for _ in range(rows)]).int()
                weights = torch.rand(rows, topk)
                expected_h, expected_part = [], []
                expected = torch.zeros(rows, 1024)
                for row in range(rows):
                    for slot in range(topk):
                        expert = ids[row, slot]
                        h = (x[row].float() @ up_ref[expert].T).bfloat16().float()
                        h = h.relu().square().bfloat16().float()
                        part = (h @ down_ref[expert].T).bfloat16().float()
                        expected_h.append(h)
                        expected_part.append(part)
                        expected[row] += weights[row, slot] * part
                actual, hidden, parts, meta = run(
                    "bolt_iq2_source",
                    "MoeVerify",
                    [x, ids, weights, up, up_scale, down, down_scale],
                    [
                        (torch.bfloat16, (rows, 1024)),
                        (torch.bfloat16, (rows * topk, inter)),
                        (torch.float32, (rows * topk, 1024)),
                        (torch.int32, (1 + rows * topk * 9,)),
                    ],
                    topk=topk,
                    iq2=2,
                )
                self.assert_projection(hidden, torch.stack(expected_h))
                self.assert_projection(parts, torch.stack(expected_part))
                self.assert_projection(actual, expected)
                self.assertEqual(int(meta[0]), len(ids.unique()))

    def test_iq2_invalid_intermediate_size_is_rejected(self):
        # 320 passed the old 64-alignment check but cannot index 256-value scales.
        rows, experts, inter = 1, 2, 320
        inputs = [
            torch.zeros(rows, 1024, dtype=torch.bfloat16),
            torch.zeros(rows, 1, dtype=torch.int32),
            torch.ones(rows, 1),
            torch.zeros(experts, inter, 32, 8, dtype=torch.uint8),
            torch.zeros(experts, inter, 4, dtype=torch.bfloat16),
            torch.zeros(experts, 1024, inter // 32, 8, dtype=torch.uint8),
            torch.zeros(experts, 1024, inter // 256, dtype=torch.bfloat16),
        ]
        with self.assertRaisesRegex(Exception, "inter size must be a multiple of 256"):
            run(
                "bolt_iq2_source",
                "MoeVerify",
                inputs,
                [
                    (torch.bfloat16, (rows, 1024)),
                    (torch.bfloat16, (rows, inter)),
                    (torch.float32, (rows, 1024)),
                    (torch.int32, (1 + rows * 9,)),
                ],
                topk=1,
                iq2=2,
            )


if __name__ == "__main__":
    unittest.main()
