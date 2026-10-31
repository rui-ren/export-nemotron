"""Guarded CUDA ONNX tests; a busy GPU is a blocker, never a successful skip."""
import argparse
import gc
import os
from pathlib import Path
import unittest

import gguf
import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
import torch

from test_onnx import packed_fixture

LIBRARY = None


def run(operation, inputs, output_types, attrs, domain="lightning_q4"):
    types = {torch.bfloat16: 16, torch.float32: 1, torch.int32: 6,
             torch.int64: 7, torch.uint8: 2}
    names = list(inputs)
    graph = helper.make_graph([
        helper.make_node(operation, names, [f"y{i}" for i in range(len(output_types))],
                         domain=domain, **attrs)], operation,
        [helper.make_tensor_value_info(name, types[value.dtype], list(value.shape))
         for name, value in inputs.items()],
        [helper.make_tensor_value_info(f"y{i}", dtype, shape)
         for i, (dtype, shape) in enumerate(output_types)])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid(domain, 1)], ir_version=10)
    options = ort.SessionOptions()
    options.register_custom_ops_library(str(LIBRARY))
    cuda = {"device_id": "0", "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)}
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options,
                                  providers=[("CUDAExecutionProvider", cuda)])
    binding = session.io_binding()
    keepalive = []
    for name, value in inputs.items():
        ort_value = ort.OrtValue.from_dlpack(value.contiguous().__dlpack__())
        keepalive.append(ort_value)
        binding.bind_ortvalue_input(name, ort_value)
    for i in range(len(output_types)):
        binding.bind_output(f"y{i}", "cuda", 0)
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return [torch.from_dlpack(value).clone() for value in binding.get_outputs()]


def unpack(packed, format):
    return torch.from_numpy(gguf.dequantize(
        packed, gguf.GGMLQuantizationType(format))).to(device="cuda", dtype=torch.bfloat16)


class CudaOperatorTests(unittest.TestCase):
    def tearDown(self):
        torch.cuda.synchronize()
        gc.collect()
        import pynvml as nv
        nv.nvmlInit()
        try:
            handle = nv.nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
            if (len(nv.nvmlDeviceGetComputeRunningProcesses(handle)) > 1
                    or nv.nvmlDeviceGetMemoryInfo(handle).free < 2 * 2**30):
                raise RuntimeError("CUDA qualification ownership/headroom guard failed")
        finally:
            nv.nvmlShutdown()

    def test_all_block_formats_and_embedding_rows(self):
        for format in (6, 8, 12):
            with self.subTest(format=format):
                packed = packed_fixture(format)
                matrix = torch.from_numpy(packed).cuda()
                expected = unpack(packed, format)
                actual = run("Dequantize", {"matrix": matrix}, [(16, [17, 512])],
                             {"n": 17, "k": 512, "format": format})[0]
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                ids = torch.tensor([[16, 0, 7, 7]], dtype=torch.int64, device="cuda")
                actual = run("Embedding", {"ids": ids, "matrix": matrix}, [(16, [1, 4, 512])],
                             {"n": 17, "k": 512, "format": format})[0]
                torch.testing.assert_close(actual, expected[ids], atol=0, rtol=0)

    def test_dense_matches_dequantized_bf16_native_rows(self):
        for format in (6, 8, 12):
            for rows in (1, 17):
                with self.subTest(format=format, rows=rows):
                    packed = packed_fixture(format, n=64)
                    x = (torch.randn((1, rows, 512), device="cuda") * 0.01).bfloat16()
                    actual = run("Linear", {"x": x, "matrix": torch.from_numpy(packed).cuda()},
                                 [(16, [1, rows, 64])], {"n": 64, "k": 512, "format": format})[0]
                    expected = run("RowLinear", {"x": x, "matrix": unpack(packed, format).t().contiguous()},
                                   [(16, [1, rows, 64])], {}, domain="lightning")[0]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_moe_decode_and_heterogeneous_cohorts_match_dequantized_native(self):
        hidden, inter, experts = 2688, 1856, 9
        for rows in (1, 3):
            with self.subTest(rows=rows):
                up = packed_fixture(6, n=experts * inter, k=hidden).reshape(experts, inter, -1)
                down = packed_fixture(8, n=experts * hidden, k=inter).reshape(experts, hidden, -1)
                x = (torch.randn((rows, hidden), device="cuda") * 0.01).bfloat16()
                indices = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 2, 3, 4, 5, 6],
                                        [1, 2, 4, 5, 6, 7]][:rows], device="cuda", dtype=torch.int32)
                probs = torch.rand((rows, 6), device="cuda")
                probs /= probs.sum(-1, keepdim=True)
                shapes = [(16, [rows, hidden]), (16, [rows * 6, inter]), (1, [rows, 6, hidden])]
                shared = {"x": x, "indices": indices, "probs": probs}
                actual = run("MoE", {**shared, "up": torch.from_numpy(up).cuda(),
                                     "down": torch.from_numpy(down).cuda()}, shapes,
                             {"hidden": hidden, "intermediate": inter, "experts": experts,
                              "up_format": 6, "down_format": 8})
                expected = run("MoeCohortBF16", {**shared, "up": unpack(up, 6),
                                                "down": unpack(down, 8)}, shapes,
                               {"topk": 6, "source_order": 1}, domain="lightning")
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a.reshape(-1), b.reshape(-1), atol=0, rtol=0)

    def test_invalid_embedding_and_expert_ids_fail_before_access(self):
        matrix = torch.from_numpy(packed_fixture(6)).cuda()
        for token in (-1, 17):
            with self.subTest(token=token), self.assertRaisesRegex(Exception, "token ID out of range"):
                run("Embedding", {"ids": torch.tensor([[token]], device="cuda"), "matrix": matrix},
                    [(16, [1, 1, 512])], {"n": 17, "k": 512, "format": 6})
        hidden, inter, experts = 2688, 1856, 6
        for expert in (-1, experts):
            with self.subTest(expert=expert), self.assertRaisesRegex(Exception, "expert ID out of range"):
                run("MoE", {
                    "x": torch.zeros((1, hidden), dtype=torch.bfloat16, device="cuda"),
                    "indices": torch.tensor([[0, 1, 2, 3, 4, expert]], dtype=torch.int32, device="cuda"),
                    "probs": torch.ones((1, 6), device="cuda"),
                    "up": torch.empty((experts, inter, hidden // 32 * 22), dtype=torch.uint8, device="cuda"),
                    "down": torch.empty((experts, hidden, inter // 32 * 22), dtype=torch.uint8, device="cuda"),
                }, [(16, [1, hidden]), (16, [6, inter]), (1, [1, 6, hidden])],
                    {"hidden": hidden, "intermediate": inter, "experts": experts,
                     "up_format": 6, "down_format": 6})


def main():
    global LIBRARY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        parser.error("CUDA_VISIBLE_DEVICES must select exactly --gpu-index")
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(args.gpu_index)
    if nv.nvmlDeviceGetComputeRunningProcesses(handle) or nv.nvmlDeviceGetMemoryInfo(handle).free < 70 * 2**30:
        raise RuntimeError("CUDA qualification blocked: requires an idle GPU with >=70 GiB free")
    nv.nvmlShutdown()
    LIBRARY = args.library.resolve()
    torch.manual_seed(47)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CudaOperatorTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
