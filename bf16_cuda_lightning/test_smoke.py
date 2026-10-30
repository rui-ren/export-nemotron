"""Standalone BF16 CUDA checks; no exporter package or checkpoint required."""
import os
from pathlib import Path
import unittest

import onnx
from onnx import helper
import onnxruntime as ort
import torch


class CudaSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; a skipped test is not validation")
        cls.library = Path(os.environ["LIGHTNING_OPS"]).resolve(strict=True)

    def setUp(self):
        torch.manual_seed(47)

    def run_op(self, operation, feeds, outputs, **attributes):
        types = {torch.bfloat16: onnx.TensorProto.BFLOAT16,
                 torch.float32: onnx.TensorProto.FLOAT, torch.int32: onnx.TensorProto.INT32}
        inputs = [helper.make_tensor_value_info(name, types[tensor.dtype], list(tensor.shape))
                  for name, tensor in feeds.items()]
        names = [f"output{i}" for i in range(len(outputs))]
        node = helper.make_node(operation, list(feeds), names, domain="lightning", **attributes)
        graph = helper.make_graph([node], operation, inputs, [
            helper.make_tensor_value_info(name, types[dtype], shape)
            for name, (shape, dtype) in zip(names, outputs)])
        model = helper.make_model(graph, opset_imports=[
            helper.make_opsetid("", 21), helper.make_opsetid("lightning", 1)], ir_version=10)
        options = ort.SessionOptions()
        options.register_custom_ops_library(str(self.library))
        session = ort.InferenceSession(
            model.SerializeToString(), sess_options=options,
            providers=[("CUDAExecutionProvider", {
                "device_id": "0", "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})])
        binding, keepalive = session.io_binding(), []
        for name, tensor in feeds.items():
            value = ort.OrtValue.from_dlpack(tensor.contiguous().__dlpack__())
            keepalive.append(value)
            binding.bind_ortvalue_input(name, value)
        for name in names:
            binding.bind_output(name, "cuda", 0)
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return [torch.from_dlpack(value) for value in binding.get_outputs()]

    def test_row_linear_matches_source_logical_rows(self):
        x = (torch.randn(1, 2, 2688, device="cuda") * 0.1).bfloat16()
        weight = (torch.randn(4096, 2688, device="cuda") * 0.1).bfloat16()
        actual = self.run_op("RowLinear", {"x": x, "weight": weight.t().contiguous()},
                             [([1, 2, 4096], torch.bfloat16)])[0]
        expected = torch.cat([torch.nn.functional.linear(x[:, row:row + 1], weight)
                              for row in range(2)], dim=1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_gpu_routed_decode_and_verification_match_reference(self):
        hidden, intermediate, experts, routes = 2688, 1856, 8, 6
        x = (torch.randn(2, hidden, device="cuda") * 0.1).bfloat16()
        up = (torch.randn(experts, intermediate, hidden, device="cuda") * 0.1).bfloat16()
        down = (torch.randn(experts, hidden, intermediate, device="cuda") * 0.1).bfloat16()
        indices = torch.tensor([[7, 0, 3, 5, 1, 6], [2, 7, 0, 4, 6, 3]],
                               dtype=torch.int32, device="cuda")
        probs = torch.rand(2, routes, device="cuda")
        probs /= probs.sum(-1, keepdim=True)
        attributes = {"topk": routes, "source_order": 1}
        reference = []
        for row in range(2):
            feeds = {"x": x[row:row + 1], "idx": indices[row:row + 1],
                     "probs": probs[row:row + 1], "up": up, "down": down}
            shapes = [([1, hidden], torch.bfloat16),
                      ([routes, intermediate], torch.bfloat16), ([routes, hidden], torch.float32)]
            expected = self.run_op("MoeDecodeBF16", feeds, shapes, **attributes)
            actual = self.run_op("MoeDeviceBF16", feeds, shapes, **attributes)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            reference.append(expected)
        actual = self.run_op(
            "MoeVerifyBF16", {"x": x, "idx": indices, "probs": probs, "up": up, "down": down},
            [([2, hidden], torch.bfloat16), ([2 * routes, intermediate], torch.bfloat16),
             ([2, routes, hidden], torch.float32)], **attributes)
        expected = [torch.cat([row[i] for row in reference], dim=0) for i in (0, 1)]
        expected.append(torch.stack([row[2] for row in reference]))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
