"""H100 operation tests, including Lightning dimensions and nonuniform RMS weights."""
import os
from pathlib import Path
import tempfile
import unittest

import onnx
from onnx import helper
import onnxruntime as ort
import torch

from shared.graph import CUSTOM, FLOAT, INT32, Graph, TensorStore
from shared.quantization import dequantize_int4, quantize_int4
from shared.reference import experts_core, mamba_core
from shared.runtime import run_cuda


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("LIGHTNING_OPS"),
                     "Requires CUDA and LIGHTNING_OPS; a skip is not GPU validation")
class KernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def session(self, operation, tensors, output_shapes, attributes):
        with TensorStore(self.root / "unused.data") as store:
            graph = Graph(store, torch.bfloat16)
            for name, tensor in tensors.items():
                dtype = FLOAT if tensor.dtype == torch.float32 else (
                    INT32 if tensor.dtype == torch.int32 else onnx.TensorProto.UINT8
                    if tensor.dtype == torch.uint8 else onnx.TensorProto.BFLOAT16)
                graph.input(name, dtype, list(tensor.shape))
            results = graph.op(operation, list(tensors), outputs=len(output_shapes),
                               domain=CUSTOM, **attributes)
            for index, (shape, dtype) in enumerate(output_shapes):
                graph.output(results[index], f"output{index}", dtype, shape)
            model = helper.make_model(helper.make_graph(graph.nodes, operation, graph.inputs, graph.outputs),
                                      opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(CUSTOM, 1)])
            model.ir_version = 10
        options = ort.SessionOptions()
        options.register_custom_ops_library(os.environ["LIGHTNING_OPS"])
        cuda = {"device_id": "0", "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)}
        if os.environ.get("CUDA_PROVIDER_LIBRARY"):
            ort.register_execution_provider_library("CUDAExecutionProvider", os.environ["CUDA_PROVIDER_LIBRARY"])
            devices = [d for d in ort.get_ep_devices()
                       if d.ep_name == "CUDAExecutionProvider" and d.ep_metadata.get("cuda_device_id") == "0"]
            if not devices:
                raise RuntimeError("CUDA plugin did not expose device 0")
            options.add_provider_for_devices(devices, cuda)
            return ort.InferenceSession(model.SerializeToString(), sess_options=options)
        return ort.InferenceSession(model.SerializeToString(), sess_options=options,
                                    providers=[("CUDAExecutionProvider", cuda)])

    def random(self, shape, dtype=torch.bfloat16):
        return (torch.randn(shape, device="cuda") * 0.1).to(dtype)

    def test_router_real_hidden_width_and_ties(self):
        for hidden in (128, 2688, 2691):
            with self.subTest(hidden=hidden):
                x, gate = self.random((1, hidden)), self.random((128, hidden))
                bias = self.random((128,), torch.float32)
                tensors = {"x": x, "gate": gate, "bias": bias}
                session = self.session("MoeRoute", tensors,
                                       [([1, 6], INT32), ([1, 6], FLOAT), ([1, 128], FLOAT)],
                                       {"topk": 6, "scale": 2.5})
                for tied in (False, True):
                    if tied:
                        tensors["x"] = torch.zeros_like(x)
                        tensors["bias"] = torch.zeros_like(bias)
                    actual = run_cuda(session, tensors)
                    scores = torch.nn.functional.linear(tensors["x"].float(), gate.float()).sigmoid()
                    indices = torch.argsort(scores + tensors["bias"], descending=True, stable=True)[:, :6]
                    weights = scores.gather(1, indices)
                    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 2.5
                    torch.testing.assert_close(actual["output0"], indices.to(torch.int32), atol=0, rtol=0)
                    torch.testing.assert_close(actual["output1"], weights, atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(actual["output2"], scores, atol=2e-6, rtol=2e-5)

    def test_mamba_real_dimensions_multistep(self):
        heads, dim, states, groups, kernel = 64, 64, 128, 8, 4
        inner, channels = heads * dim, heads * dim + 2 * groups * states
        tensors = {
            "projection": self.random((1, inner + channels + heads)),
            "past_conv": self.random((1, channels, kernel - 1)),
            "past_ssm": self.random((1, heads, states, dim), torch.float32),
            "conv_w": self.random((channels, 1, kernel)), "conv_b": self.random((channels,)),
            "dt_bias": self.random((heads,), torch.float32),
            "a": -torch.exp(self.random((heads,), torch.float32)),
            "skip": torch.ones(heads, dtype=torch.float32, device="cuda"),
            "norm": (1 + self.random((inner,))).to(torch.bfloat16).float(),
        }
        shapes = [([1, inner], onnx.TensorProto.BFLOAT16),
                  ([1, channels, kernel - 1], onnx.TensorProto.BFLOAT16),
                  ([1, heads, states, dim], FLOAT),
                  ([1, channels], onnx.TensorProto.BFLOAT16),
                  ([1, inner], onnx.TensorProto.BFLOAT16)]
        session = self.session("MambaStep", tensors, shapes,
                               {"heads": heads, "head_dim": dim, "d_state": states,
                                "n_groups": groups, "epsilon": 1e-5, "time_step_min": 0.0})
        reference_conv, reference_ssm = tensors["past_conv"], tensors["past_ssm"]
        for _ in range(4):
            tensors["projection"] = self.random(tensors["projection"].shape)
            inputs = list(tensors.values())
            inputs[1], inputs[2] = reference_conv, reference_ssm
            expected = mamba_core(*inputs, heads, dim, states, groups, 1e-5, 0.0)
            actual = run_cuda(session, tensors)
            for index in range(3):
                torch.testing.assert_close(actual[f"output{index}"], expected[index], atol=0.012, rtol=0.025)
            tensors["past_conv"], tensors["past_ssm"] = actual["output1"], actual["output2"]
            reference_conv, reference_ssm = expected[1], expected[2]

    def test_experts_real_dimensions_bf16_and_w4(self):
        for hidden, intermediate in ((2688, 1856), (131, 67)):
            for quantized in (False, True):
                with self.subTest(hidden=hidden, intermediate=intermediate, quantized=quantized):
                    x = self.random((1, hidden))
                    indices = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
                    probabilities = torch.tensor([[1.1, 1.4]], dtype=torch.float32, device="cuda")
                    up, down = self.random((3, intermediate, hidden)), self.random((3, hidden, intermediate))
                    tensors = {"x": x, "indices": indices, "probabilities": probabilities}
                    attributes = {"topk": 2}
                    if quantized:
                        up_pairs = [quantize_int4(value.cpu(), 32, torch.bfloat16) for value in up]
                        down_pairs = [quantize_int4(value.cpu(), 32, torch.bfloat16) for value in down]
                        qup, sup = (torch.stack([pair[index] for pair in up_pairs]) for index in (0, 1))
                        qdown, sdown = (torch.stack([pair[index] for pair in down_pairs]) for index in (0, 1))
                        tensors.update(up=qup.cuda(), up_scales=sup.cuda(),
                                       down=qdown.cuda(), down_scales=sdown.cuda())
                        up = dequantize_int4(qup, sup, hidden).to("cuda", torch.bfloat16)
                        down = dequantize_int4(qdown, sdown, intermediate).to("cuda", torch.bfloat16)
                        attributes["block_size"] = 32
                    else:
                        tensors.update(up=up, down=down)
                    session = self.session("MoeDecode" if quantized else "MoeDecodeBF16", tensors,
                                           [([1, hidden], onnx.TensorProto.BFLOAT16),
                                            ([2, intermediate], onnx.TensorProto.BFLOAT16),
                                            ([2, hidden], FLOAT)], attributes)
                    actual = run_cuda(session, tensors)
                    expected = experts_core(x, indices, probabilities, up, down)
                    torch.testing.assert_close(actual["output0"], expected, atol=0.08, rtol=0.04)


if __name__ == "__main__":
    unittest.main()
