"""Small helpers for describing ONNX operations, not executing them."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from onnx import NodeProto, TensorProto, ValueInfoProto, helper

from tensor_store import ExternalTensorStore

if TYPE_CHECKING:
    import torch

BF16 = TensorProto.BFLOAT16
F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
I32 = TensorProto.INT32
U8 = TensorProto.UINT8

MS = "com.microsoft"
BOLT = "bolt"


def value_info(name, dtype, shape) -> ValueInfoProto:
    """Describe an input/output. A string dimension means its size is dynamic."""
    return helper.make_tensor_value_info(name, dtype, shape)


class GraphBuilder:
    def __init__(self, store: ExternalTensorStore):
        self.store = store
        self.nodes: list[NodeProto] = []
        self.inputs: list[ValueInfoProto] = []
        self.outputs: list[ValueInfoProto] = []
        self.counter = 0

    def unique_name(self, prefix="tensor") -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def op(self, op_type, inputs, nout=1, domain="", name=None, **attributes):
        """Record an operation and return the names of its output tensors."""
        outputs = [self.unique_name(op_type.lower()) for _ in range(nout)]
        self.nodes.append(helper.make_node(
            op_type, inputs, outputs,
            name=name or self.unique_name("node_" + op_type),
            domain=domain, **attributes,
        ))
        return outputs[0] if nout == 1 else outputs

    def const(self, value, dtype=I64):
        """Keep tiny shape/axis constants inside the graph for readability."""
        if isinstance(value, (int, float)):
            values, shape = [value], []
        else:
            values = list(value)
            shape = [len(values)]
        tensor = helper.make_tensor(self.unique_name("constant"), dtype, shape, values)
        return self.op("Constant", [], value=tensor)

    def init(self, name: str, tensor: np.ndarray | torch.Tensor) -> str:
        """Store a weight externally; the graph only keeps its name."""
        if isinstance(tensor, np.ndarray):
            return self.store.add(name, tensor)

        import torch

        tensor = tensor.detach().contiguous().cpu()
        if tensor.dtype == torch.bfloat16:
            return self.store.add_raw(
                name, BF16, list(tensor.shape), tensor.view(torch.int16).numpy().tobytes(),
            )
        return self.store.add(name, tensor.numpy())
