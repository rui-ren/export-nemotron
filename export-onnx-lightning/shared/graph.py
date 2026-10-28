"""Record nodes and stream external tensors without materializing the full model."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from onnx import TensorProto, helper

FLOAT = TensorProto.FLOAT
BF16 = TensorProto.BFLOAT16
INT64 = TensorProto.INT64
INT32 = TensorProto.INT32
UINT8 = TensorProto.UINT8
MS = "com.microsoft"
CUSTOM = "lightning"

TYPES = {torch.float32: FLOAT, torch.bfloat16: BF16, torch.uint8: UINT8,
         torch.int64: INT64, torch.int32: INT32}


def raw(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return tensor.numpy().tobytes()


class TensorStore:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("wb")
        self.initializers = []
        self.names = set()

    def add(self, name: str, tensor: torch.Tensor) -> str:
        return self.add_chunks(name, tuple(tensor.shape), tensor.dtype, [tensor])

    def add_chunks(self, name: str, shape: tuple[int, ...], dtype: torch.dtype, chunks) -> str:
        if name in self.names:
            raise ValueError(f"Duplicate initializer: {name}")
        self.file.write(b"\0" * ((-self.file.tell()) % 64))
        offset = self.file.tell()
        for tensor in chunks:
            if tensor.dtype != dtype:
                raise TypeError(f"{name}: inconsistent streamed tensor dtype")
            self.file.write(raw(tensor))
        length = self.file.tell() - offset
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if length != expected:
            raise ValueError(f"{name}: wrote {length} bytes, expected {expected}")
        proto = TensorProto(name=name, data_type=TYPES[dtype], dims=shape)
        proto.data_location = TensorProto.EXTERNAL
        for key, value in (("location", self.path.name), ("offset", str(offset)), ("length", str(length))):
            entry = proto.external_data.add()
            entry.key, entry.value = key, value
        self.initializers.append(proto)
        self.names.add(name)
        return name

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.file.close()


class Graph:
    def __init__(self, store: TensorStore, dtype: torch.dtype):
        self.store, self.dtype = store, dtype
        self.type = TYPES[dtype]
        self.nodes, self.inputs, self.outputs = [], [], []
        self.state = []
        self.counter = 0

    def name(self, prefix):
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def op(self, kind, inputs, outputs=1, domain="", **attrs):
        names = [self.name(kind.lower()) for _ in range(outputs)]
        self.nodes.append(helper.make_node(kind, inputs, names, name=self.name(kind), domain=domain, **attrs))
        return names[0] if outputs == 1 else names

    def constant(self, values, dtype=INT64):
        scalar = isinstance(values, (int, float))
        tensor = helper.make_tensor(self.name("constant"), dtype, [] if scalar else [len(values)],
                                    [values] if scalar else values)
        return self.op("Constant", [], value=tensor)

    def cast(self, value, dtype):
        return self.op("Cast", [value], to=dtype)

    def reshape(self, value, shape):
        return self.op("Reshape", [value, self.constant(shape)])

    def input(self, name, dtype, shape):
        self.inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def output(self, value, name, dtype, shape):
        self.nodes.append(helper.make_node("Identity", [value], [name]))
        self.outputs.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def cache(self, name, dtype, shape):
        past = self.input("past_" + name, dtype, shape)
        self.state.append({"input": past, "output": "present_" + name, "dtype": dtype, "shape": shape})
        return past

    def present(self, value, name, dtype, shape):
        return self.output(value, "present_" + name, dtype, shape)
