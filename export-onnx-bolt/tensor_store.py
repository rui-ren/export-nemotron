"""Write large weight arrays separately from the ONNX graph."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from onnx import TensorProto, helper


class ExternalTensorStore:
    """Append weights to a .data file and remember their ONNX references.

    An initializer describes a constant tensor: its name, shape and data type.
    Here it also records where the tensor's bytes live, instead of embedding
    those bytes in the graph.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file = self.path.open("wb")
        self.offset = 0
        self.initializers: list[TensorProto] = []
        self.names: set[str] = set()

    def add(self, name: str, array: np.ndarray) -> str:
        array = np.asarray(array)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        data_type = helper.np_dtype_to_tensor_dtype(array.dtype)
        return self.add_raw(name, data_type, array.shape, array.tobytes())

    def add_raw(self, name: str, data_type: int, shape: Sequence[int], data: bytes) -> str:
        """Store raw bytes (also supports BF16, which NumPy does not provide)."""
        if self.file.closed:
            raise ValueError("Cannot add a weight after the store has closed")
        if name in self.names:
            raise ValueError(f"Duplicate weight name: {name}")
        padding = (-self.offset) % 64
        self.file.write(b"\0" * padding)
        self.offset += padding
        self.file.write(data)

        tensor = TensorProto(name=name, data_type=data_type, dims=list(shape))
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in (
            ("location", self.path.name),
            ("offset", str(self.offset)),
            ("length", str(len(data))),
        ):
            entry = tensor.external_data.add()
            entry.key, entry.value = key, value
        self.offset += len(data)
        self.names.add(name)
        self.initializers.append(tensor)
        return name

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> ExternalTensorStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
