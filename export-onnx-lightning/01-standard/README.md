# 1. Readable BF16 baseline

```bash
python 01-standard/export.py --model-dir /models/lightning --out /models/onnx-standard
```

No `lightning` custom operations are emitted. By default attention uses ORT's
`com.microsoft::GroupQueryAttention`; Mamba, routing and expert math use
ordinary ONNX nodes.

For an explicit attention graph too:

```bash
python 01-standard/export.py --model-dir /models/lightning \
  --out /models/onnx-standard-explicit --attention standard
```

Read [checkpoint.py](../shared/checkpoint.py), [graph.py](../shared/graph.py),
then the non-custom branches in [layers.py](../shared/layers.py).
The selected experts are gathered before evaluation; this baseline prioritizes
readability and correctness, not efficient expert-weight memory movement.
Ordinary ONNX does not mean every operator/dtype is supported by every runtime.

The [main guide](../README.md) covers H100 validation and limitations.
