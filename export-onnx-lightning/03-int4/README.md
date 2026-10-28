# 3. W4A16 RTN

```bash
python 03-int4/export.py --model-dir /models/lightning --out /models/onnx-int4
```

The input remains the original BF16 safetensors checkpoint. Quantization occurs
during export, before external weights are written. No intermediate BF16 ONNX
export is required, although that baseline should be validated first.

- Dense linears: ORT `MatMulNBits`, bits 4.
- Routed experts: Bolt-style `lightning::MoeRoute` -> `lightning::MoeDecode`.
- Mamba update: `lightning::MambaStep`; attention: ORT GQA.
- Embedding and LM head remain BF16. Router multiplication/normalization and
  recurrent state stay FP32. Activations remain BF16.
- Uses the same [custom library](../../src/lightning/lightning_ops.cu).

For each K-axis block, RTN uses `scale=max(abs(weight))/7`, rounds the scale to
the stored dtype, then rounds/clamps signed values to `[-7,7]`. The unsigned
stored value adds zero point 8. Two values share a byte, **low nibble first**.
All-zero blocks use scale 1; padding stores zero point 8.

Expert packed weights have shape `[E,N,ceil(K/block),block/2]`;
scales have shape `[E,N,ceil(K/block)]`. The default block is 32.
This is symmetric INT4 RTN, **not NVFP4, IQ2, GPTQ or AWQ**. No calibration
dataset is used and accuracy preservation is not guaranteed.

Read [quantization.py](../shared/quantization.py), then
[expert_weights and linear](../shared/layers.py).
Use the [main guide's comparisons](../README.md) to measure W4 quality against
the BF16 custom model. The output is not entirely INT4 and its size should not
be estimated simply as parameter count / 2.
