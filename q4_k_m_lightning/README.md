# Lightning Q4_K_M ONNX workflow (not yet GPU-qualified)

The deliverable is an **ONNX Runtime GenAI package** for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, not a GGUF inference model.
It contains `model.onnx`, mixed quantized external weights, a CUDA custom-op
library, tokenizer files and `genai_config.json`. Its runtime does **not** link
to llama.cpp or read GGUF files.

**Status:** the actual ONNX package has been exported, structurally audited and
its CUDA library compiled. **CUDA execution, full-model qualification and ONNX
benchmarks have not run.** GPU use is paused at the user's request while other
tests run. Do not treat compilation or CPU checks as GPU qualification.
[`ONNX_STATUS.json`](ONNX_STATUS.json) records the current evidence and blockers.

The existing BF16 artifacts and default runtime are unchanged. This first
ONNX implementation is target-only, without MTP. It is an experimental
correctness-first path, not a claim of faster decode or accepted model quality.

## What is actually quantized

All **164 matrices** selected by the pinned stock Q4_K_M preset are stored as
native packed UINT8 blocks in ONNX:

| Block type | Matrices |
| --- | --- |
| Q5_0 | 110 |
| Q8_0 | 25 |
| Q4_K | 29 |

Lightning's expert widths 2688/1856 are not divisible by the 256-element K-quant
block. Stock upstream policy therefore falls back to Q5_0/Q8_0 for 134 matrices.
The packing is not symmetric block-32 RTN and must not be reinterpreted as such.
The 237 nonquantized initializers retain the original ONNX bytes and precision.
The converter's Q/K row permutation is reversed before packing ONNX weights.

The created external ONNX weights are **24,489,993,984 bytes = 24.490 GB =
22.808 GiB**. This is a disk-weight size, **not measured GPU allocation**.
The 24.515 GB file and 23.604 GiB GPU result in the historical reference report
belong to a different runtime; they are not ONNX results.

Custom domain `lightning_q4` provides:

| Operator | Execution |
| --- | --- |
| `Embedding` | Dequantize only requested embedding rows to BF16 |
| `Linear` | Temporarily dequantize one matrix, then source-row BF16 cuBLAS |
| `MoE` | Group routes, temporarily dequantize one expert at a time, run source-row-configured BF16 cohort projections |
| `Dequantize` | Diagnostic operator for block-format qualification |

The graph replaces one embedding, 117 dense/head projections and 23 MoE nodes.
Mamba, attention, normalization, routing, recurrent-state ownership and cache
contracts remain on the existing BF16 implementation. No full resident BF16
expert copy is created. Projection/activation rounding and FP32 route reduction
are retained. Host route-count synchronization and temporary weight expansion
may cost performance; only actual ONNX benchmarks can establish the trade-off.

## Build and export

Use the qualified H100/ORT environment. The shared library compiles the BF16
kernel snapshot together with the new operators, registering both domains
through one `RegisterCustomOps` entry point.
Set `BF16_SOURCE_DIR` to `bf16_cuda_lightning/` from commit
`a77d0522eab5536d6c980c361f8aa93f8a50389b`
([BF16 PR](https://github.com/rui-ren/export-nemotron/pull/2)).

```bash
cmake -S q4_k_m_lightning/cuda -B build/lightning-q4-onnx \
  -DCMAKE_BUILD_TYPE=Release \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/session/headers \
  -DBF16_SOURCE_DIR=/path/to/qualified/bf16_cuda_lightning
cmake --build build/lightning-q4-onnx --parallel 2
```

`onnx_export.py` starts with the original safetensors checkpoint and the
qualified chunked BF16 ONNX package. The pinned upstream quantizer is an
**offline weight-preparation tool only**; its staging files are not included
in the runtime package. Use clean llama.cpp revision
`5047748f114564fcf271e6a648f60e2d2eae6264`.
The original BF16 checkpoint and quantizer staging need additional disk space.

```bash
python q4_k_m_lightning/onnx_export.py \
  --model-dir /models/lightning-bf16 \
  --bf16-package /models/genai-chunked-device \
  --llama-source /path/to/pinned/llama.cpp \
  --quantizer /path/to/llama-quantize \
  --work-dir /models/quantizer-staging \
  --custom-library build/lightning-q4-onnx/lib/liblightning_q4_ops.so \
  --out /models/genai-q4km-onnx
```

Destinations must be new. `--quantized-reference` optionally reuses the already
audited offline weight cache instead of repeating quantization. It is never a
GenAI input. The output configuration retains the original **262144** context
limit, not the offline converter's larger metadata value.

The machine-local package is
`/home/lightning/export-onnx/artifacts/genai-q4km-onnx`.
It remains explicitly marked `qualified: false` and `benchmarked: false`.

## CPU checks that have passed

- All 164 packed ONNX matrices match the quantized reference bits, accounting
  for independently checked Q/K row restoration.
- All 237 retained initializers match the qualified BF16 ONNX bytes.
- Original graph inputs/outputs and their types/shapes are unchanged.
- The actual shared host/device C++ decoder matches reference FP32 bits for
  synthetic blocks and **1,458,240 real values**, sampling all 164 matrices.
- **12 CPU regression tests** pass, including incompatible route-semantics
  rejection. Python compilation and the CUDA build pass.
- The built library registers both custom-op domains on ORT 1.32.0.
  Registration alone does not execute or qualify CUDA kernels.

```bash
c++ -O3 -ffp-contract=off -std=c++17 -fPIC -shared \
  -I/usr/local/cuda/include q4_k_m_lightning/cuda/test_blocks.cpp \
  -o /tmp/libq4_block_test.so
Q4_BLOCK_TEST_LIBRARY=/tmp/libq4_block_test.so \
PYTHONPATH=/path/to/pinned/llama.cpp/gguf-py \
python -m unittest discover -s q4_k_m_lightning -p 'test_*.py' -v
```

`audit_onnx.py` independently checks every packed/retained initializer against
the source packages and validates the unchanged decoder ABI. Its offline
reference dependency is not a runtime dependency.

## Pending GPU gates -- do not run during the user's GPU hold

The prepared `validate_ops.py` exercises all three formats, embedding rows,
decode/multirow dense projections, heterogeneous/shared/singleton expert
cohorts and invalid IDs. It compares against dequantized BF16 ONNX operators,
not an alternate inference engine. It requires the existing PyTorch/ORT
DLPack-enabled CUDA environment, `nvidia-ml-py`, and the pinned offline Python
block reader. It fails instead of skipping when a GPU is busy.

After GPU use is explicitly released, run operator tests and CUDA sanitizers;
then use `validate_model.py` to qualify full-model finite logits, generation,
reset consistency and quantization drift against the native BF16 ONNX model.
It runs both models in isolated child processes and retains independent NVML
samples plus raw final-prompt logits. Only then benchmark
native ONNX Runtime GenAI at 512 and 32768 input tokens using the existing
`export-onnx-lightning/genai/benchmark.py` harness, matched BF16 controls,
no MTP and repeated requests. Exclude the first prediction from decode rate.
Retain the >=70 GiB idle-start and >=2 GiB running-headroom/unique-owner guards.

No automatic GPU worker, watcher or queued benchmark is started by this code.
No ONNX prefill/decode token/s or GPU-memory value is available yet.

The earlier GGUF-only reference experiment is preserved separately in
[`REFERENCE_GGUF.md`](REFERENCE_GGUF.md) and [`RESULTS.json`](RESULTS.json).
