# Lightning BF16 CUDA custom kernels

Standalone source snapshot for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.
It adapts the coworker's Bolt kernels and device-dispatch strategies to
Lightning's dimensions and source BF16 arithmetic. No model weights, exported
ONNX models, runtime binaries or changes to the existing Bolt build are included.

## Contents

| File | Purpose |
| --- | --- |
| `lightning_ops.cu` | ORT registration, Mamba, normalization, routing and reference expert implementations |
| `chunked_prefill.cuh` | Source-faithful chunked prefill, cohort experts, two-slot recurrent state and two-row verification |
| `device_decode.cuh` | GPU-routed BF16 expert tensor-core kernels |
| `CMakeLists.txt` | Independent H100/sm90 shared-library build |
| `test_smoke.py` | Standalone GPU registration, dense projection and expert-path checks |

The original snapshot matched the locally qualified implementation. Subsequent
review fixes validate attention lengths before subtraction, return cohort
activations in token/route order and preserve verification shape fallbacks.
Current source hashes are in `SOURCE_SHA256SUMS`.
The shared registration source also retains legacy `W4Linear`/`MoeDecode`
entry points for compatibility. **The BF16 paths do not call those quantized
operators or quantize any weights.**

## Build

Requirements: Linux, H100, CUDA 12.6, CMake >=3.24, C++17, and the complete
C/C++ API headers from a compatible CUDA-enabled ONNX Runtime main build.
The qualified ORT revision was
`74bda515ee979679592b8b65814589b1465d6352`, with local Flash-arithmetic fixes;
an arbitrary stock ORT wheel is not claimed equivalent.

Run from the repository root:

```bash
cmake -S bf16_cuda_lightning -B build/bf16_cuda_lightning \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/include/onnxruntime/core/session
cmake --build build/bf16_cuda_lightning --parallel 8
```

Output: `build/bf16_cuda_lightning/lib/liblightning_ops.so`.
The build preserves `-O3 --fmad=false -lineinfo`, CUDA architecture 90 and
cuBLAS/cuBLASLt linkage. It does not link an ORT binary: `RegisterCustomOps`
initializes the ORT API table at registration.

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.register_custom_ops_library(
    "/absolute/path/to/build/bf16_cuda_lightning/lib/liblightning_ops.so"
)
session = ort.InferenceSession(
    "/path/to/compatible-lightning.onnx",
    sess_options=options,
    providers=["CUDAExecutionProvider"],
)
```

Custom nodes use domain `lightning`, version 1. This library is not a drop-in
replacement for domain `bolt`. Compatible exported nodes and cache layouts are
required; loading the library alone does not optimize an arbitrary ONNX graph.

## BF16 paths

- `RMSNorm`, `MambaStep`, `MoeRoute`, `MoeDecodeBF16`: source-aware decode
  normalization, recurrence, routing and diagnostic expert execution.
- `RowLinear`, `CachedAttentionChunk`, `MambaChunk`, `MoeChunkBF16`:
  logical one-row projections, ORT GQA invocation and source-faithful prefill.
- `MoeCohortBF16`: expert-weight reuse with the source one-row cuBLASLt
  algorithm, preserving BF16 rounding and split-K configuration.
- `MoeDeviceBF16`: GPU-resident single-token routed experts, retaining cohort
  prefill. The optimized shape is `H=2688,I=1856,topk=6,source_order=1`.
- `MambaWindow`, `MoeVerifyBF16`: two-slot recurrent states and source-faithful
  two-row expert verification for single-draft MTP.

Batch size is one. Verification rows are consecutive tokens, not independent
requests. Other supported dimensions/row counts retain the existing fallback
paths. Valid, unique expert IDs are required; device-path bounds guards do not
provide the diagnostic path's synchronous host-validation guarantee.
Multi-row expert activation output is `[T*topk,I]` in token/route order,
matching indices and the `[T,topk,H]` contribution output. Expert/cohort packing
is internal scratch, not the public activation layout.

FC1 retains two contiguous split-K partitions, BF16 projection/ReLU-squared
rounding and ascending MMA order; FC2 retains its source rounding. Finalization
preserves source FP32 contribution order. Exactness depends on the qualified
GPU/CUDA/cuBLAS configuration, not just the BF16 storage type.

MTP operators alone are not a speculative controller. The qualified native
integration additionally required aligned head prefill, two-row logits,
explicit-length rewind, shared weights and cache reconciliation. Those runtime
and exporter changes are intentionally outside this kernel-only snapshot.

## Validation and measured scope

Before publication the implementation passed 6,330 bounded independent HF
target comparison fields plus 1,284 two-row fields bitwise, 108 actual-model
greedy MTP comparisons and 45 focused regression tests. CUDA memory/race checks
reported zero errors/hazards. These checks do **not** prove universal numerical
identity or independent full-32K equivalence against PyTorch.

Native end-to-end measurements used one H100, original BF16, batch one and
ten requests per variant. They are not isolated-kernel performance promises:

| Input tokens | Configuration | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- |
| 32768 | Optimized target-only | 437.224 | 134.462 |
| 32768 | Native MTP | 401.527 | 131.673 |
| 512 | Optimized target-only | 573.378 | 155.929 |
| 512 | Native MTP | 547.805 | 169.915 |

MTP improves short-context decode by 8.97%, but regresses 32K decode by 2.07%
and adds prompt latency. It is not enabled by this library and is not the
recommended long-context configuration. No 400 tok/s claim is made.

For standalone smoke tests, use PyTorch, ONNX and the compatible CUDA ORT build
with `OrtValue.from_dlpack`/DLPack output support:

```bash
CUDA_VISIBLE_DEVICES=0 \
LIGHTNING_OPS="$PWD/build/bf16_cuda_lightning/lib/liblightning_ops.so" \
python -m unittest discover -s bf16_cuda_lightning -p test_smoke.py -v
```

These smoke tests fail, rather than silently skip, when CUDA or the library is
missing. They do not download weights or replace the full numerical gates.
They include unequal-size shared-expert cohorts, all three expert outputs,
non-specialized two-row fallback, and invalid/edge attention lengths.
Historical full-model measurements above precede these review fixes; the
review fixes have focused GPU regression coverage, not new performance claims.
