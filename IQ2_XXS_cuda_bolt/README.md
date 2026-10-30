# Native Bolt CUDA kernels with scale fixes

This is the **rounding-only native implementation** measured on H100, not the
slower diagnostic implementation. The two CUDA source snapshots are byte-for-byte
copies of the deployed sources. The original implementation is
[`rui-ren/nemotron_ops@e4f63ffd`](https://github.com/rui-ren/nemotron_ops/tree/e4f63ffdcb830e2faf5ddb149047953d922a6afb).
The repository's existing `src/bolt_ops.cu` matches that baseline.

## What changed

* **IQ2_XXS:** reconstruct `BF16(grid * FP32_scale)`, not
  `BF16(grid * BF16(scale))`, in `iq2-compat/bolt_ops.cu::DequantRow`.
  This applies to native `iq2=2` shared and routed experts.
* **Q8_0:** consume the checkpoint's genuine **FP16** scales, without first
  converting them to BF16. `compat/q8_rank_compat.cu` preserves the input's
  leading dimensions, flattens them to 1..8 rows, and uses the original MMA.

Mamba, routing, attention, NVFP4, expert grouping, split-K, MMA operand order and
expert finalization remain native. No source operators, canonical/sequential
expert accumulation, Mamba cast corrections, observers, or debug fallbacks are
activated. The separate source copies and historical `bolt_iq2_source` name are
intentional: they preserve the exact deployed library boundaries. Despite that
name, this library is the original native kernel with the IQ2 scale correction.
The plugin wrapper additionally rejects native IQ2 intermediate sizes not
divisible by 256, before any routed kernel launches. The kernel snapshots
themselves remain unchanged.

The original top-level exporter and kernels are **not modified** by this folder.
In particular, its existing BF16 Q8 scale export is not compatible with the new
FP16 Q8 input. Casting those already-rounded values back to FP16 does **not**
recover source scales.

## Build

Requires Linux, CUDA with Hopper `sm_90` support, CMake >=3.22 and the qualified
ORT headers. The measured build used CUDA 12.6, GCC 11.4 and ORT 1.31 headers.

```bash
cmake -S IQ2_XXS_cuda_bolt -B IQ2_XXS_cuda_bolt/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DORT_INCLUDE_DIR=/path/to/ort/include/onnxruntime/core/session \
  -DORT_LIBRARY=/path/to/qualified/runtime/libonnxruntime.so.1
cmake --build IQ2_XXS_cuda_bolt/build -j2
CUDA_VISIBLE_DEVICES=0 ctest --test-dir IQ2_XXS_cuda_bolt/build --output-on-failure
python -m unittest discover -s IQ2_XXS_cuda_bolt/tests -p native_runtime_test.py
CUDA_VISIBLE_DEVICES=0 BOLT_PLUGIN=/path/to/qualified/runtime/libonnxruntime_providers_cuda.so \
  python IQ2_XXS_cuda_bolt/tests/operators_test.py -v
```

Outputs in `build/lib/`:

| Library | Domain / operators |
|---|---|
| `libbolt_upstream.so` | Original `bolt` operators, unchanged |
| `libbolt_q8_fp16.so` | `bolt_fast_q8::Q8GemvFp16Scales` |
| `libbolt_iq2_source.so` | `bolt_iq2_source::{Q4Gemv,MoeVerify}`, `iq2=2` |

The CUDA regressions exercise the actual IQ2 dequantization and Q8 FP16 scale
loader across scales, signs, codebook entries and quantized integers. Each test
requires exact reconstructed BF16 values and actual counterexamples to early
scale rounding. These are component tests, not full-model acceptance.
The Q8 test initializes its read-only-cache inputs in a separate CUDA launch.

The ORT integration tests require Torch, NumPy, `onnx_ir` and `gguf`. They
exercise the actual new domains with single-/multi-row inputs, Q8 `[2,3,K]`
shape preservation, independent GGUF IQ2 dequantization, routed hidden/part/final
outputs, and invalid IQ2 geometry rejection. CPU fallback is disabled; CUDA
absence fails rather than skips. `libort_cuda_runner.so` links directly to the
qualified ORT library, avoiding an incompatible installed ORT Python wheel.
`ORT_LIBRARY` is optional when not building these integration tests.

## Load the measured graph

`model.onnx` is the lightweight, **weight-free** graph from the measured native
package: 1,312 nodes / 1,196 initializers, unchanged original I/O. It references
external data; **this folder is not a standalone downloadable model**.

Use the already prepared **HF-derived native-layout asset package**, preserving
its tokenizer, original drafter, configuration and these exact data files:

| File | Purpose |
|---|---|
| `model.onnx.data` | HF-derived native-layout target parameters |
| `native-iq2-memory.bin` | Native IQ2 records and original FP16 superblock-scale bits |
| `q8-source-scales.fp16.bin` | Original Q8 source scales, genuine FP16 |
| `dflash2.onnx`, `dflash2.onnx.data` | Original speculative drafter |
| `libbolt_drafter_original.so` | Unchanged drafter operators |

Hashes of graph, source snapshots, data assets and measured runtime are in
`provenance.json`. The HF checkpoint revision is
`c6cc2e8ffa353403213d6b9b18aaf8eb684d705a` of
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`.
Do not substitute a GGUF-derived asset package or an arbitrary re-export: different
weight realizations/scale layouts are not interchangeable. This PR does not
reimplement the complete HF weight conversion pipeline or distribute weights.

The existing package configuration must retain its qualified target/drafter
settings. Copy the three newly built DSOs into that package, then:

```python
import sys

sys.path.insert(0, "IQ2_XXS_cuda_bolt")
from native_runtime import create_model

model = create_model(
    "/path/to/hf-derived-native-assets",
    cuda_provider_library="/path/to/qualified/runtime/libonnxruntime_providers_cuda.so",
)
```

The optional provider argument registers the qualified CUDA provider explicitly.
Repeated calls reuse that registration; conflicting provider paths fail rather
than silently taking over. An application-owned existing registration is accepted
only when Linux process mappings identify exactly the requested provider.
If its path cannot be verified, omit the argument to explicitly use the
application's registration. Do not unregister it while using this loader.
The loader uses the supported `Config.add_model_data` graph overlay, retaining
the graph bytes for the model lifetime and resolving external data inside the
asset root. It sets the decoder filename to `model.onnx` so a differently named
decoder in the existing configuration cannot bypass the overlay.
Do not call `Model("IQ2_XXS_cuda_bolt")` or use escaping external-data
symlinks. No runtime security checks are bypassed.

**Runtime is not bundled.** The benchmark used a custom ORT/GenAI build supporting
`nemotron_bolt`, DFlash2, multiple custom-op libraries, and speculative options.
An arbitrary stock `onnxruntime-genai` wheel is not a supported replacement.
Runtime binary hashes are supplied for identification, not a claim that the
kernel sources alone reproduce the entire runtime build.

## Recorded results and limits

Public `onnxruntime-genai` `Model` / `Generator` API, H100, one fixed 71-token
prompt, 512 generated tokens, two warmups plus three measured runs per process:

| Mode | Weighted decode tok/s | Warm TTFT |
|---|---:|---:|
| Speculative k=7, primary process | 434.74 | approximately 85 ms |
| Speculative k=7, independent repeat | **434.12** | approximately 85 ms |
| Ordinary greedy, separate five-run check | **122.23** | **81.99 ms** |

Decode excludes the first generated token: `511 / decode_seconds`.
TTFT starts before `append_tokens`; it is not isolated GPU prefill throughput.
All ten speculative sequences were identical; the ordinary/speculative check
also matched all 512 tokens for this prompt. Guards passed; measured peak device
memory was 28.27 GiB. CUDA-graph configuration stayed enabled, but actual capture
was **not independently verified**.

`benchmark.json` retains the exact prompt/output IDs, all measured speculative
rates, sequence digest and runtime identification. These are recorded
machine-local results, not a portable performance guarantee or a matched speedup
against the older diagnostic implementation.
The review fixes add wrapper validation and loader/tests only; these historical
rates were not remeasured on the rebuilt plugin.

**Accuracy is not qualified for this variant.** The earlier corrected diagnostic
implementation scored 168/200 (84%) on MMLU-Pro; that score must not be attributed
to these kernels. MMLU-Pro has not been rerun here. Full HF/diagnostic numerical
parity is unresolved; this native sequence differs from the earlier diagnostic
sequence after 18 matching tokens. Native ordinary/speculative agreement is not
proof of HF equivalence.

## License

MIT, under the repository's `LICENSE`. The copied IQ2 codebook also retains its
ggml/llama.cpp attribution in each header.
