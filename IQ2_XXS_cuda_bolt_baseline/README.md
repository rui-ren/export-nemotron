# Historical Bolt diagnostic baseline: 83.59 tok/s

This archives the earlier **approximately 84 tok/s implementation**, separately
from [`../IQ2_XXS_cuda_bolt`](../IQ2_XXS_cuda_bolt). It preserves the historical
graph, configuration, operator source dependency closure, timing helper and
benchmark records. It is intentionally not cleaned up into the faster native
implementation.

The measured candidate was `native-r8-v1/integration-v3`: diagnostic source
operators plus the C8 NVFP4 optimization. Its exact source snapshots are in
`sources/`; original paths and SHA-256 hashes are in `provenance.json`.
`build.py` verifies those bytes and rewrites **only absolute local includes**
in an ignored staging tree. It does not change arithmetic.

## What the historical result means

H100, exclusive GPU 5 on a shared host, public ORT GenAI API, fixed 71-token
prompt, 512 generated tokens, DFlash2 k=7, two warmups and three measured runs:

| Historical matched arm | Median decode tok/s |
|---|---:|
| Preceding diagnostic pipeline | 75.3839 |
| This C8 diagnostic pipeline | **83.5923** |

Both arms produced the same complete 512-token sequence in all five runs.
Decode is `511 / decode_seconds`, excluding the first generated token.
TTFT starts before `append_tokens` and includes first-token materialization;
it is not isolated GPU prefill throughput. `benchmark.json` retains the exact
prompt/output IDs, per-run timings and speculative statistics.

The recorded full diagnostic qualification covered four frozen cases and
3,136 raw-bitwise observations. This is **diagnostic-reference parity**, not
independently executed native HuggingFace full-model equivalence.
The earlier diagnostic lineage scored 168/200 (84%) on MMLU-Pro; that
evaluation was **not rerun on this exact C8 build**.

**83.59 versus approximately 434 tok/s is not a controlled kernel-only A/B.**
The later pipeline changes the graph, capture settings, operator algorithms,
generated sequence and speculative work/acceptance. This archive is useful
for studying the whole implementation, not attributing the entire difference
to one bad kernel.

## Where the overhead lives

* Source-faithful fallbacks reconstruct weights, use padded source-logical
  GEMM geometry and allocate substantial scratch space.
* Attention and router operators retain ATen/Torch dependencies and source
  execution contracts. Inspect `sources/attention` and `sources/router`.
* Mamba retains source SSD-prefill profiles, explicit rounding boundaries and
  embedded generated kernel images, plus the historical coalesced cached path.
* The target runs with CUDA graph capture **off**, TF32 off; the drafter's
  configuration enables capture. Host waits and diagnostic shape/history
  contracts remain present.

This was already an optimized diagnostic baseline, not an all-naive kernel
implementation. Do not remove its graph-visible history/position/normalization
contracts and continue calling it the historical baseline.

## Rebuild the ten archived operator libraries

Requires Linux, Python 3.11+ (the measured runtime uses **3.12**), Hopper H100,
CUDA 12.6, ORT 1.31 headers, GCC 11, and the qualified Torch
2.14.1+cu130 installation. The source fallback binds specifically to
`libcublas.so.13`; CUDA 12 compiler headers and CUDA 13 runtime libraries were
used together in the recorded environment.

```bash
python IQ2_XXS_cuda_bolt_baseline/build.py --prepare-only
python IQ2_XXS_cuda_bolt_baseline/build.py \
  --cuda-home /usr/local/cuda-12.6 \
  --ort-include /path/to/ort-1.31-headers \
  --torch /path/to/site-packages/torch \
  --cublas /path/to/site-packages/nvidia/cu13/lib/libcublas.so.13 \
  --cudart /path/to/site-packages/nvidia/cu13/lib/libcudart.so.13 \
  --cxx /usr/bin/g++-11
python -m unittest discover -s IQ2_XXS_cuda_bolt_baseline -p '*_test.py'
```

Outputs are in `build/lib/`; executed commands and rebuilt hashes go to
`build/receipt.json`. Historical binary hashes are identification evidence,
not a promise of reproducible binary identity after include/build relocation.
Rebuilt libraries require fresh numerical/performance qualification.
The build is sequential and does not run GPU inference.

`build-plan.json` preserves the NVFP4 object symbol renames and private operator
registrations. The ten targets are NVFP4, Q8, wide Q8, shared IQ2, routed IQ2,
Mamba, attention, router, RMS and expert summation.

## Assets and runtime are not bundled

This folder is **not a standalone model**. It requires the already prepared
HF-derived diagnostic-layout assets:

| File | Recorded size |
|---|---:|
| `model.onnx.data` | 43,085,691,904 bytes |
| `native-iq2-memory.bin` | 17,506,959,360 bytes |
| `q8-source-scales.fp16.bin` | 72,351,744 bytes |
| `dflash2.onnx`, `dflash2.onnx.data` | Original drafter graph and parameters |
| `libbolt_ops.so`, `libbolt_drafter_original.so` | Unchanged original operators |

Their hashes are in `provenance.json`. The source model is
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` at revision
`c6cc2e8ffa353403213d6b9b18aaf8eb684d705a`. This PR does not include
the HF conversion pipeline, large weights, built libraries or runtime sources.
Do not substitute arbitrary GGUF-derived assets.

Use the qualified custom ORT/GenAI build supporting `nemotron_bolt`, DFlash2,
multiple custom-op libraries and these history contracts. Stock wheels are
not interchangeable. Runtime hashes are recorded. The Mamba source fallbacks
also require substantial device/host memory; this is **not** a portable 24GB
deployment despite the checkpoint name.

For a fresh benchmark, prepare a **separate complete asset package**, install
these `model.onnx` and `genai_config.json` files there, and add the ten rebuilt
libraries alongside the two unchanged libraries. Keep external data inside the
package's real asset root; escaping external-data symlinks are not supported.
Do not overwrite the fast package or the preserved historical package.

Activate the qualified runtime environment, expose its CUDA/Torch libraries
and preload the **same cuBLAS path passed at build time** if needed to avoid
loading a different cuBLAS first. Then:

```bash
CUDA_VISIBLE_DEVICES=0 BOLT_MAMBA_CUMSUM_HEAD_TILE=1 \
  python IQ2_XXS_cuda_bolt_baseline/benchmark.py \
    --model /path/to/separate-complete-diagnostic-package \
    --runtime-dir /path/to/qualified/runtime/lib \
    --allow-rebuilt-libraries \
    --out /path/to/new-baseline-results.json
```

The replay checks graph, configuration, weights, unchanged libraries and runtime
hashes before timing. `--allow-rebuilt-libraries` permits only the ten rebuilt
operator hashes to differ, records them, and does not waive output checks.
Omit it when using the exact historical binaries. All five runs must emit
the recorded 512 IDs; partial/mismatched results are preserved instead of
silently accepted.

`BOLT_MAMBA_CUMSUM_HEAD_TILE=1` is the frozen source-autotuner profile for
**this benchmark prompt**, not a universal setting for arbitrary prompt lengths.
Historical generation and profile metadata remain unmodified. Some intermediate
metadata labels say "unqualified"; the later archived receipt hashes describe
the qualification actually performed, without rewriting older evidence.

## Archive validation

The relocated sources built all ten shared libraries with CUDA 12.6, GCC 11
and ORT 1.31 headers. All ten loaded in the qualified environment and exported
`RegisterCustomOps`. Three wrapper regressions and Ruff lint/format checks
passed. The 1,507-node / 1,200-initializer graph's external-data bounds and
all ten recorded 512-token sequences were checked.

These are source/archive/build checks. **Full-model inference, throughput and
accuracy were not rerun with the rebuilt binaries for this publication.**

## Attribution

Original Bolt kernels: `rui-ren/nemotron_ops`, MIT; see the repository license
and retained source headers. IQ2 tables retain ggml attribution. The embedded
source-derived Mamba/Triton kernels originate from `state-spaces/mamba`,
Apache-2.0; see `licenses/mamba-LICENSE`.
