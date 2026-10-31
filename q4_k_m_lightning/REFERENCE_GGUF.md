# Historical offline GGUF reference experiment

This records the earlier reference phase, not the requested ONNX endpoint.
See [README.md](README.md) for the actual ONNX package and current qualification
status. The memory and drift numbers below are **not ONNX measurements**.

This exports `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` through the
**pinned upstream llama.cpp Q4_K_M preset**, with MTP excluded. It is a GGUF
reference artifact, **not an ONNX graph or an optimized native GenAI package**.
The existing BF16 exports, CUDA libraries and default runtime are unchanged.

Do not pass these packed weights to `W4Linear` or `MoeDecode`. Their symmetric
INT4 RTN packing is incompatible with GGML Q4_K/Q5_0/Q8_0 blocks. ONNX deployment
needs separate compatible operators, tensor-layout conversion and numerical
qualification; this folder does not claim to implement that runtime.

## Important Lightning shape limitation

Q4_K uses blocks of 256 weights. Lightning's expert input widths **2688 and
1856 are not divisible by 256**. The stock quantizer therefore falls back from
Q4_K to Q5_0 and from Q6_K to Q8_0 where required. We do not silently pad the
matrices, replace the preset, or label uniform RTN as Q4_K_M.

The measured full-target artifact is **24,515,129,024 bytes = 24.515 GB =
22.831 GiB**, rather than the earlier estimated 20-22 GB:

| Actual tensor type | Tensors | Weight bytes, decimal GB |
| --- | --- | --- |
| Q5_0 | 110 | 16.336320 |
| Q8_0 | 25 | 7.955856 |
| Q4_K | 29 | 0.179601 |
| F32 | 237 | 0.035456 |

134 of 401 tensors need shape fallbacks. The file's Q4_K_M label describes the
upstream preset, not a guarantee that every matrix is four-bit. Actual weights
average about **6.21 bits/parameter**, including the retained precision mix.
The embedding is Q5_0 and output head is Q8_0 under the stock preset; this
is not the earlier proposed expert-only policy with BF16 embedding/head.

This first experiment uses no importance matrix or calibration data. An
importance-weighted run is a distinct follow-up, not implied by this result.

## Reproduce

Use the existing PyTorch/safetensors/NumPy/Transformers environment and a CUDA
toolchain; do not replace the qualified ORT installation.
The converter also needs the pinned upstream conversion requirements; the
optional GPU validation script uses `nvidia-ml-py` (`import pynvml`).
The reference source revision is
`5047748f114564fcf271e6a648f60e2d2eae6264`.

```bash
git clone https://github.com/ggml-org/llama.cpp.git /path/to/llama-q4km
git -C /path/to/llama-q4km checkout 5047748f114564fcf271e6a648f60e2d2eae6264
cmake -S /path/to/llama-q4km -B /path/to/build-llama-q4km \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build /path/to/build-llama-q4km --target llama-quantize --parallel 8

python q4_k_m_lightning/export.py --model-dir /models/lightning-bf16 \
  --llama-source /path/to/llama-q4km \
  --quantizer /path/to/build-llama-q4km/bin/llama-quantize \
  --out /models/lightning-q4km
```

Destinations must be new or empty; completed audit evidence is never overwritten.
The script requires the clean pinned source. Quantize directly from original
BF16, never from the earlier RTN/another lossy export.
The retained BF16 GGUF needs approximately 63.18 GB of disk in addition to the
quantized file. This is staging/reference storage, not a claim about GPU memory.

`quantization.json` records SHA256 hashes, actual type/byte counts, per-tensor
weight errors and complete finite-value/shape checks. Weight error is not an
accuracy score. `--audit-only` can audit existing files with the exact output
names `target-bf16.gguf` and `target-Q4_K_M.gguf`.

Audit every target tensor against the original checkpoint:

```bash
python q4_k_m_lightning/audit_source.py --model-dir /models/lightning-bf16 \
  --gguf /models/lightning-q4km/target-bf16.gguf \
  --llama-source /path/to/llama-q4km --out source-audit.json
```

This checks all 6,243 source target tensors against 401 GGUF tensors,
including expert stacking, FP32 negative exponential and the converter's
per-head Q/K row permutation. It does not confuse representation changes with
unchanged raw bytes. The upstream converter declares a 1,048,576-token context;
the original checkpoint declares **262,144**. Do not infer extra context support
from the GGUF metadata. The bounded probe below uses only a 2048-token context.

## Bounded inference and quality checks

Build the small probe against the pinned reference runtime:

```bash
c++ -O2 -std=c++17 \
  -I /path/to/llama-q4km/include -I /path/to/llama-q4km/ggml/include \
  q4_k_m_lightning/probe.cpp -L /path/to/build-llama-q4km/bin \
  -Wl,-rpath,/path/to/build-llama-q4km/bin \
  -lllama -lggml -lggml-base -o /path/to/lightning-q4km-probe

CUDA_VISIBLE_DEVICES=0 python q4_k_m_lightning/validate.py \
  --baseline /models/lightning-q4km/target-bf16.gguf \
  --model /models/lightning-q4km/target-Q4_K_M.gguf \
  --probe /path/to/lightning-q4km-probe \
  --reference /path/to/independent-hf-capture.npz \
  --tokenizer-dir /models/lightning-bf16 --gpu-index 0 --out q4km-validation
```

The probe compares identical token IDs, captures every prompt-position logit
and generates up to 32 greedy tokens, with normal EOS. It reloads BF16 and
quantized models sequentially and clears recurrent/KV state between prompts.
The monitor requires an initially idle GPU with >=70 GiB free and retains
all >=2 GiB headroom/unique-owner samples. Output directories must be new.

The captured five histories are **four unique prompts plus one repeat**,
129 total prompt positions. Same-engine BF16/quantized top-1 agreement ranges
from 75.0% to 91.7%; examples generate coherent recursion explanations and
prime-number lists. The repeat verifies state-reset consistency.
That is a smoke/drift check, **not representative accuracy acceptance**.
The reference runtime's BF16 arithmetic itself differs from native HF, so the
report separates HF-to-reference-runtime drift from quantization drift.
Neither exact BF16 parity nor a throughput improvement is claimed.

Reference runtime peak GPU allocation was **59.184 GiB for BF16** and
**23.604 GiB for Q4_K_M** in the bounded probe. These are not ONNX GenAI
memory/performance measurements and do not include a 32K KV workload.
Keep BF16 as the deployment baseline until representative reasoning/code and
long-context quality checks and compatible ONNX kernels are qualified.

Machine-local outputs:

- `artifacts/q4km-lightning/`: BF16 staging GGUF, Q4_K_M GGUF and weight audit.
- `evidence/q4km-source-audit.json`: full source-conversion checks.
- `evidence/q4km-native-validation-namespace/`: final raw logits, generated IDs/text
  and resource samples. Earlier captures, including the rejected
  container-PID/host-PID comparison, remain separate.

[`RESULTS.json`](RESULTS.json) records the artifact hashes, tensor mix and
bounded comparison results without bundling model weights or raw prompt data.
All 401 tensors passed finite-value/shape checks, all 6,243 source tensors
passed conversion checks, and all 237 retained tensors passed a separate
byte-for-byte check. Six focused audit tests pass. Both model runs pass exact
repeat-logit/generation checks and all 215 final GPU guard samples pass.

Focused audit tests need no CUDA/checkpoint:

```bash
PYTHONPATH=/path/to/llama-q4km/gguf-py \
python -m unittest discover -s q4_k_m_lightning -p test_export.py -v
```
