# Lightning: four export versions

Source model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.
Input is the complete, locally downloaded **safetensors checkpoint**, not GGUF.

| Folder | What you learn | Graph |
| --- | --- | --- |
| [01-standard](01-standard/README.md) | Explicit BF16 math and cache updates | Standard ONNX, with optional ORT GQA |
| [02-custom](02-custom/README.md) | Coworker-style fusion and CUDA contracts | BF16 `MambaStep`, `MoeRoute`, `MoeDecodeBF16`, ORT GQA |
| [03-int4](03-int4/README.md) | BF16-to-W4A16 round-to-nearest quantization | `MambaStep`, `MoeRoute`, `MoeDecode`, ORT `MatMulNBits` and GQA |
| [04-mtp](04-mtp/README.md) | The separate attention+MoE draft stage | BF16, explicit or custom MoE |

Each folder has its own export command. Shared checkpoint, graph, layer and
runtime code lives in [shared](shared), rather than four diverging copies.
The original Bolt exporter and library are not modified.

## What is and is not implemented

- Each export writes `model.onnx` (recipe), `model.onnx.data` (ingredients)
  and `manifest.json` (state and deployment contract).
- All versions are **batch 1, one-token decode**. Replay prompt tokens in order
  to populate caches. This is not an optimized parallel-prefill exporter.
- Each graph runs on **one GPU**. Having eight H100s does not automatically
  shard these graphs. BF16 fit on an 80 GB device must be measured, including
  ORT weight copies, workspaces, KV and recurrent caches.
- The runner uses ORT Python directly, not an ORT GenAI model configuration.
- Custom operations are in domain `lightning`, not `bolt` or `com.microsoft`.
  Their shared library is required by both the custom and INT4 versions.
- Multi-row `MambaVerify`/`MoeVerify`, speculative acceptance/rejection and
  cache rollback are **not implemented**. An MTP draft graph is not a
  complete speculative-decoding engine.
- No production performance or real-model accuracy result is claimed.
  CUDA compilation, H100 numerical tests and full checkpoint inference
  must be performed on the target machine. Dependency-free CLI checks do
  not establish model correctness.

## H100 prerequisites

Use your existing Linux/H100 environment. No environment-creation step is needed.
It must provide CUDA-enabled PyTorch, NumPy, ONNX >= 1.18, safetensors, and an
ORT main CUDA build supporting BF16 GQA, BF16 `MatMulNBits`, and Python DLPack
IO binding. See [requirements.txt](requirements.txt) for exporter dependencies.
Do not install a CPU-only ORT wheel over your GPU/main build.

Run the following commands from this directory. Paths below are Linux examples:

```bash
MODEL=/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

python 01-standard/export.py --model-dir "$MODEL" --out /models/onnx-standard
python 02-custom/export.py   --model-dir "$MODEL" --out /models/onnx-custom
python 03-int4/export.py     --model-dir "$MODEL" --out /models/onnx-int4
python 04-mtp/export.py      --model-dir "$MODEL" --out /models/onnx-mtp --mtp-backend custom
```

Destinations must be new or empty. Start with `--max-length 128` for short
validation runs, then increase deliberately. Default capacity is 2048.
Export uses CPU checkpoint streaming; inference and kernel validation use H100.
Never use debug `--num-layers` graphs as complete model baselines.

## Build and test the custom library

```bash
cmake -S ../src/lightning -B /tmp/lightning-build \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/include/onnxruntime/core/session \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
cmake --build /tmp/lightning-build -j
export LIGHTNING_OPS=/tmp/lightning-build/lib/liblightning_ops.so

# These diagnostic tests may run in your H100 environment, without creating
# a separate CPU environment. They require the existing ORT Python package.
python -m unittest discover -s tests -p test_export.py -v
python -m unittest discover -s tests -p test_gpu.py -v
python -m unittest discover -s tests -p test_gpu_ops.py -v
```

The GPU tests skip without CUDA and `LIGHTNING_OPS`; a skipped test is not a pass.
For an ORT CUDA plugin build, additionally set `CUDA_PROVIDER_LIBRARY` for
tests, or pass `--cuda-provider-library` to the runner.

The tiny checkpoint tests cover multi-step logits, hidden states and all
cache outputs. The operator tests also exercise actual Lightning dimensions.
Neither substitutes for a real-checkpoint comparison.

## Compare the original model and the exported models

Run each process separately so two complete models do not occupy GPU memory
at the same time. Use exactly the same teacher-forced tokens:

```bash
TOKENS=1,42,100,200,300,400

# Requires a compatible Transformers build. This tool rejects missing weights
# instead of accepting randomly initialized layers.
python tools/capture_hf.py --model-dir "$MODEL" --token-ids "$TOKENS" --record hf.npz

python run.py --export /models/onnx-standard --token-ids "$TOKENS" --record standard.npz
python run.py --export /models/onnx-custom --custom-library "$LIGHTNING_OPS" \
  --token-ids "$TOKENS" --record custom.npz

python tools/compare.py --reference hf.npz --candidate standard.npz
python tools/compare.py --reference standard.npz --candidate custom.npz

python run.py --export /models/onnx-int4 --custom-library "$LIGHTNING_OPS" \
  --token-ids "$TOKENS" --record int4.npz
python tools/compare.py --reference custom.npz --candidate int4.npz --metrics-only
```

Comparison reports per-tensor maximum absolute and relative L2 errors, logits
top-1 agreement and next-token NLL on the supplied sequence. Default
`--atol 0.1 --rtol 0.05` are starting thresholds, not established accuracy
guarantees. Non-finite values, mismatched fields/shapes/tokens, or exceeded
thresholds fail explicitly. `--metrics-only` reports W4 changes without
claiming equivalence; evaluate representative prompts/corpora before accepting
quantized quality. Greedy generation alone is not a correctness test.

For timing without record-transfer overhead:

```bash
python run.py --export /models/onnx-custom --custom-library "$LIGHTNING_OPS" \
  --token-ids "$TOKENS" --max-new-tokens 32
```

This measures sequential replay/decode, not optimized serving throughput.

## Numerical and cache contracts

- Attention does not apply RoPE in the inspected Nemotron-H implementations;
  GQA is emitted with `do_rotary=0`.
- Recurrent SSM state is FP32 `[batch, heads, state_size, head_dim]`, following
  Bolt's layout. Convolution history is BF16 `[batch, channels, kernel-1]`.
- Router logits, sigmoid and probability normalization accumulate in FP32.
  Correction bias affects selection only; probabilities use unbiased scores.
- Mamba's default `--dt-min 0` selects unclamped recurrent updates.
  Upstream fast/chunked prefill and fallback recurrent implementations can
  differ in clamping and convolution rounding. The reference capture sets the
  same minimum explicitly, but that does not make different upstream paths
  bit-identical. Investigate discrepancies rather than relaxing tolerances
  until tests pass.
- Manifest format 2 records the new state layout and
  `bolt_style_lightning_v1` custom contract. Re-export older draft artifacts;
  their renamed operations and state layouts are not interchangeable.

Implementation references are the local original [Bolt exporter](../export_bolt.py),
[Bolt CUDA source](../src/bolt/bolt_ops.cu), upstream
[Transformers Nemotron-H](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py)
and [vLLM MTP](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nemotron_h_mtp.py).
These upstream links track main; record the actual installed framework/ORT
commit revisions with benchmark results. Input config and index hashes are
recorded in each manifest; they do not hash every weight shard.
