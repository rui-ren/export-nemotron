# Lightning single-token CUDA custom ops

This is a local, source-derived adaptation of the coworker's [Bolt source](../bolt/bolt_ops.cu),
not an implementation of Bolt's multi-row `MambaVerify`/`MoeVerify` or tensor-core
verification paths. All operators use the `lightning` domain and
`CUDAExecutionProvider`, with batch **1** and one token per invocation. No
performance equivalence is claimed.

## Operation contracts

`BF16`, `FLOAT`, `INT32`, and `UINT8` below mean ONNX tensor element types.
All tensors are contiguous CUDA tensors. Parameters and tensors are required;
there are no optional inputs or outputs.

### `MambaStep`

Attributes: positive integers `heads`, `head_dim`, `d_state`, `n_groups`;
finite positive `epsilon`; finite nonnegative `time_step_min`.
`heads` must be divisible by `n_groups`. Define
`D = heads * head_dim`, `C = D + 2 * n_groups * d_state`.
The convolution width `L >= 2` is inferred from `conv_w`.

| Input (in order) | Type | Shape |
|---|---|---|
| proj: gate, xBC, dt concatenation | BF16 | `[1,D+C+heads]` |
| conv_state | BF16 | `[1,C,L-1]` |
| ssm_state | FLOAT | `[1,heads,d_state,head_dim]` |
| conv_w | BF16 | `[C,1,L]` |
| conv_b | BF16 | `[C]` |
| dt_bias, A_neg, D_skip | FLOAT | each `[heads]` |
| norm_w | FLOAT | `[D]` |

Outputs, in order: `y` BF16 `[1,D]`, `conv_state_out` BF16 `[1,C,L-1]`,
`ssm_state_out` FLOAT `[1,heads,d_state,head_dim]`, `xbc_act` BF16 `[1,C]`,
`ypre` FLOAT `[1,D]`. Outputs do not require aliasing inputs.

The channel-owned convolution (including the kernel-4 `uint2` load), the
`head_dim=64,d_state=128` two-part state update, FP32 recurrent-state layout,
softplus threshold, and warp-reduced grouped gated RMS normalization derive
from Bolt's `MambaConvKernel`, `MambaSsmKernel`, and `MambaNormKernel`.
Other positive state/head dimensions and convolution widths have scalar-safe
fallbacks with the same recurrence; the state reduction order may differ.

**Intentional numerical difference from actual Bolt code:** Bolt computes
`bf16(v * rsqrt(mean(v*v)+epsilon) * norm_w)` with FP32 `v`, inverse RMS, and
norm weights, rounding only the final result. Lightning instead computes
`bf16(float(bf16(v * rsqrt(mean(v*v)+epsilon))) * norm_w)`:
the normalized gated value is rounded to BF16 **before** the FP32 norm-weight
multiplication. The FLOAT norm-weight tensor is expected to contain
BF16-representable values, as produced by the Lightning reference; the kernel
does not silently round arbitrary FLOAT weights. `ypre` remains FP32, not a
BF16 scratch tensor. Gate application and RMS accumulation remain FP32.

### `MoeRoute`

Inputs: `x` BF16 `[1,H]`, `gate` BF16 `[E,H]`, correction `bias` FLOAT `[E]`.
Attributes: integer `topk=K`, finite nonnegative FLOAT `scale`;
`1 <= K <= min(E,32)`, `1 <= E <= 512`, `H > 0`.
Outputs: indices INT32 `[1,K]`, probabilities FLOAT `[1,K]`, and uncorrected
sigmoid scores FLOAT `[1,E]` (not logits).

Bolt's one-warp-per-expert logits and iterative warp top-k are retained.
Selection uses `sigmoid(x @ gate.T) + bias`; returned probabilities use the
uncorrected selected sigmoid scores, divided by their sum plus `1e-20`, then
multiplied by `scale`. Equal corrected scores choose the smallest expert ID.
NaN corrected selection scores are treated as negative infinity; NaN source
scores can still propagate to probabilities. Finite model inputs are expected.
Vector loads are guarded by extent and alignment, with scalar tails, so
`H=2688`, `H=128`, and unaligned/non-multiple-of-eight fixture rows are safe.

### `MoeDecodeBF16` and `MoeDecode`

Both require integer attribute `topk=K`, with `1 <= K <= min(E,32)`.
Common inputs: `x` BF16 `[1,H]`, selected `idx` INT32 `[1,K]`, and
`probs` FLOAT `[1,K]`.

`MoeDecodeBF16` then takes `up` BF16 `[E,I,H]` and `down` BF16 `[E,H,I]`.

`MoeDecode` additionally requires positive **even** integer `block_size=B`
and takes, after the common inputs:

1. `q_up` UINT8 `[E,I,ceil(H/B),B/2]`
2. `scale_up` BF16 `[E,I,ceil(H/B)]`
3. `q_down` UINT8 `[E,H,ceil(I/B),B/2]`
4. `scale_down` BF16 `[E,H,ceil(I/B)]`

Packed weights use unsigned low-nibble-first values with zero point 8.
Padded final-block elements are ignored. Each value is dequantized as
`bf16((nibble-8) * float(scale))` **before** FP32 dot-product multiplication.
Unlike Bolt's original `DotInt4x32`, scale is not factored out of an
integer-valued dot product: that would omit Lightning's BF16 weight rounding.

Both return three outputs: `y` BF16 `[1,H]`, `h` BF16 `[K,I]`, and `part`
FLOAT `[K,H]`. For each selected expert:

```
h = bf16(relu(float(bf16(x @ up.T))) ** 2)
part = float(bf16(h @ down.T))
y = bf16(sum_k(probs[k] * part[k]))
```

`part` is FP32 storage of BF16-rounded down-projection values, matching Bolt's
decode rounding boundary. Bolt's pair/row grid, eight warps per block,
`WarpSum`, packed 32-value W4 chunks, ReLU-squared rounding, and float4
finalization are adapted rather than using a generic matrix library.
The BF16 sibling uses guarded eight-value vector dots. Alignment guards and
scalar tails cover partial chunks and padded rows. `H=2688,I=1856` requires no
latent-1024 assumption. Positive `H,I` are limited to 524280 by CUDA grid.y.

Expert indices must be in `[0,E)`; their device values are not copied to the
host or synchronized on every decode. Device assertions catch invalid IDs in
assert-enabled builds; bounds guards prevent invalid weight accesses even
when assertions are disabled. Invalid IDs are outside the operation contract,
and their outputs are undefined. Router output satisfies this precondition.

## ABI, validation, and build

The status-returning `Ort::CustomOpBase<...,true>` ABI, ORT-provided CUDA stream,
static custom domain lifetime, and one-time registration follow Bolt.
Input/output counts, tensor element types, exact ranks/shapes/counts, integer
ranges, required attributes and CUDA grid limits are validated explicitly;
invalid contracts return `ORT_INVALID_ARGUMENT`. Each launch is checked with
`cudaGetLastError`, returning `ORT_FAIL` on launch failure. Like Bolt, execution
is asynchronous: later device execution errors are observed by ORT at its
normal synchronization points, not by launch checks alone.

The kernel source and its standalone CMake project live in `src/lightning`,
beside `src/bolt`. Export scripts remain in
[export-onnx-lightning](../../export-onnx-lightning/README.md).

Configure this folder separately from the original Bolt build, using the
ONNX Runtime **main** C/C++ API headers (including their companion headers):

```powershell
cmake -S src\lightning -B build\lightning -DORT_INCLUDE_DIR=C:\path\to\onnxruntime\include\onnxruntime\core\session
cmake --build build\lightning --config Release
```

CMake requests CUDA architecture 90 (sm90), CUDA/C++17, PIC, and
`-O3 --fmad=false -lineinfo`; the shared library exports `RegisterCustomOps`.
There is no ORT binary link dependency because the API table is initialized
at registration. Use a matching CUDA-enabled ORT build and compatible CUDA
toolchain on the target machine.

The commands above run from the repository root. See the
[H100 guide](../../export-onnx-lightning/README.md#build-and-test-the-custom-library)
for Linux build and test commands.

**Validation limitation:** this adaptation has not been compiled or run on a
GPU in the authoring environment. Source review is not evidence of compilation,
numerical parity, correct runtime loading, or performance.
