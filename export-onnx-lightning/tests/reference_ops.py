"""CPU evaluators for graph tests, not replacements for the CUDA custom library."""
import numpy as np
import torch
from onnx.reference.op_run import OpRun

from shared.quantization import dequantize_int4
from shared.reference import experts_core, mamba_core


class GroupQueryAttention(OpRun):
    op_domain = "com.microsoft"

    def _run(self, q, k, v, pk, pv, lengths, total, num_heads=None, kv_num_heads=None, **attributes):
        position = int(lengths[0])
        dimension = pk.shape[-1]
        nk, nv = pk.copy(), pv.copy()
        nk[:, :, position] = k.reshape(1, kv_num_heads, dimension)
        nv[:, :, position] = v.reshape(1, kv_num_heads, dimension)
        query = q.reshape(1, 1, num_heads, dimension).transpose(0, 2, 1, 3)
        keys = np.repeat(nk[:, :, :position + 1], num_heads // kv_num_heads, axis=1)
        values = np.repeat(nv[:, :, :position + 1], num_heads // kv_num_heads, axis=1)
        logits = query @ keys.transpose(0, 1, 3, 2) * dimension ** -0.5
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        output = (probs @ values).transpose(0, 2, 1, 3).reshape(q.shape)
        return output.astype(q.dtype), nk, nv


class MatMulNBits(OpRun):
    op_domain = "com.microsoft"

    def _run(self, x, packed, scales, K=None, **attributes):
        weight = dequantize_int4(torch.from_numpy(packed.copy()), torch.from_numpy(scales.copy()), K).numpy()
        return ((x @ weight.T).astype(x.dtype),)


class MambaStep(OpRun):
    op_domain = "lightning"

    def _run(self, *inputs, heads=None, head_dim=None, d_state=None, n_groups=None,
             epsilon=None, time_step_min=None, **attributes):
        tensors = [torch.from_numpy(value.copy()) for value in inputs]
        results = mamba_core(*tensors, heads, head_dim, d_state, n_groups, epsilon, time_step_min)
        return tuple(result.numpy() for result in results)


class MoeRoute(OpRun):
    op_domain = "lightning"

    def _run(self, x, gate, bias, topk=None, scale=None, **attributes):
        scores = torch.nn.functional.linear(torch.from_numpy(x.copy()).float(),
                                            torch.from_numpy(gate.copy()).float()).sigmoid()
        # Stable sort defines the same smallest-expert-ID tie rule as the CUDA op.
        indices = torch.argsort(scores + torch.from_numpy(bias.copy()), dim=-1,
                                descending=True, stable=True)[:, :topk]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * scale
        return indices.to(torch.int32).numpy(), weights.numpy(), scores.numpy()


def expert_outputs(x, indices, probabilities, up, down):
    result = experts_core(x, indices, probabilities, up, down)
    activated, partial = [], []
    for expert in indices.reshape(-1):
        hidden = torch.nn.functional.relu(torch.nn.functional.linear(x, up[int(expert)])).square()
        activated.append(hidden)
        partial.append(torch.nn.functional.linear(hidden, down[int(expert)]).float())
    return result.numpy(), torch.cat(activated).numpy(), torch.cat(partial).numpy()


class MoeDecodeBF16(OpRun):
    op_domain = "lightning"

    def _run(self, *inputs, **attributes):
        tensors = [torch.from_numpy(value.copy()) for value in inputs]
        return expert_outputs(*tensors)


class MoeDecode(OpRun):
    op_domain = "lightning"

    def _run(self, x, indices, probabilities, up, up_scales, down, down_scales,
             block_size=None, **attributes):
        up_values = dequantize_int4(torch.from_numpy(up.copy()), torch.from_numpy(up_scales.copy()), x.shape[-1])
        down_values = dequantize_int4(torch.from_numpy(down.copy()), torch.from_numpy(down_scales.copy()),
                                    up.shape[1])
        return expert_outputs(torch.from_numpy(x.copy()), torch.from_numpy(indices.copy()),
                              torch.from_numpy(probabilities.copy()), up_values, down_values)


OPERATORS = [GroupQueryAttention, MatMulNBits, MambaStep, MoeRoute, MoeDecodeBF16, MoeDecode]
