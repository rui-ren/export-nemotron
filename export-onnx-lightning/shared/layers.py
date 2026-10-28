"""Single-token Lightning layers. Operations record a graph; they do not infer."""
from __future__ import annotations

import torch

from .graph import BF16, CUSTOM, FLOAT, INT32, INT64, MS
from .quantization import quantize_int4


class Layers:
    def __init__(self, checkpoint, graph, options):
        self.checkpoint, self.g, self.options = checkpoint, graph, options
        self.c = checkpoint.config
        self.hidden = self.c["hidden_size"]
        self.custom = options.variant in ("custom", "int4") or (
            options.variant == "mtp" and options.mtp_backend == "custom"
        )
        self.quantized = options.variant == "int4"

    def weight(self, name, shape, dtype=None):
        return self.g.store.add(name, self.checkpoint.get(name, shape).to(dtype or self.g.dtype))

    def linear(self, x, name, rows, columns, quantize=True):
        g = self.g
        weight = self.checkpoint.get(name, (rows, columns))
        if self.quantized and quantize:
            packed, scales = quantize_int4(weight, self.options.block_size, g.dtype)
            qname = g.store.add(name + ".int4", packed)
            sname = g.store.add(name + ".scales", scales)
            return g.op("MatMulNBits", [x, qname, sname], domain=MS, K=columns, N=rows,
                        bits=4, block_size=self.options.block_size)
        name = g.store.add(name, weight.to(g.dtype).t().contiguous())
        return g.op("MatMul", [x, name])

    def rms(self, x, name, width=None):
        g, width = self.g, width or self.hidden
        weight = self.weight(name, (width,), torch.float32)
        value = g.cast(x, FLOAT)
        squared = g.op("Mul", [value, value])
        variance = g.op("ReduceMean", [squared, g.constant([-1])], keepdims=1)
        denominator = g.op("Sqrt", [g.op("Add", [
            variance, g.constant(float(self.c["layer_norm_epsilon"]), FLOAT)])])
        normalized = g.op("Div", [value, denominator])
        return g.cast(g.op("Mul", [normalized, weight]), g.type)

    def relu2(self, x):
        positive = self.g.op("Relu", [x])
        return self.g.op("Mul", [positive, positive])

    def expert_weights(self, prefix, direction):
        g, c = self.g, self.c
        experts, inter = c["n_routed_experts"], c["moe_intermediate_size"]
        rows, columns = (inter, self.hidden) if direction == "up" else (self.hidden, inter)
        names = [f"{prefix}experts.{index}.{direction}_proj.weight" for index in range(experts)]
        output_name = prefix + direction + "_experts"
        if not self.quantized:
            return (g.store.add_chunks(
                output_name, (experts, rows, columns), g.dtype,
                (self.checkpoint.get(name, (rows, columns)).to(g.dtype) for name in names),
            ),)
        block = self.options.block_size
        count = (columns + block - 1) // block
        # Two bounded passes avoid retaining an entire layer's expert weights.
        packed = g.store.add_chunks(
            output_name + ".int4", (experts, rows, count, block // 2), torch.uint8,
            (quantize_int4(self.checkpoint.get(name, (rows, columns)), block, g.dtype)[0] for name in names),
        )
        scales = g.store.add_chunks(
            output_name + ".scales", (experts, rows, count), g.dtype,
            (quantize_int4(self.checkpoint.get(name, (rows, columns)), block, g.dtype)[1] for name in names),
        )
        return packed, scales

    def moe(self, x, prefix):
        g, c = self.g, self.c
        experts, topk = c["n_routed_experts"], c["num_experts_per_tok"]
        inter, shared = c["moe_intermediate_size"], c["moe_shared_expert_intermediate_size"]
        flat = g.reshape(x, [1, self.hidden])
        bias = self.weight(prefix + "gate.e_score_correction_bias", (experts,), torch.float32)
        if self.custom:
            router_name = self.weight(prefix + "gate.weight", (experts, self.hidden))
            indices, probabilities, _ = g.op(
                "MoeRoute", [flat, router_name, bias], outputs=3, domain=CUSTOM,
                topk=topk, scale=float(c["routed_scaling_factor"]),
            )
        else:
            router = self.checkpoint.get(prefix + "gate.weight", (experts, self.hidden)).float().t().contiguous()
            router_name = g.store.add(prefix + "gate.weight", router)
            scores = g.op("Sigmoid", [g.op("MatMul", [g.cast(flat, FLOAT), router_name])])
            choice = g.op("Add", [scores, bias])
            _, indices = g.op("TopK", [choice, g.constant([topk])], outputs=2, axis=-1, largest=1, sorted=1)
            selected = g.op("GatherElements", [scores, indices], axis=1)
            denominator = g.op("ReduceSum", [selected, g.constant([-1])], keepdims=1)
            denominator = g.op("Add", [denominator, g.constant(1e-20, FLOAT)])
            probabilities = g.op("Div", [selected, denominator])
            probabilities = g.op("Mul", [probabilities, g.constant(float(c["routed_scaling_factor"]), FLOAT)])
        up, down = self.expert_weights(prefix, "up"), self.expert_weights(prefix, "down")
        if self.custom:
            kind = "MoeDecode" if self.quantized else "MoeDecodeBF16"
            inputs = [flat, indices, probabilities, *up, *down]
            routed, _, _ = g.op(kind, inputs, outputs=3, domain=CUSTOM, topk=topk,
                                **({"block_size": self.options.block_size} if self.quantized else {}))
        else:
            # Gather only selected experts, rather than evaluating all experts.
            chosen = g.reshape(indices, [topk])
            up_selected = g.op("Gather", [up[0], chosen], axis=0)
            down_selected = g.op("Gather", [down[0], chosen], axis=0)
            up_transposed = g.op("Transpose", [up_selected], perm=[0, 2, 1])
            expanded_x = g.reshape(flat, [1, 1, self.hidden])
            activated = self.relu2(g.op("MatMul", [expanded_x, up_transposed]))
            down_transposed = g.op("Transpose", [down_selected], perm=[0, 2, 1])
            values = g.op("MatMul", [activated, down_transposed])
            weighted = g.op("Mul", [g.cast(values, FLOAT), g.reshape(probabilities, [topk, 1, 1])])
            routed = g.cast(g.op("ReduceSum", [weighted, g.constant([0])], keepdims=0), g.type)
        shared_up = self.linear(flat, prefix + "shared_experts.up_proj.weight", shared, self.hidden)
        shared_down = self.linear(self.relu2(shared_up), prefix + "shared_experts.down_proj.weight",
                                  self.hidden, shared)
        return g.reshape(g.op("Add", [routed, shared_down]), [1, 1, self.hidden])

    def attention(self, x, prefix, layer):
        g, c = self.g, self.c
        heads, kv, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        query = self.linear(x, prefix + "q_proj.weight", heads * dim, self.hidden)
        key = self.linear(x, prefix + "k_proj.weight", kv * dim, self.hidden)
        value = self.linear(x, prefix + "v_proj.weight", kv * dim, self.hidden)
        shape = [1, kv, self.options.max_length, dim]
        past_key = g.cache(f"key.{layer}", g.type, shape)
        past_value = g.cache(f"value.{layer}", g.type, shape)
        if self.options.attention == "gqa":
            output, new_key, new_value = g.op(
                "GroupQueryAttention",
                [query, key, value, past_key, past_value, "seqlens_k", "total_sequence_length"],
                outputs=3, domain=MS, num_heads=heads, kv_num_heads=kv, do_rotary=0,
            )
        else:
            position = g.op("Expand", ["seqlens_k", g.constant([kv, 1])])
            head_ids = g.reshape(g.constant(list(range(kv))), [kv, 1])
            indices = g.op("Concat", [
                g.reshape(g.constant([0] * kv), [kv, 1]), head_ids, g.cast(position, INT64),
            ], axis=1)
            new_key = g.op("ScatterND", [past_key, indices, g.reshape(key, [kv, dim])])
            new_value = g.op("ScatterND", [past_value, indices, g.reshape(value, [kv, dim])])
            end = g.op("Unsqueeze", [g.cast("total_sequence_length", INT64), g.constant([0])])
            key_valid = g.op("Slice", [new_key, g.constant([0]), end, g.constant([2])])
            value_valid = g.op("Slice", [new_value, g.constant([0]), end, g.constant([2])])
            repeats = g.constant([head // (heads // kv) for head in range(heads)])
            keys = g.op("Gather", [key_valid, repeats], axis=1)
            values = g.op("Gather", [value_valid, repeats], axis=1)
            queries = g.op("Transpose", [g.reshape(query, [1, 1, heads, dim])], perm=[0, 2, 1, 3])
            scores = g.op("MatMul", [g.cast(queries, FLOAT),
                                    g.op("Transpose", [g.cast(keys, FLOAT)], perm=[0, 1, 3, 2])])
            scores = g.op("Mul", [scores, g.constant(dim ** -0.5, FLOAT)])
            probs = g.cast(g.op("Softmax", [scores], axis=-1), g.type)
            context = g.op("MatMul", [probs, values])
            output = g.reshape(g.op("Transpose", [context], perm=[0, 2, 1, 3]), [1, 1, heads * dim])
        g.present(new_key, f"key.{layer}", g.type, shape)
        g.present(new_value, f"value.{layer}", g.type, shape)
        return self.linear(output, prefix + "o_proj.weight", self.hidden, heads * dim)

    def mamba(self, x, prefix, layer):
        g, c = self.g, self.c
        heads, dim, state, groups = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
        inner, kernel = heads * dim, c["conv_kernel"]
        channels = inner + 2 * groups * state
        projected = self.linear(x, prefix + "in_proj.weight", inner + channels + heads, self.hidden)
        conv_shape, state_shape = [1, channels, kernel - 1], [1, heads, state, dim]
        past_conv = g.cache(f"conv.{layer}", g.type, conv_shape)
        past_ssm = g.cache(f"ssm.{layer}", FLOAT, state_shape)
        conv_w = self.weight(prefix + "conv1d.weight", (channels, 1, kernel))
        conv_b = self.weight(prefix + "conv1d.bias", (channels,))
        dt_bias = self.weight(prefix + "dt_bias", (heads,), torch.float32)
        a = -torch.exp(self.checkpoint.get(prefix + "A_log", (heads,)).float())
        a_name = g.store.add(prefix + "A_negative", a)
        skip = self.weight(prefix + "D", (heads,), torch.float32)
        norm = self.weight(prefix + "norm.weight", (inner,), torch.float32)
        if self.custom:
            y, new_conv, new_ssm, _, _ = g.op(
                "MambaStep",
                [g.reshape(projected, [1, inner + channels + heads]), past_conv, past_ssm,
                 conv_w, conv_b, dt_bias, a_name, skip, norm],
                outputs=5, domain=CUSTOM, heads=heads, head_dim=dim, d_state=state,
                n_groups=groups, epsilon=float(c["layer_norm_epsilon"]),
                time_step_min=float(self.options.dt_min),
            )
        else:
            gate, xbc, dt = g.op("Split", [
                g.reshape(projected, [1, inner + channels + heads]),
                g.constant([inner, channels, heads])], outputs=3, axis=1)
            history = g.op("Concat", [past_conv, g.reshape(xbc, [1, channels, 1])], axis=2)
            new_conv = g.op("Slice", [history, g.constant([1]), g.constant([kernel]), g.constant([2])])
            product = g.op("Mul", [g.cast(history, FLOAT), g.reshape(g.cast(conv_w, FLOAT), [1, channels, kernel])])
            conv = g.op("ReduceSum", [product, g.constant([2])], keepdims=0)
            conv = g.op("Add", [conv, g.cast(conv_b, FLOAT)])
            conv = g.cast(g.op("Mul", [conv, g.op("Sigmoid", [conv])]), g.type)
            xs, bs, cs = g.op("Split", [conv, g.constant([inner, groups * state, groups * state])],
                              outputs=3, axis=1)
            group_ids = g.constant([head // (heads // groups) for head in range(heads)])
            bs = g.op("Gather", [g.reshape(g.cast(bs, FLOAT), [1, groups, state]), group_ids], axis=1)
            cs = g.op("Gather", [g.reshape(g.cast(cs, FLOAT), [1, groups, state]), group_ids], axis=1)
            dt = g.op("Softplus", [g.op("Add", [g.cast(dt, FLOAT), dt_bias])])
            if self.options.dt_min:
                dt = g.op("Max", [dt, g.constant(float(self.options.dt_min), FLOAT)])
            dt = g.reshape(dt, [1, heads, 1, 1])
            decay = g.op("Exp", [g.op("Mul", [dt, g.reshape(a_name, [1, heads, 1, 1])])])
            xs = g.reshape(g.cast(xs, FLOAT), [1, heads, dim, 1])
            delta = g.op("Mul", [g.op("Mul", [dt, g.reshape(bs, [1, heads, 1, state])]), xs])
            previous = g.op("Transpose", [past_ssm], perm=[0, 1, 3, 2])
            new_ssm = g.op("Add", [g.op("Mul", [previous, decay]), delta])
            yc = g.op("Mul", [new_ssm, g.reshape(cs, [1, heads, 1, state])])
            y = g.op("ReduceSum", [yc, g.constant([-1])], keepdims=0)
            dskip = g.op("Mul", [g.reshape(xs, [1, heads, dim]), g.reshape(skip, [1, heads, 1])])
            y = g.reshape(g.cast(g.op("Add", [y, dskip]), g.type), [1, inner])
            gate = g.cast(gate, FLOAT)
            activated_gate = g.op("Mul", [gate, g.op("Sigmoid", [gate])])
            gated = g.op("Mul", [g.cast(y, FLOAT), activated_gate])
            grouped = g.reshape(gated, [1, groups, inner // groups])
            variance = g.op("ReduceMean", [g.op("Mul", [grouped, grouped]), g.constant([-1])], keepdims=1)
            denom = g.op("Sqrt", [g.op("Add", [variance, g.constant(float(c["layer_norm_epsilon"]), FLOAT)])])
            normalized = g.reshape(g.op("Div", [grouped, denom]), [1, inner])
            y = g.cast(g.op("Mul", [g.cast(g.cast(normalized, g.type), FLOAT), norm]), g.type)
            new_ssm = g.op("Transpose", [new_ssm], perm=[0, 1, 3, 2])
        g.present(new_conv, f"conv.{layer}", g.type, conv_shape)
        g.present(new_ssm, f"ssm.{layer}", FLOAT, state_shape)
        return self.linear(g.reshape(y, [1, 1, inner]), prefix + "out_proj.weight", self.hidden, inner)
