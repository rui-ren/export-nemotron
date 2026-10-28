"""Build the operations inside a Bolt layer.

Each method receives tensor *names* and returns a new tensor name. It does
not run inference. graph.op records an operation; graph.init stores a weight.
The original mathematical operations and quantized layouts are preserved.
"""
from __future__ import annotations

import torch
from onnx import helper

from checkpoint import BACKBONE_PREFIX
from graph import BF16, F32, I64, I32, U8, MS, BOLT, GraphBuilder, value_info
from quantization import (
    dequantize_iq2xxs, dequantize_nvfp4, dequantize_q8_0,
    iq2xxs_to_int4, iq2xxs_to_int8, iq2xxs_to_lut4, iq2xxs_to_native, q8_0_split,
)


class LayerBuilder:
    def __init__(self, options, checkpoint, store):
        self.options = options
        self.checkpoint = checkpoint
        self.config = checkpoint.config
        self.device = options.device
        self.store = store
        self.graph = GraphBuilder(store)
        self.layer_types = self.config["layers_block_type"]
        self.num_layers = options.num_layers or len(self.layer_types)
        self.hidden_size = self.config["hidden_size"]
        self.epsilon = self.config["layer_norm_epsilon"]

    def linear(self, x, weight_name: str, out_features: int, in_features: int, kind: str, tag: str):
        """kind (checkpoint format): 'q8' | 'iq2' | 'nvfp4'; the exported format is selected by args."""
        graph, options = self.graph, self.options
        if kind == "q8":
            blocks = self.checkpoint.get(weight_name, self.device)
            if options.q8 == "int8":
                q, d = q8_0_split(blocks)
                B = graph.init(tag + ".qw", q.reshape(out_features, in_features // 32, 32).cpu())
                S = graph.init(tag + ".sc", d.to(torch.bfloat16).cpu())
                if options.fused:
                    return graph.op("Q8Gemv", [x, B, S], domain=BOLT)
                return graph.op("MatMulNBits", [x, B, S], domain=MS, K=in_features, N=out_features, bits=8, block_size=32)
            w = dequantize_q8_0(blocks).to(torch.bfloat16)
        elif kind == "iq2":
            blocks = self.checkpoint.get(weight_name, self.device)
            if options.iq2 == "native":
                rec, dd = iq2xxs_to_native(blocks)
                B = graph.init(tag + ".qw", rec.reshape(out_features, in_features // 32, 8).cpu())
                S = graph.init(tag + ".sc", dd.reshape(out_features, in_features // 256).cpu())
                return graph.op("Q4Gemv", [x, B, S], domain=BOLT, iq2=2)
            if options.iq2 in ("int4", "lut4"):
                p, s = (iq2xxs_to_lut4 if options.iq2 == "lut4" else iq2xxs_to_int4)(blocks)
                B = graph.init(tag + ".qw", p.reshape(out_features, in_features // 32, 16).cpu())
                S = graph.init(tag + ".sc", s.to(torch.bfloat16).cpu())
                if options.fused:
                    return graph.op("Q4Gemv", [x, B, S], domain=BOLT, **({"iq2": 1} if options.iq2 == "lut4" else {}))
                return graph.op("MatMulNBits", [x, B, S], domain=MS, K=in_features, N=out_features, bits=4, block_size=32)
            w = dequantize_iq2xxs(blocks).to(torch.bfloat16)
        elif kind == "nvfp4" and options.fused:
            base = weight_name[: -len(".weight")]
            wq = graph.init(tag + ".fp4", self.checkpoint.get(weight_name))
            ws = graph.init(tag + ".fp4s", self.checkpoint.get(base + ".weight_scale").view(torch.uint8))
            gs = graph.init(tag + ".fp4g", self.checkpoint.get(base + ".weight_scale_2").to(torch.float32).reshape(1))
            xin = graph.op("Reshape", [x, graph.const([-1, in_features])])
            y = graph.op("Nvfp4Gemv", [xin, wq, ws, gs], domain=BOLT)
            return graph.op("Reshape", [y, graph.const([1, -1, out_features])])
        elif kind == "nvfp4":
            base = weight_name[: -len(".weight")]
            w = dequantize_nvfp4(
                self.checkpoint.get(weight_name, self.device),
                self.checkpoint.get(base + ".weight_scale", self.device),
                self.checkpoint.get(base + ".weight_scale_2", self.device),
            ).to(torch.bfloat16)
        else:
            raise ValueError(kind)
        assert w.shape == (out_features, in_features), (weight_name, w.shape, out_features, in_features)
        W = graph.init(tag + ".w", w.t().contiguous().cpu())
        return graph.op("MatMul", [x, W])


    def rms(self, x, weight_name):
        """Normalize the last dimension, then multiply by a learned scale."""
        w = self.graph.init(weight_name, self.checkpoint.get(weight_name).to(torch.bfloat16))
        return self.graph.op("SimplifiedLayerNormalization", [x, w], epsilon=self.epsilon, axis=-1)


    def skip_rms(self, x, skip, weight_name):
        """Add the residual stream and normalize; return both results."""
        w = self.graph.init(weight_name, self.checkpoint.get(weight_name).to(torch.bfloat16))
        out = self.graph.op("SkipSimplifiedLayerNormalization", [x, skip, w], nout=4, domain=MS, epsilon=self.epsilon)
        return out[0], out[3]


    def mamba(self, i, x):
        """Project -> causal convolution -> recurrent state update -> normalize."""
        graph, config, prefix = self.graph, self.config, f"{BACKBONE_PREFIX}layers.{i}.mixer."
        batch_size = 1 if self.options.decode else "batch"
        heads, head_dim, state_size, num_groups = config["mamba_num_heads"], config["mamba_head_dim"], config["ssm_state_size"], config["n_groups"]
        inner_size = heads * head_dim
        conv_dim = inner_size + 2 * num_groups * state_size
        proj = self.linear(x, prefix + "in_proj.weight", inner_size + conv_dim + heads, self.hidden_size, "nvfp4", f"l{i}.in_proj")
        if self.options.fused:
            return self.mamba_fused(i, proj, batch_size)
        gate, xbc, dt = graph.op("Split", [proj, graph.const([inner_size, conv_dim, heads])], nout=3, axis=-1)
        xbc_t = graph.op("Transpose", [xbc], perm=[0, 2, 1])
        cw = graph.init(f"l{i}.conv_w", self.checkpoint.get(prefix + "conv1d.weight").to(torch.bfloat16))
        cb = graph.init(f"l{i}.conv_b", self.checkpoint.get(prefix + "conv1d.bias").to(torch.bfloat16))
        pc = f"past_conv.{i}"
        graph.inputs.append(value_info(pc, BF16, [batch_size, conv_dim, config["conv_kernel"] - 1]))
        conv_out, new_conv = graph.op("CausalConvWithState", [xbc_t, cw, cb, pc], nout=2, domain=MS, activation="silu")
        graph.outputs.append(value_info(f"present_conv.{i}", BF16, [batch_size, conv_dim, config["conv_kernel"] - 1]))
        graph.nodes.append(helper.make_node("Identity", [new_conv], [f"present_conv.{i}"]))
        xbc_a = graph.op("Transpose", [conv_out], perm=[0, 2, 1])
        xs, bm, cm = graph.op("Split", [xbc_a, graph.const([inner_size, num_groups * state_size, num_groups * state_size])], nout=3, axis=-1)
        dtf = graph.op("Cast", [dt], to=F32)
        dtb = graph.init(f"l{i}.dt_bias", self.checkpoint.get(prefix + "dt_bias").to(torch.float32))
        dts = graph.op("Softplus", [graph.op("Add", [dtf, dtb])])
        if self.options.dt_min > 0:
            dts = graph.op("Max", [dts, graph.const(float(self.options.dt_min), F32)])
        a_neg = graph.init(f"l{i}.A_neg", -torch.exp(self.checkpoint.get(prefix + "A_log").to(torch.float32)))
        decay = graph.op("Mul", [dts, a_neg])
        xf = graph.op("Cast", [xs], to=F32)
        x4 = graph.op("Reshape", [xf, graph.const([0, 0, heads, head_dim])])
        time_step_per_head = graph.op("Unsqueeze", [dts, graph.const([-1])])
        weighted_values = graph.op("Mul", [x4, time_step_per_head])
        value = graph.op("Reshape", [weighted_values, graph.const([0, 0, inner_size])])
        qf = graph.op("Cast", [cm], to=F32)
        kf = graph.op("Cast", [bm], to=F32)
        ps = f"past_ssm.{i}"
        graph.inputs.append(value_info(ps, F32, [batch_size, heads, state_size, head_dim]))
        y, new_ssm = graph.op(
            "LinearAttention",
            [qf, kf, value, ps, decay],
            nout=2,
            domain=MS,
            update_rule="gated",
            scale=1.0,
            q_num_heads=num_groups,
            kv_num_heads=heads,
        )
        graph.outputs.append(value_info(f"present_ssm.{i}", F32, [batch_size, heads, state_size, head_dim]))
        graph.nodes.append(helper.make_node("Identity", [new_ssm], [f"present_ssm.{i}"]))
        Dp = graph.init(f"l{i}.D", self.checkpoint.get(prefix + "D").to(torch.float32).reshape(1, 1, heads, 1))
        dskip = graph.op("Reshape", [graph.op("Mul", [x4, Dp]), graph.const([0, 0, inner_size])])
        y = graph.op("Add", [y, dskip])
        gf = graph.op("Cast", [gate], to=F32)
        gate_sigmoid = graph.op("Sigmoid", [gf])
        activated_gate = graph.op("Mul", [gf, gate_sigmoid])
        yg = graph.op("Mul", [y, activated_gate])
        y4 = graph.op("Reshape", [yg, graph.const([0, 0, num_groups, inner_size // num_groups])])
        if "ones_gn" not in self.store.names:
            graph.init("ones_gn", torch.ones(inner_size // num_groups, dtype=torch.float32))
        yn = graph.op("SimplifiedLayerNormalization", [y4, "ones_gn"], epsilon=float(config["rms_norm_eps"]), axis=-1)
        norm_weight = graph.init(f"l{i}.norm_w", self.checkpoint.get(prefix + "norm.weight").to(torch.float32).reshape(1, 1, num_groups, inner_size // num_groups))
        yn = graph.op("Cast", [graph.op("Reshape", [graph.op("Mul", [yn, norm_weight]), graph.const([0, 0, inner_size])])], to=BF16)
        return self.linear(yn, prefix + "out_proj.weight", self.hidden_size, inner_size, "nvfp4", f"l{i}.out_proj")


    def mamba_fused(self, i, proj, batch_size):
        """Record a CUDA op replacing the middle of the explicit Mamba chain."""
        graph, config, prefix = self.graph, self.config, f"{BACKBONE_PREFIX}layers.{i}.mixer."
        heads, head_dim, state_size, num_groups = config["mamba_num_heads"], config["mamba_head_dim"], config["ssm_state_size"], config["n_groups"]
        inner_size = heads * head_dim
        conv_dim = inner_size + 2 * num_groups * state_size
        p2 = graph.op("Reshape", [proj, graph.const([-1, inner_size + conv_dim + heads])])
        cw = graph.init(f"l{i}.conv_w", self.checkpoint.get(prefix + "conv1d.weight").to(torch.bfloat16))
        cb = graph.init(f"l{i}.conv_b", self.checkpoint.get(prefix + "conv1d.bias").to(torch.bfloat16))
        pc, ps = f"past_conv.{i}", f"past_ssm.{i}"
        SR = self.options.verify_rows or batch_size
        graph.inputs.append(value_info(pc, BF16, [SR, conv_dim, config["conv_kernel"] - 1]))
        graph.inputs.append(value_info(ps, F32, [SR, heads, state_size, head_dim]))
        dtb = graph.init(f"l{i}.dt_bias", self.checkpoint.get(prefix + "dt_bias").to(torch.float32))
        a_neg = graph.init(f"l{i}.A_neg", -torch.exp(self.checkpoint.get(prefix + "A_log").to(torch.float32)))
        Dp = graph.init(f"l{i}.D", self.checkpoint.get(prefix + "D").to(torch.float32))
        norm_weight = graph.init(f"l{i}.norm_w", self.checkpoint.get(prefix + "norm.weight").to(torch.float32))
        y, nconv, nssm, _, _ = graph.op(
            "MambaVerify" if self.options.verify_rows else "MambaStep",
            [p2, pc, ps, cw, cb, dtb, a_neg, Dp, norm_weight] + (["state_row"] if self.options.verify_rows else []), nout=5, domain=BOLT,
            heads=heads, head_dim=head_dim, d_state=state_size, n_groups=num_groups, epsilon=float(config["rms_norm_eps"]),
            time_step_min=float(self.options.dt_min),
        )
        graph.outputs.append(value_info(f"present_conv.{i}", BF16, [SR, conv_dim, config["conv_kernel"] - 1]))
        graph.nodes.append(helper.make_node("Identity", [nconv], [f"present_conv.{i}"]))
        graph.outputs.append(value_info(f"present_ssm.{i}", F32, [SR, heads, state_size, head_dim]))
        graph.nodes.append(helper.make_node("Identity", [nssm], [f"present_ssm.{i}"]))
        yb = graph.op("Reshape", [y, graph.const([1, -1, inner_size])])
        return self.linear(yb, prefix + "out_proj.weight", self.hidden_size, inner_size, "nvfp4", f"l{i}.out_proj")


    def attention(self, i, x, seqlens_k, total_len):
        """Project Q/K/V, attend using cached keys/values, then project back."""
        graph, config, prefix = self.graph, self.config, f"{BACKBONE_PREFIX}layers.{i}.mixer."
        num_heads, num_kv_heads, head_dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        q = self.linear(x, prefix + "q_proj.weight", num_heads * head_dim, self.hidden_size, "q8", f"l{i}.q")
        k = self.linear(x, prefix + "k_proj.weight", num_kv_heads * head_dim, self.hidden_size, "q8", f"l{i}.k")
        v = self.linear(x, prefix + "v_proj.weight", num_kv_heads * head_dim, self.hidden_size, "q8", f"l{i}.v")
        pk, pv = f"past_key.{i}", f"past_value.{i}"
        ml = self.options.max_len if self.options.decode else "past_len"
        graph.inputs.append(value_info(pk, BF16, [1 if self.options.decode else "batch", num_kv_heads, ml, head_dim]))
        graph.inputs.append(value_info(pv, BF16, [1 if self.options.decode else "batch", num_kv_heads, ml, head_dim]))
        o, nk, nv = graph.op(
            "GroupQueryAttention",
            [q, k, v, pk, pv, seqlens_k, total_len],
            nout=3,
            domain=MS,
            num_heads=num_heads,
            kv_num_heads=num_kv_heads,
            do_rotary=0,
        )
        ml = self.options.max_len if self.options.decode else "total_len"
        graph.outputs.append(value_info(f"present_key.{i}", BF16, [1 if self.options.decode else "batch", num_kv_heads, ml, head_dim]))
        graph.outputs.append(value_info(f"present_value.{i}", BF16, [1 if self.options.decode else "batch", num_kv_heads, ml, head_dim]))
        graph.nodes.append(helper.make_node("Identity", [nk], [f"present_key.{i}"]))
        graph.nodes.append(helper.make_node("Identity", [nv], [f"present_value.{i}"]))
        return self.linear(o, prefix + "o_proj.weight", self.hidden_size, num_heads * head_dim, "q8", f"l{i}.o")


    def relu2(self, x):
        """The expert activation is max(x, 0) squared, not ordinary ReLU."""
        if self.options.fused:
            return self.graph.op("Relu2", [x], domain=BOLT)
        r = self.graph.op("Relu", [x])
        return self.graph.op("Mul", [r, r])


    def moe(self, i, x):
        """Route tokens to experts, run in latent space, add a shared expert."""
        graph, config, prefix = self.graph, self.config, f"{BACKBONE_PREFIX}layers.{i}.mixer."
        bc = config["block_configs"][i]
        E, topk = bc["n_routed_experts"], bc["num_experts_per_tok"]
        lat = config["moe_latent_size"]
        sh = config["moe_shared_expert_intermediate_size"]
        x2 = graph.op("Reshape", [x, graph.const([-1, self.hidden_size])])  # [T,H]
        if self.options.fused:
            gw = graph.init(f"l{i}.gate_w", self.checkpoint.get(prefix + "gate.weight").to(torch.bfloat16))
            gb = graph.init(f"l{i}.gate_bias", self.checkpoint.get(prefix + "gate.e_score_correction_bias").to(torch.float32))
            ridx, rw, _ = graph.op("MoeRoute", [x2, gw, gb], nout=3, domain=BOLT, topk=topk,
                               scale=float(config["routed_scaling_factor"]))
        else:
            gw = graph.init(f"l{i}.gate_w", self.checkpoint.get(prefix + "gate.weight").to(torch.float32).t().contiguous())
            gb = graph.init(f"l{i}.gate_bias", self.checkpoint.get(prefix + "gate.e_score_correction_bias").to(torch.float32))
            logits = graph.op("MatMul", [graph.op("Cast", [x2], to=F32), gw])
            scores = graph.op("Sigmoid", [logits])
            # The correction bias selects experts; it must not change their output weights.
            sel = graph.op("Add", [scores, gb])
            _, idx = graph.op("TopK", [sel, graph.const([topk])], nout=2, axis=-1, largest=1, sorted=0)
            tw = graph.op("GatherElements", [scores, idx], axis=1)
            weight_sum = graph.op("ReduceSum", [tw, graph.const([1])], keepdims=1)
            safe_weight_sum = graph.op("Add", [weight_sum, graph.const(1e-20, F32)])
            tw = graph.op("Div", [tw, safe_weight_sum])
            tw = graph.op("Mul", [tw, graph.const(float(config["routed_scaling_factor"]), F32)])
            zeros = graph.op("ConstantOfShape", [graph.op("Shape", [scores])], value=helper.make_tensor("z", F32, [1], [0.0]))
            ones = graph.op("Expand", [graph.const(1.0, F32), graph.op("Shape", [idx])])
            probs = graph.op("Cast", [graph.op("ScatterElements", [zeros, idx, ones], axis=1)], to=BF16)
            wdense = graph.op("Cast", [graph.op("ScatterElements", [zeros, idx, tw], axis=1)], to=BF16)
        z = self.linear(x2, prefix + "fc1_latent_proj.weight", lat, self.hidden_size, "q8", f"l{i}.fc1_lat")
        up = self.checkpoint.get_many([f"{prefix}experts.{e}.up_proj.weight" for e in range(E)], self.device)
        conv = {"int8": iq2xxs_to_int8, "lut4": iq2xxs_to_lut4, "native": iq2xxs_to_native}.get(self.options.experts, iq2xxs_to_int4)
        p1, s1 = conv(up)
        del up
        dn = self.checkpoint.get_many([f"{prefix}experts.{e}.down_proj.weight" for e in range(E)], self.device)
        p2, s2 = conv(dn)
        del dn
        w1 = graph.init(f"l{i}.fc1_w", p1.cpu())
        sc1 = graph.init(f"l{i}.fc1_s", s1.to(torch.bfloat16).cpu())
        w2 = graph.init(f"l{i}.fc2_w", p2.cpu())
        sc2 = graph.init(f"l{i}.fc2_s", s2.to(torch.bfloat16).cpu())
        if self.options.fused:
            assert self.options.experts in ("int4", "lut4", "native")
            assert self.options.experts != "native" or self.options.verify_rows, "native IQ2 experts need the multi-row MoeVerify path (--verify-rows)"
            if self.options.verify_rows:  # experts shared across the rows are read once
                moe_out = graph.op("MoeVerify", [z, ridx, rw, w1, sc1, w2, sc2], nout=4, domain=BOLT, topk=topk, iq2={"lut4": 1, "native": 2}.get(self.options.experts, 0))[0]
            else:
                moe_out, _, _ = graph.op("MoeDecode", [z, ridx, rw, w1, sc1, w2, sc2], nout=3, domain=BOLT, topk=topk)
            out = self.linear(moe_out, prefix + "fc2_latent_proj.weight", self.hidden_size, lat, "q8", f"l{i}.fc2_lat")
            s_up = self.linear(x2, prefix + "shared_experts.up_proj.weight", sh, self.hidden_size, "iq2", f"l{i}.sh_up")
            s_dn = self.linear(self.relu2(s_up), prefix + "shared_experts.down_proj.weight", self.hidden_size, sh, "iq2", f"l{i}.sh_dn")
            out = graph.op("Add", [out, s_dn])
            return graph.op("Reshape", [out, graph.const([1, -1, self.hidden_size]) if self.options.decode else graph.op("Shape", [x])])
        ins = [z, probs, w1, sc1, "", w2, sc2, "", "", "", "", "", "", "", wdense]
        moe_out = graph.op(
            "QMoE",
            ins,
            domain=MS,
            k=topk,
            activation_type="relu2",
            normalize_routing_weights=0,
            expert_weight_bits=8 if self.options.experts == "int8" else 4,
            block_size=32,
            weights_prepacked=0,
        )
        out = self.linear(moe_out, prefix + "fc2_latent_proj.weight", self.hidden_size, lat, "q8", f"l{i}.fc2_lat")
        s_up = self.linear(x2, prefix + "shared_experts.up_proj.weight", sh, self.hidden_size, "iq2", f"l{i}.sh_up")
        s_dn = self.linear(self.relu2(s_up), prefix + "shared_experts.down_proj.weight", self.hidden_size, sh, "iq2", f"l{i}.sh_dn")
        out = graph.op("Add", [out, s_dn])
        return graph.op("Reshape", [out, graph.const([1, -1, self.hidden_size]) if self.options.decode else graph.op("Shape", [x])])
