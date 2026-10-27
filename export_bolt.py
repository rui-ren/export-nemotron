"""Export NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB (text backbone) to ONNX (ORT CUDA EP, contrib ops).

Layer types: mamba (CausalConvWithState + LinearAttention[gated]), attention (GroupQueryAttention, no rope), moe (QMoE relu2,
latent experts, shared expert).  Weights are streamed from the safetensors checkpoint into one external data file.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import onnx
import torch
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(__file__))
from bolt_weights import (  # noqa: E402
    PFX,
    Checkpoint,
    GgufCheckpoint,
    dq_iq2xxs,
    dq_nvfp4,
    dq_q8_0,
    iq2xxs_to_int4,
    iq2xxs_to_int8,
    iq2xxs_to_lut4,
    iq2xxs_to_native,
    q8_0_split,
)

BF16 = TensorProto.BFLOAT16
F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
I32 = TensorProto.INT32
U8 = TensorProto.UINT8

_NP2ONNX = {
    torch.bfloat16: BF16,
    torch.float32: F32,
    torch.float16: F16,
    torch.int64: I64,
    torch.int32: I32,
    torch.uint8: U8,
}


def _raw(t: torch.Tensor) -> bytes:
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


class Pool:
    """Writes initializers to a single external data file."""

    def __init__(self, path: str):
        self.f = open(path, "wb")
        self.name = os.path.basename(path)
        self.off = 0
        self.protos: list[TensorProto] = []
        self.names: set[str] = set()

    def add(self, name: str, t: torch.Tensor) -> str:
        assert name not in self.names, name
        self.names.add(name)
        pad = (-self.off) % 64
        if pad:
            self.f.write(b"\0" * pad)
            self.off += pad
        data = _raw(t)
        self.f.write(data)
        tp = TensorProto()
        tp.name = name
        tp.data_type = _NP2ONNX[t.dtype]
        tp.dims.extend(list(t.shape))
        tp.data_location = TensorProto.EXTERNAL
        for k, v in (("location", self.name), ("offset", str(self.off)), ("length", str(len(data)))):
            e = tp.external_data.add()
            e.key, e.value = k, v
        self.off += len(data)
        self.protos.append(tp)
        return name

    def close(self):
        self.f.close()


class G:
    def __init__(self, pool: Pool):
        self.pool = pool
        self.nodes: list = []
        self.n = 0
        self.inputs: list = []
        self.outputs: list = []

    def uid(self, p="t"):
        self.n += 1
        return f"{p}_{self.n}"

    def op(self, typ, ins, nout=1, domain="", name=None, **attrs):
        outs = [self.uid(typ.lower()) for _ in range(nout)]
        self.nodes.append(helper.make_node(typ, ins, outs, name=name or self.uid("n_" + typ), domain=domain, **attrs))
        return outs[0] if nout == 1 else outs

    def const(self, val, dtype=I64):
        if isinstance(val, (int, float)):
            arr, shape = [val], []
        else:
            arr = list(val)
            shape = [len(arr)]
        t = helper.make_tensor(self.uid("c"), dtype, shape, arr)
        return self.op("Constant", [], value=t)

    def init(self, name, t):
        return self.pool.add(name, t)


def vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


AUX_LAYERS = (1, 18, 35, 51, 68, 85)  # LiLiCorr dflash_config.target_layer_ids
MS = "com.microsoft"
BOLT = "bolt"


class Exporter:
    def __init__(self, args):
        self.a = args
        self.ck = GgufCheckpoint() if args.weights == "gguf" else Checkpoint()
        self.cfg = self.ck.cfg
        self.dev = "cuda"
        os.makedirs(args.out, exist_ok=True)
        self.pool = Pool(os.path.join(args.out, "model.onnx.data"))
        self.g = G(self.pool)
        self.layers_type = self.cfg["layers_block_type"]
        self.nl = args.num_layers if args.num_layers else len(self.layers_type)
        self.H = self.cfg["hidden_size"]
        self.eps = self.cfg["layer_norm_epsilon"]

    # ---------------------------------------------------------------- dense linears
    def linear(self, x, wname: str, N: int, K: int, kind: str, tag: str):
        """kind (checkpoint format): 'q8' | 'iq2' | 'nvfp4'; the exported format is selected by args."""
        g, a = self.g, self.a
        if kind == "q8":
            blk = self.ck.get(wname, self.dev)
            if a.q8 == "int8":
                q, d = q8_0_split(blk)
                B = g.init(tag + ".qw", q.reshape(N, K // 32, 32).cpu())
                S = g.init(tag + ".sc", d.to(torch.bfloat16).cpu())
                if a.fused:
                    return g.op("Q8Gemv", [x, B, S], domain=BOLT)
                return g.op("MatMulNBits", [x, B, S], domain=MS, K=K, N=N, bits=8, block_size=32)
            w = dq_q8_0(blk).to(torch.bfloat16)
        elif kind == "iq2":
            blk = self.ck.get(wname, self.dev)
            if a.iq2 == "native":
                rec, dd = iq2xxs_to_native(blk)
                B = g.init(tag + ".qw", rec.reshape(N, K // 32, 8).cpu())
                S = g.init(tag + ".sc", dd.reshape(N, K // 256).cpu())
                return g.op("Q4Gemv", [x, B, S], domain=BOLT, iq2=2)
            if a.iq2 in ("int4", "lut4"):
                p, s = (iq2xxs_to_lut4 if a.iq2 == "lut4" else iq2xxs_to_int4)(blk)
                B = g.init(tag + ".qw", p.reshape(N, K // 32, 16).cpu())
                S = g.init(tag + ".sc", s.to(torch.bfloat16).cpu())
                if a.fused:
                    return g.op("Q4Gemv", [x, B, S], domain=BOLT, **({"iq2": 1} if a.iq2 == "lut4" else {}))
                return g.op("MatMulNBits", [x, B, S], domain=MS, K=K, N=N, bits=4, block_size=32)
            w = dq_iq2xxs(blk).to(torch.bfloat16)
        elif kind == "nvfp4" and a.fused:
            base = wname[: -len(".weight")]
            wq = g.init(tag + ".fp4", self.ck.get(wname))
            ws = g.init(tag + ".fp4s", self.ck.get(base + ".weight_scale").view(torch.uint8))
            gs = g.init(tag + ".fp4g", self.ck.get(base + ".weight_scale_2").to(torch.float32).reshape(1))
            xin = g.op("Reshape", [x, g.const([-1, K])])
            y = g.op("Nvfp4Gemv", [xin, wq, ws, gs], domain=BOLT)
            return g.op("Reshape", [y, g.const([1, -1, N])])
        elif kind == "nvfp4":
            base = wname[: -len(".weight")]
            w = dq_nvfp4(
                self.ck.get(wname, self.dev),
                self.ck.get(base + ".weight_scale", self.dev),
                self.ck.get(base + ".weight_scale_2", self.dev),
            ).to(torch.bfloat16)
        else:
            raise ValueError(kind)
        assert w.shape == (N, K), (wname, w.shape, N, K)
        W = g.init(tag + ".w", w.t().contiguous().cpu())
        return g.op("MatMul", [x, W])

    # ---------------------------------------------------------------- blocks
    def rms(self, x, wname):
        w = self.g.init(wname, self.ck.get(wname).to(torch.bfloat16))
        return self.g.op("SimplifiedLayerNormalization", [x, w], epsilon=self.eps, axis=-1)

    def skip_rms(self, x, skip, wname):
        w = self.g.init(wname, self.ck.get(wname).to(torch.bfloat16))
        out = self.g.op("SkipSimplifiedLayerNormalization", [x, skip, w], nout=4, domain=MS, epsilon=self.eps)
        return out[0], out[3]

    def mamba(self, i, x):
        g, c, L = self.g, self.cfg, f"{PFX}layers.{i}.mixer."
        B_ = 1 if self.a.decode else "batch"
        heads, hd, ds, ng = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
        di = heads * hd
        conv_dim = di + 2 * ng * ds
        proj = self.linear(x, L + "in_proj.weight", di + conv_dim + heads, self.H, "nvfp4", f"l{i}.in_proj")
        if self.a.fused:
            return self.mamba_fused(i, proj, B_)
        gate, xbc, dt = g.op("Split", [proj, g.const([di, conv_dim, heads])], nout=3, axis=-1)
        xbc_t = g.op("Transpose", [xbc], perm=[0, 2, 1])
        cw = g.init(f"l{i}.conv_w", self.ck.get(L + "conv1d.weight").to(torch.bfloat16))
        cb = g.init(f"l{i}.conv_b", self.ck.get(L + "conv1d.bias").to(torch.bfloat16))
        pc = f"past_conv.{i}"
        g.inputs.append(vi(pc, BF16, [B_, conv_dim, c["conv_kernel"] - 1]))
        conv_out, new_conv = g.op("CausalConvWithState", [xbc_t, cw, cb, pc], nout=2, domain=MS, activation="silu")
        g.outputs.append(vi(f"present_conv.{i}", BF16, [B_, conv_dim, c["conv_kernel"] - 1]))
        g.nodes.append(helper.make_node("Identity", [new_conv], [f"present_conv.{i}"]))
        xbc_a = g.op("Transpose", [conv_out], perm=[0, 2, 1])
        xs, bm, cm = g.op("Split", [xbc_a, g.const([di, ng * ds, ng * ds])], nout=3, axis=-1)
        dtf = g.op("Cast", [dt], to=F32)
        dtb = g.init(f"l{i}.dt_bias", self.ck.get(L + "dt_bias").to(torch.float32))
        dts = g.op("Softplus", [g.op("Add", [dtf, dtb])])
        if self.a.dt_min > 0:
            dts = g.op("Max", [dts, g.const(float(self.a.dt_min), F32)])
        a_neg = g.init(f"l{i}.A_neg", -torch.exp(self.ck.get(L + "A_log").to(torch.float32)))
        decay = g.op("Mul", [dts, a_neg])
        xf = g.op("Cast", [xs], to=F32)
        x4 = g.op("Reshape", [xf, g.const([0, 0, heads, hd])])
        value = g.op("Reshape", [g.op("Mul", [x4, g.op("Unsqueeze", [dts, g.const([-1])])]), g.const([0, 0, di])])
        qf = g.op("Cast", [cm], to=F32)
        kf = g.op("Cast", [bm], to=F32)
        ps = f"past_ssm.{i}"
        g.inputs.append(vi(ps, F32, [B_, heads, ds, hd]))
        y, new_ssm = g.op(
            "LinearAttention",
            [qf, kf, value, ps, decay],
            nout=2,
            domain=MS,
            update_rule="gated",
            scale=1.0,
            q_num_heads=ng,
            kv_num_heads=heads,
        )
        g.outputs.append(vi(f"present_ssm.{i}", F32, [B_, heads, ds, hd]))
        g.nodes.append(helper.make_node("Identity", [new_ssm], [f"present_ssm.{i}"]))
        Dp = g.init(f"l{i}.D", self.ck.get(L + "D").to(torch.float32).reshape(1, 1, heads, 1))
        dskip = g.op("Reshape", [g.op("Mul", [x4, Dp]), g.const([0, 0, di])])
        y = g.op("Add", [y, dskip])
        gf = g.op("Cast", [gate], to=F32)
        yg = g.op("Mul", [y, g.op("Mul", [gf, g.op("Sigmoid", [gf])])])
        y4 = g.op("Reshape", [yg, g.const([0, 0, ng, di // ng])])
        if "ones_gn" not in self.pool.names:
            g.init("ones_gn", torch.ones(di // ng, dtype=torch.float32))
        yn = g.op("SimplifiedLayerNormalization", [y4, "ones_gn"], epsilon=float(c["rms_norm_eps"]), axis=-1)
        nw = g.init(f"l{i}.norm_w", self.ck.get(L + "norm.weight").to(torch.float32).reshape(1, 1, ng, di // ng))
        yn = g.op("Cast", [g.op("Reshape", [g.op("Mul", [yn, nw]), g.const([0, 0, di])])], to=BF16)
        return self.linear(yn, L + "out_proj.weight", self.H, di, "nvfp4", f"l{i}.out_proj")

    def mamba_fused(self, i, proj, B_):
        g, c, L = self.g, self.cfg, f"{PFX}layers.{i}.mixer."
        heads, hd, ds, ng = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
        di = heads * hd
        conv_dim = di + 2 * ng * ds
        p2 = g.op("Reshape", [proj, g.const([-1, di + conv_dim + heads])])
        cw = g.init(f"l{i}.conv_w", self.ck.get(L + "conv1d.weight").to(torch.bfloat16))
        cb = g.init(f"l{i}.conv_b", self.ck.get(L + "conv1d.bias").to(torch.bfloat16))
        pc, ps = f"past_conv.{i}", f"past_ssm.{i}"
        SR = self.a.verify_rows or B_
        g.inputs.append(vi(pc, BF16, [SR, conv_dim, c["conv_kernel"] - 1]))
        g.inputs.append(vi(ps, F32, [SR, heads, ds, hd]))
        dtb = g.init(f"l{i}.dt_bias", self.ck.get(L + "dt_bias").to(torch.float32))
        a_neg = g.init(f"l{i}.A_neg", -torch.exp(self.ck.get(L + "A_log").to(torch.float32)))
        Dp = g.init(f"l{i}.D", self.ck.get(L + "D").to(torch.float32))
        nw = g.init(f"l{i}.norm_w", self.ck.get(L + "norm.weight").to(torch.float32))
        y, nconv, nssm, _, _ = g.op(
            "MambaVerify" if self.a.verify_rows else "MambaStep",
            [p2, pc, ps, cw, cb, dtb, a_neg, Dp, nw] + (["state_row"] if self.a.verify_rows else []), nout=5, domain=BOLT,
            heads=heads, head_dim=hd, d_state=ds, n_groups=ng, epsilon=float(c["rms_norm_eps"]),
            time_step_min=float(self.a.dt_min),
        )
        g.outputs.append(vi(f"present_conv.{i}", BF16, [SR, conv_dim, c["conv_kernel"] - 1]))
        g.nodes.append(helper.make_node("Identity", [nconv], [f"present_conv.{i}"]))
        g.outputs.append(vi(f"present_ssm.{i}", F32, [SR, heads, ds, hd]))
        g.nodes.append(helper.make_node("Identity", [nssm], [f"present_ssm.{i}"]))
        yb = g.op("Reshape", [y, g.const([1, -1, di])])
        return self.linear(yb, L + "out_proj.weight", self.H, di, "nvfp4", f"l{i}.out_proj")

    def attention(self, i, x, seqlens_k, total_len):
        g, c, L = self.g, self.cfg, f"{PFX}layers.{i}.mixer."
        nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        q = self.linear(x, L + "q_proj.weight", nh * hd, self.H, "q8", f"l{i}.q")
        k = self.linear(x, L + "k_proj.weight", nkv * hd, self.H, "q8", f"l{i}.k")
        v = self.linear(x, L + "v_proj.weight", nkv * hd, self.H, "q8", f"l{i}.v")
        pk, pv = f"past_key.{i}", f"past_value.{i}"
        ml = self.a.max_len if self.a.decode else "past_len"
        g.inputs.append(vi(pk, BF16, [1 if self.a.decode else "batch", nkv, ml, hd]))
        g.inputs.append(vi(pv, BF16, [1 if self.a.decode else "batch", nkv, ml, hd]))
        o, nk, nv = g.op(
            "GroupQueryAttention",
            [q, k, v, pk, pv, seqlens_k, total_len],
            nout=3,
            domain=MS,
            num_heads=nh,
            kv_num_heads=nkv,
            do_rotary=0,
        )
        ml = self.a.max_len if self.a.decode else "total_len"
        g.outputs.append(vi(f"present_key.{i}", BF16, [1 if self.a.decode else "batch", nkv, ml, hd]))
        g.outputs.append(vi(f"present_value.{i}", BF16, [1 if self.a.decode else "batch", nkv, ml, hd]))
        g.nodes.append(helper.make_node("Identity", [nk], [f"present_key.{i}"]))
        g.nodes.append(helper.make_node("Identity", [nv], [f"present_value.{i}"]))
        return self.linear(o, L + "o_proj.weight", self.H, nh * hd, "q8", f"l{i}.o")

    def relu2(self, x):
        if self.a.fused:
            return self.g.op("Relu2", [x], domain=BOLT)
        r = self.g.op("Relu", [x])
        return self.g.op("Mul", [r, r])

    def moe(self, i, x):
        g, c, L = self.g, self.cfg, f"{PFX}layers.{i}.mixer."
        bc = c["block_configs"][i]
        E, topk = bc["n_routed_experts"], bc["num_experts_per_tok"]
        lat = c["moe_latent_size"]
        sh = c["moe_shared_expert_intermediate_size"]
        x2 = g.op("Reshape", [x, g.const([-1, self.H])])  # [T,H]
        if self.a.fused:
            gw = g.init(f"l{i}.gate_w", self.ck.get(L + "gate.weight").to(torch.bfloat16))
            gb = g.init(f"l{i}.gate_bias", self.ck.get(L + "gate.e_score_correction_bias").to(torch.float32))
            ridx, rw, _ = g.op("MoeRoute", [x2, gw, gb], nout=3, domain=BOLT, topk=topk,
                               scale=float(c["routed_scaling_factor"]))
        else:
            gw = g.init(f"l{i}.gate_w", self.ck.get(L + "gate.weight").to(torch.float32).t().contiguous())
            gb = g.init(f"l{i}.gate_bias", self.ck.get(L + "gate.e_score_correction_bias").to(torch.float32))
            logits = g.op("MatMul", [g.op("Cast", [x2], to=F32), gw])
            scores = g.op("Sigmoid", [logits])
            sel = g.op("Add", [scores, gb])
            _, idx = g.op("TopK", [sel, g.const([topk])], nout=2, axis=-1, largest=1, sorted=0)
            tw = g.op("GatherElements", [scores, idx], axis=1)
            tw = g.op("Div", [tw, g.op("Add", [g.op("ReduceSum", [tw, g.const([1])], keepdims=1), g.const(1e-20, F32)])])
            tw = g.op("Mul", [tw, g.const(float(c["routed_scaling_factor"]), F32)])
            zeros = g.op("ConstantOfShape", [g.op("Shape", [scores])], value=helper.make_tensor("z", F32, [1], [0.0]))
            ones = g.op("Expand", [g.const(1.0, F32), g.op("Shape", [idx])])
            probs = g.op("Cast", [g.op("ScatterElements", [zeros, idx, ones], axis=1)], to=BF16)
            wdense = g.op("Cast", [g.op("ScatterElements", [zeros, idx, tw], axis=1)], to=BF16)
        z = self.linear(x2, L + "fc1_latent_proj.weight", lat, self.H, "q8", f"l{i}.fc1_lat")
        up = self.ck.get_many([f"{L}experts.{e}.up_proj.weight" for e in range(E)], self.dev)
        conv = {"int8": iq2xxs_to_int8, "lut4": iq2xxs_to_lut4, "native": iq2xxs_to_native}.get(self.a.experts, iq2xxs_to_int4)
        p1, s1 = conv(up)
        del up
        dn = self.ck.get_many([f"{L}experts.{e}.down_proj.weight" for e in range(E)], self.dev)
        p2, s2 = conv(dn)
        del dn
        w1 = g.init(f"l{i}.fc1_w", p1.cpu())
        sc1 = g.init(f"l{i}.fc1_s", s1.to(torch.bfloat16).cpu())
        w2 = g.init(f"l{i}.fc2_w", p2.cpu())
        sc2 = g.init(f"l{i}.fc2_s", s2.to(torch.bfloat16).cpu())
        if self.a.fused:
            assert self.a.experts in ("int4", "lut4", "native")
            assert self.a.experts != "native" or self.a.verify_rows, "native IQ2 experts need the multi-row MoeVerify path (--verify-rows)"
            if self.a.verify_rows:  # experts shared across the rows are read once
                moe_out = g.op("MoeVerify", [z, ridx, rw, w1, sc1, w2, sc2], nout=4, domain=BOLT, topk=topk, iq2={"lut4": 1, "native": 2}.get(self.a.experts, 0))[0]
            else:
                moe_out, _, _ = g.op("MoeDecode", [z, ridx, rw, w1, sc1, w2, sc2], nout=3, domain=BOLT, topk=topk)
            out = self.linear(moe_out, L + "fc2_latent_proj.weight", self.H, lat, "q8", f"l{i}.fc2_lat")
            s_up = self.linear(x2, L + "shared_experts.up_proj.weight", sh, self.H, "iq2", f"l{i}.sh_up")
            s_dn = self.linear(self.relu2(s_up), L + "shared_experts.down_proj.weight", self.H, sh, "iq2", f"l{i}.sh_dn")
            out = g.op("Add", [out, s_dn])
            return g.op("Reshape", [out, g.const([1, -1, self.H]) if self.a.decode else g.op("Shape", [x])])
        ins = [z, probs, w1, sc1, "", w2, sc2, "", "", "", "", "", "", "", wdense]
        moe_out = g.op(
            "QMoE",
            ins,
            domain=MS,
            k=topk,
            activation_type="relu2",
            normalize_routing_weights=0,
            expert_weight_bits=8 if self.a.experts == "int8" else 4,
            block_size=32,
            weights_prepacked=0,
        )
        out = self.linear(moe_out, L + "fc2_latent_proj.weight", self.H, lat, "q8", f"l{i}.fc2_lat")
        s_up = self.linear(x2, L + "shared_experts.up_proj.weight", sh, self.H, "iq2", f"l{i}.sh_up")
        s_dn = self.linear(self.relu2(s_up), L + "shared_experts.down_proj.weight", self.H, sh, "iq2", f"l{i}.sh_dn")
        out = g.op("Add", [out, s_dn])
        return g.op("Reshape", [out, g.const([1, -1, self.H]) if self.a.decode else g.op("Shape", [x])])

    # ---------------------------------------------------------------- model
    def build(self):
        g, a = self.g, self.a
        t0 = time.time()
        dec = a.decode
        g.inputs += [
            vi("input_ids", I64, [1, a.verify_rows or 1] if dec else ["batch", "seq"]),
            vi("seqlens_k", I32, [1] if dec else ["batch"]),
            vi("total_sequence_length", I32, []),
        ]
        if a.verify_rows:
            g.inputs.append(vi("state_row", I32, [1]))
        if not dec:
            g.inputs.append(vi("logits_rows", I64, []))
        E = g.init("embed_tokens", self.ck.get(PFX + "embeddings.weight"))
        resid = g.op("Gather", [E, "input_ids"], axis=0)
        mixer_out = None
        aux = []
        for i in range(self.nl):
            t = self.layers_type[i]
            nw = f"{PFX}layers.{i}.norm.weight"
            if mixer_out is None:
                normed = self.rms(resid, nw)
            else:
                normed, resid = self.skip_rms(mixer_out, resid, nw)
            if a.aux and i in AUX_LAYERS:
                aux.append(resid)  # residual stream entering layer i (= vLLM aux_hidden_states[i])
            if t == "mamba":
                mixer_out = self.mamba(i, normed)
            elif t == "attention":
                mixer_out = self.attention(i, normed, "seqlens_k", "total_sequence_length")
            elif t == "moe":
                mixer_out = self.moe(i, normed)
            else:
                raise ValueError(t)
            if a.debug_layers and i in a.debug_layers:
                name = f"debug_mixer.{i}"
                g.outputs.append(vi(name, BF16, ["batch", "seq", self.H]))
                g.nodes.append(helper.make_node("Identity", [mixer_out], [name]))
            print(f"layer {i} {t} done {time.time()-t0:.0f}s pool={self.pool.off/2**30:.2f}GiB", flush=True)
        if a.aux:
            assert len(aux) == len(AUX_LAYERS), "--aux needs all layers"
            cat = g.op("Concat", aux, axis=-1)
            g.outputs.append(vi("aux_hidden_states", BF16, [1, a.verify_rows or 1, self.H * len(AUX_LAYERS)] if dec else ["batch", "seq", self.H * len(AUX_LAYERS)]))
            g.nodes.append(helper.make_node("Identity", [cat], ["aux_hidden_states"]))
        final, _ = self.skip_rms(mixer_out, resid, PFX + "norm_f.weight")
        if dec:
            rows = final
        else:
            S = g.op("Gather", [g.op("Shape", [final]), g.const(1)], axis=0)
            start = g.op("Sub", [S, "logits_rows"])
            rows = g.op("Slice", [final, g.op("Unsqueeze", [start, g.const([0])]), g.const([2**62]), g.const([1])])
        lm = self.linear(rows, "language_model.lm_head.weight", self.cfg["vocab_size"], self.H, "q8", "lm_head")
        g.outputs.insert(0, vi("logits", BF16, [1, a.verify_rows or 1, self.cfg["vocab_size"]] if dec else ["batch", "rows", self.cfg["vocab_size"]]))
        g.nodes.append(helper.make_node("Identity", [lm], ["logits"]))
        if a.verify_rows:
            g.outputs.append(vi("pred_ids", I64, [1, a.verify_rows]))
            lm32 = g.op("Cast", [lm], to=F32)  # the CUDA ArgMax has no bf16 kernel
            g.nodes.append(helper.make_node("ArgMax", [lm32], ["pred_ids"], axis=-1, keepdims=0))
        graph = helper.make_graph(g.nodes, "bolt", g.inputs, g.outputs, initializer=self.pool.protos)
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(MS, 1)],
            producer_name="bolt_export",
        )
        model.ir_version = 10
        self.pool.close()
        path = os.path.join(self.a.out, "model.onnx")
        onnx.save(model, path)
        print("saved", path, f"{time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-layers", type=int, default=0)
    ap.add_argument("--q8", choices=["bf16", "int8"], default="bf16")
    ap.add_argument("--iq2", choices=["bf16", "int4", "lut4", "native"], default="bf16", help="lut4 = exact IQ2_XXS as 4-bit codes (bolt iq2=1); native = exact, 2.06 bits (bolt iq2=2)")
    ap.add_argument("--aux", action="store_true", help="also output aux_hidden_states (residual stream entering layers 1,18,35,51,68,85, concatenated)")
    ap.add_argument("--fused", action="store_true", help="use bolt custom ops (decode graph)")
    ap.add_argument("--decode", action="store_true", help="static S=1 graph with fixed-size shared KV buffers (CUDA graph)")
    ap.add_argument("--verify-rows", type=int, default=0, help="static R-row verify graph (implies --decode --fused): per-row Mamba states, pred_ids output")
    ap.add_argument("--dt-min", type=float, default=0.0, help="lower clamp of the Mamba dt after softplus. 0 = none (llama.cpp / vLLM / the training kernels); "
                    "the HF slow path clamps at config.time_step_min=0.001, which degrades long generations")
    ap.add_argument("--weights", choices=["hf", "gguf"], default="hf", help="gguf: IQ2/NVFP4/Q8_0 blocks and the router from NVIDIA's GGUF (same formats)")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--experts", choices=["int4", "int8", "lut4", "native"], default="int4")
    ap.add_argument("--debug-layers", type=int, nargs="*", default=[])
    args = ap.parse_args()
    if args.verify_rows:
        args.decode = args.fused = True
    Exporter(args).build()


if __name__ == "__main__":
    main()

