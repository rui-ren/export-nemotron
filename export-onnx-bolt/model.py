"""Connect embeddings, the layer stack, final normalization and logits."""
from __future__ import annotations

import time

from onnx import helper

from checkpoint import BACKBONE_PREFIX
from graph import BF16, F32, I64, I32, MS, BOLT, value_info
from layers import LayerBuilder

AUX_LAYERS = (1, 18, 35, 51, 68, 85)


def build_model(layers: LayerBuilder):
    graph, options = layers.graph, layers.options
    t0 = time.time()
    decode = options.decode
    graph.inputs += [
        value_info("input_ids", I64, [1, options.verify_rows or 1] if decode else ["batch", "seq"]),
        value_info("seqlens_k", I32, [1] if decode else ["batch"]),
        value_info("total_sequence_length", I32, []),
    ]
    if options.verify_rows:
        graph.inputs.append(value_info("state_row", I32, [1]))
    if not decode:
        graph.inputs.append(value_info("logits_rows", I64, []))
    E = graph.init("embed_tokens", layers.checkpoint.get(BACKBONE_PREFIX + "embeddings.weight"))
    residual = graph.op("Gather", [E, "input_ids"], axis=0)
    mixer_out = None
    aux = []
    for i in range(layers.num_layers):
        t = layers.layer_types[i]
        norm_weight = f"{BACKBONE_PREFIX}layers.{i}.norm.weight"
        if mixer_out is None:
            normalized = layers.rms(residual, norm_weight)
        else:
            normalized, residual = layers.skip_rms(mixer_out, residual, norm_weight)
        if options.aux and i in AUX_LAYERS:
            aux.append(residual)  # residual stream entering layer i (= vLLM aux_hidden_states[i])
        if t == "mamba":
            mixer_out = layers.mamba(i, normalized)
        elif t == "attention":
            mixer_out = layers.attention(i, normalized, "seqlens_k", "total_sequence_length")
        elif t == "moe":
            mixer_out = layers.moe(i, normalized)
        else:
            raise ValueError(t)
        if options.debug_layers and i in options.debug_layers:
            name = f"debug_mixer.{i}"
            graph.outputs.append(value_info(name, BF16, ["batch", "seq", layers.hidden_size]))
            graph.nodes.append(helper.make_node("Identity", [mixer_out], [name]))
        print(f"layer {i} {t} done {time.time()-t0:.0f}s pool={layers.store.offset/2**30:.2f}GiB", flush=True)
    if options.aux:
        assert len(aux) == len(AUX_LAYERS), "--aux needs all layers"
        cat = graph.op("Concat", aux, axis=-1)
        graph.outputs.append(value_info("aux_hidden_states", BF16, [1, options.verify_rows or 1, layers.hidden_size * len(AUX_LAYERS)] if decode else ["batch", "seq", layers.hidden_size * len(AUX_LAYERS)]))
        graph.nodes.append(helper.make_node("Identity", [cat], ["aux_hidden_states"]))
    final, _ = layers.skip_rms(mixer_out, residual, BACKBONE_PREFIX + "norm_f.weight")
    if decode:
        rows = final
    else:
        S = graph.op("Gather", [graph.op("Shape", [final]), graph.const(1)], axis=0)
        start = graph.op("Sub", [S, "logits_rows"])
        rows = graph.op("Slice", [final, graph.op("Unsqueeze", [start, graph.const([0])]), graph.const([2**62]), graph.const([1])])
    lm = layers.linear(rows, "language_model.lm_head.weight", layers.config["vocab_size"], layers.hidden_size, "q8", "lm_head")
    graph.outputs.insert(0, value_info("logits", BF16, [1, options.verify_rows or 1, layers.config["vocab_size"]] if decode else ["batch", "rows", layers.config["vocab_size"]]))
    graph.nodes.append(helper.make_node("Identity", [lm], ["logits"]))
    if options.verify_rows:
        graph.outputs.append(value_info("pred_ids", I64, [1, options.verify_rows]))
        lm32 = graph.op("Cast", [lm], to=F32)  # the CUDA ArgMax has no bf16 kernel
        graph.nodes.append(helper.make_node("ArgMax", [lm32], ["pred_ids"], axis=-1, keepdims=0))
    graph = helper.make_graph(graph.nodes, "bolt", graph.inputs, graph.outputs, initializer=layers.store.initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(MS, 1)]
        + ([helper.make_opsetid(BOLT, 1)] if options.fused else []),
        producer_name="bolt_export",
    )
    model.ir_version = 10
    return model
