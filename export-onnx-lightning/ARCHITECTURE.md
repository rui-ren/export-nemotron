# Exporter code relationships

These are source-backed relationship diagrams, in the same source-file grouping
style as the requested example. Click an image to inspect its full resolution.

**Source commit:** [`f8ea83d57e8a0f77483805a7cb932e77277e0283`](https://github.com/rui-ren/export-nemotron/commit/f8ea83d57e8a0f77483805a7cb932e77277e0283).

Generated locally using Python AST, selected C++ symbols, registration bindings,
and CUDA launch sites, then rendered with Graphviz. No model weights were
downloaded or exported. Graph-generation scripts/data remain outside the
repository in the authoring session's artifact folder.

## 1. Lightning: separate entry points and shared exporter

![Lightning export code relationships](https://github.com/user-attachments/assets/5b172711-ebcb-4713-994e-c0a0b8c1e7cf)

Scope: four entry modules, shared exporter, checkpoint/storage classes, selected
layer methods, and quantization calls. Start at the left and follow the solid
call edges. Calls to `Graph.op` record a graph; they do not execute inference.

## 2. Lightning: custom operations and CUDA

![Lightning native kernel relationships](https://github.com/user-attachments/assets/0533d096-5c6e-4c75-a214-7ca6f4d86c30)

Scope: graph emission, ORT library registration, native operation adapters,
kernel-state classes and selected CUDA launches. Dotted orange edges mean an
exporter **emits an ONNX node**, not that Python directly calls the CUDA kernel.
`MoeDecodeOp<false>` binds BF16 decode; `<true>` binds W4 decode.
`ComputeV2` is represented inside its kernel-state class node.

## 3. Bolt: readable exporter

![Bolt export code relationships](https://github.com/user-attachments/assets/2e8e1e65-3cfb-4bf4-91f7-b28a98196ff4)

Scope: the new Bolt coordinator, layer builder, graph helpers, safetensors reader
and external-weight writer. Original Bolt CUDA files were relocated unchanged;
their internal kernels are not expanded in this graph.

## Legend and limitations

- Blue nodes: functions, selected methods or module entry points.
- Green nodes: classes or native operation adapters.
- Orange nodes: CUDA kernel functions. Purple nodes: external ORT APIs.
- Solid blue: implemented calls. Dashed green: construction.
- Dotted orange: graph-node emission. Dashed purple: native type binding,
  registration or ABI loading. Solid orange: CUDA launch.
- Enclosures group nodes by source module. `L` labels are source line numbers.

These are selected code relationships, **not complete call graphs or execution
traces**. Most small helpers, standard ONNX nodes, tests and reference tools
are omitted for readability. Conditional BF16/W4, layer-type and specialized
kernel paths are shown together, not as one unconditional execution.
The images do not establish CUDA compilation, full-model correctness, memory
fit, or performance. Those require validation on the Linux/H100 machine.

See the [usage and validation guide](README.md) for export and test commands.
