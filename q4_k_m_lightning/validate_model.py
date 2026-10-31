"""Guarded native ONNX GenAI smoke/reset and quantization-drift qualification."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from export import digest


def cases(path):
    with np.load(path) as reference:
        return [(key.split(".")[0], reference[key].reshape(-1).astype(np.int32),
                 reference[key.split(".")[0] + ".prompt_logits"].reshape(-1, 131072)[-1].copy())
                for key in reference.files if key.endswith(".token_ids")]


def logit_metrics(reference, candidate):
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite native ONNX logits")
    la, lb = a - a.max(), b - b.max()
    la -= np.log(np.exp(la).sum())
    lb -= np.log(np.exp(lb).sum())
    return {
        "top1_equal": bool(a.argmax() == b.argmax()),
        "reference_top1": int(a.argmax()), "candidate_top1": int(b.argmax()),
        "kl_reference_to_candidate": float((np.exp(la) * (la - lb)).sum()),
        "relative_logit_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)),
        "max_absolute_logit_error": float(np.abs(a - b).max()),
        "bitwise_equal": bool(np.array_equal(reference.view(np.uint32), candidate.view(np.uint32))),
    }


def worker(args):
    import onnxruntime_genai as og

    label = args.worker_label
    directory = args.baseline if label == "bf16" else args.model
    model = og.Model(str(directory))
    scores, generated, repeats, seen = {}, {}, [], {}
    for name, tokens, _ in cases(args.reference):
        if not 2 <= len(tokens) <= 512:
            raise ValueError("Native smoke captures must have 2..512 tokens")
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(tokens) + 32, do_sample=False)
        generator = og.Generator(model, params)
        generator.append_tokens(tokens.reshape(1, -1))
        logits = generator.get_output("logits").reshape(-1, 131072)[-1].astype(np.float32).copy()
        if not np.isfinite(logits).all():
            raise RuntimeError(f"Nonfinite {label} prefill logits")
        sequence = []
        for _ in range(32):
            if generator.is_done():
                break
            generator.generate_next_token()
            sequence.append(int(generator.get_next_tokens()[0]))
            if not np.isfinite(generator.get_output("logits")).all():
                raise RuntimeError(f"Nonfinite {label} decode logits")
        key = tuple(tokens.tolist())
        if key in seen:
            previous = seen[key]
            if sequence != generated[previous] or not np.array_equal(
                    logits.view(np.uint32), scores[previous].view(np.uint32)):
                raise RuntimeError(f"{label} state-reset repeat mismatch")
            repeats.append({"case": name, "matches_case": previous, "exact": True})
        else:
            seen[key] = name
        scores[name], generated[name] = logits, sequence
        del generator, params
        gc.collect()
    if not repeats:
        raise ValueError("Include a repeated prompt to qualify state reset")
    with (args.out / f"{label}.npz").open("xb") as output:
        np.savez(output, **scores)
    with (args.out / f"{label}.json").open("x") as output:
        json.dump({"complete": True, "generated_ids": generated, "repeat_checks": repeats}, output, indent=2)


def monitored_worker(args, label, handle, nv):
    if nv.nvmlDeviceGetComputeRunningProcesses(handle) or nv.nvmlDeviceGetMemoryInfo(handle).free < 70 * 2**30:
        raise RuntimeError("Requires an idle GPU with >=70 GiB free; no shared-GPU qualification")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-label", label]
    peak, samples, owner = 0, 0, None
    with (args.out / f"{label}.log").open("x") as log, (
            args.out / f"{label}.samples.jsonl").open("x") as monitor:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                running = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                memory = nv.nvmlDeviceGetMemoryInfo(handle)
                if len(running) == 1 and owner is None:
                    owner = running[0].pid
                invalid = (len(running) > 1 or any(p.pid != owner for p in running)
                           or memory.free < 2 * 2**30)
                allocated = running[0].usedGpuMemory if len(running) == 1 else 0
                peak = max(peak, allocated)
                monitor.write(json.dumps({"time": time.time(), "free_bytes": memory.free,
                    "gpu_bytes": allocated, "pids": [p.pid for p in running], "guard_failure": invalid}) + "\n")
                samples += 1
                if invalid:
                    raise RuntimeError("Native ONNX ownership/headroom guard failed")
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(f"{label} native ONNX worker failed; see {log.name}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if not peak:
        raise RuntimeError("No GPU execution observed")
    return {"guard_samples": samples, "peak_process_gpu_gib": peak / 2**30,
            "nvml_owner_pid": owner, "all_guard_samples_pass": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker-label", choices=("bf16", "q4km"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        parser.error("CUDA_VISIBLE_DEVICES must select exactly --gpu-index")
    if args.worker_label:
        worker(args)
        return
    if args.out.exists():
        parser.error("Refusing to overwrite native ONNX evidence")
    import pynvml as nv
    nv.nvmlInit()
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(args.gpu_index)
        report = {"complete": False, "runtime": "native ONNX Runtime GenAI", "runs": {},
                  "reference_sha256": digest(args.reference), "comparisons": {},
                  "limitation": "Short smoke/drift checks, not representative accuracy or performance."}
        args.out.mkdir(parents=True)
        for label, directory in (("bf16", args.baseline), ("q4km", args.model)):
            report["runs"][label] = monitored_worker(args, label, handle, nv)
            report["runs"][label].update({
                "graph_sha256": digest(directory / "model.onnx"),
                "library_sha256": digest(directory / "liblightning_ops.so")})
        with np.load(args.out / "bf16.npz") as a, np.load(args.out / "q4km.npz") as b:
            for name, _, hf in cases(args.reference):
                report["comparisons"][name] = {
                    "native_bf16_vs_quantized": logit_metrics(a[name], b[name]),
                    "hf_vs_native_bf16": logit_metrics(hf, a[name]),
                    "hf_vs_quantized": logit_metrics(hf, b[name]),
                }
        report["complete"] = True
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        nv.nvmlShutdown()


if __name__ == "__main__":
    main()
