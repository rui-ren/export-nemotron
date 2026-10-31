"""Bounded same-engine BF16/quantized checks with independent HF drift reporting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pynvml as nv

from export import digest


def compare(reference, candidate, tokens):
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite comparison logits")
    log_a = a - np.max(a, axis=-1, keepdims=True)
    log_a -= np.log(np.exp(log_a).sum(axis=-1, keepdims=True))
    log_b = b - np.max(b, axis=-1, keepdims=True)
    log_b -= np.log(np.exp(log_b).sum(axis=-1, keepdims=True))
    indices = np.arange(len(tokens) - 1)
    next_tokens = tokens[1:]
    nll_a = -log_a[indices, next_tokens].mean()
    nll_b = -log_b[indices, next_tokens].mean()
    return {
        "positions": len(tokens),
        "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).mean()),
        "mean_kl_reference_to_candidate": float((np.exp(log_a) * (log_a - log_b)).sum(-1).mean()),
        "relative_logit_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)),
        "max_absolute_logit_error": float(np.abs(a - b).max()),
        "reference_next_token_nll": float(nll_a), "candidate_next_token_nll": float(nll_b),
        "next_token_nll_delta": float(nll_b - nll_a),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite validation evidence")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        parser.error("CUDA_VISIBLE_DEVICES must select exactly --gpu-index")
    with np.load(args.reference) as reference:
        cases = []
        for key in reference.files:
            if key.endswith(".token_ids"):
                name = key.split(".")[0]
                tokens = reference[key].reshape(-1).astype(np.int32)
                scores = reference[name + ".prompt_logits"].reshape(-1, 131072).copy()
                cases.append((name, tokens, scores))
    if not cases:
        raise ValueError("No independent HF prompt captures")
    args.out.mkdir(parents=True)
    prompts = args.out / "prompts.tokens"
    prompts.write_text("".join(
        f"{len(tokens)} 32 " + " ".join(map(str, tokens)) + "\n" for _, tokens, _ in cases))
    nv.nvmlInit()
    gpu = nv.nvmlDeviceGetHandleByIndex(args.gpu_index)
    report = {
        "complete": False, "gpu_uuid": nv.nvmlDeviceGetUUID(gpu),
        "gpu_name": nv.nvmlDeviceGetName(gpu),
        "probe_sha256": digest(args.probe), "reference_sha256": digest(args.reference),
        "prompts_sha256": digest(prompts), "runs": {}, "comparisons": {},
        "limitation": "Bounded smoke/drift checks only, not a representative accuracy evaluation. "
                      "llama.cpp BF16 arithmetic is not certified identical to the qualified ONNX target.",
    }
    for label, model in (("bf16", args.baseline), ("q4km", args.model)):
        if (nv.nvmlDeviceGetComputeRunningProcesses(gpu)
                or nv.nvmlDeviceGetMemoryInfo(gpu).free < 70 * 2**30):
            raise RuntimeError("Probe requires an initially idle GPU with >=70GiB free")
        command = [str(args.probe.resolve()), str(model.resolve()), str(prompts.resolve()),
                   str((args.out / f"{label}.f32").resolve()),
                   str((args.out / f"{label}.generated").resolve())]
        peak, samples, owner = 0, 0, None
        start = time.monotonic()
        with (args.out / f"{label}.log").open("w") as log, (
                args.out / f"{label}.samples.jsonl").open("w") as monitor:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    processes = nv.nvmlDeviceGetComputeRunningProcesses(gpu)
                    memory = nv.nvmlDeviceGetMemoryInfo(gpu)
                    # NVML exposes host PIDs even when Popen uses container PIDs.
                    if len(processes) == 1 and owner is None:
                        owner = processes[0].pid
                    invalid = (len(processes) > 1
                               or any(p.pid != owner for p in processes)
                               or memory.free < 2 * 2**30)
                    allocated = processes[0].usedGpuMemory if len(processes) == 1 else 0
                    peak = max(peak, allocated)
                    monitor.write(json.dumps({"time": time.time(), "free_bytes": memory.free,
                                              "gpu_bytes": allocated, "guard_failure": invalid,
                                              "pids": [p.pid for p in processes]}) + "\n")
                    samples += 1
                    if invalid:
                        raise RuntimeError("GPU ownership/headroom guard failed")
                    if worker.poll() is not None:
                        break
                    time.sleep(0.1)
                if worker.returncode:
                    raise RuntimeError(f"{label} probe failed; see {log.name}")
            finally:
                if worker.poll() is None:
                    worker.terminate()
                    worker.wait(timeout=30)
        if not peak:
            raise RuntimeError("Probe never observed on GPU")
        report["runs"][label] = {
            "command": command, "worker_pid": worker.pid, "nvml_owner_pid": owner,
            "peak_process_gpu_gib": peak / 2**30, "guard_samples": samples,
            "elapsed_seconds_including_load": time.monotonic() - start,
            "scores_sha256": digest(args.out / f"{label}.f32"),
        }
        time.sleep(1)
    values = {}
    rows = sum(len(tokens) for _, tokens, _ in cases)
    for label in ("bf16", "q4km"):
        values[label] = np.fromfile(args.out / f"{label}.f32", dtype=np.float32).reshape(rows, 131072)
    offset = 0
    for name, tokens, hf in cases:
        stop = offset + len(tokens)
        a, b = values["bf16"][offset:stop], values["q4km"][offset:stop]
        report["comparisons"][name] = {
            "same_engine_bf16_vs_q4km": compare(a, b, tokens),
            "hf_vs_llama_bf16": compare(hf, a, tokens),
            "hf_vs_q4km": compare(hf, b, tokens),
        }
        offset = stop
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    report["repeat_checks"] = []
    for label in ("bf16", "q4km"):
        sequences = [[int(token) for token in row.split()]
                     for row in (args.out / f"{label}.generated").read_text().splitlines()]
        if len(sequences) != len(cases):
            raise RuntimeError("Generation output case count mismatch")
        report["runs"][label]["generated"] = [
            {"case": name, "ids": ids, "text": tokenizer.decode(ids)}
            for (name, _, _), ids in zip(cases, sequences)]
        seen, offset = {}, 0
        for (name, tokens, _), ids in zip(cases, sequences):
            scores = values[label][offset:offset + len(tokens)]
            key = tuple(tokens.tolist())
            if key in seen:
                previous_name, previous_scores, previous_ids = seen[key]
                if ids != previous_ids or not np.array_equal(scores, previous_scores):
                    raise RuntimeError(f"{label} state-reset repeat mismatch: {name}")
                report["repeat_checks"].append({
                    "model": label, "case": name, "matches_case": previous_name, "exact": True,
                })
            else:
                seen[key] = (name, scores, ids)
            offset += len(tokens)
    report["complete"] = True
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    nv.nvmlShutdown()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
