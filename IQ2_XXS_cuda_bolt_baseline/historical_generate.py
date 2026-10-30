"""Generate text or resumable LiveCodeBench outputs through the native GenAI public API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def prompt_tokens(tokenizer, messages: list[dict], thinking: bool) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=False
    )
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def generate(og, model, prompt: list[int], max_new: int, capacity: int) -> dict:
    if not prompt:
        raise ValueError("Prompt must contain at least one token")
    if max_new < 1 or len(prompt) + max_new > capacity:
        raise ValueError(f"Prompt ({len(prompt)}) + output budget ({max_new}) exceeds capacity ({capacity})")
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, batch_size=1, max_length=len(prompt) + max_new)
    generator = og.Generator(model, params)
    start = time.perf_counter()
    generator.append_tokens(np.asarray(prompt, dtype=np.int32))
    generator.generate_next_token()
    generator.get_sequence(0)
    first = time.perf_counter()
    while not generator.is_done():
        generator.generate_next_token()
    sequence = generator.get_sequence(0).tolist()
    end = time.perf_counter()
    if sequence[:len(prompt)] != prompt:
        raise RuntimeError("GenAI changed the appended prompt sequence")
    tokens = sequence[len(prompt):]
    if not tokens or len(tokens) > max_new:
        raise RuntimeError(f"Invalid generated token count: {len(tokens)}")
    stats = generator.get_speculative_stats()
    result = {
        "tokens": tokens,
        "prompt_tokens": len(prompt),
        "generated_tokens": len(tokens),
        "prompt_sha256": token_hash(prompt),
        "generated_sha256": token_hash(tokens),
        "truncated": tokens[-1] not in (2, 11),
        "ttft_s": first - start,
        "decode_s": end - first,
        "tok_s": (len(tokens) - 1) / max(end - first, 1e-9),
        "speculative_stats": stats,
    }
    del generator
    return result


def evaluation_protocol(args: argparse.Namespace, og) -> dict:
    model = args.model
    return {
        "model": str(model.resolve()),
        "graph_sha256": file_hash(model / "model.onnx"),
        "drafter_sha256": file_hash(model / "dflash2.onnx"),
        "config_sha256": file_hash(model / "genai_config.json"),
        "tokenizer_sha256": file_hash(model / "tokenizer.json"),
        "chat_template_sha256": file_hash(model / "chat_template.jinja"),
        "runtime_sha256": file_hash(Path(og.__file__)),
        "native_runtime_sha256": file_hash(model / "runtime/lib/libonnxruntime-genai.so"),
        "custom_ops_sha256": file_hash(model / "libbolt_ops.so"),
        "cuda_plugin_sha256": file_hash(model / "runtime/lib/libonnxruntime_providers_cuda.so"),
        "samples_sha256": file_hash(args.samples),
        "driver_sha256": file_hash(Path(__file__)),
        "thinking": args.thinking,
        "max_new": args.max_new,
        "shard": args.shard,
        "shards": args.shards,
        "limit": args.limit,
    }


def prepare_resume(output: Path, protocol: dict) -> dict[int, dict]:
    protocol_path = output.with_suffix(".protocol.json")
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Evaluation protocol changed; use a new output path: {output}")
    elif output.exists():
        raise ValueError(f"Cannot resume results without a protocol manifest: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    completed = {}
    if output.exists():
        for line in output.read_text().splitlines():
            record = json.loads(line)
            index = record["index"]
            if index in completed:
                raise ValueError(f"Duplicate evaluation record: {index}")
            completed[index] = record
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt")
    mode.add_argument("--samples", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.limit < 0:
        parser.error("Require shards >= 1, 0 <= shard < shards, and limit >= 0")
    if args.samples is not None and args.out is None:
        parser.error("--samples requires --out")
    if args.max_new < 1:
        parser.error("--max-new must be positive")
    import onnxruntime_genai as og
    from transformers import AutoTokenizer

    bundled_runtime = (args.model / "runtime/lib").resolve()
    if Path(og.__file__).resolve().parent != bundled_runtime:
        raise RuntimeError(
            f"Wrong GenAI runtime loaded: {og.__file__}. Source {args.model / 'runtime-env.sh'} first."
        )
    og.register_execution_provider_library(
        "CUDAExecutionProvider", str(bundled_runtime / "libonnxruntime_providers_cuda.so")
    )
    config = json.loads((args.model / "genai_config.json").read_text())
    capacity = config["model"]["context_length"]
    model = og.Model(str(args.model))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if args.prompt is not None:
        ids = prompt_tokens(tokenizer, [{"role": "user", "content": args.prompt}], args.thinking)
        result = generate(og, model, ids, args.max_new, capacity)
        result["text"] = tokenizer.decode(result["tokens"], skip_special_tokens=False)
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(result["text"])
        print(json.dumps({key: value for key, value in result.items() if key not in {"tokens", "text"}}))
        return
    protocol = evaluation_protocol(args, og)
    completed = prepare_resume(args.out, protocol)
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    indices = [sample["index"] for sample in samples]
    if len(indices) != len(set(indices)):
        raise ValueError("Dataset contains duplicate sample indices")
    samples = [sample for sample in samples if sample["index"] % args.shards == args.shard]
    if args.limit:
        samples = samples[:args.limit]
    expected = {sample["index"] for sample in samples}
    if not completed.keys() <= expected:
        raise ValueError("Output contains records outside this shard's dataset")
    with args.out.open("a") as sink:
        for sample in samples:
            index = sample["index"]
            ids = prompt_tokens(tokenizer, sample["input"], args.thinking)
            if index in completed:
                if completed[index]["prompt_sha256"] != token_hash(ids):
                    raise ValueError(f"Resume prompt mismatch for sample {index}")
                continue
            record = generate(og, model, ids, args.max_new, capacity)
            record.update({
                "index": index,
                "question_id": sample["question_id"],
                "test_file": sample["test_file"],
                "text": tokenizer.decode(record["tokens"], skip_special_tokens=False),
            })
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            completed[index] = record
            print(json.dumps({key: record[key] for key in (
                "index", "generated_tokens", "truncated", "tok_s"
            )}), flush=True)
    print(json.dumps({"completed": len(completed), "expected": len(samples), "output": str(args.out)}))


if __name__ == "__main__":
    main()
