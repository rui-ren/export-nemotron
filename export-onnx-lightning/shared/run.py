"""Run greedy target decode, or teacher-forced MTP using an aligned target record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .runtime import Decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--custom-library", type=Path)
    parser.add_argument("--cuda-provider-library", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--token-ids", help="Comma-separated input IDs; no tokenizer dependency")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--record", type=Path, help="Write teacher-forced logits, hidden states and final caches")
    parser.add_argument("--mtp-target-record", type=Path, help="Target NPZ; pairs h_t with token x_(t+1)")
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be >= 0")
    if args.record and args.max_new_tokens:
        parser.error("Use --max-new-tokens 0 when recording comparable teacher-forced states")
    manifest = json.loads((args.export / "manifest.json").read_text())
    if manifest["variant"] == "mtp":
        if not args.mtp_target_record or args.max_new_tokens or args.token_ids:
            parser.error("MTP run requires --mtp-target-record and does not use --token-ids/--max-new-tokens")
        with np.load(args.mtp_target_record, allow_pickle=False) as record:
            tokens = record["token_ids"][1:].astype(np.int64).tolist()
            hidden = record["hidden_states"][:-1].copy()
        if not tokens or len(tokens) != len(hidden):
            parser.error("MTP record must contain at least two aligned target positions")
    else:
        if not args.token_ids or args.mtp_target_record:
            parser.error("Target run requires --token-ids and does not use --mtp-target-record")
        tokens = [int(value.strip()) for value in args.token_ids.split(",")]
        hidden = None
    if len(tokens) + args.max_new_tokens > manifest["max_length"]:
        parser.error("Requested tokens exceed the export's cache capacity")
    decoder = Decoder(args.export, args.custom_library, args.device, args.cuda_provider_library)
    logits, hidden_outputs = [], []
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    outputs = None
    for index, token in enumerate(tokens):
        target = torch.from_numpy(hidden[index]) if hidden is not None else None
        outputs = decoder.step(token, target)
        if args.record:
            logits.append(outputs["logits"].float().cpu().numpy())
            hidden_outputs.append(outputs["hidden_states"].float().cpu().numpy())
    torch.cuda.synchronize(args.device)
    prompt_seconds = time.perf_counter() - started
    generated = []
    started = time.perf_counter()
    for _ in range(args.max_new_tokens):
        token = int(outputs["logits"].argmax(-1).item())
        generated.append(token)
        outputs = decoder.step(token)
    torch.cuda.synchronize(args.device)
    decode_seconds = time.perf_counter() - started
    print(json.dumps({
        "teacher_forced_tokens": len(tokens), "teacher_forced_seconds": prompt_seconds,
        "generated_ids": generated, "decode_seconds": decode_seconds,
        "decode_tokens_per_second": len(generated) / decode_seconds if generated else None,
        "note": "Sequential prompt replay, not an optimized parallel prefill benchmark; recording adds transfers.",
    }, indent=2))
    if args.record:
        data = {"token_ids": np.asarray(tokens, np.int64), "logits": np.stack(logits),
                "hidden_states": np.stack(hidden_outputs)}
        for state in decoder.manifest["states"]:
            value = outputs[state["output"]]
            if state["output"].startswith(("present_key.", "present_value.")):
                value = value[:, :, :len(tokens)]
            data[state["output"]] = value.float().cpu().numpy()
        np.savez(args.record, **data)


if __name__ == "__main__":
    main()
