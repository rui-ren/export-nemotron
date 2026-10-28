"""Compare separate reference/ORT records, including final recurrent and KV states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare_records(reference, candidate, atol, rtol):
    if set(reference.files) != set(candidate.files):
        raise ValueError(f"Record fields differ: {set(reference.files) ^ set(candidate.files)}")
    if not np.array_equal(reference["token_ids"], candidate["token_ids"]):
        raise ValueError("Records must use identical teacher-forced token IDs")
    report, passed = {}, True
    for name in reference.files:
        if name == "token_ids":
            continue
        expected, actual = reference[name].astype(np.float64), candidate[name].astype(np.float64)
        if expected.shape != actual.shape:
            raise ValueError(f"{name}: shapes differ: {expected.shape} vs {actual.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError(f"{name}: non-finite output")
        delta = actual - expected
        close = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
        report[name] = {
            "max_absolute_error": float(np.max(np.abs(delta))),
            "relative_l2_error": float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-20)),
            "within_tolerance": close,
        }
        passed &= close
    expected, actual = reference["logits"], candidate["logits"]
    report["logits"]["top1_agreement"] = float(np.mean(expected.argmax(-1) == actual.argmax(-1)))
    # Token t's output predicts token t+1. This is a fixed-sequence quality proxy,
    # not a substitute for evaluation on a representative corpus.
    labels = reference["token_ids"][1:]
    if len(labels):
        for label, record in (("reference", reference), ("candidate", candidate)):
            scores = record["logits"][:-1].reshape(len(labels), -1).astype(np.float64)
            maximum = scores.max(-1)
            log_normalizer = maximum + np.log(np.exp(scores - maximum[:, None]).sum(-1))
            nll = log_normalizer - scores[np.arange(len(labels)), labels]
            report["logits"][label + "_mean_next_token_nll"] = float(nll.mean())
    return report, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--metrics-only", action="store_true",
                        help="Report quantization differences without asserting BF16-equivalence")
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("Tolerances must be nonnegative")
    with np.load(args.reference, allow_pickle=False) as reference:
        with np.load(args.candidate, allow_pickle=False) as candidate:
            report, passed = compare_records(reference, candidate, args.atol, args.rtol)
    print(json.dumps(report, indent=2))
    if not passed and not args.metrics_only:
        raise SystemExit("FAIL: at least one output/state exceeded the specified tolerances")


if __name__ == "__main__":
    main()
