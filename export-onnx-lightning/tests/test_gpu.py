"""Run on the H100 machine after building the custom library.

LIGHTNING_OPS=/path/liblightning_ops.so python -m unittest discover -s tests -p test_gpu.py -v
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from shared.checkpoint import Checkpoint
from shared.export import main, parse_args
from fixtures import create_checkpoint
from shared.reference import TorchDecoder
from shared.runtime import Decoder


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("LIGHTNING_OPS"),
                     "Requires CUDA and LIGHTNING_OPS")
class GpuTests(unittest.TestCase):
    def test_all_variants_and_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            config = create_checkpoint(source, gpu=True)
            for variant in ("bf16", "custom", "int4", "mtp"):
                with self.subTest(variant=variant):
                    output = root / variant
                    arguments = ["--model-dir", str(source), "--out", str(output), "--variant", variant,
                                 "--max-length", "8", "--mtp-backend", "custom"]
                    options = parse_args(arguments)
                    with contextlib.redirect_stdout(io.StringIO()):
                        main(arguments)
                    decoder = Decoder(output, os.environ["LIGHTNING_OPS"],
                                      provider_library=os.environ.get("CUDA_PROVIDER_LIBRARY"))
                    reference = TorchDecoder(Checkpoint(source), options, "cuda")
                    generator = torch.Generator(device="cuda").manual_seed(19)
                    for token in (1, 3, 5, 2):
                        hidden = torch.randn((1, 1, config["hidden_size"]), device="cuda", generator=generator,
                                             dtype=torch.bfloat16) if variant == "mtp" else None
                        actual, expected = decoder.step(token, hidden), reference.step(token, hidden)
                        for name in expected:
                            torch.testing.assert_close(actual[name], expected[name], atol=0.025, rtol=0.04,
                                                       msg=f"{variant}: {name}")
                    del decoder
                    torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
