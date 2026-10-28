"""Source-contract checks runnable without CUDA, PyTorch or a checkpoint."""
import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class KernelContractTests(unittest.TestCase):
    def test_mamba_gpu_fixtures_match_registered_output_types(self):
        source = (ROOT / "src" / "lightning" / "lightning_ops.cu").read_text(encoding="utf-8")
        adapter = source.split("struct MambaStepOp :", 1)[1].split("struct MoeRouteOp :", 1)[0]
        outputs = adapter.split("GetOutputType(size_t", 1)[1]
        match = re.search(r"types\[\]\s*=\s*\{([^}]+)\}", outputs)
        self.assertIsNotNone(match)
        registered = [item.strip() for item in match.group(1).split(",")]
        self.assertEqual(registered, ["BF", "BF", "F32", "BF", "F32"])
        fixture_path = ROOT / "export-onnx-lightning" / "tests" / "test_gpu_ops.py"
        tree = ast.parse(fixture_path.read_text(encoding="utf-8"))
        checked = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            sessions = [item for item in ast.walk(node) if isinstance(item, ast.Call)
                        and ast.unparse(item.func) == "self.session"
                        and item.args and isinstance(item.args[0], ast.Constant)
                        and item.args[0].value == "MambaStep"]
            if not sessions:
                continue
            assignment = next(item for item in node.body if isinstance(item, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == "shapes"
                                      for target in item.targets))
            names = [ast.unparse(item.elts[1]) for item in assignment.value.elts]
            mapped = [{"onnx.TensorProto.BFLOAT16": "BF", "FLOAT": "F32"}[name] for name in names]
            self.assertEqual(mapped, registered, node.name)
            checked += 1
        self.assertEqual(checked, 2)

    def test_mamba_rounds_readout_in_both_normalization_passes(self):
        source = (ROOT / "src" / "lightning" / "lightning_ops.cu").read_text(encoding="utf-8")
        norm = source.split("__global__ void MambaNormKernel(", 1)[1].split("struct MambaStepState", 1)[0]
        values = re.findall(r"const float v = ([^;]+);", norm)
        self.assertEqual(values, ["RoundBF(ypre[base + i]) * Silu(B2F(proj[base + i]))"] * 2)


if __name__ == "__main__":
    unittest.main()
