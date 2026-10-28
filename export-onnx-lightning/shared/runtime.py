"""CUDA IO binding for one-token graphs; no implicit weight downloading."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def run_cuda(session, feeds, device=0):
    """Bind torch tensors without NumPy copies; the session must share torch's stream."""
    import onnxruntime as ort

    binding = session.io_binding()
    keepalive = []
    for name, tensor in feeds.items():
        value = ort.OrtValue.from_dlpack(tensor.contiguous().__dlpack__())
        keepalive.append(value)
        binding.bind_ortvalue_input(name, value)
    for item in session.get_outputs():
        binding.bind_output(item.name, "cuda", device)
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return {
        item.name: torch.from_dlpack(value)
        for item, value in zip(session.get_outputs(), binding.get_outputs())
    }


class Decoder:
    def __init__(self, directory, library=None, device=0, provider_library=None, allow_partial=False):
        import onnxruntime as ort

        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 2:
            raise ValueError("Re-export this model: the draft cache/kernel contract is incompatible")
        if (self.manifest["custom_library_required"]
                and self.manifest.get("kernel_contract") != "bolt_style_lightning_v1"):
            raise ValueError("Unsupported custom kernel contract; re-export with the matching library")
        if self.manifest["debug_partial_backbone"] and not allow_partial:
            raise ValueError("This is a partial debug graph, not the complete model")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch is required for GPU IO binding")
        if self.manifest["custom_library_required"] and not library:
            raise ValueError("Pass --custom-library for this export")
        if self.manifest["custom_library_required"] and self.manifest["dtype"] != "bfloat16":
            raise ValueError("The CUDA custom kernels accept BF16; FP32 is for CPU diagnostics only")
        if not hasattr(ort.OrtValue, "from_dlpack"):
            raise RuntimeError("Use an ORT build exposing OrtValue.from_dlpack for BF16 IO binding")
        self.ort, self.device_id = ort, device
        self.device = torch.device("cuda", device)
        self.dtype = torch.bfloat16 if self.manifest["dtype"] == "bfloat16" else torch.float32
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        cuda_options = {"device_id": str(device),
                        "user_compute_stream": str(torch.cuda.current_stream(self.device).cuda_stream)}
        if library:
            options.register_custom_ops_library(str(library))
        if provider_library:
            ort.register_execution_provider_library("CUDAExecutionProvider", str(provider_library))
            devices = [d for d in ort.get_ep_devices()
                       if d.ep_name == "CUDAExecutionProvider" and d.ep_metadata.get("cuda_device_id") == str(device)]
            if not devices:
                raise RuntimeError(f"No CUDA provider device {device}")
            options.add_provider_for_devices(devices, cuda_options)
            self.session = ort.InferenceSession(str(self.directory / "model.onnx"), sess_options=options)
        else:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("This ONNX Runtime build has no CUDAExecutionProvider")
            self.session = ort.InferenceSession(
                str(self.directory / "model.onnx"), sess_options=options,
                providers=[("CUDAExecutionProvider", cuda_options)],
            )
        self.reset()

    def reset(self):
        self.length = 0
        self.states = {}
        for state in self.manifest["states"]:
            dtype = torch.float32 if state["dtype"] == 1 else self.dtype
            self.states[state["input"]] = torch.zeros(state["shape"], dtype=dtype, device=self.device)

    @torch.inference_mode()
    def step(self, token, target_hidden_states=None):
        if self.length >= self.manifest["max_length"]:
            raise ValueError("Cache capacity exceeded; export with a larger --max-length")
        if not 0 <= int(token) < self.manifest["config"]["vocab_size"]:
            raise ValueError("Token ID is outside the vocabulary")
        feeds = {
            "input_ids": torch.tensor([[token]], dtype=torch.int64, device=self.device),
            "seqlens_k": torch.tensor([self.length], dtype=torch.int32, device=self.device),
            "total_sequence_length": torch.tensor(self.length + 1, dtype=torch.int32),
            **self.states,
        }
        if self.manifest["variant"] == "mtp":
            if target_hidden_states is None:
                raise ValueError("MTP requires an aligned target hidden state")
            expected = (1, 1, self.manifest["config"]["hidden_size"])
            if tuple(target_hidden_states.shape) != expected:
                raise ValueError(f"MTP hidden state must have shape {expected}")
            feeds["target_hidden_states"] = target_hidden_states.to(self.device, self.dtype)
        outputs = run_cuda(self.session, feeds, self.device_id)
        for state in self.manifest["states"]:
            self.states[state["input"]] = outputs[state["output"]]
        self.length += 1
        return outputs
