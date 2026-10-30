"""Apply opt-in lifetime fixes to staged copies, never the historical sources."""

from __future__ import annotations

from pathlib import Path


def apply(stage: Path) -> None:
    directory = stage / "mamba/source"
    replacements = {
        "ssd_binary_source.cuh": [
            (
                '#include "state_cubin.h"',
                '#include "state_cubin.h"\n#include "safe_cuda_module.h"',
            ),
            (
                "CUmodule cumsum_module, state_module;",
                "SafeCudaModule cumsum_module, state_module;",
            ),
            (
                "thread_local std::vector<std::pair<CUcontext, SourceSsdBinary*>> modules;",
                "thread_local std::vector<std::pair<CUcontext, std::unique_ptr<SourceSsdBinary>>> modules;",
            ),
            (
                (
                    "auto* module = new SourceSsdBinary;\n"
                    "  modules.emplace_back(current, module);\n"
                    "  return *module;"
                ),
                (
                    "auto module = std::make_unique<SourceSsdBinary>();\n"
                    "  modules.emplace_back(current, std::move(module));\n"
                    "  return *modules.back().second;"
                ),
            ),
        ],
        "parallel_source_readout.cuh": [
            (
                "CUmodule cb_module, readout_module;",
                "SafeCudaModule cb_module, readout_module;",
            ),
            (
                "thread_local std::vector<std::pair<CUcontext,E6ParallelModules*>> entries;",
                "thread_local std::vector<std::pair<CUcontext,std::unique_ptr<E6ParallelModules>>> entries;",
            ),
            (
                (
                    "auto* value = new E6ParallelModules;\n"
                    "  entries.emplace_back(context,value);\n"
                    "  return *value;"
                ),
                (
                    "auto value = std::make_unique<E6ParallelModules>();\n"
                    "  entries.emplace_back(context,std::move(value));\n"
                    "  return *entries.back().second;"
                ),
            ),
        ],
    }
    for name, changes in replacements.items():
        path = directory / name
        text = path.read_text()
        for before, after in changes:
            if text.count(before) != 1:
                raise ValueError(f"Cleanup patch does not match historical source: {name}")
            text = text.replace(before, after)
        path.write_text(text)
    header = Path(__file__).with_name("safe_cuda_module.h")
    (directory / header.name).write_bytes(header.read_bytes())
