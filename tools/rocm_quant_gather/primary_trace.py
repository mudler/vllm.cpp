#!/usr/bin/env python3
"""Trace only the pinned production gather; dense controls run in primary.py.

Capture both output dtypes on the same packed table and IDs as the native tool.
Allocator counters are PyTorch telemetry, separate from rocprofv3 trace evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from primary import seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    import gguf
    import numpy as np
    import torch
    import vllm_gguf_plugin.ops as ops
    from vllm_gguf_plugin.quantization.vocal_embeds import apply_gguf_embedding

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    report = {"manifest": seal(args.manifest), "torch": torch.__version__,
              "torch_git": torch.version.git_version, "hip": torch.version.hip,
              "plugin_pin": "d4c1f0d082fc7cd4350da56689109a01c1f29d6c",
              "method": "vllm_gguf_plugin.quantization.vocal_embeds.apply_gguf_embedding",
              "scope": "production method only; no full-table dense control in this process",
              "allocator_counters": "PyTorch telemetry, not profiler allocation events",
              "cases": []}
    for case in manifest["cases"]:
        quant_type = int(case["type"])
        if quant_type in (15, 39, 66):
            raise ValueError("the common trace manifest must contain only primary codecs")
        block_elements, block_bytes = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(quant_type)]
        row_bytes = case["width"] // block_elements * block_bytes
        packed_path = args.manifest.parent / case["packed"]
        ids_path = args.manifest.parent / case["ids"]
        packed = np.frombuffer(packed_path.read_bytes(), dtype=np.uint8).copy().reshape(case["rows"], row_bytes)
        host_ids = np.frombuffer(ids_path.read_bytes(), dtype="<i4").copy()
        if host_ids.size != 4:
            raise ValueError("the upstream shape adaptation requires four IDs")
        weight = torch.tensor(packed, device="cuda")
        ids = torch.tensor(host_ids.reshape(2, 2), device="cuda", dtype=torch.long)
        for name, output_dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            with torch.inference_mode():
                actual = apply_gguf_embedding(ids, weight, quant_type, case["width"], dtype=output_dtype)
                torch.cuda.synchronize()
            after = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            raw = actual.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            result_path = args.output / f"{case['name']}-{name}.bin"
            result_path.write_bytes(raw)
            report["cases"].append({"name": case["name"], "type": quant_type, "dtype": name,
                                    "packed": seal(packed_path), "ids": seal(ids_path),
                                    "table_bytes": packed.nbytes, "selected_packed_bytes": 4 * row_bytes,
                                    "id_dtype": "i64", "id_bytes": ids.numel() * ids.element_size(),
                                    "output_bytes": len(raw), "allocated_before": before,
                                    "allocated_after": after, "peak_allocated": peak,
                                    "native_extension": ops._cuda_kernel_available("ggml_dequantize", quant_type),
                                    "result": result_path.name, "seal": seal(result_path)})
            print(f"PRIMARY TRACE {case['name']} {name} output={len(raw)}", flush=True)
            del actual
        del weight, ids
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
