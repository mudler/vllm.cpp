#!/usr/bin/env python3
"""Capture the pinned plugin embedding method without downloading fixtures.

The embedding parameters come from plugin d4c1f0d tests/test_kernels.py:102-124.
The shared local ABI flattens the original 2-D IDs. This harness restores that
shape for the plugin and keeps the upstream atol=0.01, rtol=0.04 comparison.
IQ1_M and f16 output exceed the shared loader and Embedding contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def seal(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def export_upstream(directory: Path, output: Path, spec: Path) -> None:
    """Keep every tensor from each complete, hash-verified upstream fixture."""
    import gguf
    import numpy as np

    files = re.findall(r"\| `(Quant_[^`]+\.gguf)` \| (\d+) \| `([0-9a-f]{64})` \|", spec.read_text())
    if len(files) != 32:
        raise ValueError("the committed spec must name all 32 applicable upstream fixtures")
    # Verify the complete corpus before exporting any selected test input.
    for filename, size, digest in files:
        if seal(directory / filename) != {"bytes": int(size), "sha256": digest}:
            raise ValueError(f"upstream fixture hash or size mismatch: {filename}")
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    sources = {}
    for filename, _, _ in files:
        path = directory / filename
        sources[filename] = seal(path)
        hidden = int(path.stem.rsplit("_", 1)[1])
        for index, tensor in enumerate(gguf.GGUFReader(path).tensors):
            packed = np.asarray(tensor.data, dtype=np.uint8)
            rows = int(tensor.shape[1])
            if len(tensor.shape) != 2 or int(tensor.shape[0]) != hidden:
                raise ValueError(f"unexpected upstream tensor geometry: {filename}:{tensor.name}")
            name = f"{path.stem}-{index}"
            (output / f"{name}.packed").write_bytes(packed.tobytes())
            (output / f"{name}.ids").write_bytes(np.array([0, rows - 1, rows // 2, 0], dtype="<i4").tobytes())
            cases.append({"name": name, "type": int(tensor.tensor_type), "rows": rows,
                          "width": hidden, "packed": f"{name}.packed", "ids": f"{name}.ids",
                          "source": filename, "tensor": tensor.name})
    (output / "manifest.json").write_text(json.dumps({"version": 1, "source": "upstream-original",
                                                     "fixtures": sources, "cases": cases}, indent=2) + "\n")


def capture(manifest_path: Path, output: Path, selected: str | None) -> None:
    import gguf
    import numpy as np
    import torch
    import vllm_gguf_plugin.ops as ops
    from vllm_gguf_plugin.quantization.vocal_embeds import apply_gguf_embedding

    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    if selected is not None:
        cases = [case for case in cases if case["name"] == selected]
        if len(cases) != 1:
            raise ValueError("--case must name exactly one manifest entry")
    report = {"version": 1, "manifest": seal(manifest_path), "torch": torch.__version__,
              "torch_git": torch.version.git_version, "hip": torch.version.hip,
              "device": torch.cuda.get_device_name(0),
              "architecture": torch.cuda.get_device_properties(0).gcnArchName,
              "method": "vllm_gguf_plugin.quantization.vocal_embeds.apply_gguf_embedding",
              "plugin_pin": "d4c1f0d082fc7cd4350da56689109a01c1f29d6c",
              "cases": [], "exclusions": {"types": ["IQ1_M", "Q8_K", "MXFP4", "IQ1_XXXS"],
                                           "output_dtypes": ["f16"]}}
    try:
        for case in cases:
            if case["type"] in (15, 39, 66):
                continue  # These three codecs have the separately pinned secondary oracle.
            packed_path = manifest_path.parent / case["packed"]
            ids_path = manifest_path.parent / case["ids"]
            dtype = gguf.GGMLQuantizationType(case["type"])
            row_bytes = case["width"] // gguf.GGML_QUANT_SIZES[dtype][0] * gguf.GGML_QUANT_SIZES[dtype][1]
            packed = np.frombuffer(packed_path.read_bytes(), dtype=np.uint8).copy().reshape(case["rows"], row_bytes)
            host_ids = np.frombuffer(ids_path.read_bytes(), dtype="<i4").copy()
            if host_ids.size != 4:
                raise ValueError("the pinned upstream embedding case requires exactly four IDs")
            weight = torch.tensor(packed, device="cuda")
            ids = torch.tensor(host_ids.reshape(2, 2), device="cuda", dtype=torch.long)
            scalar = gguf.dequantize(packed, dtype)
            for name, output_dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
                with torch.inference_mode():
                    actual = apply_gguf_embedding(ids, weight, int(dtype), case["width"], dtype=output_dtype)
                    torch.cuda.synchronize()
                    # The original test's dense control remains independent of the plugin decoder.
                    dense = torch.tensor(scalar, device="cuda").to(output_dtype)
                    expected = torch.embedding(dense, ids)
                    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.04)
                raw = actual.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                result_path = output / f"{case['name']}-{name}.bin"
                result_path.write_bytes(raw)
                report["cases"].append({"name": case["name"], "type": int(dtype), "dtype": name,
                                        "packed": seal(packed_path), "ids": seal(ids_path),
                                        "native_extension": ops._cuda_kernel_available("ggml_dequantize", int(dtype)),
                                        "table_bytes": packed.nbytes, "selected_packed_bytes": 4 * row_bytes,
                                        "output_bytes": len(raw), "upstream_assert_close": "PASS",
                                        "result": result_path.name, "seal": seal(result_path)})
                print(f"PRIMARY {case['name']} {name} PASS", flush=True)
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-upstream")
    export.add_argument("directory", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--spec", type=Path, default=Path(__file__).resolve().parents[2] / ".agents/specs/rocm-quant-gather.md")
    run = sub.add_parser("capture")
    run.add_argument("manifest", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--case")
    args = parser.parse_args()
    if args.command == "export-upstream":
        export_upstream(args.directory, args.output, args.spec)
    else:
        capture(args.manifest, args.output, args.case)


if __name__ == "__main__":
    main()
