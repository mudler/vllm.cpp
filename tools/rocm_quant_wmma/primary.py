#!/usr/bin/env python3
"""Port plugin d4c1f0d tests/test_kernels.py::test_mmq for Q4_K and Q6_K.

Preserve every original tensor, tokens 7/83/128/2048, widths 256/1024,
F16/BF16/F32 inputs, seed zero, and the original dtype-specific tolerances.
The sole dtype adaptation narrows native F32 output to F16 after capture,
because the native MatmulBTQuant contract has no F16 output.
Run capture inside the task's pinned primary runtime under the GPU mutex.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

PIN = "d4c1f0d082fc7cd4350da56689109a01c1f29d6c"
ATOL = {"f16": 1, "bf16": 1.5, "f32": 1.2}
RTOL = {"f16": 1e-1, "bf16": 1e4, "f32": 2e1}


def seal(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def raw(tensor) -> bytes:
    import torch
    return tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def capture(fixture_manifest: Path, fixtures: Path, output: Path) -> None:
    import gguf
    import numpy as np
    import torch
    import vllm_gguf_plugin.ops as ops

    source = json.loads(fixture_manifest.read_text())
    expected_tensors = [entry for entry in source["cases"] if entry["type"] in (12, 14)]
    if len(expected_tensors) != 20:
        raise ValueError("the original Q4_K/Q6_K corpus must contain all 20 tensors")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "plugin_pin": PIN, "fixture_manifest": seal(fixture_manifest),
                "torch": torch.__version__, "torch_git": torch.version.git_version,
                "hip": torch.version.hip, "architecture": torch.cuda.get_device_properties(0).gcnArchName,
                "ops_source": seal(Path(ops.__file__)), "fixtures": {}, "cases": []}
    for quant_name in ("Q4_K", "Q6_K"):
        quant_type = gguf.GGMLQuantizationType[quant_name]
        for hidden in (256, 1024):
            filename = f"Quant_{quant_name}_{hidden}.gguf"
            fixture = fixtures / filename
            expected = source["fixtures"][filename]
            if seal(fixture) != expected:
                raise ValueError(f"original fixture hash or size differs: {filename}")
            manifest["fixtures"][filename] = expected
            tensors = gguf.GGUFReader(fixture).tensors
            selected = [entry for entry in expected_tensors if entry["source"] == filename]
            if [tensor.name for tensor in tensors] != [entry["tensor"] for entry in selected]:
                raise ValueError("original tensor names or ordering differs")
            for name, dtype in (("f16", torch.half), ("bf16", torch.bfloat16), ("f32", torch.float32)):
                for tokens in (7, 83, 128, 2048):
                    # Exact upstream seed_everything and rand device/dtype.
                    random.seed(0)
                    np.random.seed(0)
                    torch.manual_seed(0)
                    x = torch.rand((tokens, hidden), dtype=dtype, device="cuda")
                    activation_name = f"{quant_name}-{hidden}-{name}-{tokens}.activation"
                    (output / activation_name).write_bytes(raw(x))
                    for index, tensor in enumerate(tensors):
                        case_name = f"{quant_name}-{hidden}-{index}-{name}-{tokens}"
                        packed_name = f"{quant_name}-{hidden}-{index}.packed"
                        packed_path = output / packed_name
                        packed_path.write_bytes(tensor.data.tobytes())
                        with torch.inference_mode():
                            weight = torch.tensor(gguf.dequantize(tensor.data, quant_type), device="cuda").to(dtype)
                            reference = x @ weight.T
                            packed = torch.tensor(tensor.data, device="cuda")
                            actual = ops.ggml_mul_mat_a8(packed, x, int(quant_type), packed.shape[0]).to(dtype)
                            torch.testing.assert_close(actual, reference, atol=ATOL[name], rtol=RTOL[name])
                        reference_name = case_name + ".reference"
                        primary_name = case_name + ".primary"
                        (output / reference_name).write_bytes(raw(reference))
                        (output / primary_name).write_bytes(raw(actual))
                        manifest["cases"].append({"name": case_name, "type": int(quant_type),
                                                  "source": filename, "tensor": tensor.name,
                                                  "tokens": tokens, "rows": int(packed.shape[0]), "width": hidden,
                                                  "dtype": name, "seed": 0, "packed": packed_name,
                                                  "packed_seal": seal(packed_path), "activation": activation_name,
                                                  "activation_seal": seal(output / activation_name),
                                                  "reference": reference_name, "primary": primary_name,
                                                  "reference_seal": seal(output / reference_name),
                                                  "primary_seal": seal(output / primary_name),
                                                  "native_extension": ops._cuda_gemm_kernel_available("ggml_mul_mat_a8", int(quant_type)),
                                                  "atol": ATOL[name], "rtol": RTOL[name]})
                        print(f"PRIMARY {case_name} PASS", flush=True)
    if len(manifest["cases"]) != 240:
        raise ValueError("original coverage is not 240 tensor cases")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def compare(manifest_path: Path, native: Path) -> None:
    import torch

    manifest = json.loads(manifest_path.read_text())
    report = json.loads((native / "report.json").read_text())
    actual_cases = {entry["name"]: entry for entry in report["cases"]}
    if len(actual_cases) != len(report["cases"]) or set(actual_cases) != {entry["name"] for entry in manifest["cases"]}:
        raise ValueError("native case coverage differs from the complete primary manifest")
    dtypes = {"f16": torch.half, "bf16": torch.bfloat16, "f32": torch.float32}
    comparisons = []
    for case in manifest["cases"]:
        entry = actual_cases[case["name"]]
        for name in ("packed", "activation", "reference", "primary"):
            if seal(manifest_path.parent / case[name]) != case[name + "_seal"]:
                raise ValueError(f"changed {name} bytes: {case['name']}")
        dtype = dtypes[case["dtype"]]
        expected_native_dtype = "bf16" if case["dtype"] == "bf16" else "f32"
        if entry["dtype"] != expected_native_dtype:
            raise ValueError("native output dtype differs from the explicit adaptation")
        result_path = native / entry["result"]
        got = torch.frombuffer(bytearray(result_path.read_bytes()), dtype=dtypes[entry["dtype"]]).clone().to(dtype)
        if got.numel() != case["tokens"] * case["rows"]:
            raise ValueError("native output shape differs")
        for field in ("reference", "primary"):
            ref = torch.frombuffer(bytearray((manifest_path.parent / case[field]).read_bytes()), dtype=dtype).clone()
            torch.testing.assert_close(got, ref, atol=ATOL[case["dtype"]], rtol=RTOL[case["dtype"]])
        comparisons.append({"name": case["name"], "verdict": "PASS", "result": seal(result_path),
                            "atol": ATOL[case["dtype"]], "rtol": RTOL[case["dtype"]]})
        print(f"NATIVE PARITY {case['name']} PASS", flush=True)
    (native / "comparison.json").write_text(json.dumps({"verdict": "PASS", "cases": comparisons}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("capture")
    run.add_argument("fixture_manifest", type=Path)
    run.add_argument("fixtures", type=Path)
    run.add_argument("output", type=Path)
    check = commands.add_parser("compare")
    check.add_argument("manifest", type=Path)
    check.add_argument("native", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.fixture_manifest, args.fixtures, args.output)
    else:
        compare(args.manifest, args.native)


if __name__ == "__main__":
    main()
