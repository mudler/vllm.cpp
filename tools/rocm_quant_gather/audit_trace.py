#!/usr/bin/env python3
"""Audit matching production gather traces and each arm's untraced outputs.

Native allocation sizes are rocprofv3 events correlated with hipMalloc.
Primary tensor sizes are PyTorch allocator telemetry; its caching allocator
appears as a backing pool in the profiler. These are separate evidence types.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from primary import seal


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(directory: Path, suffix: str) -> tuple[Path, list[dict]]:
    paths = list(directory.glob(f"*_{suffix}.csv"))
    require(len(paths) == 1, f"expected one complete process trace for {suffix}")
    with paths[0].open() as file:
        return paths[0], list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("untraced_native", type=Path)
    parser.add_argument("untraced_primary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs = json.loads(args.manifest.read_text())["cases"]
    expected_outputs = 2 * len(inputs)
    summary = {"manifest": seal(args.manifest), "outputs_per_arm": expected_outputs,
               "scope": "same-byte operation dispatch and memory; no model or performance claim",
               "arms": {}, "files": {}}
    for arm in ("native", "primary"):
        directory = args.traces / arm
        report = json.loads((directory / "report.json").read_text())
        previous_dir = getattr(args, "untraced_" + arm)
        previous = json.loads((previous_dir / "report.json").read_text())
        previous_entries = {(c["name"], c["dtype"]): c for c in previous["cases"]}
        require(len(report["cases"]) == expected_outputs, "missing traced output cases")
        expected_keys = [(c["name"], dtype) for c in inputs for dtype in ("f32", "bf16")]
        require([(c["name"], c["dtype"]) for c in report["cases"]] == expected_keys,
                "trace case sequence differs from the common manifest")
        for entry in report["cases"]:
            prior = previous_entries[(entry["name"], entry["dtype"])]
            actual_path = directory / entry["result"]
            require(actual_path.read_bytes() == (previous_dir / prior["result"]).read_bytes(),
                    "profiling or I64 ID adaptation changed an output byte")
            require(entry["id_dtype"] == "i64" and entry["id_bytes"] == 32,
                    "both arms must consume the same four I64 IDs")
            summary["files"][str(actual_path)] = seal(actual_path)
        profiler = args.traces / (arm + "-profiler")
        kernel_file, kernels = read_csv(profiler, "kernel_trace")
        api_file, api = read_csv(profiler, "hip_api_trace")
        allocation_file, allocations = read_csv(profiler, "memory_allocation_trace")
        api_by_id = {r["Correlation_Id"]: r["Function"] for r in api}
        marker = "EmbeddingQuantGatherKernel<" if arm == "native" else "void dequantize_block"
        decode = sorted((r for r in kernels if marker in r["Kernel_Name"]),
                        key=lambda r: int(r["Start_Timestamp"]))
        require(len(decode) == expected_outputs, "each output must have exactly one GPU decoder")
        for entry, event in zip(report["cases"], decode):
            name = event["Kernel_Name"]
            require(event["Correlation_Id"] in api_by_id, "decoder lacks its HIP API event")
            if arm == "native":
                required_type = "<float, long," if entry["dtype"] == "f32" else "<unsigned short, long,"
            else:
                required_type = "float>" if entry["dtype"] == "f32" else "c10::BFloat16>"
            require(required_type in name, "executed decoder output or ID template dtype differs")
            if arm == "native":
                require("Dq" + entry["name"].rsplit("-", 1)[0] + ">" in name,
                        "native decoder format differs from the selected packed table")
        result = {"unchanged_output_bytes": expected_outputs, "gpu_decoders": len(decode),
                  "decoder_template_dtypes": {"f32": expected_outputs // 2, "bf16": expected_outputs // 2},
                  "kernel_histogram": dict(Counter(r["Kernel_Name"] for r in kernels))}
        if arm == "native":
            ordered_allocations = sorted((r for r in allocations
                                          if r["Operation"] == "MEMORY_ALLOCATION_ALLOCATE"
                                          and api_by_id.get(r["Correlation_Id"]) == "hipMalloc"),
                                         key=lambda r: int(r["Start_Timestamp"]))
            expected_sizes = []
            for case in inputs:
                table_bytes = (args.manifest.parent / case["packed"]).stat().st_size
                expected_sizes.extend([table_bytes, 32, 4 * case["width"] * 4, 16,
                                       4 * case["width"] * 2, 16])
            require([int(r["Allocation_Size"]) for r in ordered_allocations] == expected_sizes,
                    "native allocation sequence has missing or extra table/ID/output/scratch storage")
            result["profiler_hipmalloc_allocations"] = len(ordered_allocations)
            result["scratch_allocations_16_bytes"] = expected_outputs
            result["extra_device_allocations"] = 0
            result["allocation_evidence"] = "rocprofv3 sizes correlated with hipMalloc; runtime host allocations excluded"
        else:
            selected = [r for r in kernels if "indexSelectSmallIndex<unsigned char, long," in r["Kernel_Name"]]
            require(len(selected) == expected_outputs, "primary must select packed bytes on the GPU for every output")
            for entry in report["cases"]:
                require(entry["native_extension"] is True, "primary capture selected a non-native decoder")
                selected_rounded = (entry["selected_packed_bytes"] + 511) // 512 * 512
                require(entry["allocated_after"] - entry["allocated_before"] == entry["output_bytes"],
                        "primary live allocator delta differs from output size")
                require(entry["peak_allocated"] - entry["allocated_before"] == entry["output_bytes"] + selected_rounded,
                        "primary peak allocator delta differs from output plus selected packed rows")
            result["packed_gpu_selections"] = len(selected)
            result["tensor_memory_evidence"] = "PyTorch telemetry: output plus selected rows rounded to observed512-byte allocator units"
            result["profiler_allocation_scope"] = "backing allocator pool; does not measure individual PyTorch tensors"
        summary["arms"][arm] = result
        for file in (kernel_file, api_file, allocation_file, directory / "report.json"):
            summary["files"][str(file)] = seal(file)
    summary["pass"] = True
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: both {expected_outputs}-output traces preserve untraced bytes and execute native decoders; "
          "native allocation sequence is exact, primary allocator telemetry is separately verified")


if __name__ == "__main__":
    main()
