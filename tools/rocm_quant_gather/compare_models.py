#!/usr/bin/env python3
"""Compare the complete bounded model matrix and its physical cache contract.

Native captures enter through the public C ABI. Every primary case is a fresh
pinned engine. Secondary model captures name their bounded allocation overlay.
Q8_K remains a decoder-only secondary qualification and is never counted as a
secondary model execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import struct
import math

from primary import seal


FORMATS = ("Q4_0", "Q5_0", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
           "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S",
           "IQ2_S", "IQ4_XS", "MXFP4", "IQ1_XXXS")
SECONDARY = {"MXFP4": "stock", "IQ1_XXXS": "fork"}
PROMPTS = ([1, 0, 63, 127, 63], [1, 127, 0, 127])


def qualify_primary(report: dict, model: dict, config: dict,
                    prompt: list[int], repeat: int) -> None:
    """Qualify either capture against the sealed runtime and bounded workload."""
    if any(key in report for key in ("exception_type", "exception", "traceback")):
        raise ValueError("primary capture contains an exception")
    # Runtime identities come from the row's successful sealed Q4_0 captures.
    # These are local qualification requirements, not a global oracle pin change.
    expected = {
        "status": "EXECUTED; token comparison remains required",
        "primary_pin": "e126687a9a828d513c01a07cd69f025f27d63280",
        "plugin_pin": "d4c1f0d082fc7cd4350da56689109a01c1f29d6c",
        "vllm": "0.28.1rc1.dev132+ge126687a9",
        "plugin": "0.0.5+d4c1f0d.gfx1100",
        "torch": "2.12.0+git6bbd260",
        "torch_git": "6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5",
        "hip": "7.2.53211",
        "model": model, "config": config, "prompt": prompt, "repeat": repeat,
        "resolved_model_dtype": "torch.bfloat16",
        "requested": {
            "model_dtype": "auto from bfloat16 config", "kv_cache_dtype": "auto",
            "block_size": 16, "num_gpu_blocks_override": 16,
            "max_model_len": 64, "max_num_seqs": 1, "seed": 0x524F434D,
            "greedy": True, "ignore_eos": True, "max_tokens": 4,
            "stop": [], "stop_token_ids": []},
        "cache_capacity_contract": {
            "block_size": 16, "physical_blocks": 16, "physical_cells": 256,
            "reserved_null_blocks": 1, "usable_blocks": 15, "usable_cells": 240,
            "logical_max_model_len": 64, "bf16_kv_payload_bytes": 65536,
            "source": "vllm/v1/core/kv_cache_utils.py:2304-2309 at e126687a9a"},
    }
    for key, value in expected.items():
        # JSON comparison preserves the distinction between booleans and numbers,
        # including prompt IDs and nested sampling fields.
        if key not in report or json.dumps(report[key], sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"primary qualification field differs or is missing: {key}")


def tokens(path: Path) -> list[int]:
    result = [int(value) for value in path.read_text().split()]
    if len(result) != 4 or any(token < 0 or token >= 128 for token in result):
        raise ValueError(f"invalid bounded completion: {path}")
    return result


def logits(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) != 4 * 128 * 4:
        raise ValueError(f"incomplete logits: {path}")
    if not all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", data)):
        raise ValueError(f"nonfinite logits: {path}")
    return data


def primary_memory(report: dict) -> dict:
    result = {}
    for phase in ("memory_before_generation", "memory_after_generation"):
        workers = report[phase]
        if len(workers) != 1:
            raise ValueError("bounded primary must have exactly one worker")
        worker = workers[0]
        if worker["physical_blocks"] != 16 or worker["resolved_model_dtype"] != "torch.bfloat16":
            raise ValueError("primary physical block count or model dtype differs")
        caches = worker["cache_tensors"]
        if (worker["resolved_layout"] != "LHBNC" or
                worker["layout_stride_order"] != [0, 2, 1, 3, 4]):
            raise ValueError("primary executing cache layout differs from the pinned full-attention path")
        if len(caches) != 1:
            raise ValueError("bounded primary must expose one full-attention cache tensor")
        unique = {tensor["storage_address"]: tensor for tensor in caches}
        if not caches or sum(tensor["storage_bytes"] for tensor in unique.values()) != 65536:
            raise ValueError("primary physical cache storage must contain exactly 65536 bytes")
        if sum(tensor["payload_bytes"] for tensor in caches) != 65536:
            raise ValueError("primary cache views do not describe the complete physical payload")
        for tensor in caches:
            if tensor["dtype"] != "torch.bfloat16" or tensor["element_bytes"] != 2:
                raise ValueError("primary actual cache tensor is not BF16")
            if math.prod(tensor["shape"]) != tensor["numel"]:
                raise ValueError("primary cache shape and element count differ")
            if tensor["shape"] != [16, 2, 16, 64] or tensor["stride_elements"] != [1024, 16384, 64, 1]:
                raise ValueError("primary actual cache shape or stride differs")
            storage_offset = tensor["storage_offset_elements"]
            if type(storage_offset) is not int:
                raise ValueError("primary cache storage offset must be an integer")
            end_element = storage_offset + 1 + sum(
                (size - 1) * stride for size, stride in zip(tensor["shape"], tensor["stride_elements"]))
            if storage_offset < 0 or end_element * tensor["element_bytes"] > tensor["storage_bytes"]:
                raise ValueError("primary strided cache view is outside its storage")
            if storage_offset != 0:
                raise ValueError("primary cache view must start at its storage origin")
            allocation = tensor["allocator"]
            if allocation is None or allocation["state"] != "active_allocated":
                raise ValueError("primary cache storage has no live allocator block")
            if allocation["block_bytes"] < tensor["storage_bytes"]:
                raise ValueError("primary cache exceeds its allocator block")
            offset = tensor["storage_address"] - allocation["block_address"]
            if offset < 0 or offset + tensor["storage_bytes"] > allocation["block_bytes"]:
                raise ValueError("primary cache storage is outside its allocator block")
        result[phase] = worker
    before = result["memory_before_generation"]["cache_tensors"]
    after = result["memory_after_generation"]["cache_tensors"]
    for left, right in zip(before, after):
        for key in ("storage_address", "storage_bytes", "storage_offset_elements", "shape",
                    "stride_elements", "dtype", "element_bytes", "numel", "payload_bytes"):
            if left[key] != right[key]:
                raise ValueError("primary physical cache changed during generation")
    return result


def compare(args) -> dict:
    log = args.native_log.read_text()
    tests = re.findall(r"^(\d+): Test command: [^\n]*/test_capi_rocm_embedding_quant$", log, re.MULTILINE)
    if len(tests) != 1:
        raise ValueError("native log must identify exactly one public CTest execution")
    allocation_lines = re.findall(r"^" + tests[0] + r": \[kv-alloc\] ([^\n]+)", log, re.MULTILINE)
    if len(allocation_lines) != 228:
        raise ValueError("native public gate must capture all 228 engine cache allocations")
    for line in allocation_lines:
        values = dict(re.findall(r"(\w+)=(\w+)", line))
        expected = {"block_size": "16", "num_kv_heads": "1", "head_size": "64",
                    "dtype": "2", "page_size_bytes": "4096", "num_blocks": "16"}
        if any(values.get(key) != value for key, value in expected.items()):
            raise ValueError(f"native runtime cache differs from matched BF16 physical capacity: {line}")
    cases = []
    first = {}
    secondary_first = {}
    files = {str(args.native_log): seal(args.native_log)}
    for name in FORMATS:
        for repeat in range(3):
            for p, prompt in enumerate(PROMPTS):
                stem = f"{name}-r{repeat}-p{p}"
                captures = {}
                for arm in ("quant", "dense"):
                    prefix = args.native / (stem + "-" + arm)
                    ids_path = Path(str(prefix) + "-tokens.txt")
                    logits_path = Path(str(prefix) + "-logits-f32.bin")
                    captures[arm] = (tokens(ids_path), logits(logits_path))
                    files[str(ids_path)] = seal(ids_path)
                    files[str(logits_path)] = seal(logits_path)
                if captures["quant"] != captures["dense"]:
                    raise ValueError(f"native quantized completion differs from dense control: {stem}")
                key = (name, p)
                if repeat == 0:
                    first[key] = captures["quant"]
                elif first[key] != captures["quant"]:
                    raise ValueError(f"native repeated engine differs: {stem}")
                actual = captures["quant"][0]
                case = {"format": name, "repeat": repeat, "prompt_index": p,
                        "prompt": prompt, "native_tokens": actual, "native_dense_exact": True,
                        "native_logits_sha256": files[str(args.native / (stem + "-quant-logits-f32.bin"))]["sha256"]}
                if name == "Q8_K":
                    case["oracle"] = "decoder-only stock adaptation; no stock GET_ROWS or model claim"
                elif name in SECONDARY:
                    prefix = args.secondary / (stem + "-oracle")
                    path = Path(str(prefix) + "-tokens.txt")
                    expected_ids = tokens(path)
                    scores_path = Path(str(prefix) + "-logits-f32.bin")
                    oracle_scores = logits(scores_path)
                    if repeat == 0:
                        secondary_first[key] = (expected_ids, oracle_scores)
                    elif secondary_first[key] != (expected_ids, oracle_scores):
                        raise ValueError(f"secondary repeated engine differs: {stem}")
                    log_path = Path(str(prefix) + ".log")
                    if "physical_context=256" not in log_path.read_text():
                        raise ValueError(f"secondary physical capacity is unmeasured: {stem}")
                    files[str(path)] = seal(path)
                    files[str(scores_path)] = seal(scores_path)
                    files[str(log_path)] = seal(log_path)
                    case.update(oracle=SECONDARY[name] + " pin plus bounded retain-hybrid-copy overlay",
                                oracle_tokens=expected_ids)
                    if actual != expected_ids:
                        raise ValueError(f"native {actual} differs from secondary {expected_ids}: {stem}")
                else:
                    path = args.primary / f"{name}-p{p}-r{repeat}.json"
                    report = json.loads(path.read_text())
                    files[str(path)] = seal(path)
                    qualify_primary(report, seal(args.fixtures / (name + ".gguf")),
                                    seal(args.config / "config.json"), prompt, repeat)
                    case.update(oracle="primary pinned vLLM and GGUF plugin", oracle_tokens=report["tokens"])
                    if actual != report["tokens"]:
                        raise ValueError(f"native {actual} differs from primary {report['tokens']}: {stem}")
                cases.append(case)
    recurrent = []
    for oracle in ("stock", "fork"):
        for p in range(2):
            baseline = None
            for repeat in range(3):
                for arm in ("pristine", "overlay"):
                    prefix = args.secondary / "controls" / f"{oracle}-{arm}-r{repeat}-p{p}"
                    ids_path = Path(str(prefix) + "-tokens.txt")
                    scores_path = Path(str(prefix) + "-logits-f32.bin")
                    actual = (tokens(ids_path), logits(scores_path))
                    if baseline is None:
                        baseline = actual
                    elif actual != baseline:
                        raise ValueError(f"recurrent model changed across repeats or overlay: {prefix}")
                    files[str(ids_path)] = seal(ids_path)
                    files[str(scores_path)] = seal(scores_path)
                    recurrent.append({"oracle": oracle, "arm": arm, "repeat": repeat,
                                      "prompt_index": p, "tokens": actual[0],
                                      "logits_sha256": files[str(scores_path)]["sha256"]})
    # Token qualification and memory instrumentation are independent executions.
    # Both gates remain required. The memory capture uses the identical Q4_0
    # workload and must preserve its completion, model, and configuration bytes.
    memory_report = json.loads(args.primary_memory.read_text())
    qualify_primary(memory_report, seal(args.fixtures / "Q4_0.gguf"),
                    seal(args.config / "config.json"), PROMPTS[0], 0)
    if memory_report["tokens"] != first[("Q4_0", 0)][0]:
        raise ValueError("primary memory capture changed the qualified Q4_0 workload")
    measured_memory = primary_memory(memory_report)
    files[str(args.primary_memory)] = seal(args.primary_memory)
    return {"status": "PASS", "comparison": "all native quantized and dense completions, followed by identical primary or named secondary model tokens",
            "native_engines": 228, "primary_engines": 96, "secondary_overlay_engines": 12,
            "q8_k_decoder_only_native_engines": 12, "physical_kv_payload_bytes": 65536,
            "logical_native_primary_limit": 64, "primary_reserved_null_bytes": 4096,
            "native_reserved_null_bytes": 4096, "primary_actual_memory": measured_memory,
            "identical_tool_backing_allocations": "separate model trace audit is required",
            "recurrent_pristine_overlay_engines": recurrent,
            "cases": cases, "sealed_outputs": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "native-log", "primary", "primary-memory", "secondary", "fixtures", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args)
    except (ValueError, KeyError, FileNotFoundError) as error:
        args.output.write_text(json.dumps({"status": "FAIL", "error": str(error)}, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("cases", "sealed_outputs")}))


if __name__ == "__main__":
    main()
