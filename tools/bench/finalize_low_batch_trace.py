#!/usr/bin/env python3
"""Fail-closed finalizer for the exact batch-2 paired execution trace.

The GPU driver intentionally stops after writing immutable raw artifacts.  This
CPU-only step revalidates the complete capture chain, extracts only steady
batch-2 vLLM decode windows, classifies bounded batch-1 drain windows, and
writes the completion marker last.  Cross-profiler kernel durations are kept
diagnostic: they select source-inspection targets but never become a binding
throughput result.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

if __package__ in (None, ""):
    script = pathlib.Path(__file__).resolve()
    if len(script.parents) > 2:
        sys.path.insert(0, str(script.parents[2]))

from tools.bench.online_gate import (
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    TRACE_CAPTURE_GRAPH_REPLAYS,
    TRACE_GDN_PACKED_COUPLED_BA_NODE_COUNT,
    TRACE_PRIMARY_GRAPH_CONTRACTS_BY_BATCH,
    TRACE_REPETITIONS,
    VLLM_DECODE_FAMILY_CONTRACTS,
    _load_json_object,
    record_trace_status,
)
from tools.bench.serve_low_common import (
    HarnessError,
    canonical_json,
    sha256_file,
    write_json_atomic,
)


MODEL_KEY = "27"
TRACE_BATCH = 2
TRACE_PROMPTS = 6
TRACE_RUNS = TRACE_REPETITIONS + 1
EXPECTED_PURE_WINDOWS = (
    TRACE_RUNS * (TRACE_PROMPTS // TRACE_BATCH) * (128 - 1)
)
EXPECTED_BASE_GENERATION_WINDOWS = TRACE_RUNS * (TRACE_PROMPTS // TRACE_BATCH) * 128
MAX_DRAIN_WINDOWS = TRACE_RUNS
EXPECTED_B2_KERNELS = 1_160
EXPECTED_B2_ORDERED_NAMES_SHA256 = (
    "858915dd7dd7a3dcd5b91d7ae9739a6e522ad36ea1364a92c584511faa3bfad0"
)
ALLOWED_B2_SIGNATURES = {
    # Accepted 3812d8 async-ON denominator.
    "b5c6fcacded1fc761f0e20a37935dfc6992cabc8526a3be9d862396d9c90dd7b": (
        "accepted-3812d8"
    ),
    # Fresh 179a0fc trace: only five generated RMSNorm partitions move from
    # 48 to 50 registers; ordered names, grids, blocks and shared memory match.
    "8de0b7fef06d88d9ea023d9ddb55b172ca92b3ae27b3a351bb14536e904da894": (
        "generated-rmsnorm-registers-50"
    ),
    # Immutable 0091cd1 GDN-BA merged oracle control.  All 1,522 steady
    # windows retain the accepted names, blocks, grids and shared memory;
    # only cached Torch/Inductor RMSNorm partitions redistribute registers:
    # 1x26, 48x28, 64x40, 1x44, 19x48 and 44x50.
    "17e1037ec3918420faf2ed4a004e10c9b0c1165504a67939e784578f3d1c14ed": (
        "gdn-ba-merged-rmsnorm-register-allocation"
    ),
    # The paired 0091cd1 split oracle control has the same exact structural
    # contract across all 1,521 steady windows, with 35 generated RMSNorm
    # launches at 48 registers and 28 at 50; its complete distribution is
    # 1x26, 48x28, 64x40, 1x44, 35x48 and 28x50.
    "f7a3ca1f31d7d5dd672fc3def715cbd025b771d552601a8521fb3768b3adcadf": (
        "gdn-ba-split-rmsnorm-register-allocation"
    ),
}
EXPECTED_B2_FP4_TACTICS = {"static_persistent": 80, "stream_k": 128}
EXPECTED_B2_GENERATED_RMSNORM_QUANT = 177
SOURCE_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
ANY_GENERATION_RE = re.compile(
    r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)"
)
PURE_GENERATION_RE = re.compile(
    r"execute_context_0\(0\)_generation_(\d+)\((\d+)\)"
)
DERIVED_PATHS = frozenset(
    {
        "trace/27/c2-manifest.json",
        "trace/27/c2-summary.json",
        "trace/27/status-c2.json",
    }
)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(value < 0.0 for value in values):
        raise HarnessError("timing statistics require non-negative samples")
    mean = statistics.mean(values)
    ordered = sorted(values)
    return {
        "count": len(values),
        "coefficient_of_variation": (
            statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0
        ),
        "max": ordered[-1],
        "mean": mean,
        "median": statistics.median(ordered),
        "min": ordered[0],
        "p05": ordered[int(0.05 * (len(ordered) - 1))],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
    }


def _family_for(name: str, contract: Mapping[str, tuple[str, int]]) -> str | None:
    matches = [
        family
        for family, (needle, _) in contract.items()
        if needle in name
        and not (
            family == "normal_fp4_producer"
            and "silu_mul_cvt_fp16_to_fp4" in name
        )
    ]
    if len(matches) > 1:
        raise HarnessError(f"kernel matches multiple low-batch families: {name}")
    return matches[0] if matches else None


def _generated_rmsnorm_quant(name: str) -> bool:
    lowered = name.lower()
    return "rms_norm" in lowered or (
        "mean_mul_pow_rsqrt_scaled_fp4_quant" in lowered
    )


def _local_rmsnorm(name: str) -> bool:
    lowered = name.lower()
    return "rmsnormrowkernel" in lowered or "rmsnormgatedrowkernel" in lowered


def _fp4_tactic(name: str) -> str | None:
    if "MainloopSm120TmaWarpSpecializedBlockScaled" not in name:
        return None
    if "StreamKScheduler" in name:
        return "stream_k"
    if "StaticPersistentScheduler" in name:
        return "static_persistent"
    return "other"


def _selected_family(name: str, broad_family: str | None, *, local: bool) -> str:
    if broad_family == "fp4_gemm":
        return "fp4_gemm"
    if "cutlass_80_" in name and "gemm_bf16" in name:
        return "bf16_cutlass_gemm"
    if (_local_rmsnorm(name) if local else _generated_rmsnorm_quant(name)):
        return "rmsnorm_generated_partition"
    if broad_family is not None:
        return broad_family
    return "other"


def _launch_signature(event: Mapping[str, Any], path: pathlib.Path) -> dict[str, Any]:
    args = event.get("args")
    if not isinstance(args, dict):
        raise HarnessError(f"{path}: steady decode kernel lacks launch metadata")
    signature = {
        "block": args.get("block"),
        "grid": args.get("grid"),
        "name": event.get("name"),
        "registers_per_thread": args.get("registers per thread"),
        "shared_memory": args.get("shared memory"),
    }
    for geometry in ("block", "grid"):
        value = signature[geometry]
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise HarnessError(f"{path}: steady decode {geometry} is incomplete")
    for resource in ("registers_per_thread", "shared_memory"):
        value = signature[resource]
        if isinstance(value, bool) or not isinstance(value, int):
            raise HarnessError(f"{path}: steady decode {resource} is incomplete")
    return signature


def summarize_vllm_c2_trace(
    path: pathlib.Path,
    *,
    expected_pure_windows: int = EXPECTED_PURE_WINDOWS,
    expected_base_generation_windows: int = EXPECTED_BASE_GENERATION_WINDOWS,
    max_drain_windows: int = MAX_DRAIN_WINDOWS,
    expected_kernel_count: int = EXPECTED_B2_KERNELS,
    expected_names_sha256: str = EXPECTED_B2_ORDERED_NAMES_SHA256,
    allowed_signature_sha256: Mapping[str, str] = ALLOWED_B2_SIGNATURES,
    family_contract: Mapping[str, tuple[str, int]] | None = None,
    expected_fp4_tactics: Mapping[str, int] = EXPECTED_B2_FP4_TACTICS,
    expected_generated_rmsnorm_quant: int = EXPECTED_B2_GENERATED_RMSNORM_QUANT,
) -> dict[str, Any]:
    """Extract exact steady B=2 windows while retaining drain diagnostics."""

    if family_contract is None:
        family_contract = VLLM_DECODE_FAMILY_CONTRACTS[MODEL_KEY]
    opener = gzip.open if path.suffix == ".gz" else pathlib.Path.open
    try:
        if path.suffix == ".gz":
            with opener(path, "rt", encoding="utf-8") as source:
                value = json.load(source)
        else:
            with opener(path, "r", encoding="utf-8") as source:
                value = json.load(source)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"{path}: invalid vLLM Torch trace: {error}") from error
    events = value.get("traceEvents") if isinstance(value, dict) else None
    if not isinstance(events, list):
        raise HarnessError(f"{path}: traceEvents array is absent")

    generation_window_count = 0
    pure_windows: list[tuple[float, float, int, str]] = []
    kernels: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        category = str(event.get("cat", "")).lower()
        name = event.get("name")
        if category == "gpu_user_annotation" and isinstance(name, str):
            if ANY_GENERATION_RE.fullmatch(name):
                generation_window_count += 1
            match = PURE_GENERATION_RE.fullmatch(name)
            if match is None:
                continue
            batch, repeated = (int(item) for item in match.groups())
            if batch != repeated or batch not in (1, TRACE_BATCH):
                raise HarnessError(f"{path}: invalid pure-decode batch {batch}/{repeated}")
            start = event.get("ts")
            duration = event.get("dur")
            if (
                event.get("ph") != "X"
                or isinstance(start, bool)
                or not isinstance(start, (int, float))
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or start < 0.0
                or duration <= 0.0
            ):
                raise HarnessError(f"{path}: pure-decode annotation is incomplete")
            pure_windows.append((float(start), float(start + duration), batch, name))
            continue
        if "kernel" not in category or not isinstance(name, str) or not name:
            continue
        start = event.get("ts")
        duration = event.get("dur")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or start < 0.0
            or duration < 0.0
        ):
            raise HarnessError(f"{path}: kernel timing is invalid")
        kernels.append(
            {
                "args": event.get("args"),
                "duration_us": float(duration),
                "end": float(start + duration),
                "name": name,
                "ph": event.get("ph"),
                "start": float(start),
            }
        )

    pure_windows.sort()
    if len(pure_windows) != expected_pure_windows:
        raise HarnessError(
            f"{path}: found {len(pure_windows)} pure windows; "
            f"expected {expected_pure_windows}"
        )
    for previous, current in zip(pure_windows, pure_windows[1:]):
        if current[0] < previous[1]:
            raise HarnessError(f"{path}: pure-decode windows overlap")
    drain_count = sum(batch == 1 for _, _, batch, _ in pure_windows)
    if drain_count > max_drain_windows:
        raise HarnessError(
            f"{path}: found {drain_count} B=1 drains; maximum is {max_drain_windows}"
        )
    expected_generation_windows = expected_base_generation_windows + drain_count
    if generation_window_count != expected_generation_windows:
        raise HarnessError(
            f"{path}: found {generation_window_count} generation annotations; "
            f"expected {expected_generation_windows}"
        )

    kernels.sort(key=lambda item: (item["start"], item["end"], item["name"]))
    cursor = 0
    steady: list[dict[str, Any]] = []
    drains: list[dict[str, Any]] = []
    expected_family_counts = {
        family: expected for family, (_, expected) in family_contract.items()
    }
    for index, (window_start, window_end, batch, label) in enumerate(
        pure_windows, start=1
    ):
        while cursor < len(kernels) and kernels[cursor]["end"] <= window_start:
            cursor += 1
        scan = cursor
        contained = []
        while scan < len(kernels) and kernels[scan]["start"] < window_end:
            event = kernels[scan]
            if event["start"] >= window_start and event["end"] <= window_end:
                contained.append(event)
            elif batch == TRACE_BATCH:
                raise HarnessError(
                    f"{path}: kernel crosses steady B=2 window {index} boundary"
                )
            scan += 1
        cursor = scan
        names = [event["name"] for event in contained]
        name_sha256 = _digest(names)
        kernel_time_us = sum(event["duration_us"] for event in contained)
        if batch == 1:
            drains.append(
                {
                    "kernel_count": len(contained),
                    "kernel_time_us": kernel_time_us,
                    "ordered_names_sha256": name_sha256,
                }
            )
            continue
        if len(contained) != expected_kernel_count:
            raise HarnessError(
                f"{path}: steady window {index} has {len(contained)} kernels; "
                f"expected {expected_kernel_count}"
            )
        if name_sha256 != expected_names_sha256:
            raise HarnessError(
                f"{path}: steady window {index} ordered kernel names drifted"
            )
        signatures = [_launch_signature(event, path) for event in contained]
        signature_sha256 = _digest(signatures)
        signature_label = allowed_signature_sha256.get(signature_sha256)
        if signature_label is None:
            raise HarnessError(
                f"{path}: steady window {index} launch signature drifted: "
                f"{signature_sha256}"
            )

        broad_counts = Counter()
        selected_counts = Counter()
        selected_times = defaultdict(float)
        fp4_tactics = Counter()
        for event in contained:
            broad_family = _family_for(event["name"], family_contract)
            if broad_family is not None:
                broad_counts[broad_family] += 1
            selected = _selected_family(event["name"], broad_family, local=False)
            selected_counts[selected] += 1
            selected_times[selected] += event["duration_us"]
            tactic = _fp4_tactic(event["name"])
            if tactic is not None:
                fp4_tactics[tactic] += 1
        if dict(broad_counts) != expected_family_counts:
            raise HarnessError(
                f"{path}: steady window {index} family counts drifted: "
                f"{dict(broad_counts)}"
            )
        if dict(fp4_tactics) != dict(expected_fp4_tactics):
            raise HarnessError(
                f"{path}: steady window {index} FP4 tactics drifted: "
                f"{dict(fp4_tactics)}"
            )
        rmsnorm_count = selected_counts["rmsnorm_generated_partition"]
        if rmsnorm_count != expected_generated_rmsnorm_quant:
            raise HarnessError(
                f"{path}: steady window {index} has {rmsnorm_count} generated "
                f"RMSNorm/quant partitions; expected {expected_generated_rmsnorm_quant}"
            )
        steady.append(
            {
                "family_counts": dict(selected_counts),
                "family_time_us": dict(selected_times),
                "fp4_tactics": dict(fp4_tactics),
                "kernel_time_us": kernel_time_us,
                "signature_label": signature_label,
                "signature_sha256": signature_sha256,
            }
        )

    if len(steady) < expected_pure_windows - max_drain_windows:
        raise HarnessError(f"{path}: too few steady B=2 windows")
    signature_distribution = Counter(item["signature_sha256"] for item in steady)
    if len(signature_distribution) != 1:
        raise HarnessError(f"{path}: steady B=2 signature varies within one trace")
    family_names = sorted(steady[0]["family_counts"])
    family_summary = {}
    for family in family_names:
        counts = [item["family_counts"][family] for item in steady]
        if len(set(counts)) != 1:
            raise HarnessError(f"{path}: steady {family} count varies")
        family_summary[family] = {
            "count_per_window": counts[0],
            "kernel_time_us": _stats(
                [item["family_time_us"][family] for item in steady]
            ),
        }
    drain_distribution = Counter(
        (item["kernel_count"], item["ordered_names_sha256"]) for item in drains
    )
    return {
        "aggregate_kernel_count": len(kernels),
        "aggregate_kernel_time_us": sum(item["duration_us"] for item in kernels),
        "drain_window_count": len(drains),
        "drain_window_topologies": [
            {
                "count": count,
                "kernel_count": kernel_count,
                "ordered_names_sha256": names_sha256,
            }
            for (kernel_count, names_sha256), count in sorted(drain_distribution.items())
        ],
        "family_summary": family_summary,
        "generation_window_count": generation_window_count,
        "ordered_names_sha256": expected_names_sha256,
        "pure_window_count": len(pure_windows),
        "selected_trace": str(path),
        "selected_trace_sha256": sha256_file(path),
        "steady_b2_kernel_count_per_window": expected_kernel_count,
        "steady_b2_kernel_time_us": _stats(
            [item["kernel_time_us"] for item in steady]
        ),
        "steady_b2_signature_distribution": dict(signature_distribution),
        "steady_b2_signature_label": steady[0]["signature_label"],
        "steady_b2_window_count": len(steady),
    }


def summarize_local_c2(
    core_status: Mapping[str, Any],
    *,
    expected_graph_contract: Mapping[str, Any] | None = None,
    expected_fp4_tactics: Mapping[str, int] = EXPECTED_B2_FP4_TACTICS,
    expected_rmsnorm_count: int = EXPECTED_B2_GENERATED_RMSNORM_QUANT,
) -> dict[str, Any]:
    if expected_graph_contract is None:
        expected_graph_contract = TRACE_PRIMARY_GRAPH_CONTRACTS_BY_BATCH[
            (MODEL_KEY, TRACE_BATCH)
        ]
    summaries = core_status.get("nsys_kernel_summaries")
    validations = core_status.get("nsys_validations")
    if (
        not isinstance(summaries, list)
        or not isinstance(validations, list)
        or len(summaries) != TRACE_REPETITIONS * TRACE_CAPTURE_GRAPH_REPLAYS
        or len(validations) != len(summaries)
    ):
        raise HarnessError("low-batch core status lacks 12 local ranges")
    topology_hashes = {
        item.get("canonical_node_multiset_sha256") for item in validations
    }
    if len(topology_hashes) != 1 or not all(
        isinstance(value, str) for value in topology_hashes
    ):
        raise HarnessError("local low-batch topology is not invariant")
    invariant_topology_hashes = {
        item.get("gdn_packed_invariant_node_multiset_sha256")
        for item in validations
    }
    invariant_topology_hash = None
    if invariant_topology_hashes != {None}:
        if len(invariant_topology_hashes) != 1 or not all(
            isinstance(value, str) for value in invariant_topology_hashes
        ):
            raise HarnessError("local mode-invariant GDN topology is not invariant")
        invariant_topology_hash = next(iter(invariant_topology_hashes))
    coupled_ba_topology_hashes = {
        item.get("gdn_packed_coupled_ba_node_multiset_sha256")
        for item in validations
    }
    coupled_ba_topology_hash = None
    if coupled_ba_topology_hashes != {None}:
        if len(coupled_ba_topology_hashes) != 1 or not all(
            isinstance(value, str) for value in coupled_ba_topology_hashes
        ):
            raise HarnessError("local coupled BA topology is not invariant")
        coupled_ba_topology_hash = next(iter(coupled_ba_topology_hashes))
    coupled_ba_node_counts = {
        item.get("gdn_packed_coupled_ba_node_count") for item in validations
    }
    coupled_ba_node_count = None
    if coupled_ba_node_counts != {None}:
        if coupled_ba_node_counts != {TRACE_GDN_PACKED_COUPLED_BA_NODE_COUNT}:
            raise HarnessError("local coupled BA node count is not invariant")
        coupled_ba_node_count = next(iter(coupled_ba_node_counts))
    optional_packed_fields = (
        invariant_topology_hash,
        coupled_ba_topology_hash,
        coupled_ba_node_count,
    )
    if any(value is None for value in optional_packed_fields) and any(
        value is not None for value in optional_packed_fields
    ):
        raise HarnessError("local packed GDN topology metadata is incomplete")
    range_records = []
    expected_family_counts = {
        family: expected
        for family, (_, expected) in expected_graph_contract["families"].items()
    }
    for index, summary in enumerate(summaries, start=1):
        if not isinstance(summary, dict):
            raise HarnessError(f"local range {index} summary is malformed")
        if summary.get("kernel_count") != expected_graph_contract["node_count"]:
            raise HarnessError(f"local range {index} kernel count drifted")
        kernels = summary.get("kernels")
        if not isinstance(kernels, list) or not kernels:
            raise HarnessError(f"local range {index} kernel inventory is absent")
        broad_counts = Counter()
        selected_counts = Counter()
        selected_times = defaultdict(float)
        fp4_tactics = Counter()
        total_count = 0
        total_time_us = 0.0
        for kernel in kernels:
            if not isinstance(kernel, dict) or not isinstance(kernel.get("name"), str):
                raise HarnessError(f"local range {index} kernel is malformed")
            name = kernel["name"]
            count = kernel.get("count")
            duration_ns = kernel.get("total_duration_ns")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or isinstance(duration_ns, bool)
                or not isinstance(duration_ns, int)
                or duration_ns <= 0
            ):
                raise HarnessError(f"local range {index} kernel totals are invalid")
            duration_us = duration_ns / 1_000.0
            total_count += count
            total_time_us += duration_us
            broad_family = _family_for(name, expected_graph_contract["families"])
            if broad_family is not None:
                broad_counts[broad_family] += count
            selected = _selected_family(name, broad_family, local=True)
            selected_counts[selected] += count
            selected_times[selected] += duration_us
            tactic = _fp4_tactic(name)
            if tactic is not None:
                fp4_tactics[tactic] += count
        if total_count != expected_graph_contract["node_count"]:
            raise HarnessError(f"local range {index} per-kernel counts do not sum")
        actual_family_counts = {
            family: broad_counts[family] for family in expected_family_counts
        }
        if actual_family_counts != expected_family_counts:
            raise HarnessError(f"local range {index} family counts drifted")
        if dict(fp4_tactics) != dict(expected_fp4_tactics):
            raise HarnessError(f"local range {index} FP4 tactics drifted")
        if (
            selected_counts["rmsnorm_generated_partition"]
            != expected_rmsnorm_count
        ):
            raise HarnessError(f"local range {index} RMSNorm count drifted")
        recorded_total = summary.get("kernel_time_ns")
        if (
            isinstance(recorded_total, bool)
            or not isinstance(recorded_total, int)
            or abs(total_time_us - recorded_total / 1_000.0) > 1e-6
        ):
            raise HarnessError(f"local range {index} kernel time does not sum")
        range_records.append(
            {
                "family_counts": dict(selected_counts),
                "family_time_us": dict(selected_times),
                "kernel_time_us": total_time_us,
            }
        )
    family_summary = {}
    for family in sorted(range_records[0]["family_counts"]):
        counts = [item["family_counts"][family] for item in range_records]
        if len(set(counts)) != 1:
            raise HarnessError(f"local {family} count varies across ranges")
        family_summary[family] = {
            "count_per_window": counts[0],
            "kernel_time_us": _stats(
                [item["family_time_us"][family] for item in range_records]
            ),
        }
    result = {
        "family_summary": family_summary,
        "kernel_count_per_window": expected_graph_contract["node_count"],
        "kernel_time_us": _stats(
            [item["kernel_time_us"] for item in range_records]
        ),
        "node_multiset_sha256": next(iter(topology_hashes)),
        "range_count": len(range_records),
    }
    if invariant_topology_hash is not None:
        result["gdn_packed_invariant_node_multiset_sha256"] = (
            invariant_topology_hash
        )
        result["gdn_packed_coupled_ba_node_multiset_sha256"] = (
            coupled_ba_topology_hash
        )
        result["gdn_packed_coupled_ba_node_count"] = coupled_ba_node_count
    return result


def _validate_model_gate(path: pathlib.Path, source_commit: str) -> dict[str, Any]:
    gate = _load_json_object(path)
    expected = {
        "model_key": MODEL_KEY,
        "passed": True,
        "test_name": "test_qwen27_paged_engine",
        "vllm_cpp_sha": source_commit,
    }
    for field, value in expected.items():
        if gate.get(field) != value:
            raise HarnessError(f"model gate {field} differs")
    log = gate.get("log")
    digest = gate.get("log_sha256")
    if not isinstance(log, str) or not isinstance(digest, str):
        raise HarnessError("model gate log record is absent")
    log_path = pathlib.Path(log)
    if not log_path.is_file() or sha256_file(log_path) != digest:
        raise HarnessError("model gate log drifted")
    return {"file": str(path), "file_sha256": sha256_file(path), **gate}


def _validate_run_log(path: pathlib.Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise HarnessError(f"run log is absent: {path}") from error
    required = {
        "c2 raw paired trace capture complete; status remains PENDING until low-batch finalization",
        "model 27 node-level paired trace complete",
    }
    if not required.issubset(set(lines)):
        raise HarnessError("run log lacks the successful raw-capture terminus")
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


def _trace_paths(evidence: pathlib.Path) -> dict[str, Any]:
    trace = evidence / "trace" / MODEL_KEY
    return {
        "trace": trace,
        "reports": sorted(trace.glob("ours-r?.?.nsys-rep")),
        "sqlites": sorted(trace.glob("ours-r?.?.sqlite")),
        "validations": sorted(trace.glob("ours-r?.?-nsys-validation.json")),
        "summaries": sorted(trace.glob("ours-r?.?-cuda_gpu_kern_sum.txt")),
        "commands": sorted(trace.glob("ours-r?-profile-command.txt")),
        "logs": sorted(trace.glob("ours-r?-profile.log")),
        "controls": sorted(trace.glob("ours-r?-profile-control.json")),
    }


def build_summary(
    evidence: pathlib.Path,
    *,
    source_commit: str,
    run_log: pathlib.Path,
) -> dict[str, Any]:
    if SOURCE_COMMIT_RE.fullmatch(source_commit) is None:
        raise HarnessError("source commit must be a lowercase 40-character Git SHA")
    paths = _trace_paths(evidence)
    trace = paths["trace"]
    vllm_kernel_summary = trace / "vllm-kernels.json"
    vllm_summary = _load_json_object(vllm_kernel_summary)
    selected_trace = vllm_summary.get("selected_trace")
    if not isinstance(selected_trace, str) or not pathlib.Path(selected_trace).is_absolute():
        raise HarnessError("vLLM aggregate summary lacks an absolute selected trace")
    vllm_trace = pathlib.Path(selected_trace)

    core = record_trace_status(
        trace / "c2-summary.json",
        model_key=MODEL_KEY,
        ours_nsys_reports=paths["reports"],
        ours_nsys_sqlites=paths["sqlites"],
        ours_nsys_validations=paths["validations"],
        ours_kernel_summaries=paths["summaries"],
        ours_commands=paths["commands"],
        ours_profile_logs=paths["logs"],
        ours_profile_controls=paths["controls"],
        ours_client_results=[
            evidence / "raw" / MODEL_KEY / "ours" / f"c2-r1-trace{index}.json"
            for index in range(1, TRACE_REPETITIONS + 1)
        ],
        ours_client_logs=[
            evidence / "logs" / MODEL_KEY / "ours" / f"c2-r1-trace{index}.log"
            for index in range(1, TRACE_REPETITIONS + 1)
        ],
        ours_probe_results=[
            evidence
            / "raw"
            / MODEL_KEY
            / "ours"
            / f"c2-r1-trace{index}-probe.json"
            for index in range(1, TRACE_REPETITIONS + 1)
        ],
        ours_probe_logs=[
            evidence
            / "logs"
            / MODEL_KEY
            / "ours"
            / f"c2-r1-trace{index}-probe.log"
            for index in range(1, TRACE_REPETITIONS + 1)
        ],
        vllm_torch_trace=vllm_trace,
        vllm_kernel_summary=vllm_kernel_summary,
        vllm_command=trace / "vllm-profile-command.txt",
        vllm_profile_log=trace / "vllm-profile.log",
        vllm_metadata=trace / "vllm-profile-metadata.json",
        vllm_corpus=evidence / "corpus" / MODEL_KEY / "vllm" / "c2-r1.jsonl",
        cache_drop_reports=[
            trace / "cache-before-ours.json",
            trace / "cache-between-engines.json",
            trace / "cache-after-vllm.json",
        ],
        execution_manifest=evidence / "execution" / "27-trace.json",
        vllm_cpp_sha=source_commit,
        expected_batch=TRACE_BATCH,
        trace_prompts=TRACE_PROMPTS,
        write_output=False,
    )
    metadata = _load_json_object(trace / "vllm-profile-metadata.json")
    if metadata.get("async_scheduling_requested") != "default":
        raise HarnessError("vLLM trace did not preserve default async scheduling")
    if metadata.get("async_scheduling_resolved") is not True:
        raise HarnessError("vLLM default async scheduling did not resolve enabled")

    model_gate = _validate_model_gate(
        evidence / "preflight" / "model-gate" / "27.json", source_commit
    )
    run_log_record = _validate_run_log(run_log)
    local = summarize_local_c2(core)
    oracle = summarize_vllm_c2_trace(vllm_trace)
    if oracle["selected_trace_sha256"] != vllm_summary.get("selected_trace_sha256"):
        raise HarnessError("vLLM selected trace hash differs from its aggregate summary")

    local_median = float(local["kernel_time_us"]["median"])
    oracle_median = float(oracle["steady_b2_kernel_time_us"]["median"])
    family_deltas = []
    shared_families = set(local["family_summary"]) & set(oracle["family_summary"])
    for family in shared_families:
        local_family = local["family_summary"][family]
        oracle_family = oracle["family_summary"][family]
        local_time = float(local_family["kernel_time_us"]["median"])
        oracle_time = float(oracle_family["kernel_time_us"]["median"])
        family_deltas.append(
            {
                "family": family,
                "local_count": local_family["count_per_window"],
                "oracle_count": oracle_family["count_per_window"],
                "local_median_us": local_time,
                "oracle_median_us": oracle_time,
                "local_minus_oracle_us": local_time - oracle_time,
            }
        )
    family_deltas.sort(
        key=lambda item: (-item["local_minus_oracle_us"], item["family"])
    )
    return {
        "schema_version": 1,
        "status": "complete-diagnostic",
        "benchmark_binding": False,
        "benchmark_binding_reason": (
            "exact topology and per-profiler kernel durations only; local Nsight and "
            "oracle Torch-profiler timings do not replace the online every-axis gate"
        ),
        "evidence_source_commit": source_commit,
        "workload": {
            "admission_mode": "closed-loop",
            "async_scheduling": "vLLM-default-enabled",
            "input_len": 1024,
            "max_concurrency": TRACE_BATCH,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS[MODEL_KEY],
            "max_num_seqs": MAX_NUM_SEQS,
            "model": "Qwen3.6-27B-NVFP4",
            "num_prompts": TRACE_PROMPTS,
            "output_len": 128,
            "prefix_caching": False,
            "profile_runs": TRACE_RUNS,
        },
        "model_gate": model_gate,
        "run_log": run_log_record,
        "core_trace_status": core,
        "local": local,
        "oracle": oracle,
        "diagnostic_comparison": {
            "family_deltas": family_deltas,
            "local_over_oracle_median_kernel_time": local_median / oracle_median,
            "local_throughput_proxy_over_oracle": oracle_median / local_median,
            "speed_credit": False,
        },
    }


def build_manifest(
    evidence: pathlib.Path, *, run_log: pathlib.Path
) -> dict[str, Any]:
    files = []
    for path in sorted(evidence.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(evidence).as_posix()
        if relative in DERIVED_PATHS:
            continue
        files.append(
            {"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size}
        )
    if not files:
        raise HarnessError(f"no immutable low-batch artifacts under {evidence}")
    external = [
        {"path": str(run_log), "sha256": sha256_file(run_log), "size": run_log.stat().st_size}
    ]
    return {
        "schema_version": 1,
        "artifact_set_sha256": _digest(files + external),
        "external_files": external,
        "files": files,
    }


def finalize(
    evidence: pathlib.Path,
    *,
    source_commit: str,
    run_log: pathlib.Path,
    finalizer: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trace = evidence / "trace" / MODEL_KEY
    summary_path = trace / "c2-summary.json"
    manifest_path = trace / "c2-manifest.json"
    marker_path = trace / "status-c2.json"
    existing = [
        str(path) for path in (summary_path, manifest_path, marker_path) if path.exists()
    ]
    if existing:
        raise HarnessError(f"refusing to overwrite derived artifact(s): {existing}")
    summary = build_summary(
        evidence, source_commit=source_commit, run_log=run_log
    )
    manifest = build_manifest(evidence, run_log=run_log)
    write_json_atomic(summary_path, summary)
    write_json_atomic(manifest_path, manifest)
    marker = {
        "schema_version": 1,
        "status": "complete-diagnostic",
        "evidence_source_commit": source_commit,
        "artifact_set_sha256": manifest["artifact_set_sha256"],
        "finalizer_sha256": sha256_file(finalizer),
        "manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
    }
    write_json_atomic(marker_path, marker)
    return summary, manifest, marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-log", type=pathlib.Path)
    args = parser.parse_args()
    run_log = args.run_log
    if run_log is None:
        run_log = args.evidence.parent.parent / "c2-run.log"
    summary, _, marker = finalize(
        args.evidence,
        source_commit=args.source_commit,
        run_log=run_log,
        finalizer=pathlib.Path(__file__),
    )
    print(
        canonical_json(
            {
                "local_over_oracle_median_kernel_time": summary[
                    "diagnostic_comparison"
                ]["local_over_oracle_median_kernel_time"],
                "oracle_steady_b2_windows": summary["oracle"][
                    "steady_b2_window_count"
                ],
                "status": marker["status"],
                "summary_sha256": marker["summary_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"finalize-low-batch-trace: {error}", file=sys.stderr)
        raise SystemExit(2) from error
