#!/usr/bin/env python3
"""Summarize the exact local Qwen3.5-4B comparison evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
from typing import Any


def numeric_series(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def parse_cpp_report(path: pathlib.Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([^:]+):\s+([0-9.]+)\s*$", line)
        if match:
            result[match.group(1).strip()] = float(match.group(2))
    required = {
        "Successful requests",
        "Total input tokens",
        "Total generated tokens",
        "Total token throughput (tok/s)",
        "Mean TTFT (ms)",
        "Mean TPOT (ms)",
        "Mean ITL (ms)",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"{path}: missing report fields: {sorted(missing)}")
    return result


def cpp_metrics(root: pathlib.Path, mode: str) -> dict[str, Any]:
    runs = [
        parse_cpp_report(root / f"performance-cpp-{mode}-r{rep}.log")
        for rep in range(1, 4)
    ]
    return {
        key: numeric_series([run[key] for run in runs])
        for key in runs[0]
    }


def vllm_metrics(root: pathlib.Path) -> dict[str, Any]:
    runs = [
        json.loads((root / f"performance-vllm-r{rep}.metrics.json").read_text())
        for rep in range(1, 4)
    ]
    scalar = [
        "duration_s",
        "successful_requests",
        "maximum_request_concurrency",
        "input_tokens",
        "output_tokens",
        "request_throughput",
        "input_token_throughput",
        "output_token_throughput",
        "total_token_throughput",
        "mean_per_stream_decode_rate",
    ]
    result = {key: numeric_series([run[key] for run in runs]) for key in scalar}
    for family in ("ttft_ms", "core_ttft_ms", "tpot_ms", "itl_ms", "e2el_ms"):
        result[family] = {
            stat: numeric_series([run[family][stat] for run in runs])
            for stat in ("mean", "median", "p99")
        }
    return result


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def memory(root: pathlib.Path, arm: str) -> dict[str, Any]:
    runs = []
    for rep in range(1, 4):
        prefix = f"memory-{arm}-r{rep}"
        summary = json.loads((root / f"{prefix}.memory-summary.json").read_text())
        samples = load_jsonl(root / f"{prefix}.memory.jsonl")
        live = [sample for sample in samples if sample["alive"]]
        resident = [
            sample
            for sample in live
            if sample["gpu_memory_mib"] is not None
            and sample["gpu_memory_mib"] > 10_000
        ][-10:]
        started = float((root / f"{prefix}.started").read_text())
        finished = float((root / f"{prefix}.finished").read_text())
        runs.append(
            {
                "peak_pss_gib": summary["peak_pss_kib"] / 1024 / 1024,
                "peak_rss_gib": summary["peak_rss_kib"] / 1024 / 1024,
                "stable_pss_gib": (
                    statistics.fmean(row["pss_kib"] for row in resident)
                    / 1024
                    / 1024
                    if resident
                    else None
                ),
                "peak_vram_mib": max(
                    sample["gpu_memory_mib"] or 0 for sample in live
                ),
                "mem_available_drop_gib": (
                    summary["peak_mem_available_drop_kib"] / 1024 / 1024
                ),
                "wall_s": finished - started,
            }
        )
    result: dict[str, Any] = {}
    for key in runs[0]:
        values = [run[key] for run in runs]
        result[key] = (
            numeric_series(values) if all(value is not None for value in values) else None
        )
    return result


def load_tokens(path: pathlib.Path) -> list[list[int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 128 or any(len(row) != 128 for row in rows):
        raise ValueError(f"{path}: expected 128 request rows of 128 tokens")
    return rows


def request_matches(left: list[list[int]], right: list[list[int]]) -> int:
    return sum(a == b for a, b in zip(left, right))


def token_comparisons(
    root: pathlib.Path, historical_root: pathlib.Path
) -> dict[str, Any]:
    current = {
        arm: [
            load_tokens(root / f"performance-{arm}-r{rep}.tokens.json")
            for rep in range(1, 4)
        ]
        for arm in ("cpp-on", "cpp-off", "vllm")
    }
    historical = {
        arm: [
            load_tokens(historical_root / f"perf-{arm}-r{rep}.tokens.json")
            for rep in range(1, 4)
        ]
        for arm in ("cpp-on", "cpp-off", "vllm")
    }
    result: dict[str, Any] = {
        "current_same_arm_vs_r1": {},
        "current_on_vs_off_by_rep": [],
        "current_vs_historical_same_arm_by_rep": {},
        "current_cpp_on_vs_vllm_by_rep": [],
    }
    for arm, runs in current.items():
        result["current_same_arm_vs_r1"][arm] = [
            request_matches(runs[0], run) for run in runs
        ]
        result["current_vs_historical_same_arm_by_rep"][arm] = [
            request_matches(run, old)
            for run, old in zip(runs, historical[arm])
        ]
    result["current_on_vs_off_by_rep"] = [
        request_matches(on, off)
        for on, off in zip(current["cpp-on"], current["cpp-off"])
    ]
    result["current_cpp_on_vs_vllm_by_rep"] = [
        request_matches(on, ref)
        for on, ref in zip(current["cpp-on"], current["vllm"])
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--historical-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    old_aggregate = json.loads(
        (args.historical_root / "aggregate.json").read_text(encoding="utf-8")
    )
    current_on = cpp_metrics(args.root, "on")
    current_off = cpp_metrics(args.root, "off")
    current_vllm = vllm_metrics(args.root)
    result = {
        "commit": (args.root / "commit.txt").read_text().strip(),
        "cpp_on": current_on,
        "cpp_off": current_off,
        "vllm": current_vllm,
        "memory": {
            "cpp_on": memory(args.root, "cpp-on"),
            "cpp_off": memory(args.root, "cpp-off"),
            "vllm": memory(args.root, "vllm"),
        },
        "tokens": token_comparisons(args.root, args.historical_root),
        "historical": {
            "commit": old_aggregate["commit"],
            "cpp_on": old_aggregate["cpp_on"],
            "cpp_off": old_aggregate["cpp_off"],
            "vllm": old_aggregate["vllm_production"],
            "vllm_client": {
                key: value
                for key, value in old_aggregate["vllm_client"].items()
                if key != "per_request"
            },
            "memory": old_aggregate["memory"],
        },
        "ratios": {
            "cpp_on_vs_historical_total_tput": (
                current_on["Total token throughput (tok/s)"]["mean"]
                / old_aggregate["cpp_on"]["Total token throughput (tok/s)"]["mean"]
            ),
            "cpp_off_vs_historical_total_tput": (
                current_off["Total token throughput (tok/s)"]["mean"]
                / old_aggregate["cpp_off"]["Total token throughput (tok/s)"]["mean"]
            ),
            "cpp_on_vs_current_vllm_total_tput": (
                current_on["Total token throughput (tok/s)"]["mean"]
                / current_vllm["total_token_throughput"]["mean"]
            ),
            "cpp_off_vs_current_vllm_total_tput": (
                current_off["Total token throughput (tok/s)"]["mean"]
                / current_vllm["total_token_throughput"]["mean"]
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"commit": result["commit"], "ratios": result["ratios"], "tokens": result["tokens"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
