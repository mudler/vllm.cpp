#!/usr/bin/env python3
"""Validate and finalize the paired vLLM async-scheduling diagnostic.

The GPU series deliberately leaves finalization to a CPU-only, fail-closed
step.  This module validates all six online timing legs and both shape-neutral
Torch-profiler summaries before atomically writing the derived summary,
artifact manifest, and completion marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

if __package__ in (None, ""):
    script = pathlib.Path(__file__).resolve()
    if len(script.parents) > 2:
        sys.path.insert(0, str(script.parents[2]))

from tools.bench.serve_low_common import (
    HarnessError,
    canonical_json,
    coefficient_of_variation,
    require_number,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)


MODES = ("on", "off")
REPETITIONS = (1, 2, 3)
SOURCE_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
DERIVED_NAMES = frozenset(
    {"summary.json", "manifest.json", "series-complete.json", "series.log"}
)
METRICS = {
    "request_throughput": "higher",
    "output_throughput": "higher",
    "total_token_throughput": "higher",
    "mean_ttft_ms": "lower",
    "mean_tpot_ms": "lower",
    "mean_itl_ms": "lower",
    "mean_e2el_ms": "lower",
}


def _load_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HarnessError(f"missing artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise HarnessError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError(f"{path}: root must be an object")
    return value


def _require_int(value: Any, field: str, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessError(f"{field} must be an integer")
    if expected is not None and value != expected:
        raise HarnessError(f"{field} must be {expected}; found {value}")
    return value


def _require_bool(value: Any, field: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise HarnessError(f"{field} must be boolean")
    if expected is not None and value is not expected:
        raise HarnessError(f"{field} must be {expected}; found {value}")
    return value


def _require_hex_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise HarnessError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _digest_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _validate_raw_leg(path: pathlib.Path) -> dict[str, Any]:
    raw = _load_object(path)
    _require_int(raw.get("completed"), f"{path}:completed", 6)
    _require_int(raw.get("failed"), f"{path}:failed", 0)
    _require_int(raw.get("num_prompts"), f"{path}:num_prompts", 6)
    _require_int(raw.get("max_concurrency"), f"{path}:max_concurrency", 2)
    _require_int(raw.get("total_input_tokens"), f"{path}:total_input_tokens", 6144)
    _require_int(raw.get("total_output_tokens"), f"{path}:total_output_tokens", 768)

    errors = raw.get("errors")
    if not isinstance(errors, list) or len(errors) != 6 or any(errors):
        raise HarnessError(f"{path}: expected six falsey request errors")
    input_lens = raw.get("input_lens")
    output_lens = raw.get("output_lens")
    if input_lens != [1024] * 6:
        raise HarnessError(f"{path}: input lengths are not six exact 1024-token prompts")
    if output_lens != [128] * 6:
        raise HarnessError(f"{path}: output lengths are not six exact 128-token generations")
    generated = raw.get("generated_texts")
    if not isinstance(generated, list) or len(generated) != 6:
        raise HarnessError(f"{path}: expected six generated texts")

    metrics = {
        name: require_number(raw.get(name), f"{path}:{name}") for name in METRICS
    }
    return {
        "file": str(path),
        "file_sha256": sha256_file(path),
        "generated_texts_sha256": _digest_value(generated),
        "metrics": metrics,
    }


def _validate_metadata(path: pathlib.Path, mode: str) -> dict[str, Any]:
    metadata = _load_object(path)
    if metadata.get("async_scheduling_requested") != mode:
        raise HarnessError(f"{path}: requested async mode is not {mode}")
    _require_bool(
        metadata.get("async_scheduling_resolved"),
        f"{path}:async_scheduling_resolved",
        mode == "on",
    )
    if metadata.get("admission_mode") != "closed-loop":
        raise HarnessError(f"{path}: admission mode is not closed-loop")
    _require_bool(metadata.get("enable_prefix_caching"), f"{path}:prefix", False)
    for field, expected in {
        "input_len": 1024,
        "output_len": 128,
        "num_prompts": 6,
        "max_concurrency": 2,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 2048,
        "profiled_warmup_prompts": 6,
        "repetitions": 3,
    }.items():
        _require_int(metadata.get(field), f"{path}:{field}", expected)
    corpus_sha256 = _require_hex_digest(
        metadata.get("corpus_sha256"), f"{path}:corpus_sha256"
    )
    digests = metadata.get("output_digests")
    if not isinstance(digests, list) or len(digests) != 4:
        raise HarnessError(f"{path}: expected warmup plus three output digests")
    for index, digest in enumerate(digests):
        _require_hex_digest(digest, f"{path}:output_digests[{index}]")
    all_equal = len(set(digests)) == 1
    _require_bool(metadata.get("output_digests_equal"), f"{path}:digests", all_equal)
    if metadata.get("output_digest") != digests[0]:
        raise HarnessError(f"{path}: output_digest is not the first recorded digest")
    return {
        "file": str(path),
        "file_sha256": sha256_file(path),
        "corpus_sha256": corpus_sha256,
        "output_digests": digests,
        "output_digests_equal": all_equal,
    }


def _validate_kernel_summary(path: pathlib.Path) -> dict[str, Any]:
    summary = _load_object(path)
    kernel_count = _require_int(summary.get("kernel_count"), f"{path}:kernel_count")
    if kernel_count <= 0:
        raise HarnessError(f"{path}: kernel_count must be positive")
    kernel_time_us = require_number(summary.get("kernel_time_us"), f"{path}:time")
    if kernel_time_us <= 0.0:
        raise HarnessError(f"{path}: kernel_time_us must be positive")
    selected_trace_sha256 = _require_hex_digest(
        summary.get("selected_trace_sha256"), f"{path}:selected_trace_sha256"
    )
    kernels = summary.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise HarnessError(f"{path}: kernels must be a non-empty list")
    by_name: dict[str, dict[str, float | int]] = {}
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, dict) or not isinstance(kernel.get("name"), str):
            raise HarnessError(f"{path}:kernels[{index}] is malformed")
        name = kernel["name"]
        if name in by_name:
            raise HarnessError(f"{path}: duplicate kernel name: {name}")
        count = _require_int(kernel.get("count"), f"{path}:{name}:count")
        duration = require_number(
            kernel.get("total_duration_us"), f"{path}:{name}:duration"
        )
        if count <= 0 or duration <= 0.0:
            raise HarnessError(f"{path}:{name}: count and duration must be positive")
        by_name[name] = {"count": count, "total_duration_us": duration}
    if sum(int(item["count"]) for item in by_name.values()) != kernel_count:
        raise HarnessError(f"{path}: per-kernel counts do not sum to kernel_count")
    return {
        "file": str(path),
        "file_sha256": sha256_file(path),
        "kernel_count": kernel_count,
        "kernel_time_us": kernel_time_us,
        "selected_trace_sha256": selected_trace_sha256,
        "by_name": by_name,
    }


def _metric_summary(values: Mapping[str, Sequence[float]], direction: str) -> dict[str, Any]:
    on = list(values["on"])
    off = list(values["off"])
    on_median = statistics.median(on)
    off_median = statistics.median(off)
    ratio = on_median / off_median if direction == "higher" else off_median / on_median
    return {
        "direction": direction,
        "on": on,
        "off": off,
        "on_median": on_median,
        "off_median": off_median,
        "on_cv": coefficient_of_variation(on),
        "off_cv": coefficient_of_variation(off),
        "direction_normalized_ratio": ratio,
    }


def _kernel_delta(on: Mapping[str, Any], off: Mapping[str, Any]) -> dict[str, Any]:
    on_names = set(on)
    off_names = set(off)
    deltas = []
    for name in on_names | off_names:
        on_item = on.get(name, {"count": 0, "total_duration_us": 0.0})
        off_item = off.get(name, {"count": 0, "total_duration_us": 0.0})
        duration_delta = float(on_item["total_duration_us"]) - float(
            off_item["total_duration_us"]
        )
        deltas.append(
            {
                "name": name,
                "on_count": int(on_item["count"]),
                "off_count": int(off_item["count"]),
                "count_delta": int(on_item["count"]) - int(off_item["count"]),
                "on_duration_us": float(on_item["total_duration_us"]),
                "off_duration_us": float(off_item["total_duration_us"]),
                "duration_delta_us": duration_delta,
            }
        )
    deltas.sort(key=lambda item: (-abs(item["duration_delta_us"]), item["name"]))
    return {
        "shared_names": len(on_names & off_names),
        "on_only_names": len(on_names - off_names),
        "off_only_names": len(off_names - on_names),
        "largest_absolute_duration_deltas": deltas[:20],
    }


def build_summary(
    evidence: pathlib.Path,
    *,
    source_commit: str,
    minimum_total_credit: float = 1.04,
) -> dict[str, Any]:
    if SOURCE_COMMIT_RE.fullmatch(source_commit) is None:
        raise HarnessError("source commit must be a lowercase 40-character Git SHA")
    if minimum_total_credit <= 1.0:
        raise HarnessError("minimum total credit must be greater than 1.0")

    raw: dict[str, dict[int, dict[str, Any]]] = {mode: {} for mode in MODES}
    metric_values: dict[str, dict[str, list[float]]] = {
        name: {mode: [] for mode in MODES} for name in METRICS
    }
    for mode in MODES:
        for repetition in REPETITIONS:
            leg = _validate_raw_leg(evidence / "raw" / f"{mode}-r{repetition}.json")
            raw[mode][repetition] = leg
            for name, value in leg["metrics"].items():
                metric_values[name][mode].append(value)

    metadata = {
        mode: _validate_metadata(evidence / "traces" / f"{mode}-metadata.json", mode)
        for mode in MODES
    }
    if metadata["on"]["corpus_sha256"] != metadata["off"]["corpus_sha256"]:
        raise HarnessError("trace modes did not use the same corpus")
    kernels = {
        mode: _validate_kernel_summary(
            evidence / "traces" / f"{mode}-kernel-summary.json"
        )
        for mode in MODES
    }
    timing = {
        name: _metric_summary(values, METRICS[name])
        for name, values in metric_values.items()
    }
    paired_outputs = {
        f"r{repetition}": (
            raw["on"][repetition]["generated_texts_sha256"]
            == raw["off"][repetition]["generated_texts_sha256"]
        )
        for repetition in REPETITIONS
    }
    total_ratio = timing["total_token_throughput"]["direction_normalized_ratio"]
    return {
        "schema_version": 1,
        "status": "complete-diagnostic",
        "evidence_source_commit": source_commit,
        "workload": {
            "model": "Qwen3.6-27B-NVFP4",
            "input_len": 1024,
            "output_len": 128,
            "num_prompts": 6,
            "max_concurrency": 2,
            "repetitions": 3,
            "admission_mode": "closed-loop",
            "prefix_caching": False,
            "corpus_sha256": metadata["on"]["corpus_sha256"],
        },
        "timing": timing,
        "speed_credit": {
            "minimum_total_ratio": minimum_total_credit,
            "observed_total_ratio": total_ratio,
            "meets_minimum": total_ratio >= minimum_total_credit,
        },
        "outputs": {
            "trace_repetitions_equal": {
                mode: metadata[mode]["output_digests_equal"] for mode in MODES
            },
            "paired_online_generated_texts_equal": paired_outputs,
        },
        "traces": {
            mode: {
                key: value
                for key, value in kernels[mode].items()
                if key not in {"by_name", "file"}
            }
            for mode in MODES
        }
        | {
            "on_over_off_kernel_count": (
                kernels["on"]["kernel_count"] / kernels["off"]["kernel_count"]
            ),
            "on_over_off_kernel_time": (
                kernels["on"]["kernel_time_us"] / kernels["off"]["kernel_time_us"]
            ),
            "kernel_delta": _kernel_delta(
                kernels["on"]["by_name"], kernels["off"]["by_name"]
            ),
        },
        "raw_legs": {
            mode: {
                f"r{repetition}": {
                    key: value
                    for key, value in raw[mode][repetition].items()
                    if key != "file"
                }
                for repetition in REPETITIONS
            }
            for mode in MODES
        },
        "metadata": {
            mode: {key: value for key, value in metadata[mode].items() if key != "file"}
            for mode in MODES
        },
    }


def build_manifest(evidence: pathlib.Path) -> dict[str, Any]:
    files = []
    for path in sorted(evidence.rglob("*")):
        if not path.is_file() or path.name in DERIVED_NAMES:
            continue
        relative = path.relative_to(evidence).as_posix()
        files.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    if not files:
        raise HarnessError(f"no immutable artifacts under {evidence}")
    categories: dict[str, list[dict[str, Any]]] = {}
    for item in files:
        category = item["path"].split("/", 1)[0]
        categories.setdefault(category, []).append(item)
    return {
        "schema_version": 1,
        "files": files,
        "artifact_set_sha256": _digest_value(files),
        "category_sha256": {
            name: _digest_value(items) for name, items in sorted(categories.items())
        },
    }


def finalize(
    evidence: pathlib.Path,
    *,
    source_commit: str,
    finalizer: pathlib.Path,
    minimum_total_credit: float = 1.04,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    outputs = [
        evidence / "summary.json",
        evidence / "manifest.json",
        evidence / "series-complete.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise HarnessError(f"refusing to overwrite derived artifact(s): {existing}")
    summary = build_summary(
        evidence,
        source_commit=source_commit,
        minimum_total_credit=minimum_total_credit,
    )
    manifest = build_manifest(evidence)
    write_json_atomic(outputs[0], summary)
    write_json_atomic(outputs[1], manifest)
    marker = {
        "schema_version": 1,
        "status": "complete-diagnostic",
        "evidence_source_commit": source_commit,
        "finalizer_sha256": sha256_file(finalizer),
        "summary_sha256": sha256_file(outputs[0]),
        "manifest_sha256": sha256_file(outputs[1]),
        "artifact_set_sha256": manifest["artifact_set_sha256"],
    }
    write_json_atomic(outputs[2], marker)
    return summary, manifest, marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--minimum-total-credit", type=float, default=1.04)
    args = parser.parse_args()
    summary, _, marker = finalize(
        args.evidence,
        source_commit=args.source_commit,
        finalizer=pathlib.Path(__file__),
        minimum_total_credit=args.minimum_total_credit,
    )
    print(
        canonical_json(
            {
                "status": marker["status"],
                "observed_total_ratio": summary["speed_credit"][
                    "observed_total_ratio"
                ],
                "meets_minimum": summary["speed_credit"]["meets_minimum"],
                "summary_sha256": marker["summary_sha256"],
                "manifest_sha256": marker["manifest_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"finalize-async-credit: {error}", file=sys.stderr)
        raise SystemExit(2) from error
