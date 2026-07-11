#!/usr/bin/env python3
"""Aggregate GPU kernel events from vLLM torch-profiler Chrome traces.

The event selection follows the execution-trace recipe in
``.agents/parity-lever-protocol.md``: Chrome ``traceEvents`` whose category is
``kernel`` are grouped by actual runtime-resolved kernel name and duration.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
from collections import defaultdict
from typing import Any

from tools.bench.serve_low_common import (
    HarnessError,
    canonical_json,
    require_number,
    sha256_file,
    write_json_atomic,
)


def _read_trace(path: pathlib.Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else pathlib.Path.open
    try:
        if path.suffix == ".gz":
            with opener(path, "rt", encoding="utf-8") as source:
                value = json.load(source)
        else:
            with opener(path, "r", encoding="utf-8") as source:
                value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"{path}: invalid profiler trace: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("traceEvents"), list):
        raise HarnessError(f"{path}: traceEvents array is absent")
    return value


def summarize(profile_dir: pathlib.Path) -> dict[str, Any]:
    traces = sorted(profile_dir.rglob("*.pt.trace.json"))
    traces.extend(sorted(profile_dir.rglob("*.pt.trace.json.gz")))
    if not traces:
        raise HarnessError(f"no torch-profiler traces under {profile_dir}")
    per_file = []
    selected: pathlib.Path | None = None
    selected_duration = -1.0
    selected_kernels: dict[str, tuple[int, float]] = {}
    for path in traces:
        grouped: dict[str, list[float]] = defaultdict(list)
        for event in _read_trace(path)["traceEvents"]:
            if not isinstance(event, dict):
                continue
            category = str(event.get("cat", "")).lower()
            name = event.get("name")
            if "kernel" not in category or not isinstance(name, str) or not name:
                continue
            duration = require_number(event.get("dur"), f"{path}:kernel duration")
            if duration >= 0.0:
                grouped[name].append(duration)
        total = sum(sum(values) for values in grouped.values())
        per_file.append(
            {
                "file": str(path),
                "kernel_events": sum(len(values) for values in grouped.values()),
                "kernel_time_us": total,
                "sha256": sha256_file(path),
            }
        )
        if total > selected_duration:
            selected = path
            selected_duration = total
            selected_kernels = {
                name: (len(values), sum(values)) for name, values in grouped.items()
            }
    if selected is None or selected_duration <= 0.0 or not selected_kernels:
        raise HarnessError("torch-profiler output contains no positive-duration kernels")
    kernels = sorted(
        (
            {
                "count": count,
                "name": name,
                "percent": duration / selected_duration * 100.0,
                "total_duration_us": duration,
            }
            for name, (count, duration) in selected_kernels.items()
        ),
        key=lambda item: (-item["total_duration_us"], item["name"]),
    )
    return {
        "files": per_file,
        "kernel_count": sum(item["count"] for item in kernels),
        "kernel_time_us": selected_duration,
        "kernels": kernels,
        "selected_trace": str(selected),
        "selected_trace_sha256": sha256_file(selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise HarnessError(f"refusing to overwrite kernel summary: {args.output}")
    result = summarize(args.profile_dir)
    write_json_atomic(args.output, result)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"summarize-torch-kernels: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
