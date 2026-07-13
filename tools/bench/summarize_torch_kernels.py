#!/usr/bin/env python3
"""Aggregate GPU kernel events from vLLM torch-profiler Chrome traces.

The event selection follows the execution-trace recipe in
``.agents/parity-lever-protocol.md``: Chrome ``traceEvents`` whose category is
``kernel`` are grouped by actual runtime-resolved kernel name and duration.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from tools.bench.online_gate import MODEL_REVISIONS, _summarize_torch_trace
from tools.bench.serve_low_common import (
    HarnessError,
    canonical_json,
    write_json_atomic,
)

def summarize(
    profile_dir: pathlib.Path, *, model_key: str | None = None
) -> dict[str, Any]:
    traces = sorted(profile_dir.rglob("*.pt.trace.json"))
    traces.extend(sorted(profile_dir.rglob("*.pt.trace.json.gz")))
    if not traces:
        raise HarnessError(f"no torch-profiler traces under {profile_dir}")
    per_file = []
    selected: pathlib.Path | None = None
    selected_duration = -1.0
    selected_summary: dict[str, Any] | None = None
    for path in traces:
        summary = _summarize_torch_trace(path, model_key=model_key)
        total = summary["kernel_time_us"]
        per_file.append(
            {
                "file": str(path),
                "kernel_events": summary["kernel_count"],
                "kernel_time_us": total,
                "sha256": summary["selected_trace_sha256"],
            }
        )
        if total > selected_duration:
            selected = path
            selected_duration = total
            selected_summary = summary
    if selected is None or selected_duration <= 0.0 or selected_summary is None:
        raise HarnessError("torch-profiler output contains no positive-duration kernels")
    return {
        **selected_summary,
        "files": per_file,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-key", choices=tuple(MODEL_REVISIONS))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise HarnessError(f"refusing to overwrite kernel summary: {args.output}")
    result = summarize(args.profile_dir, model_key=args.model_key)
    write_json_atomic(args.output, result)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"summarize-torch-kernels: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
