#!/usr/bin/env python3
"""Sample aggregate process-tree RSS/PSS for unified-memory comparisons.

The common metric follows ``.agents/specs/cuda-sglang-low-concurrency.md``:
sum ``/proc/<pid>/smaps_rollup`` across the owned server tree and retain host
``MemAvailable``.  Numeric accelerator residency is optional; an ``N/A`` value
is represented as JSON ``null`` and is never converted to zero.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import time
from collections.abc import Iterable
from typing import Any

from tools.bench.serve_low_common import HarnessError, canonical_json


def _integer_prefix(value: str, field: str) -> int:
    token = value.strip().split()[0]
    try:
        return int(token)
    except (IndexError, ValueError) as error:
        raise HarnessError(f"invalid {field} value: {value!r}") from error


def read_process_memory(
    pid: int, proc_root: pathlib.Path = pathlib.Path("/proc")
) -> dict[str, int]:
    values: dict[str, int] = {}
    path = proc_root / str(pid) / "smaps_rollup"
    with path.open(encoding="utf-8") as source:
        for line in source:
            key, separator, value = line.partition(":")
            if separator and key in {"Rss", "Pss"}:
                values[f"{key.lower()}_kib"] = _integer_prefix(value, key)
    if "rss_kib" not in values or "pss_kib" not in values:
        raise HarnessError(f"{path}: missing Rss/Pss")
    return values


def read_mem_available(proc_root: pathlib.Path = pathlib.Path("/proc")) -> int:
    with (proc_root / "meminfo").open(encoding="utf-8") as source:
        for line in source:
            key, separator, value = line.partition(":")
            if separator and key == "MemAvailable":
                return _integer_prefix(value, key)
    raise HarnessError(f"{proc_root / 'meminfo'}: missing MemAvailable")


def _direct_children(pid: int, proc_root: pathlib.Path) -> list[int]:
    path = proc_root / str(pid) / "task" / str(pid) / "children"
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    children = []
    for value in text.split():
        try:
            children.append(int(value))
        except ValueError:
            continue
    return children


def _is_live_process(pid: int, proc_root: pathlib.Path) -> bool:
    process = proc_root / str(pid)
    if not process.exists():
        return False
    try:
        stat = (process / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        # Synthetic proc fixtures need not reproduce the kernel stat file. A
        # real process that races away is caught again while reading smaps.
        return process.exists()
    close = stat.rfind(")")
    if close < 0:
        return True
    fields = stat[close + 1 :].split()
    return not fields or fields[0] != "Z"


def process_tree(root_pid: int, proc_root: pathlib.Path = pathlib.Path("/proc")) -> list[int]:
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen or not _is_live_process(pid, proc_root):
            continue
        seen.add(pid)
        pending.extend(_direct_children(pid, proc_root))
    return sorted(seen)


def cgroup_processes(cgroup: pathlib.Path | None) -> list[int]:
    if cgroup is None:
        return []
    files = [cgroup] if cgroup.is_file() else sorted(cgroup.rglob("cgroup.procs"))
    pids: set[int] = set()
    for path in files:
        try:
            values = path.read_text(encoding="utf-8").split()
        except (FileNotFoundError, PermissionError):
            continue
        for value in values:
            try:
                pids.add(int(value))
            except ValueError:
                continue
    return sorted(pids)


def query_gpu_memory_mib(pids: Iterable[int]) -> int | None:
    target = set(pids)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    total = 0
    found = False
    for line in result.stdout.splitlines():
        left, separator, right = line.partition(",")
        if not separator:
            continue
        try:
            pid = int(left.strip())
        except ValueError:
            continue
        if pid not in target:
            continue
        found = True
        value = right.strip()
        if value.upper() in {"N/A", "[N/A]", "NA"}:
            return None
        try:
            total += int(value)
        except ValueError:
            return None
    return total if found else 0


def sample_once(
    root_pid: int,
    *,
    proc_root: pathlib.Path = pathlib.Path("/proc"),
    include_gpu: bool = False,
    cgroup: pathlib.Path | None = None,
) -> dict[str, Any]:
    roots = [root_pid, *cgroup_processes(cgroup)]
    pids = sorted(
        {
            pid
            for root in roots
            for pid in process_tree(root, proc_root)
        }
    )
    rss_kib = 0
    pss_kib = 0
    sampled: list[int] = []
    vanished: list[int] = []
    for pid in pids:
        try:
            memory = read_process_memory(pid, proc_root)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            vanished.append(pid)
            continue
        rss_kib += memory["rss_kib"]
        pss_kib += memory["pss_kib"]
        sampled.append(pid)
    return {
        "alive": bool(pids),
        "gpu_memory_mib": query_gpu_memory_mib(sampled) if include_gpu else None,
        "mem_available_kib": read_mem_available(proc_root),
        "pids": sampled,
        "pss_kib": pss_kib,
        "rss_kib": rss_kib,
        "vanished_pids": vanished,
    }


def run_sampler(
    root_pid: int,
    output: pathlib.Path,
    *,
    interval_s: float = 0.1,
    duration_s: float | None = None,
    include_gpu: bool = False,
    proc_root: pathlib.Path = pathlib.Path("/proc"),
    cgroup: pathlib.Path | None = None,
) -> dict[str, Any]:
    if interval_s <= 0.0:
        raise HarnessError("sampling interval must be positive")
    if duration_s is not None and duration_s <= 0.0:
        raise HarnessError("duration must be positive")
    if output.exists():
        raise HarnessError(f"refusing to overwrite memory evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    baseline_available: int | None = None
    peak_rss = 0
    peak_pss = 0
    peak_available_drop = 0
    samples = 0
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        while True:
            sample = sample_once(
                root_pid,
                proc_root=proc_root,
                include_gpu=include_gpu,
                cgroup=cgroup,
            )
            now = time.monotonic()
            baseline_available = (
                sample["mem_available_kib"]
                if baseline_available is None
                else baseline_available
            )
            peak_rss = max(peak_rss, sample["rss_kib"])
            peak_pss = max(peak_pss, sample["pss_kib"])
            peak_available_drop = max(
                peak_available_drop,
                baseline_available - sample["mem_available_kib"],
            )
            sample.update(
                elapsed_s=now - start,
                peak_mem_available_drop_kib=peak_available_drop,
                peak_pss_kib=peak_pss,
                peak_rss_kib=peak_rss,
                timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            sink.write(canonical_json(sample) + "\n")
            sink.flush()
            samples += 1
            if not sample["alive"]:
                break
            if duration_s is not None and now - start >= duration_s:
                break
            time.sleep(interval_s)
    return {
        "peak_mem_available_drop_kib": peak_available_drop,
        "peak_pss_kib": peak_pss,
        "peak_rss_kib": peak_rss,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument(
        "--cgroup",
        type=pathlib.Path,
        help="owned cgroup directory or cgroup.procs file (needed for containers)",
    )
    args = parser.parse_args()
    summary = run_sampler(
        args.pid,
        args.output,
        interval_s=args.interval,
        duration_s=args.duration,
        include_gpu=args.include_gpu,
        cgroup=args.cgroup,
    )
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"process-memory: {error}", file=os.sys.stderr)
        raise SystemExit(2) from error
