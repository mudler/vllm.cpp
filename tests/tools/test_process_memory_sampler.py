"""Process-tree memory sampler tests for unified-memory benchmark evidence."""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from tools.bench.sample_process_memory import (
    process_tree,
    query_gpu_memory_mib,
    run_sampler,
    sample_once,
)


def _process(root: pathlib.Path, pid: int, *, children: str, rss: int, pss: int) -> None:
    process = root / str(pid)
    task = process / "task" / str(pid)
    task.mkdir(parents=True)
    (task / "children").write_text(children)
    (process / "smaps_rollup").write_text(
        f"Rss: {rss} kB\nPss: {pss} kB\n"
    )


class MemorySamplerTests(unittest.TestCase):
    def test_parent_children_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proc = pathlib.Path(temporary)
            (proc / "meminfo").write_text("MemAvailable: 9000 kB\n")
            _process(proc, 10, children="20 30", rss=100, pss=80)
            _process(proc, 20, children="", rss=50, pss=40)
            _process(proc, 30, children="40", rss=25, pss=20)
            _process(proc, 40, children="", rss=10, pss=5)
            _process(proc, 50, children="", rss=30, pss=25)
            cgroup = proc / "owned-cgroup"
            cgroup.mkdir()
            (cgroup / "cgroup.procs").write_text("10\n50\n")
            self.assertEqual(process_tree(10, proc), [10, 20, 30, 40])
            sample = sample_once(10, proc_root=proc, cgroup=cgroup)
            self.assertEqual(sample["rss_kib"], 215)
            self.assertEqual(sample["pss_kib"], 170)
            self.assertEqual(sample["mem_available_kib"], 9000)
            self.assertEqual(sample["pids"], [10, 20, 30, 40, 50])

    def test_gpu_na_is_null_not_zero(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="10, N/A\n", stderr=""
        )
        with mock.patch("subprocess.run", return_value=completed):
            self.assertIsNone(query_gpu_memory_mib([10]))
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="10, 123\n11, 7\n", stderr=""
        )
        with mock.patch("subprocess.run", return_value=completed):
            self.assertEqual(query_gpu_memory_mib([10, 11]), 130)

    def test_short_lived_pid_finishes_with_valid_jsonl_and_monotonic_peaks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "memory.jsonl"
            process = subprocess.Popen(
                ["python3", "-c", "import time; x=bytearray(1024*1024); time.sleep(.08)"]
            )
            summary = run_sampler(
                process.pid,
                output,
                interval_s=0.01,
                duration_s=1.0,
            )
            process.wait()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertGreaterEqual(summary["samples"], 2)
            self.assertFalse(rows[-1]["alive"])
            peaks = [row["peak_rss_kib"] for row in rows]
            self.assertEqual(peaks, sorted(peaks))
            self.assertTrue(all(row["gpu_memory_mib"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
