#!/usr/bin/env python3
"""Black-box contract for the BACKEND-CPU PMU benchmark CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

if len(sys.argv) != 2:
    raise RuntimeError("expected the benchmark executable path")
BENCH_BINARY = Path(sys.argv.pop())


class CpuKernelBenchTests(unittest.TestCase):
    def run_bench(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(BENCH_BINARY), *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_json_schema_and_deterministic_checksum(self) -> None:
        args = (
            "--dtype", "q4_k", "--m", "1", "--n", "8", "--k", "256",
            "--warmup", "1", "--iterations", "2", "--counters", "off",
            "--format", "json",
        )
        first = self.run_bench(*args)
        second = self.run_bench(*args)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        got = json.loads(first.stdout)
        again = json.loads(second.stdout)
        self.assertEqual(got["schema"], "vllm-cpu-kernel-bench/v1")
        self.assertEqual(got["scope"], "vt-op")
        self.assertEqual(got["shape"], {"m": 1, "n": 8, "k": 256})
        self.assertTrue(got["timing"]["valid"])
        self.assertGreater(got["timing"]["calls_per_sample"], 1)
        self.assertEqual(got["checksum"], again["checksum"])
        self.assertEqual(got["counter_passes"], [])
        self.assertIn("compiler", got["metadata"])
        self.assertIn("throttle_before", got["metadata"])

    def test_invalid_dtype_fails_loudly(self) -> None:
        got = self.run_bench("--dtype", "invented")
        self.assertEqual(got.returncode, 2)
        self.assertIn("unsupported --dtype", got.stderr)

    def test_counter_unavailability_is_structured(self) -> None:
        got = self.run_bench(
            "--dtype", "q8_0", "--m", "1", "--n", "8", "--k", "256",
            "--warmup", "0", "--iterations", "1", "--counters", "a76",
            "--format", "json",
        )
        self.assertEqual(got.returncode, 0, got.stderr)
        passes = json.loads(got.stdout)["counter_passes"]
        self.assertGreaterEqual(len(passes), 1)
        self.assertIn(passes[0]["status"], {"ok", "partial", "multiplexed", "unsupported"})


if __name__ == "__main__":
    unittest.main()
