"""Exercise the lease-2 clock reproduction command without a GPU."""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "docs/bench-evidence/qwen38-27b-q4km-gfx1151-ourarm-head-20260905/clock_windows.py"


class ClockWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tags = [f"n{n}-r{r}" for n in (64, 128) for r in range(1, 5)]
        self.rows = []
        for run in range(1, 5):
            for timestamp in (10 * run, 10 * run + 2):
                self.rows.append(dict(timestamp=timestamp, sclk_mhz=100 * run, busy_percent=10 * run))
            for timestamp in (10 * run - .5, 10 * run + 2.5):
                self.rows.append(dict(timestamp=timestamp, sclk_mhz=9999, busy_percent=99))
        self.rows.sort(key=lambda sample: sample["timestamp"])
        for tag in self.tags:
            (self.root / f"{tag}.err.txt").write_text("\n".join(
                f"vllm-cli: run={run}/4 generate_start_unix={run * 10}.000000 generate_end_unix={run * 10 + 2}.000000"
                for run in range(1, 5)))
            self.write_samples(tag, self.rows)

    def write_samples(self, tag, rows):
        (self.root / f"clock-{tag}.jsonl.gz").write_bytes(gzip.compress(
            "\n".join(json.dumps(row) for row in rows).encode()))

    def command(self):
        return subprocess.run([sys.executable, str(SCRIPT), "--evidence-dir", str(self.root)],
                              capture_output=True, text=True, timeout=10)

    def test_cli_keeps_warm_samples_inside_inclusive_bounds(self):
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["pooled_warm"]["samples"], 48)
        self.assertEqual(report["pooled_warm"]["sclk_mhz_mean"], 300)
        self.assertEqual(report["pooled_warm"]["busy_percent_mean"], 30)
        self.assertEqual(report["pooled_whole_leg"]["samples"], 128)
        self.assertEqual(len(report["legs"]), 8)
        for leg in report["legs"]:
            self.assertEqual(leg["warm"]["samples"], 6)
            self.assertEqual([run["samples"] for run in leg["generations"]], [2] * 4)
            self.assertEqual([run["sclk_mhz_mean"] for run in leg["generations"]], [100, 200, 300, 400])
        self.assertIn("not occupancy", report["interpretation"])
        self.assertEqual(report["token_gate"], "FAIL (carried, not remeasured)")

    def test_cli_pools_samples_instead_of_leg_means(self):
        self.write_samples(self.tags[0], self.rows + [dict(timestamp=21, sclk_mhz=800, busy_percent=80)])
        result = self.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["pooled_warm"]["samples"], 49)
        self.assertAlmostEqual(report["pooled_warm"]["sclk_mhz_mean"], (48 * 300 + 800) / 49)

    def test_cli_rejects_missing_or_overlapping_window(self):
        path = self.root / f"{self.tags[0]}.err.txt"
        original = path.read_text()
        for bad in (original.split("vllm-cli: run=4")[0],
                    original.replace("run=4/4", "run=3/4"),
                    original.replace("generate_start_unix=20", "generate_start_unix=11")):
            with self.subTest(bad=bad):
                path.write_text(bad)
                result = self.command()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("generation windows", result.stderr)

    def test_cli_rejects_window_without_samples(self):
        self.write_samples(self.tags[0], [row for row in self.rows if not 20 <= row["timestamp"] <= 22])
        result = self.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no clock samples", result.stderr)

    def test_cli_rejects_invalid_sample(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["sclk_mhz"] = float("nan")
        self.write_samples(self.tags[0], rows)
        result = self.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid clock sample", result.stderr)


if __name__ == "__main__":
    unittest.main()
