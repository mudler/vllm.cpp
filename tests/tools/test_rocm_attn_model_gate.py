"""Reject missing production dispatches and altered attention model evidence."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

VALIDATOR = (
    Path(__file__).resolve().parents[2] / "tools/rocm_attn_wmma/validate_model.py"
)


class ModelGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.case = dict(name="request", prompt_token_ids=list(range(64)))
        self.result = dict(cases=[dict(self.case, output_token_ids=[7, 8, 9])])
        self.prefill = dict(
            Kernel_Name="PagedAttnPrefillSharedKWmma<2>",
            Workgroup_Size_X=512,
            Grid_Size_X=2048,
            Grid_Size_Y=4,
            Grid_Size_Z=1,
            Scratch_Size=0,
            VGPR_Count=100,
            SGPR_Count=100,
        )
        self.decode = dict(
            self.prefill,
            Kernel_Name="Rdna3DecodeWmma<16, 16>",
            Workgroup_Size_X=128,
            Grid_Size_X=512,
            Grid_Size_Y=1,
        )
        self.rows = [self.prefill.copy() for _ in range(2)] + [
            self.decode.copy() for _ in range(4)
        ]

    def run_gate(
        self, rows=None, actual=None, arm="wmma", graph=None, launches=1, captures=1
    ):
        files = {
            "manifest.json": dict(output_len=3, cases=[self.case]),
            "primary.json": self.result,
            "native.json": self.result if actual is None else actual,
        }
        for name, data in files.items():
            (self.root / name).write_text(json.dumps(data))
        with (self.root / "trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.prefill.keys())
            writer.writeheader()
            writer.writerows(self.rows if rows is None else rows)
        options = []
        if graph is not None:
            with (self.root / "hip.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["Function"])
                writer.writeheader()
                writer.writerows(
                    [dict(Function="hipGraphLaunch")] * launches
                    + [dict(Function="hipStreamEndCapture")] * captures
                )
            options = ["--graph", graph, "--hip-trace", str(self.root / "hip.csv")]
        return subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                *[str(self.root / name) for name in files],
                str(self.root / "trace.csv"),
                "--layers",
                "2",
                "--arm",
                arm,
                "--output",
                str(self.root / "receipt.json"),
                *options,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_complete_default_and_scalar(self):
        self.assertEqual(self.run_gate().returncode, 0)
        for row in self.rows[:2]:
            row.update(Kernel_Name="PagedAttnPrefillSharedK<2>", Grid_Size_X=1024)
        self.assertEqual(self.run_gate(arm="scalar").returncode, 0)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertEqual(receipt["decode_dispatches"], 4)

    def test_missing_prefill(self):
        result = self.run_gate(self.rows[1:])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prefill dispatches", result.stderr)

    def test_missing_decode(self):
        result = self.run_gate(self.rows[:-1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("decode dispatches", result.stderr)

    def test_decode_geometry(self):
        self.rows[-1]["Grid_Size_X"] = 128
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("decode dispatch geometry", result.stderr)

    def test_decode_scratch(self):
        self.rows[-1]["Scratch_Size"] = 24
        result = self.run_gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("decode dispatch uses scratch", result.stderr)

    def test_graph_replays_after_warmup(self):
        self.assertEqual(self.run_gate(graph="on").returncode, 0)

    def test_missing_graph_production_call(self):
        result = self.run_gate(graph="on", launches=0, captures=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("graph replay dispatch differs", result.stderr)

    def test_graph_not_reused(self):
        result = self.run_gate(graph="on", captures=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("graph replay dispatch differs", result.stderr)

    def test_graph_opt_out(self):
        self.assertEqual(
            self.run_gate(graph="off", launches=0, captures=0).returncode, 0
        )
        self.assertNotEqual(self.run_gate(graph="off").returncode, 0)

    def test_changed_completion(self):
        actual = copy.deepcopy(self.result)
        actual["cases"][0]["output_token_ids"][1] = 10
        result = self.run_gate(actual=actual)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output token IDs differ", result.stderr)


if __name__ == "__main__":
    unittest.main()
