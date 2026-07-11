"""Contracts for rootless, verified page-cache eviction."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from tools.bench.drop_file_cache import drop_file_cache


class DropFileCacheTests(unittest.TestCase):
    def test_posix_fadvise_evicts_all_inventoried_resident_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "first.bin"
            second = root / "second.bin"
            first.write_bytes(b"a" * (4 * 1024 * 1024))
            second.write_bytes(b"b" * (2 * 1024 * 1024))
            (root / "duplicate.bin").symlink_to(first)
            os.sync()
            first.read_bytes()
            second.read_bytes()

            report = drop_file_cache([root])

            self.assertTrue(report["succeeded"])
            self.assertEqual(report["file_count"], 2)
            self.assertEqual(report["duplicate_inode_count"], 1)
            self.assertGreater(report["resident_before_bytes"], 0)
            self.assertEqual(report["resident_after_bytes"], 0)
            self.assertRegex(report["file_inventory_sha256"], r"^[0-9a-f]{64}$")

    def test_cli_writes_failure_report_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "report.json"
            repo = pathlib.Path(__file__).resolve().parents[2]
            command = [
                sys.executable,
                "-m",
                "tools.bench.drop_file_cache",
                "--root",
                str(root / "missing"),
                "--output",
                str(output),
            ]
            result = subprocess.run(
                command,
                cwd=repo,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["succeeded"])
            self.assertIn("absent", report["error"])

            repeated = subprocess.run(
                command,
                cwd=repo,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                report,
            )


if __name__ == "__main__":
    unittest.main()
