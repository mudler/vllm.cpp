#!/usr/bin/env python3
"""The SGLang oracle identity gate detects what it claims to detect.

`scripts/sglang_lease_identity.py` is the assertion that stands in for a runtime
commit check, because `sglang/_version.py` in the PyPI wheel sets
`__commit_id__ = None`. The gate runs inside an `rc` lease against a GB10, which
CI has none of -- so what CI CAN hold is that the gate goes RED on each defect it
promises to catch, and that the committed manifest is internally consistent.

Row `SGLANG-ORACLE-LEASE-WHEEL`, spec `.agents/specs/sglang-wheel-in-lease.md`,
issue #1265.

Each mutation is applied to a SCRATCH fixture and reverted, and the suite
re-asserts the green case afterwards, so a mutation that silently failed to
apply cannot read as a passing test.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts/sglang_lease_identity.py"
MANIFEST = ROOT / ".agents/specs/sglang-wheel-in-lease.json"
ORACLE = ROOT / ".agents/oracles/sglang.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManifestShapeTests(unittest.TestCase):
    """The committed manifest is the thing the gate compares against.

    Reading it here is also what stops it landing unreached: no other executing
    code in this tree opens it.
    """

    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST.read_text())

    def test_root_and_exclude_are_what_the_gate_assumes(self) -> None:
        self.assertEqual(self.manifest["root"], "sglang/")
        self.assertEqual(self.manifest["exclude"], ["__pycache__/"])

    def test_file_count_agrees_with_the_file_table(self) -> None:
        self.assertEqual(self.manifest["file_count"], len(self.manifest["files"]))

    def test_every_key_is_under_the_declared_root(self) -> None:
        root = self.manifest["root"]
        offenders = [k for k in self.manifest["files"] if not k.startswith(root)]
        self.assertEqual(offenders[:5], [], f"{len(offenders)} keys outside {root}")

    def test_no_key_names_an_excluded_directory(self) -> None:
        offenders = [k for k in self.manifest["files"] if "__pycache__/" in k]
        self.assertEqual(offenders[:5], [])

    def test_every_value_is_a_sha256(self) -> None:
        bad = [
            k
            for k, v in self.manifest["files"].items()
            if len(v) != 64 or not all(c in "0123456789abcdef" for c in v)
        ]
        self.assertEqual(bad[:5], [])

    def test_the_manifest_pin_is_the_oracle_pin(self) -> None:
        """A manifest under a different pin would assert the wrong tree."""
        text = ORACLE.read_text()
        pins = [
            line.split("=", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("pin =")
        ]
        self.assertEqual(len(pins), 1, "expected exactly one `pin =` in the pin block")
        self.assertEqual(self.manifest["pin"], pins[0])
        self.assertEqual(self.manifest["oracle_id"], "sglang")

    def test_the_wheel_hashes_are_the_ones_the_oracle_records(self) -> None:
        text = ORACLE.read_text()
        for key in ("wheel", "kernel_wheel"):
            entry = self.manifest[key]
            self.assertIn(entry["sha256"], text, f"{key} sha256 absent from {ORACLE}")
            self.assertIn(entry["filename"], text, f"{key} filename absent")


class GateDetectionTests(unittest.TestCase):
    """Mutate each guarantee. Reading the gate is not proving it."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sglang-identity-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.site = self.tmp / "site-packages"
        self.pkg = self.site / "sglang"
        (self.pkg / "srt" / "__pycache__").mkdir(parents=True)
        (self.pkg / "__init__.py").write_text('__version__ = "0.5.15"\n')
        (self.pkg / "srt" / "a.py").write_text("X = 1\n")
        # pip writes these and the wheel does not carry them. The gate must
        # ignore them, so the fixture has one.
        (self.pkg / "srt" / "__pycache__" / "a.cpython-312.pyc").write_text("junk\n")
        files = {}
        for dirpath, dirnames, filenames in os.walk(self.pkg):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                p = Path(dirpath) / name
                files["sglang/" + p.relative_to(self.pkg).as_posix()] = _sha256(p)
        self.manifest_path = self.tmp / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "pin": "TESTPIN",
                    "root": "sglang/",
                    "exclude": ["__pycache__/"],
                    "file_count": len(files),
                    "files": files,
                }
            )
        )

    def run_gate(self, pythonpath: Path | None = None, cwd: str = "/"):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(pythonpath or self.site)
        return subprocess.run(
            [sys.executable, str(GATE), "--manifest", str(self.manifest_path)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )

    def assert_green(self) -> None:
        proc = self.run_gate()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("IDENTITY OK", proc.stdout)

    def test_unmutated_fixture_is_green(self) -> None:
        self.assert_green()

    def test_pycache_is_ignored_rather_than_reported_extra(self) -> None:
        proc = self.run_gate()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("derived_files=2", proc.stdout)

    def test_one_changed_byte_is_red(self) -> None:
        target = self.pkg / "srt" / "a.py"
        original = target.read_bytes()
        target.write_text("X = 2\n")
        self.assertNotEqual(target.read_bytes(), original, "the mutation never applied")
        proc = self.run_gate()
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("DIFFERING: sglang/srt/a.py", proc.stdout)
        target.write_bytes(original)
        self.assert_green()

    def test_a_missing_file_is_red(self) -> None:
        target = self.pkg / "srt" / "a.py"
        original = target.read_bytes()
        target.unlink()
        self.assertFalse(target.exists(), "the mutation never applied")
        proc = self.run_gate()
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("MISSING: sglang/srt/a.py", proc.stdout)
        target.write_bytes(original)
        self.assert_green()

    def test_an_extra_file_is_red(self) -> None:
        extra = self.pkg / "srt" / "b.py"
        extra.write_text("Y = 2\n")
        self.assertTrue(extra.exists(), "the mutation never applied")
        proc = self.run_gate()
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("EXTRA: sglang/srt/b.py", proc.stdout)
        extra.unlink()
        self.assert_green()

    def test_running_outside_root_is_refused(self) -> None:
        """`cd /` is what stops the gate reading a source checkout on sys.path."""
        proc = self.run_gate(cwd=str(self.tmp))
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("run this from /", proc.stderr)

    def test_a_source_tree_is_refused_even_when_its_bytes_match(self) -> None:
        checkout = self.tmp / "checkout"
        checkout.mkdir()
        shutil.copytree(self.pkg, checkout / "sglang")
        proc = self.run_gate(pythonpath=checkout)
        self.assertEqual(proc.returncode, 4, proc.stdout + proc.stderr)
        self.assertIn("not an installed package", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
