#!/usr/bin/env python3
"""The gate bring-up script resolves its machine values or refuses.

Issue #1190. `scripts/dgx-bringup.sh` used to hard-code one operator's
architecture and CUTLASS checkout as defaults. A second developer running it
built for the wrong architecture, and the CUTLASS default had already drifted
from the path `.agents/environment.md` records as mandatory on the very box the
file is named after. A configure that does not find CUTLASS drops the sm120a
NVFP4 GEMM and FlashAttention-2 without saying so, which is a false green
rather than a missing feature, so an unset value must refuse before the build.

The refusal is asserted BEFORE any configure step, so a failure here cannot
consume a GPU box or a build tree.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dgx-bringup.sh"
REQUIRED = ("DEVICE_ARCH", "CUTLASS_DIR")


def staged_copy(directory: Path, env_text: str | None) -> Path:
    """Copy the script into a scratch root so a scratch `.env` is the one read.

    The script resolves `.env` next to its own parent directory, so the copy
    keeps the `scripts/` level. Writing into the real checkout would clobber
    the developer's own untracked file.
    """
    (directory / "scripts").mkdir(parents=True, exist_ok=True)
    target = directory / "scripts" / SCRIPT.name
    shutil.copy2(SCRIPT, target)
    if env_text is not None:
        (directory / ".env").write_text(env_text, encoding="utf-8")
    return target


def blind_path(directory: Path) -> str:
    """A PATH whose `nvcc` always fails, so no configure or build can start."""
    binary = directory / "bin"
    binary.mkdir(parents=True, exist_ok=True)
    stub = binary / "nvcc"
    stub.write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")
    stub.chmod(0o755)
    return f"{binary}:{os.environ.get('PATH', '/usr/bin:/bin')}"


def run(target: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    clean = {
        key: value
        for key, value in os.environ.items()
        if key not in REQUIRED and key != "DEVICE_TOOLKIT_ROOT"
    }
    clean.update(env)
    return subprocess.run(
        ["bash", str(target)],
        cwd=target.parent.parent,
        env=clean,
        capture_output=True,
        text=True,
        check=False,
    )


class RefusalTests(unittest.TestCase):
    def test_each_required_value_refuses_by_name_before_any_build(self):
        for missing in REQUIRED:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                target = staged_copy(directory, None)
                supplied = {
                    key: "placeholder" for key in REQUIRED if key != missing
                }
                supplied["PATH"] = blind_path(directory)
                result = run(target, supplied)
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertIn(f"REFUSED: {missing} is unset", result.stderr)
                self.assertIn(".env", result.stderr)
                self.assertIn("Never substitute another", result.stderr)
                # The refusal is the FIRST thing that happens.
                self.assertNotIn("=== nvcc ===", result.stdout)
                self.assertFalse((directory / "build-cuda").exists())

    def test_no_hardcoded_fallback_survives_in_the_script(self):
        """A `${KEY:-literal}` default is what this row removed.

        The behavioral cases above cannot see a default that is merely wrong
        for another developer, because a wrong value still runs. This one reads
        the source, so reintroducing a resolved literal goes red.
        """
        text = SCRIPT.read_text(encoding="utf-8")
        code = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        for key in REQUIRED:
            # A NON-EMPTY default is the defect. `${KEY:-}` is how the shell's
            # own answer is saved before the file is sourced, and it supplies
            # nothing.
            self.assertIsNone(
                re.search(r"\$\{" + key + r":-[^}]", code),
                f"{key} carries a fallback value",
            )
        # One operator's resolved values, in the executable half. The comment
        # half may still name `121a` as an example of the vendor naming, which
        # is documentation and not a fallback.
        for literal in ("cutlass_probe", "121a", "/usr/local/cuda"):
            self.assertNotIn(literal, code)
        # A host name is wrong in either half, because a reader copies it.
        self.assertNotIn("dgx.casa", text)


class ResolutionTests(unittest.TestCase):
    def test_values_resolve_from_the_untracked_env_file(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            target = staged_copy(
                directory, "DEVICE_ARCH=121a\nCUTLASS_DIR=/scratch/cutlass\n"
            )
            result = run(target, {"PATH": blind_path(directory)})
            # Past both refusals and stopped at the toolchain probe, which the
            # stub PATH guarantees fails, so nothing configures or builds.
            self.assertNotIn("REFUSED", result.stderr)
            self.assertIn("=== nvcc ===", result.stdout)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertFalse((directory / "build-cuda").exists())

    def test_the_process_environment_wins_over_the_file(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            target = staged_copy(directory, "DEVICE_ARCH=\nCUTLASS_DIR=\n")
            supplied = {key: "from-shell" for key in REQUIRED}
            supplied["PATH"] = blind_path(directory)
            # An empty assignment in the file must not blank a value the shell
            # already answered, or a half-filled .env would refuse a session
            # that is correctly configured.
            result = run(target, supplied)
            self.assertNotIn("REFUSED", result.stderr)
            self.assertIn("=== nvcc ===", result.stdout)


if __name__ == "__main__":
    unittest.main()
