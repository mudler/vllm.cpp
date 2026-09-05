#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-lease-ccache.py (#2473).

The defect this gate exists for costs 23 minutes per lease and reads as
compliance. `rc describe`'s usage sheet requires ccache and instructs that its
cache live on `/workspace`; `/workspace` is CIFS with `nounix`; ccache 4.9.1
takes every lock with `symlink(2)`, which that mount refuses with EOPNOTSUPP. So
every store AND every counter update fails, `ccache -s` reads zero of
everything, and the job pays a full cold build believing it did not.

TWO DIRECTIONS IN EVERY CASE, because a gate that only refuses is half a
contract. A script that puts the cache on local disk must come back green as
firmly as a `/workspace` cache comes back red. Fifteen scripts here configure
CMake inside a lease and the wrong recipe spreads by copying a neighbour, so the
green direction is what keeps the gate usable rather than routed around.

THE MUTATION THAT MATTERS is the one that deletes the checker's refusal. A gate
whose suite still passes with its own assertion removed measures nothing, and
`test_refusal_deleted_reds_the_suite` runs the checker with that line removed in
a scratch copy and requires the tree it guards to come back green -- which is
the failure the real defect wore.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-lease-ccache.py"
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"

EXAMINED = re.compile(r"examined ([0-9]+) shell scripts")

LOCAL_RECIPE = """#!/bin/bash
set -u
export CCACHE_DIR=/root/ccache
export CCACHE_REMOTE_STORAGE=file:/workspace/ccache-remote
cmake -S . -B b -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
"""

NAS_RECIPE = """#!/bin/bash
set -u
export CCACHE_DIR=/workspace/ccache
cmake -S . -B b -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
"""

NO_CCACHE = """#!/bin/bash
set -u
cmake -S . -B b
"""


class Base(unittest.TestCase):
    def run_checker(
        self, root: Path, checker: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(checker or CHECKER), "--root", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )

    def scratch(self, files: dict[str, str]) -> Path:
        root = Path(tempfile.mkdtemp(prefix="vllm-lease-ccache-"))
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (root / name).write_text(text, encoding="utf-8")
        return root


class RefusesTheNasCache(Base):
    def test_workspace_ccache_dir_is_refused(self) -> None:
        root = self.scratch({"scripts/lease.sh": NAS_RECIPE})
        got = self.run_checker(root)
        self.assertEqual(got.returncode, 1, got.stdout + got.stderr)
        self.assertIn("scripts/lease.sh", got.stdout + got.stderr)
        self.assertIn("symlink", (got.stdout + got.stderr).lower())

    def test_the_message_names_the_replacement(self) -> None:
        """A refusal that does not say what to do instead gets routed around."""
        root = self.scratch({"scripts/lease.sh": NAS_RECIPE})
        got = self.run_checker(root)
        out = got.stdout + got.stderr
        self.assertIn("CCACHE_REMOTE_STORAGE", out)

    def test_a_default_expansion_is_still_the_nas(self) -> None:
        """`${CCACHE_DIR:-/workspace/ccache}` is the shape the tree carried."""
        script = "#!/bin/bash\nexport CCACHE_DIR=${CCACHE_DIR:-/workspace/ccache}\n"
        root = self.scratch({"scripts/lease.sh": script})
        self.assertEqual(self.run_checker(root).returncode, 1)

    def test_a_quoted_path_is_still_the_nas(self) -> None:
        script = '#!/bin/bash\nexport CCACHE_DIR="/workspace/build-cache"\n'
        root = self.scratch({"scripts/lease.sh": script})
        self.assertEqual(self.run_checker(root).returncode, 1)


class AcceptsTheWorkingRecipe(Base):
    def test_local_ccache_dir_is_accepted(self) -> None:
        root = self.scratch({"scripts/lease.sh": LOCAL_RECIPE})
        got = self.run_checker(root)
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)

    def test_remote_storage_on_workspace_is_accepted(self) -> None:
        """The remote store MAY live on the NAS: ccache's `file:` backend uses
        open+rename, which this mount serves, and takes no symlink lock. A gate
        that refused the whole word `/workspace` would forbid the fix."""
        script = (
            "#!/bin/bash\nexport CCACHE_DIR=/root/cc\n"
            "export CCACHE_REMOTE_STORAGE=file:/workspace/ccache-remote\n"
        )
        root = self.scratch({"scripts/lease.sh": script})
        self.assertEqual(self.run_checker(root).returncode, 0)

    def test_a_script_with_no_ccache_is_accepted(self) -> None:
        root = self.scratch({"scripts/lease.sh": NO_CCACHE})
        self.assertEqual(self.run_checker(root).returncode, 0)

    def test_a_comment_is_not_a_setting(self) -> None:
        """This spec and these scripts DISCUSS `/workspace/ccache` at length."""
        script = "#!/bin/bash\n# never: export CCACHE_DIR=/workspace/ccache\n"
        root = self.scratch({"scripts/lease.sh": script})
        got = self.run_checker(root)
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)


class TheGateReadsSomething(Base):
    def test_it_reports_how_many_scripts_it_read(self) -> None:
        """A checker that silently examined zero files exits 0 and proves
        nothing. The count is the instrument's own precondition."""
        root = self.scratch(
            {"scripts/a.sh": LOCAL_RECIPE, "scripts/b.sh": NO_CCACHE}
        )
        got = self.run_checker(root)
        found = EXAMINED.search(got.stdout)
        self.assertIsNotNone(found, got.stdout)
        self.assertEqual(int(found.group(1)), 2)

    def test_an_empty_tree_is_refused_rather_than_passed(self) -> None:
        root = self.scratch({})
        got = self.run_checker(root)
        self.assertNotEqual(got.returncode, 0, got.stdout)


class TheShippedTree(Base):
    def test_the_tree_is_green(self) -> None:
        got = self.run_checker(ROOT)
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)

    def test_it_actually_read_the_lease_scripts(self) -> None:
        got = self.run_checker(ROOT)
        found = EXAMINED.search(got.stdout)
        self.assertIsNotNone(found, got.stdout)
        self.assertGreater(int(found.group(1)), 20, got.stdout)

    def test_preflight_runs_it(self) -> None:
        """A gate no lane runs is the same defect one step up."""
        self.assertIn("check-lease-ccache", PREFLIGHT.read_text(encoding="utf-8"))


class Mutations(Base):
    def mutate(self, drop: str) -> Path:
        """A scratch copy of the checker with one line removed."""
        text = CHECKER.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines(True) if drop not in ln]
        self.assertLess(len(lines), len(text.splitlines(True)), f"{drop!r} not found")
        out = Path(tempfile.mkdtemp(prefix="vllm-lease-ccache-mut-")) / "checker.py"
        out.write_text("".join(lines), encoding="utf-8")
        return out

    def test_refusal_deleted_reds_the_suite(self) -> None:
        """Delete the failure exit and a NAS cache comes back green.

        This is the shape of the real defect: the reference job printed its
        ccache counters and continued, and a dead cache reads exactly like a
        live one in a log that does not refuse."""
        mutant = self.mutate("return 1")
        root = self.scratch({"scripts/lease.sh": NAS_RECIPE})
        self.assertEqual(self.run_checker(root, mutant).returncode, 0)
        self.assertEqual(self.run_checker(root).returncode, 1)

    def test_comment_stripping_deleted_reds_the_green_direction(self) -> None:
        """Without it every document that discusses the defect becomes a red."""
        mutant = self.mutate("strip_comment")
        root = self.scratch(
            {"scripts/lease.sh": "#!/bin/bash\n# export CCACHE_DIR=/workspace/ccache\n"}
        )
        self.assertEqual(self.run_checker(root, mutant).returncode, 1)
        self.assertEqual(self.run_checker(root).returncode, 0)


class TheRefusalParsesTheREALCounters(Base):
    """The in-script refusal, run against the bytes the defect actually made.

    `scripts/ltx25-render-confirm.sh` decides whether the cache took part by
    reading `Cacheable calls` out of `ccache -s -v`. That decision is only worth
    anything if it fires on the output a DEAD cache produces and stays quiet on
    the output a live one produces, so both fixtures below are transcripts, not
    inventions:

      dead -- `ccache-after.txt` from rc job 93a60151, the 1404 s build. ccache
              4.x omits a section whose counters are all zero, so a cache that
              could not write a single counter prints ONLY its storage line.
      live -- the `remote-cold` arm of rc job e4793984.

    The extraction is read out of the shipped script rather than restated here.
    A transcription cannot gate the thing it transcribes.
    """

    SCRIPT = ROOT / "scripts/ltx25-render-confirm.sh"

    DEAD = "Local storage:\n  Cache size (GB): 0.0 / 20.0 ( 0.00%)\n"
    LIVE = (
        "Cache directory:    /tmp/ccp2-r2.180233\n"
        "Cacheable calls:      8 /   8 (100.0%)\n"
        "  Hits:               8 /   8 (100.0%)\n"
        "  Misses:             0 /   8 ( 0.00%)\n"
        "Local storage:\n"
        "  Cache size (GB):  0.0 / 5.0 ( 0.00%)\n"
    )

    def extraction(self) -> str:
        """The `sed` program the shipped script uses, taken from the script."""
        for line in self.SCRIPT.read_text(encoding="utf-8").splitlines():
            if "Cacheable calls" in line and "sed -n" in line:
                start = line.index("'") + 1
                return line[start : line.index("'", start)]
        self.fail("the script no longer extracts Cacheable calls with sed")

    def calls(self, text: str) -> str:
        path = Path(tempfile.mkdtemp(prefix="vllm-ccache-stats-")) / "s.txt"
        path.write_text(text, encoding="utf-8")
        got = subprocess.run(
            ["sed", "-n", self.extraction(), str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return got.stdout.strip().splitlines()[0] if got.stdout.strip() else ""

    def test_the_dead_cache_yields_no_count_so_the_refusal_fires(self) -> None:
        self.assertEqual(self.calls(self.DEAD), "")

    def test_the_live_cache_yields_its_count(self) -> None:
        self.assertEqual(self.calls(self.LIVE), "8")

    def test_the_script_refuses_on_zero_rather_than_logging_it(self) -> None:
        """The line this replaced printed the counters and carried on."""
        text = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ccache_cacheable_calls", text)
        self.assertRegex(text, r"cacheable calls[\s\S]{0,600}?exit 37")

    def test_the_script_probes_symlink_before_the_build(self) -> None:
        """The precondition that separates a usable CCACHE_DIR from a no-op."""
        text = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn(".symlink-probe", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
