#!/usr/bin/env python3
"""The GPU file mutex has exactly ONE truth: `${GPU_LOCK:-$HOME/gpu.lock}`.

Issue #777. This repo carried TWO GPU mutexes. The documented procedure
(`.env.example`, `.agents/environment.md`, `.agents/coordination.md`) named
`${GPU_LOCK}`, `.env.example` shipped that variable EMPTY, and
`.agents/coordination.md` recorded the developer profile's value as `/tmp/gpu` --
while the harness scripts hardcoded `$HOME/gpu.lock`. Follow the docs and you
took one file; run a harness and you took another. The two never serialised.

No runtime check can catch that. A `flock` on the wrong file SUCCEEDS -- that is
what a mutex does -- so the failure is silent and surfaces only as timing noise
that gets attributed to whoever else was on the box. It cost a full standalone
Marlin benchmark series (every absolute downgraded to an upper bound in
`.agents/benchmark-record.md`), and `.agents/specs/muse-glimmer.md` records an
earlier bite: "That job also did NOT hold `${GPU_LOCK}`, so the mutex protected
nothing."

So the invariant is asserted STRUCTURALLY here:

  A. No script names a bare GPU-lock path. `/tmp/gpu` is gone entirely; the
     `$HOME/gpu.lock` default may only appear on a line that also names
     `GPU_LOCK`, so the path is never readable without the variable that
     resolves it.
  B. A script that TAKES a lock must spell the canonical default, so the
     variable and the script cannot disagree.
  C/D/E. The three instruction surfaces -- `.env.example` and the two `.agents/`
     guides -- name the variable AND its default, so a `.env` that predates the
     fix is still repairable from the docs alone.

Historical records (`.agents/benchmark-record.md`, `.agents/parity-ledger.md`,
`state-events/`, and every spec recording a past run) are deliberately NOT
scanned: they say what was actually used at the time and are evidence. Rewriting
them would falsify the record.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ENV_EXAMPLE = ROOT / ".env.example"
COORDINATION = ROOT / ".agents/coordination.md"
ENVIRONMENT = ROOT / ".agents/environment.md"

# The one truth, in the two spellings this repo needs.
CANONICAL_SHELL = "${GPU_LOCK:-$HOME/gpu.lock}"
CANONICAL_ENV_LINE = "GPU_LOCK=$HOME/gpu.lock"

# The abandoned mutex. Nothing under scripts/ or tools/ may name it again.
RETIRED_PATH = "/tmp/gpu"

# The default path, in every spelling. Legal only alongside `GPU_LOCK`.
_DEFAULT_PATH = re.compile(r"(?:\$HOME/gpu\.lock|\$\{HOME\}/gpu\.lock|~/gpu\.lock)")

# A `flock` that actually takes something: a path, a quoted/expanded word, or a
# file descriptor (`flock 9`, paired with `exec 9>`). A prose "no flock needed"
# has no target and is not an invocation.
_FLOCK_CALL = re.compile(r"(?<![\w./-])flock\s+(?:-[\w-]+\s+)*([\"'$/~]\S*|\d+)")
# The fd form's other half: `exec 9>/tmp/gpu` is where the path really lives.
_EXEC_REDIRECT = re.compile(r"(?<![\w./-])exec\s+\d+>\s*(\S+)")


def scanned_scripts() -> list[Path]:
    """Every tracked shell/Python file under scripts/ and tools/."""
    out = subprocess.run(
        ["git", "ls-files", "scripts", "tools"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [
        ROOT / p
        for p in out
        if p.endswith((".sh", ".py")) and (ROOT / p).is_file()
    ]


def bare_path_offenders(path: Path, text: str) -> list[str]:
    """Lines naming a GPU-lock path with no `GPU_LOCK` to resolve it."""
    offenders: list[str] = []
    rel = path.relative_to(ROOT)
    for number, line in enumerate(text.splitlines(), start=1):
        if RETIRED_PATH in line:
            offenders.append(f"{rel}:{number}: names the retired mutex {RETIRED_PATH}")
        elif _DEFAULT_PATH.search(line) and "GPU_LOCK" not in line:
            offenders.append(
                f"{rel}:{number}: names the lock path with no GPU_LOCK beside it"
            )
    return offenders


def takes_a_lock(text: str) -> bool:
    """Does this file invoke `flock`, or open the fd one is taken on?"""
    return bool(_FLOCK_CALL.search(text) or _EXEC_REDIRECT.search(text))


def section(text: str, heading: str) -> str:
    """The body under `heading`, up to the next same-or-higher heading."""
    match = re.search(rf"(?m)^(#{{1,6}})\s*{re.escape(heading)}\s*$", text)
    assert match is not None, f"missing heading: {heading}"
    depth = len(match.group(1))
    rest = text[match.end() :]
    nxt = re.search(rf"(?m)^#{{1,{depth}}}\s", rest)
    return rest[: nxt.start()] if nxt else rest


class GpuLockOneTruthTests(unittest.TestCase):
    def test_no_script_names_a_bare_gpu_lock_path(self) -> None:
        """Rule A. A path readable without its variable is half a mutex."""
        offenders: list[str] = []
        for path in scanned_scripts():
            text = path.read_text(encoding="utf-8", errors="replace")
            offenders.extend(bare_path_offenders(path, text))
        self.assertEqual(
            offenders,
            [],
            "the GPU lock has one truth, `"
            + CANONICAL_SHELL
            + "`:\n  " + "\n  ".join(offenders),
        )

    def test_every_script_that_locks_spells_the_canonical_default(self) -> None:
        """Rule B. You cannot take the GPU lock without resolving GPU_LOCK."""
        offenders: list[str] = []
        for path in scanned_scripts():
            text = path.read_text(encoding="utf-8", errors="replace")
            if not takes_a_lock(text):
                continue
            if "GPU_LOCK" not in text:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(
            offenders,
            [],
            "these take a lock without resolving GPU_LOCK: " + ", ".join(offenders),
        )

    def test_env_example_ships_the_default_rather_than_a_blank(self) -> None:
        """Rule C. A blank is what let a filled-in `.env` quietly diverge."""
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        self.assertIn(
            CANONICAL_ENV_LINE,
            text.splitlines(),
            ".env.example must ship `" + CANONICAL_ENV_LINE + "`, never `GPU_LOCK=`",
        )

    def test_coordination_names_the_one_lock_and_not_the_retired_one(self) -> None:
        """Rule D. coordination.md was the ROOT: it documented `/tmp/gpu`."""
        body = section(COORDINATION.read_text(encoding="utf-8"), "GPU scheduling")
        self.assertNotIn(
            RETIRED_PATH,
            body,
            "coordination.md's GPU scheduling section still names the retired mutex",
        )
        self.assertIn("${GPU_LOCK}", body)
        self.assertIn("$HOME/gpu.lock", body)

    def test_environment_states_the_variable_and_its_default(self) -> None:
        """Rule E. Usable by someone whose `.env` predates this change."""
        text = ENVIRONMENT.read_text(encoding="utf-8")
        mutex = [line for line in text.splitlines() if "**GPU mutex:**" in line]
        self.assertEqual(len(mutex), 1, "expected exactly one GPU mutex bullet")
        start = text.index(mutex[0])
        # The bullet runs to the next bullet at the same indent.
        rest = text[start:]
        nxt = re.search(r"(?m)^  - ", rest[len(mutex[0]) :])
        bullet = rest[: len(mutex[0]) + nxt.start()] if nxt else rest
        self.assertIn("${GPU_LOCK}", bullet)
        self.assertIn(
            "$HOME/gpu.lock",
            bullet,
            "the bullet must state the DEFAULT too, or a stale .env stays wrong",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1)
