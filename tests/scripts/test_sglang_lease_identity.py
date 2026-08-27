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

import functools
import hashlib
import json
import os
import re
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

# The records that quote the manifest population as MEASURED are DERIVED, never
# listed. Any markdown in this corpus that states the population in one of the
# shapes below is bound to the literal, so a FOURTH record quoting a wrong count
# is not invisible: the fresh review of this row swept the tree and found the
# figure in five files against an allowlist of three.
RECORD_GLOBS = (".agents/**/*.md", "docs/**/*.md", "*.md")

# `.agents/issue-index.md` states the population too, and is EXCLUDED because it
# is append-only by policy: its rows are history, GitHub holds their state, and
# a pin advance cannot rewrite them. Excluding it is a policy boundary, not a
# licence for it to be wrong.
APPEND_ONLY_RECORDS = (ROOT / ".agents/issue-index.md",)

# The three records #1832 names. The sweep must still FIND each of them: a
# record that stops quoting the population, or that is renamed away, reds here
# rather than silently shrinking the swept set.
QUOTING_RECORDS = (
    ROOT / ".agents/environment.md",
    ORACLE,
    ROOT / ".agents/sglang-matrix.md",
)

_N = r"([0-9][0-9,]*)"  # a decimal, carrying the thousands separator prose uses
_B = r"\*{0,2}"  # markdown bold around a figure

# A number stands in a POPULATION SLOT when the prose reads it AS the file
# population of the installed `sglang/` tree. EVERY capture of EVERY slot is
# asserted, which is the half the presence test that stood here could not do:
# `(?<!\d)3338(?!\d)` anywhere in a record was satisfied by any ONE correct
# occurrence, so `.agents/oracles/sglang.md` could read `**3337 of 3338**` -- a
# manifest MISMATCH, the exact defect this gate exists to catch -- and stay
# green, and so could `3337 files of the installed` in `../sglang-matrix.md`.
#
# The slots read this manifest's VOCABULARY, and only two of them read its
# IDENTITY: `files in the wheel's \`sglang/\` tree | N` and `N of N files
# against the committed manifest` name the thing they count, while the other ten
# read any number written in the same phrase shape and compare it to the literal
# below. That DEFERS the false-positive class the broad alternatives were
# rejected for rather than avoiding it. A rule broad enough to read every
# `N files` fires on the 287-file and 81-file populations two other rows measure
# TODAY, and a near-miss band fires on the 3335 and 3336 source-tarball figures
# the lease spec carries a table of; these ten fire on the same populations only
# once someone writes them in one of these ten shapes, and prose a future row
# could plausibly write -- `the staged 81-file manifest`, `287 of 287, zero
# missing`, `manifest_files=81`, `IDENTITY OK: 81 files` -- reds here. `## Risks`
# in `.agents/specs/gate-sglang-manifest-and-suite-registration.md` records that
# bound in both directions.
#
# The corpus moves, so a count of it names its tree. At `16ebcac4b` these globs
# reach 888 markdown files and sweep 887 -- all but the append-only index -- and
# ELEVEN of these twelve shapes select 22 figures in six files and nothing else.
# The twelfth selects nothing there, and the comment beside it says why it is
# kept. Re-derive rather than trust: `_records_stating_a_population()` returns
# the breakdown per record.
#
# The separator is stripped before comparing, so `3,338` is a CORRECT statement
# of the count and passes. It read as a defect while the assertion looked for
# one exact decimal.
#
# Every gap between tokens is `\s`, never a literal space, because these records
# are hard-wrapped at about 78 columns. `3338-file manifest` re-wrapped to
# `3338-file` + newline + `manifest` is an ordinary edit, and against a literal
# space it does not fail the record -- it drops the record out of the swept set
# entirely, which is the worse half.
POPULATION_SLOTS = (
    re.compile(rf"{_N}[-\s]file\s+manifest"),
    re.compile(rf"{_N}\s+files\s+of\s+the\s+installed"),
    re.compile(rf"files\s+in\s+the\s+wheel's\s+`sglang/`\s+tree\s*\|\s*{_N}"),
    re.compile(rf"{_N}{_B}\s+of\s+{_B}{_N}{_B}\s+manifest\s+files"),
    re.compile(
        rf"{_N}{_B}\s+of\s+{_B}{_N}{_B}\s+files\s+against\s+the\s+committed\s+manifest"
    ),
    re.compile(rf"{_N}\s+of\s+{_N},?\s+(?:0|zero)\s+missing"),
    re.compile(rf"IDENTITY_RC=0`?,?\s+{_N}\s+of\s+{_N}"),
    re.compile(rf"manifest_files={_N}"),
    re.compile(rf"derived_files={_N}"),
    re.compile(rf"{_N}\s+derived\s+against\s+{_N}\s+in\s+the\s+manifest"),
    re.compile(rf"IDENTITY\s+OK:\s+{_N}\s+files"),
    # SELECTS NOTHING in the swept corpus today. Its only occurrence in the
    # tree is the #1832 row of `.agents/issue-index.md` -- "3338 files, from one
    # generation run on 2026-08-19" -- which APPEND_ONLY_RECORDS excludes by
    # policy, not because the phrasing is unreachable. It is kept because that
    # sentence is the honest-record wording this tree already wrote once, and it
    # is the row a later record would copy it from; a shape that currently
    # matches nothing costs one `finditer` over text already in memory, and
    # deleting it would make the sweep blind to the one phrasing the issue this
    # row repairs used in its own words.
    re.compile(rf"{_N}\s+files,\s+from\s+one\s+generation\s+run"),
)


def _population_slots(text: str) -> list[tuple[int, str]]:
    """Every population figure the text states, with the phrase that states it."""

    found: list[tuple[int, str]] = []
    for pattern in POPULATION_SLOTS:
        for match in pattern.finditer(text):
            phrase = " ".join(match.group(0).split())
            for group in match.groups():
                found.append((int(group.replace(",", "")), phrase))
    return found


@functools.lru_cache(maxsize=1)
def _records_stating_a_population() -> tuple[tuple[Path, tuple[tuple[int, str], ...]], ...]:
    skip = {path.resolve() for path in APPEND_ONLY_RECORDS}
    seen: set[Path] = set()
    out: list[tuple[Path, tuple[tuple[int, str], ...]]] = []
    for glob in RECORD_GLOBS:
        for path in sorted(ROOT.glob(glob)):
            resolved = path.resolve()
            if resolved in seen or resolved in skip:
                continue
            seen.add(resolved)
            slots = _population_slots(path.read_text(encoding="utf-8", errors="replace"))
            if slots:
                out.append((path, tuple(slots)))
    return tuple(out)


# The file population of the installed `sglang/` tree at pin `f63458b5`,
# derived in job `86282a1a` on 2026-08-23 and asserted there as
# `IDENTITY_RC=0`, 3338 of 3338.
#
# THE LITERAL LIVES HERE, and is never read back out of the manifest (#1832).
# `manifest["file_count"] == len(manifest["files"])` compares one JSON document
# to itself: on the base of this change, emptying `files` and setting the
# header to 0 left the suite at `Ran 14 tests ... OK`, rc=0, and so did dropping
# `sglang/README.md` and decrementing the header to 3337 -- the exact shape a
# mis-generated manifest takes. A checker that reads its expectation from the
# file it checks is a tautology, which is the rule
# `scripts/check-test-registration.py` already states for its label selection.
#
# EXACT, with no tolerance. A wheel at a pinned revision has exactly one file
# population, so any movement is either a new pin -- which moves `pin`,
# `wheel.sha256` and this literal together, in a change that says so -- or a
# defective manifest. A floor set below the real count would admit the silent
# shrink it exists to catch, the way `NORETURN_POPULATION_FLOOR = 40` against a
# real 53 does elsewhere in this tree.
#
# This pins the manifest AS COMMITTED. It does not re-derive it: that needs a
# second independent install inside an `rc` lease and stays owed under #1265.
EXPECTED_FILE_COUNT = 3338


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
        """Kept, and insufficient on its own: this is the one case that catches
        a header edited without the table, and it is the case #1832 shows can
        see nothing else."""

        self.assertEqual(self.manifest["file_count"], len(self.manifest["files"]))

    def test_the_file_table_holds_the_pinned_population(self) -> None:
        self.assertEqual(len(self.manifest["files"]), EXPECTED_FILE_COUNT)

    def test_the_declared_file_count_is_the_pinned_population(self) -> None:
        """Asserted separately from the table so each half reds on its own name."""

        self.assertEqual(self.manifest["file_count"], EXPECTED_FILE_COUNT)

    def test_the_records_quote_the_pinned_population(self) -> None:
        """The count is quoted as measured in three records and was in no
        executing code (#1832). Binding them to the same literal means
        regenerating the manifest without correcting the prose reds, and
        correcting the prose without the manifest reds.

        A phrase, never a line number: these are files other changes append to,
        and a recorded line anchor goes stale inside one pull request.
        """

        stating = {path.resolve() for path, _ in _records_stating_a_population()}
        silent = [
            record.relative_to(ROOT).as_posix()
            for record in QUOTING_RECORDS
            if record.resolve() not in stating
        ]
        self.assertEqual(silent, [], "these records state no population figure at all")

    def test_no_record_states_a_population_other_than_the_pinned_one(self) -> None:
        """EVERY figure, not one of them. The presence test this sits beside
        went green on `**3337 of 3338**` in the oracle record and on `3337 files
        of the installed` in the matrix row, because a second correct occurrence
        in the same file answered for the mutated one.

        The swept set is derived, so a record that quotes the population without
        being on any list is bound the moment it says so.
        """

        wrong = [
            f"{path.relative_to(ROOT).as_posix()}: {phrase!r}"
            for path, slots in _records_stating_a_population()
            for value, phrase in slots
            if value != EXPECTED_FILE_COUNT
        ]
        self.assertEqual(wrong, [], f"population figures that are not {EXPECTED_FILE_COUNT}")

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
