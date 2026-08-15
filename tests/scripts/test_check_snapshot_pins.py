"""GATE-PIN-UNPINNED-SNAPSHOTS (#471) — the checker's own gate.

A checker that cannot be shown failing has proven nothing, and a checker whose
pattern is quietly narrowed goes green while the defect walks back in. Each case
here synthesises a tree, so none of them depends on what the repo happens to
contain today.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CHECKER = REPO_ROOT / "scripts" / "check-snapshot-pins.py"
PIN_HEADER = REPO_ROOT / "tests" / "parity" / "hf_snapshot.h"

# A candidate that could not be LAUNCHED, as opposed to one that ran and
# rejected the header: 127 is "not found", 126 is "found, not executable". Only
# these fall through to the next candidate; any other status is a compiler's
# verdict on the header and is reported as such.
_CANNOT_LAUNCH = (126, 127)


def _cxx_candidates() -> list[list[str]]:
    """C++ commands to syntax-check the pin header with, best first.

    `$CXX` is honored as a WHOLE command, not as its first token. `ccache g++`,
    `sccache clang++` and `distcc g++` are ordinary spellings of it, and keeping
    only the first token runs the LAUNCHER with the compiler's flags and no
    compiler behind it -- a red that names this header while the real complaint
    is `/usr/bin/env: invalid option -- 's'`. Fail-closed, so never dangerous,
    but it accuses the wrong file.

    Each entry is a full argv prefix whose executable is resolved through PATH;
    an unresolvable one is dropped here rather than raising later.
    """

    wanted: list[list[str]] = []
    configured = os.environ.get("CXX", "").split()
    if configured:
        wanted.append(configured)
    wanted += [["c++"], ["g++"], ["clang++"]]

    resolved: list[list[str]] = []
    for command in wanted:
        found = shutil.which(command[0])
        if found is not None:
            resolved.append([found, *command[1:]])
    return resolved


_spec = importlib.util.spec_from_file_location("check_snapshot_pins", CHECKER)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class SnapshotPinChecker(unittest.TestCase):
    def _tree(self, tmp: pathlib.Path, body: str, rel: str) -> pathlib.Path:
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        return target

    def test_an_unpinned_resolver_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, mod.UNPINNED_FIXTURE, "tests/parity/test_new_gate.cpp")
            problems = mod.check(tmp)
        self.assertTrue(
            any("UNPINNED checkpoint resolution" in p for p in problems),
            "a new directory_iterator over <repo>/snapshots/ must fail the gate",
        )

    def test_a_pinned_resolver_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, mod.PINNED_FIXTURE, "tests/parity/test_new_gate.cpp")
            problems = [p for p in mod.check(tmp) if "UNPINNED" in p]
        self.assertEqual([], problems)

    def test_shard_iteration_inside_a_resolved_dir_is_not_a_resolution(self) -> None:
        """The converted gates still iterate their OWN checkpoint for shards.

        If that tripped the checker, the only way to green would be to weaken the
        pattern, which is how a gate stops gating.
        """
        body = (
            '#include <filesystem>\n'
            'namespace fs = std::filesystem;\n'
            '// resolved via parity::HfSnapshot, then enumerate shards:\n'
            'void Shards(const std::string& dir) {\n'
            '  std::error_code ec;\n'
            '  for (const auto& e : fs::directory_iterator(dir, ec)) (void)e;\n'
            '}\n'
            '// the word snapshots appears in prose above\n'
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, body, "tests/parity/test_shards.cpp")
            problems = [p for p in mod.check(tmp) if "UNPINNED" in p]
        self.assertEqual([], problems)

    def test_a_commented_out_resolver_is_not_a_resolution(self) -> None:
        body = (
            '#include <filesystem>\n'
            'namespace fs = std::filesystem;\n'
            '// for (const auto& e : fs::directory_iterator(snaps, ec)) {}\n'
            '// snapshots\n'
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, body, "tests/parity/test_comment.cpp")
            problems = [p for p in mod.check(tmp) if "UNPINNED" in p]
        self.assertEqual([], problems)

    def test_a_stale_ledger_entry_fails(self) -> None:
        """A ledger that outlives its debt starts excusing unreviewed code."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            (tmp / "tests").mkdir()
            problems = mod.check(tmp)
        self.assertTrue(
            any("STALE ledger entry" in p for p in problems),
            "every ledger line must still name a real unpinned resolution",
        )

    def test_the_repo_itself_is_clean(self) -> None:
        self.assertEqual([], mod.check(REPO_ROOT))

    def test_the_ledger_holds_no_file_this_row_pinned(self) -> None:
        """The eight gates GATE-PIN-UNPINNED-SNAPSHOTS converted must not reappear.

        Re-adding one to the ledger would be the cheapest way to un-fix this row,
        so it is spelled out rather than left to the STALE check.
        """
        pinned_by_this_row = {
            "tests/parity/test_qwen3_dflash_draft_parity.cpp",
            "tests/parity/test_qwen3_dflash_kvprep_parity.cpp",
            "tests/parity/test_qwen27_dflash_spec_decode.cpp",
            "tests/parity/test_op_parity.cpp",
            "tests/parity/test_qwen36_paged_engine.cpp",
            "tests/parity/test_qwen36_async_serving.cpp",
            "tests/parity/test_qwen36_spec_decode.cpp",
            "tests/vllm/test_qwen36_weights.cpp",
        }
        self.assertEqual(set(), pinned_by_this_row & set(mod.LEDGER))

    def test_the_fixture_corpus_sweeps_both_directions(self) -> None:
        """The claim "it cannot be greened by narrowing its pattern", BOUNDED.

        One fixture proves only that the checker matches THAT fixture. This drives
        the whole corpus: every positive is an idiom some plausible narrowing would
        drop, every negative is code some plausible widening would falsely flag.

        This test used to close on "so a mutation to any single branch of the
        pattern lands on one of them". That was not true and this row exists
        because it was not: dropping the three environment `CACHE_MARKERS`, or
        `scandir`/`walk`/`iglob`/`rglob`, or `listdir`, or `readdir` each left
        this test, the self-test and the checker green. Only 4 of 11 enumerators
        and 2 of 5 markers were isolated by a fixture.

        The claim it makes now is the bounded one: the corpus catches narrowings
        the corpus COVERS, and what it covers is no longer a matter of trust.
        `test_narrowing_any_single_branch_drops_a_positive_fixture` performs each
        single-branch drop of `ENUMERATORS` and `CACHE_MARKERS` and requires a
        fixture to notice. Narrowings outside those two lists -- the `fs::`
        spelling, the scanned roots and suffixes, the subject-binding rule -- are
        covered only where a fixture happens to exercise them, and an idiom
        nobody wrote a fixture for is not covered at all.
        """
        self.assertGreaterEqual(len(mod.FIXTURES), 13, "the corpus must cover a class")
        self.assertTrue(any(f.unpinned for f in mod.FIXTURES))
        self.assertTrue(any(not f.unpinned for f in mod.FIXTURES))
        for fixture in mod.FIXTURES:
            with self.subTest(fixture=fixture.name):
                with tempfile.TemporaryDirectory() as raw:
                    tmp = pathlib.Path(raw)
                    self._tree(tmp, fixture.body, fixture.rel)
                    reported = [
                        p
                        for p in mod.check(tmp)
                        if "UNPINNED checkpoint resolution" in p
                    ]
                self.assertEqual(
                    fixture.unpinned,
                    bool(reported),
                    f"{fixture.name} ({fixture.rel}): {fixture.kills or 'must stay clean'}",
                )

    def _reports(self, fixture) -> bool:
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, fixture.body, fixture.rel)
            return any(
                "UNPINNED checkpoint resolution" in p for p in mod.check(tmp)
            )

    def _positives_lost_to(self, **narrowed) -> list[str]:
        """Names of positive fixtures the checker stops reporting under `narrowed`.

        The narrowed pattern is built by the checker's OWN `enumerator_re` /
        `marker_re`, never restated here: a pattern copied into a test drifts
        from the one that ships and then proves nothing about it.
        """
        saved = {name: getattr(mod, name) for name in narrowed}
        try:
            for name, value in narrowed.items():
                setattr(mod, name, value)
            return [
                f.name for f in mod.FIXTURES if f.unpinned and not self._reports(f)
            ]
        finally:
            for name, value in saved.items():
                setattr(mod, name, value)

    def test_narrowing_any_single_branch_drops_a_positive_fixture(self) -> None:
        """The corpus's bound, PERFORMED rather than asserted.

        The failure this row repairs was not a wrong pattern, it was an unearned
        claim: the corpus asserted it caught every single-branch narrowing while
        7 of 11 enumerators and 3 of 5 markers had no fixture that isolated them,
        so deleting them was free. Restating the claim more carefully fixes the
        honesty and not the hole.

        So the coverage is checked one branch at a time, by actually removing the
        branch. An enumerator or marker added to the checker without a fixture
        that isolates it fails HERE, which is what keeps the two lists from
        outgrowing the corpus again. It says nothing about narrowings outside
        those two lists; see the docstring above for what stays uncovered.
        """
        for enumerator in mod.ENUMERATORS:
            with self.subTest(enumerator=enumerator):
                narrowed = tuple(e for e in mod.ENUMERATORS if e != enumerator)
                self.assertTrue(
                    self._positives_lost_to(
                        ENUMERATOR_RE=mod.enumerator_re(narrowed)
                    ),
                    f"dropping the `{enumerator}` enumerator leaves every positive "
                    "fixture still detected: no fixture isolates it, so the "
                    "branch is unenforced. Add one that uses it and no other "
                    "enumerator.",
                )
        for marker in mod.CACHE_MARKERS:
            with self.subTest(marker=marker):
                narrowed = tuple(m for m in mod.CACHE_MARKERS if m != marker)
                self.assertTrue(
                    self._positives_lost_to(CACHE_MARKER_RE=mod.marker_re(narrowed)),
                    f"dropping the `{marker}` cache marker leaves every positive "
                    "fixture still detected: no fixture isolates it, so the "
                    "branch is unenforced. Add one that carries it and no other "
                    "marker.",
                )

    def test_the_narrowing_sweep_restores_the_checker(self) -> None:
        """The sweep above monkeypatches module state; a leak would fake a pass."""
        before = (mod.ENUMERATOR_RE.pattern, mod.CACHE_MARKER_RE.pattern)
        self._positives_lost_to(
            ENUMERATOR_RE=mod.enumerator_re(("directory_iterator",)),
            CACHE_MARKER_RE=mod.marker_re(("models--",)),
        )
        self.assertEqual(
            before, (mod.ENUMERATOR_RE.pattern, mod.CACHE_MARKER_RE.pattern)
        )
        self.assertEqual(
            mod.ENUMERATOR_RE.pattern, mod.enumerator_re(mod.ENUMERATORS).pattern
        )
        self.assertEqual(
            mod.CACHE_MARKER_RE.pattern, mod.marker_re(mod.CACHE_MARKERS).pattern
        )

    def test_self_exclusion_refuses_a_path_that_is_not_this_checker(self) -> None:
        """SELF_EXCLUDED was the cheapest way to green the checker for a new gate.

        `LEDGER` owes a tracking issue and is held to the STALE ratchet.
        SELF_EXCLUDED, as a bare frozenset, owed neither -- so adding any real
        gate path to it left checker, self-test and suite all green, and the gate
        it named was never inspected again. It is not a second ledger and now
        refuses to be used as one.
        """
        rogue = "tests/parity/test_qwen36_paged_engine.cpp"
        self.assertNotIn(rogue, mod.SELF_EXCLUDED)
        mod.SELF_EXCLUDED[rogue] = "borrowed for a gate (#471)"
        try:
            problems = mod.check_self_exclusion(REPO_ROOT)
        finally:
            del mod.SELF_EXCLUDED[rogue]
        self.assertTrue(
            any("not this checker" in p for p in problems),
            f"{rogue} was accepted into SELF_EXCLUDED",
        )

    def test_self_exclusion_refuses_a_line_with_no_tracking_issue(self) -> None:
        mod.SELF_EXCLUDED["tests/scripts/test_check_snapshot_pins.py"] = "because"
        try:
            problems = mod.check_self_exclusion(REPO_ROOT)
        finally:
            mod.SELF_EXCLUDED["tests/scripts/test_check_snapshot_pins.py"] = (
                "this checker's own suite (#471)"
            )
        self.assertTrue(any("without a tracking issue" in p for p in problems))

    def test_self_exclusion_refuses_a_line_whose_file_is_now_clean(self) -> None:
        """The STALE ratchet LEDGER has, applied to the exclusion list too."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, "int main() { return 0; }\n", "scripts/check-snapshot-pins.py")
            problems = mod.check_self_exclusion(tmp)
        self.assertTrue(
            any("STALE self-exclusion" in p for p in problems),
            "an excluded file carrying no resolver text must lose its line",
        )

    def test_the_real_self_exclusion_list_is_well_formed(self) -> None:
        self.assertEqual([], mod.check_self_exclusion(REPO_ROOT))

    def test_every_positive_fixture_records_the_narrowing_it_kills(self) -> None:
        """A fixture with no stated claim cannot be maintained; it just accretes."""
        for fixture in mod.FIXTURES:
            if fixture.unpinned:
                with self.subTest(fixture=fixture.name):
                    self.assertTrue(fixture.kills.strip(), fixture.name)

    def test_every_ledger_line_names_a_tracking_issue(self) -> None:
        """A debt with no issue owing its removal is an exemption wearing a reason."""
        self.assertEqual([], mod.check_ledger_shape())

    def test_the_checker_runs_in_ci_and_not_only_in_preflight(self) -> None:
        """AGENTS.md: "Hooks are bypassable convenience, never proof."

        A checker wired only into scripts/agent-preflight.sh gates whoever chooses
        to run it. Both surfaces are asserted here so removing either is RED.

        Each CI leg is asserted SEPARATELY, and that is not pedantry. The bare
        `assertIn("scripts/check-snapshot-pins.py", workflow)` this test used to
        make is satisfied by the plain invocation on its own, so deleting just
        the `--self-test` leg -- the leg that sweeps the fixture corpus, which is
        the entire substance of this row -- left CI green. A checker that runs
        without its corpus is a checker nobody would notice going blind.
        """
        workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        preflight = (REPO_ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        for leg in (
            "python3 scripts/check-snapshot-pins.py\n",
            "python3 scripts/check-snapshot-pins.py --self-test\n",
            "python3 tests/scripts/test_check_snapshot_pins.py\n",
        ):
            with self.subTest(leg=leg.strip()):
                self.assertIn(leg, workflow, f"CI no longer runs `{leg.strip()}`")
        self.assertIn("check-snapshot-pins", preflight)
        self.assertIn("test_check_snapshot_pins", preflight)

    def test_text_the_toolchain_never_sees_is_not_a_resolution(self) -> None:
        """Python's arm of the rule scripts/checker_text.py states for C++.

        A `#` comment and a usage-example docstring are both text the interpreter
        never runs, and a `re.search` reads either as live code.
        """
        body = (
            'import glob\n'
            'import os\n'
            '\n'
            'HELP = 1\n'
            '\n'
            'def resolve(cache):\n'
            '    """Usage: glob.glob(os.path.join(cache, "snapshots", "*"))[0]"""\n'
            '    # return sorted(glob.glob(os.path.join(cache, "snapshots", "*")))[0]\n'
            '    return None\n'
        )
        with tempfile.TemporaryDirectory() as raw:
            tmp = pathlib.Path(raw)
            self._tree(tmp, body, "tools/bench/commented.py")
            problems = [p for p in mod.check(tmp) if "UNPINNED" in p]
        self.assertEqual([], problems)

    def test_every_header_pin_has_an_accessor(self) -> None:
        """A constant with no accessor is a pin nothing resolves through."""
        import re

        header = (REPO_ROOT / "tests/parity/hf_snapshot.h").read_text(encoding="utf-8")
        constants = re.findall(r"inline constexpr const char\* (k\w*Revision)\s*=", header)
        self.assertGreaterEqual(len(constants), 3, constants)
        for name in constants:
            with self.subTest(constant=name):
                self.assertEqual(
                    1,
                    len(re.findall(rf"\b{name}\b,", header)),
                    f"{name} is never passed to HfSnapshot; it pins nothing",
                )

    def _no_compiler(self, why: str) -> None:
        """Skip on a developer box, FAIL in CI.

        `skipTest` exits 0, and `scripts/agent-preflight.sh`'s `run()` prints a
        green `ok` while swallowing stdout, so a host with no C++ compiler turns
        the guard below into a silent no-op that LOOKS like it ran. Measured:
        `env PATH=/nonexistent python3 -m unittest ...` reported `OK (skipped=1)`
        and exit 0. That is tolerable on a laptop and is not tolerable in the
        lane that is supposed to be the backstop, so `CI` makes it a failure --
        CI's runner ships `g++`, so this can only fire if that stops being true.
        """

        if os.environ.get("CI", "").strip():
            self.fail(
                "CI must be able to syntax-check "
                f"{PIN_HEADER.relative_to(REPO_ROOT)}, and cannot: {why}. "
                "A skip here exits 0 and reads as a pass."
            )
        self.skipTest(why)

    def test_the_pin_header_compiles_on_its_own(self) -> None:
        """Issue #558 (guard) / #551, #546 (the defect it is a guard against).

        `af8170154` added `Nemotron35LightningSnapshot()` eleven lines ABOVE the
        `HfSnapshot` it calls, and `main` went red: 14 translation units failed
        with `'HfSnapshot' was not declared in this scope`. Every one of them is
        checkpoint-gated, so a serial `ctest` reported them `***Not Run` -- which
        reads as a missing checkpoint on a box that has none, not as a compile
        break. `fafa16f0f` (PR #556) fixed the ordering; this is the guard, which
        that fix did not carry.

        A full C++ build does catch it (`test_hf_snapshot_pinning` includes this
        header and always builds), and that is exactly why the guard belongs
        here instead: the change that broke it never built C++ at all. This suite
        runs from `scripts/agent-preflight.sh` and from CI's record lane, needs
        no CMake, no build tree, no GPU and no checkpoint, and takes about a
        quarter of a second -- so the header cannot go back to being validated
        only by a build that a records-and-evidence commit has no reason to run.

        Compiling the header ALONE, rather than as part of some TU that happens
        to include other things first, is the property under test: a pin nobody
        can include is a pin nobody resolves through. It also catches the
        merge hazard the two independent repairs of #551 produced -- moving the
        same block to two different places auto-merges with no conflict into a
        header that DEFINES the accessor twice.
        """
        candidates = _cxx_candidates()
        if not candidates:
            self._no_compiler("no C++ compiler on PATH")
        with tempfile.TemporaryDirectory() as raw:
            tu = pathlib.Path(raw) / "hf_snapshot_alone.cc"
            tu.write_text(f'#include "{PIN_HEADER.name}"\n', encoding="utf-8")
            attempted: list[str] = []
            for command in candidates:
                proc = subprocess.run(
                    [
                        *command,
                        "-std=c++20",
                        "-fsyntax-only",
                        f"-I{PIN_HEADER.parent}",
                        str(tu),
                    ],
                    capture_output=True,
                    text=True,
                )
                attempted.append(" ".join(command))
                if proc.returncode not in _CANNOT_LAUNCH:
                    break
            else:
                self._no_compiler(f"none of {attempted} could be launched")
        self.assertEqual(
            0,
            proc.returncode,
            f"{PIN_HEADER.relative_to(REPO_ROOT)} does not compile on its own "
            f"under {' '.join(command)}:\n{proc.stderr.strip()}",
        )


if __name__ == "__main__":
    unittest.main()
