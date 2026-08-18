#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-symbol-anchors.py (#1143, #1139).

Fixture paths are `alpha/beta/...`, not `src/vllm/...`, and that is load-bearing.
This file is scanned by the checker it tests, so a fixture written as a real
local path is a real citation of a file that does not exist -- twelve of them,
which is how the first committed version reported itself red. `alpha/` exists
in no tree, so the fixtures fall into the skipped upstream bucket in the real
repository while still resolving inside each temporary one.

The case this suite exists to pin is `test_verdict_depends_on_the_citing_text`.
#911 shipped an anchor checker that read its expectation out of the file it was
checking and reported 27/27 FRESH while five anchors pointed at unrelated code.
A checker of that shape cannot distinguish two citations that name DIFFERENT
symbols in the SAME file, because nothing it reads varies. This one must, and
that case asserts it does.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-symbol-anchors.py"
CHECKED_RE = re.compile(r"in-repo checked (\d+) \(fresh (\d+), stale (\d+)\)")


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


class Tree:
    """A throwaway tree with one cited file and one citing file."""

    def __init__(self, cited_path: str, cited_body: str, citing_body: str,
                 citing_path: str = "note.md"):
        self.box = tempfile.TemporaryDirectory()
        root = Path(self.box.name)
        for rel, body in ((cited_path, cited_body), (citing_path, citing_body)):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        self.root = root

    def __enter__(self) -> Path:
        return self.root

    def __exit__(self, *exc) -> None:
        self.box.cleanup()


def init_repo(root: Path, tracked: list[str]) -> None:
    """Make `root` a real repository with exactly `tracked` in the index.

    The untracked case needs one: outside a repository `git ls-files` fails and
    the checker falls back to walking the filesystem, where nothing is untracked
    and the blind spot cannot exist.
    """

    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "add", *tracked], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "seed"], check=True, env=env)


class SymbolAnchorTests(unittest.TestCase):
    def test_self_test_sweeps_a_non_empty_corpus(self):
        result = run("--self-test")
        self.assertEqual(result.returncode, 0, result.stdout)
        match = re.search(r"self-test fixtures: (\d+), failures: (\d+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        self.assertGreater(int(match.group(1)), 0, "a corpus of zero fixtures proves nothing")
        self.assertEqual(int(match.group(2)), 0, result.stdout)

    def test_this_tree_is_green_over_a_non_zero_population(self):
        result = run()
        self.assertEqual(result.returncode, 0, result.stdout)
        match = CHECKED_RE.search(result.stdout)
        self.assertIsNotNone(match, result.stdout)
        self.assertGreater(
            int(match.group(1)), 0,
            "a green run that checked nothing is a skip wearing a pass",
        )
        self.assertEqual(int(match.group(3)), 0, result.stdout)

    def test_fresh_symbol_passes(self):
        with Tree("alpha/beta/a.cpp", "struct Widget {};\n",
                  "see `alpha/beta/a.cpp::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("in-repo checked 1 (fresh 1, stale 0)", result.stdout)

    def test_renamed_symbol_is_reported(self):
        with Tree("alpha/beta/a.cpp", "struct Gadget {};\n",
                  "see `alpha/beta/a.cpp::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("does not contain `Widget`", result.stdout)
        self.assertIn("note.md:1", result.stdout)

    def test_verdict_depends_on_the_citing_text(self):
        """The anti-tautology case (#911).

        One cited file, unchanged, cited twice. The two verdicts must differ,
        which is only possible if the expectation comes from the CITING side.
        """

        body = "struct Widget {};\n"
        with Tree("alpha/beta/a.cpp", body, "`alpha/beta/a.cpp::Widget`\n") as root:
            good = run("--root", str(root))
        with Tree("alpha/beta/a.cpp", body, "`alpha/beta/a.cpp::Gadget`\n") as root:
            bad = run("--root", str(root))
        self.assertEqual(good.returncode, 0, good.stdout)
        self.assertEqual(bad.returncode, 1, bad.stdout)
        self.assertIn("fresh 1, stale 0", good.stdout)
        self.assertIn("fresh 0, stale 1", bad.stdout)

    def test_a_call_site_satisfies_a_citation(self):
        with Tree("alpha/beta/a.cpp", "  ModelRegistry::Load(config);\n",
                  "`alpha/beta/a.cpp::ModelRegistry::Load`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_longer_identifier_does_not_satisfy_a_shorter_citation(self):
        with Tree("alpha/beta/a.cpp", "  LoadShards(dir);\n",
                  "`alpha/beta/a.cpp::Load`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_a_local_path_that_does_not_exist_is_reported(self):
        with Tree("alpha/beta/a.cpp", "struct Widget {};\n",
                  "`alpha/beta/a.cpp::Widget` and `alpha/beta/gone.cpp::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("names a path under this repository that does not exist", result.stdout)

    def test_an_upstream_path_is_bucketed_not_failed(self):
        with Tree("alpha/beta/a.cpp", "struct Widget {};\n",
                  "`alpha/beta/a.cpp::Widget` and `vllm/config/model.py::ModelConfig`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("upstream/unknown 1", result.stdout)

    def test_zero_checked_citations_is_a_failure(self):
        with Tree("alpha/beta/a.cpp", "struct Widget {};\n",
                  "`vllm/config/model.py::ModelConfig` only\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("0 in-repo citations were checked", result.stdout)

    def test_the_frozen_archive_is_skipped_and_counted(self):
        with Tree("alpha/beta/a.cpp", "struct Widget {};\n",
                  "`alpha/beta/a.cpp::Gadget` is stale here\n",
                  citing_path=".agents/completed/old/STATE-1.md") as root:
            (root / "live.md").write_text("`alpha/beta/a.cpp::Widget`\n", encoding="utf-8")
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("1 frozen files skipped", result.stdout)

    def test_prose_without_a_path_is_not_a_citation(self):
        with Tree("alpha/beta/a.cpp", "struct Widget {};\n",
                  "`Qwen3_5MTPKind::kMoe` and `alpha/beta/a.cpp::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("1 citations", result.stdout)
        self.assertIn("in-repo checked 1", result.stdout)

    def test_a_citation_span_is_not_its_own_evidence(self):
        """A self-citation must not be satisfied by the citation text (#911 again).

        The immunity argument is "expectation from the CITING file, evidence
        from the CITED file". When they are the SAME file those two sides
        collapse: the checker searched the whole body, and the citation span
        itself contains the symbol, so `GhostSymbol` -- which exists nowhere but
        inside its own citation -- read as fresh. Zero of the tree's anchors are
        self-citations today, and `.agents/porting.md` now tells authors to
        write the full path, which makes a file documenting its own symbols the
        most natural next thing anyone writes.
        """

        with Tree("alpha/beta/a.cpp",
                  "// see `alpha/beta/a.cpp::GhostSymbol` for details\n"
                  "struct Widget {};\n",
                  "`alpha/beta/a.cpp::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("does not contain `GhostSymbol`", result.stdout)

    def test_a_citation_in_the_cited_file_is_not_evidence_either(self):
        """The cross-file form of the same laundering.

        `a.cpp` names `Ghost` only inside a citation of a DIFFERENT file. That
        span is someone else's claim, never a definition or a call, so it may
        not satisfy a citation of `a.cpp`.
        """

        with Tree("alpha/beta/a.cpp", "// see `alpha/beta/b.cpp::Ghost`\n",
                  "`alpha/beta/a.cpp::Ghost`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("does not contain `Ghost`", result.stdout)

    def test_untracked_files_carrying_citations_are_counted(self):
        """The blind spot this change reported about itself, made visible.

        An untracked file is not in `git ls-files`, so its citations are never
        scanned and its path never resolves. CI is safe because everything is
        tracked there; a local pre-commit run is not. The skip is not closed --
        it is COUNTED, the same way `frozen_files` is.
        """

        with Tree("alpha/beta/a.cpp", "struct Widget {};\n",
                  "`alpha/beta/a.cpp::Widget`\n") as root:
            init_repo(root, ["alpha/beta/a.cpp", "note.md"])
            (root / "scratch.md").write_text(
                "`alpha/beta/a.cpp::Gadget`\n", encoding="utf-8")
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("1 untracked file", result.stdout)

    def test_the_floor_reds_when_the_population_collapses(self):
        """A floor of zero is a mute switch.

        One added FROZEN_PREFIXES entry, or a narrowed CITATION_RE, drops the
        in-repo population from 91 to 1 and a `checked == 0` guard stays green
        the whole way down.
        """

        with Tree("alpha/beta/a.cpp", "struct Widget {};\n",
                  "`alpha/beta/a.cpp::Widget`\n") as root:
            result = run("--root", str(root), "--min-checked", "80")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("below the recorded floor", result.stdout)
        self.assertIn("80", result.stdout)

    def test_this_tree_meets_the_recorded_floor(self):
        """The floor is asserted here with a LITERAL, not read from the checker.

        Reading the module's constant would make this a tautology: lowering the
        floor would lower the expectation with it.
        """

        result = run()
        self.assertEqual(result.returncode, 0, result.stdout)
        match = CHECKED_RE.search(result.stdout)
        self.assertIsNotNone(match, result.stdout)
        self.assertGreaterEqual(int(match.group(1)), 85, result.stdout)

    def test_the_buckets_sum_to_the_citation_count(self):
        """A floor below the real count is a mute switch unless the buckets sum.

        Every citation lands in exactly one of checked / upstream / missing, so
        a citation that quietly stops being counted anywhere is arithmetic, not
        judgement.
        """

        result = run()
        self.assertEqual(result.returncode, 0, result.stdout)
        total = int(re.search(r"symbol anchors: (\d+) citations", result.stdout).group(1))
        checked = int(CHECKED_RE.search(result.stdout).group(1))
        upstream = int(re.search(r"upstream/unknown (\d+)", result.stdout).group(1))
        missing = int(re.search(r"missing local path (\d+)", result.stdout).group(1))
        self.assertEqual(checked + upstream + missing, total, result.stdout)
        self.assertIn(f"buckets sum {total} vs {total} citations", result.stdout)

        # This tree has zero missing local paths, so dropping THAT bucket from
        # the sum leaves the arithmetic intact and the mute switch survives.
        # One fixture tree populates all three at once.
        with Tree("alpha/beta/a.cpp", "struct Widget {};\n",
                  "`alpha/beta/a.cpp::Widget` `alpha/beta/gone.cpp::Widget` "
                  "`vllm/config/model.py::ModelConfig`\n") as root:
            three = run("--root", str(root))
        self.assertIn("in-repo checked 1", three.stdout)
        self.assertIn("upstream/unknown 1", three.stdout)
        self.assertIn("missing local path 1", three.stdout)
        self.assertIn("buckets sum 3 vs 3 citations", three.stdout)

    def test_an_ambiguous_basename_is_checked_against_every_candidate(self):
        """The symbol lives in the LAST candidate, not the first.

        Candidate order is sorted, so `alpha/beta/a.cpp` is consulted first and
        does not carry `Widget`. A checker that stops at the first candidate
        therefore reds here, deterministically.
        """

        with Tree("alpha/beta/a.cpp", "struct Other {};\n",
                  "`a.cpp::Widget`\n") as root:
            (root / "alpha/gamma").mkdir(parents=True, exist_ok=True)
            (root / "alpha/gamma/a.cpp").write_text("struct Widget {};\n", encoding="utf-8")
            good = run("--root", str(root))
        with Tree("alpha/beta/a.cpp", "struct Other {};\n",
                  "`a.cpp::Ghost`\n") as root:
            (root / "alpha/gamma").mkdir(parents=True, exist_ok=True)
            (root / "alpha/gamma/a.cpp").write_text("struct Widget {};\n", encoding="utf-8")
            bad = run("--root", str(root))
        self.assertEqual(good.returncode, 0, good.stdout)
        self.assertIn("of which ambiguous basename 1", good.stdout)
        self.assertEqual(bad.returncode, 1, bad.stdout)
        self.assertIn("no file named `a.cpp`", bad.stdout)

    def test_a_dot_leading_path_is_a_citation(self):
        """A dot-leading path used to match ZERO times, silently.

        Any citation of a file under `.agents/` or `.github/` was invisible to
        the grammar, which is latent only until someone cites a spec by symbol.
        """

        with Tree(".alpha/beta/note.md", "the Widget seam\n",
                  "`.alpha/beta/note.md::Widget` and `.alpha/beta/gone.md::Widget`\n") as root:
            result = run("--root", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("in-repo checked 1", result.stdout)
        self.assertIn("names a path under this repository that does not exist", result.stdout)

    def test_parity_pin_is_read_from_the_recorded_block(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_csa", CHECKER)
            module = importlib.util.module_from_spec(spec)
            # `from __future__ import annotations` + @dataclass needs the module
            # visible in sys.modules while its body executes.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            pin = module.parity_pin(ROOT / ".agents/upstream-sync.md")
        finally:
            sys.path.pop(0)
        self.assertRegex(pin, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
