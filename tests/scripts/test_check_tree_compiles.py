#!/usr/bin/env python3
"""`scripts/check-tree-compiles.py` compiles what a change can break (#2401).

Row `ENG-PREFLIGHT-COMPILES`, spec `.agents/specs/preflight-compiles.md`.

The defect this suite pins: on 2026-08-31 `main` was pushed twice in a state
that does not compile, and `scripts/agent-preflight.sh` was green both times.
`5263ac31f` was a shell line continuation inside a `//` comment, which
`-Wcomment` under this tree's `-Werror` rejects. `08fa2f5aa` was
`MlaSharedSelection*` bound to `ForwardMlaAttentionBlock`'s `vt::Tensor*`
parameter (#2395). The 30 record checkers and 60 suites preflight runs validate
records, prose, anchors and trailers against a tree, and every one of them
passes on a tree that does not build.

## Why this suite builds a real scratch project

Every assertion below could be spelled as a text match against the checker, and
a text match cannot tell a reverse-include closure that computes the right set
from one that computes it and then throws it away. What is under test is which
translation units get handed to a compiler, so each case runs `cmake` and a real
`c++` over a scratch repository it owns end to end. The projects are four files
and compile in about a second.

The scratch repository is a real git repository with real commits, because the
checker resolves its scope from `git diff` and its flags from a
`compile_commands.json` that CMake writes with absolute source paths. A fixture
that only wrote files into a directory would exercise neither.

## The two cases that carry the row

`test_a_broken_translation_unit_in_scope_does_not_compile` is `5263ac31f`'s
shape: the offending line is in a file the diff names, and a scope of "the
`.cpp` files this diff names" would find it.

`test_a_changed_header_reaches_a_translation_unit_the_diff_does_not_name` is
`08fa2f5aa`'s shape and is the case that decides the design. Neither
contributing commit carried the broken call: one added the parameter to the
header, the other added the caller, and each side compiled alone. The scratch
project reproduces that exactly -- the diff touches the header and its
definition, and the file that stops compiling is the one the diff does NOT
touch. A changed-`.cpp` scope is green here. Delete the reverse-include closure
from the checker and only this case fails.

## Un-run is not passing

`test_a_source_no_target_compiles_is_named_and_not_counted` holds the
distinction that a "681 of 685 tests failed" summary could not: a translation
unit that was never handed to a compiler must be reported as such, by name, and
must not be added to the compiled total. A count that folds the two together
reads a build that never ran as a build that passed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "check-tree-compiles.py"
PREFLIGHT = ROOT / "scripts" / "agent-preflight.sh"


CMAKELISTS = """\
cmake_minimum_required(VERSION 3.16)
project(scratch LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wall -Wextra -Werror)
add_library(scratch_lib STATIC src/lib.cpp src/user.cpp)
target_include_directories(scratch_lib PUBLIC include)
"""

HEADER = """\
#pragma once
int Add(int a, int b);
"""

LIB = """\
#include "lib.h"
int Add(int a, int b) { return a + b; }
"""

USER = """\
#include "lib.h"
int Use() { return Add(1, 2); }
"""


class Report:
    def __init__(self, returncode: int, text: str) -> None:
        self.returncode = returncode
        self.text = text

    def __str__(self) -> str:  # shown verbatim on any failure below
        return f"exit {self.returncode}\n{self.text}"


class ScratchProject(unittest.TestCase):
    """A four-file CMake project in a real git repository."""

    def setUp(self) -> None:
        if shutil.which("cmake") is None:
            self.skipTest("cmake is not on PATH")
        self.tmp = Path(tempfile.mkdtemp(prefix="tree-compiles-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "include").mkdir()
        (self.tmp / "src").mkdir()
        self.write("CMakeLists.txt", CMAKELISTS)
        self.write("include/lib.h", HEADER)
        self.write("src/lib.cpp", LIB)
        self.write("src/user.cpp", USER)
        self.write("notes.md", "notes\n")

        self.git("init", "--quiet", ".")
        self.git("config", "user.email", "tree-compiles@test.invalid")
        self.git("config", "user.name", "tree compiles test")
        self.git("config", "commit.gpgsign", "false")
        self.commit("base")
        self.base = self.rev("HEAD")

    # -- scratch helpers -----------------------------------------------------

    def write(self, relative: str, text: str) -> None:
        path = self.tmp / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.tmp, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    def commit(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", message)

    def rev(self, revision: str) -> str:
        return self.git("rev-parse", revision)

    def check(self, *args: str) -> Report:
        result = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--source-dir",
                str(self.tmp),
                "--build-dir",
                str(self.tmp / ".compilecheck"),
                "--jobs",
                "2",
                *args,
            ],
            cwd=self.tmp,
            capture_output=True,
            text=True,
            check=False,
        )
        return Report(result.returncode, result.stdout + result.stderr)


class ScopeTests(ScratchProject):
    def test_a_broken_translation_unit_in_scope_does_not_compile(self) -> None:
        """`5263ac31f`: a `//` comment ending in a shell line continuation."""

        self.write(
            "src/user.cpp",
            '#include "lib.h"\n'
            "// built with: c++ -c src/user.cpp \\\n"
            "//     -Iinclude\n"
            "int Use() { return Add(1, 2); }\n",
        )
        self.commit("break user.cpp")
        report = self.check("--base", self.base)
        self.assertEqual(1, report.returncode, report)
        self.assertIn("src/user.cpp", report.text, report)
        self.assertIn("multi-line comment", report.text, report)

    def test_a_changed_header_reaches_a_translation_unit_the_diff_does_not_name(
        self,
    ) -> None:
        """`08fa2f5aa`: the broken caller is in no contributing diff.

        RED before the reverse-include closure: the diff names `include/lib.h`
        and `src/lib.cpp`, both of which compile, and `src/user.cpp` -- the only
        file that stops compiling -- is not in it.
        """

        self.write("include/lib.h", "#pragma once\nint Add(int a, int b, int c);\n")
        self.write(
            "src/lib.cpp",
            '#include "lib.h"\nint Add(int a, int b, int c) { return a + b + c; }\n',
        )
        self.commit("widen Add")

        changed = self.git("diff", "--name-only", self.base, "HEAD").split()
        self.assertNotIn(
            "src/user.cpp",
            changed,
            "harness precondition failed: the diff names the file that breaks, "
            "so this case would pass under a changed-.cpp scope and proves nothing",
        )

        report = self.check("--base", self.base)
        self.assertEqual(1, report.returncode, report)
        self.assertIn("src/user.cpp", report.text, report)

    def test_a_unit_whose_dependency_scan_fails_is_checked_anyway(self) -> None:
        """No include edges is not "reaches no header".

        RED before the forced inclusion: `src/broken.cpp` exists at the base,
        is not in the diff, and does not preprocess, so it contributes no edges
        and falls out of the closure of a header change. The run then exits 0
        over a unit that does not compile.
        """

        self.write("src/broken.cpp", "#if 1\nint Broken() { return 1; }\n")
        self.write(
            "CMakeLists.txt",
            CMAKELISTS.replace(
                "src/lib.cpp src/user.cpp", "src/lib.cpp src/user.cpp src/broken.cpp"
            ),
        )
        self.commit("a unit the preprocessor cannot read")
        base = self.rev("HEAD")

        # A header edit that breaks nothing, so the ONLY way to a non-zero exit
        # is the unscannable unit being forced into the closure.
        self.write("include/lib.h", HEADER + "int Subtract(int a, int b);\n")
        self.commit("declare Subtract")
        changed = self.git("diff", "--name-only", base, "HEAD").split()
        self.assertEqual(
            ["include/lib.h"],
            changed,
            "harness precondition failed: the diff must name the header alone",
        )

        report = self.check("--base", base)
        self.assertEqual(1, report.returncode, report)
        self.assertIn("src/broken.cpp", report.text, report)
        self.assertIn("would not preprocess", report.text, report)

    def test_an_empty_scope_exits_zero_and_says_the_set_is_empty(self) -> None:
        """A records-only change is 30 of the last 60 commits on `main`."""

        self.write("notes.md", "notes, revised\n")
        self.commit("notes only")
        report = self.check("--base", self.base)
        self.assertEqual(0, report.returncode, report)
        self.assertIn("no translation unit is in scope", report.text, report)
        self.assertNotIn(
            "compiled",
            report.text,
            "an empty scope must not report a compiled count it did not earn",
        )

    def test_a_clean_code_change_in_scope_compiles(self) -> None:
        self.write(
            "src/user.cpp",
            '#include "lib.h"\nint Use() { return Add(2, 3); }\n',
        )
        self.commit("touch user.cpp")
        report = self.check("--base", self.base)
        self.assertEqual(0, report.returncode, report)
        self.assertIn("1 of 1 translation unit", report.text, report)

    def test_an_uncommitted_change_is_in_scope(self) -> None:
        """A pre-commit run has to see the tree in hand, not only the range."""

        self.write(
            "src/user.cpp",
            '#include "lib.h"\n'
            "// recipe \\\n"
            "//   continued\n"
            "int Use() { return Add(1, 2); }\n",
        )
        report = self.check("--base", self.base)
        self.assertEqual(1, report.returncode, report)
        self.assertIn("src/user.cpp", report.text, report)


class CannotVerifyTests(ScratchProject):
    def test_an_unresolvable_base_is_cannot_verify_and_not_an_empty_range(
        self,
    ) -> None:
        report = self.check("--base", "refs/heads/no-such-branch")
        self.assertEqual(2, report.returncode, report)
        self.assertIn("CANNOT-VERIFY", report.text, report)

    def test_a_configure_failure_is_cannot_verify_and_not_a_pass(self) -> None:
        self.write("CMakeLists.txt", CMAKELISTS + "\nmessage(FATAL_ERROR \"nope\")\n")
        self.write("src/user.cpp", USER.replace("Add(1, 2)", "Add(3, 4)"))
        self.commit("break the configure")
        report = self.check("--base", self.base)
        self.assertEqual(2, report.returncode, report)
        self.assertIn("CANNOT-VERIFY", report.text, report)

    def test_a_source_no_target_compiles_is_named_and_not_counted(self) -> None:
        """Un-run is not passing, and it is not failing either.

        A `.cpp` no target compiles -- a `.cu` behind an OFF option, an
        orphaned file -- is reported by name so a reader can see it was never
        handed to a compiler. Folding it into the compiled total is how a build
        that never ran reads as a build that passed.
        """

        self.write("src/orphan.cpp", "int Orphan() { return 1; }\n")
        self.commit("add a file no target builds")
        report = self.check("--base", self.base)
        self.assertEqual(0, report.returncode, report)
        self.assertIn("src/orphan.cpp", report.text, report)
        self.assertIn("no target in this configuration compiles", report.text, report)
        self.assertNotIn("1 of 1 translation unit", report.text, report)


class ReportTests(ScratchProject):
    def test_the_report_names_the_base_the_head_and_the_count(self) -> None:
        """The instrument states what it compared, in its own output.

        A reader who cannot see the range and the derived count in the report
        cannot tell a scope that covered the change from one that covered
        nothing, and both print the same verdict.
        """

        self.write("src/user.cpp", USER.replace("Add(1, 2)", "Add(4, 5)"))
        self.commit("touch user.cpp")
        report = self.check("--base", self.base)
        self.assertEqual(0, report.returncode, report)
        self.assertIn(self.base[:12], report.text, report)
        self.assertIn(self.rev("HEAD")[:12], report.text, report)
        self.assertIn("in scope:", report.text, report)


INERT_PYTHON3 = """\
#!/usr/bin/env bash
# Every record checker succeeds; the compile gate returns what the test chose.
# Each invocation of the compile gate is LOGGED, because how many times preflight
# runs it is itself under test: the discovered `scripts/check-*.py` sweep runs
# every name not listed in NAMED_CHECKERS, and a second run of this one is a
# second full -fsyntax-only pass rather than a cheap usage error.
for arg in "$@"; do
  case "$arg" in
    */check-tree-compiles.py|check-tree-compiles.py)
      [ -n "${VLLM_TEST_COMPILE_LOG:-}" ] && printf '%s\\n' "$*" >> "$VLLM_TEST_COMPILE_LOG"
      exit "${VLLM_TEST_COMPILE_RC:-0}" ;;
  esac
done
exit 0
"""


class PreflightMappingTests(unittest.TestCase):
    """Preflight maps the checker's third exit code onto its third state.

    Executed rather than grepped: a text assertion is satisfied by a `case`
    branch that has moved rather than by one that runs.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="tree-compiles-preflight-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "scripts").mkdir()
        (self.tmp / "bin").mkdir()
        shutil.copy2(PREFLIGHT, self.tmp / "scripts" / PREFLIGHT.name)
        self.script = self.tmp / "scripts" / PREFLIGHT.name
        stub = self.tmp / "bin" / "python3"
        stub.write_text(INERT_PYTHON3, encoding="utf-8")
        stub.chmod(0o755)
        self.invocations = self.tmp / "compile-gate.log"
        # The scratch `scripts/` must CONTAIN the checker, or the discovered
        # `scripts/check-*.py` sweep has nothing to find and the double-run case
        # below would pass over a sweep that never ran.
        (self.tmp / "scripts" / "check-tree-compiles.py").write_text(
            "# stand-in; the stub python3 above decides the exit code\n",
            encoding="utf-8",
        )

        self.git("init", "--quiet", ".")
        self.git("config", "user.email", "tree-compiles@test.invalid")
        self.git("config", "user.name", "tree compiles test")
        self.git("config", "commit.gpgsign", "false")
        (self.tmp / "a").write_text("a\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "base")
        self.git("update-ref", "refs/remotes/origin/main", self.git("rev-parse", "HEAD"))

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.tmp, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    def preflight(self, rc: str) -> Report:
        environment = dict(os.environ)
        environment["PATH"] = f"{self.tmp / 'bin'}{os.pathsep}{environment['PATH']}"
        environment["VLLM_TEST_COMPILE_RC"] = rc
        environment["VLLM_TEST_COMPILE_LOG"] = str(self.invocations)
        self.invocations.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(self.script), "--quiet", "--no-require-role"],
            cwd=self.tmp,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        return Report(result.returncode, result.stdout + result.stderr)

    def compile_line(self, report: Report) -> str:
        for line in report.text.splitlines():
            if "tree-compiles" in line and line.startswith("  "):
                return line
        return ""

    def test_exit_zero_reports_ok(self) -> None:
        report = self.preflight("0")
        self.assertIn("ok", self.compile_line(report), report)

    def test_exit_one_reports_fail(self) -> None:
        report = self.preflight("1")
        self.assertIn("FAIL", self.compile_line(report), report)
        self.assertNotIn("All gates green.", report.text, report)

    def test_preflight_runs_the_compile_gate_exactly_once(self) -> None:
        """The discovered sweep must not start a second compile pass.

        RED before `check-tree-compiles.py` joined `NAMED_CHECKERS`: the
        `scripts/check-*.py` sweep runs every name that list does not carry, so
        the gate ran twice. For the four names already on that list the second
        run is a cheap argparse usage error. For this one it is a second full
        `-fsyntax-only` pass at -j8 -- 65 s, 39 units and 531 MB measured -- and
        it starts while the first has only just finished. Parallel builds have
        OOM-killed this box before.
        """

        report = self.preflight("0")
        runs = [
            line
            for line in self.invocations.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(
            1,
            len(runs),
            f"preflight invoked the compile gate {len(runs)} time(s):\n"
            + "\n".join(runs)
            + f"\n{report}",
        )
        # Precondition: the one run is the deliberate block, which passes a base.
        self.assertIn("--base", runs[0], report)

    def test_exit_two_reports_skip_and_denies_the_green_banner(self) -> None:
        report = self.preflight("2")
        self.assertIn("SKIP", self.compile_line(report), report)
        self.assertNotIn("All gates green.", report.text, report)


class RegistrationTests(unittest.TestCase):
    def test_the_checker_exists_and_preflight_runs_it(self) -> None:
        self.assertTrue(CHECKER.is_file(), f"{CHECKER} does not exist")
        text = PREFLIGHT.read_text(encoding="utf-8")
        self.assertIn("check-tree-compiles.py", text)
        self.assertIn("test_check_tree_compiles", text)

    def test_the_suite_is_registered_on_the_ci_script_lane(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("tests/scripts/test_check_tree_compiles.py", workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
