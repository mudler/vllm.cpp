#!/usr/bin/env python3
"""`.githooks/pre-push` refuses a checker it names but cannot find (#1779).

Row `GATE-PREPUSH-FAIL-LOUD`, spec `.agents/specs/gate-prepush-fail-loud.md`.

The defect this suite pins: the hook declared six checkers and three of them had
already been deleted from `scripts/` -- `check-policy.py` and
`check-state-record.py` by `0f3e44eee`, `check-public-doc-tables.py` by
`1db7e59cf` (#1714). The loop guarded each name with

    [ -f "$work/scripts/$checker" ] || continue

so an absent checker was skipped without a word and the hook still exited 0. It
presented as six gates and was three. `core.hooksPath` is set to `.githooks` in
this checkout, so the hook runs on every push and reported success over half a
list that no longer existed.

The harm is not the three that were deliberately removed. It is that the hook
could not tell a deliberate removal from an accidental one: the next checker to
vanish would have been swallowed the same way, and the only signal available to
a reader was the absence of a line they had no reason to count.

## Why this suite executes the hook instead of reading it

Every assertion below could be spelled as a text match against the hook, and a
text match is satisfied by a `continue` that has moved three lines down rather
than disappeared. What is under test is control flow -- whether the loop body
reaches `failed=1` -- so the hook is executed against a scratch repository it
owns end to end. `tests/scripts/test_agent_preflight_skip_report.py` made the
same choice about `agent-preflight.sh` for the same reason.

The scratch repository is a real git repository with real commits, because the
hook materialises the pushed commit with `git worktree add` and then looks for
the checker inside THAT tree. A fixture that only wrote files into a directory
would exercise none of that.

## Why the scratch hook's list is rewritten

The list under test has to contain a name that does not exist, and the shipped
list must not: that is the whole point of the fix. So the copy in the scratch
repository has its `CHECKERS=(...)` array rewritten. The rewrite goes through
`checkers_in()`, the same parser `test_every_named_checker_exists` uses against
the shipped hook, and `test_the_rewrite_applies` asserts the copy actually
changed. A parser that stopped matching would fail those two rather than pass
this one vacuously.
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
HOOK = ROOT / ".githooks/pre-push"
HOOK_README = ROOT / ".githooks/README.md"
SCRIPTS = ROOT / "scripts"
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"
CI = ROOT / ".github/workflows/ci.yml"
SUITE_NAME = Path(__file__).stem

def _array_span(text: str) -> tuple[int, int, list[str]]:
    """`(first, last, names)` for the `CHECKERS=(...)` array, by line index.

    A line scanner rather than one regular expression, because the two arrays
    have different shapes: the hook writes one line, and
    `scripts/agent-preflight.sh` writes one name per line with comments between
    them -- and one of those comments contains `(#1417)`, which a `[^)]*` body
    would stop at, silently reading a short list as the whole list.
    """

    lines = text.splitlines()
    first = None
    for index, line in enumerate(lines):
        if line.startswith("CHECKERS=("):
            first = index
            break
    if first is None:
        raise AssertionError("no CHECKERS=( array found")
    names: list[str] = []
    for index in range(first, len(lines)):
        body = lines[index].split("#", 1)[0]
        if index == first:
            body = body[len("CHECKERS=(") :]
        closed = body.rstrip().endswith(")")
        names.extend(body.rstrip().removesuffix(")").split())
        if closed:
            return first, index, names
    raise AssertionError("the CHECKERS=( array is never closed")


def checkers_in(text: str) -> list[str]:
    """The names in a `CHECKERS=(...)` array, comments and blanks dropped."""

    return _array_span(text)[2]


def rewrite_checkers(text: str, names: list[str]) -> str:
    """Replace the `CHECKERS` array with `names`, preserving everything else."""

    first, last, _ = _array_span(text)
    lines = text.splitlines(keepends=True)
    replacement = "CHECKERS=(" + " ".join(names) + ")\n"
    return "".join(lines[:first]) + replacement + "".join(lines[last + 1 :])


PASSING_CHECKER = "import sys\n\nsys.exit(0)\n"
FAILING_CHECKER = "import sys\n\nprint('the tree is wrong')\nsys.exit(1)\n"


class ScratchPush:
    """A git repository the hook can be run against for real.

    `run` returns the completed process for a push of `head` over `base`, with
    the ref line the hook reads on stdin.
    """

    def __init__(self, root: Path, present: dict[str, str], named: list[str]):
        self.root = root
        env = dict(os.environ)
        env.update(
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@example.invalid",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@example.invalid",
        )
        self.env = env
        self._git("init", "--quiet", "-b", "main")
        (root / "scripts").mkdir()
        for name, body in present.items():
            (root / "scripts" / name).write_text(body, encoding="utf-8")
        hooks = root / ".githooks"
        hooks.mkdir()
        self.hook = hooks / "pre-push"
        self.hook.write_text(
            rewrite_checkers(HOOK.read_text(encoding="utf-8"), named), encoding="utf-8"
        )
        self.hook.chmod(0o755)
        self._git("add", "-A")
        self._git("commit", "--quiet", "-m", "base")
        self.base = self._git("rev-parse", "HEAD").stdout.strip()
        (root / "README.md").write_text("head commit\n", encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "--quiet", "-m", "head")
        self.head = self._git("rev-parse", "HEAD").stdout.strip()

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            text=True,
            capture_output=True,
            check=True,
            env=self.env,
        )

    def run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.hook), "origin", str(self.root)],
            cwd=self.root,
            input=f"refs/heads/main {self.head} refs/heads/main {self.base}\n",
            text=True,
            capture_output=True,
            check=False,
            env=self.env,
        )


class HookExecutionTests(unittest.TestCase):
    """What the hook DOES when a name in its list has no file."""

    def scratch(self, present: dict[str, str], named: list[str]) -> ScratchPush:
        directory = tempfile.mkdtemp(prefix="prepush-")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return ScratchPush(Path(directory), present, named)

    def test_a_named_checker_with_no_file_fails_the_push(self) -> None:
        """THE defect. Absent must not read as absent-and-fine."""

        push = self.scratch(
            {"check-present.py": PASSING_CHECKER},
            ["check-present.py", "check-vanished.py"],
        )
        result = push.run()
        self.assertNotEqual(
            result.returncode,
            0,
            "the hook accepted a push while a checker it names does not exist:\n"
            f"stdout={result.stdout!r} stderr={result.stderr!r}",
        )
        self.assertIn("check-vanished.py", result.stderr)

    def test_the_message_names_the_missing_checker_not_only_a_count(self) -> None:
        """A reader who cannot see WHICH name is dead cannot prune the list."""

        push = self.scratch(
            {"check-present.py": PASSING_CHECKER},
            ["check-gone-a.py", "check-present.py", "check-gone-b.py"],
        )
        result = push.run()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("check-gone-a.py", result.stderr)
        self.assertIn("check-gone-b.py", result.stderr)

    def test_a_list_that_is_wholly_present_and_passing_still_pushes(self) -> None:
        """The other direction. A gate that reds every push gets turned off."""

        push = self.scratch(
            {
                "check-present.py": PASSING_CHECKER,
                "check-second.py": PASSING_CHECKER,
            },
            ["check-present.py", "check-second.py"],
        )
        result = push.run()
        self.assertEqual(
            result.returncode,
            0,
            "the hook refused a push whose every named checker exists and "
            f"passes:\nstdout={result.stdout!r} stderr={result.stderr!r}",
        )

    def test_a_present_checker_that_fails_still_fails_the_push(self) -> None:
        """The behaviour that already worked is not traded for the new one."""

        push = self.scratch(
            {
                "check-present.py": PASSING_CHECKER,
                "check-angry.py": FAILING_CHECKER,
            },
            ["check-present.py", "check-angry.py"],
        )
        result = push.run()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("check-angry.py", result.stderr)
        self.assertIn("the tree is wrong", result.stderr)

    def test_the_rewrite_applies(self) -> None:
        """Guards every case above: a no-op rewrite would test the real list."""

        push = self.scratch({"check-present.py": PASSING_CHECKER}, ["check-only.py"])
        self.assertEqual(
            checkers_in(push.hook.read_text(encoding="utf-8")), ["check-only.py"]
        )
        self.assertNotEqual(
            push.hook.read_text(encoding="utf-8"), HOOK.read_text(encoding="utf-8")
        )


class ShippedListTests(unittest.TestCase):
    """The invariant the fix establishes over the tree as it stands."""

    def test_every_named_checker_exists(self) -> None:
        """What will catch the NEXT deletion, in review rather than at a push."""

        names = checkers_in(HOOK.read_text(encoding="utf-8"))
        self.assertTrue(names, "the hook names no checkers at all")
        missing = [name for name in names if not (SCRIPTS / name).is_file()]
        self.assertEqual(
            missing,
            [],
            f"{HOOK.relative_to(ROOT)} names checkers that scripts/ does not "
            "have. Delete the name in the same change that deletes the file",
        )

    def test_the_hook_and_preflight_agree(self) -> None:
        """The hook's own comment says to keep the two lists in sync (#1779)."""

        hook_names = {
            name.removesuffix(".py") for name in checkers_in(HOOK.read_text("utf-8"))
        }
        preflight_names = set(checkers_in(PREFLIGHT.read_text("utf-8")))
        self.assertEqual(
            sorted(hook_names - preflight_names),
            [],
            "the hook runs a checker that scripts/agent-preflight.sh does not, "
            "against the instruction in the hook's own comment",
        )

    def test_the_hook_readme_names_only_files_that_exist(self) -> None:
        """The same lie in prose: the README listed the deleted table gate."""

        text = HOOK_README.read_text(encoding="utf-8")
        referenced = sorted(set(re.findall(r"scripts/[\w.-]+\.(?:py|sh)", text)))
        self.assertTrue(referenced, "the README names no scripts at all")
        missing = [path for path in referenced if not (ROOT / path).is_file()]
        self.assertEqual(
            missing,
            [],
            f"{HOOK_README.relative_to(ROOT)} names scripts that do not exist",
        )


class RegistrationTests(unittest.TestCase):
    """A suite nothing runs is the shape this issue is about."""

    def test_preflight_runs_this_suite(self) -> None:
        self.assertIn(SUITE_NAME, PREFLIGHT.read_text(encoding="utf-8"))

    def test_ci_runs_this_suite(self) -> None:
        self.assertIn(f"tests/scripts/{SUITE_NAME}.py", CI.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
