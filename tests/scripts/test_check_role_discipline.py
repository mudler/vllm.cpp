#!/usr/bin/env python3
"""Mutation tests for the arrival rule: every change lands on a task branch.

The checker's own docstring is the authority on what it enforces. These tests
pin the two regimes it now has:

* pre-worktree-cutover, integration paths (scripts/, .agents/, docs/, .github/,
  AGENTS.md) are EXEMPT, because that history was made under the direct-push
  rule and reddening it retroactively would be dishonest;
* post-cutover, nothing is exempt -- every tracked path must arrive on a task
  branch, which is what makes "the shared checkout is never a work surface"
  enforceable rather than merely written down.

The `govern_integration=True` cases are the red-before evidence: on the parent
commit the parameter does not exist and the exemption is unconditional.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import subprocess
import sys
import tokenize
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "check_role_discipline", ROOT / "scripts/check-role-discipline.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


DIRECT = dict(parents=["a" * 40], subject="policy: tighten a rule", body="")


def violations(paths, *, govern_integration=False, **overrides):
    """Run the arrival rule over *paths* for a single-parent direct commit.

    The keyword is omitted unless it is needed, so the legacy-regime cases call
    the pre-change signature and stay green on both sides of the cutover; only
    the worktree-regime cases depend on the new parameter existing.
    """
    call = {**DIRECT, **overrides}
    extra = {"govern_integration": True} if govern_integration else {}
    return checker.policy_commit_violations(
        "deadbee",
        call["parents"],
        call["subject"],
        call["body"],
        paths,
        (),
        **extra,
    )


class IntegrationPathClassification(unittest.TestCase):
    def test_integration_trees_and_top_files_are_integration(self) -> None:
        for path in (
            "scripts/agent-start.py",
            "scripts/check-role-discipline.py",
            ".agents/NOW.md",
            ".agents/specs/worktree-isolation.md",
            "docs/STATUS.md",
            ".github/workflows/ci.yml",
            "tests/scripts/test_check_role_discipline.py",
            "AGENTS.md",
            "CLAUDE.md",
            "README.md",
        ):
            with self.subTest(path=path):
                self.assertTrue(checker.is_integration_path(path))

    def test_product_paths_are_not_integration(self) -> None:
        for path in (
            "src/engine.cpp",
            "include/vllm.h",
            "tests/vt/test_gemv.cpp",
            "CMakeLists.txt",
            "cmake/cuda.cmake",
        ):
            with self.subTest(path=path):
                self.assertFalse(checker.is_integration_path(path))
                self.assertTrue(checker.is_feature_path(path))

    def test_malformed_paths_fail_closed_as_feature(self) -> None:
        for path in ("/etc/passwd", "../escape.cpp", "a//b.cpp", "", "src\\win.cpp"):
            with self.subTest(path=path):
                self.assertTrue(checker.is_feature_path(path))


class LegacyRegimeKeepsTheIntegrationExemption(unittest.TestCase):
    """Before the worktree cutover, a direct integration push is allowed."""

    def test_integration_only_commit_is_exempt(self) -> None:
        self.assertEqual(violations(["AGENTS.md", ".agents/NOW.md"]), [])

    def test_feature_path_is_still_governed(self) -> None:
        problems = violations(["src/engine.cpp"])
        self.assertEqual(len(problems), 1)
        self.assertIn("src/engine.cpp", problems[0])


class WorktreeRegimeGovernsEverything(unittest.TestCase):
    """RED BEFORE: on the parent commit this parameter does not exist."""

    def test_integration_only_commit_is_now_governed(self) -> None:
        problems = violations(["AGENTS.md", ".agents/NOW.md"], govern_integration=True)
        self.assertEqual(len(problems), 1)
        self.assertIn("AGENTS.md", problems[0])
        self.assertIn("task branch", problems[0])

    def test_a_checker_repair_may_no_longer_go_straight_to_main(self) -> None:
        problems = violations(
            ["scripts/check-role-discipline.py"], govern_integration=True
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("shared checkout", problems[0])

    def test_arriving_on_a_row_branch_satisfies_the_rule(self) -> None:
        self.assertEqual(
            violations(
                ["AGENTS.md"],
                govern_integration=True,
                subject="policy: worktree isolation (#210)",
            ),
            [],
        )

    def test_a_merge_naming_the_row_branch_satisfies_the_rule(self) -> None:
        self.assertEqual(
            violations(
                ["AGENTS.md", "scripts/check-role-discipline.py"],
                govern_integration=True,
                parents=["a" * 40, "b" * 40],
                subject="Merge branch 'row/POLICY-WORK-WORKTREE'",
            ),
            [],
        )

    def test_preview_truncates_and_counts_the_remainder(self) -> None:
        paths = [f"docs/d{i}.md" for i in range(7)]
        problems = violations(paths, govern_integration=True)
        self.assertIn("... (+3)", problems[0])


class ArrivalDetection(unittest.TestCase):
    def test_squash_merge_carrying_a_pr_number_arrives(self) -> None:
        self.assertTrue(
            checker.arrives_via_row_pr(["a" * 40], "fix(ci): unbreak it (#194)", "")
        )

    def test_plain_direct_commit_does_not_arrive(self) -> None:
        self.assertFalse(
            checker.arrives_via_row_pr(["a" * 40], "fix: quick repair", "")
        )

    def test_synthetic_merge_inherits_arrival_from_its_second_parent(self) -> None:
        # GitHub's refs/pull/N/merge names neither the row nor the PR; the
        # reviewed content is the second parent, one hop away.
        self.assertTrue(
            checker.arrives_via_row_pr(
                ["a" * 40, "b" * 40],
                "Merge 1234567 into 89abcde",
                "",
                ("policy: tighten arrival\n\nRow: row/POLICY-WORK-WORKTREE",),
            )
        )

    def test_synthetic_merge_of_an_unnamed_branch_does_not_arrive(self) -> None:
        self.assertFalse(
            checker.arrives_via_row_pr(
                ["a" * 40, "b" * 40],
                "Merge 1234567 into 89abcde",
                "",
                ("wip: local scratch",),
            )
        )


def executable_source(path: Path) -> str:
    """`path`'s source with its PROSE blanked: comments and docstrings.

    The assertion below forbids three ref-resolving substrings anywhere in the
    checker. Applied to the RAW text it forbade them in English too, so a
    comment explaining why this checker does not consult `origin/main` reddened
    the suite with a message about a lookup that was not there (#1776). This is
    the `code_lines` idiom from `test_main_baseline.py`, widened from whole-line
    YAML comments to Python's two prose surfaces.

    String LITERALS are deliberately KEPT. A real ref lookup is spelled as one
    -- `git("rev-parse", "origin/main")` -- so dropping strings would delete the
    obligation instead of narrowing it. Line numbering is preserved, so a hit
    reports where it is.
    """

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    prose: set[int] = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            prose.update(range(first.lineno - 1, first.end_lineno))
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        row = token.start[0] - 1
        if row not in prose:
            lines[row] = lines[row][: token.start[1]]
    return "\n".join("" if index in prose else line
                     for index, line in enumerate(lines))


class ArrivalDiscriminatorTests(unittest.TestCase):
    """WHERE the evidence of arrival must live, pinned so it cannot be widened.

    #1764 and #1773 both carry the hypothesis that these commits fail because an
    external contributor's branch lives on a fork and cannot be found on
    `origin`. The hypothesis is refuted by this class's first test: the checker
    resolves no ref at all. For a single-parent commit the whole decision is
    `ROW_BRANCH.search(subject + body) or PR_REFERENCE.search(SUBJECT)`, and the
    one NON-fork commit of the five being re-flagged fails identically to the
    four fork ones.

    What the five actually share is a subject with no `(#N)`, because the merger
    supplied an explicit `commit_title` and suppressed the append GitHub
    otherwise makes. The tempting repair is to look for `#N` anywhere in the
    message instead of in the subject. That deletes the obligation: AGENTS.md
    requires every change to start from an issue, so EVERY commit body in this
    repository names one, and a body-wide match passes every direct-to-main push
    ever made. `test_a_body_only_issue_reference_does_not_satisfy_arrival` is
    that mutation, held shut.
    """

    # `dd8a3b0e1`, verbatim: a real fork squash that the gate flags, with the
    # body reference that must NOT rescue it.
    FORK_SUBJECT = "windows: fix native MSVC/Vulkan build portability"
    FORK_BODY = "Removes POSIX-only constructs ...\n\nIssue: #503\n"

    def test_the_checker_resolves_no_ref_to_decide_arrival(self) -> None:
        source = executable_source(ROOT / "scripts/check-role-discipline.py")
        for forbidden in ("ls-remote", "for-each-ref", "origin/"):
            hits = [
                f"{number}: {line.strip()}"
                for number, line in enumerate(source.splitlines(), start=1)
                if forbidden in line
            ]
            self.assertEqual(
                hits, [],
                f"arrival is decided from the commit message, and {forbidden!r} "
                "resolves a ref. A lookup here would make the fork hypothesis "
                "testable, and that is not what this checker does",
            )

    def test_a_fork_squash_carrying_the_pr_number_arrives(self) -> None:
        """The rule is satisfiable by an external contributor. A fork pull
        request merged with the DEFAULT squash title passes, because GitHub
        appends the number to it."""
        self.assertTrue(
            checker.arrives_via_row_pr(
                ["a" * 40], f"{self.FORK_SUBJECT} (#640)", self.FORK_BODY
            )
        )

    def test_a_body_only_issue_reference_does_not_satisfy_arrival(self) -> None:
        """The widening that must never land."""
        self.assertFalse(
            checker.arrives_via_row_pr(
                ["a" * 40], self.FORK_SUBJECT, self.FORK_BODY
            )
        )

    def test_a_direct_push_naming_its_issue_in_the_body_still_fails(self) -> None:
        """The same widening, seen from the case the gate exists for."""
        self.assertTrue(
            violations(
                ["src/vllm/engine.cpp"],
                govern_integration=True,
                subject="fix: quick repair on the shared checkout",
                body="Issue: #1773\n\nFOLLOWING_AGENTS_PROTOCOL\n",
            )
        )


class CutoverWiring(unittest.TestCase):
    def test_role_cutover_is_a_full_sha(self) -> None:
        self.assertRegex(checker.ROLE_DISCIPLINE_SINCE or "", r"\A[0-9a-f]{40}\Z")

    def test_worktree_cutover_is_unset_or_a_full_sha(self) -> None:
        since = checker.WORKTREE_DISCIPLINE_SINCE
        if since is not None:
            self.assertRegex(since, r"\A[0-9a-f]{40}\Z")

    def test_cutover_shas_resolve_to_real_commits(self) -> None:
        """A dangling cutover SHA fails OPEN, silently disabling the rule.

        `_since` swallows CalledProcessError and returns False, so a typo'd or
        rebased-away SHA turns enforcement off with a green gate -- exactly the
        "turn a red gate green by widening a scope" failure AGENTS.md forbids.
        """
        for name in ("ROLE_DISCIPLINE_SINCE", "WORKTREE_DISCIPLINE_SINCE"):
            since = getattr(checker, name)
            if since is None:
                continue
            with self.subTest(cutover=name):
                resolved = subprocess.run(
                    ["git", "cat-file", "-e", f"{since}^{{commit}}"],
                    cwd=ROOT,
                    capture_output=True,
                )
                self.assertEqual(
                    resolved.returncode, 0, f"{name}={since} is not a commit in this repo"
                )

    def test_unset_cutover_never_enforces(self) -> None:
        original = checker.WORKTREE_DISCIPLINE_SINCE
        checker.WORKTREE_DISCIPLINE_SINCE = None
        try:
            self.assertFalse(checker.worktree_enforced("HEAD"))
        finally:
            checker.WORKTREE_DISCIPLINE_SINCE = original


if __name__ == "__main__":
    unittest.main()
