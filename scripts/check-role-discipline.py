#!/usr/bin/env python3
"""Feature code reaches main only through a reviewed row/* PR. (W1)

This is how "the operator drives features only via sub-agents" is enforced. It
deliberately does NOT try to detect who typed the code -- authorship is
self-asserted and unprovable. It enforces the PATH instead: feature code arrives
on `main` through a merged `row/<ROW-ID>` PR, whoever produced it. A sub-agent, a
helper session, or the developer all satisfy it the same way, and a direct push
of feature code does not.

Integration work stays direct on purpose. `scripts/`, `.agents/`, `docs/` and
`.github/` are NOT feature paths, because the operator must be able to fix a
gate, resolve a conflict or repair the record without a round trip -- an operator
who cannot touch anything cannot review, which is the rubber-stamp failure the
protocol is designed to avoid.

ACTIVATION. ENFORCING since the cutover commit 44e8225cf (user-directed
2026-08-05). Every commit from that one ONWARD must land feature code through a
merged `row/*` PR; everything before it is exempt, because it was created under
the previous direct-push policy and rewriting that judgement retroactively would
redden honest history. The cutover itself is a records-only commit, so it passes.

What this changes in practice: feature paths (src/, include/, tests/, examples/,
cmake/, CMakeLists.txt) can no longer be pushed straight to main. Integration
paths (scripts/, .agents/, docs/, .github/) still can, deliberately, so the
operator can fix a gate or repair the record without a round trip.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Set to the cutover commit SHA to switch enforcement on. None = report only.
ROLE_DISCIPLINE_SINCE: str | None = "44e8225cf95fff12de6c5d4f3c3b4ecc9f0b1f94"

# Product code. A change here must arrive through a reviewed row/* PR.
FEATURE_PREFIXES = (
    "src/",
    "include/",
    "examples/",
    "tools/",
    "tests/",
    "cmake/",
)
FEATURE_FILES = {"CMakeLists.txt"}
# Protocol/record tooling the operator legitimately maintains in place.
INTEGRATION_PREFIXES = (
    "tests/scripts/",
    "scripts/",
    ".agents/",
    "docs/",
    ".github/",
)

ROW_BRANCH = re.compile(r"row/[A-Za-z0-9_.-]+")
PR_REFERENCE = re.compile(r"\(#\d+\)|#\d+")


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def is_feature_path(path: str) -> bool:
    if path.startswith(INTEGRATION_PREFIXES):
        return False
    return path in FEATURE_FILES or path.startswith(FEATURE_PREFIXES)


def arrives_via_row_pr(parents: list[str], subject: str, body: str) -> bool:
    """Whether this commit reached main through a reviewed row/* PR."""
    message = f"{subject}\n{body}"
    if len(parents) >= 2:
        # A merge commit is a PR merge when it names the branch or the PR.
        return bool(ROW_BRANCH.search(message) or PR_REFERENCE.search(message))
    # GitHub squash-merges land a single commit carrying "(#N)".
    return bool(ROW_BRANCH.search(message) or PR_REFERENCE.search(subject))


def commit_violations(
    commit: str, parents: list[str], subject: str, body: str, paths: list[str]
) -> list[str]:
    """Return the reasons this commit breaks role discipline (empty if fine)."""
    features = sorted(p for p in paths if is_feature_path(p))
    if not features:
        return []
    if arrives_via_row_pr(parents, subject, body):
        return []
    preview = ", ".join(features[:4])
    if len(features) > 4:
        preview += f", ... (+{len(features) - 4})"
    return [
        f"{commit}: feature code ({preview}) reached main without a reviewed "
        "row/* PR. Feature work goes through a helper session or a sub-agent on "
        "a `row/<ROW-ID>` branch; the operator merges it. Integration paths "
        "(scripts/, .agents/, docs/, .github/) are exempt by design"
    ]


def commit_paths(commit: str) -> list[str]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if parents:
        out = git("diff", "--name-only", parents[0], commit)
    else:
        out = git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit)
    return [line for line in out.splitlines() if line]


def inspect(commit: str) -> list[str]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    subject = git("log", "-1", "--format=%s", commit)
    body = git("log", "-1", "--format=%b", commit)
    short = git("rev-parse", "--short", commit)
    return commit_violations(short, parents, subject, body, commit_paths(commit))


def enforced(commit: str) -> bool:
    """True when this commit is after the cutover."""
    if ROLE_DISCIPLINE_SINCE is None:
        return False
    try:
        git("merge-base", "--is-ancestor", ROLE_DISCIPLINE_SINCE, commit)
        return True
    except subprocess.CalledProcessError:
        return False


def commits_in_range(base: str, head: str) -> list[str]:
    try:
        git("cat-file", "-e", f"{base}^{{commit}}")
    except subprocess.CalledProcessError:
        return [head]
    return [c for c in git("rev-list", "--reverse", f"{base}..{head}").splitlines() if c]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", help="check one commit")
    parser.add_argument("--base", help="check every commit after this revision")
    parser.add_argument("--head", help="range endpoint (requires --base)")
    args = parser.parse_args()
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be supplied together")

    if args.base is not None:
        commits = commits_in_range(args.base, args.head)
    elif args.commit:
        commits = [args.commit]
    else:
        commits = ["HEAD"]

    failures, reported = [], []
    for commit in commits:
        for problem in inspect(commit):
            (failures if enforced(commit) else reported).append(problem)

    for problem in reported:
        print(f"REPORT: {problem}", file=sys.stderr)
    for problem in failures:
        print(f"ERROR: {problem}", file=sys.stderr)

    if failures:
        return 1
    if ROLE_DISCIPLINE_SINCE is None:
        print(
            "OK: role discipline is REPORT-ONLY "
            f"({len(reported)} commit(s) would fail once ROLE_DISCIPLINE_SINCE "
            "names the cutover commit)."
        )
    else:
        print("OK: feature code on main arrived through reviewed row/* PRs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
