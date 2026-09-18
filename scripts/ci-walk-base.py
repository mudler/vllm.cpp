#!/usr/bin/env python3
"""Resolve the BASE commit for the diff-scoped gates.

`.github/workflows/ci.yml` runs diff-scoped steps over `BASE..HEAD`: the trailer
steps of `commit-protocol-tag` and `documentation-checkpoint`. Each one used to
choose its own base in a byte-similar copy of the same inline shell, and no test
in this tree executed any of those blocks.

WHAT THE BASE IS. On the pull-request lane it is `pull_request.base.sha`. On the
push lane it is the head of the last SUCCESSFUL push run of this workflow,
falling back to `github.event.before`. The successful-run base is deliberate: a
cancelled run must not advance the base, or its commits are skipped and nothing
re-covers them, which is what lets the push lane be latest-only (#822, #863).

NEWER MEANS ANCESTRY, NOT DATE. A commit date is author-controlled and can move
backwards across a rebase, so comparing dates can pick the wrong commit.
`git merge-base --is-ancestor` is the honest primitive and is what is used here.

Unit-tested by tests/scripts/test_ci_walk_base.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def known(repo: Path, revision: str) -> bool:
    """Whether git can resolve ``revision`` to a commit object in ``repo``."""

    if not revision:
        return False
    return _git(repo, "cat-file", "-e", f"{revision}^{{commit}}").returncode == 0


def resolve_base(
    *,
    repo: Path,
    event: str,
    head: str,
    pr_base: str,
    push_base: str,
    last_green: str,
) -> str:
    """Return the base commit the diff-scoped walk starts from."""

    if event == "pull_request":
        return pr_base

    base = last_green or push_base
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, help="github.event_name")
    parser.add_argument("--head", default="", help="the walk's head commit")
    parser.add_argument("--pr-base", default="", help="pull_request.base.sha")
    parser.add_argument("--push-base", default="", help="github.event.before")
    parser.add_argument("--last-green", default="", help="head of the last successful push run")
    parser.add_argument("--repo", default=".", help="repository to resolve commits in")
    args = parser.parse_args(argv)

    base = resolve_base(
        repo=Path(args.repo),
        event=args.event,
        head=args.head,
        pr_base=args.pr_base,
        push_base=args.push_base,
        last_green=args.last_green,
    )
    print(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
