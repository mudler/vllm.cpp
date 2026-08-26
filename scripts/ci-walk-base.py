#!/usr/bin/env python3
"""Resolve the BASE commit for the diff-scoped gates, clamped by a recorded floor.

`.github/workflows/ci.yml` runs four diff-scoped steps over `BASE..HEAD`: both
trailer steps of `commit-protocol-tag`, `documentation-checkpoint`, and
`agent-record`'s role-discipline step. Each one used to choose its own base in a
byte-similar copy of the same inline shell, and no test in this tree executed any
of those four blocks.

WHAT THE BASE IS. On the pull-request lane it is `pull_request.base.sha`. On the
push lane it is the head of the last SUCCESSFUL push run of this workflow,
falling back to `github.event.before`. The successful-run base is deliberate: a
cancelled run must not advance the base, or its commits are skipped and nothing
re-covers them, which is what lets the push lane be latest-only (#822, #863).

WHY IT NEEDS A FLOOR. That design assumes a green run is eventually reachable.
A commit already on `main` that violates a gate cannot be repaired, because the
only repair is a rewrite `AGENTS.md` forbids, so no green run is reachable, the
base freezes, and every later push re-walks the same violations over a range one
commit wider (#1809). The floor -- `scripts/ci-enforcement-floor.txt` -- is one
commit the walk never goes behind.

CANCELLED RUNS STAY LOSSLESS. The floor is a LOWER CLAMP on an otherwise
unchanged base. While it sits behind the last green commit, which is the steady
state because that commit advances on every green push and the floor advances
only when a human commits an advance, the resolved base is byte-identical to
what it was before this script existed. The only window that skips anything is
`last_green..floor` immediately after an advance, which is the forgiveness being
asked for and is enumerated in `.agents/specs/ci-enforcement-floor.md`.

NEWER MEANS ANCESTRY, NOT DATE. A commit date is author-controlled and can move
backwards across a rebase, so comparing dates can pick the wrong commit.
`git merge-base --is-ancestor` is the honest primitive and is what is used here.

Unit-tested by tests/scripts/test_ci_walk_base.py.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOOR_FILE = ROOT / "scripts/ci-enforcement-floor.txt"

SHA = re.compile(r"\A[0-9a-f]{40}\Z")


class FloorError(ValueError):
    """The recorded floor cannot be read as one commit sha."""


def read_floor(path: Path) -> str:
    """Return the single sha recorded in ``path``.

    Fails closed. A floor record that cannot be read as exactly one lowercase
    40-byte sha is an error rather than "no floor": silently reading a broken
    record as absent restores the ratchet this file exists to break, and does it
    without saying so.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FloorError(f"cannot read the enforcement floor {path}: {exc}") from exc
    values = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != 1:
        raise FloorError(
            f"{path} must hold exactly one commit sha outside its comments, found {len(values)}"
        )
    if SHA.fullmatch(values[0]) is None:
        raise FloorError(
            f"{path} must hold one lowercase 40-byte commit sha, found {values[0]!r}"
        )
    return values[0]


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


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    """Whether ``older`` is an ancestor of ``newer``. Equal commits count."""

    return _git(repo, "merge-base", "--is-ancestor", older, newer).returncode == 0


def resolve_base(
    *,
    repo: Path,
    event: str,
    head: str,
    pr_base: str,
    push_base: str,
    last_green: str,
    floor: str,
    warn=lambda message: print(message, file=sys.stderr),
) -> str:
    """Return the base commit the diff-scoped walk starts from."""

    if event == "pull_request":
        # UNCHANGED, on purpose. This lane bases on the merge base and has been
        # green throughout the freeze the floor exists to end. Applying the
        # floor here could only ever RAISE a base, which narrows what the lane
        # enforces, and no defect asks for that.
        return pr_base

    base = last_green or push_base

    if not base or not known(repo, base):
        # The all-zero sha of a new branch, or a `before` the history no longer
        # contains. The caller's own guard degrades to the tip commit alone, and
        # that behaviour is preserved byte-for-byte: the floor RAISES a usable
        # base and never substitutes for an unusable one.
        return base

    if not floor:
        return base

    if not known(repo, floor):
        warn(f"enforcement floor {floor} is not in this checkout; base left at {base}")
        return base

    if not is_ancestor(repo, floor, head):
        # A floor the current history does not contain cannot bound it, and
        # `floor..head` across unrelated history is not a range anybody asked
        # for. This is also the guard against a floor typed AHEAD of HEAD.
        warn(f"enforcement floor {floor} is not an ancestor of {head}; base left at {base}")
        return base

    if is_ancestor(repo, floor, base):
        return base

    warn(f"base {base} is behind the enforcement floor; walking from {floor} instead")
    return floor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, help="github.event_name")
    parser.add_argument("--head", default="", help="the walk's head commit")
    parser.add_argument("--pr-base", default="", help="pull_request.base.sha")
    parser.add_argument("--push-base", default="", help="github.event.before")
    parser.add_argument("--last-green", default="", help="head of the last successful push run")
    parser.add_argument(
        "--floor",
        default=None,
        help="override the recorded floor; an empty value means no floor",
    )
    parser.add_argument(
        "--floor-file",
        default=None,
        help=f"read the floor from this file instead of {DEFAULT_FLOOR_FILE}",
    )
    parser.add_argument("--repo", default=".", help="repository to resolve commits in")
    args = parser.parse_args(argv)

    if args.floor is not None:
        floor = args.floor.strip()
        if floor and SHA.fullmatch(floor) is None:
            print(f"--floor must be one lowercase 40-byte commit sha, got {floor!r}", file=sys.stderr)
            return 2
    else:
        path = Path(args.floor_file) if args.floor_file else DEFAULT_FLOOR_FILE
        try:
            floor = read_floor(path)
        except FloorError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    base = resolve_base(
        repo=Path(args.repo),
        event=args.event,
        head=args.head,
        pr_base=args.pr_base,
        push_base=args.push_base,
        last_green=args.last_green,
        floor=floor,
    )
    print(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
