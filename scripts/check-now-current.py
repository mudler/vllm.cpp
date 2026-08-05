#!/usr/bin/env python3
"""Keep .agents/NOW.md a short, current, one-Read resume surface.

The canonical record is large by design (state.md and parity-ledger.md are
megabytes of append-only evidence, and that is correct -- evidence must not be
deleted). But that made the cold-start path expensive: an agent was told to
orient from the roadmap, the owning matrix, coordination.md and the newest state
entries, none of which is cheap to locate inside files of that size.

NOW.md is the fix: the single small file a cold session reads first to become
productive. It is a SNAPSHOT, never a log -- it is rewritten in place, and the
detail it summarises stays in the append-only record.

Two obligations are enforced:
  * structure and budget, so it cannot decay into another status log;
  * freshness, so it cannot silently go stale. Any change that appends to
    .agents/state.md must refresh NOW.md in that same change, because a state
    append is exactly the event that moves what is live.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOW = ROOT / ".agents/NOW.md"
NOW_PATH = ".agents/NOW.md"

# A state append is the event that changes what is live, so it is the trigger.
FRESHNESS_TRIGGERS = {".agents/state.md"}

# Budgets. NOW.md exists to be read in full, every session, by every agent. The
# moment it stops fitting in one screenful of attention it has become the thing
# it was meant to replace.
MAX_LINES = 100
MAX_CHARS = 6000
MAX_ENTRY_CHARS = 400

REQUIRED_HEADINGS = (
    "live claims",
    "current gate",
    "next actions",
)

STAMP = re.compile(r"^<!--\s*now-updated:\s*(\d{4}-\d{2}-\d{2})\s*-->$", re.MULTILINE)


def structure_errors(text: str) -> list[str]:
    """Return budget/shape problems with the NOW digest."""
    errors: list[str] = []

    if not STAMP.search(text):
        errors.append(
            "missing the freshness stamp <!-- now-updated: YYYY-MM-DD -->; it "
            "records when this snapshot was last known true"
        )

    lowered = text.lower()
    for heading in REQUIRED_HEADINGS:
        if f"## {heading}" not in lowered:
            errors.append(
                f"missing the '## {heading}' section; a cold session needs all "
                f"of {', '.join(REQUIRED_HEADINGS)} to resume without reading "
                "the full record"
            )

    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        errors.append(
            f"is {len(lines)} lines, over the {MAX_LINES}-line budget; move "
            "detail to .agents/state.md and keep only the live position here"
        )
    if len(text) > MAX_CHARS:
        errors.append(
            f"is {len(text)} characters, over the {MAX_CHARS}-character budget; "
            "this is a digest, not a status log"
        )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("-", "|")) and len(stripped) > MAX_ENTRY_CHARS:
            errors.append(
                f"an entry is {len(stripped)} characters, over the "
                f"{MAX_ENTRY_CHARS}-character budget: {stripped[:60]!r}...; "
                "link the spec or state entry instead of inlining the narrative"
            )

    return errors


def state_entries_changed(commit: str) -> bool:
    """Whether this commit ADDED or REMOVED a state entry, vs merely reordering.

    `sort-state-tail.py` rewrites the log to repair an interleave caused by
    concurrent sessions. That moves no entry in or out and changes nothing about
    what is live, so demanding a NOW.md refresh for it is an over-trigger: the
    rule is "a state APPEND moves what is live", not "any byte changed".
    """
    import re
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if not parents:
        return True
    def headings(rev: str) -> set[str]:
        try:
            blob = git("show", f"{rev}:.agents/state.md")
        except subprocess.CalledProcessError:
            return set()
        return set(re.findall(r"^## (.+)$", blob, re.M))
    return headings(parents[0]) != headings(commit)


def freshness_errors(paths: set[str], entries_changed: bool = True) -> list[str]:
    """Return staleness problems for one atomic change."""
    triggers = sorted(FRESHNESS_TRIGGERS & paths)
    if triggers and not entries_changed:
        return []  # a pure reorder: no entry added, nothing live moved
    if triggers and NOW_PATH not in paths:
        return [
            f"{', '.join(triggers)} changed but {NOW_PATH} did not; a state "
            "append moves what is live, so refresh the digest in the same "
            "change (live claims, current gate, next actions, stamp)"
        ]
    return []


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def commit_paths(commit: str) -> set[str]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if parents:
        output = git("diff", "--name-only", parents[0], commit)
    else:
        output = git(
            "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit
        )
    return {line for line in output.splitlines() if line}


def commits_in_range(base: str, head: str) -> list[str]:
    try:
        git("cat-file", "-e", f"{base}^{{commit}}")
    except subprocess.CalledProcessError:
        return [head]
    output = git("rev-list", "--reverse", "--no-merges", f"{base}..{head}")
    return [line for line in output.splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--commit", default=None, help="check one commit")
    source.add_argument(
        "--staged", action="store_true", help="check the current staged change"
    )
    parser.add_argument("--base", help="check every commit after this revision")
    parser.add_argument("--head", help="range endpoint (requires --base)")
    args = parser.parse_args()
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be supplied together")
    if args.base is not None and (args.commit is not None or args.staged):
        parser.error("a revision range cannot be combined with --commit/--staged")
    return args


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    if not NOW.exists():
        print(f"ERROR: {NOW_PATH} does not exist", file=sys.stderr)
        return 1

    failures.extend(
        f"{NOW_PATH} {error}"
        for error in structure_errors(NOW.read_text(encoding="utf-8"))
    )

    if args.staged:
        paths = set(git("diff", "--cached", "--name-only").splitlines())
        failures.extend(f"staged change: {error}" for error in freshness_errors(paths))
    elif args.base is not None:
        for commit in commits_in_range(args.base, args.head):
            short = git("rev-parse", "--short", commit)
            failures.extend(
                f"commit {short}: {error}"
                for error in freshness_errors(
                    commit_paths(commit), state_entries_changed(commit))
            )
    elif args.commit is not None:
        short = git("rev-parse", "--short", args.commit)
        failures.extend(
            f"commit {short}: {error}"
            for error in freshness_errors(
                commit_paths(args.commit), state_entries_changed(args.commit))
        )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "NOW.md is the one-Read resume surface: the live claims, the gate "
            "being chased, and the next actions, rewritten in place. Detail "
            "belongs in .agents/state.md and the area matrices.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {NOW_PATH} is a current, in-budget resume digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
