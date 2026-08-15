#!/usr/bin/env python3
"""The issue index is append-only, proven against the merge base.

`.agents/issue-index.md` carries `merge=union` so two branches that each append
a row merge without a conflict. That driver is only safe while rows are never
edited and never deleted: union merge DUPLICATES an edited line rather than
merging it, and it does so silently.

`check-agent-record.py` sees one tree and cannot tell an appended row from an
edited one. That needs a base, so it lives here.

Deliberately NETWORK-FREE, like every other checker in this repository except
`check-release-workflow.py`. It reads Git objects that are already local.

Absence of a base is reported as SKIP, never as PASS. An unknown is not a
success, and it must not read as one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ".agents/issue-index.md"


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()

    if git("rev-parse", "--verify", "-q", args.base).returncode != 0:
        print(f"SKIP: issue-index append-only: {args.base} does not resolve here")
        return 0
    merge_base = git("merge-base", args.base, args.head)
    if merge_base.returncode != 0 or not merge_base.stdout.strip():
        print(f"SKIP: issue-index append-only: no merge base for {args.base}..{args.head}")
        return 0

    diff = git(
        "diff", "--unified=0", f"{merge_base.stdout.strip()}..{args.head}", "--", INDEX
    )
    if diff.returncode != 0:
        print(f"SKIP: issue-index append-only: cannot diff {INDEX}")
        return 0

    removed = [
        line[1:]
        for line in diff.stdout.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    if not removed:
        print("OK: issue index append-only")
        return 0

    print(f"FAIL: {INDEX} is append-only, and this range removes or edits lines.")
    print("A union merge duplicates an edited line instead of merging it.")
    print("Append a new row at the end. GitHub holds the open and closed state,")
    print("so closing an issue costs no edit here.")
    for line in removed[:10]:
        print(f"  removed: {line[:100]}")
    if len(removed) > 10:
        print(f"  ... and {len(removed) - 10} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
