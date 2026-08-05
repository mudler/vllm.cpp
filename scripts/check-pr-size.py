#!/usr/bin/env python3
"""Keep helper PRs cheap to review. (W4)

Parallel helpers plus a serial operator means review is the bottleneck, and the
mitigation is that each PR is small enough to review quickly: one row, bounded
diff. The cap applies to `row/*` PRs, which are the protocol's PRs. Other
branches are reported, not failed, so pre-existing work is not retroactively
punished for a rule it was not written under.

Record and protocol paths are exempt: an honest state entry or a matrix update
can be long, and shrinking evidence to satisfy a line cap would be the wrong
incentive entirely.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAX_FEATURE_LINES = 900
EXEMPT_PREFIXES = (".agents/", "docs/", "tests/scripts/", "scripts/", ".github/")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def counted_lines(base: str, head: str) -> tuple[int, list[str]]:
    out = git("diff", "--numstat", f"{base}...{head}")
    total, files = 0, []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, removed, path = parts
        if path.startswith(EXEMPT_PREFIXES):
            continue
        if added == "-" or removed == "-":
            continue
        total += int(added) + int(removed)
        files.append(path)
    return total, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--branch", default="", help="head branch name")
    args = parser.parse_args()

    total, files = counted_lines(args.base, args.head)
    is_row_pr = args.branch.startswith("row/")
    if total <= MAX_FEATURE_LINES:
        print(f"OK: {total} non-exempt changed lines across {len(files)} file(s).")
        return 0

    message = (
        f"{total} non-exempt changed lines across {len(files)} file(s), over the "
        f"{MAX_FEATURE_LINES} cap. Split into one row per PR so review stays "
        "cheap; record and protocol paths are already exempt."
    )
    if is_row_pr:
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    print(f"REPORT: {message} (not a row/* PR, not enforced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
