#!/usr/bin/env python3
"""Fail if a lease script keeps its ccache cache on the CIFS share (#2473).

WHY THIS GATE EXISTS, in one paragraph, because the reading is misleading in a
specific direction. `rc describe <device>`'s host-wide usage sheet requires
ccache for every C, C++, CUDA and HIP build on this fleet and instructs that the
cache be kept on the NAS as `CCACHE_DIR=/workspace/ccache`. `/workspace` is CIFS
mounted `nounix`, and ccache 4.9.1 takes every cache AND stats lock by creating a
symlink, which that mount refuses with EOPNOTSUPP. So nothing is stored, no
counter is written, and `ccache -s` reports zero hits, zero misses and zero
stores. A cache that was consulted and empty records MISSES, so zero-of-
everything reads as "the launcher never ran" when it in fact means "every write
was refused". `rc` job 93a60151 paid a 1404 s build in full under that reading.

IT IS WORSE THAN A NO-OP. Each failed acquisition costs a retry timeout of
roughly 0.4 to 0.6 s and `ccache -s` walks all 256 stats buckets; against a
`/workspace` cache `ccache -s -v` did not return at all within a probe window,
after 278 consecutive lock failures. The cache adds time to every build it
cannot accelerate.

WHAT IS REFUSED AND WHAT IS NOT. Only `CCACHE_DIR` is refused on `/workspace`,
because that is the path needing symlink(2). `CCACHE_REMOTE_STORAGE=file:...`
on `/workspace` is ACCEPTED and is the remedy: ccache's file backend stores
through open plus rename, which this mount serves, and takes no lock. A gate
that refused the string `/workspace` near the word ccache would forbid the fix,
so the two names are separated rather than pattern-matched together.

A COMMENT IS NOT A SETTING. These scripts and their spec discuss the broken
recipe at length in order to stop it coming back, so every line is stripped of
its comment before it is read. Without that, documenting the defect would be the
thing the gate refuses.

SCOPE. Every `*.sh` under `scripts/`. It is a static read: nothing is executed,
nothing is written, and no shared file is touched.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# `CCACHE_DIR=<path>`, with or without `export`, and with the `${NAME:-default}`
# form the tree actually carried. The value may be quoted.
ASSIGNMENT = re.compile(
    r"""(?:^|[;&|(]|\bexport\s+)\s*CCACHE_DIR=(?P<value>[^\s;&|)]*)""",
    re.VERBOSE,
)
# `${CCACHE_DIR:-/workspace/ccache}` puts the real value inside the expansion.
DEFAULTED = re.compile(r"\$\{CCACHE_DIR:?[-=]([^}]*)\}")

CIFS_ROOTS = ("/workspace", "/mnt/nas_share", "/usr/local/nas_share")

REMEDY = (
    "Put CCACHE_DIR on local disk and let ccache persist it through its own\n"
    "  remote storage, which is the arm that was measured:\n"
    "      export CCACHE_DIR=/root/ccache\n"
    "      export CCACHE_REMOTE_STORAGE=file:/workspace/ccache-remote\n"
    "  CCACHE_DIR is the path that needs symlink(2); the remote store is not,\n"
    "  because ccache's file backend uses open+rename and takes no lock."
)


def strip_comment(line: str) -> str:
    """Drop a `#` comment, honouring quotes just well enough for these files.

    A comment is not a setting: the scripts and the spec discuss the broken
    recipe on purpose, and a gate that refused the discussion would be refusing
    its own explanation.
    """
    out: list[str] = []
    quote = ""
    previous = ""
    for char in line:
        if quote:
            out.append(char)
            if char == quote and previous != "\\":
                quote = ""
        elif char in "'\"":
            quote = char
            out.append(char)
        elif char == "#" and (not out or out[-1].isspace()):
            break
        else:
            out.append(char)
        previous = char
    return "".join(out)


def cache_dir_values(text: str) -> list[tuple[int, str]]:
    """Every value `CCACHE_DIR` is assigned, with its 1-based line number."""
    found: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = strip_comment(raw)
        for match in ASSIGNMENT.finditer(line):
            value = match.group("value").strip().strip("\"'")
            defaulted = DEFAULTED.search(value)
            if defaulted:
                value = defaulted.group(1).strip().strip("\"'")
            if value:
                found.append((number, value))
    return found


def on_cifs(value: str) -> bool:
    return any(
        value == root or value.startswith(root + "/") for root in CIFS_ROOTS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scripts = sorted((root / "scripts").glob("*.sh"))

    # A checker that examined nothing exits 0 and proves nothing. The count is
    # this instrument's own precondition, printed so a reader can see it moved.
    if not scripts:
        print(
            f"check-lease-ccache: FAIL - no shell scripts under {root}/scripts,"
            " so this gate verified nothing",
            file=sys.stderr,
        )
        return 2

    offences: list[str] = []
    for script in scripts:
        text = script.read_text(encoding="utf-8", errors="replace")
        for number, value in cache_dir_values(text):
            if on_cifs(value):
                offences.append(
                    f"{script.relative_to(root)}:{number}: CCACHE_DIR={value}"
                )

    print(f"check-lease-ccache: examined {len(scripts)} shell scripts")
    if offences:
        print(
            "check-lease-ccache: FAIL - CCACHE_DIR is on the CIFS share, where\n"
            "  ccache cannot take its locks. symlink(2) returns EOPNOTSUPP on\n"
            "  this mount, so every store and every counter update is refused\n"
            "  and `ccache -s` reads zero hits, zero misses AND zero stores.\n"
            "  That reads as a cache nobody consulted; it is a cache that could\n"
            "  not write. rc job 93a60151 paid 1404 s for it (#2473).\n",
            file=sys.stderr,
        )
        for offence in offences:
            print(f"  {offence}", file=sys.stderr)
        print(f"\n  {REMEDY}", file=sys.stderr)
        return 1

    print("check-lease-ccache: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
