#!/usr/bin/env python3
"""Refuse a tracked file that carries a literal merge conflict marker (#1417).

Splice a conflict into `docs/STATUS.md` and every record gate passes.
`check-public-doc-tables.py` measures cell and row budgets, `check-agent-record.py`
measures counts and anchors, and neither asks whether the file is well-formed in
the first place, so a table row that is half one branch and half another
satisfies every budget it is measured against. Measured on a scratch commit at
`b537a5344`: all four record gates returned 0 with their normal OK messages.

That is not hypothetical. An earlier revision of the branch for #1414 carried a
`docs/STATUS.md` mangled by stale working-tree state, the full record gate set
ran, and it reported clean.

THE RULE, and why each half of it is shaped the way it is.

  * A line that starts with seven `<` and a space opens a hunk. A line that
    starts with seven `>` and a space closes one. Either one fails, anywhere.
  * A line that is EXACTLY seven `=` and nothing else fails ONLY inside an open
    hunk.

The separator is conditional because a bare row of `=` is legal markdown: it is
the setext heading underline and it is a horizontal rule. A `^=+` rule would
have fired on five shipped files on arrival -- three logs under
`docs/bench-evidence/gdn-replayssm-w0-20260818/` and two tokenizer corpora under
`tests/parity/goldens/` -- every one of which carries a rule longer than seven.
A gate that fires on ordinary work is the defect, not the discipline. So the
separator adds no detection power over the two markers, deliberately: its job is
to name the middle of the hunk in the report, so the reader sees the whole
conflict rather than its first line.

The diff3 `|||||||` marker is absent on purpose. A diff3 conflict still carries
the start and end markers, so nothing escapes, and a line of seven `|` is a
plausible empty row in a repository whose records are wide markdown tables.

NO ALLOWLIST, AND NO SELF-EXCLUSION. Every pattern below is built by character
repetition rather than written as a literal, so this file and its suite carry no
marker at the start of any line and need no exemption from the gate they
implement. An allowlist would be a file that every change must append to, and
AGENTS.md forbids that surface: if N concurrent pull requests edit file F, F is
a lock. A document that quotes a marker on purpose indents it, which is what the
failure message says.

    scripts/check-conflict-markers.py                # the tree this script is in
    scripts/check-conflict-markers.py --root PATH    # an explicit tree

Exit 0 clean, 1 with findings, 2 when the run could not examine anything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Built, never written: see the docstring. `START`/`END` carry the trailing
# space that `git` writes before the branch label; `SEPARATOR` is the whole line.
# Bytes, not text: the scan reads 142 MB of tracked text, and decoding all of it
# to find a marker in none of it cost more than reading it.
START = b"<" * 7 + b" "
SEPARATOR = b"=" * 7
END = b">" * 7 + b" "

# git's own binary heuristic: a NUL byte in the first block. Reading only this
# much of a binary keeps the scan off the 175 MB of tracked tensor fixtures.
SNIFF_BYTES = 8192


def tracked_paths(root: Path) -> list[str]:
    """Every path in the index, once each, in `git`'s order.

    The index rather than a commit: the incident this gate closes happened in a
    working tree, and a fresh clone's index matches its checkout anyway.

    DEDUPLICATED, and that is not a tidiness measure. An index holding an
    unresolved merge carries stages 1, 2 and 3 for every conflicted path, so
    `git ls-files` names such a path three times. Without this, the file is read
    three times, its findings print three times, and `examined` over-counts by
    two per conflicted path -- measured as `9 findings in 1 file; examined 3
    tracked text files` for a single real conflict. The verdict was still 1, so
    this never caused a miss, but the count was wrong in exactly the state this
    gate exists for, and a gate that cannot say how many things it examined has
    not reported.
    """
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"git ls-files exited {result.returncode} in {root}: {message}\n"
            "Unknown is not absence: this run examined nothing."
        )
    names = [
        name
        for name in result.stdout.decode("utf-8", "surrogateescape").split("\0")
        if name
    ]
    # `dict.fromkeys` rather than `set`, so git's order survives the dedupe and
    # the report stays reproducible run to run.
    return list(dict.fromkeys(names))


def scan_text(data: bytes) -> list[tuple[int, str, str]]:
    """Findings in one file, as `(line number, shape, line)`.

    A cheap whole-buffer reject first, and it is a PURE OPTIMIZATION: it decides
    nothing the per-line pass would decide differently. A marker line contains
    the marker as a substring, so a buffer holding neither substring anywhere
    cannot hold one at the start of a line. The separator needs no test of its
    own here, because it is only ever a finding inside a hunk and a hunk needs a
    start marker in the same buffer. Almost every file takes this arm, which is
    what holds the whole-tree scan at 0.29 s of CPU.

    Deleting this reject therefore leaves the suite green, and that green is the
    expected result rather than a gap. The SEMANTIC guard in this function is
    `open_at is not None`, and dropping it reds
    `test_a_separator_after_the_hunk_closes_is_not_named`, which exists because
    the first mutation run found nothing else could tell.
    """
    if START not in data and END not in data:
        return []
    findings: list[tuple[int, str, str]] = []
    open_at: int | None = None
    for number, raw in enumerate(data.split(b"\n"), start=1):
        # A conflict written on Windows arrives with a carriage return, and a
        # separator compared without stripping it would never match.
        line = raw.rstrip(b"\r")
        if line.startswith(START):
            shape = "conflict start marker"
            open_at = number
        elif line.startswith(END):
            shape = "conflict end marker"
            open_at = None
        elif line == SEPARATOR and open_at is not None:
            shape = f"conflict separator inside the hunk opened at line {open_at}"
        else:
            continue
        findings.append((number, shape, line.decode("utf-8", "replace")))
    return findings


def scan_tree(root: Path) -> tuple[list[str], set[str], list[str], dict[str, int]]:
    """Findings over every tracked text file, with the counts that were skipped.

    Returns the finding lines, the SET of paths they name, the unreadable-file
    lines, and the counts. The offender set is collected here rather than parsed
    back out of the report: splitting a report line on its first colon
    mis-attributes any path that contains one.
    """
    counts = {"tracked": 0, "examined": 0, "binary": 0, "symlink": 0, "absent": 0}
    report: list[str] = []
    offenders: set[str] = set()
    unreadable: list[str] = []
    for name in tracked_paths(root):
        counts["tracked"] += 1
        path = root / name
        # Checked before `is_file()`, which follows a symlink and would file a
        # broken link under the wrong count.
        if path.is_symlink():
            counts["symlink"] += 1
            continue
        if not path.is_file():
            counts["absent"] += 1
            continue
        try:
            with path.open("rb") as handle:
                head = handle.read(SNIFF_BYTES)
                if b"\0" in head:
                    counts["binary"] += 1
                    continue
                data = head + handle.read()
        except OSError as error:
            # NOT a finding, and not a skip either. A file this run could not
            # read is a file this run cannot call clean, so it takes its own
            # list, its own exit status and its own remedy. Filing it under
            # `report` made an I/O error exit 1 and print "Resolve the merge
            # before committing", which is the wrong remedy for the wrong
            # problem.
            unreadable.append(f"{name}: unreadable: {error}")
            continue
        counts["examined"] += 1
        for number, shape, line in scan_text(data):
            report.append(f"{name}:{number}: {shape}: {line[:80]}")
            offenders.add(name)
    return report, offenders, unreadable, counts


def summary(findings: int, files: int, counts: dict[str, int]) -> str:
    plural = "file" if files == 1 else "files"
    return (
        f"conflict markers: {findings} findings in {files} {plural}; "
        f"examined {counts['examined']} tracked text files "
        f"({counts['binary']} binary, {counts['symlink']} symlink, "
        f"{counts['absent']} absent skipped, {counts['tracked']} tracked paths)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "the tree to examine. Defaults to the repository holding this "
            "script, which from a linked worktree is that worktree and from the "
            "shared checkout is the shared checkout."
        ),
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]

    # Printed first and always. A checker resolves its root from its own path,
    # so a run whose root is not the tree the reader means can print OK about
    # somebody else's checkout.
    print(f"root: {root}")
    try:
        report, offenders, unreadable, counts = scan_tree(root)
    except RuntimeError as error:
        print(str(error))
        print(summary(0, 0, {"tracked": 0, "examined": 0, "binary": 0, "symlink": 0, "absent": 0}))
        return 2

    for line in report:
        print(line)
    for line in unreadable:
        print(line)
    print(summary(len(report), len(offenders), counts))

    if counts["examined"] == 0:
        print(
            "A gate that examined nothing has not reported. Check --root, and "
            "check that it names a git repository with tracked text files."
        )
        return 2
    if unreadable:
        # Ordered BEFORE the findings arm on purpose. A run that could not read
        # part of the tree cannot say the tree is clean, and it must not say so
        # by reporting only the part it managed to read.
        print(
            f"{len(unreadable)} tracked file(s) could not be read, so this run "
            "cannot report on them. Unknown is not absence. Repair the "
            "permissions or the checkout and rerun."
        )
        return 2
    if report:
        print(
            "Resolve the merge before committing. A document that quotes a "
            "conflict marker on purpose must indent it, so that it does not "
            "start at column 0."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
