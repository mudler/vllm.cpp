#!/usr/bin/env python3
"""Keep .agents/NOW.md a short, current, one-Read resume surface.

NOW.md is the single small file a cold session reads first to become productive.
It is a SNAPSHOT, never a log: it is rewritten in place, and the history it used
to summarise now lives where history belongs, in git.

This checker owns exactly one obligation -- structure and budget, so NOW.md
cannot decay into another status log. The other half, "NOW.md must be refreshed
when the live position moves", is owned by check-doc-checkpoint.py, which
already requires NOW.md on a lifecycle change. Splitting one obligation across
two checkers is how it ends up enforced twice and satisfiable by neither.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

NOW = ROOT / ".agents/NOW.md"
NOW_PATH = ".agents/NOW.md"

# Budgets. NOW.md exists to be read in full, every session, by every agent. The
# moment it stops fitting in one screenful of attention it has become the thing
# it was meant to replace.
#
# MAX_CHARS WAS REMOVED 2026-08-11 (ENG-RECORD-CONFLICT-SURFACES, #364). It was
# 6000, and the tracked file measured EXACTLY 6000: tuned to the byte, with no
# headroom at all. That made adding a row require EVICTING one, so every PR
# performed a read-modify-write of a single shared global, and NOW.md conflicted
# in 5 of the 16 conflicting open PRs measured at origin/main d928e2c3.
#
# The conflict was the LUCKY outcome. Concurrent read-modify-write loses
# updates: a clean three-way merge of two such PRs applies BOTH evictions and
# BOTH additions, silently dropping two live rows and blowing the very budget
# this constant existed to defend. A gate whose SUCCESS mode is unsafe is worse
# than no gate.
#
# MAX_LINES and MAX_ENTRY_CHARS are KEPT and carry the obligation between them.
# A line cap bounds the page just as a byte cap does, but a row costs ONE line
# rather than a variable number of bytes, so an ordinary edit no longer forces
# an unrelated deletion; and MAX_ENTRY_CHARS bounds each entry LOCALLY, which is
# what actually stops a digest decaying into the status log it replaced.
MAX_LINES = 100
MAX_ENTRY_CHARS = 400

REQUIRED_HEADINGS = (
    "live claims",
    "current gate",
    "next actions",
)

ROW_TABLE_LINE = re.compile(r"^\|\s*`[A-Z0-9][A-Za-z0-9_.-]*`\s*\|")

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
            "detail to the row's spec and keep only the live position here"
        )
    # REGROWTH GUARD (ENG-NOW-DERIVED, #374). The per-row claims table left this
    # file because requiring it made NOW.md a surface every row-advancing PR had
    # to write -- a lock under AGENTS.md §Records, and 5 of the 16 conflicting
    # open PRs at d928e2c3. Removing it once is not enough: the decay path is
    # someone re-adding "just one row", and then the file is a status log again
    # and every PR is back in it. So the SHAPE is enforced, not just the state.
    #
    # A row here is a line whose first cell is a backticked stable ID. Ordinary
    # tables (the gate, the invariants) have prose first cells and still pass.
    for lineno, line in enumerate(lines, 1):
        if ROW_TABLE_LINE.match(line.strip()):
            errors.append(
                f"line {lineno}: a per-row table row is back in NOW.md "
                f"({line.strip()[:48]!r}...). The live position is DERIVED -- run "
                "scripts/now.py -- and a row's next step belongs in that row's "
                "own spec under `## Now`, which has one writer. Putting rows here "
                "again makes this file a surface every PR must write"
            )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("-", "|")) and len(stripped) > MAX_ENTRY_CHARS:
            errors.append(
                f"an entry is {len(stripped)} characters, over the "
                f"{MAX_ENTRY_CHARS}-character budget: {stripped[:60]!r}...; "
                "link the row's spec instead of inlining the narrative"
            )

    return errors


def main(argv: list[str]) -> int:
    # --base/--head/--commit/--staged are accepted and ignored: CI passes a
    # range, and this check is range-independent now that freshness coupling
    # belongs to check-doc-checkpoint.py. Silently accepting them keeps the CI
    # invocation stable.
    del argv

    if not NOW.exists():
        print(f"ERROR: {NOW_PATH} does not exist", file=sys.stderr)
        return 1

    failures = [
        f"{NOW_PATH} {error}"
        for error in structure_errors(NOW.read_text(encoding="utf-8"))
    ]

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "NOW.md is the one-Read resume surface: the live claims, the gate "
            "being chased, and the next actions, rewritten in place. Detail "
            "belongs in the row's spec and the area matrices; history belongs "
            "in git.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {NOW_PATH} is a current, in-budget resume digest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
