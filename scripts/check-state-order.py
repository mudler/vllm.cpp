#!/usr/bin/env python3
"""Keep the tail of .agents/state.md genuinely chronological.

`.agents/state.md` declares "append; newest last", and AGENTS.md sends every
cold session to the newest entries for its resume context. That contract was
silently broken: union-merging state appends from parallel worktrees interleaves
them, so the tail carried entries dated 07-27 AFTER entries dated 07-30. An
agent reading the tail therefore got a jumble that both included stale entries
and omitted recent ones -- the cold-resume primitive the whole protocol rests on
was unsound.

Heading dates cannot be parsed reliably (some entries put the date in a trailing
parenthetical, some lead with it), so entries below the enforcement marker carry
an explicit sortable anchor:

    ## Some entry title
    <!-- state: 2026-08-04T16:30 -->

This gate gets applied only BELOW the marker, so the existing history stays
untouched and every new entry is enforced. When a merge interleaves entries, the
repair is mechanical: scripts/sort-state-tail.py --apply.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".agents/state.md"

# Everything after this marker is enforced; everything before it is frozen
# history that predates the contract.
MARKER = "<!-- state-order:enforced-below -->"

ANCHOR = re.compile(r"^<!--\s*state:\s*(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2}))?\s*-->$")


def split_at_marker(text: str) -> list[str] | None:
    """Return the enforced lines after the marker, or None when it is absent."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == MARKER:
            return lines[index + 1 :]
    return None


def entry_errors(text: str) -> list[str]:
    """Return ordering/anchor problems in the enforced tail of a state log."""
    enforced = split_at_marker(text)
    if enforced is None:
        return [
            f"the enforcement marker {MARKER} is missing; append it once at the "
            "end of the frozen history so every later entry is order-checked"
        ]

    errors: list[str] = []
    anchors: list[tuple[tuple[str, str], int, str]] = []
    seen_heading = False
    pending_heading: tuple[int, str] | None = None

    for offset, line in enumerate(enforced):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("## "):
            if pending_heading is not None:
                errors.append(
                    f"entry {pending_heading[1]!r} has no "
                    "<!-- state: YYYY-MM-DD --> anchor on the line after its "
                    "heading"
                )
            pending_heading = (offset, stripped[3:].strip())
            seen_heading = True
            continue

        match = ANCHOR.match(stripped)
        if match:
            if pending_heading is None:
                errors.append(
                    f"anchor {stripped!r} is not attached to an entry heading; "
                    "each anchor belongs on the line after its '## ' heading"
                )
                continue
            key = (match.group(1), match.group(2) or "00:00")
            anchors.append((key, pending_heading[0], pending_heading[1]))
            pending_heading = None
            continue

        if not seen_heading:
            errors.append(
                "content appears below the marker before any '## ' entry "
                f"heading: {stripped[:60]!r}"
            )
            seen_heading = True

    if pending_heading is not None:
        errors.append(
            f"entry {pending_heading[1]!r} has no <!-- state: YYYY-MM-DD --> "
            "anchor on the line after its heading"
        )

    for (key, _, title), (previous_key, _, previous_title) in zip(
        anchors[1:], anchors
    ):
        if key < previous_key:
            errors.append(
                f"entry {title!r} is anchored {key[0]}T{key[1]} but follows "
                f"{previous_title!r} anchored {previous_key[0]}T{previous_key[1]}; "
                "the log is newest-last. Repair with "
                "`python3 scripts/sort-state-tail.py --apply`"
            )

    return errors


def main() -> int:
    if not STATE.exists():
        print(f"ERROR: {STATE} does not exist", file=sys.stderr)
        return 1

    failures = entry_errors(STATE.read_text(encoding="utf-8"))
    if failures:
        for failure in failures:
            print(f"ERROR: .agents/state.md: {failure}", file=sys.stderr)
        print(
            "Every state entry appended below the marker carries a sortable "
            "<!-- state: YYYY-MM-DD --> (or ...THH:MM) anchor on the line after "
            "its heading, and the anchors run oldest to newest.",
            file=sys.stderr,
        )
        return 1

    print("OK: the enforced tail of .agents/state.md is anchored and chronological.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
