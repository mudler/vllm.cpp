#!/usr/bin/env python3
"""Restore chronological order to the enforced tail of .agents/state.md.

A union merge of state appends from two worktrees interleaves entries, which is
what broke the "newest last" contract in the first place. Resolving that by hand
invites dropped entries, so the repair is mechanical: stable-sort the anchored
entries below the marker by their <!-- state: ... --> anchor and rewrite the
file. Content above the marker (the frozen history) is never touched, and no
entry is ever dropped -- this only reorders whole entries.

    python3 scripts/sort-state-tail.py            # report what would move
    python3 scripts/sort-state-tail.py --apply    # rewrite the file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".agents/state.md"

sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


order = _load("state_order", "scripts/check-state-order.py")


class Entry:
    """One state-log entry: its heading, its anchor key, and its whole body."""

    def __init__(self, key: tuple[str, str], title: str, lines: list[str]) -> None:
        self.key = key
        self.title = title
        self.lines = lines


def parse_entries(enforced: list[str]) -> tuple[list[Entry], list[str]]:
    """Split the enforced tail into entries; returns (entries, problems)."""
    entries: list[Entry] = []
    problems: list[str] = []
    current: Entry | None = None

    for line in enforced:
        stripped = line.strip()
        if stripped.startswith("## "):
            if current is not None:
                entries.append(current)
            current = Entry(("", ""), stripped[3:].strip(), [line])
            continue

        if current is None:
            if stripped:
                problems.append(
                    f"content before the first entry heading: {stripped[:60]!r}"
                )
            continue

        current.lines.append(line)
        match = order.ANCHOR.match(stripped)
        if match and current.key == ("", ""):
            current.key = (match.group(1), match.group(2) or "00:00")

    if current is not None:
        entries.append(current)

    for entry in entries:
        if entry.key == ("", ""):
            problems.append(f"entry {entry.title!r} has no anchor; cannot sort it")

    return entries, problems


def rstrip_trailing_blanks(lines: list[str]) -> list[str]:
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="rewrite the file instead of reporting"
    )
    args = parser.parse_args()

    text = STATE.read_text(encoding="utf-8")
    enforced = order.split_at_marker(text)
    if enforced is None:
        print(f"ERROR: {order.MARKER} is missing from .agents/state.md", file=sys.stderr)
        return 1

    entries, problems = parse_entries(enforced)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        print(
            "Fix the anchors first; sorting an unanchored entry would place it "
            "arbitrarily.",
            file=sys.stderr,
        )
        return 1

    ordered = sorted(entries, key=lambda entry: entry.key)
    moved = [
        entry.title
        for entry, sorted_entry in zip(entries, ordered)
        if entry.title != sorted_entry.title
    ]

    if not moved:
        print("OK: the enforced tail of .agents/state.md is already chronological.")
        return 0

    if not args.apply:
        print(f"{len(moved)} entries are out of order; re-run with --apply:")
        for title in moved[:10]:
            print(f"  - {title[:100]}")
        if len(moved) > 10:
            print(f"  ... (+{len(moved) - 10} more)")
        return 1

    head = text.split(order.MARKER)[0]
    body: list[str] = []
    for entry in ordered:
        body.extend(rstrip_trailing_blanks(list(entry.lines)))
        body.append("")

    STATE.write_text(
        f"{head}{order.MARKER}\n\n" + "\n".join(body).lstrip("\n") + "\n",
        encoding="utf-8",
    )
    print(f"Reordered {len(moved)} entries in .agents/state.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
