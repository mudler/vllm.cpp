#!/usr/bin/env python3
"""Roll accumulated sections out of docs/BENCHMARKS.md into the record.

docs/BENCHMARKS.md is a keyed-table scoreboard: a checkpoint updates its ROW.
When a checkpoint appends an H2 narrative section instead,
scripts/check-benchmarks-structure.py fails and points here. This script does
the move mechanically, so what gets relocated is not a judgement call: any H2
section whose title is not on the CANONICAL_SECTIONS allowlist moves verbatim
into .agents/benchmark-record.md, oldest content first, nothing edited or
dropped.

The allowlist is CANONICAL_SECTIONS in scripts/check-public-doc-tables.py, read
from there so this script and the CI gate can never disagree. Adding a genuinely
new comparison subject (a new reference engine, a new hardware class) means
adding one line to it. That friction is deliberate: it is what separates "we now
benchmark against X" from "here is a paragraph about last night's run".

    scripts/roll-benchmark-record.py            # dry run, lists what would move
    scripts/roll-benchmark-record.py --apply    # perform the move
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "docs/BENCHMARKS.md"
RECORD = ROOT / ".agents/benchmark-record.md"


def _load_checker():
    """Load the CI checker so the allowlist has exactly ONE definition.

    What this script moves and what CI rejects must never drift apart, so the
    canonical-section list lives in check-public-doc-tables.py and is read from
    there rather than copied.
    """
    path = ROOT / "scripts/check-public-doc-tables.py"
    spec = importlib.util.spec_from_file_location("public_doc_tables", path)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RULES = _load_checker().BENCHMARKS_RULES


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split into (preamble, [(title, body_including_heading), ...]) on H2s.

    Fenced code blocks are respected so a "## " inside a fence is not a
    heading.
    """
    lines = text.splitlines(keepends=True)
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            sections.append((line[3:].strip(), [line]))
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble.append(line)
    return "".join(preamble), [(t, "".join(b)) for t, b in sections]


def is_canonical(title: str) -> bool:
    return RULES.is_canonical(title)


def plan(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (kept, rollable) sections for the given scoreboard text."""
    _, sections = split_sections(text)
    kept = [s for s in sections if is_canonical(s[0])]
    rollable = [s for s in sections if not is_canonical(s[0])]
    return kept, rollable


def rebuild(preamble: str, kept: list[tuple[str, str]]) -> str:
    # Removing a section leaves behind the blank line that separated it, which
    # would accumulate one line per roll. Normalize to a single trailing
    # newline so repeated rolls are idempotent.
    return (preamble + "".join(body for _, body in kept)).rstrip("\n") + "\n"


def roll(today: str) -> tuple[int, str]:
    """Perform the move. Returns (count_moved, human-readable summary)."""
    text = BENCHMARKS.read_text(encoding="utf-8")
    preamble, _ = split_sections(text)
    kept, rollable = plan(text)
    if not rollable:
        return 0, "nothing to roll: every section is canonical"

    moved = "".join(body for _, body in rollable)
    banner = (
        f"\n## Rolled out of the scoreboard on {today}\n\n"
        "Moved verbatim from `docs/BENCHMARKS.md` by "
        "`scripts/roll-benchmark-record.py`. Nothing edited or deleted.\n\n"
    )
    with RECORD.open("a", encoding="utf-8") as handle:
        handle.write(banner + moved)
    BENCHMARKS.write_text(rebuild(preamble, kept), encoding="utf-8")
    titles = "\n".join(f"  - {title}" for title, _ in rollable)
    return len(rollable), f"moved {len(rollable)} section(s):\n{titles}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the move (default is a dry run)",
    )
    args = parser.parse_args()

    if not BENCHMARKS.exists():
        print("ERROR: docs/BENCHMARKS.md is missing", file=sys.stderr)
        return 1
    if not RECORD.exists():
        print(
            "ERROR: .agents/benchmark-record.md is missing; refusing to roll "
            "sections into a file that does not exist",
            file=sys.stderr,
        )
        return 1

    _, rollable = plan(BENCHMARKS.read_text(encoding="utf-8"))
    if not rollable:
        print("OK: every docs/BENCHMARKS.md section is canonical, nothing to roll.")
        return 0

    if not args.apply:
        print("These sections are not canonical and would move to the record:")
        for title, _ in rollable:
            print(f"  - {title}")
        print("\nRe-run with --apply to move them.")
        return 1

    today = datetime.date.today().isoformat()
    count, summary = roll(today)
    print(f"OK: {summary}")
    print(f"Destination: {RECORD.relative_to(ROOT)}")
    return 0 if count else 0


if __name__ == "__main__":
    raise SystemExit(main())
