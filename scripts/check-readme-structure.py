#!/usr/bin/env python3
"""Enforce that README.md stays a human-readable, user-facing document.

Per AGENTS.md, README.md is the LocalAI house-style user-facing document, not a
status-tracking log. This checker fails if the README loses one of the required
user-facing sections (Features / Build / usage-CLI / OpenAI server / Consuming),
grows a table cell into a "wall of prose", or contains an em-dash (house style).

The validation logic is a pure function `readme_errors(text) -> list[str]` so it
is unit-testable and mutation-testable (see
tests/scripts/test_check_readme_structure.py), mirroring check-doc-checkpoint.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
STATUS = ROOT / "docs/STATUS.md"

# Each required user-facing section is (label, matchers): the README must have an
# H2 heading whose lowercased text contains ANY of the matcher substrings.
REQUIRED_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Features", ("features",)),
    ("Supported models", ("supported models", "models")),
    ("Performance", ("performance",)),
    ("Build", ("build",)),
    ("Usage / CLI", ("cli", "running inference", "usage")),
    ("OpenAI server", ("server",)),
    ("Consuming (library / C API)", ("library", "consuming", "c api", "c-api")),
)

# A table cell longer than this is the "wall of prose" smell: forensic detail
# belongs in docs/STATUS.md and docs/BENCHMARKS.md, not in a README table cell.
MAX_CELL_CHARS = 220

# The README is a landing page, not the status ledger, and what keeps it one is
# a budget on each ENTRY: one prose paragraph, one table cell. Measured in
# characters, not lines, so the budget does not move with how the prose happens
# to be wrapped.
#
# There is deliberately NO whole-file budget. `MAX_README_CHARS = 30000` was
# removed under #498, when README.md stood 7 characters below it. Per AGENTS.md
# Records a budget on a shared file makes every addition evict someone else's
# content, and merging two such edits cleanly is worse than conflicting because
# it applies both evictions. It was the fourth budget of that shape to be
# retired, after the per-class line limits, `check-now-current.py`'s MAX_CHARS
# and STATUS_RATCHET (#364), and PageRules `max_chars` (#460). Do not
# reintroduce it under a larger value:
# `test_checker_declares_no_whole_file_budget` fails if the constant returns at
# all. See .agents/specs/readme-budget-retire.md.
MAX_PARAGRAPH_CHARS = 900

# The README must point at the status ledger, and the ledger must actually carry
# the capability table (otherwise "move it to STATUS.md" silently loses it).
STATUS_LINK = "docs/STATUS.md"
CONTRIBUTOR_LINK = "CONTRIBUTING.md"
STATUS_REQUIRED_HEADINGS = ("capability status",)


def _h2_headers(text: str) -> list[str]:
    return [ln[3:].strip() for ln in text.splitlines() if ln.startswith("## ")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell) <= set("-: ") for cell in cells)


def _prose_paragraphs(text: str) -> list[tuple[int, str]]:
    """Yield (start_line, paragraph) for prose only.

    Fenced code blocks, tables, headings, and list items are excluded: the rule
    targets the wall-of-prose narrative paragraph, not legitimate long tables or
    code samples.
    """
    paragraphs: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    in_fence = False

    def flush() -> None:
        if current:
            paragraphs.append((start, " ".join(current)))
            current.clear()

    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            flush()
            continue
        if in_fence:
            continue
        is_prose = bool(stripped) and not (
            stripped.startswith("|")
            or stripped.startswith("#")
            or stripped.startswith("-")
            or stripped.startswith("*")
            or stripped.startswith(">")
        )
        if is_prose:
            if not current:
                start = lineno
            current.append(stripped)
        else:
            flush()
    flush()
    return paragraphs


def status_errors(text: str) -> list[str]:
    """Return problems with docs/STATUS.md, the per-capability status ledger."""
    errors: list[str] = []
    headers_lower = [h.lower() for h in _h2_headers(text)]
    for needle in STATUS_REQUIRED_HEADINGS:
        if not any(needle in h for h in headers_lower):
            errors.append(
                f"docs/STATUS.md is missing the '{needle}' section (it is the "
                "surface AGENTS.md points the per-capability obligation at)"
            )
    return errors


def readme_errors(text: str) -> list[str]:
    """Return a list of human-readable problems with the README text."""
    errors: list[str] = []

    headers_lower = [h.lower() for h in _h2_headers(text)]
    for label, matchers in REQUIRED_SECTIONS:
        if not any(any(m in h for m in matchers) for h in headers_lower):
            errors.append(f"missing required user-facing section: {label}")

    if "—" in text:  # em-dash
        count = text.count("—")
        errors.append(
            f"README contains {count} em-dash(es); house style forbids them "
            "(use commas, periods, parentheses, or hyphens)"
        )

    if STATUS_LINK not in text:
        errors.append(
            f"README does not link to {STATUS_LINK}; the landing page must "
            "point at the per-capability status ledger"
        )

    if CONTRIBUTOR_LINK not in text:
        errors.append(
            f"README does not link to {CONTRIBUTOR_LINK}; contributors need a "
            "public entry point to the agent protocol"
        )

    for lineno, para in _prose_paragraphs(text):
        if len(para) > MAX_PARAGRAPH_CHARS:
            errors.append(
                f"line {lineno}: prose paragraph of {len(para)} chars exceeds "
                f"{MAX_PARAGRAPH_CHARS} (wall-of-prose smell; move the detail "
                "to docs/STATUS.md and link to it)"
            )

    in_fence = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if _is_separator_row(cells):
                continue
            for cell in cells:
                if len(cell) > MAX_CELL_CHARS:
                    errors.append(
                        f"line {lineno}: table cell of {len(cell)} chars exceeds "
                        f"{MAX_CELL_CHARS} (wall-of-prose smell; move forensic "
                        "detail to docs/STATUS.md / docs/BENCHMARKS.md)"
                    )
    return errors


def main() -> int:
    if not README.exists():
        print("ERROR: README.md is missing", file=sys.stderr)
        return 1
    if not STATUS.exists():
        print("ERROR: docs/STATUS.md is missing (it is the per-capability "
              "status ledger AGENTS.md requires)", file=sys.stderr)
        return 1
    errors = readme_errors(README.read_text(encoding="utf-8"))
    errors += status_errors(STATUS.read_text(encoding="utf-8"))
    if errors:
        print("ERROR: the user-facing docs are not valid:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("OK: README.md is a valid landing page and docs/STATUS.md carries "
          "the capability ledger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
