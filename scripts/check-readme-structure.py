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
# belongs in .agents/state.md and docs/BENCHMARKS.md, not in a README table cell.
MAX_CELL_CHARS = 400


def _h2_headers(text: str) -> list[str]:
    return [ln[3:].strip() for ln in text.splitlines() if ln.startswith("## ")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell) <= set("-: ") for cell in cells)


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
                        "detail to .agents/state.md / docs/BENCHMARKS.md)"
                    )
    return errors


def main() -> int:
    if not README.exists():
        print("ERROR: README.md is missing", file=sys.stderr)
        return 1
    errors = readme_errors(README.read_text(encoding="utf-8"))
    if errors:
        print("ERROR: README.md is not a valid user-facing document:",
              file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("OK: README.md has the required user-facing sections and no "
          "wall-of-prose cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
