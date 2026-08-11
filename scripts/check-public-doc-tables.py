#!/usr/bin/env python3
"""Enforce that the public keyed-table docs stay human-readable.

Per AGENTS.md, docs/BENCHMARKS.md and docs/FEATURES.md are KEYED TABLES in the
same class as .agents/roadmap_v1.md and the area matrices: one row per subject,
updated in place. They are NOT append logs. Left unchecked they drift back into
one a checkpoint at a time (BENCHMARKS.md reached 11,405 lines and 171 sections
before the 2026-08-04 conversion), which is exactly what makes them unreadable
to users.

This checker fails if either page loses a required user-facing section, grows
past its budget, accumulates sections or prose instead of rows, or stops
pointing at the record it relocates detail into. When BENCHMARKS.md fails
because sections accumulated, `scripts/roll-benchmark-record.py` moves them into
.agents/benchmark-record.md.

The validation logic is the pure functions `benchmarks_errors(text)` and
`features_errors(text)` so they are unit-testable and mutation-testable (see
tests/scripts/test_check_public_doc_tables.py), mirroring
check-readme-structure.py and check-doc-checkpoint.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "docs/BENCHMARKS.md"
FEATURES = ROOT / "docs/FEATURES.md"
RECORD = ROOT / ".agents/benchmark-record.md"

# A table cell longer than this is the wall-of-prose smell: forensic detail
# belongs in the record, not in a keyed-table cell. Shared by both pages.
MAX_CELL_CHARS = 220

# Likewise shared: a single paragraph past this is narrative, not a caption.
MAX_PARAGRAPH_CHARS = 700

# The scoreboard must point at the record, and the record must exist, otherwise
# "move it to the archive" silently loses the evidence.
RECORD_LINK = ".agents/benchmark-record.md"


class PageRules:
    """The budget for one public keyed-table page.

    Every limit is set from what the page measured when it was written, with
    headroom for genuinely new subjects but not for accumulated entries. That
    is the whole mechanism: growth that is new ROWS passes, growth that is new
    SECTIONS or new PROSE fails.
    """

    def __init__(
        self,
        *,
        name: str,
        kind: str,
        required_sections: tuple[tuple[str, tuple[str, ...]], ...],
        max_h2_sections: int,
        max_prose_paragraphs: int,
        max_chars: int,
        min_table_rows: int,
        required_links: tuple[str, ...],
        canonical_sections: tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.required_sections = required_sections
        self.max_h2_sections = max_h2_sections
        self.max_prose_paragraphs = max_prose_paragraphs
        self.max_chars = max_chars
        self.min_table_rows = min_table_rows
        self.required_links = required_links
        # When set, EVERY H2 must match one of these substrings. This is the
        # sharp form of the append-log rule: the FIRST appended section fails,
        # rather than the seventh once a count budget fills up. Left None for a
        # page where new sections are a normal, legitimate way to grow.
        self.canonical_sections = canonical_sections

    def is_canonical(self, title: str) -> bool:
        if self.canonical_sections is None:
            return True
        lowered = title.lower()
        return any(matcher in lowered for matcher in self.canonical_sections)


# docs/BENCHMARKS.md: 10 sections, 24 prose paragraphs, 12,078 chars, 68 rows
# when it was converted from an 11,405-line append log on 2026-08-04. Required
# sections are the four things a reader arriving from the README badge needs:
# the summary, the method behind the numbers, what is NOT measured, and how to
# re-run it.
BENCHMARKS_RULES = PageRules(
    name="docs/BENCHMARKS.md",
    kind="scoreboard",
    required_sections=(
        ("At a glance", ("at a glance", "summary", "scoreboard")),
        ("How we measure", ("how we measure", "methodology", "method")),
        ("Open gaps", ("open gaps", "not measured", "pending")),
        ("Reproduce", ("reproduce", "reproduction")),
    ),
    max_h2_sections=16,
    max_prose_paragraphs=35,
    max_chars=45000,
    min_table_rows=40,
    required_links=(RECORD_LINK,),
    # The scoreboard's sections ARE its schema: a reference engine, a resource
    # axis, or one of the four required reader sections. Anything else is a
    # per-attempt entry. Adding a genuinely new comparison subject means adding
    # one line here, which is a deliberate act; appending an entry is not.
    canonical_sections=(
        "at a glance",
        "vllm",
        "llama.cpp",
        "mlx",
        "dwarfstar",
        "memory",
        "speculative decoding",
        "how we measure",
        "open gaps",
        "reproduce",
    ),
)

# docs/FEATURES.md: 13 sections, 10 prose paragraphs, 8,275 chars, 104 rows when
# it was written on 2026-08-04. It must keep pointing at the two surfaces that
# carry the detail it deliberately does not: the lifecycle caveats behind each
# mark, and the measured speed behind each supported row.
FEATURES_RULES = PageRules(
    name="docs/FEATURES.md",
    kind="feature matrix",
    required_sections=(
        ("At a glance", ("at a glance", "summary", "overview")),
        ("Not supported yet", ("not supported", "gaps", "limitations")),
        ("How to read this page", ("how to read", "legend", "reading")),
    ),
    max_h2_sections=20,
    max_prose_paragraphs=20,
    max_chars=30000,
    min_table_rows=60,
    required_links=("STATUS.md", "BENCHMARKS.md"),
)


def _h2_headers(text: str) -> list[str]:
    return [ln[3:].strip() for ln in text.splitlines() if ln.startswith("## ")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell) <= set("-: ") for cell in cells)


def _prose_paragraphs(text: str) -> list[tuple[int, str]]:
    """Yield (start_line, paragraph) for prose only.

    Fenced code blocks, tables, headings, and list items are excluded: the rule
    targets the narrative paragraph, not legitimate tables or code samples.
    """
    paragraphs: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    in_fence = False

    def flush() -> None:
        nonlocal current
        if current:
            paragraphs.append((start, " ".join(current)))
            current = []

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


def _table_rows(text: str) -> list[tuple[int, list[str]]]:
    """Yield (line_number, cells) for every non-separator table row."""
    rows: list[tuple[int, list[str]]] = []
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
            if not _is_separator_row(cells):
                rows.append((lineno, cells))
    return rows


def page_errors(text: str, rules: PageRules) -> list[str]:
    """Return human-readable problems with one keyed-table page."""
    errors: list[str] = []

    headers = _h2_headers(text)
    headers_lower = [h.lower() for h in headers]
    for label, matchers in rules.required_sections:
        if not any(any(m in h for m in matchers) for h in headers_lower):
            errors.append(f"missing required user-facing section: {label}")

    non_canonical = [h for h in headers if not rules.is_canonical(h)]
    if non_canonical:
        listed = "; ".join(non_canonical[:3])
        if len(non_canonical) > 3:
            listed += f"; ... (+{len(non_canonical) - 3})"
        errors.append(
            f"{rules.name} has {len(non_canonical)} non-canonical H2 "
            f"section(s) ({listed}); it is a KEYED TABLE, so a checkpoint "
            "updates its ROW, it does not append a section. Run "
            "scripts/roll-benchmark-record.py --apply to move them verbatim "
            f"into {RECORD_LINK}, or add a genuinely new comparison subject to "
            "CANONICAL_SECTIONS deliberately"
        )

    if len(headers) > rules.max_h2_sections:
        errors.append(
            f"{rules.name} has {len(headers)} H2 sections, over the "
            f"{rules.max_h2_sections}-section budget; it is a KEYED TABLE, so "
            "a checkpoint updates its ROW, it does not append a section"
        )

    if "—" in text:  # em-dash
        count = text.count("—")
        errors.append(
            f"{rules.name} contains {count} em-dash(es); house style forbids "
            "them (use commas, periods, parentheses, or hyphens)"
        )

    if len(text) > rules.max_chars:
        errors.append(
            f"{rules.name} is {len(text)} chars, over the {rules.max_chars}-char "
            f"{rules.kind} budget; per-attempt detail belongs in the record and "
            "the lifecycle surfaces, not here"
        )

    for link in rules.required_links:
        if link not in text:
            errors.append(
                f"{rules.name} does not link to {link}; the {rules.kind} must "
                "point at the surface carrying the detail it omits, so nothing "
                "it drops becomes unreachable"
            )

    paragraphs = _prose_paragraphs(text)
    if len(paragraphs) > rules.max_prose_paragraphs:
        errors.append(
            f"{rules.name} has {len(paragraphs)} prose paragraphs, over the "
            f"{rules.max_prose_paragraphs} budget; content belongs in table "
            "ROWS and prose only explains them"
        )
    for lineno, para in paragraphs:
        if len(para) > MAX_PARAGRAPH_CHARS:
            errors.append(
                f"line {lineno}: prose paragraph of {len(para)} chars exceeds "
                f"{MAX_PARAGRAPH_CHARS} (wall-of-prose smell; put the result "
                f"in a table row and the detail in {RECORD_LINK})"
            )

    rows = _table_rows(text)
    if len(rows) < rules.min_table_rows:
        errors.append(
            f"{rules.name} has {len(rows)} table rows, under the "
            f"{rules.min_table_rows} minimum; the {rules.kind} carries its "
            "content as keyed table rows, not as prose"
        )
    for lineno, cells in rows:
        for cell in cells:
            if len(cell) > MAX_CELL_CHARS:
                errors.append(
                    f"line {lineno}: table cell of {len(cell)} chars exceeds "
                    f"{MAX_CELL_CHARS} (wall-of-prose smell; move the detail "
                    f"to {RECORD_LINK})"
                )
    return errors


def benchmarks_errors(text: str) -> list[str]:
    """Return problems with docs/BENCHMARKS.md, the measured scoreboard."""
    return page_errors(text, BENCHMARKS_RULES)


def features_errors(text: str) -> list[str]:
    """Return problems with docs/FEATURES.md, the feature comparison matrix."""
    return page_errors(text, FEATURES_RULES)


# docs/STATUS.md is the per-capability ledger. It is legitimately larger than a
# scoreboard, and it is ALSO the page BENCHMARKS.md's forensics were told to
# move to, so a hard budget today would either fail on landing or push detail
# somewhere worse. But it is currently in the exact shape BENCHMARKS.md was in
# before its conversion (291k chars, 91 paragraphs over the prose budget, 47
# oversized cells, one cell of 16,181 chars), and it is the only public surface
# with no size gate at all - which is precisely how BENCHMARKS.md reached 11,127
# lines unnoticed.
#
# So this is a RATCHET, not a budget: every limit is pinned to what the page
# measured on 2026-08-04 and may only go DOWN. New rows and tighter prose pass;
# growth fails. The same mechanism as the device-leakage DSR ratchet in CI.
# Lowering these numbers as the page is compacted is the gate closing.
#
# THE `chars` KEY WAS REMOVED 2026-08-11 (ENG-RECORD-CONFLICT-SURFACES, #364).
# It was a byte count of docs/STATUS.md stored in THIS file and permitted to move
# only downward, so a PR owing the page one lifecycle line had to delete
# unrelated prose from some other row to pay for it AND edit this file — two
# shared surfaces per PR, both of which every other PR was also editing. It made
# `scripts/check-public-doc-tables.py` a merge hotspot in 4 of the 16 conflicting
# open PRs and docs/STATUS.md one in another 4, measured at origin/main d928e2c3.
# The comment that used to sit here already recorded the failure — "a ratchet
# pinned to the byte turns every concurrently merged row's one-line status edit
# into a spurious failure" — and answered it by adding slack to the constant,
# which only postponed it to the next cadence of parallel work.
#
# The three keys below are DELIBERATELY KEPT. Each counts a QUALITY defect
# (sections, paragraphs over MAX_PARAGRAPH_CHARS, cells over MAX_CELL_CHARS)
# rather than a length, so an ordinary lifecycle line moves none of them and two
# concurrent PRs do not collide on them. They carry the whole anti-bloat
# obligation the `chars` key was claimed to serve: a page cannot decay into
# wall-of-prose without tripping one, and BENCHMARKS.md's 11,127-line decay would
# have been caught by `long_paragraphs` and `oversized_cells` alone.
STATUS = ROOT / "docs/STATUS.md"
STATUS_RATCHET = {
    "h2_sections": 11,
    "long_paragraphs": 82,
    "oversized_cells": 44,
}
STATUS_REQUIRED = (
    ("Parity pin", ("parity pin",)),
    ("Capability status", ("capability status",)),
    ("Not supported yet", ("not supported yet", "not yet supported")),
)


def status_errors(text: str) -> list[str]:
    """Return ratchet violations for docs/STATUS.md, the per-capability ledger."""
    errors: list[str] = []

    headers = _h2_headers(text)
    lowered = [h.lower() for h in headers]
    for label, matchers in STATUS_REQUIRED:
        if not any(any(m in h for m in matchers) for h in lowered):
            errors.append(
                f"docs/STATUS.md is missing its '{label}' section; it is the "
                "per-capability status surface every checkpoint updates"
            )

    measured = {
        "chars": len(text),
        "h2_sections": len(headers),
        "long_paragraphs": sum(
            1 for _, para in _prose_paragraphs(text) if len(para) > MAX_PARAGRAPH_CHARS
        ),
        "oversized_cells": sum(
            1
            for _, cells in _table_rows(text)
            for cell in cells
            if len(cell) > MAX_CELL_CHARS
        ),
    }
    for key, cap in STATUS_RATCHET.items():
        if measured[key] > cap:
            errors.append(
                f"docs/STATUS.md {key.replace('_', ' ')} is {measured[key]}, over "
                f"the {cap} ratchet: this page may only shrink. Collapse the "
                "superseded narrative into the binding result and move the "
                f"detail to {RECORD_LINK} or structured state evidence, then lower the "
                "ratchet in the same change"
            )

    return errors


def record_errors(path: Path) -> list[str]:
    """Return problems with the append-only benchmark record."""
    if not path.exists():
        return [
            f"{RECORD_LINK} is missing; it is the append-only record the "
            "scoreboard relocates forensics into, and losing it loses the "
            "evidence"
        ]
    return []


def main() -> int:
    errors: list[str] = []
    for path, check in (
        (BENCHMARKS, benchmarks_errors),
        (FEATURES, features_errors),
        (STATUS, status_errors),
    ):
        if not path.exists():
            errors.append(f"{path.relative_to(ROOT)} is missing")
            continue
        errors += check(path.read_text(encoding="utf-8"))
    errors += record_errors(RECORD)
    if errors:
        print(
            "ERROR: the public keyed-table docs are not valid:",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(
        "OK: docs/BENCHMARKS.md and docs/FEATURES.md are human-readable keyed "
        f"tables, docs/STATUS.md is inside its size ratchet, and {RECORD_LINK} "
        "carries the append-only record."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
