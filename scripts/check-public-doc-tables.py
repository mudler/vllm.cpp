#!/usr/bin/env python3
"""Enforce that the public keyed-table docs stay human-readable.

Per AGENTS.md, docs/BENCHMARKS.md and docs/FEATURES.md are KEYED TABLES in the
same class as .agents/roadmap_v1.md and the area matrices: one row per subject,
updated in place. They are NOT append logs. Left unchecked they drift back into
one a checkpoint at a time (BENCHMARKS.md reached 11,405 lines and 171 sections
before the 2026-08-04 conversion), which is exactly what makes them unreadable
to users.

This checker fails if either page loses a required user-facing section, appends
a section or a per-attempt dated heading instead of updating a row, lets one
entry grow into a wall of prose, or stops pointing at the record it relocates
detail into. When BENCHMARKS.md fails because sections accumulated,
`scripts/roll-benchmark-record.py` moves them into .agents/benchmark-record.md.

Nothing here budgets the WHOLE FILE. Adding a measurement row must never
require deleting a row someone else owns: see the MAX_ROW_CHARS comment for the
measurement behind that, and AGENTS.md, Records, "cap the entry, never the
file".

The validation logic is the pure functions `benchmarks_errors(text)` and
`features_errors(text)` so they are unit-testable and mutation-testable (see
tests/scripts/test_check_public_doc_tables.py), mirroring
check-readme-structure.py and check-doc-checkpoint.py.
"""

from __future__ import annotations

import re
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

# THE PER-PAGE `max_chars` BUDGET WAS REMOVED 2026-08-12
# (ENG-RECORD-CONFLICT-SURFACES, #460). It was 45,000 for the scoreboard and
# 30,000 for the feature matrix, and both pages sat against it: BENCHMARKS.md
# measured 44,795 of 45,000 and FEATURES.md 29,740 of 30,000. A budget on a
# SHARED file makes every addition an eviction of somebody else's row, which is
# the corollary AGENTS.md Records states outright ("cap the entry, never the
# file"), and 87308dea already removed the two sibling budgets on this argument
# under #364: MAX_CHARS in check-now-current.py and the `chars` key of
# STATUS_RATCHET below. This was the third, left standing in that pass.
#
# The measured consequences, over the last 25 commits touching BENCHMARKS.md:
# free space ranged from 421 characters down to MINUS SEVEN; row count fell 165
# to 162 while the project gained measurements; two commits (93613baa, 887e04ff)
# exist for no purpose but to pay rent; and 04b2b9fa is a CLEAN automatic merge
# that landed the page at 45,007 chars, over the cap, because two PRs each paid
# by evicting a different row and the three-way merge applied both additions and
# neither eviction. A gate whose success mode is unsafe is worse than no gate.
#
# MAX_ROW_CHARS and DATED_HEADING_RE carry the obligation between them, and both
# are ENTRY-scoped, so an author bounds their own row and never anyone else's.

# One table row is one ENTRY, and this is its budget. Set from the shipped
# pages: the longest live row is 520 chars on BENCHMARKS.md and 580 on
# FEATURES.md. It is a real constraint and a tighter one than the cell cap it
# joins, which alone permits a five-column row of 1,100 characters.
MAX_ROW_CHARS = 600

# REGROWTH GUARD. What actually bloated BENCHMARKS.md to 11,405 lines was
# PER-ATTEMPT sections, appended one checkpoint at a time, and the page's own
# archive records their shape: 287 of the 310 sections already rolled into
# .agents/benchmark-record.md name a DATE in the heading. Zero of the 36 live
# headings across the two public pages do (18 each; reproduce with _headings).
# So a dated heading is the append-log entry, and it fails at the FIRST one.
#
# WHAT THIS GUARD IS AND IS NOT TIGHTER THAN. It is strictly tighter than the
# canonical-section allowlist it joins, which runs over _h2_headers and so
# matches "## " only: an appended DATED "### " subsection was rejected by
# nothing but the retired character budget, and is now rejected here at every
# depth. It is NOT tighter than the retired byte cap in general, and an earlier
# revision of this comment claimed that it was. An UNDATED appended subsection
# passes this guard by construction, because the guard fires on a date. What
# bounds that case is the paragraph budget, once _prose_paragraphs counts the
# bulleted forensics such a section carries: see the fold recorded there, and
# the two mutations that measured the gap before it existed.
DATED_HEADING_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")

# The scoreboard must point at the record, and the record must exist, otherwise
# "move it to the archive" silently loses the evidence.
RECORD_LINK = ".agents/benchmark-record.md"


class PageRules:
    """The budget for one public keyed-table page.

    Every limit is set from what the page measured when it was written, with
    headroom for genuinely new subjects but not for accumulated entries. That
    is the whole mechanism: growth that is new ROWS passes, growth that is new
    SECTIONS or new PROSE fails.

    NONE of these limits is a budget on the whole file: see the MAX_ROW_CHARS
    comment above for why the per-page `max_chars` key was removed on
    2026-08-12. A limit here either counts a QUALITY defect (sections,
    paragraphs) or bounds ONE entry, so adding a ROW never requires deleting a
    row someone else owns.

    THAT SENTENCE IS TRUE OF ROWS, AND ONLY OF ROWS. `max_h2_sections` and
    `max_prose_paragraphs` are still whole-page COUNTS, and both live pages sit
    on the paragraph one: docs/BENCHMARKS.md measures 35 of 35 and
    docs/FEATURES.md 21 of 21. So adding a PARAGRAPH does still cost somebody's
    paragraph. That is deliberate rather than overlooked. Rows are the growth
    mode of a keyed table and prose is the decay mode this checker exists to
    stop, which is the same argument #364 used to keep `long_paragraphs` in
    STATUS_RATCHET while deleting `chars`. It is a real cost, not a free one,
    and it is recorded here so the next author meets it in the docstring instead
    of in CI.
    """

    def __init__(
        self,
        *,
        name: str,
        kind: str,
        required_sections: tuple[tuple[str, tuple[str, ...]], ...],
        max_h2_sections: int,
        max_prose_paragraphs: int,
        min_table_rows: int,
        required_links: tuple[str, ...],
        canonical_sections: tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.required_sections = required_sections
        self.max_h2_sections = max_h2_sections
        self.max_prose_paragraphs = max_prose_paragraphs
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
    # 21, not 20, since 2026-08-12: _prose_paragraphs now folds list items in,
    # and this page carries one (the C-ABI capability note). Re-baselined to
    # what the page measures under the new definition, exactly as the old number
    # was pinned to what it measured under the old one. Nothing was widened: the
    # population counted grew, the headroom did not.
    max_prose_paragraphs=21,
    min_table_rows=60,
    required_links=("STATUS.md", "BENCHMARKS.md"),
)


def _h2_headers(text: str) -> list[str]:
    return [ln[3:].strip() for ln in text.splitlines() if ln.startswith("## ")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell) <= set("-: ") for cell in cells)


LIST_ITEM_RE = re.compile(r"^(?:[-*+](?:\s|$)|\d+[.)]\s)")


def _prose_paragraphs(text: str) -> list[tuple[int, str]]:
    """Yield (start_line, paragraph) for narrative prose.

    Fenced code blocks, tables and headings are excluded: the rule targets the
    narrative paragraph, not legitimate tables or code samples.

    LIST ITEMS AND BLOCKQUOTE LINES ARE PROSE HERE, folded into the paragraph
    that runs through them, and that is a 2026-08-12 repair, not the original
    behaviour (#460, review finding F1). Excluding them meant a bulleted or
    quoted wall was counted by NOTHING: not this budget, not MAX_PARAGRAPH_CHARS,
    not MAX_CELL_CHARS, not MAX_ROW_CHARS, and not the dated-heading guard, which
    only sees a DATE. Two mutations proved it on the real checker: 3,000 appended
    bullet lines took docs/BENCHMARKS.md to 113,833 characters and 500 appended
    UNDATED "### Attempt N" sections with bulleted forensics took it to 117,222,
    and both reported no errors. Folding makes a contiguous run one paragraph, so
    the wall trips MAX_PARAGRAPH_CHARS, and a run per section trips the paragraph
    COUNT. Neither live page gains a paragraph from the fold except
    docs/FEATURES.md, which gains its one list item; see max_prose_paragraphs.

    KNOWN RESIDUE, deliberately left and filed, not silently kept: a line
    beginning with EMPHASIS rather than a list marker ("**Protocol.** ...") is
    still excluded, because it starts with "*". That is the accident this
    exclusion always was, and closing it turns four paragraphs already shipped on
    docs/BENCHMARKS.md red at 717, 719, 748 and 1,084 characters against
    MAX_PARAGRAPH_CHARS = 700. Fixing it therefore owes an edit to a page #481
    holds open, so it takes its own spec and its own red-before rather than
    riding along here: issue #507, spec W8. A prose wall led by "**bold**" is
    UNBOUNDED until then.
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
        # "-", "+" and ">" are gone from this list entirely, which IS the fold:
        # a list item or a quoted line now joins the paragraph running through
        # it. Only "|", "#" and an emphasis-lead "*" still break a paragraph,
        # and the last of those is the residue documented above, not a rule.
        is_prose = bool(stripped) and not (
            stripped.startswith("|")
            or stripped.startswith("#")
            or (stripped.startswith("*") and not LIST_ITEM_RE.match(stripped))
        )
        if is_prose:
            if not current:
                start = lineno
            current.append(stripped)
        else:
            flush()
    flush()
    return paragraphs


def _table_rows(text: str) -> list[tuple[int, list[str], str]]:
    """Yield (line_number, cells, raw_row) for every non-separator table row.

    The raw row travels with the cells because MAX_ROW_CHARS bounds the ENTRY,
    which is the whole row, not any one of its cells.
    """
    rows: list[tuple[int, list[str], str]] = []
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
                rows.append((lineno, cells, stripped))
    return rows


def _headings(text: str) -> list[tuple[int, str]]:
    """Yield (line_number, title) for every ATX heading, fences excluded.

    Every DEPTH, unlike _h2_headers: the regrowth guard has to see the "### "
    subsections the canonical-section allowlist never covered.
    """
    headings: list[tuple[int, str]] = []
    in_fence = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                headings.append((lineno, title))
    return headings


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

    # REGROWTH GUARD (see DATED_HEADING_RE). A dated heading is what a
    # per-attempt entry looks like on this page, at any depth, and it fails at
    # the FIRST one rather than once a count budget fills up.
    for lineno, title in _headings(text):
        if DATED_HEADING_RE.search(title):
            errors.append(
                f"line {lineno}: heading {title[:60]!r} names a date, so it is "
                f"a PER-ATTEMPT entry; {rules.name} is a KEYED TABLE and a "
                "checkpoint updates its ROW. Run "
                "scripts/roll-benchmark-record.py --apply to move it verbatim "
                f"into {RECORD_LINK}, and put the date in the row or the prose"
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
    for lineno, cells, raw_row in rows:
        for cell in cells:
            if len(cell) > MAX_CELL_CHARS:
                errors.append(
                    f"line {lineno}: table cell of {len(cell)} chars exceeds "
                    f"{MAX_CELL_CHARS} (wall-of-prose smell; move the detail "
                    f"to {RECORD_LINK})"
                )
        # THE ENTRY CAP (see MAX_ROW_CHARS). This bounds YOUR row, so the cost
        # of a new measurement is paid by shortening it, never by deleting a
        # row someone else owns.
        if len(raw_row) > MAX_ROW_CHARS:
            errors.append(
                f"line {lineno}: table row of {len(raw_row)} chars exceeds the "
                f"{MAX_ROW_CHARS}-char ENTRY budget; shorten THIS row and move "
                f"its forensics to {RECORD_LINK}. The page itself has no budget "
                "and never needs an unrelated row deleted to make space"
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
    # 75, down from 82 on 2026-08-12. Not a compaction of the page:
    # _prose_paragraphs now folds list items and blockquote lines into the
    # paragraph running through them, so docs/STATUS.md's 29 list items join
    # neighbouring paragraphs instead of splitting them, and the page measures
    # 75 long paragraphs where it measured 82. A ratchet is pinned to what the
    # page measures, so it follows the measurement DOWN in the same change that
    # moved it. Leaving 82 would have banked 7 units of slack this row did not
    # earn.
    "long_paragraphs": 75,
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
            for _, cells, _raw in _table_rows(text)
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
