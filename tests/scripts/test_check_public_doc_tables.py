#!/usr/bin/env python3
"""Unit and mutation checks for the public keyed-table doc tooling.

Covers scripts/check-public-doc-tables.py (the CI gate over docs/BENCHMARKS.md
and docs/FEATURES.md) and scripts/roll-benchmark-record.py (the mechanical
rollup the gate points at).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


doc_tables = _load("doc_tables", "scripts/check-public-doc-tables.py")
roll_record = _load("roll_record", "scripts/roll-benchmark-record.py")


# The whole-file character budget retired on 2026-08-12 (#460). Used by exactly
# one test, on a SYNTHETIC page that test builds itself, so that the "a row no
# longer has to evict one" claim can name the size it is talking about. It must
# never be compared against a real page: that would store a measurement of one
# file inside another, which is the defect this row removed.
RETIRED_PAGE_BUDGET = 45000

# A minimal scoreboard that satisfies every rule, used as the mutation baseline.
# MIN_TABLE_ROWS forces a real table, so the baseline carries one.
_ROWS = "\n".join(f"| Model {i} | {i}.0x |" for i in range(60))

VALID = "\n".join(
    [
        "# Benchmarks",
        "",
        "Measured results. See .agents/benchmark-record.md for the record.",
        "",
        "## At a glance",
        "",
        "| Reference | Result |",
        "|---|---|",
        _ROWS,
        "",
        "## How we measure",
        "",
        "Greedy, closed loop, three interleaved repetitions.",
        "",
        "## Open gaps",
        "",
        "| Track | Status |",
        "|---|---|",
        "| Thing | Pending |",
        "",
        "## Reproduce",
        "",
        "```sh",
        "vllm-bench run",
        "```",
        "",
    ]
)


RELEASE_FACTS = (
    "v0.0.2",
    "eight primary archive/checksum/provenance triplets",
    "two indexes",
    "26 assets",
    "7020de93652ca920424a10ac5255b34810dd2f24",
    "31466516224",
    "Windows W14-W16 implemented",
    "native hosted gates",
    "merged-SHA ten-tuple dry run",
    "matching-hardware evidence",
    "v0.0.3-pre.1 publication",
    "32-asset audit",
    "W12 optional/non-primary",
)


def _release_rows(text: str) -> list[str]:
    at_a_glance = text.split("## At a glance", 1)[1].split("\n## ", 1)[0]
    return [
        line for line in at_a_glance.splitlines()
        if line.startswith("| **Binary release")
    ]


def _release_projection_errors(text: str) -> list[str]:
    rows = _release_rows(text)
    errors = []
    if len(rows) != 1:
        errors.append(f"expected one keyed Binary release row, found {len(rows)}")
    projection = "\n".join(rows)
    for fact in RELEASE_FACTS:
        if fact not in projection:
            errors.append(f"release projection is missing {fact!r}")
    return errors


def _project_release_rows(onto: str, from_text: str) -> str:
    """Apply only PR #446's heading and release projection to another page."""
    release_rows = _release_rows(from_text)
    projected = []
    inserted = False
    for line in onto.splitlines():
        if line.startswith("# Benchmarks"):
            projected.append(from_text.splitlines()[0])
        elif line.startswith("| **Binary release"):
            if not inserted:
                projected.extend(release_rows)
                inserted = True
        else:
            projected.append(line)
    return "\n".join(projected) + "\n"


class BenchmarksStructureTests(unittest.TestCase):
    def test_minimal_valid_document_passes(self) -> None:
        self.assertEqual(doc_tables.benchmarks_errors(VALID), [])

    def test_shipped_benchmarks_page_passes(self) -> None:
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        self.assertEqual(doc_tables.benchmarks_errors(text), [])

    def test_release_projection_is_single_and_complete(self) -> None:
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        self.assertEqual(_release_projection_errors(text), [])

    def test_each_release_projection_fact_is_load_bearing(self) -> None:
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        for fact in RELEASE_FACTS:
            with self.subTest(fact=fact):
                mutated = text.replace(fact, "[REMOVED]", 1)
                self.assertTrue(
                    any(fact in error for error in _release_projection_errors(mutated))
                )

    def test_second_release_projection_row_is_rejected(self) -> None:
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        row = _release_rows(text)[0]
        mutated = text.replace(row, f"{row}\n{row}", 1)
        self.assertIn(
            "expected one keyed Binary release row, found 2",
            _release_projection_errors(mutated),
        )

    def test_release_projection_fits_after_current_main_merge(self) -> None:
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        main = subprocess.run(
            ["git", "show", "bbc482a2:docs/BENCHMARKS.md"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        merged = _project_release_rows(main, text)
        self.assertEqual(_release_projection_errors(merged), [])
        # The whole-page size assertion this line used to carry went with
        # `max_chars` on 2026-08-12 (#460). What has to hold after a merge is
        # that every ENTRY is in budget, which the full check below asserts.
        for row in _release_rows(merged):
            self.assertLessEqual(len(row), doc_tables.MAX_ROW_CHARS)
        self.assertEqual(doc_tables.benchmarks_errors(merged), [])

    def test_shipped_record_exists(self) -> None:
        self.assertEqual(
            doc_tables.record_errors(doc_tables.RECORD), []
        )

    def test_missing_record_fails(self) -> None:
        errors = doc_tables.record_errors(ROOT / "does/not/exist.md")
        self.assertTrue(any("benchmark-record.md" in e for e in errors), errors)

    def test_missing_at_a_glance_section_fails(self) -> None:
        mutated = VALID.replace("## At a glance", "## Numbers")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("At a glance" in e for e in errors), errors)

    def test_missing_methodology_section_fails(self) -> None:
        mutated = VALID.replace("## How we measure", "## Notes")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("How we measure" in e for e in errors), errors)

    def test_missing_open_gaps_section_fails(self) -> None:
        mutated = VALID.replace("## Open gaps", "## Misc")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("Open gaps" in e for e in errors), errors)

    def test_missing_reproduce_section_fails(self) -> None:
        mutated = VALID.replace("## Reproduce", "## Commands")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("Reproduce" in e for e in errors), errors)

    def test_appended_sections_fail_as_an_append_log(self) -> None:
        # The drift this checker exists to stop: one narrative H2 per
        # checkpoint until the page is a log again.
        appended = "\n".join(
            f"## CLAIM-SOMETHING-{i} (2026-08-0{i % 9}) - PENDING\n\nProse.\n"
            for i in range(20)
        )
        errors = doc_tables.benchmarks_errors(VALID + "\n" + appended)
        self.assertTrue(any("H2 sections" in e for e in errors), errors)

    def test_a_single_appended_section_fails_immediately(self) -> None:
        # The sharp rule: the FIRST appended entry fails, not the seventh once
        # a count budget fills. This is what actually stops the drift.
        mutated = VALID + "\n## CLAIM-FAKE-LEVER (2026-08-04) - PENDING\n\nProse.\n"
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("non-canonical" in e for e in errors), errors)
        self.assertTrue(any("roll-benchmark-record.py" in e for e in errors), errors)

    def test_a_new_canonical_subject_is_allowed(self) -> None:
        # Adding a real comparison subject named in CANONICAL_SECTIONS passes.
        mutated = VALID + "\n## MLX-LM, Apple M4\n\n| Axis | Ratio |\n|---|---|\n"
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])

    def test_em_dash_fails(self) -> None:
        mutated = VALID.replace("Measured results.", "Measured results — really.")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("em-dash" in e for e in errors), errors)

    # THE PAGE HAS NO SIZE BUDGET as of 2026-08-12 (#460), so the test that
    # used to sit here, asserting a 45,000-char page fails, is gone. It is
    # replaced, not deleted: the obligation it carried moved to the entry cap
    # and the regrowth guard below, and the reason it had to move is
    # test_a_new_row_costs_no_eviction, which the old rule made impossible.

    def test_a_new_row_costs_no_eviction(self) -> None:
        """The acceptance test for #460: a measurement row lands on its own.

        A page already at the retired 45,000-char budget gains one row and
        stays valid. Under the old rule this required deleting somebody else's
        row, and a clean merge of two such payments landed the real page at
        45,007 chars (04b2b9fa), applying both additions and neither eviction.
        """
        filler = "\n".join(
            f"| Subject {i:04d} | {'measured, byte exact, ' * 6}{i}.0x |"
            for i in range(300)
        )
        big = VALID.replace(_ROWS, _ROWS + "\n" + filler)
        self.assertGreater(len(big), RETIRED_PAGE_BUDGET)
        self.assertEqual(doc_tables.benchmarks_errors(big), [])

        added = "| Qwen3.6-35B canonical regrid | PENDING, stale grid |"
        grown = big.replace(filler, filler + "\n" + added)
        self.assertEqual(doc_tables.benchmarks_errors(grown), [])
        self.assertEqual(len(grown), len(big) + len(added) + 1)

    def test_the_shipped_page_can_accept_the_next_measurement_row(self) -> None:
        """#460 on the REAL page: add a row, keep every existing row.

        The row added here is the 35B canonical regrid PR #481 records as owed.
        It is added and dropped again inside this test, so the page is not
        edited; what is asserted is that the surface would accept it.

        The row is inserted DIRECTLY BELOW the last row of the Open gaps table,
        with no blank line between, so markdown renders it inside that table
        rather than it merely satisfying _table_rows. An earlier revision
        separated it by blank lines, which proved the checker claim but not the
        placement.

        Nothing here compares the grown page against the retired 45,000-char
        budget. That assertion would store a measurement of docs/BENCHMARKS.md
        inside this test file, which is exactly what AGENTS.md, Records, forbids
        and what #460 removed: it had 173 characters of margin, so any PR
        compacting the page by more than that turned it red in a file it does
        not own, and 887e04ff shrank the page by 280. The above-the-budget point
        is made by test_a_new_row_costs_no_eviction, on a synthetic page that
        test builds itself.
        """
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        self.assertEqual(doc_tables.benchmarks_errors(text), [])
        owed = (
            "| Qwen3.6-35B canonical regrid (`ROAD-V1-A`, #378) | "
            "**STALE, no current number.** The canonical 0.918x-0.972x grid "
            "predates 136 src commits, one worth +2.05% at c8; holding vLLM "
            "fixed only IMPLIES c8 ~0.937, an estimate | Recapture the 6-point "
            "c1-c32 grid on current main, both arms in one session, before any "
            "35B residual is attributed |"
        )
        # The median Open gaps row on this page is 328 chars and 205 were free,
        # so a REALISTIC row was unaffordable, not just a pathological one.
        self.assertGreater(len(owed), 328)
        self.assertLessEqual(len(owed), doc_tables.MAX_ROW_CHARS)

        before, sep, after = text.partition("\n## Reproduce")
        self.assertEqual(sep, "\n## Reproduce")
        open_gaps = before.rstrip("\n")
        self.assertTrue(
            open_gaps.endswith("|"),
            "the last thing before ## Reproduce is expected to be the final Open "
            "gaps table row; if it is not, this test is inserting the owed row "
            "somewhere markdown will not render it in that table",
        )
        grown = f"{open_gaps}\n{owed}\n{sep}{after}"
        self.assertNotEqual(grown, text)
        self.assertIn(f"|\n{owed}\n", grown)
        self.assertEqual(doc_tables.benchmarks_errors(grown), [])
        # Nothing was evicted to make room.
        for _, _, row in doc_tables._table_rows(text):
            self.assertIn(row, grown)

    def test_oversized_row_fails(self) -> None:
        # Cap the ENTRY: one row past budget is rejected, and the fix is to
        # shorten THAT row.
        wide = "| Subject | " + " | ".join(["x" * 200] * 4) + " |"
        mutated = VALID.replace("| Thing | Pending |", wide)
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("ENTRY budget" in e for e in errors), errors)

    def test_the_row_cap_is_not_subsumed_by_the_cell_cap(self) -> None:
        # Every cell legal, the row illegal: without the row cap a five-column
        # row of 1,100 chars passes.
        cells = " | ".join(["y" * (doc_tables.MAX_CELL_CHARS - 1)] * 4)
        mutated = VALID.replace("| Thing | Pending |", f"| {cells} |")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertEqual([e for e in errors if "wall-of-prose" in e], [])
        self.assertTrue(any("ENTRY budget" in e for e in errors), errors)

    def test_a_row_at_the_cap_is_allowed(self) -> None:
        # Exactly at MAX_ROW_CHARS, with every cell inside MAX_CELL_CHARS.
        # "| a | b | c | d |" costs the four cells plus 13 characters.
        budget = doc_tables.MAX_ROW_CHARS - 13
        widths = [budget // 4] * 4
        widths[0] += budget - sum(widths)
        cells = ["z" * w for w in widths]
        self.assertTrue(all(w <= doc_tables.MAX_CELL_CHARS for w in widths))
        row = "| " + " | ".join(cells) + " |"
        self.assertEqual(len(row), doc_tables.MAX_ROW_CHARS)
        mutated = VALID.replace("| Thing | Pending |", row)
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])

    def test_a_dated_h2_is_rejected(self) -> None:
        mutated = VALID + "\n## vLLM re-grid 2026-08-12\n\nA paragraph.\n"
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("PER-ATTEMPT entry" in e for e in errors), errors)

    def test_a_dated_h3_is_rejected(self) -> None:
        # The hole the retired byte cap was silently covering: the canonical
        # allowlist runs over "## " only, so this was caught by nothing else.
        mutated = VALID.replace(
            "## How we measure",
            "### Qwen3.6-27B by concurrency, 2026-08-12 rerun\n\n"
            "| Point | Ratio |\n|---|---|\n| c1 | 0.9x |\n\n## How we measure",
        )
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("PER-ATTEMPT entry" in e for e in errors), errors)
        self.assertTrue(any("names a date" in e for e in errors), errors)

    def test_a_dated_h3_is_rejected_on_the_feature_matrix_too(self) -> None:
        mutated = VALID_FEATURES + "\n### Coverage sweep 2026-08-12\n\nText.\n"
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("PER-ATTEMPT entry" in e for e in errors), errors)

    def test_a_new_subject_subsection_is_allowed(self) -> None:
        # The guard is not a section freeze: a genuinely new subject passes.
        mutated = VALID.replace(
            "## How we measure",
            "### Laguna-S-2.1 (NVFP4)\n\n"
            "| Point | Ratio |\n|---|---|\n| c1 | 1.03x |\n\n## How we measure",
        )
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])

    def test_a_date_inside_a_fence_is_not_a_heading(self) -> None:
        # Sample output that happens to contain a heading-shaped line is not a
        # section, so the regrowth guard must not fire on it.
        mutated = VALID.replace(
            "vllm-bench run", "vllm-bench run\n### historical 2026-08-04 output"
        )
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])

    def test_a_dated_row_is_still_allowed(self) -> None:
        # The date belongs in the ROW. Only the heading form is the append log.
        mutated = VALID.replace(
            "| Thing | Pending |", "| Thing | Pending, captured 2026-08-12 |"
        )
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])

    def test_the_shipped_pages_carry_no_dated_heading(self) -> None:
        for page in ("docs/BENCHMARKS.md", "docs/FEATURES.md"):
            text = (ROOT / page).read_text(encoding="utf-8")
            dated = [
                title
                for _, title in doc_tables._headings(text)
                if doc_tables.DATED_HEADING_RE.search(title)
            ]
            self.assertEqual(dated, [], page)

    def test_no_page_carries_a_whole_file_size_budget(self) -> None:
        # The invariant behind #460, held as a rule rather than as a habit: a
        # budget on a shared file makes every addition an eviction.
        for rules in (doc_tables.BENCHMARKS_RULES, doc_tables.FEATURES_RULES):
            self.assertFalse(hasattr(rules, "max_chars"), rules.name)
        self.assertNotIn("chars", doc_tables.STATUS_RATCHET)

    def test_a_wall_of_BULLETS_fails(self) -> None:
        """M8: the exact mutant the retired byte cap used to catch.

        3,000 appended bullet lines. Under `max_chars` this was rejected as
        "113833 chars, over the 45000-char scoreboard budget". Between the entry
        cap landing and the 2026-08-12 fold it was rejected by NOTHING:
        _prose_paragraphs excluded every line starting with "-", so the wall was
        outside the paragraph count, MAX_PARAGRAPH_CHARS, MAX_CELL_CHARS,
        MAX_ROW_CHARS and the dated-heading guard at once. It is the append-log
        class the spec's stop condition names, so it is the condition of the
        removal, not a nicety.
        """
        wall = "\n".join(
            f"- attempt {i}: lever measured neutral, reverted, see the record"
            for i in range(3000)
        )
        errors = doc_tables.benchmarks_errors(VALID + "\n\n" + wall + "\n")
        self.assertTrue(any("prose paragraph of" in e for e in errors), errors[:3])

    def test_a_wall_of_BLOCKQUOTES_fails(self) -> None:
        # Same channel, the other excluded prefix.
        wall = "\n".join(f"> quoted forensic note {i}" for i in range(3000))
        errors = doc_tables.benchmarks_errors(VALID + "\n\n" + wall + "\n")
        self.assertTrue(any("prose paragraph of" in e for e in errors), errors[:3])

    def test_appended_UNDATED_subsections_with_bullets_fail(self) -> None:
        """M7: the dated-heading guard fires on a DATE, so it never saw this.

        500 appended UNDATED "### Attempt N" sections carrying bulleted
        forensics measured 117,222 characters with the checker exiting 0 and all
        68 tests green. The heading guard cannot reach it, by construction. The
        paragraph COUNT does, once the bullets are counted.
        """
        appended = "\n".join(
            f"### Attempt {i}\n\n- lever {i} measured neutral on a warm box\n"
            f"- reverted, probe kept\n"
            for i in range(500)
        )
        errors = doc_tables.benchmarks_errors(VALID + "\n" + appended)
        self.assertTrue(any("prose paragraphs" in e for e in errors), errors[:3])

    def test_a_short_bullet_list_is_still_allowed(self) -> None:
        # The fold is a budget, not a ban: a legitimate short list passes.
        mutated = VALID.replace(
            "Greedy, closed loop, three interleaved repetitions.",
            "Greedy, closed loop.\n\n- three interleaved repetitions\n"
            "- one flock holder\n- cold leg discarded\n",
        )
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])

    def test_a_bullet_run_is_ONE_paragraph_not_many(self) -> None:
        # Folding must not turn every legitimate list item into a paragraph, or
        # the count budget would fire on ordinary documents instead of walls.
        text = "intro line\n\n- a\n- b\n- c\n\ntail line\n"
        self.assertEqual(
            [para for _, para in doc_tables._prose_paragraphs(text)],
            ["intro line", "- a - b - c", "tail line"],
        )

    def test_an_EMPHASIS_lead_wall_is_a_KNOWN_residue(self) -> None:
        """Characterisation, not approval: issue #507, spec W8.

        A line beginning with emphasis still starts with "*" and is still
        excluded, so a wall led by "**bold**" is unbounded. Closing it turns
        four paragraphs already shipped on docs/BENCHMARKS.md red against
        MAX_PARAGRAPH_CHARS (717, 719, 748, 1,084 chars), which owes an edit to
        a page #481 holds open, so it takes its own spec and red-before.

        This test goes RED the day that lands. That is the point: the residue is
        recorded where the next author will meet it, and closing it is a
        deliberate act rather than a silent one.
        """
        wall = "\n\n".join(f"**Attempt {i}.** " + "x " * 200 for i in range(500))
        self.assertEqual(doc_tables.benchmarks_errors(VALID + "\n\n" + wall + "\n"), [])

    def test_the_shipped_pages_sit_at_their_folded_paragraph_budget(self) -> None:
        # The budgets are pinned to what the pages MEASURE under the fold, which
        # is what makes re-baselining FEATURES from 20 to 21 a re-measurement
        # and not slack. If either page shrinks, these numbers come down.
        for page, rules, expected in (
            ("docs/BENCHMARKS.md", doc_tables.BENCHMARKS_RULES, 35),
            ("docs/FEATURES.md", doc_tables.FEATURES_RULES, 21),
        ):
            with self.subTest(page=page):
                text = (ROOT / page).read_text(encoding="utf-8")
                self.assertEqual(
                    len(doc_tables._prose_paragraphs(text)), expected
                )
                self.assertEqual(rules.max_prose_paragraphs, expected)

    def test_missing_record_link_fails(self) -> None:
        mutated = VALID.replace(
            "See .agents/benchmark-record.md for the record.", "No record."
        )
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("does not link to" in e for e in errors), errors)

    def test_too_many_prose_paragraphs_fails(self) -> None:
        # Results migrating back into prose, one short paragraph at a time.
        prose = "\n\n".join(f"A result sentence number {i}." for i in range(40))
        errors = doc_tables.benchmarks_errors(VALID + "\n\n" + prose)
        self.assertTrue(any("prose paragraphs" in e for e in errors), errors)

    def test_long_prose_paragraph_fails(self) -> None:
        mutated = VALID.replace(
            "Greedy, closed loop, three interleaved repetitions.", "word " * 300
        )
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("prose paragraph" in e for e in errors), errors)

    def test_wall_of_prose_table_cell_fails(self) -> None:
        wall = "x " * 200
        mutated = VALID.replace("| Thing | Pending |", f"| Thing | {wall} |")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("wall-of-prose" in e for e in errors), errors)

    def test_page_without_tables_fails(self) -> None:
        # Shedding prose by shedding content must not pass.
        mutated = VALID.replace(_ROWS, "| Model 0 | 1.0x |")
        errors = doc_tables.benchmarks_errors(mutated)
        self.assertTrue(any("table rows" in e for e in errors), errors)

    def test_long_code_block_is_allowed(self) -> None:
        mutated = VALID.replace(
            "vllm-bench run", "\n".join(["vllm-bench run"] * 60)
        )
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])

    def test_many_short_table_rows_are_allowed(self) -> None:
        rows = "\n".join(["| Thing | Pending |"] * 120)
        mutated = VALID.replace("| Thing | Pending |", rows)
        self.assertEqual(doc_tables.benchmarks_errors(mutated), [])


VALID_FEATURES = "\n".join(
    [
        "# Features",
        "",
        "See STATUS.md for lifecycle state and BENCHMARKS.md for speed.",
        "",
        "## At a glance",
        "",
        "| Feature | vllm.cpp | vLLM |",
        "|---|---|---|",
        "\n".join(f"| Feature {i} | yes | yes |" for i in range(80)),
        "",
        "## Not supported yet",
        "",
        "| Gap | State |",
        "|---|---|",
        "| Thing | Not started |",
        "",
        "## How to read this page",
        "",
        "A mark means implemented and gated.",
        "",
    ]
)


class FeaturesStructureTests(unittest.TestCase):
    def test_minimal_valid_document_passes(self) -> None:
        self.assertEqual(doc_tables.features_errors(VALID_FEATURES), [])

    def test_shipped_features_page_passes(self) -> None:
        text = (ROOT / "docs/FEATURES.md").read_text(encoding="utf-8")
        self.assertEqual(doc_tables.features_errors(text), [])

    def test_missing_at_a_glance_section_fails(self) -> None:
        mutated = VALID_FEATURES.replace("## At a glance", "## Stuff")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("At a glance" in e for e in errors), errors)

    def test_missing_not_supported_section_fails(self) -> None:
        # The honesty section: a feature matrix without it reads as a sales
        # sheet, which is exactly the drift this row guards.
        mutated = VALID_FEATURES.replace("## Not supported yet", "## Roadmap")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("Not supported yet" in e for e in errors), errors)

    def test_missing_legend_section_fails(self) -> None:
        mutated = VALID_FEATURES.replace("## How to read this page", "## Notes")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("How to read" in e for e in errors), errors)

    def test_missing_status_link_fails(self) -> None:
        mutated = VALID_FEATURES.replace("STATUS.md", "nothing")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("STATUS.md" in e for e in errors), errors)

    def test_missing_benchmarks_link_fails(self) -> None:
        mutated = VALID_FEATURES.replace("BENCHMARKS.md for speed", "speed")
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("BENCHMARKS.md" in e for e in errors), errors)

    def test_prose_replacing_the_matrix_fails(self) -> None:
        mutated = VALID_FEATURES.replace(
            "\n".join(f"| Feature {i} | yes | yes |" for i in range(80)),
            "| Feature 0 | yes | yes |",
        )
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("table rows" in e for e in errors), errors)

    def test_appended_sections_fail_as_an_append_log(self) -> None:
        appended = "\n".join(
            f"## Feature note {i} (2026-08-04)\n\nProse.\n" for i in range(25)
        )
        errors = doc_tables.features_errors(VALID_FEATURES + "\n" + appended)
        self.assertTrue(any("H2 sections" in e for e in errors), errors)

    def test_em_dash_fails(self) -> None:
        mutated = VALID_FEATURES.replace(
            "A mark means implemented and gated.", "A mark means gated — really."
        )
        errors = doc_tables.features_errors(mutated)
        self.assertTrue(any("em-dash" in e for e in errors), errors)

    def test_the_two_pages_have_distinct_budgets(self) -> None:
        # A regression guard on the refactor: FEATURES must not silently
        # inherit the scoreboard's looser limits. The `max_chars` comparison
        # this used to lead with went with the key itself on 2026-08-12 (#460);
        # the remaining per-page limits still have to differ.
        self.assertNotEqual(
            doc_tables.FEATURES_RULES.max_prose_paragraphs,
            doc_tables.BENCHMARKS_RULES.max_prose_paragraphs,
        )
        self.assertNotEqual(
            doc_tables.FEATURES_RULES.min_table_rows,
            doc_tables.BENCHMARKS_RULES.min_table_rows,
        )
        self.assertNotEqual(
            doc_tables.FEATURES_RULES.required_sections,
            doc_tables.BENCHMARKS_RULES.required_sections,
        )


class RollRecordTests(unittest.TestCase):
    def test_shipped_page_has_nothing_to_roll(self) -> None:
        text = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        _, rollable = roll_record.plan(text)
        self.assertEqual([title for title, _ in rollable], [])

    def test_non_canonical_section_is_rollable(self) -> None:
        text = VALID + "\n## CLAIM-FOO decode attribution (2026-08-04)\n\nProse.\n"
        kept, rollable = roll_record.plan(text)
        self.assertEqual(len(rollable), 1)
        self.assertIn("CLAIM-FOO", rollable[0][0])
        self.assertTrue(all("CLAIM-FOO" not in t for t, _ in kept), kept)

    def test_canonical_sections_are_kept(self) -> None:
        kept, rollable = roll_record.plan(VALID)
        self.assertEqual(rollable, [])
        self.assertEqual(len(kept), 4)

    def test_rebuild_drops_only_the_rolled_sections(self) -> None:
        extra = "## CLAIM-BAR (2026-08-04)\n\nForensics.\n"
        text = VALID + "\n" + extra
        preamble, _ = roll_record.split_sections(text)
        kept, rollable = roll_record.plan(text)
        rebuilt = roll_record.rebuild(preamble, kept)
        self.assertNotIn("CLAIM-BAR", rebuilt)
        self.assertIn("## At a glance", rebuilt)
        # The moved text is preserved verbatim for the record.
        self.assertIn("Forensics.", rollable[0][1])

    def test_rebuilt_page_still_passes_the_checker(self) -> None:
        text = VALID + "\n## CLAIM-BAZ (2026-08-04)\n\n" + ("word " * 300) + "\n"
        self.assertTrue(doc_tables.benchmarks_errors(text))
        preamble, _ = roll_record.split_sections(text)
        kept, _ = roll_record.plan(text)
        rebuilt = roll_record.rebuild(preamble, kept)
        self.assertEqual(doc_tables.benchmarks_errors(rebuilt), [])

    def test_allowlist_has_exactly_one_definition(self) -> None:
        # The rollup and the CI gate must never disagree about what moves. The
        # rollup reads the list from the checker rather than copying it, so the
        # two allowlists are the same values, loaded from one file.
        self.assertEqual(
            roll_record.RULES.canonical_sections,
            doc_tables.BENCHMARKS_RULES.canonical_sections,
        )
        for title in ("At a glance", "vLLM, online serving", "Reproduce"):
            with self.subTest(title=title):
                self.assertEqual(
                    roll_record.is_canonical(title),
                    doc_tables.BENCHMARKS_RULES.is_canonical(title),
                )

    def test_what_the_checker_rejects_is_what_the_rollup_moves(self) -> None:
        text = VALID + "\n## CLAIM-QUX attribution (2026-08-04)\n\nProse.\n"
        self.assertTrue(
            any("non-canonical" in e for e in doc_tables.benchmarks_errors(text))
        )
        _, rollable = roll_record.plan(text)
        self.assertEqual(len(rollable), 1)
        self.assertIn("CLAIM-QUX", rollable[0][0])

    def test_rolling_is_byte_exact_on_the_kept_sections(self) -> None:
        # Rolling must change nothing except removing the rolled sections, so
        # repeated rolls are idempotent and never churn the scoreboard.
        base = (ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
        text = base + "\n## CLAIM-CHURN (2026-08-04)\n\nProse.\n"
        preamble, _ = roll_record.split_sections(text)
        kept, _ = roll_record.plan(text)
        self.assertEqual(roll_record.rebuild(preamble, kept), base)

    def test_features_page_has_no_canonical_allowlist(self) -> None:
        # New feature AREAS are a legitimate way for that page to grow, so it
        # is governed by the section budget alone.
        self.assertIsNone(doc_tables.FEATURES_RULES.canonical_sections)
        self.assertIsNotNone(doc_tables.BENCHMARKS_RULES.canonical_sections)

    def test_heading_inside_a_fence_is_not_a_section(self) -> None:
        text = VALID.replace("vllm-bench run", "## not a heading")
        _, sections = roll_record.split_sections(text)
        self.assertTrue(all("not a heading" not in t for t, _ in sections))



# docs/STATUS.md is guarded by a RATCHET, not a budget: it may only shrink. The
# mutations therefore prove both directions -- growth fails, shrinkage passes --
# and that the required sections cannot be dropped.
STATUS_VALID = "\n".join(
    [
        "# Status",
        "",
        "## Parity pin",
        "",
        "vLLM 0.26.0.dev0.",
        "",
        "## Capability status",
        "",
        "| Capability | State |",
        "|---|---|",
        "| Thing | Works |",
        "",
        "## Not supported yet",
        "",
        "| Thing | Why |",
        "|---|---|",
        "| Other | Not started |",
    ]
)


class StatusRatchet(unittest.TestCase):
    def test_small_valid_page_passes(self) -> None:
        self.assertEqual(doc_tables.status_errors(STATUS_VALID), [])

    def test_each_required_section_is_enforced(self) -> None:
        for label, matchers in doc_tables.STATUS_REQUIRED:
            text = STATUS_VALID.replace(f"## {label}", "## Something else")
            with self.subTest(section=label):
                self.assertTrue(
                    any(label in e for e in doc_tables.status_errors(text)),
                    f"dropping '{label}' was not rejected",
                )

    def test_growth_past_the_section_ratchet_is_rejected(self) -> None:
        extra = "\n".join(
            f"## Extra {i}\n\nprose\n" for i in range(doc_tables.STATUS_RATCHET["h2_sections"] + 1)
        )
        errors = doc_tables.status_errors(STATUS_VALID + "\n" + extra)
        self.assertTrue(any("h2 sections is" in e for e in errors))

    def test_growth_past_the_paragraph_ratchet_is_rejected(self) -> None:
        para = "y" * (doc_tables.MAX_PARAGRAPH_CHARS + 1)
        extra = "\n\n".join([para] * (doc_tables.STATUS_RATCHET["long_paragraphs"] + 1))
        errors = doc_tables.status_errors(STATUS_VALID + "\n\n" + extra)
        self.assertTrue(any("long paragraphs is" in e for e in errors))

    def test_growth_past_the_cell_ratchet_is_rejected(self) -> None:
        cell = "z" * (doc_tables.MAX_CELL_CHARS + 1)
        rows = "\n".join(
            f"| {cell} | {cell} |" for _ in range(doc_tables.STATUS_RATCHET["oversized_cells"])
        )
        errors = doc_tables.status_errors(STATUS_VALID + "\n\n| a | b |\n|---|---|\n" + rows)
        self.assertTrue(any("oversized cells is" in e for e in errors))

    def test_a_lifecycle_line_costs_no_unrelated_deletion(self) -> None:
        """The defect ENG-RECORD-CONFLICT-SURFACES (#364) removed.

        This is the RED-BEFORE case. With the `chars` key in place it failed:
        a page that gained one ordinary lifecycle line was over its byte
        ratchet, so the PR had to find the bytes by deleting prose from some
        UNRELATED row and re-pin the constant in the checker. That is what
        coupled every concurrent PR to lines it did not own and made both
        docs/STATUS.md and this checker merge hotspots.

        A lifecycle line is the most ordinary edit this page takes. It must
        cost nothing but itself.
        """
        line = "| New capability | Works |\n"
        grown = STATUS_VALID + "\n" + line
        self.assertEqual(doc_tables.status_errors(grown), [])

        # And on the LIVE page, which is the case that actually failed.
        live = doc_tables.STATUS.read_text(encoding="utf-8")
        self.assertEqual(doc_tables.status_errors(live + line), [])

    def test_two_concurrent_lifecycle_lines_do_not_couple(self) -> None:
        """Two PRs adding a row each must both pass, independently.

        The byte ratchet made this impossible by construction: whichever landed
        second was over the number the first had re-pinned, so it had to be
        rebased and re-paid even though the two edits touch nothing in common.
        Asserting BOTH arms and their union is what proves the coupling is gone
        rather than merely relaxed for one of them.
        """
        live = doc_tables.STATUS.read_text(encoding="utf-8")
        a = "| Capability A | Works |\n"
        b = "| Capability B | Works |\n"
        self.assertEqual(doc_tables.status_errors(live + a), [])
        self.assertEqual(doc_tables.status_errors(live + b), [])
        self.assertEqual(doc_tables.status_errors(live + a + b), [])

    def test_the_retained_keys_still_bind_on_the_live_page(self) -> None:
        """Removing `chars` must not have quietly disarmed the other three.

        Each retained key counts a QUALITY defect rather than a length, so this
        mutates the LIVE page past each one in turn and requires a failure. If a
        later edit lands the page on a cap, this goes red instead of the cap
        silently ceasing to bind.
        """
        live = doc_tables.STATUS.read_text(encoding="utf-8")
        self.assertEqual(doc_tables.status_errors(live), [])

        over_sections = live + "\n" + "\n".join(
            f"## Extra {i}\n\nprose\n"
            for i in range(doc_tables.STATUS_RATCHET["h2_sections"] + 1)
        )
        self.assertTrue(
            any("h2 sections is" in e for e in doc_tables.status_errors(over_sections))
        )

        para = "y" * (doc_tables.MAX_PARAGRAPH_CHARS + 1)
        over_paras = live + "\n\n" + "\n\n".join(
            [para] * (doc_tables.STATUS_RATCHET["long_paragraphs"] + 1)
        )
        self.assertTrue(
            any("long paragraphs is" in e for e in doc_tables.status_errors(over_paras))
        )

        cell = "z" * (doc_tables.MAX_CELL_CHARS + 1)
        over_cells = live + "\n\n| a | b |\n|---|---|\n" + "\n".join(
            f"| {cell} | {cell} |"
            for _ in range(doc_tables.STATUS_RATCHET["oversized_cells"] + 1)
        )
        self.assertTrue(
            any("oversized cells is" in e for e in doc_tables.status_errors(over_cells))
        )

    def test_the_live_page_is_inside_its_ratchet(self) -> None:
        self.assertEqual(
            doc_tables.status_errors(doc_tables.STATUS.read_text(encoding="utf-8")), []
        )

    def test_a_retired_claim_cannot_come_back_for_free(self) -> None:
        """A claim the page RETIRED must cost something to reinstate.

        The ratchet is what makes a deletion permanent. #277 paid for its STATUS
        edit by deleting two claims that had become false -- that `/metrics` has
        no live async backing, once in the OpenAI-server row and once in the
        metrics paragraph. The char ratchet it also lowered is gone (#364);
        the deletion obligation asserted here is not.

        Two things are asserted, and neither is the byte-tightness above.
        First, the retired wording is genuinely GONE: a page that still says
        `/metrics` is unbacked while the AsyncLLM output handler records into it
        is lying to a reader, and no size ratchet notices a false sentence.
        Second, putting it back is rejected -- so a future edit reinstating it
        has to find the space, out loud, instead of spending headroom the
        deletion left behind.
        """
        text = doc_tables.STATUS.read_text(encoding="utf-8")
        self.assertEqual(doc_tables.status_errors(text), [])

        retired = (
            "metrics and cache reset lack live async backing",
            "the async production-serving path wiring",
        )
        for claim in retired:
            with self.subTest(claim=claim):
                self.assertNotIn(
                    claim,
                    text,
                    "docs/STATUS.md still carries a claim about /metrics that "
                    "the AsyncLLM wiring made false",
                )

        restored = text.replace(
            "cache reset lacks live async backing",
            retired[0],
            1,
        ).replace(
            "The remaining work is the chat/completion",
            f"The remaining work is {retired[1]}, the chat/completion",
            1,
        )
        self.assertGreater(
            len(restored),
            len(text),
            "the mutation must actually re-add the retired claims; if the "
            "anchors stopped matching, this test is asserting nothing",
        )
        # The char ratchet used to be the second half of this test: reinstating
        # the claims had to break it. That half is gone with the ratchet
        # (ENG-RECORD-CONFLICT-SURFACES, #364) and is deliberately NOT replaced
        # by a weaker assertion, because it was never what made the claim false
        # -- no size gate can tell a true sentence from a lying one. What
        # survives is the obligation that actually mattered and that the ratchet
        # never carried: the retired wording is gone from the live page.
        self.assertEqual(doc_tables.status_errors(restored), [])

    def test_the_status_ratchet_only_ever_moves_down(self) -> None:
        """A ratchet that can be RAISED is a budget with extra steps.

        The checker states the rule in prose -- "every limit is pinned to what
        the page measured ... and may only go DOWN" -- and nothing enforced it.
        Every other ratchet test here measures the LIVE page, so raising a cap
        and growing the page by the same amount in one change is green in all of
        them: raising a cap and growing the page by the same amount in one
        change was green in all of them. That is the cheap way out the ratchet
        exists to block. The `chars` key it was written for is gone (#364) --
        it was the one every STATUS edit touched, which is what made it a merge
        hotspot -- and the guard still binds the three that remain.

        STATUS_RATCHET_CEILING is the ratchet as it stands. Lowering a cap keeps
        passing; raising one fails here and can only be unblocked by editing the
        ceiling in the same change -- a deliberate, reviewable act instead of a
        silent bump. The caps have in fact only ever fallen (chars 289727 ->
        243584 over 48 commits), so the ceiling never needs to rise.
        """
        ceiling = {
            "h2_sections": 11,
            # 75, down from 82 on 2026-08-12: _prose_paragraphs now folds list
            # items in, so docs/STATUS.md measures 75 long paragraphs where it
            # measured 82, and the ratchet followed the measurement DOWN in the
            # same change. Lowered here in the same commit, which is what this
            # test's own message demands.
            "long_paragraphs": 75,
            "oversized_cells": 44,
        }
        self.assertEqual(
            set(doc_tables.STATUS_RATCHET),
            set(ceiling),
            "every ratcheted metric needs a ceiling here, and a ceiling with no "
            "metric behind it is a metric that quietly stopped being enforced",
        )
        for key, cap in doc_tables.STATUS_RATCHET.items():
            with self.subTest(metric=key):
                self.assertLessEqual(
                    cap,
                    ceiling[key],
                    f"the {key} ratchet moved UP to {cap} from {ceiling[key]}; "
                    "docs/STATUS.md may only shrink. Collapse the superseded "
                    "narrative instead, and lower this ceiling in the same "
                    "change that lowers the ratchet -- never the reverse",
                )
