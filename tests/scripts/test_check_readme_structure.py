#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-readme-structure.py."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-readme-structure.py"
SPEC = importlib.util.spec_from_file_location("readme_structure", CHECKER)
assert SPEC is not None and SPEC.loader is not None
readme_structure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = readme_structure
SPEC.loader.exec_module(readme_structure)


# Test-owned literal: never derive this from the production checker constant,
# or deleting/changing that constant could change both setup and expectation.
EXPECTED_CONTRIBUTOR_LINK = "CONTRIBUTING.md"


# A minimal document that satisfies every rule, used as the mutation baseline.
VALID = "\n".join(
    [
        "# vllm.cpp",
        "",
        "One-paragraph intro.",
        "",
        "## Features",
        "",
        "| Capability | State |",
        "|---|---|",
        "| Thing | Works |",
        "",
        "## Supported models",
        "",
        "A short list.",
        "",
        "## Performance",
        "",
        "Measured numbers.",
        "",
        "## Build",
        "",
        "```sh",
        "cmake -S . -B build",
        "```",
        "",
        "## Running inference (CLI)",
        "",
        "vllm-cli usage.",
        "",
        "## OpenAI-compatible server",
        "",
        "server usage.",
        "",
        "## Consuming it as a library (C API and C++)",
        "",
        "Link libvllm.",
        "",
        "Contributor guide: CONTRIBUTING.md",
        "",
    ]
)

class ReadmeStructureTests(unittest.TestCase):
    def test_minimal_valid_document_passes(self) -> None:
        self.assertEqual(readme_structure.readme_errors(VALID), [])

    def test_shipped_readme_passes(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(readme_structure.readme_errors(text), [])

    def test_missing_features_section_fails(self) -> None:
        mutated = VALID.replace("## Features", "## Feetures")
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any("Features" in e for e in errors), errors)

    def test_missing_build_section_fails(self) -> None:
        mutated = VALID.replace("## Build", "## Compilation notes")
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any("Build" in e for e in errors), errors)

    def test_missing_cli_usage_section_fails(self) -> None:
        mutated = VALID.replace("## Running inference (CLI)", "## Notes")
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any("Usage / CLI" in e for e in errors), errors)

    def test_missing_server_section_fails(self) -> None:
        mutated = VALID.replace("## OpenAI-compatible server", "## Endpoints")
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any("OpenAI server" in e for e in errors), errors)

    def test_missing_consuming_section_fails(self) -> None:
        mutated = VALID.replace(
            "## Consuming it as a library (C API and C++)", "## Embedding"
        )
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any("Consuming" in e for e in errors), errors)

    def test_em_dash_fails(self) -> None:
        mutated = VALID.replace("One-paragraph intro.", "An intro — with a dash.")
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any("em-dash" in e for e in errors), errors)

    def test_wall_of_prose_table_cell_fails(self) -> None:
        wall = "x " * 300  # ~600 chars, well over the threshold
        mutated = VALID.replace("| Thing | Works |", f"| Thing | {wall} |")
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any("wall-of-prose" in e for e in errors), errors)

    def test_long_prose_paragraph_fails(self) -> None:
        # A wall-of-prose paragraph is the drift this rule exists to stop. It is
        # reported as a paragraph problem, not as a table-cell one.
        mutated = VALID.replace("Measured numbers.", "word " * 300)
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any("prose paragraph" in e for e in errors), errors)

    def test_long_code_block_is_allowed(self) -> None:
        # Fenced code is exempt: a long build recipe is not wall-of-prose.
        mutated = VALID.replace(
            "cmake -S . -B build", "\n".join(["cmake -S . -B build"] * 60)
        )
        errors = readme_structure.readme_errors(mutated)
        self.assertEqual(errors, [])

    def test_long_table_is_allowed(self) -> None:
        # Many short rows are fine; it is long CELLS that are the smell.
        rows = "\n".join(["| Thing | Works |"] * 80)
        mutated = VALID.replace("| Thing | Works |", rows)
        errors = readme_structure.readme_errors(mutated)
        self.assertEqual(errors, [])

    def test_oversized_readme_passes_when_every_entry_is_small(self) -> None:
        # #498: the landing page carries no whole-file budget. A README far past
        # the retired 30,000-char cap is valid so long as every ENTRY is small,
        # because a shared-file budget makes each addition evict someone else's
        # content and lets two individually-valid edits merge into a failure.
        mutated = VALID + "\n" + ("- a filler bullet line\n" * 3000)
        self.assertGreater(len(mutated), 30000)
        self.assertEqual(readme_structure.readme_errors(mutated), [])

    def test_checker_declares_no_whole_file_budget(self) -> None:
        # The anti-regression tooth: fails if the constant returns under ANY
        # value, not merely if the current threshold is raised.
        self.assertFalse(
            hasattr(readme_structure, "MAX_README_CHARS"),
            "MAX_README_CHARS is back; AGENTS.md Records forbids a budget on a "
            "whole shared file (see .agents/specs/readme-budget-retire.md)",
        )

    def test_entry_budgets_survive_the_file_budget_removal(self) -> None:
        # Mutation guard: removing the file cap must not disable the per-entry
        # caps. An oversized paragraph and an oversized cell must still fail,
        # each on its own, inside a document that is otherwise valid.
        long_para = VALID.replace("Measured numbers.", "word " * 300)
        self.assertTrue(
            any("wall-of-prose" in e for e in readme_structure.readme_errors(long_para)),
            readme_structure.readme_errors(long_para),
        )
        long_cell = VALID.replace("| Thing | Works |", f"| Thing | {'x' * 300} |")
        self.assertTrue(
            any("wall-of-prose" in e for e in readme_structure.readme_errors(long_cell)),
            readme_structure.readme_errors(long_cell),
        )

    def test_checker_exposes_the_literal_contributor_link(self) -> None:
        self.assertEqual(
            getattr(readme_structure, "CONTRIBUTOR_LINK", None),
            EXPECTED_CONTRIBUTOR_LINK,
        )

    def test_missing_contributor_link_fails(self) -> None:
        mutated = VALID.replace(
            f"Contributor guide: {EXPECTED_CONTRIBUTOR_LINK}", "No guide."
        )
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any(EXPECTED_CONTRIBUTOR_LINK in e for e in errors), errors)

    def test_main_runs_the_contributor_link_check(self) -> None:
        missing_link = VALID.replace(
            f"Contributor guide: {EXPECTED_CONTRIBUTOR_LINK}", "No guide."
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readme = root / "README.md"
            readme.write_text(missing_link, encoding="utf-8")

            saved_readme, readme_structure.README = readme_structure.README, readme
            out, err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = readme_structure.main()
            finally:
                readme_structure.README = saved_readme

        self.assertEqual(code, 1)
        self.assertIn(EXPECTED_CONTRIBUTOR_LINK, err.getvalue())

    def test_tightened_cell_budget_catches_mid_length_cells(self) -> None:
        # 300 chars passed under the old 400-char threshold; it must not now.
        cell = "x" * 300
        mutated = VALID.replace("| Thing | Works |", f"| Thing | {cell} |")
        errors = readme_structure.readme_errors(mutated)
        self.assertTrue(any("wall-of-prose" in e for e in errors), errors)
if __name__ == "__main__":
    unittest.main()
