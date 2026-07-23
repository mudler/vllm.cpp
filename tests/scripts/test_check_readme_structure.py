#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-readme-structure.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-readme-structure.py"
SPEC = importlib.util.spec_from_file_location("readme_structure", CHECKER)
assert SPEC is not None and SPEC.loader is not None
readme_structure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = readme_structure
SPEC.loader.exec_module(readme_structure)


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

    def test_long_prose_outside_a_table_is_allowed(self) -> None:
        # A long paragraph (not a table cell) must NOT trip the cell check.
        mutated = VALID.replace("Measured numbers.", "word " * 300)
        errors = readme_structure.readme_errors(mutated)
        self.assertFalse(any("wall-of-prose" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
