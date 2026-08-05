#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-protocol-consistency.py.

The gate exists to catch prose drifting away from the checker that enforces it,
so the mutations here are the real historical failure: a document that still
names README.md as a checkpoint surface, and a document whose contract silently
disagrees with scripts/check-doc-checkpoint.py.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
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


consistency = _load("protocol_consistency", "scripts/check-protocol-consistency.py")

EXPECTED = ("docs/STATUS.md", "docs/BENCHMARKS.md", "docs/FEATURES.md")


def document(*paths: str) -> str:
    rows = "\n".join(f"| `{path}` | every checkpoint |" for path in paths)
    return "\n".join(
        [
            "# Some normative document",
            "",
            consistency.BEGIN,
            "| Public surface | Owed by |",
            "|---|---|",
            rows,
            consistency.END,
            "",
            "Trailing prose.",
        ]
    )


class ContractParsing(unittest.TestCase):
    def test_extracts_paths_in_order(self) -> None:
        self.assertEqual(
            consistency.contract_paths(document(*EXPECTED)), list(EXPECTED)
        )

    def test_absent_block_is_none(self) -> None:
        self.assertIsNone(consistency.contract_paths("# No contract here"))

    def test_end_before_begin_is_none(self) -> None:
        text = f"{consistency.END}\n| `docs/STATUS.md` |\n{consistency.BEGIN}"
        self.assertIsNone(consistency.contract_paths(text))


class Mutations(unittest.TestCase):
    def test_baseline_passes(self) -> None:
        self.assertEqual(
            consistency.document_errors("doc", document(*EXPECTED), EXPECTED), []
        )

    def test_missing_block_is_rejected(self) -> None:
        errors = consistency.document_errors("doc", "# nothing", EXPECTED)
        self.assertTrue(any("missing the doc-obligation contract" in e for e in errors))

    def test_readme_in_contract_is_rejected_by_name(self) -> None:
        """The exact historical regression: README named as a checkpoint."""
        text = document("README.md", "docs/BENCHMARKS.md", "docs/FEATURES.md")
        errors = consistency.document_errors("doc", text, EXPECTED)
        self.assertTrue(any("README.md" in e and "landing page" in e for e in errors))

    def test_dropped_surface_is_rejected(self) -> None:
        text = document("docs/STATUS.md", "docs/BENCHMARKS.md")
        errors = consistency.document_errors("doc", text, EXPECTED)
        self.assertTrue(any("enforces" in e for e in errors))

    def test_reordered_surfaces_are_rejected(self) -> None:
        text = document("docs/BENCHMARKS.md", "docs/STATUS.md", "docs/FEATURES.md")
        self.assertNotEqual(
            consistency.document_errors("doc", text, EXPECTED), []
        )

    def test_extra_surface_is_rejected(self) -> None:
        text = document(*EXPECTED, "docs/USAGE.md")
        self.assertNotEqual(consistency.document_errors("doc", text, EXPECTED), [])


class LiveTree(unittest.TestCase):
    def test_expected_surfaces_come_from_the_checker(self) -> None:
        self.assertEqual(consistency.obligated_surfaces(), EXPECTED)

    def test_repository_contract_is_consistent(self) -> None:
        self.assertEqual(consistency.main(), 0)

    def test_every_contract_document_exists(self) -> None:
        for name in consistency.CONTRACT_DOCUMENTS:
            self.assertTrue((ROOT / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
