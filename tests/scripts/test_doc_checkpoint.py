#!/usr/bin/env python3
"""Unit checks for scripts/check-doc-checkpoint.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-doc-checkpoint.py"
SPEC = importlib.util.spec_from_file_location("doc_checkpoint", CHECKER)
assert SPEC is not None and SPEC.loader is not None
doc_checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = doc_checkpoint
SPEC.loader.exec_module(doc_checkpoint)


class DocumentationCheckpointTests(unittest.TestCase):
    def test_production_change_requires_both_public_documents(self) -> None:
        errors = doc_checkpoint.checkpoint_errors({"src/vllm/example.cpp"})
        self.assertEqual(len(errors), 2)
        self.assertIn("README.md", errors[0])
        self.assertIn("docs/BENCHMARKS.md", errors[1])

    def test_one_public_document_does_not_satisfy_checkpoint(self) -> None:
        errors = doc_checkpoint.checkpoint_errors(
            {"tests/vllm/test_example.cpp", "README.md"}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("docs/BENCHMARKS.md", errors[0])

    def test_lifecycle_and_diagnostic_records_are_checkpoints(self) -> None:
        for path in (
            ".agents/roadmap_v1.md",
            ".agents/state.md",
            ".agents/specs/example.md",
            "tools/bench/example.py",
        ):
            with self.subTest(path=path):
                self.assertTrue(doc_checkpoint.checkpoint_errors({path}))

    def test_both_public_documents_satisfy_checkpoint(self) -> None:
        self.assertEqual(
            doc_checkpoint.checkpoint_errors(
                {
                    ".agents/engine-matrix.md",
                    "README.md",
                    "docs/BENCHMARKS.md",
                }
            ),
            [],
        )

    def test_unrelated_documentation_correction_is_not_a_checkpoint(self) -> None:
        self.assertEqual(
            doc_checkpoint.checkpoint_errors({"docs/design-notes.md"}), []
        )


if __name__ == "__main__":
    unittest.main()
