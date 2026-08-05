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
        self.assertIn("docs/STATUS.md", errors[0])
        self.assertIn("docs/BENCHMARKS.md", errors[1])

    def test_one_public_document_does_not_satisfy_checkpoint(self) -> None:
        errors = doc_checkpoint.checkpoint_errors(
            {"tests/vllm/test_example.cpp", "docs/STATUS.md"}
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
                    "docs/STATUS.md",
                    "docs/BENCHMARKS.md",
                }
            ),
            [],
        )

    def test_unrelated_documentation_correction_is_not_a_checkpoint(self) -> None:
        self.assertEqual(
            doc_checkpoint.checkpoint_errors({"docs/design-notes.md"}), []
        )


class FeatureSurfaceCheckpointTests(unittest.TestCase):
    """docs/FEATURES.md is the public mirror of the area matrices."""

    def test_feature_matrix_change_requires_the_public_feature_page(self) -> None:
        errors = doc_checkpoint.checkpoint_errors(
            {".agents/feature-matrix.md", "docs/STATUS.md", "docs/BENCHMARKS.md"}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("docs/FEATURES.md", errors[0])

    def test_every_area_matrix_triggers_the_feature_page(self) -> None:
        for path in (
            ".agents/feature-matrix.md",
            ".agents/model-matrix.md",
            ".agents/backend-matrix.md",
            ".agents/quantization-matrix.md",
            "src/vllm/model_executor/models/qwen3.cpp",
        ):
            with self.subTest(path=path):
                self.assertTrue(doc_checkpoint.is_feature_path(path))

    def test_updating_the_feature_page_satisfies_the_obligation(self) -> None:
        self.assertEqual(
            doc_checkpoint.checkpoint_errors(
                {
                    ".agents/model-matrix.md",
                    "docs/STATUS.md",
                    "docs/BENCHMARKS.md",
                    "docs/FEATURES.md",
                }
            ),
            [],
        )

    def test_non_feature_code_does_not_owe_the_feature_page(self) -> None:
        # A scheduler or kernel change is a checkpoint, but it moves no public
        # feature row, so it must not be forced to touch FEATURES.md.
        errors = doc_checkpoint.checkpoint_errors(
            {"src/vt/cuda/matmul.cu", "docs/STATUS.md", "docs/BENCHMARKS.md"}
        )
        self.assertEqual(errors, [])

    def test_feature_page_alone_is_not_a_general_checkpoint_surface(self) -> None:
        # Editing FEATURES.md must not satisfy the STATUS/BENCHMARKS obligation.
        errors = doc_checkpoint.checkpoint_errors(
            {".agents/model-matrix.md", "docs/FEATURES.md"}
        )
        self.assertTrue(any("docs/STATUS.md" in e for e in errors), errors)
        self.assertTrue(any("docs/BENCHMARKS.md" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
