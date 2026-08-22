#!/usr/bin/env python3
"""Mutation tests for the mechanical public benchmark index."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_index", ROOT / "scripts/check-benchmark-index.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)

VALID = """# Benchmarks

| Benchmark ID | Subject | Disposition | Detail |
|---|---|---|---|
| `alpha` | Alpha | Measured | [Details](benchmarks/alpha.md) |
"""


class BenchmarkIndexTests(unittest.TestCase):
    def errors(self, text: str, files: set[str] | None = None) -> list[str]:
        return checker.benchmark_index_errors(
            text, {"alpha.md"} if files is None else files
        )

    def test_valid_index_passes(self) -> None:
        self.assertEqual(self.errors(VALID), [])

    def test_duplicate_id_fails(self) -> None:
        duplicate = VALID + "| `alpha` | Again | Measured | [Details](benchmarks/alpha.md) |\n"
        self.assertTrue(any("duplicate" in error for error in self.errors(duplicate)))

    def test_missing_detail_file_fails(self) -> None:
        self.assertTrue(any("missing detail" in error for error in self.errors(VALID, set())))

    def test_orphan_detail_file_fails(self) -> None:
        self.assertTrue(any("orphan" in error for error in self.errors(VALID, {"alpha.md", "beta.md"})))

    def test_id_must_match_file_stem(self) -> None:
        mismatch = VALID.replace("benchmarks/alpha.md", "benchmarks/beta.md")
        errors = self.errors(mismatch, {"beta.md"})
        self.assertTrue(any("does not match" in error for error in errors), errors)

    def test_shipped_index_passes(self) -> None:
        self.assertEqual(checker.validate_root(ROOT), [])


if __name__ == "__main__":
    unittest.main()
