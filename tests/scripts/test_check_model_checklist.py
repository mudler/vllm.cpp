#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-model-checklist.py.

Mirrors tests/scripts/test_check_readme_structure.py: a small self-consistent
matrix is the mutation baseline, and each mutation proves the checker has teeth
(catches a mark the row does not back, a drifted rollup count, and an omitted
engaged architecture). The shipped matrix is also asserted to pass.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-model-checklist.py"
SPEC = importlib.util.spec_from_file_location("check_model_checklist", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


# A minimal self-consistent matrix: two detailed rows in a claim table (ID+State
# header) plus the checklist section. One ACTIVE row (engaged, ✅), one
# INVENTORIED row (rollup tail only), one SPIKE row (engaged, 📋).
VALID = "\n".join(
    [
        "# Model parity matrix",
        "",
        "## Architecture-support checklist",
        "",
        "Intro prose.",
        "",
        "| State | Rows |",
        "|---|---|",
        "| ACTIVE | 1 |",
        "| SPIKE | 1 |",
        "| INVENTORIED | 1 |",
        "| **Total** | **3** |",
        "",
        "| Support | Architecture | Family | Status | Row |",
        "|---|---|---|---|---|",
        "| ✅ | `FooForCausalLM` | Foo | token-exact 6/6; speed pending | `MODEL-TEXT-foo` |",
        "| 📋 | `BarForCausalLM` | Bar | scoped, not implemented | `MODEL-TEXT-bar` |",
        "",
        "## MODEL-TEXT",
        "",
        "| ID | Item | Upstream | Task | Deps | Spike/spec | State | Evidence | Owner |",
        "|---|---|---|---|---|---|---|---|---|",
        "| `MODEL-TEXT-foo` | `FooForCausalLM` | up | t | d | s | `ACTIVE` | e | o |",
        "| `MODEL-TEXT-bar` | `BarForCausalLM` | up | t | d | s | `SPIKE` | e | o |",
        "| `MODEL-TEXT-baz` | `BazForCausalLM` | up | t | d | s | `INVENTORIED` | e | o |",
        "",
    ]
)


class ModelChecklistTests(unittest.TestCase):
    def test_minimal_valid_matrix_passes(self) -> None:
        self.assertEqual(mod.checklist_errors(VALID), [])

    def test_shipped_matrix_passes(self) -> None:
        text = (ROOT / ".agents/model-matrix.md").read_text(encoding="utf-8")
        self.assertEqual(mod.checklist_errors(text), [])

    def test_check_on_a_spike_row_fails(self) -> None:
        # Mark the SPIKE row ✅ (correctness-complete). SPIKE does not back ✅.
        mutated = VALID.replace(
            "| 📋 | `BarForCausalLM` | Bar | scoped, not implemented | `MODEL-TEXT-bar` |",
            "| ✅ | `BarForCausalLM` | Bar | scoped, not implemented | `MODEL-TEXT-bar` |",
        )
        errors = mod.checklist_errors(mutated)
        self.assertTrue(
            any("MODEL-TEXT-bar" in e and "back" in e for e in errors),
            errors,
        )

    def test_check_on_a_blocked_row_via_wrong_mark_fails(self) -> None:
        # 🚫 (blocked) on an ACTIVE row: ACTIVE does not back 🚫.
        mutated = VALID.replace(
            "| ✅ | `FooForCausalLM` | Foo | token-exact 6/6; speed pending | `MODEL-TEXT-foo` |",
            "| 🚫 | `FooForCausalLM` | Foo | token-exact 6/6; speed pending | `MODEL-TEXT-foo` |",
        )
        errors = mod.checklist_errors(mutated)
        self.assertTrue(any("MODEL-TEXT-foo" in e and "back" in e for e in errors), errors)

    def test_wrong_rollup_count_fails(self) -> None:
        mutated = VALID.replace("| ACTIVE | 1 |", "| ACTIVE | 2 |")
        errors = mod.checklist_errors(mutated)
        self.assertTrue(any("ACTIVE" in e and "matrix has 1" in e for e in errors), errors)

    def test_wrong_total_fails(self) -> None:
        mutated = VALID.replace("| **Total** | **3** |", "| **Total** | **4** |")
        errors = mod.checklist_errors(mutated)
        self.assertTrue(any("Total" in e for e in errors), errors)

    def test_omitted_engaged_architecture_fails(self) -> None:
        # Drop the ✅ entry for the ACTIVE row: an engaged row is now uncovered.
        mutated = VALID.replace(
            "| ✅ | `FooForCausalLM` | Foo | token-exact 6/6; speed pending | `MODEL-TEXT-foo` |\n",
            "",
        )
        errors = mod.checklist_errors(mutated)
        self.assertTrue(
            any("MODEL-TEXT-foo" in e and "missing" in e for e in errors), errors
        )

    def test_inventoried_row_listed_individually_fails(self) -> None:
        mutated = VALID.replace(
            "| 📋 | `BarForCausalLM` | Bar | scoped, not implemented | `MODEL-TEXT-bar` |\n",
            "| 📋 | `BarForCausalLM` | Bar | scoped, not implemented | `MODEL-TEXT-bar` |\n"
            "| ✅ | `BazForCausalLM` | Baz | listed by mistake | `MODEL-TEXT-baz` |\n",
        )
        errors = mod.checklist_errors(mutated)
        self.assertTrue(any("MODEL-TEXT-baz" in e for e in errors), errors)

    def test_unknown_row_reference_fails(self) -> None:
        mutated = VALID.replace(
            "speed pending | `MODEL-TEXT-foo` |",
            "speed pending | `MODEL-TEXT-ghost` |",
        )
        errors = mod.checklist_errors(mutated)
        self.assertTrue(any("MODEL-TEXT-ghost" in e for e in errors), errors)

    def test_legit_new_engaged_row_added_correctly_passes(self) -> None:
        # Add a new ACTIVE row + its ✅ entry + bump the rollup: must pass clean.
        mutated = (
            VALID.replace(
                "| ACTIVE | 1 |",
                "| ACTIVE | 2 |",
            )
            .replace("| **Total** | **3** |", "| **Total** | **4** |")
            .replace(
                "| 📋 | `BarForCausalLM` | Bar | scoped, not implemented | `MODEL-TEXT-bar` |\n",
                "| 📋 | `BarForCausalLM` | Bar | scoped, not implemented | `MODEL-TEXT-bar` |\n"
                "| ✅ | `NewForCausalLM` | New | token-exact 8/8 | `MODEL-TEXT-new` |\n",
            )
            .replace(
                "| `MODEL-TEXT-baz` | `BazForCausalLM` | up | t | d | s | `INVENTORIED` | e | o |",
                "| `MODEL-TEXT-new` | `NewForCausalLM` | up | t | d | s | `ACTIVE` | e | o |\n"
                "| `MODEL-TEXT-baz` | `BazForCausalLM` | up | t | d | s | `INVENTORIED` | e | o |",
            )
        )
        self.assertEqual(mod.checklist_errors(mutated), [])


if __name__ == "__main__":
    unittest.main()
