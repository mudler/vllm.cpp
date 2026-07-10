#!/usr/bin/env python3
"""Mutation checks for scripts/check-agent-record.py lifecycle enforcement."""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-agent-record.py"
SPEC = importlib.util.spec_from_file_location("agent_record", CHECKER)
assert SPEC is not None and SPEC.loader is not None
agent_record = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agent_record
SPEC.loader.exec_module(agent_record)


def with_field(row, field: str, value: str):
    index = agent_record.field_index(row.header, field)
    assert index is not None, (row.item_id, field)
    cells = list(row.cells)
    cells[index] = value
    return replace(row, cells=tuple(cells), raw="| " + " | ".join(cells) + " |")


def validate_mutation(rows, changed):
    mutated = [changed if row.item_id == changed.item_id else row for row in rows]
    by_id = {row.item_id: row for row in mutated}
    errors: list[str] = []
    agent_record.check_row_contracts(mutated, by_id, errors)
    return errors


def require(errors: list[str], pattern: str) -> None:
    if not any(re.search(pattern, error) for error in errors):
        raise AssertionError(f"missing expected error {pattern!r}:\n" + "\n".join(errors))


class AgentRecordMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        parse_errors: list[str] = []
        cls.rows, _ = agent_record.check_matrices(parse_errors)
        if parse_errors:
            raise AssertionError("\n".join(parse_errors))
        cls.by_id = {row.item_id: row for row in cls.rows}

        baseline_errors: list[str] = []
        agent_record.check_row_contracts(cls.rows, cls.by_id, baseline_errors)
        if baseline_errors:
            raise AssertionError("\n".join(baseline_errors))

    def test_ready_requires_real_spec(self) -> None:
        ready = with_field(
            self.by_id["PAR-TP"], "spec", "`planned: specs/tensor-parallelism.md`"
        )
        require(
            validate_mutation(self.rows, ready),
            r"PAR-TP READY has no real .agents/specs link",
        )

    def test_ready_spec_names_stable_row(self) -> None:
        wrong_spec = with_field(
            self.by_id["PAR-TP"], "spec", "[wrong spike](specs/mtp-spec-decode.md)"
        )
        require(
            validate_mutation(self.rows, wrong_spec),
            r"no linked spec names exact stable token `PAR-TP`",
        )

    def test_prose_keyword_bag_is_not_a_spike_contract(self) -> None:
        prose = (
            "`PAR-TP` scope upstream chain our baseline port map tests to port "
            "gates dependencies work breakdown risks and decisions"
        )
        missing = agent_record.missing_spec_requirements(prose)
        self.assertEqual(set(missing), set(agent_record.SPEC_REQUIREMENTS))

    def test_empty_structured_sections_are_not_a_spike_contract(self) -> None:
        empty_headings = "`PAR-TP`\n" + "\n".join(
            f"## {label}" for label in agent_record.SPEC_REQUIREMENTS
        )
        self.assertEqual(
            set(agent_record.missing_spec_requirements(empty_headings)),
            set(agent_record.SPEC_REQUIREMENTS),
        )
        empty_rows = "\n".join(
            f"| {label} | - |" for label in agent_record.SPEC_REQUIREMENTS
        )
        self.assertEqual(
            set(agent_record.missing_spec_requirements(empty_rows)),
            set(agent_record.SPEC_REQUIREMENTS),
        )
        placeholder_tables = "`PAR-TP`\n" + "\n".join(
            f"## {label}\n| Key | Value |\n|---|---|\n| thing | - |"
            for label in agent_record.SPEC_REQUIREMENTS
        )
        self.assertEqual(
            set(agent_record.missing_spec_requirements(placeholder_tables)),
            set(agent_record.SPEC_REQUIREMENTS),
        )

    def test_active_requires_claim_owner(self) -> None:
        active = with_field(self.by_id["KERNEL-GDN-AOT-BF16"], "owner", "-")
        require(
            validate_mutation(self.rows, active),
            r"ACTIVE row KERNEL-GDN-AOT-BF16 has no CLAIM-\* owner",
        )

    def test_implemented_state_requires_exact_code_anchor(self) -> None:
        partial = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "code",
            "implementation exists somewhere",
        )
        require(
            validate_mutation(self.rows, partial),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact local code anchor",
        )

    def test_unrelated_local_link_is_not_code_or_test_evidence(self) -> None:
        bad_code = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "code",
            "[not evidence](roadmap_v1.md#L1)",
        )
        require(
            validate_mutation(self.rows, bad_code),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact local code anchor",
        )

        bad_test = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "tests",
            "[not evidence](roadmap_v1.md#L1)",
        )
        require(
            validate_mutation(self.rows, bad_test),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact test/evidence anchor",
        )

    def test_out_of_range_line_is_not_an_anchor(self) -> None:
        partial = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "code",
            "[bad line](../src/vt/cuda/cuda_moe.cu#L999999), "
            "[test](../tests/vt/test_ops_moe_grouped.cpp#L453)",
        )
        require(
            validate_mutation(self.rows, partial),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact local code anchor",
        )

    def test_out_of_range_raw_range_is_not_an_anchor(self) -> None:
        partial = with_field(
            self.by_id["KERNEL-MOE-UNQUANTIZED"],
            "code",
            "`src/vt/cuda/cuda_moe.cu:349-999999`",
        )
        require(
            validate_mutation(self.rows, partial),
            r"KERNEL-MOE-UNQUANTIZED PARTIAL lacks exact local code anchor",
        )

    def test_done_requires_exact_ledger_link(self) -> None:
        done = self.by_id["QUANT-NVFP4-MO-W4A16"]
        evidence = done.field("tests").replace(
            "parity-ledger.md#L284", "state.md#L1"
        )
        done = with_field(done, "tests", evidence)
        require(
            validate_mutation(self.rows, done),
            r"DONE row QUANT-NVFP4-MO-W4A16 lacks exact parity-ledger link",
        )

    def test_done_requires_closing_commit_in_owner(self) -> None:
        done = with_field(self.by_id["QUANT-NVFP4-MO-W4A16"], "owner", "-")
        require(
            validate_mutation(self.rows, done),
            r"DONE row QUANT-NVFP4-MO-W4A16 owner is not the hexadecimal closing commit",
        )

    def test_done_closing_commit_must_exist(self) -> None:
        done = with_field(
            self.by_id["QUANT-NVFP4-MO-W4A16"], "owner", "deadbee"
        )
        require(
            validate_mutation(self.rows, done),
            r"DONE row QUANT-NVFP4-MO-W4A16 closing commit deadbee does not exist",
        )

    def test_tables_require_semantic_owner_column(self) -> None:
        factory = self.by_id["MODEL-FACTORY-registry"]
        owner_index = agent_record.field_index(factory.header, "owner")
        if owner_index is None:
            raise AssertionError("baseline MODEL-FACTORY-registry has no owner column")
        malformed = replace(
            factory,
            header=factory.header[:owner_index] + factory.header[owner_index + 1 :],
            cells=factory.cells[:owner_index] + factory.cells[owner_index + 1 :],
        )
        require(
            validate_mutation(self.rows, malformed),
            r"MODEL-FACTORY-registry table lacks semantic owner column",
        )


if __name__ == "__main__":
    unittest.main()
