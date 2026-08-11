#!/usr/bin/env python3
"""Mutation checks for scripts/check-agent-record.py lifecycle enforcement."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


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
        # The fixture row must actually be `ACTIVE` today, or the mutation stops
        # exercising the ACTIVE branch and the test passes vacuously. It was
        # pinned to KERNEL-GDN-AOT-BF16 until the 2026-08-06 live-state audit
        # moved that row to READY, so it is now picked from the live record.
        active_id = next(
            row.item_id
            for row in self.rows
            if row.state == "ACTIVE" and row.path.name == "kernel-matrix.md"
        )
        active = with_field(self.by_id[active_id], "owner", "-")
        require(
            validate_mutation(self.rows, active),
            rf"ACTIVE row {re.escape(active_id)} has no CLAIM-\* owner",
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
            "parity-ledger.md#L284", "NOW.md#L1"
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

    def test_engine_row_ratchet_is_load_bearing(self) -> None:
        """The ENGINE_ROWS pin must catch a row appearing or vanishing.

        The constant is re-pinned by hand whenever a real row lands, so it is
        worth proving it is not decorative: a matrix carrying one row fewer than
        the pin has to be an error, in both directions. Without this, bumping
        the number to silence a failure would look exactly like bumping it for a
        new row.
        """
        clean: list[str] = []
        agent_record.check_matrices(clean)
        self.assertEqual([error for error in clean if "engine rows" in error], [])

        errors: list[str] = []
        with mock.patch.object(
            agent_record, "ENGINE_ROWS", agent_record.ENGINE_ROWS - 1
        ):
            agent_record.check_matrices(errors)

        require(errors, r"\d+ engine rows; expected \d+")

    def test_model_row_ratchet_is_load_bearing(self) -> None:
        """The MODEL row pin must catch a row appearing or vanishing.

        Mirrors the ENGINE ratchet above, for the same reason and with more
        force: the MODEL count is the one that actually moves, because every new
        architecture re-pins it by hand. Muse Glimmer took it 361 -> 362. Without
        this, bumping the number to silence a failure is indistinguishable from
        bumping it because a row really landed.
        """
        clean: list[str] = []
        agent_record.check_matrices(clean)
        self.assertEqual([error for error in clean if "MODEL rows" in error], [])

        path, expected = agent_record.MATRICES["MODEL"]
        errors: list[str] = []
        with mock.patch.dict(
            agent_record.MATRICES, {"MODEL": (path, expected - 1)}
        ):
            agent_record.check_matrices(errors)
        require(errors, r"\d+ MODEL rows; expected \d+")

        # Both directions: a pin ABOVE the tree must fail too, so a count can
        # never be inflated ahead of the rows that justify it.
        errors = []
        with mock.patch.dict(
            agent_record.MATRICES, {"MODEL": (path, expected + 1)}
        ):
            agent_record.check_matrices(errors)
        require(errors, r"\d+ MODEL rows; expected \d+")

    def test_engine_summary_rejects_stale_area_rollup(self) -> None:
        source = agent_record.ENGINE_MATRIX.read_text(encoding="utf-8")
        current = next(
            line
            for line in source.splitlines()
            if line.startswith("| Serving, API, CLI, library |")
        )
        cells = [cell.strip() for cell in current.strip().strip("|").split("|")]
        self.assertGreater(int(cells[6]), 0)
        cells[5] = str(int(cells[5]) + 1)
        cells[6] = str(int(cells[6]) - 1)
        stale = "| " + " | ".join(cells) + " |"

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            matrix = Path(temp_dir) / "engine-matrix.md"
            with mock.patch.object(agent_record, "ENGINE_MATRIX", matrix):
                matrix.write_text(source, encoding="utf-8")
                baseline_errors: list[str] = []
                baseline_rows = agent_record.parse_claim_rows(matrix, baseline_errors)
                agent_record.check_engine_summary(baseline_rows, baseline_errors)
                self.assertEqual(baseline_errors, [])

                matrix.write_text(source.replace(current, stale), encoding="utf-8")
                errors: list[str] = []
                rows = agent_record.parse_claim_rows(matrix, errors)
                agent_record.check_engine_summary(rows, errors)

        require(errors, r"Serving, API, CLI, library summary ready=\d+; actual \d+")


class MigratedLegacyLinks(unittest.TestCase):
    def test_legacy_payload_keeps_original_agents_relative_link_base(self) -> None:
        source = ROOT / ".agents/completed/state-events/0000-00/STATE-LEGACY-000001.md"
        text = "<!-- legacy-payload:begin -->\n[spec](specs/example.md)"
        self.assertEqual(agent_record.link_base(source, text), ROOT / ".agents")

    def test_post_cutover_event_links_remain_event_relative(self) -> None:
        source = ROOT / ".agents/completed/state-events/2026-08/STATE-20260808T120000-001.md"
        self.assertEqual(agent_record.link_base(source, "[local](note.md)"), source.parent)




class IssueIntakeTable(unittest.TestCase):
    """The roadmap issue table is the intake surface: no work without an issue.

    The validator is deliberately NETWORK-FREE, so these mutations are all about
    FORM and internal consistency. Whether an issue is still open is the agent's
    job at intake; making CI ask GitHub would reintroduce exactly the kind of
    connectivity flake this protocol was slimmed down to remove.
    """

    GOOD = (
        "## Open issues\n\n"
        "| Issue | Row | Title | Kind |\n"
        "|---:|---|---|---|\n"
        "| [#201](https://github.com/mudler/vllm.cpp/issues/201) | `BACKEND-ROCM` | x | bug |\n"
        "| [#85](https://github.com/mudler/vllm.cpp/issues/85) | — | y | bug |\n"
        "\n## Top-level portfolio\n"
    )

    def run_check(self, section):
        import importlib.util

        path = ROOT / ".agents/roadmap_v1.md"
        original = path.read_text(encoding="utf-8")
        try:
            path.write_text(section, encoding="utf-8")
            errors = []
            agent_record.check_issue_table(errors)
            return errors
        finally:
            path.write_text(original, encoding="utf-8")

    def test_a_well_formed_table_passes(self):
        self.assertEqual(self.run_check(self.GOOD), [])

    def test_a_missing_table_is_rejected(self):
        errors = self.run_check("# Roadmap\n\n## Top-level portfolio\n")
        self.assertTrue(any("Open issues" in e for e in errors), errors)

    def test_an_empty_table_is_rejected(self):
        section = "## Open issues\n\n| Issue | Row | Title | Kind |\n|---:|---|---|---|\n\n## Top-level portfolio\n"
        errors = self.run_check(section)
        self.assertTrue(any("no rows" in e for e in errors), errors)

    def test_a_bare_issue_number_without_a_link_is_rejected(self):
        section = self.GOOD.replace(
            "| [#85](https://github.com/mudler/vllm.cpp/issues/85) |", "| #85 |"
        )
        errors = self.run_check(section)
        self.assertTrue(any("malformed issue row" in e for e in errors), errors)

    def test_a_link_pointing_at_a_different_issue_is_rejected(self):
        """The number and its URL must agree, or the table lies."""
        section = self.GOOD.replace(
            "[#201](https://github.com/mudler/vllm.cpp/issues/201)",
            "[#201](https://github.com/mudler/vllm.cpp/issues/999)",
        )
        errors = self.run_check(section)
        self.assertTrue(any("a different issue" in e for e in errors), errors)

    def test_a_duplicated_issue_is_rejected(self):
        section = self.GOOD.replace(
            "\n## Top-level portfolio",
            "| [#201](https://github.com/mudler/vllm.cpp/issues/201) | — | dup | bug |\n"
            "\n## Top-level portfolio",
        )
        errors = self.run_check(section)
        self.assertTrue(any("listed twice" in e for e in errors), errors)

    def test_the_tracked_roadmap_table_is_valid(self):
        errors = []
        agent_record.check_issue_table(errors)
        self.assertEqual(errors, [])



class PerClaimFileSource(unittest.TestCase):
    """A claim may live in its own file under .agents/claims/ (#364).

    The claims TABLE in coordination.md is insert-at-one-anchor, so every
    concurrent claim appended a row at the same line. It conflicted in 8 of the
    16 conflicting open PRs measured at origin/main d928e2c3 -- six of them one
    author's sequential ROCm GDN stack whose ONLY conflict was this file. A
    claim in its own file has one writer and cannot collide.

    parse_active_claims reads BOTH sources, so no existing row had to be
    migrated and the SPIKE/ACTIVE cross-check is unchanged.
    """

    def test_both_sources_are_read(self) -> None:
        sources = [str(p.relative_to(ROOT)) for p in agent_record.claim_sources()]
        self.assertIn(".agents/coordination.md", sources)
        self.assertTrue(
            (ROOT / ".agents/claims").is_dir(),
            "the per-claim directory must exist for new claims to have a home",
        )

    def test_a_claim_file_registers_exactly_like_a_table_row(self) -> None:
        probe = ROOT / ".agents/claims/CLAIM-AGENT-RECORD-PROBE.md"
        probe.write_text(
            "# CLAIM-AGENT-RECORD-PROBE\n\n"
            "| Claim | Row IDs |\n|---|---|\n"
            "| `CLAIM-AGENT-RECORD-PROBE` | `ENG-RECORD-CONFLICT-SURFACES` |\n",
            encoding="utf-8",
        )
        try:
            errors: list[str] = []
            claims = agent_record.parse_active_claims(errors)
            self.assertIn("CLAIM-AGENT-RECORD-PROBE", claims)
            self.assertEqual(
                claims["CLAIM-AGENT-RECORD-PROBE"], {"ENG-RECORD-CONFLICT-SURFACES"}
            )
        finally:
            probe.unlink()

    def test_dropping_the_directory_from_the_sources_is_caught(self) -> None:
        """MUTATION: read coordination.md alone and the claim disappears.

        This is the semantic being added. Without it a claim filed in its own
        file is invisible to the gate, and the row it owns fails the
        SPIKE/ACTIVE cross-check while looking correctly claimed to a human.
        """
        probe = ROOT / ".agents/claims/CLAIM-AGENT-RECORD-PROBE.md"
        probe.write_text(
            "# CLAIM-AGENT-RECORD-PROBE\n\n"
            "| Claim | Row IDs |\n|---|---|\n"
            "| `CLAIM-AGENT-RECORD-PROBE` | `ENG-RECORD-CONFLICT-SURFACES` |\n",
            encoding="utf-8",
        )
        try:
            with mock.patch.object(
                agent_record,
                "claim_sources",
                lambda: [ROOT / ".agents/coordination.md"],
            ):
                errors: list[str] = []
                self.assertNotIn(
                    "CLAIM-AGENT-RECORD-PROBE",
                    agent_record.parse_active_claims(errors),
                    "the mutation must hide the claim, or this test proves nothing",
                )
            errors = []
            self.assertIn("CLAIM-AGENT-RECORD-PROBE", agent_record.parse_active_claims(errors))
        finally:
            probe.unlink()

    def test_a_claim_declared_twice_across_sources_is_rejected(self) -> None:
        """Reading a second source must not weaken the duplicate check."""
        probe = ROOT / ".agents/claims/CLAIM-AGENT-RECORD-DUP.md"
        row = (
            "| Claim | Row IDs |\n|---|---|\n"
            "| `CLAIM-AGENT-RECORD-DUP` | `ENG-RECORD-CONFLICT-SURFACES` |\n"
        )
        probe.write_text("# dup\n\n" + row + row, encoding="utf-8")
        try:
            errors: list[str] = []
            agent_record.parse_active_claims(errors)
            self.assertTrue(
                any("duplicate active claim" in e for e in errors),
                f"a duplicate must be caught; got {errors}",
            )
        finally:
            probe.unlink()


if __name__ == "__main__":
    unittest.main()
