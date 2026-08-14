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

    def test_windows_release_row_is_inside_the_engine_ratchet(self) -> None:
        """The #117 row and its ratchet bump are one semantic change."""

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual([error for error in errors if "engine rows" in error], [])
        windows = [row for row in rows if row.item_id == "ENG-RELEASE-WINDOWS"]
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].path.name, "engine-matrix.md")

    def test_serve_recipe_args_row_is_inside_the_engine_ratchet(self) -> None:
        """The #606 row and its 152 -> 153 ratchet bump are one semantic change.

        Same shape as the #117 assertion above, and it exists for the same
        reason: the bump and the row have to arrive together, or a number was
        moved to silence a failure. This one also pins WHICH matrix owns the
        row, because SERVE-* IDs are reachable from more than one, and a row
        that drifted into another matrix would leave the engine count short
        while the pin still read 153.
        """

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual([error for error in errors if "engine rows" in error], [])
        recipe = [row for row in rows if row.item_id == "SERVE-RECIPE-ARGS"]
        self.assertEqual(len(recipe), 1)
        self.assertEqual(recipe[0].path.name, "engine-matrix.md")

    def test_omni_pin_row_is_inside_the_engine_ratchet(self) -> None:
        """The #633 row and its 153 -> 154 ratchet bump are one semantic change.

        Same shape as the #117 and #606 assertions above, and it carries one
        extra hazard worth pinning. This bump COLLIDED: `main` took the constant
        152 -> 153 for `SERVE-RECIPE-ARGS` while the omni-pin branch took the
        same 152 -> 153 for its own row, so both sides read 153 and the merge
        looked clean. Resolving it by keeping either 153 would have dropped a
        real row while leaving the matrix internally consistent, which is
        exactly the state no other assertion here can see. Naming BOTH rows is
        what makes 154 checkable rather than plausible.
        """

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual([error for error in errors if "engine rows" in error], [])

        for item_id in ("ENG-UPSTREAM-OMNI-PIN", "SERVE-RECIPE-ARGS"):
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "engine-matrix.md", item_id)

    def test_music3_and_indextts_rows_both_survive_their_collision(self) -> None:
        """373 needs BOTH rows named, because the merge that produced it collided.

        The same hazard the omni-pin assertion above records, on the MODEL pin
        and on the same day. `main` took the constant 370 -> 372 for IndexTTS-2.5
        (two architectures) while the Music3 branch took 370 -> 371 for its own
        row. Neither side was wrong about its own change, and neither number was
        373 -- so whichever side an auto-merge kept, the tree would have been
        internally consistent while silently short a real architecture.

        A count assertion alone cannot see that: it only knows the pin matches
        the rows it can find. Naming the three rows is what makes 373 checkable
        rather than plausible.
        """

        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual([error for error in errors if "MODEL rows" in error], [])

        collided = (
            "MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation",
            "MODEL-MM-indextts2-index-tts2-talker-for-conditional-generation",
            "MODEL-MM-indextts2-index-tts2-s2-mel-decoder",
        )
        for item_id in collided:
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "model-matrix.md", item_id)

    def test_model_row_ratchet_is_load_bearing(self) -> None:
        """The MODEL row pin must catch a row appearing or vanishing.

        Mirrors the ENGINE ratchet above, for the same reason and with more
        force: the MODEL count is the one that actually moves, because every new
        architecture re-pins it by hand. Muse Glimmer took it 361 -> 362
        (`c8fc24a50`); the seven recipe architectures that had no row at all took
        it 362 -> 369 (#609, #610, `eba6ab7c7`); LTX-2.5 took it 369 -> 370
        (#435, `cefacd2d0`); IndexTTS-2.5 took it 370 -> 372, being two
        architectures (#634); MiniMax-Music3 took it to 373 (#672); and the two
        text-only Qwen3.5 arms took it 373 -> 375 (#490). Without this,
        bumping the number to silence a failure is indistinguishable from bumping
        it because a row really landed.
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

    def test_indextts_rows_are_inside_the_model_ratchet(self) -> None:
        """The #634 rows and the 370 -> 372 bump are one semantic change.

        IndexTTS-2.5 is registered by vLLM-Omni as TWO architectures, a talker
        and an S2Mel decoder, so it moves the pin by two rather than one. That
        is the hazard worth pinning: a port described in prose as "a model" is
        the shape that lands one row and a bump of two, and the count alone
        cannot tell that from two rows landing. Both are named here, and both
        are asserted `INVENTORIED` rather than `SPIKE` — they are unclaimed and
        blocked on #633, and `SPIKE` would owe a `CLAIM-*` owner they do not
        have.
        """
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual([error for error in errors if "MODEL rows" in error], [])

        for item_id in (
            "MODEL-MM-indextts2-index-tts2-talker-for-conditional-generation",
            "MODEL-MM-indextts2-index-tts2-s2-mel-decoder",
        ):
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "model-matrix.md", item_id)
            self.assertEqual(
                found[0].field("state").strip().strip("`"), "INVENTORIED", item_id
            )

    def test_dots3_rows_are_inside_the_model_ratchet(self) -> None:
        """The #699 rows and the 373 -> 375 bump are one semantic change.

        dots3-note is the IndexTTS-2.5 shape again on a different lane: vLLM
        registers it as TWO architectures, `Dots3NoteForCausalLM` and its
        speculative head `Dots3NoteMTPModel`, so a port described in prose as
        "a model" moves the pin by two. Naming both is what makes 375 checkable
        rather than plausible.

        What this catches that nothing else does, measured: RENAMING the MTP row
        leaves the count at 375, touches no claim, and every other check stays
        green -- only this assertion goes red. That is the whole point of naming
        rows rather than counting them.

        The state assertions are deliberately weaker evidence, and the record
        says so rather than implying otherwise: mutating either row's lifecycle
        is already caught upstream of here by the claim-ownership and
        spec-structure rules (INVENTORIED -> SPIKE trips "SPIKE row has no
        CLAIM-* owner"; SPIKE -> ACTIVE trips the structured-spec requirement).
        They are pinned anyway because the asymmetry is intentional -- the
        target row is `SPIKE` with a committed spec and an owner, the MTP row is
        `INVENTORIED` because it is unclaimed and blocked behind the target's
        oracle and hardware gaps -- and a future refactor of those rules should
        not silently take the pin with it.
        """
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual([error for error in errors if "MODEL rows" in error], [])

        for item_id, state in (
            ("MODEL-MM-dots3-note-dots3-note-for-causal-lm", "SPIKE"),
            ("MODEL-SPEC-dots3-note-dots3-note-mtp", "INVENTORIED"),
        ):
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "model-matrix.md", item_id)
            self.assertEqual(found[0].field("state").strip().strip("`"), state, item_id)

    def test_recipe_backfill_rows_are_inside_the_model_ratchet(self) -> None:
        """The #609/#610 rows and the 362 -> 369 bump are one semantic change.

        Mirrors `test_windows_release_row_is_inside_the_engine_ratchet`: name
        the rows the bump was taken FOR, so a count raised to silence a broken
        parse cannot look identical to a count raised because rows landed. Two
        of the seven are pinned, one per issue; seven near-identical assertions
        would add repetition, not force.
        """
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        self.assertEqual([error for error in errors if "MODEL rows" in error], [])

        for item_id in (
            "MODEL-TEXT-bailing-moe-v3-bailing-moe-v3-for-causal-lm",
            "MODEL-MM-moss-tts-moss-tts-delay-talker-for-generation",
        ):
            found = [row for row in rows if row.item_id == item_id]
            self.assertEqual(len(found), 1, item_id)
            self.assertEqual(found[0].path.name, "model-matrix.md")
            self.assertEqual(
                found[0].field("state").strip().strip("`"), "INVENTORIED", item_id
            )

    def test_beyond_pin_rows_stay_out_of_the_at_pin_model_inventory(self) -> None:
        """A beyond-pin row must not inflate the AT-THE-PIN model inventory.

        `check_model_invariants` pins what vLLM's OWN registry holds at
        `555967922`: 324 category/target rows, 373 memberships, 356
        architectures, 310 targets, 261 modules. It counts a row only when the
        Upstream cell carries a backticked `module`-colon-colon-`class` token.
        So a row for an architecture that is NOT at the pin -- MuseGlimmer,
        KimiK3, MiniMaxH3DiT, and the seven recipe architectures rowed for
        #609/#610 -- deliberately spells its module and class as separate
        fields instead, and contributes nothing.

        That is a convention, and an unenforced convention drifts. The moment
        someone "helpfully" anchors one of those rows the pinned inventory
        silently gains a target vLLM does not register at the pin, and the next
        person to re-pin the counts bakes the error in. This proves the
        omission is load-bearing rather than an oversight.
        """
        clean: list[str] = []
        agent_record.check_model_invariants(clean)
        self.assertEqual(clean, [])

        beyond_pin = (
            "MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation",
            "MODEL-MM-kimi-k3-kimi-k3-for-conditional-generation",
            "MODEL-DIFFUSION-minimax-h3-mini-max-h3-dit",
            "MODEL-TEXT-bailing-moe-v3-bailing-moe-v3-for-causal-lm",
            "MODEL-MM-moss-tts-moss-tts-delay-talker-for-generation",
            "MODEL-MM-moss-tts-moss-tts-realtime-talker-for-generation",
            "MODEL-MM-qwen3-tts-qwen3-tts-talker-for-conditional-generation",
            "MODEL-MM-higgs-audio-v3-higgs-audio-v3-talker-for-conditional-generation",
            "MODEL-MM-voxtral-realtime-voxtral-realtime-for-conditional-generation",
            "MODEL-MM-bailing-mm-native-bailing-mm-native-for-conditional-generation",
        )
        matrix = agent_record.AGENTS / "model-matrix.md"
        lines = matrix.read_text(encoding="utf-8").splitlines(keepends=True)
        for item_id in beyond_pin:
            rows = [line for line in lines if line.startswith(f"| `{item_id}` |")]
            self.assertEqual(len(rows), 1, item_id)
            upstream = agent_record.split_cells(rows[0])[2]
            self.assertEqual(
                [v for v in re.findall(r"`([^`]+)`", upstream) if "::" in v],
                [],
                item_id,
            )

        # Anchoring exactly ONE of them the at-the-pin way must move the
        # inventory off its pin, in the file rather than in a stub.
        victim = "MODEL-DIFFUSION-minimax-h3-mini-max-h3-dit"
        mutated: list[str] = []
        for line in lines:
            if line.startswith(f"| `{victim}` |"):
                cells = line.split("|")
                cells[3] += " `vllm/model_executor/models/minimax_h3.py::MiniMaxH3DiTModel` "
                line = "|".join(cells)
            mutated.append(line)
        self.assertNotEqual(mutated, lines)

        errors: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            agents = Path(tmp)
            (agents / "model-matrix.md").write_text("".join(mutated), encoding="utf-8")
            # ROOT moves with AGENTS: the checker reports the path relative to
            # it, so leaving ROOT pointing at the real tree raises instead of
            # producing the error we are asserting on.
            with (
                mock.patch.object(agent_record, "AGENTS", agents),
                mock.patch.object(agent_record, "ROOT", agents),
            ):
                agent_record.check_model_invariants(errors)

        require(errors, r"model inventory .*expected")

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
        self.assertEqual(agent_record.link_bases(source, text), (ROOT / ".agents",))

    def test_post_cutover_event_links_remain_event_relative(self) -> None:
        source = ROOT / ".agents/completed/state-events/2026-08/STATE-20260808T120000-001.md"
        self.assertEqual(
            agent_record.link_bases(source, "[local](note.md)"), (source.parent,)
        )


class LinkExtraction(unittest.TestCase):
    """#460: what the checker calls a link must be a link a READER can follow.

    docs/BENCHMARKS.md is compacted by MOVING a superseded row into
    .agents/benchmark-record.md byte-for-byte. Before this, any row carrying a
    docs/-relative evidence link dangled the moment it was archived, whether it
    was quoted inside a fence or moved as live markdown, so the documented
    payment mechanism did not work for exactly the rows that carry evidence.
    """

    def test_fenced_link_is_not_extracted(self) -> None:
        text = "```text\n| row | [evidence](bench-evidence/x.md) |\n```\n"
        self.assertEqual(agent_record.extract_links(text), [])

    def test_tilde_fenced_link_is_not_extracted(self) -> None:
        text = "~~~console\n$ see [evidence](bench-evidence/x.md)\n~~~\n"
        self.assertEqual(agent_record.extract_links(text), [])

    def test_inline_code_link_is_not_extracted(self) -> None:
        self.assertEqual(
            agent_record.extract_links("write `[label](target.md)` to link\n"), []
        )

    def test_live_link_is_still_extracted(self) -> None:
        self.assertEqual(
            agent_record.extract_links("see [spec](specs/example.md) now\n"),
            ["specs/example.md"],
        )

    def test_a_backticked_label_is_still_a_link(self) -> None:
        # The overwhelmingly common form in this tree: [`name`](path).
        self.assertEqual(
            agent_record.extract_links("[`workflow.md`](workflow.md)\n"),
            ["workflow.md"],
        )

    def test_link_after_a_closed_fence_is_still_extracted(self) -> None:
        text = "```sh\nrun [x](nope.md)\n```\n\nreal [spec](specs/example.md)\n"
        self.assertEqual(agent_record.extract_links(text), ["specs/example.md"])

    def test_link_beside_an_inline_span_is_still_extracted(self) -> None:
        text = "`VT_FLAG=1` and [spec](specs/example.md)\n"
        self.assertEqual(agent_record.extract_links(text), ["specs/example.md"])

    def test_a_fence_with_an_INFO_STRING_does_not_close_a_block(self) -> None:
        """The F2 defect, minimised.

        CommonMark: a closing fence carries no info string. Treating ```sh as a
        close INVERTS fence phase for the rest of the file, so prose after the
        next real fence gets blanked and text inside a block gets validated.
        The two assertions are the two halves of the inversion. First: with no
        real close, everything after ```sh is still inside. Second, and this is
        the half that bit the tree: the BARE fence is the close, so the link
        below it is live and the one above it is not. Under the loose rule both
        answers were exactly backwards.
        """
        self.assertEqual(
            agent_record.extract_links("```\nopen\n```sh\ninside [a](nope-a.md)\n"),
            [],
        )
        self.assertEqual(
            agent_record.extract_links(
                "```\nopen\n```sh\ninside [a](nope-a.md)\n```\n"
                "after [b](specs/example.md)\n"
            ),
            ["specs/example.md"],
        )

    def test_a_closing_fence_must_match_the_opener(self) -> None:
        # Wrong character, then too short: neither closes, so the link after it
        # is still inside the block.
        for closer in ("~~~", "``"):
            with self.subTest(closer=closer):
                text = f"````\ncode\n{closer}\n[x](nope.md)\n"
                self.assertEqual(agent_record.extract_links(text), [])

    def test_a_LONGER_closing_fence_does_close(self) -> None:
        text = "```\ncode\n`````\n\nreal [spec](specs/example.md)\n"
        self.assertEqual(agent_record.extract_links(text), ["specs/example.md"])

    def test_prose_two_lines_below_a_closed_fence_is_still_scanned(self) -> None:
        """The live case, in the file that actually mis-paired.

        .agents/completed/state-events/0000-00/STATE-LEGACY-000001.md has an
        unclosed fence at :17697. Under the loose rule the ```sh at :17948
        "closed" it, phase inverted, and the ordinary prose link at :18297 was
        blanked: a reader-followable link silently stopped being validated.
        """
        source = ROOT / ".agents/completed/state-events/0000-00/STATE-LEGACY-000001.md"
        targets = agent_record.extract_links(source.read_text(encoding="utf-8"))
        self.assertIn("specs/cpu-llamacpp-floor-remeasure-2026-07-22.md", targets)

    def test_a_link_straddled_by_two_INLINE_SPANS_is_not_extracted(self) -> None:
        """F5: `[`name`](path)` is four backticks, and not a link.

        CommonMark reads it as the code span `[`, the literal text name, and the
        code span `](path)`, so there is no link and the checker agrees. It
        earns a test because the effect is to HIDE a target that does not exist:
        this row's own spec carried the form and hid `path`. An author who wants
        to SHOW the form writes it with a double-backtick delimiter, which is
        also code and also correctly skipped; an author who wants a real link
        writes it without the outer pair, which still resolves.
        """
        self.assertEqual(agent_record.extract_links("`[`name`](path)`\n"), [])
        self.assertEqual(agent_record.extract_links("`` [`name`](path) ``\n"), [])
        self.assertEqual(
            agent_record.extract_links("[`name`](specs/example.md)\n"),
            ["specs/example.md"],
        )

    def test_stripping_preserves_line_and_column_positions(self) -> None:
        # Blanked, not deleted, so every line and column offset survives. NOT
        # evidence of anything today: check_links reports no line numbers at
        # all. Held so a caller that does report them cannot be broken here.
        text = "a\n```\nbbbb\n```\nc `dd` e\n"
        stripped = agent_record.strip_code_spans(text)
        self.assertEqual(len(stripped.splitlines()), len(text.splitlines()))
        for original, blanked in zip(text.splitlines(), stripped.splitlines()):
            self.assertEqual(len(original), len(blanked))

    def test_the_benchmark_record_also_resolves_from_docs(self) -> None:
        # It is the declared archive of docs/BENCHMARKS.md, so a row moved into
        # it verbatim keeps its docs/-relative evidence link resolvable.
        source = ROOT / ".agents/benchmark-record.md"
        self.assertEqual(
            agent_record.link_bases(source, ""), (source.parent, ROOT / "docs")
        )

    def test_an_archived_row_with_a_docs_relative_link_is_accepted(self) -> None:
        record = ROOT / ".agents/benchmark-record.md"
        original = record.read_text(encoding="utf-8")
        moved = (
            "\n## Assembly vs compiler SDOT\n\n| Result | Evidence |\n|---|---|\n"
            "| leaf wall | [assembly evidence]"
            "(bench-evidence/rpi5-a76-q8-dot-20260806.md) |\n"
        )
        errors: list[str] = []
        try:
            record.write_text(original + moved, encoding="utf-8")
            agent_record.check_links(errors)
        finally:
            record.write_text(original, encoding="utf-8")
        self.assertEqual(
            [e for e in errors if "rpi5-a76-q8-dot" in e], [], errors[:5]
        )

    def test_an_archived_row_with_a_MISSING_link_still_dangles(self) -> None:
        # The second base is a base, not an amnesty.
        record = ROOT / ".agents/benchmark-record.md"
        original = record.read_text(encoding="utf-8")
        errors: list[str] = []
        try:
            record.write_text(
                original + "\n[gone](bench-evidence/no-such-file-20260812.md)\n",
                encoding="utf-8",
            )
            agent_record.check_links(errors)
        finally:
            record.write_text(original, encoding="utf-8")
        require(errors, r"dangling link bench-evidence/no-such-file-20260812\.md")

    def test_the_tree_has_no_dangling_link(self) -> None:
        errors: list[str] = []
        agent_record.check_links(errors)
        self.assertEqual(errors, [])



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


class TenstorrentResidualGoldenRowIsCounted(unittest.TestCase):
    """The BACKEND ratchet bump is backed by a real row (#393).

    The count is re-pinned by hand, so a bump with no row behind it looks
    exactly like a bump for a new row. This ties this bump to this row.
    """

    ROW = "BACKEND-TENSTORRENT-RESIDUAL-GOLDEN"

    def test_the_row_exists_in_the_backend_matrix(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_backend_pin_is_load_bearing(self) -> None:
        """MUTATION: moving the pin by one must make the count disagree."""
        clean: list[str] = []
        agent_record.check_matrices(clean)
        self.assertEqual([e for e in clean if "backend rows" in e.lower()], [])

        path, count = agent_record.MATRICES["BACKEND"]
        errors: list[str] = []
        with mock.patch.dict(
            agent_record.MATRICES, {"BACKEND": (path, count - 1)}
        ):
            agent_record.check_matrices(errors)
        self.assertTrue(
            any("backend rows" in e.lower() for e in errors),
            f"the BACKEND pin must bind; got {errors}",
        )


class Qwen35TextOnlyRowsAreCounted(unittest.TestCase):
    """The MODEL ratchet bump 373 -> 375 is backed by two real rows (#490).

    Same shape, and the same reason, as the BACKEND class above: the count is
    re-pinned by hand, so a bump with nothing behind it is indistinguishable
    from a bump for rows that really landed. `test_model_row_ratchet_is_
    load_bearing` proves the pin BINDS by moving it, which holds for any value
    of the pin; it cannot say whether THIS value is the right one. These two
    tests do, by tying the pin to the rows the matrix actually carries.
    """

    ROWS = (
        "MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm",
        "MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm",
    )

    def test_both_text_only_rows_exist_in_the_model_matrix(self) -> None:
        lines = (
            (ROOT / ".agents/model-matrix.md")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for row in self.ROWS:
            matching = [line for line in lines if line.startswith(f"| `{row}` |")]
            self.assertEqual(len(matching), 1, f"{row} must appear exactly once")

    def test_the_model_pin_equals_the_rows_the_matrix_carries(self) -> None:
        """MUTATION: the pin and the tree disagreeing by one row must be RED.

        Counted the way `check_matrices` counts, so a pin left behind by a
        landing row -- or moved ahead of one -- fails here and not only inside
        the checker's own error list.
        """
        path, expected = agent_record.MATRICES["MODEL"]
        errors: list[str] = []
        rows, _ = agent_record.check_matrices(errors)
        actual = sum(
            row.item_id.startswith("MODEL-") for row in rows if row.path == path
        )
        self.assertEqual(
            actual,
            expected,
            "the MODEL pin must equal the MODEL rows model-matrix.md carries",
        )
        self.assertEqual([error for error in errors if "MODEL rows" in error], [])


class TenstorrentMistralRowIsCounted(unittest.TestCase):
    """The BACKEND ratchet bump to 82 is backed by a real row (#670).

    Same shape as TenstorrentResidualGoldenRowIsCounted and for the same
    reason: the count is re-pinned by hand, so a bump with no row behind it is
    indistinguishable from a bump for a new row. `b55f6ec14` set the precedent
    that a ratchet bump lands with a case keyed to ITS OWN row; this is that
    case for BACKEND-TENSTORRENT-MISTRAL.
    """

    ROW = "BACKEND-TENSTORRENT-MISTRAL"

    def test_the_row_exists_in_the_backend_matrix(self) -> None:
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        matching = [
            line for line in text.splitlines() if line.startswith(f"| `{self.ROW}` |")
        ]
        self.assertEqual(len(matching), 1, f"{self.ROW} must appear exactly once")

    def test_the_row_names_its_issue_and_its_spec(self) -> None:
        """A row whose issue is only in the PR body is untraceable from the tree.

        This row shipped originally citing PR #354 -- a merged PR, not an issue
        -- so nothing in the repository pointed at anything trackable. Pin both
        links here so a future edit cannot quietly drop them again.
        """
        text = (ROOT / ".agents/backend-matrix.md").read_text(encoding="utf-8")
        row = next(l for l in text.splitlines() if l.startswith(f"| `{self.ROW}` |"))
        self.assertIn("tenstorrent-mistral.md", row)
        roadmap = (ROOT / ".agents/roadmap_v1.md").read_text(encoding="utf-8")
        self.assertIn("issues/670", roadmap)

    def test_the_backend_pin_is_load_bearing_for_this_row(self) -> None:
        """MUTATION: with this row removed, the pinned count must disagree.

        Redirects only the BACKEND entry at a mutated copy on disk. Patching
        `Path.read_text` globally would feed backend content to every matrix and
        this test would then pass on errors that have nothing to do with the
        removal -- green for the wrong reason, which is the failure mode these
        cases exist to catch.
        """
        clean: list[str] = []
        agent_record.check_matrices(clean)
        self.assertEqual([e for e in clean if "backend rows" in e.lower()], [])

        path, count = agent_record.MATRICES["BACKEND"]
        text = path.read_text(encoding="utf-8")
        without = "\n".join(
            l for l in text.splitlines() if not l.startswith(f"| `{self.ROW}` |")
        )
        self.assertNotEqual(without, text, "the row must be present to remove")

        # Under ROOT, not /tmp: check_matrices reports via
        # `path.relative_to(ROOT)`, which raises on a path outside the repo.
        # And BOTH tables need redirecting -- rows are parsed from
        # MATRIX_PATHS while the count is pinned in MATRICES, so patching only
        # the latter counts zero rows for a reason unrelated to the removal.
        with tempfile.TemporaryDirectory(dir=agent_record.ROOT) as tmp:
            mutated = Path(tmp) / "backend-matrix.md"
            mutated.write_text(without, encoding="utf-8")
            paths = [mutated if q == path else q for q in agent_record.MATRIX_PATHS]
            errors: list[str] = []
            with mock.patch.object(agent_record, "MATRIX_PATHS", paths), \
                 mock.patch.dict(
                     agent_record.MATRICES, {"BACKEND": (mutated, count)}
                 ):
                agent_record.check_matrices(errors)
        self.assertTrue(
            any("backend rows" in e.lower() for e in errors),
            f"removing {self.ROW} must break the BACKEND count; got {errors}",
        )


if __name__ == "__main__":
    unittest.main()
