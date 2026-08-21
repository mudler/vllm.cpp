#!/usr/bin/env python3
"""Mutation tests for scripts/check-doc-checkpoint.py.

The rewritten gate asks "did a row change lifecycle state", not "which directory
did you touch". These tests pin BOTH halves of that:

  * the obligation still fires on a real claim (a lifecycle move, a measurement),
    so a capability cannot land undocumented;
  * it does NOT fire on ordinary code edits, which is the defect that produced
    16 of the last 20 red CI runs and six hardcoded escape-hatch path sets.

The second half matters as much as the first. A gate that over-fires gets
worked around, and the workarounds were what made the old gate unrepairable.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "check_doc_checkpoint", ROOT / "scripts/check-doc-checkpoint.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


ROW_TABLE = """# Kernel matrix

| ID | Item | Spike/spec | State | Owner |
|---|---|---|---|---|
| `KERNEL-ALPHA` | alpha | [alpha](specs/alpha.md) | `{alpha}` | ops |
| `KERNEL-BETA` | beta | [beta](specs/beta.md) | `{beta}` | ops |
"""

# A row spec carrying the relocated live position (ENG-NOW-DERIVED, #374).
SPEC_WITH_NOW = "# Alpha\n\n## Scope\n\nthings\n\n## Now\n\nRun the focused gate.\n"
SPEC_WITHOUT_NOW = "# Alpha\n\n## Scope\n\nthings\n"
SPEC_EMPTY_NOW = "# Alpha\n\n## Scope\n\nthings\n\n## Now\n\n## Gates\n\nx\n"


class RowStateParsing(unittest.TestCase):
    def test_row_states_are_read_from_the_keyed_table(self):
        states = checker.row_states(ROW_TABLE.format(alpha="READY", beta="DONE"))
        self.assertEqual(states, {"KERNEL-ALPHA": "READY", "KERNEL-BETA": "DONE"})

    def test_prose_states_in_evidence_columns_do_not_win(self):
        """The state cell is the LAST backticked state on the row."""
        line = "| `KERNEL-X` | supersedes the `DONE` attempt | `ACTIVE` | ops |\n"
        self.assertEqual(checker.row_states(line), {"KERNEL-X": "ACTIVE"})

    def test_non_row_lines_are_ignored(self):
        self.assertEqual(checker.row_states("| not a row | `DONE` |\n"), {})


class LifecycleTrigger(unittest.TestCase):
    """classify() via a stubbed blob reader, so no git fixture is needed."""

    def classify(self, paths, before_text, after_text):
        original = checker.blob
        checker.blob = lambda rev, path: before_text if rev == "BEFORE" else after_text
        try:
            return checker.classify(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    def test_lifecycle_move_is_a_claim(self):
        classes, reasons = self.classify(
            [".agents/kernel-matrix.md"],
            ROW_TABLE.format(alpha="READY", beta="DONE"),
            ROW_TABLE.format(alpha="DONE", beta="DONE"),
        )
        self.assertIn("lifecycle", classes)
        self.assertTrue(any("READY -> DONE" in r for r in reasons), reasons)

    def test_editing_a_row_without_moving_its_state_is_not_a_claim(self):
        table = ROW_TABLE.format(alpha="READY", beta="DONE")
        classes, _ = self.classify(
            [".agents/kernel-matrix.md"], table, table.replace("alpha", "alpha prime")
        )
        self.assertNotIn("lifecycle", classes)

    def test_a_new_row_only_counts_once_it_claims_something(self):
        one_row = "| `KERNEL-ALPHA` | alpha | `{}` | ops |\n"
        for state, expected in (("TODO", False), ("READY", False), ("DONE", True)):
            with self.subTest(state=state):
                classes, _ = self.classify(
                    [".agents/kernel-matrix.md"], "", one_row.format(state)
                )
                self.assertEqual("lifecycle" in classes, expected)

    def test_a_measurement_is_a_claim(self):
        classes, reasons = self.classify(
            [".agents/benchmark-record.md"], "27B: 4.28 tok/s", "27B: 4.36 tok/s"
        )
        self.assertIn("lifecycle", classes)
        self.assertTrue(any("measurement" in r for r in reasons), reasons)


class ObligationsFire(unittest.TestCase):
    def errors(self, paths, before_text="", after_text="", specs=None):
        """Stub blob() PER PATH so a row table and its spec can differ.

        specs maps a spec path to its AFTER content; anything unnamed falls back
        to a spec that carries a `## Now`, so a test that is not about the spec
        obligation does not have to declare one.
        """
        specs = specs or {}
        original = checker.blob

        def fake(rev, path):
            if path.startswith(".agents/specs/"):
                return specs.get(path, SPEC_WITH_NOW)
            return before_text if rev == "BEFORE" else after_text

        checker.blob = fake
        try:
            return checker.errors_for(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    MOVED = (
        ROW_TABLE.format(alpha="READY", beta="DONE"),
        ROW_TABLE.format(alpha="DONE", beta="DONE"),
    )

    def test_lifecycle_move_demands_status_and_benchmarks(self):
        errors = self.errors([".agents/kernel-matrix.md"], *self.MOVED)
        self.assertTrue(errors)
        joined = " ".join(errors)
        for surface in ("docs/STATUS.md", "docs/BENCHMARKS.md"):
            self.assertIn(surface, joined)

    def test_a_lifecycle_move_no_longer_demands_the_shared_digest(self):
        """RED-BEFORE: .agents/NOW.md was in the lifecycle triple (#374).

        That one requirement is why every row-advancing PR wrote one shared
        file, and why NOW.md conflicted in 5 of the 16 conflicting open PRs
        measured at d928e2c3. A complete checkpoint must now pass WITHOUT it.
        """
        paths = [
            ".agents/kernel-matrix.md",
            ".agents/specs/alpha.md",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
        ]
        self.assertEqual(self.errors(paths, *self.MOVED), [])

    def test_each_required_surface_is_individually_load_bearing(self):
        required = ["docs/STATUS.md", "docs/BENCHMARKS.md"]
        for omitted in required:
            with self.subTest(omitted=omitted):
                paths = [".agents/kernel-matrix.md", ".agents/specs/alpha.md"] + [
                    s for s in required if s != omitted
                ]
                errors = self.errors(paths, *self.MOVED)
                self.assertTrue(errors, f"omitting {omitted} was not caught")
                self.assertIn(omitted, " ".join(errors))

    def test_a_complete_checkpoint_passes(self):
        paths = [
            ".agents/kernel-matrix.md",
            ".agents/specs/alpha.md",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
        ]
        self.assertEqual(self.errors(paths, *self.MOVED), [])

    def test_the_moved_rows_spec_is_required(self):
        """The relocated obligation, and the reason NOW could leave the triple.

        The live position still has to be recorded on a lifecycle move; it is
        recorded in the ROW's spec instead of the shared digest. Omit the spec
        and this must fail, or the requirement was dropped rather than moved.
        """
        paths = [".agents/kernel-matrix.md", "docs/STATUS.md", "docs/BENCHMARKS.md"]
        errors = self.errors(paths, *self.MOVED)
        self.assertTrue(errors)
        self.assertIn(".agents/specs/alpha.md", " ".join(errors))

    def test_a_spec_without_a_now_section_fails(self):
        paths = [
            ".agents/kernel-matrix.md",
            ".agents/specs/alpha.md",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
        ]
        errors = self.errors(
            paths, *self.MOVED, specs={".agents/specs/alpha.md": SPEC_WITHOUT_NOW}
        )
        self.assertTrue(errors)
        self.assertIn("`## Now`", " ".join(errors))

    def test_an_empty_now_section_fails(self):
        """A heading with nothing under it is not a recorded position."""
        paths = [
            ".agents/kernel-matrix.md",
            ".agents/specs/alpha.md",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
        ]
        errors = self.errors(
            paths, *self.MOVED, specs={".agents/specs/alpha.md": SPEC_EMPTY_NOW}
        )
        self.assertTrue(errors)
        self.assertIn("empty", " ".join(errors))

    def test_a_row_linking_no_spec_is_reported_not_skipped(self):
        """Fail closed: an unresolvable row must be named, never waved through."""
        no_spec = ROW_TABLE.replace(" [alpha](specs/alpha.md) |", " - |")
        before = no_spec.format(alpha="READY", beta="DONE")
        after = no_spec.format(alpha="DONE", beta="DONE")
        errors = self.errors(
            [".agents/kernel-matrix.md", "docs/STATUS.md", "docs/BENCHMARKS.md"],
            before,
            after,
        )
        self.assertTrue(any("links no spec" in e for e in errors), errors)

    def test_restoring_now_to_the_triple_breaks_the_new_contract(self):
        """MUTATION: proves the removal is what makes the NOW-free case pass."""
        paths = [
            ".agents/kernel-matrix.md",
            ".agents/specs/alpha.md",
            "docs/STATUS.md",
            "docs/BENCHMARKS.md",
        ]
        self.assertEqual(self.errors(paths, *self.MOVED), [])
        original = dict(checker.REQUIRED)
        checker.REQUIRED["lifecycle"] = (
            checker.STATUS,
            checker.BENCHMARKS,
            checker.NOW,
        )
        try:
            errors = self.errors(paths, *self.MOVED)
            self.assertTrue(errors, "the mutation must reinstate the shared write")
            self.assertIn(".agents/NOW.md", " ".join(errors))
        finally:
            checker.REQUIRED.clear()
            checker.REQUIRED.update(original)
        self.assertEqual(self.errors(paths, *self.MOVED), [])


class ObligationsDoNotOverfire(unittest.TestCase):
    """The regression this rewrite exists to prevent."""

    def errors(self, paths):
        original = checker.blob
        checker.blob = lambda rev, path: (
            "old claim" if rev == "BEFORE" else "new claim"
        )
        try:
            return checker.errors_for(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    def test_editing_source_owes_nothing(self):
        for path in (
            "src/vt/cuda/cuda_backend.cu",
            "tests/vt/test_backend.cpp",
            "scripts/check-agent-record.py",
            "tools/bench/serve.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.errors([path]), [])

    def test_a_governance_change_is_not_a_feature_checkpoint(self):
        """Staging this very design must not demand a benchmark update."""
        self.assertEqual(
            self.errors(
                [
                    "AGENTS.md",
                    ".agents/workflow.md",
                    "scripts/check-doc-checkpoint.py",
                    "tests/scripts/test_doc_checkpoint.py",
                    "docs/superpowers/specs/2026-08-09-policy-simplification-design.md",
                ]
            ),
            [],
        )

    def test_no_hardcoded_escape_hatches_remain(self):
        """Six exact-path-set exemptions were fossils of the wrong trigger."""
        source = (ROOT / "scripts/check-doc-checkpoint.py").read_text(encoding="utf-8")
        for fossil in (
            "POLICY_CONSOLIDATION_FILES",
            "POLICY_CUTOVER_FILES",
            "PR_SIZE_BOOTSTRAP_FILES",
            "PENDING_PR_RANGE_FILES",
            "SYNTHETIC_MERGE_RANGE_FILES",
            "CLAIM_CUTOVER_FILES",
        ):
            self.assertNotIn(fossil, source)


class PublicDocumentPolicyTests(unittest.TestCase):
    def test_lifecycle_projection_names_the_row_spec_not_now_md(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        section = agents.split("## Public documents", 1)[1].split(
            "## Work happens in a worktree", 1
        )[0]
        self.assertIn("| `docs/STATUS.md` | a row changes lifecycle state |", section)
        self.assertIn(
            "| `docs/BENCHMARKS.md` | a row gains an accepted or explicitly "
            "pending/failed/void measurement |",
            section,
        )
        self.assertIn("| the moved row spec's `## Now` |", section)
        self.assertNotIn("| `.agents/NOW.md` |", section)
        prose = " ".join(section.split())
        self.assertIn(
            "A lifecycle change owes `STATUS`, `BENCHMARKS`, and the moved row "
            "spec's `## Now`.",
            prose,
        )


class SupportSurfaces(unittest.TestCase):
    def errors(self, paths):
        original = checker.blob
        checker.blob = lambda rev, path: (
            "old claim" if rev == "BEFORE" else "new claim"
        )
        try:
            return checker.errors_for(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    def test_a_matrix_record_still_owes_the_feature_surface(self):
        # The four .agents/*-matrix.md records ARE claim surfaces, so they keep
        # the path trigger. Pins what #595's fix must not widen away.
        errors = self.errors([".agents/model-matrix.md"])
        self.assertTrue(errors)
        self.assertIn("docs/FEATURES.md", errors[0])

    def test_a_user_entrypoint_owes_usage(self):
        errors = self.errors(["include/vllm.h"])
        self.assertTrue(errors)
        self.assertIn("docs/USAGE.md", errors[0])

    def test_readme_needs_a_landing_source(self):
        errors = self.errors(["README.md"])
        self.assertTrue(errors)
        self.assertIn("landing source", errors[0])

    def test_readme_with_a_landing_source_passes(self):
        self.assertEqual(self.errors(["README.md", ".agents/mission.md"]), [])

    def test_a_coedited_projection_never_licenses_readme_churn(self):
        """Deliberate: STATUS.md changing is not permission to rewrite README."""
        errors = self.errors(["README.md", "docs/STATUS.md"])
        self.assertTrue(errors)
        self.assertIn("landing source", errors[0])

    def test_a_link_repair_is_not_a_claim(self):
        """Moving a file forces every document linking it to be edited.

        Those edits assert nothing, so they must not demand a benchmark update
        -- that demand is how the old gate grew six hardcoded escape hatches.
        Link TARGETS are normalised away; link TEXT is not.
        """
        before = "See [current state](.agents/state.md) for details.\n"
        after = "See [current state](.agents/NOW.md) for details.\n"
        self.assertFalse(checker.claims_changed(before, after))

        reworded = "See [the live snapshot](.agents/NOW.md) for details.\n"
        self.assertTrue(checker.claims_changed(before, reworded))

        original = checker.blob
        checker.blob = lambda rev, path: before if rev == "BEFORE" else after
        try:
            self.assertEqual(
                checker.errors_for({"README.md"}, "BEFORE", "AFTER"), []
            )
            self.assertEqual(
                checker.errors_for(
                    {".agents/benchmark-record.md"}, "BEFORE", "AFTER"
                ),
                [],
            )
        finally:
            checker.blob = original

    def test_a_landing_source_permits_but_does_not_demand_readme(self):
        self.assertEqual(self.errors([".agents/mission.md"]), [])

    def test_the_quickstart_page_is_a_landing_source(self):
        """#1520: the page the README now defers to for starting a model.

        The README `## Quickstart` block used to carry the commands. It now
        points at `docs/QUICKSTART.md`, so the claim "this is where a reader
        starts" changed BECAUSE that page exists. The page is not a projection
        of a claim recorded elsewhere, so a change to it is a real reason for
        the README pointer at it to change.
        """
        self.assertEqual(self.errors(["README.md", "docs/QUICKSTART.md"]), [])

    def test_the_quickstart_page_permits_but_does_not_demand_readme(self):
        """`landing_page` never demands the README, and #1520 does not add one."""
        self.assertEqual(self.errors(["docs/QUICKSTART.md"]), [])

    def test_an_unrelated_document_never_licenses_readme_churn(self):
        """The property #1520 must not break, stated on a NON-projection doc.

        `test_a_coedited_projection_never_licenses_readme_churn` covers a public
        projection. This covers an ordinary document under `docs/`, which is the
        class the quickstart page belongs to by path. Exactly one member of that
        class is a landing source, and admitting it must not admit the class.
        """
        for document in ("docs/BUILD.md", "docs/ROCM.md", "docs/RELEASES.md"):
            with self.subTest(document=document):
                errors = self.errors(["README.md", document])
                self.assertTrue(errors)
                self.assertIn("landing source", errors[0])




class FeatureSurfaceTrigger(unittest.TestCase):
    """#595: a model file owes docs/FEATURES.md when its REGISTRATIONS change.

    Editing the internals of an already-registered architecture is not a claim
    about what the project supports. Keying on the path instead is what made a
    one-line compile fix owe a public-doc edit, and #1054 answered that demand
    with prose that crossed the check-public-doc-tables budgets and blocked
    every push in the repository (#1055, #1058, #1062).
    """

    MODEL = "src/vllm/model_executor/models/somemodel.cpp"
    REGISTERED = 'REGISTER_VLLM_MODEL("SomeForCausalLM", SomeModel);\nint f() { return 1; }\n'
    EDITED = 'REGISTER_VLLM_MODEL("SomeForCausalLM", SomeModel);\nint f() { return 2; }\n'
    ADDED = (
        'REGISTER_VLLM_MODEL("SomeForCausalLM", SomeModel);\n'
        'REGISTER_VLLM_MODEL("OtherForCausalLM", OtherModel);\n'
    )

    def errors(self, paths, before_text, after_text):
        original = checker.blob

        def fake(rev, path):
            if path.startswith(".agents/specs/"):
                return SPEC_WITH_NOW
            return before_text if rev == "BEFORE" else after_text

        checker.blob = fake
        try:
            return checker.errors_for(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    def test_editing_a_registered_model_owes_nothing(self):
        # RED before the #595 trigger change: the path alone fired.
        errors = self.errors([self.MODEL], self.REGISTERED, self.EDITED)
        self.assertEqual(errors, [], f"a registration-preserving edit demanded: {errors}")

    def test_a_new_model_registration_owes_the_feature_surface(self):
        errors = self.errors([self.MODEL], self.REGISTERED, self.ADDED)
        self.assertTrue(errors, "adding an architecture must still owe FEATURES.md")
        self.assertIn("docs/FEATURES.md", errors[0])

    def test_a_brand_new_model_file_owes_the_feature_surface(self):
        errors = self.errors([self.MODEL], "", self.REGISTERED)
        self.assertTrue(errors, "a new registered architecture must owe FEATURES.md")
        self.assertIn("docs/FEATURES.md", errors[0])

    def test_removing_a_registration_owes_the_feature_surface(self):
        errors = self.errors([self.MODEL], self.ADDED, self.REGISTERED)
        self.assertTrue(errors, "removing an architecture must still owe FEATURES.md")
        self.assertIn("docs/FEATURES.md", errors[0])

    def test_a_satisfied_registration_change_passes(self):
        errors = self.errors(
            [self.MODEL, "docs/FEATURES.md"], self.REGISTERED, self.ADDED
        )
        self.assertEqual(errors, [])


class PartialIsALifecycleState(unittest.TestCase):
    """GATE-DOC-CHECKPOINT-STATES (#1434).

    `STATES` is the whole definition of what a lifecycle state IS for this gate.
    `PARTIAL` was absent from it while being the second most used state in the
    matrices -- 118 backticked cells at 947e5f648, against 77 for `DONE`. The
    consequence is not a wrong label: `row_states` DROPS a row it cannot match,
    and both `lifecycle_moves` and `moved_rows` iterate the AFTER map, so a row
    leaving the matched set is never compared against its predecessor at all.

    Measured with scratch commits at 947e5f648 on the unmodified checker, the
    two transitions below returned rc 0 and this file had nothing that could
    fail. The third returned rc 1 for the WRONG reason.
    """

    TABLE = ROW_TABLE

    def errors(self, paths, before_text, after_text):
        original = checker.blob

        def fake(rev, path):
            if path.startswith(".agents/specs/"):
                return SPEC_WITH_NOW
            return before_text if rev == "BEFORE" else after_text

        checker.blob = fake
        try:
            return checker.errors_for(set(paths), "BEFORE", "AFTER")
        finally:
            checker.blob = original

    def reasons(self, before_text, after_text):
        original = checker.blob
        checker.blob = lambda rev, path: (
            before_text if rev == "BEFORE" else after_text
        )
        try:
            return checker.classify(
                {".agents/kernel-matrix.md"}, "BEFORE", "AFTER"
            )
        finally:
            checker.blob = original

    def test_a_move_into_partial_is_a_lifecycle_move(self):
        """RED-BEFORE. The reported case: LOAD-GGUF-MMPROJ moved READY -> PARTIAL.

        The row vanishes from the AFTER map, so nothing iterates it and the
        checkpoint surfaces are owed by a claim no gate can observe.
        """
        errors = self.errors(
            [".agents/kernel-matrix.md"],
            self.TABLE.format(alpha="READY", beta="DONE"),
            self.TABLE.format(alpha="PARTIAL", beta="DONE"),
        )
        self.assertTrue(errors, "READY -> PARTIAL must be a lifecycle move")
        joined = " ".join(errors)
        for surface in ("docs/STATUS.md", "docs/BENCHMARKS.md", ".agents/specs/alpha.md"):
            self.assertIn(surface, joined)

    def test_a_move_out_of_partial_is_a_lifecycle_move(self):
        """RED-BEFORE. Absent from BEFORE, and READY is not in the claim set."""
        errors = self.errors(
            [".agents/kernel-matrix.md"],
            self.TABLE.format(alpha="PARTIAL", beta="DONE"),
            self.TABLE.format(alpha="READY", beta="DONE"),
        )
        self.assertTrue(errors, "PARTIAL -> READY must be a lifecycle move")
        self.assertIn("docs/STATUS.md", " ".join(errors))

    def test_a_move_out_of_partial_names_the_transition_not_a_new_row(self):
        """PARTIAL -> ACTIVE already red, for the wrong reason.

        Unseen in BEFORE, the row looked BRAND NEW, so the gate said `added as
        ACTIVE` about a row that had existed for months. The verdict was right
        by accident; the reason a reader is handed has to be right too.
        """
        classes, reasons = self.reasons(
            self.TABLE.format(alpha="PARTIAL", beta="DONE"),
            self.TABLE.format(alpha="ACTIVE", beta="DONE"),
        )
        self.assertIn("lifecycle", classes)
        joined = " ".join(reasons)
        self.assertIn("KERNEL-ALPHA PARTIAL -> ACTIVE", joined)
        self.assertNotIn("added as", joined)

    def test_a_complete_partial_checkpoint_passes(self):
        """The widening must be payable, not merely loud."""
        errors = self.errors(
            [
                ".agents/kernel-matrix.md",
                ".agents/specs/alpha.md",
                "docs/STATUS.md",
                "docs/BENCHMARKS.md",
            ],
            self.TABLE.format(alpha="READY", beta="DONE"),
            self.TABLE.format(alpha="PARTIAL", beta="DONE"),
        )
        self.assertEqual(errors, [])

    def test_prose_partial_does_not_beat_the_state_cell(self):
        """The last-match heuristic still holds for the new token.

        It is load-bearing here: `KV-BLOCK-POOL` writes `PARTIAL` (not `DONE`)
        in its own evidence prose, which is why the gate resolved that row as
        `DONE` before this change.
        """
        line = "| `KERNEL-X` | `PARTIAL` (not `DONE`) because two arms refuse | `PARTIAL` | ops |\n"
        self.assertEqual(checker.row_states(line), {"KERNEL-X": "PARTIAL"})

    def test_removing_partial_from_states_restores_the_blind_spot(self):
        """MUTATION: proves the tuple entry is what makes the cases above fire.

        STATE_CELL is derived from STATES at import, so a mutation that only
        rebound STATES would apply to nothing and read as a passing test.
        Both are rebound, and the assertion below is that the gate goes BLIND.
        """
        moved = (
            self.TABLE.format(alpha="READY", beta="DONE"),
            self.TABLE.format(alpha="PARTIAL", beta="DONE"),
        )
        self.assertTrue(self.errors([".agents/kernel-matrix.md"], *moved))

        states, cell = checker.STATES, checker.STATE_CELL
        narrowed = tuple(s for s in states if s != "PARTIAL")
        self.assertNotEqual(narrowed, states, "the mutation applied to nothing")
        checker.STATES = narrowed
        checker.STATE_CELL = re.compile(
            r"`(" + "|".join(re.escape(s) for s in narrowed) + r")`"
        )
        try:
            self.assertEqual(
                self.errors([".agents/kernel-matrix.md"], *moved),
                [],
                "without PARTIAL the gate must go blind; it did not, so these "
                "tests are measuring something other than the tuple entry",
            )
        finally:
            checker.STATES, checker.STATE_CELL = states, cell
        self.assertTrue(self.errors([".agents/kernel-matrix.md"], *moved))

    def test_a_new_row_added_as_partial_is_still_not_a_claim(self):
        """A deliberate exclusion, pinned so a later widening is a decision.

        The claim set governs rows ABSENT from the BEFORE map, and the dominant
        real cause of that here is a record RELOCATION between matrices, not a
        new capability. Widening it would red legitimate record moves for 118
        more rows while fixing nothing #1434 reports.
        """
        one_row = "| `KERNEL-ALPHA` | alpha | `PARTIAL` | ops |\n"
        classes, _ = self.reasons("", one_row)
        self.assertNotIn("lifecycle", classes)


class AnchorBackfillIsDeliberatelyExcluded(unittest.TestCase):
    """The ruling #1434 asked for, kept executable rather than only argued.

    `.agents/feature-matrix.md` names `PARTIAL`, `BLOCKED` and `ANCHOR-BACKFILL`
    together, so "the matrix calls it a state" cannot be the test. The test is
    whether `docs/STATUS.md` PROJECTS it, because `REQUIRED["lifecycle"]` is
    (STATUS, BENCHMARKS) and carries all of them or none.
    """

    def test_partial_is_a_public_status_term(self):
        status = (ROOT / "docs/STATUS.md").read_text(encoding="utf-8")
        self.assertIn(
            "| Partial | A usable path exists with named missing behavior or "
            "evidence |",
            status,
        )
        self.assertIn("PARTIAL", checker.STATES)

    def test_anchor_backfill_is_not_a_public_status_term(self):
        """If this ever fails, the exclusion below has to be revisited."""
        status = (ROOT / "docs/STATUS.md").read_text(encoding="utf-8")
        self.assertNotIn("ANCHOR-BACKFILL", status)
        self.assertNotIn("Anchor-backfill", status)

    def test_anchor_backfill_stays_out_of_the_tuple(self):
        """Admitting it demands a STATUS edit with nothing true to write.

        That is the shape check-doc-checkpoint.py:4-17 records as the reason
        this file was rewritten: 16 of 20 red CI runs and six hardcoded
        escape-hatch path sets. Paying the `## Now` half it genuinely owes needs
        REQUIRED to carry a spec-only class, which is a different change and is
        listed under `## Owed` in doc-checkpoint-lifecycle-states.md.
        """
        self.assertNotIn("ANCHOR-BACKFILL", checker.STATES)
        spec = (
            ROOT / ".agents/specs/doc-checkpoint-lifecycle-states.md"
        ).read_text(encoding="utf-8")
        self.assertIn("## Owed", spec)
        self.assertIn("ANCHOR-BACKFILL", spec.split("## Owed", 1)[1])


if __name__ == "__main__":
    unittest.main()
