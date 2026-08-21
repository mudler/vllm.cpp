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

    def test_a_new_row_added_as_partial_is_a_claim(self):
        """OVERTURNED in W2 of this row (#1434), and the reversal is the point.

        This test previously asserted the opposite. The exclusion it pinned
        rested on "the dominant real cause of a row being absent from BEFORE is
        a record RELOCATION between matrices", which was never measured. It is
        now: over the 400 non-merge commits ending at e2a9e035d there were 45
        arrivals, ALL genuinely new, and ZERO relocations. The reasoning and the
        mutation live in ANewPartialRowIsAClaim below; this case stays here so
        that the two halves of the PARTIAL decision are read together.
        """
        one_row = "| `KERNEL-ALPHA` | alpha | `PARTIAL` | ops |\n"
        classes, _ = self.reasons("", one_row)
        self.assertIn("lifecycle", classes)


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



# A table whose row links no spec at all. Pre-claim rows are shaped like this by
# protocol -- .agents/feature-matrix.md gives a SPIKE row a `CLAIM-*`, not a spec
# -- and 416 of the 463 INVENTORIED rows in the seven tables carry no spec link.
UNLINKED_ROW_TABLE = """# Kernel matrix

| ID | Item | Spike/spec | State | Owner |
|---|---|---|---|---|
| `KERNEL-ALPHA` | alpha | `CLAIM-ALPHA` | `{alpha}` | ops |
"""


class RecordStatesAreResolvedButNotClaims(unittest.TestCase):
    """GATE-DOC-CHECKPOINT-STATES W2 (#1434), residuals 1 and 3.

    INVENTORIED, SPIKE and ANCHOR-BACKFILL are real lifecycle positions that
    docs/STATUS.md carries no term for. They were unresolvable, so `row_states`
    dropped the row and NOTHING downstream ran: a row moving out of the matched
    set looked like a deletion, and an ANCHOR-BACKFILL move was invisible.

    The first wave recorded the fix as "REQUIRED needs a spec-only class". That
    premise is wrong and measuring it is what closed the residual: `errors_for`
    calls `spec_now_errors` DIRECTLY, never through `classify()` or `REQUIRED`,
    so a spec-only obligation already existed. What was missing was resolution.
    """

    def errors(self, paths, before_text, after_text, specs=None):
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

    def classify(self, before_text, after_text):
        original = checker.blob
        checker.blob = lambda rev, path: (
            before_text if rev == "BEFORE" else after_text
        )
        try:
            return checker.classify({".agents/kernel-matrix.md"}, "BEFORE", "AFTER")
        finally:
            checker.blob = original

    def test_record_states_resolve(self):
        """RED-BEFORE: `row_states` dropped every one of these rows."""
        for state in ("INVENTORIED", "SPIKE", "ANCHOR-BACKFILL"):
            with self.subTest(state=state):
                states = checker.row_states(
                    ROW_TABLE.format(alpha=state, beta="DONE")
                )
                self.assertEqual(states.get("KERNEL-ALPHA"), state)

    def test_a_move_into_anchor_backfill_owes_the_spec_now(self):
        """RED-BEFORE. Residual 1: the obligation #1434 named and did not pay.

        DONE -> ANCHOR-BACKFILL is a record-hygiene move: the capability is
        already implemented and only the row's anchors are missing.
        """
        errors = self.errors(
            [".agents/kernel-matrix.md"],
            ROW_TABLE.format(alpha="DONE", beta="DONE"),
            ROW_TABLE.format(alpha="ANCHOR-BACKFILL", beta="DONE"),
        )
        self.assertTrue(errors, "DONE -> ANCHOR-BACKFILL must be seen")
        self.assertIn(".agents/specs/alpha.md", " ".join(errors))

    def test_a_record_move_does_not_demand_the_public_surfaces(self):
        """The other half of the ruling, and the reason it is payable.

        Admitting these states into `STATES` would have triggered
        REQUIRED["lifecycle"] = (STATUS, BENCHMARKS), which carries all of its
        surfaces or none. docs/STATUS.md has nothing true to write about a row
        whose capability did not change, and demanding an edit with nothing to
        say is the shape check-doc-checkpoint.py:4-17 records as the reason this
        file was rewritten.
        """
        errors = self.errors(
            [".agents/kernel-matrix.md", ".agents/specs/alpha.md"],
            ROW_TABLE.format(alpha="DONE", beta="DONE"),
            ROW_TABLE.format(alpha="ANCHOR-BACKFILL", beta="DONE"),
        )
        self.assertEqual(errors, [])

        classes, _ = self.classify(
            ROW_TABLE.format(alpha="DONE", beta="DONE"),
            ROW_TABLE.format(alpha="ANCHOR-BACKFILL", beta="DONE"),
        )
        self.assertNotIn("lifecycle", classes)

    def test_leaving_the_claim_set_is_no_longer_silent(self):
        """RED-BEFORE. Residual 3: READY -> INVENTORIED looked like a deletion.

        `transitions` iterates the AFTER map, so an unresolvable destination
        state removed the row from the comparison entirely. Resolving the record
        states fixes this WITHOUT reporting disappearances, which would red a
        legitimate relocation between matrices.
        """
        for destination in ("INVENTORIED", "SPIKE", "ANCHOR-BACKFILL"):
            with self.subTest(destination=destination):
                errors = self.errors(
                    [".agents/kernel-matrix.md"],
                    ROW_TABLE.format(alpha="READY", beta="DONE"),
                    ROW_TABLE.format(alpha=destination, beta="DONE"),
                )
                self.assertTrue(errors, f"READY -> {destination} must be seen")
                self.assertIn(".agents/specs/alpha.md", " ".join(errors))

    def test_entering_the_claim_set_from_a_record_state_is_seen(self):
        """RED-BEFORE. INVENTORIED -> READY: absent from BEFORE, and READY is
        not in the arrival set, so the row was invisible from both sides.

        This is not hypothetical: ab6e65216 on main moved ENG-CUDAGRAPH-BREAK
        INVENTORIED -> READY and the gate reported nothing.
        """
        errors = self.errors(
            [".agents/kernel-matrix.md"],
            ROW_TABLE.format(alpha="INVENTORIED", beta="DONE"),
            ROW_TABLE.format(alpha="READY", beta="DONE"),
        )
        self.assertTrue(errors)
        self.assertIn(".agents/specs/alpha.md", " ".join(errors))

    def test_a_record_move_reaching_a_claim_state_stays_a_record_move(self):
        """INVENTORIED -> DONE crosses the boundary, so which side wins?

        The record side. STATUS.md can say what a DONE row does, but the move
        itself starts from a position that page never carried, and the arrival
        rule below already governs a row that appears as DONE. Pinned so that
        the polarity is a decision rather than an accident of the `and`.
        """
        classes, _ = self.classify(
            ROW_TABLE.format(alpha="INVENTORIED", beta="DONE"),
            ROW_TABLE.format(alpha="DONE", beta="DONE"),
        )
        self.assertNotIn("lifecycle", classes)

    def test_a_record_move_on_a_row_linking_no_spec_is_not_an_error(self):
        """The one narrowing in this change, and it is stated as such.

        For a CLAIM move a row linking no spec is an error naming the row. For a
        RECORD move it cannot be: .agents/feature-matrix.md gives a pre-claim
        row a `CLAIM-*` rather than a spec, and 416 of 463 INVENTORIED rows link
        none. Demanding one would demand a document the protocol says does not
        exist yet. Nothing is loosened -- before this change the move was not
        observed at all.
        """
        errors = self.errors(
            [".agents/kernel-matrix.md"],
            UNLINKED_ROW_TABLE.format(alpha="INVENTORIED"),
            UNLINKED_ROW_TABLE.format(alpha="SPIKE"),
        )
        self.assertEqual(errors, [])

    def test_a_claim_move_on_a_row_linking_no_spec_is_still_an_error(self):
        """The carve-out above must not leak into the claim path."""
        errors = self.errors(
            [".agents/kernel-matrix.md"],
            UNLINKED_ROW_TABLE.format(alpha="READY"),
            UNLINKED_ROW_TABLE.format(alpha="DONE"),
        )
        self.assertTrue(errors)
        self.assertIn("links no spec", " ".join(errors))

    def test_a_record_move_still_needs_a_non_empty_now(self):
        """A linked spec is held to the same `## Now` contract as a claim move."""
        for body, expected in (
            (SPEC_WITHOUT_NOW, "has no `## Now` section"),
            (SPEC_EMPTY_NOW, "`## Now` is empty"),
        ):
            with self.subTest(expected=expected):
                errors = self.errors(
                    [".agents/kernel-matrix.md", ".agents/specs/alpha.md"],
                    ROW_TABLE.format(alpha="DONE", beta="DONE"),
                    ROW_TABLE.format(alpha="ANCHOR-BACKFILL", beta="DONE"),
                    specs={".agents/specs/alpha.md": body},
                )
                self.assertTrue(errors)
                self.assertIn(expected, " ".join(errors))

    def test_a_row_added_as_a_record_state_claims_nothing(self):
        """A newly inventoried row has no prior position and asserts nothing."""
        for state in ("INVENTORIED", "SPIKE", "ANCHOR-BACKFILL"):
            with self.subTest(state=state):
                errors = self.errors(
                    [".agents/kernel-matrix.md"],
                    "",
                    ROW_TABLE.format(alpha=state, beta=state),
                )
                self.assertEqual(errors, [])

    def test_prose_naming_a_record_state_does_not_beat_the_state_cell(self):
        """MEASURED, not stylistic: this is why RECORD_CELL is cell-anchored.

        The evidence prose in this repository narrates record states mid
        sentence far more often than claim ones. Two model rows literally write
        "the row stays `SPIKE`" while their State cells read `BLOCKED` and
        `ACTIVE`, so a record state matched anywhere on the line resolved both
        rows to `SPIKE` -- turning a correct resolution into a wrong one.
        """
        line = (
            "| `KERNEL-X` | claimed by two campaigns, so the row stays `SPIKE` "
            "until W3 lands | `BLOCKED` | ops |\n"
        )
        self.assertEqual(checker.row_states(line), {"KERNEL-X": "BLOCKED"})

    def test_a_record_state_opening_a_cell_wins_over_later_prose(self):
        """The converse, and the two rows this repairs.

        engine-matrix KV-EXTERNAL-CACHE resolved `ACTIVE` and quantization-matrix
        QUANT-GGUF-PRESETS resolved `READY`, both from prose after their real
        State cells. A row whose State cell is a record state is by definition
        making no claim, and resolving it to a claim state is the one error that
        makes this gate demand the wrong surfaces.
        """
        line = (
            "| `KERNEL-X` | anchors owed | `ANCHOR-BACKFILL` (W1-W5 landed) | "
            "the seam is `ACTIVE` upstream | ops |\n"
        )
        self.assertEqual(checker.row_states(line), {"KERNEL-X": "ANCHOR-BACKFILL"})

    def test_unanchoring_the_record_cell_breaks_two_real_rows(self):
        """MUTATION on the ANCHORING, which nothing else here proves.

        Drop the cell boundary from RECORD_CELL and the same prose line resolves
        to `SPIKE`. That is not a fixture: `MODEL-TEXT-deepseek-v2-deepseek-v3`
        and `MODEL-TEXT-kimi-linear` both narrate "the row stays `SPIKE`" while
        their State cells read `BLOCKED` and `ACTIVE`, so the unanchored form
        turns two correct resolutions into wrong ones.
        """
        line = (
            "| `KERNEL-X` | claimed by two campaigns, so the row stays `SPIKE` "
            "until W3 lands | `BLOCKED` | ops |\n"
        )
        self.assertEqual(checker.row_states(line), {"KERNEL-X": "BLOCKED"})

        cell = checker.RECORD_CELL
        unanchored = re.compile(
            r"`(" + "|".join(re.escape(s) for s in checker.RECORD_STATES) + r")`"
        )
        self.assertNotEqual(unanchored.pattern, cell.pattern, "mutation applied to nothing")
        checker.RECORD_CELL = unanchored
        try:
            self.assertEqual(
                checker.row_states(line),
                {"KERNEL-X": "SPIKE"},
                "unanchored, the evidence prose must win; it did not, so the "
                "anchoring is not what this test measures",
            )
        finally:
            checker.RECORD_CELL = cell
        self.assertEqual(checker.row_states(line), {"KERNEL-X": "BLOCKED"})

    def test_removing_the_record_states_restores_the_blind_spot(self):
        """MUTATION. RECORD_CELL is derived from RECORD_STATES at import, so a
        mutation that rebound only the tuple would apply to nothing and read as
        a passing test. Both are rebound.
        """
        moved = (
            ROW_TABLE.format(alpha="DONE", beta="DONE"),
            ROW_TABLE.format(alpha="ANCHOR-BACKFILL", beta="DONE"),
        )
        self.assertTrue(self.errors([".agents/kernel-matrix.md"], *moved))

        record, cell = checker.RECORD_STATES, checker.RECORD_CELL
        checker.RECORD_STATES = ()
        checker.RECORD_CELL = re.compile(r"(?!x)x")
        self.assertNotEqual(checker.RECORD_STATES, record, "mutation applied to nothing")
        try:
            self.assertEqual(
                self.errors([".agents/kernel-matrix.md"], *moved),
                [],
                "without the record states the gate must go blind; it did not, "
                "so these tests measure something other than the tuple",
            )
        finally:
            checker.RECORD_STATES, checker.RECORD_CELL = record, cell
        self.assertTrue(self.errors([".agents/kernel-matrix.md"], *moved))

    def test_treating_a_record_move_as_a_claim_move_demands_status(self):
        """MUTATION on the SPLIT, not on the tuple.

        Resolution alone is not the change; classifying the move as RECORD is.
        Force every move to CLAIM and the same DONE -> ANCHOR-BACKFILL edit
        starts demanding docs/STATUS.md, which is the outcome the ruling rejects.
        """
        paid = [".agents/kernel-matrix.md", ".agents/specs/alpha.md"]
        moved = (
            ROW_TABLE.format(alpha="DONE", beta="DONE"),
            ROW_TABLE.format(alpha="ANCHOR-BACKFILL", beta="DONE"),
        )
        self.assertEqual(self.errors(paid, *moved), [])

        original = checker.transitions

        def all_claim(paths, before, after):
            return [
                (path, row, previous, state, checker.CLAIM)
                for path, row, previous, state, _kind in original(
                    paths, before, after
                )
            ]

        checker.transitions = all_claim
        try:
            errors = self.errors(paid, *moved)
            self.assertTrue(errors)
            self.assertIn("docs/STATUS.md", " ".join(errors))
        finally:
            checker.transitions = original
        self.assertEqual(self.errors(paid, *moved), [])


class ANewPartialRowIsAClaim(unittest.TestCase):
    """GATE-DOC-CHECKPOINT-STATES W2 (#1434), residual 4.

    This OVERTURNS a decision the first wave pinned. That exclusion rested on
    "the dominant real cause of a row being absent from BEFORE is a record
    RELOCATION between matrices", and the premise was never measured. Replayed
    over the 400 non-merge commits ending at e2a9e035d, 126 of which touch a row
    table: 45 arrivals, ALL genuinely new, ZERO relocations, ZERO departures.
    Two arrived as PARTIAL. The case the exclusion protected does not occur.

    Admitting it costs nothing historically: the one commit with PARTIAL
    arrivals, 5498b4aea, already paid STATUS, BENCHMARKS and the spec.
    """

    def classify(self, before_text, after_text):
        original = checker.blob
        checker.blob = lambda rev, path: (
            before_text if rev == "BEFORE" else after_text
        )
        try:
            return checker.classify({".agents/model-matrix.md"}, "BEFORE", "AFTER")
        finally:
            checker.blob = original

    ROW = "| `MODEL-ALPHA` | alpha | [alpha](specs/alpha.md) | `{}` | ops |\n"

    def test_a_new_row_added_as_partial_is_a_claim(self):
        """RED-BEFORE: PARTIAL was not in the arrival set."""
        classes, reasons = self.classify("", self.ROW.format("PARTIAL"))
        self.assertIn("lifecycle", classes)
        self.assertIn("MODEL-ALPHA added as PARTIAL", " ".join(reasons))

    def test_the_pre_claim_arrivals_are_still_not_claims(self):
        """The polarity is unchanged for everything that claims nothing."""
        for state in ("TODO", "READY", "INVENTORIED", "SPIKE", "ANCHOR-BACKFILL"):
            with self.subTest(state=state):
                classes, _ = self.classify("", self.ROW.format(state))
                self.assertNotIn("lifecycle", classes)

    def test_removing_partial_from_the_arrival_set_restores_the_blind_spot(self):
        """MUTATION. CLAIM_ON_ARRIVAL is read at call time, so rebinding it is
        enough -- unlike STATE_CELL, nothing is derived from it at import.
        """
        self.assertIn("lifecycle", self.classify("", self.ROW.format("PARTIAL"))[0])

        arrival = checker.CLAIM_ON_ARRIVAL
        narrowed = frozenset(arrival - {"PARTIAL"})
        self.assertNotEqual(narrowed, arrival, "the mutation applied to nothing")
        checker.CLAIM_ON_ARRIVAL = narrowed
        try:
            self.assertNotIn(
                "lifecycle",
                self.classify("", self.ROW.format("PARTIAL"))[0],
                "without PARTIAL in the arrival set the gate must go blind",
            )
        finally:
            checker.CLAIM_ON_ARRIVAL = arrival
        self.assertIn("lifecycle", self.classify("", self.ROW.format("PARTIAL"))[0])


class TheSglangMatrixIsNotALifecycleTable(unittest.TestCase):
    """GATE-DOC-CHECKPOINT-STATES W2 (#1434), residual 2: DECLINED, with the
    measurement that declines it kept executable.

    `.agents/sglang-matrix.md` is a keyed table with `SGLANG-*` row IDs, so it
    LOOKS like a seventh sibling of ROW_TABLES. It is not: its own header says it
    carries "a **classification** in place of a lifecycle state", and that axis
    is FUSED / SGLANG-DISTINCT / INVENTORIED / OUT-OF-SCOPE, of which only
    INVENTORIED is a lifecycle token and it is written unbackticked there.

    Admitting the file would observe ZERO of its 46 keyed rows on their own axis
    and three on prose about OTHER matrices' rows -- `ENG-ASYNC-SCHED` `DONE`,
    `SPEC-MTP` `DONE`, `SPEC-DFLASH` `DONE`, each quoted inside the "our
    implementation" column. Rewording one of those sentences would then read as
    an SGLANG row changing lifecycle state and would demand docs/STATUS.md,
    docs/BENCHMARKS.md and a spec `## Now` for a move that never happened.
    """

    LEGEND = ("FUSED", "SGLANG-DISTINCT", "OUT-OF-SCOPE")

    def setUp(self):
        self.text = (ROOT / ".agents/sglang-matrix.md").read_text(encoding="utf-8")

    def test_the_file_stays_out_of_row_tables(self):
        self.assertNotIn(".agents/sglang-matrix.md", checker.ROW_TABLES)

    def test_its_axis_is_a_classification_not_a_lifecycle_state(self):
        self.assertIn("**classification** in place of a lifecycle state", self.text)
        for value in self.LEGEND:
            self.assertNotIn(value, checker.LIFECYCLE_STATES)

    def test_every_row_the_checker_could_resolve_there_is_a_phantom(self):
        """The load-bearing half: each resolvable row HAS its own classification
        cell, so the token the checker matched came from somewhere else.

        If the file ever grows a real lifecycle State cell, that row will have no
        classification cell and this fails -- which is the point. The exclusion
        is then a decision someone makes again, not a drift.
        """
        resolved = checker.row_states(self.text)
        self.assertTrue(
            resolved, "the trap is that the file resolves rows, not that it is inert"
        )
        for line in self.text.splitlines():
            identifier = checker.ROW_ID.match(line)
            if not identifier or identifier.group(1) not in resolved:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            classification = [cell for cell in cells if cell in self.LEGEND]
            self.assertEqual(
                len(classification),
                1,
                f"{identifier.group(1)} has no classification cell, so its "
                f"resolved state {resolved[identifier.group(1)]!r} may be real; "
                "re-decide whether this file belongs in ROW_TABLES",
            )


if __name__ == "__main__":
    unittest.main()
