#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-oracle-pins.py.

The checker's claim is that the oracle registry cannot rot: not into a record
missing its revision, not into a second `primary`, and not into an AGENTS.md
table that names a different set of oracles than the directory holds. This file
PERFORMS each of those mutations against a synthetic registry and fails if the
checker shrugs one off, which is what makes the claim testable rather than
asserted.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-oracle-pins.py"
SPEC = importlib.util.spec_from_file_location("check_oracle_pins", CHECKER)
assert SPEC is not None and SPEC.loader is not None
check_oracle_pins = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_oracle_pins
SPEC.loader.exec_module(check_oracle_pins)


GOOD_SECONDARY = """# Fixture

```oracle-pin
id = fixture
role = secondary
upstream = https://github.com/example/fixture
scope = a path vLLM does not implement
pin = UNPINNED
pin_label = none
pinned_on = 2026-08-13
gateable = no
evidence = #647
```
"""

GOOD_PRIMARY = """# Fixture primary

```oracle-pin
id = vllm
role = primary
upstream = https://github.com/vllm-project/vllm
scope = everything vLLM implements
pin = 5559679229bc961848b121ccdeaa8fa5d79bec98
pin_label = 0.26.0.dev0
pinned_on = 2026-07-26
gateable = yes
evidence = .agents/upstream-sync.md
```
"""


PARITY_RECORD = """# Fixture primary, paired with PARITY_SYNC

```oracle-pin
id = vllm
role = primary
upstream = https://github.com/vllm-project/vllm
scope = everything vLLM implements
pin = 1111111111111111111111111111111111111111
pin_label = 9.9.9rc1.dev1
pinned_on = 2026-09-04
gateable = yes
evidence = AGENTS.md
```
"""

PARITY_SYNC = """# Fixture authority

```parity-pin
vllm_commit = 1111111111111111111111111111111111111111
vllm_runtime_version = 9.9.9rc1.dev1+g1111111111
vllm_distribution_version = 9.9.9rc1.dev1+g1111111111.precompiled
flashinfer_version = 0.6.18
```
"""

# A secondary oracle for the SCOPING case: its `pin` and `pin_label` name a
# revision and a release PARITY_SYNC never mentions, which is the ordinary
# state of all thirteen of them. Reconciling this record against the authority
# reports it, so a registry-wide rule cannot stay green over this fixture.
UNRECONCILED_SECONDARY = """# Fixture secondary, pinned to something else entirely

```oracle-pin
id = fixture
role = secondary
upstream = https://github.com/example/fixture
scope = a path vLLM does not implement
pin = 3333333333333333333333333333333333333333
pin_label = 8.8.8
pinned_on = 2026-09-04
gateable = no
evidence = #647
```
"""


def record(text: str, name: str = "fixture") -> check_oracle_pins.Record | None:
    """Parse one fixture, returning None when the block will not parse."""
    return check_oracle_pins.parse_record(Path(f"{name}.md"), text, [])


def errors_for(files: dict[str, str], agents_ids: list[str]) -> list[str]:
    """Run the whole registry check over a synthetic tree."""
    return check_oracle_pins.check_registry(
        {Path(f"{name}.md"): text for name, text in files.items()},
        agents_ids,
    )


class RecordParsingTests(unittest.TestCase):
    def test_wellformed_record_parses(self) -> None:
        parsed = record(GOOD_SECONDARY)
        assert parsed is not None
        self.assertEqual(parsed.fields["id"], "fixture")
        self.assertEqual(parsed.fields["gateable"], "no")

    def test_missing_block_is_reported(self) -> None:
        errs: list[str] = []
        self.assertIsNone(
            check_oracle_pins.parse_record(Path("fixture.md"), "# no block here\n", errs)
        )
        self.assertTrue(errs)

    def test_two_blocks_are_reported(self) -> None:
        errs: list[str] = []
        check_oracle_pins.parse_record(
            Path("fixture.md"), GOOD_SECONDARY + GOOD_SECONDARY, errs
        )
        self.assertTrue(any("exactly one" in e for e in errs))

    def test_every_required_key_is_load_bearing(self) -> None:
        # Dropping ANY single required key must be reported. A required-key list
        # that outgrows its enforcement is the failure this performs, not asserts.
        for key in check_oracle_pins.REQUIRED_KEYS:
            with self.subTest(key=key):
                mutated = "\n".join(
                    line
                    for line in GOOD_SECONDARY.splitlines()
                    if not line.startswith(f"{key} =")
                )
                errs = errors_for({"fixture": mutated}, ["fixture"])
                self.assertTrue(
                    any(key in e for e in errs),
                    f"dropping {key!r} was not reported: {errs}",
                )

    def test_empty_value_is_reported(self) -> None:
        mutated = GOOD_SECONDARY.replace("scope = a path vLLM does not implement", "scope =")
        errs = errors_for({"fixture": mutated}, ["fixture"])
        self.assertTrue(any("scope" in e for e in errs), errs)

    def test_unknown_key_is_reported(self) -> None:
        mutated = GOOD_SECONDARY.replace("gateable = no", "gateable = no\nnotes = smuggled")
        errs = errors_for({"fixture": mutated}, ["fixture"])
        self.assertTrue(any("notes" in e for e in errs), errs)


class RegistryConsistencyTests(unittest.TestCase):
    def test_clean_registry_passes(self) -> None:
        self.assertEqual(
            errors_for({"vllm": GOOD_PRIMARY, "fixture": GOOD_SECONDARY}, ["vllm", "fixture"]),
            [],
        )

    def test_id_must_match_filename(self) -> None:
        errs = errors_for({"renamed": GOOD_SECONDARY}, ["renamed"])
        self.assertTrue(any("filename" in e for e in errs), errs)

    def test_two_primaries_are_reported(self) -> None:
        second = GOOD_PRIMARY.replace("id = vllm", "id = usurper")
        errs = errors_for(
            {"vllm": GOOD_PRIMARY, "usurper": second}, ["vllm", "usurper"]
        )
        self.assertTrue(any("primary" in e for e in errs), errs)

    def test_missing_primary_is_reported(self) -> None:
        errs = errors_for({"fixture": GOOD_SECONDARY}, ["fixture"])
        self.assertTrue(any("primary" in e for e in errs), errs)

    def test_primary_must_be_vllm(self) -> None:
        usurper = GOOD_PRIMARY.replace("id = vllm", "id = sglang")
        errs = errors_for({"sglang": usurper}, ["sglang"])
        self.assertTrue(any("vllm" in e for e in errs), errs)

    def test_registry_entry_absent_from_agents_is_reported(self) -> None:
        errs = errors_for(
            {"vllm": GOOD_PRIMARY, "fixture": GOOD_SECONDARY}, ["vllm"]
        )
        self.assertTrue(any("AGENTS.md" in e and "fixture" in e for e in errs), errs)

    def test_agents_entry_absent_from_registry_is_reported(self) -> None:
        errs = errors_for({"vllm": GOOD_PRIMARY}, ["vllm", "ghost"])
        self.assertTrue(any("ghost" in e for e in errs), errs)


class GateabilityTests(unittest.TestCase):
    def test_gateable_yes_needs_a_real_pin(self) -> None:
        # An oracle cannot be gateable against a revision it does not name.
        mutated = (
            GOOD_PRIMARY.replace(
                "pin = 5559679229bc961848b121ccdeaa8fa5d79bec98", "pin = UNPINNED"
            )
        )
        errs = errors_for({"vllm": mutated}, ["vllm"])
        self.assertTrue(any("UNPINNED" in e for e in errs), errs)

    def test_gateable_yes_needs_evidence_that_exists(self) -> None:
        mutated = GOOD_PRIMARY.replace(
            "evidence = .agents/upstream-sync.md",
            "evidence = .agents/specs/this-file-does-not-exist.md",
        )
        errs = errors_for({"vllm": mutated}, ["vllm"])
        self.assertTrue(any("evidence" in e for e in errs), errs)

    def test_gateable_yes_may_not_cite_an_issue_instead_of_evidence(self) -> None:
        # "#123" is a promise of measurement, not a measurement.
        mutated = GOOD_PRIMARY.replace("evidence = .agents/upstream-sync.md", "evidence = #647")
        errs = errors_for({"vllm": mutated}, ["vllm"])
        self.assertTrue(any("evidence" in e for e in errs), errs)

    def test_gateable_no_must_name_the_owing_issue(self) -> None:
        mutated = GOOD_SECONDARY.replace("evidence = #647", "evidence = not measured yet")
        errs = errors_for({"vllm": GOOD_PRIMARY, "fixture": mutated}, ["vllm", "fixture"])
        self.assertTrue(any("issue" in e for e in errs), errs)

    def test_gateable_must_be_yes_or_no(self) -> None:
        mutated = GOOD_SECONDARY.replace("gateable = no", "gateable = partly")
        errs = errors_for({"vllm": GOOD_PRIMARY, "fixture": mutated}, ["vllm", "fixture"])
        self.assertTrue(any("gateable" in e for e in errs), errs)

    def test_pinned_on_must_be_an_iso_date(self) -> None:
        mutated = GOOD_SECONDARY.replace("pinned_on = 2026-08-13", "pinned_on = last tuesday")
        errs = errors_for({"vllm": GOOD_PRIMARY, "fixture": mutated}, ["vllm", "fixture"])
        self.assertTrue(any("pinned_on" in e for e in errs), errs)


class DeclaredOracleTests(unittest.TestCase):
    """A spec that opts into the `**Secondary oracle:**` syntax must name a real id."""

    def test_declared_id_must_be_registered(self) -> None:
        errs = check_oracle_pins.check_declarations(
            {Path("spec.md"): "**Secondary oracle:** `nonesuch`\n"},
            {"vllm", "sglang"},
        )
        self.assertTrue(any("nonesuch" in e for e in errs), errs)

    def test_registered_id_passes(self) -> None:
        self.assertEqual(
            check_oracle_pins.check_declarations(
                {Path("spec.md"): "**Secondary oracle:** `sglang`\n"},
                {"vllm", "sglang"},
            ),
            [],
        )

    def test_a_spec_that_never_declares_is_untouched(self) -> None:
        # No false positives on the hundreds of specs that mention llama.cpp or
        # SGLang in prose; only the opt-in syntax is enforced.
        self.assertEqual(
            check_oracle_pins.check_declarations(
                {Path("spec.md"): "we compared against llama.cpp and SGLang here\n"},
                {"vllm"},
            ),
            [],
        )


class ParityReconciliationTests(unittest.TestCase):
    """The vllm record's pin must agree with the AUTHORITY (#2829).

    Every case pairs a synthetic authority with a synthetic record, so nothing
    here moves when the real pin advances, and the expectation never comes from
    the file under test.
    """

    def reconcile(self, record_text: str, sync_text: str) -> list[str]:
        return check_oracle_pins.parity_errors(record_text, sync_text)

    def test_agreeing_surfaces_reconcile(self) -> None:
        self.assertEqual(self.reconcile(PARITY_RECORD, PARITY_SYNC), [])

    def test_a_stale_pin_is_reported(self) -> None:
        # THE #2829 SHAPE: the authority advanced and the record did not.
        mutated = PARITY_RECORD.replace(
            "pin = 1111111111111111111111111111111111111111",
            "pin = 2222222222222222222222222222222222222222",
        )
        errs = self.reconcile(mutated, PARITY_SYNC)
        self.assertTrue(any("vllm_commit" in e for e in errs), errs)

    def test_a_stale_label_is_reported(self) -> None:
        mutated = PARITY_RECORD.replace("pin_label = 9.9.9rc1.dev1", "pin_label = 9.8.0")
        errs = self.reconcile(mutated, PARITY_SYNC)
        self.assertTrue(any("pin_label" in e for e in errs), errs)

    def test_a_truncated_label_is_reported(self) -> None:
        # A PROPER PREFIX of the right label. `startswith` accepts this and the
        # decomposition rule refuses it; that difference is the whole reason
        # the rule is stated as equality against the public version.
        mutated = PARITY_RECORD.replace("pin_label = 9.9.9rc1.dev1", "pin_label = 9.9.9rc1.dev")
        errs = self.reconcile(mutated, PARITY_SYNC)
        self.assertTrue(any("pin_label" in e for e in errs), errs)

    def test_a_label_carrying_the_local_segment_is_reported(self) -> None:
        # The other direction: the full runtime string is not the label either.
        mutated = PARITY_RECORD.replace(
            "pin_label = 9.9.9rc1.dev1", "pin_label = 9.9.9rc1.dev1+g1111111111"
        )
        errs = self.reconcile(mutated, PARITY_SYNC)
        self.assertTrue(any("pin_label" in e for e in errs), errs)

    def test_a_label_with_no_local_segment_in_the_authority_still_reconciles(self) -> None:
        # A pin taken from a tagged build reports a bare public version, and
        # the rule degrades to plain equality rather than going red on it.
        sync = PARITY_SYNC.replace(
            "vllm_runtime_version = 9.9.9rc1.dev1+g1111111111",
            "vllm_runtime_version = 9.9.9rc1.dev1",
        )
        self.assertEqual(self.reconcile(PARITY_RECORD, sync), [])

    def test_a_missing_authority_block_fails_closed(self) -> None:
        # Not a skipped rule. A checker that stops checking when it cannot find
        # its expectation is a mute switch.
        errs = self.reconcile(PARITY_RECORD, "# the block was renamed\n")
        self.assertTrue(any("parity-pin" in e for e in errs), errs)

    def test_two_authority_blocks_fail_closed(self) -> None:
        errs = self.reconcile(PARITY_RECORD, PARITY_SYNC + PARITY_SYNC)
        self.assertTrue(any("exactly one" in e for e in errs), errs)

    def test_an_unparsable_authority_line_is_reported(self) -> None:
        sync = PARITY_SYNC.replace("flashinfer_version = 0.6.18", "flashinfer 0.6.18")
        errs = self.reconcile(PARITY_RECORD, sync)
        self.assertTrue(any("is not `key = value`" in e for e in errs), errs)

    def test_an_unknown_authority_key_is_reported(self) -> None:
        sync = PARITY_SYNC.replace("flashinfer_version = 0.6.18", "torch_version = 2.13.0")
        errs = self.reconcile(PARITY_RECORD, sync)
        self.assertTrue(any("torch_version" in e for e in errs), errs)

    def test_a_duplicate_authority_key_is_reported(self) -> None:
        sync = PARITY_SYNC.replace(
            "flashinfer_version = 0.6.18",
            "flashinfer_version = 0.6.18\nflashinfer_version = 0.6.15",
        )
        errs = self.reconcile(PARITY_RECORD, sync)
        self.assertTrue(any("duplicate" in e for e in errs), errs)

    def test_an_empty_authority_value_is_reported(self) -> None:
        sync = PARITY_SYNC.replace(
            "vllm_commit = 1111111111111111111111111111111111111111", "vllm_commit ="
        )
        errs = self.reconcile(PARITY_RECORD, sync)
        self.assertTrue(any("vllm_commit" in e for e in errs), errs)

    def test_an_unreadable_record_fails_closed(self) -> None:
        errs = self.reconcile("# no oracle-pin block here\n", PARITY_SYNC)
        self.assertTrue(any("unchecked" in e for e in errs), errs)

    def test_public_version_strips_only_the_local_segment(self) -> None:
        cases = {
            "0.28.1rc1.dev132+ge126687a9": "0.28.1rc1.dev132",
            "0.28.1rc1.dev132+ge126687a9.precompiled": "0.28.1rc1.dev132",
            "0.29.0": "0.29.0",
        }
        for given, expected in cases.items():
            with self.subTest(given=given):
                self.assertEqual(check_oracle_pins.public_version(given), expected)

    def test_the_live_surfaces_agree(self) -> None:
        # The reconciliation over the REAL two files, expectation from the
        # authority and value from the record, neither one a literal here.
        sync_text = check_oracle_pins.UPSTREAM_SYNC.read_text(encoding="utf-8")
        record_text = (check_oracle_pins.ORACLES / "vllm.md").read_text(encoding="utf-8")
        self.assertEqual(check_oracle_pins.parity_errors(record_text, sync_text), [])


class ParityThroughMainTests(unittest.TestCase):
    """The reconciliation is REACHED from the checker's entry point.

    Every other case here calls the rule directly, which proves the rule works
    and not that anything runs it. These three point `main` at a synthetic
    registry and assert on its return code, so deleting the call site in
    `main` turns this red -- the mutation AGENTS.md "Nothing lands dead" asks
    a reviewer to perform. The scoping case is here for the same reason: the
    scope is a property of `main`, so only `main` can be asked about it.
    """

    def agents_table(self, ids: tuple[str, ...]) -> str:
        """The AGENTS.md admissible-oracle table, naming exactly *ids*."""
        rows = "".join(f"| Fixture | `{oracle_id}` | a fixture |\n" for oracle_id in ids)
        return (
            "<!-- oracle-registry:begin -->\n"
            "| Oracle | Registry id | Reach for it when |\n"
            "|---|---|---|\n"
            f"{rows}"
            "<!-- oracle-registry:end -->\n"
        )

    def run_main(
        self, record_text: str, sync_text: str, others: dict[str, str] | None = None
    ) -> int:
        others = others or {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracles = root / "oracles"
            oracles.mkdir()
            (oracles / "vllm.md").write_text(record_text, encoding="utf-8")
            for stem, text in others.items():
                (oracles / f"{stem}.md").write_text(text, encoding="utf-8")
            (root / "upstream-sync.md").write_text(sync_text, encoding="utf-8")
            (root / "AGENTS.md").write_text(
                self.agents_table(("vllm", *others)), encoding="utf-8"
            )
            with mock.patch.multiple(
                check_oracle_pins,
                # ROOT too: `evidence` existence and the declaration glob are
                # both resolved against it, so a half-patched module would read
                # the real tree from inside a synthetic one.
                ROOT=root,
                ORACLES=oracles,
                AGENTS_MD=root / "AGENTS.md",
                UPSTREAM_SYNC=root / "upstream-sync.md",
                DECLARATION_ROOTS=(root,),
            ):
                return check_oracle_pins.main([])

    def test_main_passes_when_the_surfaces_agree(self) -> None:
        self.assertEqual(self.run_main(PARITY_RECORD, PARITY_SYNC), 0)

    def test_main_fails_when_the_surfaces_disagree(self) -> None:
        stale = PARITY_RECORD.replace(
            "pin = 1111111111111111111111111111111111111111",
            "pin = 2222222222222222222222222222222222222222",
        )
        self.assertEqual(self.run_main(stale, PARITY_SYNC), 1)

    def test_main_does_not_reconcile_a_secondary_oracle(self) -> None:
        # THE SCOPING TEST, and it has to go through `main`, because `main` is
        # where the scope lives: it selects the record to reconcile by filename
        # stem. `UNRECONCILED_SECONDARY` names a commit and a release the
        # authority never mentions, which is the ordinary state of the thirteen
        # secondary oracles, so a registry-wide version of this rule reports it
        # and returns 1. `check_registry` cannot be asked this question at
        # all, because `check_registry` never runs the parity rule. The live
        # `test_the_repository_registry_is_clean` does red under that same
        # mutation -- 26 parity errors over the 13 real secondaries -- but its
        # catch is incidental: it would evaporate if the registry ever held
        # `vllm.md` alone, and it reads whatever the tree happens to hold, so
        # it has no controlled fixture and cannot be the scoping test.
        self.assertEqual(
            self.run_main(
                PARITY_RECORD, PARITY_SYNC, {"fixture": UNRECONCILED_SECONDARY}
            ),
            0,
        )


class LiveRegistryTests(unittest.TestCase):
    def test_the_repository_registry_is_clean(self) -> None:
        self.assertEqual(check_oracle_pins.main([]), 0)

    def test_self_test_corpus_passes_in_both_directions(self) -> None:
        self.assertEqual(check_oracle_pins.main(["--self-test"]), 0)


if __name__ == "__main__":
    unittest.main()
