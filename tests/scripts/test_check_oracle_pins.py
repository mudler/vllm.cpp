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
import unittest
from pathlib import Path


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


class LiveRegistryTests(unittest.TestCase):
    def test_the_repository_registry_is_clean(self) -> None:
        self.assertEqual(check_oracle_pins.main([]), 0)

    def test_self_test_corpus_passes_in_both_directions(self) -> None:
        self.assertEqual(check_oracle_pins.main(["--self-test"]), 0)


if __name__ == "__main__":
    unittest.main()
