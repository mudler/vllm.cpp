#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-gate-commands.py.

A gate command that cannot fail collapses "done" into the implementer's opinion
of its own work. This classifier's only job is to tell a runnable command from
prose that looks like one.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gates = _load("check_gate_commands", "scripts/check-gate-commands.py")


class GatesSectionTests(unittest.TestCase):
    def test_finds_the_gates_heading_at_any_level(self):
        for heading in ("## Gates", "### Gates", "#### Gates and evidence"):
            text = f"# Spec\n\nintro\n\n{heading}\n\nrun `ctest -R foo`\n\n## Next\n\ntail\n"
            section = gates.gates_section(text)
            self.assertIsNotNone(section, heading)
            self.assertIn("ctest", section)
            self.assertNotIn("tail", section, "must stop at the next heading")

    def test_returns_none_when_there_is_no_gates_section(self):
        self.assertIsNone(gates.gates_section("# Spec\n\n## Scope\n\nnothing here\n"))

    def test_is_not_fooled_by_the_word_gates_in_prose(self):
        # "the gates are green" is not a section heading.
        self.assertIsNone(gates.gates_section("# Spec\n\nAll the gates are green.\n"))


class RunnableCommandTests(unittest.TestCase):
    def test_recognises_a_real_command(self):
        for body in [
            "run `ctest -R test_foo`",
            "`python3 scripts/check-agent-record.py`",
            "```\nbash scripts/agent-preflight.sh\n```",
            "`cmake --build build -j`",
            # The repo's MANDATED shape for a gate that touches the GPU. The
            # wrapper quotes the real command, so nothing else can reach it.
            "`flock /tmp/gpu -c 'ctest -R qwen36_paged_engine'`",
            # A built test binary, invoked with no arguments at all.
            "`./build-cuda-121a/tests/test_dropin_abi`",
        ]:
            self.assertTrue(gates.runnable_commands(body), body)

    def test_rejects_prose_that_merely_mentions_gating(self):
        for body in [
            "Correctness, e2e and performance gates apply.",
            "The SACRED gate must pass on GB10.",
            "`docs/BENCHMARKS.md`",
            # Every one of these is on the shipped record and was credited as a
            # runnable command. A backticked FILENAME is not a command, and a
            # tool name must be a whole word: `sha256_cbor` is not `sh`, and
            # `python@3.14` is not `python`.
            "`tests/vllm/models/test_model_registry.cpp`",
            "`tests/`",
            "`sha256_cbor`",
            "`python@3.14`",
            # The GPU lock IDIOM, named in three specs. A wrapper with nothing
            # to run is not a gate; a plain `flock` vocabulary entry credits
            # all three of those rows with a command they do not have.
            "`flock`",
            "`flock /tmp/gpu`",
        ]:
            self.assertEqual(gates.runnable_commands(body), [], body)
        # ...without rejecting a path that really is invoked.
        self.assertTrue(gates.runnable_commands("`scripts/check-agent-record.py`"))
        self.assertTrue(gates.runnable_commands("`./build/vllm-cli --model x`"))

    def test_rejects_a_command_that_cannot_fail(self):
        # These are the exact shapes the spec forbids: a Verify that always
        # succeeds turns "done" into an opinion.
        for body in ["`true`", "`echo ok`", "`:`", "`echo done && true`"]:
            self.assertEqual(gates.runnable_commands(body), [], body)
        # And they are rejected for the RIGHT reason. Without this, the loop
        # above passes vacuously on any implementation that simply fails to
        # recognise `true` as a command at all, and the cannot-fail rule --
        # the point of this classifier -- is pinned by nothing.
        for candidate in ("true", "echo ok", ":", "echo done && true"):
            self.assertTrue(gates.is_command(candidate), candidate)

    def test_rejects_a_piped_command(self):
        # `cmd | tail` reports tail's exit status, so the gate cannot fail.
        self.assertEqual(gates.runnable_commands("`ctest -R foo | tail -5`"), [])


class ShippedRecordTests(unittest.TestCase):
    def test_the_audit_covers_every_gated_state(self):
        self.assertEqual(
            gates.GATED_STATES,
            frozenset({"READY", "ACTIVE", "GATING", "BLOCKED"}),
        )
        # ...and audit() must actually FILTER on it. Asserting the constant's
        # literal value pins nothing about the denominator: deleting the state
        # filter in audit() leaves every other assertion in this file green
        # while the report goes from 97 rows to 726. Task 4 ratchets on that
        # number, so it is pinned here.
        audited = gates.audit()
        self.assertTrue(audited)
        self.assertLessEqual({item["state"] for item in audited}, gates.GATED_STATES)
        # And the filter is only load-bearing if the matrices really do carry
        # rows it excludes -- otherwise the assertion above is vacuous.
        on_record = set()
        for path in gates.AUDITED_MATRIX_PATHS:
            for row in gates.record.parse_claim_rows(path, []):
                on_record.add(row.state)
        self.assertTrue(on_record - gates.GATED_STATES, "filter excludes nothing")

    def test_the_six_lifecycle_matrices_are_audited(self):
        names = {p.name for p in gates.AUDITED_MATRIX_PATHS}
        self.assertIn("feature-matrix.md", names)
        self.assertEqual(len(names), 6)
        # The LIST length too, not just the set of names. If check-agent-record's
        # MATRIX_PATHS ever gains the one appended here, audit() parses that file
        # twice and double-counts every row in it -- the denominator moving
        # silently again, which a set comparison reads as still 6.
        self.assertEqual(len(gates.AUDITED_MATRIX_PATHS), 6)

    def test_sglang_is_excluded_and_the_exclusion_is_justified(self):
        # It was listed and contributed 0 rows: "audited in name only", an
        # absence that reads as a pass (audit artifact risk 6). Dropping it is
        # only honest if it genuinely has no gated rows, so pin THAT rather than
        # the bare absence. sglang-matrix.md carries a CLASSIFICATION column
        # (FUSED / INVENTORIED / NOT-APPLICABLE), not a lifecycle state, so the
        # row parser recognises nothing in it -- and reports no error either.
        # If it ever gains real lifecycle rows this goes red, and the matrix must
        # come back into the audited set rather than stay silently skipped.
        names = {p.name for p in gates.AUDITED_MATRIX_PATHS}
        self.assertNotIn("sglang-matrix.md", names)
        sglang = gates.record.AGENTS / "sglang-matrix.md"
        self.assertTrue(sglang.is_file())
        errors: list[str] = []
        self.assertEqual(gates.record.parse_claim_rows(sglang, errors), [])
        self.assertEqual(errors, [])

    def test_every_record_carries_a_known_verdict(self):
        known = {"runnable", "gates-no-command", "no-gates-section", "no-spec"}
        records = gates.audit()
        self.assertTrue(records)
        for item in records:
            self.assertIn(item["verdict"], known)

    def test_a_matrix_that_does_not_parse_is_a_hard_failure(self):
        # audit() used to build a fresh `errors` list per matrix, hand it to
        # parse_claim_rows and never read it. Stripping rows from a matrix then
        # surfaced as "these baseline rows left the gated population ... re-pin
        # RUNNABLE_BASELINE" -- a parse FAILURE wearing the face of a legitimate
        # record edit, recommending the one action the audit forbids doing
        # blindly. Every mode must go red, --json included.
        original = gates.record.parse_claim_rows

        def broken(path, errors):
            rows = original(path, errors)
            errors.append(f"{path.name}:1: SOME-ROW has 4 cells; header has 6")
            return rows

        gates.record.parse_claim_rows = broken
        try:
            with self.assertRaises(gates.RecordParseError):
                gates.audit()
            noise = io.StringIO()
            with contextlib.redirect_stderr(noise), contextlib.redirect_stdout(noise):
                statuses = [
                    gates.main(argv)
                    for argv in ([], ["--json"], ["--check"], ["--json", "--check"])
                ]
            self.assertEqual(statuses, [1, 1, 1, 1])
        finally:
            gates.record.parse_claim_rows = original

    def test_a_parse_failure_never_reads_as_a_legitimate_record_edit(self):
        # The message is the finding: a broken matrix must not be reported in
        # the words that describe a row correctly leaving the population, and
        # must not steer the reader toward re-pinning the baseline.
        original = gates.record.parse_claim_rows

        def broken(path, errors):
            errors.append(f"{path.name}:1: SOME-ROW must have exactly one canonical state")
            return []

        gates.record.parse_claim_rows = broken
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stderr(buffer):
                self.assertEqual(gates.main(["--check"]), 1)
        finally:
            gates.record.parse_claim_rows = original
        message = buffer.getvalue()
        self.assertIn("did not PARSE", message)
        self.assertIn("must have exactly one canonical state", message)
        self.assertNotIn("left the gated population", message)
        self.assertNotIn("re-pin RUNNABLE_BASELINE in the", message)

    def test_report_mode_exits_zero_even_with_debt(self):
        # 67 of 97 rows cannot state a command today. Report mode must still
        # exit 0 -- the ratchet is step 4, after the debt is recorded.
        self.assertEqual(gates.main([]), 0)


def _bash_array(text: str, name: str) -> list[str]:
    """The entries of a `NAME=(\n ... \n)` array in a bash script.

    Membership in the ARRAY, never a substring of the whole file: preflight
    mentions `check-gate-commands` twice -- once in `CHECKERS`, once in the
    `case` branch that adds `--check` -- so a substring test stays green with
    the CHECKERS entry deleted. That is this repo's recorded defect class (a
    substring `--grep` crediting a row with another row's commits), and it
    would hide the exact mutation step 6 requires to go red.
    """
    match = re.search(rf"(?m)^{re.escape(name)}=\($(.*?)^\)$", text, re.S)
    assert match is not None, f"{name}=( ... ) not found"
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


class RatchetTests(unittest.TestCase):
    def test_the_container_rows_credit_rests_on_scripts_that_exist(self):
        """Why ENG-RELEASE-CONTAINERS is in the baseline, not merely that it is.

        The row was credited on arrival because its spec binds the release chain
        it inherits -- scripts that exist and genuinely fail on a broken staged
        tree. Its OWN gates did not exist when it was pinned. If the credit ever
        came to rest only on scripts the tree does not have, the pin would be a
        certificate for nothing, and this is the case that catches that.
        """
        spec = ROOT / ".agents/specs/container-images.md"
        self.assertTrue(spec.exists(), "the container spec is the source of the credit")
        commands = gates.runnable_commands(spec.read_text(encoding="utf-8"))
        existing = sorted({c for c in commands if (ROOT / c).exists()})
        self.assertTrue(
            existing,
            "every command the container spec cites is absent from the tree; "
            "the runnable credit rests on nothing",
        )
        self.assertIn("scripts/validate-release-archive.py", existing)

    def test_the_baseline_matches_the_shipped_record(self):
        # EXACT equality, in both directions, and that is the whole contract:
        # this is an exact pin, not a shrink-only floor. Lowering the baseline
        # to make a red gate green goes red here, which is the point -- and so
        # does GROWTH. Add a real gate command to a row's spec and `--check`
        # stays 0 while this assertion, preflight and CI go red until the set
        # below is re-pinned. Growth is welcome; silent growth is not.
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_now_derived_left_the_gated_population_cleanly(self):
        # ENG-NOW-DERIVED (#374) shipped W1-W5 and reached DONE. Closure removes
        # it from both sides of the exact pin; it is not assigned a weaker
        # verdict and its implementation evidence remains in the record.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertIsNone(verdicts.get("ENG-NOW-DERIVED"))
        self.assertNotIn("ENG-NOW-DERIVED", gates.RUNNABLE_BASELINE)

    def test_re_adding_done_to_the_gated_population_breaks_the_pin(self):
        # MUTATION: restoring the departed lifecycle state must expose the four
        # runnable DONE rows and disagree with the re-pinned baseline.
        original = gates.GATED_STATES
        gates.GATED_STATES = frozenset(original | {"DONE"})
        try:
            runnable = {
                r["id"] for r in gates.audit() if r["verdict"] == "runnable"
            }
        finally:
            gates.GATED_STATES = original
        departed = {
            "ENG-ASYNC-SCHED",
            "SERVE-HTTP-TRANSPORT",
            "ENG-NOW-DERIVED",
            "ENG-TRAILER-MERGE-ARTIFACTS",
        }
        self.assertEqual(runnable - set(gates.RUNNABLE_BASELINE), departed)
        self.assertNotEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_trailer_merge_artifacts_left_the_gated_population_cleanly(self):
        # Credited runnable on arrival at ACTIVE, then gone the same day on
        # reaching DONE (157080c8). Assert the departure on BOTH sides -- gone
        # from the audit AND gone from the baseline -- because a row present in
        # one and not the other is exactly what the exact pin exists to catch.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertIsNone(verdicts.get("ENG-TRAILER-MERGE-ARTIFACTS"))
        self.assertNotIn("ENG-TRAILER-MERGE-ARTIFACTS", gates.RUNNABLE_BASELINE)

    def test_forge_coauthor_is_credited_for_real_commands(self):
        # ENG-FORGE-COAUTHOR (#418) joins the runnable population on arrival, so
        # it earns the credit the same way: its spec's Gates section must name
        # commands that can actually fail, including the per-commit re-check of
        # the real f64f2b71 the row exists to unblock.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-FORGE-COAUTHOR"), "runnable")
        spec = (ROOT / ".agents/specs/forge-coauthor-attribution.md").read_text(
            encoding="utf-8"
        )
        for command in ("scripts/agent-preflight.sh", "agent-integration.py"):
            with self.subTest(command=command):
                self.assertIn(command, spec)

    def test_record_conflict_surfaces_is_credited_for_real_commands(self):
        # ENG-RECORD-CONFLICT-SURFACES (#364) joined the runnable population on
        # arrival, so the credit has to be earned the same way ENG-DOCS-SITE
        # earns it: the spec's Gates section must name commands that can
        # actually fail, not prose. This row's gate is the record gate -- no
        # CUDA, GPU or SACRED gate is implicated because no product source is
        # touched, and the spec says so rather than leaving the absence
        # unexplained.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-RECORD-CONFLICT-SURFACES"), "runnable")
        spec = (ROOT / ".agents/specs/retire-shared-record-surfaces.md").read_text(
            encoding="utf-8"
        )
        for command in ("scripts/agent-preflight.sh", "agent-integration.py"):
            with self.subTest(command=command):
                self.assertIn(command, spec)

    def test_serve_recipe_args_is_credited_for_real_commands(self):
        # SERVE-RECIPE-ARGS (#606) enters the runnable population by leaving
        # SPIKE for ACTIVE, which is the first state that puts it in
        # GATED_STATES at all. The credit is earned the same way the two rows
        # above earn it: the spec's Gates section names commands that can
        # actually fail. The second assertion is the one specific to this row --
        # its gate is CPU-only by construction, so the spec must say why no GPU
        # or oracle leg is implicated rather than leaving the absence to be read
        # as an omission.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("SERVE-RECIPE-ARGS"), "runnable")
        spec = (ROOT / ".agents/specs/serve-recipe-args.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/agent-preflight.sh", spec)
        self.assertIn("No GPU, no oracle run", spec)

    def test_dropping_serve_recipe_args_from_the_pin_breaks_it(self):
        # MUTATION, in the direction this re-pin actually moved: the entry added
        # for #606 must be what keeps the exact pin agreeing with the audit.
        # Remove it and the equality assertion above has to go red, which is
        # what proves the row was pinned because it entered the population and
        # not to quiet a gate.
        reduced = set(gates.RUNNABLE_BASELINE) - {"SERVE-RECIPE-ARGS"}
        self.assertNotEqual(reduced, set(gates.RUNNABLE_BASELINE))
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(runnable, reduced)
        self.assertEqual(runnable - reduced, {"SERVE-RECIPE-ARGS"})

    def test_the_baseline_re_pin_is_load_bearing(self):
        # MUTATION: the re-pin that added this row must be what makes the audit
        # agree with the baseline. Drop the entry and the exact-pin assertion
        # above has to go red -- otherwise the baseline is decorative and a row
        # could enter or leave the runnable population unnoticed, which is the
        # failure the exact pin exists to catch.
        reduced = set(gates.RUNNABLE_BASELINE) - {"ENG-RECORD-CONFLICT-SURFACES"}
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertNotEqual(
            runnable,
            reduced,
            "removing the row from the baseline must break the pin",
        )
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_eng_docs_site_is_credited_for_real_commands(self):
        # ENG-DOCS-SITE joined the runnable population on arrival rather than
        # being parked as gates-no-command, so the credit has to be earned by
        # the spec actually naming commands that can fail. Pin that: the row is
        # runnable AND its Gates section carries the two that do the work. A
        # spec rewritten into prose gates goes red here rather than silently
        # keeping a credit it no longer deserves.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("ENG-DOCS-SITE"), "runnable")

        spec = ROOT / ".agents" / "specs" / "gh-pages-docs-site.md"
        body = spec.read_text(encoding="utf-8")
        self.assertIn("python3 scripts/check-site.py", body)
        self.assertIn("hugo --minify -s website", body)

    def test_the_newly_pinned_row_carries_a_reproducible_invocation(self):
        # The baseline is hand-edited and `audit()` is not, so the failure mode a
        # re-pin invites is adding an id whose spec does not actually tell anyone
        # how to run the gate. Set-equality above cannot catch that: both sides
        # move together.
        #
        # It is worth being precise about what this does and does not prove. The
        # obvious mutation -- strip the fenced block, expect the credit to vanish
        # -- does NOT hold, and finding that out is the useful part: with both
        # blocks removed the section still yields `ctest`, harvested from an
        # inline code span in the surrounding prose. That is the weak-credit debt
        # the checker's own header already admits to (see the RUNNABLE_BASELINE
        # comment and gate-command-audit-2026-08-06.md risk 3), not something
        # this row introduced, so it is recorded here rather than worked around.
        #
        # What is asserted instead is the thing a reader actually needs: the row
        # is pinned, it audits runnable, and its Gates section carries a real
        # fenced invocation naming the binaries the gate was run with.
        row = "SAMPLE-PROMPT-LOGPROBS"
        self.assertIn(row, gates.RUNNABLE_BASELINE)
        record = next(r for r in gates.audit() if r["id"] == row)
        self.assertEqual(record["verdict"], "runnable", record)

        section = gates.gates_section(
            (ROOT / ".agents/specs/prompt-logprobs.md").read_text(encoding="utf-8")
        )
        self.assertIsNotNone(section)
        blocks = re.findall(r"```sh\n(.*?)```", section, flags=re.DOTALL)
        self.assertTrue(blocks, "the Gates section carries no fenced invocation")
        body = "\n".join(blocks)
        for fragment in ("cmake -S . -B", "cmake --build", "test_llm_engine", "ctest"):
            self.assertIn(fragment, body, fragment)

    def test_lora_runtime_is_credited_for_its_own_invocation(self):
        # LORA-RUNTIME re-entered the runnable population when the row went back
        # to ACTIVE for W2 (#278), and the credit it arrived with was WEAK: the
        # UPSTREAM path `tests/lora/test_qwen35_densemodel_lora.py`, named in
        # the spec's prose as the eventual model gate, which nothing here runs.
        # The pin was taken only alongside the row's real CPU invocation, so
        # assert the real one is what the spec carries. Delete the ctest lines
        # and this goes red, instead of the row quietly keeping a credit that
        # rests on an upstream filename.
        verdicts = {r["id"]: r["verdict"] for r in gates.audit()}
        self.assertEqual(verdicts.get("LORA-RUNTIME"), "runnable")

        spec = ROOT / ".agents" / "specs" / "lora-adapter.md"
        section = gates.gates_section(spec.read_text(encoding="utf-8"))
        self.assertIsNotNone(section)
        commands = gates.runnable_commands(section)
        self.assertIn(
            "ctest --test-dir build-cpu -R test_lora_layers --output-on-failure",
            commands,
        )
        self.assertIn(
            "ctest --test-dir build-cpu -R test_punica_cpu --output-on-failure",
            commands,
        )

    def test_a_row_that_loses_its_command_is_refused(self):
        # Still present, still gated, no longer runnable -- a real regression.
        victim = sorted(gates.RUNNABLE_BASELINE)[0]
        records = [{"verdict": "gates-no-command", "id": victim, "state": "READY",
                    "path": "p", "line": 1, "detail": "d"}]
        errors = gates.ratchet_errors(records)
        self.assertTrue(errors)
        self.assertIn(victim, errors[0])
        self.assertIn("Repair the row", errors[0])

    def test_a_row_that_left_the_population_reports_differently(self):
        # Deleted or transitioned out: legitimate, but must re-pin. The two
        # cases MUST be distinguishable -- that is why the baseline is a set.
        errors = gates.ratchet_errors([])
        self.assertTrue(errors)
        self.assertTrue(any("left the gated population" in e for e in errors))
        self.assertFalse(any("Repair the row" in e for e in errors))

    def test_a_lost_row_and_a_departed_row_are_reported_separately(self):
        # The two cases in ONE run, which is the only arrangement that proves
        # they are distinguishable rather than merely differently worded. The
        # two tests above each see a single case, so both stay green if the
        # `& present` split is deleted and every drop is called a loss; here
        # the departed row would then be named in the "Repair the row" message
        # and the assertion below goes red.
        ordered = sorted(gates.RUNNABLE_BASELINE)
        self.assertGreaterEqual(len(ordered), 2)
        lost, departed = ordered[0], ordered[1]
        records = [{"verdict": "gates-no-command", "id": lost, "state": "READY",
                    "path": "p", "line": 1, "detail": "d"}]
        records += [
            {"verdict": "runnable", "id": rid, "state": "READY",
             "path": "p", "line": 2, "detail": "d"}
            for rid in ordered[2:]
        ]
        errors = gates.ratchet_errors(records)
        self.assertEqual(len(errors), 2, errors)
        loss_msg = [e for e in errors if "Repair the row" in e]
        gone_msg = [e for e in errors if "left the gated population" in e]
        self.assertEqual(len(loss_msg), 1, errors)
        self.assertEqual(len(gone_msg), 1, errors)
        self.assertIn(lost, loss_msg[0])
        self.assertNotIn(departed, loss_msg[0])
        self.assertIn(departed, gone_msg[0])
        self.assertNotIn(lost, gone_msg[0])

    def test_an_improvement_is_allowed(self):
        records = [
            {"verdict": "runnable", "id": rid, "state": "READY",
             "path": "p", "line": 1, "detail": "d"}
            for rid in sorted(gates.RUNNABLE_BASELINE)
        ] + [{"verdict": "runnable", "id": "NEW-ROW", "state": "READY",
              "path": "p", "line": 2, "detail": "d"}]
        self.assertEqual(gates.ratchet_errors(records), [])

    def test_check_mode_passes_on_the_shipped_record(self):
        # The gate ships GREEN. It was wired after the debt was recorded, so it
        # never had to be relaxed to pass.
        self.assertEqual(gates.main(["--check"]), 0)

    def test_check_mode_is_never_silently_swallowed_by_json(self):
        # --json used to be the only mode, and it returns 0 unconditionally. If
        # it short-circuits ahead of --check, `--json --check` is a gate that
        # CANNOT FAIL -- the precise shape this classifier exists to detect,
        # wearing this classifier's own face.
        self.assertEqual(gates.main(["--json", "--check"]), 0)
        original = gates.RUNNABLE_BASELINE
        try:
            gates.RUNNABLE_BASELINE = frozenset(original | {"ROW-THAT-IS-NOT-THERE"})
            self.assertEqual(gates.main(["--check"]), 1)
            self.assertEqual(gates.main(["--json", "--check"]), 1)
        finally:
            gates.RUNNABLE_BASELINE = original

    def test_the_checker_is_wired_into_preflight_and_ci(self):
        preflight = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        self.assertIn("check-gate-commands", _bash_array(preflight, "CHECKERS"))
        self.assertIn("test_check_gate_commands", _bash_array(preflight, "SUITES"))
        # ...and dispatched WITH --check. Report mode exits 0 whatever the
        # record says, so a CHECKERS entry without the flag installs a gate
        # that cannot fail.
        self.assertRegex(
            preflight, r"(?m)^\s*[\w|-]*check-gate-commands\)\s+run\b.*--check"
        )
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/check-gate-commands.py --check", ci)
        self.assertIn("tests/scripts/test_check_gate_commands.py", ci)


if __name__ == "__main__":
    unittest.main()
