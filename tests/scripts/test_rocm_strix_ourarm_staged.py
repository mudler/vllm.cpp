#!/usr/bin/env python3
"""The staged gfx1151 paired measurement must refuse to run (#2497).

`.agents/scripts/rocm-strix-ourarm-staged.sh` carries the paired decode
measurement for `strix:gpu0` in full -- order-alternated rounds, leg count, clock
windows, the reference-tier assertion -- so that when our arm's token gate is
ratified the run costs one lease and not a design session.

Until then it may not run. `AGENTS.md` Gates admits a performance result from an
arm only after that arm's declared token-exact gate passes, and this arm's gate
reads `TOKEN_GATE=FAIL` at 3 of 6
(`docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`).
#2497 has already had one measurement retracted for being taken ahead of that
gate, so a staged script that starts on a bare invocation is the same defect
parked in a file.

WHAT THIS SUITE COMPARES AGAINST WHAT, in words, because a test that does not
narrate its own wiring cannot be audited: it invokes the staged script itself, in
a scratch working directory, with `STRIX_ARM_SPEED_RATIFIED_BY` set to each of
the values a careless caller would reach for, and requires a refusal from every
one of them. The green direction is exercised too, through `--guard-only`, which
exists so the ACCEPTING branch is reachable off the box; a guard that can only
ever be observed refusing is indistinguishable from a script that always exits
non-zero.

THE MUTATION THAT MATTERS deletes the guard in a scratch copy and requires the
unset case to stop refusing. A suite that still passes with the refusal removed
is measuring the file's existence.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".agents/scripts/rocm-strix-ourarm-staged.sh"

GUARD_BEGIN = "# --- RATIFICATION GUARD begin ---"
GUARD_END = "# --- RATIFICATION GUARD end ---"
REFUSAL_EXIT = 3
RATIFIED = "ratified 2026-09-02 by the operator on #2497 after the token gate passed"


def run(env_value: object, *args: str) -> subprocess.CompletedProcess:
    """Invoke the staged script in a scratch cwd with a chosen guard value.

    `env_value is None` means the variable is absent, which is a different case
    from the variable being present and empty; both are exercised.
    """
    env = dict(os.environ)
    env.pop("STRIX_ARM_SPEED_RATIFIED_BY", None)
    if env_value is not None:
        env["STRIX_ARM_SPEED_RATIFIED_BY"] = str(env_value)
    with tempfile.TemporaryDirectory() as scratch:
        proc = subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=scratch,
            timeout=120,
        )
        proc.scratch_entries = sorted(  # type: ignore[attr-defined]
            p.name for p in Path(scratch).iterdir()
        )
    return proc


class StagedScriptRefuses(unittest.TestCase):
    def test_script_is_present_and_executable(self) -> None:
        self.assertTrue(SCRIPT.is_file(), f"{SCRIPT} is missing")
        self.assertTrue(os.access(SCRIPT, os.X_OK), f"{SCRIPT} is not executable")

    def test_refuses_when_the_variable_is_absent(self) -> None:
        proc = run(None)
        self.assertEqual(
            proc.returncode,
            REFUSAL_EXIT,
            f"absent variable must refuse with {REFUSAL_EXIT}; got {proc.returncode}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}",
        )
        self.assertIn("REFUSED", proc.stdout + proc.stderr)

    def test_refuses_an_empty_value(self) -> None:
        proc = run("")
        self.assertEqual(proc.returncode, REFUSAL_EXIT, proc.stdout + proc.stderr)

    def test_refuses_a_value_that_names_no_decision(self) -> None:
        """`=1` and friends assert nothing. The variable must NAME the decision.

        The guard has two conditions and this suite exercises each one ALONE,
        because a population that only ever trips both tests them as one and a
        mutation deleting either condition then reads as a pass. Both mutations
        were run and both read NOT DETECTED until the values below were split:

          - LONG and naming nothing -- refused only by the issue-reference term
          - SHORT and carrying a reference -- refused only by the length floor.
            `#2497` is an issue number with no decision, no date and nobody
            attached to it, which is a citation and not a ratification.

        A THIRD split matters and was missing, which left a live mutant. The
        issue-reference term is `#[0-9]+`, and every long refused value above
        carried no digit at all -- so weakening it to `#?[0-9]+` left this whole
        suite green at 13 passed, while
        `STRIX_ARM_SPEED_RATIFIED_BY='ratified 2026-09-02 by the operator'` went
        from rc 3 to rc 0. The contract "naming means an issue reference" had
        silently degraded to "contains any digit", and no assertion could see
        it, because a date is the digit a real ratification always carries. The
        values marked below close that: LONG, carrying digits, carrying no `#`.
        """
        for value in (
            "1",
            "true",
            "yes",
            "ok",
            "ratified",
            "please",
            "yes it is fine to run this now",
            "the operator said it was ratified, honestly",
            "#1",
            "#2497",
            "see #2497",
            "ok #2497",
            # LONG, with digits, no `#`. These are what kill `#?[0-9]+`.
            "ratified 2026-09-02 by the operator",
            "ratified on 2026-09-02, token gate PASS",
            "approved 2026-09-02 by the operator, issue 2497",
            "token gate passed 6 of 6 on 2026-09-02",
            "1111111111111111",
        ):
            with self.subTest(value=value):
                proc = run(value)
                self.assertEqual(
                    proc.returncode,
                    REFUSAL_EXIT,
                    f"{value!r} must be refused; got {proc.returncode}\n{proc.stdout}",
                )

    def test_refusal_names_the_gate_that_is_failing(self) -> None:
        """A refusal a reader cannot act on sends them to guess."""
        text = run(None).stdout + run(None).stderr
        self.assertIn("token gate", text.lower())
        self.assertIn("qwen38-27b-q4km-rocm-gfx1151-token-gate-v2", text)
        self.assertIn("2497", text)
        self.assertIn("STRIX_ARM_SPEED_RATIFIED_BY", text)

    def test_refusal_writes_nothing(self) -> None:
        """A refused run must not have started staging anything."""
        proc = run(None)
        self.assertEqual(
            proc.scratch_entries,  # type: ignore[attr-defined]
            [],
            "a refused run created files in its working directory",
        )

    def test_a_named_decision_gets_past_the_guard(self) -> None:
        """The accepting branch is reachable, or the guard is untestable."""
        proc = run(RATIFIED, "--guard-only")
        self.assertEqual(
            proc.returncode,
            0,
            f"a well-formed ratification must pass the guard\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}",
        )
        self.assertIn("RATIFICATION_OK", proc.stdout)
        self.assertIn(RATIFIED, proc.stdout, "the guard must echo the decision it accepted")

    def test_guard_only_still_refuses(self) -> None:
        """`--guard-only` is a probe, never a bypass."""
        self.assertEqual(run(None, "--guard-only").returncode, REFUSAL_EXIT)


class StagedScriptShape(unittest.TestCase):
    """Static properties the run itself would be too expensive to prove."""

    def setUp(self) -> None:
        self.text = SCRIPT.read_text(encoding="utf-8")

    def test_the_guard_precedes_every_device_touch(self) -> None:
        """The refusal is the FIRST thing, before a path is read or a device opened."""
        end = self.text.index(GUARD_END)
        prologue = self.text[:end]
        for token in ("podman", "vllm-cli", "llama-bench", "/dev/kfd", "sha256sum"):
            self.assertNotIn(
                token,
                prologue,
                f"{token!r} appears before the ratification guard closes",
            )
        for token in ("podman", "vllm-cli", "llama-bench", "/dev/kfd"):
            self.assertIn(token, self.text, f"{token!r} is absent; this is not the paired job")

    def test_hsa_enable_sdma_is_never_set(self) -> None:
        """Retired by #2511. A partial mitigation of a symptom is not a knob."""
        for line in self.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertNotRegex(
                stripped,
                r"(^|\s|-e\s*[\"']?)HSA_ENABLE_SDMA\s*=",
                f"HSA_ENABLE_SDMA is set on a live line: {line}",
            )

    def test_the_design_is_declared_not_counted(self) -> None:
        """N comes from the design. A tee'd log reads every leg twice."""
        # re.MULTILINE must be COMPILED IN: assertRegex's third argument is the
        # failure message, so a flag passed there is silently a no-op.
        self.assertRegex(self.text, re.compile(r"^ROUNDS=\d+", re.MULTILINE))
        self.assertRegex(self.text, re.compile(r"^REPEAT=\d+", re.MULTILINE))

    def test_it_asserts_the_reference_tier_is_absent(self) -> None:
        """gfx1151 is integrated, so a silent host fallback is invisible in tok/s.

        The assertion is on the ENABLED form and not on the bare name. A
        substring test passes over `VT_OP_PROVIDER_STATS_DISABLED=1`, which is
        the mutation that turns the counter off while leaving the name in the
        file; it read NOT DETECTED against the looser assertion.
        """
        self.assertRegex(
            self.text,
            re.compile(r"-e VT_OP_PROVIDER_STATS=1\b"),
            "the reference-tier counter is not switched on for our arm's legs",
        )


class GuardMutation(unittest.TestCase):
    """Prove the suite detects the defect, rather than reading the file."""

    def test_deleting_the_guard_stops_the_refusal(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        begin = text.index(GUARD_BEGIN)
        end = text.index(GUARD_END) + len(GUARD_END)
        mutated = text[:begin] + "true\n" + text[end:]
        self.assertNotEqual(mutated, text, "the mutation did not apply")
        with tempfile.TemporaryDirectory() as scratch:
            victim = Path(scratch) / "mutated.sh"
            victim.write_text(mutated, encoding="utf-8")
            env = dict(os.environ)
            env.pop("STRIX_ARM_SPEED_RATIFIED_BY", None)
            work = Path(scratch) / "cwd"
            work.mkdir()
            proc = subprocess.run(
                ["bash", str(victim), "--guard-only"],
                capture_output=True,
                text=True,
                env=env,
                cwd=work,
                timeout=120,
            )
        self.assertNotEqual(
            proc.returncode,
            REFUSAL_EXIT,
            "the guard was deleted and the script still refused; "
            "this suite would pass over a removed refusal",
        )


if __name__ == "__main__":
    unittest.main()
