#!/usr/bin/env python3
"""Every value the LTX-2.5 A/B harnesses set for `VLLM_LTX2_DIT_FLASH_ATTN` is a
rung the dispatch actually parses, and it is the rung the arm's own label claims.

#1751, row `LTX25-DIT-ATTN-ARM-PARSE`.

WHY THIS SUITE EXISTS. `src/vllm/model_executor/models/ltx2_device.cpp` reads one
environment variable to pick which attention op the DiT self-attention calls, and
three committed harnesses set it. The variable's ACCEPTED VALUES HAVE ALREADY
CHANGED ONCE under those harnesses: at #1549 the knob was binary, `=0` selected
`vt::Attention` and every other value selected `vt::AttentionDenseFlash`; #1551
made it three-way, moved the unset default up to `vt::AttentionDenseFa2` and gave
the flash rung the exact spelling `flash`. Two harnesses were left behind by that
rename and kept exporting the #1549 spellings, so each ran an arm it did not name:

  * `ltx25-dit-attn-flash-pixel-ab.sh` set `=1` for `flash` and `flash-ctl`.
  * `ltx25-dit-attn-flash-ab.sh` left the variable UNSET for its `flash` arm.

Neither was a typo. Both were true when they were written, and both became false
in a commit that touched neither file. That is what makes it a gate's job rather
than a reviewer's: nothing in this tree connected the literal in a shell script
to the literal in the C++ dispatch, so the two drifted apart in silence.

WHAT IT COMPARES, AND WHY THAT IS NOT A TRANSCRIPTION. The accepted set is READ
out of the dispatch's `std::strcmp(arm, "...")` calls rather than written here,
so a fourth arm or a renamed one moves this suite's expectation with it. The used
set is READ out of each harness's own arm invocations. Neither side is copied
into this file; what this file holds is the requirement that the second is a
subset of the first, and that each arm's LABEL agrees with the rung it selects.

THE LABEL HALF IS THE HALF THAT CATCHES THE REAL DEFECT. A value can be perfectly
valid and still be the wrong arm: `unset` is an accepted value and it is what the
flash arm of `ltx25-dit-attn-flash-ab.sh` exported for a whole row, while that
harness's phase [F] only PRINTS the op-provider selections and asserts nothing
about them. Subset-alone would have called that green.

It reads text and runs nothing, so it needs no GPU, no lease and no toolchain.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DISPATCH = ROOT / "src/vllm/model_executor/models/ltx2_device.cpp"

# The one spelling that is not a string literal in the dispatch: absence. It is
# spelled "" at a harness call site, because both `run_arm` helpers treat an
# empty third argument as `unset VLLM_LTX2_DIT_FLASH_ATTN`.
UNSET = ""

# label stem -> the rung that label claims. Read off the harnesses' own headers,
# which name what each arm is: `flash`/`flash-ctl` are the `vt::AttentionDenseFlash`
# rung, `naive` is `vt::Attention`, `fa2` is the unset `vt::AttentionDenseFa2`
# default. An arm whose label matches no stem is REFUSED rather than skipped --
# see `test_every_arm_label_is_one_this_suite_knows`.
LABEL_TO_ARM = {
    "flash": "flash",
    "flash-ctl": "flash",
    "naive": "0",
    "fa2": UNSET,
    "fa2-ctl": UNSET,
}


def dispatch_code() -> str:
    """`ltx2_device.cpp` with its `//` comments removed.

    THE ONE READER OF THE DISPATCH, and it has to be one. Both things this suite
    derives from that file -- the accepted set below and the refusal tripwires in
    `TheDispatchRefusesAFourthValue` -- must agree on what counts as code, because
    they check the two halves of one claim: that the dispatch matches these values
    and refuses everything else. Read the raw file for the first and stripped text
    for the second, and a comment quoting `std::strcmp(arm, "X")` widens the
    accepted set while the tripwires still read the real dispatch, so the subset
    half admits a value the code rejects. The dispatch's own comment QUOTES the
    defective `arm[0] == '0'` it replaced, so comments here do carry code-shaped
    text, and stripping them is what keeps the record of the defect writable.
    """
    return re.sub(r"//[^\n]*", "", DISPATCH.read_text(encoding="utf-8"))


def accepted_values() -> set[str]:
    """The values the dispatch matches EXACTLY, read from its own `strcmp` calls."""
    literals = set(re.findall(r'std::strcmp\(arm,\s*"([^"]*)"\)', dispatch_code()))
    # THE INSTRUMENT'S OWN PRECONDITION. A regex that matched nothing would make
    # every subset assertion below vacuously true, which is a passing suite over
    # an unread file. Two is the number of named arms the knob has ever had; if
    # the dispatch is restructured so this cannot read it, that is a finding.
    assert len(literals) >= 2, (
        f"read {len(literals)} strcmp arm literals from {DISPATCH}; this suite "
        "cannot check a subset it could not derive"
    )
    return literals | {UNSET}


def arms_of(path: Path, pattern: str, want: int) -> list[tuple[str, str]]:
    """(label, knob) for each arm invocation in `path`, with a count precondition."""
    text = path.read_text(encoding="utf-8")
    found = [(m.group("label"), m.group("knob")) for m in re.finditer(pattern, text, re.M)]
    # SAME PRECONDITION, per harness. A restructured harness that this pattern no
    # longer matches must fail here rather than report zero arms and pass.
    assert len(found) == want, (
        f"{path.name}: matched {len(found)} arm invocations, expected {want}: {found}"
    )
    return found


HARNESSES = {
    # `render <label> "<knob>" <timeout>` -- the knob is QUOTED here as it is in
    # the two `run_arm` harnesses below, because the fa2 rung's value is the EMPTY
    # STRING and an unquoted `\S+` cannot express it. Before the fa2 arms landed
    # the pixel harness wrote its knobs bare and this pattern read them bare; one
    # spelling across the three files is what lets one reviewer check all three.
    "ltx25-dit-attn-flash-pixel-ab.sh": (
        r'^render\s+(?P<label>\S+)\s+"(?P<knob>[^"]*)"\s+"\$\{TMO_',
        4,
    ),
    # `run_arm <label> <timeout> "<knob>"`
    "ltx25-dit-attn-flash-ab.sh": (
        r'^run_arm\s+(?P<label>\S+)\s+"\$\{TMO_[^"]*"\s+"(?P<knob>[^"]*)"',
        2,
    ),
    # the same shape inside a `case` arm, so it is not line-anchored
    "ltx25-dit-attn-fa2-hd128-ab.sh": (
        r'run_arm\s+(?P<label>\S+)\s+"\$\{TMO_[^"]*"\s+"(?P<knob>[^"]*)"',
        3,
    ),
}


def all_arms() -> list[tuple[str, str, str]]:
    """(harness, label, knob) over every committed LTX-2.5 attention A/B arm."""
    out: list[tuple[str, str, str]] = []
    for name, (pattern, want) in HARNESSES.items():
        for label, knob in arms_of(ROOT / "scripts" / name, pattern, want):
            out.append((name, label, knob))
    return out


class TheHarnessesSetArmsTheDispatchParses(unittest.TestCase):
    def test_the_harness_files_all_exist(self) -> None:
        for name in HARNESSES:
            self.assertTrue((ROOT / "scripts" / name).is_file(),
                            f"scripts/{name} is gone; this suite would silently check nothing")
        self.assertTrue(DISPATCH.is_file(), f"{DISPATCH} is gone")

    def test_every_arm_value_is_one_the_dispatch_accepts(self) -> None:
        """The subset half. Since #1751 an unrecognised value REFUSES at the first
        DiT forward, so a stale literal here costs a lease rather than producing a
        wrong number -- but it still costs the lease, and this catches it on a
        laptop instead."""
        accepted = accepted_values()
        for harness, label, knob in all_arms():
            with self.subTest(harness=harness, label=label):
                self.assertIn(
                    knob, accepted,
                    f"scripts/{harness} arm {label!r} exports "
                    f"VLLM_LTX2_DIT_FLASH_ATTN={knob!r}, which "
                    f"{DISPATCH.name} does not parse (it accepts "
                    f"{sorted(accepted)!r}, where '' means unset)")

    def test_every_arm_label_is_one_this_suite_knows(self) -> None:
        """A new label must be MAPPED, never silently unchecked. Without this, an
        arm added under a name `LABEL_TO_ARM` has no entry for would skip the
        assertion below and read as covered."""
        for harness, label, _ in all_arms():
            with self.subTest(harness=harness, label=label):
                self.assertIn(label, LABEL_TO_ARM,
                              f"scripts/{harness} has an arm labelled {label!r} that this "
                              "suite has no rung for; add it to LABEL_TO_ARM")

    def test_every_arm_selects_the_rung_its_label_claims(self) -> None:
        """The half that catches the defect the subset half cannot see: a value
        that is accepted and is still the wrong arm."""
        for harness, label, knob in all_arms():
            with self.subTest(harness=harness, label=label):
                want = LABEL_TO_ARM[label]
                self.assertEqual(
                    knob, want,
                    f"scripts/{harness} arm {label!r} exports "
                    f"VLLM_LTX2_DIT_FLASH_ATTN={knob!r} but its label claims the "
                    f"{want!r} rung ('' means unset). An arm that names one rung and "
                    "runs another makes an A/B measure a kernel against itself.")

    def test_at_least_one_arm_of_each_rung_is_still_reachable(self) -> None:
        """The A/B's whole premise: all three rungs run from ONE binary. If the
        committed harnesses between them stop naming a rung, that rung's recorded
        denominator is no longer reproducible from this tree."""
        used = {knob for _, _, knob in all_arms()}
        for rung in sorted(accepted_values()):
            with self.subTest(rung=rung or "<unset>"):
                self.assertIn(rung, used,
                              f"no committed harness arm selects {rung!r} any more")


class TheDispatchRefusesAFourthValue(unittest.TestCase):
    """TEXT TRIPWIRES, and labelled as such. The EXECUTABLE gate on the refusal is
    `tests/vllm/models/test_ltx2_device.cpp`'s "an unrecognised
    VLLM_LTX2_DIT_FLASH_ATTN value is REFUSED by name", which runs nine values
    through the production forward. These two assertions exist because this suite
    derives the accepted set from the dispatch's shape, and would keep passing on
    a dispatch that had gone back to a prefix test or a silent fall-through."""

    def setUp(self) -> None:
        # CODE ONLY, through the SAME `dispatch_code` helper `accepted_values`
        # uses, so the two readers cannot drift apart again. `//` comments are
        # stripped before either assertion runs, because the dispatch's own
        # comment QUOTES the defective `arm[0] == \'0\'` it replaced -- that is the
        # record of why the arm reads the way it does, and a tripwire that fired
        # on it would be pressure to delete the reason.
        self.text = dispatch_code()
        self.assertIn('std::getenv("VLLM_LTX2_DIT_FLASH_ATTN")', self.text,
                      "the knob is not read in code any more, so both assertions "
                      "below would be measuring a file that no longer has a dispatch")

    def test_no_arm_is_selected_by_a_prefix(self) -> None:
        self.assertFalse(
            "arm[0]" in self.text,
            "the knob is matched by a PREFIX again in `ltx2_device.cpp`: `0x`, `07` "
            "and `0flash` would each select an arm nobody asked for (#1751)")

    def test_an_unrecognised_value_throws_rather_than_defaulting(self) -> None:
        block = self.text.split('std::getenv("VLLM_LTX2_DIT_FLASH_ATTN")', 1)[1]
        block = block.split("\n    } else {", 1)[0]
        self.assertTrue(
            "throw std::invalid_argument" in block,
            "the last arm of the knob dispatch no longer refuses; an unrecognised "
            "value falls through to a default again (#1751)")
        self.assertTrue(
            "VLLM_LTX2_DIT_FLASH_ATTN=" in block,
            "the refusal no longer names the variable it refuses")


class ThisSuiteIsRegistered(unittest.TestCase):
    """A gate nothing runs is not a gate. Asserted here rather than trusted,
    because both registrations are one line in a file this suite does not own."""

    def test_preflight_runs_it(self) -> None:
        text = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        self.assertTrue("test_ltx2_dit_attn_knob_arms" in text,
                        "scripts/agent-preflight.sh does not list this suite")

    def test_the_ci_record_lane_runs_it(self) -> None:
        text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertTrue("tests/scripts/test_ltx2_dit_attn_knob_arms.py" in text,
                        ".github/workflows/ci.yml does not run this suite")


if __name__ == "__main__":
    unittest.main(verbosity=2)
