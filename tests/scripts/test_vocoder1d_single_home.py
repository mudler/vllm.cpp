#!/usr/bin/env python3
"""The 1-D vocoder core has exactly ONE home, and it is not a model's header.

`ltx2_audio_vae.cpp` records why this matters: LTX-2.5 deliberately did not copy
MiniMax-H3's alias-free primitives, because "a second copy of the alias-free trim
geometry is the duplicate that goes wrong quietly -- each copy keeps its own green
gate while the two audio VAEs drift apart." IndexTTS-2.5 is the third consumer.

No numeric test can see that failure: a fork passes every tensor comparison on
both sides on the day it is made, and only drifts later. So the invariant is
asserted structurally here -- one declaration site, one definition site, and no
consumer reaching for a model-specific spelling.
"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HEADER = ROOT / "include/vllm/model_executor/models/vocoder1d.h"
SOURCE = ROOT / "src/vllm/model_executor/models/vocoder1d.cpp"
H3_HEADER = ROOT / "include/vllm/model_executor/models/minimax_h3.h"

# The core: every symbol both audio VAEs share.
SYMBOLS = (
    "Conv1d",
    "ConvTranspose1d",
    "Pad1d",
    "SnakeActivation",
    "AliasFreeActivation1d",
    "KaiserSincFilter1d",
    # MiniMax-Music3's vocoder (#672) made the weight-norm fold a second
    # consumer's problem. `w = g * v / ||v||` over every dim but dim 0 is one
    # line of arithmetic with two ways to get the axis wrong, and a fork agrees
    # on the day it is made -- the same reason every symbol above is here.
    "MaterializeWeightNorm",
)

# The spellings the move retires. A reappearance means someone re-forked, or
# re-anchored the shared core back onto one model's name.
RETIRED = tuple("MiniMaxH3" + name for name in SYMBOLS) + ("kMiniMaxH3SnakeEps",)


def tracked_sources() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "src", "include", "tests", "examples"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [ROOT / p for p in out if p.endswith((".h", ".cpp", ".cu", ".hip"))]


class Vocoder1dSingleHomeTests(unittest.TestCase):
    def test_the_neutral_header_declares_every_shared_symbol(self) -> None:
        self.assertTrue(HEADER.is_file(), f"{HEADER} must exist")
        text = HEADER.read_text(encoding="utf-8")
        self.assertIn("namespace vocoder1d", text)
        for name in SYMBOLS:
            self.assertIn(name, text, f"{name} must be declared in the neutral header")
        self.assertIn("kSnakeEps", text)

    def test_the_definitions_live_beside_the_declarations(self) -> None:
        self.assertTrue(SOURCE.is_file(), f"{SOURCE} must exist")
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("namespace vocoder1d", text)

    def test_the_model_header_no_longer_owns_the_core(self) -> None:
        """A model's header is not a home for something three lanes share."""
        text = H3_HEADER.read_text(encoding="utf-8")
        for name in RETIRED:
            self.assertNotIn(
                name,
                text,
                f"{name} still lives in minimax_h3.h; the core moved to vocoder1d.h",
            )

    def test_no_tracked_source_uses_a_retired_model_specific_spelling(self) -> None:
        offenders: list[str] = []
        for path in tracked_sources():
            if path.name.startswith("vocoder1d"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for name in RETIRED:
                if re.search(rf"\b{re.escape(name)}\b", text):
                    offenders.append(f"{path.relative_to(ROOT)}: {name}")
        self.assertEqual(offenders, [], "retired spellings survive: " + "; ".join(offenders))

    def test_exactly_one_definition_of_each_shared_symbol(self) -> None:
        """Two definitions is the fork this guard exists to catch."""
        pattern = re.compile(
            r"^(?:std::vector<float>|void|double)\s+(?:vocoder1d::)?"
            r"(Conv1d|ConvTranspose1d|Pad1d|SnakeActivation|KaiserSincFilter1d"
            r"|MaterializeWeightNorm)\s*\(",
            re.M,
        )
        counts: dict[str, int] = {name: 0 for name in pattern.pattern and SYMBOLS}
        examined = 0
        for path in tracked_sources():
            if path.suffix != ".cpp":
                continue
            # The `vt` KERNEL SEAM is not a candidate home for this core, and
            # since #672 it holds `vt::Conv1d` / `vt::ConvTranspose1d` -- the ops
            # `vocoder1d` DELEGATES to. Those are the opposite of the fork this
            # file exists to catch: they are what removed the second copy of the
            # arithmetic. This pattern is a line-anchored TEXT match and cannot
            # see a namespace, so it read `vt::Conv1d`'s definition in
            # src/vt/ops.cpp as a duplicate of `vocoder1d::Conv1d`.
            #
            # The exclusion is scoped to the seam and PAID FOR by the new test
            # below, which is added coverage rather than a subtraction: excluding
            # a tree would otherwise let the core quietly re-grow its own loops
            # while the op sat unused, so this file now pins the delegation it is
            # trading for.
            if path.is_relative_to(ROOT / "src" / "vt"):
                continue
            examined += 1
            for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
                counts[match.group(1)] += 1
        # Say HOW MANY files were read. A walk that examined only vocoder1d.cpp
        # would report every count as 1 and pass while seeing none of the tree --
        # a green that means nothing, and one no count-of-1 assertion can detect.
        self.assertGreater(examined, 100, f"only {examined} .cpp files scanned; the walk is broken")
        for name, count in counts.items():
            if name == "AliasFreeActivation1d":
                continue
            self.assertEqual(count, 1, f"{name} has {count} definitions; exactly one is allowed")

    def test_the_core_delegates_its_convolutions_to_the_shared_vt_ops(self) -> None:
        """The price of excluding `src/vt/` from the count above.

        Since #672 the convolution ARITHMETIC lives once, in the `vt::Conv1d` /
        `vt::ConvTranspose1d` providers, and `vocoder1d` is their caller. If the
        core ever re-grows its own loop, the count above would still read 1 --
        one definition, in the right file, quietly doing the work itself again --
        and all six consumers would silently leave the shared seam while every
        numeric gate stayed green, because a re-grown loop computes the same
        thing. That is the same class of failure as the fork this file was
        written for, so it is asserted here, beside the exclusion it pays for.
        """
        text = SOURCE.read_text(encoding="utf-8")
        for op in ("vt::Conv1d(", "vt::ConvTranspose1d("):
            # assertTrue, not assertIn: assertIn prints the whole HAYSTACK on
            # failure, and the haystack here is the entire source file.
            self.assertTrue(
                op in text,
                f"{SOURCE.name} no longer calls {op} -- the core has left the shared vt seam",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1)
