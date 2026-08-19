#!/usr/bin/env python3
"""Unit checks for scripts/nemotron-h-a2q1-per-token.py (#810 A2-Q1).

THE REFUSALS ARE THE POINT. This helper exists because
`examples/nemotron_h_gen` reports neither a rate nor a duration, so the
per-output-token number A2-Q1 is measured on has to be DERIVED — and a derived
number that prints 0, or a negative, or a rate over an unknown denominator, is
indistinguishable from a measurement once it reaches a report.

Three of its four behaviours are refusals, and nothing else in the tree pins
them. A later edit that "simplified" any of them would go unnoticed until it had
already produced a number somebody quoted, which is the failure this repository
keeps paying for.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/nemotron-h-a2q1-per-token.py"
SPEC = importlib.util.spec_from_file_location("a2q1_per_token", HELPER)
assert SPEC is not None and SPEC.loader is not None
a2q1_per_token = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = a2q1_per_token
SPEC.loader.exec_module(a2q1_per_token)

LOADED = "[nemotron-h] engine loaded in 280.9s\n"
MATCH = ("[nemotron-h] TOKEN MATCH: 96/96 over 3 prompt(s) "
         "(full rows=3, short rows=0, mode=decode)\n")


def run(text: str, t0: float, t1: float, arch: str | None = None) -> str:
    with NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(text)
        path = fh.name
    argv = ["prog", path, "lbl", str(t0), str(t1)]
    if arch is not None:
        argv.append(arch)
    buf = io.StringIO()
    with redirect_stdout(buf):
        a2q1_per_token.main(argv)
    return buf.getvalue()


class PerTokenTests(unittest.TestCase):
    def test_reports_the_rate_with_every_term_it_divided(self) -> None:
        out = run(LOADED + MATCH, 1000.0, 1013.8)
        # The window, the excluded load and the token count all have to be on the
        # page, or the rate cannot be checked by the person reading it.
        self.assertIn("decode window 13.800 s", out)
        self.assertIn("engine load 280.9 s, excluded", out)
        self.assertIn("tokens compared 96", out)
        self.assertIn("per output token 0.143750 s", out)

    def test_the_load_is_excluded_not_subtracted_twice(self) -> None:
        # The caller brackets the DECODE, so the 280.9 s load must not be taken
        # off again. 13.8/96 == 0.14375; subtracting the load would go negative.
        out = run(LOADED + MATCH, 1000.0, 1013.8)
        self.assertIn("per output token 0.143750 s", out)
        self.assertNotIn("-0.", out)

    def test_a_non_positive_window_refuses_rather_than_printing_a_rate(self) -> None:
        out = run(LOADED + MATCH, 1000.0, 1000.0)
        self.assertIn("is not positive", out)
        self.assertNotIn("per output token", out)

    def test_a_missing_token_line_refuses_rather_than_printing_zero(self) -> None:
        out = run(LOADED, 1000.0, 1013.8)
        self.assertIn("NOT derivable, not 0", out)
        self.assertNotIn("per output token", out)

    def test_zero_compared_tokens_refuses_rather_than_dividing_by_nothing(self) -> None:
        out = run(LOADED + "[nemotron-h] TOKEN MATCH: 0/0 over 3 prompt(s)\n",
                  1000.0, 1013.8)
        self.assertIn("divide by nothing", out)
        self.assertNotIn("per output token", out)

    def test_the_vllm_ratio_is_withheld_on_an_arch_it_was_not_measured_on(self) -> None:
        # 0.014369 s is a GB10 figure. The first Thor run printed `ratio 54.7x`
        # for a number never measured against vLLM on that box -- the same
        # cross-silicon defect the busy-fraction reporter carried. The rate
        # itself still prints; only the comparison is withheld.
        out = run(LOADED + MATCH, 1000.0, 1075.418, "110")
        self.assertIn("per output token 0.785604 s", out)
        self.assertIn("NO vLLM ratio for arch 110", out)
        self.assertNotIn("ratio 54.7x", out)

    def test_the_vllm_ratio_is_quoted_on_the_arch_it_was_measured_on(self) -> None:
        out = run(LOADED + MATCH, 1000.0, 1075.418, "121a")
        self.assertIn("ratio 54.7x", out)
        self.assertNotIn("NO vLLM ratio", out)

    def test_the_vllm_denominator_is_the_pinned_one(self) -> None:
        # The ratio is only meaningful against the number this row's gap is
        # quoted against; a drifting constant would silently restate the gap.
        self.assertEqual(a2q1_per_token.VLLM_PER_TOKEN_S, 0.014369)


if __name__ == "__main__":
    unittest.main()
