#!/usr/bin/env python3
"""The logit comparator must report a delta against the margin it has to cross (#2590).

Spec: .agents/specs/rocm-tier-hidden-state-bisect.md section 6.

A cross-tier logit delta decides a token only when it is of the size of the
decision it would have to overturn. Reporting the delta alone is the mistake this
file exists to prevent, so four properties are pinned:

  * THE VOCABULARY IS DERIVED, not assumed. A dump read at a guessed vocabulary
    produces a plausible table over reinterpreted bytes, and no threshold
    downstream can see it. A length that is not a whole multiple of the sidecar's
    step count is a refusal.
  * THE MARGIN IS EACH TIER'S OWN top1 - top2, because that is what its argmax
    actually turned on.
  * A STEP IS AT RISK when the smaller margin is below that step's delta, and the
    tool reports whether any flip happened OUTSIDE that set. A flip the delta
    cannot account for would be a finding rather than a rounding, and nothing
    else in the pipeline would surface it.
  * POST-DIVERGENCE STEPS ARE EXCLUDED FROM THE STATISTIC and still printed,
    marked, so the exclusion is visible rather than silent.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(ROOT, "scripts", "tier-logit-delta.py")


def write_arm(directory, rows, req="0"):
    """rows: list of per-step logit lists. Writes the dump plus its ids sidecar."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"ours_{req}.f32"), "wb") as fh:
        for r in rows:
            fh.write(struct.pack("<%df" % len(r), *r))
    with open(os.path.join(directory, f"ours_{req}.ids.txt"), "w",
              encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            fh.write(f"{i} {max(range(len(r)), key=lambda v: r[v])}\n")


def run(a, b, *extra):
    return subprocess.run([sys.executable, TOOL, "--a", a, "--label-a", "rocm",
                           "--b", b, "--label-b", "cpu", *extra],
                          capture_output=True, text=True)


class LogitDeltaTests(unittest.TestCase):
    def test_a_flip_inside_the_at_risk_set_is_accounted_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            # Step 0: a wide margin, both tiers agree. Step 1: a margin of 0.20
            # on each tier and a cross-tier delta of 0.60, which overturns it.
            write_arm(a, [[10.0, 5.0, 0.0], [1.00, 1.20, 0.0]])
            write_arm(b, [[10.2, 5.0, 0.0], [1.20, 1.00, 0.6]])
            r = run(a, b)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("argmax flips: 1 of 2", r.stdout)
            self.assertIn("steps AT RISK", r.stdout)
            self.assertIn("of those, flipped: 1", r.stdout)
            self.assertIn("flips OUTSIDE the at-risk set: 0", r.stdout)

    def test_at_risk_is_necessary_and_not_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            # A narrow margin and a delta larger than it, but the delta lands on
            # a THIRD logit and the ranking survives. At risk, not flipped --
            # which is what most at-risk steps do, and a tool that equated the
            # two would over-explain every profile.
            write_arm(a, [[2.00, 1.99, 0.0]])
            write_arm(b, [[2.00, 1.99, 0.5]])
            r = run(a, b)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("argmax flips: 0 of 1", r.stdout)
            self.assertIn("steps AT RISK (min margin < that step's max_abs delta): 1 of 1",
                          r.stdout)
            self.assertIn("of those, flipped: 0", r.stdout)

    def test_at_risk_uses_the_SMALLER_of_the_two_margins(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            # One tier won by 0.02, the other by 0.50, and the tiers differ by
            # 0.40. The step IS at risk, because the 0.02 decision is the one the
            # delta can overturn. Keying on the larger margin would call it safe,
            # and a real flip would then land outside the at-risk set and be
            # reported as a finding the tool cannot account for.
            write_arm(a, [[1.00, 0.98, 0.0]])
            write_arm(b, [[1.40, 0.90, 0.0]])
            r = run(a, b)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("argmax flips: 0 of 1", r.stdout)
            self.assertIn("steps AT RISK (min margin < that step's max_abs delta): 1 of 1",
                          r.stdout)

    def test_post_divergence_steps_leave_the_statistic_but_stay_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            write_arm(a, [[10.0, 5.0, 0.0], [1.0, 1.2, 0.0], [50.0, 0.0, 0.0]])
            write_arm(b, [[10.2, 5.0, 0.0], [1.2, 1.0, 0.6], [0.0, 50.0, 0.0]])
            r = run(a, b, "--last-comparable-step", "1")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("comparable steps: 2 of 3", r.stdout)
            self.assertIn("post-divergence: NOT comparable", r.stdout)
            # Step 2's enormous delta must not enter the at-risk count.
            self.assertIn("steps AT RISK (min margin < that step's max_abs delta): 1 of 2",
                          r.stdout)
            self.assertIn("of those, flipped: 1", r.stdout)

    def test_vocab_is_derived_and_a_ragged_dump_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            write_arm(a, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            write_arm(b, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            ok = run(a, b)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("vocab=3 (derived from the ids sidecar)", ok.stdout)
            # Truncate one float off: the length is no longer a whole multiple.
            path = os.path.join(a, "ours_0.f32")
            with open(path, "rb") as fh:
                raw = fh.read()
            with open(path, "wb") as fh:
                fh.write(raw[:-4])
            bad = run(a, b)
            self.assertEqual(bad.returncode, 2)
            self.assertIn("not a whole multiple", bad.stderr)

    def test_an_empty_dump_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a)
            write_arm(b, [[1.0, 2.0]])
            r = run(a, b)
            self.assertEqual(r.returncode, 2)
            self.assertIn("looks exactly like a dump that was switched off",
                          r.stderr)

    def test_a_vocabulary_disagreement_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            write_arm(a, [[1.0, 2.0, 3.0]])
            write_arm(b, [[1.0, 2.0]])
            r = run(a, b)
            self.assertEqual(r.returncode, 2)
            self.assertIn("vocabulary sizes differ", r.stderr)


if __name__ == "__main__":
    unittest.main()
