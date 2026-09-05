#!/usr/bin/env python3
"""The attribution tool must not read a defect as arithmetic, or the reverse (#2590).

Spec: .agents/specs/rocm-tier-hidden-state-bisect.md section 6.

This tool decides the row's outcome: it is the thing that says whether a
cross-tier delta profile is one op being wrong or two bf16 residual streams doing
what they must. Four properties are pinned, each one a way of getting that
backwards:

  * POST-DIVERGENCE STEPS ARE EXCLUDED, and the exclusion is required rather
    than offered. Once the tiers emit different tokens they consume different
    inputs, so a later step compares two different computations; including them
    inflates every statistic. The tool takes the divergence step as an argument
    and never guesses it.
  * THE FAMILY SPLIT IS BY VARIANCE, not by the ratio to the previous layer. A
    ratio cannot rank families: early layers ramp by 2x on rounding alone,
    because `rel_l2` there is 1e-3. The increment of `rel_l2` SQUARED is what is
    additive under independent rounding, and it is what the share is computed
    from.
  * THE bf16 PREDICTION IS PRINTED BESIDE THE MEASUREMENT. "The tiers differ by
    2.6%" decides nothing until it sits next to what they must differ by.
  * A PLANTED DEFECT MUST NOT READ AS ARITHMETIC. The last case doubles one
    family's per-layer error and requires the ratio to leave 1.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(ROOT, "scripts", "tier-delta-attribution.py")


def family(layer, interval=4):
    if layer < 0:
        return "embed"
    return "full" if (layer + 1) % interval == 0 else "gdn"


def make(path, steps, layers, per_layer, divergence_step=None, post_scale=40.0):
    """A synthetic profile whose rel_l2 is an exact random walk of `per_layer`."""
    rows = []
    for s in range(steps):
        acc = 0.0
        blown = divergence_step is not None and s > divergence_step
        rows.append({"step": s, "layer": -1, "stage": "stream", "family": "embed",
                     "rel_l2": (post_scale if blown else 0.0),
                     "max_abs": (post_scale if blown else 0.0), "rms": 0.0})
        for l in range(layers):
            acc += per_layer[family(l)] ** 2
            v = math.sqrt(acc) * (post_scale if blown else 1.0)
            rows.append({"step": s, "layer": l, "stage": "stream",
                         "family": family(l), "rel_l2": v, "max_abs": v, "rms": v})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"a": "A", "b": "B", "label_a": "rocm", "label_b": "cpu",
                   "joined": len(rows), "only_in_a": 0, "only_in_b": 0,
                   "rows": rows}, fh)


def run(path, div, *extra):
    return subprocess.run([sys.executable, TOOL, "--json", path,
                           "--first-divergence", str(div), *extra],
                          capture_output=True, text=True)


class AttributionTests(unittest.TestCase):
    # eps/sqrt(3)*sqrt(2) for bf16, the per-store two-tier difference.
    STORE = 2.0 ** -8 / math.sqrt(3.0) * math.sqrt(2.0)

    def test_pure_bf16_walk_reads_as_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.json")
            make(p, steps=6, layers=64,
                 per_layer={"gdn": self.STORE, "full": self.STORE})
            r = run(p, 5)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("a random walk over 64 layers reaches", r.stdout)
            ratio = float([l for l in r.stdout.splitlines()
                           if "MEASURED / PREDICTED" in l][0].split("=")[1])
            self.assertAlmostEqual(ratio, 1.0, places=2)

    def test_a_planted_defect_does_not_read_as_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.json")
            # One family carrying four times the rounding it should.
            make(p, steps=6, layers=64,
                 per_layer={"gdn": self.STORE, "full": 4 * self.STORE})
            r = run(p, 5)
            self.assertEqual(r.returncode, 0, r.stderr)
            ratio = float([l for l in r.stdout.splitlines()
                           if "MEASURED / PREDICTED" in l][0].split("=")[1])
            self.assertGreater(ratio, 1.5)

    def test_family_share_is_by_variance_not_by_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.json")
            # 16 `full` layers at 4x the step of 48 `gdn` layers:
            # variance share = 16*16 / (16*16 + 48*1) = 84.2%.
            make(p, steps=4, layers=64,
                 per_layer={"gdn": self.STORE, "full": 4 * self.STORE})
            r = run(p, 3)
            self.assertEqual(r.returncode, 0, r.stderr)
            full = [l for l in r.stdout.splitlines() if l.strip().startswith("full")][0]
            gdn = [l for l in r.stdout.splitlines() if l.strip().startswith("gdn")][0]
            self.assertIn("84.2%", full)
            self.assertIn("15.8%", gdn)

    def test_post_divergence_steps_are_excluded_and_said_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.json")
            make(p, steps=8, layers=64,
                 per_layer={"gdn": self.STORE, "full": self.STORE},
                 divergence_step=5)
            r = run(p, 5)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("comparable steps = 0..5 (6)", r.stdout)
            self.assertIn("excluded as post-divergence = 2", r.stdout)
            self.assertIn("the exclusion is load-bearing", r.stdout)
            ratio = float([l for l in r.stdout.splitlines()
                           if "MEASURED / PREDICTED" in l][0].split("=")[1])
            self.assertAlmostEqual(ratio, 1.0, places=2)

            # Including them puts the incomparable steps into the spread. The
            # HEADLINE ratio is a median and survives two of eight, which is
            # exactly why the tool prints min and max beside it and why the
            # exclusion is an argument rather than a convenience: a reader who
            # only sees the median cannot tell this apart from a clean profile.
            def last_layer_max(out):
                line = [l for l in out.splitlines() if "rel_l2  median=" in l][0]
                return float(line.split("max=")[1].split()[0])
            self.assertLess(last_layer_max(r.stdout), 0.1)
            r_all = run(p, 7)
            self.assertEqual(r_all.returncode, 0, r_all.stderr)
            self.assertGreater(last_layer_max(r_all.stdout), 1.0)
            self.assertIn("excluded as post-divergence = 0", r_all.stdout)

    def test_bit_identical_input_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.json")
            make(p, steps=3, layers=8,
                 per_layer={"gdn": self.STORE, "full": self.STORE})
            r = run(p, 2)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("3 of 3 steps BIT-IDENTICAL", r.stdout)

    def test_missing_stage_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.json")
            make(p, steps=2, layers=4,
                 per_layer={"gdn": self.STORE, "full": self.STORE})
            r = run(p, 1, "--stage", "block_out")
            self.assertEqual(r.returncode, 2)
            self.assertIn("REFUSED", r.stderr)


if __name__ == "__main__":
    unittest.main()
