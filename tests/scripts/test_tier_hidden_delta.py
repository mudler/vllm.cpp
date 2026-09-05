#!/usr/bin/env python3
"""The comparator must refuse a join it cannot trust (#2590).

Spec: .agents/specs/rocm-tier-hidden-state-bisect.md

WHAT IT GUARDS.  `scripts/tier-hidden-delta.py` is the only reader of the
activation dumps, so every way it can produce a confident number over the wrong
bytes has to be closed here.  Four of them:

  * a MISSING manifest -- an empty table would otherwise read exactly like two
    tiers that agree everywhere;
  * a PARTIAL join -- intersecting two half-populated directories reports a
    clean profile over almost nothing, and nothing downstream can see it;
  * a SHAPE or DTYPE disagreement on a joined key -- two dumps describing
    different computations diff to a plausible-looking number;
  * a TRUNCATED blob whose length disagrees with its manifest row.

It also pins the arithmetic that decides the row: `stream` is `hidden + res`,
because neither half alone is this model's hidden state, and `growth` is the
ratio to the previous layer, because "the first non-zero layer" is layer 0 on
any two tiers and is not a finding.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(ROOT, "scripts", "tier-hidden-delta.py")

COLUMNS = ("step", "layer", "stage", "dtype", "rows", "cols", "bytes", "file")


def bf16_bytes(values):
    """Encode float32s the way the writer stores a bf16 activation buffer."""
    out = bytearray()
    for v in values:
        bits = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        out += struct.pack("<H", bits >> 16)
    return bytes(out)


def write_blob(directory, step, layer, stage, values, rows, cols,
               dtype="bf16", truncate=0, declared_bytes=None):
    raw = bf16_bytes(values)
    name = f"s{step}_l{layer}_{stage}.bin"
    with open(os.path.join(directory, name), "wb") as fh:
        fh.write(raw[: len(raw) - truncate] if truncate else raw)
    n = declared_bytes if declared_bytes is not None else len(raw)
    with open(os.path.join(directory, "manifest.tsv"), "a", encoding="utf-8") as fh:
        fh.write(f"{step}\t{layer}\t{stage}\t{dtype}\t{rows}\t{cols}\t{n}\t{name}\n")


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run(a, b, *extra):
    return subprocess.run(
        [sys.executable, TOOL, "--a", a, "--label-a", "rocm", "--b", b,
         "--label-b", "cpu", *extra],
        capture_output=True, text=True)


class ComparatorTests(unittest.TestCase):
    def test_joins_and_reports_the_stream_as_hidden_plus_res(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a), os.makedirs(b)
            for layer in (0, 1):
                write_blob(a, 0, layer, "hidden", [1.0, 2.0], 1, 2)
                write_blob(a, 0, layer, "res", [4.0, 8.0], 1, 2)
                write_blob(b, 0, layer, "hidden", [1.0, 2.0], 1, 2)
                write_blob(b, 0, layer, "res", [4.0, 8.0], 1, 2)
            out = os.path.join(tmp, "o.json")
            r = run(a, b, "--json", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            # The wiring is narrated, not assumed by the reader.
            self.assertIn("A = rocm", r.stdout)
            self.assertIn("B = cpu", r.stdout)
            self.assertIn("joined=4", r.stdout)
            data = load_json(out)
            streams = [x for x in data["rows"] if x["stage"] == "stream"]
            self.assertEqual(len(streams), 2)
            # Identical arms: every delta is exactly zero, and that is what the
            # run-to-run floor of a self-comparison has to read.
            self.assertTrue(all(x["max_abs"] == 0.0 for x in streams))
            self.assertIn("bit-identical `stream` positions: 2 of 2", r.stdout)

    def test_stream_is_the_sum_not_either_half(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a), os.makedirs(b)
            # The halves differ, and they cancel in the SUM. A comparator that
            # read `hidden` alone would report a divergence where the model's
            # hidden state is identical -- which is the predecessor dump's exact
            # defect, carried into the reader.
            write_blob(a, 0, 0, "hidden", [3.0], 1, 1)
            write_blob(a, 0, 0, "res", [1.0], 1, 1)
            write_blob(b, 0, 0, "hidden", [1.0], 1, 1)
            write_blob(b, 0, 0, "res", [3.0], 1, 1)
            out = os.path.join(tmp, "o.json")
            r = run(a, b, "--json", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = {x["stage"]: x for x in load_json(out)["rows"]}
            self.assertEqual(rows["hidden"]["max_abs"], 2.0)
            self.assertEqual(rows["res"]["max_abs"], 2.0)
            self.assertEqual(rows["stream"]["max_abs"], 0.0)

    def test_growth_is_the_ratio_to_the_previous_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a), os.makedirs(b)
            # A ramp on layers 0 and 1, then a 10x jump at layer 2. The tool must
            # name layer 2, and must NOT name layer 0 -- "the first non-zero
            # layer" is the answer this design exists to refuse.
            for layer, delta in ((0, 1.0), (1, 2.0), (2, 20.0)):
                write_blob(a, 0, layer, "hidden", [100.0 + delta], 1, 1)
                write_blob(a, 0, layer, "res", [0.0], 1, 1)
                write_blob(b, 0, layer, "hidden", [100.0], 1, 1)
                write_blob(b, 0, layer, "res", [0.0], 1, 1)
            out = os.path.join(tmp, "o.json")
            r = run(a, b, "--json", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("largest single-layer growth on `stream`", r.stdout)
            self.assertIn("layer=2", r.stdout)
            rows = {(x["layer"], x["stage"]): x
                    for x in load_json(out)["rows"]}
            self.assertIsNone(rows[(0, "stream")]["growth"])
            self.assertAlmostEqual(rows[(1, "stream")]["growth"], 2.0, places=3)
            self.assertAlmostEqual(rows[(2, "stream")]["growth"], 10.0, places=3)

    def test_layer_family_splits_gdn_from_full_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a), os.makedirs(b)
            for layer in (-1, 2, 3):
                for d in (a, b):
                    write_blob(d, 0, layer, "hidden", [1.0], 1, 1)
                    write_blob(d, 0, layer, "res", [0.0], 1, 1)
            out = os.path.join(tmp, "o.json")
            r = run(a, b, "--json", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            fam = {x["layer"]: x["family"]
                   for x in load_json(out)["rows"]
                   if x["stage"] == "stream"}
            self.assertEqual(fam[-1], "embed")   # the post-embedding snapshot
            self.assertEqual(fam[2], "gdn")
            self.assertEqual(fam[3], "full")     # (3+1) % 4 == 0

    def test_missing_manifest_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a), os.makedirs(b)
            write_blob(a, 0, 0, "hidden", [1.0], 1, 1)
            r = run(a, b)
            self.assertEqual(r.returncode, 2)
            self.assertIn("REFUSED", r.stderr)
            self.assertIn("no manifest.tsv", r.stderr)

    def test_partial_join_refuses_and_allow_drops_opts_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a), os.makedirs(b)
            for layer in (0, 1):
                write_blob(a, 0, layer, "hidden", [1.0], 1, 1)
                write_blob(a, 0, layer, "res", [0.0], 1, 1)
            write_blob(b, 0, 0, "hidden", [1.0], 1, 1)
            write_blob(b, 0, 0, "res", [0.0], 1, 1)
            r = run(a, b)
            self.assertEqual(r.returncode, 2)
            self.assertIn("do not cover the same keys", r.stderr)
            r2 = run(a, b, "--allow-drops")
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertIn("only_in_A=2", r2.stdout)

    def test_shape_disagreement_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a), os.makedirs(b)
            write_blob(a, 0, 0, "hidden", [1.0, 2.0], 1, 2)
            write_blob(b, 0, 0, "hidden", [1.0, 2.0], 2, 1)
            r = run(a, b)
            self.assertEqual(r.returncode, 2)
            self.assertIn("different computations", r.stderr)

    def test_truncated_blob_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a), os.makedirs(b)
            write_blob(a, 0, 0, "hidden", [1.0, 2.0], 1, 2, truncate=2)
            write_blob(b, 0, 0, "hidden", [1.0, 2.0], 1, 2)
            r = run(a, b)
            self.assertEqual(r.returncode, 2)
            self.assertIn("bytes, manifest says", r.stderr)

    def test_empty_manifest_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(a), os.makedirs(b)
            open(os.path.join(a, "manifest.tsv"), "w").close()
            write_blob(b, 0, 0, "hidden", [1.0], 1, 1)
            r = run(a, b)
            self.assertEqual(r.returncode, 2)
            self.assertIn("zero blobs", r.stderr)


if __name__ == "__main__":
    unittest.main()
