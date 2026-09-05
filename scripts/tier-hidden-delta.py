#!/usr/bin/env python3
"""Join two activation dumps and print the per-layer cross-tier delta profile.

ROCM-TIER-DIVERGENCE (#2590).
Spec: .agents/specs/rocm-tier-hidden-state-bisect.md

WHAT THIS ANSWERS. At three decode steps our ROCm tier and our CPU tier compute
a different argmax over an identical prefix on an identical artifact, with no
oracle in the comparison (docs/bench-evidence/
qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md, the `A vs D` column).
Two implementations of one computation disagree.  This tool reads the per-layer
hidden state both of them wrote and says WHERE they separate.

WHAT IT REFUSES TO DO. It never reports "the first layer whose delta is
non-zero".  Two tiers reduce in different orders and differ in the last bits at
every layer, so that answer is "layer 0" every time and it is not a finding.
What it prints instead is the SHAPE: the per-layer `rel_l2` and the ratio of
each layer's `rel_l2` to the layer before it.  Under pure rounding that ratio
sits near 1 and the profile is a ramp.  A layer where it jumps is a defect
signature, and that is the one this tool points at.

WHAT IT SAYS ABOUT ITSELF.  The header names which directory was taken as A and
which as B, how many manifest rows each side declared, how many keys joined, and
how many were dropped.  A comparator that silently intersects two half-populated
directories reports a clean profile over almost nothing, and the drop count is
the only thing that makes that visible.  Drops are a refusal by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np

MANIFEST = "manifest.tsv"
COLUMNS = ("step", "layer", "stage", "dtype", "rows", "cols", "bytes", "file")

# Element widths, keyed by the spelling `vt::Name(DType)` writes.  Only the
# dtypes an activation buffer can carry are listed: a block-quantized activation
# does not exist on this path, and a manifest naming one is a bug in the writer
# rather than a case to decode.
DTYPES = {"f32": (np.float32, 4), "f16": (np.float16, 2), "bf16": (None, 2)}


def die(msg: str) -> "None":
    print(f"REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_manifest(directory: str, label: str) -> "OrderedDict":
    path = os.path.join(directory, MANIFEST)
    if not os.path.isfile(path):
        die(
            f"arm {label}: no {MANIFEST} in {directory!r}. The dump writes one row "
            f"per blob, so a missing manifest means the run never dumped -- which "
            f"is exactly what a dump that is switched off looks like, and is why "
            f"this is a refusal and not an empty table."
        )
    rows: "OrderedDict" = OrderedDict()
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != len(COLUMNS):
                die(
                    f"arm {label}: {path}:{lineno} has {len(parts)} columns, "
                    f"expected {len(COLUMNS)} ({', '.join(COLUMNS)})"
                )
            rec = dict(zip(COLUMNS, parts))
            for k in ("step", "layer", "rows", "cols", "bytes"):
                rec[k] = int(rec[k])
            key = (rec["step"], rec["layer"], rec["stage"])
            if key in rows:
                die(
                    f"arm {label}: duplicate key {key} at {path}:{lineno}. The key "
                    f"is (step, layer, stage) and it must identify one blob; a "
                    f"duplicate means the writer was keyed on something else."
                )
            rows[key] = rec
    if not rows:
        die(f"arm {label}: {path} declares zero blobs")
    return rows


def load_blob(directory: str, rec: dict, label: str) -> np.ndarray:
    path = os.path.join(directory, rec["file"])
    if not os.path.isfile(path):
        die(f"arm {label}: manifest names {rec['file']!r} but the file is absent")
    size = os.path.getsize(path)
    if size != rec["bytes"]:
        die(
            f"arm {label}: {rec['file']} is {size} bytes, manifest says "
            f"{rec['bytes']}. A truncated blob would otherwise read as a shape "
            f"mismatch or, worse, as a plausible profile."
        )
    dt = rec["dtype"]
    if dt not in DTYPES:
        die(f"arm {label}: {rec['file']} has dtype {dt!r}, which is not decodable")
    raw = np.fromfile(path, dtype=np.uint8)
    n = rec["rows"] * rec["cols"]
    if dt == "bf16":
        u16 = raw.view("<u2")
        if u16.size != n:
            die(f"arm {label}: {rec['file']} holds {u16.size} elems, expected {n}")
        wide = np.zeros(u16.size, dtype=np.uint32)
        wide[:] = u16.astype(np.uint32) << 16
        vals = wide.view(np.float32)
    else:
        np_dt, _ = DTYPES[dt]
        vals = raw.view(np_dt).astype(np.float32)
        if vals.size != n:
            die(f"arm {label}: {rec['file']} holds {vals.size} elems, expected {n}")
    return vals.reshape(rec["rows"], rec["cols"]).astype(np.float64)


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    denom = float(np.sqrt(np.sum(b * b)))
    return {
        "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
        "rms": float(np.sqrt(np.mean(d * d))) if d.size else 0.0,
        "rel_l2": (float(np.sqrt(np.sum(d * d)) / denom) if denom > 0 else float("nan")),
        "argmax_flat": int(np.argmax(np.abs(d))) if d.size else -1,
        "b_absmax": float(np.max(np.abs(b))) if b.size else 0.0,
    }


def family(layer: int, interval: int) -> str:
    if layer < 0:
        return "embed"
    if interval <= 0:
        return "gdn"
    return "full" if (layer + 1) % interval == 0 else "gdn"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="dump directory of arm A")
    ap.add_argument("--b", required=True, help="dump directory of arm B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--step", type=int, default=None,
                    help="report only this forward-step ordinal")
    ap.add_argument("--stage", action="append", default=None,
                    help="report only these stages (repeatable)")
    ap.add_argument("--full-attention-interval", type=int, default=4,
                    help="layer l is full-attention when (l+1) %% interval == 0")
    ap.add_argument("--allow-drops", action="store_true",
                    help="do not refuse when the two manifests do not cover the "
                         "same keys; the drop count is printed either way")
    ap.add_argument("--json", default=None, help="also write the table as JSON")
    args = ap.parse_args()

    ma = read_manifest(args.a, args.label_a)
    mb = read_manifest(args.b, args.label_b)

    keys_a, keys_b = set(ma), set(mb)
    joined = keys_a & keys_b
    only_a, only_b = keys_a - keys_b, keys_b - keys_a

    # Say what is being compared with what, in words, before any number. A
    # reader who cannot see the wiring in the report is auditing the source
    # instead of the run.
    print("== tier-hidden-delta ==")
    print(f"A = {args.label_a}  dir={args.a}  manifest_rows={len(ma)}")
    print(f"B = {args.label_b}  dir={args.b}  manifest_rows={len(mb)}")
    print(f"join key = (step, layer, stage); joined={len(joined)} "
          f"only_in_A={len(only_a)} only_in_B={len(only_b)}")
    print(f"delta is A minus B; rel_l2 is ||A-B|| / ||B||, so B ({args.label_b}) "
          f"is the denominator")
    print(f"full-attention interval = {args.full_attention_interval} "
          f"(layer l is full-attn when (l+1) mod interval == 0)")

    if (only_a or only_b) and not args.allow_drops:
        die(
            f"the two manifests do not cover the same keys: {len(only_a)} only in "
            f"A, {len(only_b)} only in B. A join over the intersection would "
            f"report a profile over fewer positions than either side dumped, and "
            f"nothing downstream could see that. Pass --allow-drops to accept it."
        )

    wanted_stages = set(args.stage) if args.stage else None

    # The stream dump is a PAIR. In this model neither `hidden` nor `res` is the
    # hidden state: the two are carried separately and summed at the final norm
    # (qwen3_5.cpp `MaybeCaptureAuxTap` computes exactly hidden+res). So where
    # both halves joined, their sum is reported as the synthetic stage `stream`,
    # and that is the row to read as "the hidden state after layer l".
    rows = []
    steps = sorted({k[0] for k in joined})
    for step in steps:
        if args.step is not None and step != args.step:
            continue
        layers = sorted({k[1] for k in joined if k[0] == step})
        for layer in layers:
            stages = sorted({k[2] for k in joined if k[0] == step and k[1] == layer})
            cache = {}
            for stage in stages:
                if wanted_stages and stage not in wanted_stages:
                    continue
                ka, kb = ma[(step, layer, stage)], mb[(step, layer, stage)]
                for col in ("dtype", "rows", "cols"):
                    if ka[col] != kb[col]:
                        die(
                            f"key (step={step}, layer={layer}, stage={stage}) has "
                            f"{col}={ka[col]!r} on {args.label_a} and "
                            f"{kb[col]!r} on {args.label_b}. Two dumps that "
                            f"describe different computations cannot be diffed."
                        )
                va = load_blob(args.a, ka, args.label_a)
                vb = load_blob(args.b, kb, args.label_b)
                cache[stage] = (va, vb)
                st = stats(va, vb)
                st.update(step=step, layer=layer, stage=stage,
                          family=family(layer, args.full_attention_interval),
                          dtype=ka["dtype"], rows=ka["rows"], cols=ka["cols"])
                rows.append(st)
            if "hidden" in cache and "res" in cache:
                sa = cache["hidden"][0] + cache["res"][0]
                sb = cache["hidden"][1] + cache["res"][1]
                st = stats(sa, sb)
                st.update(step=step, layer=layer, stage="stream",
                          family=family(layer, args.full_attention_interval),
                          dtype=ma[(step, layer, "hidden")]["dtype"],
                          rows=sa.shape[0], cols=sa.shape[1])
                rows.append(st)

    if not rows:
        die("the join produced no comparable rows after filtering")

    hdr = (f"{'step':>5} {'layer':>6} {'fam':>5} {'stage':>24} "
           f"{'max_abs':>12} {'rms':>12} {'rel_l2':>12} {'growth':>8}")
    print()
    print(hdr)
    print("-" * len(hdr))
    prev = {}
    for r in sorted(rows, key=lambda r: (r["step"], r["layer"], r["stage"])):
        # `growth` is this layer's rel_l2 divided by the same stage's rel_l2 one
        # layer earlier. Near 1 across the whole model is a ramp; a jump is the
        # localisation. It is blank on the first layer a stage appears at,
        # because there is nothing to divide by and a 1.0 there would read as a
        # measurement.
        key = (r["step"], r["stage"])
        g = ""
        p = prev.get(key)
        if p is not None and p > 0 and math.isfinite(r["rel_l2"]):
            g = f"{r['rel_l2'] / p:8.3f}"
        if math.isfinite(r["rel_l2"]):
            prev[key] = r["rel_l2"]
        r["growth"] = float(g) if g else None
        print(f"{r['step']:>5} {r['layer']:>6} {r['family']:>5} {r['stage']:>24} "
              f"{r['max_abs']:12.6e} {r['rms']:12.6e} {r['rel_l2']:12.6e} {g:>8}")

    # The headline, computed rather than eyeballed: the largest single-layer jump
    # in the `stream` profile, which is the quantity the spec's P1 names.
    stream = [r for r in rows if r["stage"] == "stream" and r.get("growth")]
    print()
    if stream:
        worst = max(stream, key=lambda r: r["growth"])
        print(f"largest single-layer growth on `stream`: {worst['growth']:.3f}x at "
              f"step={worst['step']} layer={worst['layer']} "
              f"family={worst['family']}")
    else:
        print("largest single-layer growth on `stream`: NOT COMPUTED "
              "(fewer than two layers carried both halves)")

    zero = [r for r in rows if r["stage"] == "stream" and r["max_abs"] == 0.0]
    print(f"bit-identical `stream` positions: {len(zero)} of "
          f"{len([r for r in rows if r['stage'] == 'stream'])}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"a": args.a, "b": args.b, "label_a": args.label_a,
                       "label_b": args.label_b, "joined": len(joined),
                       "only_in_a": len(only_a), "only_in_b": len(only_b),
                       "rows": rows}, fh, indent=1, sort_keys=True)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
