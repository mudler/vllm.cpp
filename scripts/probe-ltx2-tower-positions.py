#!/usr/bin/env python3
"""Ask the Gemma-4 tower ORACLE whether left-padded positions may be renumbered.

The question #1467 raised: `test_ltx2_text_encoder.cpp` measured the
position-renumbering MUTANT scoring CLOSER to the oracle than the port, which
reads as "our positions are wrong". This answers it somewhere the end-to-end
bf16 conditioning cannot — against the oracle's own left-padded run, at both
arithmetic widths, with no projection stack in between.

It imports scripts/gen-ltx2-gemma-tower-goldens.py and calls that generator's
own `build_tower` and `run_tower`, so there is no second implementation of the
fixture to drift. CPU only, no checkpoint, no network.

Usage:
    scripts/probe-ltx2-tower-positions.py <repo-root>

Needs the same interpreter the goldens need: a `transformers` that registers
`gemma4_unified` (>= 5.8), plus torch and numpy. MEASURED identical at 5.12.1
(the goldens' oracle) and at 5.14.1 (the parity pin).
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "gen", root / "scripts" / "gen-ltx2-gemma-tower-goldens.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

import transformers
print(f"oracle: transformers {transformers.__version__} at {Path(transformers.__file__).resolve().parent}")
print(f"        torch {torch.__version__}")

real = json.loads((root / "tests/vllm/models/ltx2_gemma4_text_config.json").read_text())
model, inner, config, filled = gen.build_tower(real)
print(f"        {len(filled)} params/buffers filled, layer_types={gen.LAYER_TYPES}")

TOK, NP_, SEQ = gen.TOKENS, gen.NUM_PAD, gen.SEQ
padded_ids = [gen.PAD_ID] * NP_ + TOK
padded_mask = [0] * NP_ + [1] * SEQ
abs_pos = list(range(NP_, NP_ + SEQ))
zero_pos = list(range(SEQ))

def legs(dt, label):
    P = gen.run_tower(inner, padded_ids, padded_mask, dt)
    A = gen.run_tower(inner, TOK, [1]*SEQ, dt, positions=abs_pos)
    Z = gen.run_tower(inner, TOK, [1]*SEQ, dt, positions=zero_pos)
    pa = [float(np.abs(a - p[NP_:]).max()) for a, p in zip(A, P)]
    pz = [float(np.abs(z - p[NP_:]).max()) for z, p in zip(Z, P)]
    az = [float(np.abs(a - z).max()) for a, z in zip(A, Z)]
    mag = [float(np.abs(p[NP_:]).max()) for p in P]
    print(f"\n=== {label} ===")
    print(" st  |padded|max   |ABS-padded|    |ZERO-padded|   |ABS-ZERO|")
    for i,(m,x,y,z) in enumerate(zip(mag,pa,pz,az)):
        print(f" {i:2d}  {m:12.6f}  {x:14.6e}  {y:14.6e}  {z:14.6e}")
    print(f" MAX {max(mag):12.6f}  {max(pa):14.6e}  {max(pz):14.6e}  {max(az):14.6e}")
    print(f" ABS is CLOSER to the padded oracle at {sum(1 for x,y in zip(pa,pz) if x<y)}"
          f"/{len(pa)} states; ZERO closer at {sum(1 for x,y in zip(pa,pz) if y<x)}")
    return pa, pz, mag

f32 = legs(torch.float32, "float32 (the arm with no bf16 rounding in it)")
bf16 = legs(torch.bfloat16, "bfloat16 (the shipped dtype)")

pa, pz, mag = f32
print(f"\nf32 relative: |ABS-padded|/|padded| = {max(pa)/max(mag):.3e}   "
      f"|ZERO-padded|/|padded| = {max(pz)/max(mag):.3e}")
