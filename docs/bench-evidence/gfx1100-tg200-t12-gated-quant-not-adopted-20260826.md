# GFX1100-TG200 — T12: gated-norm producer-quant fusion attempted, NOT adopted

Date: 2026-08-26. Branch `row/GFX1100-TG200` at the T10+T11 landing
(`7c518f6a`). This lever was implemented, gate-green, measured inert at
engine level, diagnosed, and REVERTED byte-restored. This file is the
record and the starting point for any successor.

## Mechanism attempted

~40 standalone QuantizeQ8KK launches/token (13 µs each ≈ 0.5 ms/tok)
remain after lever C because their producer activations are not
rmsnorm-row outputs. The dominant group is the FFN gate_up/down consumers
whose input rows come from the GATED norm (24 layers). T12 extended
lever-C's producer-token fusion into `RmsNormGatedCoopK`: a cooperative
Q8_K epilogue (left-biased-max tree — bitwise-equal to the scalar
first-occurrence scan) behind its own knob `VT_GDN_NORMGATED_QUANT=1`,
recording the token through the shared bridge so existing consumers take
it without changes.

## What was proven

- Focused gate: full suite **16/16 cases, 850 assertions**, including the
  new case asserting the gated-norm scratch is BYTE-IDENTICAL to the
  standalone quantizer AND to the CPU host oracle on random,
  tied-amax(sign), and zero rows for nsb∈{1,3,10}, plus flag-inertness.
- In-process consumer probe: producers>=1 AND consumers_fused>=1 with a
  same-pointer K-quant matvec — the bridge contract works.
- rocpd at the ON config in-engine: QuantizeQ8KK standalone stayed at
  **40.0/tok** — no consumer took the token through the model executor.

## Verdict

Engine-level A/B wash (77.250 OFF vs 77.269 ON medians; all pairs
byte-identical) WITH engagement unproven end-to-end: the executor's FFN
matvec activation does not match the recorded producer output pointer
(different buffer or a strided/reshaped view). The unit-level mechanism is
correct; the missing piece is engine plumbing — either pass the gated
norm's device buffer identity through to the matvec call, or register the
producer against the buffer the matvec actually reads.

REVERTED byte-restored per the non-winner precedent. A successor should
start from ops.cpp dispatch tracing of the qwen3_5.cpp FFN call sites to
identify the exact pointer/view mismatch, not from this kernel again.

## Context for the ranking

This was ranked #3 (~0.48 ms/tok upper bound) in the corrected budget.
With it closed, the remaining order is: dispatch-gap audit (up to ~1.0),
rmsnorm_row second pass (+0.38), streaming micro-tuning (+0.5 spread).
Position stands at **84.3 tok/s median** (T11 window).
