# GFX1100-TG200 — negative result: pointer-keyed quantized-activation cache

Date: 2026-08-23. Follows `gfx1100-tg200-t2a-20260822.md`.

## What was tried

A per-stream cache in front of `QuantizeQ8KK` keyed on
`(activation ptr, row stride, activation dtype, m, nsb, weight dtype)`:
the first kMatmulBTQuant call over a given activation launches the quant
kernel; later calls with the same key reuse the scratch buffer.

## Result: REJECTED — unsound under the block-recycling allocator

- First cut (pointer-only key): throughput rose to ~45 tok/s median, but the
  generated text degenerated into repeated garbage (`heimerheimer...`) — the
  DevicePool recycles activation blocks across steps, so the same pointer
  carried different content on the next step and stale quantized data was
  served. Correctness gate caught it exactly as designed.
- Second cut (epoch keying via vt::BumpQuantEpoch/CurrentQuantEpoch, bumped
  once per model forward): still degenerate. Within ONE step the pool hands
  the SAME address to DIFFERENT activations (DBuf freed and re-allocated mid-
  forward), so even intra-step pointer identity does not imply content
  identity.
- Reverted completely; revert verified by coherent output on the acceptance
  workload (the run reproduces the T1a-style coherent transformer explana-
  tion). Both cuts were never committed.

## Why this matters for the campaign

1. The "129 QuantizeQ8KK launches/token" cost is real GPU-busy time (~59us
   each profiled), but it CANNOT be eliminated by result-caching without a
   content-identity signal the allocator does not provide.
2. The sound levers for this budget are structural, not caching:
   - merge gate+up into one keep-quant GEMM (halves the quant sites),
   - MMVQ-style dequant-in-register decode GEMV (removes the separate quant
     kernel entirely, following SGLang's mmvq.cuh pattern),
   - ROCm decode-graph capture (removes the launch overhead that makes each
     tiny kernel cost ~59 us of queue time).
3. The probe instrumentation (VT_MATMUL_BT_QUANT_PROBE) also stays out of
   the tree; it served its one-shot purpose.

## Measured (for the record)

| Arm | median tok/s | notes |
|---|---|---|
| baseline (T1a) | 40.65 | idle host |
| cache v1 (ptr key) | 45.0 | DEGENERATE OUTPUT — rejected |
| cache v2 (epoch) | 44.9 | STILL DEGENERATE — root cause above |
| reverted build | coherent | matches T1a-class output |

Per working rule 3: perf wins that break correctness do not land. This is
the documented rejection, not a silent drop.
