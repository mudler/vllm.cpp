# GFX1100-TG200 — T8: cooperative single-row rmsnorm remap adopted (+3.2%)

Date: 2026-08-25. Host: local RX 7900 XTX (gfx1100), native build
`build-hip`, branch `row/GFX1100-TG200` at the T7-revert head plus this
change. Checkpoint sha256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
All legs in gpu-ctl-held windows; A/B window 23:10:08Z–23:12:02Z.

## Change

`RmsNormRowCoopKernel` behind `VT_RMSNORM_ROW_COOP=1` (default OFF;
registered on the kernel-internal allowlist). Decode launches ONE
256-thread block per norm row; the ported body chains three strided scalar
passes, a nine-step `__syncthreads()` shared-memory tree, and — under
lever-C's fused epilogue — a per-superblock serial `QuantQ8KSBlock` walk
on one thread. The arm rebuilds the internals:

1. Two-level reduction: wavefront `shfl_down` trees + one cross-wavefront
   combine through shared memory — two barriers instead of nine. Wave
   width is taken from `warpSize` at runtime (RDNA default 32); the first
   cut hardcoded 64 and silently dropped whole wavefronts' sums — caught
   by the focused gate (NMSE 0.086), fixed before any perf claim.
2. Vector passes: 16-byte loads/stores where alignment holds, scalar
   fallback otherwise (uniform per launch).
3. Cooperative q8 epilogue: the whole block quantizes ONE superblock at a
   time, thread i owning element i. The Lever C byte contract survives BY
   CONSTRUCTION: `(mx, amax)` comes from a LEFT-BIASED max over ascending
   positions — bitwise identical to the scalar first-occurrence scan,
   including sign ties — and iscale/DNearestInt/clamp/bsums arithmetic is
   verbatim.

The float association of the RMS reduction changes, so outputs may move
within float ULPs; the flag rides the campaign config as an opt-in like
GQA4 / GDN_SCAN_COOP / PREAMBLE_COOP, and the teacher-forced logprob-band
ceremony stays owed before any default flip.

## Correctness gate

`tests/vt/test_rocm_quant_dot`: new T8 case — epilogue scratch
BYTE-IDENTICAL to the standalone quantizer AND to the CPU host oracle on
random, tied-amax (sign tie: |x0|==|x17|==|x291|), and zero rows for
nsb∈{1,3,10}; COOP-vs-plain op output NMSE ≤ 1e-6; flag-inert leg.
Full suite **13/13 cases, 821 assertions SUCCESS** under the lock.

## Acceptance A/B — interleaved x5 pairs, full campaign config

Config: MMVQ+SKINNY+GQA4+SCAN_COOP+PREAMBLE_COOP+NORM_QUANT_FUSED, only
`VT_RMSNORM_ROW_COOP` varied; pinned prompt, 256 gen tokens, greedy,
batch 1, `examples/vllm-cli`; warm rep discarded per arm.

| Arm | runs (tok/s) | median |
|---|---|---|
| COOP unset | 73.305, 73.271, 73.228, 73.085, 73.108 | **73.228** |
| COOP=1 | 75.762, 75.799, 75.584, 75.467, 75.504 | **75.584** |

ON wins ALL five pairs, **+3.2% median**. Token identity: outputs diverge
from byte 149 (greedy tie flips from the changed reduction order — the
ratified adjudication case, coherent analytic prose both arms; raw
divergence is never presented as quality).

## Attribution

rocpd `-r true` capture at the ON config (512 tokens):
`/home/ghazni/agent-artifacts/tg200-t7/cap/jarvis/687945_results.db`.

| Kernel | /tok | avg us | ms/tok |
|---|---|---|---|
| RmsNormRowCoopKernel fused q8 | 65.0 | **11.22** | **0.729** (was 18.06 us / 1.174) |

Kernel time −38% (−0.445 ms/tok busy); GPU busy/tok 11.61 → 11.14 across
captures. Process note recorded honestly: the FIRST A/B window ran a
stale `vllm-cli` (linked before the T8 edit) and measured an inert wash —
the rocpd engagement check (Coop symbol absent) caught it, the binary was
relunk, and only then was any number recorded. Engagement evidence is now
part of the landing checklist for every engine-level A/B.

## Position

**75.6 tok/s median** on the acceptance workload this window (host load
1.3–2.1). Next budget items from the T7 re-ranking: the GDN latency trio
(Scan 0.730 + PostConv ~0.67 + NormGated 0.414 ≈ 1.81 ms/tok combined),
then wvSplitKSml's 408 GB/s vs the 598 reference. Failed-attempt ledger:
2 of 10 (T7 wash carried no kernel regression; T8 adopted).
