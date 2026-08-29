# GFX1100-TG200 — T9: cooperative gated-norm remap adopted (+2.6% median)

Date: 2026-08-25. Host: local RX 7900 XTX (gfx1100), native build
`build-hip`, branch `row/GFX1100-TG200` at the T8 landing plus this change.
Checkpoint sha256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
A/B window 23:33:49Z–23:35:35Z under gpu-ctl hold.

## Change

`RmsNormGatedCoopK` behind `VT_GDN_NORMGATED_COOP=1` (default OFF;
allowlist-registered). The donor kernel runs ONE THREAD PER ROW
(`<<<rows, 1>>>`) — each row walks d twice serially, 24 launches/tok x
17.25–18.4 us = ~0.42 ms/tok of pure single-thread latency. The arm gives
each row a 256-thread block: strided-per-thread sumsq (the coalesced
pattern for a streaming pass), wavefront-shfl reduction with width from
`warpSize`, one cross-wavefront combine, then a strided gated store. The
float association changes; the flag rides the campaign config opt-in and
the teacher-forced ceremony stays owed before any default flip.

## Correctness gate

Full suite **14/14 cases, 825 assertions SUCCESS**, including the new T9
case: COOP-vs-donor output NMSE <= 1e-6 on bf16 rows x d∈{256, 2560}, and
flag-inertness asserted byte-level.

## Acceptance A/B — interleaved x5 pairs, full campaign config

Config: MMVQ+SKINNY+GQA4+SCAN_COOP+PREAMBLE_COOP+NORM_QUANT_FUSED+
RMSNORM_ROW_COOP, only `VT_GDN_NORMGATED_COOP` varied; pinned prompt,
256 gen tokens, greedy, batch 1.

| Arm | runs (tok/s) | median |
|---|---|---|
| COOP unset | 75.815, 75.815, 75.722, 75.715, 74.172 | **75.722** |
| COOP=1 | 77.789, 77.705, 77.719, 77.557, 77.397 | **77.705** |

ON wins ALL five pairs, **+2.6% median**. Outputs diverge from byte 55 —
greedy tie flips from the changed reduction order, coherent analytic prose
both arms (ratified adjudication case).

## Attribution

rocpd capture at the ON config: `RmsNormGatedCoopK` 24/tok at **2.04us**
(0.049 ms/tok) vs donor `RmsNormGatedK` 18.38us (0.441 ms/tok) — a 9x
kernel-time reduction.

## Process notes (recorded honestly)

Two inert windows preceded the valid measurement, both caused by stale
artifacts rather than the lever: (1) the T9 test initially set the WRONG
env var (the T8 guard's) and could not witness engagement; (2) the engine
ran the T8-era `libvllm.so` until it was relinked after the T9 edits —
diagnosed via the rocpd DONOR-ONLY symbol check. Standing rule going
forward: every engine-level A/B window starts with an engagement witness
(kernel symbol present in the capture, or equivalent counter), and every
source edit relinks ALL consumed targets (static lib, shared lib, CLI)
before any measurement.

## Position

**77.7 tok/s median** this window (host load 2.5–4.1). Next budget by the
T7 re-ranking: GdnScanCoop (0.730 ms/tok) and GdnPostConvChunked (~0.67),
then wvSplitKSml's 408 GB/s vs the 598 reference. Failed-attempt ledger:
2 of 10 (T7 wash; T8/T9 adopted).
