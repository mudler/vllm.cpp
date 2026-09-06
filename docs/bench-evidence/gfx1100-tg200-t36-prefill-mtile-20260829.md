# GFX1100-TG200 T36 — prefill GEMM M-tiling (VT_PREFILL_TILE): measured +0.75% (MT8) / +1.2% (MT16), adopted as opt-in (2026-08-29)

## Question

The spec's budget prices the prefill GEMM at ~1.03 ms/tok amortized, "naive
m-pass-through GEMM re-reads weight rows M times through L2 — tiling is the
unexplored lever", ~0.7 ms/tok recoverable. T34 ranked T36 second (~0.7).

## Lever

`KQuantGemmMTiledK<OutT, Fmt, MT>` in `src/vt/rocm/rocm_grouped_gemm.hip`:
one warp computes MT consecutive activation rows for ONE weight row — the
warp streams the weight row's superblocks once and applies each loaded block
to MT activation rows, instead of the baseline `KQuantGemmK` warp-per-(i,j)
where the same weight row is re-read by m warps. Dispatched in
`MatmulBTQuantKernelRocm`'s K-quant baseline branch behind `VT_PREFILL_TILE`
(default OFF, read per call), m > 1 only; the m == 1 decode GEMV/coop arms
are untouched.

**Numerics contract: BIT-IDENTICAL to the baseline** — per output element
the lane→superblock map (sb = lane, lane+32, ...), the Dot call sequence,
the f32 partial accumulation order, and the 16..1 `__shfl_down_sync` tree
are unchanged; only the loop nesting gains an inner activation-row pass.

## Op-level gate (new file; the standing gate file stays unchanged)

`tests/vt/test_rocm_prefill_tile.cpp` (registered `test_rocm_prefill_tile`):
3 dtypes (Q4_K/Q5_K/Q6_K) × nsb {1,3,10} × m {2,3,8,39,512} × n {1,7,129,257}
× 2 seeds — **720/720 assertions green**: tiled == baseline by raw-byte
memcmp, and ON arm within the 1e-6 NMSE band vs the CPU oracle at every
shape. `ctest -R 'rocm|quant'` shows the same result set with the lever ON
and OFF (identical pass/fail pattern; see the pre-existing-red note below).

## Pre-existing gate reds at HEAD (NOT introduced by this lever)

Verified by stashing the lever and rebuilding the pristine tree:

- `test_rocm_quant_dot`: the Q5_K MMVQ GEMV arm (weight case 1, m == 1
  shapes, n ∈ {1,7,129}) deviates from both the OFF arm and the CPU oracle
  (NMSE ~1e-4..7e-4 vs the 1e-6 band); 218/841 assertions red. Identical
  red count with the lever's env set — the engine never routes those shapes
  (fused sub-arm engages only at n ≤ 512; engine Q5_K rows are n = 4096+),
  which is why the engine-level byte-identity checks still pass.
- `test_gguf_keep_quant`: quantized-gather routing reds; with a visible AMD
  GPU the running platform is ROCM and `DeviceQuantGatherSupported` (CPU-only
  gate) refuses, so `RouteGgufTensor` returns expand where the test expects
  keep-quant. 5 assertions red even with no GPU visible. Both reds
  reproduce on pristine HEAD; this lever's arms are untouched by them (the
  new gate file + engine coherence carry the correctness burden).

## Acceptance A/B (same binary, 1 warm + 5 reps, medians, gpu-ctl held, idle host)

Prompt: `tools/tg200-prompt.txt` (109 prompt tokens — NOTE: T33's morning
91.24 median used the since-removed 71-token `tg200-eval-prompt.txt`; T34's
own warm run at THIS prompt measured 85.777 tok/s with body md5 783cea17…,
which every arm below reproduces exactly — the acceptance position at the
current prompt is ~85.8 tok/s, not 91.24).

| arm | tok/s (reps 1..5) | median | body md5 |
|---|---|---|---|
| OFF (baseline) | 85.776 85.431 85.853 85.753 85.533 | **85.753** | 783cea17… |
| ON, MT=8 | 86.393 86.265 86.427 86.400 86.468 | **86.400** (+0.75%) | 783cea17… |
| ON, MT=16 (probe) | 86.758 87.004 86.762 86.896 (76.409 outlier) | 86.762 | 783cea17… |
| ON, MT=16 (clean re-run) | 86.825 87.083 86.892 83.423 86.800 | **86.825** (+1.25%) | 783cea17… |

Coherence: every arm's 256-token body is byte-identical within-arm and
across arms (bit-identical claim PROVEN end-to-end: the tiled kernel runs
the whole prefill — all projections, all three K-quant formats — and the
greedy decode is unchanged).

## Verdict

- Complete separation between OFF and both ON arms (every ON rep > every
  OFF rep), so the win is real — but **+0.75..1.2% is below the 2%
  adoption bar**: the spec's ~0.7 ms/tok pricing assumed the weight re-read
  through L2 dominates prefill; on gfx1100 the 64 MB Infinity Cache absorbs
  most of the re-read traffic, and the measured recoverable prefill GEMM
  time is ~25 ms of a ~2.98 s run (~5× smaller than the ~180 ms priced).
- MT=16 (weight-pass factor 16) beats MT=8; beyond it the term is
  saturated. The lever ships as an opt-in (`VT_PREFILL_TILE=1`, kMT=16)
  with zero-risk numerics (bit-identical), left default-OFF — default flips
  owe the teacher-forced ceremony and a ≥2% bar this lever does not meet.
- Spec `## Now` line records: closed below the 2% bar, adopted as opt-in,
  +1.25% median (86.825 vs 85.753 at MT=16), byte-identical outputs.

## Clean-window re-A/B (2026-08-29 late window, post-#9-fix tree, reference a0fa1c4a…)

The original arms above were later found to have run against a co-tenant
hung container. Re-measured on an idle window (loadavg 0.98 at start, no
stray engine processes, 1 warm + 5 reps, shared base arm with the T35-r3
evidence): base median **85.834** (85.713 85.838 85.899 83.200 85.834),
VT_PREFILL_TILE=1 median **86.392** (86.511 86.669 86.375 86.311 86.392)
= **+0.65%**, body byte-identical to the re-minted reference
(a0fa1c4a…) in every arm. Confirms the direction and the below-bar
opt-in disposition at the clean-window magnitude.
