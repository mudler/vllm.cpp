# GFX1100-TG200 — T5-era baseline, lever-C 4B adjudication, fresh budget table

Date: 2026-08-25. Host: local RX 7900 XTX (gfx1100), NATIVE host build (no
container): ROCm userland 7.2.53211 at `/opt/rocm`, driver reports gfx1100,
`-DVLLM_CPP_HIP_ARCHITECTURES=gfx1100`. Build `build-hip` at branch head
`e0586593`. Checkpoint sha256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
(re-verified lineage from levc attribution; file unchanged since Aug 21).
All GPU legs inside one gpu-ctl lock window; standing serve parked via
reservation; host load 0.45 at window start.

## Baseline acceptance gate (full-stack config)

`VT_GEMV_MMVQ=1 VT_SKINNY_BF16=1 VT_NORM_QUANT_FUSED=1`, canonical prompt
(109 prompt tokens), `--max-tokens 256 --temperature 0 --seed 0`, batch 1,
`--repeat 6` (rep 1 warmup discarded, T1a convention):

47.517 (warmup), 50.032, 49.971, 49.970, 49.934, 49.586 →
**median 49.97 tok/s** (reps 2-6). Coherent analytic prose, all length-finish.

## Lever-C adjudication ON THE 4B (the adoption measurement was 0.8B-only)

Interleaved same-window pairs, warm reps, 5 pairs, only flag varied:

| Arm | warm runs | median |
|---|---|---|
| `VT_NORM_QUANT_FUSED=1` | 49.993, 49.954, 49.822, 49.818, 49.887 | 49.887 |
| `VT_NORM_QUANT_FUSED=0` | 50.827, 50.794, 50.718, 50.741, 50.672 | **50.718** |

OFF wins ALL five pairs, −1.6% for ON. Token coherence: both arms stream
coherent text. Verdict: **lever-C's default-config enablement does not carry
to the 4B gate workload.** Root cause below; the fusion CONCEPT survives only
if the epilogue stops being slower than the launch it removes.

## Fresh attribution (rocprofv3 rocpd, head e0586593, full-stack config)

Capture `/tmp/tg200-prof-base/jarvis/879532_results.db`, 2 reps = 512 tokens.
GPU busy 9732 ms / 512 tok = **19.0 ms busy/tok** vs 20.0 ms wall/tok: the
dispatch gap is ~1 ms/tok (graph capture working); the budget is GPU-busy
dominated now. Per-token table (family level):

| Kernel | /tok | avg µs | ms/tok | note |
|---|---|---|---|---|
| RmsNormRowKernel (FUSED q8 epilogue instantiation) | 64.7 | 53.8 | **3.48** | was 29.3/tok @ 7.5µs pre-lever-C |
| KQuantGemvMmvqK Li0/Li2 (all grids) | ~85 | 27–57 | **~4.0** | FFN/attn proj matvecs, 194 GB/s effective at the dominant grid |
| wvSplitKSml<1,bf16> o_proj | 71.7 | 32.1 | 2.31 | 13 MB weights/call ≈ 408 GB/s, near-roofline-ish |
| PagedAttnOnlineIf | 8.0 | 277.0 | 2.21 | grows with context |
| QuantizeQ8KK standalone (non-fusable sites) | 39.8 | 49.7 | 1.98 | sites 5+6 from levc census |
| GdnScanK | 24.0 | 60.7 | 1.46 | |
| KQuantGemmK large-grid (lm_head class) | ~1.0 | 1319–5760 | 1.24 | |
| AttnQkNormRopeGateK | 8.0 | 88.5 | 0.70 | |
| GdnPostConvChunkedK | 23.9 | 27.1 | 0.65 | |
| RmsNormGatedK | 23.9 | 17.2 | 0.41 | |

## The pathology (root cause, one shared body)

`QuantQ8KSBlock` (src/vt/rocm/rocm_act_quant.h) is a SINGLE-THREAD serial
routine: 2 passes over 256 elements, scalar loads through a `const void*`
with the ActDT `switch` re-executed per element, serial bsums. Every consumer
instantiates it: the standalone quant (128 threads = 128 sbs in parallel, each
serial), the fused norm epilogue (nsb ≤ 10 of 256 threads active), and the
MMVQ LDS prologue. ~50µs per super-block-set against a <2µs memory floor is
the same 25–100× waste class the spec predicted under the next rock.

## Next hypothesis (top-item attack)

Rewrite the SHARED body only: unswitch ActDT, vectorize loads (elem0 is a
multiple of 256 → 16 B alignment guaranteed for bf16/f32), keep the amax scan
in strict element order (first-occurrence lowest-index tie-break preserved
exactly), quant pass element-independent, bsums integer-exact. Byte-exact vs
CPU oracle asserted by the existing `tests/vt/test_rocm_quant_dot.cpp`
(132k assertions incl. tied-amax adversarial). Expected: epilogue + standalone
quant drop from ~50µs toward ~10µs ⇒ up to ~4.5 ms/tok.

## Honest notes

- Native-host build is a NEW configuration for this campaign (prior evidence
  ran in `rocm-dev:7.14.0` containers, `/work` scratch which no longer
  exists). Absolute numbers here are the first native-build baselines;
  cross-era deltas are indicative, not paired.
- `.env` created in the shared checkout (DEVICE_ARCH/TOOLKIT/COMPILER/
  CHECKPOINT_ROOT observed on this machine; GPU_LOCK pointed at
  `/home/ghazni/gpu-coord/gpu.lock` so script fallbacks serialize with

## T5a result — shared-body vectorization (same binary, interleaved x5 pairs)

`QuantQ8KSBlock` unswitched per dtype and vectorized to 16-byte loads (amax
scan kept in strict ascending element order; quant pass element-independent;
bsums integer-exact; scalar fallback on any misalignment). Gate:
`test_rocm_quant_dot` 12/12 cases, 797 assertions SUCCESS under the lock.

Acceptance workload, only `VT_NORM_QUANT_FUSED` varied, other levers ON:

| Arm | warm runs | median |
|---|---|---|
| FUSED=1 | 61.665, 61.499, 61.553, 61.466, 61.412 | 61.499 |
| FUSED=0 | 61.787, 61.741, 61.606, 60.978, 61.609 | 61.609 |

- vs the 49.97 baseline: **+23.1%** (FUSED=0 arm) — from the quant-body fix
  alone; both arms benefit because all three consumers share the body.
- Lever-C fusion is now a near-tie wash (−0.2%, winners mixed): the ~49µs
  launch it removes shrank to roughly the kernel's real cost. Adjudication
  deferred until the next budget table decides whether the epilogue stays.
- Token identity: engine output BYTE-IDENTICAL to the pre-change baseline
  build on the gate prompt (cmp over stdout bodies, 1415 bytes,
  `/tmp/base.body` vs `/tmp/t5.body`), matching the bit-exactness claim.

New position: **~61.6 tok/s median** (16.2 ms/tok) against the 200 tok/s /
5.00 ms/tok target. Next attribution re-take prices what the ~3 ms/tok of
killed pathology left at the top.
