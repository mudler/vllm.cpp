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
CPU oracle asserted by the existing `tests/vt/test_rocm_quant_dot.cpp`.
A fresh configure and build at source commit
`09da0553c880a9233dc80aba26ae8aab97aaa825` recorded 841 assertions across
19 cases. The earlier 132,094-assertion claim came from a stale ROCm 7.14-era
binary whose test lattice no longer matched the source; it is historical
provenance, not a current gate count. Expected: epilogue + standalone quant
drop from ~50µs toward ~10µs ⇒ up to ~4.5 ms/tok.

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

## T5a re-attribution and T5b — the attention fallback

Fresh rocpd capture at a5bfddb0 (512 tokens): GPU busy 15.21 ms/tok.
Top items: wvSplitKSml bf16 o_proj 2.31 (408 GB/s ≈ 68% of the ~598 GB/s
board peak with the donor-tuned split-K kernel — recorded near-roofline, no
ceiling declared); PagedAttnOnlineIf 2.20; KQuantGemvMmvq Li0 big-grid 1.81
(194 GB/s effective); GdnScanK 1.45.

The attention item was NOT a kernel deficiency but a ROUTING hole: the GGUF
dense path feeds f32 queries, which excludes every bf16 decode kernel, and
the f32-Q DecodeGqa arm (T3a) hard-required d == 256 while this model has
d == 128. T5b (`5b71c8a4`) adds the EPL=4 instantiation behind the existing
opt-in `VT_ATTN_DECODE_GQA4=1`. 276µs/call of serial per-key __syncthreads
walk replaced by the warp-strided geometry.

## T5b result — acceptance A/B, interleaved x5 pairs

| Arm | warm runs | median |
|---|---|---|
| GQA4=1 | 69.851, 69.902, 67.660, 69.764, 69.780 | **69.780** |
| GQA4 unset | 61.519, 61.468, 61.475, 61.441, 60.661 | 61.468 |

ON wins all five pairs, **+13.5% median**. Near-tie adjudication: the ON
arm's 256-token gate-prompt output is BYTE-IDENTICAL to the original
pre-campaign baseline output (cmp over completion bodies) — zero tie flips
on this workload despite the reduction-order change. Owed before any
DEFAULT flip of `VT_ATTN_DECODE_GQA4`: the full teacher-forced logprob-band
ceremony per `.agents/specs/rocm-m4-oracle.md` on a gate model; until then
the flag rides the campaign config like its siblings.

Pre-existing-failure note: `test_gguf_keep_quant` (7 cases) and one
`test_backend_cross_device` case fail identically on the pristine head
without T5b — native-build configuration issues owned separately from this
lever.

Position after T5b: **69.8 tok/s median** (14.3 ms/tok) vs the 200 tok/s /
5.00 ms/tok target. Next budget: GemvMmvq weight-streaming efficiency,
GdnScan latency, RmsNorm epilogue residue (~18µs × 65/tok).

## T5c — nontemporal weight loads in KQuantGemvMmvqRow: CLOSED NEGATIVE

Hypothesis: the donor wvSplitKSml streams weights with
__builtin_nontemporal_load; the MMVQ row body's memcpy weight loads might
gain the same way (weights stream once per token). Implementation touched
only load policy (Wq/Wh/W0-W2 nontemporal; shared activation q8 temporal);
bit-exact by construction, test_rocm_quant_dot 12/12·797 green.

Acceptance window x5 (same config as T5b ON): 69.358, 69.247, 67.775,
69.294, 69.218 → median **69.294** vs T5b's 69.780 — no win (-0.7%,
cross-window noise at best). REVERTED (byte-restored via git checkout,
rebuilt clean). The donor's policy does not transfer: the MMVQ row body is
dp4a/reduction-latency bound, not L2-capacity bound. Next attack on this
family would need a geometry change (row-per-wavefront coalesced ki walk),
which is a rewrite, not a lever.

## T6a result — cooperative GDN scan (VT_GDN_SCAN_COOP=1)

Warp-per-row remap of GdnScanK (commit 640d9418): lanes walk ki coalesced,
dots reduce through a fixed shfl_down tree, rows iterate warp-strided.
Acceptance A/B interleaved x5:

| Arm | warm runs | median |
|---|---|---|
| COOP=1 | 73.017, 73.061, 73.068, 71.863, 73.144 | **73.061** |
| donor walk | 69.942, 66.846, 69.641, 69.823, 69.820 | 69.820 |

COOP wins all five pairs, +4.6%. cross_device recurrence NMSE green under
the flag (24/25; the one failure is the pre-existing native-build case).
Near-tie adjudication: gate-prompt output diverges at char 204
("Transformers process input..." vs baseline "it processes input...") — a
greedy tie flip from the changed dot-reduction order; both streams are
coherent analytic prose with identical structure. Full teacher-forced
logprob-band ceremony owed before any default flip; until then the flag
rides the campaign config.

Position: **73.1 tok/s median** (13.7 ms/tok wall). Next budget:
AttnQkNormRopeGateK (8 calls/tok @ 88us on one 256-thread block),
RmsNormRow fused-epilogue residue (~18us x 65/tok), GemvMmvq geometry.

## T6b result — cooperative attention preamble (VT_ATTN_PREAMBLE_COOP=1)

Warp-per-item remap of AttnQkNormRopeGateK. Acceptance A/B interleaved x5:

| Arm | warm runs | median |
|---|---|---|
| COOP=1 | 76.667, 76.595, 76.396, 76.334, 76.204 | **76.595** |
| donor walk | 73.220, 73.196, 73.176, 73.022, 73.205 | 73.196 |

ON wins all five pairs, +4.6%. cross_device green under the flag.
Near-tie adjudication: output diverges from the T6a stream at char 285
("...mechanism to weigh the import..." vs "...to capture long-ran...") —
another greedy tie flip, coherent prose both sides. Teacher-forced
ceremony remains owed before default flips of the three opt-in arms
(GQA4 / GDN_SCAN_COOP / PREAMBLE_COOP).

## Session-close attribution (T6b config, rocpd 512 tokens)

GPU busy **12.13 ms/tok** (wall ~13.1 = 76.6 tok/s); dispatch gap ~1 ms.
Next-session starting table:

| Kernel | ms/tok | note |
|---|---|---|
| wvSplitKSml<1,bf16> o_proj | 2.30 | 408 GB/s of ~598 peak; donor-tuned; near-roofline |
| KQuantGemvMmvqK Li0 big-grid | 1.81 | 56.8us/call; dp4a-tuned; needs GEOMETRY rewrite (coalesced ki walk) not a load-policy tweak |
| RmsNormRowKernel fused | 1.18 | epilogue residue: nsb threads still serial-ish per row |
| GdnScanCoopK | 0.78 | was 1.46 pre-T6a |
| KQuantGemmK lm_head class | ~1.17 total | large-grid GEMMs |
| GdnPostConvChunkedK | 0.65 | |
| GemvMmvq other grids | ~1.48 | |
| QuantizeQ8KK standalone | 0.53 | post-T5a |

Session ledger: baseline 49.97 -> 76.60 tok/s median (+53%). Adopted:
T5a shared-quant-body vectorization (+23%), T5b d128 f32-Q DecodeGqa arm
(+13.5%), T6a cooperative GDN scan (+4.6%), T6b cooperative attn preamble
(+4.6%). Closed negative: T5c MMVQ nontemporal loads (wash, reverted).
Failed-attempt count against the goal's cap: 1 of 10.
