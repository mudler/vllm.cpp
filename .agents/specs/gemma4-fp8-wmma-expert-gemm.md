# GEMMA4-FP8-WMMA-EXPERT-GEMM — fused-dequant FP8 WMMA expert GEMM (M>1)

Issue: [#1762](https://github.com/mudler/vllm.cpp/issues/1762)

Row: `GEMMA4-FP8-WMMA-EXPERT-GEMM`

Spec-first on origin/main `66d1b0a9044e581085b0a2766246e20b8dc9c5d0`
(Bakon push-hard / coord `5183`). Not branched from PR #1758.
Dirty lab `~/llms/vllm.cpp` is not a donor.

Status: **spec only — gate amended** (Researcher `5a54` / coord `3c63`).
No kernel, no product edit, no GPU, no serve, no default-ON.
Acceptance `B/C>1.3` is GPU-dependent; this commit is paper.

## Git integration

Separate pull requests. Tonight is spec + issue + index only. The
implementation PR starts only after this spec is reviewed. One later
product PR is still the default for the kernel itself.

## Problem

KEEP-class Gemma-4-26B FP8 prefill on dual R9700 is still far under
the llama.cpp Vulkan Q8 bar (~2014 t/s @11k / ~1099 @42k vs ~3503 /
~2714). `ROCPROF_P11K_SHAREDK` (2026-08-11, KEEP, PC=0) already
resolved the whole-prefill breakdown:

| bucket | share |
|---|---:|
| attention (SharedK-WMMA) | 46.4% |
| MoE hipBLAS Tensile GemmEx (BF16) | 21.4% |
| FP8→BF16 `DequantFp8ChannelBf16` | 10.1% |
| RmsNorm | 7.8% |
| pack/gather | 2.2% |

L1 (attention) is a proven kernel win and a separate correctness
hold. This row is **L2 only**: dequant + expert GEMM = **31.5%**.

Amdahl (drop the 10.1% tax, ~2× the 21.4% GEMM):
`1 / (1 - 0.101 - 0.214 + 0.214/2) ≈ 1.263×` → ~3130 t/s @11k from
the KEEP class. Vision, not a GREEN.

## Current product path (main `66d1b0a9`)

`src/vllm/model_executor/models/gemma4_moe.cpp` `ExpertGeGLUFp8Native`
(`:96-131`):

- `T==1 && beta==0`: `vt::ExpertGeGLUFp8TopKM1`. **Out of scope.**
- `T>1`: sticky `vt::DequantFp8ChannelBf16` of `W_gu` and `W_dn` into
  BF16 scratch, then `vt::MatmulBT` + `vt::GeluAndMul` +
  `vt::MatmulBTAlphaBeta`. Comment at `:111-112` states the existing
  `Fp8ChannelGemmMKernel` is serial-in-M and is **not** the product
  path.

`VT_GEMMA4_CUSTOM_EXPERT=1` today gates a **BF16** custom decode
path (`:1078-1081`, `:1499-1529`). It does not replace the FP8 T>1
dequant+Tensile path. This row must not change T=1 decode meaning.

`VT_GEMMA4_PREFILL_GEMM_M` default **2048** (`docs/ENVIRONMENT.md`).
hipBLASLt FP8 W8A8 was already slower than BF16 GemmEx on gfx1201
(`:1102-1103`). Lt FP8 is not the comparator.

## Parked probe (binding)

2026-08-16 isolated scratch probe `a70e` is **not** a product kernel
and is **not** "the kernel has never been written" in the throwaway
sense. It **failed both gates** at M=2048: **0.126×** production and
final-Y NRMSE 0.02468 / maxrel 0.117. Notes:
`~/llms/llms-research-notes/ROCM_GEMMA4_FP8_WMMA_EXPERT_SPEC_DRAFT_2026-08-16.md`.

Reopen only a **materially different construction**. Do not land or
re-time that probe. Do not treat scalar-vs-microkernel uplift as
progress.

## Upstream / hardware anchors

- rocWMMA `float8_t,float8_t,float32_t,16,16,16` on ROCm 7.2.4.
  No BF16×FP8 specialization. Activation E4M3 quant is required.
- gfx1201 (R9700) first slice only.
- Gemma-4-26B dims: **H=2816**, **I=704** (N and K are multiples of 16).
- Profile: `~/llms/llms-research-notes/ROCPROF_P11K_SHAREDK_2026-08-11.md`
- Histogram: `~/llms/llms-research-notes/GEMMA4_P11K_M_HIST_2026-08-16.md`
  (raw `~/llms/logs/gemma4-m-hist.jsonl`)

## Real M distribution (measured count / token mass)

One KEEP p11k request, T=11051 (8192+2859), E=128, top_k=8,
`GEMM_M=2048`, `DEQUANT_PIPE=0`, PC=0.

| | value |
|---|---:|
| routed expert calls | 5,115 |
| mean M | 518.5 |
| p50 M | **29** |
| p90 / p95 / max | 1,561 / 2,837 / 8,185 |
| token mass M≥2048 | **70.8%** |
| after cap, exact-2048 chunks | **57.2%** of tokens (741 / 5856 launches) |
| 256≤M<2048 | 23.3% tokens |
| M<256 | 5.9% tokens (76.1% of **calls**) |

**Optimise the 2048 chunk, not p50=29.**

Count / token-mass data are **not** time data. Token mass is **not**
an acceptable proxy for the integration gate (Researcher `5a54` /
coord `3c63`). Per-chunk GPU µs were never captured (SIGTERM /
no `atexit`). Do not relabel tok% as a 1.3× claim.

Dequant cache on that run: **86.4% miss / 13.6% hit**, not overlapped.
**Independence of that mix from M is unmeasured.** Do not assume
86.4/13.6 applies uniformly across the frozen M buckets.

## Scope

In:

- default-OFF `VT_GEMMA4_CUSTOM_EXPERT=1` branch for **FP8 T>1 only**;
- rowwise BF16→E4M3 activation quant (timed, not hidden);
- rocWMMA 16×16×16 FP8×FP8→F32 GEMM for gate/up and down;
- channel-scale + row-scale epilogue; existing `GeluAndMul` between;
- exact fallback to current dequant+hipBLAS on unset / 0 / T=1 /
  wrong arch / wrong dims / unsupported layout / setup failure;
- isolated full-pipeline gate vs **production**, then one product
  wire in `ExpertGeGLUFp8Native` immediately before the T>1 dequant.

Out:

- T=1 decode GEMV / WMMA changes;
- attention / SharedK / SWA / RmsNorm;
- GGUF / #523;
- hipBLASLt FP8 as the win condition;
- default-ON;
- GPU/serve on this spec commit;
- repeating the `a70e` construction.

## Design

Gate/up: `A[M,H] @ W_gu[2I,H]^T -> GU[M,2I]`
Down: `Act[M,I] @ W_dn[H,I]^T -> Y[M,H]`

Weights stay OCP E4M3 with one BF16 scale per output channel.
Activations are quantized per row (scale = max(abs)/448, satfinite
E4M3). Accumulators are F32. This is **not** bit-identical to the
BF16-activation production path.

Host eligibility (cached once, no getenv on the hot path):

- `VT_GEMMA4_CUSTOM_EXPERT=1`
- ROCm gfx1201
- FP8 native expert, `M>1`, `H=2816`, `I=704`
- bool API: false means the caller **must** run today's path
- no `hipMalloc` / `hipFree` / `hipSetDevice` on the hot path
- default-OFF must not allocate eager VRAM

First candidate: single-wave CTA per 16×16 C tile. Multi-wave is a
later isolated arm (known gfx1201 hang class). Compile gate: zero
spills, static LDS ≤ 65536. N/K tails fail closed.

## Tests and gates

### Numerical (before any timing claim)

Against production on identical inputs, final Y:

- no nonfinite values
- correlation ≥ 0.999
- NRMSE ≤ 0.02
- max-abs / max(1, max-ref) ≤ 0.05

Predeclared. Reviewer may tighten, never relax after seeing speed.
Candidate-only RED: mutate the candidate epilogue scale or M-tail
mask and show the tensor gate fails while production is unchanged.

Shapes: M=2048 primary; `{512,1024,1536}` mid; `{2,15,16,17,32,64,128}`
correctness / launch-tax. Do not use M=1 to decide prefill viability.

### Integration speed

Full sequence (both quants + WMMA GU + existing GeLU + WMMA down)
vs the actual production dequant+Tensile path.

Delete the phrase *time-weighted M histogram*. The scoreable gate is
the count-weighted measured-time aggregate over frozen buckets:

```
B = sum_{i,s} n[i,s] * latency_prod[i,s]
C = sum_{i,s} n[i,s] * latency_cand[i,s]
PASS iff B/C > 1.3
```

- `i` = frozen exact routed/chunked M
- `s` = frozen cold/hit cache state
- `n[i,s]` = call counts from the frozen histogram (not token mass)
- `latency_*[i,s]` = **measured** device latency for that exact
  `(M, cache-state)` cell

**Conjunctive, not substitutes:**

1. literal `>1.3×` at **M=2048**, **and**
2. literal `>1.3×` at every preregistered mid-mass shape
   `{512,1024,1536}`, **and**
3. the aggregate `B/C > 1.3`.

The aggregate alone can hide a regression in a shape that carries
real mass. Token-mass weighting is forbidden. Do not fill missing
`(i,s)` cells by assuming the 86.4/13.6 mix is independent of M.

A **direct replay** of the frozen M/state sequence with total elapsed
comparison is stronger than reconstructing `B`/`C` from per-bucket
means; prefer it when the vehicle can be built.

This gate is **GPU-dependent**. Spec and static implementation may
proceed; speed GREEN, promotion, and default-ON stay withheld until
a reviewed device arm scores `B` and `C`. If only the inner WMMA
kernel clears 1.3×, park. Scalar-only uplift is insufficient.

### Later (not this spec commit)

Paris / arithmetic `63` with candidate witness >0. Serving A/B
≥5% at both ~11k and ~42k under KEEP topology, `VT_ROCM_SKINNY=0`
recorded, before any default change. Strategic 3k is context, not a
waiver.

## Risks

- `a70e` already lost to production by ~8× at the shape that matters.
- Activation E4M3 can fail the numerical gate even if WMMA is fast.
- Token-mass ≠ measured latency; a 1.3× claim from tok% is a false GREEN.
- `VT_GEMMA4_CUSTOM_EXPERT` already means something else on decode.
- wvSplitK is default-on for gfx12 on this base; any later GPU arm
  records `VT_ROCM_SKINNY=0` or the pre-sync gate.

## Stop conditions

- Repeat of `a70e` construction or any candidate ≤1.3× vs production
  at M=2048.
- Numerical gate or RED fails.
- Spills, LDS > 64 KiB, or only a multi-wave-hang geometry is fast.
- Win requires changing T=1 decode, checkpoint format, or KEEP
  defaults.
- Relabelling token mass as `B`/`C`, or filling `(i,s)` cells from
  an unmeasured cache-state×M independence assumption.

## Evidence to attach later

Profile + histogram SHAs; compiler/gfx/HSACO; microbench CSV;
tensor + RED logs; host fallback/witness matrix; Paris/63; serving
A/B only after review. Outcome section on promote or park.
