# `PERF-LAGUNA-GROUPED-GEMV` — what bounds the grouped Q4_K/Q5_K GEMV

Issue [#2250](https://github.com/mudler/vllm.cpp/issues/2250). Owning row
`MODEL-TEXT-laguna-laguna-for-causal-lm`. Lever #2 of
[`laguna-s21-w7-speed-2026-07-31.md`](laguna-s21-w7-speed-2026-07-31.md) §W11,
opened after lever #1 closed ([#2061](https://github.com/mudler/vllm.cpp/issues/2061)).

## The row starts from prior measurement, not from the lever's name

W11 measured the keep-quant GEMVs at ~87% of Laguna decode GPU and labelled the
lever "BW-tuning". **That label is not load-bearing and this row does not inherit
it**, because the sibling kernel carrying the other 24.7% was measured and turned
out to be something else entirely.

**The Q8_0 half is CLOSED and must not be re-opened on a bandwidth premise.**
`ds4-q8-ncu-2026-07-30.md` plus `74a70427a` and `8779eccb7` establish, at the
counter rather than by inference:

| counter | value | reading |
|---|---|---|
| `long_scoreboard` | 54-57 | memory-LATENCY exposure |
| achieved occupancy | 72-75% | NOT occupancy-starved |
| `l1tex__t_sector_hit_rate` | 96.6% | the 16x sector over-fetch never reaches DRAM |
| `lg_throttle` | 74.2 | LSU global-load pipe saturated on the weight unpack |
| `local_ld` / `local_st` | 0 / 0 | no register spill — the SINK4 hypothesis REFUTED |

Five structural levers came back flat or refuted there: aligned repack (Brick 4),
sub-warp occupancy (Brick 11), launch consolidation (Brick 12), the register-spill
hypothesis, and multi-row/prefetch ILP ("RE-CONFIRMED WASH"). `74a70427a` records
a MEASURED FLOOR.

## Scope

| Field | Content |
|---|---|
| In | `QuantDotGemmGroupedKernel` (`src/vt/cuda/cuda_quant_dot.cu:815`), the Q4_K/Q5_K grouped routed-expert GEMV, 62.1% of Laguna decode GPU by W11's `cuda_gpu_kern_sum`. W1 measures what bounds it; any lever is chosen from those counters and is a LATER wave |
| Out | `QuantDotGemmQ8_0Kernel` — closed above, and re-opening it needs new evidence rather than a new attempt; the fused gate/up arm (#2061, done); the fp4/NVFP4 Laguna path, a different branch with a different bottleneck; device-residency, DEMOTED by W11 |
| Gate model | `unsloth/Laguna-S-2.1-GGUF UD-Q4_K_XL` @ `750f92f9`, staged at `/workspace/ckpt/laguna-s21-ud-q4kxl/` |

## Why the kernel is a plausible candidate ANYWAY

Not because of W11's label, but because its own spec says so.
[`cuda-keepquant-gemm.md`](cuda-keepquant-gemm.md) records the current structure as
correctness-first and names the deferral in its scope: *"MMVQ warp-per-output is
the correctness-first structure; tensor-core tiling is a later speed brick."*

The kernel is one warp per output element, lanes striding the K super-blocks with
a warp reduce, mirroring llama.cpp's `mmvq.cu` structure but with our Q8_K
numerics. It has never been bandwidth- or ILP-tuned on CUDA. So there is a
documented, deliberately-deferred axis here — which is a different thing from
assuming the kernel is bandwidth-bound.

## W1 — the measurement, and what would make it wrong

Profile `QuantDotGemmGroupedKernel` with `sudo ncu` inside a real Laguna decode on
`dgx:gpu0`, capturing the SAME counter set the Q8_0 work used so the two kernels
are directly comparable: `long_scoreboard`, achieved occupancy, `l1tex` and `lts`
sector hit rates, bytes per sector, `lg_throttle`, `local_ld`/`local_st`, and
compute/memory SOL.

**It must profile DECODE, not the whole run.** `nsys`/`ncu` over a whole
invocation aggregates prefill with decode and folds in one-time load-path work;
that trap has already produced one wrong attribution in this tree, where a
whole-run `kern_sum` contaminated by load-time Marlin repack yielded a
"kernels already at parity" claim that a clean graph A/B later reversed. Target
the kernel by name and skip past prefill.

**The reading decides the next wave, and the four readings are not the same
lever:**

- **Bandwidth-bound** (high memory SOL, low L1 hit): vectorised loads, a wider
  per-thread footprint. This is what W11's label assumed.
- **Latency-bound** (high `long_scoreboard`, healthy occupancy): the Q8_0 story
  repeating, in which case its five refuted levers are the prior and the expected
  value of trying them again is low.
- **LSU-pipe-bound** (high `lg_throttle`): the unpack instruction count is the
  cost, and the lever is fewer, wider loads per block rather than more of them.
- **Occupancy-bound**: register or shared-memory pressure — and note the Q8_0
  kernel was NOT occupancy-starved at 72-75%, so this would be a genuine
  difference between the two rather than a shared cause.

**A null result is a result.** If the counters say the grouped kernel is bound the
same way Q8_0 is, this row records that and stops, because that is the finding —
the two kernels sharing a floor is more useful than a sixth refuted brick.

## Gates

- W1 is measurement-only. NO product code, no default flipped, no speed claimed.
- Counters recorded with the run's own artefact paths, kernel name, launch
  configuration and the checkpoint revision, so the numbers can be re-derived
  rather than re-quoted.
- Any later lever gates on bit-exactness against the current kernel first, then
  on a warm order-balanced A/B — the shape #2061's W3b arrived at, after its
  single-run predecessor produced a 2.15x figure that was a cold-cache artefact.
- No llama.cpp denominator is quoted. `27.8 tok/s` and every ratio derived from it
  stay superseded under [#1003](https://github.com/mudler/vllm.cpp/issues/1003);
  W11's own "~22% of peak vs llama.cpp ~76%" inherits that supersession and is
  therefore ALSO not quotable as a target.

## Now

`READY`. Spec committed, no implementation. Next action is W1, which needs a GPU
lease and the staged checkpoint, and produces counters rather than code.
