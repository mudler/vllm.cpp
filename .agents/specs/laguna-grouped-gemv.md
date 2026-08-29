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

## W1 — MEASURED: latency-bound, the same floor as the Q8_0 sibling. The row STOPS.

Profiled on `dgx:gpu0` (GB10, sm_121a) on 2026-08-29 with Nsight Compute 2025.3.1,
`--kernel-name regex:QuantDotGemmGrouped --launch-skip 200 --launch-count 8
--set full`, against the real `UD-Q4_K_XL` @ `750f92f9`. Eight decode launches of
`QuantDotGemmGroupedKernel<5, float>` (Q5_K). Evidence:
[`docs/bench-evidence/laguna-grouped-gemv-ncu-20260829.csv`](../../docs/bench-evidence/laguna-grouped-gemv-ncu-20260829.csv).

| metric | value | reading |
|---|---:|---|
| Achieved Occupancy | **101.5%** (theoretical 93.75) | fully occupied |
| Compute (SM) Throughput | **26.1%** | not compute-bound |
| Memory Throughput | **28.0%** | **NOT bandwidth-bound** |
| L1/TEX Hit Rate | **95.1%** | over-fetch absorbed, never reaches DRAM |
| L2 Hit Rate | 42.4% | |
| **Eligible Warps Per Scheduler** | **0.42** | **the finding** |
| Active Warps Per Scheduler | 10.87 | |
| Issue Slots Busy | 14.5% | |
| Warp Cycles Per Issued Instruction | 77.6 | |
| Registers Per Thread | 43 | |
| Duration / launch | 166.5 us | grid 4480, block 128 |

### The reading: latency, not bandwidth

**W11's "BW-tuning" label is WRONG for this kernel, as it was for its sibling.**
Memory throughput is 28% — there is no bandwidth wall to tune against.

The decisive counter is **0.42 eligible warps per scheduler**. Each scheduler
holds 10.87 ACTIVE warps and yet fewer than one is READY TO ISSUE at any instant,
so issue slots are busy only 14.5% of the time at 101% occupancy. That is the
signature of memory-LATENCY exposure, and it is why compute and memory SOL are
BOTH low at once: the warps are resident and waiting, not competing for a pipe.

### It is the same floor as the Q8_0 kernel

| | grouped Q4_K/Q5_K (this) | Q8_0 (`ds4-q8-ncu`) |
|---|---:|---:|
| occupancy | 101.5% | 72-75% |
| L1 hit | 95.1% | 96.6% |
| bound by | latency (0.42 eligible warps) | latency (long_scoreboard 54-57) |

Two different kernels, two different occupancies, the same conclusion: the
dependent load-to-unpack-to-dot chain is the cost, and neither is starved of
bandwidth or of warps.

### What this REFUTES before it was attempted

The levers W11's label implied are refuted by these counters rather than by
experiment, which is the point of measuring first:

- **Vectorised loads / wider footprint** — the bandwidth lever. Memory SOL is 28%.
  There is nothing to widen into.
- **Occupancy tuning** — achieved occupancy is 101.5% of theoretical. There is no
  occupancy to recover, and this kernel has MORE than the Q8_0 one, which was
  itself not occupancy-starved.
- **A dp4a pass** — compute SOL is 26.1%; the arithmetic is not the wall.

And the axis that IS implicated, latency hiding, is the one the Q8_0 campaign
already spent five structural bricks on — aligned repack, sub-warp occupancy,
launch consolidation, the register-spill hypothesis, and multi-row/prefetch ILP —
all flat or refuted, with a recorded MEASURED FLOOR. Those are this kernel's prior
too, not a fresh menu.

### The row stops here, as `## Gates` said it would

The spec committed before the measurement: "A null result is a result. If the
counters say the grouped kernel is bound the same way Q8_0 is, this row records
that and stops, because the two kernels sharing a floor is more useful than a
sixth refuted brick." That is the measured case, and it is applied.

### What is NOT claimed

Eight launches of ONE kernel specialisation (Q5_K, `<5, float>`) on one prompt at
one context length. The Q4_K specialisation was not separately captured, and
context length moves the grid. Nothing here is a speed claim, no default changed,
and no llama.cpp denominator is quoted — W11's "~22% of peak vs llama.cpp ~76%"
inherits the #1003 supersession and is not a target.

**Tensor-core tiling was examined as the reopening candidate and does NOT apply
to decode.** It is the one mechanism that would shorten the unpack dependency
chain rather than feed more warps to it, and `cuda-keepquant-gemm.md` defers it,
so it was the obvious next row. Reading the upstream reference closes it for this
shape.

llama.cpp's MoE dispatch at `b10451` (`ggml-cuda.cu:1916`) is:

```c
const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
if (ne2 <= mmvq_mmid_max) { /* MMVQ */ }   // else MMQ
```

For **Q4_K and Q5_K**, which are exactly Laguna's expert dtypes, the Turing+ table
(`mmvq.cu:141`) falls through to `default: return MMVQ_MAX_BATCH_SIZE`, and that
is **8** (`mmvq.cuh:3`). Laguna decodes one token, so `ne2 = 1 <= 8` and **llama.cpp
takes MMVQ — warp-per-output, the structure we already have.** It reaches for
tensor-core MMQ tiles only above batch 8, which is prefill.

So there is no upstream existence proof that MMQ wins at decode on this dtype; the
reference deliberately chooses our structure at this batch size. The shape agrees:
tensor-core tiles want at least 16 rows and Laguna's decode grouped GEMM has
`P = 10` top-k experts, so roughly six of sixteen rows would be padding even if it
were built.

Tensor-core tiling therefore remains a real deferral — for **prefill**, not for the
decode 62% that motivated this lever. A prefill row would need its own attribution
first, because Laguna's measured gap was decode and prefill was never attributed.

## The W11 lever list is now EXHAUSTED

| Lever | Disposition |
|---|---|
| #1 fused gate/up (`QuantizeQ8K` dedup) | MEASURED, [#2061](https://github.com/mudler/vllm.cpp/issues/2061): works, ~+4.28% warm, moves a token on 6 of 6 prompts, ships default-OFF |
| #2 keep-quant GEMV "BW-tuning" | REFUTED AT THE COUNTER, this row: latency-bound at 101.5% occupancy, memory SOL 28%, so the bandwidth, occupancy and dp4a levers are all refuted before attempt |
| #3 device-resident decode | DEMOTED by W11 itself: GPU-busy ~= host sync time, worth ~0.02 s/tok |
| (reopening candidate) tensor-core MMQ | NOT APPLICABLE to decode: upstream uses MMVQ below batch 8 for Q4_K/Q5_K |

Together with the Q8_0 kernel's five refuted structural bricks and its recorded
MEASURED FLOOR, the ranked plan that came out of the W7/W11 attribution is
complete. **Laguna's remaining decode cost is a memory-latency dependency chain in
the keep-quant unpack, and no lever on that list moves it.**

That is a real answer rather than an absence of one, and it is what closes the
campaign. What it does NOT say is that Laguna is at its floor for all time — it
says the enumerated levers are spent. A new lever needs a new mechanism and a
fresh attribution, not another pass at this list.

## Now

`DONE` for W1. The row's question — what bounds the grouped Q4_K/Q5_K GEMV — is
answered: memory latency, at full occupancy, sharing the Q8_0 kernel's floor. The
bandwidth, occupancy and dp4a levers are refuted at the counter. No product code
was written and none should be, on this evidence.

The W11 lever list is exhausted (see the table above). Tensor-core tiling stays a
genuine deferral for PREFILL only, and would need its own attribution first —
Laguna's measured gap was decode, and prefill has never been attributed.
