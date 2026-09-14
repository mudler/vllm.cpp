ID: ISSUE-LOCAL-01M2EK69SESGH6ST1ESMFZC808
Title: qwen4_exp decode sustains ~4% of GB10 memory bandwidth while this tree sustains 81% on another model, so the reference gap is an efficiency deficit and not a ceiling
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

DERIVED from the model's own config and this row's measured step, with no GPU
and no new capture. It reframes the sojufx gap from a ceiling into a defect, and
it contradicts two conclusions this row published earlier today.

## The arithmetic

Active weights per decode token, from `qwen4_exp.h`: `hidden_size` 2560, 48
layers, MoE 512 experts top-10 with `moe_intermediate_size` 640 plus one shared
expert, attention `num_attention_heads` 24 / `num_key_value_heads` 2 /
`head_dim` 256.

```
per layer   10 x 3 x 2560 x 640   (routed)   =  49.15 M
          +  3 x 2560 x 640       (shared)   =   4.92 M
          + q 15.7 M + kv 2.6 M + o 15.7 M   =  34.00 M
                                               ---------
                                                88.15 M
x 48 layers                                  =   4.231 G active params
at IQ1_S (~1.6 bit/param)                    =   ~0.85 GB per step
```

| | step | achieved | % of 273 GB/s |
|---|---|---|---|
| **us, 400-token reference workload** | 76.8 ms | **11.0 GB/s** | **4.0%** |
| us, 600-token workload | 87.6 ms | 9.7 GB/s | 3.5% |
| sojufx at 66.17 tok/s (NVFP4, ~4 bit) | 15.1 ms | 140.1 GB/s | 51.3% |
| **this tree, ANOTHER model, SAME box** | -- | **220.8 GB/s** | **81%** |

That last row is the one that makes this a defect rather than a limit: this
repository already sustains 81% of peak on `dgx:gpu0`, recorded in its own
measurements. **Two models in one tree, twenty times apart in memory
efficiency.**

## What it means

**The bandwidth FLOOR for this model is ~3.1 ms per step, about 320 tok/s.** The
reference's 66.17 tok/s is a 15.1 ms step -- five times SLOWER than what the
memory system permits. At even 50% of peak we would be at ~161 tok/s, 2.4x past
the reference.

**So the 5.08x gap is not a physical limit and never was.** It is the symptom of
a decode path running at roughly one-thirteenth of the reference's memory
efficiency and one-twentieth of this tree's own demonstrated efficiency.

## TWO CONCLUSIONS THIS ROW PUBLISHED TODAY ARE WRONG, and are corrected here

1. **"No combination of the identified kernel leads reaches 66 without
   speculation"** was withdrawn once already for being written from two of
   twenty-five kernel rows. It is now wrong a second and more basic way: it
   assumed the remaining distance was a shortfall to be scraped together, when
   the measurement says there is roughly **20x of headroom** against the memory
   system.

2. **The MTP finding is TRUE but was over-weighted.**
   `ISSUE-LOCAL-01M2EG6R3MRCB9ENZB9840KAXX` establishes that the artifact carries
   no MTP head and no converter will emit one, and that stands. But speculative
   decoding multiplies throughput at a GIVEN step cost; it cannot explain, and
   would not fix, running at 4% of bandwidth. **It was framed as the blocker when
   the real problem is an order of magnitude larger and entirely ours.**

## Owed, and this is the next row

The 68.9% of kernel time that is GEMM/GEMV splits evenly between the cuBLAS
`gemvx` family (26.76 ms/step) and our own `QuantDotGemm*` kernels (26.22
ms/step). Neither has a per-kernel bandwidth attribution.

1. **Bytes-per-call for the top GEMM kernels.** For each of the ~12 GEMM rows,
   derive the weight bytes its shape implies and divide by its measured average.
   That gives achieved bandwidth PER KERNEL and separates the pathological ones
   from the merely imperfect. It needs the shapes, not a profiler, so `ncu`'s
   refusal (`ERR_NVGPUCTRPERM`) does not block it.
2. One datum already exists and suggests the spread is wide:
   `QuantDotGemmKernel<WType 4, float>` runs ONCE per step at **2.49 ms**, the
   `lm_head` shape; against a vocab-sized IQ1_S matrix that is roughly 19% of
   peak -- five times better than the 4% average, which means the many small
   kernels are far worse than the average.
3. Only then scope a fix. Do not start from the q/k/v merge: at batch 1 the
   activation is ~5 KB against tens of MB of weights, so merging saves launches
   and activation re-reads, neither of which is where 96% of the bandwidth went.
   **The seam requirement in `AGENTS.md` still stands on its own terms** and is
   recorded separately, but it is not the lever this issue points at.


## Resolution

-
