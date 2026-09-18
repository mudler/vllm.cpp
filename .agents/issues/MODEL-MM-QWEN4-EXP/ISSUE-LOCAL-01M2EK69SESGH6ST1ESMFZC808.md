ID: ISSUE-LOCAL-01M2EK69SESGH6ST1ESMFZC808
Title: qwen4_exp decode sustains ~4% of GB10 memory bandwidth while this tree sustains 81% on another model, so the reference gap is an efficiency deficit and not a ceiling
Row: MODEL-MM-QWEN4-EXP
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-14
Closed: 2026-09-14

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

**ATTRIBUTED 2026-09-14, and the attribution FALSIFIES this issue's own
headline.** The per-kernel work item above is done, from
`tests/vllm/models/qwen4_exp_gguf_manifest.inc`, the loader's residency rules and
the committed kernel table, with no GPU, no lease and no `ncu`. The full section
is `.agents/specs/qwen4-exp-flash-next.md`, "### THE PER-KERNEL BANDWIDTH
ATTRIBUTION".

**The stop condition fired.** This issue derives ~0.85 GB per step. The manifest
gives **6.792 GB**, an **8.0x** disagreement, so both derivations are stated
rather than one being preferred:

| | this issue | the manifest |
|---|---|---|
| active GEMM params/token | 4.231 G | 6.625 G |
| average width | IQ1_S, ~1.6 bit | 8.20 bit |
| bytes/step | ~0.85 GB | **6.792 GB** |
| achieved | 11.0 GB/s, 4.0% of peak | **88.3 GB/s, 32.4% of peak** |
| floor | 3.1 ms, ~320 tok/s | **24.88 ms, ~40 tok/s** |

This issue's derivation priced the entire model at IQ1_S, which is in fact only
68 of the 96 routed expert gate/up towers; omitted the hyper-connections
(668.4 MB/step), the `lm_head` (357.6 MB) and the MoE router (125.8 MB); and gave
all 48 layers one attention shape when 36 are Gated DeltaNet and 12 are QSA. Its
LARGEST omission is not a pricing error at all: `attn_qkv`, `attn_gate` and
`ssm_out` of all 36 linear layers **dequantize to bf16 at load**, because the
V-head reorder the released 16-vs-48 config forces cannot be applied to a k-quant
block stream (`qwen4_exp_weights.cpp:277-395`, and its own comment at `:300-302`).
That is 4.152 GB/step against 1.508 GB in the file: **2.75x, and 61% of all
decode weight traffic.**

**What was attributed.** Four kernel rows carry a committed average duration, and
ranked by achieved bandwidth ascending they are: `QuantDotGemmGroupedKernel`
(IQ1_S expert gate/up, 68 calls/step, 3.200 MB) at **41.6 GB/s, 15.2%**;
`QuantDotGemmGrouped32Kernel` (IQ4_NL `ffn_down_exps`, 48 calls, 9.216 MB) at
**56.5 GB/s, 20.7%**; cuBLAS `gemvx` #1 (71.9 calls, two of the three large GDN
projections) at **129-172 GB/s, 47-63%**; and
`QuantDotGemmKernel<WType 4, float>` (the Q4_K `lm_head`, 357.581 MB, 2.49 ms) at
**143.6 GB/s, 52.6%**. At group level the cuBLAS family runs **4.343 GB in
26.76 ms = 162.3 GB/s (59.5%)** and our own `QuantDotGemm*` family **2.449 GB in
26.22 ms = 93.4 GB/s (34.2%)**. A full per-call-site byte inventory covering all
6.792 GB is in the spec section.

**So the separation this issue asked for exists, but not where it expected.**
Nothing is at 1-5% of peak. cuBLAS is the larger half of the GEMM bill because it
moves 1.8x the bytes, not because it is slow, and it is the MORE efficient half.
The one genuinely low band is our own grouped k-quant kernels at 15-21% over
778 MB/step. And item 2 of this issue is also corrected: the `lm_head` is not
"roughly 19% of peak", it is **52.6%**, because it is Q4_K and not IQ1_S.

**What could NOT be attributed**, all recorded in the spec section: ten of the
roughly fourteen GEMM rows have no committed average duration, because the full
25-row `nsys stats` output was never committed (the capture `4e36bbae` is
retained, so re-running `nsys stats` on it closes this with no model load); which
two of the three large GDN projections form `gemvx` row #1, hence that row's
range; the two grouped-kernel durations, which come from the ~1600-token capture
at `ee0644eab` rather than from `4e36bbae`; `QuantDotGemmQ8_0Kernel`'s 242
calls/step, which resolves two ways 28 MB apart; and whether the PLE block runs
on every decode step (34.9 MB, 0.5%).

**Item 3 of this issue stands and is now quantified.** Do not start from the
q/k/v merge: at batch 1 the activation is under 0.1% of the bytes a GEMV moves.
Start with the Gated DeltaNet keep-quant repair, which is 2.644 GB/step and the
only item that moves the floor from ~40 tok/s to ~66 tok/s -- that is, the only
item without which the reference's rate is not reachable on this artifact at all.

Resolved by row `MODEL-MM-QWEN4-EXP`, branch `row/MODEL-MM-QWEN4-EXP-bwattrib`.
