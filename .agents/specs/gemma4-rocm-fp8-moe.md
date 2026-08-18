# SPIKE / SPEC — Gemma-4 MoE ROCm FP8 + SharedK-WMMA (#317)

**Status:** ACTIVE lab path (dual R9700 gfx1201) · PR tip `feat/gemma4-rocm-fp8-split`  
**Claim class:** ROCm serve correctness + decode/prefill knobs (not CUDA SACRED parity)  
**Related PRs:** #234 sampler · #227 KV fail-fast · #316 SSE · #328 bare toolcall · **this**

## Problem
Gemma-4-26B MoE FP8 on consumer RDNA4 needs: resident dual-GPU experts, peer mix, SharedK-WMMA prefill, decode multi-CTA KV splits, and portable `vt::` seams so `models/` never names HIP (device-leakage).

## In scope (this PR)
| Area | Files (representative) | Gate |
|------|------------------------|------|
| Portable fused_ops / backend hooks | `include/vt/fused_ops.h`, `src/vt/fused_ops.cpp`, `backend.h` | CPU link + device-leakage |
| FP8 ExpertGeGLU / channel GEMV | `rocm_fp8_channel_gemv.hip`, `rocm_gemma4_experts.hip` | Lab A/B + quality (Paris/arith) |
| MoE host policy | `gemma4_moe.cpp` / `.h` | Lab load + decode |
| Prefill SharedK-WMMA | `rocm_paged_attn.hip` (prefill path) | Prefill eng ~2k class @11–12k |
| Decode KV splits / slide | `rocm_paged_attn.hip` | `VT_ATTN_DECODE_*` recipe |
| Env surface | `docs/ENVIRONMENT.md` `VT_GEMMA4_*` / `VT_ATTN_*` | check-env-doc |

## Out of scope
- SSE keepalives (#316), ROCm sampler (#234), bare tool parser (#328)
- CUDA SACRED gemma4-E4B text golden (dgx) — unchanged by ROCm HIP paths when `HasRocm` off
- Full MoE token-exact vs vLLM on RDNA4 (no AMD CI runners)

## Env recipe (lab binding)
See `docs/ENVIRONMENT.md` and lab `gemma4-fp8-recipe.env`:
`VT_GEMMA4_FP8_HW_CVT`, `VT_ATTN_DECODE_KV_SPLITS`, `VT_ATTN_DECODE_SLIDE_*`, resident experts, prefill GEMM_M / PEER_ACT / SharedK-WMMA.

## Automated coverage in-tree
1. **Device-leakage** — models/ must not call `vt::rocm::*` (CI `check-device-leakage`).
2. **CPU unit** — `test_gemma4_rocm_fp8_seams` (this PR): fused_ops symbols resolve; env knobs parse inert defaults on CPU.
3. **Existing** — `test_gemma4_paged_engine` / registry e2e remain CUDA/dgx optional skips.
4. **Lab (not CI)** — exclusive decode_depth_curve + Paris/READY on dual R9700; evidence in contributor lab notes, not forced into STATUS ratchet growth.

## Residuals (named, not silent)
- HIP guards polish in dense `gemma4.cpp` paths
- Broader parity matrix rows when AMD CI exists
- Further decode micro-opts stay lab-recipe until KEEP+quality

## Merge criteria (maintainer)
- Spec present (this file)
- Device-leakage OK; env documented
- CPU seam test green
- No SSE/sampler/toolcall scope creep
- Sanitize ambient reds treated as main baseline unless unique product fail

## Scope note (2026-08-11 rebuild on main)

This tip is **ROCm FP8 MoE + SharedK-WMMA + fused_ops + expert LRU/prewarm**.

**Explicitly deferred** (localai-bot hold on prior tip):
- `ForwardGemma4Layers` extract / layer-loop restructure in `gemma4.cpp`
- `Gemma4DecodeGraph` / pure-decode hipGraph driver
- Any unguarded CUDA-path forward refactor requiring GB10 token-exact golden

Rationale: those changes touch the gate model forward on all backends. ROCm kernels
and env-gated MoE paths do not. Revisit as a **separate** PR with CUDA golden.

`VT_GEMMA4_MLP_MOE_PARALLEL` is documented but **not** wired in this tip (was only
in the deferred layer-loop path).

## Measured KEEP (2026-08-13, contributor lab)

Hardware: 2x R9700 gfx1201, ROCm 7.2.4, kernel 7.0.0-29-generic.
Model: Gemma-4-26B-A4B-it FP8. Fair protocol: `PREFIX_CACHE=0` + unique pads.

Recipe as run: SharedK-WMMA on, FLASH/FMHA/scoreless off,
`VT_GEMMA4_PREFILL_GEMM_M=2048`, `VT_GEMMA4_PREFILL_PEER_ACT=1`, batch MoE
`T>=64`, decode `KV_SPLITS=16` / `SLIDE_SPLITS=8` / `SLIDE_WARPS=16` /
`SPLIT_WARPS=12`. Speculative, ngram, layer-split: **off**.

**Four of those names are not read by any product code in this tree, so this
recipe is not reproducible as written.** `VT_ATTN_DECODE_KV_SPLITS`,
`VT_ATTN_DECODE_SLIDE_SPLITS` and `VT_ATTN_DECODE_SPLIT_WARPS` appear only in
`tests/vt/test_gemma4_rocm_fp8_seams.cpp`, and `VT_ATTN_DECODE_SLIDE_WARPS`
appears nowhere at all; `git grep` over `src/` and `include/` returns zero hits
for all four. That is [#845](https://github.com/mudler/vllm.cpp/issues/845): the
seam test asserts `EnvInt(name, 16) == 16` with the variable unset, a tautology
that passes whether or not the knob exists, which is how three names that
nothing reads came to look real. The first two knob names above were also
abbreviated and are spelled here as the product spells them
(`PEER_ACT` is `VT_GEMMA4_PREFILL_PEER_ACT`, `gemma4_moe.cpp:1026`;
`PREFILL_GEMM_M` is `VT_GEMMA4_PREFILL_GEMM_M`, `:1016`, and `2048` is its
default, not an override).

This file's `**Status:**` line records the run as `PR tip
feat/gemma4-rocm-fp8-split` with **no commit SHA**, so which tree produced these numbers cannot be
established from this repository. Either the four decode splits were live on that
tree and it is not this one, or they were inert and the numbers were produced by
the shipped defaults. Owed to the contributor: the tree SHA, and which of the two
it was. Until then the decode figure in particular has no recipe behind it, and
`docs/USAGE.md` publishes only the knobs a reader can actually set.

| Depth | median prefill t/s | notes |
|------:|-------------------:|-------|
| ~3k | 2112 | 3 reps, Paris |
| ~11k | 2014 | first rep 1170 outlier dropped from claim; median 2014 |
| ~18k | 1705 | 3 reps |
| ~42k | 1099 | 2 reps |

Decode stream 64 tok: **55.5 t/s** temp=0, **49.1 t/s** temp=0.7.
Quality: Paris, arith `63`, `gemma4` parser `tool_calls`, `/health` `/metrics`.
No engine-fatal / hipError in the closeout log.

Same-box Vulkan Q8 unique-pad bar: **3503 @11k / 2714 @42k**. KEEP gap ~1.74x / 2.47x.
This row ships the reliable ROCm plateau. It does **not** close that bar.

### Rejected (do not default-on)

FMHA_WMMA2 (quality fail + 0.67x @11k). Isolated P1 cm1 wg256 (~1.13x KEEP; need ~3.35x isolated). Layer-split FIFO (~0.60x vs the Q8 bar). Head-TP peer-read (peer BW). hipBLASLt dual-GPU (fatal or slower).

### Residual compiler / ACO

Faithful HIP cm1 hsaco on gfx1201: 192 VGPR, 339 spills, 940 B private, vs KEEP SharedK 35 spills / 120 B. Output-slicing bottoms out at 235 spills / 736 B (D_PT=1). Mechanism of the isolated 1.13x, not by itself a named LLVM defect.

Matched RADV/ACO (Mesa 26.0.3, discrete R9700, llama.cpp coopmat1 spec Br=16 Bc=64 wg=128 sg=32 row_split=4): official pipeline stats are **0 spilled VGPR, 0 scratch** at VGPR=256 (f16 and q8_0 d=512). That is an (a)-leaning HIP-LLVM vs ACO disparity on equivalent ownership. Smallest LLVM/rocWMMA component is not named. Phase-2 rebuild stays blocked until that component and a measurable prediction exist. A general codegen fix, if one exists, is a separate claim from the 3.35x bar.

## Owed

- **A vLLM-ROCm denominator for every number in "Measured KEEP".** They are
  engine-side figures with nothing on the other side, so under AGENTS.md "Gates"
  they are not a throughput result at all, and `docs/BENCHMARKS.md` keeps this
  backend at `PENDING: no binding throughput number`. The oracle is not
  hypothetical: `docs/ROCM.md` §5 documents two working Docker vLLM-ROCm recipes
  on this hardware family, tier 2 building this project's pinned commit
  `555967922` inside `rocm/vllm-dev:base` in about 6.5 minutes. Owed: the same
  model, quantization, prompt set, request shape, concurrency and cache policy on
  both sides of one idle 2x R9700 box, and the ratio per depth. Only the
  contributor has that hardware.
- **The tree SHA behind the 2026-08-13 run**, and whether the four decode splits
  named in the recipe were live on it. See the note under "Measured KEEP" and
  [#845](https://github.com/mudler/vllm.cpp/issues/845).
- **`.agents/benchmark-record.md` is deliberately not appended here.** It is the
  append-only measurement log, and a number whose tree is unidentified and whose
  denominator is missing must not enter it: once logged, a figure gets quoted as
  measured. It goes in when the two items above are discharged.
- **The Vulkan Q8 comparison stays open, not closed.** Same box, ~1.74x at 11k
  and ~2.47x at 42k in Vulkan's favour. AGENTS.md forbids reading that as a
  ceiling. The next traceable hypothesis is the one this file already names: the
  HIP-LLVM versus ACO register-allocation disparity, 339 spills against ACO's 0,
  and it needs the smallest LLVM/rocWMMA component named before a phase-2
  rebuild.
