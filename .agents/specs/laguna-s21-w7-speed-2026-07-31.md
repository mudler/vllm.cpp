# Laguna-S-2.1 (`LagunaForCausalLM` / `laguna`) — W7: decode-speed ATTRIBUTION + ranked lever plan (profile-only)

**Row:** `MODEL-TEXT-laguna-laguna-for-causal-lm` — stays **RUNNABLE / ACTIVE** (speed research; NO code change).
**Claim context:** `CLAIM-LAGUNA-W7-SPEED`. **Date:** 2026-07-31. **Base:** W6 `08eec2a8` (HEAD `ace8c210`).
**HW:** dgx.casa GB10 (sm_121a), 119 GiB unified. **Branch:** `laguna-s21-w7-speed-profile` (NOT pushed).
**Method:** `nsys profile --trace=cuda` of `examples/laguna-gen --gpu` on the REAL 3-shard
`unsloth/Laguna-S-2.1-GGUF UD-Q4_K_XL` (69 GiB keep-quant), prompt "The capital of France is",
`--max-tokens 8` (1 prefill T=6 + 7 steady-state decode T=1). Worker down, `flock $HOME/gpu.lock`,
one model resident (peak 71.6 GiB), build tree cleaned after. This is a READ/PROFILE-ONLY brick —
the levers are the follow-on implementation.

## 0. The gap being attributed

W6 decode = **0.66 s/tok cold (~1.5 tok/s)**; under this nsys warm run **0.53 s/tok (~1.9 tok/s)**.
llama.cpp Poolside-fork on the IDENTICAL UD-Q4_K GGUF = **27.8 tok/s** (36 ms/tok). Gap ≈ **15×
(warm) – 18× (cold)**. This mirrors the DeepSeek-V4 speed campaign start (host-orchestrated
keep-quant forward, `.agents/specs/deepseek-v4-device-decode.md` / `deepseek-v4-last-mile.md`), and
the same playbook applies.

> **SUPERSEDED DENOMINATOR (2026-08-16, #1003).** The `27.8 tok/s` above, and
> every ratio in this spec derived from it (the §2 roofline reference row, the
> §4 reachable verdict, the W8 and W9 SPEED lines, and the GEMV-ceiling lever in
> §W11), came from a **Poolside fork** of llama.cpp on branch `laguna`
> (`github.com/poolsideai/llama.cpp@laguna`, named at
> `laguna-s21-w4-2026-07-31.md:65`). **No commit SHA for it is recorded anywhere
> in this tree.** A branch is a moving reference, not a revision, so this
> denominator cannot be reproduced and never could be. It is neither the former
> pin `237ad9b96` nor the current stock pin `b10451`. Our own side of every
> comparison here is untouched, and the same-binary W8 and W9 A/B ratios
> (`3.9×`, `1.38×`, `5.1×`) are ours-versus-ours and stand unchanged. What is
> owed is the `15x`, `18x`, `4.7x` and `3.6x` framing against llama.cpp.
> Enumerated as **row 13** in
> [`oracle-llamacpp-repin-stock.md`](oracle-llamacpp-repin-stock.md), re-take
> owed under [#1003](https://github.com/mudler/vllm.cpp/issues/1003). Do not
> quote `27.8` without re-measuring the reference arm against a named commit.
>
> Sections rather than line numbers. The first version of this list cited five
> line numbers in this file, and inserting the mark that carried them pushed
> every one of those lines down by the mark's own length. A citation into the
> file it is written in is stale before it is saved, so it should not be a line
> number.

## 1. MEASURED attribution (nsys, 8-step forward, 5.14 s wall = 1.41 prefill + 3.73 decode)

**Where the time is (GPU vs host):**

| bucket | measured | note |
|---|---|---|
| GPU kernel total | **1.682 s = 32.7% of wall** | the GPU is IDLE ~67% of every step |
| **host / idle (not overlapped)** | **~3.46 s = 67.3% of wall** | glue + per-op serialization gaps |
| `cudaStreamSynchronize` | 1.729 s wall, **22,115 calls** (~2,764/step) | ≈ kernel wall ⇒ host FULLY blocked, ZERO overlap |
| `cudaLaunchKernel` | 0.108 s = **2.1%** | launch overhead is NOT the bottleneck |
| H2D / D2H copies | **none** (nsys has no GPU-mem data) | unified memory — GPU reads host `std::vector` in place; NOT copy-bound |

**Root cause = the host-orchestrated composition.** `LagunaForwardGgufCached` (`laguna.cpp`) issues
**~1,795 keep-quant GEMMs per token**, and every one calls `DrainQueue` (a `cudaStreamSynchronize`)
right after (`LqGemm`/`LqGemmRowSlice`, `laguna.cpp:193,229` — NO `defer_sync`, unlike ds4's
`TimedMatmul(defer_sync)`/`DrainDevice`). All the glue (RMSNorm, dual-RoPE, GQA attention softmax,
per-head softplus out-gate, router `MatmulNK`, top-k, SwiGLU, combine, embed gather) runs SCALAR on
the HOST in `std::vector` loops between the synced GEMMs. So the step is a serial chain of
{launch tiny GEMV → block in sync → host glue → repeat} with the GPU idle in every gap.

Per-token GEMM count (all synced, all M=1 GEMVs): attn 5/layer × 48 = 240; MoE experts
**30/layer × 47 = 1,410 UN-GROUPED** (`LqGemmRowSlice` per selected expert, gate/up/down × 10) +
shared 3/layer × 47 = 141; dense-L0 3; lm_head 1 → **~1,795**.

**Where the GPU 32.7% goes** (per-kernel, whole run):

| kernel | % of GPU time | instances | what it is |
|---|---|---|---|
| `QuantizeQ8KKernel` | **39.4%** | 18,330 | **ACTIVATION quantizer** — re-quantizes the hidden row to Q8_K, once PER GEMM |
| `QuantDotGemmKernel<Q4_K>` | 25.6% | 11,960 | routed expert gate/up GEMV (un-grouped) |
| `QuantDotGemmKernel<Q5_K>` | 16.7% | 6,110 | routed expert down GEMV (un-grouped) |
| `QuantDotGemmQ8_0Kernel` | 16.7% | 3,785 | attn q/k/v/o/g + shared + dense + lm_head |
| `QuantDotGemmKernel<Q6_K>` + `QuantizeQ8_0Preq` | ~1.6% | | tail |

So **40% of the GPU time is activation-quant glue**, not the weight matmul — the keep-quant path
re-quantizes the SAME hidden row separately for each of the ~1,795 GEMMs (~2,291 `QuantizeQ8K`/step).
The 60% that IS weight-GEMM runs at ~126 ms/step = **~52 GB/s = 22% of the 240 GB/s peak** (un-grouped,
one-warp-per-output tiny-M GEMVs).

## 2. The roofline (why 27.8 is the reference, and what is reachable)

Active keep-quant weight bytes read per decoded token (each weight read once, M=1):
attn Q8_0 **2.98 GB** + routed experts Q4_K/Q5_K (10 of 256) **2.68 GB** + shared Q8_0 0.47 GB +
dense-L0 Q8_0 0.12 GB + lm_head Q8_0 0.33 GB = **~6.58 GB/token**.

| | tok/s | ms/tok | effective BW | % of 240 GB/s peak |
|---|---|---|---|---|
| **hard roofline** (6.58 GB @ 240 GB/s, dequant+compute free) | **36.5** | 27.4 | 240 | 100% |
| **llama.cpp (the reference)** | **27.8** | 36.0 | 183 | **76%** |
| ours (nsys warm) | 1.9 | 530 | 12.4 | **5.2%** |
| ours (W6 cold) | 1.5 | 660 | 10.0 | **4.2%** |

We read the SAME 6.58 GB as llama.cpp but at ~1/15 the memory efficiency, because the GPU sits idle
between synced tiny GEMVs. **The 15–18× is ~⅔ host-orchestration + ~⅓ GPU kernel inefficiency**
(activation-quant + un-grouped GEMV), NOT kernel compute — a T=1 decode matvec is memory-bound and
tensor cores do not help it (same finding as ds4 `deepseek-v4-last-mile.md §1`).

## 3. Ranked lever plan (grounded in the profile + the ds4 device-decode playbook)

The whole ds4 apparatus these need is ALREADY in-tree (`deepseek_v4.cpp`
`ForwardResidentDecodeGguf`, `defer_sync`/`DrainDevice`, `vt::MatmulBTQuantGrouped`,
`V4Graph::Step` capture, the dp4a-tuned `QuantDotGemmGrouped` kernels).

| # | lever | where the time is now (measured) | expected | risk | effort |
|---|---|---|---|---|---|
| **1** | **Device-resident decode** (`ForwardResidentDecodeGguf` analog): issue every GEMM `defer_sync`, ONE `DrainDevice` at the step boundary, replace host RMSNorm/RoPE/attention/softplus-gate/router/SwiGLU/combine with in-place device kernels on the unified buffers (most exist from ds4/qwen3_5). | host = **67% of step**; 22,115 syncs; GPU idle 2/3 of the time | **1.5 → ~5–7 tok/s (3–4×)** | med (near-tie where device reductions reorder; large surface — stage A/B/C/D like ds4) | large |
| **2** | **Grouped expert GEMM** (`vt::MatmulBTQuantGrouped` + fused gate+up+SwiGLU): replace the 30 per-expert `LqGemmRowSlice` launches/layer with ~2–3 grouped launches over the 10 selected experts; quantize the shared activation ONCE (broadcast). | Q4_K 25.6% + Q5_K 16.7% GPU as **1,410 un-grouped GEMVs/step**, each driving a separate `QuantizeQ8K` | **+1.5–2× on top of L1**; also dedupes most of the 40% `QuantizeQ8K`. Bit-exact by construction (ds4 Brick 2/6). Independent of L1 (helps even the host path). | low–med | med |
| **3** | **Kill the activation-quant overhead** (`QuantizeQ8K` = **39.4% of GPU**): quantize each site's shared input ONCE, not per projection. NOTE ds4 Brick 8 REFUTED fusing quant INTO the GEMM prologue (per-block re-quant = −22%) — the win is DEDUPE + fewer standalone launches, largely SUBSUMED by L1+L2. | 18,330 `QuantizeQ8K` launches (~2,291/step) | realized via L1+L2; standalone ~+0.3–0.5× | low | low–med |
| **4** | **Capture the decode CUDA graph** (after L1's sync-free step; `V4Graph::Step`/`Qwen3_5DenseDecodeGraph` analog): collapse all launches into one `cudaGraphLaunch`. | launch overhead only 2.1% today, but the graph guarantees zero host gaps between kernels | +10–20% on top of L1 | med (capture hazards — `[[cudagraph-capture-bakes-stack-addresses]]`) | med |
| **5** | **GEMM BW-efficiency (tuned MMVQ)**: ensure the Q4_K/Q5_K grouped experts use the ds4 Brick-1/1b dp4a vectorized-dequant kernels and Q8_0 the Brick-3/9 preq path; push 22% → ds4-class ~58% of peak. | weight GEMMs at **22% of peak** vs llama.cpp 76% | final push ~12 → ~15–20 tok/s | med (near-tie where fp reassociates) | med–large |
| **6** | **Host per-token waste (cheap, do FIRST, bit-exact):** (a) `LagunaEmbed` calls `ReadF32(embed)` which COPIES the ENTIRE [100352×3072] f32 embed table (**1.23 GB/token**) to gather ONE row — gather the row directly; (b) `BuildLaguna{FullYarn,Sliding}CosSin` rebuild the full cos/sin tables (O(ctx)) every token — build once; (c) `ReadF32(router)` re-copies the 3 MB router f32 every layer every token — cache. | pure host time inside the 67% bucket; the embed full-copy alone ~1.23 GB memcpy/token | small but trivial + safe; the free win | negligible | tiny |
| **7** | **fp8 KV + sliding-window KV**: parity/long-context/footprint, NOT the 24-tok lever (at ctx≤512 eviction is a no-op and attention host time is tiny). Mirror ds4 Brick 5; measure at 256+ ctx. | ~0% of the short-ctx step | scaling/footprint only | — | defer |

**Sequence:** L6 (free host cleanups) → L2 (grouped experts — big, independent, bit-exact) → L1
(device-resident decode — the dominant win) → L4 (capture graph) → L5 (tuned MMVQ BW). L3 falls out
of L1+L2; L7 is a deferred long-context/parity lever.

## 4. Honest reachable verdict (vs 27.8)

- **Roofline ceiling** = 36.5 tok/s (out of reach on this quant mix).
- **llama.cpp reference** = 27.8 (76% of peak, hand-fused CUDA-graph MMVQ).
- **Realistic target ~13–20 tok/s (8–13×)**: L6+L2 alone (still host-orchestrated, far fewer
  launches/syncs + dedup quant) ≈ 3–4 tok/s; +L1 residency ≈ 6–9; +L4 graph ≈ 8–11; +L5 kernel BW
  ≈ 13–20. Matching ds4's realized ~58%-of-peak fully-fused efficiency would put Laguna ≈ 21 tok/s.
- **27.8 is the STRETCH** and likely just short — this is the SAME shape as ds4's own outcome
  (host-orchestrated → ~12.7 tok/s = ~77% of its 16.5 reference, full parity NOT reached; the
  irreducible residual is the K-quant GEMV BW gap vs the hand-tuned reference).
- The dominant, highest-confidence win is **L1 device-residency + L2 grouped experts** — together
  they attack the measured 67% host + the 40% activation-quant + the un-grouped-GEMV inefficiency in
  one coherent rewrite, and every piece is already in-tree from the ds4 campaign.

## 5. Reproduce

```
# DGX GB10, worker down, flock $HOME/gpu.lock, TMPDIR on a writable disk
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o nsys-laguna \
  build/examples/laguna-gen --model .../Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf --gpu --max-tokens 8
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum nsys-laguna.nsys-rep
```
Token stream matched the W6 golden (`22345 83 350 785 989 395 13259 330`), decode 0.49–0.57 s/tok
steady. nsys report retained on dgx at `~/_git/laguna-w7-profile/nsys-laguna.nsys-rep`.

## Decision

Row stays **RUNNABLE / ACTIVE**. This is a profile-only attribution: the 0.66 s/tok is **~67%
host-orchestration / per-GEMM sync + ~40%-of-GPU activation-quant glue + un-grouped tiny-M expert
GEMVs at 22% of BW peak** — NOT kernel compute. The ranked levers (device-resident decode + grouped
expert GEMM first, then decode-graph + tuned MMVQ) are the DeepSeek-V4 playbook, all in-tree. Honest
reachable ~13–20 tok/s, 27.8 a stretch. Numbers are measured (nsys) or explicitly computed (roofline);
hypotheses are labelled. No code changed.

---

## W8 — lever #5 (embed gather) LANDED + GATED (2026-07-31, `CLAIM-LAGUNA-W8-EMBED`)

The W7 profile filed `LagunaEmbed`'s full-table copy under "#5 free host cleanups", but
it was the **dominant** decode cost, not minor: `LagunaEmbed` called `ReadF32(embed_t)`
which converted the ENTIRE `[Vsz,H]` f32/bf16 embed table (~311M element-converts,
~1.23 GB) on the HOST **every token** just to gather T rows. Fixed: gather only the T
needed rows directly from the table bytes (bit-identical — same per-element f32/bf16→f32
conversion, same rows; the untouched rows never affected the output).

**GATED on the real 3-shard UD-Q4_K_XL GGUF (GB10 sm_121a, `--gpu`, W6 cached path,
drop_caches cold, prompt "The capital of France is", 24 tokens):**
- **TOKEN-IDENTICAL PASS** — `generated ids` byte-equal to the W5/W6 golden
  (`22345 83 350 785 989 395 13259 330 4159 9431 377 340 4328 377 444 136 22029 9626 71 493 6396 565 7760 10291`),
  coherent " Paris." continuation. Bit-exact confirmed on the real model.
- **SPEED: decode 0.66 → 0.17 s/tok = 3.9×** (Laguna **1.5 → 5.9 tok/s**; 18× → 4.7× vs
  llama.cpp 27.8). Prefill 0.98s, peak 69.96 GiB. Single bit-exact host fix.

The measured 0.49 s/tok saved per token ≈ the host cost of the eliminated 311M-element
embed conversion — consistent with the host-orchestration attribution. Remaining levers
unchanged (grouped-expert GEMM = A3, then device-resident decode); the per-token
RoPE-cos/sin rebuild + norm/router `ReadF32` are the smaller residual #5 items.

---

## W9 — lever #2 (grouped-expert GEMM) LANDED + GATED (2026-07-31, `CLAIM-LAGUNA-W9-GROUPED`)

Laguna's 30 un-grouped per-expert GEMV launches/step (top_k experts × {gate,up,down},
each a separate `LqGemmRowSlice` + `DrainQueue`) fold onto the SHARED keep-quant grouped
op `vt::MatmulBTQuantGrouped`: per token, the Pk selected experts' gate/up/down each
collapse to ONE grouped launch over the already-stacked `[E*N,H]` expert tower (NO loader
change — Laguna's experts are stored stacked, sliced by `id*moe_I`). New `LqGemmGrouped`
thin wrapper (mirror of ds4 `GemmGroupedExpertsKq`) + `LagunaGroupedMoeEnabled()` gate;
rows built in SLOT ORDER so `GateUpSilu` (Laguna's own) + the `acc` combine run in the
identical order as the per-expert fallback ⇒ bit-identical by construction.

**GATED — same-binary A/B on the real UD-Q4_K_XL GGUF (GB10 sm_121a, `--gpu`, W6 cached,
drop_caches cold, 24 tok):**
- **BYTE-IDENTICAL** — grouped (`VT_LAGUNA_GROUPED_MOE=1`, default) and per-expert (`=0`)
  `generated ids` are byte-equal (md5 `754728c6…`) and BOTH == the W6/W5 golden (a wrong
  row-order/combine would diverge). Bit-exact confirmed on the real model, both ways.
- **SPEED: decode 0.18 → 0.13 s/tok = 1.38×** (same-binary grouped-vs-per-expert). Prefill
  0.99 → 0.70s. Cumulative with W8 embed: **0.66 → 0.13 s/tok = 5.1× (1.5 → 7.7 tok/s);
  18× → 3.6× vs llama.cpp 27.8**.

This routes through the SHARED `vt::MatmulBTQuantGrouped` (fold policy: no per-model kernel)
and proves the grouped-consumption pattern qwen3_5's A3 reuses (after its loader-stacking —
qwen3_5 stores experts per-`OwnedTensor`, not stacked). Remaining Laguna lever: #1
device-resident decode (the 22k-syncs kill) toward the ~13–20 tok/s ceiling.

---

## W10 — RE-PROFILE after W8+W9 (2026-07-31, measurement-only, `CLAIM-LAGUNA-W10-REPROFILE`)

The W7 attribution profiled the pre-W8/W9 0.66 s/tok path; re-profiled current main
(`a402eb6b`, W8+W9) to re-rank levers with real data. nsys `--trace=cuda`, real
UD-Q4_K_XL GGUF (GB10, `--gpu`, W6 cached, drop_caches cold, 24 tok, TPOT 0.13 s/tok):

| CUDA API | Time% | calls | reading |
|---|---|---|---|
| **`cudaStreamSynchronize`** | **89.7%** (2.59 s) | **14,034 (~610/step)** | the per-GEMM `DrainQueue` — STILL the dominant cost |
| `cudaStreamCreate` | 7.4% | 1 | one-time queue create (213 ms, amortized, not per-step) |
| `cudaLaunchKernel` | 2.6% | 28,068 (~1,220/step) | launch overhead is NOT the bottleneck |

**VERDICT: lever #1 (device-resident decode) CONFIRMED as the top remaining lever.**
Syncs dropped 22,115 → 14,034 (W9's grouped-MoE effect) but decode is STILL 89.7%
host-sync-bound: every `LqGemm`/`LqGemmGrouped` drains the stream so the host reads the
output for the next f32 glue op (RmsNorm, RoPE, attention, SwiGLU, router, softplus
out-gate — all host today). The fix (mirror ds4 `ForwardResidentDecodeGguf` + decode
CUDA-graph): keep intermediates ON-DEVICE and run the glue on-device so the whole
step defers to ONE `DrainDevice` (610 → ~1 sync/step). This is a MULTI-BRICK campaign
(port each Laguna glue op to a device kernel + chain them), NOT a bit-exact host fix
like W8/W9 — governed by the CUDA-graph-capture-safety hazards (see
[[cudagraph-capture-bakes-stack-addresses]]). GPU-kernel breakdown (post-residency
residual: `QuantizeQ8K` dedup, weight-GEMV BW) owed on the next profile. Reachable
still ~13–20 tok/s; killing ~600 syncs/step is the path there.

---

## W11 — GO/NO-GO on device-residency: decode is now GPU-COMPUTE-BOUND, not host-bound (2026-07-31, `CLAIM-LAGUNA-W11-GONOGO`)

Before committing the LARGE device-resident campaign (W10's inferred lever #1), measured
GPU-busy time (`nsys cuda_gpu_kern_sum`) to check the recoverable headroom. Real
UD-Q4_K_XL (GB10, `--gpu`, 24 tok, TPOT 0.13 s/tok):

| GPU kernel | Time% | Total | reading |
|---|---|---|---|
| `QuantDotGemmGroupedKernel` Q4_K/Q5_K/Q6_K | **62.1%** | 1.59 s | MoE expert GEMVs (W9-grouped; must read all top_k experts) |
| `QuantDotGemmQ8_0Kernel` | 24.7% | 0.63 s | attention q/k/v/o + shared + dense + lm_head GEMVs |
| `QuantizeQ8KKernel` | 12.4% | 0.32 s | activation re-quant (per GEMM) |
| **Σ GPU-busy** | | **2.56 s** | ≈ `cudaStreamSynchronize` 2.59 s ≈ decode wall |

**VERDICT: device-residency (task #228) DEMOTED — decode is GPU-compute-bound.** GPU-busy
(2.56 s) ≈ the host sync time (2.59 s): the 89.7% "host-sync" from W10 is the host
SERIALLY WAITING on real GPU kernels, NOT idle host-gap. After W8 (embed) + W9 (expert
launch collapse), what remains IS the keep-quant GEMV compute. Deferring syncs recovers
only the host-glue serialization (~0.8 s across prefill+decode) → maybe 0.13 → ~0.11
s/tok, modest — NOT worth a multi-brick CUDA-graph campaign. **This measurement reversed
the W7/W10 "host-orchestration" framing** (which was true at 0.66 s/tok, before W8/W9).

**RE-RANKED remaining levers (real data):**
1. **`QuantizeQ8K` dedup (~12% GPU, tractable):** W9 uses 2× `MatmulBTQuantGrouped`
   (gate,up) each re-quantizing the activation; switch to the FUSED
   `vt::MoeGateUpSwiGLUGrouped` (quantizes act ONCE for gate+up, ds4's op) → removes a
   chunk of the 12.4%. Bit-exactness vs Laguna's `GateUpSilu` must be A/B-verified
   (limit=+inf ⇒ plain silu·up; the ds4-gated composite identity should hold).
2. **keep-quant GEMV BW-tuning (~87% GPU, HARD — the ceiling lever):** the
   `QuantDotGemmGroupedKernel` (Q4_K/Q5_K, 62%) + `QuantDotGemmQ8_0Kernel` (25%) run at
   ~22% of the 240 GB/s peak (W7) vs llama.cpp ~76%. This is the ds4-Q8_0-saga class
   (7 levers largely refuted, compiler/SASS-limited) — but the GROUPED Q4_K/Q5_K kernel
   is DISTINCT from the ds4 Q8_0 kernel and may not have been BW-tuned; a dp4a /
   vectorized-load / occupancy pass on it is the real ~13–20 tok/s headroom.
3. device-resident decode (#1, task #228): DEMOTED to a modest ~0.02 s/tok recovery,
   deprioritized below the two GPU-kernel levers.
