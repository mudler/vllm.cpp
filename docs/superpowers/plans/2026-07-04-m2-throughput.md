# M2 — Throughput parity vs vLLM (gate #1), profile-prioritized

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. This is GPU-kernel work — build + validate on GB10 via `ssh dgx.casa` (non-interactive SSH works; sync `~/work/vllm.cpp`, `export PATH=/usr/local/cuda/bin:$PATH`, build `-DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`).

**Goal:** Close gate #1 — throughput parity vs vLLM for Qwen3.6-35B-A3B-NVFP4 on GB10 at
large concurrency (prefill + decode). **Latest free-box measurement (2026-07-04, GB10, 0%
contention, 8×1024×128 enforce-eager, AFTER M2.7): vLLM 0.24.0 = 1181.78 total / 131.31 output
tok/s; ours = 157.27 total / 17.47 output → ~7.5× slower** (down from ~1000× at M2 start, and
from ~16.8× before M2.7). The M2.1 harness (`examples/bench` → `vllm-bench`) measures our side;
run `vllm bench throughput` on `~/venvs/vllm-oracle` (`ninja` on PATH) for the baseline. Record
numbers here as each kernel lands. STANDING DIRECTIVE: always compare vs vLLM apples-to-apples,
and STATE THE CONFIG each time — ~7.5× is vs enforce-eager (1181/131); the gap vs full-graph
vLLM (1377/153) is larger (AGENTS.md, gates.md §PROTOCOL DIRECTIVE).

**THE PROFILE — v3 (2026-07-04, nsys free-box, load-bearing; SUPERSEDES the v1 "GPU-idle,
CPU-dequant-bound" and v2 "88% host-API launch-overhead" theories — BOTH now fixed).** After
the device-resident-weights work (M2.2b) and the async-on-stream device-resident forward
(Phase 1, `f95b131`), **both prefill and decode are GPU-COMPUTE-bound.** CUDA-graph capture
(`2999431`) confirmed it: graphs bought only +2.6% single-stream / −7% at batch-8 because host
launch overhead is already hidden behind the GPU. The remaining ~16× is raw fp4-kernel speed:
- **PREFILL (14.7s single-prompt TTFT; the batched-TTFT ~100s is the #1 throughput killer):**
  GPU-bound on the fp4 path. **fp4 MoE grouped GEMM = 70.7%** (`MoeGroupedGemmNvfp4Tiled`
  gate/up 47% + down 23%), GdnScan 19%, PagedAttn 5%, fp4 projections 4%. The fp4 GEMMs are
  hand-tiled **CUDA-core** at ~20 TFLOP/s (tensor-core `nvjet` ops are <0.2%) — far below
  GB10 fp4 tensor-core peak. → **tensor-core NVFP4 MMA is THE prefill unlock (~75% of prefill).**
- **DECODE (65ms TPOT, M=1, weight-bandwidth-bound):** naive fp4 GEMV ~55%
  (`MatmulNvfp4KernelNaive` 28% + `MoeGroupedGemmNvfp4KernelNaive` 22% + down 5%), cublas bf16
  `gemvx` 14% (should be on the fp4 path), **`GreedyArgmaxKernel` 11.8% = 7.5ms/token, a
  single-block vocab reduction ~100× improvable**, GdnScan 5%, PagedAttn 2%. MMA does NOT help
  M=1 → decode needs a separate **fp4-GEMV + fast-argmax** pass, not the prefill MMA.

## Priority order (highest-leverage first, per the v3 GPU-compute-bound profile)

### PROFILE MOVED after M2.7 (2026-07-04, `bc0a8d7`) — re-profile the prefill
M2.7 landed and the fp4 MoE GEMM dropped **70.7% → 11.6%** of prefill. The prefill bottleneck
is now **`GdnScanKernel` = 63.8%** (the gated-delta-net recurrence). So the current top prefill
lever is **M2.3 (GDN chunked scan)**, NOT the MoE GEMM. Post-M2.7 prefill shares: GdnScan 63.8%,
fp4 MoE GEMM 11.6% (wmma), PagedAttn ~5%, fp4 projections ~4%.

### M2.3 — GDN chunk-parallel prefill scan (THE prefill lever now — IN PROGRESS 2026-07-04)
`GdnScanKernel` (`cuda_gdn.cu:350`) is a SEQUENTIAL recurrence used for both prefill (scans all
prompt tokens one-at-a-time per layer → 63.8% of prefill) and decode (1-step, fine). Replace the
PREFILL path with a **chunk-parallel scan** mirroring vLLM's FLA `model_executor/layers/fla/ops/
chunk.py` (+ chunk_scaled_dot_kkt / solve_tril / wy_fast / chunk_delta_h / chunk_o / cumsum):
parallelize within-chunk via matmuls, pass the `[Dk×Dv]` state sequentially across chunks. Keep
the decode 1-step recurrence untouched. Reference: `.agents/gdn-semantics.md`. Correctness:
chunked == sequential (synthetic, ≥2 chunks + tail) → paged gate 16/16 → measure prefill TTFT.

### M2.7 — tensor-core NVFP4 MMA GEMM (THE earlier prefill unlock — DONE `bc0a8d7`)
Moved the fp4 W4A16 GEMMs (MoE grouped + dense projections) off hand-tiled CUDA cores onto
tensor cores (`nvcuda::wmma` bf16×bf16→f32, dequant-into-shared) + device-side expert-grouping
(counting sort → dense per-expert GEMM). **RESULT: prefill TTFT 14.4→6.1s (2.36×), 8×1024×128
total 70→157 tok/s (2.24×), gap 16.8×→7.5×; paged gate 16/16.** Tuning headroom remains (kernel
~3-4 TFLOP/s, below bf16 peak — larger tiles / cp.async / drop ragged-BM waste). Native fp4-MMA
stretch not attempted (low leverage now GDN dominates).

### M2.8 — decode fp4-GEMV + fast-argmax (the decode gap, separate from MMA)
(a) An optimized M=1 fp4 GEMV replacing the `...Naive` kernels (~55% of decode); (b) route the
14% cublas bf16 `gemvx` (out_proj/router) onto the fp4 path; (c) a multi-block/warp-reduce
`GreedyArgmaxKernel` (11.8% of decode, ~100× headroom — cheap, high-ROI).

### (historical) Priority order per the earlier GPU-idle / host-overhead profiles — DONE

### M2.2a — NVFP4 W4A16 dequant-GEMM kernel (THE unblocker)
The dominant cost. Implement a CUDA GEMM that takes NVFP4 weights **directly in device
memory** (the U8 fp4 packed + fp8-e4m3 group-16 scale + weight_scale_2 amax/2688 — the
modelopt W4A16 layout already in `DequantNvfp4ToBf16`) × bf16 activations → bf16 output,
dequantizing on-the-fly in the kernel (no host bf16 tensor). Correctness-grade first
(a straightforward per-tile dequant-then-mma or dequant-into-shared), then tune. Reference:
mudler's killgate NVFP4 prior art (`~/killgate_series/` on dgx, `.agents/environment.md`
§ prior art + `.agents/gguf-nvfp4-notes.md` § M2.2 — the UE4M3 LUT ×0.5 trap, QR/QI packing).
Unit-test on GB10 vs the existing bf16 matmul (small synthetic NVFP4 tensor → dequant-GEMM
== bf16-matmul(dequant(w)) within tolerance). This is the compute primitive.

### M2.2b — keep NVFP4 weights fp4 in device memory (the storage refactor)
Wire M2.2a: the model weights stay NVFP4 (~22GB) **resident on the GPU** instead of
dequanting to 70GB bf16 host tensors. Kills the 40-min CPU dequant, the 70GB host
tensors, AND the per-op weight upload (the weight lives on-device, the GEMM reads it in
place). This + M2.2a is expected to be the bulk of the speedup (GPU stops being idle).
Requires a device-resident weight representation + the forward calling the NVFP4 GEMM.
Validate: the 35B paged greedy gate still passes (correctness) + re-measure `vllm-bench`
(the load time + the decode tok/s should jump).

### M2.x — staged on-device KV/weight storage (the MRV2 axis, deferred at M1.5/M1.8)
Eliminate the remaining per-op host↔device copies (activations, KV cache) — keep the
step's tensors on-device across the forward. The `vt` runtime's arena + the paged KV
cache move to device-resident. Re-measure.

### M2.5 — CUDA graphs (decode capture/replay)
GB10 is launch-overhead + bandwidth bound; decode is many tiny kernels. Capture the decode
step into a CUDA graph over persistent buffers, replay per token. Big decode-throughput win.

### M2.3 / M2.4 — fused GDN + fused MoE + paged-attn tuning
Fused post-conv/recurrence GDN kernels (M2.3), fused MoE (256e top-8) token-tile selection
(M2.4-ish, killgate patch 0015 prior art), FlashInfer-class paged attention for GQA 16/2
on sm_121 (M2.4). Profile-driven after M2.2/graphs.

### M2.6 — fusions + async scheduling; **gate #1 measurement → ledger**
qk-norm+rope+gate, rmsnorm+residual fusions; async scheduling if CPU-bound. DoD: the
parity table (ours vs vLLM at large concurrency, prefill + decode) committed to the ledger.
**← gate #1 met.**

## Global Constraints
- Every kernel: correctness FIRST (unit-test on GB10 vs the bf16 reference, small synthetic
  case — fast, no 35B load needed), THEN measure the speedup with `vllm-bench`. Keep the
  35B paged greedy gate (test_qwen36_paged_engine) passing after each change.
- 1:1 where a vLLM/csrc kernel maps (backends.md §drop-in — CUDA/AMD csrc alignment); the
  NVFP4/GDN/MoE fused kernels are original (§9) until the csrc drop-in.
- Build + validate on GB10 (ssh dgx.casa). Record each measured number in the ledger.

## Self-review notes
- The order is PROFILE-DRIVEN: GPU-idle ⇒ M2.2 (on-device fp4 weights, no CPU dequant/upload)
  is THE unblocker; do it first, re-measure, then attack the next bottleneck the profile shows.
- Don't optimize blind — measure each kernel on GB10 against the vLLM baseline (1377 tok/s).
- Correctness gate (test_qwen36_paged_engine) must stay green through every kernel change.
