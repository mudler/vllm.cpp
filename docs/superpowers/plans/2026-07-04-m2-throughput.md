# M2 — Throughput parity vs vLLM (gate #1), profile-prioritized

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. This is GPU-kernel work — build + validate on GB10 via `ssh dgx.casa` (non-interactive SSH works; sync `~/work/vllm.cpp`, `export PATH=/usr/local/cuda/bin:$PATH`, build `-DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`).

**Goal:** Close gate #1 — throughput parity vs vLLM for Qwen3.6-35B-A3B-NVFP4 on GB10 at
large concurrency (prefill + decode). **Measured baseline (2026-07-04, GB10, 8×256×32):
vLLM 0.24.0 = 1377 total tok/s / 153 output tok/s; ours (correctness-grade) = ~1000×
slower, GPU-idle, CPU-dequant/staging-bound.** The M2.1 harness (`examples/bench`
→ `vllm-bench`) measures our side; run `vllm bench throughput` on `~/venvs/vllm-oracle`
(needs `ninja` on PATH) for the baseline. Record numbers here as each kernel lands.

**THE PROFILE (measured, load-bearing):** the correctness-grade forward runs GPU at ~0%
with 82GB host RSS. The bottleneck is NOT GPU compute — it is (a) the CPU-side
NVFP4→bf16 dequant at LOAD (builds ~70GB of bf16 host tensors, ~40 min), and (b) per-op
host↔device staging (each op uploads bf16 weights from host, runs, discards). So the
priority order below is dictated by the profile, NOT the roadmap's original numbering.

## Priority order (highest-leverage first, per the GPU-idle profile)

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
