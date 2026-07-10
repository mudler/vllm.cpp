# vllm.cpp Post-MVP Expansion Map — Synthesis

## Bottom line (decisive order of attack)

The gate campaign already built the assets that make expansion cheap: an sm_80+ WMMA/FA-2/Marlin/cuBLASLt kernel set, a Triton-AOT pipeline that retargets by one flag, and a clean `vt::` op-table seam. The winning sequence exploits that, front-loads the near-free reach, runs the two hardware-independent tracks in parallel, and defers the from-scratch backends.

- **Wave 0 (prereq, ~1 wk):** close the two `backends.md:38` structural items — per-`Device` backend registry (currently per-`DeviceType`) and widen the `out=f32` op-validation — plus a CUDA device-capability probe. This unblocks *every* downstream track.
- **Wave 1 (~2 wk, huge reach, nearly free):** NVIDIA arch fan-out (sm_89/90 first, then sm_80/86). No new kernels; it also *forces the capability-fallback machinery into existence*, which every other GPU vendor reuses.
- **Wave 2 (parallel, different engineers, no contention):**
  - (a) Async/overlap scheduling — engine-level, benefits all backends, a mirror obligation.
  - (b) CPU threadpool → compute-in-quant GEMM — self-contained in `src/vt/cpu/`.
- **Wave 3 (~4–6 wk):** AMD MI300X (gfx942). The Triton-GDN one-flag lever makes the hardest kernel near-free on AMD.
- **Wave 4 (hardware-enabled):** Apple Metal via MLX. The M4/16 GB Mac mini at
  `ssh 192.168.68.103` supports op parity and small-model milestones now; a
  larger-memory Mac is still required for 27B/35B gate-model measurements.
- **Wave 5 (long poles, workload-driven):** Vulkan (broad cross-vendor bet), then EAGLE/MTP spec decode, Intel SYCL, grammar jump-forward.

## Ranked expansion map (reach × effort)

Effort: 🟢 days–2 wk · 🟡 4–6 wk · 🔴 8–16 wk.

| # | Target | User reach | Effort | First milestone | Key risk |
|---|---|---|---|---|---|
| **0** | **Structural prereq** — per-`Device` registry + widen `out=f32` + CUDA capability probe | (unblocks all) | 🟢 ~1 wk | Op-table selection keyed on `(OpId, Device)`; descending native-fp4→cuBLASLt-fp8→Marlin→bf16 chain wired | Touches every backend seam; do before parallelizing |
| **1** | **NVIDIA arch fan-out** (sm_89/90 → sm_80/86: RTX-4090, L40S, H100, A100, RTX-3090) | Very high — entire non-Blackwell NVIDIA base | 🟢 ~2 wk | Correctness gate green on one H100/4090 (fp8 via cuBLASLt, nvfp4→Marlin, native-fp4 gated off), then A100 bf16 | Per-arch Triton GDN autotune (GB10-pinned configs; hand-C++ fallback correct meanwhile); Hopper peak-fp8 needs a Sm90 cutlass config (+2–3 d) |
| **2** | **Async/overlap scheduling** (engine, cross-cutting) | All backends / all users (latency) | 🟡 medium | nsys the decode steady-state to size the prize, then port `async_scheduler.cpp` + `step_with_batch_queue` + event-based sampling | Token-exact stop/abort/preemption with in-flight steps; honest win is TPOT/ITL, **not** throughput |
| **3** | **CPU threadpool + compute-in-quant GEMM** (+ tiled/KV-split CPU FA) | High — CPU-only / edge / no-GPU + GB10's 20 Arm cores | 🟢 1 wk + 🟡 2–4 wk | `src/vt/cpu/cpu_threadpool.{h,cpp}` (port ggml pool 1:1), convert `Matmul(BT)` to chunked parallel-for (bit-identical, measurable) | Compute-in-quant ≠ bit-identical to dequant-to-bf16 → need a new CPU parity oracle (keep dequant path as `VT_CPU_REF`) |
| **4** | **AMD MI300X** (gfx942 / CDNA3) | High — #2 datacenter accelerator | 🟡 ~4–6 wk | gfx942 bf16 gate via hip-Triton GDN (one `--target hip:gfx942:64` flag) + hipified glue + vendored `csrc/rocm/attention.cu` MFMA | cutlass fp8/fp4 + FA-2-CuTe are NVIDIA-locked; fp8/fp4 peak must come from AITER/hipBLASLt (follow-on ~3–4 wk) |
| **5** | **Apple Metal via MLX** (M-series dev/consumer) | High — Apple Silicon base | 🟡 ~6–10 wk | Vendor MLX, stand up `src/vt/metal/mlx_ops.cpp` against CPU-ref goldens, then run a small bf16 dense model on the M4/16 GB host | Available host cannot fit the 27B/35B gates; paged KV needs MLX-internal primitives (ported from vllm-metal Apache-2.0); nvfp4 E4M3-vs-UE4M3 scale parity (#2962) |
| **6** | **Vulkan** (cross-vendor: AMD/Intel/NVIDIA consumer + mobile) | Very high but *shared* | 🔴 ~10–16 wk | bf16 dense decode of one gate model via `VK_KHR_cooperative_matrix` + Vulkan paged-KV + command-buffer replay | Fully hand-written SPIR-V (no Triton/cutlass leverage); longest sustained effort; amortizes across all vendors |
| **7** | **EAGLE-3 / MTP spec decode** | Medium — latency serving, low concurrency | 🔴 large | New `src/vllm/v1/spec_decode/` mirroring vLLM + rejection sampler; MTP heads exist upstream | Batch-decay (1.38× @ bs64, can go <1× @ bs24); MoE verification activates more experts; GDN linear-state rollback on rejection is hard |
| **8** | **Intel XPU (SYCL)** (Arc/Battlemage, DC Max) | Medium (Arc growing) | 🔴 ~8–12 wk | Mirror `platforms/xpu` + vendor `vllm-xpu-kernels` SYCL attention/MoE | Highest per-vendor effort; weakest Triton lever; oneDNN dep tension with "we own the code" |
| **9** | **Deferred tail** — grammar jump-forward (JSON serving), q8_0 KV cache, HiCache/router | Niche / workload-specific | mixed | (workload-driven) | jump-forward is a deliberate vLLM deviation → opt-in only, §9 |
| **✗** | **RadixAttention / token-level cache** | — | large | — | **Rejected per MIRROR policy** — vLLM's hash-block APC is already ported; ≈0 gain on gate (random prompts); revisit only if multi-turn/RAG serving is benchmarked (then steal LPM admission ordering, not a KV rewrite) |

## Per-track single most important finding

- **GPU vendors/arches (Report 1):** The bf16 correctness path is **already sm_80+** (WMMA m16n16k16, FA-2 sources literally `_sm80.cu`, Marlin 8.0+, arch-agnostic cuBLASLt + CUDA-graphs). A100/H100/4090/5090 need *essentially no new kernels* — just CMake gating + a capability probe. And the 5 Triton GDN kernels retarget to AMD by one `--target hip:gfx942:64` flag (confirmed in-toolchain), making the hardest kernel we own near-free on ROCm.
- **SGLang steals (Report 2):** Async/overlap scheduling is **vLLM's default at our pin** (`config/scheduler.py:158` → `True`) while we run fully synchronous — it is simultaneously the biggest SGLang-proven lever *and* an unmet mirror obligation. The honest correction: because we already ≥1.0× against async-vLLM, its throughput win is low single-digit %; the real payoff is TPOT/ITL for the serve-latency A/B (task #42) and closing the mirror gap. No SGLang-proprietary kernel beats vLLM's resolved kernels — the kernel "steal" is a no-op.
- **CPU / llama.cpp (Report 3):** The structural fix is **compute-in-quant** — keep GGUF blocks resident and quantize the *activation* to int8 to meet them (like llama.cpp) instead of dequantizing weights to bf16 at load. That buys ~3.3× RAM, ~3.3× decode, and order-of-magnitude prefill on the 35B GGUF. But the **threadpool is the prerequisite** — we have *zero* threads today; ~1 week gives ~N_cores× on every op before any kernel work.
- **Apple/Metal (Report 0):** Ollama's **production swap from mature hand-MSL (llama.cpp) to MLX** (~1.6× prefill, ~2× decode) decisively resolves our `backends.md` E1-vs-E2 question in MLX's favor. Vendor MLX (MIT, C++) under `vt::` exactly as cuBLASLt/cutlass sit on CUDA; port vllm-metal's Apache-2.0 paged/GDN MSL as MLX-internal primitives (the one gap). As in-process C++ embedders we can implement RFC #188 (paged kernels as in-graph primitives) that Python plugins cannot — avoiding vllm-metal's 56-syncs-per-step pathology.

## Disagreements, corrections & unverified flags

**Strategic fork (Report 1's own call, worth escalating):** Vulkan vs SYCL. Report 1 argues Vulkan *dominates* SYCL on reach-per-effort (one backend covers AMD+Intel+NVIDIA consumer + mobile; llama.cpp shows coopmat can beat ROCm on AMD consumer), yet ranks it #4 for effort and calls it "the strategic bet." Decision needed: broad total reach (Vulkan) vs peak Intel-DC throughput (SYCL). My call: Vulkan before dedicated SYCL, and only do SYCL under a specific Intel-DC mandate.

**Corrections to task premises made by the researchers (accept these):**
- xgrammar is **not** vendored — we ship a native GBNF backend (`backend_native.cpp`); grammar-overlap falls out of async scheduling for free, jump-forward is a vLLM-rejected deviation → opt-in §9 only.
- CPU baseline is a **naive single-threaded scalar** triple loop with per-element dtype switch (single-digit GFLOP/s), not a vectorized reference — the win multipliers are off that honest floor.
- Async scheduling is **not a throughput win for us** (we already beat async-vLLM) — reframe its target as latency + mirror parity.

**Must-measure-before-committing (unverified):**
- **Async prize:** nsys the decode steady-state and quantify GPU-idle-between-steps *first* — that number sizes the port. (Adversarial: sglang#19347 shows the overlap scheduler not always delivering even in SGLang.)
- **MLX numeric parity:** built-in `fast::rope`/`rms_norm`/`sdpa` won't be bit-identical to our CPU refs — needs A/B, with `fast::metal_kernel` CPU-ref ports as fallback.
- **nvfp4 cross-backend parity:** MLX uses **signed E4M3** block scales vs NVIDIA's **UE4M3** (#2962) — run the scale-format check before loading our modelopt NVFP4 27B through MLX. Note llama.cpp's new `GGML_TYPE_NVFP4` (PR #19769) uses UE4M3 and *matches* NVIDIA — so nvfp4 portability is backend-specific and unverified across the three.
- **Per-arch Triton autotune:** GDN launch configs are pinned to GB10; each new NVIDIA/AMD arch wants its own sweep for peak (fallback stays correct).
- **AMD consumer risk:** upstream Triton RDNA3 (gfx1100) fused-MoE codegen bug (sglang#30245) — not in blast radius for CDNA GDN, but gates the RDNA consumer follow-on; lean on hand-C++ MoE fallback.
- **CPU numerics:** compute-in-quant changes gate goldens — keep dequant-to-bf16 as `VT_CPU_REF` oracle and add the scalar `vec_dot` reference.

**Environment update (2026-07-10):** an Apple M4 Mac mini with 16 GB is now
reachable at `ssh 192.168.68.103` and recorded in `environment.md`. This removes
the hardware block for MLX compilation, op parity, and small-model milestones.
It does not satisfy 27B/35B gate-model memory requirements; those measurements
still need a larger-memory Apple Silicon host.

---
*Synthesis of 4 research tracks (Metal/MLX, arches/vendors, SGLang, llama.cpp CPU) — workflow wf_24e95c3b-7a5, 2026-07-10. Full per-track reports in the workflow journal.*
