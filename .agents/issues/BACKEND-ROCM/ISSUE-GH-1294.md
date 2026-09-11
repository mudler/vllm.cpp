ID: ISSUE-GH-1294
Title: ROCm decode is kernel-bound: the activation quantizer is 35% of GPU time and GdnPostConvK 19%, while H2D+D2H is under 1%
Row: BACKEND-ROCM
State: OPEN
Kind: record
GitHub: 1294
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-19
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> ROCm decode on gfx1200 is **kernel-bound, not transfer-bound**. Profiled with
> `rocprofv3` (rocprofiler-sdk 7.2.3), the largest costs are two `vt::rocm`
> kernels, and host/device data movement is under 1% of wall time.
>
> This corrects a plausible-sounding but wrong hypothesis: that the MoE reference
> path's host round-trips (`qwen3_5.cpp:6736` "download the hidden once, then
> gather + per-expert MLP") dominate. They do not. The measurement is below.
>
> ## Method
>
> - `rocprofv3 --runtime-trace --stats -f csv`, ROCm 7.2.3, gfx1200 (RX 9060 XT,
>   16 GB, discrete), gcc 15.2.0, NixOS.
> - Model `Qwen3.6-14B-A3B-VibeForged-v2-Q4_K_M.gguf` (7.87 GiB, all experts
>   Q4_K/Q6_K), greedy, `--max-num-seqs 1`.
> - **Decode isolated by differencing**: identical prompt at `--max-tokens 4` and
>   `--max-tokens 36`, subtracted, divided by 32. This removes model load and
>   prefill.
> - Idle host: load average 0.50, no compositor-adjacent load beyond the desktop,
>   GPU 6% at rest. Dispatch counts were byte-identical across a contended and an
>   idle run (281/30/60/20/21), so the structure is deterministic and only
>   durations moved.
> - Profiler overhead is negligible here: profiled wall is 88.7 ms/token
>   (11.3 tok/s) against 11.06-11.18 tok/s unprofiled.
>
> ## Per-token decode
>
> | Domain | per token |
> |---|---|
> | Wall | 88.7 ms |
> | GPU kernel dispatch | 59.99 ms (1738 dispatches) |
> | **Real H2D + D2H transfer** | **0.885 ms** (160 copies) |
>
> | Kernel class | dispatches/tok | ms/tok | share of GPU |
> |---|---|---|---|
> | `vt::` kernels | 976 | 44.92 | 74.9% |
> | hipBLASLt GEMM | 230 | 13.81 | 23.0% |
> | copy/fill plumbing | 532 | 1.26 | 2.1% |
>
> `hipMemcpyAsync` shows 59.31 ms/token of **API** time across 689 calls, but only
> 0.885 ms/token of actual transfer. That API time is the CPU blocking on GPU work
> that has not finished, not PCIe cost. Transfer is not the problem.
>
> ## The two leads
>
> | Kernel | calls/tok | ms/tok | per call | share of GPU |
> |---|---|---|---|---|
> | `QuantizeQ8KK` | 281 | 21.25 | 75.6 us | 35% |
> | `GdnPostConvK` | 30 | 11.24 | 375 us | 19% |
>
> ### 1. Redundant activation quantization (this repo, backend-agnostic)
>
> `MoeBlock` passes the **same** activation buffer to `KqGrouped` twice:
>
> ```cpp
> const std::vector<float> g = KqGrouped(d, act, P, I, H, w.expert_gate_kq, eids);
> const std::vector<float> u = KqGrouped(d, act, P, I, H, w.expert_up_kq, eids);
> ```
>
> (`src/vllm/model_executor/models/qwen3_5.cpp:6832-6833` on `main`.) Each call
> uploads `act` and quantizes it to the weight type's `vec_dot_type`, so the
> identical activation is quantized twice per layer per token. `KqGrouped` is
> backend-agnostic, so CPU and CUDA pay this too; ROCm is only where it was
> measured. Hoisting one quantization removes roughly a third of the quantize
> calls on the MoE path.
>
> ### 2. `GdnPostConvK` at 375 us per call
>
> `src/vt/rocm/rocm_gdn_postconv.hip:57`, 30 calls/token (one per GDN layer),
> 11.24 ms/token = 19% of GPU time, second only to the quantizer. Untouched by any
> open ROCm PR. No hypothesis offered here; it simply has not been looked at.
>
> ## Scope note on the largest kernel
>
> `QuantizeQ8KK` lives in `src/vt/rocm/rocm_grouped_gemm.hip`, which is **not on
> `main`** - it arrives with #523, whose keep-quant registration is what makes the
> model fit on a 16 GB card at all. The 35% figure is therefore feedback on that
> PR rather than a defect on `main`, and is noted there. It is recorded here
> because it is the same measurement and because lead 1 above is on `main` and is
> one of the reasons the call count is what it is.
>
> `main` today registers no `kMatmulBTQuant` on ROCm, so this whole path is
> reachable only with #523 applied. The build measured was `main` @ `4ee5f4a6`
> plus #523 plus the #559/#570 `AttnQkNormRopeGate` fix.
>
> ## What this does not claim
>
> One model, one prompt, batch 1, one board. No prefill attribution. No comparison
> against a pinned oracle - a llama.cpp Vulkan build on the same file and board was
> reported at roughly 77 tok/s by the reporter, which is not a controlled
> comparison and is recorded only as evidence the hardware is not the limit.
>

## Resolution

-
