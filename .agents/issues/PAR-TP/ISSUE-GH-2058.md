ID: ISSUE-GH-2058
Title: BACKEND-DISTRIBUTED-TP-CLEAN: tp>1 fp8w tail-MLP shard host-dequantizes every layer (~1.7 s/layer) — make it a per-rank device-resident W8A16 GEMM
Row: PAR-TP
State: OPEN
Kind: UNKNOWN
GitHub: 2058
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-27
Updated: 2026-08-27
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

>  Claim
>
>  On the 27B NVFP4 checkpoint's 8 tail layers (fp8w=1, layers 56–63), the tp>1 DenseMlpBlock shard host-dequantizes the resident per-channel FP8 weights every
>  layer: D2H of the e4m3fn [n,k] bytes + f32 [n] per-output-column scale, a row-parallel host decode to f32, then an f32 slice upload per step. Measured
>  ~1600–1780 ms/layer. After the attention NCCL-churn fix (retained comm group, ~520→3 ms/layer) and the fp4 lazy-alloc fix (~340 ms/layer), this fp8w tail is
>  the dominant remaining TP>1 step cost — 8 × ~1.7 s ≈ 13.6 s of the step. These weights already load and run correctly at tp=1 (per-channel keep-quant W8A16
>  via LoadFp8PerChannelRawNK); this is a routing/compute gap, not a loading or format gap.
>
>  Mechanism on the tp>1 path
>
>  src/vllm/model_executor/models/qwen3_5.cpp DenseMlpBlock fp8w arm: ResidentWeight keeps the fp8 bytes on device; the shard D2H's them, dequantizes host-side
>  out[n,k] = F8E4M3ToF32(w[n,k]) * scale[n] (row-parallel), and uploads f32 gate/up/down slices before the host runT + one AllReduceSum over [O].
>
>  Proposed fix (mirrors M-B3 fp4, vt_cuda_mlp_shard_runT_fp4)
>
>  - Build per-rank resident fp8 slices on each lane device (thread-per-lane, one-time lazy, process-lifetime static cache — the M-B3 pattern).
>  - In-kernel W8A16 GEMM: act[i] = siLU(Σ_k x[k]·F8(gate[i,k])·gs[i]) · (Σ_k x[k]·F8(up[i,k])·us[i]), out[o] = Σ_q act[q]·F8(down[o,q])·ds[o], per-rank
>    column-slice over the intermediate (row-slice gate/up, column-slice down), no bf16 rounding — matches the host arm exactly.
>  - One AllReduceSum over [O], reduced result on rank 0; same group AllReduce + process-lifetime per-rank stream/buffer reuse as runT_fp4.
>
>  Logic anchor
>
>  out[n,k] = F8E4M3ToF32(w[n,k]) * scale[n] is byte-identical to the host fp8w arm and to vt::MatmulFp8W8a16 when the activation is f32-exact. Mirror
>  MlpGuActFp4/MlpDownFp4 scheduling/reduction order so the device path reproduces the host runT path it gates on.
>
>  Correctness (each before any accept)
>
>  1. Model-free 2-GPU selfcheck vt_cuda_mlp_shard_runT_fp8w_selfcheck: synthetic fp8 e4m3fn bytes + per-column scales (both signs, subnormal/normal
>     exponents), device kernel vs host-decode + runT reference, tolerance 1e-3.
>  2. Full test_nccl_group 20/20 (197,529 assertions) on GPUs 2,3.
>  3. G5 TP2 serve on 2×V100 (snapshot 7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108): output byte-identical to the certified tp2==tp1 baseline ' Paris.\nThe
>     capital of Germany is Berlin.'.
>
>  Risks
>
>  - Dtype width precisely mirrors the fp4 port: keep the fp8→f32 decode (never silently widen or narrow); a collapse to a numerically-better dequant is
>    invisible to the token gate, so the shape/bytes-moved contract is asserted, not assumed.
>  - Multiplication order must match the host runT reference (accumulator + tree order), not just the value.
>  - A built-kernel failure faults loudly (return rc), never a silent host fallback.
>
>  Out of scope
>
>  - fp4 body layers (done: M-B3, token-exact).
>  - Block-wise 128×128 FP8 format port (#1189) — different checkpoint family, format-level.
>  - tp=1 fp8 input-projection output-dtype/template selection (#339) — different lever.
>
>

## Resolution

-
