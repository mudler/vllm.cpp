ID: ISSUE-GH-325
Title: NVFP4 W4A16 decode runs ~10x below the memory-bandwidth roof on sm_110 (Jetson Thor): the dense Marlin GEMM is excluded by the `marlin-nvfp4` FEATURE-TABLE cell
Row: BENCH-QWEN38-27B-SOTA
State: OPEN
Kind: UNKNOWN
GitHub: 325
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-11
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ### Summary
>
> On NVIDIA Jetson AGX Thor (sm_110, compute 11.0, 20 SMs, 122 GB unified LPDDR5X @ 273 GB/s
> nominal), single-stream NVFP4 W4A16 decode is roughly an order of magnitude below what the
> device's memory bandwidth allows. The cause is not a missing kernel — the dense Marlin NVFP4
> W4A16 GEMM already vendored in `src/vt/cuda/marlin/` is simply not built for this
> architecture, because `cmake/CudaArchFeatures.cmake` restricts the `marlin-nvfp4` cell to
> `12.0a,12.1a`.
>
> Widening that one cell to include `11.0` makes the existing kernel build and run on sm_110,
> and it is **8.0x-29.0x faster per GEMM at decode shapes** than the kernel currently selected.
>
> **This is the root cause of the flat-with-concurrency throughput I reported in #168.** That
> report observed serving throughput on sm_110 barely improving from concurrency 1 to 4 and
> could not explain it. The explanation is below: every decode GEMM takes an early return into
> a kernel whose cost is strictly linear in the batch dimension, so batching amortises nothing.
> The measurements here are at the GEMM level and reproduce that flatness exactly.
>
> ### Environment
>
> - NVIDIA Jetson AGX Thor, sm_110 / CC 11.0, 20 SMs, 32 MB L2, 122 GB unified memory
> - CUDA 13.2, driver via JetPack 7.2, built in `nvcr.io/nvidia/vllm:26.04-py3`
> - vllm.cpp @ `e3cc4f6`
> - Configure: `-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`
>   (`TRITON=OFF` is required — there is no sm_110 AOT tree)
> - Model shapes: Qwen3-32B (hidden 5120, 64 layers, 64 q heads x 128, 8 KV heads,
>   intermediate 25600), NVFP4A16
>
> ### What happens today
>
> `vt::MatmulNvfp4` dispatches in `src/vt/cuda/cuda_matmul_nvfp4.cu::Launch()`:
>
> ```cpp
> constexpr int64_t kTileMinRows = 32;
> ...
> if (m < kTileMinRows) {
>     MatmulNvfp4KernelNaive<<<dim3((n+255)/256, m), 256>>>(...);
>     return;                      // returns BEFORE the WMMA branch
> }
> if constexpr (bf16) { if (WmmaEnabled()) { MatmulNvfp4Wmma<...> } }
> ```
>
> At decode the activation row count `m` equals the batch (1, 2, 4, ...), always `< 32`, so
> every decode GEMM takes the early return into `MatmulNvfp4KernelNaive` — a CUDA-core,
> one-thread-per-output-column kernel. Confirmed with Nsight Compute 2026.1.1:
> `sm__inst_executed_pipe_tensor.sum` is **0** at M=1 and **1310720** at M=32.
>
> The threshold behaves exactly as written. Measured, q_proj (N=8192, K=5120), default dispatch:
>
> | M | 29 | 30 | 31 | **32** | 33 | 34 |
> |---|---|---|---|---|---|---|
> | ms/call | 21.97 | 22.44 | 23.37 | **1.79** | 1.80 | 1.89 |
>
> A 13.1x discontinuity at exactly `kTileMinRows`. Output checksums rise monotonically straight
> through the boundary, so both kernels are correct — the step is dispatch, not breakage.
>
> ### Why this is not simply a threshold-tuning bug
>
> Lowering `kTileMinRows` does **not** fix single-stream decode. Forcing M=1 onto the existing
> bf16 WMMA path makes it *slower*, because a BM=64 tile wastes 63/64 of its rows:
>
> | shape | M=1 naive | M=1 forced WMMA |
> |---|---|---|
> | q_proj | **0.945 ms** | 1.696 ms |
> | gate_proj | **2.361 ms** | 4.742 ms |
> | down_proj | **2.561 ms** | 5.406 ms |
>
> Break-even between the two is M=2 on this device (naive 0.945 / 1.888 / 2.352 vs WMMA
> 1.696 / 1.696 / 1.702 at M=1/2/3), so the `>= 32` threshold is mistuned for a ~20-SM part —
> but retuning it only optimises between two kernels that are both far off the roof.
>
> ### Where the time actually goes
>
> Nsight Compute on `MatmulNvfp4KernelNaive`, gate_proj, M=1:
>
> ```
> Mem Pipes Busy                     98.26 %      <- saturated
> Max Bandwidth (DRAM)               11.47 %      <- nowhere near
> Compute (SM) Throughput            77.55 %  (q_proj)
> Executed IPC                        0.90
> L1/TEX Hit Rate                    76.90 %
> Achieved Occupancy                 28.58 %
> Waves Per SM                        0.27
> l1tex ... sectors_per_request       3.66 sector  <- loads ARE coalesced
> ```
>
> The kernel is **LSU-issue-bound**, not DRAM-bound and not math-bound. Each thread owns one
> output column and re-reads the whole activation row with 2-byte scalar loads: for gate_proj
> that is ~131M activation loads against ~4M vectorised weight loads, a ~32:1 ratio. Note the
> loads are well coalesced (3.66 sectors/request) — the problem is the *number* of requests,
> not their shape.
>
> For contrast, the Marlin kernel below issues far fewer, much wider requests
> (16 sectors/request) at *lower* occupancy (16.67%) and *lower* SM throughput (28.38%), and is
> 11x faster. Occupancy and SM-throughput heuristics both point the wrong way here.
>
> ### The fix: the kernel is already in the tree
>
> `src/vt/cuda/cuda_marlin_dense.cu` is a 1:1 lift of vLLM's dense Marlin NVFP4 W4A16 GEMM, and
> the host wiring is complete behind `#ifdef VT_MARLIN_NVFP4`
> (`include/vllm/model_executor/models/dense_nvfp4_gemm.h`). It is excluded from an sm_110 build
> by one line, `cmake/CudaArchFeatures.cmake:333`:
>
> ```
> "marlin-nvfp4|12.0a,12.1a|vendored Marlin NVFP4 W4A16 MoE GEMM (VT_MARLIN_NVFP4)"
> ```
>
> The row's own comment already flags this as a policy choice rather than a technical limit:
>
> > upstream: vLLM MARLIN_ARCHS "8.0+PTX;12.0a;12.1a" (CMakeLists.txt:558) — the sm80+PTX leg is
> > NOT claimed here: our vendored slice is the bf16 NVFP4 instantiation only and has never been
> > built or run outside sm_12x.
>
> This issue is that claim being tested on one more architecture. See the linked PR for the
> one-line change, measurements and numerics validation.
>
> ### Measured result (details in the PR)
>
> M=1, Qwen3-32B shapes, same binary, same random operands:
>
> | shape | naive | Marlin | speedup |
> |---|---|---|---|
> | q_proj | 0.9461 ms | **0.0829 ms** | **11.4x** |
> | kv_proj | 0.5009 ms | **0.0173 ms** | **29.0x** |
> | o_proj | 0.8093 ms | **0.0794 ms** | **10.2x** |
> | gate_proj | 2.3675 ms | **0.2943 ms** | **8.0x** |
> | down_proj | 2.6945 ms | **0.2870 ms** | **9.4x** |
>
> Marlin is also flat in M (0.0821 / 0.0824 / 0.0819 / 0.0819 / 0.0826 ms at M=1..5 for q_proj),
> which is the concurrency scaling the naive path lacks entirely.
>
> ### Scope of the claim
>
> Validated on **sm_110 only**, on one device, with one model's shapes. I have not built or run
> this for sm_80 / sm_86 / sm_89, and I am not claiming the `8.0+PTX` leg — only that sm_110
> works. Whether to widen the cell further is a separate question.
>
> ---
>
> ## Separate observed defect (worth its own issue)
>
> `vllm-bench` with `--model <Qwen3-32B-NVFP4A16 dir> --num-prompts 2 --input-len 32
> --output-len 32 --concurrency 1` on sm_110: after loading the 20 GB checkpoint, RSS grew
> steadily at ~13 MB/s for 17 minutes with no benchmark output ever produced (host memory 84 GB
> -> 111 GB used). GPU power draw was non-zero throughout, so it was doing work.
>
> I killed the run to protect other services on the box, so this is **unreproduced by design** and
> I have no end-to-end tok/s from it. Reported as an observation, not a diagnosis — it may be
> specific to this arch/model/build combination.
>

## Resolution

-
