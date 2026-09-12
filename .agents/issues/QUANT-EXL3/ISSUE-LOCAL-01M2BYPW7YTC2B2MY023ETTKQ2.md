ID: ISSUE-LOCAL-01M2BYPW7YTC2B2MY023ETTKQ2
Title: EXL3 reconstruct dispatch (#3150) refuses M>144 on non-CUDA backends, crosses into batched decode, and has no serving measurement
Row: QUANT-EXL3
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

#3150 (`39d3af455`, closes #3124) ported exllamav3's `AUTO_RECONSTRUCT_THRESHOLD` dispatch: `Exl3MatmulD` (`include/vllm/model_executor/models/dense_attn_block.h:304`) sends M > 144 to `vt::Exl3ReconstructGemm`, a reconstruct plus cuBLASLt fp16 GEMM. Its evidence is 264/264 kernel assertions on GB10 and Thor. No serving measurement exists on a tree that contains it, and a read of the landed code finds three defects or risks that no test covers.

1. Non-CUDA backends refuse EXL3 prefill above 144 tokens. The dispatch does not depend on the device, but `kExl3ReconstructGemm` is registered for `kCUDA` only (`src/vt/cuda/cuda_exl3.cu:3081`). `kExl3Gemm` is also registered for CPU (`cpu_exl3_kernels.cpp:510`), ROCm (`rocm_ops.hip:359`) and Vulkan (`vulkan_ops.cpp:1822`). `Resolve` then fails with "no kernel for op Exl3ReconstructGemm" (`src/vt/op_provider.cpp:625`), and the reference tier cannot stand in because the op has no CPU registration. Before #3150 these backends served every M through `Exl3Gemm`. This is read from the code and has not been run. The upstream reference is exllamav3 `exl3.py:132-139`, which is CUDA only, so the port must keep the fallback on the backends that have no reconstruct kernel.

2. Batched decode crosses the threshold. exllamav3 serializes generation (`tools/serve_openai.py` `gen_lock`), so its decode M is 1 + k = 8 and never reaches 144. Our DFlash2 verify runs 8 tokens per sequence, so from c >= 19 a decode step has M > 144. Every EXL3 linear then reconstructs its whole weight to fp16 on each step and allocates a [K, min(N, 32768)] fp16 scratch per call. The effect on c = 32 throughput and TPOT is unmeasured.

3. Divisibility precondition. `vt::Exl3ReconstructGemm` (`src/vt/ops.cpp`) requires both k and n to be multiples of 128 on every path. exllamav3 requires that for the fused Hadamard variant only. No EXL3 tensor has been checked against it, so an EXL3 checkpoint with a 16-divisible, non-128-divisible dimension may now refuse at M > 144.

Early measurement, one leg per engine, which is not publishable alone: another session's variadic run at `f847e1c7` (the row/QUANT-EXL3 head, #3150's content) recorded OURS-r1-c1 and THEIRS-r1-c1 on dgx:gpu0 on 2026-09-12 before its worker was lost. Evidence: `/workspace/exl3-variadic/out/` on the rc share. Median TTFT by band, ours against theirs: S 611 / 549 ms, M 832 / 745, L 1564 / 2319, XL 3699 / 4612. The published pre-#3150 XL value was 9913 against 4900. Output tok/s is 41.82 against 32.48, TPOT p50 16.9 against 21.4 ms, and TTFT p95 3602 against 4380 ms. This suggests that #3150 closed and reversed the long-prompt prefill gap at c = 1.

Owed: a full A/B that is rc job resubmitted after `ba8079ae` failed its corpus checksum on dgx:gpu0 (`/workspace/exl3-3150-ab/`). It runs `39d3af455` (both engines) against its parent `3cafbcaf` (our engine, sharing the exllamav3 legs) at c = 1, 16 and 32, two rounds, with the published c16/c32 recipe. It answers the #3150 effect and item 2, and it adds per-band TPOT from the per-request records. Items 1 and 3 need a code fix with a red-first test through `Exl3MatmulD` on a CPU queue at M > 144.

## Resolution

-
