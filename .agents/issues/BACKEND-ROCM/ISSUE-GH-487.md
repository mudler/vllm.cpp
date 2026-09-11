ID: ISSUE-GH-487
Title: ROCm gfx1200: decode M=1 matmuls go to hipBLASLt 128x128-tile GEMM, not a skinny-GEMM kernel — 77% of decode GPU time vs vLLM's 4%
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 487
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-12
Updated: 2026-08-24
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Summary
>
> On gfx1200, **77.4% of decode GPU time goes to hipBLASLt GEMM kernels selecting
> 128x128 macro-tiles for M=1 shapes.** vLLM on the same board spends **4.1%**
> there, because it routes decode-shaped matmuls to a dedicated skinny-GEMM
> kernel (`wvSplitK`) instead of a BLAS GEMM. This is the single largest
> component of the ROCm decode throughput gap.
>
> Measured with the **same tool on both sides** (`rocprofv3 --kernel-trace`,
> inside the pinned oracle container), same model, same prompt, same 32 generated
> tokens, both traces windowed to the decode phase only.
>
> | | vllm.cpp | vLLM `555967922` | ratio |
> |---|---|---|---|
> | dispatches in window | 11,845 | 11,561 | ~equal |
> | **GPU busy** | **303.7 ms** | **158.1 ms** | **1.92x** |
> | hipBLASLt GEMM share | **77.4%** | **4.1%** | — |
> | dominant kernel | `Cijk_…MT128x128x32` @ 73.7 us | `wvSplitK_hf_sml_` @ 36.3 us | 2.03x |
>
> Our top entries in the decode window:
>
> ```text
>  42.1%   1736 calls    73.70 us  Cijk_Alik_Bljk_BBS_…_MT128x128x32_MI16x16x1_…
>  13.8%     32 calls  1308.74 us  Cijk_Alik_Bljk_BSS_…_MT128x128x32_…      <- vocab projection
>  12.3%    868 calls    43.16 us  Cijk_Alik_Bljk_BBS_…_MT64x64x32_…
>   9.2%    868 calls    32.02 us  Cijk_Alik_Bljk_BBS_…_MT64x64x32_…
> ```
>
> vLLM's, same window:
>
> ```text
>  80.5%   3504 calls    36.31 us  wvSplitK_hf_sml_<__hip_bfloat16, 32, 2, 16, 8, 2, 1>
>   2.7%     56 calls    75.26 us  Cijk_Alik_Bljk_BBS_…_MT128x128x32_…
> ```
>
> ## Why this is the shape of the problem
>
> `MT128x128x32` is a **128x128 macro-tile**. Decode is **M=1** — one row. The
> tile computes 128 rows' worth of work to produce one useful row, so roughly
> 127/128 of each tile is discarded. hipBLASLt is doing what it was asked to do;
> it is being asked the wrong question.
>
> The worst single instance is the vocab projection — `[1,1024] x [151936,1024]`
> — at **1308 us per token**, 13.8% of decode time in 32 dispatches.
>
> Dispatch counts being near-identical (11,845 vs 11,561) rules out launch
> overhead: both sides issue the same amount of work, ours is slower per kernel.
> That is independently confirmed by the decode-graph capture row (#332), where
> capturing hipGraphs — removing essentially all per-step launch cost — moved
> throughput by **+3.2%** and GPU utilisation sits at **100%** throughout decode.
>
> ## Upstream has this, for this exact arch
>
> `csrc/rocm/skinny_gemms.cu` (2,360 lines at the pin) provides `wvSplitK`,
> `wvSplitKrc` and `LLMM1`. The dispatch is
> `vllm/model_executor/layers/utils.py:172-187`:
>
> ```python
> use_skinny = (
>     envs.VLLM_ROCM_USE_SKINNY_GEMM
>     and (on_gfx9() or on_gfx1x())          # gfx1x INCLUDES gfx1200
>     and x.dtype in [torch.float16, torch.bfloat16]
>     and k % 8 == 0
> )
> if use_skinny:
>     if m > 8 and 0 < n <= 5:               # n = token count -> decode
>         out = ops.wvSplitK(weight, x_view, cu_count, bias)
>     elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:
>         out = ops.LLMM1(weight, x_view, 4)
> ```
>
> `on_gfx1x()` covers this board, so the path is not gfx9-only. We have no
> equivalent: `grep -rlniE 'skinny|wvsplitk' src/vt/rocm/` returns nothing, and
> `rocm_matmul_hipblaslt.hip` sends every shape to hipBLASLt regardless of M.
>
> ## Suggested scope
>
> Port the M<=5 decode path and route to it, mirroring upstream's own condition
> rather than inventing one. `rocm_fp8_channel_gemv.hip` shows the project
> already has a GEMV-shaped kernel pattern to follow for a different dtype.
>
> Worth measuring before committing to the full port: a cheap experiment is to
> force a smaller hipBLASLt tile for M=1 and see how much of the 2x is
> tile-selection versus kernel design. If tile choice alone closes most of it,
> that is a much smaller change than a kernel port.
>
> ## Not claimed
>
> - **No ceiling claim.** This names one cause worth ~2x of GPU time; it does not
>   assert it is the only one. See the companion paged-attention issue.
> - Single board (RX 9060 XT, gfx1200, RDNA4, ROCm 7.2.3). The four #41 boards
>   have not been traced.
> - Both traces are single runs. The kernel-share split is large enough that
>   run-to-run noise is not a plausible explanation, but the ms figures are not
>   2-3x-reproduced measurements.
>
> ## Reproduce
>
> Our binary profiles inside the oracle container, which is what makes the
> same-tool comparison possible. It needs the nix store bind-mounted, and
> `LD_LIBRARY_PATH` scoped to the child so `rocprofv3`'s own tools do not pick up
> the nix glibc:
>
> ```sh
> docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
>   -v /nix:/nix:ro -v "$PWD:/work:ro" -v "$HOME/.cache/vllm-cpp-rocm-overlay:$HOME/.cache/vllm-cpp-rocm-overlay:ro" \
>   -v <model>:/models/Qwen3-0.6B:ro -v "$OUT:/prof" \
>   vllm-rocm-oracle:555967922-gfx1200 \
>   rocprofv3 --kernel-trace --stats -d /prof -o ours -- \
>   bash -c "export LD_LIBRARY_PATH=/work/build-hip:<overlay>:<nix lib dirs>:/opt/rocm/lib:/usr/lib/x86_64-linux-gnu; \
>            exec /work/build-hip/examples/vllm-cli --model /models/Qwen3-0.6B \
>              --prompt 'The capital of france is' --max-tokens 32 --temperature 0"
> ```
>
> Both `*_results.db` files are rocpd SQLite; the decode window is the final
> burst of dispatches (bucket by `start` to find the phase boundary).
>

## Resolution

-
