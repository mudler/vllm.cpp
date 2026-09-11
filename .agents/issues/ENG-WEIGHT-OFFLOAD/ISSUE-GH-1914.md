ID: ISSUE-GH-1914
Title: ENG-WEIGHT-OFFLOAD on ROCm: four measured device facts from a throwaway gfx1200 spike, including a non-monotonic budget cliff and hipMallocManaged not migrating
Row: ENG-WEIGHT-OFFLOAD
State: OPEN
Kind: record
GitHub: 1914
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-25
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `ENG-WEIGHT-OFFLOAD`
>
> `ENG-WEIGHT-OFFLOAD` is being built from vLLM's `cpu_offload_gb` mirror
> ([#797](https://github.com/mudler/vllm.cpp/issues/797), the dense half of
> [#149](https://github.com/mudler/vllm.cpp/issues/149)); the config surface has
> landed. **None of it has been measured on ROCm**, and the spec's scope table
> does not mention AMD.
>
> A throwaway spike ran the device-behaviour questions on gfx1200 in August and
> answered four of them. It was never merged, deliberately — it is not the feature
> and must not be mistaken for one. This issue records what it measured so the
> numbers survive the branch, and names the two that contradict a reasonable prior.
>
> **Provenance.** Branch `spike/rocm-523`, commit `5056bbf90`, 2026-08-19, based on
> `7b9e207b1` and now 259 commits behind `main`. RX 9060 XT (gfx1200, 15.92 GiB),
> ROCm 7.2.3. It touched three files (`include/vt/backend.h`,
> `src/vllm/model_executor/models/qwen3_5.cpp`, `src/vt/rocm/rocm_backend.hip`,
> +86 lines) and is inert with no environment variable set, so the OFF arm is the
> unmodified upload path in the same binary.
>
> ## 1. The premise works: a 19.45 GiB checkpoint runs on a 15.92 GiB card
>
> `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` is 19.45 GiB and dies on `hipMalloc: out of
> memory`. With a byte budget of large weights kept in host memory and handed to
> the kernel as a device-readable pointer, **it loads and generates**.
>
> Kernels read host-resident weights **correctly**: on the 14B, offloading 2.00 of
> 6.39 GiB of experts (31%) gave **byte-identical tokens** and cost **10.6%**.
>
> **No new backend virtual was needed for the pinned arm.** On ROCm
> `hipHostMalloc` returns a pointer the device reads directly, and
> `hipHostGetDevicePointer` returns the **same value**.
>
> ## 2. The cost does not scale the way a slab-read microbenchmark predicts
>
> The microbenchmark measured **23 GB/s** on idealised streaming. The real GEMM
> gets roughly **8**. At 47% offloaded, the same 60 GEMM dispatches went
> **8.03 → 31.20 ms/token**.
>
> Anyone sizing an offload budget from a streaming-bandwidth number will be wrong
> by about 3x in the optimistic direction.
>
> ## 3. The budget is NOT monotonic, and the cliff is reproducible
>
> | budget | tok/s |
> |---|---|
> | 6 GiB | **3.21** |
> | 7 GiB | **7.67** |
>
> A *smaller* offload budget is 2.4x slower. Six leaves almost nothing for KV and
> allocator slack. **Unexplained beyond that** — the mechanism was not isolated,
> only the cliff's existence and reproducibility. A naive "offload as little as
> possible" policy walks straight into it.
>
> ## 4. `hipMallocManaged` does NOT migrate on this part
>
> It allocates past VRAM and the device can write to it, so it looks like it
> works. It does not migrate: a paired A/B against the pinned arm was **identical**
> (7.64/7.77 vs 7.63/7.77 tok/s), and `mem_info_gtt_used` stayed **flat at 0.43
> GiB** while `vram_used` filled to 15.76.
>
> Consistent with `docs/ROCM.md` (the managed-memory note, at `:56` on current
> `main`; the spike cited `:148` before that file was rewritten). The practical
> consequence for the offload design: on discrete AMD, managed memory is not a
> shortcut to a UVA tier — the pinned-host path is what actually works.
>
> `Backend::AllocManaged` / `FreeManaged` were added defaulting to `nullptr`,
> meaning "this backend has no managed allocator", so a caller falls back rather
> than assuming.
>
> ## What the spike is NOT
>
> No `WeightOffloader`, no canonical-name targeting, no `cpu_offload_gb`, no
> `supports_weight_offload`. Selection is **raw byte size against a counter**. It
> answers device-behaviour questions; it does not implement the mirrored design,
> and its code should not be lifted.
>
> ## Related, and already recorded elsewhere
>
> llama.cpp's **Vulkan** backend loads the same 19.45 GiB file by spilling into
> the 31.35 GiB GTT the amdgpu driver exposes (`mem_info_gtt_total`, confirmed
> still 31.35 GiB on this box today), while its **HIP** backend refuses exactly as
> we do. That is a backend memory-strategy difference and is recorded in
> [#1400](https://github.com/mudler/vllm.cpp/issues/1400).
>
> Adjacent constraint from the same board:
> [#1870](https://github.com/mudler/vllm.cpp/issues/1870) — `VT_GGUF_KEEP_QUANT=0`
> now OOMs on a 16 GiB card, so keep-quant residency is load-bearing rather than
> optional, which changes what an offload budget is competing for.
>
> ## Caveats
>
> - **One board, one ROCm version, one model family.** gfx1200 / ROCm 7.2.3,
>   Qwen3.5/3.6 GDN-MoE. Nothing here is claimed for CDNA, for another vendor, or
>   for dense architectures.
> - The measurements are from **2026-08-19 on a 259-commit-stale base**. The
>   `GdnPostConvK` fix ([#1402](https://github.com/mudler/vllm.cpp/pull/1402)) and
>   the ROCm keep-quant GEMM ([#523](https://github.com/mudler/vllm.cpp/pull/523))
>   both landed afterwards and change decode cost, so the **ratios** are the
>   durable part and the absolute tok/s figures are not.
> - Host contention was not controlled to benchmark standard.
> - Finding 3's mechanism is unexplained; findings 2 and 4 are single-board
>   observations that a second board would strengthen.
>
> Owned by `ENG-WEIGHT-OFFLOAD`.
>

## Resolution

-
