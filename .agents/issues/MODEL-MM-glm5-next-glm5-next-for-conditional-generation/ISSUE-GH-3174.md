ID: ISSUE-GH-3174
Title: MODEL-MM-GLM53-FLASH: the compose — glm5_next_device.cpp and the remaining nine arms
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: OPEN
Kind: UNKNOWN
GitHub: 3174
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`
>
> `Glm5NextForConditionalGeneration` (GLM-5.3-Flash) has three of eleven arms on a
> device queue after W9c-3a (expert GEMM), W9c-2 (KDA recurrence, MoE router
> topk), and W9c-3b (KV binding). The other eight still run on the interposed CPU
> queue that `Glm5NextHostForward` constructs at `glm5_next_forward.cpp:288`. O43
> discloses that, and this issue owns the wave that closes it.
>
> ## Scope
>
> Create `glm5_next_device.cpp` following `kimi_linear_device.cpp`'s
> single-queue, device-resident pattern. Move six arms from host to device where
> providers exist on both CUDA and ROCm:
>
> - RMSNorm (`vt::RmsNorm`)
> - Embedding gather (`vt::Embedding` / `vt::EmbeddingQuant`)
> - Chunked lm_head (`vt::Matmul`)
> - DSA k-pool indexer (`vt::Glm5NextKpoolCompress` / `Select` — O36)
> - MoE combine (`vt::MoeCombine`)
> - Dense and shared MLPs (`vt::Matmul`)
>
> Two arms stay as host-fallback islands:
> - Eager MLA attention (needs W9c-1's port to `mla::ForwardMlaAttentionBlock`)
> - mHC sites (O34 — `kDeepseekV4Mhc` has no ROCm provider)
>
> ## Owed
>
> - W9c-1 (MLA attention onto `mla::ForwardMlaAttentionBlock`) — REFUSED, owns
>   the MLA host-fallback island
> - O34 (`kDeepseekV4Mhc` ROCm provider) — owns the mHC host-fallback island
> - O36 (k-pool device ops wired) — DISCHARGED by this wave

## Resolution

-
