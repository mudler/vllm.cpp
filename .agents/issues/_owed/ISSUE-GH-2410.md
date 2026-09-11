ID: ISSUE-GH-2410
Title: MODEL-MM-GLM53-FLASH-CUDA: GLM-5.3-Flash has no device arm, and it is a port onto existing kernels rather than a kernel campaign
Row: -
State: OPEN
Kind: UNKNOWN
GitHub: 2410
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-MM-GLM53-FLASH-CUDA`
>
> `Glm5NextForConditionalGeneration` (GLM-5.3-Flash) has no device arm.
> `glm5_next_forward.cpp:231-238` refuses a non-CPU queue by name, and
> `--device cuda` has never produced a token from this model. The only generation
> this row has ever observed is O30's ` Paris.` on `dgx:gpu0` with `--device cpu`
> at 195.5 s/token.
>
> `.agents/specs/glm5-next-flash.md` §W9c priced that debt as a kernel campaign on
> the premise that the six forward files "carry ZERO `vt::Tensor`". Measured on
> `0b4766c96`, they carry **25 across 2,783 lines** — 15 in `glm5_next_moe.cpp`
> and 10 in `glm5_next_kda.cpp`, which also make five `vt::` compute-op calls
> between them. Every primitive family this model needs already has a registered
> CUDA provider, and the sibling `GlmMoeDsaForCausalLM` already drives
> `mla::ForwardMlaAttentionBlock` on a GPU through a seam W3 widened to admit this
> row's NoPE geometry. The debt is a PORT, not a kernel campaign.
>
> This issue tracks the port, which the rescoping splits into four waves:
>
> - **W9b** — keep-quant device residency of the 101.24 GiB artifact via
>   `dense_attn::ResidentWeight`. Open question: unified memory on GB10.
> - **W9c-1** — retire `glm5_next_attn.cpp` + `glm5_next_dsa.cpp`'s hand-rolled
>   MLA/DSA block (991 host lines) onto `mla::ForwardMlaAttentionBlock`. 11 of 45
>   layers. This is also the AGENTS.md §"Shared seams" parallel-path defect (O33).
> - **W9c-2** — lift the two CPU-only refusals (`glm5_next_kda.cpp:322-325`,
>   `glm5_next_moe.cpp:222-225`) so the KDA and MoE arms reach the CUDA providers
>   they already call. 34 of 45 layers plus every sparse block.
> - **W9c-3** — the compose: `Dev`/`DBuf`, a device-carrying `ForwardLogits`, the
>   `input.gather_logits` dispatch predicate with a residency clause, RMSNorm /
>   embedding / lm_head onto their ops, and the mHC decision (O34: mHC has a CUDA
>   kernel and no CPU registration). This wave deletes the refusal.
>
> None is started. The rescoping itself lands no product code.
>
> Fleet constraint measured while scoping: on `thor:gpu0` (compute_cap 11.0) a
> CUDA build with `-DVLLM_CPP_FLASH_ATTN=ON` and CUTLASS 4.5.0 configures to
> `CUDA FA2 compiled-arch manifest: []` — FA2's arch table is
> `8.0,8.6,8.7,8.9,12.0a,12.1a` (`cmake/CudaArchFeatures.cmake:349`). MLA prefill
> on this family IS FlashAttention with no fallback beneath it, so the eventual
> `--device cuda` end-to-end test can only ever run on `dgx:gpu0` (`sm_121a`).
>
> Owed entries: O32 (the arm, as four waves), O33 (the parallel path), O34 (mHC's
> inverted gap), O35 (two drifted anchors in this row's own briefing).
>

## Resolution

-
