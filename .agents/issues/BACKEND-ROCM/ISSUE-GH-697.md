ID: ISSUE-GH-697
Title: feat(rocm/gemma4): integrate bc64 flash-attention prefill kernel (2.76x isolated) env-gated on gfx1201
Row: BACKEND-ROCM
State: OPEN
Kind: feature
GitHub: 697
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-14
Updated: 2026-08-14
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Goal
>
> Integrate the validated bc64 flash-attention prefill kernel (striped online softmax + WG256 packaging + PV 8-wave, isolated best 8744us = 2.76x over the current SharedK path at T=2048/d512, identity-verified) into the production ROCm gfx1201 prefill attention path, **env-gated, default OFF**.
>
> ## Context
>
> Isolated kernel development (thread on the bus) took the gfx1201 prefill attention from SharedK-class to 2.76x via three evidence-backed fixes: eliminating a 21x-redundant lane-replicated softmax, WG256 active-wave packaging, and PV 8-wave distribution. The remaining ~1.94x to Vulkan/ACO is dynamic-scheduling/codegen and was ruled a ceiling. This issue tracks landing the 2.76x construction into the serving path (projected ~1.63x whole-prefill once integrated).
>
> ## Scope
>
> - Add the bc64 kernel adjacent to `PagedAttnPrefillSharedKWmma` (`src/vt/rocm/rocm_paged_attn.hip:1307`), adapted to the **paged-KV** interface (block_table / seq_lens / query_start_loc), not the isolated contiguous form.
> - Add a default-OFF `VT_ATTN_PREFILL_BC64_FA` host branch inside the eligible BF16/GQA block, before the SharedK dispatch (~:1694).
> - Supported shapes only: d512 qg8 full window; d256 qg2 sliding window. Unsupported shape or env-absent/0 → byte-identical fallthrough.
>
> ## Out of scope
>
> - Replacing `PagedAttnPrefillSharedKWmma`. Importing the dirty lab `VT_ATTN_PREFILL_*` experiment matrix. Decode path. Non-BF16.
>
> ## Gates
>
> - off/default path byte-identical to current KEEP.
> - on: shape reachability for d512-full + d256-sliding via real block_table/seq_lens/query_start_loc.
> - token-exact vs the current SharedK path on real serving prompts BEFORE any perf number.
> - same-binary A/B (flag on vs off) product p42k, fair PC=0, reverse-order medians.
>
> ## Owner
>
> Coordinator + hermes-vllm implement (don-agent offline, developer-directed); research fresh-review + p42k gate rerun.
>
> Kind: feature

## Resolution

-
