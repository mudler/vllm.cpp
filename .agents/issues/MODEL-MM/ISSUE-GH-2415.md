ID: ISSUE-GH-2415
Title: MODEL-MM-GLM53-FLASH-CUDA: GLM-5.3-Flash's k-pool indexer has no device op, so the device arm is not a routing-only port
Row: MODEL-MM
State: OPEN
Kind: UNKNOWN
GitHub: 2415
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
> `.agents/specs/glm5-next-flash.md` §W9c's rescoping (#2410, #2413) priced the
> device arm as "a PORT, not a kernel campaign", on the finding that "Every
> primitive family this model needs already has a registered CUDA provider" and
> that therefore **no kernel needs writing for correctness**.
>
> That is false for one family, and it is the family on the critical path of all
> 11 DSA layers.
>
> ## GLM-5.3-Flash's indexer is a k-pool indexer, and nothing in this tree implements it on a device
>
> Measured on `e4aa7c527`.
>
> - `include/vt/ops.h:130-131` defines exactly two indexer ops, `kDsaIndexerLogits`
>   and `kDsaTopkSelect`. Its own comment at `:123` names them "the DSA 'Lightning
>   Indexer' selection pair". `kDeepseekV4Dsa` (`:295`) is the same family
>   (`:284-285`). **There is no pooled-selection op of any kind.**
> - `git grep -l 'kpool\|index_kpool\|compress_ape\|compress_gate' src/ include/`
>   returns 11 files, and every one is a `glm5_next_*` file. Zero in the shared
>   seam, zero under `src/vt/`, zero in any other model.
> - `mla::MlaBlockWeights` carries exactly five indexer tensors
>   (`mla_attention.h:510-514`). This model has all five **plus** `kpool_ape`
>   `[index_kpool, head_dim]` and `kpool_gate` `[head_dim, hidden_size]`
>   (`glm5_next_dsa.h:137`, `:141`) — the two learned weights that ARE the
>   pooling. They are named tensors in the published artifact,
>   `indexer_compressor_ape.weight` and `indexer_compressor_gate.weight`, loaded
>   at `glm5_next_loader.cpp:378-380`.
>
> The selection is structurally different rather than a variant. The seam picks
> the top `index_topk` **tokens** from per-token logits. This model pools
> `index_kpool = 4` consecutive valid tokens under a learned per-channel 4-way
> softmax (`glm5_next_dsa.cpp:168-304`), picks the top
> `index_topk / index_kpool = 512` **pools**, expands them back to member token
> indices, then appends the ragged visible tail raw and unscored (`:307-362`). The
> output is **2051** wide, not 2048 (`OutputWidth()`, `glm5_next_dsa.cpp:56-59`),
> and it carries `-1` sentinels and duplicates, which upstream absorbs with
> `scatter_add_` + `ne(0)`.
>
> `glm5_next_dsa.cpp` carries **zero** `vt::Tensor` across its 534 lines.
>
> ## What this costs
>
> The 11 DSA layers cannot reach a GPU by routing alone. Either
>
> 1. a k-pool indexer CUDA kernel is written (a new op family: the pooled
>    compression, the pool-level top-k, and the visible-tail append), or
> 2. the indexer stays on the host, and every DSA layer costs a device-to-host
>    round trip on every step — 11 per step, on the critical path.
>
> Neither is priced in `.agents/specs/glm5-next-flash.md` today. This issue owns
> the measurement and the decision.
>
> ## What it also corrects
>
> O33 states that `glm5_next_attn.cpp` + `glm5_next_dsa.cpp` are 991 host lines
> reimplementing a shipped block. **It is 457 lines, not 991.** `glm5_next_dsa.cpp`
> duplicates nothing, because the seam does not implement what it computes. The
> parallel-path defect is `glm5_next_attn.cpp` alone. O33's text lands with
> #2413; the correction is recorded in the row's spec under
> "### W9c-1 — the MLA route, PRICED and REFUSED".
>

## Resolution

-
