ID: ISSUE-GH-1715
Title: Tenstorrent cannot run the Qwen3.5/3.8 GDN-hybrid family: the GDN op chain (kGdnPrefill/kGdnDecode/conv/state-I/O) has no TT kernels
Row: BACKEND-TENSTORRENT-GDN
State: CLOSED
Kind: feature
GitHub: 1715
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-22
Updated: 2026-08-30
Closed: 2026-08-30

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-TENSTORRENT-GDN`
>
> ## What
>
> The Qwen3.5/3.8 family (GDN-hybrid: `Qwen3_5ForConditionalGeneration`,
> `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`) cannot run on the Tenstorrent
> backend at all. The Tenstorrent platform allow-list admits only dense
> standard-attention archs (`src/vllm/platforms/tenstorrent.cpp:55`:
> `OPTForCausalLM`, `Qwen3ForCausalLM`, `MistralForCausalLM`), and the GDN
> linear-attention op chain those models require has no Tenstorrent kernel:
> `kGdnPrefill`, `kGdnDecode`, `kL2Norm`, `kRmsNormGated`, `kCausalConv1dFwd`,
> `kCausalConv1dUpdate`, `kGdnStateGather`, `kGdnStateScatter` (further:
> `kGdnSpecDecode`, `kGdnPackedDecode`) all exist as CPU and CUDA arms only.
>
> Because the P150 is a **discrete** PCIe device (`UnifiedMemory() == false`),
> the portable reference tier is withheld and `GetOp` on a miss refuses **by
> name** (`src/vt/op_provider.cpp` `Resolve`), so registering the archs without
> native kernels would hard-fail at the first GDN layer. The ops must land
> first — the same ordering `BACKEND-TENSTORRENT-MISTRAL` used.
>
> ## The mirror-source fact
>
> vLLM has no Tenstorrent platform, so there is no vLLM mirror for any TT kernel.
> The correctness reference is our own CPU f32 arm (the residual-golden /
> `BACKEND-TENSTORRENT-RESIDUAL-GOLDEN` pattern, already the precedent for every
> TT op), and tt-metal's own op library is the implementation substrate, exactly
> as `ttnn::sdpa_decode` is for `kPagedAttention`.
>
> ## The anchor that makes this tractable now
>
> The pinned tt-metal checkout ships a standalone FLA chunked GDN forward:
> `ttnn::operations::transformer::chunk_gated_delta_rule` (q/k/v/g/beta,
> optional initial/final recurrent state, `use_qk_l2norm`, one Tensix core per
> (B*HV) head, on-core state, "matches FLA naive_chunk_gated_delta_rule
> numerics"). That op maps onto `kGdnPrefill`'s contract (q/k pre-normalized by
> the caller, `scale` on q, state `[N,Hv,Dv,Dk]` — a last-two-dims permute away
> from tt-metal's `[B,HV,K,V]` final_state). The decode step is a rank-1
> delta-rule update composable from matmul + eltwise ttnn primitives if the
> chunked op is too heavy at T=1.
>
> ## Scope (row `BACKEND-TENSTORRENT-GDN`, child of `BACKEND-TENSTORRENT`)
>
> - Op-level registration + gating vs the CPU f32 oracle on synthetic shapes
>   (no checkpoint needed, no capacity problem): prefill set first
>   (`kCausalConv1dFwd`, `kL2Norm`, `kRmsNormGated`, `kGdnPrefill`), then the
>   decode set (`kCausalConv1dUpdate`, `kGdnDecode`, `kGdnStateGather`,
>   `kGdnStateScatter`).
> - Out of scope here: arch allow-list registration for `Qwen3_5*` (own row,
>   once the ops exist), MoE, quant arms, `kGdnSpecDecode`/`kGdnPackedDecode`
>   (spec-decode on TT owes async readback #1627 first).
>
> This is the hard prerequisite for Qwen3.8 on Tenstorrent, chosen at planning
> on 2026-08-22.
>

## Resolution

GitHub records closing pull request #1716 (https://github.com/mudler/vllm.cpp/pull/1716) merged on 2026-08-23 as commit `1757330006f670572526d3d97cb36718c3007119`. GitHub closed issue #1715 on 2026-08-30.
