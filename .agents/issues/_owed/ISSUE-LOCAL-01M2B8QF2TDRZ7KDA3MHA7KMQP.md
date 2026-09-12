ID: ISSUE-LOCAL-01M2B8QF2TDRZ7KDA3MHA7KMQP
Title: backend(ROCm): the qwen4_exp forward still refuses after W4, at the DSA indexer pair
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

MODEL-MM-QWEN4-EXP W4 landed the last two of the NINE ops `.agents/specs/qwen4-exp-rocm-ops.md` scopes, and the `qwen4_exp` forward still refuses on `rocm`. The scope of nine was measured against the ops this model's blocks call DIRECTLY; it missed two the QSA block COMPOSES its indexer from.

`src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp:391,403` calls `vt::DsaIndexerLogits` and `vt::DsaTopkSelect` on the model's own queue, unconditionally — the second one runs even when `nb == 0`, because the all-`-1` selection it writes IS upstream's `num_complete_blocks == 0` branch. Neither op has a `kROCM` registration: `grep -rn 'kDsaIndexerLogits|kDsaTopkSelect' src/vt/` reaches `src/vt/cpu/`, `src/vt/cuda/cuda_dsa_indexer.cu:320,322` and two ROCm COMMENTS, one of which (`src/vt/rocm/rocm_ops.hip:374`) already records the absence from the GLM-5.3 side: 'The DSA indexer pair (kDsaIndexerLogits, kDsaTopkSelect) is still absent, so the GLM-5.3 speed axis stays VOID and a SPARSE step still refuses.'

On gfx1151 that is a refusal by name and not a slow path, for the reason spec D2 measures: `hipDeviceAttributePageableMemoryAccess = 0` since #2511, so the portable CPU reference tier is never installed and `GetOp` throws. The first refusal a `qwen4_exp` step on ROCm now meets is `kDsaIndexerLogits`, one call AFTER `vt::Qwen4ExpQsaCompress` which W4 just landed.

What this owes: a wave that gives both indexer ops a ROCm arm, gated against the CPU oracle the way W1-W4 were, after which `ModelRegistry::Forward` can be attempted on that board for the first time. The gap is filed rather than fixed inside W4 because it is two more ops with their own mirror question — vLLM's AMD backend DIVERGES on exactly this pair, replacing `nvidia/ops/qsa.py`'s tensor-core `tl.dot` scoring kernel with a per-head `tl.sum` loop and routing top-k to `ops.top_k_per_row_decode` instead of `torch.ops._C.{cooperative,persistent}_topk` — and spec D3 makes that divergence binding rather than optional. It is ALSO not qwen4_exp-only: GLM-5.3's sparse step refuses on the same two.

## Resolution

-
