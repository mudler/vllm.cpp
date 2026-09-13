ID: ISSUE-LOCAL-01M2B8QF2TDRZ7KDA3MHA7KMQP
Title: backend(ROCm): the qwen4_exp forward still refuses after W4, at the DSA indexer pair
Row: MODEL-MM-QWEN4-EXP
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

MODEL-MM-QWEN4-EXP W4 landed the last two of the NINE ops `.agents/specs/qwen4-exp-rocm-ops.md` scopes, and the `qwen4_exp` forward still refuses on `rocm`. The scope of nine was measured against the ops this model's blocks call DIRECTLY; it missed two the QSA block COMPOSES its indexer from.

`src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp:391,403` calls `vt::DsaIndexerLogits` and `vt::DsaTopkSelect` on the model's own queue, unconditionally — the second one runs even when `nb == 0`, because the all-`-1` selection it writes IS upstream's `num_complete_blocks == 0` branch. Neither op has a `kROCM` registration: `grep -rn 'kDsaIndexerLogits|kDsaTopkSelect' src/vt/` reaches `src/vt/cpu/`, `src/vt/cuda/cuda_dsa_indexer.cu:320,322` and two ROCm COMMENTS, one of which (`src/vt/rocm/rocm_ops.hip:374`) already records the absence from the GLM-5.3 side: 'The DSA indexer pair (kDsaIndexerLogits, kDsaTopkSelect) is still absent, so the GLM-5.3 speed axis stays VOID and a SPARSE step still refuses.'

On gfx1151 that is a refusal by name and not a slow path, for the reason spec D2 measures: `hipDeviceAttributePageableMemoryAccess = 0` since #2511, so the portable CPU reference tier is never installed and `GetOp` throws. The first refusal a `qwen4_exp` step on ROCm now meets is `kDsaIndexerLogits`, one call AFTER `vt::Qwen4ExpQsaCompress` which W4 just landed.

What this owes: a wave that gives both indexer ops a ROCm arm, gated against the CPU oracle the way W1-W4 were, after which `ModelRegistry::Forward` can be attempted on that board for the first time. The gap is filed rather than fixed inside W4 because it is two more ops with their own mirror question — vLLM's AMD backend DIVERGES on exactly this pair, replacing `nvidia/ops/qsa.py`'s tensor-core `tl.dot` scoring kernel with a per-head `tl.sum` loop and routing top-k to `ops.top_k_per_row_decode` instead of `torch.ops._C.{cooperative,persistent}_topk` — and spec D3 makes that divergence binding rather than optional. It is ALSO not qwen4_exp-only: GLM-5.3's sparse step refuses on the same two.

## Resolution

RESOLVED 2026-09-12 by W5 of `.agents/specs/qwen4-exp-rocm-ops.md`. `src/vt/rocm/rocm_dsa_indexer.hip` registers `vt::DsaIndexerLogits` and `vt::DsaTopkSelect` on `kROCM`, self-registering in W1's pattern with `rocm_ops.hip` untouched. Measured on `strix:gpu0` (gfx1151, ROCm 7.2.4 / HIP 7.2.53211). **TWO FIXTURES ARE QUOTED AND THIS BLOCK SAYS WHICH**, because a fresh review of the first green returned FAIL and the repair changed the logits case: `kS` went from 40 keys to 41 and the logits buffer gained a guard band, which adds one `guard_touched` assertion per arm over eight arms. RED at `3c415b750`, 60 cases / 58 passed / 2 failed / 0 skipped, 84565 assertions, both failures the registration REQUIREs and neither a number; first GREEN at `044d9da17`, 60 cases / 60 passed / 0 failed / 0 skipped, 84825 assertions, at the PRE-REPAIR fixture; GREEN AT THE SHIPPING ARM at `8eef4a429`, 60 cases / 60 passed / 0 failed / 0 skipped, 84833 assertions, at the REPAIRED fixture. `8eef4a429` is the last commit on this branch that touches a translation unit the suite compiles, so it is the head this record ships. The red and the first green were measured at `0b605b80a` and `4c9e2042c`, which the rebase onto `63abad75c` replayed as the two commits named above; spec D3h checks that the replay changed no file the suite compiles, so the binary sha256 values there still identify the arms. At the repaired fixture the logits arm is BIT-IDENTICAL to the CPU oracle in all four arms at the released 64-head geometry (0 differing bytes of 820) and 1.99e-14 to 2.84e-14 NMSE at 96 heads, fourteen orders under the `5e-4` band; the four zeroes are zero at both fixtures. The selector is INDEX-EXACT (0 differing indices of 6, 30 and 72 and 0 differing counts at topk 1, 5 and 12), and the fixture repair did not touch the selector case. FIFTEEN mutations, THIRTEEN red and TWO equivalent mutants with their analyses, `G3` (a redundant guard) and `T5` (the short-context branch at `n == topk`, where both branches compute the same answer so no fixture can separate them); spec D3h has the table, says which fixture each mutation was measured at, and D3g has the vLLM AMD mirror comparison. TWO ROWS OWNED THIS PAIR: it is also W2 of `.agents/specs/rocm-mla-dsa-ops.md` under BACKEND-ROCM / #2715, reconciled onto the one arm in the same change. WHAT THIS DOES NOT MEAN: it removes the last KNOWN op refusal for qwen4_exp on ROCm, and nothing has loaded a checkpoint or completed a forward on that board; W6 owns that and no token or performance claim follows from this wave.
