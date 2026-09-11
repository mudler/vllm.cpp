ID: ISSUE-GH-1561
Title: **vllm#52816 MERGED, so this row's gate head and O21's parked D12 decision both come due.** Read from the forge 2026-08-21: `merged: true`, `merged_at 2026-08-21T05:27:22Z`, merge commit `b389ac29465b33f9e9c534df221ea3c129e9793f`, head `3406ec1dae9916f920b90f0dbf90dcf54923d042`. The merge landed 46 minutes BEFORE `SPEC-DFLASH2` W6's work commit `bb416e0ae` was authored (`06:13:50Z`), so five statements in the spec plus the `#1538` index row recorded an open pull request that had already closed. `## Gates` G2's own rule is "`66e5414c` if #52816 has not merged, and the merge commit if it has", so the head it selects today is `b389ac29`. W6's capture stays pinned to `66e5414c` because that is the wheel that ran and it predates the merge -- a dated exception, not the rule -- and what is owed here is moving the gate head and re-reading G2 and G3 at the merged head. O21 additionally parked the `UnquantizedEmbeddingMethod`/`UnquantizedLinearMethod` guard deletion as "the decision when #52816 settles"; it has settled onto vLLM's `main`, our `RefuseQuantizedDflash2LmHead` stands on its own independent reason (the GGUF arm dequantizes `output.weight` to bf16), and writing that decision down against a merged upstream rather than a branch is owed. Re-verified at `3406ec1d`: `logits_processor.py:241-286` preserves the padding mask, the id rebase, the TP all-gather and the scale-THEN-softcap order, so O16's reading holds at the merged head too. Records corrected on `row/SPEC-DFLASH2-W6`; the WORK is owed here
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: verification
GitHub: 1561
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:588`

### Frozen archive evidence

> | [#1561](https://github.com/mudler/vllm.cpp/issues/1561) | `SPEC-DFLASH2` | **vllm#52816 MERGED, so this row's gate head and O21's parked D12 decision both come due.** Read from the forge 2026-08-21: `merged: true`, `merged_at 2026-08-21T05:27:22Z`, merge commit `b389ac29465b33f9e9c534df221ea3c129e9793f`, head `3406ec1dae9916f920b90f0dbf90dcf54923d042`. The merge landed 46 minutes BEFORE `SPEC-DFLASH2` W6's work commit `bb416e0ae` was authored (`06:13:50Z`), so five statements in the spec plus the `#1538` index row recorded an open pull request that had already closed. `## Gates` G2's own rule is "`66e5414c` if #52816 has not merged, and the merge commit if it has", so the head it selects today is `b389ac29`. W6's capture stays pinned to `66e5414c` because that is the wheel that ran and it predates the merge -- a dated exception, not the rule -- and what is owed here is moving the gate head and re-reading G2 and G3 at the merged head. O21 additionally parked the `UnquantizedEmbeddingMethod`/`UnquantizedLinearMethod` guard deletion as "the decision when #52816 settles"; it has settled onto vLLM's `main`, our `RefuseQuantizedDflash2LmHead` stands on its own independent reason (the GGUF arm dequantizes `output.weight` to bf16), and writing that decision down against a merged upstream rather than a branch is owed. Re-verified at `3406ec1d`: `logits_processor.py:241-286` preserves the padding mask, the id rebase, the TP all-gather and the scale-THEN-softcap order, so O16's reading holds at the merged head too. Records corrected on `row/SPEC-DFLASH2-W6`; the WORK is owed here | verification |

## Resolution

-
