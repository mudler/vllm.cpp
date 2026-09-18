ID: ISSUE-LOCAL-01M2CKN5516AKE7W2JVDV86Z8X
Title: OWED: dense_attn::ResidentWeight's staged-borrow release is UNCOVERED on the four other GGUF model families that execute it
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

Fix 1 of .agents/specs/rocm-host-residency-after-upload.md put MaybeReleaseStagedBorrowSource inside dense_attn::ResidentWeight (include/vllm/model_executor/models/dense_attn_block.h:246-274), which is a SHARED seam. Four production model families besides qwen4_exp reach it with weights that satisfy both of the release's predicates (a BORROWED span with mmap_fd >= 0, which only the GGUF keep-quant borrow producers set, and a platform plus backend that cannot dereference host memory): GLM-MoE-DSA (glm_moe_dsa_forward.cpp:162-186, :269, :334), GLM5-Next (glm5_next_moe.cpp:243-245), Muse-Glimmer (muse_glimmer.cpp:156, 251, 259, 273, 295, 309, 315, 326, 395, 421, 434, 435 and muse_glimmer_mm.cpp:264) and the Qwen3.5 DFlash draft head sharing a GGUF target's kept-F16 embedding table (qwen3_dflash.cpp:597, 906, 1851, 1956). The release therefore fires on dgx, thor, orin and strix for all five families, and NO test covers any family but qwen4_exp. The focused harness cannot reach the others today: its fake backend registers memory operations only, so every entry point above the seam refuses on a missing op before residency is asked about, and section 4a of the spec records this rather than contorting a case. What is owed is a harness that can drive a second family's production entry point on a non-host-addressable fake platform, and one case per family that convicts the release at that family's own call site. See .agents/specs/rocm-host-residency-after-upload.md section 4a for the verified enumeration and for the two families checked and excluded (DeepSeek-V4, Laguna).

## Resolution

-
