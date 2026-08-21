# CLAIM-ROCM-GEMMA4-INDEXED-MAX-T

| Claim | Row IDs | Agent | Worktree | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-ROCM-GEMMA4-INDEXED-MAX-T` | `BACKEND-ROCM` (slug `ROCM-GEMMA4-INDEXED-MAX-T`, issue #838) | hermes-vllm (lab), helper | `/home/don/llms/vllm.cpp-indexed-max-t` | `row/ROCM-GEMMA4-INDEXED-MAX-T` | Owns ONLY: widen `gemma4_moe.cpp` T==1 indexed gate to T≤63 via `VT_GEMMA4_DECODE_INDEXED_MAX_T` default 63, using existing per-token indexed helpers, plus tensor oracle vs serial reference. **EXCLUDED:** packed `ExpertGeGLUFp8TopKIndexedBatched`, DEVICE_GROUP, INDEXED_NOSYNC, #837, #839, #697. Independent history from abandoned `row/ROCM-GEMMA4-XDEV-MOE`. | `IMPLEMENTING` | 2026-08-15 — d973/0f32 repair: retire inside acc_idx scope, Retire fail-closed, independent serial ref, injectable arms, owner kind |
