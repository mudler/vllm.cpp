# CLAIM-ROCM-GEMMA4-GETBLAS-DUALSLOT

| Claim | Row IDs | Agent | Worktree | Branch | Owned scope | State | Last update |
|---|---|---|---|---|---|---|---|
| `CLAIM-ROCM-GEMMA4-GETBLAS-DUALSLOT` | `BACKEND-ROCM` (slug `ROCM-GEMMA4-GETBLAS-DUALSLOT`, issue #837) | hermes-vllm (lab), helper | `/home/don/llms/vllm.cpp-getblas` | `row/ROCM-GEMMA4-GETBLAS-DUALSLOT` | Owns ONLY: `GetBlas` `tls_slots[2]` in `src/vt/rocm/rocm_matmul_hipblaslt.hip` plus host lifetime seam tests. **EXCLUDED:** Launch/Finish (#839), indexed T (#838), #697 / `rocm_paged_attn.hip`. Independent history from the abandoned combined branch `row/ROCM-GEMMA4-XDEV-MOE`. | `IMPLEMENTING` | 2026-08-15 — 6195 production capture hook load-bearing |
