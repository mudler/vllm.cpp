ID: ISSUE-LOCAL-01M36X2DRZFFF2PY26DPXMDNJP
Title: Q4_K_M quant-block loading fails in qwen3 GGUF loader (RequireExpand vs keep_quant conflict)
Row: BACKEND-GATE-ROCM-SGLANG
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

LoadQwen3FromGguf in qwen3_gguf_weights.cpp uses RequireExpand to assert that merged matmul weights (q/k/v, gate/up) expand to bf16. On CPU, the default policy has keep_quant=true, so Q4_K_M matmul weights route to kKeepQuant and RequireExpand fails with: 'a matmul_weight tensor must not keep quant blocks: blk.0.attn_q.weight'. The fix is to narrow the policy with NoKeepQuant(pol) for the RequireExpand calls, following the pattern used in laguna_weights.cpp and deepseek_v4_weights.cpp.

## Resolution

Fixed in this commit: added `const GgufLoadPolicy expand_pol = NoKeepQuant(pol);` after
the policy resolution in LoadQwen3FromGguf, and changed all 10 RequireExpand calls
to use expand_pol instead of pol. This forces Q4_K_M matmul weights to route to
kExpandBf16 instead of kKeepQuant on CPU, satisfying the assertion. o_proj and
down_proj still use the original pol (they may keep quant). The unit test
test_qwen3_gguf_weights passes (5/5 tests, 117 assertions).
