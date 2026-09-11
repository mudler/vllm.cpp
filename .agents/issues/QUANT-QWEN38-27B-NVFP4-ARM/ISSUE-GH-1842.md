ID: ISSUE-GH-1842
Title: Target-only Qwen3.8 W4 sampled decode selects mixed-dtype packed GDN
Row: QUANT-QWEN38-27B-NVFP4-ARM
State: OPEN
Kind: UNKNOWN
GitHub: 1842
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-24
Updated: 2026-08-24
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Native Windows CUDA target-only decode for the Qwen3.8-27B W4A16 gate model fails before producing a sampled token when temperature is 0.7 and top-p is 0.8. The production path reaches vt::GdnPackedDecode with mixed_qkv/a/b/out not sharing one FP16/BF16/F32 dtype and throws at src/vt/ops.cpp:2496. The same target model runs through speculative DFlash2 verification because that step takes a different GDN recurrence route. Reproduction evidence: J:\\vllmcpp-dflash2-w4-runs\\73-target-only-repro-t07-p08-1x4. Owning row: SPEC-REJECTION. The fix must preserve the packed-decode dtype predictor/producer invariant and add a target-only production regression test.\n\nFOLLOWING_AGENTS_PROTOCOL\n\nFollowing-Agents-Protocol: true\nAI-Assisted: true\nAssisted-by: AGENT:GPT-5 [CODEX]

## Resolution

-
