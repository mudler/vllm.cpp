ID: ISSUE-LOCAL-01M2E91MVJ9GV3PAKCF144SVZJ
Title: vLLM merges the HC down and inject projections into ONE padded MergedColumnParallelLinear; we run three GEMMs
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

Verdict (b) at `.agents/specs/qwen4-exp-flash-next.md:176` and `:4626-4629`, carried as 'NO issue yet' until now. At the active parity pin `e126687a9a`, vLLM stacks the hyper-connection down and inject projections into a single `MergedColumnParallelLinear` of shape [lora_rank, hc_count] padded to a multiple of 16 rows, with a WeightsMapper that stacks the two checkpoint tensors (`nvidia/hyperconnection.py:98-110`, `nvidia/model.py:146-155`), and its decode plan is KEYED on that merged (336, 10240) shape (`nvidia/low_latency_gemm.py:72-79`). This tree runs three separate GEMMs and never stacks hc_*_down with hc_*_inject (`cuda_qwen4_exp.cu:414`, `:418`, `:428`; `rocm_qwen4_exp.hip` mirrors it). AGENTS.md mandates routing mergeable projections through `layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`, so upstream's merge is exactly the seam this tree already has and is not reaching. NOT MEASURED: no A/B has been run, and the win is asserted from upstream's structure rather than from a number on this tree.

## Resolution

-
