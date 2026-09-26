ID: ISSUE-LOCAL-01M3F8A6CJCJHH40P873BT9DW2
Title: Generalise EXL3 loader beyond DeepSeek-V4 (accept version 1.5.1 + codebook mul1)
Row: QUANT-EXL3
State: OPEN
Kind: QUANT-EXL3
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-26
Updated: 2026-09-26
Closed: -

## Problem

The EXL3 quantized-linear loader is hard-gated to DeepSeek-V4. deepseek_v4_weights.cpp:1010 rejects version != rank-sliced-deepseek-v4-v1 and codebook != mcg. The checkpoint vcruz305/MiMo-V2.6-Flash-RL-EXL3 carries version: 1.5.1 and codebook: mul1, so it fails two VT_CHECKs before any tensor loads. The mul1 kernel and marker acceptance already landed (quant-exl3-mul1.md slices A/B), but the loader that consumes them is the DSV4 row — it does not generalise to a non-DSV4 model. IsExl3Checkpoint lives in deepseek_v4_weights.cpp and no other architecture can reach the scheme.

## Resolution

Move IsExl3Checkpoint out of deepseek_v4_weights.cpp to a shared header. Generalise version/codebook acceptance: accept version 1.5.1 and codebook mul1 as informational fields, not reject gates. DeepSeek-V4 keeps its own version == rank-sliced-deepseek-v4-v1 check in its own loader. Each model that wants EXL3 registers its linear layers with Exl3LinearMethod when IsExl3Checkpoint(config) is true.
