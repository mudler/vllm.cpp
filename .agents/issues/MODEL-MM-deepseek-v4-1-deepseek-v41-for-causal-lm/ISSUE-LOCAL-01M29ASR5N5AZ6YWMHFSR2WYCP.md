ID: ISSUE-LOCAL-01M29ASR5N5AZ6YWMHFSR2WYCP
Title: DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`) is registered on vLLM `main` at `e77daef89e` and MEASURABLY absent at our parity pin `e126687a9a`, 566 commits away, so no primary oracle exists and AGENTS.md forecloses every secondary for a path vLLM implements. Upstream FORKED the V4 tree rather than extending it (`vllm/models/deepseek_v4_1/` duplicates attention, compressor, sparse_mla and quant_config, and ships no cpu/, no xpu/ and no mtp.py), so this is not a delta on our DeepSeek-V4 port: Engram n-gram conditional memory (1033 lines, two 384M-row tables, 39.8% of the checkpoint by measured bytes, cross-forward hash state), MXFP8 32x32 UE8M0 linears against V4's 128x128 fp8, a second indexer block size, DSpark drafting with no MTP head, a vision tower, and the CED encoder-decoder KV split with its SWA Bounded Replay all have no counterpart here. Every published arm is also HW-blocked on one GB10: the release is 475.27 GiB, the EXL3 3.5bpw hybrid 428.49 GiB, and the only GGUF rung that fits is degenerate by construction. Scoped 2026-09-11 as records only; the row is BLOCKED on the pin advance AND on that hardware, so a pin advance alone does not unblock it, and the advance is a separate owned decision
Row: MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: -

## Problem

A published model this project has no row for, whose oracle cannot exist at the current pin.

## Resolution

-
