ID: ISSUE-LOCAL-01M29AT0818WGXFSS4E03TT3KA
Title: `DSparkV41DraftModel`, DeepSeek-V4.1-Flash's ONLY speculative arm, is registered on vLLM `main` at `registry.py:647` and absent at our pin, where `DSparkDraftModel` (`registry.py:624`) is a different class against the V4 target. V4.1 ships no MTP head at any revision (`vllm/models/deepseek_v4_1/` has no `mtp.py` and `DeepseekV41MTP` does not exist), so V4's MTP+DSpark pair collapses to DSpark alone and `config/speculative.py` branches on `deepseek_v41` to select it. Blocked on the same pin advance as its target, and behind the target itself: a draft head is not portable before the model it drafts for
Row: MODEL-SPEC-deepseek-v4-1-dspark-v41-draft-model
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: -

## Problem

The V4.1 speculative arm has no row, and differs in kind from the V4 one.

## Resolution

-
