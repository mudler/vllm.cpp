ID: ISSUE-GH-1598
Title: **A chain entry of `--speculative-config` called `draft_sample_method` and `rejection_sample_method` a typo, when this engine HONOURS both at the top level of the same document.** Measured at `31cefe631` + `e2a9e035d`: `{"vllm_cpp":{"drafter_chain":[{"method":"mtp","draft_sample_method":"greedy"}]}}` returned `unknown key "vllm_cpp.drafter_chain[0].draft_sample_method"` — the SAME message, modulo the name, that a real misspelling `num_speculatve_tokens` returns. Both keys are genuine `SpeculativeConfig` fields at the parity pin `555967922` (`Literal` aliases at `vllm/config/speculative.py:77,78`, field declarations at `:283` and `:216`), and `CheckValueGatedKey` value-gates both at the top level to upstream's own default, beside a chain, because the verify is engine-wide. Inside an ENTRY they were in neither `kHonouredKeys` nor `kUpstreamUnimplementedKeys`, so `CheckEntryKeys` fell them through to the typo branch — #1160's failure inverted, since #1160 split the classes precisely so that a key vLLM declares does not read as "unknown". Two prose claims were false with it: the function's own comment said it reused "the SAME three classes #1160 established" (it reused two), and `docs/SPECULATIVE-DECODING.md` said entry keys are refused "in the same two classes as above". **FIXED IN FLOW** in the `SPEC-DRAFTER-CHAIN` W1 repair. The correct class is neither honoured nor unimplemented — the keys ARE implemented, they are simply not per-drafter — so class 2 gets its own refusal naming the key, saying the engine honours it, and saying to spell it at the TOP LEVEL beside the chain. Class 2 is now one named `kEngineWideValueGatedKeys` set read by both admissions rather than two hand-written comparisons, which is how the split was lost. Red-first: the new subcase failed 4 assertions before the fix; mutating the class-2 branch out reddens it again. Found by a fresh review of `row/SPEC-DRAFTER-CHAIN-W1`
Row: SPEC-DRAFTER-CHAIN
State: UNKNOWN
Kind: bug
GitHub: 1598
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:563`

### Frozen archive evidence

> | [#1598](https://github.com/mudler/vllm.cpp/issues/1598) | `SPEC-DRAFTER-CHAIN` | **A chain entry of `--speculative-config` called `draft_sample_method` and `rejection_sample_method` a typo, when this engine HONOURS both at the top level of the same document.** Measured at `31cefe631` + `e2a9e035d`: `{"vllm_cpp":{"drafter_chain":[{"method":"mtp","draft_sample_method":"greedy"}]}}` returned `unknown key "vllm_cpp.drafter_chain[0].draft_sample_method"` — the SAME message, modulo the name, that a real misspelling `num_speculatve_tokens` returns. Both keys are genuine `SpeculativeConfig` fields at the parity pin `555967922` (`Literal` aliases at `vllm/config/speculative.py:77,78`, field declarations at `:283` and `:216`), and `CheckValueGatedKey` value-gates both at the top level to upstream's own default, beside a chain, because the verify is engine-wide. Inside an ENTRY they were in neither `kHonouredKeys` nor `kUpstreamUnimplementedKeys`, so `CheckEntryKeys` fell them through to the typo branch — #1160's failure inverted, since #1160 split the classes precisely so that a key vLLM declares does not read as "unknown". Two prose claims were false with it: the function's own comment said it reused "the SAME three classes #1160 established" (it reused two), and `docs/SPECULATIVE-DECODING.md` said entry keys are refused "in the same two classes as above". **FIXED IN FLOW** in the `SPEC-DRAFTER-CHAIN` W1 repair. The correct class is neither honoured nor unimplemented — the keys ARE implemented, they are simply not per-drafter — so class 2 gets its own refusal naming the key, saying the engine honours it, and saying to spell it at the TOP LEVEL beside the chain. Class 2 is now one named `kEngineWideValueGatedKeys` set read by both admissions rather than two hand-written comparisons, which is how the split was lost. Red-first: the new subcase failed 4 assertions before the fix; mutating the class-2 branch out reddens it again. Found by a fresh review of `row/SPEC-DRAFTER-CHAIN-W1` | bug |

## Resolution

-
