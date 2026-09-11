ID: ISSUE-GH-809
Title: GGUF dispatch had no default and fell through to qwen3_5's config builder, so EVERY architecture outside its five dispatched names was refused with `qwen3_5 gguf: unexpected architecture` — the wrong model's name, in an unrelated translation unit. `HfConfigFromGgufDispatch` now refuses by the file's OWN `general.architecture` and names the supported set, and `nemotron_h`/`nemotron_h_moe` reach NemotronH's own W7 refusal, which the W3 subcase had claimed while only ever exercising a hand-built `Kind::kGguf` `ModelSource`. FIXED IN FLOW
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 809
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:256`

### Frozen archive evidence

> | [#809](https://github.com/mudler/vllm.cpp/issues/809) | `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` | GGUF dispatch had no default and fell through to qwen3_5's config builder, so EVERY architecture outside its five dispatched names was refused with `qwen3_5 gguf: unexpected architecture` — the wrong model's name, in an unrelated translation unit. `HfConfigFromGgufDispatch` now refuses by the file's OWN `general.architecture` and names the supported set, and `nemotron_h`/`nemotron_h_moe` reach NemotronH's own W7 refusal, which the W3 subcase had claimed while only ever exercising a hand-built `Kind::kGguf` `ModelSource`. FIXED IN FLOW | bug |

## Resolution

-
