ID: ISSUE-GH-1097
Title: `ltx2-gen --lora a --lora b` fuses `b`, DISCARDS `a` and exits 0. `SetExtra` (`examples/ltx2_gen/main.cpp:212-221`) overwrites an existing key in place, so N invocations of `--lora` (`:255-262`) leave exactly one `lora_path` extra. `docs/USAGE.md:809-813` published the opposite - "a second `--lora`" as one of three things that refuse by name - and is corrected in the change that filed this. The wider half, measured after filing: the refusal is unreachable from EVERY production entry point, not only the CLI. `ltx2_video.cpp:813` is the only `dit_options.loras.push_back` in the tree and runs at most once under `if (!lora_path.empty())`, so `options.loras.size()` is 0 or 1 for the CLI, for `vllm_video_engine_load` and for the server alike; `Ltx2ResolveLoraReferenceFactors`'s `> 1` branch (`ltx2_lora.cpp:243-248`) is reached only by `test_ltx2_lora.cpp:384,:492`. So it is correct code guarding a state nothing can construct yet, and the state it guards is what N-adapter fusion ([#932](https://github.com/mudler/vllm.cpp/issues/932)) introduces - which is where the reachability half belongs. Two fix shapes, neither chosen here: refuse the second `--lora` in the CLI, or accumulate and let the library refusal fire. Second defect in the same area and from the same landing: `ltx2_video.cpp:362-363` says "nine of these ten reach a reader" about `kKnownLoadExtras`, which now holds TWELVE entries (`:377-383`) after `lora_path` and `lora_strength` landed with #923; eleven of twelve reach a reader and `duration_head_path` is still the one that does not
Row: ROAD-V1-LTX25
State: UNKNOWN
Kind: bug
GitHub: 1097
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:325`

### Frozen archive evidence

> | [#1097](https://github.com/mudler/vllm.cpp/issues/1097) | `ROAD-V1-LTX25` | `ltx2-gen --lora a --lora b` fuses `b`, DISCARDS `a` and exits 0. `SetExtra` (`examples/ltx2_gen/main.cpp:212-221`) overwrites an existing key in place, so N invocations of `--lora` (`:255-262`) leave exactly one `lora_path` extra. `docs/USAGE.md:809-813` published the opposite - "a second `--lora`" as one of three things that refuse by name - and is corrected in the change that filed this. The wider half, measured after filing: the refusal is unreachable from EVERY production entry point, not only the CLI. `ltx2_video.cpp:813` is the only `dit_options.loras.push_back` in the tree and runs at most once under `if (!lora_path.empty())`, so `options.loras.size()` is 0 or 1 for the CLI, for `vllm_video_engine_load` and for the server alike; `Ltx2ResolveLoraReferenceFactors`'s `> 1` branch (`ltx2_lora.cpp:243-248`) is reached only by `test_ltx2_lora.cpp:384,:492`. So it is correct code guarding a state nothing can construct yet, and the state it guards is what N-adapter fusion ([#932](https://github.com/mudler/vllm.cpp/issues/932)) introduces - which is where the reachability half belongs. Two fix shapes, neither chosen here: refuse the second `--lora` in the CLI, or accumulate and let the library refusal fire. Second defect in the same area and from the same landing: `ltx2_video.cpp:362-363` says "nine of these ten reach a reader" about `kKnownLoadExtras`, which now holds TWELVE entries (`:377-383`) after `lora_path` and `lora_strength` landed with #923; eleven of twelve reach a reader and `duration_head_path` is still the one that does not | bug |

## Resolution

-
