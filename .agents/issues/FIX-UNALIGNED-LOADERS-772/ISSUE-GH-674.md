ID: ISSUE-GH-674
Title: `main` is RED on `sanitize-cpu (address,undefined)` since `cefacd2d0`: `Ltx2LoadVaeWeights` (`ltx2_loader.cpp:1325`) reinterpret_casts the safetensors mmap to `const uint16_t*`, and that payload offset carries NO alignment guarantee — UB everywhere, a real fault on `build-test-cpu-arm64` and Jetson/Orin sm_110. THIRD recurrence of one class after [#301](https://github.com/mudler/vllm.cpp/issues/301) (closed; it left the `vt::LoadUnaligned` seam) and [#627](https://github.com/mudler/vllm.cpp/issues/627) (`qwen3_5_weights.cpp`, still open). The coverage that caught it was ACCIDENTAL — the fixture's JSON header happens to land that tensor odd — so the fix owes a case that FORCES the odd offset and asserts the parity
Row: FIX-UNALIGNED-LOADERS-772
State: UNKNOWN
Kind: bug
GitHub: 674
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:190`

### Frozen archive evidence

> | [#674](https://github.com/mudler/vllm.cpp/issues/674) | — | `main` is RED on `sanitize-cpu (address,undefined)` since `cefacd2d0`: `Ltx2LoadVaeWeights` (`ltx2_loader.cpp:1325`) reinterpret_casts the safetensors mmap to `const uint16_t*`, and that payload offset carries NO alignment guarantee — UB everywhere, a real fault on `build-test-cpu-arm64` and Jetson/Orin sm_110. THIRD recurrence of one class after [#301](https://github.com/mudler/vllm.cpp/issues/301) (closed; it left the `vt::LoadUnaligned` seam) and [#627](https://github.com/mudler/vllm.cpp/issues/627) (`qwen3_5_weights.cpp`, still open). The coverage that caught it was ACCIDENTAL — the fixture's JSON header happens to land that tensor odd — so the fix owes a case that FORCES the odd offset and asserts the parity | bug |

## Resolution

-
