ID: ISSUE-GH-2261
Title: **The G4 llama.cpp ladder cannot run: `llama-server` at the `qwen4exp` pin reports NO KV size, so `KV_BYTES_PER_TOKEN` has to be measured on a lease.** `scripts/qwen4exp-llamacpp-ladder.sh` extracted `KV self size = N` from the server log and, finding nothing, set the term to 0 and passed. Measured on the row's own production capture — `decode-proof/llama-server.log`, the COMPLETE unfiltered server output at 1,862 bytes — there is no `KV self size`, no `llama_kv_cache:` sizing line and no allocation summary at all, with sixteen minutes between `load_model:` and `threadpool init` and nothing printed in between; `/props` carries no KV bytes either, its only sizing fields being `n_ctx = 4096` and `total_slots = 1`. `KV_BYTES_PER_TOKEN` also defaults to 0, so on the real box NEITHER check carried a KV term while the ladder configures `CTX_TOTAL=49152` over 32 slots against a 67.5 GiB model on a 119 GiB unified-memory device that reboots rather than swaps — the 128-GiB-on-a-119-GB-box family, wearing the guard's own name. Nothing caught it because every server stub in the test suite emitted the line, so the fixture and the measured denominator disagreed. The guard now REFUSES (`E_KV_UNREPORTED`, 21) when neither the engine nor the operator supplies a term, and the fixture is silent about KV as the real server is. `/metrics` was read and rejected (it publishes a usage RATIO, not a size); a post-launch RSS check was rejected because whether GB10's unified `cudaMalloc` shows in `smaps_rollup` cannot be settled without a lease. Owed: a per-token cost measured under `-np 32 -c 49152` on a leased load
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: bug
GitHub: 2261
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:905`

### Frozen archive evidence

> | [#2261](https://github.com/mudler/vllm.cpp/issues/2261) | `MODEL-MM-QWEN4-EXP` | **The G4 llama.cpp ladder cannot run: `llama-server` at the `qwen4exp` pin reports NO KV size, so `KV_BYTES_PER_TOKEN` has to be measured on a lease.** `scripts/qwen4exp-llamacpp-ladder.sh` extracted `KV self size = N` from the server log and, finding nothing, set the term to 0 and passed. Measured on the row's own production capture — `decode-proof/llama-server.log`, the COMPLETE unfiltered server output at 1,862 bytes — there is no `KV self size`, no `llama_kv_cache:` sizing line and no allocation summary at all, with sixteen minutes between `load_model:` and `threadpool init` and nothing printed in between; `/props` carries no KV bytes either, its only sizing fields being `n_ctx = 4096` and `total_slots = 1`. `KV_BYTES_PER_TOKEN` also defaults to 0, so on the real box NEITHER check carried a KV term while the ladder configures `CTX_TOTAL=49152` over 32 slots against a 67.5 GiB model on a 119 GiB unified-memory device that reboots rather than swaps — the 128-GiB-on-a-119-GB-box family, wearing the guard's own name. Nothing caught it because every server stub in the test suite emitted the line, so the fixture and the measured denominator disagreed. The guard now REFUSES (`E_KV_UNREPORTED`, 21) when neither the engine nor the operator supplies a term, and the fixture is silent about KV as the real server is. `/metrics` was read and rejected (it publishes a usage RATIO, not a size); a post-launch RSS check was rejected because whether GB10's unified `cudaMalloc` shows in `smaps_rollup` cannot be settled without a lease. Owed: a per-token cost measured under `-np 32 -c 49152` on a leased load | bug |

## Resolution

-
