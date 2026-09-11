ID: ISSUE-GH-2002
Title: With `--speculative-config` set, `GPUModelRunner::sample_tokens` branches on `num_draft_tokens > 0` alone and returns the greedy-only `RejectionSampler`'s output, so `Sampler::forward` and `vt::RandomSample` are never called and a `temperature: 1.0` request decodes GREEDILY. `include/vllm/v1/spec_decode/rejection_sampler.h` states the contract it violates in its own deferral list ("a temperature > 0 request must NOT be routed here yet"); neither the runner nor `RejectionSampler::forward` enforces it. Found while writing #1984's acceptance measurement against a baseline recipe carrying `--speculative-config`, where the sampler under test would never have been launched and the null result would have read as "the change did nothing"
Row: SAMPLE-CORE
State: UNKNOWN
Kind: bug
GitHub: 2002
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:784`

### Frozen archive evidence

> | [#2002](https://github.com/mudler/vllm.cpp/issues/2002) | `SAMPLE-CORE` | With `--speculative-config` set, `GPUModelRunner::sample_tokens` branches on `num_draft_tokens > 0` alone and returns the greedy-only `RejectionSampler`'s output, so `Sampler::forward` and `vt::RandomSample` are never called and a `temperature: 1.0` request decodes GREEDILY. `include/vllm/v1/spec_decode/rejection_sampler.h` states the contract it violates in its own deferral list ("a temperature > 0 request must NOT be routed here yet"); neither the runner nor `RejectionSampler::forward` enforces it. Found while writing #1984's acceptance measurement against a baseline recipe carrying `--speculative-config`, where the sampler under test would never have been launched and the null result would have read as "the change did nothing" | bug |

## Resolution

-
