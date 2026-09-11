ID: ISSUE-GH-931
Title: Our OpenAI server fails requests under concurrency where vLLM does not, and the failures silently corrupt every throughput ratio taken against it. Found benchmarking Qwen3.8-27B bf16 ([#915](https://github.com/mudler/vllm.cpp/issues/915)) on GB10: identical client, identical workload, one interleaved series under one GPU lock, ours `failed` 1/6 at c1 and 12/48 at c8 against vLLM's 0/6. `output_throughput` divides tokens by a wall duration that still contains the dead request, so our c1 leg reads 0.675x while median TPOT in the SAME file reads 1.017x in our favour. No harness asserted `failed == 0`, and our server logged nothing — the captured log is 27 startup lines, taken at readiness rather than after the leg
Row: MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 931
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:254`

### Frozen archive evidence

> | [#931](https://github.com/mudler/vllm.cpp/issues/931) | `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation` | Our OpenAI server fails requests under concurrency where vLLM does not, and the failures silently corrupt every throughput ratio taken against it. Found benchmarking Qwen3.8-27B bf16 ([#915](https://github.com/mudler/vllm.cpp/issues/915)) on GB10: identical client, identical workload, one interleaved series under one GPU lock, ours `failed` 1/6 at c1 and 12/48 at c8 against vLLM's 0/6. `output_throughput` divides tokens by a wall duration that still contains the dead request, so our c1 leg reads 0.675x while median TPOT in the SAME file reads 1.017x in our favour. No harness asserted `failed == 0`, and our server logged nothing — the captured log is 27 startup lines, taken at readiness rather than after the leg | bug |

## Resolution

-
