ID: ISSUE-GH-489
Title: GDN `output_gate_type` is never parsed: a "sigmoid" checkpoint silently computes silu gating, and no token gate can see it
Row: MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 489
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:111`

### Frozen archive evidence

> | [#489](https://github.com/mudler/vllm.cpp/issues/489) | `MODEL-MM-qwen3-5-qwen3-5-moe-for-conditional-generation` | GDN `output_gate_type` is never parsed: a "sigmoid" checkpoint silently computes silu gating, and no token gate can see it | bug |

## Resolution

-
