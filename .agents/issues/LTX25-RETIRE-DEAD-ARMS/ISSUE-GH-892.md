ID: ISSUE-GH-892
Title: The `kMultiGpuParallelism` marker shipped "zero `cfg` hits in either multigpu tree" in the header and "three forms" in the user-visible refusal. At LTX-2 `fd4ded7f` it is 5 hits against 33 files as the control, and there is a fourth form, `BatchParallelGemmaBuilder` (`multigpu/bp_gemma_builder.py:42`). The spec's grep was path-filtered past `docs/multigpu/`, which is #604 again
Row: LTX25-RETIRE-DEAD-ARMS
State: UNKNOWN
Kind: bug
GitHub: 892
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:235`

### Frozen archive evidence

> | [#892](https://github.com/mudler/vllm.cpp/issues/892) | `LTX25-RETIRE-DEAD-ARMS` | The `kMultiGpuParallelism` marker shipped "zero `cfg` hits in either multigpu tree" in the header and "three forms" in the user-visible refusal. At LTX-2 `fd4ded7f` it is 5 hits against 33 files as the control, and there is a fourth form, `BatchParallelGemmaBuilder` (`multigpu/bp_gemma_builder.py:42`). The spec's grep was path-filtered past `docs/multigpu/`, which is #604 again | bug |

## Resolution

-
