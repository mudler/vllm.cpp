ID: ISSUE-GH-2046
Title: `qwen3_5.cpp` kept private `Dev`/`DBuf`/`MakeTensor`/`Reshape` copies instead of the shared `dense_device_glue.h` set — the off-framework divergence its own `ResidentWeight` comment records, where a repair reached 25 model files and not this one. The private types also had INTERNAL LINKAGE, which is what forced the MoE placement seam to carry a glue-templated second spelling; migrating collapses it back to one
Row: ENG-HYBRID-PLACEMENT
State: UNKNOWN
Kind: bug
GitHub: 2046
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:780`

### Frozen archive evidence

> | [#2046](https://github.com/mudler/vllm.cpp/issues/2046) | `ENG-HYBRID-PLACEMENT` | `qwen3_5.cpp` kept private `Dev`/`DBuf`/`MakeTensor`/`Reshape` copies instead of the shared `dense_device_glue.h` set — the off-framework divergence its own `ResidentWeight` comment records, where a repair reached 25 model files and not this one. The private types also had INTERNAL LINKAGE, which is what forced the MoE placement seam to carry a glue-templated second spelling; migrating collapses it back to one | bug |

## Resolution

-
