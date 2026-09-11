ID: ISSUE-GH-1371
Title: CPU had NO attention backend for a head size FLASH_ATTN declines, and the summary line said `0 failed` while it happened. `CpuPlatform::get_attn_backend_priority` returned `{CPU_ATTN, FLASH_ATTN}` where `CPU_ATTN` was a name in a list and nothing else — never registered, so every CPU selection fell through to `FLASH_ATTN`, recorded as behavior-preserving because our CPU paged-attention kernel reads that backend's NHD layout. It was behavior-preserving only while `FLASH_ATTN` accepted everything CPU asked. [#1332](https://github.com/mudler/vllm.cpp/issues/1332) M1 (`369ea7fd4`) gave it `flash_attn.py:170-178`'s `head_size % 8 == 0` rule, and a CPU request with `head_size` 6 then matched no registered backend at all: `SelectAttentionBackendName` throws out of `GPUModelRunner::initialize_kv_cache` with `No valid attention backend for device type 0 from {FLASH_ATTN: [head_size not supported]}`. Attributed by clean before/after on one row's own two shas across two boxes and two architectures (`68a0ff378` on thor sm_110, 13/13 passed; `e35c14d52` on dgx sm_121a, 11 of 13 thrown), so neither box- nor arch-specific. **The reporting shape is the second half of this issue**: assertions collapse `3272 -> 18` while doctest still prints `0 failed`, because a THROWN case runs no assertions — a gate grepping `assertions:` reads eighteen assertions, eighteen passed, zero failed, and calls it clean. Fixed by registering `CPU_ATTN` (`src/vllm/v1/attention/backends/cpu_attn.cpp`), which is the answer upstream `cpu.py:75-87` gives for every CPU request; the FA2 rule is untouched, because it is right about the FA2 kernel and was never about ours
Row: BACKEND-ATTN-REGISTRY
State: UNKNOWN
Kind: bug
GitHub: 1371
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:469`

### Frozen archive evidence

> | [#1371](https://github.com/mudler/vllm.cpp/issues/1371) | `BACKEND-ATTN-REGISTRY` | CPU had NO attention backend for a head size FLASH_ATTN declines, and the summary line said `0 failed` while it happened. `CpuPlatform::get_attn_backend_priority` returned `{CPU_ATTN, FLASH_ATTN}` where `CPU_ATTN` was a name in a list and nothing else — never registered, so every CPU selection fell through to `FLASH_ATTN`, recorded as behavior-preserving because our CPU paged-attention kernel reads that backend's NHD layout. It was behavior-preserving only while `FLASH_ATTN` accepted everything CPU asked. [#1332](https://github.com/mudler/vllm.cpp/issues/1332) M1 (`369ea7fd4`) gave it `flash_attn.py:170-178`'s `head_size % 8 == 0` rule, and a CPU request with `head_size` 6 then matched no registered backend at all: `SelectAttentionBackendName` throws out of `GPUModelRunner::initialize_kv_cache` with `No valid attention backend for device type 0 from {FLASH_ATTN: [head_size not supported]}`. Attributed by clean before/after on one row's own two shas across two boxes and two architectures (`68a0ff378` on thor sm_110, 13/13 passed; `e35c14d52` on dgx sm_121a, 11 of 13 thrown), so neither box- nor arch-specific. **The reporting shape is the second half of this issue**: assertions collapse `3272 -> 18` while doctest still prints `0 failed`, because a THROWN case runs no assertions — a gate grepping `assertions:` reads eighteen assertions, eighteen passed, zero failed, and calls it clean. Fixed by registering `CPU_ATTN` (`src/vllm/v1/attention/backends/cpu_attn.cpp`), which is the answer upstream `cpu.py:75-87` gives for every CPU request; the FA2 rule is untouched, because it is right about the FA2 kernel and was never about ours | bug |

## Resolution

-
