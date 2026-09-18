# ROCm wvSplitK: gfx1151 dispatch tuning

Row: `BACKEND-ROCM`. Issue: `ISSUE-LOCAL-01M2JXGESJW624DWDHBM4SF8KS`.

## Problem

The wvSplitK skinny GEMM kernel (`src/vt/rocm/rocm_skinny_gemm.hip`) accounts
for **28% of decode time** on Qwen3.8-27B (gfx1151, ROCm 7.2.4). The kernel
body was ported faithfully in #506, but the dispatch configuration remained
fixed at `YTILE=2, UNRL=2` for all shapes.

Upstream vLLM (`csrc/rocm/skinny_gemms.cu:1240-1258`) uses gfx1151-specific
dispatch logic that selects different YTILE/UNRL combinations based on K, N,
and LDS fit. Our fixed configuration is suboptimal for typical decode shapes.

## Scope

- **In scope**: `wvSplitKSml` kernel dispatch in `rocm_skinny_gemm.hip`
- **Formats**: bf16 only (the existing port's dtype)
- **Shapes**: decode shapes (M=1-4, N > 8, K varies by layer)
- **Architectures**: gfx1151 (RDNA3.5) with fallback to generic gfx1x

## Design

Mirror vLLM's dispatch logic:

```
sYT = ceil(M / (CuCount * 4))

if sYT <= 1:              return {YTILE=1, UNRL=4}
elif K%1024==512 && K>=1536 && (sYT>=40 || K>=4096): return {4, 1}
elif K < 1024:            return {2, 4}
elif K <= 2048 && (N>=2 || sYT<=26): return {1, 4}
elif N >= 2 && !fit_lds:  return {1, 4}
elif N == 1:              return {1, 2}
else:                     return {1, 1}
```

Implementation:
1. Template-parameterize `wvSplitKSml<N, YTILE, UNRL>` kernel
2. Add `SelectYtileUnrl()` host function with gfx1151 detection
3. Expand dispatch table to cover all YTILE/UNRL combinations
4. Fallback to `YTILE=2, UNRL=2` for non-gfx1151 architectures

## Expected impact

- **wvSplitKSml = 28% of decode time** on Qwen3.8-27B
- For typical decode shapes:
  - Small K (<1024): YTILE=2, UNRL=4 (more unrolling)
  - Medium K (1024-2048): YTILE=1, UNRL=4 (better occupancy)
  - Large K (>2048): YTILE=1, UNRL=1 or 2 (reduce register pressure)

## Evidence

- Kernel trace: `docs/bench-evidence/strix-kernel-trace-3015-20260907/`
- Upstream anchor: vLLM `csrc/rocm/skinny_gemms.cu:1240-1258`

## Owed

- gfx9 (wave64) wvSplitK tuning - separate port needed
- Formal benchmark comparing before/after on gfx1151
