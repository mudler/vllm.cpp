ID: ISSUE-GH-1867
Title: **The DFlash2 selector's per-step top-k cost 683 us/step where FlashInfer's radix top-k does the same work in 40 us** -- `TopKValuesIndicesRowKernel`, 8 rows x 248320 vocab, K=16, measured on `dgx:gpu0` with nsys against SGLang on the identical checkpoint and workload (#1857's kernel table, the artifact-verified re-take): +0.65 ms/step, the fourth-largest per-step lever there. The cost was the ITERATION COUNT -- a ternary bisection of the threshold in float VALUE space under `kThreshMaxIter = 64`, every iteration a full pass over a 248320-wide row -- where a radix narrowing over a monotone key fixes the same threshold EXACTLY in four rounds, two of which read global memory at all. W12 ports the arithmetic (`include/vt/radix_topk.h`, anchored on `flashinfer/topk_common.cuh:35-39` and `flashinfer/topk.cuh:683-691` at FlashInfer `0.6.12`, the wheel vLLM's own `_topk` dispatches to at merge `b389ac29`) and rewrites the CUDA arm around it as `TopKValuesIndicesRadixRowKernel`; the multi-CTA grid barrier and workspace `## Risks/decisions` D2 refused stay refused, ONE CTA PER ROW. **The tie-break does not move**: upstream leaves FlashInfer's `tie_break` at `NONE`, ours is index-ascending and `include/vt/ops.h` pins it, so the port mirrors FlashInfer's algorithm and our contract -- which is FlashInfer's own `TopKTieBreak::Small`. The CPU reference is UNCHANGED, so the two arms still answer by different routes. Gated on a host with no `nvcc` by `tests/vt/test_ops_radix_topk` against a full stable sort, including on the production shape. **The GPU number and the device run are OWED** (`## Owed` O34, operator-run, `-DVLLM_CPP_CUTLASS_FETCH=ON` plus an `nm` assertion on the new kernel name before any timing); occupancy is the named residual (O35). Nothing here claims a measured speedup
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: perf
GitHub: 1867
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:730`

### Frozen archive evidence

> | [#1867](https://github.com/mudler/vllm.cpp/issues/1867) | `SPEC-DFLASH2` | **The DFlash2 selector's per-step top-k cost 683 us/step where FlashInfer's radix top-k does the same work in 40 us** -- `TopKValuesIndicesRowKernel`, 8 rows x 248320 vocab, K=16, measured on `dgx:gpu0` with nsys against SGLang on the identical checkpoint and workload (#1857's kernel table, the artifact-verified re-take): +0.65 ms/step, the fourth-largest per-step lever there. The cost was the ITERATION COUNT -- a ternary bisection of the threshold in float VALUE space under `kThreshMaxIter = 64`, every iteration a full pass over a 248320-wide row -- where a radix narrowing over a monotone key fixes the same threshold EXACTLY in four rounds, two of which read global memory at all. W12 ports the arithmetic (`include/vt/radix_topk.h`, anchored on `flashinfer/topk_common.cuh:35-39` and `flashinfer/topk.cuh:683-691` at FlashInfer `0.6.12`, the wheel vLLM's own `_topk` dispatches to at merge `b389ac29`) and rewrites the CUDA arm around it as `TopKValuesIndicesRadixRowKernel`; the multi-CTA grid barrier and workspace `## Risks/decisions` D2 refused stay refused, ONE CTA PER ROW. **The tie-break does not move**: upstream leaves FlashInfer's `tie_break` at `NONE`, ours is index-ascending and `include/vt/ops.h` pins it, so the port mirrors FlashInfer's algorithm and our contract -- which is FlashInfer's own `TopKTieBreak::Small`. The CPU reference is UNCHANGED, so the two arms still answer by different routes. Gated on a host with no `nvcc` by `tests/vt/test_ops_radix_topk` against a full stable sort, including on the production shape. **The GPU number and the device run are OWED** (`## Owed` O34, operator-run, `-DVLLM_CPP_CUTLASS_FETCH=ON` plus an `nm` assertion on the new kernel name before any timing); occupancy is the named residual (O35). Nothing here claims a measured speedup | perf |

## Resolution

-
