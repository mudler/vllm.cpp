# NVFP4 model-owned device alpha (W3-F)

Status: **READY — spike accepted; implementation and all new gates PENDING**

Owning row: `KERNEL-GEMM-NVFP4-W4A4`

Claim: `CLAIM-NVFP4-SMALL-M-4`

This spike selects one bounded execution difference after W3-C made FP4 tactic
selection reproducible and the corrected W3-E component failed the strict
every-axis gate. It does not stack a repair into W3-E, claim an end-to-end
speedup, authorize the exact oracle grid, or change 35B W4A16 behavior. W3-F
only replaces the CUDA CUTLASS W4A4 adapter's per-GEMM host-scalar staging with
the model-owned device scalar used by vLLM and FlashInfer. The current staging
path remains a same-binary diagnostic fallback.

## Why this row is selected

The accepted traces and the complete source/dependency chain agree on one
executed mismatch:

- immutable local node trace SQLite `99cbd04d…93f8` contains **296,575 graph
  child `SetScalar` launches / 1,425 forward markers = 208.123 per forward**,
  totaling 207.124383 ms or 0.145350 ms/forward; its 23,514 eager calls include
  warmup/autotune and are not used for the steady-state normalization;
- the later W3-E direct trace SQLite `bb6c6fe…966` contains exactly **624
  `SetScalar` launches across three forwards = 208 per forward**, totaling
  1.005728 ms. This confirms the launch count after the 208-GEMM packed-QKV
  topology closed; the profiled duration is diagnostic, not an E2E prediction;
- binding vLLM v0.25 kernel summary `e4e916d1…565` and Torch-profiler trace
  `0c6f859f…0e2` contain **zero** `SetScalar` occurrences;
- local `MatmulNvfp4CutlassKernelCuda` obtains a stream scalar and launches
  `SetScalar<<<1,1>>>` before every GEMM, while vLLM constructs `layer.alpha`
  as a non-trainable device `Parameter` once and passes that tensor through
  FlashInfer to the CUTLASS `global_sf` pointer.

The 0.145 ms/forward clean graph slice is an upper bound, not a promised
throughput gain: graph scheduling, launch overlap and changed model-owned
memory can make the E2E result smaller or neutral. A frozen-plan same-binary
A/B remains the acceptance gate.

## Complete upstream and dependency chain

All vLLM paths are at v0.25.0 tag
`702f4814fe54fabff350d43cb753ae3e47c0c276`; installed dependency paths are
from FlashInfer 0.6.13 in the validated oracle environment.

| Concern | Exact upstream/dependency source | Contract to mirror |
|---|---|---|
| compressed-tensors load | `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:95-141` | take max shard divisors, reciprocate them, and retain `layer.alpha = input_global_scale * weight_global_scale` as a non-trainable device parameter |
| executed linear backend | `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:125-170` | the SM121 FlashInfer CUTLASS path passes `layer.alpha` unchanged with packed operands and swizzled scales |
| vLLM FlashInfer wrapper | `vllm/utils/flashinfer.py:780-816,819-854` | `alpha` is a tensor; if absent a one-element F32 tensor is allocated on the activation device, and the tensor is forwarded to `flashinfer.mm_fp4` |
| direct vLLM CUTLASS ABI | `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu:24-62`; `nvfp4_scaled_mm_sm120_kernels.cu:127-185,235-260` | require a CUDA F32 alpha tensor and assign `fusion_args.alpha_ptr = alpha.data_ptr()`; no per-call scalar kernel is launched |
| FlashInfer Python runner | installed `flashinfer/gemm/gemm_base.py:1307-1350` | retain the alpha tensor in the tunable input list and pass it directly to the compiled FP4 GEMM module |
| FlashInfer SM121 binding | installed `flashinfer/data/csrc/fp4_gemm_cutlass_sm120.cu:52-77,82-105,135-175` | validate one CUDA F32 `globalScale`, then pass `globalScale.data_ptr()` to the runner on the active stream |
| FlashInfer CUTLASS runner | installed `flashinfer/data/include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h:159-183` | forward `float const* global_sf` through the architecture dispatch into the epilogue |
| CUTLASS epilogue | installed CUTLASS `include/cutlass/epilogue/thread/linear_combination.h` and the local lifted tactic adapter | the epilogue dereferences a persistent device `alpha_ptr`; pointer identity may remain fixed across graph replay |
| executable specs | `tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py:42-168`, `test_nvfp4_scaled_mm.py:27-100`, `test_nvfp4_qutlass.py:202-268` | create alpha as a CUDA F32 tensor, cover small/padded shapes and non-unit values, consume it through the FP4 GEMM, and compare against dequantized reference output |

Relevant vLLM blob IDs are `c737b057…572ca` (CT scheme),
`b721a135…3d9` (FlashInfer linear backend), `3a45ede8…6e81` (SM120 stable
kernel), `a1e76d73…8cc` and `e7e16817…49e` (two GEMM tests). Installed
FlashInfer source SHA-256 values are `d9b3fd63…3c43` (runner),
`a1c4a0e3…6ee` (SM120 binding) and `446daaf3…b90e` (SM120 template).

The trace proves this chain executes on the 27B gate. Source availability alone
is not used as execution evidence.

## Current local chain and exact files to adapt

| Local concern | Current anchor | W3-F adaptation |
|---|---|---|
| model ownership | `include/vllm/model_executor/models/qwen3_5_weights.h:74-106,165-210`; `qwen3_5_dense.h:42-75` | add model-lifetime device-alpha ownership for individual, packed-QKV and merged-gate/up true-W4A4 linears; retain the exact host alpha values |
| resident construction | `src/vllm/model_executor/models/qwen3_5.cpp:492-709` | upload each immutable F32 scalar once on the model queue before first GEMM/capture and return a one-element device tensor alongside each resident operand |
| model dispatch | `src/vllm/model_executor/models/qwen3_5.cpp:1154-1264` | default CUTLASS W4A4 calls pass the resident alpha tensor; `VT_FP4_DEVICE_ALPHA=0` calls the old host-float overload |
| typed operation | `include/vt/ops.h:332-333,578-585`; `src/vt/ops.cpp:332-357` | add the upstream-shaped device-tensor overload, validate F32/one element/contiguous/same device, and retain an explicit host-scalar overload for the diagnostic arm |
| CUDA adapter | `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:76-120,914-968` | use the supplied tensor pointer directly; only the host fallback obtains stream scratch and launches `SetScalar` |
| CUTLASS tactics | `src/vt/cuda/nvfp4_cutlass_tactics.h:13-25`; `nvfp4_cutlass_tactic_impl.cuh:84-154` | unchanged: the existing `const float* alpha` ABI already matches upstream and accepts the model-owned pointer |
| tests | `tests/vt/test_ops_nvfp4_fp4.cpp:1140-1305,1340-1445,1540-1635` | port device-alpha shape/value/reference cases, compare device/host arms, cover graph replay and invalid tensor contracts, then retain forced-plan and packed-QKV coverage |

No quantization producer, tactic, on-disk layout, scheduler, KV cache,
attention, GDN, server API, W4A16 path, CPU kernel, or non-CUDA backend changes.

## Dispatch and fallback rules

1. The upstream-shaped public overload accepts a contiguous F32 tensor with
   exactly one element on the queue device. Rank zero or one is valid; shape is
   not otherwise interpreted.
2. Model-owned device alpha is eligible only for compiled CUDA CUTLASS true
   W4A4 execution. CPU/emulation and the 35B W4A16/Marlin path remain unchanged.
3. Every independent `Nvfp4Weight` owns its scalar for the same lifetime as its
   packed weight. Merged gate/up and packed QKV own their merged scalar next to
   their already-merged packed/scale buffers, using the exact existing max-before-
   reciprocal globals.
4. The one-element allocation and H2D copy occur on the engine queue during
   lazy resident initialization, which pre-serve warmup reaches before graph
   capture. The host source is model-owned for at least as long as the async
   copy; no stack-backed async source is allowed.
5. Default true-W4A4 model calls pass the resident tensor directly through
   `MatmulNvfp4Cutlass` to `LaunchParams::alpha`. The adapter launches no scalar
   producer and creates no alpha scratch entry in this arm.
6. `VT_FP4_DEVICE_ALPHA=0` selects the exact old host-float API in the same
   binary. That arm retains stream-scoped alpha allocation and one `SetScalar`
   per GEMM so topology and performance can be compared without a rebuild.
7. The host-float overload remains for focused compatibility tests and
   diagnostic callers; it is not the production true-W4A4 default.
8. Alpha value, packed/scale operands, selected frozen tactic, output dtype,
   GEMM count and token stream must be identical between arms. Only scalar
   ownership/staging may differ.

## Tests to port with the code

| Upstream executable specification | Required local port |
|---|---|
| `test_flashinfer_nvfp4_scaled_mm.py::test_flashinfer_nvfp4_gemm` | device F32 alpha over M=1/2/3 and aligned/padded shapes; device-alpha output matches the existing host-alpha and dequantized reference for every locally supported BF16 shape |
| `test_nvfp4_scaled_mm.py::test_nvfp4_gemm` | non-unit alpha derived from two global scales; direct tensor consumption preserves established CUTLASS tolerance |
| `test_nvfp4_qutlass.py::test_fused_quantization` / `test_llama_shapes` | one-element device alpha feeds the same packed operands and produces byte-equal BF16 output where the upstream case requires equality |
| stable-op input checks | reject non-F32, non-contiguous, zero/multiple-element and cross-device alpha before dispatch; accept rank-0/rank-1 one-element views |
| local capture and forced-plan cases | warmed device-alpha capture/replay is byte-identical; host fallback remains capture-safe; identical frozen/forced tactic produces identical output between arms |
| real model gates | default and `VT_FP4_DEVICE_ALPHA=0` 27B gates each pass 235/235 plus 16/16; 35B correctness-only proves W4A16 inertness |

Upstream FP16 output remains checked in as skipped under the existing W4
breadth leaf; W3-F does not silently claim that dtype.

## Correctness, safety and performance gates

### G1 — build/API and canonical record

- CPU Release and CUDA 13.0.88 sm_121a warning-as-error builds pass.
- Device-alpha validation and the host fallback compile in one binary.
- README, BENCHMARKS, roadmap, kernel/feature/quantization matrices, inventory,
  coordination, ledger and state advance together; record/doc mutation checks
  pass.

### G2 — operator and model correctness

- Device and host alpha arms match for non-unit alpha across upstream small,
  aligned, padded and real gate shapes under the same forced/frozen tactic.
- Device alpha remains valid through repeated eager calls and CUDA graph replay;
  no allocation or H2D copy occurs inside capture.
- Focused CUDA tests and real 27B default/fallback each pass. The 35B
  correctness-only gate proves the W4A16 route is inert; it is not a 35B
  performance authorization.

### G3 — safety and lifecycle

- Compute-sanitizer covers device-alpha initialization, eager use and graph
  replay with zero errors. A no-pool run must return all new allocations.
- The production model creates one 4-byte device scalar per executed logical
  FP4 linear (expected 208 for the packed topology, 832 bytes total), not per
  request or per forward. Model teardown frees them with the owning backend.
- No partial resident state is accepted for packed/scale/alpha tuples.

### G4 — structural trace

Profile the pushed default and fallback 27B model with `nsys --cuda-graph-trace=node`
and retain the exact vLLM Torch-profiler comparison:

- default steady-state forward markers contain **zero** local `SetScalar`;
- fallback retains approximately **208 `SetScalar` launches/forward**;
- vLLM remains zero; FP4 GEMM/producer counts and tactic IDs do not change;
- record eager/graph counts, kernel duration, output/lifecycle hashes and exact
  artifacts. Profiled throughput remains diagnostic.

### G5 — frozen-plan same-binary component A/B

On an idle DGX under one `/tmp/gpu` lock, run default device alpha versus
`VT_FP4_DEVICE_ALPHA=0` at cache-off input-1,024/output-128, greedy seed 0,
c2 and c16 in AB/BA/AB order. Reuse the accepted frozen 64-plan map; require
both model gates, exact request/token counts, lifecycle return, all 40 timing
and all 8 peak-memory axes. Mean-only or trace-only improvement is insufficient.

### G6 — conditional exact vLLM grid

Only if G1-G5 pass every axis, run the complete vLLM v0.25.0 27B
c1/2/4/8/16/32 three-repetition grid and paired node trace. All 124 axes still
bind. Do not run 35B performance until the complete 27B gate passes.

## Exact read-only baseline reproduction

```sh
ssh dgx.casa 'DB=~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27/ours.sqlite; sqlite3 -separator "|" "$DB" "SELECT CASE WHEN k.graphNodeId IS NULL THEN '\''eager'\'' ELSE '\''graph'\'' END, COUNT(*), ROUND(SUM(k.end-k.start)/1000000.0,6) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE instr(s.value,'\''SetScalar'\'') > 0 GROUP BY k.graphNodeId IS NULL ORDER BY 1;"'
ssh dgx.casa 'ROOT=~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27; python3 -c "import json; x=json.load(open(\"$ROOT/vllm-kernels.json\")); print(\"SetScalar\" in str(x))"'
```

Expected local rows are `eager|23514|23.263872` and
`graph|296575|207.124383`; the vLLM expression prints `False`. No GPU command is
required for this spike checkpoint.

## Work breakdown

| Work | Deliverable | State |
|---|---|---|
| W3-F0 | whole-chain execution/source/dependency audit, tests, dispatch and gates | **complete in this spike** |
| W3-F1 | tensor alpha operation/validation plus model-owned individual/merged residency and same-binary fallback | `READY` |
| W3-F2 | ported device-alpha/reference/capture/invalid-input tests and local build gates | `PENDING` |
| W3-F3 | immutable CUDA safety, 27B/35B correctness and paired node trace | `PENDING` |
| W3-F4 | frozen-plan c2/c16 AB/BA/AB every-axis component | `PENDING` |
| W3-F5 | exact v0.25 27B grid/trace, conditional on F4 passing every axis | `BLOCKED on F4` |

Any neutral or negative result is recorded honestly. A failed W3-F component
returns the parity loop to a fresh executed-path scan; it does not authorize
stacking another unmeasured lever or running 35B performance.
