# NVFP4 direct swizzled activation-scale emission (W3-E)

Status: **IMPLEMENTED / GATING; component strict gate FAILED (32/40 timing,
6/8 memory)**

Owning row: `KERNEL-GEMM-NVFP4-W4A4`

Claim: `CLAIM-NVFP4-SMALL-M-3`

This spike selects one bounded, executed difference after the immutable
`3f256ab` vLLM 0.25.0 27B grid failed strict parity at 55/124 axes. It does not
claim parity closure, 35B performance, or any broader quantization support.
The completed component shows small mean-throughput gains but fails the
every-axis gate, so it supplies no accepted speed credit. W3-E changes only how
the CUDA CUTLASS W4A4 path materializes dynamic
activation block scales: write the tensor-core swizzled layout directly in the
quant producer and retain the current linear-scale producer as the explicit
fallback.

## Why this row is selected

The generated-body audit refuted the apparent norm+FP4 fusion gap. Continuing
the same trace at kernel-body level exposes a real local-only operation:

- exact ours Nsight SQLite SHA `99cbd04d…93f8` contains **320,099**
  `SwizzleBlockscaleKernel` launches totaling **1.238881054 s** (23,524 eager /
  506.032544 ms and 296,575 graph-child / 732.848510 ms);
- normalized by 1,425 local forward markers, that is **224.631 launches and
  0.869390 ms of GPU kernel time per forward**;
- exact vLLM kernel summary SHA `e4e916d1…565` contains zero standalone
  `SwizzleBlockscaleKernel` / `swizzle_blockscale` launches;
- vLLM 0.25.0's executed `cvt_fp16_to_fp4` and
  `silu_mul_cvt_fp16_to_fp4` compute the tensor-core scale address before their
  `STG.8`, whereas our producers write `[M,K/16]` linearly and launch a second
  kernel to reorder it.

The clean local slice is launch/time attribution only. It is not an E2E gain:
CUDA-graph overlap, padded-scale initialization and changed producer occupancy
can reduce or erase the recovered time. Therefore same-binary A/B remains the
acceptance gate.

The same trace's FA2-vs-custom decode-attention difference is not selected.
After counting vLLM's main, split alternate and combine kernels, it exposes no
large missing family and its cross-profiler durations are not a valid speed
ratio. Attention remains available for later re-audit if a clean same-profiler
slice demonstrates a larger residual.

## Complete upstream and dependency chain

All paths below are relative to vLLM v0.25.0 tag
`702f4814fe54fabff350d43cb753ae3e47c0c276` unless noted.

| Concern | Exact upstream/dependency source | Contract to mirror |
|---|---|---|
| public allocation/layout mode | `vllm/_custom_ops.py:33-89,1580-1647` | `scaled_fp4_quant` supports swizzled and linear scale layouts; swizzled storage is padded to rows/128 and scale-cols/4 |
| stable custom-op allocation/dispatch | `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu:75-115,178-249` | the functional/out variants preserve explicit layout selection and dispatch the SM100/SM120 implementation |
| direct normal quant producer | `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu:41-105,178-249` | iterate padded rows/columns, use zero input outside the logical tensor, and pass the final scale address to the quant helper |
| swizzled address and FP8 scale write | `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh:168-206,244-289` | scale layout is `[numMTiles,numKTiles,32,4,4]`; the producer writes E4M3 directly to that byte offset and uses the same approximate-reciprocal NVFP4 math |
| fused SiLU producer | `csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu:40-116,120-163` | fused SiLU×up quant uses the same direct swizzled-scale address and packed-FP4 output contract |
| tests as executable spec | `tests/kernels/quantization/test_nvfp4_quant.py:15-287`, `test_nvfp4_scaled_mm.py:15-91`, `test_silu_mul_nvfp4_quant.py:16-73` | exact packed values/scales, swizzle recovery, padded rows/columns, linear fallback, fused producer, GEMM consumption and BF16/FP16 shape coverage |
| compressed-tensors consumer | `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_w4a4_nvfp4.py:95-138` and the selected CUTLASS linear backend | swizzled activation scales feed W4A4 CUTLASS; emulation retains linear scales |

The runtime trace proves the normal and fused producers execute on the 27B
gate. Source availability alone is not used as execution evidence.

## Current local chain and exact files to adapt

| Local concern | Current anchor | W3-E adaptation |
|---|---|---|
| typed op contract | `include/vt/ops.h:519-542`, `src/vt/ops.cpp:173-244` | add an explicit scale-layout enum/argument; validate either exact linear or exact padded-swizzled shape without ambiguous shape inference |
| CPU/reference implementation | `src/vt/cpu/cpu_ops.cpp` NVFP4 producers | preserve linear default; implement the direct offset/padding contract as the portable oracle |
| CUDA normal quant | `src/vt/cuda/cuda_matmul_nvfp4.cu:960-1037` | store `sc8` at the upstream swizzled byte offset when requested; cover every padded scale slot with zero in the same launch |
| CUDA fused producers | `src/vt/cuda/cuda_matmul_nvfp4.cu:1040-1185` | apply the same layout mode to two-input and one-input SiLU+quant, preserving their BF16 rounding boundary |
| standalone fallback | `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:552-584` | retain `SwizzleBlockscaleKernel` for linear compatibility and `VT_FP4_DIRECT_SF=0`; do not delete the public op |
| model dispatch | `src/vllm/model_executor/models/qwen3_5.cpp:1132-1253,1308-1338,3300-3373` | CUTLASS true-W4A4 producers allocate padded swizzled scales and pass them directly; emulation/CPU/non-W4A4 paths remain linear |
| tests | `tests/vt/test_ops_nvfp4_fp4.cpp:351-825,935-1042` | port the upstream padded/layout cases and require direct scales/packed FP4/GEMM output to match linear+standalone-swizzle |

No loader, on-disk weight layout, tactic family, KV cache, scheduler, serving
API, 35B W4A16 path, or non-CUDA backend selection changes.

## Dispatch and fallback rules

1. Add `Fp4ScaleLayout::{kLinear,kCutlassSwizzled}` to the typed producer API.
   Existing callers default explicitly to `kLinear`; no output-shape heuristic
   is allowed because aligned tensors can have identical apparent dimensions.
2. W3-E is eligible only when all are true: CUDA, compiled CUTLASS NVFP4,
   `NvfpCutlassEnabled()`, true compressed-tensors W4A4 activation quantization,
   and `VT_FP4_DIRECT_SF` is not `0`.
3. Eligible producers allocate the scale buffer as raw E4M3 bytes with logical
   shape `[round_up(M,128), round_up(K/16,4)]` and hand it directly to
   `MatmulNvfp4Cutlass`.
4. The direct kernel visits logical and padded scale positions. Padded rows or
   columns write E4M3 zero and never read input or write packed FP4. This makes
   correctness independent of allocator contents and avoids adding a separate
   memset launch. The result must equal zero-initialized upstream storage.
5. `VT_FP4_DIRECT_SF=0` restores the exact current sequence:
   linear producer → `SwizzleBlockscale` → CUTLASS GEMM. CPU, emulation and any
   unsupported backend always use that linear contract.
6. Do not remove `VT_SWIZZLE_IN_QUANT`: it remains the older swizzle-once
   diagnostic for split/shared callers. W3-E subsumes it only on direct-eligible
   production sites; compatibility behavior remains testable.
7. The packed FP4 bytes, recovered logical scales, CUTLASS output dtype and
   token stream must not change between arms.

## Tests to port with the code

| Upstream executable specification | Required local port |
|---|---|
| `test_nvfp4_quant.py::test_quantize_to_fp4` | BF16 and the locally supported F32 producer over aligned M/K; direct packed bytes and recovered scales equal the linear reference |
| `test_nvfp4_quant.py::test_python_util_matches_cpp_allocation` | exact direct scale shapes for `(1,64)`, `(32,4096)`, `(127,1024)`, `(128,4096)`, `(256,16384)` |
| `test_nvfp4_quant.py::test_quantize_to_fp4_with_padded_output` | padded M and scale columns are byte-zero; valid packed/scales match linear+swizzle |
| `test_nvfp4_quant.py::test_quantize_to_fp4_padded{,_no_sf_swizzled}` | port every listed padded shape relevant to our no-K-padding API, including `(32,14336)`; keep the linear arm checked |
| `test_nvfp4_scaled_mm.py::test_nvfp4_gemm` | direct scales consumed by `MatmulNvfp4Cutlass` match the existing swizzled-composition output/tolerance |
| `test_silu_mul_nvfp4_quant.py::test_silu_mul_nvfp4_quant` | both local fused SiLU producer forms have direct-vs-composed packed/scale/GEMM parity |
| local QKV/gate-up tests | packed QKV and merged gate-up direct/fallback outputs are byte-identical before their GEMMs; `VT_FP4_MERGED_QKV=0` remains independent |

Checked-in skipped coverage is required for upstream FP16 until the existing W4
breadth leaf supplies FP16. No upstream case is silently dropped.

## Correctness, safety and performance gates

### G1 — build/API

- CPU Release and CUDA 13.0.88 `sm_121a` warning-as-error builds pass.
- Existing linear callers compile unchanged in behavior.
- Record, documentation checkpoint and mutation tests pass.

### G2 — operator and model correctness

- Direct scale bytes equal `SwizzleBlockscale(linear)` for every ported shape;
  packed FP4 bytes are identical.
- CUTLASS direct/fallback GEMM outputs are byte-identical where the existing
  composition is byte-identical and otherwise meet its established tolerance.
- Focused compute-sanitizer covers M=1, M=127, padded rows/columns, real
  K=5,120/17,408 and packed QKV N=14,336 with zero errors/leaks.
- Real 27B default and `VT_FP4_DIRECT_SF=0` each pass 235/235 plus the committed
  16/16 token gate. Run the 35B token gate to prove the W4A16 path is inert; it
  is not a 35B performance authorization.

### G3 — structural trace

- Same-workload node tracing must show no hot-path standalone
  `SwizzleBlockscaleKernel` for the direct arm and the expected retained calls
  in the fallback arm.
- Direct producer counts replace, not duplicate, the old producer+swizzle
  chain. FP4 GEMM/forward topology remains 208.
- Record graph/eager counts, total kernel time, output digest, lifecycle and
  exact artifact hashes. Profiled throughput remains non-binding.

### G4 — same-binary component A/B

On an idle DGX under one `/tmp/gpu` lock, run direct vs
`VT_FP4_DIRECT_SF=0` in AB/BA/AB order at cache-off input-1,024/output-128,
greedy seed 0, at c2 and c16. Each arm gets three retained repetitions, exact
native output counts, memory return and the standard 20 timing + 4 memory
component axes. Acceptance requires reproducible non-regression on every axis;
mean-only or trace-only improvement is insufficient.

**Result at immutable pushed `53ab1492983282a9858cc301d4f7e9aad4784c48`:**
the one-lock AB/BA/AB series completed all 12 legs, 612/612 requests, 12/12
memory returns and both 235/235 model gates with zero lazy plan-cache misses.
At c2, direct/fallback mean total throughput is
**150.116922/149.801191 = 1.002107665x**, with **16/20 timing + 4/4 memory**
axes. At c16 it is **796.834440/791.907102 = 1.006222116x**, with **16/20
timing + 2/4 memory** axes. The combined disposition is therefore **32/40
timing + 6/8 memory: FAILED**. c2 misses median TPOT, p90 TPOT, p90 TTFT and
p99 TTFT at 0.998753/0.998498/0.975935/0.993449x. c16 misses mean TTFT, p90
TPOT, p99 E2EL and p99 TTFT at
0.995423/0.994162/0.997763/0.966681x; PSS/RSS also miss at
0.999435/0.999435x. Five of six paired 128-token generated-text hashes differ,
while the fixed oracle gates remain exact. Only 18--33 of 64 paired tactic IDs
match and only 9--17 keys per arm retain one ID across all three c16 runs, so
plan selection is recorded as a confounder rather than assigned as the cause.
Summary/selection SHA-256 are `cfff5711...50e9` / `ceaa5296...47b4`.

W3-C3R later resolves the text-hash observation without changing this failed
performance disposition. With one frozen 64-plan map, direct and fallback are
**6/6 x 128-token equal** when requests are sent sequentially with identical
batch shape, while each arm is **0/6** equal to its own earlier c2 output.
Production-default vLLM v0.25.0, with `VLLM_BATCH_INVARIANT` unset, is likewise
**0/6** equal between sequential and c2 execution. Cross-run online text hashes
are therefore diagnostic, not a correctness predicate; the component still
fails independently at **32/40 timing + 6/8 memory** and receives no speed
credit. Reclassification summary SHA is `a1c500b3...41de`.

### G5 — oracle grid

If G1-G4 pass, run the full immutable vLLM v0.25.0 27B c1/2/4/8/16/32
three-repetition grid and paired trace. The project gate remains every one of
124 axes at or above/below the oracle as appropriate. W3-E cannot close the row
by itself while any axis fails. Do not run the 35B performance grid first.

G4 failed, so **G5 was not run**. Immutable `3f256ab` remains the binding
vLLM result at 55/124; no 35B performance command was authorized.

## Exact reproduction checkpoint

Current structural baseline (read-only, no GPU execution):

```sh
ssh dgx.casa 'DB=~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27/ours.sqlite; sqlite3 -separator "|" "$DB" "SELECT CASE WHEN k.graphNodeId IS NULL THEN '\''eager'\'' ELSE '\''graph'\'' END, COUNT(*), ROUND(SUM(k.end-k.start)/1000000.0,6), ROUND(AVG(k.end-k.start)/1000.0,3) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE instr(s.value,'\''SwizzleBlockscaleKernel'\'') > 0 GROUP BY k.graphNodeId IS NULL ORDER BY 1;"'
```

Expected baseline rows are `eager|23524|506.032544|21.511` and
`graph|296575|732.84851|2.471`. The completed component evidence is
`~/work/vllm.cpp-direct-sf/53ab1492983282a9858cc301d4f7e9aad4784c48/component-ab-c2-c16`.
The original retained driver SHA is `7d81edb2...0640`; it executed the correct
committed online-gate request counts but its temporary summarizer incorrectly
asserted 96 requests at c2 and failed closed after all legs. The corrected
summary-only driver SHA `828ec34e...a63d` uses the committed c2=6/c16=96
contract and performs no GPU/model execution. Reproduce only the summary with:

```sh
SUMMARY_ONLY=1 ~/work/vllm.cpp-direct-sf/53ab1492983282a9858cc301d4f7e9aad4784c48/summary-driver-corrected.sh
```

## Work breakdown

| Work | Deliverable | State |
|---|---|---|
| W3-E0 | whole-chain source/body audit, upstream tests, dispatch, files and gates | **complete in this spike** |
| W3-E1 | explicit layout API + CPU/direct CUDA normal quant + ported padded/layout tests | **complete** |
| W3-E2 | fused producer layouts + true-W4A4 model dispatch + fallback + model tests | **complete** |
| W3-E3 | sanitizer, real-model gates, node trace and c2/c16 same-binary A/B | **complete; strict A/B failed at 32/40 timing + 6/8 memory** |
| W3-E4 | exact v0.25 27B grid/trace and lifecycle classification | **not run because E3 failed** |

Any negative or neutral result is recorded honestly. A failed W3-E component
does not permit stacking another unmeasured lever into the same A/B.
