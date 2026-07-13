# NVFP4 BF16 normal-producer vectorized I/O (W3-H)

Status: **ACTIVE — H1a/H1b/H1c are VOID; H1d collector and semantic-evidence
hardening is pending and implementation remains prohibited**

Owning row: `KERNEL-GEMM-NVFP4-W4A4`

Claim: `CLAIM-SERVE-GATE-1`

This spike selects one bounded CUDA execution difference after W3-G's
correctness-faithful FA2 path strict-failed its c2/c16 component. It does not
claim an end-to-end win, change the binding `3f256ab` result of 55/124 axes,
authorize an exact grid or authorize any 35B performance command. W3-H is only
the **normal** BF16 activation-to-NVFP4 producer used by the 27B direct-swizzled
CUTLASS W4A4 path. The separate fused SiLU producer, FP4 GEMMs, GDN, attention,
scheduler and host-weight lifetime are excluded.

H1 has now completed two paired attempts from immutable `5d8af792`, but neither
is binding. H1a used a writable native tactic cache and changed 37/64 selected
plans. H1b restored the exact read-only 64-plan fixture and completed every
workload/lifecycle arm, but Nsight reported dropped CUDA events and the dominant
graph had uneven node replay counts. Immutable H1c at `d1f8e33` then passed
the 27B model gate in 19.20 seconds and completed 48/48 capture-1 requests, but
Nsight emitted severity-2 `Not all CUDA events might have been collected`
after reporting 818,537 collected CUDA events. The fail-closed driver stopped
before captures 2/3 and vLLM. Its retained process log has zero plan
lifecycle/selection records, so the exact frozen-map contract is not
recoverable post-hoc. No runtime implementation or speed credit follows from
any of the three void traces.

The first implementation leaf is intentionally narrower than vLLM's whole
kernel: one aligned 256-bit BF16 load and one packed 64-bit FP4 store while
retaining the current scalar max, scale, reciprocal selection, E2M1 bucket
semantics, flat launch geometry and padding behavior. This isolates I/O and
instruction-count changes from arithmetic and scheduling changes. A scalar
same-binary fallback remains present, and the candidate begins default-off.

## Scope

### In scope

- CUDA `DType::kBF16` plus `Fp4ScaleLayout::kCutlassSwizzled` normal
  `ScaledFp4Quant` on CUDA 12.9+ and compute capability 100--129.
- The Qwen3.6-27B true-W4A4 production shapes, especially the 80 K=5,120 and
  64 K=6,144 normal producers executed per decode forward.
- A process-cached `VT_FP4_QUANT_VEC=1` candidate and exact scalar fallback;
  after every gate passes, unset may become vectorized and `=0` becomes the
  rollback. Until then unset and `=0` both select scalar. No environment or
  device-capability query may execute in the per-producer hot call.
- Aligned 32-byte BF16 input loads, unchanged scalar arithmetic over the loaded
  values, unchanged scale-byte store, and one little-endian 64-bit packed-FP4
  store.
- Exact output-byte, padding, capture/replay, SASS, counter, sanitizer,
  real-model, structural-trace and strict serving gates.

### Out of scope

- `SiluAndMulFp4QuantKernel`, `SiluMulFp4QuantKernel`, their model dispatch or
  their diagnostic 0.154-ms/decode-forward opportunity.
- Changing exact division to vLLM's `rcp.approx.ftz`, changing FP4 bucket
  semantics, changing the two-pass arithmetic order, or combining arithmetic
  and I/O into one acceptance result.
- vLLM's occupancy-capped 2D grid, block-size selection, FP16 support,
  `padded_n` logical/physical output-width split, CUDA 12.8 128-bit path,
  expert quantization, UE8M0/MXFP4 or non-CUDA backends.
- FP4 GEMM tactic changes, direct-scale layout changes, model-owned alpha,
  packed projection topology, FA2/GDN/scheduler/KV changes, host-weight release,
  prefix caching, SGLang or 35B performance.

These exclusions are independent leaves, not statements that the upstream
modes are unnecessary. Feature-parity policy still requires them to be
inventoried and spiked separately.

## Why W3-H is selected

### H1 attempts and lossless-trace contract

H1a at
`~/work/vllm.cpp-executed-path-refresh/5d8af792a0010434fa9681a9cf46b6a5cdbfc77b/evidence/5d8af792a0010434fa9681a9cf46b6a5cdbfc77b`
is **VOID**. It ran the tactic cache read-write, loaded/saved 64 native plans and
selected 37/64 tactics differently from the accepted v0.25 fixture, including
M=16/N=34,816/K=5,120 tactic 4 instead of tactic 14. Its Nsight report SHA is
`471e30d620236789cfafa6d5ca52a98272e0e93161c3aafe29262ec4f7deb7fb`;
a later SQLite audit also finds the CUDA-event-loss warning. The corrected
manual plan validator passes; an earlier invocation without the required
`PYTHONPATH` and a separate fail-closed precreated-evidence dry-run are setup
diagnostics, not benchmarks.

H1b at
`~/work/vllm.cpp-executed-path-refresh-frozen-h1b/5d8af792a0010434fa9681a9cf46b6a5cdbfc77b/evidence/5d8af792a0010434fa9681a9cf46b6a5cdbfc77b`
is also **VOID**. It loaded 64 FlashInfer and zero native plans in read-only
mode, tuned/rejected/saved zero, reproduced selected-plan SHA
`f2d9be7fc4a89de1cfa994ab9be08a423e0c4f6981fe46cb808cef485f4c1fa4`,
and left the forbidden native-cache target absent. The model gate, three local
48/48-request legs, vLLM Torch trace and all cache drops completed. Ours Nsight
report/SQLite SHA are `a76a6ed30239eab52587700f73267ab9be70788ae2f012761c5637228c7bfd11`
and `b6dcd5d69900010d953d817db612083ffc92d94b17177b9348f8d99fc0485165`.
The SQLite contains severity-2 `Not all CUDA events might have been collected.`
Its dominant 1,107-node graph has 930 nodes replayed 1,372 times and 177 nodes
replayed 1,373 times, proving at least 930 missing graph-node events. The old
status `passed:true` did not inspect SQLite and is a historical harness false
positive.

Retained H1b events diagnose normal production at 0.638331 versus 0.342777
ms/decode forward, fused production at about 0.543321 versus 0.257276 ms, and
the frozen FP4 GEMM set at about 54.676 versus 54.792 ms. These cross-profiler
values cannot bind or authorize H2; they only leave W3-H as the leading
eligible candidate because no verified larger residual displaced it.

H1c replaces the single long local profile with three independent Nsight
reports, one 48-request repetition each, while retaining one uninterrupted GPU
lock for the series. Each capture uses node-level CUDA-graph tracing, a
10,000-ms CUDA flush interval, no CPU context-switch sampling and an 11-second
final idle drain. Before the next capture or vLLM arm, the harness exports and
hashes SQLite, rejects every non-whitelisted severity>=2 diagnostic, requires
graph-node rows and requires uniform replay counts across the dominant graph.
All three reports, SQLite files, validation reports, commands, logs and kernel
summaries are separately hashed. Every capture must contain exactly 1,107
dominant-graph nodes, including 208 FP4 GEMMs, 144 normal producers, 64 fused
producers, 48 GDN recurrence nodes and 16 FA2 main plus 16 combine nodes. H1c
remains **PENDING** until that contract runs from a clean pushed immutable SHA.

### Executed topology and clean diagnostic slice

Current immutable local evidence is the W3-G default node trace at
`~/work/vllm.cpp-fa2-decode/ae9e8ff0576badabdda7289beeacaa1041c55d21/evidence/trace`.
Its SQLite SHA-256 is
`6b9a0ddbbf33b2f3b152f5d1f95428958ebd04d16cf28d968dde589627a766dc`.
The normal producer summary SHA-256 is
`910bd8df71118c2c2f7081c9f4f70c6a697c1316a904ee37636f6b8497d956f7`.
Graph-node rows, excluding eager/capture work, contain:

| Normal shape | Current geometry | Calls / graph forwards | Mean | Per-forward time |
|---|---|---:|---:|---:|
| K=5,120 | grid 160, block 256 | 1,120 / 14 = 80 | 4.349314 us | 0.347945 ms |
| K=6,144 | grid 192, block 256 | 896 / 14 = 64 | 4.374821 us | 0.279988 ms |
| **total** | | **144 / forward** | | **0.627934 ms** |

The vLLM v0.25.0 raw Torch-profiler trace is
`~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27/vllm-profile/online-gate_rank0.1783906481468807023.pt.trace.json.gz`,
SHA-256
`0c6f859f916ceb15970f3f6896c84bc37f4e2152c8aedef2c7b893b40e6ee0e2`.
Geometry separates 1,484 decode forwards from 104 prefill forwards:

| Normal shape | vLLM decode geometry | Decode calls | Mean | Per-forward time |
|---|---|---:|---:|---:|
| K=5,120 | grid 128, block 320 | 118,720 = 80 x 1,484 | 2.297265 us | 0.183781 ms |
| K=6,144 | grid 128, block 384 | 94,976 = 64 x 1,484 | 2.496197 us | 0.159757 ms |
| **total** | | **144 / forward** | | **0.343538 ms** |

The clean diagnostic difference is therefore **0.284396 ms/decode forward**.
The aggregate vLLM value of 2,189.809 ms / 1,588 forwards is invalid for this
comparison because it includes exactly 8,320 + 6,656 calls from the 104 large
prefill forwards. The profilers and capture windows still differ, so 0.284396
ms is a candidate ceiling, not a binding ratio or accepted speed credit.

Padding is not the explanation: both decode geometries launch the same total
threads, 40,960 for K=5,120 and 49,152 for K=6,144. Both traced kernels use 30
registers/thread and no shared memory. The separate fused producer is excluded
from every W3-H estimate.

### Executed SASS difference

The current source blob is identical at `b96fc33`, `ae9e8ff` and `7517af4`:
`4441e63b86d6d203c6292d5e1d73a9f48c1373a0`. The clean W3-G object
`build-cuda/CMakeFiles/vllm.dir/src/vt/cuda/cuda_matmul_nvfp4.cu.o` has
SHA-256 `843a42aeda711a937f5e2151f47781f270fb6027e241ac76c1765d15aad247b9`.
`cuobjdump --dump-sass` of the BF16/swizzled normal symbol shows a 1,056-
instruction function with 32 `LDG.E.U16` instructions, one scale-byte plus
eight packed-byte stores on a valid group, and no 256-bit load or 64-bit store.
The 32 loads are two complete reads of the same 16 BF16 values: once for max,
again for packing.

The executed vLLM symbol lives in
`~/venvs/vllm-oracle-v0.25.0-stage/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so`,
SHA-256 `56c647dd3130e097dcd44e219b639ae76b3c4618d8d679235798f2ce94193df4`.
Its 160-instruction SASS has one predicated 256-bit load, eight packed E2M1
conversions, one scale-byte store and one 64-bit packed store. The requested
BF16 input-load model over 80 x K5,120 + 64 x K6,144 at batch 16 is about
49.0 MiB locally versus 24.5 MiB upstream; both write about 12.25 MiB. Cache
may hide part of that byte difference, so counters and same-binary timing are
mandatory.

### Defensible ceiling

The observed normal-producer difference is too small to close parity alone.
Even a zero-cost local normal producer can remove at most 0.627934 ms/decode
forward, and the observed vLLM-shaped target removes about 0.284396 ms. Binding
TPOT/ITL gaps are larger on almost every point. W3-H is one ranked component,
not a claim that the remaining gap is a ceiling. A fresh full-workload trace
must continue ranking FP4 GEMM, GDN post-convolution/recurrence, fused SiLU and
the rest of the executed path.

## Prior negative experiment and why this is not a repeat

Commit `f787cf8` on `origin/perf/hw-fp4-quant` implemented
`VT_HW_FP4_QUANT` with vector loads, hardware E2M1 conversion, approximate
reciprocal and both normal and fused producers on the older linear-scale path.
Its normal helpers/dispatch are
`f787cf8:src/vt/cuda/cuda_matmul_nvfp4.cu:947-1110,1158-1209` and its fused
body is `:1281-1343`. The old normal launch used flat `m * (K / 16)` geometry
(B16 grids 20/24), not today's direct padded grids 160/192, and it performed
an uncached `getenv` in the per-call path.
On the historical input-1,024/output-8, c16/48 workload it measured
1,584/1,573 tok/s versus 1,585/1,576 tok/s and was correctly classified
**NEUTRAL / SHELVED**. It also produced 2,468/37,888 packed-byte mismatches
(6.5%) versus the exact-division ladder, attributed to the approximate
reciprocal.

That result remains a binding negative constraint. W3-H does not cherry-pick
it away or copy the old branch wholesale. The current production path now has
direct swizzled scales, packed QKV, frozen tactics, a different v0.25 oracle
and a decode-heavy online gate. W3-H isolates only normal-producer I/O and
requires byte identity. If the isolated candidate is neutral, the work stops;
upstream arithmetic, fused production or 2D geometry may not be added silently
to manufacture a combined result.

## Complete upstream and dependency chain

All vLLM paths below are at v0.25.0 tag
`702f4814fe54fabff350d43cb753ae3e47c0c276`.

| Concern | Exact upstream/dependency source | Contract to mirror |
|---|---|---|
| FlashInfer-CUTLASS eligibility | `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:97-123` | require supported FlashInfer/CUTLASS FP4 and capability >=100 |
| activation quant dispatch | `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:135-170` | unquantized input is dynamically quantized with swizzled scales before the CUTLASS GEMM |
| public allocation/out call | `vllm/_custom_ops.py:33-92,1492-1563` | allocate packed FP4 and physical swizzled-scale bytes, support padded output, dispatch the stable op |
| stable SM dispatch | `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu:63-117` | compiled SM100--119 or SM120--129 path, functional and out variants |
| normal kernel and launch | `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu:38-107,178-250` | predicated packed load, all padded scale slots, quant helper, packed store, occupancy-aware 2D launch |
| vector/load primitives | `csrc/cuda_vec_utils.cuh:20-39,123-132,264-288` | aligned packed-vector representation and predicated 256-bit load |
| quant helper/layout | `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh:25-36,118-200,219-301` | CUDA>=12.9 16-element thread, BF16x2 max, exact swizzled address, E4M3 scale, reciprocal and packed E2M1 |
| separate fused producer | `csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu:38-162` | inventoried but excluded from W3-H |
| upstream tests | `tests/kernels/quantization/test_nvfp4_quant.py:11-308`; `test_nvfp4_scaled_mm.py:17-100`; `test_silu_mul_nvfp4_quant.py:16-78` | capability/dtype/shape/layout/padding, GEMM consumption and separate fused behavior |
| determinism breadth | `tests/v1/determinism/test_nvfp4_batch_invariant.py:27-100`; `test_nvfp4_scaled_mm.py:25-100` | opt-in batch-invariant mode is a separate inventoried feature, not production-default W3-H |

The `backend="flashinfer-cutlass"` label does not move activation quantization
into FlashInfer: `_custom_ops.py` invokes the vLLM stable custom op. The runtime
trace names `vllm::cvt_fp16_to_fp4<__nv_bfloat16,false>`, and the oracle log
selects `FlashInferCutlassNvFp4LinearKernel`; therefore the source and executed
binary chain agree.

## Current local chain and port map

| Local concern | Current anchor | W3-H change |
|---|---|---|
| public validation | `src/vt/ops.cpp:173-203` | no contract change; preserve rank-2, K%16, contiguity and exact output shapes |
| scale address | `src/vt/cuda/cuda_matmul_nvfp4.cu:958-981` | preserve byte-for-byte |
| scalar normal body | `src/vt/cuda/cuda_matmul_nvfp4.cu:983-1037` | retain as fallback and arithmetic reference |
| normal launcher | `src/vt/cuda/cuda_matmul_nvfp4.cu:1039-1079` | add host eligibility/toggle and launch the BF16/swizzled vector body only when safe |
| direct-scale/model eligibility | `src/vllm/model_executor/models/qwen3_5.cpp:1026-1078,1253-1324,1394-1505` | no model topology change; existing direct normal calls select through the op launcher |
| separate fused producer | `src/vt/cuda/cuda_matmul_nvfp4.cu:1201-1306`; `src/vllm/model_executor/models/qwen3_5.cpp:3465-3584` | unchanged and explicitly excluded |
| tests | `tests/vt/test_ops_nvfp4_fp4.cpp:374-476,628-728` | add OFF/ON bytes, eligibility, capture, boundary and exact real-shape cases |

No header/API, loader, on-disk format, weight layout, GEMM tactic, scale layout,
model graph, serving API or non-CUDA backend changes.

## Dispatch and fallback rules

1. `VT_FP4_QUANT_VEC=1` and the device capability are resolved once per process
   before model execution, then passed through a direct internal dispatch seam.
   Initially unset and `=0` select the scalar kernel. The two kernels are
   compiled into the same binary. Tests and microbenchmarks use the internal
   seam to exercise OFF/ON in one process; they do not mutate or reread the
   environment between hot calls.
2. The candidate is eligible only for BF16, direct CUTLASS-swizzled scales,
   CUDA toolkit >=12.9, runtime capability 100--129, 32-byte-aligned input and
   8-byte-aligned packed output. Any failed predicate uses scalar without
   changing output.
3. Keep block 256 and the flat one-thread-per-padded-scale-slot mapping for the
   first component. Do not combine 2D geometry with I/O vectorization.
4. One aligned 256-bit load obtains all 16 BF16 values. Those exact bits feed
   the existing scalar max/scale/reciprocal/nibble operations in the same order.
   Combine the same eight bytes in little-endian order and issue one 64-bit
   store. The scale byte and padded-zero branches remain unchanged.
5. Both the default exact reciprocal and opt-in `VT_NVFP4_FP4_NATIVE=1`
   approximate reciprocal modes must be byte-identical between vector and
   scalar. W3-H does not choose between those modes.
6. Captured graphs bind the host-selected kernel. Replay performs no allocation,
   free, synchronization or host/device copy.
7. If H1--H4 all pass, unset becomes vectorized for the eligible slice and
   `VT_FP4_QUANT_VEC=0` is the exact rollback. If any strict axis fails, default
   remains scalar and W3-H receives no speed credit.

## Tests to port with the code

| Upstream executable specification | Required local port/disposition |
|---|---|
| `test_nvfp4_quant.py::test_quantize_to_fp4` | BF16 core/reference shapes plus exact OFF/ON packed and scale bytes; FP16 checked in as skipped/tracked because the local op lacks FP16 |
| `test_python_util_matches_cpp_allocation` | retain exact direct/linear physical shape tests; no allocation behavior change |
| `test_quantize_to_fp4_with_padded_output` | **SKIPPED/tracked:** local op has no distinct logical K versus physical `padded_n`; do not claim the existing scale-padding test ports packed-output padding |
| `test_quantize_to_fp4_padded{,_no_sf_swizzled}` | retain padded M/K, direct/linear and every padded scale byte; W3-H changes only BF16/direct eligibility |
| `test_nvfp4_scaled_mm.py::test_nvfp4_gemm` | unchanged downstream CUTLASS output and frozen tactics for candidate/fallback |
| `test_silu_mul_nvfp4_quant.py::test_silu_mul_nvfp4_quant` | existing stronger fused tests remain green; kernel/count must be unchanged |
| batch-invariant tests | remain tracked under `ENG-BATCH-INVARIANT`; production-default token gates are required here |

Local W3-H cases cover BF16 M=`1,2,8,16,17,32,127,128,129,2048` and
K=`64,5120,6144,14336,17408`; threshold-adjacent finite values, `+0/-0`,
extrema and the existing NaN policy; misaligned input/output fallback; both
reciprocal modes; and cold capture plus two replays. Candidate/fallback packed
bytes and every padded scale byte must match exactly.

## Gates

### G0 — fresh exact-workload executed-path refresh

Before implementation, run current pushed source and the validated vLLM
v0.25.0 oracle under one `/tmp/gpu` lock using the exact 27B cache-off c16,
input-1,024/output-128, 48-prompt x three-repetition contract in
`scripts/dgx-online-serving.sh --trace-only`. Ours uses **three independent**
Nsight Systems captures, one repetition each. H1c used
`--cuda-graph-trace=node --cuda-flush-interval=10000 --sample=none
--cpuctxsw=none` plus a final 11-second drain and still lost events, so those
flags are now a failed constraint rather than an H1d recipe. H1d must retain
lossless node-level attribution while bounding the capture range and/or trace
buffers enough to avoid loss, and must record the exact profiler version,
flags, exit status and capture-range markers.
Export and validate every SQLite before starting the next capture and before
the vLLM arm. Any non-whitelisted severity>=2 diagnostic, absent graph-node
rows, uneven dominant-graph replay count, missing indexed artifact or hash
drift voids the attempt. H1d must also reconcile graph-node events against
runtime graph launches and the exact 16-warmup/48-measured workload; require
distinct capture IDs and path/command/report/SQLite linkage; canonicalize and
hash the full primary-node name/geometry/resource multiset; require identical
structure across all three captures; and recompute every retained kernel
summary from the indexed raw artifact. The exact 27B c16 graph contract is
1,107 primary nodes with 208 FP4 GEMM, 144 normal-producer, 64 fused-producer,
48 recurrence and 16+16 FA2 nodes; any family drift also voids the attempt.
The remaining 611 primary nodes may not stay unconstrained. vLLM uses the
mandated Torch profiler because nsys breaks its EngineCore on this host; its
raw trace, selected hash, exact command/corpus/workload and clean decode-window
family counts must be independently recomputed rather than trusted from the
aggregate summary. Geometry/window slicing must exclude prefill, eager and
graph-capture contamination.

The report must inventory current normal/fused producers, FP4 GEMMs, GDN
post-convolution/recurrence, FA2, graph-node counts, the frozen 64-plan map,
output digests, cache eviction and lifecycle. The plan cache must be read-only,
load all 64 accepted FlashInfer plans, load/save/tune/reject zero native plans
and reproduce selected-plan SHA `f2d9be7f…1fa4` independently in every server
process. This is a parsed semantic gate, not a log-presence/hash check; missing
records void the attempt. The forbidden native-cache target must remain absent.
Legacy status without the complete H1d schema remains visible but cannot bind.
If a larger verified current residual displaces normal production, leave W3-H
ACTIVE without implementation and spike that row instead. Cross-profiler time
is diagnostic only.

### G1 — build, operator bytes and capture

- CPU Release remains green; CUDA 13.0.88/GCC 13.3/sm_121a warning-as-error
  build passes.
- All ported cases above are exact for packed bytes and every scale byte.
- Misalignment/capability/toolkit eligibility falls back exactly.
- Candidate and scalar graph capture/replay are byte-identical and allocation-
  free during capture/replay.
- Focused compute-sanitizer reports zero errors and zero leaked bytes.

### G2 — SASS, microbenchmark and counters

- Candidate BF16/direct symbol has one 256-bit input load, one packed 64-bit
  store, no second input read, no local/stack spill; fallback symbol remains.
- Same-binary CUDA-event ABBA microbenchmarks run after warmup for K=5,120 and
  K=6,144 at M=1/2/4/8/16/32 and for representative M=128/512/2,048 prefill.
  Report median, p90 and spread; candidate must beat scalar at both decode K
  classes and must not regress any retained large-M class.
- Nsight Compute on M=16/K=5,120 and M=16/K=6,144 records duration,
  instructions, global/L1/L2/DRAM traffic, occupancy and stalls for both arms.
  M=2,048 is required before claiming TTFT/prefill credit.
- A positive kernel delta is necessary but not sufficient for E2E acceptance.

### G3 — real model and structural trace

- 27B candidate and scalar each pass 235/235 plus committed 16/16 oracle
  tokens from the identical frozen plan map. Correctness-only 35B passes
  315/315 and proves the W4A16 path inert; this is not performance authority.
- Paired node traces show exactly 144 normal and 64 fused producers per steady
  decode forward, 208 FP4 GEMMs/forward, unchanged kernel identities and 64/64
  tactics, no extra copy/allocation/synchronization and no fused-producer change.
- The same-profiler candidate/scalar slice must improve both normal K classes.
  Otherwise stop before the serving component.

### G4 — strict same-binary serving component

On an idle DGX under one uninterrupted lock, run candidate versus
`VT_FP4_QUANT_VEC=0` in AB/BA/AB order at cache-off input-1,024/output-128,
greedy seed 0, c2 and c16. Both model gates precede timing. The 12-leg/612-
request series requires all lifecycle/cache/thermal/frozen-plan checks plus
**40/40 timing and 8/8 memory axes**. Mean-only improvement, SASS improvement,
or a failed memory/tail axis gives W3-H no speed credit and leaves default off.

### G5 — exact vLLM grid and downstream gates

Only a 48/48 G4 pass authorizes a fresh production-vLLM v0.25.0 c1/2/4/8/16/32
three-repetition 27B grid and paired trace. Every one of 124 axes must pass.
Only after 27B closes may the held 35B performance campaign run. SGLang
cache-off/prefix gates and the broader roadmap retain their existing ordering.

## Exact read-only reproduction

Current local normal graph slice:

```sh
ssh dgx.casa 'DB=$HOME/work/vllm.cpp-fa2-decode/ae9e8ff0576badabdda7289beeacaa1041c55d21/evidence/trace/fa2-default.sqlite; sqlite3 -header -column "$DB" "SELECT CASE WHEN k.graphNodeId IS NULL THEN '\''eager'\'' ELSE '\''graph'\'' END mode,k.gridX,k.gridY,k.blockX,COUNT(*) calls,ROUND(SUM(k.end-k.start)/1e6,6) ms,ROUND(AVG(k.end-k.start)/1e3,6) us FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE instr(s.value,'\''ScaledFp4QuantKernel<__nv_bfloat16'\'')>0 AND instr(s.value,'\''Silu'\'')=0 GROUP BY mode,k.gridX,k.gridY,k.blockX ORDER BY mode,k.gridX,k.blockX;"'
```

Fresh G0 uses a new immutable pushed SHA and evidence root:

```sh
SHA=$(git rev-parse HEAD)
ROOT="$HOME/work/vllm.cpp-executed-path-refresh/$SHA"
scripts/dgx-online-serving.sh --trace-only --model 27 \
  --snapshot "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/$(readlink "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/refs/main")" \
  --source-corpus "$ROOT/evidence/corpus/27" \
  --evidence "$ROOT/evidence" --build-dir "$ROOT/build-cuda" \
  --client "$HOME/venvs/vllm-oracle/bin/vllm"
```

The setup checkpoint must prepare and hash the source corpus and manifest
before the lock is acquired. No partial trace duration or throughput binds.

## Dependencies

- vLLM v0.25.0 target `702f481`, validated oracle environment and FlashInfer
  0.6.13/CUTLASS source used by the binding gate.
- CUDA 13.0.88, GCC 13.3, sm_121a/GB10, Nsight Systems, Nsight Compute and
  `cuobjdump`.
- Qwen3.6-27B NVFP4 snapshot, exact online-gate token corpus, frozen 64-plan
  fixture and existing 27B/35B token gates.
- W3-C plan reproducibility and W3-E direct swizzled-scale production. W3-F and
  W3-G stay held constant; their component failures remain independent.
- One uncontended `/tmp/gpu` lock for each complete paired profile or A/B
  series. No new dependency or license is introduced.

## Work breakdown

| Work | Deliverable | State |
|---|---|---|
| W3-H0 | whole-chain source/SASS/trace/history/test/gate inventory | **complete in this spike** |
| W3-H1 | fresh exact-workload current ours/vLLM paired trace and residual re-ranking | **ACTIVE: H1a/H1b/H1c complete but VOID; H1d lossless collector + exact plan/workload/graph/vLLM validation pending** |
| W3-H2 | I/O-only BF16/direct vector kernel, host toggle/eligibility and scalar fallback | **pending; prohibited until H1** |
| W3-H3 | ported byte/alignment/capture tests, sanitizer, SASS, microbench/NCU, model and paired structure gates | **pending** |
| W3-H4 | frozen c2/c16 40+8 strict component | **pending** |
| W3-H5 | conditional default flip, exact v0.25 27B 124-axis grid and lifecycle classification | **blocked on 48/48 H4** |

## Risks and decisions

- **Insufficient ceiling:** 0.284 ms/forward cannot close the gate alone. Keep
  scanning after this isolated leaf; never call the remaining gap diffuse.
- **Profiler mismatch:** current ours/vLLM times identify a target but cannot
  prove a ratio. Same-binary candidate/fallback owns performance acceptance.
- **Alignment:** CUDA allocations and K%16 normally preserve 32-byte group
  alignment, but public tensors can be manually offset. Check and fall back.
- **Numerical drift:** load/store vectorization must not change arithmetic
  order, reciprocal mode, nibble order, NaN/zero handling or padded bytes.
- **Old neutral result:** `f787cf8` prevents assuming vectorization wins. Do
  not repeat its per-call `getenv`, and cache capability before model
  execution. The strict stop rule is part of the contract, not optional
  context.
- **Scope stacking:** fused SiLU, hardware conversion, approximate reciprocal
  and 2D geometry require independent measured checkpoints.
- **Portability:** keep the scalar CUDA and CPU/reference paths. The vector
  path is a capability specialization, not a replacement for backend-neutral
  behavior.
