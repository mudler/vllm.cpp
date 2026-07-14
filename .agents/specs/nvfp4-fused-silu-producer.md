# NVFP4 fused SiLU→FP4 producer (W3-I)

Status: **ACTIVE — W3-I1 is structurally accepted at clean `15c6b89` behind a
default-off rollback toggle; the complete c2/c16 component remains pending**

Owning row: `KERNEL-GEMM-NVFP4-W4A4`

Claim: `CLAIM-SERVE-GATE-1`

The accepted schema-v5 trace ranks the dense fused SiLU→FP4 producer as the
largest positive mapped GPU residual: **+0.357354 ms/decode window**, ahead of
the normal BF16→FP4 producer in all 12 local reports. This spike scopes the
first implementation that follows that evidence. It does not change the
binding `3f256ab` result (**55/124**), confer speed credit, authorize the exact
grid, or authorize 35B performance.

## Scope and stop boundary

W3-I first mirrors only the production path executed by Qwen3.6-27B on GB10:

- contiguous BF16 merged `gate_up` input `[M, 2I]`;
- dense one-input `SiluAndMulFp4Quant`;
- 16 values per quantization group;
- direct CUTLASS-swizzled E4M3 block scales;
- packed E2M1 output consumed immediately by the existing CUTLASS FP4 GEMM;
- CUDA 13 / sm_121a, with the current F32, linear-layout, CPU, two-input, and
  expert paths retained as fallbacks.

W3-I does **not** mix in the displaced normal-producer W3-H2, GEMM tactics,
GDN, attention, scheduler, host-memory, or 35B changes. FP16 input and fused
expert variants are inventoried below but are separate breadth leaves. One
speed-sensitive candidate is implemented and fully classified before another
is stacked on it.

## Binding executed evidence

The immutable local trace root is
`~/work/vllm.cpp-executed-path-refresh-h1d/c498a4131af7e6cf0ac678841212af80f4f12d53`.
Validator `71128642ce04c191f559ea4ccabe4b7e33a66b0f` accepts all 12 lossless
one-replay reports; status SHA-256 is
`84d15970d5a68e8a6307949a78eb33fbe5db3104c70129abd3d2ae0bb3696e66`.
The derived ranking lives at
`~/work/vllm.cpp-trace-validator/71128642ce04c191f559ea4ccabe4b7e33a66b0f/c498a413-residual-ranking.json`
with SHA-256
`7c3232487e414d5d0087d310c4189c7c8ab356399bba1f640af4c07478c32456`.

| Evidence | vllm.cpp | vLLM v0.25.0 |
|---|---:|---:|
| Fused calls per clean decode window | 64 | 64 |
| Time per window | 0.595536 ms median of 12 reports | 0.238182 ms mean of 1,476 clean windows |
| Mapped residual | \+0.357354 ms | reference |
| Executed local launch | grid `(544,1,1)`, block `(256,1,1)`, 38 registers | 2-D grid with block 512, 40 registers; clean grids include `(1,3)`, `(4,3)`, `(8,3)`, `(16,3)` |
| Graph attribution | one fused kernel includes actual work and padded-scale sweep | scale buffers are zeroed by generated compiler work; the custom op writes actual rows/groups |

The local object is
`.../c498a413.../build-cuda/CMakeFiles/vllm.dir/src/vt/cuda/cuda_matmul_nvfp4.cu.o`,
SHA-256 `7f06f46d…e965e`. The oracle binary is
`~/venvs/vllm-oracle-v0.25.0-stage/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so`,
SHA-256 `56c647dd…df4`. `cuobjdump` extraction followed by `nvdisasm -b SM121`
grounds the compiled-body comparison:

| Compiled BF16 fused body | vllm.cpp | vLLM |
|---|---:|---:|
| SASS instructions | 1,480 | 384 |
| Gate/up loads | 32 scalar `LDG.E.U16` | 2 × 256-bit `LDG...256` |
| Packed-output stores | 8 × `STG.E.U8` | 1 × `STG.E.64` |
| Scale stores | padded sweep plus actual `STG.E.U8` | 1 actual `STG.E.U8`; padding is pre-zeroed |
| FP4 conversion | scalar threshold/nibble path | 8 packed hardware `cvt...e2m1x2` operations |

Resource counts alone are close; the instruction and memory-operation shape is
not. W3-I therefore ports the resolved packed execution rather than tuning the
existing scalar loop.

## Upstream and dependency chain

The porting pin is vLLM `e24d1b24`; the executable v0.25.0 target is
`702f4814`. The fused kernel, helpers, model route, and tests below are unchanged
between those commits. The only listed-file delta is an unrelated CPU MoE fake
registration in `_custom_ops.py`.

| Layer | Exact upstream source | Required behavior |
|---|---|---|
| Qwen3.5/3.6 dense model | `vllm/model_executor/models/qwen3_5.py:72,147-161`; `vllm/model_executor/models/qwen2_moe.py:75-120` | Qwen3.6 dense MLP uses merged gate/up → `SiluAndMul` → down projection |
| Quantized-activation contract | `vllm/model_executor/layers/fusion/quant_activation.py:18-70` | producer carries packed data, scale, original metadata, and the consumer's `kNvfp4Dynamic` key |
| Consumer dispatch | `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:97-176`; `compressed_tensors_w4a4_nvfp4.py:87-141` | FlashInfer CUTLASS accepts a matching pre-quantized activation and consumes its swizzled scales |
| Fusion selection | `vllm/compilation/passes/fusion/act_quant_fusion.py:36-40,128-181,283-300` | replace `SiluAndMul` + dynamic NVFP4 quant with the fused custom op when the CUDA op exists |
| Mutation lowering | `vllm/compilation/passes/utility/fix_functionalization.py:127-155` | expose the fused op's packed result and block-scale mutations to the compiled graph |
| Allocation contract | `vllm/_custom_ops.py:50-92,1492-1563` | packed bytes are empty; padded swizzled scale storage is zero-initialized because downstream Blackwell GEMM can read padding |
| Stable op entry | `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu:43-49,63-72,140-154`; `torch_bindings.cpp:252,631`; `ops.h:151` | capability-gated CUDA stable ABI, FP16/BF16 only |
| Fused CUDA body | `csrc/libtorch_stable/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu:30-116,120-163` | 16 values/thread, two predicated 256-bit loads, BF16/FP16 packed SiLU, swizzled scale address, packed conversion/store, 512-thread 2-D launch and row loop |
| Quant helpers | `csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh:25-36,118-162,164-200,219-329` | BF16x2 max, approximate reciprocal, E4M3 scale, packed hardware E2M1 conversion, exact tensor-core scale address |
| Vector primitives | `csrc/libtorch_stable/cuda_vec_utils.cuh:123-175,264-288,331-350` | aligned `PackedVec`, predicated 256-bit cache-global loads, packed BF16/FP16 conversion |
| Dense tests | `tests/kernels/quantization/test_silu_mul_nvfp4_quant.py:25-78` | FP16/BF16 shapes and fused-vs-unfused dequantized agreement |
| Graph tests | `tests/compile/passes/test_silu_mul_quant_fusion.py:100-142,238-375`; `tests/compile/passes/test_functionalization.py:263-339` | fused op replaces the two-op graph and survives functionalization |
| Breadth variants | `csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu:33-249,332-446` | expert-offset forms, including fused SiLU, are real upstream surface but are outside dense W3-I1 |

The generated executable graph was also dumped, rather than inferred from
source. Oracle cache file
`~/.cache/vllm/torch_compile_cache/torch_aot_compile/e9ba3b4b4e571c3eb4b6fdb087a24e650940a06403112c39401a583fbaedbaa7/inductor_cache/5w/c5witfuvalucva6yzxyahzqeuejurui2tvihcy3m424u5lj57hdl.py`
has SHA-256 `6e2ee70d…a5ba`. Lines 1263-1272 allocate three padded scale
buffers and zero them together in `triton_poi_fused_0`; lines 1329-1338 pass
the already-zeroed fused buffer to the custom op and then the CUTLASS FP4 GEMM.
That dump explains why padding work is absent from the vLLM custom-kernel SASS.

## Local chain and exact gap

| Layer | Current local anchor | Present behavior / W3-I obligation |
|---|---|---|
| Public op contract | `include/vt/ops.h:518-557`; `src/vt/ops.cpp:206-282` | retain layout/shape validation and CPU-neutral API; clarify that the full operation, not necessarily the custom body, zeroes padding |
| CPU reference | `src/vt/cpu/cpu_ops.cpp:384-410` | retain composite `SiluAndMul` → `ScaledFp4Quant` behavior |
| CUDA fused body | `src/vt/cuda/cuda_matmul_nvfp4.cu:1207-1362,1471-1500` | default-off packed BF16/direct-swizzled specialization uses two 256-bit loads, BF16x2 max, hardware E2M1 packing, signed-zero canonicalization, one 64-bit store and explicit async pre-zero; scalar path remains the fallback |
| Shared scale address/math | `src/vt/cuda/cuda_matmul_nvfp4.cu:913-981` | reuse exact E4M3 and swizzle contract; add packed helpers only where needed |
| Model dispatch | `src/vllm/model_executor/models/qwen3_5.cpp:1071-1082,1210-1219,3468-3506,3566-3587` | production executes one-input BF16/swizzled form; add a cached opt-in eligibility check without changing other paths |
| Buffer lifecycle | `src/vllm/model_executor/models/qwen3_5.cpp:334-395` | pooled `DBuf` memory is dirty, so a packed kernel may not omit padding initialization; W3-I1 must explicitly pre-zero it |
| Existing tests | `tests/vt/test_ops_nvfp4_fp4.cpp:731-817,820-957,959-1127` | two-input, CPU one-input, direct/padded scale, and CUDA one-input tests now cover candidate/fallback, production M classes, poisoned padding, capture/replay and misaligned fallback |

The accepted 27B route has `VT_W4A4_TRUE`, fused SiLU quant, direct scales, and
merged SiLU quant default-on. It calls `SiluAndMulFp4Quant` at
`qwen3_5.cpp:3493-3496` once per dense layer and consumes the result at
`qwen3_5.cpp:3504-3506`. F32/linear and split/two-input routes are not present
in the accepted production trace and do not block the first specialization.

## Semantic contract

1. Each thread owns 16 adjacent logical outputs. Gate and up halves are read
   from one contiguous `[M,2I]` row; nibble order remains low=even/high=odd.
2. SiLU and multiplication execute in F32, then pass through the input dtype's
   packed rounding boundary before the per-group maximum and quantization.
3. The group scale remains E4M3 `input_global_scale_inv * max(abs(x))/6`; the
   existing exact-versus-approximate reciprocal selection is preserved in
   W3-I1 so vector/geometry work is isolated from numeric-policy work.
4. The direct scale byte uses the existing CUTLASS 128×4 atom address. Every
   padded byte must be zero before the downstream GEMM, including dirty-pool
   and graph-replay cases.
5. W3-I1 may use hardware packed E2M1 conversion only if candidate/fallback
   bytes agree in the current exact mode. Any mismatch is diagnosed against
   the upstream dequantized reference and the real-model token gate; it is
   never waived for speed.
6. Misaligned input/output pointers, unsupported dtype/layout, non-sm1xxa
   builds, and disabled candidate mode dispatch to the existing implementation.

## Implementation leaves

| Leaf | Deliverable | State |
|---|---|---|
| W3-I0 | source, generated-code, SASS, executed-trace, dispatch, test, dependency and gate inventory | **COMPLETE in this spike** |
| W3-I1 | opt-in `VT_FP4_FUSED_VEC=1` dense BF16/swizzled specialization: explicit async scale pre-zero, 16-value packed body, 256-bit gate/up loads, BF16x2 max, packed 64-bit output, 512-thread 2-D row-loop launch; cached eligibility and scalar fallback | **IMPLEMENTED / IMMUTABLE STRUCTURE PASS; 48-AXIS COMPONENT PENDING** |
| W3-I2 | if a post-I1 trace shows per-call zeroing is material, aggregate/fuse scale initialization at the model/graph lifecycle seam to mirror the generated vLLM zero kernel | **INVENTORIED; prohibited before I1 classification** |
| W3-I3 | upstream approximate-reciprocal numeric mode plus FP16 packed input, independently correctness/performance-gated | **INVENTORIED** |
| W3-I4 | two-input and expert-offset packed breadth, each with the matching upstream tests | **INVENTORIED** |
| W3-I5 | conditional default flip, exact v0.25.0 27B 124-axis grid, then lifecycle classification | **BLOCKED on a passing I1 component** |

The W3-I1 pre-zero is a safe local equivalent, not a claim that one memset per
producer matches Inductor's multi-buffer fusion. The trace must count those
nodes explicitly. W3-I2 exists only if that measured topology is the next
largest local residual.

## W3-I1 current checkpoint

The immutable source/build/trace root is
`~/work/vllm.cpp-nvfp4-fused/15c6b8933d982019aa8965d218deb0eb1d9dc3f4-r2`.
It is detached at exact clean commit `15c6b8933d982019aa8965d218deb0eb1d9dc3f4`.
All CUDA series below ran uncontended under GPU ownership.

| Gate | Result |
|---|---|
| Build provenance | **PASS — 158/158 CUDA targets** with CUDA 13.0.88, sm_121a, external FlashInfer CUTLASS, vendored Triton AOT and FA2; configure/build log SHA `c8e131d4…0d1c` / `f1d76f28…8f07` |
| CUDA candidate OFF / ON | **PASS — 22/22 cases and 26,916/26,916 assertions per arm**; immutable logs are byte-identical, SHA `d51911e6…5aca` |
| Capture, poisoned padding, production M classes, alignment fallback | **PASS** in the focused operator suite |
| Strict memcheck | **PASS — 22/22, 26,916/26,916, zero errors, zero leaks** with `VT_CUTLASS_NOPOOL=1`; log SHA `3aebdc8c…6964` |
| 27B candidate / fallback | **PASS — 235/235 + 16/16 each**, one frozen 64/64 plan map; immutable logs are byte-identical, SHA `8b150a2f…8670` |
| 35B candidate-on correctness + inertness | **PASS — 2/2 + 315/315**; log SHA `fba79fbb…65f1`; paired trace contains **zero** W3-I calls |
| Packed compiled body | object/SASS SHA `d6ca771b…65bb` / `662f2c54…4102`: **816 instructions, 36 registers, zero stack/local/shared, two 256-bit loads, one 64-bit output store, one scale-byte store, eight packed E2M1 conversions** |
| Paired 27B graph trace | **PASS — 896/896** eligible graph calls use the packed body, with **896** explicit 139,264-byte zero nodes and no scalar call. Fallback fused slice is **6.064064 ms**; candidate body + zero is **3.839808 ms**, down **36.68%**. Allocation/free/sync/capture/copy counts are identical |
| c2+c16 component | **PENDING — no rate or speed credit**. First start is **VOID before measurement** on stale native-cache build ID; repaired exact-fixture driver SHA `06a5f72e…488d` |

Negative zero is part of the byte-exact semantic contract: packed hardware
conversion is retained, then zero-magnitude nibbles have their sign bit cleared
branch-free to match the established exact-mode fallback.

## Tests to port and extend

| Upstream case | Local expression |
|---|---|
| `test_silu_mul_nvfp4_quant` FP16/BF16 × four shapes | keep BF16 in W3-I1; add FP16 as skipped/tracked W3-I3 breadth; compare fused dequantized output to the composite reference at upstream tolerances |
| `TestSiluMulNvfp4QuantModel` fusion result | exercise merged `[M,2I]` → fused packed activation → existing CUTLASS consumer and compare downstream BF16 output |
| activation-quant graph replacement | local model trace must show one fused producer and no materialized BF16 activation or normal quant fallback |
| functionalization mutated outputs | public op must write both packed and scale outputs correctly under CUDA graph capture/replay |

W3-I1 strengthens those tests with BF16 `M=1,2,4,8,16,32,48,128` and
`I=64,128,2048,17408`, exact/tail grid cases, 32-byte aligned and deliberately
misaligned tensors, scale storage poisoned with `0xA5`, all padded bytes zero,
candidate/fallback packed/scale bytes, downstream GEMM output, empty shapes,
repeated capture/replay, and disabled-toggle fallback. Existing F32/linear,
two-input, CPU, 35B, and validation tests remain green.

## Correctness, safety, and structural gates

W3-I1 earns no performance run until all of these pass:

1. CUDA-off build and full CPU suite; exact sm_121a RelWithDebInfo build.
2. Focused CUDA operator suite for every shape/alignment/layout case above.
3. `compute-sanitizer` memcheck with zero errors/leaks, including poisoned
   padding and graph replay.
4. Candidate/fallback SASS check: candidate has two 256-bit BF16 loads, one
   64-bit packed store, one actual scale-byte store, no 32-scalar-load/eight-
   byte-store body, and a materially smaller instruction count.
5. Paired graph trace: exactly 64 fused logical producers/window, candidate
   geometry matches the 2-D row/group contract, no scalar fallback on eligible
   calls, no hidden allocation/synchronization, and every explicit scale-zero
   node is counted.
6. Qwen3.6-27B candidate and fallback each pass the mandatory 16/16
   token-for-token oracle gate with the frozen 64/64 plan fixture. Qwen3.6-35B
   correctness remains unchanged and the W3-I dispatch count is zero there.

Any semantic, memory, graph, or model-gate failure turns W3-I1 off and blocks
speed credit. Correctness is never traded for a faster producer.

## Performance gate and reproduction

Run on an idle DGX under one `/tmp/gpu` ownership window. Use the exact binding
Qwen3.6-27B snapshot, frozen corpus, vLLM v0.25.0 fixture, input 1,024 → output
128, cache off, greedy, and frozen 64/64 tactic map. A same-binary
candidate/fallback `AB/BA/AB` series at c2 and c16 covers all **40 timing + 8
memory axes** and all request/lifecycle checks.

The candidate passes only if:

- both arms pass token correctness and use identical plans;
- candidate matches or beats fallback on every applicable throughput, latency,
  and memory axis across the complete repeated series;
- the improvement reproduces and the trace proves it came from the selected
  fused path; and
- there is no graph-node, allocation, synchronization, or teardown regression.

A positive mean with any red axis is a failed component. A passing component
authorizes the exact vLLM v0.25.0 27B 124-axis grid; it does not itself replace
the binding result. Only **124/124** closes 27B and authorizes 35B performance.

The repaired driver is
`~/work/vllm.cpp-nvfp4-fused/15c6b8933d982019aa8965d218deb0eb1d9dc3f4-r2/w3i-component-driver-r2.sh`,
SHA-256 `06a5f72eaecc0d7d3300be75b6589c7bdcc4590203296926bd1dae0acc89488d`.
It runs `VT_FP4_FUSED_VEC=1` versus `0` against the exact accepted v0.25
FlashInfer fixture (SHA `e81e9181…7edd`) and a guaranteed-absent read-only
native path, under the same binary, corpus, port allocation, one GPU lock, and
AB/BA/AB manifest. The first driver/root remains `VOID`: its reused native
document failed build-ID validation before timing. The repaired result root is
SHA-owned and immutable; partial or contended output is `VOID`.

## Dependencies and risks

- Requires CUDA 12.9+ 256-bit load and packed FP4 PTX; the gate build uses CUDA
  13 and sm_121a. Other CUDA targets retain the scalar fallback.
- `DBuf` reuse makes padding initialization mandatory. Omitting it can corrupt
  output and produce misleading Blackwell throughput, exactly as upstream's
  allocation comment warns.
- A 256-bit load requires aligned base and group addresses. Public/manual
  tensor views must fail eligibility rather than invoke undefined access.
- Cross-profiler residuals select work but do not prove a same-binary win.
- Per-producer memset may erase part of the kernel gain. That is a measured
  W3-I1 outcome, not permission to skip zeroing or silently add W3-I2.
- Host PSS/RSS and low-concurrency serving gaps remain independent even if the
  fused GPU component passes; the full every-axis gate remains binding.
