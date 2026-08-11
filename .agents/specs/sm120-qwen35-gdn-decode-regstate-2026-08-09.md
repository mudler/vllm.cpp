# sm_120 Qwen3.5 fused GDN decode register-resident state discriminator

**Lifecycle:** `MEASURED/PROVISIONAL`; exact specialization SASS-size and NCU
diagnostics `PENDING`

**Owner rows:** `KERNEL-SSM-MAMBA`, `ROAD-V1-C2-LOCAL-BF16`

**Hardware/workload:** RTX 5070 Ti (`sm_120`), Qwen3.5-4B plain BF16

**Scope:** one strict, default-OFF, numerical-order-preserving register-state
variant of the accepted BV16+NW8+shared-swizzle production kernel. The spike
commit was spec-only; the measured outcome below remains default-OFF and makes
no default, release, or 27B/35B claim.

## Gap, evidence, and falsifiable hypothesis

Product `824370396` accepted `VT_GDN_DECODE_SWIZZLE=1` as an exact opt-in. Its
counterbalanced means improve all fused decode 7.4555% and grid-y 800 7.5123%,
but the accepted y800 call is still **166.566 us** versus pinned vLLM
**128.061 us** (**1.301x** slower). Both local arms execute gx8/bx128/reg56;
swizzled shared memory is 9,728 bytes. The accepted token SHA-256 is
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.

The accepted local kernel cooperatively stages the F32 `[BV,Dk]` state into
swizzled shared memory (`cuda_gdn.cu:2758-2783`), then each lane traverses its
same 16 logical columns twice: decay/dot at `:2794-2810`, followed by
update/output at `:2813-2828`. It finally synchronizes and rereads shared for
the coalesced global writeback at `:2830-2838`. Thus each logical state element
participates in an initial shared store, two shared read-modify-writes, and a
writeback shared load.

**Hypothesis:** on the exact production specialization, retaining the lane's
16 F32 state values in registers between the two recurrence loops removes one
shared read and one shared write per element without changing any arithmetic
or logical access. If compiler resource growth, spills, lost occupancy, or
shared-to-register moves consume the saving, the candidate will not clear the
gates and must be removed.

## Complete execution-chain ground

The local production path is:

1. Qwen's paged step selects the decode segment and indexed persistent cache at
   `src/vllm/model_executor/models/qwen3_5.cpp:3950-3969`.
2. Public validation and dispatch are `src/vt/ops.cpp:1946-1955`.
3. CUDA validates dtypes/state, resolves NW8 and the shared-budget fallback at
   `src/vt/cuda/cuda_gdn.cu:2920-2950`; BV and swizzle environment dispatch is
   at `:2968-2969`.
4. The portable strict selectors, production-only swizzle eligibility and
   9,728-byte launch contract are
   `src/vt/cuda/gdn_decode_fused.h:16-78`.
5. The executing template, shared staging, recurrence and launch binding are
   `src/vt/cuda/cuda_gdn.cu:2729-2871`.

Pinned vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` calls the vendored FLA
Triton kernel in
`vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:24-178`.
Its logical F32 `[BV,BK]` value `b_h` is created and loaded at `:102-120`, kept
across decay/dot/update/output at `:122-154`, and stored at `:156-170`. The
wrapper resolves `BK=128`, `BV=32`, grid `(NK,NV,N*HV)`, four warps and three
stages at `:205-212,241-277`.

The exact pin-generated artifact is
`.triton-cache/D5PNU7TCKM4WKZZ6SN22CRUFGNRRSCWEBBGHZEFA6WQ3V7FVKNAA/`:
`fused_sigmoid_gating_delta_rule_update_kernel.source:1-28` binds the generated
IR back to upstream line 24; its JSON records sm_120, Triton 3.7.1, four warps,
three stages and 512 shared bytes. The matched production trace records
grid `1x4x800`, block 128, 96 registers and 128.061 us/call. This is evidence
that a register-heavier schedule exists, not permission to copy Triton's
different BV32 reduction schedule or its fused gate arithmetic.

## Exact candidate and rollback contract

Add strict opt-in `VT_GDN_DECODE_REGSTATE=1`. Only exact string `"1"` selects
the candidate. Unset, empty, whitespace, prefixes/suffixes, truthy words and
every other value retain the accepted shared-state kernel.

Eligibility is the conjunction of:

- resolved `VT_GDN_DECODE_BV=16`;
- resolved `VT_GDN_DECODE_SWIZZLE=1`;
- `Dv==Dk==128` and resolved `NW==8`;
- the regular, non-speculative `GdnDecodeFusedKernel` path.

Every partial shape, BV32/default, unswizzled BV16, other NW, sequential
fallback, and speculative-decode kernel remains byte-for-byte on its incumbent
path. Specialize a compile-time `REGSTATE` boolean; do not add a runtime branch
inside either recurrence loop.

For `REGSTATE=true`, after the existing cooperative load and synchronization:

1. declare `float rr[16]` and load in increasing `j` order
   `rr[j] = sbh[vi*136 + j*8 + wk]`;
2. execute the existing decay/dot loop in the same `j=0..15` order using
   `rr[j]`, with identical expressions and accumulator;
3. retain the existing XOR shuffle sequence `off=1,2,4`;
4. execute the existing update/output loop in the same `j=0..15` order using
   `rr[j]`, with identical expressions and accumulator;
5. retain the existing output conversion/store and shuffle sequence;
6. write `rr[j]` back to the same swizzled shared addresses in increasing `j`,
   synchronize, then use the existing coalesced global state writeback.

Do not change state/q/k/v global layout, shared mapping or 136-float row
stride; gx8/bx128/smem9728; input/output/state dtype conversion; `expf`, beta,
`vp`, FMA spelling; accumulator order; shuffle order; null/index semantics; or
public surface. Rollback is `VT_GDN_DECODE_REGSTATE=0` or unset. The feature
remains explicit opt-in even if accepted.

## Red-first portable and CUDA tests

Extend `tests/vt/test_gdn_decode_fused.cpp` (existing selector/layout coverage
at `:17-160`) before implementation:

1. exact `"1"` parser plus unset/empty/whitespace/prefix/suffix/truthy invalids;
2. production eligibility requires BV16+swizzle+Dv128+Dk128+NW8;
3. every individual failed predicate selects the incumbent specialization;
4. the resolved contract remains gx8/bx128/smem9728 and exposes exactly one
   shared-state or register-state callback;
5. for all `wk=0..7,j=0..15`, register slot `j` maps to shared `j*8+wk` and
   logical column `wk*16+j`, bijectively covering 0..127.

Extend the CUDA exact matrix rooted at `tests/vt/test_ops_gdn.cpp:1867-1953`
to compare accepted SWZ versus REGSTATE byte-for-byte for complete output and
persistent state across production and fallback/partial shapes, compact and
indexed state, null indices, BF16/F32 I/O, and BF16/F16/F32 state where the
public op supports them. Extend the public graph test at `:3488-3577` to prove
REGSTATE executes exactly one gx8/bx128/smem9728 node. Run cached Qwen3.5
paged-forward and require the exact production token SHA above.

Scratch mutations must make the focused test fail when they corrupt or remove
the strict parser, production eligibility, one-callback dispatch, `rr[j]`
load mapping, `rr[j]` writeback mapping, output store, or persistent-state
store. A candidate that merely compiles while silently taking SWZ is a failure.

## Compiler/SASS and resource gate

Build baseline and candidate into one immutable binary, then capture the two
template specializations from the same toolchain:

```sh
nix develop .#cuda --command cmake --build build-nix-cuda-transplant-triton --target vllm-bench test_gdn_decode_fused test_ops_gdn
nix develop .#cuda --command cuobjdump --dump-resource-usage --demangle build-nix-cuda-transplant-triton/examples/vllm-bench
nix develop .#cuda --command cuobjdump --dump-sass --demangle build-nix-cuda-transplant-triton/examples/vllm-bench
nix develop .#cuda --command cuobjdump --dump-resource-usage .triton-cache/D5PNU7TCKM4WKZZ6SN22CRUFGNRRSCWEBBGHZEFA6WQ3V7FVKNAA/fused_sigmoid_gating_delta_rule_update_kernel.cubin
```

Record exact binary/cubin SHA-256, compiler versions, registers/thread, static
and dynamic shared, stack/local bytes, spill loads/stores, static SASS
instruction count, and shared-load/shared-store instruction counts. Candidate
must have zero spills and zero local state array, unchanged dynamic shared and
block geometry, fewer static shared-memory instructions, and no more than 5%
total static SASS instruction growth. Use
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the two actual function
pointers; a predicted active-warps drop greater than 12.5% is material and
rejects the arm. Resource data describe compiler output only; they do not
substitute for hardware counters.

## Fallback performance experiment without NCU

Nsight Compute is **unavailable on this local workstation**: `ncu --version`,
`/usr/local/cuda-13.0/bin/ncu --version`, and
`nix develop .#cuda --command ncu --version` all fail because no local binary
is installed; `flake.nix` supplies Nsight Systems, not Nsight Compute. The
repository's `ncu 2025.3.1.0` evidence belongs to DGX, whose use is not
authorized for this task. Counter permission and any local installation are
external authorization gates. Therefore no bank-conflict, scoreboard, DRAM,
or achieved-occupancy conclusion may be inferred in this experiment.

The falsifiable fallback is same-binary graph-node `nsys`, under the configured
GPU lock, in counterbalanced order `SWZa -> REGa -> REGb -> SWZb`, using the
accepted cached checkpoint and exact workload:

```sh
GPU_LOCK=/tmp/gpu
MODEL=/home/rich/c/vllm.cpp/.hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
BIN=/home/rich/c/vllm.cpp/build-nix-cuda-transplant-triton/examples/vllm-bench
flock "$GPU_LOCK" env LD_LIBRARY_PATH=/run/opengl-driver/lib VT_RELEASE_HOST_WEIGHTS=1 VT_DIRECT_DEVICE_LOAD=1 VT_GDN_DECODE_BV=16 VT_GDN_DECODE_SWIZZLE=1 VT_GDN_DECODE_REGSTATE=0 nsys profile --force-overwrite=true --sample=none --cpuctxsw=none --trace=cuda,nvtx --cuda-graph-trace=node --output=/tmp/qwen35-gdn-regstate-swza "$BIN" --model "$MODEL" --dataset-path /tmp/qwen35-4b-sharegpt-1024.json --num-prompts 128 --output-len 128 --concurrency 32 --temperature 0 --max-num-batched-tokens 2048 --num-blocks 1280 --output-token-ids /tmp/qwen35-gdn-regstate-swza.tokens.json
```

Repeat the exact command for `REGa`, `REGb`, and `SWZb`, changing only
`VT_GDN_DECODE_REGSTATE`, output names, and using one lock across the complete
series. Record all-fused and grid-y 800 sums/counts, launch geometry/resources,
token SHA, total/output throughput, TTFT, TPOT/ITL, E2E, peak GPU allocation,
peak/stable host PSS and available-memory drop. Run a separate counterbalanced
SWZ/REG memory pair without `nsys`.

## Acceptance, rejection, and owed diagnostic

Retain REGSTATE only if all correctness, mutation, graph and resource gates
pass; both REG samples beat their paired SWZ control; the counterbalanced y800
mean improves by at least **1.00%**; all-fused decode improves; total/output
throughput do not decrease; TTFT, TPOT/ITL and E2E do not increase; and GPU
allocation does not materially increase. A host-PSS increase is material when
it exceeds both **1.00% and 64 MiB**; repeat such a move once, and reject if it
reproduces. Any spill/local array, >12.5% predicted active-warp loss, material
memory regression, mixed sample direction, or enclosing regression rejects the
candidate. Rejection removes the selector, specialization and tests while
preserving the refutation in this spec, benchmark record, kernel/roadmap rows,
STATUS and BENCHMARKS. Acceptance updates the same records but remains opt-in;
default/release/27B/35B gates stay open.

When a local NCU binary and counter authority become available, the still-owed
diagnostic profiles the 400th per-launch-configuration y800 node under
`--replay-mode application --graph-profiling node --target-processes all` and
records shared requests/wavefronts/bank conflicts, DRAM bytes/throughput,
short/long-scoreboard stalls, achieved occupancy and executed instructions for
SWZ, REGSTATE and pinned vLLM. It is diagnostic attribution, not retroactive
speed credit, and must never be marked complete from SASS or timing inference.

## Measured outcome, 2026-08-09

**Disposition:** `MEASURED/PROVISIONAL`, retained as an explicit opt-in at
product commit `7476818c1`. Correctness, graph geometry, observed compiler
resources, counterbalanced timing, enclosing performance and memory satisfy
their measured bars, but the exact-specialization static SASS-size <=5% check
and the NCU diagnostic remain pending external download/tool authority.
Therefore this is not a fully `ACCEPTED` candidate and receives no default,
release, or 27B/35B credit.

The full canonical preflight is green and the fresh static plus targeted
mutation re-review reports `PASS`. Operator gates are portable selector/mapping
**10/10 · 1,962**, CUDA GDN **70/70 · 4,830**, and cached Qwen3.5
paged-forward **4/4 · 8**, with all observed diffs zero. Every performance and
memory token file has SHA-256
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.
The graph contract is exactly grid x/y `8/800`, block x 128, dynamic shared
9,728 bytes, 56 registers and zero local bytes.

The same-tool ptxas log
`/tmp/cuda_gdn-regstate-candidate-7476818c1.log` identifies the production
`GdnDecodeFusedKernel<bf16,bf16,float,8,true,true>` specialization at 56
registers, one barrier, zero-byte stack frame and zero spill loads/stores. The
accepted `SWIZZLED=true, REGSTATE=false` specialization has the same reported
resources. This proves no observed register, barrier, stack or spill regression;
it does not substitute for the still-missing exact static SASS instruction-size
and shared-instruction counts.

The retained nvcc 12.9 intermediate build at `/tmp/cuda-gdn-keep.gXumAL/`
adds non-substitute compiler structure. `readelf` reports exact sm_120a cubin
ELF symbol sizes 27,008 bytes for SWZ and 26,496 bytes for REGSTATE
(**-1.896%**). Exact PTX function counts are 922/894 total instructions,
69/53 `ld.shared`, 47/31 `st.shared`, unchanged 19/7 global loads/stores and
two barriers for SWZ/REGSTATE. Thus the generated PTX removes exactly 16 shared
loads and 16 shared stores and the cubin symbol shrinks, supporting the intended
mechanism and <=5% byte-size direction. These are not the required SASS
instruction/shared-op counts, so the cuobjdump gate remains pending. Artifact
SHA-256 values are `dc500503...e2e6` for `cuda_gdn.ptx` and
`5169611e...d913` for `cuda_gdn.sm_120a.cubin`.

The prescribed four-leg series is
`SWZa -> REGa -> REGb -> SWZb`. Both raw REGSTATE legs beat both bracketing SWZ
legs on grid-y 800 and all fused decode. Counterbalanced means are:

| Axis | SWZ | REGSTATE | Change |
|---|---:|---:|---:|
| grid-y 800, 816 calls | 135.7758125 ms | 134.2911050 ms | **-1.093499%** |
| all fused decode, 1,704 calls | 268.1141915 ms | 264.7265645 ms | **-1.263502%** |
| total throughput | 6726.240 tok/s | 6739.125 tok/s | **+0.191563%** |
| output throughput | 743.770 tok/s | 745.195 tok/s | **+0.191591%** |
| mean TTFT | 1026.800 ms | 1021.745 ms | **-0.492306%** |
| mean TPOT / ITL | 35.000 ms | 35.000 ms | unchanged |
| mean E2E | 5471.960 ms | 5466.420 ms | **-0.101243%** |

Evidence roots are
`/tmp/qwen35-gdn-regstate-{swza,rega,regb,swzb}-7476818c1.*`. The separate
non-profiled memory pair is
`/tmp/qwen35-gdn-regstate-{swz,reg}-mem-7476818c1.*`: peak GPU allocation is
13,058/13,060 MiB and peak host PSS is 2,436,964/1,989,479 KiB for SWZ/REGSTATE.
The candidate's PSS is lower, so no repeat is required; timing from this memory
pair is non-binding.

`cuobjdump` and NCU are not locally available without the separately requested
downloads, those downloads are not authorized, and the Ordino service is
unavailable. The next disposition gate is to capture exact SWZ/REGSTATE static
SASS size and shared load/store counts, then run the specified NCU diagnostic
when both the binary and counter authority become available. Until then the
selector stays strict opt-in only.
