# sm_120 Qwen3.5 fused GDN decode direct-state-store discriminator

**Lifecycle:** `REJECTED/REMOVED`; decisive timing loss, product/tests restored
to parent `a9d7789579582872d5063620acd9eb33b73ed1a3`

**Owner rows:** `KERNEL-SSM-MAMBA`, `ROAD-V1-C2-LOCAL-BF16`

**Hardware/workload:** RTX 5070 Ti (`sm_120`), Qwen3.5-4B plain BF16

**Authority:** records-only rejection/cleanup. No retained product or test code,
new GPU run, remote operation, default, release, acceptance, or 27B/35B claim.

## Gap, evidence, and falsifiable hypothesis

This spike is based on clean disposition head
`63842df4b85a5eaf3b703ff7d3627d04340a881d`. REGK was structurally rejected
and removed: same-tool nvcc 12.9 emitted byte-identical exact PTX for its
baseline and either recurrence loop restored to `bk`, proving the compiler
already keeps K live. Do not resurrect REGK or add q+k caching; q is single-use.

The incumbent is exact, provisional `REGSTATE` product `7476818c1`. Its y800
mean is **164.57 us/call**, **-1.093499%** versus SWZ but about **1.285x**
pinned vLLM's 128.061 us/call. Its exact compiler structure is 894 PTX
instructions, 53 `ld.shared`, 31 `st.shared`, two barriers, 56 registers and
zero stack/spills; gx8/bx128/smem9728 is unchanged from SWZ. Exact token
SHA-256 is
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.

In `src/vt/cuda/cuda_gdn.cu:GdnDecodeFusedKernel`, REGSTATE finishes the
recurrence with each lane's 16 updated F32 values in `rr`. It then writes
`rr[j]` back to swizzled shared, executes a second `__syncthreads()`, and
cooperatively rereads shared before converting/storing persistent state. The
sequence exists only to recover coalesced global stores.

Pinned vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` directly stores its F32
register state tile: vendored FLA
`vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:156-170`
forms `p_ht` from the resolved state index and executes
`tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)`. This grounds the
direct register-to-persistent-state lifetime, not FLA's layout or schedule.

**Hypothesis:** on the exact full REGSTATE production tile, each thread can
store its own `rr[16]` directly to logical persistent-state columns, removing
the final shared writes, barrier and rereads while preserving every state byte.
The deliberate counter-cost is less-coalesced global traffic. Static structure
and same-binary timing decide; inferred barrier or coalescing credit is invalid.

## Exact selector, eligibility, and mapping

Add strict default-OFF `VT_GDN_DECODE_DIRECT_STORE=1`. Only complete string
`"1"` selects it. Unset, empty, whitespace, prefixes/suffixes, numeric and
truthy lookalikes preserve the current path.

Eligibility is only the regular, non-speculative, non-fallback, full-tile
`REGSTATE=true` production specialization: exact BV16+SWZ, `Dv==Dk==128`,
resolved NW8. The flag never promotes another path. Invalid/unset or any failed
predicate retains current REGSTATE when otherwise eligible; partial shapes,
BV32, unswizzled BV16, other NW, sequential fallback, speculative decode and
null/index corner behavior remain incumbent.

Add a host/device constexpr mapping helper used by both product and portable
tests. For `vi=0..15`, `wk=0..7`, `j=0..15`, the direct offset from `s_head` is:

```cpp
vi * dk + GdnDecodeRegisterLogicalColumn(wk, j)
// exact production form: vi * 128 + wk * 16 + j
```

The mapping is bijective over all 2,048 offsets in a full BV16×Dk128 state
tile. `s_head` already contains the resolved compact/indexed persistent row,
head and `vbase`; the helper must not repeat those terms.

Specialize a compile-time `DIRECT_STORE` boolean with
`static_assert(!DIRECT_STORE || REGSTATE)`. For eligible valid threads, after
the existing output store, execute increasing `j=0..15`:

```cpp
Store(s_head, GdnDecodeDirectStateOffset(vi, wk, j, dk), rr[j]);
```

In `DIRECT_STORE=true` only, omit the complete final `rr→r` loop, second
barrier, and cooperative `sbh→s_head` loop. `DIRECT_STORE=false` must compile
the current REGSTATE body. Preserve the initial cooperative state load and its
barrier, q/k shared staging, `rr` load, all recurrence arithmetic and shuffles,
output store, dtype conversion, state index/null resolution and global layout.
Keep gx8/bx128/smem9728 even though the final shared phase disappears.

## Red-first correctness and mutation gates

Before product changes, extend the portable contract tests for:

1. exact parser semantics and invalid fallback;
2. eligibility requiring every REGSTATE production/full-tile predicate;
3. exactly one incumbent or direct-store callback and unchanged geometry;
4. representative direct offsets plus exhaustive bijection of all 2,048 full
   tile elements, with no duplicate, missing or out-of-range address.

Extend CUDA comparison of REGSTATE versus DIRECT_STORE to require byte-exact
complete output and full persistent state across production and every
ineligible/fallback surface: compact/indexed state, null indices, partial
dimensions, BF16/F32 I/O, and BF16/F16/F32 state where supported. The public
graph must execute exactly one gx8/bx128/smem9728 candidate node. Graph geometry
does not prove barrier removal; compiler evidence does.

Scratch mutations must kill the strict parser, each eligibility predicate,
callback binding, direct logical address, direct store, output store, indexed
row and null handling. Replacing DIRECT_STORE with the incumbent shared
fallback must fail exact compiler shared-op/barrier discrimination. Forcing a
partial/ineligible path into direct storage must fail portable eligibility or
CUDA exact coverage. Restore the tree byte-for-byte after every mutation.

## Compiler and resource gate

Build REGSTATE and DIRECT_STORE into one immutable binary and compare the exact
`bf16,bf16,float,NW8,SWIZZLED=true,REGSTATE=true` specializations with the same
toolchain. Record compiler/binary hashes, registers, stack/local/spills,
barriers, PTX/SASS totals and shared/global load/store counts.

DIRECT_STORE must have fewer PTX **and** SASS shared loads, fewer shared stores,
and fewer barriers than REGSTATE; zero spills, zero stack and zero local array;
unchanged gx8/bx128/smem9728; and at most **5%** total SASS instruction growth.
Record global-store count because this arm trades coalescing for static shared
work, but do not infer transaction efficiency from instruction count. Use
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on actual function pointers;
an active-warps drop greater than **12.5%** rejects the arm. Missing exact
symbols, compiler no-op, or any static/resource miss removes the candidate
before timing credit.

NCU remains owed diagnostic when local binary and counter authority exist:
profile matched y800 nodes and record global transactions/sectors, shared
requests/wavefronts, barrier/scoreboard stalls, DRAM, achieved occupancy and
instructions for REGSTATE, DIRECT_STORE and pinned vLLM. It is attribution,
never speed credit or a substitute for static/timing gates.

## Same-binary performance and disposition

Under one `flock /tmp/gpu` lock, use the immutable REGSTATE Qwen3.5-4B BF16
128-request, 128-output-token, concurrency-32 workload. Hold every other
selector fixed and run:

`REGa -> DSTa -> DSTb -> REGb`.

Collect graph-node `nsys`, exact call/geometry/resource evidence, token hashes,
all-fused and y800 sums/counts, total/output throughput, TTFT, TPOT/ITL, E2E,
peak GPU allocation, peak/stable PSS and available-memory drop. Run a separate
counterbalanced REG/DST memory pair without profiling.

Retain DIRECT_STORE only if all correctness, mutation, graph and resource gates
pass; both DST legs beat their paired REG controls; counterbalanced y800
improves by at least **1.00%**; all fused decode improves; total/output
throughput do not decrease; TTFT, TPOT/ITL and E2E do not increase; every
enclosing axis and paired memory are non-regressing; and tokens are exact. A
host-PSS increase is material only when it exceeds both 1.00% and 64 MiB, then
must reproduce once before rejection.

If compiler structure is unchanged or any bar fails, remove selector, product
and tests in the disposition change and preserve the rejection in this spec,
benchmark record, kernel/roadmap rows, STATUS and BENCHMARKS. If every bar
passes, retain only as explicit opt-in. No default, release, pinned-vLLM,
27B or 35B acceptance follows from this local discriminator.

## Outcome — rejected and removed

Committed experiment `9485a551473534af9af348c10aff31b949fdd840` preserved
exact tokens and the intended gx8/gy800/bx128/smem9728 graph. REGSTATE/DIRECT
STORE used 56/55 registers and zero local storage. Static PTX improved from
848 to 694 instructions, 53 to 48 shared loads, 31 to 15 shared stores and two
to one barrier, but global stores increased from 7 to 18.

The prescribed same-binary `REGa -> DSTa -> DSTb -> REGb` series decisively
rejects the trade: y800 **134.3005475 -> 196.3294475 ms (+46.1866%)**,
**164.584 -> 240.600 us/call**; all fused decode **264.656741 ->
388.7906645 ms (+46.904%)**. Both raw DST legs lose. Enclosing REG/DST means
also regress: total **6725.73 -> 6694.67 tok/s (-0.4618%)**, output **743.715
-> 740.28 tok/s (-0.4619%)**, TTFT **1029.48 -> 1031.48 ms (+0.1943%)**,
TPOT/ITL **35.02 -> 35.21 ms (+0.5425%)**, and E2E **5477.21 -> 5502.83 ms
(+0.4678%)**. Every token SHA-256 remains exact. Evidence roots are
`/tmp/qwen35-gdn-direct-store-{rega,dsta,dstb,regb}-9485a5514.*`.

The lower shared traffic cannot offset the loss of global-store coalescing.
No memory pair is owed after this hard timing rejection. The exact unpublished
tip was reverted without committing; its seven product/test/env/pending-doc
paths match parent `a9d778957` byte-for-byte. DIRECT_STORE is therefore
**REJECTED/REMOVED** with no acceptance/default claim. This disposition names
no new implementation target.
