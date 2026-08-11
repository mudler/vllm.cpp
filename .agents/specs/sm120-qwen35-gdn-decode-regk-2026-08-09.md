# sm_120 Qwen3.5 fused GDN decode register-key discriminator

**Lifecycle:** `REJECTED/REMOVED`; structural compiler no-op, no product or
test residue at disposition head `28d3a9334d50a80e16f0f3d2ccc0e0cc1b9a4593`

**Owner rows:** `KERNEL-SSM-MAMBA`, `ROAD-V1-C2-LOCAL-BF16`

**Hardware/workload:** RTX 5070 Ti (`sm_120`), Qwen3.5-4B plain BF16

**Authority:** this disposition is records-only. No product or test code, GPU
run, download, remote operation, default, release, acceptance, or 27B/35B
claim is authorized.

## Intake, incumbent evidence, and hypothesis

This spike is based on immutable head
`f157ed6332ab9e5a9e78b77569eac4ae29b08eba`, with current REGSTATE product
`7476818c1b2615a07f6163c0ce192a91518e1ee2`. The regular production chain is
Qwen paged decode in `src/vllm/model_executor/models/qwen3_5.cpp:3950-3969`,
public validation in `src/vt/ops.cpp:1946-1955`, strict portable selection in
`src/vt/cuda/gdn_decode_fused.h:16-161`, and CUDA dispatch/recurrence in
`src/vt/cuda/cuda_gdn.cu:2729-3014`. Current portable coverage is
`tests/vt/test_gdn_decode_fused.cpp:17-279`; CUDA exact and graph coverage is
rooted at `tests/vt/test_ops_gdn.cpp:1867-1953,3488-3645`.

REGSTATE is exact and provisionally faster than accepted SWZ. Its prescribed
counterbalanced means are **164.57 us/call** on grid-y 800, **-1.093499%**
versus SWZ, and **-1.263502%** over all fused calls. Both REGSTATE legs beat
both bracketing SWZ legs and every enclosing mean is non-regressing. It remains
about **1.285x** the pinned-vLLM 128.061 us/call. The production graph is
gx8/bx128/smem9728, ptxas reports 56 registers, one barrier and zero stack or
spills, and every token file is exact at SHA-256
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.

The retained same-tool nvcc 12.9 artifact gives the exact
`bf16,bf16,float,NW8,SWIZZLED=true` structures: SWZ/REGSTATE cubin symbols are
27,008/26,496 bytes, PTX instruction counts are 922/894, and PTX
`ld.shared`/`st.shared` counts are 69/47 versus 53/31. Global loads/stores stay
19/7. These data establish the current baseline; PTX and ELF size do not stand
in for the required exact SASS counts.

Current REGSTATE loads `bk[sc]` in both recurrence loops:
decay/dot at `cuda_gdn.cu:2806-2811` and update/output at `:2830-2834`.
The lane requests the identical 16 K values in the identical `j=0..15` order
both times. Thus K is the next unstacked, independently falsifiable repeated
shared-read lever.

Pinned vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` executes vendored FLA
`vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:24-178`.
Its F32 `b_k` register tile is loaded once at `:124`, optionally normalized at
`:140`, and stays live for both `b_h` reductions at `:148` and `:151`. This
grounds the K lifetime only. It does not authorize FLA's different BV32
schedule, gate arithmetic, normalization, layout, or reduction order.

**Hypothesis:** on the exact REGSTATE production specialization, loading only
the lane's 16 K values once into registers and reusing them in both recurrence
loops removes the repeated shared K read without changing a numerical
operation. If the compiler already performs this caching, if shared loads do
not fall, or if resource growth consumes the saving, the candidate is a no-op
or a miss and must be removed before any q+k experiment.

## Exact selector, eligibility, and implementation

Add strict opt-in `VT_GDN_DECODE_REGK=1`. Only the complete string `"1"`
selects the candidate. Unset, empty, whitespace, prefixes/suffixes, truthy
words and every other value preserve the currently resolved path.

REGK is eligible only when the existing contract has already resolved the
regular, non-speculative `REGSTATE=true` specialization: exact BV16, exact
SWIZZLE, `Dv==Dk==128`, resolved NW8, and the fused kernel rather than the
sequential fallback. REGK must never promote an otherwise ineligible launch.
Invalid/unset REGK and every failed predicate retain current REGSTATE when that
path is otherwise eligible; BV32, unswizzled BV16, partial dimensions, other
NW, speculative decode and sequential fallback retain their current incumbent.

Specialize a compile-time `REGK` boolean with
`static_assert(!REGK || REGSTATE)`. In `REGK=true`, after the existing
cooperative load and synchronization, declare `float rk[16]` and preload once,
in increasing order:

```cpp
for (int64_t j = 0; j < 16; ++j)
  rk[j] = bk[GdnDecodeRegisterSharedColumn(wk, j)];
```

Use the exact `rk[j]` value in both existing REGSTATE loops: replace only the
K operand in `pdot += rr[j] * bk[sc]` and the K operand in
`rr[j] += vp * bk[sc]`; the following `po` expression continues to use
`bq[sc]`. Preserve increasing
`j=0..15`, expression spelling, accumulator sequence, shuffles, and every
other operation. `REGK=false` must compile the current REGSTATE body.

Do **not** cache q in this candidate. Do not change `rr`, state/q/k/v global
or shared layout, the 136-float row stride, global traffic, conversions,
`expf`, beta, `vp`, FMA spelling, output/state stores, accumulator/shuffle
order, null/index semantics, graph capture, public surface, or
gx8/bx128/smem9728. Rollback is `VT_GDN_DECODE_REGK=0` or unset.

## Red-first tests and mutation obligations

Before product code, extend the CUDA-free contract and its focused test to
prove:

1. the REGK parser accepts only exact `"1"`, rejecting unset, empty,
   whitespace, prefix/suffix, numeric and truthy lookalikes;
2. eligibility is exactly current REGSTATE plus REGK, and failing each of BV16,
   SWIZZLE, Dv128, Dk128, NW8, REGSTATE or regular fused-path eligibility
   selects the current incumbent rather than REGK;
3. gx8/bx128/smem9728 is unchanged and exactly one current-REGSTATE or REGK
   callback fires;
4. for every `wk=0..7,j=0..15`, `rk[j]` maps to shared column `j*8+wk` and
   logical column `wk*16+j`, bijectively covering 0..127.

Extend the CUDA exact matrix to compare REGSTATE and REGK byte-for-byte for the
complete output and persistent state. Cover the production and fallback/
partial shapes, compact and indexed state, null indices, BF16/F32 I/O, and
BF16/F16/F32 state where the public operation supports them. The public graph
test must show exactly one gx8/bx128/smem9728 REGK node. Cached Qwen3.5
paged-forward must retain the token SHA above.

Correctness cannot distinguish a K operand that falls back to the same shared
value, so mutation review has two structural obligations in addition to
numerical tests. Mutating the `rk` preload mapping must fail the mapping/exact
gate. Mutating **either** REGK recurrence loop back to `bk[sc]` must fail the
same-tool exact PTX shared-load discriminator (and its SASS counterpart when
available). If the compiler makes either mutation indistinguishable, REGK is a
compiler no-op and is rejected. Also kill the parser, every eligibility
predicate, callback binding, output store and persistent-state store. Restore
the tree byte-for-byte after each mutation.

## Build, resource, correctness, and graph gates

Build REGSTATE and REGK into one immutable CUDA binary. Run the portable
selector/mapping test, full CUDA GDN suite, cached Qwen3.5 paged-forward, staged
and full preflight, then fresh static/targeted-mutation review and independent
operator rerun. Record the compiler/binary hashes and exact specializations.

For exact `bf16,bf16,float,NW8,SWIZZLED=true,REGSTATE=true`, compare REGSTATE
and REGK using the same compiler/toolchain. Both PTX and SASS shared-load
counts must fall; zero stack, zero spills and zero local array are mandatory;
gx8/bx128/smem9728 must be unchanged; total SASS instruction growth must be
at most **5%**. Record registers, barriers, local/stack/spill bytes, exact
PTX/SASS totals and shared/global load/store counts. Use
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the actual function
pointers; an active-warps drop greater than **12.5%** rejects REGK. A compiler
no-op, missing exact specialization, unchanged shared loads, or any resource
bar failure triggers cleanup before timing credit.

NCU remains an owed diagnostic when a local binary and counter authority exist:
profile the matched 400th y800 node and record shared requests/wavefronts/bank
conflicts, DRAM, scoreboard stalls, achieved occupancy and instructions for
REGSTATE, REGK and pinned vLLM. NCU is attribution only, never speed credit and
never a substitute for the static or timing gates.

## Same-binary measurement and disposition

Under one `flock /tmp/gpu` lock, use the immutable cached Qwen3.5-4B BF16
128-request, 128-output-token, concurrency-32 workload from the REGSTATE spec.
Hold every selector except REGK constant and run:

`REGa -> KREGa -> KREGb -> REGb`.

Collect graph-node `nsys`, exact kernel/call/geometry/resource evidence, token
hashes, total and grid-y 800 fused sums/counts, total/output throughput, TTFT,
TPOT/ITL, E2E, peak GPU allocation, peak/stable PSS and available-memory drop.
Run a separate counterbalanced REG/KREG memory pair without profiling.

Retain REGK only if all correctness, mutation, graph and resource gates pass;
both KREG legs beat their paired REG controls; the counterbalanced y800 mean
improves by at least **1.00%**; all fused decode improves; total/output
throughput do not decrease; TTFT, TPOT/ITL and E2E do not increase; every
other enclosing axis is non-regressing; tokens are exact; and paired memory
does not materially increase. Use REGSTATE's material-host-PSS rule: an
increase must exceed both 1.00% and 64 MiB, then reproduce once before reject.

If the compiler makes REGK a no-op or any bar fails, remove its selector,
product specialization and tests in the same disposition change and preserve
the rejection in this spec, benchmark record, kernel/roadmap rows, STATUS and
BENCHMARKS. Any later hypothesis requires a separate spike and must account for
the evidence recorded in the outcome below.
If every bar passes, retain REGK only as an explicit opt-in. No default,
release, pinned-vLLM, 27B or 35B acceptance follows from this 4B discriminator.

## Outcome — structurally rejected and removed

The red-first missing-contract compile was captured before candidate product
work. The candidate selector, specialization and tests existed only as
uncommitted experiment changes and were restored completely; disposition head
`28d3a9334d50a80e16f0f3d2ccc0e0cc1b9a4593` contains no REGK selector, product
code or test residue. The restored portable gate passed **13/13 · 2,005**.

The contract mutations were effective: the parser prefix mutation killed
**0/1** with two failures, regular-path eligibility failed at `-Werror`, and
both compile-time assertions killed the shared-column mapping mutation. The
critical structural mutation did not. With the same nvcc 12.9 toolchain, exact
`bf16,bf16,float,NW8,SWIZZLED=true,REGSTATE=true,REGK=true` PTX for the baseline,
the first recurrence loop changed back to `bk`, and the second loop changed
back to `bk` is byte-identical. Each exact symbol has **848 PTX instructions**
and **53 `ld.shared`**; full PTX SHA-256 is
`1fec62b7f48dc5fa2a33bfe414ce6a10aa01979c41489375a8d67b07c88a97a0`, and
both loop comparisons are identical. Incumbent REGSTATE without the candidate
already reports the same **53 `ld.shared`**, so the compiler already performs
the intended K reuse. The mandatory shared-load reduction and loop-mutation
discriminator therefore fail: REGK is a compiler no-op and is structurally
**REJECTED/REMOVED before product commit, GPU timing, or memory measurement**.

No performance, GPU, memory, default, release, pinned-vLLM, 27B or 35B claim
results. REGSTATE remains measured/provisional with exact SASS and NCU still
pending. The next direction requires a new spike: test direct register-to-global
REGSTATE writeback that removes the final `rr` to shared write, synchronization
and shared reread, with global-store coalescing recorded as the falsifiable
tradeoff. Do not attempt q+k caching: q is single-use and K caching is now a
proven compiler no-op.
