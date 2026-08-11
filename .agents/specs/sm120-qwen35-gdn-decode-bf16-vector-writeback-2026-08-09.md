# sm_120 Qwen3.5 GDN decode BF16 vector-writeback discriminator

**Lifecycle:** `REJECTED/REMOVED`; exact but below the timing gate

**Owner rows:** `KERNEL-SSM-MAMBA`, `ROAD-V1-C2-LOCAL-BF16`

**Hardware/workload:** RTX 5070 Ti (`sm_120`), Qwen3.5-4B plain BF16

**Scope/authority:** spec only. No product, tests, GPU work, remote operation,
default, release, acceptance, or 27B/35B claim is part of this checkpoint.

## Ground and falsifiable hypothesis

Exact provisional REGSTATE `7476818c1` retains `rr[16]` per lane but writes it
to swizzled shared, synchronizes, and rereads it for coalesced global state
writeback (`src/vt/cuda/cuda_gdn.cu:2789-2865`). Its y800 mean is 134.291105 ms
in the accepted comparison and exact tokens hash to
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.

The removed scalar-direct experiment `9485a5514` proved the final shared phase
is removable but the replacement transport was wrong. PTX fell 848->694,
shared loads 53->48, shared stores 31->15 and barriers 2->1, yet global stores
rose 7->18; y800/all-fused regressed 46.1866%/46.904%. Product was removed at
`d357be92b`. Do not resurrect that scalar loop.

For exact Qwen production state `TState=bf16`, lane `wk` already owns logical
columns `wk*16..wk*16+15`: a contiguous 32-byte range. The state allocation and
128-element BF16 row stride are at least 16-byte aligned, and each lane start
adds a multiple of 32 bytes. The existing scalar `Store(__nv_bfloat16*,...)`
uses `__float2bfloat16`, round-to-nearest-even (`cuda_gdn.cu:209-219`).

**Hypothesis:** convert `rr[0..15]` elementwise with that exact operation, pack
each group of eight into a 16-byte value, and issue two aligned vector stores
per lane directly to its contiguous logical columns. This can remove the final
shared write/barrier/reread without the scalar-direct arm's sixteen store
instructions per lane. If nvcc scalarizes or spills the pack, alignment is not
proved, state bytes differ, or either timing leg loses, reject and remove it.

Pinned vLLM remains grounded by its FLA register tile and direct persistent
store at `${VLLM_SOURCE}/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:156-170`.
That establishes register-to-state lifetime, not this transport schedule.

## Exact candidate contract

Add strict default-OFF `VT_GDN_DECODE_BF16_VECSTORE=1`; only exact `"1"`
selects it. Eligibility is the complete regular REGSTATE production path:
BV16, shared swizzle, Dv=Dk=128, NW8, non-speculative fused decode, and
`TState=__nv_bfloat16`. F16/F32 state, partial shapes, alternate BV/NW/layout,
sequential fallback, invalid/null index paths and every unset/invalid spelling
remain incumbent REGSTATE.

Use a host/device constexpr mapping/alignment contract for lane, pack and
element:

```text
logical_column(wk, pack, element) = wk*16 + pack*8 + element
pack in {0,1}, element in [0,8)
byte offset from state value-row = logical_column * 2
```

The 128 columns must be a bijection; each pack begins at a 16-byte boundary,
the two packs cover exactly the lane's current `rr[16]`, and four 8-lane
subgroups in a warp address four independent value rows. Form each packed value
with the same per-element BF16 RN conversion as scalar `Store`; use a bit-safe
aligned aggregate (`int4` or equivalent), never type-pun through an unaligned
object. Static assertions must bind element size, pack size, Dk and alignment.

Only replace REGSTATE's final `rr` shared write, final barrier, and shared
reread/writeback. Preserve recurrence expression/order, shuffles, output store,
initial coalesced state load/barrier, q/k staging, state row/index/null
resolution, conversions, gx8/bx128/smem9728 and every fallback.

An 8-lane register transpose is a **fallback discriminator**, not the first
implementation. If the vector arm fails only because compiler output cannot
issue aligned vector stores, write a new spike for two 8x8 half-transposes that
make each scalar store instruction address adjacent subgroup columns. Do not
combine transpose and vector packing in this row: the extra shuffles obscure
which transport won.

## Red-first tests and mutations

Before product work, add portable tests for strict parsing, every eligibility
predicate, exactly one incumbent/vector callback, the complete 128-column
mapping bijection, two disjoint aligned packs per lane, and BF16 pack bytes
matching sixteen independent reference RN conversions. Capture the intended
red result before adding selector/product support.

Extend CUDA REGSTATE-versus-vector coverage to require byte-exact output and
the complete persistent state for compact/indexed storage, valid/null indices,
production BF16 input/output/state and every ineligible dtype/shape fallback.
The graph test must distinguish the vector callback and still observe exactly
one gx8/gy800/bx128/smem9728 node. Cached Qwen tokens must match the hash above.

Scratch mutations must kill parser, each eligibility predicate, lane/pack
addressing, either pack store, BF16 conversion, indexed row, null handling and
callback identity. Replacing vector writeback with incumbent shared writeback
must fail compiler discrimination. Restore the tree byte-for-byte after each
mutation.

## Compiler, correctness, and performance gates

Compare exact REGSTATE and vector specializations from one immutable nvcc
binary. Record compiler/binary hashes, PTX/cubin/SASS symbol identity,
registers, local/stack/spills, barriers, total instructions, shared/global
load/store instructions, and actual-function occupancy. The vector arm must:

- contain two 16-byte vector global state stores per lane (or an equivalent
  lower instruction form), with no scalarized sixteen-store sequence;
- materially reduce global-store instructions versus removed scalar-direct
  (18) and reduce shared loads/stores plus barriers versus REGSTATE;
- have zero local/stack/spills, unchanged geometry/shared bytes, no occupancy
  loss over 12.5%, and no total SASS growth over 5%.

Missing symbols, unproved alignment, scalarization, compiler no-op, or any
resource miss rejects/removes the candidate before performance credit. When
authorized, same-tool NCU on matched y800 nodes records global sectors and
transactions, shared traffic, barrier/scoreboard stalls, DRAM and occupancy for
REGSTATE/vector/pinned-vLLM; counters diagnose but never substitute for timing.

After exact portable, CUDA, graph and cached-Qwen correctness, hold one GPU lock
and run same-binary `REGa -> VECa -> VECb -> REGb`. Both VEC raw y800 legs must
beat both controls; counterbalanced y800 must improve at least 1.00%; all fused
decode must improve; total/output throughput, TTFT, TPOT/ITL, E2E, GPU memory
and material host PSS must not regress. Reproduce a passing result with the
same binary. Any miss removes selector/product/tests and records the rejection.
A pass remains explicit opt-in: it grants no default, release, pinned-vLLM or
unavailable 27B/35B acceptance.

## First implementation step

Add only portable parser/eligibility/mapping/pack-reference tests and capture
their red failure. Do not edit CUDA recurrence code until that evidence exists.

## Outcome — rejected and removed

Reviewed implementation `29de225c8` plus repair `3ca49e926` passed the fresh
mutation re-review. The repaired portable suite ran 16 cases / 2,568 assertions;
independent mutations killed every eligibility predicate, BF16 RN conversion,
either pack, a hardwired-false dispatch, the CUDA route and the vector store.
The operator rebuilt the CUDA translation unit and passed
`test_gdn_decode_fused` plus `test_ops_gdn`. The immutable four-leg run was
token-exact throughout (SHA-256
`be20ffbceb61f0264ca21d972bfc5fc51e855e64f2b945de71669cae666aa702`).

| Arm | all fused decode, 1,656 calls | grid-y 800, 864 calls | total / output tok/s | TTFT | TPOT / ITL | E2E |
|---|---:|---:|---:|---:|---:|---:|
| REG-a | 251.210058 ms | 139.127859 ms | 6760.82 / 747.59 | 1027.95 ms | 34.81 ms | 5449.30 ms |
| VEC-a | 250.472657 ms | 138.702659 ms | 6802.32 / 752.18 | 1015.76 ms | 34.66 ms | 5417.91 ms |
| VEC-b | 251.214368 ms | 139.138796 ms | 6804.21 / 752.39 | 1015.60 ms | 34.65 ms | 5416.35 ms |
| REG-b | 251.404693 ms | 139.111428 ms | 6807.51 / 752.76 | 1013.82 ms | 34.65 ms | 5413.73 ms |

The counterbalanced means improve y800 only **0.1430%** and all fused decode
only **0.1846%**. VEC-b is slower than both controls on y800 and slower than
REG-a on all fused decode, so the candidate fails both the required 1.00% gain
and the every-vector-leg stability gate. The enclosing differences are run
order drift: the final REG-b also beats both vector legs on every reported
wall axis. `nsys` confirms unchanged gx8/gy800/bx128, 56 registers/thread,
9,728 dynamic shared bytes and zero local bytes in every arm; static inspection
shows two 16-byte stores, but the old PTX artifact predates the repaired pack
seam and receives no current-head codegen credit.

Raw roots are
`/tmp/qwen35-gdn-vecstore-{rega,veca,vecb,regb}-3ca49e926.*`; the GPU was idle
before the locked, cgroup-contained series (RTX 5070 Ti, driver 595.71.05,
P8, 40 C, 0% utilization). The experiment is therefore rejected and its
selector/product/tests are removed. REGSTATE remains the accepted opt-in; no
default, release, pinned-vLLM, 27B or 35B claim changes.
