# sm_120 Qwen3.5 fused GDN decode BV16/NW8 shared-bank swizzle

**Lifecycle:** `COMPLETE` — accepted opt-in at product `824370396`; defaults
and release gates remain open.

**Owner rows:** `KERNEL-SSM-MAMBA`, `ROAD-V1-C2-LOCAL-BF16`

**Hardware/workload:** RTX 5070 Ti (`sm_120`), Qwen3.5-4B plain BF16

**Authority:** records and docs only in this checkpoint; no product, tests, GPU,
remote operations or release-default change.

## Gap and ground truth

BV16 is the sole retained opt-in (`VT_GDN_DECODE_BV=16`); BV8, BV24 and BV32
RPT2 were exact but performance-rejected and removed. The latest
counterbalanced series measures BV16's dominant grid-y 800 call at **167.82
us/call** versus pinned vLLM **128.061 us/call** (**1.310x** slower), so the
residual remains open. The incumbent implementation is
`src/vt/cuda/cuda_gdn.cu:GdnDecodeFusedKernel` and its portable selector and
geometry contract is `src/vt/cuda/gdn_decode_fused.h`; coverage lives in
`tests/vt/test_gdn_decode_fused.cpp` and `tests/vt/test_ops_gdn.cpp`. Pinned
vLLM/FLA ground remains `fla/ops/gated_delta_rule/fused_recurrent.py` and
`fla/ops/gated_delta_rule/wy_fast.py` through vLLM's fused recurrent wrapper,
as recorded by the BV16 spike. This is a local layout discriminator, not a
claim that the candidate mirrors FLA's layout.

The present BV16/Dk128/NW8 schedule gives lane `wk` the 16 contiguous columns
`c=wk*16+j`. For each memory instruction those lanes cluster in banks, while
one warp contains four independent eight-lane value-row groups. State shared
rows use stride `dk+1=129`, so corresponding groups do not deliberately cover
disjoint bank ranges. The hypothesis is that an exact shared-only swizzle can
reduce conflicts without changing global traffic, arithmetic or launch width.

## Exact discriminator and layout

Add strict opt-in `VT_GDN_DECODE_SWIZZLE=1`: only exact string `"1"` enables
it. Unset, empty, whitespace, prefixes/suffixes, numeric lookalikes and every
other value select the incumbent layout. It is eligible only when the resolved
value tile is BV16, `Dv==Dk==128`, and resolved `NW==8`; every other geometry,
including partial tiles and all BV32/default paths, uses the old layout.

Specialize the kernel/launcher on a compile-time layout boolean. With
`ck=Dk/NW=16`, map logical column `c` to shared column
`sw=(c%ck)*NW+c/ck`. Thus lane `wk` at its existing local iteration `j` reads
and writes `sw=j*NW+wk`. Apply the same mapping to shared q and k; the four row
groups in a warp request the same q/k address and retain multicast behavior.
Use swizzled state row stride `Dk+NW=136`, shifting adjacent value-row bases by
eight banks so the four eight-lane groups cover disjoint ranges. The production
shared allocation is
`(2*128 + 16*136)*sizeof(float) = 9,728 bytes`; launch geometry remains grid-x
8, block-x 128. Keep the incumbent `dk+1` state stride and linear q/k layout in
the false specialization.

The load, two recurrence loops, writeback and output/state stores address the
same logical elements. Per-row operation order remains `j=0..15` for each lane,
so output and persistent state must be byte-exact. Resolve all constants and
address forms outside or at compile time: the recurrence inner loops contain
no division or modulo. Do not change input/output/state dtype conversion,
shuffle reduction order, global layout, null-slot semantics, cache indexing,
grid topology, default selection, BV16 selection, or any public surface.

## Red-first tests and mutation obligations

Extend the CUDA-free contract tests for strict parser behavior, eligibility and
fallback, the 128-column bijection, representative indices (including
`c=0,15,16,127` and `j*8+wk`), state stride 136, shared bytes 9,728, and the
unchanged grid-x 8/block-x 128 geometry. Prove partial Dv/Dk, NW other than 8,
BV32/default and invalid selectors retain the incumbent layout and allocation.

CUDA must compare swizzle OFF/ON byte-for-byte for both output and the complete
persistent state across the production 128/128 shape, partial/fallback shapes,
compact and indexed caches, negative/null indices, and the supported BF16/F32
activation/output and cache-state paths. Capture the public `vt::GdnDecode`
graph and require exactly one production kernel node with grid-x 8, block-x
128 and 9,728 dynamic shared bytes when both `BV=16` and `SWIZZLE=1` are set.
Run the full CUDA GDN and Qwen3.5 paged-forward gates after the focused tests.

Scratch mutations must make the focused gate fail when they corrupt or remove:
the swizzle mapping, state-row stride, production eligibility/binding, output
store, or state writeback. Record red-before implementation and restore every
mutation byte-for-byte.

## Performance gate and disposition

Build once, pin the exact executable/model/prompt/token SHA, acquire the local
GPU contention lock, and run graph-node `nsys` in counterbalanced order
`BV16a -> SWZa -> SWZb -> BV16b` on the identical cached Qwen3.5-4B BF16
128-request/128-output-token c32 workload. Record raw reports, kernel geometry,
registers, all-call and grid-y 800 sums/counts, token SHA-256, total/output
throughput, TTFT, TPOT/ITL, E2E, peak VRAM and peak host PSS.

Retain the swizzle only as an opt-in if both swizzled samples are byte/token
exact, the counterbalanced grid-y 800 mean improves by at least **1.00%** over
BV16, the all-fused-decode mean improves, both samples move in the winning
direction, and every enclosing axis is non-regressing (throughput may not
decrease; latency and memory may not increase). A passing 4B result does not
close default, release, Qwen3.6-27B or 35B gates. If any correctness, geometry,
microbenchmark or enclosing bar fails, remove the selector and all candidate
product/test arms while preserving the measured refutation in this spec and
the benchmark record. BV16 remains the incumbent in either disposition.

## Measured result — accepted opt-in

Spike `bdfaf823e` and product `824370396` implement the strict shared-swizzle
arm. Operator gates pass: portable **6/6 · 633**, CUDA GDN
**69/69 · 4,742**, and Qwen3.5 paged-forward **4/4 · 8**. Parser and mapping
negative mutations fail with 5 and 132 assertions respectively. All six
production token files have SHA-256
`83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.
Both arms execute grid-x 8 / block 128 at 56 registers; shared memory changes
from 9,280 to 9,728 bytes as designed.

The counterbalanced `BV16a -> SWZa -> SWZb -> BV16b` series has 1,704 total
and 816 grid-y 800 calls per arm. Swizzle improves all fused decode
**289.8737275 -> 268.254130 ms (-7.4555%)** and y800
**146.959362 -> 135.9178535 ms (-7.5123%)**. Both swizzle samples win.
Enclosing means also improve: total/output throughput
**6724.18/743.54 -> 6739.86/745.275 tok/s**, TTFT
**1027.34 -> 1023.005 ms**, TPOT/ITL **35.045 -> 34.98 ms**, and E2E
**5478.205 -> 5465.725 ms**.

The memory pair remains positive for the user-visible and device axes:
total/output **6784.64/750.23 -> 6793.05/751.16 tok/s**, TTFT
**1020.77 -> 1018.67 ms**, TPOT **34.72 -> 34.68 ms**, E2E
**5430.24 -> 5423.44 ms**, and GPU allocation **13,060 -> 13,054 MiB**.
Peak host PSS alone rises **1,909,259 -> 2,038,631 KiB (+6.78%)**, while
available memory drops less in the swizzle arm (**1,641,476 -> 1,529,724 KiB**
available-drop basis supplied by the run). This noisy transient host-load axis
is preserved as an acceptance caveat; the arm remains explicit opt-in and
receives no default or release credit.

Evidence is
`/tmp/qwen35-gdn-swizzle-{bv16a,swza,swzb,bv16b}-8243703.{nsys-rep,sqlite,log,tokens.json}`
and `/tmp/qwen35-gdn-swizzle-{bv16,swz}-mem-8243703.{jsonl,log,tokens.json}`.
Trace SHA-256 prefixes are `59bb6081` (BV16a), `082cd6d4` (SWZa),
`b44792c8` (SWZb), and `633fad31` (BV16b). The accepted swizzle y800 call is
**166.566 us** versus pinned vLLM **128.061 us**, leaving a **1.301x** residual.
Default, release, 27B and 35B gates remain open.
