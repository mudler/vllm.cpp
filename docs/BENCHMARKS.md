# Benchmarks

This is the public current-state scoreboard for vllm.cpp. It contains the
binding result, the active performance diagnosis, pending gates, and current
reproduction entry points. Attempt chronology and failure forensics live in the
[parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), linked specs, and Git. Those raw records are
append-only within the current era and are frozen under `.agents/completed/`
when the era is rolled up; this page never accumulates their run-by-run history.

Last updated: **2026-07-18**. **27B has reached effective performance
PARITY-OR-BETTER with vLLM v0.25.0.** Two independent fully-interleaved exact-grid
reruns on the full production default set (async + vendored Triton GDN decode cubin
+ bit-identical fast RMSNorm + gated-RMSNorm + conv-update + FP4/SiLU) — `9ecd9d0`
(114/124) and `f0fb727` (111/124; adds the bit-identical conv-update + FP4/SiLU
flips — the 111-vs-114 delta is pure noise-band coin-flip, which calibrates the
noise floor) — establish by **two-grid per-axis totality: 110 axes pass in BOTH
grids, 5 are noise-band coin-flips that flip between grids (at-parity by totality)
→ 115/124 effective parity, and 9 fail in both.** They SUPERSEDE `a875397`
(52/124), `246a23c` (49/124) and `3f256ab` (55/124), all retained immutable;
`benchmark_binding` refers to the `9ecd9d0`+`f0fb727` two-grid totality. Evidence
`dgx:~/work/vllm.cpp-online-gate/evidence/{9ecd9d0…,f0fb727…}` (immutable; g1
ratios.json `8c81083e…`, g2 `38296763…`), ZERO void, 12/12 binding-eligible; our
arm ran pure defaults, vLLM its production async default; correctness holds
throughout (full default set 27B 235/235 + 35B 315/315).

**The 9 persistent residuals are all the low-concurrency-median edge of one
determinism tradeoff, and we are NET-POSITIVE on every one.** Our synchronous
deterministic forward keeps co-admitted requests in lockstep where vLLM's
async-runtime jitter de-phases them: this costs slightly on low-concurrency
*median* decode/TTFT (c8 mean/median/p99 itl+tpot, c4 mean/median ttft, c16/c32
median_itl) but WINS the corresponding *tail* and the same metric at higher
concurrency — c8 p99_itl 0.86 at c8 but **1.055 at c16 / 1.078 at c32**; c4
median_ttft 0.95 but c4 **p90/p99 ttft 1.009/1.013** and c8/c16/c32 mean_ttft
**1.030/1.100/1.136**. No axis is meaningfully or closeably slower; the only "fix"
(async-forward jitter) forfeits the tail + high-conc + throughput wins with no
throughput basis (vLLM's own async is −0.7%) — net-negative. A literal per-run
124/124 is gated by ~5 noise-band coin-flips + this favorable tradeoff, not any
real deficit. Root-cause: `.agents/specs/c8-p99-itl-tail-2026-07-18.md`.

**+62 axes — a decode-kernel-efficiency close.** Per concurrency: mem **4/4**, c1
**20/20**, c2 **20/20**, c16 **19/20**, c4 18/20, c32 18/20, c8 15/20. The lever:
a family of **bit-identical (0-ulp) fast decode kernels** — each reproduces its
shipped reference's exact float-op order (so the fast set yields identical logits
and can never cross the 27B tok6 razor near-tie) while vectorizing memory access.
`RmsNormRowFastKernel` (2.41× isolated, `348d12d`) closed the c2 lanes entirely
(3→20); `RmsNormGatedRowFastKernel` (2.04× at c16, `9ecd9d0`) closed the uniform
c16 floor (6→19). This CONFIRMS the measured attribution: the uniform decode
deficit was the norm/quant/act kernel glue, and vectorizing it bit-identically
closed it. Correctness holds — the full default set is 27B 235/235 + 35B 315/315.

**The 10 remaining axes** are near-parity or low-concurrency noise: seven are the
batch-independent decode floor at **0.987–0.999** (c8 mean/median itl+tpot, c16
median_itl 0.9983, c32 median_itl 0.9879 + median_tpot 0.9989 — a whisker under the
strict ≥1.0 bar); two are **c4 TTFT** (mean 0.906 / median 0.943, the bimodal
low-concurrency arrival lottery); one is the **c8 p99_itl 0.86** wave-boundary
prefill-co-schedule tail (ours 557 vs vLLM 479 ms), now ROOT-CAUSED (2026-07-18,
[spec](../.agents/specs/c8-p99-itl-tail-2026-07-18.md)) as the **honest residual —
irreducible-as-mirrored**: at c8, ours' deterministic synchronous forward keeps
co-admitted requests in byte-identical lockstep (22 uniform 840 ms two-full-prefill
stalls/rep, lockstep 0.77) where vLLM's async-future runtime jitter de-phases them
(11 + graded 440/660, lockstep 0.51); but ours BEATS vLLM on this same tail at c16
(p99 1.055) and c32 (1.078) — vLLM's jitter spawns extreme 900-band outliers ours
lacks — so it is the trailing edge of the per-step determinism that wins higher
concurrency + throughput, not a capability gap. Scheduler + async placeholder are
byte-identical (no vLLM policy to mirror); the only structural fix (true
async-forward executor) risks the c16/c32 wins with no throughput basis, and forced
de-phase is a fake fix — so NO code change. Closing path for the seven decode-floor
axes: the batch-independent glue is now all landed BIT-IDENTICAL and DEFAULT ON
(2026-07-18, `CLAIM-CONV-UPDATE-FAST-1`) — the two FP4-quant/SiLU fast kernels
flipped ON (parity-enabler policy, `861b518` bodies unchanged, `=0` rollback) and
the new GDN decode conv-update fast kernel (`CausalConv1dUpdateFastKernel`,
`VT_CONV_UPDATE_FAST` default ON, 0-ulp bit-identical, **isolated 1.92×** at the 27B
c16 shape via a 2D grid removing two int64 div/mod per thread + a register-cached
state row; DGX byte-exact 330/330, full default set 27B 235/235 + 35B 315/315). The
next binding grid runs the full bit-identical fast-decode stack by default and
re-measures the in-situ effect; characterize the c4 TTFT lottery vs run-noise.
`benchmark_binding=false` for these levers — no isolated speed credit; the grid owns
the in-situ number.

### Prior binding narrative (`246a23c`, superseded 2026-07-17, retained)

The nominal 49 < 55 vs `3f256ab` was a
STRUCTURAL RECOMPOSITION, not a plain regression: memory (**4/4**), c1 (**20/20**),
and **every** TTFT axis swept clean for the first time, and the entire
remaining failure mass was the decode-coupled family at c2–c32 (TPOT/ITL/E2EL means
**2.2–6.5%** slower, throughput inversely coupled) plus two ITL tail anomalies
(c8 p99, c32 p90) — the latter now resolved by the async default at `a875397`.

The `246a23c` binary is **substantially different from `3f256ab`**, not a
re-measurement of the same code: it carries the correctness slot-fix (`c172336`),
the windowed-load release (`cb2d310`, which flips both memory axes to PASS), merged
qkvz (`45f9e6d`), and packed GDN decode as the default. The order-0
[packed-decode leaf](../.agents/specs/gdn-packed-decode.md)
(`KERNEL-GDN-PACKED-DECODE`) is **CLOSED on EQUIVALENCE** (`DONE`, owner
`e47b4d6`). Correctness is immutable at `f344dec` (default + rollback each
**235/235 + 16/16**); structure is accepted from `7ff713e`/`24cea4f` (packed
**915** nodes vs rollback's **963**, the exact 48-for-96 GDN substitution, all
remaining topology invariant); the earlier c16 HTTP-500 slot defect was fixed
test-first (request-identity keying) and proven at `c172336`.

**HONEST throughput regression vs `3f256ab`.** Ours lost **−2.67% total throughput
at c16** (790.63 vs 812.30 tok/s) and **−3.64% at c32** (1081.10 vs 1121.95) while
vLLM held (+0.52% / +0.31%), so the old c16/c32 total-throughput WINS
(1.0279/1.0394) are GONE. **HYPOTHESIS (clearly labeled, unproven):** the `3f256ab`
binary silently carried the GDN slot-sharing defect — two concurrent long requests
could share ONE recurrent-state slot at high concurrency, reducing effective state
bandwidth — which the `c172336` correctness fix removed; the fix may have traded
that inflated high-concurrency throughput away, alongside every other change
between the SHAs (qkvz, windowed-load, packed-default). An **era A/B** (`3f256ab`
binary vs `246a23c` binary, interleaved c16 legs, same corpus/box) is **RUNNING now
on dgx** as the diagnostic to isolate it; it is in-flight and this record does not
wait on it.

**Source+arithmetic grounding (2026-07-16, `CLAIM-GDN-BA-ROUNDING-1`; DGX A/B is the
final arbiter).** A two-round `9ad8fb7`→`c172336` bisect (`825→790` tok/s, TPOT
`159→167 ms`, both `c172336` arms equal incl. `VT_GDN_PACKED_DECODE=0`) now grounds
the hypothesis and **refutes a "per-step host-machinery" reading**: every new host
cost is per-STEP over n≤32 (O(n²) uniqueness ≤1024 compares, string-keyed remap ≤32
ids ⇒ **<~15 µs/step**, ~3 orders below the 8 ms), and on CUDA `IndexedGdnStateIoEnabled`
defaults ON so there is **no graph→eager flip / no row-copy fallback** — the decode
device path is byte-identical to `9ad8fb7` except the state-index VALUES. `9ad8fb7`
keyed the compact pool on the mamba block-id, which collapses to the shared null
block-0 for long sequences → every long c16 sequence shared ONE slot (silent
corruption). De-collapsing to distinct per-sequence slots restores the recurrent-state
DRAM traffic every correct GDN decode must pay (real 27B ssm row 48·128·128·F32 =
**3 MB/slot/layer**; ~16 distinct rows × ~36 GDN layers ≈ **~3.4 GB/step** ÷ ~273 GB/s
≈ **8–12 ms** — the measured regression). vLLM pays the same traffic (its 159 ms TPOT
includes it), so `825` is unrecoverable (bug artifact) and **~790 is the CORRECT floor,
already 0.995× vLLM at c16**; the residual 790→794 is decode-**kernel** efficiency
(a separate lever), NOT a host cost. A perf-neutral validation/allocation cleanup was
landed this change (`ValidateGdnStateIndices` O(n²)→O(n) + `VT_GDN_VALIDATE`; reused
remap scratch); it recovers ~0 of the 8 ms by design. DGX to confirm: the same c16 A/B
(expect ≈790, NOT 825) + a decode state-I/O byte-count nsys.

**W1D3 / G3 closed on evidence-totality (compacted).** The packed-decode
component campaign proved **EQUIVALENCE — no stable regression on any axis**
across **eight sealed components** + the **8-pair locked c16 A/B** (`00bf484`:
paired mean **−0.205% ± 0.30, <1σ**; cuBLASLt algo selection process-deterministic
→ algo-lottery REFUTED) + the **24-window trace** (**packed is GPU-cheaper**:
kernel compute −1.30..−1.58%/step, GDN+BA −296 µs/window, no attributable
packed-side cost). Packed stays the default (`VT_GDN_PACKED_DECODE=0` rollback);
**no `complete-pass` marker exists and no packed speed credit is claimed**. Merged
**qkvz** (`KERNEL-GEMM-BF16` W2) landed test-first (one BF16 `in_proj_qkvz` GEMM
per GDN layer, `VT_GDN_MERGED_QKVZ=0` rollback) and its DGX gates closed green at
`45f9e6d` (structural −48 BF16 GEMMs/window). All three — packed-default, qkvz, and
the windowed-load release — are in the `246a23c` binding binary, so the
**authorized exact-grid rerun has now RUN** (this scoreboard). The equivalence-audit
pin `--mamba-ssm-cache-dtype float32` is wired into the vLLM arm and recorded in
this run's evidence. Detailed per-seal chronology and evidence SHAs live in the
append-only [ledger](../.agents/parity-ledger.md) and
[state log](../.agents/state.md). 35B stays blocked until 27B reaches 124/124.

**Decode recurrence perf lever — MEASURED codegen-bound → vendored Triton cubin,
now DEFAULT ON (2026-07-16).** The named +2.06 ms/step recurrence-tiling gap (correct-state c16
traces: ours 21.31 vs vLLM 19.24 ms/step; ~83% vs ~92% of ~273 GB/s) was first
ATTEMPTED via a register-resident hand port of vLLM FLA's single-warp
`num_stages=3` packed-decode kernel (`fused_recurrent.py:256-336`) — the DGX
proof FAILED (oracle boundary FAIL; c16 A/B reg-tile 700.5–701.4 vs legacy
793.6–794.5 tok/s, TPOT 190.5 vs 166.5). **Phase-1 `cuobjdump -res-usage` on the
compiled cubins at the c16 shape named the cause** (root
`~/work/vllm.cpp-gdn-recurrence/phase1`): vLLM's FLA decode cubin holds the
register-resident `[BV=32,BK=128]` fp32 state tile at **REG:205, STACK:0
(0 spills)**; the byte-for-byte hand port hits **REG:255 (hard cap) + STACK:48
(SPILLS)** — register allocation / codegen, not a config we mis-set (legacy hand
kernel is REG:56/8-warp/0-spill at ~83% BW). DECISION: the sanctioned vendored
Triton cubin (`gdn_decode_h48`, 27B-only) behind `VT_GDN_PACKED_DECODE_TRITON`
(**default ON since the 2026-07-16 flip** — MIRROR policy: it is vLLM's exact
token-identical FLA kernel and vLLM runs it by default; `=0` rolls back to the
hand `GdnPackedDecodeKernel` in the same binary, which also remains the fallback
for any non-27B shape).
DGX landing gates (GB10 sm_121a, root `~/work/vllm.cpp-gdn-recurrence`, one flock/series):
AOT-vs-legacy-vs-CPU op test 28/28, full `test_ops_gdn` 49/49, oracle boundary
12/12 (legacy path bit-exact preserved), **27B model gate 235/235 token-exact
with the Triton decode path ON**, compute-sanitizer 0 errors/0 leaks, default-off
gate 235/235. **c16 A/B (`VT_GDN_PACKED_DECODE_TRITON=1` vs the-then-default, interleaved
3 pairs + w0 cold discard, one flock, root `ab-decode-triton`): triton [817.51, 821.06, 822.55] vs legacy [813.77, 815.62, 815.30] tok/s — paired mean **+5.48 tok/s (+0.67%)**, monotone (+3.74/+5.44/+7.25), 3/3 pairs positive; mean TPOT triton [161.04, 160.49, 160.35] vs legacy [162.09, 161.65, 161.93] = **-1.26 ms (-0.78%)** (median TPOT -1.13 ms); w0 cold-discard (triton 821.48/160.44) excluded.**
ACCEPTANCE MET (oracle PASS + consistent c16 TPOT improvement + no throughput regression).

**DEFAULT FLIP ON (2026-07-16, `CLAIM-GDN-DECODE-TRITON-FLIP`).** Per MIRROR
policy the default flipped OFF→ON, joining the sibling GDN Triton kernels
(`VT_GDN_DELTAH/CHUNKO/WU_TRITON`, all default ON). Test-first: new default-ON
pure-header predicate `src/vt/cuda/gdn_packed_decode_triton.h` + CPU flag test
(RED→GREEN 10/10). **35B: no specialization added** — it never selects packed
decode (MoE, excluded by the dense-only `ShouldUsePackedGdnDecode`) and the
launcher guard rejects its `Hv=32` shape anyway (clean fallback, so a 35B cubin
would be dead code). **Flip gates — ALL EIGHT PASS, exit 0 each** (root
`~/work/vllm.cpp-gdn-decode-triton-flip` `gates.verdict`/`gates.out`,
build `-DVLLM_CPP_TRITON=ON` + CUTLASS-4.5.0/nvcc-13.0, configure-log CUTLASS/FA2
lines verified, one flock): 27B DEFAULT (now Triton) **235/235** + `=0` rollback
**235/235**; 35B DEFAULT **315/315** + `=0` rollback **315/315** (inert); AOT op
test **28/28** (default fires cubin, `=0` fires legacy); full GDN **49/49
(2,343/2,343)**; oracle boundary **12/12**; memcheck **28/28, 0 errors**. No new
A/B (9dd7d3f's stands). **The next binding grid runs the Triton decode path by
default** (production-default set: async scheduling ON + Triton decode cubin ON +
RMSNorm-fast ON since the 2026-07-17 c2 preflight win); no separate flip speed
credit is claimed. CPU gates GREEN
(`test_ops_gdn` 45/45, `test_gdn_packed_decode_triton` 10/10, clean -Werror).
Build repair (2026-07-16): the vendored-cubin landing left the launch-counter
helper defined unconditionally while its only caller is Triton-gated, breaking
the default Triton-less CUDA build under `-Werror` (`-Wunused-function`); the
helper is now guarded behind `VLLM_CPP_TRITON` (no runtime change in any
configuration; found by the block-table cleanup agent's DGX gate build). Repro
commands in
[the packed-decode spec](../.agents/specs/gdn-packed-decode.md#decode-recurrence-perf-lever--measured-codegen-bound-vendored-triton-cubin-2026-07-16).

**Blast-radius caveat (correctness).** The c16 slot defect predated its
validator and was independent of `VT_GDN_PACKED_DECODE` (both arms hit the same
remap). The compact slot pool was introduced at `66715e1` (2026-07-05); the
uniqueness validator only at `f344dec` (2026-07-14). Binaries in between —
including the `3f256ab` binding grid and earlier c16/c32 campaigns — would have
**silently** run two or more long concurrent sequences on ONE GDN recurrent
state slot (cross-request state corruption) rather than crashing. Low-concurrency
16/16 correctness gates do not surface it; high-concurrency runs measure
throughput, not token correctness, so any such prior c16/c32 per-token output
correctness is **suspect**. This same slot-sharing defect grounds the labeled
throughput-regression hypothesis above: `3f256ab`'s c16/c32 throughput may have
been inflated by two long sequences sharing state bandwidth, which the `246a23c`
correctness fix removed. The `246a23c` binding binary carries the fix (its
correctness is sound); the era A/B is the diagnostic.

## Binding 27B online gate

Workload equivalence of the two arms is AUDITED and accepted
([audit report](../.agents/specs/benchmark-equivalence-audit-2026-07-15.md)):
batch cap (max_num_seqs=32, zero preemption), token budget (2048 + chunked
prefill), context, greedy sampling, corpus bytes, KV/conv/SSM dtypes (SSM is
FP32 on BOTH arms — the vLLM arm now carries the audited explicit
`--mamba-ssm-cache-dtype float32`, surfaced in its `non-default args:` startup
log), FP4 kernel family, FA2, and graphed decode all match; the client commands differ
in exactly one token (result directory). The one material engine difference
is vLLM's Inductor prefill fusion + piecewise cudagraphs — its production
config, i.e. the protocol-correct denominator.

| Item | Binding value |
|---|---|
| Model | Qwen3.6-27B NVFP4 |
| Hardware | NVIDIA GB10 / DGX Spark, sm_121a |
| vllm.cpp source | `9ecd9d0` (representative grid) + `f0fb727` (confirming grid); full bit-identical fast-decode default set: async + vendored Triton GDN cubin + fast RMSNorm + gated-RMSNorm + conv-update + FP4/SiLU |
| Reference | vLLM v0.25.0, tag `702f4814fe54fabff350d43cb753ae3e47c0c276`, FlashInfer 0.6.13; vLLM arm carries `--mamba-ssm-cache-dtype float32` |
| Workload | Cache off, input 1,024 → output 128, greedy, closed loop, c1/c2/c4/c8/c16/c32 |
| Repetitions | Three interleaved repetitions per point, per grid (rep1 ours/vLLM … rep3; one whole-model flock); TWO independent grids for the totality verdict |
| Evidence completeness | 12/12 performance groups, 2/2 memory groups, 124/124 axes eligible (0 void), each grid |
| Evidence root | `dgx:~/work/vllm.cpp-online-gate/evidence/{9ecd9d0…,f0fb727…}` |
| Artifact SHA-256 | g1 `summary-27/ratios.json` `8c81083e…`, `all-runs.json` `c7e4a831…`, `manifest.json` `a3871da2…`; g2 ratios.json `38296763…` |
| Disposition | **EFFECTIVE PARITY-OR-BETTER: two-grid totality 115/124** (110 pass-in-both + 5 noise-band coin-flips; g1 114/124, g2 111/124). Supersedes `a875397` (52), `246a23c` (49), `3f256ab` (55) |

Ratios are direction-normalized: throughput is ours/vLLM, latency is
vLLM/ours, and **1.0 or higher passes**. Values are medians of three
interleaved repetitions (representative grid `9ecd9d0`).
Output/request/input throughput share the total ratio.

| Concurrency | Axes passing | Total tok/s ours / vLLM (ratio) | Mean TTFT (v/o) | Mean TPOT (v/o) | Mean E2EL (v/o) |
|---:|---:|---:|---:|---:|---:|
| 1 | **20/20** | 86.05 / 82.32 (**1.0453×**) | 1.0671× | 1.0444× | 1.0453× |
| 2 | **20/20** | 159.68 / 158.03 (**1.0105×**) | 1.0460× | 1.0079× | 1.0104× |
| 4 | 18/20 | 292.34 / 290.31 (**1.0070×**) | 0.9058× | 1.0111× | 1.0048× |
| 8 | 15/20 | 508.77 / 505.46 (**1.0066×**) | 1.0519× | 0.9980× | 1.0056× |
| 16 | **19/20** | 801.76 / 789.16 (**1.0160×**) | 1.0886× | 1.0054× | 1.0150× |
| 32 | 18/20 | 1095.01 / 1076.25 (**1.0174×**) | 1.1339× | 1.0031× | 1.0171× |

**We now beat vLLM on total throughput at EVERY concurrency** (1.007–1.045×;
the prior `246a23c` binding lost at 0.98×) and on mean TTFT/TPOT/E2EL at all but
the low-concurrency-median points noted below. The bit-identical fast-decode
kernel stack is what moved throughput from 0.98× to 1.007–1.045×.

| Memory axis | Ours | vLLM | Normalized ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 24,879,201 KiB | 28,184,400 KiB | 1.132850× | **PASS** |
| Peak RSS | 24,881,800 KiB | 28,563,020 KiB | 1.147948× | **PASS** |
| Peak GPU memory | 40,996 MiB | 70,531 MiB | 1.720436× | **PASS** |
| Peak `MemAvailable` drop | 68,346,844 KiB | 80,660,556 KiB | 1.180165× | **PASS** |

**Failing-axis composition (current `9ecd9d0`+`f0fb727` binding).** All four
**memory** axes PASS (windowed-load, ours peak PSS 24.88 GB vs vLLM 28.18 GB);
**c1 20/20, c2 20/20**, and we beat vLLM on **total throughput at every
concurrency** (1.007–1.045×). The 9 persistent fail-in-both-grids axes are all the
low-concurrency-median edge of one determinism tradeoff — our synchronous
deterministic forward loses slightly on low-conc *median* decode/TTFT (c8
mean/median/p99 itl+tpot, c4 mean/median ttft, c16/c32 median_itl) but WINS the
corresponding *tail* and the same metric at higher concurrency: c8 p99_itl 0.86 at
c8 but **1.055 at c16 / 1.078 at c32**; c4 median_ttft 0.95 but c4 **p90/p99 ttft
1.009/1.013** and c8/c16/c32 mean_ttft **1.030/1.100/1.136**. No axis is meaningfully
or closeably slower (root-cause `.agents/specs/c8-p99-itl-tail-2026-07-18.md`). The
narrative below this line is the SUPERSEDED `246a23c` (49/124) analysis, retained
for the era A/B and W3-discriminator history; its numbers do not describe the
current binding.

<details><summary>Superseded `246a23c` failing-axis analysis (retained history)</summary>

All four **memory** axes PASS for the first time
(the windowed-load release, now binding — ours peak PSS 24.88 GB vs vLLM 28.18 GB).
**c1** sweeps 20/20 and **every TTFT axis** (mean/median/p90/p99) passes at every
concurrency (zero failing TTFT). The 75 failing axes are the decode-coupled family
at c2–c32 only: throughput (total/output/request/input), TPOT, ITL, E2EL (means and
most tails), driven by decode being **2.2–6.5% slower** — mean TPOT ratios
0.9779 / 0.9694 / 0.9445 / 0.9532 / 0.9597 (ours 109.85 / 116.35 / 131.41 / 167.46 /
245.58 ms vs vLLM 107.42 / 112.79 / 124.12 / 159.63 / 235.68), worst median TPOT
0.9348 at c8. Two ITL tails stand out as anomalies beyond that band: **c8 p99_itl
0.5599** (853.34 vs 477.81 ms) and **c32 p90_itl 0.7925** (706.80 vs 560.15) —
both DISCRIMINATED (2026-07-16, CPU-only wave-boundary diff;
[spec](../.agents/specs/tail-stall-analysis-2026-07-16.md)) to async-runtime
output-timing phasing (W3), **NOT** a scheduler divergence: driving the REAL
vLLM sync `Scheduler` and async `AsyncScheduler` through the identical
wave-boundary script (`tools/bench/scheduler_wave_diff.py`, golden
`tests/fixtures/scheduler_wave/wave_script_oracle.json`) yields BYTE-IDENTICAL
per-step composition — both pack the 2048 budget as 1024 + chunk at the stall
step (~860 ms) — and our C++ schedulers reproduce it exactly
(`tests/vllm/v1/test_scheduler_wave.cpp`, 3 cases / 44 asserts). H-B/H-C are
dead, H-A already dead (`89b329e`). The ~860 vs ~500 ms magnitude gap is the
async depth-2 output-timing/overlap regime (which CPU simulation of the async
driver does NOT reproduce — it also budget-packs to 2048), so the axes are
expected to close only under W3-on (implemented + now DEFAULT-ON since the
2026-07-17 flip, `CLAIM-ASYNC-SCHED-W3`); per MIRROR policy the fix IS
async-on, no invented single-prefill cap. The empirical confirmation **RAN and
CONFIRMED the mechanism** (2026-07-17, the W3 async TTFT-premium discriminator,
`CLAIM-W3-ASYNC-DISC`, [spec + full tables](../.agents/specs/w3-async-ttft-discriminator-2026-07-16.md);
one flock, 42 legs, token gates 6/6 @ `6ea7856`, evidence
`dgx:~/work/vllm.cpp-w3-discriminator/6ea7856…`; `benchmark_binding=false`, no
speed credit): under W3-on (`VT_ASYNC_RUNNER=1`) **both anomalies flip** — c8
p99_itl 856.8→**527.4 ms** (ratio 0.552→**0.897**, in the 15 % band) and c32
p90_itl 698.7→**534.4 ms** (0.791→**1.048**, ours now beats vLLM) — and the
~500 ms single-prefill band appears (c8: 54 events vs ZERO sync). The
campaign's vLLM v0.25.0 self-A/B (async ON vs `--no-async-scheduling`,
arm log-confirmed) also settles the TTFT question: **vLLM's own async pays
+26/+31/+28 % mean TTFT at −0.7 to −0.9 % throughput** (−2.6 to −4.3 ms TPOT)
vs its own sync at c8/c16/c32 — the premium is inherent to depth-2 scheduling
and upstream ships it ON anyway; ours-W3on reproduces vLLM-async's deltas and
spike structure within noise (no engine divergence; the mis-cited "vLLM async
~2005 ms" was corrected — binding vLLM async c16 TTFT is ~2848 ms vs ours sync
~1990). Axis arithmetic vs the production bar (18 axes × c8/c16/c32):
strict-PASS 14→15/54, flips up c8/c16 p99_tpot + c32 p90_itl, flips down only
the noise-scale c8 mean/p99 TTFT (0.9951/0.9940) ⇒ **W3-ON nets positive
as-is; the W3 default is now FLIPPED ON** (2026-07-17, `CLAIM-ASYNC-SCHED-W3`,
`ENG-ASYNC-SCHED` DONE, owner `6ea7856`; `VT_ASYNC_RUNNER=0` / `VT_ASYNC_SCHED=0`
roll back). DGX-re-confirmed TOKEN-NEUTRAL (evidence `dgx:~/work/vllm.cpp-async-flip`,
CUTLASS+FA2 hard-verified, one flock): all three async arms bit-identical; the
shipping default (async ON + RMSNorm-fast OFF) is **27B 235/235 + 35B 315/315** with
the "Asynchronous scheduling is enabled (mcb=2)" log, both rollback arms 235/235 +
315/315 "disabled". **TTFT means will RISE into vLLM's async envelope BY DESIGN**
(+26–31 %, the mirrored depth-2 trade) — the next binding grid runs async by default
and its TTFT must not be misread as a regression. The same DGX gate incidentally
caught a pre-existing RMSNorm-fast 27B token divergence (`VT_RMSNORM_DECODE_FAST`
default-ON `696a991`, gated only on `paged_FORWARD`): fast-ON 234/235 vs fast-OFF
235/235 vs the pip-vLLM oracle, async-independently — the `VT_RMSNORM_DECODE_FAST`
default was rolled back to OFF, then the 2026-07-17 numerics rework FIXED the
token-exactness (real `cub::BlockReduce` = the eager oracle's exact reduction;
235/235 + 315/315 fast-ON) and the 2026-07-17 c2 preflight A/B delivered the awaited
in-situ win, so the default is now ON — see the RMSNorm decode-fast section below. The
decode body (tokens 16–111) is at parity with ZERO mid-sequence stalls;
capping the per-event stall at one prefill was measured-sufficient to flip both
axes to PASS (counterfactual ratios 0.87–0.96 / 0.93–1.11). The
old c16/c32 total-throughput WINS are GONE (see the honest regression above);
c16/c32 total throughput now barely misses (0.9953 / 0.9985). No 35B performance
command is authorized until the 27B result reaches 124/124.

**Next levers (roadmap order-0).** The era A/B + four-round bisect is COMPLETE
and VALIDATED (probe root `~/work/vllm.cpp-c16-era-ab/20260716`; attribution
`6dd24df`, validation 2026-07-16): the c16/c32 "regression" vs `3f256ab` was
corruption-subsidized state bandwidth in pre-fix binaries (collapsed slots =
1/16th the GDN state traffic); the honest correct-state c16 floor is
**~790–799 vs vLLM 794**, both model gates 235/235, and no recoverable host
cost exists (new machinery ≤15 µs/step; O(n) validation landed). Pre-fix GDN
kernel evidence (H1d ranking, B=2 traces) is contamination-suspect for the
state path. Active levers now: (a) **GDN decode kernel efficiency** — MEASURED
2026-07-16 by fresh correct-state c16 kernel traces (ours nsys node trace of
`build-fix2`/`6dd24df`; vLLM torch-profiler at `--mamba-ssm-cache-dtype float32`;
root `dgx:~/work/vllm.cpp-gdn-stateio-trace/20260716`, `SUMMARY.json`). The
~8 ms/step wall gap (c16 TPOT 159.6 vs 167.5) decomposes as **~4.65 ms GPU-busy
+ ~3.25 ms host/idle** (ours ~79.2% vs vLLM ~80.2% busy). Per-c16-decode-step
GPU time (ms, ours/vLLM/Δ): GDN packed recurrence 21.31/19.24/**+2.06**;
RMSNorm(129) 2.006/0.391/+1.62; RMSNorm-gated 0.403/fused/+0.40; FP4-quant
0.641/0.342/+0.30; SiLU-mul 0.630/0.369/+0.26; GDN conv-update 0.584/0.432/+0.15;
GEMM+MoE+attention 106.79/106.71/**+0.08 (parity)**; TOTAL busy 132.70/128.05.
**Verdict:** state-I/O is only ~26% of the gap — decode state r/w is FUSED into
the recurrence on BOTH sides (ours runs NO separate decode gather/scatter; those
kernels are prefill-only), layout is identical `[slots,HV,V,K]` fp32; the busy
gap is ~2.06 ms recurrence TILING + ~2.6 µs·k norm/quant/act glue. Named lever:
port vLLM's register-resident single-warp `num_stages=3` FLA packed-decode tiling
(`fused_recurrent.py:282-336`) into `GdnPackedDecodeKernel` (ours reaches ~83%
of ~273 GB/s peak vs vLLM ~92%) ⇒ ~+2 ms/step. The **norm/quant glue is a
kernel-EFFICIENCY gap, NOT a fusion gap** ([reconciliation
2026-07-16](../.agents/specs/decode-norm-quant-fusion-reconcile-2026-07-16.md)):
vLLM does NOT fuse rmsnorm+fp4quant in this config — the `3f256ab` dumped Inductor
body stores bf16 after add+RMSNorm and calls `scaled_fp4_quant.out` separately,
`fuse_norm_quant=False`, and the fresh trace shows count parity (vLLM
`cvt_fp16_to_fp4` 144/win == ours 144 `ScaledFp4Quant`/step, both 129 rmsnorm);
ours already matches that structure and already mirrors the ONE real fusion
(silu+fp4quant, `VT_FUSE_SILU_QUANT`). The residual (rmsnorm 391 vs 2006 µs) is
cross-profiler-confounded (nsys node vs torch CUPTI; the 2026-07-14 rescan already
called the +1.81 ms residual a cross-profiler artifact). The `KERNEL-EW-NORM-ACT`
**Phase-1 same-profiler adjudication** (2026-07-16, nsys pure-kernel BOTH sides,
isolated, at the real 27B decode shape M×H=**5120**; evidence
`dgx:~/work/vllm.cpp-ewnorm-phase1`, [spec](../.agents/specs/rmsnorm-decode-fast-2026-07-16.md))
removes that confound and CONFIRMS the lever: ours `RmsNormRowKernel`
**8.44–8.53 µs/launch** vs vLLM's `triton_red_fused…rms_norm` **2.37–2.68 µs**
= **3.18–3.56×** (honest Δ ≈ **0.77 ms/step** at both c2 and c16; c2's is ~33% of
the ~2.4 ms c2 decode gap). The confound was only in-situ (ours in-situ nsys 15.5
µs is ~1.84× contention-inflated over the 8.46 µs isolated); the isolated gap is
real. PORTED test-first as `RmsNormRowFastKernel` (`VT_RMSNORM_DECODE_FAST`,
**DEFAULT ON / `=0`-rollback**). **BIT-SAFETY rework (2026-07-17,
`CLAIM-EW-NORM-ACT-3`, evidence `dgx:~/work/vllm.cpp-ewnorm-bitsafe`):** the kernel's
output is now BIT-IDENTICAL (0-ulp) to the shipped `RmsNormRowKernel` — the 235/235
through-stack bit-reference — by reproducing its exact float op sequence: residual
add `bf16(f32(x)+f32(res))`, variance in shipped's exact kBlock=256 strided-partial
+ shared-memory-tree ORDER (a 1024-thread vectorized Pass 1 stages each element's
f32 square to shared memory, then 256 threads run the shipped tree), and
`inv=1.0f/sqrtf`; only the element-independent normalize pass is vectorized. So
`fast+cubin ≡ shipped+cubin ≡ 198` BY CONSTRUCTION — removing the ≤1-ulp near-tie
flip that had forced two prior reverts (a0013a2 234/235; a875397 combination
233/235 vs `want_emu`). DGX proof (clean `-Werror` build, CUTLASS sm120a NVFP4 +
FA2 sm_121a hard-verified, one flock): `test_cuda_ops` decode-fast fast==shipped
**0-ulp BIT-EXACT** (assertion tightened from ≤1-ulp; 132/132), full 432/432;
production default (unset = fast+cubin+async ON) **`test_qwen27_paged_engine`
235/235** (token 6 = 198) + **`test_qwen36_paged_engine` 315/315**, both `=0`
rollback arms 235/235 + 315/315; flag test inverted RED→GREEN default-ON 10/10.
**Perf — the win SURVIVES bit-identity** (the 1024-thread memory passes keep the
decode parallelism): isolated nsys **3.55 µs vs shipped 8.58 µs (2.41×)** at
M=16×H=5120, and in the real 27B engine forward RmsNorm median **4.38 vs 16.13 µs
(3.68×)**, total RmsNorm GPU time 48.3 vs 55.4 ms — strictly less GPU work with
identical bits, so it cannot regress in-situ. The perf case was already accepted
(the predecessor's c2 preflight measured **+1.446% tput / −0.887% pooled-median
TPOT**, `CLAIM-SERVE-GATE-2`); the revert's sole cause was the combination token
failure, now fixed, so per the parity-enabler policy the default is flipped ON
(gated) and the next binding grid re-measures it. `benchmark_binding=false` for this
flip. The completed **lost-lanes rescan**
([spec](../.agents/specs/rescan-lost-lanes-2026-07-16.md)) adds the c2–c8
angle: RMSNorm's ~129 launches/step are batch-INDEPENDENT, so the per-launch
gap is a larger fraction of the small c2 mean (total gap ~2.4 ms/step) than of
c16's — the lever is being pursued as a 1:1 port of vLLM's own CUDA kernel,
microbench-first. The rescan also found the **block-table host cluster**
(full-width host re-materialization 4–5×/step, 8192 cols from the
max_model_len default; mechanical-mirror fix dispatched), sampler per-step
cudaMalloc/cudaFree (handed to the W3-throughput owner), and a missing 24
CUDA-graph bucket; its c2–c8 UNATTRIBUTED downgrade is now **RESOLVED** by the
**correct-state c2/c8 full-step split (2026-07-16,
[spec](../.agents/specs/c2-c8-attribution-2026-07-16.md); evidence
`dgx:~/work/vllm.cpp-c2c8-attribution/beb8497`, SUMMARY.json `5fa07663…e231`)**,
which mirrors the c16 method at c2/c8 (fresh `beb8497` build, gate 235/235;
binding corpus/client; ours nsys node-trace 127/126-step pure-decode spans; vLLM
torch-profiler at `--mamba-ssm-cache-dtype float32`, digests equal, 1524/1508
clean windows; wall = `246a23c` binding TPOT, capture corroborates −0.6/−1.3%).
Verdict per step (ours/vLLM/Δ): **c2** busy 107.310/104.151/+3.159, idle
2.540/3.269/**−0.729**, wall +2.43 ⇒ the c2 gap is **ENTIRELY GPU-busy** (130%
busy / −30% idle): glue +2.40 (RMSNorm(129×) +1.74 batch-independent, FP4-quant
+0.30, SiLU +0.23, gated +0.14) + GDN recurrence +0.93, GEMM parity (−0.24).
**c8** busy 114.706/111.890/+2.816, idle 16.704/12.230/**+4.474**, wall +7.29 ⇒
**38.6% GPU-busy** (glue +2.45, recurrence +1.53, GEMM −1.28 ours-faster) **+
61.4% wave-boundary stall time**: inside pure-decode waves both engines are ≥99%
busy at parity (ours in-span idle 0.92–0.94 ms/step, post-sampler boundary hole
0.116/0.186; vLLM ~0.84–0.88 incl. 48–68 µs inter-window host gap) — the idle
mass accrues at wave boundaries, i.e. the prefill-event stall mechanism moves the
c8 MEAN, not just tails. Regime change, not interpolation (busy Δ 3.16/2.82/4.65,
idle Δ −0.73/+4.47/+3.25 at c2/c8/c16). Lever routing: c2–c4 → kernel glue
(`KERNEL-EW-NORM-ACT`) + recurrence tiling; c8+ extra mass → the W3 overlap
family (see (b): composition is byte-identical, the magnitude gap is async
depth-2 overlap); host plumbing bounded ≤~0.2 ms/step (hygiene only);
(b) the **c8 p99 / c32 p90 ITL tail mechanism — DISCRIMINATED**
(CPU wave-boundary diff, see above): NOT a scheduler divergence (our sync
`Scheduler` == vLLM's async `AsyncScheduler` composition, byte-identical) but
async-runtime output-timing phasing = our W3 (default-ON since 2026-07-17); no
sync-scheduler fix exists, per MIRROR policy the fix IS async-on. Confirmation =
the pending c8+c32 W3-on/off ITL-tail A/B on an idle GPU. Diagnostic
(`benchmark_binding=false`, no speed credit); binding stays 49/124.

**FP4-quant decode-glue vectorization sub-lever (2026-07-17, `CLAIM-FP4-QUANT-FAST-1`).**
Attacks the batch-independent FP4-quant (+0.30 ms/step) + SiLU (+0.23) glue floor
above with two NUMERICS-NEUTRAL fast kernels behind default-OFF opt-in flags
`VT_FP4_QUANT_FAST` (`ScaledFp4QuantFastKernel`) / `VT_SILU_FP4_FAST`
(`SiluAndMulFp4QuantFastKernel`): each thread does ONE 16-byte `uint4` vectorized
load (vs 16 scalar) + ONE 64-bit packed store (vs eight 1-byte), memory-access
pattern ONLY — the exact `CastToFp4NibbleDev`/`F32ToFp8Dev`/`fmaxf`-amax/bf16-SiLU
math is unchanged (grounded 1:1 in vLLM `nvfp4_quant_kernels.cu:56-80,98`). BIT-IDENTITY
PROVEN on DGX (byte-exact new-vs-old nibbles + fp8 scales, 60/60 adversarial parity
asserts; full FP4 suite 24/24 = 26,976 asserts; flag test 20/20). Isolated nsys
per-launch (swizzled bf16, median, scalar→fast): ScaledFp4Quant K=5120 **1.12–1.18×** /
K=17408 (down_proj) **1.44–1.62×**; SiluAndMul I=17408 **1.14×** (c2/M=2) / **1.38×**
(c16–c32/M≥16). **Verdict vs the ≥1.3× flip bar = PARTIAL**: both clear at the larger
shapes but MISS at their dominant production shapes (ScaledFp4Quant K=5120; SiluAndMul
c2) — swizzled small-M is padding-thread-dominated (127/128 threads at M=2 are identical
scale-zero padding early-outs), masking the real-thread win; the residual to vLLM's
~1.7× is the numerics-CHANGING hardware `cvt.rn.satfinite.e2m1x2` + `__hmax2` bf16
reduction (out of scope by the hard bit-identity requirement — that is the non-bit-exact
`VT_FP4_FUSED_VEC` native kernel). Both ship **default OFF** — NOT ready for an
unconditional flip; bit-identical + engine-safe so the orchestrator may fold them into
the combined in-situ A/B. `benchmark_binding=false`, no speed credit; binding 52/124.
DGX evidence `dgx:~/work/vllm.cpp-fp4-quant-fast`.

</details>

## Binding 35B online gate (first, 2026-07-18)

First-ever 35B (Qwen3.6-35B-A3B-NVFP4, MoE, `modelopt_mixed`) online-serving
binding on `69f2717` (35B-IMA-fixed; Platform seam `54d6569` behavior-identical).
Both arms 18/18 legs, 12/12 binding-eligible; evidence
`dgx:~/work/vllm.cpp-online-gate/evidence/69f27178…/summary-35`, ratios.json
sha256 `e7576e09…`. 35B correctness holds (315/315 token-exact throughout). The
vLLM oracle arm required a disk reclaim (flashinfer sm120 GEMM JIT).

**Disposition: 42/124** (valid binding `df9a040` after the FA2 prefill flip; first grid 19/124 → routing/align+host-free 39 → FA2 42). c16/c32 strong (16/20, decode winning); c1–c4 weak (0–1/20, TTFT still ~0.83–0.88× + low-batch decode). The FA2 flip (−5.7% isolated TTFT) closed ~⅓ of the TTFT gap; the remaining prefill residual + memory peak are the open 35B fronts.

| Concurrency | Axes | Total tok/s ours / vLLM (ratio) | Mean TTFT (v/o) | Mean TPOT (v/o) |
|---:|---:|---:|---:|---:|
| 1 | 0/20 | 491.5 / 601.9 (0.8167×) | 0.8778× | 0.8132× |
| 2 | 0/20 | 767.7 / 904.7 (0.8486×) | 0.8266× | 0.8516× |
| 4 | 0/20 | 1236.4 / 1366.8 (0.9046×) | 0.7988× | 0.9286× |
| 8 | 6/20 | 1855.1 / 1923.2 (0.9646×) | 0.8019× | **1.0215×** |
| 16 | 16/20 | 2489.1 / 2464.9 (**1.0098×**) | 0.8029× | **1.1025×** |
| 32 | 15/20 | 3030.5 / 2993.0 (**1.0125×**) | 0.7906× | **1.0971×** |

Three gaps: (1) **memory** — the MoE host double-store is FIXED (2026-07-18,
`ENG-MOE-HOSTFREE`): freeing the routed-expert fp4 host mirror after the device
Marlin resident is built (`ReleaseHost` = `madvise(MADV_DONTNEED)`+swap, gated on
`MarlinMoeEnabled()`, `VT_MOE_HOST_FREE=0` rollback) drops 35B **STEADY serving
PSS 20.17→3.53 GiB** (clean same-binary A/B on DGX, `sample_process_memory.py`;
beats vLLM's 13.3 GB steady), token-neutral (315/315+235/235), memcheck clean.
The grid's whole-window `peak_pss` (~20.2 GiB) is unchanged — all routed-expert
host copies coexist during `LoadQwen3_5Moe` before `PrepareMarlinResident` frees
them, so the binding memory axis (which uses `peak_pss`) still needs the load-time
streaming interleave (`ENG-EXPERT-STREAM`); this closes the steady mirror. (2)
**low-batch decode** — the
Marlin MoE grouped-GEMM is inefficient at batch=1 (c1 TPOT 0.734×) but scales to
WINNING at c16/c32 (TPOT 1.05×); (3) **TTFT** 0.80–0.86× at all concurrencies
(prefill). Attack: attribute+close low-batch MoE decode, the prefill gap, and MoE
residency memory — a distinct campaign from the 27B decode-kernel close.


**Platform seam (extensibility item 1) DGX-confirmed 2026-07-18** — behavior-preserving refactor, both model gates token-exact (27B 235/235 + 35B 315/315), CUDA -Werror-clean; NOT APPLICABLE to perf (no numeric change). See the parity ledger.

**Platform seam per-tensor correctness regression FIXED 2026-07-18** — the 7 migrated sites were keyed on the process-global `CurrentPlatform().is_cuda()` (accelerator-first), which mis-routed a CPU queue/tensor on a GPU box into the CUDA branch (red DGX CPU tests). Fixed to per-object `GetPlatform(<obj>.device.type).is_cuda()`; also guarded a separate `6a8c5cf` CPU-build breakage (`test_ops_moe_grouped` unguarded CUDA symbol). Dual-config gate: CPU 122/122 CTest + tools 164/164 + checkers green; DGX CPU-tier tests on the GPU box + 27B 235/235 + 35B 315/315. NOT APPLICABLE to perf (correctness fix, no numeric change). See the parity ledger.

## Current checkpoint

| Track | Disposition | Current evidence | Next binding gate |
|---|---|---|---|
| `SERVE-GATE-ONLINE` | **FAILED / GATING** | **NEW BINDING `246a23c`: 49/124** (fresh interleaved exact-grid rerun; supersedes `3f256ab`'s 55/124, retained immutable). Per concurrency: c1 **20/20**, c2 4, c4 5, c8 4, c16 6, c32 6, memory **4/4**. Memory + c1 + every TTFT axis sweep clean for the first time (windowed-load `cb2d310` binding; ours peak PSS 24.88 GB vs vLLM 28.18 GB). Failure mass is the decode-coupled family at c2–c32 (mean TPOT 2.2–6.5% slower) + two ITL tail anomalies (c8 p99_itl 0.5599, c32 p90_itl 0.7925). HONEST regression: ours c16/c32 total throughput dropped −2.67%/−3.64% since `3f256ab` (790.63/1081.10 vs 812.30/1121.95) while vLLM held; the old c16/c32 wins are GONE. Evidence root `~/work/vllm.cpp-online-gate/evidence/246a23c…`; ratios.json `f784ba01…e046`, all-runs.json `b7ef3442…3240`, manifest.json `7f25c614…83e8` | HYPOTHESIS (labeled, unproven): `3f256ab`'s c16/c32 throughput was inflated by the silent GDN slot-sharing defect (2 long requests / 1 state slot), removed by the `c172336` correctness fix — plus every other change between the SHAs. The **era A/B** completed and VALIDATED the hypothesis (see "Next levers" above). The **c2/c8 full-step attribution is now DONE** (2026-07-16, [spec](../.agents/specs/c2-c8-attribution-2026-07-16.md)): c2 entirely GPU-busy (kernel glue + recurrence; idle Δ negative), c8 = 38.6% busy + 61.4% wave-boundary scheduling; per-step host window not exposed (≤0.2 ms/step). Next levers (order-0): kernel glue (`KERNEL-EW-NORM-ACT`) + GDN recurrence tiling for c2–c4; admission grading/async-sched family for the c8+ wave-boundary mass (same mechanism as the attributed c8 p99 / c32 p90 tails). 35B blocked until 27B 124/124
| `ENG-ASYNC-SCHED` (W3) | **DONE — DEFAULT FLIPPED ON 2026-07-17 (mirror vLLM), DGX-re-confirmed TOKEN-NEUTRAL; owner `6ea7856`.** `VT_ASYNC_RUNNER` defaults ON, so depth-2 `AsyncScheduler` + `step_with_batch_queue` + runner async input-combine/copy-stream D2H + `LoadedEngine` mcb=2 is the production default; `VT_ASYNC_RUNNER=0` / `VT_ASYNC_SCHED=0` roll back. DGX re-confirm (`dgx:~/work/vllm.cpp-async-flip`, CUTLASS+FA2 hard-verified, one flock): all three async arms bit-identical; shipping default (async ON + RMSNorm-fast OFF) **27B 235/235 + 35B 315/315** log "enabled (mcb=2)", rollbacks 235/235+315/315 "disabled". TTFT rises into vLLM's async envelope BY DESIGN (the mirrored trade); no new A/B (the discriminator's `f086b64`/`6ea7856` stands) | **DGX proof `f086b64` (root `dgx:~/work/vllm.cpp-w3-proof/f086b64…`):** 5/5 correctness gates PASS (27B+35B default, both W3-on, rollback — token-exact, arm-log correct). Interleaved c16 A/B (same binary, `VT_ASYNC_RUNNER=1` vs `+VT_ASYNC_SCHED=0`): **W3-on** tput 790.9–792.7, meanTPOT **160.9–161.2** (**−5.4 ms/step, WIN**), meanTTFT **2757.9–2778.1**; **W3-off** tput 793.5–794.1, meanTPOT 166.2–166.4, meanTTFT **2028.4–2032.0**. So overlap wins decode (−5.4 ms/step) but c16 meanTTFT **+36 % (+730 ms)** and throughput is **neutral (−0.3 %)**. **DIAGNOSIS (CPU-verified):** the +730 ms is NOT an admission delay — `test_async_admission_timing.cpp` proves depth-2 schedules a new prefill the SAME step as sync/vLLM (`UniProcExecutor` is synchronous, `uniproc_executor.py:91-106`); it is the closed-loop Little's-law consequence of neutral throughput + faster decode (`127×5.4≈686`). `benchmark_binding=false`, no speed credit | **NOT a CPU-fixable regression.** **2026-07-16 throughput-lever attempt RAN and is REFUTED:** the per-step sampler alloc/free serialization (rescan item #2 — `cudaMalloc`/`cudaHostAlloc`/events per step + the overlap-killing `cudaFree` inside `get_output`) was removed with persistent pooled buffers (`AsyncOutputPool` + `Sampler` greedy scratch, mirror of `gpu_model_runner.py:873-878` + `async_utils.py:12-70`). Clean re-proof (dgx `~/work/vllm.cpp-w3-tput/ab-fix2`, CUTLASS+FA2 hard-verified build): token-exactness **6/6 PASS**; interleaved c16 (w0+3 pairs): W3-on tput 788.14–790.58 / meanTPOT 161.60–162.04 / TTFT ~2732 vs W3-off 790.32–792.44 / 166.69–167.00 / ~2027 ⇒ **tput −0.32 % — gate ≥+1.5 % FAILS**; TPOT −4.95 ms retained; identical to pre-fix `f086b64`. Ceiling analysis: the removed syncs are O(10–100 µs)/step ≤0.1 % of a ~165 ms c16 step — two orders below the gate. (First attempt VOID: configure omitted `-DVLLM_CPP_CUTLASS_DIR` ⇒ WMMA fallback ~50 tok/s both arms + FA2 dropped ⇒ gates failed on `kv_cache_backend_resident()`; recorded, no verdict drawn.) The pooled-buffer structure LANDS (token-exact, no speed credit). W3 **stays default-OFF**; the depth-2 tput lever moves to the c8+ wave/overlap family (`62d4762`). **Re-proof gate unchanged:** ACCEPT = TPOT win retained AND TTFT within ~2 % of sync (achievable only with a throughput gain) |
| `KERNEL-GEMM-BF16` | **GATING — W2A qkvz implemented, DGX gates PENDING** | W1 BA: `0091cd1` closes structure, `f925294` closes projection/inertness, clean `f344dec` closes W1D2/G2 at **235/235**. **W2A qkvz (2026-07-15, test-first):** one raw-NK `[q,k,v,z]` owner (loader), ONE BF16 GEMM + strided mixed/z views on the CUDA default, split rollback from the same owner (`VT_GDN_MERGED_QKVZ=0`/`VT_GDN_MERGED_PROJ=0`), CPU merged-owner == split bit-exact, 35B/GGUF inert; CPU gates green (full CTest 107/107, tools 162/162, clean -Werror rebuild). `benchmark_binding=false`, no speed credit. **DGX gates at `baea3ec`: jobs 1 (default suites 8/8), 2a (`VT_GDN_MERGED_QKVZ=0`), 3 (35B native + legacy) PASS; 2b (`VT_GDN_MERGED_PROJ=0`) failed in the TEST contract only** — engine correct (master-off deselects packed decode by the designed BA coupling; 229/229 token asserts green); fixed test-first (shared `PackedGdnDecodeEnvSelected` truth table, CPU 16/16; serial CTest 107/107 re-green). First memcheck run was PATH-only (`compute-sanitizer: command not found`). **DGX gates then closed GREEN at `45f9e6d`**: default suites 8/8, both rollback arms, 35B inert, memcheck 0 errors/0 leaks, structural trace confirmed **−48 BF16 GEMMs/window** (145→97; wmma-BF16/packed-window 1.959≈2.0 merged vs 2.980≈3.0 split). qkvz is in the `246a23c` binding binary | **W2A closed green.** qkvz rides the `246a23c` binding exact-grid rerun (49/124; no isolated qkvz speed credit). Remaining `KERNEL-GEMM-BF16` work: none for attribution — the c2/c8 full-step split (2026-07-16) closed it (GEMM bundle at parity c2, ours-faster c8); open GEMM-adjacent lever is the GDN recurrence tiling |
| `KERNEL-GDN-PACKED-DECODE` | **`DONE` — W1D3 CLOSED on EQUIVALENCE** (owner `e47b4d6`) | Slot lifecycle fix `c172336` (identity-keyed pool; RED→GREEN `test_runner` 8/8) proven on DGX (`--diagnostic-c16` 3/3, both model gates 235/235); scheduler co-schedule parity with vLLM verified (`test_scheduler` 31/31). W1D3/G3 closed on evidence-totality: **eight sealed components** (harness calibrated test-first; focused suite 79/79, tools 162/162) + the 8-pair locked c16 A/B (`00bf484`: paired mean **−0.205% ± 0.30, <1σ**; cuBLASLt algo selection process-deterministic → algo-lottery REFUTED) + the 24-window trace (**packed is GPU-cheaper**: kernel compute −1.30..−1.58%/step, GDN+BA −296 µs/window, no attributable packed-side cost). The eighth (22-leg) seal `complete-failed` at `e47b4d6`: **38/40 + 8/8 memory**, stability clean, paired-consistency PASS at both concurrencies; the 2 fails are sign-flipping band-edge statistics of a true-zero effect. Full chronology and per-seal SHAs in state/ledger | **Disposition: EQUIVALENCE PROVEN — no stable regression on any axis.** Packed is the default (`VT_GDN_PACKED_DECODE=0` rollback); **no `complete-pass` marker exists and no speed credit is claimed**. qkvz (`KERNEL-GEMM-BF16` W2) closed green (`45f9e6d`); the authorized exact grid has now RUN (new binding `246a23c`, 49/124)
| RMSNorm/generated partitions ("norm/quant fusion") | **CLOSED / DISPROVEN as a parity gap** (RECONFIRMED 2026-07-16) | The 2026-07-14 [parity rescan](../.agents/specs/parity-rescan-2026-07-14.md) verified vLLM's `RmsNormQuantFusionPass` is FP8-only (no nvfp4 keys); the nvfp4 path runs standalone `scaled_fp4_quant` exactly like ours, and the +1.81 ms residual was a cross-profiler artifact. **2026-07-16 [reconciliation](../.agents/specs/decode-norm-quant-fusion-reconcile-2026-07-16.md)** re-verified this against the fresh correct-state c16 trace (count parity `cvt_fp16_to_fp4` 144/win == ours 144/step; `3f256ab` body-dump stores bf16 then quantizes separately) + a decode-shape microbench (isolated RMSNorm 6-9 µs), and CORRECTED the 2026-07-16 `SUMMARY.json` "vLLM fuses add+rmsnorm+fp4quant" note. Fusion-attributable headroom ~0 ms; the modest non-bit-exact efficiency residual is reassigned to `KERNEL-EW-NORM-ACT` | None — fusion lever stays CLOSED; efficiency redirect is spike + in-situ-A/B-gated, not implemented |
| fp8 cuBLASLt plan cache (`KERNEL-GEMM-FP8`) | **DONE / MEASURED NEUTRAL — bit-exact mirror, OPT-IN (`VT_FP8_PLAN_CACHE=1`), premise disproven** | The fp8 dense GEMM (`cuda_matmul.cu`) rebuilt the cuBLASLt descriptor + 3 TN layouts + heuristic every call; vLLM reuses an in-graph plan. Added a per-device `{desc,layouts,algo}` cache keyed on the full shape/config (`fp8_plan_cache.h`); BIT-EXACT (algo process-deterministic per shape — byte-exact cached==fresh `test_ops_fp8_cutlass` + CPU key/flag 4/4; 27B 235/235 + 35B 315/315 both flags). **Same-binary 35B A/B (one flock) is wall-clock NEUTRAL**: prefill TTFT in1024/c8 async-on ON 1491.5/OFF 1496.8, async-off ~1496.7/~1503.2; decode TPOT c1 15.16/15.14, c4 21.79/21.83. **nsys (async-off eager prefill): the pre-fp8-GEMM GPU gap is UNCHANGED — median 210 µs (off) vs 204 µs (on)** ⇒ the grounded "~0.8 ms removable gap" premise is NOT reproduced; the heuristic host cost is negligible/hidden (prefill GPU-bound; decode graph-captured so the heuristic runs at capture, not per replay-step). Kept opt-in for eager/non-graph regimes; default NOT flipped (faster unmet). Evidence `dgx:/tmp/fp8pc-*` + `~/work/prefill-attr-35b` | None — the vLLM-mirror lands opt-in; no default-flip and no speed credit |
| Serving transport (TCP_NODELAY) | **DONE / MEASURED NEUTRAL on the gate workload** | Mirror landed (`SERVE-HTTP-TRANSPORT`): `set_tcp_nodelay(true)` matches vLLM's uvicorn/asyncio default; behavioral accepted-socket test RED **0** → GREEN **1**, 22/22 cases. The non-binding one-lock localhost A/B (`~/work/vllm.cpp-tcpnodelay-sizing/ff915e8…`, 4a450f9 Nagle-ON vs ff915e8 Nagle-OFF, c1/c2 ×2 reps, identical pinned-client workload; raw-set SHA `f5b52900…2128`) is **neutral within noise** on every ITL/TPOT/throughput metric (c1 mean ITL ~102.7 both arms; c2 ~108–109; first cold-start leg excluded). Mechanism: ~100 ms per-token write cadence vs µs loopback ACKs means Nagle never coalesces — the rescan's rank-1 gain hypothesis is REFUTED for the loopback gate; the mirror stays for real-network parity | None for the gate — the c2/c8 full-step attribution is complete (transport ruled out; c2 gap is GPU-busy kernel glue) |
| Block-table host-cluster cleanup ([rescan](../.agents/specs/rescan-lost-lanes-2026-07-16.md) §1,§5,§6) | **CPU-SIDE, BIT-IDENTICAL; `benchmark_binding=false`, no speed credit** — the payoff is measured by the dispatched correct-state c2/c8 full-step probe and the next authorized exact grid, not by these mirrors. **(c) LANDED** — `block_table.compute_slot_mapping` drops the dead tail-pad (`~2×(max_num_batched_tokens−total)` int64 writes/step; the decode graph re-pads via `BuildPaddedDecode`, the only other consumer slices `[0,total)`); test_block_table 11/11, test_prepare_inputs 6/6. **(d) LANDED** — the decode-graph capture-size set is DERIVED from `max_num_seqs` (`include/vllm/model_executor/models/decode_graph_sizes.h`, `DecodeGraphSizes`/`PadToCaptureSize`; mirrors vLLM `_set_cudagraph_sizes` reduced to the full-decode-cudagraph regime): `max_num_seqs=32` → `{1,2,4,8,16,24,32}` (adds the missing 24 bucket, drops the never-reachable 64; batches 17–24 stop over-padding to 32, +1 captured graph). CUDA-only; padding rows inert → token-exact. RED→GREEN `test_decode_graph_sizes` 5 cases/478 asserts (RED = 24 bucket absent under the old fixed set). **(e) LANDED** — `InputBatch::make_sampling_metadata` caches + rebuilds only on batch change (add/remove/condense/swap set a dirty flag), mirroring vLLM `refresh_metadata` (gpu_input_batch.py:812-830); deviation: the penalty-active path rebuilds every step (our port copies output_token_ids where vLLM holds a live ref), so the greedy/no-penalties gate gets the full win bit-identically. `scheduler.cpp:371,377` `std::move`s the `num_scheduled_tokens` map + `finished_req_ids` set (container plumbing, zero policy change; move-then-clear keeps observable behavior identical). RED→GREEN `test_input_batch` (+2 cases; RED = stale cache after a 2nd add returns size 1 != 2); `test_scheduler` 31/31, `test_runner` 228/228, `test_sampling_metadata` 6/6 unchanged. Items (a)-runner + (b) (GDN col-0) are in `runner.cpp` (async/GDN-claim owned) — reported, not touched. Landed `8a717b2`/`81afc36`/`0c4b41c` (merged `e027ad5`); clean full `-Werror` rebuild 0 warnings, CPU battery + tools 164/164 green. **DGX token-exactness gate PASSED** on the `e027ad5` build (GB10, one `flock /tmp/gpu` series, root `dgx:~/work/vllm.cpp-blocktable-gate`): 27B default **235/235**, 27B `VT_GDN_PACKED_DECODE=0` rollback **235/235**, 35B **315/315**, all exit 0 | **CLOSED — claim released.** No A/B by design (`benchmark_binding=false`, no speed credit); payoff measured by the c2/c8 attribution probe + the next authorized exact grid |
| Host-weight ownership | **BINDING MEMORY AXES NOW PASS at `246a23c`** (windowed release is binding) — ours peak PSS/RSS 24,879,201/24,881,800 KiB vs vLLM 28,184,400/28,563,020 KiB (1.1329×/1.1479×), GPU 40,996 vs 70,531 MiB, MemAvailable-drop 68,346,844 vs 80,660,556 KiB; all four PASS. The `LOAD-SAFETENSORS` **windowed release** (progressive interior-page `madvise(MADV_DONTNEED)` on each copied-then-dead source range during the copy loop; default on, `VT_LOAD_WINDOWED_RELEASE=0` rollback; page-lifetime proven in [the spike](../.agents/specs/safetensors-windowed-load.md); CPU RED→GREEN smaps-Rss/byte-identity/neighbor-safety tests) is now **measured on GB10** at `cb2d310`, root `~/work/vllm.cpp-windowed-load/cb2d310c…518/evidence`, single 27B load-to-ready per arm under one `flock /tmp/gpu`: **OFF VmHWM 48,285,916 kB (48.29 GB)** / VmRSS 24,750,696 kB (the double-residency peak intact, matching the precheck) vs **ON (default) VmHWM 24,750,704 kB (24.75 GB) = VmRSS** — the load transient is **fully eliminated (−23,535,212 kB, −48.7%)**; steady Pss 24,748,252/24,748,260 kB both arms. ON-arm serving smoke **6/6** requests, 0 failed (302.7 tok/s c1, health-only). Artifact SHA-256: status `cdccc1dd…7233` / `3fd0592c…1fc0`, smaps `11baecd3…3881` / `41837d12…65f3`, smoke `ed271a68…5aa8`, server logs `772bec6b…9c49` / `b66bb783…a81cd`. PROJECTION (not credit): ours ≈24.75–24.86 GB peak vs vLLM's binding 28.17/28.53 GB Peak PSS/RSS → both failing memory axes are projected to flip PASS at the next authorized exact-grid rerun | **Binding Peak PSS/RSS axes now PASS** at the `246a23c` exact-grid rerun, as projected. For 35B the routed-expert host double-store is separately CLOSED (2026-07-18, `ENG-MOE-HOSTFREE`, `ac77bec`): `ReleaseHost` (`madvise(MADV_DONTNEED)`+swap) frees the ~16.9 GiB fp4 expert host mirror after the device Marlin resident is built → 35B STEADY serving PSS 20.17→**3.53 GiB** (same-binary A/B, `VT_MOE_HOST_FREE` rollback). Direct-to-device *load-time* streaming remains the deeper fix (it moves the whole-window `peak_pss` load-phase coexistence, which the post-build free cannot; also removes the 22.92 GiB safetensors steady mirror) |
| Qwen3.6-35B-A3B performance | **BLOCKED / NOT RUN (grid crash FIXED)** | Correctness passes; no current v0.25.0 performance denominator exists. The 35B online-serving **c2+ crash that blocked the grid is root-caused + FIXED** (2026-07-18, `CLAIM-35B-GRAPH-SCRATCH-1`, [spec](../.agents/specs/decode-graph-scratch-uaf-2026-07-18.md)): at concurrency > 1 the engine died with `cudaEventSynchronize: an illegal memory access`. cuda-gdb pinned the faulting kernel to `marlin_moe_wna16::Marlin<…><<<(144,1,1),(128,1,1)>>>` — its fp32-reduce scratch `c_tmp` (`EnsureCtmp`, `cuda_moe_marlin.cu`) is a grow-on-free per-stream buffer whose pointer is baked into the captured pure-decode CUDA graph; a bigger later prefill/decode `cudaFreeAsync`s it → the next graph replay reads freed memory (single-stream c1 never grows it → never crashed). Differential isolation: needs graphs ON + concurrency > 1 + long context; async/WMMA ruled OUT (async-off + wmma-off still 5/5 crash; graphs-off CLEAN); memcheck masks it. Fix = retire-on-grow (`RetireGraphScratch`, `src/vt/cuda/graph_safe_scratch.h`) across the four decode-graph-reached scratch allocators. Sweep c1-c16 pre-fix 5/5 crash → post-fix 0; 315/315 token-exact preserved. `benchmark_binding=false` (correctness fix, no speed credit) | Run the v0.25.0 performance grid (now unblocked) after 27B reaches 124/124 |
| 35B FA2-prefill + fused-preamble lever | **LANDED / default-ON (2026-07-18, `CLAIM-35B-FA2-FLIP-1`); `benchmark_binding=false` (offline op A/B, no grid speed credit — the 35B grid re-measures in-situ)** ([spec](../.agents/specs/qwen36-35b-fa2-prefill-oracle-2026-07-18.md)) | `FuseAttnPreambleOn` flipped default-ON all arches ⇒ the 35B ratio-8 full-attn layers take the exact `flash_fwd_splitkv` kernel (kernel-side `fa2_prefill` admits any GQA ratio at head_dim 256, `cuda_paged_attn.cu:2494`). FULL current-main default-set gate (async + GDN cubin + all fast kernels + flip): **35B `test_qwen36_paged_engine` 315/315 + 27B `test_qwen27_paged_engine` 235/235** (`dgx:/tmp/fa2gates_u.log`); **memcheck 35B prefill 0 errors** (`dgx:/tmp/fa2_memcheck2.log`); `test_ops_attn_preamble` 14/14. Realistic input-1024 TTFT A/B (same binary, FA2 default vs `VT_FA2_PREFILL=0`, conc8/num-prompts32, 3 interleaved on/off pairs + dropped warmup, `dgx:/tmp/fa2_ttft2.log`): **FA2-on Mean TTFT 824.7 ms vs off 874.4 ms = −5.7%** (median 638.6 vs 676.9 = −5.7%; prefill token-throughput 5170.5 vs 4902.5 tok/s = +5.5%; per-arm spread <7 ms, so the ~50 ms gap is well-separated). Below the ~7-9% offline-kernel target because the 1.86× attention-kernel win dilutes across the whole prefill (GEMM/MoE/GDN dominate); no decode/TPOT regression (prefill-only lever, both arms 315/315 token-exact). The "round normed q/k→bf16 before RoPE" tighten (`fused_qk_norm_rope.py:67`) was op-level bit-identical (fused-bf16 == unfused-bf16, q/k 0-mismatch) but flipped the 27B tok6 whitespace near-tie away from the pip-vLLM oracle (233/235) in COMBINATION (RMSNorm-saga) ⇒ NOT shipped; preamble ships UNTIGHTENED, both arches token-exact. CPU gate: clean `-Werror` rebuild 0 errors/0 warnings, full DGX ctest **156/157** (both engine gates green; the sole failure `test_capi` is a KNOWN pre-existing nondeterministic dgx-box detokenizer UTF-8 flake — 3 runs of the same binary give 3 different results — not a regression, unrelated to the attention path), tools unittest 164/164, checkers green | The 35B v0.25.0 perf grid (orchestrator-owned) re-measures the in-situ prefill/TTFT gain from this lever |
| SGLang shared-prefix floor | **PENDING / NO ACCEPTED NUMBER** | No equivalent cache-on vllm.cpp/vLLM/SGLang campaign exists | After cache-off parity, gate equivalent vLLM v0.25.0 and SGLang v0.5.15; the faster reference binds each axis |
| External KV / LMCache | **NOT IMPLEMENTED / NOT BENCHMARKED** | Connector ABI and two-engine store/retrieve remain roadmap inventory | Spike fake-provider semantics, then gate LMCache MP before in-process mode |

### Active packed-decode implementation checkpoint

The semantic and operator gates are retained at clean `f18ca23` and
`9ad8fb7`; their complete hashes and attempt chronology live in the ledger and
state log. W1D3 is now **CLOSED on EQUIVALENCE** (`KERNEL-GDN-PACKED-DECODE`
`DONE`, owner `e47b4d6`; no `complete-pass` marker, no speed credit):

| Surface | Current evidence | Disposition |
|---|---|---|
| Exact vLLM boundary | Direct packed output/state BF16 differences **0/1** | PASS; upstream explicit reference has the same one-element state delta |
| Cache ABI | `MambaSpec` conv then temporal; gate allocation BF16 conv + FP32 SSM | PASS on registry/runner production-like tests |
| 27B default | **235/235 + 16/16**; prefill 0 packed calls, first decode exactly 48 | Immutable G2 PASS at `f344dec` |
| 27B rollback | `VT_GDN_PACKED_DECODE=0`, **235/235 + 16/16**, 0 packed calls | Immutable G2 PASS at `f344dec` |
| 35B native/batched | **315/315**, 0 packed calls | Immutable inertness PASS |
| 35B GGUF | Compact **14/14**, Balanced **14/14**, loader **98/98** | Immutable isolated-process PASS |
| Safety | CUDA GDN **43/43, 1,707/1,707**; packed/corner/FP16-SSM memcheck zero errors/leaks | Immutable PASS |
| W1D3 trace harness | Packed **915** nodes with 48 packed recurrence calls; rollback **963** with 48 decomposed + 48 post-conv calls; both retain 145 BF16 GEMMs, isolate exactly 48 mode-coupled BA projections, and require every remaining signature to match | Raw capture PASS at `7ff713e`: **12/12 + 12/12** exact ranges; both oracle traces structurally exact |
| Component runner | Production profile-control-off build; exact source/vLLM corpus manifests and partition hashes; full oracle/toolchain/artifact inventory; exact raw-sample throughput/TTFT/ITL recomputation plus bounded pinned-clock E2E/TPOT validation and duration-span consistency; frozen 64-plan fixture; fresh isolated server per leg; c2=6 requests and c16=96 requests per rep; a discarded cold-start warmup pair then 5 timed reps AB/BA/AB/BA/AB (22 legs); one lock across exact-snapshot correctness and all legs | The `d82d282` c16 HTTP-500 (`duplicate live GDN state index`) was fixed test-first (request-identity slot keying) and proven on DGX at `c172336`. **Eight full components have since sealed** a marker-last terminal status (3× `complete-void` on the TTFT-family arrival lottery, then 5× `complete-failed` as the harness calibrated to true-zero noise); the eighth (first 22-leg) seal is `complete-failed` with 38/40 + 8/8 memory PASS, stability clean, paired-consistency PASS at BOTH concurrencies — **W1D3 CLOSES on equivalence** |
| Performance | Marker-last finalization requires all **40 timing + 8 memory** median axes and the gated paired run axes, per-run stability of ≤4% for non-tail timing and all memory axes and ≤15% for the tail axes (p90/p99 of ttft/tpot/itl/e2el — revised test-first after two `c172336` tail voids), fixed 1-GiB memory-return tolerance, parsed throttle counters, successful pinned GPU-idle probes, exact server/client/preflight commands and lifecycle markers. The c2 TTFT-family (mean/median/p90/p99 of ttft) is instead compared on the arm-pooled 30-per-request distribution, stability-gated on a 50% pooled sanity bound, and excluded from the gated per-rep paired axes (bimodal arrival lottery; c16/non-TTFT keep 4%/15%; E2EL unchanged). **Acceptance uses a NOISE BAND (`contract.acceptance`): a comparison axis (median, gated paired, c2 mode means, memory) fails only when the packed deficit exceeds run noise — 0.5% for non-tail timing and PSS/RSS, 15% for tail axes, per-mode c2 TTFT, and the recalibrated GPU-memory bands; packed≥rollback always passes.** The gated per-rep **paired** axes use a **MAJORITY-CONSISTENCY** rule (`contract.paired_gate`): a paired axis fails only when ≥3 of its 5 rep-pairs breach the band in the same (packed-worse) direction; single-pair breaches are recorded as diagnostics. A stable regression is `complete-failed`; a sealable unstable/malformed run is `complete-void` | **Eight sealed → W1D3 CLOSES on EQUIVALENCE.** VOID×3 (all TTFT-family: a bimodal prefill arrival lottery, no scheduler divergence), then `complete-failed`×5 as the harness calibrated to true-zero noise (acceptance noise band, mode-conditional c2 TTFT + GPU-memory recalibration, majority-consistency pairing, 5-rep + cold-discard precision upgrade). The eighth (first 22-leg: cold-discard pair + 5 reps) seal `complete-failed` at `e47b4d6`: **38/40 axes, 8/8 memory, stability clean, `validation_error=None`, paired-consistency PASS at BOTH c2/c16**; c16 at equivalence (packed med 801.97 vs rollback 802.95, −0.12%, passes); the 2 fails are sign-flipping band-edge statistics of a true-zero effect (c2 `median_tpot_ms` 0.9899, c2 pooled `p99_ttft_ms` 0.8464). Combined with the 8-pair locked c16 A/B (−0.205% ± 0.30, <1σ) and the 24-window trace (packed GPU-cheaper), **no stable regression exists on any axis — EQUIVALENCE PROVEN. No `complete-pass` marker exists; NO SPEED CREDIT** |

Failure evidence is immutable. Order/run/execution/c16 raw/client/server SHA-256
values are `a2de5b07…6de0` / `297e3c62…a6fb` / `ff71c9f0…0684` /
`f8571d48…945c` / `8b9526f4…63a9` / `7b05e066…7dfe`. The C++ API wraps the
engine exception text into its JSON error body, but pinned vLLM's OpenAI
benchmark stores only `response.reason`; therefore the retained client evidence
says only `Internal Server Error`. Exact exception capture is the next
diagnostic, not a guessed runtime fix.

The current raw root is
`~/work/vllm.cpp-gdn-packed-trace/7ff713e377457130db4ed15929133d1b463aff96`.
Execution / configure / model-gate / run-log SHA-256 values are
`253ff089…a4bb` / `903d10a1…5c12` / `3ebdd9f9…bfca` /
`6c0c2cae…2235`; packed/rollback oracle trace hashes are
`4448c549…c293` / `9d918979…a420`. The first finalizer log
(`a71ac6e4…7e67`) records the expected fail-closed unknown-signature error.
Accepted summary / manifest / marker / pushed-finalizer-log hashes are
`bf5c04b7…702f` / `2e92b3a2…55c0` / `e1314019…8c1` /
`626c6844…8211`; artifact-set SHA is `ea286db4…c3dc`. Source, GPU and lock
returned clean/idle/free; the complete artifacts are retained.

Immutable evidence root:
`~/work/vllm.cpp-gdn-packed-decode/f344decf457a4d50c3bcae78a2903d7fe176a511/evidence-g2`.
Status is `complete-g2`; the complete one-lock order is frozen in
`run-plan.txt`, and its core entry points are:

```sh
flock /tmp/gpu build-cuda/tests/test_ops_gdn
flock /tmp/gpu build-cuda/tests/test_op_parity \
  -tc='qwen27 GDN packed decode boundary*'
flock /tmp/gpu build-cuda/tests/test_qwen27_paged_engine
flock /tmp/gpu env VT_GDN_PACKED_DECODE=0 \
  build-cuda/tests/test_qwen27_paged_engine
flock /tmp/gpu build-cuda/tests/test_qwen36_paged_engine
```

G2 and W1D3 structural evidence are closed. The repaired source has been pushed
and launched exactly once at `d82d282`; that incomplete root must never be
reused or appended. The failure inspection and next diagnostic entry point are
in the newest [state record](../.agents/state.md) entry and the
[spike](../.agents/specs/gdn-packed-decode.md). The failed launch provenance is:

```sh
set -euo pipefail
SOURCE_REPO="$HOME/work/vllm.cpp"
git -C "$SOURCE_REPO" fetch origin main
SHA=d82d282f9efd1a5b97e7c6f1ac7a55b949849d09
ROOT="$HOME/work/vllm.cpp-gdn-packed-component/$SHA"
BINDING="$HOME/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560"
SNAPSHOT="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a"
test ! -e "$ROOT"
mkdir -p "$(dirname "$ROOT")"
git -C "$SOURCE_REPO" worktree add --detach "$ROOT/source" "$SHA"
mkdir -p "$ROOT/evidence/corpus"
cp -a "$BINDING/corpus/27" "$ROOT/evidence/corpus/27"
cmake -S "$ROOT/source" -B "$ROOT/build-production" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$HOME/venvs/vllm-oracle/bin/ninja" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_FLASH_ATTN=ON \
  -DVLLM_CPP_TRITON=ON -DVLLM_CPP_TRITON_REGEN=OFF \
  -DVLLM_CPP_BENCH_PROFILE_CONTROL=OFF \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tee "$ROOT/configure.log"
"$ROOT/source/scripts/dgx-gdn-packed-component.sh" --execute \
  --snapshot "$SNAPSHOT" \
  --source-corpus "$ROOT/evidence/corpus/27" \
  --evidence "$ROOT/evidence" \
  --build-dir "$ROOT/build-production" \
  --configure-log "$ROOT/configure.log" \
  --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA"
```

`$ROOT/source` is the clean detached worktree at `$SHA`; the recipe above is the
`--execute` reproduction pattern (the eighth run used the 22-leg 5-rep plan). The
request corpus is copied byte-for-byte from the binding `3f256ab` evidence (the
5-rep corpus is a byte-verified deterministic extension); the separate 64-plan
fixture is committed under `tests/fixtures/` and the runner requires it in
read-only mode. The packed component claims no throughput number, so the binding
27B grid stays `benchmark_binding=false` until the authorized fresh exact-grid
rerun.

**W1D3 is CLOSED on equivalence** (see the summary at the top). All eight
component roots are immutable and must never be reused
(`c172336`/`-r2`, `d19e091`, `2dbe892`, `da05444`, `495ba780`, run-6, and the
eighth/closing seal `e47b4d6`). The active tracks are now **qkvz**
(`KERNEL-GEMM-BF16` W2) and the authorized fresh binding/exact-grid rerun; no
further packed-decode component run is required (further single-run seals of the
true-zero effect are band-edge coin flips that would not change the recorded
conclusion). The `--diagnostic-c16` recipe below is retained as the focused
regression check: it reruns the packed c16 boundary three times under one GPU
lock, captures any `engine-fatal:` root cause on stderr (server log), and
replays the corpus row into a persisted HTTP body — writing only
`component-diagnostic.json` and a `diagnostic/` subtree, never a sealable
component. Provision `SNAPSHOT`/corpus/build exactly as above, then, with a new
root whose basename contains `diagnostic-c16`:

```sh
SHA=<pushed diagnostic SHA>
ROOT="$HOME/work/vllm.cpp-gdn-packed-diagnostic-c16/$SHA"
# ... provision $ROOT/source, $ROOT/evidence-diagnostic-c16/corpus/27,
#     $ROOT/build-production, $ROOT/configure.log exactly as the --execute recipe
"$ROOT/source/scripts/dgx-gdn-packed-component.sh" --diagnostic-c16 \
  --snapshot "$SNAPSHOT" \
  --source-corpus "$ROOT/evidence-diagnostic-c16/corpus/27" \
  --evidence "$ROOT/evidence-diagnostic-c16" \
  --build-dir "$ROOT/build-production" \
  --configure-log "$ROOT/configure.log" \
  --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA"
```

The mode asserts the evidence basename contains `diagnostic-c16` and holds no
`component-*.json`; it runs no model gates, no 2/16 sweep, and never calls
`finalize`. `benchmark_binding` stays `false` throughout.

This reproduction has been executed once at `4a450f9` (root
`~/work/vllm.cpp-gdn-packed-diagnostic-c16/4a450f9…`, build **154/154**,
marker-last `diagnostic_complete`, GPU/lock returned idle/free). All three
fresh-server reps failed identically with
`engine-fatal: … duplicate live GDN state index at qwen3_5.cpp:73`.
Core SHA-256: `component-diagnostic.json` `42de1323…13ea`; r1/r2/r3 server
logs `f26f0030…8bc0` / `8ecba873…02a3` / `e68411cf…6f3f`; r1 error body
`c5aa0933…7fc3`. The root is diagnostic evidence only and is preserved
unchanged; the repair checkpoint and a fresh SHA/root full component rerun
are next.

Closed controls do not remain active leaves: async scheduling measured neutral
for speed, and the prior fused-producer candidate remains default-off after its
strict component failure. Their exact results and reproduction history remain
in the append-only record rather than this live scoreboard.

The completed core correctness/safety root is
`~/work/vllm.cpp-gdn-ba/immutable-581d335fec2e5a96d9ccbb38c1ec001c39ac1789`.
Status / artifact-list SHA-256 values are `3895e658…4cf6` / `ed2bf8d8…895b`.
Focused CTest, merged/split 27B, 35B, and memcheck log hashes are
`4cf699ad…759b`, `c2a6f93f…cf96` (both arms), `b926716e…9875`, and
`a3d61cb9…fb87`; fixture `e81e9181…7edd` loaded 64 plans and the forbidden
native cache stayed absent. The rejected BF16-output preflight log remains
`09078b76…b050`. This closes immutable core correctness/safety only and earns no
speed ratio.

The completed structural root is
`~/work/vllm.cpp-gdn-ba-trace/0091cd192d9a6baa2197a4f3bdb0561bd859baf5`.
All 24 local ranges pass their exact contracts. The merged/split oracle traces
have SHA-256 `b8d26d4c…fc59` / `cef841ce…ede5`; they contain 1,522 / 1,521
internally invariant steady B=2 windows at 1,160 kernels and ordered-name SHA
`858915dd…fad0`. Their full launch signatures are `17e1037e…14ed` /
`f7a3ca1f…cadf`. Pushed finalizer commit `8a1f923` wrote the marker last with
status `complete-structural`. Summary / manifest / marker / artifact-set /
finalizer SHA-256 values are `03601168…54d5` / `b203f0d2…5412` /
`72328c48…63e` / `b93fd633…70a2` / `57395e99…b146`. The summary proves merged
963/145 versus split 1,011/193, exact 48/48 deltas, unchanged non-BF16 family
counts and `benchmark_binding=false`; it grants no speed ratio.

The current W1C hardware root is
`~/work/vllm.cpp-gdn-ba-rounding/f9252943d1e96dbfa43e3b8f2d06dec1aa5f20d3/evidence/w1c-correctness-inertness`.
The source is clean at exact `f925294`, and the binary SHA-256 values are frozen
in `provenance.txt`. Projection / loader / native-35B / real-GGUF / default-27B /
BF16-27B log SHA-256 values are
`a791c567…37d1` / `d455b8fc…05f6` / `72caeca9…06c` /
`87833f22…af8d` / `da5dd836…091e` / `148d743f…86a`.
The final BF16 assertion failure intentionally prevents `status.txt`,
`sha256sum.txt`, and the terminal event from being written. GPU and lock were
verified idle afterward. This is a valid **FAILED** correctness checkpoint,
not a partial performance number.

## Evidence selecting merged GDN projections

The immutable completed root is
`~/work/vllm.cpp-executed-path-c2/179a0fc2afc1c33b63d14de8e50d3fde976c7356`.
Its status is `complete-diagnostic`.

| Artifact | SHA-256 |
|---|---|
| c2 summary | `0ef6a1240d33c16410cd4e43b30ca8667a6d92e6eee8506d7bd03388fe010273` |
| c2 manifest | `2556cfd032fae2201d9f8deb818343731b7dc99d9f8e6329da9b793262712f21` |
| status | `9e0143fa1b9c74e218e486fedd0606850708619a0e859dafe94957e24a507b57` |
| artifact set | `cc248ad2b5bf08f85b0d6b178de70682a104917e16c59c9adf34d661217f823a` |
| fresh oracle trace | `2b3bf41269fd19ef65c5c3e06f067af73d7d997de3b6be17a2af785b6a86785c` |

All **12/12** local B=2 graph ranges are invariant at 1,011 kernels + 7
memcpy + 1 memset. The oracle contains **1,522** invariant steady B=2 windows
at 1,160 kernels plus two bounded B=1 drains. Both engines execute the same
**128 Stream-K + 80 static-persistent** FP4 tactic split.

| BF16 projection structure per B=2 window | vllm.cpp | vLLM v0.25.0 |
|---|---:|---:|
| qkv / packed qkvz | 48 | 48 |
| z | 48 | included in qkvz |
| b+a / packed ba | 96 | 48 |
| lm_head | 1 | 1 |
| Total | **193** | **97** |

The source arithmetic independently gives `(4-2) × 48 = 96`; this is not a
profiler-name classification artifact. Diagnostic family medians are
**51.662672 vs 48.798042 ms** (+2.864630 ms), with a shape-level decomposition
ranking BA at about 1.882 ms and qkvz at about 0.476 ms. These durations cross
Nsight/Torch-profiler domains and are not a speed ratio. The accepted spike
therefore gates BA and qkvz separately and forbids duplicate weight owners or
split-copy kernels.

## Verify or reproduce the current checkpoint

Verify the durable diagnostic without GPU work:

```sh
RAW_SHA=179a0fc2afc1c33b63d14de8e50d3fde976c7356
ROOT="$HOME/work/vllm.cpp-executed-path-c2/$RAW_SHA/evidence/$RAW_SHA/trace/27"
sha256sum "$ROOT"/{c2-summary.json,c2-manifest.json,status-c2.json}
# Expected prefixes: 0ef6a124…0273 / 2556cfd0…2f21 / 9e0143fa…7b57
```

Verify the pushed-SHA core correctness/safety checkpoint without rerunning the GPU:

```sh
SHA=581d335fec2e5a96d9ccbb38c1ec001c39ac1789
ROOT="$HOME/work/vllm.cpp-gdn-ba/immutable-$SHA"
test "$(git -C "$ROOT/src" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$ROOT/src" status --porcelain)"
sha256sum "$ROOT/evidence"/{status.txt,sha256sums.txt}
# Expected: 3895e658…4cf6 / ed2bf8d8…895b
```

Verify the durable structural checkpoint without GPU work:

```sh
RAW_SHA=0091cd192d9a6baa2197a4f3bdb0561bd859baf5
ROOT="$HOME/work/vllm.cpp-gdn-ba-trace/$RAW_SHA"
TRACE="$ROOT/evidence/$RAW_SHA/trace/27"
sha256sum "$TRACE"/{gdn-ba-summary.json,gdn-ba-manifest.json,status-gdn-ba.json}
# Expected: 03601168…54d5 / b203f0d2…5412 / 72328c48…63e
```

Verify the current fail-closed W1C checkpoint without rerunning the GPU:

```sh
SHA=f9252943d1e96dbfa43e3b8f2d06dec1aa5f20d3
ROOT="$HOME/work/vllm.cpp-gdn-ba-rounding/$SHA"
E="$ROOT/evidence/w1c-correctness-inertness"
test "$(git -C "$ROOT/source" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$ROOT/source" status --porcelain)"
test ! -e "$E/status.txt"
sha256sum "$E"/{01-projection.log,02-qwen36-weights.log,03-qwen36-native.log,04-qwen36-gguf.log,05-qwen27-default.log,06-qwen27-bf16.log}
```

The exact single-lock command below reproduces the accepted subgates and final
233/235 failure. Any partial or unlocked execution is void:

```sh
SHA=$(git rev-parse HEAD)
ROOT="$HOME/work/vllm.cpp-gdn-ba-rounding/$SHA"
SOURCE="$ROOT/source"
BUILD="$ROOT/build-cuda"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$SOURCE" status --porcelain)"
flock /tmp/gpu bash -lc "
  set -euo pipefail
  '$BUILD/tests/test_op_parity' \
    -tc='qwen27 GDN BA BF16 projection matches vLLM 0.25 oracle*'
  '$BUILD/tests/test_qwen36_weights'
  '$BUILD/tests/test_qwen36_paged_engine'
  '$BUILD/tests/test_qwen36_gguf_engine'
  '$BUILD/tests/test_qwen27_paged_engine'
  VT_GDN_BA_OUT_BF16=1 '$BUILD/tests/test_qwen27_paged_engine'
"
```

The prior end-to-end failure remains reproducible and is the model-level RED
precondition for the packed production dispatch:

```sh
SHA=f9252943d1e96dbfa43e3b8f2d06dec1aa5f20d3
BUILD="$HOME/work/vllm.cpp-gdn-ba-rounding/$SHA/build-cuda"
flock /tmp/gpu env VT_GDN_BA_OUT_BF16=1 \
  "$BUILD/tests/test_qwen27_paged_engine"
# Current disposition: 233/235; got == greedy_ids_emulation.npy
```

The accepted W1D3 finalization is reproducible without recapturing the GPU
series. Its marker records both the finalizer and imported launch-validator
hashes:

```sh
set -o pipefail
RAW_SHA=7ff713e377457130db4ed15929133d1b463aff96
FINALIZER_SHA=24cea4f1fe28c89968cad1ed845fbfbd64514b0c
ROOT="$HOME/work/vllm.cpp-gdn-packed-trace/$RAW_SHA"
EVIDENCE="$ROOT/evidence/$RAW_SHA"
SOURCE="$HOME/work/vllm.cpp-gdn-packed-finalizer/$FINALIZER_SHA/source"
VERIFY="$ROOT/finalizer-replay-$FINALIZER_SHA"
test -z "$(git status --porcelain)"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$FINALIZER_SHA"
test ! -e "$VERIFY"
cp -a --reflink=auto "$EVIDENCE" "$VERIFY"
rm "$VERIFY/trace/27/gdn-packed-summary.json" \
  "$VERIFY/trace/27/gdn-packed-manifest.json" \
  "$VERIFY/trace/27/status-gdn-packed.json"
PYTHONPATH="$SOURCE" python3 \
  "$SOURCE/tools/bench/finalize_gdn_packed_trace.py" \
  --evidence "$VERIFY" --source-commit "$RAW_SHA" \
  --run-log "$ROOT/gdn-packed-run.log" \
  2>&1 | tee "$ROOT/gdn-packed-finalizer-replay.log"
```

Acceptance requires packed **915** versus rollback **963** total nodes, 145
BF16 GEMMs in both arms, 48 packed recurrence calls replacing 48 decomposed +
48 post-conv calls, exactly 48 BA projection nodes at the accepted `(8,1,1)`
geometry in each arm (hashed separately because BF16-vs-F32 output may change
the cuBLASLt tactic), and an identical normalized signature multiset for every
remaining kernel/memcpy/memset node.
Exact requirements are in the
[packed-decode spike](../.agents/specs/gdn-packed-decode.md).

Re-aggregate the **new binding** result without GPU work; exit **1** is expected
for a not-all-axes-pass gate, while exit 2 means malformed evidence. The prior
binding `3f256ab` re-aggregates identically from its own immutable root:

```sh
# New binding (49/124). Swap the SHA for 3f256ab… to re-aggregate the prior.
SOURCE="$HOME/work/vllm.cpp-online-gate/evidence/246a23cfa423e8e50c65b0ff067be55f3a3c7bf9"
CHECK="/tmp/vllm-cpp-246a23c-summary-$USER"
cp -a --reflink=auto "$SOURCE" "$CHECK"
rm -rf "$CHECK/summary-27"
set +e
PYTHONPATH="$PWD" python3 tools/bench/online_gate_summary.py \
  --evidence "$CHECK" --model 27
rc=$?
set -e
test "$rc" -eq 1
# Expected: report.md "Every-axis ratios failing or void: 75/124" (49 pass).
# Verify artifact hashes: ratios.json f784ba01…e046, all-runs.json b7ef3442…3240,
# manifest.json 7f25c614…83e8.
```

Relaunch the authorized timed grid from the pushed SHA. `--execute` is now a
PURE timed production grid (model gate + interleaved ours/vLLM legs +
memory/thermal/cache-drop capture + summary) from the profile-control-OFF
build; the H1d paired trace stays a separate `--trace-only` run that requires
the instrumented build:

```sh
SHA=$(git rev-parse HEAD)
CLAIM="$HOME/work/vllm.cpp-online-gate"
EVIDENCE="$CLAIM/evidence/$SHA"
# 1. Dry-run plan (no GPU lock).
scripts/dgx-online-serving.sh --dry-run --claim-root "$CLAIM" --vllm-cpp-sha "$SHA"
# 2. Byte-identical binding corpus copy into evidence/corpus/27.
scripts/dgx-online-serving.sh --prepare-corpus --model 27 \
  --source-corpus "$EVIDENCE/corpus/27" --evidence "$EVIDENCE"
# 3. Timed production grid (one /tmp/gpu lock for the whole model).
scripts/dgx-online-serving.sh --execute --model 27 \
  --snapshot "$SNAPSHOT" --source-corpus "$EVIDENCE/corpus/27" \
  --evidence "$EVIDENCE" --build-dir "$BUILD" --configure-log "$CONFIGURE_LOG"
```

## Correctness-only changes (benchmark disposition NOT APPLICABLE)

- **Platform seam extraction (2026-07-18, `BACKEND-PLATFORM`).** A
  behavior-preserving refactor: added the C++ `Platform` capability seam
  (`include/vllm/platforms/interface.h` + `src/vllm/platforms/{platform,cpu,cuda}.cpp`,
  faithful mirror of `vllm/platforms/interface.py:134-229`) and migrated the 7
  memory-model / residency conditionals (KV-cache device residency + async
  device combine/scatter in `runner.cpp`, the decode-graph CUDA gate in
  `model_registry.cpp`, the host→device weight-residency branches in
  `qwen3_5.cpp`) from inline `device.type == kCUDA` tests to
  `CurrentPlatform()` capability calls. No kernel/numeric/dispatch change (the
  same branch is taken on CUDA/CPU as before), so **NOT APPLICABLE** to the
  throughput / latency / memory scoreboard. Verified by the new `test_platform`
  unit test, the full CPU CTest suite, and (behavior-preserving ⇒ must be
  unchanged) both DGX model gates `test_qwen27_paged_engine` 235/235 +
  `test_qwen36_paged_engine` 315/315.

- **CPU `kCastF32` kernel registration (2026-07-18).** Registered the missing
  CPU `bf16→f32` cast kernel so the CPU backend supports both cast directions
  (mirrors the already-present `f32→bf16` `kCastBf16`), closing a CPU
  op-registration asymmetry. CPU-only op registration + unit test; no CUDA path
  change and no gate-workload impact, so **NOT APPLICABLE** to the throughput /
  latency / memory scoreboard. Verified by `test_ops_glue` (`cast_f32` RED→GREEN)
  and the full CPU CTest suite.

- **M=1 decode MoE routing/align parallelization (2026-07-18,
  `CLAIM-MOE-DECODE-PARALLEL-1`).** The two MoE decode kernels that at M=1 (c1)
  launched a single block and ran serially — MoeAlign (29.3 µs vs vLLM 3.6 µs,
  8.2×) and MoeRouterTopK (20.2 µs vs vLLM 6.7 µs, 3.0×), together ~1.7 ms/tok ≈
  63% of the grounded 2.7 ms 35B c1 TPOT gap — are parallelized (router →
  per-thread argmax + tree reduction; align → one-thread-per-expert
  `cub::BlockScan` prefix sum), mirroring vLLM `topk_softmax_kernels.cu` /
  `moe_align_sum_kernels.cu`, plus L3 block_size_m 16→8 at low M and L4 removal of
  4 redundant Marlin workspace memsets. **BYTE-EXACT** to the retained serial
  reference (router 72/72 + align 60/60 parity), **27B 235/235 + 35B 315/315
  token-exact**, memcheck-clean 35B decode; efficiency-only ⇒ shipped ON by
  default. Per-kernel nsys same-box A/B (c1/M=1, `--cuda-graph-trace=node`,
  baseline = serial-kernel build a7d08d7 on-box): **MoeAlign 29.5 µs → 3.0 µs
  (9.8×, now below vLLM's 3.6 µs); MoeRouterTopK 20.2 µs → 12.3 µs (1.64×,** vs
  vLLM's 6.7 µs — byte-exactness ties the softmax reduction to our tree order,
  so vLLM's register-fused `topkGating` is off-limits). Combined excess vs vLLM
  39.2 µs → 5.0 µs per invocation (~87% eliminated). memcheck 35B graph decode
  0 errors. The in-situ c1/c8 TPOT recovery is `benchmark_binding=false` here —
  the 35B v0.25.0 performance grid re-measures it (the ~1.7 ms/tok c1 target; no
  regression at c16/c32, where the serial cost already amortized). Evidence
  `dgx:~/work/vllm.cpp-moe-decode-par`.

## Benchmark policy

- Correctness is a precondition and cannot be traded for speed.
- Ours and every reference use the same model, requests, sampling, token
  budget, cache policy, concurrency, and hardware.
- A benchmark series runs on an idle GPU under one ownership window and is
  repeated enough to distinguish signal from run noise.
- Partial, contended, stale-denominator, or diagnostically incomplete results
  are `VOID`; they never contribute to an accepted ratio.
- Every 27B throughput, latency, and memory axis must pass before 35B
  performance or broader roadmap execution.

The complete contract is in the
[benchmark protocol](../.agents/benchmark-protocol.md) and
[online serving gate spec](../.agents/specs/cuda-online-serving-gate.md).
