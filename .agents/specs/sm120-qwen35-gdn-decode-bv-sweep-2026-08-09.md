# sm_120 Qwen3.5 fused GDN decode value-tile sweep

**Rows:** `KERNEL-SSM-MAMBA`, feeding `ROAD-V1-C2-LOCAL-BF16`.
**Lifecycle:** `COMPLETE` local discriminator. BV8 and BV24 were byte-exact but
performance-rejected and removed; BV16 remains the one opt-in incumbent and
BV32 remains shipped default. The generic Mamba row remains `INVENTORIED`;
default and Qwen3.6-27B/35B release gates stay open.

## Intake, evidence and fixed scope

This records-only spike is based on current head `26122c7e5debaf152b29f589fc778e532f6e5eb9`
and `upstream/main` `ba8d867c9ea1aab0896af9c741924ba9855e0400`.
The accepted BV16 result at product commit `92256e6a9` keeps NW8 and every
per-row operation byte-ordered while changing the production launch from
grid-x 4 / block 256 to grid-x 8 / block 128. On the dominant grid-y 800 shape
it improves **183.526→163.432 us/call (-10.95%)**, but remains **1.276x** behind
the pinned same-tool vLLM result, **128.061 us/call**. BV16 is the incumbent;
shipped BV32 is the default control. The correctness-invalid NW4 experiment
changed the Dk partition/reduction order and is forbidden here.

The local execution chain is `GdnDecodeFusedCuda` →
`DispatchGdnDecodeValueTile` → `LaunchGdnDecodeFused{S,NW}` →
`GdnDecodeFusedKernel` at `src/vt/cuda/cuda_gdn.cu:2700-2931`, with the
CUDA-free parser, ceil geometry and callback dispatcher at
`src/vt/cuda/gdn_decode_fused.h:11-67`. The same-process BV16/BV32 byte matrix
is at `tests/vt/test_ops_gdn.cpp:1865-1951,3464-3481`; its portable contract is
`tests/vt/test_gdn_decode_fused.cpp:14-85`.

Pinned vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` vendors FLA at
`${VLLM_SOURCE}/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py:24-277`:
the recurrence is `:122-170`, `BV=min(next_power_of_2(V),32)` is `:205-212`,
grid `(NK,NV,N*HV)` is `:241`, and `num_warps=4` is `:212,275`. This sweep is
a local schedule discriminator, not a claim that BV8/BV16/BV24 mirrors FLA's
BV32 choice. No CUTLASS, cuBLASLt or DeepGEMM layer executes this recurrence.

## Exact design and launch contracts

Change only independent value-row tiling. Keep `NW=8`, each row's eight
consecutive Dk-slice lanes, `c0/c1`, shuffle offsets, decay, dot/update/output
arithmetic, tail guards, output store and state writeback unchanged. Do not
specialize, duplicate or reorder the recurrence. For Dv=Dk=128:

| Candidate | grid-x `ceil(128/BV)` | block `BV*NW` | dynamic shared bytes `(2*Dk + BV*(Dk+1))*4` |
|---|---:|---:|---:|
| BV8 | 16 | 64 | 5,152 |
| BV16 incumbent | 8 | 128 | 9,280 |
| BV24 | 6 | 192 | 13,408 |
| BV32 shipped | 4 | 256 | 17,536 |

The BV24 sixth block is a guarded eight-row tail; grid rounding must be ceiling,
never floor. General geometry remains `BV=min(Dv,requested)`,
`NV=ceil(Dv/BV)`, `NW_eff=(Dv>=32 ? requested_nw : 1)`, block
`BV*NW_eff`, and the shared-byte formula above. Zero Dv/Dk is the existing
no-launch contract and the existing >48 KiB sequential fallback is unchanged.

Extend `VT_GDN_DECODE_BV` with strict full-string parsing. Exactly `"8"`,
`"16"`, `"24"` and `"32"` resolve BV8/BV16/BV24/BV32 respectively. Unset,
empty, whitespace, signed or zero-padded forms, prefixes/suffixes, overflow and
every other value resolve shipped BV32. No `atoi` or prefix parsing. Production
must use the same four-callback portable dispatcher that tests exercise.

## RED-first correctness and mutation contract

Before product edits, extend the portable test so it fails on the current
two-arm contract. It must prove exact parser/fallback semantics; all four
production geometries/shared sizes; BV24 ceiling/tail and partial dimensions;
the Dv<32 NW1 corner; zero-dimension no-launch; and exactly one callback for
each candidate. Deleting or rerouting either new arm, relaxing parsing, changing
ceil to floor, or corrupting a block/shared calculation must make it RED.

Expand the CUDA matrix to run BV8, BV16, BV24 and BV32 in one process and
require raw-byte equality of output and final state against BV32. Cover compact
and indexed state, BF16 and F32 state, production Dv=Dk=128, a partial Dv, and
an indexed null row whose output is zero and unreferenced state stays untouched.
Mutate each new dispatch, the BV24 last-tile guard, row mapping, and both
output/state stores; each mutation must fail focused coverage and restoration
must return the tree byte-for-byte.

Build and run `test_gdn_decode_fused`, full CUDA `test_ops_gdn`,
`test_qwen35_paged_forward`, and `vllm-bench` from one CUDA/Triton/CUTLASS
build. The production 128-request greedy token output for every arm must equal
SHA-256 `83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.
Any byte mismatch stops performance work. Run focused, staged and full gates,
then fresh static and targeted-mutation review; the operator reruns the declared
gate independently.

## Same-binary trace and selection gate

Under one GPU lease and `flock /tmp/gpu`, use one rebuilt binary and the cached
Qwen3.5-4B BF16 workload: 128 ShareGPT requests, 128 output tokens, c32,
`max_num_batched_tokens=2048`, 1,280 KV blocks and greedy sampling. Keep every
other selector identical. Run one mirrored eight-leg series:

`BV32a → BV16a → BV8a → BV24a → BV24b → BV8b → BV16b → BV32b`.

Collect `nsys --cuda-graph-trace=node`, exported SQLite, raw summaries, token
hashes, binary/build SHA, clocks/contention, peak VRAM and peak PSS for every
leg. Launch evidence must show all four exact grid/block/shared contracts above,
NW8, the fused kernel (not scan fallback), identical candidate call counts, and
the dominant grid-y 800 group separately from all fused-decode calls.

A new candidate is eligible only when its counterbalanced mean grid-y 800 time
is at least **1.00% lower than BV16**, its all-fused-decode mean does not regress
against BV16, both samples have the same direction, and every token hash is
identical. If both qualify, select the lower grid-y 800 mean; a noisy/mixed
ordering retains BV16. The selected opt-in then must not regress against BV16
on any enclosing mean: total and output throughput may not decrease; TTFT,
TPOT, ITL, E2E, peak VRAM and peak PSS may not increase. Record values and
ratios for every axis. Default BV32 receives no displacement credit.

At disposition, retain exactly shipped BV32 plus one opt-in incumbent: BV16 if
neither new candidate clears every bar, otherwise the selected BV8 or BV24.
Remove every losing dormant selector, enum/dispatcher arm, instantiation and
test expectation in the same implementation change; the removed selector
strings must then fall back to BV32. Preserve rejected measurements in the
append-only benchmark record and measured result section here. Even a local win
stays opt-in: default/release gates remain open and the 27B/35B vehicles are
hardware-unavailable on this host.

## Measured result — BV8/BV24 rejected, BV16 retained opt-in

Sweep implementation `e102a14de` passed operator runtime verification: portable
selector/geometry **3/3 · 100**, CUDA GDN **69/69 · 4,850** (including public
production-graph geometry) and Qwen3.5 paged-forward **4/4 · 8**. The standard
128-request greedy workload ran in the prescribed order
BV32a→BV16a→BV8a→BV24a→BV24b→BV8b→BV16b→BV32b. All eight token files have
SHA-256 `83fcdc45f79ddb06a634c7d7d95eba3384543b3cd781a45a8db1fc4e2a453545`.

| Arm | all fused decode, 1,704 calls | grid-y 800, 816 calls | total / output tok/s | TTFT | TPOT / ITL | E2E |
|---|---:|---:|---:|---:|---:|---:|
| BV32a | 302.606110 ms | 153.728820 ms | 6733.45 / 744.57 | 1021.35 ms | 35.04 ms | 5471.00 ms |
| BV16a | 271.472694 ms | 136.909597 ms | 6743.80 / 745.71 | 1020.07 ms | 34.98 ms | 5462.59 ms |
| BV8a | 277.729795 ms | 140.484486 ms | 6741.12 / 745.41 | 1021.16 ms | 34.99 ms | 5464.71 ms |
| BV24a | 279.819526 ms | 141.534196 ms | 6734.28 / 744.66 | 1022.07 ms | 35.03 ms | 5470.34 ms |
| BV24b | 279.639897 ms | 141.524051 ms | 6731.40 / 744.34 | 1022.61 ms | 35.04 ms | 5472.57 ms |
| BV8b | 279.157494 ms | 141.735960 ms | 6738.32 / 745.10 | 1021.23 ms | 35.01 ms | 5466.97 ms |
| BV16b | 271.707232 ms | 137.034656 ms | 6736.22 / 744.87 | 1022.23 ms | 35.01 ms | 5468.79 ms |
| BV32b | 302.982021 ms | 153.870033 ms | 6727.95 / 743.96 | 1023.06 ms | 35.06 ms | 5475.47 ms |

The counterbalanced means are:

| Arm | all fused decode | vs BV16 | grid-y 800 | vs BV16 |
|---|---:|---:|---:|---:|
| BV32 | 302.7940655 ms | +11.4894% | 153.7994265 ms | +12.2852% |
| BV16 | 271.5899630 ms | control | 136.9721265 ms | control |
| BV8 | 278.4436445 ms | **+2.5235%** | 141.1102230 ms | **+3.0211%** |
| BV24 | 279.7297115 ms | **+2.9971%** | 141.5291235 ms | **+3.3270%** |

Both samples of each new candidate are slower than their corresponding BV16
sample. BV8 and BV24 therefore fail both the required ≥1% dominant-shape win
and the all-fused non-regression bar; neither reaches downstream selection.
BV16 remains a repeatable win over BV32 in this series: **-10.3054%** across all
fused calls and **-10.9411%** on grid-y 800.

| Arm mean | total tok/s | output tok/s | TTFT | TPOT / ITL | E2E |
|---|---:|---:|---:|---:|---:|
| BV32 | 6730.700 | 744.265 | 1022.205 ms | 35.050 ms | 5473.235 ms |
| BV16 | 6740.010 | 745.290 | 1021.150 ms | 34.995 ms | 5465.690 ms |
| BV8 | 6739.720 | 745.255 | 1021.195 ms | 35.000 ms | 5465.840 ms |
| BV24 | 6732.840 | 744.500 | 1022.340 ms | 35.035 ms | 5471.455 ms |

The profiler confirms exact Dv=Dk=128 launches at 52 registers/thread: BV8
grid-x 16 / block 64 / 5,152 shared bytes; BV16 8 / 128 / 9,280; BV24 6 /
192 / 13,408; and BV32 4 / 256 / 17,536. Disposition cleanup commit
`634ccba70` removes the rejected BV8/BV24 selectors, dispatch arms,
instantiations and expectations, so exact strings `"8"` and `"24"` again fall
back to BV32. Exact `"16"` remains the sole opt-in and unset remains BV32.

The current series' dominant BV16 mean is **136.9721265 ms / 816 =
167.858 us/call** versus pinned vLLM **128.061 us/call**, about **1.311x**
slower. The prior accepted BV16 series reported **1.276x**; that separate run is
preserved rather than overwritten, and the difference is treated as run
variance rather than a new oracle claim. The residual, default gate and
hardware-unavailable 27B/35B release gates remain open.

Evidence is preserved as
`/tmp/qwen35-gdn-bvsweep-{32a,16a,8a,24a,24b,8b,16b,32b}-e102a14de.{nsys-rep,sqlite,log,tokens.json}`.
Trace SHA-256 values in that order are
`323bf5a4c28b46bc2b4bdd1ee7469abb0ad3f2205d6866c11c8d07fff17942e8`,
`21a48aa428eebf5c85ded20eb309a7f8ca5272a3e6c8d34aa5ca18e66f336aef`,
`8266020bd8dae1a1580484e9e8bf6ce3ac427598d588393704793d47b7e1e260`,
`bd36b8470dd3f563cc494e9e9e5215b846cd38ec0e2f34d104406586ff0c3229`,
`ed046407514480c281e6de2511c27a566c7da7504970159b0b0f574605f003f1`,
`f7c9931f3953a836f612d8d940ef91b8657b3f90940e66a75e95cc4f744ee374`,
`e22aa55a1094ef9282a28a107d26a3297409ff5f74d24ce6f138380ce2a21db8`
and `7a6cbb1281acf9fd23c9923156d744d76d54af1765b7752345e18dc4c50cfcce`.
