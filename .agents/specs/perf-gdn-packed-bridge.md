# GDN packed decode: bridge the dtype PREDICTION to the fp8 GEMM that produces it

**Row:** `PERF-GDN-PACKED-BRIDGE` · **issue:**
[#365](https://github.com/mudler/vllm.cpp/issues/365) · **composes:**
[`perf-27b-gdn-packed-reachable.md`](perf-27b-gdn-packed-reachable.md) `ab0b14a9`
× [`perf-fp8-alpha-fold.md`](perf-fp8-alpha-fold.md) `918d4546` · **status:**
`SPIKE`.

Composed against `918d4546` itself, not `row/PERF-MAXSTACK-27B` `e5407053`
which integrates it: tighter (only the row depended on), and `e5407053` is a
bare `Merge branch ...` commit with NO trailers, which reds `commit-trailers` on
every gate. That is a pre-existing maxstack defect, reported not propagated.

## Issue-table keying (#365 is an umbrella)

`#365` is the multi-lever "27B gap DECOMPOSED" issue: the dense-Marlin `gate_up`
half AND this GDN packed-decode half are both inside it. The issue table is keyed
by issue NUMBER and `scripts/check-agent-record.py` refuses a number listed
twice, so it can hold exactly one row per issue. On `origin/main` that entry
belongs to `PERF-27B-DENSE-MARLIN-GATEUP`, and this row leaves it byte-for-byte
intact rather than clobbering a key it does not own (AGENTS.md, Records).

That means `ab0b14a9` carries a record defect worth reporting rather than
propagating: its roadmap edit REWROTE main's `#365` row in place, retargeting it
from `PERF-27B-DENSE-MARLIN-GATEUP` to `PERF-27B-GDN-PACKED-REACHABLE`. Merging
it forward reintroduces the duplicate and reds `check-agent-record`. The
composition here drops that edit; whoever lands `ab0b14a9` owes the same fix or a
deliberate re-key of the umbrella.

Issues FILED from this row's fresh review and keyed to it in the roadmap table:

| Issue | What it holds |
|---|---|
| [#491](https://github.com/mudler/vllm.cpp/issues/491) | The two toggles are NOT independent — no 2x2, the bf16 lever is decode-negative, its PREFILL justification unmeasured, and the three preconditions for any default flip |
| [#492](https://github.com/mudler/vllm.cpp/issues/492) | The fp8 qkvz `qqsss`->`qqtst` reselection is measured only at M=1, and its reason to exist is a PREFILL cost |
| [#493](https://github.com/mudler/vllm.cpp/issues/493) | doctest 2.5.2 `MESSAGE` renders a string-literal ternary as its first branch unconditionally |

## THIS ROW HAS NO TOKEN GATE ON THE PATH IT MODIFIES

Stated first because it bounds every claim below.

SACRED `test_qwen27_paged_engine` pins `models--unsloth--Qwen3.6-27B-NVFP4` at
revision `890bdef7` (`tests/parity/hf_snapshot.h:31,36`). That is a **bf16-tower**
checkpoint: `LoadGdnDense` only populates `in_proj_qkv_fp8` when the tensor dtype
is `F8_E4M3` (`qwen3_5_dense_weights.cpp:427-430`), and on `890bdef7` it is not.
Every line this row changes is reachable **only** on a native-fp8 GDN tower —
`nvidia/Qwen3.6-27B-NVFP4`@`0893e160`. So a green 235/235 says nothing whatsoever
about this change: the gate never executes the branch.

Consequently:

- **Correctness is NOT claimed by this row.** Not "probably fine", not "byte
  identical by construction" — unclaimed. (`GATE-27B-FP8-TOWER-GOLDEN` has since
  landed and §4 records what its gate does and does not establish: no gross
  defect, at ~50x coarser sensitivity than the perturbation introduced. That
  still is not a correctness claim.)
- Speed MAY be measured, because a speed A/B does not depend on a golden.
- The token gate is owed by `row/GATE-27B-FP8-TOWER-GOLDEN`, which is capturing
  real goldens for `nvidia@0893e160`. **Both toggles must stay DEFAULT OFF until
  that lands and passes**, and no default flip may be argued from this spec.
- The composed arm rounds the fp8 GEMM's f32 accumulator to bf16 *before* the
  alpha multiply instead of after (`perf-fp8-alpha-fold.md`), and swaps the
  decomposed recurrence for a different kernel. Tokens CAN move. A lost token is
  `NEEDS_DECISION`, never a re-cut golden.

## What is broken

Composing the two rows does nothing. Both toggles ON still leaves
`vt::GdnPackedDecode` deselected, because the composed tree carries **two
independent sources for one dtype**:

| | the merged fp8 `mixed_qkv` dtype | consumed by |
|---|---|---|
| maxstack | `fp8_indt` = `GdnFp8InBf16Enabled() && indt==BF16 && outdt==BF16` -> **BF16** | `MergedFp8QkvzD`, i.e. the GEMM that actually allocates the buffer |
| REACHABLE | `GdnFp8MixedQkvDType()` = `return DType::kF32;` — no env dependency at all | `GdnProjectedMixedQkvDType` -> `GdnPackedDecodeDTypesCompatible` |

`GdnPackedDecodeDTypesCompatible` requires `mixed_qkv == kBF16`, so it returns
false, `ShouldUsePackedGdnDecode` returns false, and the arm is inert. ON == OFF.
Any A/B run in that state measures nothing and would be VOID.

This is fail-safe (the prediction under-reports capability, so nothing aborts and
no wrong kernel runs) but it is exactly the drift `ab0b14a9` introduced the helper
to prevent.

### The guard that was supposed to catch it tested the wrong side

`ab0b14a9` guarded the merged arm with

```cpp
VT_CHECK(GdnFp8MixedQkvDType() == DType::kF32,
         "qwen3_5 merged FP8 GDN qkvz: the merged arm emits F32; "
         "GdnFp8MixedQkvDType must agree");
```

That asserts a property of the PREDICTOR against a literal. It cannot observe
what the GEMM allocates, so it passes unchanged while the merged arm emits bf16.
Resolving the compose conflict to maxstack's side deletes it outright. Either
way the invariant it names — predictor agrees with producer — is unprotected.

## Design

### 1. The bridge

`GdnFp8MixedQkvDType()` stops being a constant and takes maxstack's own
three-term expression, plus the arm selector, because **the toggle is
merged-arm-only**: `fp8_indt` reaches `MergedFp8QkvzD` and nothing else; the
split fp8 arm still hardcodes `DType::kF32` on both of its call paths
(`MatmulFp8CutlassPreQuantD` / `MatmulFp8CutlassD`). A predictor that ignored
the arm would claim BF16 on a checkpoint that takes the split path and be wrong
in the UNSAFE direction (predict BF16, produce F32 -> `vt::GdnPackedDecode`
throws on the uniformity check).

So one new input joins `GdnMixedQkvDTypeInputs`:

```
fp8_merged_arm  // ShouldUseMergedGdnFp8Qkvz(...) held for this layer
```

and the fp8 leg of the prediction becomes
`fp8_merged_arm ? fp8_out_dtype : kF32`.

`GdnMergedFp8QkvzEligibilityFor(d, w, conv_dim, value_dim)` is already in scope
at the eligibility call site, and `ProjectGdnQkvz` selects the arm with the same
predicate over the same inputs, so the two cannot disagree.

### 2. The replacement guard — asserts the real invariant, both directions

The point of the helper is that the PREDICTION equals what the GEMM ALLOCATES.
So the guard compares those two, on both arms, in both directions:

- merged arm: `plan dtype == predictor(fp8_merged_arm=true)`
- split arm: `hardcoded F32 == predictor(fp8_merged_arm=false)`

A one-sided `== kF32` is what failed; an equality between producer and predictor
fails whichever side moves. Placed where the buffer is allocated, so it observes
the value actually used rather than re-deriving it.

### 3. Deliberately unchanged

- Every default. Both toggles stay OFF.
- `ShouldUsePackedGdnDecode`, `GdnPackedDecodeDTypesCompatible`,
  `vt::GdnPackedDecode`'s contract, and every kernel.
- The split arm's dtype. Narrowing it is `perf-fp8-alpha-fold.md`'s to make, not
  this row's.

## Tests (RED first)

CPU tier, `tests/vllm/models/test_qwen27_paged_forward.cpp`. Composed-tree
baseline before any edit: **27 cases / 752 assertions, `Status: SUCCESS`**. A
changed assertion count is RED even when it prints "passed".

RED must fail for the intended reason — that the predictor is a constant:

1. merged arm + toggle ON -> predictor says BF16 (fails at HEAD: says F32).
2. merged arm + toggle OFF -> F32.
3. **split** arm + toggle ON -> F32 (the merged-only property; this is the term
   that keeps the prediction from lying in the unsafe direction).
4. `indt`/`outdt` rollback each independently force F32 even with the toggle ON.
5. end-to-end: `GdnPackedDecodeDTypesCompatible` fed the predictor's output is
   true only in case 1.

## Gates

- `scripts/agent-preflight.sh --staged` and on committed HEAD.
- CPU: `test_qwen27_paged_forward`, case AND assertion counts before/after.
- SELECTION (GPU, `-DVLLM_CPP_TRITON=ON` — without it
  `RecordGdnPackedDecodeTritonLaunch` is compiled out and `triton_launches`
  reads 0, indistinguishable from "did not fire"): after
  `ResetGdnPackedDecodeDebugStats()`,
  `packed_launches == 48 * steps && triton_launches == 48 * steps` is the cubin
  firing; `packed_launches == 48*steps && triton_launches == 0` is the hand
  kernel with the cubin REJECTED; `packed_launches == 0` is the model never
  selecting packed.
- NOT a gate here: any token count. See the first section.

## Expected payoff

`GdnDecodeFusedKernel` 28.08 us/call x48 vs vLLM's packed 19.21 us/call
(1.3109 vs 0.9084 ms/step) — about **+0.425 ms/step** of a +1.81 ms/step decode
deficit, plus `GdnPostConvFastKernel` (+0.131 ms/step, no vLLM counterpart)
which the packed kernel absorbs. Anything materially larger is a measurement
defect, not a windfall.

## Stop conditions

- Predictor and producer disagree anywhere -> the new guard fires; fix the
  bridge, never widen the guard.
- Selection counters show `packed_launches == 0` with both toggles on -> report
  the actual deselecting term; do not widen `ShouldUsePackedGdnDecode`.
- `triton_launches == 0` with `packed_launches > 0` -> report which
  `TryTritonPackedDecode` guard rejected it; do not relax the AOT predicate.
- Any token claim -> refused until `row/GATE-27B-FP8-TOWER-GOLDEN` lands.

## Measured (dgx, GB10 sm_121a, 2026-08-12)

Merged forward onto `origin/main` `a2ca83a8`, which lands
`row/GATE-27B-FP8-TOWER-GOLDEN` — the first gate that EXECUTES the fp8 tower, and
the token gate this spec recorded as owed. CPU tier unchanged at **29 cases /
765 assertions, SUCCESS**.

Build: `RelWithDebInfo`, `-DVLLM_CPP_TRITON=ON`,
`-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`,
`-DVLLM_CPP_CUDA_ARCHITECTURES=121a`, 0 warnings. Configure log verified to
print `CUTLASS found … enabling sm120a NVFP4 cutlass GEMM`,
`CUDA feature fa2: ENABLED for [121a]` and `Triton AOT: gdn_decode_h48 <-
sm_121a`. One `flock $HOME/gpu.lock` acquisition, queued behind two other jobs,
never contended.

### 1. SELECTION — PROVEN, from an EAGER step

The counters are HOST-DISPATCH counts and CUDA graph REPLAY performs no host
dispatch, so they read 0 in a graphed throughput run whatever was selected. They
are therefore read from `test_qwen27n_fp8_tower_paged_engine`, the only harness
that loads `nvidia@0893e160` AND steps eagerly, after
`ResetGdnPackedDecodeDebugStats()` and one pure non-spec decode step:

| arm | `packed_launches` | `triton_launches` |
|---|---|---|
| default (no lever) | **0** | 0 |
| `VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1` | **48** | **48** |

`48 && 48` is the vendored FLA cubin firing on every GDN layer — the goal state,
not the `48 && 0` hand-kernel-with-cubin-rejected state and not the
`0` never-selected state. Corroborated independently: the arm exactly as it
landed on main, whose contract is the constant `packed_launches == 0`, PASSES
without the lever and FAILS with it (`231 assertions | 1 failed`, exit 1).

### 2. Decode A/B — two independent instruments agree

Per-kernel (`nsys -t cuda --cuda-graph-trace=node`, `cuda_gpu_kern_sum`, 63
decode steps, one run per arm). Only kernels whose identity or count CHANGED are
attributable; same-kernel/same-count rows are run-to-run variance:

| change | OFF ms/step | ON ms/step | delta |
|---|---|---|---|
| `GdnDecodeFusedKernel` -> `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 1.3647 (28.43 us/call) | 0.9645 (**20.09 us/call**) | **-0.4002** |
| `GdnPostConvFastKernel` absorbed (3024 decode calls gone) | 0.1348 | 0.0039 | -0.1309 |
| `CastBf16Kernel` 4096 -> 1024 instances | 0.0758 | 0.0191 | -0.0567 |
| `CausalConv1dUpdateFastKernel` f32 -> bf16 | 0.1392 | 0.1219 | -0.0173 |
| `MulColVecF32Kernel` f32 -> bf16 | 0.1113 | 0.1090 | -0.0023 |
| `gemvx` template swap | 0.5001 | 0.4988 | -0.0013 |
| merged fp8 qkvz GEMM `nvjet…qqsss_192x48x128` -> `…qqtst…` (D f32->bf16) | 18.6259 (381.98 us/call) | 18.7209 (383.92 us/call) | **+0.0950** |
| **net attributable** | | | ~~-0.5137~~ **do not quote (below)** |

**Corrected on review: only the STRUCTURAL terms survive the run-to-run drift,
and `-0.5137` must not be quoted to four figures.** The control is
`marlin::Marlin` — 8,128 instances in BOTH traces, touched by nothing in this
row — and it moves **+0.58% = +0.262 ms/step** between them, half the size of the
"net attributable" above. What is larger than that drift is the pair whose kernel
identity CHANGES or that DISAPPEARS: **-0.400** (recurrence swap) and **-0.131**
(post-conv absorbed). The `+-0.09` rows — the -0.0776 bf16 cascade and the
+0.0950 fp8 qkvz GEMM — are INSIDE the drift and are candidates, not findings.
One run per arm cannot separate them.

End-to-end same-binary A/B (`vllm-bench`, c1, in 16 / out 256, 3 requests per
leg, order-alternated `off,on,on,off,off,on`):

| arm | n | legs (median TPOT ms) | median | spread |
|---|---|---|---|---|
| OFF | 3 | 83.17, 83.33, 83.33 | **83.33** | 0.16 ms (0.19%) |
| ON | 3 | 82.66, 82.72, 82.79 | **82.72** | 0.13 ms (0.16%) |

The arms SEPARATE: every ON leg is below every OFF leg. **-0.61 ms/step
(-0.73%)** by median, -0.56 by mean, against a within-arm spread of <0.2% — and
the ordering rules out drift, because the run warms slightly (the earliest OFF
leg is the fastest OFF leg and the latest ON leg the slowest ON leg), which
works AGAINST the effect rather than for it.

### 3. Against the PIN

Denominator: `~/venvs/vllm-oracle-next` addressed by EXPLICIT path, never the
canonical `vllm-oracle` symlink, which still points at the v0.25.0 ROLLBACK
(#375). Identity ASSERTED before the run and aborting otherwise —
`vllm.__version__ = 0.23.1rc1.dev1511+g555967922`, FlashInfer `0.6.15.post1`,
Torch `2.13.0+cu130`. GRAPHED (`vllm serve`, no `--enforce-eager`),
`--language-model-only`, `--gpu-memory-utilization 0.55`, 115 GiB host RAM free
at start. Same checkpoint, same in 16 / out 256, same 3 requests, same
`--max-concurrency 1`, same greedy sampling, `--ignore-eos`.

| arm | median TPOT | spread | vs PIN | decode gap |
|---|---|---|---|---|
| PIN (81.39, 81.40, 81.39) | **81.39 ms** | 0.01% | 1.000x | — |
| ours OFF | 83.33 ms | 0.19% | **0.977x** | +1.94 ms/step (+2.38%) |
| ours ON | 82.72 ms | 0.16% | **0.984x** | +1.33 ms/step (+1.63%) |

**Corrected on review: `0.977x -> 0.984x` is INDICATIVE, not binding.** Two
methodology gaps separate the numerator from the denominator. (1) The oracle
driver `bridge-oracle.sh:51` stops `local-ai-worker` before its legs; the script
that produced ALL SIX of our legs and BOTH nsys traces, `bridge-measure.sh`, does
not — so the arms did not share a machine state. (2) The arms were NOT
interleaved: ours 16:52-17:04, the pin 17:14-17:18, under separate lock
acquisitions, and `oracle.log` pin-leg 3 reports Mean 83.57 / P99 87.88 against
an 81.39 median, which is a real interference event inside the PIN legs rather
than a tail artefact. Both defects inflate the pin's time, so the true ratio is
no better than quoted and this number is CONSERVATIVE — but an interleaved re-run
under one machine state is OWED before it is binding.

The OFF row REPRODUCES the deficit this row was scoped from (+1.81 ms/step,
+2.33%) at +1.94 / +2.38%, which is the strongest available check that the
harness is measuring the same thing the original profile did. The lever closes
**31%** of the measured decode gap.

Medians are used and the reason is recorded rather than hidden: the PIN's per-leg
MEANS are 81.40, 81.39 and **83.57** — the third leg carries one slow request
whose tail moves the mean 2.7% while the median does not move at all. Ours has no
such leg (mean equals median to 0.01 ms on five of six legs). A 2.7% mean spread
on one arm is inside the ~5% band but it is the largest single source of doubt in
this comparison, and it is exactly why the ratio is quoted from medians.

TTFT is NOT the story here and is recorded so nobody has to guess: ours ~96 ms
vs the pin's 131.6 ms at this input length, i.e. we are AHEAD on prefill on this
workload, so the decode ratio above is not hiding a prefill regression.

The output-token-throughput axis DISAGREES with the decode axis and is not
quoted as decode: ours 11.20 (OFF) / 11.26 (ON) vs the pin's 12.17 tok/s, a
0.92x that TTFT and TPOT together do not account for — 96.5 ms + 255 x 83.33 ms
predicts ~12.0 tok/s for our OFF arm, so about 1.4 s per request sits outside
BOTH reported metrics in our frontend. That is a real open item, it is larger
than this lever, and it belongs to whoever owns the bench frontend rather than to
this row. It is named here so nobody quotes 0.92x as a decode ratio or 0.98x as a
throughput one.

### 3b. Against the prediction

`## Expected payoff` predicted -0.425 (packed swap) -0.131 (post-conv absorbed)
= -0.556 ms/step and warned that anything materially larger is a measurement
defect. Those two terms MEASURE -0.4002 -0.1309 = **-0.5311**, within 4.5% of
prediction, and the e2e -0.56/-0.61 lands on the same number from a different
instrument. Nothing here is a windfall.

### 4. Tokens

`test_qwen27n_fp8_tower_paged_engine` on `nvidia@0893e160`: **236 assertions,
SUCCESS, 16/16 token-exact** with the lever OFF *and* ON. The packed kernel is a
DIFFERENT kernel and the arm rounds the fp8 accumulator earlier, so this was not
byte-exact by construction — it is measured. SACRED `test_qwen27_paged_engine`
(`unsloth@890bdef7`): **235 assertions, SUCCESS, 16/16 token-exact** with the
lever OFF and ON, confirming the lever is inert on a BF16 tower.

**What that does and does not establish, stated exactly.** The 27n arm admits a
**x1.10** fp8 scale perturbation and still passes. The perturbation this row
actually introduces is bf16 rounding of the `in_proj` D, ~2^-9 ~ **0.2%**. The
gate is therefore about **50x coarser than the change it is asked to bound**.
And no test anywhere compares `vt::GdnPackedDecode` against `vt::GdnDecode` —
precisely the substitution this row makes reachable; the only op-level bound is
packed-vs-CPU-reference at 2% rtol (`tests/vt/test_ops_gdn.cpp:~1503`), so the
mutual bound between the two kernels is only ~4%. The ON arm additionally
BUNDLES two numerical changes — bf16 D narrowed before the alpha multiply, and a
different recurrence kernel — and neither is isolated from the other.

So the correct phrasing, and the only one this row licenses, is: **no gross
defect, at ~50x coarser sensitivity than the perturbation introduced, with two
changes confounded.** Not "token-safe", not "numerically equivalent". A direct
op-level `GdnPackedDecode` vs `GdnDecode` bound at real 27B decode shapes is
**OWED** and is a precondition for any default flip.

### 5. What is NOT established

- **The residual against vLLM's own launch of the same cubin.** Ours runs the
  FLA packed decode at 20.09 us/call; the profile this row was scoped from
  records vLLM at 19.21 us/call — about 4.6% above, ~0.042 ms/step, on the
  IDENTICAL kernel, which is a launch/argument-side difference and not a
  ceiling. Not re-measured here.
- **The +0.0950 ms/step on the merged fp8 qkvz GEMM.** Narrowing D to bf16
  reselects the cuBLASLt template (`qqsss` -> `qqtst`) and that template is
  ~0.51% slower per call at this shape. It is one run per arm and smaller than
  the Marlin run-to-run variance in the same table, so it is a candidate, not a
  finding. It is the `VT_GDN_FP8_IN_BF16` half of the composition and it pays
  for itself several times over, but it should be measured per-shape before
  anyone quotes it.
- **Absolute scale.** 83 ms/step (~11.2 tok/s) is this harness at its defaults
  (auto-fit `max_model_len` 8192, 256 blocks), not a tuned production point.
  Only the paired delta is claimed. Likewise the nsys per-step totals (~42
  ms/step of kernel time) are instrument-inflated and only their paired
  difference is used.
- **The ~1.4 s/request our frontend spends outside TTFT and TPOT** (§3). It is
  the reason the throughput and decode ratios differ by ~6%, and nothing here
  attributes it.
- **Whether the win survives at the gate's concurrency.** Everything above is
  c1. The GDN recurrence is per-sequence, so the saving should scale with batch,
  but that is an expectation and not a measurement.
- **The two toggles are NOT independent, so there is no 2x2 and no attribution
  between them.** `VT_GDN_PACKED_DECODE_FP8_TOWER` is INERT without
  `VT_GDN_FP8_IN_BF16`, so the only reachable arms are OFF/OFF and ON/ON. Read
  that way, the packed kernel earns the whole structural -0.531 while the bf16
  lever's own decode terms sum to **+0.017 — slightly NEGATIVE in decode**. It is
  carried here purely as the price of admission. Its actual justification — a
  122.99 ms/req PREFILL pass at T~4096 — is entirely UNMEASURED by this row,
  which ran `input_len=16`.
- **The `qqsss`->`qqtst` reselection at large M.** The +0.095 ms/step is a real
  deterministic cuBLASLt reselection for a bf16 D, and its whole reason to exist
  is a PREFILL cost, so the shape that would justify it is the one shape not
  measured.

Before any default flip, three things are required and none of them exist yet:
prefill / long-input both-arms nsys at T~4096; a concurrency sweep (c8/c16/c32);
and an op-level `GdnPackedDecode` vs `GdnDecode` bound at real 27B decode shapes.

## Now

Selection PROVEN (48/48). Decode A/B MEASURED, but only its two STRUCTURAL terms
(-0.400, -0.131) survive the untouched control's own +0.262 ms/step drift, and
the `0.977x -> 0.984x` against the pin is INDICATIVE, not binding: the arms ran
under different background conditions and were not interleaved. Tokens do not
move on either checkpoint, which establishes **no gross defect at ~50x coarser
sensitivity than the perturbation introduced, with two changes confounded** —
not equivalence. Both toggles remain DEFAULT OFF and this row does not argue a
flip; the preconditions for one are listed at the end of §5.
