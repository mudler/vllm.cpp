# Qwen3.8-27B Q4_K_M token exactness against llama.cpp `b10451`

Row: `QUANT-QWEN38-27B-GGUF-ARM`.
Issue: [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
Prior measurement:
[`docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`](../../docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md).

## Why this exists

The arm's declared token gate ran on 2026-08-23 and FAILED: tokenizer exact
6 of 6, generation divergent 5 of 6, every divergence a rank-2 loss under 0.18
logits with 282 of 288 steps at rank 1. `AGENTS.md` §Gates admits no speed or
memory result from this arm on any backend until that gate passes, so the whole
k-quant throughput lane is blocked behind it, and it already invalidated one
gfx1151 measurement (#2497).

The prior evidence named the next hypothesis and did not test it: "suspect the
quantized dot product and the activation dtype first". This row tests it.

## What the prior run actually measured, restated precisely

The issue title says "on CUDA". That means the HOST was an NVIDIA box and NOT
that the compute ran on a GPU. `/mnt/nas_share/rc/qwen38w3/job/job.sh` builds
llama.cpp with `-DGGML_CUDA=OFF` and vllm.cpp with `-DVLLM_CPP_CUDA=OFF`, and
runs `vllm-cli --model "$GGUF" --device cpu`. **Both engines executed their CPU
tiers, on aarch64 (`thor:gpu0`, 14 cores).** Every conclusion below is about the
CPU path. That is not a weakening of the issue: it is where the defect is, and
the ROCm arm the developer's goal names runs the same model code.

## The two candidate error terms, and their measured sizes

### Term A: our activations are BF16 where the oracle's are F32

vllm.cpp's `qwen35` CPU decode carries the residual stream and most projection
outputs in `bf16`:

- `hidden` is a literal `DType::kBF16` at every allocation site, with no toggle
  ([qwen3_5.cpp:8885](../../src/vllm/model_executor/models/qwen3_5.cpp#L8885)).
- `res` is `ResidualDType()`, `kBF16` unless `VT_BF16_RESIDUAL=0`
  ([qwen3_5.cpp:3483](../../src/vllm/model_executor/models/qwen3_5.cpp#L3483)),
  and its own comment records the choice as a memory-traffic one.
- The RMSNorm residual is added in f32, **rounded to the residual dtype, and
  then re-read**, twice per layer
  ([cpu_ops.cpp:576-583](../../src/vt/cpu/cpu_ops.cpp#L576)).
- The quantized GEMM accumulates in f32 and rounds once on store,
  `F32ToBF16`, for qkv, o_proj, down_proj, the GDN in/out projections and the
  lm_head GEMM
  ([cpu_quant_gemm.cpp:52-57](../../src/vt/cpu/cpu_quant_gemm.cpp#L52), reached
  from the `nrc == 1` decode path at
  [:107](../../src/vt/cpu/cpu_quant_gemm.cpp#L107)).
- The paged KV cache is `bf16` by default
  ([kv_cache_dtype.h:29](../../include/vllm/v1/kv_cache_dtype.h#L29)).
- The GDN conv and SSM state caches are `bf16`, gathered to f32 and scattered
  back **every token**, so their rounding is recurrent
  ([qwen3_5_common.cpp:52-63](../../src/vllm/model_executor/models/qwen3_5_common.cpp#L52)).

llama.cpp at `b10451` has none of that on this path. Its CPU graph is f32, and
its KV cache defaults to `GGML_TYPE_F16`, not bf16
(`src/llama-context.cpp:3538-3539`). BF16 carries 8 explicit mantissa bits, so
one store costs up to `2^-9 = 1.95e-3` relative; F16 carries 11 and F32 carries
24.

### Term B: ggml does not agree with itself across its own k-quant kernels

We port ggml's `_generic` tier only, by the header comment's own statement
([cpu_quant_dot.cpp:22-26](../../src/vt/cpu/cpu_quant_dot.cpp#L22)), and
`VecDotQ4_KQ8_K` is byte-for-byte `ggml_vec_dot_q4_K_q8_K_generic` at the pin.
The oracle does not run that body. At `b10451` a Q4_K matvec has four distinct
fp32 rounding schedules in one source tree, and the aarch64 build the gate used
takes the repacked one:

| schedule | where | fp32 applications of `d` per superblock |
|---|---|---|
| `ggml_gemv_q4_K_8x8_q8_K` (aarch64) | `ggml-cpu/arch/arm/repack.cpp:709` | 8 |
| `ggml_vec_dot_q4_K_q8_K` (SVE/NEON) | `ggml-cpu/arch/arm/quants.c:2334` | 1 |
| `ggml_vec_dot_q4_K_q8_K_generic` — **ours** | `ggml-cpu/quants.c:696` | 8 lanes, tail-folded |
| `ggml_gemv_q4_K_8x8_q8_K_generic` | `ggml-cpu/repack.cpp:958` | 16, deferred min |

The integer inner product is exact in all four; only the `int32 -> float`
conversion granularity and the order of the `d`/`dmin` multiplies differ.

**Measured, on this dev box (x86-64 AVX-512, Ryzen 9 9950X3D), against OUR OWN
compiled kernels rather than by reading source.** The harness links
`src/vt/cpu/cpu_quant_dot.cpp` and `cpu_quant_act.cpp` -- compiled with the
project's own `-ffp-contract=off` polarity -- against `libggml` built at
`b10451`, and asks three answers for the same bytes: ours, ggml's `_generic`,
and ggml's dispatched arch kernel. 2000 random rows per shape, `K` in
{2048, 5120, 12288}:

| quantity | result |
|---|---|
| our `QuantizeRowQ8_K` vs `quantize_row_q8_K_ref` | **byte-identical**, 0 of 18,000 rows, 0 of 4.6 M quants |
| our `VecDotQ6_KQ8_K` vs `ggml_vec_dot_q6_K_q8_K_generic` | **bit-identical**, 0 of 6000 rows |
| our `VecDotQ4_K/Q5_K` vs their `_generic` | differ on 80-89% of rows, by `max_abs` 1.1e-06 to 4.3e-06, `rms_rel` 5.5e-06 to 1.1e-04 |
| ggml arch vs ggml `_generic` | differ by `max_abs` 7.2e-07 to 5.4e-06, `rms_rel` 1.4e-05 to 9.8e-05 |

Two things follow. **The `QuantizeQ8KK` hypothesis the issue lists is refuted on
this path**: the activation encoder reproduces the reference byte for byte. And
our Q4_K/Q5_K last-ULP disagreement with the body we port is the SAME SIZE as
ggml's disagreement with itself, so it is a rounding schedule and not a port
defect -- Q6_K, whose generic body is the same shape MINUS the
`sumf -= dmin * sumi` term, is bit-identical, which locates the disagreement at
that one expression and at the `-O3 -ffp-contract=fast` versus
`-O2 -ffp-contract=off` difference between ggml's build and ours.

**So term A is about two hundred times term B.** `2e-3` per bf16 store, applied
twice per layer to the residual plus once per projection over 64 layers and
recurrently to the GDN state, against `1e-5` relative on a row dot. Term A is
the hypothesis this row tests first.

## The oracle disagrees with ITSELF on 1 of 6 prompts

Measured 2026-09-02, `thor:gpu0`, rc job `deb6322d-bd06-4dd1-a5ac-2dec9987fbe1`,
worker `rc-worker-n8smh`, log
`/mnt/nas_share/rc/q4ktok-thor/out/rc-worker-n8smh-20260902T000824Z/`.

Stock `llama-completion` at `b10451`, the recorded gate recipe, run twice over
the identical artifact and prompts. The only difference is `use_extra_bufts`,
which the stock `-nr/--no-repack` flag sets (`common/arg.cpp:2413-2416`): with it
on, Q4_K weights are repacked to `block_q4_Kx8` and the gemv kernel runs; with it
off, the plain `ggml_vec_dot_q4_K_q8_K` runs. Nothing else changes, and the
pinned tree's porcelain was asserted empty before and after.

**Controls first.** `R1_IDENTITY=EXACT`: the repack-on run reproduced the
2026-08-23 recorded ids on 6 of 6 prompts, from a fresh build on a DIFFERENT
worker, which re-confirms the oracle's determinism a third time and validates
this harness against the gate it is being compared to. Both configurations
re-emitted exactly the 15 `unused tensor blk.64.*` warnings. The lever is
observed rather than assumed: the harness prints `USE_EXTRA_BUFTS 1` and `0`
respectively, and llama.cpp's own `load time` falls from 651.47 ms to 302.30 ms
when the repack work is skipped.

**Result: `ORACLE_SELF_DIVERGENCES=1/6`.**

```text
ORACLE_SELF prompt=1 DIVERGE first_diff_index=34 repack_on=3095 repack_off=198
```

That step is one of the six the gate lost, and the recorded margin line for it is

```text
MARGIN 1 34  198 rank2 19.653543  top1 3095 19.738977  gap 0.085434  our="\n" top1=" When"
```

**Token 198 is what vllm.cpp produced there and was scored wrong for.** Given a
different one of its own Q4_K kernels, the oracle produces our token at the exact
step it convicted us on.

Two conclusions, and they point in opposite directions.

1. **Term B can flip a near-tie in this model.** That is now measured rather than
   argued: a per-row-dot difference of order 1e-05 relative reaches the final
   logits and crosses an 0.085-logit gap after 64 layers. No amplification
   estimate is needed.
2. **Term B does not explain the arm.** It accounts for ONE of the five divergent
   prompts. Prompts 0, 2, 4 and 5 diverge with the oracle in full agreement with
   itself, so four fifths of the failure needs a term substantially larger than
   1e-05, which is what Term A is and what `## Scope` item 3 measures.

**This does not excuse the arm and must not be read as doing so.** Four prompts
remain ours. What it does establish is that the gate as declared is not
satisfiable by any engine, llama.cpp included: `b10451` fails its own 6-of-6
token gate under a stock runtime flag. So `TOKEN_GATE=PASS` requires the gate to
pin the oracle's EXECUTED kernel path -- host architecture, `-mcpu` feature set,
repack state and batch size -- and not only its revision. Adding that pin is a
tightening of the gate's definition, not a widening of its tolerance; the
tolerance stays exact.

## All six contested gaps lie INSIDE the oracle's own noise floor

Same job. `R1` (repack on) and `R3` (repack off) walked the IDENTICAL token
sequence -- `R3` was teacher-forced along `R1`'s own ids -- so the full logit
vectors are comparable at every step rather than describing diverged contexts.
71,516,160 logits compared elementwise by `cmp_logits.c`.

The repack is a byte PERMUTATION of the same quantized values
(`make_block_q4_Kx8`, `repack.cpp:2836-2870`: it copies the eight deltas and
mins and interleaves the quants), so the dequantized weights are identical on
both sides and the ONLY thing that differs is the order and granularity of the
fp32 arithmetic.

```text
GLOBAL max_abs=1.365718e+00 rms=7.886647e-02 n=71516160 argmax_flips=1
```

| per step, over 288 steps | min | median | max |
|---|---:|---:|---:|
| max abs logit delta | **0.2020** | 0.3790 | 1.3657 |
| rms logit delta | 0.0412 | 0.0729 | 0.1856 |

The six gaps the 2026-08-23 gate convicted us on were 0.027185, 0.058050,
0.085434, 0.115482, 0.124247 and 0.178236. **Every one of them is smaller than
the MINIMUM per-step perturbation the oracle inflicts on itself**, and five of
the six are below its MEDIAN rms. There is no outlier artefact: the minimum over
all 288 steps already exceeds the largest contested gap.

### What this settles, and what it does not

**Settled: a decision at these margins is not a property of llama.cpp.** It is a
property of which of ggml's four Q4_K rounding schedules executed. The oracle
cannot resolve a 0.03-to-0.18-logit distinction, because it disagrees with itself
by more. A gate that demands 6-of-6 agreement at that scale is asking for
bit-reproduction of one specific aarch64 repacked gemv, not for arithmetic
quality, and `b10451` scores 5 of 6 against itself when asked.

**`VT_BF16_RESIDUAL` disposition (measured, so nobody re-runs it).** The bf16
default STAYS. It is a memory-traffic choice that mirrors vLLM's bf16 residual,
plus a diagnostic A/B, and it is **not** a correctness lever: the `=0` rollback
was run against this gate and did not help (5 of 6 either way, two prompts
worse). That disposition is recorded at the knob's own definition in
`qwen3_5.cpp` as well as here, because the next reader reaches the code first.

**Not settled, and still ours: the RATE.** A perturbation of this exact size
flips 1 of 6 prompts. Ours flips 5 of 6. If our error were the same size as the
oracle's own, we would flip at about its rate. We do not, so we carry an
ADDITIONAL term that is materially larger, and that term is ours to remove. This
is the quantitative case for Term A rather than an assertion of it: bf16 is
1.95e-3 relative per store against the kernel term's 1e-05, applied roughly eight
times per layer across 64 layers, plus a recurrent GDN state re-rounded every
token and a bf16 KV cache against the oracle's f16.

So the target this row can honestly aim at is **the noise floor, not zero**:
remove Term A and the arm should approach 1 of 6, which is where an engine that
does not bit-reproduce the oracle's kernel schedule sits. Reaching 0 of 6
additionally requires porting `arch/arm/repack.cpp`'s `ggml_gemv_q4_K_8x8_q8_K`,
which is the `## Owed` item and is architecture-specific by construction -- it
would not transfer to the gfx1151 arm the standing goal names.

## The red-before, reproduced on this branch

rc job `68d8899a-307e-46d3-bd2b-c021adc27ead`, `thor:gpu0`, worker
`rc-worker-n8smh`, log `out/ours-rc-worker-n8smh-20260902T005602Z/`. One binary
built CPU-only from this branch, run at its SHIPPED defaults against the recorded
`b10451` ids:

```text
TOKENIZER_DIVERGENCES=0/6
GENERATION_DIVERGENCES=5/6
TOKEN_GATE=FAIL
```

Prompt for prompt, index for index, token for token, this is the 2026-08-23
result: first differing index 7 / 34 / 20 / - / 14 / 32, prompt 3 token-exact
48 of 48, and the same id pairs (9338 vs 9564, 198 vs 3095, 13 vs 539, 4593 vs
22486, 15 vs 16). The base is ten days of `main` newer than `ff8f728071`, so
nothing in between moved this arm. That is the red-before.

## Two defects the second test registration caught, and what they teach

Arm B (`VT_ACT_F32=1`) did NOT run: it aborted at 25.7 s. Both causes were
already visible in the focused unit gate, which the default arm passed 33 of 33
and 780 of 780 through both of them -- which is precisely why the second
registration exists.

1. **The resolver was not total.** It asked
   `GetPlatform(dev_type).is_cpu()`, and the platform registry holds only the
   platforms a binary linked, so asking it about an unregistered type THROWS
   (`no platform registered for device type 1`, `platform.cpp:74`). A dtype
   resolver is asked what a device type WOULD resolve to, including from a
   CPU-only test binary, so it must answer rather than abort. It now compares the
   enum.

2. **Not every `DType::kBF16` literal is the model dtype.** Some are an OP's
   contract. `vt::SigmoidGateBf16` requires a bf16 output (`ops.cpp:5105`), and
   routing the trunk dtype into it aborted 9 cases of the 27B dense paged forward
   with `sigmoid_gate_bf16: out must be bf16`. The mechanical rewrite conflated
   the two categories. An audit of the other 28 sites against their consumers
   found no second instance: `SiluAndMul`, `MoeSiluMul` and `RmsNormGated` all
   take `IsOutFloat`, which admits f32.

Neither was reachable by reading. Both were found by running the lever, and the
default arm stayed green through both -- the exact shape of defect this row
warned about in `## Tests`.

### Why there is no third instance, by construction rather than by inspection

`ActDType` returns a non-BF16 answer for `DeviceType::kCPU` and for nothing else.
So every op whose contract fixes bf16 and which is reachable ONLY on a device
tier can never see an f32 operand from this lever, whatever a call-site scan
says. `ops.cpp` has four such contracts, and three of them --
`moe_grouped_gemm_bf16` (`act must be bf16`),
`moe_grouped_gemm_bf16_gate_up_silu` and `moe_marlin`/`marlin_dense` (`a/c must
be bf16`) -- have no CPU kernel at all and are reached through
`MoeBlockBf16Cuda`/`MoeBlockFusedCuda`, which are CUDA-only by their own
declaration. Their operands (`dact`, `ddown`) were never retyped either.

`vt::SigmoidGateBf16` is the ONLY one of the four that is backend-generic
(`cpu_ops.cpp:3728` supplies its CPU kernel), which is exactly why it is the one
that fired, and it is now the one exception carrying its reason. This argument
does not depend on having read every call site correctly; it depends only on the
resolver's device predicate, which the second test registration pins in both
directions.


## The residual dtype is NOT the dominant term: measured, and it cuts against the hypothesis

rc job `25cbca74-df63-43e4-b7db-1315497c4d41`, `thor:gpu0`, worker
`rc-worker-n8smh`, log `out/lever-rc-worker-n8smh-20260902T035957Z/`. Same binary
as the red-before, same artifact, prompts, token count and sampling; the only
difference is `VT_BF16_RESIDUAL=0`, a lever that PREDATES this row and whose own
comment documents it as the f32 rollback. Its unit arm is 33 of 33 green, and
`vllm-bench` exited 0.

```text
ARM A (shipped default, bf16 residual)  GENERATION_DIVERGENCES=5/6  TOKEN_GATE=FAIL
ARM E (VT_BF16_RESIDUAL=0, f32 residual) GENERATION_DIVERGENCES=5/6  TOKEN_GATE=FAIL
```

| prompt | A first diff | E first diff | |
|---|---:|---:|---|
| 0 | 7 | 7 | same |
| 1 | 34 | **21** | earlier |
| 2 | 20 | **4** | earlier |
| 3 | 48 (exact) | 48 (exact) | same |
| 4 | 14 | 14 | same |
| 5 | 32 | 32 | same |

Agreeing-prefix total 155 tokens against 126. **Making the residual stream f32 did
not reduce the divergence count, and it made two prompts diverge EARLIER.**

**The lever is live, which is the control that makes this readable.** The outputs
changed, so this is not a knob that did nothing; it is a knob that did something
and did not help.

### What this does and does not license

**It refutes the residual store as the dominant term.** The residual was the
single largest suspect -- two roundings per layer on the 64-layer accumulator --
and removing it moved the rate not at all.

**It does NOT refute the bf16 activation hypothesis as a whole, and saying so
would be overclaiming.** Arm E removes ONE of roughly eight bf16 stores per
layer. `hidden`, the six `MatmulBf16D` projection outputs, the KV cache and the
GDN state are all still bf16 in this arm, because the lever that would move them
together is incomplete (`VT_ACT_F32`, above) and the KV lever does not run at all
([#2548](https://github.com/mudler/vllm.cpp/issues/2548)). A partial dose that
does not move a rate is weak evidence about the full dose.

**What it does support is the chaotic reading.** A perturbation large enough to
relocate two first-divergence indices by 13 and 16 tokens left the RATE at 5 of
6. That is the same behaviour the oracle showed against itself: a change of
comparable size reshuffles which near-ties flip without changing how many do. On
that reading the rate is set by the total perturbation budget rather than by any
one term, and no single dtype will move it.

**Consequence for the row.** The bf16 term is no longer the leading named cause;
it is one term among several, and the honest position is that the DOMINANT term
is not yet identified. The instrument that would identify it is the one the
2026-08-23 evidence already said was owed and that this row still has not built:
a logit dump on OUR side, so our per-step logit delta against the oracle can be
compared with the 0.2020-to-1.3657 self-perturbation band measured for the oracle.
Without it, every remaining hypothesis is ranked by per-store magnitude rather
than by measurement.


## Scope

1. Measure whether the oracle's own greedy decode is stable across its own
   kernel schedules, using the stock `-nr/--no-repack` flag (`common/arg.cpp:2413`)
   — no patch to the pinned tree. If the oracle's own tokens move, the target
   "token-exact vs llama.cpp" is only defined once host arch, `-mcpu` feature
   set, repack state and batch size are pinned, and this row says so.
2. Give the CPU `qwen35` path a resolved compute dtype so the arm can run at the
   precision its oracle uses, instead of six hardcoded `kBF16` sites and four
   independent env vars that between them cannot reach f32.
3. Re-run the declared token gate red-before (bf16, must reproduce 5 of 6) and
   green-after (f32).
4. Whatever the outcome, name the cause against specific arithmetic and record
   the residual.

Out of scope: the ROCm hang (#2511), IQ3_S (#2510), any throughput number, and
porting ggml's aarch64 kernels (that is term B and stays owed).

## Design

`hidden`, `res`, and the bf16-out projection helpers on the dense `qwen35` CPU
path resolve one dtype rather than each carrying its own literal. The default
stays `kBF16` on a device tier where it is a measured traffic win; the CPU tier
of this arm resolves `kF32`, because on CPU every consumer of a bf16 store
widens it straight back to f32 (`LoadActF32`,
[cpu_quant_gemm.cpp:41](../../src/vt/cpu/cpu_quant_gemm.cpp#L41);
`WidenRowToF32`, [cpu_ops.cpp:577](../../src/vt/cpu/cpu_ops.cpp#L577)), so the
narrow store buys bandwidth and loses mantissa with no compensating compute win.

The switch is one resolver, reachable from `ModelRegistry::Forward` through the
ordinary `--device cpu` load, so the gate measures a capability and not a class.

## Risks

- Changing the resolved dtype touches the CUDA arm's recorded perf defaults.
  Every device default must stay byte-identical; the mutation that proves it is
  in `## Tests`.
- F32 activations raise CPU resident bytes. The arm has no admissible memory
  number yet, so this is recorded, not gated.
- Term B may still keep the gate red. That is an acceptable, reportable outcome
  and must not be answered by widening the gate.

## Tests

- A red-first unit test that the CPU dense `qwen35` path resolves the compute
  dtype from one place, and that flipping the resolver changes the dtype of the
  hidden buffer, the residual and the bf16-out projections together.
- The device default is unchanged: a mutation that makes the resolver return
  `kF32` unconditionally must red a test that pins the CUDA-tier dtype.
- The ported comparison of `VecDotQ4_KQ8_K` against `ggml_vec_dot_q4_K_q8_K_generic`
  already exists in spirit in `tests/vt/test_ops_quant_dot.cpp`; this row adds
  the measured `b10451` arch-versus-generic spread as a recorded figure, not a
  gate, because the arch tier is not ported.

## Gates

Unchanged from
[`qwen38-27b-quant-arms.md`](qwen38-27b-quant-arms.md) §Gates. llama.cpp
`b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`, this artifact
(`sha256 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`,
17,106,775,008 B), greedy, 6 prompts, 48 tokens, concurrency 1, MTP off.
`TOKEN_GATE=PASS` requires 6 of 6.

**The near-tie band is not available and is not reached for.** It is admitted
only where the oracle's greedy decode is non-deterministic; this oracle
reproduced #857 byte for byte from a different build. The ratified near-tie
DISTRIBUTIONAL gate was scored against this arm anyway, so that the refusal above
rests on a measurement rather than only on a precondition;
[`qwen38-27b-q4km-neartie-band-adjudication.md`](qwen38-27b-q4km-neartie-band-adjudication.md)
carries the rule and the result.

## Evidence required

- The rc job ids, the device, and the raw log paths.
- The kernel-level figures above, with the exact build recipe. Harness:
  `ourdot.cpp`, linking this tree's `cpu_quant_dot.cpp` + `cpu_quant_act.cpp`
  objects against `libggml-cpu.a`/`libggml-base.a`/`libggml.a` from a
  `GGML_NATIVE=ON` `b10451` build. It is deliberately NOT committed: this
  project ships without a ggml dependency, and a test that needs one is a
  dependency by another name.
- The oracle-against-itself result (repack on versus repack off).
- The final-logit delta distribution between two oracle configurations, measured
  by teacher-forcing one along the other's token sequence.
- Red-before and green-after gate output on the same binary pair.
- Every refuted hypothesis.

## Stop conditions

- Do not widen, weaken or delete the gate.
- Do not fabricate a root cause. A narrowed cause with a quantified residual is
  a valid outcome and gets posted to #2534.
- `NEEDS_DECISION` if making f32 the shipped CPU default for this arm is the
  only way to pass, since that is a product default and not a mirrored one.

## Owed

- Term B: ggml's aarch64 `arch/arm/quants.c` and `arch/arm/repack.cpp` k-quant
  kernels are not ported, so our CPU decode cannot be bit-identical to an
  aarch64 oracle build even at equal precision. Tracked by #2534 until this row
  closes and then re-filed if it survives.
- The KV cache cannot be made to MATCH the oracle, only to out-precision it.
  `ResolveKvCacheDType` ([kv_cache_dtype.h:28](../../include/vllm/v1/kv_cache_dtype.h#L28))
  offers bf16 or f32 and no f16, while llama.cpp's context defaults to
  `GGML_TYPE_F16` (`src/llama-context.cpp:3538-3539`). The f32 arm is strictly
  more precise than the oracle's store, which removes our error but leaves the
  oracle's own f16 rounding unmatched, so a KV-cache-driven residual cannot be
  closed from our side alone.
- ONE bf16 rounding survives the f32 trunk. `vt::SigmoidGateBf16` fixes its
  output dtype by contract ("out must be bf16", `ops.cpp:5105`), so the
  gated-attention output at
  [qwen3_5.cpp `SigmoidGateOProjD`](../../src/vllm/model_executor/models/qwen3_5.cpp)
  stays bf16 under `VT_ACT_F32=1`. Closing it needs an f32-capable gate kernel,
  which is a new op and not a dtype change. Every other consumer this row
  retyped takes `IsOutFloat` and admits f32; this is the only exception, and it
  was found by running, not by reading.
- **`VT_ACT_F32`'s conversion is INCOMPLETE and must not be measured around.**
  rc job `18fc60f0`, 2026-09-02: the default arm is 33 of 33 and 780 of 780, and
  the f32 arm fails 9 cases with 293 assertions. The failures are cross-PATH
  consistency assertions -- `MaxAbsDiff` between the indexed and fallback GDN
  pools, `memcmp` between the tap and plain routes, `aux_col(k,t,h) == rb[...]`,
  and one `CHECK( 2.69807 < 0.001 )` -- plus a SIGABRT in the NVFP4 lm_head case.
  Those are paired paths required to agree bitwise, so one side is retyped and
  its reference is not. A magnitude of 2.7 is not a rounding tail. Finishing the
  conversion means retyping every paired path together and resolving the NVFP4
  head's own dtype contract, which is more than this row scoped. Until then the
  lever is a staged instrument that does not work, and the arm it was meant to
  measure is measured through the pre-existing levers instead.
- **`VT_KV_CACHE_F32=1` is broken on this model path, and it PREDATES this row.**
  rc job `25cbca74`: the focused unit suite passes under it, and the real engine
  dies in the first forward with `vt: reshape_and_cache: k/v/k_cache/v_cache must
  share one float dtype (auto cache path)` (`ops.cpp:3947`). The cache becomes
  f32 while the attention path still produces bf16 k and v. So the KV term cannot
  be isolated today, and a documented same-binary A/B knob does not run. This is
  not caused by this row's change; it is found by it, and it is filed as
  [#2548](https://github.com/mudler/vllm.cpp/issues/2548). The focused unit suite
  is 33 of 33 green under that lever, so the knob reads as gated and is not.
- `VT_ACT_F32=1` is **REFUSED at the resolver**, not honoured. Honouring it does
  not give an f32 engine, it gives a numerically inconsistent one, and a SIGABRT
  in the ninth test case is exactly the "discovered later" `AGENTS.md` forbids.
  The refusal names the missing work -- retype every PAIRED path together so the
  cross-path equality gates still hold, and resolve the two bf16 dtype contracts
  the trunk feeds (`vt::SigmoidGateBf16`, the NVFP4 lm_head) -- and cites this
  spec and #2534. The test asserts the refusal AND its message on every device
  type, so a bare throw that said nothing would not pass.
- `VT_ACT_F32` is an instrument, not a default. Whether the CPU tier SHIPS f32 is
  a product default and is decided by the A/B this row runs, not by the commit
  that added the resolver.

## Now

`ACTIVE`.
