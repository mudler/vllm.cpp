# The logit dump: measure our per-step delta against the oracle instead of ranking terms by argument

Row: `QUANT-QWEN38-27B-GGUF-ARM`.
Issue: [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
Predecessor: [`qwen38-27b-q4km-token-exactness.md`](qwen38-27b-q4km-token-exactness.md),
landed as `15f750fb9`.

## Why this exists

The predecessor row established what the Q4_K_M token gate can and cannot mean,
and it killed four candidate causes. Every one of them was the best-motivated
candidate on its list at the time:

| candidate | how it died |
|---|---|
| `QuantizeQ8KK` / the Q8_K activation quantizer | byte-identical to `quantize_row_q8_K_ref`, 0 of 4.6 M quants |
| a Q6_K port defect | bit-identical to the generic body |
| a Q4_K/Q5_K port defect | differs by the same magnitude ggml differs from itself |
| the bf16 RESIDUAL, two roundings per layer on the 64-layer accumulator | `VT_BF16_RESIDUAL=0` measured: 5 of 6 either way, two prompts WORSE |

The last one was the leading hypothesis of both the implementer and the
operator. **Ranking hypotheses by per-store magnitude is not working**, and it
has now cost four dispatches. Every remaining hypothesis is ranked by argument
because our tree exposes no logit vector on any production path. The 2026-08-23
evidence said that instrument was owed; it is still owed; this row builds it.

## What exists already, and what is missing

The ORACLE half is committed and was exercised on `thor:gpu0`:
`/mnt/nas_share/rc/q4ktok-thor/job/oracle_logits.cpp` teacher-forces stock
`llama.cpp` `b10451` along a supplied id sequence and dumps
`n_predict x n_vocab` f32; `cmp_logits.c` diffs two dumps elementwise. Together
they produced the band this row measures against.

Missing is OUR half: one env-gated dump of the final logit vector per decode
step, reachable from `vllm-bench` through the same seam `--output-token-ids`
already uses.

## The denominator this measures against

Measured 2026-09-02, rc job `deb6322d`, `thor:gpu0`: the oracle against ITSELF,
one artifact and one recipe, differing only in which of ggml's own Q4_K kernels
ran, teacher-forced along one sequence so the vectors are comparable.

```text
GLOBAL max_abs=1.365718e+00 rms=7.886647e-02 n=71516160 argmax_flips=1
```

| per step, 288 steps | min | median | max |
|---|---:|---:|---:|
| max abs logit delta | 0.2020 | 0.3790 | 1.3657 |
| rms logit delta | 0.0412 | 0.0729 | 0.1856 |

That is the noise floor of "which kernel ran" on this model. Our delta gets
placed against it.

## PRE-REGISTERED OUTCOMES

**Written before the instrument exists and before any of our logits have been
seen.** Recorded in advance because this row's whole purpose is to stop
conclusions being assembled after the fact, and because four hypotheses have
already died on this row after sounding decisive beforehand. Whichever branch the
data selects, it selects it here and not in a later reading.

The statistic is the per-step delta between our final logit vector and the
oracle's, teacher-forced along the SAME token sequence so the vectors describe
the same context, summarised the same way `cmp_logits.c` already summarises the
oracle against itself: per-step `max_abs` and `rms`, plus the global figures.

1. **INSIDE the band** -- our per-step `max_abs` distribution overlaps
   0.2020 to 1.3657 and our median `rms` is at or below ~0.0729.
   Then we are AT the noise floor: the 5-of-6 rate is near-tie luck rather than
   an error of ours, no further precision term is worth chasing, and the gate
   needs the kernel-path pin the oracle record now carries and little else. The
   row would then close by tightening the gate's definition, not by changing
   arithmetic.
2. **A FEW TIMES the band** -- our per-step `max_abs` runs roughly 2x to 20x the
   oracle's, i.e. medians in the ~0.8 to ~8 range.
   Then ONE localisable term remains, and the next dispatch is the per-layer
   hidden-state bisect the 2026-08-23 evidence listed third and nobody has run:
   one prompt, one position, our hidden state against llama.cpp's at each of the
   64 layers, with the GDN (`ssm_*`) layers and the 16 full-attention layers
   separated because they are different code.
3. **ORDERS above the band** -- our per-step `max_abs` is 100x or more, i.e.
   medians above ~40 against absolute logits of 15.9 to 22.6.
   Then the precision framing is WRONG and the 282-of-288 rank-1 evidence needs
   re-reading: a difference that large cannot leave the vocabulary ordering
   almost everywhere intact, so either the comparison is mis-aligned (a harness
   defect, and the first suspect is teacher-forcing alignment) or something
   structural survives that the rank statistic hid.

**A fourth result is admissible and must be reported if it occurs:** the deltas
are not stationary across the 48 steps -- for instance small early and growing,
or spiking only at the six contested steps. That shape would say the error
ACCUMULATES rather than being a fixed per-step offset, which none of the three
branches above assumes, and it would redirect the bisect from "which layer" to
"which step".

## The first build of the instrument was in the wrong function

rc jobs `c636b3c0` (the measurement, aborted) and `e1875966` (the diagnostic),
`thor:gpu0`, worker `rc-worker-n8smh`, 2026-09-02.

The measurement run completed, both engines exited 0, and the controls did
exactly what they were written for:

```text
DUMP_PERTURBS=NO (ids byte-identical)     <- passed
ALIGNMENT=BROKEN   checked=0  bad=6       <- 6 of 6 MISSING sidecar
FATAL: alignment control failed
```

Those two results together are the signature of an instrument that did nothing:
a dump that writes nothing perturbs nothing. **No delta was computed, and no
pre-registered branch was selected.**

The diagnostic settled which of the remaining causes it was, in one 45-second
run on the warm tree:

| probe | result |
|---|---|
| the dump's own breadcrumb | **no line at all** |
| `VT_DEBUG_SAMPLED=1`, which predates this row and sits a few lines BELOW the dump in the SAME function | **0 hits** |
| tokens produced | `[[11751,13,198,760]]`, correct |

Two independent probes silent in `GPUModelRunner::sample_tokens` while sampling
demonstrably happens is not a bounds bug. It is the wrong function, and the
verdict does not rest on this row's own code.

**Why it was the wrong function, and the reading error that hid it.**
`async_input_combine_` is assigned DIRECTLY in the runner constructor, at both
`async_input_combine_ = AsyncRunnerEnvDefault()` assignments in `runner.cpp`,
and `AsyncRunnerFlagIsOn` answers TRUE when `VT_ASYNC_RUNNER` is unset
(`async_runner_flag.h`). So the production default samples in the
DEVICE-RESIDENT branch of `runner.cpp::sample_tokens_async`, which calls
`runner.cpp::assemble_sample_logits` itself instead of reaching the call in
`runner.cpp::sample_tokens`. `sample_tokens` is the fallback, not the default.

I had ruled this branch out and was wrong. I grepped the SETTER
`set_async_input_combine`, found no caller, and concluded the flag stayed false.
The field is not written through the setter. **A predicate is not ruled out by
finding no caller of one way of setting it.**

**The hermetic gate did not catch this, and that is its own finding.** The
original case called `sample_tokens` DIRECTLY and passed 13 of 13 while the
instrument wrote nothing in the real engine -- a gate measuring a class and not
the capability, which is the exact shape `AGENTS.md` "Nothing lands dead" names.
The dump is now ONE helper called from both sampling paths, and a second case
drives the async branch explicitly with `set_async_input_combine(true)`, so
deleting either call site reds exactly one case.

## The cross-tier discriminator is gone

The gfx1151 ROCm token gate returned `TOKEN_GATE=NOT_MEASURABLE`: 12 of 12 legs
faulted on both arms, zero harness errors, zero clean legs,
`reference_tier_hits=0`, native ROCm path engaged throughout. So the check this
row offered -- if a shared term drives both tiers, ROCm should diverge at the
same indices and ids as the CPU arm -- cannot be run until
[#2511](https://github.com/mudler/vllm.cpp/issues/2511) is fixed.

That removes the only fallback comparison and leaves this measurement as the one
live path to the dominant term. It does not widen this row's scope, and the
pre-registered branches are unchanged.

## Scope

1. One env-gated final-logit dump, written where the logits actually live,
   reachable from `vllm-bench` on its ordinary configuration.
2. Teacher forcing on our side is NOT in scope: the comparison forces the ORACLE
   along OUR ids, which is what the existing harness already does and what the
   recorded margins used.
3. One `rc` job: our run dumping logits, the oracle run teacher-forced along our
   ids dumping logits, then `cmp_logits.c`.
4. Report against the pre-registered branches above. No new hypothesis is ranked
   by argument in this row.

Out of scope: fixing whatever the measurement finds, the incomplete `VT_ACT_F32`
conversion, [#2548](https://github.com/mudler/vllm.cpp/issues/2548), and any
throughput number.

## Risks

- **The dump must not change what it measures.** It is a read of a buffer that
  already exists, env-gated off by default, and the gate is that the token ids
  produced with the dump ON are byte-identical to the same run with it OFF.
  Without that control the instrument could be reporting its own perturbation.
- **Alignment is the first suspect on any surprising result.** Step `i` of our
  dump must be the same context as step `i` of the oracle's. The control is that
  our argmax at each dumped step equals the id already recorded in
  `--output-token-ids` for that step.
- Size: 248320 x 4 bytes x 48 steps x 6 prompts is ~286 MB per side. Written to
  worker-local `/tmp`, never to CIFS mid-run.

## Gates

This row produces a MEASUREMENT and no gate verdict. `TOKEN_GATE` stays as the
predecessor left it: `FAIL`, 5 of 6, and no speed or memory axis is admissible.

## Evidence required

- The rc job ids, the device, the raw log paths.
- The identity control: ids with the dump on == ids with the dump off.
- The alignment control: our dumped argmax == our recorded `--output-token-ids`.
- The per-step delta table in the same shape as the oracle's, and the branch it
  selects, named against the pre-registration above.

## Stop conditions

- Do not weaken the token gate; this row does not touch it.
- If the deltas select branch 3, treat harness alignment as the first suspect and
  prove alignment before concluding anything about the engine.
- Report an unstationary delta shape if it appears, even though no pre-registered
  branch predicts it.

## Outcome

**Branch 1 selected on the pre-registered statistic, and the pre-registration's
conclusion for that branch is WRONG.** Measured `d72baf2c`, `thor:gpu0`: our
per-step delta against the oracle is median `max_abs` 0.4165 against the band's
0.3790 (**1.10x**) and median `rms` 0.07903 against 0.0729, so we are at the
oracle's own kernel-schedule noise floor. Stationary at 1.07x growth, so the
admitted fourth shape did not occur.

Branch 1 said that means "no further precision term is worth chasing". That is
false here, because the registered statistic measures the MAGNITUDE of our error
while the gate is decided by the RESOLUTION of our logits. **288 of 288 of our
top-1 logits lie exactly on the bf16 grid**, our smallest representable non-zero
gap is 0.0625 rising to 0.125 at magnitude 16-32, and five of the six contested
gaps (0.027 to 0.178) are at or below it. Six steps in 288 are EXACT ties in our
arithmetic; the six argmax flips are exactly the six recorded rank-2 steps.

The site is `qwen3_5.cpp::DenseLogitsF32D` routing to
`qwen3_5.cpp::MatmulBf16LogitsF32D`, which computes the `[M, vocab]` product in
bf16 and widens with `vt::CastF32`. Widening cannot recover a discarded mantissa.

Evidence:
[`qwen38-27b-q4km-logit-delta-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-logit-delta-20260902.md).

**Lesson for the pre-registration method, kept because it cost four dispatches.**
Pre-registering the branches was right and stopped the conclusion being assembled
after the fact. But every branch was framed on ONE statistic, and the cause lived
in a property that statistic could not express. A pre-registration should name
the statistic AND admit that a finding outside it may decide the row -- which is
what the fourth shape did for time, and what nothing did for resolution.

## Owed

- Confirm which arm of `qwen3_5.cpp::DenseLogitsF32D` this checkpoint takes (is
  its GGUF `lm_head` loaded `nk`). The bf16 grid is certain at 288 of 288; the
  routing predicate is inferred from being the only matching site.
- Whether an f32 logits head makes the arm PASS is not established. Two contested
  gaps sit below the f32-vs-oracle agreement measured at those steps, so widening
  is necessary and not obviously sufficient.
- The heavy tail: our `max_abs` reaches 17.16 against the oracle's 1.3657, at
  `p5/2`, `p5/11` and `p0/45`. Not temporal, flipped no argmax, unexplained.

## Now

`DONE` -- the instrument exists, is gated on both sampling paths, and produced
the measurement this row was opened for.
