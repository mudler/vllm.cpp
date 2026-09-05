# Our logits are on a BF16 grid, and that is why the Q4_K_M gate loses six near-ties

Row `QUANT-QWEN38-27B-GGUF-ARM`, for
[#2534](https://github.com/mudler/vllm.cpp/issues/2534). Spec:
[`qwen38-27b-q4km-logit-dump.md`](../../.agents/specs/qwen38-27b-q4km-logit-dump.md),
whose three outcomes were committed at `8a63518e1` BEFORE this instrument
existed and before any of our logits had been seen.

This is a MEASUREMENT and produces no gate verdict. `TOKEN_GATE` stays `FAIL`,
5 of 6, and no speed or memory axis becomes admissible.

## What ran

`rc` job `d72baf2c-4993-4eb7-ba93-078a48c18198`, `thor:gpu0`, worker
`rc-worker-n8smh`, 2026-09-02, log
`/mnt/nas_share/rc/q4klogit-thor/out/logit-rc-worker-n8smh-20260902T090805Z/`.
`oracle_rc=0`, `cmp_rc=0`, `JOB_LOGIT_DONE`.

thor deliberately: the denominator below was measured there on aarch64 with
`REPACK=1`, and this row's predecessor established that the oracle's kernel path
is host-dependent, so the comparison is only valid on the same host.

## Both controls passed, and they had already failed once

```text
bench_off_rc=0   bench_on_rc=0
DUMP_PERTURBS=NO (ids byte-identical)
ALIGNMENT checked=288 bad=0
ALIGNMENT=EXACT
vt-dump-logits: dir=... rows=1 vocab=248320 req_ids=1 discard=1   (printed once)
```

`vocab=248320` is the oracle's own `ORACLE_N_VOCAB`. All 288 steps (6 prompts x
48) have our dumped argmax equal to the id `--output-token-ids` recorded, so both
sides provably describe the same context at every step.

An earlier run of the same job produced `DUMP_PERTURBS=NO` beside
`ALIGNMENT=BROKEN checked=0 bad=6` and was ABORTED without computing a delta.
That pair is the signature of an instrument that did nothing: a dump that writes
nothing perturbs nothing. See the spec for the cause.

## The measurement

Our final logit vector against the oracle's, teacher-forced along OUR ids so the
vectors describe the same context. 71,516,160 logits compared elementwise.

```text
GLOBAL max_abs=1.716056e+01 rms=1.578703e-01 n=71516160 argmax_flips=6
```

| per step, 288 steps | min | median | max |
|---|---:|---:|---:|
| **ours vs oracle**, max abs | 0.2492 | **0.4165** | 17.1606 |
| **ours vs oracle**, rms | 0.04760 | **0.07903** | 1.21881 |
| oracle vs ITSELF (the pre-registered band) max abs | 0.2020 | **0.3790** | 1.3657 |
| oracle vs ITSELF, rms | 0.0412 | **0.0729** | 0.1856 |

Median ratio **1.10x**. Stationary: `steps<12` median 0.3951 against `steps>=36`
median 0.4244, growth **1.07x**, so the fourth shape the spec admitted in advance
-- an error that accumulates across the 48 steps -- **did not occur**.

**This selects pre-registered branch 1: INSIDE the band.** On the statistic the
spec named, our per-step delta is indistinguishable from the oracle's own
kernel-schedule self-perturbation.

## The finding the pre-registered statistic could not see

Branch 1 was written to conclude "no further precision term is worth chasing".
That conclusion is WRONG here, and the reason is that the registered statistic
measures the MAGNITUDE of our error and the gate is decided by the RESOLUTION of
our logits.

The six argmax flips are **exactly** the six rank-2 steps the 2026-08-23 margin
dump recorded -- `p0/7, p1/34, p1/35, p2/20, p4/14, p5/32` -- and at every one of
them our `max_abs` (0.368 to 0.563) is inside the oracle's band. Our top-1 logits
there are `15.875, 19.750, 19.125, 22.500, 16.000, 21.000` and our top1-top2 gaps
are `0.000, 0.125, 0.000, 0.125, 0.000, 0.000`.

Every one of those is a multiple of 1/8, and 0.125 is exactly the **bf16 ULP** at
magnitude 16 to 32. Tested over the whole run:

```text
our top-1 logits exactly on the bf16 grid: 288 / 288  (100.0%)
steps where our top1 and top2 are EXACTLY TIED (gap12 == 0): 6 / 288
smallest non-zero gaps we can represent: 0.0625 0.1250 0.1875 0.2500 ...
```

**Our final logits carry only bf16 precision, although they are stored as f32.**
The gaps the gate contests are 0.027185, 0.058050, 0.085434, 0.115482, 0.124247
and 0.178236. Five of the six are at or below our smallest representable non-zero
gap. Six steps in 288 are EXACT TIES in our arithmetic: we lose four of them to
the index tie-break and win two by the same accident, and the remaining two flips
sit at exactly one ULP.

So the engine is not making a worse decision than the oracle at these steps. **It
cannot represent the distinction the gate is asking about.**

## The site, named

[`qwen3_5.cpp:3205`](../../src/vllm/model_executor/models/qwen3_5.cpp) routes the
dense logits GEMM:

```cpp
return lm_head.nk ? MatmulBf16LogitsF32D(d, x, lm_head)
                  : MatmulF32D(d, x, lm_head);
```

and `MatmulBf16LogitsF32D` (`:1721`) computes the `[M, vocab]` product into a
**bf16** buffer and then widens it:

```cpp
DBuf bf16 = MatmulBf16D(d, x, w);
DBuf f32(d, DType::kF32, {...});
vt::CastF32(d.q, f32.t(), bf16.t());
```

`CastF32` cannot recover a discarded mantissa. This is the only site in the dense
path that produces exactly the grid measured above.

**One check is owed before this is called confirmed rather than identified:**
which arm of `:3205` this checkpoint takes, i.e. whether its GGUF `lm_head` is
loaded `nk`. The empirical grid is certain at 288 of 288; the routing predicate
is inferred from being the only matching site and has not been observed directly.

## Why every earlier negative result now makes sense

- **`VT_BF16_RESIDUAL=0` did not help** (5 of 6 either way, two prompts worse).
  It widens the residual stream, not the lm_head output. It was the wrong bf16.
- **282 of 288 steps are rank-1.** Away from ties, a 0.125 resolution is ample.
- **Our delta sits inside the oracle's noise band.** The defect is resolution at
  ties, not error magnitude -- which is precisely why four dispatches of
  magnitude-ranked hypotheses failed to find it.

## What this does NOT establish

- It does not show the arm would pass at f32 logits. Two of the six contested
  gaps (0.027185 and 0.058050) are below the f32-vs-oracle agreement this run
  measures at those steps, so widening the head is necessary, not obviously
  sufficient. The next run answers that, and it is one env-gated dtype away.
- It does not revisit `TOKEN_GATE`, which stays `FAIL`.
- The heavy tail (our max 17.16 against the oracle's 1.3657, at `p5/2`, `p5/11`,
  `p0/45`) is real, is NOT temporal, and did not flip any argmax. It is
  unexplained and is recorded as owed rather than folded into this conclusion.
