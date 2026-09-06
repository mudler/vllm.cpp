# What each `qwen4_exp` arm emits now that BOTH run vLLM's chunked GDN prefill, 4 September 2026

Wave ARMTOKENS of [`KERNEL-GDN-CHUNKED-MIRROR`](../../.agents/specs/gdn-chunked-mirror.md),
[#2858](https://github.com/mudler/vllm.cpp/issues/2858), under
[#2612](https://github.com/mudler/vllm.cpp/issues/2612).

**The one-line result. Neither sequence moved, and they still agree on five of
eight. No `VT_Q4EXP_LAYER_FP` fingerprint has yet observed a step where they
disagree.**

**CORRECTION 2026-09-05
([#2969](https://github.com/mudler/vllm.cpp/issues/2969)).** This line said "no
instrument", and that is wrong for the other tap. `VT_MOE_SEL_FP` counts MoE
block invocations and not model forwards, so its 384-call window reaches
forwards 4, 6 and 7. **That does not supply a reading.** The comparison taken in
that window is VOID, because its CUDA arm answered the degenerate pre-#2550
sequence `11751 271 271 271 271 271 0 0`. What is missing is a non-degenerate
arm, not a wider window.

| arm | token ids | agrees with CPU |
|---|---|---|
| `--device cpu`, production default | `11751 13 15767 411 2029 11 1092 369` | — |
| `--device cuda`, production default | `11751 13 15767 411 1928 11 628 567` | **5 of 8** (indices 0,1,2,3,5) |

Both are **byte-identical to the sequences measured before the default moved**
([PREFILLDIV](qwen4exp-cuda-prefill-divergence-20260902.md) §2,
[#2547](https://github.com/mudler/vllm.cpp/issues/2547)). The CPU arm did **not**
change. The registry comment predicting the ids "are EXPECTED TO DIFFER" is
falsified, and this file is what corrects it.

**That is not because the new arm is inert.** The chunked CPU arm runs, and §5
measures it: `VT_GDN_CHUNKED` moves decoder layer 0's Gated DeltaNet block output
by `3.702e-04` on the same binary. The ids did not follow it, in either
direction. This is exactly what the spec's `## Scope` says to expect: agreement
between two of our arms is an argmax over near-ties and is not a monotone
function of the distance between them.

**EVERY RATIO IN THIS FILE IS WITHDRAWN — READ §5's ANNOTATIONS BEFORE QUOTING
ONE** ([#2877](https://github.com/mudler/vllm.cpp/issues/2877)). This paragraph
first went on to say that the port "moves the CPU arm to within `1.772e-05` of
CUDA where the sequential arm sat `3.525e-04` away — a **19.9x** reduction in the
divergence #2547 opened". The sentence is quoted rather than deleted so a later
reader sees the shape of the error. Two reasons, in the order that matters.

**First, and it survives everything below.** `LayerFp` returns early on
`s.step >= s.budget`
(`src/vllm/model_executor/models/qwen4_exp_forward.cpp:118`), so
`VT_Q4EXP_LAYER_FP=3` fingerprints model forwards **0, 1 and 2** — tokens
`11751 13 15767`, **which the two arms AGREE on**. The three ids that disagree
are emitted at forwards **4, 6 and 7**, outside the window. **No
`VT_Q4EXP_LAYER_FP` fingerprint has yet observed a single disagreeing step.**
Every tap in §5 measures three forwards on which both arms produce the same
token. The scope of that sentence is this tap alone. See the correction at the
top of this file for `VT_MOE_SEL_FP`, whose window is counted in MoE block
invocations and does reach forwards 4, 6 and 7
([#2969](https://github.com/mudler/vllm.cpp/issues/2969)).

**Second, that ratio changes algorithm on one side, and the metric behind it
cannot rank even the pairs it did observe.** `rel(sumabs)` is a difference of
NORMS, not a norm of DIFFERENCES. Read as algorithm-**matched** CPU-vs-CUDA
pairs, the same three measurements say `L00 blk` moved **16.7x FURTHER**; over
400 seeds of a committed control those two ratios sit at **6%** and **7%** of
what no change at all produces. "The residue grew" and "the residue did not grow"
are equally unsupported, and so is a 19.9x or a 16.7x at the block. §5's two
ANNOTATION blocks carry both framings in full.

**Nothing here is a token gate, and nothing here is a speed.** No oracle decoded
this prompt. Each arm is n=1.

## 1. Why this measurement is possible now, and was not before

[#2849](https://github.com/mudler/vllm.cpp/pull/2849) landed a chunked CPU GDN
prefill mirroring vLLM's bf16 intermediate placement and **made it the default
for bf16**. Before it the two arms ran two different *algorithms* — the CPU arm
the exact sequential recurrence, the CUDA arm the chunked WY decomposition. The
port wave deferred re-deriving the sequences because it needed the 67.564 GiB
artifact and a lease; the spec's `## Now` names it. Nobody had measured either
arm under the new default.

## 2. What was run, and on what

**Tree.** `b767ebda4e55122b1a5473b9aa4027da67f77b75` — `origin/main` `d2b1bda2b`
plus **one deletion**, §3. No instrument was patched in; the fingerprint used in
§5 is committed on `main`. Source tarball sha256
`9a4a2934652d4cc9cfe63511eb8fd043f945f4f977bd1ca74412277f569a45d1`; the job
refuses at `exit 11`/`exit 12` if the tarball digest or `HEAD_SHA` differs from
the value written into the script before the lease was taken.

**Host.** `thor:gpu0` under `rc`, pod `rc-worker-n8smh`, aarch64, NVIDIA Thor,
compute capability 11.0, driver 595.78, nvcc 13.0.88, built `sm_110`, 14 cpus,
122 GiB. Jobs `77476a46-465b-429f-bfd4-0a0ed81ec011` (the build that failed, §3),
`c31b56ca-2394-4846-8f3c-6da7cdbed2b5` (run 1) and
`417fb134-5e55-4e48-83a6-2e1c43f08a9b` (run 2). `thor` is where the 5-of-8 was
measured, which is why it is where this was.

`i8mm` is **PRESENT**, so `vt::cpu::QuantRepackActive()` is true by default and
the repack chain is LIVE. That is asserted on the box rather than assumed,
because it is the precondition the repack check in §4 needs to be a
discriminator rather than a no-op.

**Request.** `examples/vllm-server`, greedy, `--temperature 0`, `max_tokens=8`,
one sequence at a time, prompt `The capital of France is`,
`--block-size 16 --num-blocks 128 --max-model-len 256 --verbose`, **no
`CUDA_LAUNCH_BLOCKING`**. `logprobs:1` is set because that is how the prior run
*read* the ids — the `"token_id:N"` fallback of `BuildCompletionLogProbs` — and
keeping it keeps the conditions equal. One binary,
sha256 `1d129fa0ab96663bea8f50f715117596241a7f2f8ae77e877ba5853bb198792f`, for
every arm in both runs; run 2 asserts that digest rather than rebuilding.

**Artifact.** The released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, staged
worker-local, identity asserted **inside the lease on the bytes the server
opened**, before any arm ran and again in run 2:

```text
RESULT ARTIFACT VERIFIED: shard1 sha256=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd staged_bytes=72546461650 shards=3
```

That is the digest [`docs/USAGE.md`](../USAGE.md)'s registry records for shard 1
of `unsloth/Qwen3.8-Flash-Next-GGUF` @ `8bdc666649440e9bdc97e16f3f75782c98478ff5`,
path `UD-IQ1_S`, and the digest PREFILLDIV asserted. The job exits 42 on a
mismatch rather than measuring a different file. Shards 2 and 3 were not
re-hashed; their digests are the registry's.

## 3. The first run produced no arm at all, because `main` did not compile

`origin/main` `d2b1bda2b` **cannot be built with `-DVLLM_CPP_CUDA=ON`**
([#2861](https://github.com/mudler/vllm.cpp/issues/2861)). `f2bda11e3` (#2849)
lifted the `VT_GDN_CHUNKED` read into `vt::GdnChunkedPrefillEnabled` and left the
CUDA-local wrapper behind with internal linkage and no caller; our CUDA flags
carry `-Werror=all-warnings`:

```text
cuda_gdn.cu(3265): error #177-D: function
  "vt::cuda::<unnamed>::ChunkedPrefillEnabled" was declared but never referenced
RESULT BUILD rc=1 wall=920s objects=549
```

The spec's `## Now` already said why it landed: *"No CUDA gate was run. D0
changed CUDA's f32 default and nothing on a GPU has executed since."* Nothing had
compiled that arm. **The measured tree is `main` plus the deletion of those nine
lines and nothing else**, and the job asserts the dead function is absent
(`dead_cuda_wrapper=0`) as a fourth source precondition.

## 4. Run 1 — the five arms, and the two controls

One build, one staged copy, five loads. Every arm returned `http=200` with
**8 ids**, none degenerate, `steps completed: 11`, and an empty first-error line.

| arm | env | ids | load | request |
|---|---|---|---|---|
| **CPU-DEFAULT** | *(production, none)* | `11751 13 15767 411 2029 11 1092 369` | 45 s | 8 s |
| **CUDA-DEFAULT** | *(production, none)* | `11751 13 15767 411 1928 11 628 567` | 30 s | 338 s |
| CPU-GDNSEQ | `VT_GDN_CHUNKED=0 VT_CPU_QUANT_REPACK=0` | `11751 13 15767 411 2029 11 1092 369` | 30 s | 7 s |
| CPU-REPACK0 | `VT_CPU_QUANT_REPACK=0` | `11751 13 15767 411 2029 11 1092 369` | 15 s | 6 s |
| CUDA-REPACK0 | `VT_CPU_QUANT_REPACK=0` | `11751 13 15767 411 1928 11 628 567` | 15 s | 275 s |

Decoded: CPU `" Paris. Given this fact, what is"`, CUDA
`" Paris. Given this information, can we"`.

**The positive control passes.** `VT_GDN_CHUNKED=0` — the exact configuration and
algorithm that produced the published sequence — reproduces it. Had it not, this
harness would not be measuring what the prior run measured and no arm here could
be quoted.

**The repack check passes.** `VT_CPU_QUANT_REPACK=0` is identical to the default
on a host where the chain is live, so the aarch64 repack that once put a NaN in
layer 0 is not perturbing this answer. `VT_CPU_QUANT_REPACK=0` is likewise
identical on CUDA, so the prior run's use of it is not what makes the two runs
comparable — the ids are the same with and without.

**And run 1 could not tell one thing from another.** All three CPU arms agreeing
*including* `VT_GDN_CHUNKED=0` has two explanations that are not the same
finding: the chunked arm runs and flips no argmax, or the chunked arm is **not
reached** and both settings run the sequential recurrence. The spec's T4 says a
flag whose two arms coincide reads as a pass either way, and no log line names
the arm — `--verbose` prints request stages, and neither branch of
`cpu_ops.cpp:2136` logs. Run 1 settles nothing here. Run 2 does.

## 5. Run 2 — the chunked CPU arm IS reached, by a committed instrument

`VT_Q4EXP_LAYER_FP=3` (`qwen4_exp_forward.cpp:95`, 15 taps) is committed on
`main`; it reads bytes a kernel already wrote and writes nothing back. Three arms
on **the same binary by digest**, with `VT_CPU_QUANT_REPACK=0` held on all three
so `VT_GDN_CHUNKED` is the only difference between the first two. Each printed
**1314** fingerprint lines and closed its three steps at `taps=437`, `taps=874`
and `taps=1311` — the same counted property PREFILLDIV read, asserted so that an
arm printing nothing and an arm whose taps agree cannot look alike.

**That counter is CUMULATIVE, and this line first read it as a per-step count**
([#2877](https://github.com/mudler/vllm.cpp/issues/2877)). `LayerFp` does
`++s.taps` on one running total that `LayerFpEndStep` never resets
(`qwen4_exp_forward.cpp:153`), so a 437-tap forward run for three steps prints
`437`, `874`, `1311` — which is what the committed `run2-results.txt` reads on
all three arms, not "`taps=437` per step on all three steps" as this file
originally said. The same misreading made `scripts/q4exp-layerfp-diff.py` refuse
every genuine multi-step fingerprint; `CumulativeTapCounter` in
`tests/scripts/test_q4exp_layerfp_diff.py` now pins the cumulative form.

Step 0, `rel = |a-b| / max(|a|,|b|)` on `sum|x|`:

| tap | CPU-CHUNKED | CPU-SEQ | rel | CUDA | rel |
|---|---|---|---|---|---|
| `emb` | 37.242316 | 37.242316 | 0.000e+00 | 37.242316 | 0.000e+00 |
| `wide` | 148.969264 | 148.969264 | 0.000e+00 | 148.969264 | 0.000e+00 |
| `L00 in` | 148.969264 | 148.969264 | 0.000e+00 | 148.969264 | 0.000e+00 |
| `L00 ahc.mix` | 6517.5925 | 6517.5925 | 0.000e+00 | 6517.5925 | 0.000e+00 |
| `L00 ahc.inj` | 0.952026367 | 0.952026367 | 0.000e+00 | 0.952026367 | 0.000e+00 |
| **`L00 blk`** | **1074.05248** | **1073.65489** | **3.702e-04** | **1074.03345** | **1.772e-05** |
| `L00 s.attn` | 232.272312 | 232.225395 | 2.020e-04 | 232.270443 | 8.047e-06 |
| `L00 mhc.mix` | 3615.47142 | 3613.82031 | 4.567e-04 | 3615.62777 | 4.324e-05 |
| `L00 mhc.inj` | 2.92224121 | 2.92212772 | 3.884e-05 | 2.92224121 | 0.000e+00 |
| `L00 moe` | 389.936505 | 389.976283 | 1.020e-04 | 390.025768 | 2.289e-04 |
| `L00 s.mlp` | 392.790737 | 392.881629 | 2.313e-04 | 392.853498 | 1.598e-04 |
| `ple` | 232.508111 | 232.752729 | 1.051e-03 | 232.818457 | 1.333e-03 |
| `s.ple` | 568.716336 | 569.04265 | 5.734e-04 | 569.080593 | 6.401e-04 |
| `out` | 27981.5263 | 27964.6752 | 6.022e-04 | 28054.1436 | 2.588e-03 |

**The flag routes, and layer 0 is where.** The first tap that differs is
`L00 blk` — the Gated DeltaNet block output — in **both** comparisons, from an
input (`L00 in`, `ahc.mix`, `ahc.inj`) that is bit-identical on all three arms.
26 of 42 taps differ between the two CPU arms. The chunked CPU arm is therefore
reached and computing, and explanation (b) of §4 is dead.

> **ANNOTATION 2026-09-04 — `42` IS LAYER-0 COVERAGE, NOT MODEL COVERAGE**
> ([#2877](https://github.com/mudler/vllm.cpp/issues/2877)). The instrument printed
> **437 taps per step, 1311 over the three steps**; the differ compared **42**.
> `run2-job.sh`'s `load()` splits each token on `'='`, but the tap prints the layer
> as `L%+03lld` — `L+00`, with no `=` — so `f.get('L')` is `None` on **every** row,
> the key collapses to `(step, None, tag)`, and `if key in rows: continue` keeps
> only the **first** occurrence. 437 taps/step therefore collapse to 14, the
> distinct tag count, and 14 x 3 steps = 42. Layers 1..47 were silently discarded.
> For the same reason **`L00` in the summary is a hardcoded label, not a field read
> from the row** — it is accidentally correct only because layer 0 prints first.
>
> **The conclusion drawn from the count survives; only the count's reach does
> not.** "The flag routes and layer 0 is where" needs *at least one* tap to have
> moved, not 42 to be the model, and `L00 blk` moving by `3.702e-04` from a
> bit-identical input carries that on its own. The same holds for §5.1's inference
> that `q_dtype == kBF16`. Read `26 of 42` as "26 of the 14 tags x 3 steps that
> reached the comparator", and read nothing about layers 1..47 into it.

### 5.1 `q`/`k`/`v` ARE bf16, and run 2 proves it without reading any source

Run 1's summary printed `FLAG ROUTES ON CPU ... NO -- either the flag is read and
ignored or q/k/v are not bf16`. **Both halves of that disjunction are false, and
the line is superseded.** It is computed from the eight ids alone, which is a
weaker instrument than the fingerprint, and it is committed in
`run1-results.txt` where a reader could mistake it for the answer. It is not.

The proof needs nothing but run 2's own numbers. `vt::GdnChunkedPrefillEnabled()`
— the `VT_GDN_CHUNKED` read — has **exactly one consumer in the tree**:

```c++
// src/vt/ops.cpp:2218
return q_dtype == DType::kBF16 && GdnChunkedPrefillEnabled();
```

If `q_dtype` were anything but bf16 the conjunction is false whatever the flag
says, and the flag could not change one bit of output. It changed `L00 blk` by
`3.702e-04` and 26 of 42 taps, on one binary by digest with `q`/`k`/`v` allocated
identically. **Therefore `q_dtype == kBF16`, as a consequence of the measurement
rather than an inference about it**, and the third possibility — that the arm is
reached, the flag routes, and the two arms simply land on the same eight argmaxes
— is what happened.

Two further readings agree. The fingerprint prints the runtime dtype of every
tap, and all 14 step-0 taps read `bf16` on all three arms. And statically,
`GdnActDType()` (`qwen3_5.cpp:3584`) is bf16 unless `VT_GDN_BF16` starts with
`0`, while `qwen4_exp_gguf_weights.cpp:206` hardcodes `torch_dtype = "bfloat16"`
rather than reading it from the GGUF.

**Run 1's `ARM ... DTYPE LINES: 0` field settled nothing and was never used.** It
greps the server log for a dtype line, and no such line exists to find:
`--verbose` prints request stages only, and neither branch of `cpu_ops.cpp:2136`
logs. A probe that produced no lines at all is not a measurement of the dtype,
and it is recorded here as a null probe so its zero cannot be read as one.

**Two independent corroborations that this is the same experiment as
PREFILLDIV.** The CPU-sequential `L00 blk` reads `1073.65489` and the CUDA one
`1074.03345`; PREFILLDIV, on a different tree two days earlier, read
`1073.65489` and `1074.03345` — byte-identical to the printed precision on both
arms. The new CPU-chunked value `1074.05248` is a third, distinct number between
them.

**The GDN divergence source is closed, and it is a 20x move.** Against CUDA, at
the taps the Gated DeltaNet feeds:

| tap | PREFILLDIV: CPU-sequential vs CUDA | **this run: CPU-chunked vs CUDA** | factor |
|---|---|---|---|
| `L00 blk` | 3.525e-04 | **1.772e-05** | **19.9x** |
| `L00 s.attn` | 1.939e-04 | **8.047e-06** | **24.1x** |
| `L00 mhc.mix` | 4.999e-04 | **4.324e-05** | **11.6x** |
| `L00 mhc.inj` | 3.884e-05 | **0.000e+00** | exact |

**ANNOTATION 2026-09-04 ([#2877](https://github.com/mudler/vllm.cpp/issues/2877)):
THE TABLE ABOVE IS MISMATCHED IN THE SAME WAY THE PARAGRAPH BELOW IS.** Its
"PREFILLDIV" column is CPU-**sequential** vs CUDA-**chunked** and its "this run"
column is CPU-chunked vs CUDA-chunked, so the factors are read across a change of
algorithm on one side. Read it with the two complete tables in the annotation
below, which apply ONE framing to every tap instead of one framing per row.

**And the second source is now the dominant one.** The same table's MoE taps move
the other way: `L00 moe` goes 1.269e-04 -> 2.289e-04 and `L00 s.mlp`
7.160e-05 -> 1.598e-04, from an input that is now 11.6x closer. That is
PREFILLDIV §3's unremoved residue, and with the GDN term gone it is what the
whole-model divergence is made of: `out` improves only 3.189e-03 -> 2.588e-03,
1.23x, against 20x at the block. **This is why the ids did not move.** The
largest divergence anywhere is at step 2's `out`: 1.471839e-02 between the two
CPU arms, 2.320338e-02 between CPU and CUDA.

> **ANNOTATION 2026-09-04 — BOTH READINGS ABOVE ARE FALSIFIED. WHETHER THE MoE
> RESIDUE GREW IS UNMEASURED, AND SO IS THE 20x AT THE BLOCK**
> ([#2877](https://github.com/mudler/vllm.cpp/issues/2877)). The text is kept, not
> rewritten: a later reader needs the shape of the error. The claim that replaces
> it is NOT "the residue did not grow" — that is the same overreach with its sign
> flipped. It is that **the metric these tables are built from cannot answer the
> question**, and that **no tap was taken at a step where the two arms disagree**.
> Four numbered points, and one correction this annotation owes itself.
>
> *(Editorial note, 2026-09-05,
> [#2969](https://github.com/mudler/vllm.cpp/issues/2969). That last clause is
> FALSE for `VT_MOE_SEL_FP`, and so are the same claim's other two occurrences in
> this annotation, at point (0) and at point (3). The full note is under point
> (0); read it before you read any of the three.)*
>
> **0. THE HEADLINE, AND IT IS THE ONE THAT SURVIVES EVERYTHING BELOW.** `LayerFp`
> returns early on `s.step >= s.budget`
> (`src/vllm/model_executor/models/qwen4_exp_forward.cpp:118`), so
> `VT_Q4EXP_LAYER_FP=3` fingerprints model forwards **0, 1 and 2** and nothing
> else. MOEDIV's committed digests count **48** MoE calls at `T=5` and **336** at
> `T=1` — 8 forwards for 8 tokens — so this window is tokens `11751 13 15767`,
> **which AGREE on both arms**. The three ids that disagree are at indices **4, 6
> and 7**. **No instrument on this row has yet observed a single disagreeing
> step.** Every number below is a measurement of three forwards on which the two
> arms produce the same token.
>
> *(Editorial note, 2026-09-05,
> [#2969](https://github.com/mudler/vllm.cpp/issues/2969). Read the attribution
> first. This annotation is THIS repository's prose. It was written under #2877
> in commit `60821f26c`, whose commit date is 2026-09-04 07:21:26 UTC, two hours
> after #2877 was filed at 05:21:23 UTC. **It is not a quotation of the issue
> body**, which contains no such paragraph. It is kept whole rather than
> rewritten, because a later reader needs the shape of the error.
>
> Three of its sentences carry the same overreach, and all three are wrong for
> `VT_MOE_SEL_FP`: the intro's "no tap was taken at a step where the two arms
> disagree", point (0)'s "No instrument on this row has yet observed a single
> disagreeing step", and point (3)'s "no tap was taken at a step where the ids
> disagree". `VT_MOE_SEL_FP`'s budget counts `MoeBlock` invocations and not model
> forwards, so wave MOEDIV's 384 calls is eight forwards, and its window reaches
> forwards 4, 6 and 7. The annotation had the arithmetic two sentences before its
> conclusion — it counts 48 MoE calls at `T=5` and 336 at `T=1`, "8 forwards for
> 8 tokens" — and applied it to the other instrument. The reading in that window
> is VOID for an unrelated reason, which is that the CUDA arm of that run was
> degenerate.
>
> #2877's own body states the SCOPED version of the same conclusion — "No tap in
> this evidence was taken at a step where the arms disagree" — and this
> correction does not overturn it. The unscoped sentences are this file's.)*
>
> **1. `rel(sumabs)` is a difference of NORMS, not a norm of DIFFERENCES, and its
> under-report is a DISTRIBUTION.** `run2-job.sh`'s `rel(a,b)` is evaluated on the
> tap's scalar `sumabs`, so the published quantity is `| S|a| - S|b| | / max`. Its
> zero means "the two tensors have equal L1 norm", not "the two tensors are
> equal", and it is not monotone in divergence. `S|x|` is sign-insensitive, so a
> zero-mean perturbation — which is every reassociation and rounding difference —
> cancels at `O(sqrt(n))`.
>
> The first version of this annotation quoted **one seed draw** for that factor
> ("122.7x at sigma 1e-3 and 229.8x at sigma 1e-4") to four significant figures,
> and said `sqrt(n) = 113` "is the observed factor" beside a table in which
> nothing equals 113. Over **400 seeds** at this tap's real size (`moe` is
> `o.tensor` `[T,H] = 5 x 2560 = 12800` bf16, `sum|x| ~ 390`):
>
> | perturbation | p05 | median | p95 |
> |---|---|---|---|
> | aligned with `sign(a)` | 1.00 | **1.00** | 1.00 |
> | zero-mean, sigma 1e-3 | 34 | **75** | 770 |
> | zero-mean, sigma 1e-4 | 48 | **140** | 1500 |
> | zero-mean, sigma 1e-5 | 48 | **140** | 1300 |
>
> The first row is the positive control, where the two measures must agree and
> do. There is no sigma dependence in the linear regime — 1e-4 and 1e-5 agree, and
> both approach `sqrt(2n/pi)/|z| = 90.3/|z|` for a standard normal `z`, median
> **134**.
>
> **TWO SIGNIFICANT FIGURES IS WHAT 400 DRAWS BUY, AND THIS TABLE FIRST CARRIED
> THREE.** The set published here until
> [#2879](https://github.com/mudler/vllm.cpp/pull/2879) — `31.4 / 69.2 / 568.2`,
> `43.9 / 125.6 / 1264.0`, `46.1 / 130.3 / 1398.7` — came from a script that was
> never committed, and it does not reproduce from `MetricSpread` over the 400
> seeds it names. It is not a different CONSTRUCTION: 14 of its 15 figures lie
> inside the committed control's own bootstrap 95% interval, and 13 of 15 inside
> the range six disjoint 400-seed blocks of that control span
> ([study](qwen4exp-gdn-chunked-token-ids-20260904/metric-spread-precision.py),
> [output](qwen4exp-gdn-chunked-token-ids-20260904/metric-spread-precision.txt)).
> Nor is it a short read of the same stream: section 3 of that study sweeps every
> prefix from 64 to 400 draws, none reproduces the set, and 11 of the 15 figures
> sit below every prefix. It was a different SAMPLE, quoted to a digit that 400
> draws do not determine.
> Every statistical figure in this annotation is now drawn over `range(400)` by
> `MetricSpread` and asserted there, rounded to two significant figures — and
> asserted against THIS FILE: `MetricSpread`'s `test_the_PUBLISHER_*` cases read
> this annotation, `docs/USAGE.md`, the tool's docstring, the tool's runtime
> stdout and both specs off disk, and compare every figure each of them quotes
> from the control to the drawn value, rendered by the one rounding function. The
> counts and six-block spans beside them are the committed precision study's, and
> it reads the same `metric_draw` by import. Moving a digit here without moving the
> draw reds a case that names this file and the figure. That publisher is what
> would have caught #2879, and it did not exist until #2879: the older
> `test_a_second_seed_block_moves_every_figure_a_third_digit_would_claim`
> compares the estimator to ITSELF and reads no document, so it measures how
> imprecise the estimator is and could not have seen a document quoting a sample
> this control never drew. What it does establish is the ceiling on the digits —
> the next disjoint 400 seeds give a median of **80** rather than 75 and a p95 of
> **574** rather than 770.
>
> **What this costs a reader is the SPREAD, and that is what settles #2877.** Hold
> the TRUE divergence fixed at sigma 1e-3 and vary only the perturbation's sign
> structure: `rel(sumabs)` spans **21x** p05..p95 while the true divergence spans
> 1.06x. Its end-to-end span is not a figure at all — one draw sets it, and it
> moves by more than 4x between seed blocks, so the **2078x** this line used to
> carry is withdrawn rather than restated. Two readings **of the same true
> divergence** differ by a median **2.1x**, 4.2x at p75, 11x at p90 and **24x** at
> p95. So, as the probability that an UNCHANGED divergence produces a ratio at
> least this large, in whole percent because that is the last digit 400 draws
> hold:
>
> | ratio | tap it is claimed for | P(no change produces it) |
> |---|---|---|
> | 24.1x | `s.attn`, defaults framing | 5% |
> | 19.9x | `blk`, defaults framing | 6% |
> | 16.7x | `blk`, matched framing | 7% |
> | 11.6x | `mhc.mix`, defaults framing | 9% |
> | 3.15x | `moe`, matched framing | **33%** |
> | 2.34x | `s.mlp`, matched framing | **45%** |
> | 2.02x | `mhc.mix`, matched framing | **52%** |
> | 1.80x | `moe`, defaults framing | **59%** |
>
> The four body rows hold to about a point between seed blocks. The four tail
> rows move by much more than a point: `P(>= 19.9x)` reads 6% on the committed
> block and 4% on the next one. The widest of the four, over the six disjoint
> blocks of the
> [precision study](qwen4exp-gdn-chunked-token-ids-20260904/metric-spread-precision.txt),
> is `P(>= 24.1x)`, which spans 2.56% to 5.42% — JUST OVER a factor of two, a
> 2.12x move. So read the tail rows as "a few percent", which is the only claim
> this row needs from them. This line first said "3% on the next one", truncating
> 3.7% where every other share on this row is rounded, which understated the move
> it was quoted to demonstrate.
>
> **Every MoE number in this row is an ordinary reading of no change at all.** The
> control is standard-library, fixed-seed and hermetic, and it is committed as
> `MetricSpread` in `tests/scripts/test_q4exp_layerfp_diff.py` so it runs on a
> lane rather than sitting in prose. It models the perturbation as i.i.d.
> zero-mean against a Gaussian signal, which is the premise under which
> `rel(sumabs)` is being read here; it bounds the METRIC's resolution and is not a
> significance test on the real tensors.
>
> **AND THAT PREMISE IS THE LOAD-BEARING ONE. THE FRESH RE-REVIEW OF THIS
> ANNOTATION FOUND THE BOUND IS MODEL-DEPENDENT, AND IT DOES NOT FAIL
> CONSERVATIVELY.** All three models below hold the same total perturbation
> energy and are drawn by the same committed control over the same 400 seeds, all
> three **at sigma 1e-3**:
>
> | perturbation, sigma 1e-3 | median under-report | pair p95 | P(no change >= 19.9x) |
> |---|---|---|---|
> | i.i.d. dense | 75x | 24x | 6% |
> | multiplicative, proportional to `a` (rounding-like) | 110x | 41x | 8% |
> | sparse, 16 elements of 12800 (like a top-k flip) | **2.9x** | 19x | 5% |
>
> A sparse perturbation is read almost in FULL — a median 2.9x under-report where
> the dense model gives 75x — so on a top-k flip the magnitude the metric reports
> is close to the real one. A multiplicative one is read WORSE than dense on every
> column **of that table**, because it carries no second-order term to hold the
> denominator away from zero.
>
> **THE SIGMA LABEL IS LOAD-BEARING, BECAUSE THE MAGNITUDE COLUMN REVERSES BELOW
> IT AND THIS ROW READS BELOW IT.** The multiplicative model is scale-INVARIANT —
> the control draws the same 110x, 41x and 8% at sigma 1e-3, 1e-4 and 1e-5 — while
> the dense model loses the second-order term that holds its denominator up as
> sigma falls, and its median under-report rises from 75x to **140x**. So in the
> LINEAR regime the DENSE model is the one read worse on magnitude, the opposite
> of the ordering the table shows. The two SPREAD columns do not reverse:
> multiplicative stays worse on both at every sigma. And the regime is not
> academic for this row — the control's median `rel(sumabs)` is **1.9e-05** at
> sigma 1e-4 and **1.9e-06** at sigma 1e-5, and the `1.772e-05` and `1.062e-06`
> the framing tables below argue over are at or below the first of those, where
> sigma 1e-3 draws **3.5e-04**. "Multiplicative is read worse" is a sigma 1e-3
> statement, and the conclusion this paragraph carries — that the bound does not
> fail conservatively — survives in both regimes, because the multiplicative
> model is read worse than the dense one on both SPREAD columns at every sigma,
> and on all three at sigma 1e-3.
>
> **The re-review first published this paragraph the other way round**
> (`35.3x -> 17.1x -> 5.3x` and `7.6% -> 4.1% -> 1.3%`, from the same uncommitted
> script as the table above, reading the multiplicative model as an improvement);
> those figures are withdrawn. The pair columns do not separate the three models
> the way the magnitude column does, and the sparse pair column is confounded
> anyway: 16 non-zero elements let the TRUE divergence span 3.2x across the same
> seeds where the dense model holds it to 1.06x, so part of that spread is real
> variation rather than metric noise. **The case is not hypothetical here:** #2552,
> cited below, names a dense reassociation term AND a bimodal top-k term at a
> 32.9% exact-bf16-tie rate, and a top-k flip is exactly the sparse case. This
> does not restore the ratio — which perturbation the real tensors carry is
> **unmeasured**, which is why the reading is withdrawn rather than reversed — but
> it names the assumption the next measurement has to establish.
>
> **2. ONE FRAMING, APPLIED TO EVERY TAP.** The first version of this annotation
> disqualified the MoE reading for mixing arms and then left the *favourable* half
> of the same three measurements standing. That is not admissible: `3.525e-04` and
> `1.269e-04` come from the same PREFILLDIV column, and a comparison that is
> mismatched for one is mismatched for the other. Four arms exist across the two
> runs, and the CPU-sequential and CUDA-chunked readings reproduce byte-for-byte
> between them, which is what makes the two runs one experiment.
>
> **Framing A — algorithm-MATCHED CPU vs CUDA** (the framing that killed the MoE
> reading, now applied to every tap). `L00 in` is `0.000e+00` on both arms, so
> layer 0's input is bit-identical and `blk` is the only tap that isolates a
> block; the rest are propagation from it.
>
> | tap | cpu-SEQ vs cuda-SEQ | cpu-CHUNKED vs cuda-CHUNKED | after the port |
> |---|---|---|---|
> | `L00 blk` | 1.062e-06 | 1.772e-05 | **16.7x FURTHER** |
> | `L00 s.attn` | 4.707e-06 | 8.047e-06 | 1.71x further |
> | `L00 mhc.mix` | 2.139e-05 | 4.324e-05 | 2.02x further |
> | `L00 mhc.inj` | 2.089e-05 | 0.000e+00 | to exactly zero |
> | `L00 moe` | 7.269e-05 | 2.289e-04 | 3.15x further |
> | `L00 s.mlp` | 6.815e-05 | 1.598e-04 | 2.34x further |
> | `out` | 1.130e-03 | 2.588e-03 | 2.29x further |
>
> **Framing B — the two SHIPPED defaults, before and after the port.** This is a
> true statement about what ships, and it is the only thing "19.9x" ever was. It
> is not algorithm-matched on the "before" side, because before #2612 the CPU
> default WAS the sequential arm.
>
> | tap | cpu-SEQ vs cuda-CHUNKED (before) | cpu-CHUNKED vs cuda-CHUNKED (after) | after the port |
> |---|---|---|---|
> | `L00 blk` | 3.525e-04 | 1.772e-05 | 19.9x closer |
> | `L00 s.attn` | 1.939e-04 | 8.047e-06 | 24.1x closer |
> | `L00 mhc.mix` | 4.999e-04 | 4.324e-05 | 11.6x closer |
> | `L00 mhc.inj` | 3.884e-05 | 0.000e+00 | to exactly zero |
> | `L00 moe` | 1.269e-04 | 2.289e-04 | 1.80x further |
> | `L00 s.mlp` | 7.160e-05 | 1.598e-04 | 2.23x further |
> | `out` | 3.189e-03 | 2.588e-03 | 1.23x closer |
>
> **The two framings disagree in DIRECTION on the same three measurements**, and
> (1) says only the top four ratios are outside the metric at all. So: the port's
> effect on the block is `19.9x closer` read one way and `16.7x further` read the
> other, at 6% and 7% under no change — neither is a result, and quoting only
> the first is what this repair exists to remove. The MoE taps move FURTHER under
> **both** framings and are inside the instrument under both, which is why
> "the residue grew" and "the residue did not grow" are equally unsupported.
>
> **Framing A's direction is not a surprise and not a defect.** The chunked
> decomposition has more reassociation freedom than an exact sequential
> recurrence, and this tree already measured that: chunked lands `2.29e-04` from
> the exact answer where sequential lands `1.15e-08`
> ([decomposition](gdn-chunked-decomposition-20260902.md)). Two chunked arms being
> further apart than two sequential arms is what that predicts. Accuracy and
> faithfulness are different things, and #2612 chose faithfulness to vLLM.
>
> **3. "This is why the ids did not move" is not supported by anything measured
> here.** The precise statement is (0)'s: **no tap was taken at a step where the
> ids disagree**. *(Editorial note, 2026-09-05,
> [#2969](https://github.com/mudler/vllm.cpp/issues/2969): that sentence is the
> third occurrence of the claim corrected under point (0), and it is FALSE for
> `VT_MOE_SEL_FP`, whose window reaches forwards 4, 6 and 7.)* The earlier
> annotation said "nothing measured on this row
> explains them", and that overshoots in the other direction — forwards 0, 1 and 2
> are causally UPSTREAM of forward 4 through the Gated DeltaNet recurrent state,
> so these taps are not irrelevant to the disagreeing ids. They simply do not
> observe them. Extending `VT_Q4EXP_LAYER_FP` past forward 3 is the measurement
> that would.
>
> **What the residue actually is** was already named by
> [#2552](https://github.com/mudler/vllm.cpp/issues/2552) and is not superseded:
> with selections held equal at layer 0 it decomposes onto the **keep-quant
> grouped expert GEMM** (`exp` 1.421e-04, 6.6x its input; `logit` only 1.1x, so
> the router GEMM is not the amplifier), above a **bimodal top-k term** at a 32.9%
> exact-bf16-tie rate. Both are **floors** that do not scale with input distance.
> Neither is a defect: #2552 read both as faithful mirrors of vLLM and llama.cpp.
> **This annotation is record and instrument work. It changes no kernel.**
>
> **The next traceable step, and the residue is NOT closed.** #2552 brackets the
> layer-0 expert-flip threshold between `2.139e-05` (no flip) and `4.999e-04`
> (flip). This run's matched pair sits at `4.324e-05` — **inside that bracket** —
> and `VT_MOE_SEL_FP` was not run on it, so whether the new reading contains a
> layer-0 selection flip is **unmeasured**. That single run, and a fingerprint
> budget that reaches forward 7, are the next two measurements. Both need the
> 68 GB released artifact and a GPU for the CUDA arm.
>
> **AND ONE CORRECTION TO THIS ANNOTATION'S OWN §5 NOTE.** "`42` is layer-0
> coverage" is wrong for 3 of the 14 tags: `emb`, `wide` and `out` are tapped with
> `il = -1`, outside the decoder loop, and are never layer 0. The committed
> differ's own largest value is at `('2', None, 'out')`. Read `42` as "the 14
> distinct tags x 3 steps that reached the comparator, at whichever `il` printed
> first for each" — layer 0 for the 11 in-loop tags, `-1` for the other three.


## 6. What this does NOT establish

- **It is not a token gate.** No oracle decoded this prompt. vLLM cannot run a
  GGUF artifact, and the llama.cpp arm aborts in `build_delta_net_chunking`
  before loading a byte — the blocker PREFILLDIV §8 recorded.
- **No speed number.** The wall times in §4 are liveness, on n=1 legs, on a box
  whose contention was not characterised.
- **One prompt, one length.** `T = 5`, a single partial chunk at `BT = 64`. The
  cross-chunk state carry — the half of the algorithm the spec's `## Owed`
  already flags as unmeasured against a real oracle — is not exercised here, and
  it is the half the bf16 state snapshot lives in.
- **The MoE residue is named, not diagnosed.** §5 first read it as "now
  dominant"; the annotation there withdraws that, because the metric the reading
  rests on cannot rank these taps in either direction. What the residue IS was
  named by [#2552](https://github.com/mudler/vllm.cpp/issues/2552) and is not
  diagnosed here. That is PREFILLDIV's open item and stays open.
- **The recorded explanation for the 5-of-8 is confirmed, not revisited.** It is
  tempting to read unchanged ids as "the divergence attributed to the algorithm
  difference did not move when the algorithms were unified". The premise is not
  established here: §5's annotations withdraw every ratio at the block, in both
  directions. What is measured is that the block output DID move — `3.702e-04`
  between the two CPU arms on the same binary, from a bit-identical input — and
  that the argmax did not.
  `.agents/specs/qwen4-exp-flash-next.md` already states the general form — token
  agreement between our two arms "is not monotone in the distance between them,
  because the decode is an argmax over near-ties", and a CPU-vs-CUDA
  token-exactness gate "is not well posed for this architecture at this
  precision". This run is a second, independent instance of exactly that, now in
  the direction that spec had not observed: PREFILLDIV saw a 332x *closer* arm
  agree on *fewer* ids; ARMTOKENS sees an arm whose layer-0 block output
  demonstrably moved agree on the *same* ids. Neither direction is monotone,
  which is the claim, and it does not need a ratio this file can defend.
- **Nothing is claimed about the other six published quants**, about
  `num_reqs > 1` (refused by name), or about ROCm, Vulkan and Tenstorrent — two
  of which still have no chunked arm.
- **The three CUDA gates the spec lists as unrun are still unrun.** §3 means they
  were unrunnable, not merely unrun; running them is not this wave.

## 7. Reproduction

`rc` jobs above on `thor:gpu0`, 03:28-04:37 UTC. The two job scripts, the two
`results.txt` files and the two tap diffs are committed beside this file in
[`qwen4exp-gdn-chunked-token-ids-20260904/`](qwen4exp-gdn-chunked-token-ids-20260904/).

```sh
vllm-server --model <shard1>.gguf --device cpu --host 127.0.0.1 --port 8171 \
    --block-size 16 --num-blocks 128 --max-model-len 256 \
    --served-model-name qwen4exp --verbose
curl -H 'Content-Type: application/json' \
     -d '{"model":"qwen4exp","prompt":"The capital of France is","max_tokens":8,"temperature":0,"logprobs":1}' \
     http://127.0.0.1:8171/v1/completions
```

`--block-size 16` because the CPU attention backend accepts only multiples of 16.
Swap `--device cuda` for the second row of the headline table, and prefix
`VT_GDN_CHUNKED=0` for the sequential arm.
