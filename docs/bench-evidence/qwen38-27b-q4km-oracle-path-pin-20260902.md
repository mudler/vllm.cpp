# The oracle emits OUR token at all three surviving divergences -- and there is still no path we pass against

Row `QUANT-QWEN38-27B-GGUF-ARM`, for
[#2534](https://github.com/mudler/vllm.cpp/issues/2534). Spec:
[`qwen38-27b-q4km-oracle-path-pin.md`](../../.agents/specs/qwen38-27b-q4km-oracle-path-pin.md),
whose denominator rule and per-outcome conclusions were committed at
`8e499f68b` **before this job ran**.

**`TOKEN_GATE=FAIL`, and the row returns `NEEDS_DECISION`.** This measurement is
about the DENOMINATOR. No engine numerics changed, and no speed or memory axis
becomes admissible.

## The two findings, and they pull in opposite directions

1. **All three surviving divergences are AMBIGUOUS.** At `p1/34`, `p2/20` and
   `p4/14`, the stock oracle at `b10451` emits **vllm.cpp's own token** under at
   least one of its own supported configurations, on the identical artifact,
   prompts, token count and sampling. Token-exactness is not a well-defined
   target at those three steps.
2. **No single pinned path is one we are token-exact against.** Of the ten paths
   measured, the best agrees with us at **3 of the 5** comparable near-tie steps,
   and the three paths that reach 3 of 5 each fail on a **different** pair. So
   the ambiguity is real and it does not rescue the gate: the gate fails against
   the stock default and against every other path measured.

The second finding is why this document does not report a pass. Being individually
inside the oracle's path-induced ambiguity at each step is not the same as
matching any oracle.

## What ran

| `rc` job | Device | Purpose | Window (UTC) |
|---|---|---|---|
| `019596fd-5e88-4d0d-aa93-2749ab618524` | `thor:gpu0` | aborted at the source fetch, no measurement | 15:51:21 to 15:59:16 |
| `8480a30e-0d6d-44a7-b1b4-00e9d36c888d` | `thor:gpu0` | this measurement, 12 oracle runs | 16:01:41 to 16:35:54 |

Worker `rc-worker-n8smh`, the same worker the 2026-09-02 self-consistency band
was measured on. `thor` deliberately: this oracle's executed kernel path is
host-dependent, so a comparison against ids recorded there is only valid there.
Evidence on the share at `/mnt/nas_share/rc/q4kpath-thor/`, which the worker sees
as `/workspace/q4kpath-thor/`: `job/` holds every script, `out/rc-worker-n8smh-20260902T160141Z/`
holds every log and every per-arm raw file. Nothing reached the box by `ssh`.
No vllm.cpp was built; this job measures the ORACLE only.

**The first job died because the worker container has lost `github.com` egress**
(`could not read Username for 'https://github.com'`, then `fatal: expected flush
after ref listing`). The pinned source is therefore STAGED rather than fetched,
and the pin is asserted more strongly than before rather than less. See
"Identity" below.

## Identity, recomputed rather than trusted

```text
llama_pin              = 10bf611e533d81f739128304991c5e133c6aebd8  (tag b10451)
llama_pinned_tree      = 870465888f30ef3cc98279f6bfa5e41a17d17477
recomputed_tree        = 870465888f30ef3cc98279f6bfa5e41a17d17477   SOURCE_TREE_VERIFIED=EXACT
recomputed_after_build = 870465888f30ef3cc98279f6bfa5e41a17d17477   SOURCE_TREE_STILL_EXACT
staged_tarball_sha256  = 12616c65e1abf8b99af392fa5ab2702e8a5ef3ff1192f745482459d310bbd84e
llama_completion_sha256= 35f5ba3ebe7e80bfbf96cfc7612375776564a482b9e7c4fafbd74d1f5df89934
oracle_paths_sha256    = f3db3be3cc91511955a0126837586c0145770e8ce766ea8d94a59cb4648850a9
gguf_sha256            = 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
gguf_size              = 17106775008
host                   = rc-worker-n8smh (thor:gpu0), aarch64, 14 cores
system_info            = CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 |
                         SVE = 1 | DOTPROD = 1 | SVE_CNT = 16 | OPENMP = 1 | REPACK = 1 |
unused blk.64 tensors  = 15   (re-observed, not cited)
```

**Why the recomputed tree is a stronger assertion than the fetch it replaces.**
A `git rev-parse HEAD` plus an empty porcelain says a checkout *claims* a name.
The job instead extracts the staged archive, runs `git add -A --force` and
`git write-tree`, and requires the result to equal commit `10bf611e`'s own tree
object. That rehashes every blob and every tree from the bytes that are about to
be compiled. Git is content-addressed, so a tree hashing to
`870465888f...` **is** that commit's tree. The same recomputation is repeated
after the build, which is what replaces the post-harness porcelain check.

## Controls, all green, each of which could have failed the job

| control | result | what it falsifies |
|---|---|---|
| 1. denominator identity | `F0_IDENTITY=EXACT`, 6 of 6 | the host, build or artifact moved since 2026-08-23 |
| 2. teacher-forcing | `CONTROL2_STEPS=288 CONTROL2_MISMATCHES=0 CONTROL2=EXACT` | the harness is not reporting argmax under a fixed context |
| 3. lever liveness | each arm prints its own `use_extra_bufts`, threads, requested and context-resolved `n_batch`/`n_ubatch`, flash-attention | a knob that never reached the runtime reads exactly like a stable oracle |
| 4. free-generation replication | `ORACLE_SELF_DIVERGENCES=1/6`, `prompt 1, index 34, 3095 -> 198` | this job is not comparable to the 2026-09-02 measurement it extends |
| 5. tree assertion | above | the built source is not the pin |
| 6. instrument self-test | `SUMMARY_SELFTEST=PASS` | the summary returns a well-formed wrong answer |

Control 6 ran on the worker, before the arms, against synthetic arms whose
answers are planted: one flips the contested step to our token, one to a third
token, one leaves it alone, one is truncated. The job aborts unless the summary
returns `UNSTABLE_TOWARD_US`, `UNSTABLE_ELSEWHERE`, `STABLE` and `NOT SCORED`
for exactly those four. It executes `job/summary_body.py`, the same file the job
runs, not a copy: a transcription cannot gate the function it transcribes.

## The arms

Twelve runs: the denominator free, ten teacher-forced along the denominator's own
ids, and the `-nr` arm free as control 4. Teacher forcing is what makes the arms
comparable -- two configurations that have already diverged cannot be compared at
a later step, so every arm's argmax at a step describes the identical preceding
context.

| arm | `use_extra_bufts` | threads | `n_ubatch` | flash-attn | diverges from the default arm |
|---|---:|---:|---:|---|---|
| `default_tf` (denominator) | 1 | 14 | 512 | auto | 0 of 6 |
| `faon_tf` | 1 | 14 | 512 | enabled | 0 of 6 |
| `t13_tf` | 1 | 13 | 512 | auto | 0 of 6 |
| `t4_tf` | 1 | 4 | 512 | auto | 0 of 6 |
| `t1_tf` | 1 | 1 | 512 | auto | 0 of 6 |
| `ub4_tf` | 1 | 14 | 4 | auto | 0 of 6 |
| `norepack_tf` | 0 | 14 | 512 | auto | 1 of 6: `p1@34 3095 -> 198` |
| `ub1_tf` | 1 | 14 | 1 | auto | 1 of 6: `p4@14 22486 -> 4593` |
| `faoff_tf` | 1 | 14 | 512 | disabled | 2 of 6: `p0@7 9564 -> 9338`, `p3@45 393 -> 25` |
| `norepack_ub1_tf` | 0 | 4 | 1 | auto | 3 of 6: `p0@7 9564 -> 9338`, `p1@34 3095 -> 198`, `p2@20 539 -> 13` |

Every arm ran 288 of 288 steps and reached its OK line; none was scored
incomplete. `system_info` is the line above for every arm.

### Two lever results worth stating separately

**The denominator's resolved attention kernel is flash-attention ENABLED, and
that is measured.** `faon_tf` is **byte-identical to `default_tf` across all 288
STEP lines**, and `faoff_tf` is not, so `AUTO` resolves to `ENABLED` on this
host. The 2026-08-23 gate therefore ran with flash attention on, which no earlier
record states.

**The thread-count control held, exactly as the spec predicted from the source.**
`t1_tf`, `t4_tf` and `t13_tf` are **byte-identical to `default_tf` across all 288
STEP lines**, at 1, 4 and 13 threads against 14. `repack.cpp:4317-4372` derives
the chunk bounds from `nth` and then rounds them to `NB_COLS`, but each output
row's dot product completes inside one `gemv`/`gemm` call over the full `ne00`,
so a chunk boundary selects WHICH rows a thread computes and not HOW any row is
computed. The prediction was committed before the run and it is reported here
because it held; had it fired, the source reading would have been wrong.

`ub4_tf` is also byte-identical: at prompt lengths 5 to 11 an ubatch of 4 leaves
the gemm/gemv split unchanged. `ub1_tf` does not, because at `n_ubatch = 1` every
prefill row takes the GEMV body (`repack.cpp:4240`, `if (nrows > 3)`).

## The three contested steps

`ours` is vllm.cpp's token on `main` at `27da7787e`, after the f32 logits head.
`margin` is that arm's own `top1 - ours`.

### `p1/34` -- ours `198`, denominator `3095`

| arm | argmax | ours logit | ours rank | margin |
|---|---:|---:|---:|---:|
| `default_tf`, `faon_tf`, `t1/t4/t13`, `ub4` | 3095 | 19.653543 | 2 | 0.085434 |
| `faoff_tf` | 3095 | 19.725327 | 2 | 0.038246 |
| `ub1_tf` | 3095 | 19.660854 | 2 | 0.065060 |
| **`norepack_tf`** | **198 (OURS)** | 19.814404 | **1** | 0.000000 |
| **`norepack_ub1_tf`** | **198 (OURS)** | 19.814404 | **1** | 0.000000 |

`STEPVERDICT p1/34 tokens_seen=[198, 3095] -> UNSTABLE_TOWARD_US`

### `p2/20` -- ours `13`, denominator `539`

| arm | argmax | ours logit | ours rank | margin |
|---|---:|---:|---:|---:|
| `default_tf`, `faon_tf`, `t1/t4/t13`, `ub4` | 539 | 22.465479 | 2 | 0.178236 |
| `faoff_tf` | 539 | 22.513199 | 2 | 0.140160 |
| `ub1_tf` | 539 | 22.412962 | 2 | 0.123640 |
| `norepack_tf` | 539 | 22.593998 | 2 | **0.013126** |
| **`norepack_ub1_tf`** | **13 (OURS)** | 22.478220 | **1** | 0.000000 |

`STEPVERDICT p2/20 tokens_seen=[13, 539] -> UNSTABLE_TOWARD_US`

This was the largest of the three margins at 0.178236. Under `norepack_tf` the
same step narrows to 0.013126 without flipping, and under `norepack_ub1_tf` it
flips to our token.

### `p4/14` -- ours `4593`, denominator `22486`

| arm | argmax | ours logit | ours rank | margin |
|---|---:|---:|---:|---:|
| `default_tf`, `faon_tf`, `t1/t4/t13`, `ub4` | 22486 | 15.919413 | 2 | 0.115482 |
| `faoff_tf` | 22486 | 15.875310 | 2 | 0.026988 |
| `norepack_tf`, `norepack_ub1_tf` | 22486 | 16.043646 | 2 | 0.098591 |
| **`ub1_tf`** | **4593 (OURS)** | 15.989957 | **1** | 0.000000 |

`STEPVERDICT p4/14 tokens_seen=[4593, 22486] -> UNSTABLE_TOWARD_US`

### The two steps the f32 head already resolved, for completeness

`p0/7` is `UNSTABLE_TOWARD_US` as well: `faoff_tf` and `norepack_ub1_tf` emit
`9338`, the token we produced BEFORE the f32 head landed and no longer produce.
`p5/32` is `STABLE`: every arm emits `16`, which is also our token, so we agree
with an oracle that agrees with itself there.

**`p1/35` is reported by the harness and must not be read as a result.** It sits
inside a prompt already diverged at 34, so under teacher forcing along the
denominator's ids its context is the denominator's and not the one vllm.cpp had.
Its `ours_rank=17016` is a probe of our token in a context we never occupied. It
is excluded from every count in this document.

## Scoring, under the rule fixed in advance

The spec's denominator rule, committed at `8e499f68b`: the oracle's **stock
default configuration** -- `use_extra_bufts = 1`, flash attention at its `AUTO`
default (resolved here, and measured, as `ENABLED`), `n_threads = nproc = 14`,
`n_batch = n_ubatch = n_ctx = 512`, aarch64 with the `system_info` above, built
`GGML_NATIVE=ON` from a tree that recomputes to the pin. It was chosen because it
is what a user gets with no flags and it is the configuration the 2026-08-23 gate
recorded; it was NOT chosen by which arm agrees with us.

Under that rule the arm diverges on prompts 1, 2 and 4, so **`TOKEN_GATE=FAIL`,
3 of 6.** The spec pre-registered that this rule yields FAIL whatever the arms
show, and it does. **The measurement did not rescue the gate and this document
does not claim it did.**

## Why the ambiguity does not become a pass under any other pinned path either

Derived from the per-arm raw files, one row per arm, N = 10 by design:

```text
PATHAGREE ub1_tf             agrees_with_us=3/5  agree=p0/7,p4/14,p5/32  differs=p1/34,p2/20
PATHAGREE norepack_ub1_tf    agrees_with_us=3/5  agree=p1/34,p2/20,p5/32 differs=p0/7,p4/14
PATHAGREE norepack_tf        agrees_with_us=3/5  agree=p0/7,p1/34,p5/32  differs=p2/20,p4/14
PATHAGREE ub4_tf             agrees_with_us=2/5  ...
PATHAGREE t4_tf              agrees_with_us=2/5  ...
PATHAGREE t1_tf              agrees_with_us=2/5  ...
PATHAGREE t13_tf             agrees_with_us=2/5  ...
PATHAGREE faon_tf            agrees_with_us=2/5  ...
PATHAGREE default_tf         agrees_with_us=2/5  ...
PATHAGREE faoff_tf           agrees_with_us=1/5  agree=p5/32
BEST_AGREEMENT=3/5 ; a path we are token-exact against would need 5/5
```

The three arms that reach 3 of 5 fail on **different** pairs, so no combination
of them is a single configuration. Repinning the denominator to any of them
would move divergences rather than remove them: `norepack_ub1_tf` would agree
with us at `p1/34` and `p2/20` and start disagreeing at `p0/7`, a step we are
currently exact on.

**A denominator chosen per step is not a denominator.** That is precisely the
failure the spec's rule exists to prevent, and this table is what it looks like
when someone tries.

## NEEDS_DECISION

The three surviving steps each landed on the spec's outcome **C2**, whose
pre-registered handling is to report the step as AMBIGUOUS and escalate rather
than score it either way. The question is a product decision and is not the
implementer's to take:

**What should a token gate do at a step where the pinned oracle's own supported
configurations disagree with each other, and one of them emits our token?**

Three options, with what this measurement says about each:

- **A. Keep the gate as it is** -- exact against the stock default, 6 of 6. The
  arm stays at FAIL. This is the status quo and what is reported above. Its cost
  is that the gate demands bit-reproduction of one aarch64 repacked kernel
  schedule at margins the oracle cannot itself resolve; `b10451` scores 5 of 6
  against itself under one stock flag.
- **B. Score a step against the SET of tokens the oracle's supported paths emit.**
  This is weaker and it is not adopted here. Note what it would also do: under it
  `p0/7` becomes ambiguous too, and prompt 3 -- which we are token-exact on --
  becomes ambiguous, because `faoff_tf` diverges from the default at `p3@45`. It
  would loosen steps we currently pass, not only steps we currently fail.
- **C. Pin a narrower path and keep exactness** -- for example declare the
  denominator to be the stock default AND require the gate to report the six
  near-tie margins beside the verdict, so a sub-0.2-logit decision is visible as
  such rather than silently binary.

This document takes none of them. It reports A's result because A is the rule
that was fixed in advance.

## What is NOT established

- **No speed or memory number.** None was run and none may be quoted; the arm's
  token gate does not pass.
- **Whether any arm reproduces our full 48-token sequences.** Only argmax under
  the denominator's context was measured. An arm that emits our token at one step
  says nothing about the 13 steps after it.
- **Anything about the ROCm arm.** This is the CPU tier on aarch64.
- **That the remaining error is not ours.** The three steps are ambiguous against
  this oracle. That is a statement about the denominator's resolving power, not a
  clean bill of health for the engine, and the `PATHAGREE` table is the reason:
  we match no path.

## Still owed

- The decision above, under [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
- The aarch64 `ggml_gemv_q4_K_8x8_q8_K` port, unchanged and still owed.
- The unexplained heavy tail (`max_abs` 17.1606 at `p5/2`, `p5/11`, `p0/45`),
  untouched by this job.
