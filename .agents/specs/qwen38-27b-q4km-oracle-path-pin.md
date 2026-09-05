# Pin the Q4_K_M oracle's EXECUTED kernel path, then score the token gate against it

Row: `QUANT-QWEN38-27B-GGUF-ARM`.
Issue: [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
Predecessors, all landed and none re-derived here:
[`qwen38-27b-q4km-token-exactness.md`](qwen38-27b-q4km-token-exactness.md),
[`qwen38-27b-q4km-logits-f32.md`](qwen38-27b-q4km-logits-f32.md),
[`docs/bench-evidence/qwen38-27b-q4km-oracle-self-consistency-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-oracle-self-consistency-20260902.md),
[`docs/bench-evidence/qwen38-27b-q4km-logits-f32-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-logits-f32-20260902.md).

## Why this exists

`.agents/oracles/llama-cpp.md` now records that this oracle's greedy decode is
**not deterministic across its own supported kernel paths**, and it obliges any
token gate against it to pin the EXECUTED path rather than only the revision.
That obligation is written and has never been discharged: the gate that stands
at `FAIL` was scored against a revision, and its denominator's executed path was
recorded only after the fact.

The arm is at 3 of 6 divergent after the f32 logits head landed. Prompt 1 (first
divergence at index 34), prompt 2 (index 20) and prompt 4 (index 14) remain.
Their first-divergence margins are 0.085434, 0.178236 and 0.115482, all below
the oracle's measured MINIMUM per-step self-perturbation of 0.2020.

This row asks one question and it is a question about the DENOMINATOR, not about
the engine: **at those three steps, does the oracle emit a different token under
its own supported kernel-path variations?** It changes no engine numerics.

## The hypothesis, and why it is being tested rather than assumed

The convenient reading is that all three steps are oracle noise, and that the
gate's `FAIL` is an artefact of an unpinned denominator. That reading is
convenient, and one measurement already points at it: `-nr/--no-repack` flips
prompt 1 index 34 to `198`, which is our token. It is nonetheless only a
hypothesis, and the `-nr` measurement covers exactly ONE lever and leaves
prompts 2 and 4 untested against any other.

The counter-hypothesis is that prompts 2 and 4 are stable under every supported
path, that they are simply ours, and that the gate is right to fail.

## The denominator rule, fixed BEFORE any new data is read

> **The pinned denominator is the oracle's STOCK DEFAULT configuration on the
> measuring host.** That is: `use_extra_bufts = 1` (llama.cpp's own default;
> `-nr` NOT passed), `flash_attn_type = AUTO` (the default), `n_threads =
> n_threads_batch = nproc`, `n_batch = n_ubatch = n_ctx = 512`, on `thor:gpu0`
> (aarch64, 14 cores), built `GGML_NATIVE=ON` from a byte-clean tree at pin
> `10bf611e533d81f739128304991c5e133c6aebd8` (`b10451`).

The rule is chosen for a reason that has nothing to do with which arm agrees
with us: it is what a user gets by running the released binary with no flags,
and it is the configuration the 2026-08-23 gate already recorded. It is fixed
here, in a commit that precedes the measurement, so that it cannot be selected
after the tokens are seen.

Every arm records `system_info`, `use_extra_bufts`, host architecture, thread
count, and the context's resolved `n_batch`/`n_ubatch`. That set is what
"pinned kernel path" means in `.agents/oracles/llama-cpp.md`, and an arm that
cannot report it is not evidence.

## Scope

1. Enumerate, from the pinned source, the stock levers that change WHICH Q4_K
   kernel executes without changing the model, the prompts, the token count or
   the sampling.
2. Run each of those arms against the identical artifact and recipe, TEACHER
   FORCED along the default arm's own ids, so every arm's argmax at a given
   step describes the identical preceding context. Two arms that have already
   diverged cannot be compared at a later step, which is why free generation
   alone cannot answer this question.
3. Report, per contested step: our token, each arm's argmax, and our token's
   logit, rank and margin under each arm.
4. Re-score the gate under the denominator rule above.

Out of scope: any engine change, any speed or memory number, the ROCm allocator
work (#2511), and porting ggml's aarch64 repacked gemv (owed under #2534).

## The levers, read from the pin rather than guessed

`ggml/src/ggml-cpu/repack.cpp` at the pin:

- **`use_extra_bufts`** (`common/arg.cpp:2411-2418`, `LLAMA_ARG_REPACK`, default
  true) selects the repacked `block_q4_Kx8` weights and `ggml_gemv_q4_K_8x8_q8_K`
  against the plain `ggml_vec_dot_q4_K_q8_K`. Already measured; re-run as a
  replication.
- **`n_ubatch`** selects GEMM against GEMV *per chunk*:
  `forward_mul_mat_one_chunk` (`repack.cpp:4240`) calls `gemm` for
  `nrows - (nrows % 4)` rows only when `nrows > 3`, and `gemv` for the tail. With
  a 5-to-11-token prompt in one ubatch the prefill takes the GEMM body; at
  `n_ubatch = 1` every prefill row takes the GEMV body. Those are different fp32
  rounding schedules, and the prefill writes the KV cache that every later step
  reads.
- **`flash_attn_type`** selects the attention kernel. The default is `AUTO`, so
  the arm the gate ran is whatever `AUTO` resolved to on this host, which the
  recorded evidence never states.
- **`n_threads`** changes the chunk boundaries in `forward_mul_mat`
  (`repack.cpp:4317-4372`): `dr0` is derived from `nth`, and chunk starts and
  ends are then rounded to `NB_COLS`. It is included as a **predicted-null
  control**: reading the pin, each output row's dot product is completed inside
  one `gemv`/`gemm` call over the full `ne00`, and a chunk boundary selects WHICH
  rows a thread computes rather than HOW any row is computed, so thread count
  should not move a token. **If a thread-count arm moves a token, that reading is
  wrong and this spec says so rather than quietly dropping the control.**

## Pre-registered conclusions

Fixed before the job runs. `s` is one of the three contested steps `p1/34`,
`p2/20`, `p4/14`; `T_def(s)` is the default arm's token there; `T_ours(s)` is
`198`, `13`, `4593` respectively.

- **C1 — STABLE.** Every arm's argmax at `s` equals `T_def(s)`. Then the oracle
  is well defined at `s`, we differ from it, and **that step is our defect. It
  stays a FAIL and this report says so plainly.**
- **C2 — UNSTABLE TOWARD US.** At least one arm's argmax at `s` equals
  `T_ours(s)`. Then token-exactness is **not well defined** at `s`. This is
  reported as **AMBIGUOUS**; it is NOT converted into a pass, and the row returns
  **NEEDS_DECISION** on what the gate should do with such a step.
- **C3 — UNSTABLE ELSEWHERE.** Some arm's argmax at `s` is a third token and no
  arm emits `T_ours(s)`. Then the oracle is unstable at `s` and still never
  agrees with us, so the step remains **our defect** and a FAIL.

### The gate verdict rule, also pre-registered

Under the denominator rule, `TOKEN_GATE = PASS` if and only if **0 of 6** prompts
diverge from the DEFAULT arm's ids. An AMBIGUOUS step never converts a
divergence into a pass. Because prompts 1, 2 and 4 already diverge from the
default arm, **this rule yields `TOKEN_GATE=FAIL` whatever the arms show.** That
is stated here deliberately: the measurement cannot rescue the gate, and any
report that claims otherwise has changed the rule after seeing the data.

A second rule exists and is NOT adopted here: score a step against the SET of
tokens the oracle's supported paths emit, and count a step as passing when our
token is in that set. It is a weaker gate and a genuine product decision, so its
result is reported as a factual count and never as the gate verdict.

## Controls, and what each one falsifies

1. **Identity.** The default arm run FREE must reproduce the 2026-08-23 recorded
   ids on 6 of 6 prompts. If it does not, the host, the build or the artifact has
   moved and no arm below is comparable to the recorded gate. The job aborts.
2. **Teacher-forcing self-check.** The default arm run TEACHER FORCED along its
   OWN ids must have `argmax == forced` at all 288 steps. A harness that fails
   this is not measuring argmax under a fixed context.
3. **Lever liveness.** Each arm prints `USE_EXTRA_BUFTS`, the requested and the
   context-resolved `n_batch`/`n_ubatch`, the thread count and the flash-attention
   request. An arm whose knob did not reach the runtime proves nothing, and a
   silent no-op arm reads exactly like a stable oracle.
4. **Free-generation replication.** The `-nr` arm is also run FREE, and must
   reproduce `ORACLE_SELF_DIVERGENCES=1/6` with `prompt 1, index 34, 3095 ->
   198`. This binds this job to the 2026-09-02 measurement it extends.
5. **Tree assertion.** `git status --porcelain` empty on the pinned source before
   AND after the harness is built, and the built binary's sha256 recorded.
6. **Instrument self-test.** The summary that produces the three step verdicts is
   run first against synthetic arms whose answers are planted by construction:
   one arm flips the contested step to OUR token, one flips it to a third token,
   one leaves it alone, and one is truncated. The job aborts unless the summary
   returns `UNSTABLE_TOWARD_US`, `UNSTABLE_ELSEWHERE`, `STABLE` and `NOT SCORED`
   for exactly those four. The self-test executes `job/summary_body.py`, the same
   file the job runs, rather than a copy of it: a transcription cannot gate the
   function it transcribes. A summary that reads the wrong field returns a
   well-formed wrong answer, and this repository has already published one that
   inverted its own result.

## Risks

- **The flattering conclusion.** Named in `## Pre-registered conclusions` and
  disarmed by fixing the denominator rule in a commit that precedes the data.
- **A no-op arm.** Control 3.
- **Leg double-counting.** The job log prints each leg twice through `tee`. Every
  count in the evidence is derived from the design (one line per arm per step)
  by a script, never read off a summary line.
- **`flash_attn_type = ENABLED` may be unsupported on this CPU build.** The
  harness reports `ARM_UNAVAILABLE` and the job continues; an unavailable arm is
  recorded as unavailable and never as agreement.

## Tests

This row runs an ORACLE measurement and changes no product code, so it adds no
unit test. Its instruments are the five controls above, each of which can fail
the job. The harness is `oracle_paths.cpp`, derived from the 2026-09-02
`oracle_logits.cpp`, which was itself derived from the W3 `oracle_tokens.cpp`
whose detokenized output was asserted byte-equal to stock `llama-completion`'s
stdout. It links the stock `libllama` built at the pin and calls the public
`llama.h` API only.

## Gates

```sh
# oracle-only; no vllm.cpp is built. Runs inside an rc lease on thor:gpu0.
setsid nohup rc run --device thor:gpu0 -- bash /workspace/q4kpath-thor/job/job.sh
```

## Evidence

`docs/bench-evidence/qwen38-27b-q4km-oracle-path-pin-20260902.md`, carrying per
arm: `system_info`, `use_extra_bufts`, host architecture, thread count, resolved
`n_batch`/`n_ubatch`, and per contested step the argmax, our token's logit, rank
and margin.

## Stop conditions

- Control 1 or 2 fails: report BLOCKED and record what moved. Do not score.
- Any contested step lands on **C2**: return **NEEDS_DECISION**. Do not pick the
  arm that agrees with us.
- A `PASS` obtained by choosing the denominator after the fact is not reportable
  under any circumstance.

## Result, 2026-09-02

rc job `8480a30e-0d6d-44a7-b1b4-00e9d36c888d`, `thor:gpu0`, worker
`rc-worker-n8smh`. Evidence:
[`docs/bench-evidence/qwen38-27b-q4km-oracle-path-pin-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-oracle-path-pin-20260902.md).
All six controls green, including the instrument self-test and
`F0_IDENTITY=EXACT`.

**All three surviving steps landed on `C2`.** The oracle emits vllm.cpp's own
token at `p1/34` (`norepack_tf`, `norepack_ub1_tf`), at `p2/20`
(`norepack_ub1_tf`) and at `p4/14` (`ub1_tf`), on the identical artifact,
prompts, token count and sampling. Per `## Pre-registered conclusions` those
steps are AMBIGUOUS, they are not converted into passes, and the row returns
**NEEDS_DECISION**.

**`TOKEN_GATE=FAIL`, 3 of 6, under the denominator rule fixed at `8e499f68b`** --
the outcome that rule was pre-registered to produce whatever the arms showed.

**And the ambiguity does not become a pass under any other pinned path.** Of the
ten paths measured, the best agrees with us at 3 of the 5 comparable near-tie
steps, and the three that reach 3 of 5 fail on different pairs, so no single
supported configuration is one this arm is token-exact against. Repinning to any
of them moves divergences rather than removing them.

**The thread-count control held.** `t1`, `t4` and `t13` are byte-identical to
the 14-thread denominator across all 288 steps, as `## The levers` predicted from
`repack.cpp:4317-4372` before the run. `ub4` is byte-identical too; `ub1` is not.

**One fact the earlier records did not carry:** the denominator's resolved
attention kernel is flash attention ENABLED. `faon_tf` is byte-identical to the
`AUTO` default over all 288 steps and `faoff_tf` is not, so `AUTO` resolves to
`ENABLED` on this host, and the 2026-08-23 gate ran that way.

### Deviation from `## Gates`

The worker container has lost `github.com` egress, so the pinned source is staged
and its tree hash is RECOMPUTED on the worker with `git write-tree` against
commit `10bf611e`'s own tree object `870465888f30ef3cc98279f6bfa5e41a17d17477`,
before and after the build. That is a stronger assertion than the fetch it
replaces, which only checked that a checkout claimed a name.
