# Limb 3, run on a vehicle for the first time

Issue [#2884](https://github.com/mudler/vllm.cpp/issues/2884), row
`QUANT-QWEN38-27B-GGUF-ARM`, spec
[`limb3-vehicle-strict-gate.md`](../../.agents/specs/limb3-vehicle-strict-gate.md).
The pin this ran on is
[`limb3-vehicle-pin-20260904.md`](limb3-vehicle-pin-20260904.md);
[#2864](https://github.com/mudler/vllm.cpp/issues/2864) is why it could not run
before.

**No speed, latency or memory figure appears below and none was taken.**
`AGENTS.md` §Gates admits a performance result from an arm only after that arm's
declared token gate passes, and
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) has already had one
measurement retracted for exactly that. `STRIX_ARM_SPEED_RATIFIED_BY` is unset
and [`../../.agents/scripts/rocm-strix-ourarm-staged.sh`](../../.agents/scripts/rocm-strix-ourarm-staged.sh)
still refuses.

## What ran

One `rc` job on `strix:gpu0`, worker `rc-worker-lcjhd`. Nothing reached the box
by `ssh`. `HSA_OVERRIDE_GFX_VERSION` is set nowhere, and the job refuses to
start if it inherited any `HSA_*`, `ROCR_*`, `PYTORCH_*`, `HIP_*` or `VT_*`
variable -- it fails on that rather than printing a line somebody has to read.

`gate.sh` is the whole job, `gen_vehicle.py` is the oracle's side and
`score_strict.py` is the verdict. All three are committed beside this file with
the job log.

## Both sides are REUSED, and each is asserted to be the object it claims

The worker has not rebooted since 2026-09-01 (`boot_id`
`a5bc8128-f6ad-4767-8614-6923f88032e1`, the same id the 2026-09-02 token gate
and the 2026-09-03 oracle run recorded), so `/tmp` still carries both builds.
That is a large saving and it is also the obvious way to measure the wrong
object, so nothing is taken on trust:

- our three build products are asserted against the sha256 values committed in
  [`qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`](qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md),
  which a different run wrote on a branch this job cannot edit. **The binary
  under test is byte-for-byte the binary the arm's own declared gate ran.**
- the vLLM venv is asserted on `vllm.__version__`, the device's `gcnArchName`,
  the compiled `vllm._C` and `vllm._rocm_C` extensions, `RocmPlatform`
  resolution and the plugin's `_C_gguf` extension, before it generates
  anything.

A rebuild would fail those assertions rather than pass silently.

## The prompts were fixed before anything was scored

`prompts_sha256` `c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e`,
six prompts, 48 tokens each, 288 decode steps. **They are not chosen by this
run.** They are the set this campaign has scored since 2026-08-23 and their
hash is committed in FOUR evidence documents that predate it (counted at
`a1ada1471`: the two `20260902` token gates, `oracle-vllm-gfx1151-20260903.md`
and `q4km-neartie-vllm-oracle-20260903.md`), so no selection
after seeing which prompts agree is possible. `gate.sh` hashes the file and
refuses on a mismatch before it scores anything, and `score_strict.py` refuses
again independently.

The prompt **token ids** are produced by our engine from the vehicle's own GGUF
vocab and FED to vLLM, so a tokenizer difference cannot contaminate a generation
comparison. vLLM's own tokenizer re-derives them from
`Qwen/Qwen3.6-27B` @ `6a9e13bd6` and the two are printed side by side.

## THE ANSWER

**`STRICT_LIMB3 = NO`. Limb 3 is NOT satisfied, and our engine is not the
reason.** The vehicle meets all six conditions #2864 pre-registered, and it
still cannot carry limb 3, because **the oracle is not deterministic on it**.
The pinned vLLM's own eager and compiled configurations disagree with each
other on 2 of the 6 prompts, so there is no single denominator for our tokens
to be exact against.

This is the outcome
[`../../.agents/specs/limb3-vehicle-strict-gate.md`](../../.agents/specs/limb3-vehicle-strict-gate.md)
§3.4 named in advance as a stop condition, and §7 declared a result rather than
a failure. **Picking whichever configuration agrees with us is forbidden**, and
it is forbidden in `score_strict.py`, which evaluates the oracle's
self-consistency first and short-circuits the verdict on it, rather than in
prose that a later reader could step around.

## Determinism, measured on the vehicle rather than inherited

Four conditions were declared before the run. Three hold and the decisive one
fails.

| # | Condition | Result |
|---|---|---|
| 1 | `EAGER1_EQ_EAGER2` | **True** |
| 2 | `COMPILED1_EQ_COMPILED2` | **True** |
| 3 | `EAGER1_EQ_COMPILED1` | **False** -- 2 of 6 prompts |
| 4 | prefill argmax reproduces its own incremental decode | **False** on both -- eager 3 steps, compiled 2 |

Each configuration is perfectly reproducible **against itself**, which is what
makes this a real disagreement rather than noise: repeating a run changes
nothing, and changing the compile path changes the tokens.

### Where the oracle's two configurations part

| prompt | first diff | eager | compiled |
|---|---:|---:|---:|
| 1 `The three primary colors are` | 8 | 1330 | 9238 |
| 3 `The Pythagorean theorem states that` | 41 | 23185 | 16134 |

Both continuations are fluent and correct English; this is a decode-path
difference, not a broken configuration. At prompt 1 the two configurations
diverge into different sentences ("they produce a secondary color" against
"they produce a neutral gray"), and at prompt 3 into "can be verified" against
"can be proven".

### The oracle also disagrees with ITSELF within one configuration

Condition 4 is the check no earlier run in this campaign has taken: feed the
oracle its own (prompt + generated tokens) in one pass and ask whether its
prefill argmax reproduces what its incremental decode emitted.

| configuration | step | prefill argmax says | its own decode said |
|---|---:|---:|---:|
| eager | prompt 2, step 26 | 1651 | 318 |
| eager | prompt 3, step 41 | 16134 | 23185 |
| eager | prompt 4, step 33 | **383** | 303 |
| compiled | prompt 1, step 15 | 13838 | 264 |
| compiled | prompt 2, step 26 | 1651 | 318 |

**At prompt 3 step 41 the oracle holds both answers at once.** Its eager decode
emits `23185`, its eager prefill emits `16134`, and `16134` is exactly what its
compiled decode emits. The disagreement is not between two engines. It is
inside one engine, on one configuration, on the same artifact.

## The strict count, reported plainly and NOT usable as a gate

Free-running greedy decode, `ignore_eos`, batch 1, MTP off, 48 tokens on each
of 6 prompts, 288 steps. Not teacher-forced. No near-tie band: a rank-2 token
is a divergence.

| comparison | token-exact prompts |
|---|---|
| ours vs vLLM eager | **3 / 6** |
| ours vs vLLM compiled | **3 / 6** |
| vLLM eager vs vLLM compiled | 4 / 6 |

**These numbers are recorded, not claimed as a limb-3 result.** A denominator
that disagrees with itself on a third of the set cannot certify anything, and
the two 3/6 figures are not even the same three prompts.

### Every divergence, with what each side actually emitted

| prompt | step | ours | eager | compiled | note |
|---|---:|---:|---:|---:|---|
| 1 `The three primary colors are` | 8 | 9238 | 1330 | **9238** | we match **compiled**; eager is the odd one |
| 2 `Water boils at a temperature of` | 11 | 87840 | 11995 | 11995 | both oracle configurations agree here, and we do not |
| 3 `The Pythagorean theorem states that` | 41 | 23185 | **23185** | 16134 | we match **eager**; compiled is the odd one |
| 4 `In 1969, humans first walked on` | 33 | 383 | 303 | 303 | both decodes agree, but eager's own **prefill argmax emits 383**, our token |

Read the table as a whole. Of the four steps at which anything disagrees with
anything, **three have the oracle disagreeing with itself**, and in two of
those three our token is one of the values the oracle itself produced. Exactly
one step -- prompt 2, step 11 -- is a divergence where both oracle
configurations agree and we differ.

That single step is the only candidate for a genuine our-side defect in these
288, and one step is far too little to call one. It is recorded here as the
thing a future run should look at first, not as a finding.

## What could NOT be established

**Whether any divergence is an exact tie.** `gen_vehicle.py` requests
`prompt_logprobs=1`, so only the winning entry's logprob was ever captured and
the runner-up's was not. The margin at each divergence is therefore unknown,
and no claim about ties -- in either direction -- is made here. Capturing
`prompt_logprobs=2` would settle it and is the cheapest next measurement.

**Whether our engine would pass against a deterministic denominator.** This run
cannot say, and reporting 3/6 as if it could is precisely the error the spec
forbade.

## A seventh condition, owed to the next vehicle search

#2864 pre-registered six conditions on the **artifact**. This run shows they
are necessary and not sufficient, because all six held and limb 3 still could
not be scored. The missing one is a condition on the **oracle**:

> 7. The pinned vLLM must be self-consistent on the vehicle: its eager and
>    compiled configurations must produce identical tokens, and each must
>    reproduce its own one-pass prefill argmax.

It cannot be checked from a header. It costs one lease to measure, and it must
be measured **before** a candidate is fetched-and-scored rather than after,
because it can veto a vehicle that every cheap check admits. #2740 had already
measured the same self-inconsistency on the arm's own artifact, at the same 2
of 6 rate; this run establishes that it is a property of this vLLM build on
this board rather than of any one checkpoint.

Whether that seventh condition can be met at all on `gfx1151` is now the
question limb 3 turns on, and it is not this issue's to answer.

## Board faults, counted rather than hidden

Our side ran 3 independent legs on the shipped default with no knobs: 2 clean,
1 `BOARD_FAULT` (`HW Exception by GPU node-1 ... reason: GPU Hang`, rc 139).
The two clean legs are token-identical to each other. The spec's rule was that
a run with fewer than two clean legs returns `NOT_MEASURED`; two is the floor
and the floor was met. Every leg asserted zero `[vt reference-tier]` hits and a
ROCm device selection, so a leg that had silently fallen to the CPU tier would
have failed rather than scored.

## What this does NOT do

It rescores nothing in #2497, #2534, #2546, #2740, #2809, #2854 or #2864. The
row stays `PARTIAL`, `TOKEN_GATE` stays declared against llama.cpp `b10451` and
stays `FAIL`, no matrix row moves, and no file under `src/`, `include/` or
`tests/` changes. `STRIX_ARM_SPEED_RATIFIED_BY` stays unset and
[`../../.agents/scripts/rocm-strix-ourarm-staged.sh`](../../.agents/scripts/rocm-strix-ourarm-staged.sh)
stays refusing. Limb 2's denominator question stays
[#2534](https://github.com/mudler/vllm.cpp/issues/2534)'s.

## Re-running it

```sh
rc run --device strix:gpu0 -- bash docs/bench-evidence/limb3-strict-gate-20260904/gate.sh
```

Needs the lease, the staged vehicle, and the worker's `/tmp` builds still
present; it asserts all three and refuses rather than rebuilding silently. Its
verbatim output is committed beside it as `job.log`, the verdict as
`strict.txt`, and every token dump as `*.json.txt`.

**The committed `gate.sh` is not byte-identical to the copy that ran, on one
line, and the difference is deliberate.** The committed line is

```sh
GGUF_SHA="${GGUF_SHA:?the staged sha256, measured on the devbox, must be passed in}"
```

and the copy that ran carried the measured digest sealed into it by the
submitting script, which refused to submit at all until it had verified the
staged bytes itself. The committed form **fails closed** for the next reader:
re-running it without `GGUF_SHA` set stops rather than scoring an unverified
artifact. `gen_vehicle.py`, `score_strict.py` and `prompts.txt` are byte-identical
to the copies that ran, verified by `diff` against
`/mnt/nas_share/rc/limb3-2884/`. Set `GGUF_SHA` to the digest in the table above
to reproduce the run exactly.

**The token dumps carry a `.json.txt` suffix rather than `.json` deliberately.**
`scripts/check-pr-size.py`'s `BENCH_EVIDENCE_RUN` admits
`txt|log|gz|sh|cu|py|jsonl|rc` inside a per-run directory and not `json`; a
`.json` file there falls through to the `public_document` class, which these
token dumps are not. The bytes are unmodified JSON. The four `gen-*.log` files
and three `ours_bench_*.log` files are the runs' stdout and stderr under the
same rule.
