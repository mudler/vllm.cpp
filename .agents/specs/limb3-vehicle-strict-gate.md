# Limb 3 has a vehicle now: pin it, fetch it, and run the strict gate

Row: `QUANT-QWEN38-27B-GGUF-ARM`
Issue: [#2884](https://github.com/mudler/vllm.cpp/issues/2884)
Refs: [#2864](https://github.com/mudler/vllm.cpp/issues/2864),
[#2854](https://github.com/mudler/vllm.cpp/issues/2854),
[#2740](https://github.com/mudler/vllm.cpp/issues/2740),
[#2497](https://github.com/mudler/vllm.cpp/issues/2497),
[#2534](https://github.com/mudler/vllm.cpp/issues/2534)

## 0. Now

**The gate has run and the answer is `STRICT_LIMB3 = NO`. Limb 3 is NOT
satisfied.** The vehicle was found, pinned, fetched, verified and driven, and
it meets all six conditions #2864 pre-registered; the gate still cannot be
scored, because **the pinned vLLM is not deterministic on it**. Its eager and
compiled configurations each reproduce themselves exactly and disagree with
each other on 2 of 6 prompts. §3.4 named that in advance as a stop condition
and §7 declared it a result rather than a failure of this spec.

The row stays `PARTIAL` and no matrix row moves. `STRIX_ARM_SPEED_RATIFIED_BY`
stays unset and
[`../scripts/rocm-strix-ourarm-staged.sh`](../scripts/rocm-strix-ourarm-staged.sh)
stays refusing, because limb 2's denominator question is
[#2534](https://github.com/mudler/vllm.cpp/issues/2534)'s and not this spec's.
Evidence: [`../../docs/bench-evidence/limb3-vehicle-pin-20260904.md`](../../docs/bench-evidence/limb3-vehicle-pin-20260904.md)
for the pin and
[`../../docs/bench-evidence/limb3-strict-gate-20260904.md`](../../docs/bench-evidence/limb3-strict-gate-20260904.md)
for the gate.

## 1. Scope

[`q4km-limb3-kquant-vehicle.md`](q4km-limb3-kquant-vehicle.md) (#2864, landed
`465cd27e3`) pre-registered six conditions, read all 77 GGUFs this fleet holds
header-first, and found the intersection **empty**. Its §7 named three ways a
vehicle could come to exist and closed option 2 with:

> in practice this means a second dense `qwen35` checkpoint that is not
> Qwen3.8-27B. Whether one exists **was not established here, because fetching
> it needs recorded authority and none was given.**

The authority is now recorded in
`.agents/developer-preferences.md` (developer,
2026-09-04), scoped to ONE vehicle meeting exactly those six conditions. This
spec spends it.

**In scope:** identify and pin a candidate, verify it against #2864's own
predicate before fetching it, stage it, and score one STRICT free-running
token-exact gate of our ROCm k-quant path against the pinned vLLM on
`strix:gpu0`.

**Out of scope, and measured to be out of scope.** No throughput, latency or
memory figure for either engine, and no cross-engine ratio. `AGENTS.md` §Gates
admits no performance result from an arm whose declared token gate has not
passed, and #2497 already carries one retraction for exactly that. This spec
rescores nothing in #2497, #2534, #2546, #2740, #2809, #2854 or #2864, and it
changes no file under `src/`, `include/` or `tests/`.

## 2. The candidate, and why it is the only one

The six conditions, unchanged from #2864 §2, are: k-quant tensors; an
architecture in `kGgufArchArms`; a family the **pinned** vLLM registers; dense;
fits the board's ~62.8 GB free carve; and not the arm's own model.

`kGgufArchArms` dispatches eight architectures. The pinned vLLM registers five
of them, and four of those five are MoE in every checkpoint the ecosystem
publishes, so condition 4 leaves `qwen35`. `vllm-gguf-plugin`'s
`_ADAPTER_REGISTRY` is `Gemma3`, `Gemma4`, `OLMoE`, `Qwen35`, `Qwen35Mtp`, and
this tree has no Gemma or OLMoE GGUF arm, which narrows it the same way from
the other side. The search space is therefore **dense `qwen35` checkpoints that
are not Qwen3.8-27B**, and the family publishes exactly two: `Qwen3.5-27B` and
`Qwen3.6-27B`.

`Qwen3.6-27B` is chosen over `Qwen3.5-27B` because its `text_config` key set is
**identical** to the arm's, `output_gate_type`, `partial_rotary_factor` and
`tie_word_embeddings` included, while `Qwen3.5-27B` carries none of the three.
The closer the config surface, the more of the same forward code both sides
execute, which is the whole point of limb 3.

**The pin is a revision, not a repo id.** #2497 refused the UD family because
`Qwen3.8-27B-UD-Q4_K_XL`'s published bytes moved in place under an unchanged
name. The plain `Q4_K_M` arm is taken rather than a `UD-*` one for the same
reason and because it is the quant tier the arm itself runs.

## 3. Method

### 3.1 Before the download

[`../../docs/bench-evidence/limb3-vehicle-pin-20260904/`](../../docs/bench-evidence/limb3-vehicle-pin-20260904/).
`vehicle_pin_check.sh` re-runs the whole determination.
`remote_gguf_header.py` reads the candidate's own header **by HTTP range
request**, and it drives #2864's committed `gguf_header.py` rather than
re-implementing the parse, so the two searches cannot disagree about what a
header says.

The four-surface oracle check of #2864 §5 is re-run for the candidate. There
the expected answer was negative and the control was positive; here the
expected answer is positive, so the control is the **negative** one:
`muse-glimmer` is re-probed on the identical four surfaces and must still read
0. A grep that matched everything would light both rows and be visible.

### 3.2 The gate

[`../../docs/bench-evidence/limb3-strict-gate-20260904/`](../../docs/bench-evidence/limb3-strict-gate-20260904/).
`gate.sh` is the `rc` job, `gen_vehicle.py` is the oracle's side and
`score_strict.py` is the verdict. One lease on `strix:gpu0`, nothing by `ssh`,
`HSA_OVERRIDE_GFX_VERSION` set nowhere, and the job refuses to start if it
inherited any `HSA_*`, `ROCR_*`, `PYTORCH_*`, `HIP_*` or `VT_*` variable.

**Both sides are REUSED, and each is asserted to be the object it claims.** The
`strix:gpu0` worker has not rebooted since 2026-09-01 (`boot_id`
`a5bc8128-f6ad-4767-8614-6923f88032e1`), so `/tmp` still carries the
token-gate-v2 build and the #2740 vLLM venv. `gate.sh` asserts our three build
products against the sha256 values committed in
[`../../docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md)
and asserts the venv's `vllm.__version__`, its device, its `RocmPlatform`
resolution and its plugin extension before it generates anything. A rebuild
would fail those assertions rather than pass silently.

### 3.3 What "strict" means, declared before the run

Free-running greedy decode, `ignore_eos`, batch 1, MTP off, **48 tokens on each
of the six pre-registered prompts**. Every token of every prompt, 288 steps.
Not teacher-forced, and no near-tie band: a rank-2 token is a divergence here.

**The prompts are pre-registered in the strongest sense available.** They are
not chosen by this spec. They are the six the declared Q4_K_M gate has scored
since 2026-08-23, `prompts_sha256`
`c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e`, a value
committed in four evidence documents that predate this run. `gate.sh` hashes
the file and refuses on a mismatch before it scores anything, and
`score_strict.py` refuses again independently.

The prompt **token ids** are produced by our engine from the vehicle's own GGUF
vocab and FED to vLLM, and vLLM's own tokenizer re-derives them and prints
whether it agrees. #2740 established the pattern: a generation comparison must
not be contaminated by a tokenizer difference.

### 3.4 The oracle's determinism is a PRECONDITION, not a footnote

#2740 measured, on the arm's own artifact, that the pinned vLLM's eager and
compiled configurations are each self-reproducible and **disagree with each
other on 2 of 6 prompts**. A vehicle inherits nothing from that measurement, so
it is re-measured here, and three conditions must all hold:

1. `EAGER1_EQ_EAGER2`
2. `COMPILED1_EQ_COMPILED2`
3. `EAGER1_EQ_COMPILED1`

plus a fourth that no earlier run in this campaign has taken: the oracle's
one-pass **prefill argmax** over (prompt + its own generated tokens) must
reproduce the tokens its incremental decode emitted.

If (3) fails, the answer is `STRICT_LIMB3=NO` with the reason "the oracle is not
deterministic on this vehicle". **Picking whichever configuration agrees with
us is forbidden**, and it is forbidden in the scoring script rather than in
prose: `score_strict.py` evaluates the oracle's self-consistency first and
short-circuits the verdict on it.

## 4. Risks

**The reused binaries could be the wrong object.** Mitigated by asserting three
sha256 values that were committed by a different run, on a branch this job
cannot edit.

**A dead instrument reads as a result.** Every leg is counted into one of three
outcomes -- OK, board fault, harness error -- and a run with fewer than two
clean legs returns `NOT_MEASURED` rather than a verdict. Our legs additionally
assert zero `[vt reference-tier]` hits and at least one `device=5` selection, so
a leg that silently ran the CPU tier under a ROCm label fails instead of
scoring.

**The board faulted 17 times in 18 legs on this workload before `27da7787e`.**
The `LEGS` budget is a scheduling parameter and not a verdict: an all-fault run
is `NOT_MEASURED`, which is a complete answer.

**The vehicle is not BIGGER than the arm.** The ratification's wording is "a
BIGGER dense model"; `Qwen3.6-27B` is 64 blocks against the arm's 65 (the arm's
65th is its `nextn` drafter) at identical width. No dense `qwen35` checkpoint
larger than 27B exists, so the strongest available vehicle is a same-class
sibling. **This is recorded as a shortfall rather than argued away**: a pass
here satisfies the six conditions #2864 pre-registered and does not by itself
satisfy the word "bigger", and whether that is enough is a ratification
decision this spec does not make.

## 5. Tests

None. No product code changes, so there is nothing a test could reach. The
re-runnable artifacts are `vehicle_pin_check.sh` and `gate.sh`, whose outputs
are committed verbatim beside them.

## 6. Gates

Run by name on this head, each exit code read from the process:

- `scripts/check-agent-record.py`
- `scripts/check-commit-style.py --range origin/main..HEAD`
- `scripts/check-commit-trailers.py --range origin/main..HEAD`
- `scripts/check-pr-size.py --base origin/main --head HEAD`, which
  `agent-preflight.sh` skips because it supplies no `--base`/`--head`

## 7. Stop conditions

- No candidate passes §2 before the download: stop and report. Do not fetch
  something that fails a condition in order to have fetched something.
- The oracle is not self-consistent on the vehicle: report that, and do not
  score against the half that agrees.
- Fewer than two clean legs on either side: `NOT_MEASURED`.
- A divergence is a complete answer. **`STRICT_LIMB3=NO` is a result, not a
  failure of this spec**, and it is more useful than a stretched pass.

## 8. Owed

- **A SEVENTH condition on any future limb-3 vehicle, this time on the ORACLE
  rather than on the artifact.** #2864's six conditions all held here and limb 3
  still could not be scored, so they are necessary and not sufficient. The
  missing one is: *the pinned vLLM must be self-consistent on the vehicle -- its
  eager and compiled configurations must produce identical tokens, and each must
  reproduce its own one-pass prefill argmax.* It cannot be read from a header,
  it costs one lease, and it must be measured before a candidate is scored
  because it can veto a vehicle every cheap check admits. Owned by this row and
  tracked under [#2884](https://github.com/mudler/vllm.cpp/issues/2884).

- **Whether the divergences are exact ties is NOT ESTABLISHED**, and it is
  cheap to settle. `gen_vehicle.py` requests `prompt_logprobs=1`, so only the
  winning entry's logprob was captured and no runner-up margin exists. Raising
  it to 2 answers it in one lease. Owned by this row.

- **Whether the pinned vLLM can be made self-consistent on `gfx1151` at all**
  is now the question limb 3 turns on. #2740 measured the same 2-of-6
  self-inconsistency on the arm's own artifact, so it is a property of this
  build on this board rather than of any one checkpoint, and no vehicle search
  can route around it. It is larger than this row and belongs to
  [#2534](https://github.com/mudler/vllm.cpp/issues/2534)'s owner to schedule.

## Outcome

**What was measured.** One `rc` job on `strix:gpu0`
(`0d0e57c3-eaad-4b1a-9d13-3cc586814d0c`, worker `rc-worker-lcjhd`,
2026-09-04T07:50Z to 08:36Z), nothing by `ssh`, `HSA_OVERRIDE_GFX_VERSION`
unset and no inherited `HSA_*`/`ROCR_*`/`PYTORCH_*`/`HIP_*`/`VT_*`. Six
pre-registered prompts, `prompts_sha256`
`c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e`, 48 tokens
each, 288 free-running greedy steps, not teacher-forced.

| | |
|---|---|
| `EAGER1_EQ_EAGER2` | True |
| `COMPILED1_EQ_COMPILED2` | True |
| `EAGER1_EQ_COMPILED1` | **False, 2 of 6** |
| prefill argmax reproduces own decode | **False**, eager 3 steps, compiled 2 |
| ours vs vLLM eager | 3/6 token-exact |
| ours vs vLLM compiled | 3/6 token-exact |
| **verdict** | **`STRICT_LIMB3 = NO`** |

**What was rejected, and why.** Scoring against whichever oracle configuration
agrees with us was rejected, and it was rejected in `score_strict.py` rather
than in prose: the script evaluates the oracle's self-consistency first and
short-circuits the verdict on it, so a later reader cannot step around it. The
temptation was real and is visible in the data -- our token matches *compiled*
at prompt 1 step 8 and *eager* at prompt 3 step 41, so either half could have
been written up as a better result than the other.

**Why each default has its value.** The vehicle is `Qwen3.6-27B` rather than
`Qwen3.5-27B` because the histogram decided it, not the name: both files carry
the `Q4_K_M` label and the Qwen3.5 one routes 96 tensors through `Q8_0`, a
different kernel from the three k-quant kernels limb 3 exists to exercise. The
pin is a revision rather than a repo id because #2497 was bitten by bytes moving
in place under an unchanged name. The prompts are the campaign's existing six
rather than a fresh set, because a set chosen after seeing which prompts agree
is not a denominator. Both sides' binaries are reused from the unrebooted
worker's `/tmp` and asserted by sha256 against digests a different run committed,
so a rebuild fails the assertion instead of passing silently.

**The most useful thing this run produced is not the verdict.** It is the
prefill-argmax check in §3.4 condition 4, which no earlier run in this campaign
had taken. It shows the oracle disagreeing with itself *inside one
configuration*: at prompt 3 step 41 eager's decode emits `23185`, eager's own
prefill emits `16134`, and `16134` is what compiled's decode emits. The
disagreement is not between two engines. That reframes limb 3's blocker from
"find a better vehicle" to "the denominator on this board is not self-consistent
yet", which is what §8 now carries.

**The shortfall §4 declared is unchanged and unresolved.** `Qwen3.6-27B` is not
BIGGER than the arm; no dense `qwen35` checkpoint larger than 27B is published.
That question never became live, because the run stopped on determinism first.
