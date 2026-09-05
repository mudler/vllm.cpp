# Spec — score the ratified near-tie band with the PRIMARY oracle

Row `QUANT-Q4K-NEARTIE-VLLM-ORACLE`. Issue
[#2809](https://github.com/mudler/vllm.cpp/issues/2809).

Sibling records: [#2497](https://github.com/mudler/vllm.cpp/issues/2497) (the
gfx1151 decode campaign, which already retracted one performance figure),
[#2534](https://github.com/mudler/vllm.cpp/issues/2534) (the ratification
decision this measurement feeds),
[#2546](https://github.com/mudler/vllm.cpp/issues/2546) (the gfx1151 token gate
that read `FAIL` at 3 of 6), and
[#2740](https://github.com/mudler/vllm.cpp/issues/2740) / #2788 (the measurement
that made the primary oracle reachable on this device).

## Now

`DONE`. The rule below was committed before the job finished; the score is in
[`../../docs/bench-evidence/q4km-neartie-vllm-oracle-20260903.md`](../../docs/bench-evidence/q4km-neartie-vllm-oracle-20260903.md).
Four of the six pre-registered outcomes fired: P1 under compiled, P2 under
eager, P4, and P5 in both configurations. See `## Outcome`.

## 1. The question

[`qwen38-27b-q4km-neartie-band-adjudication.md`](qwen38-27b-q4km-neartie-band-adjudication.md)
scored the four conjuncts of the ratified near-tie band on this arm and read
`DIST_GATE=FAIL` at `n_divergent = 3` of 288 on the ROCm tier. It scored them
against **llama.cpp `b10451`**, a secondary oracle, and it says in §7 why it had
no choice:

> **No harness exists either.** The ratified instrument reads vLLM
> `prompt_logprobs`.

#2788 removed that constraint. The pinned vLLM `5559679229` builds and runs on
`strix:gpu0`, loads the gated `Qwen3.8-27B-Q4_K_M.gguf` through
`vllm-gguf-plugin`, and generates 48 greedy tokens for all six gate prompts,
twice per configuration. So the ratified instrument can now be pointed at the
oracle it was written for.

**The question is narrow: what do the four §12.2 conjuncts read when the oracle
is vLLM rather than llama.cpp?** Nothing here decides whether the band may be
applied to this arm at all. §8 of the adjudication spec records that the decision
is the maintainer's, and this spec does not take it.

## 2. Scope

**In:** teacher-forced vLLM `prompt_logprobs` over `prompt_ids + our arm's 48
emitted tokens`, for all six gate prompts, scored against **both** of the
oracle's supported configurations. Per step: the oracle's top-1 token and
logprob, our token's logprob under the oracle, our token's rank, and the gap.

**Out, and each for a stated reason:**

- Any throughput, latency or memory figure for vllm.cpp, and any cross-engine
  ratio. `AGENTS.md` §Gates admits no performance result from an arm whose
  declared token gate has not passed, and this arm's reads `FAIL`. #2497 already
  carries one retraction for exactly this.
- Any ratification, and any change to #2497, #2534, #2546 or #2740's verdicts.
- Any change to a gate's declared oracle. The arm's declared token gate stays
  llama.cpp `b10451` and stays `FAIL`.
- `src/` and `include/`.

## 3. The rule, fixed before the numbers exist

This section is committed while the job is still building, so a later reading
does not get to choose a rule after seeing a number. That is the same discipline
§3 of the adjudication spec applied.

### 3.1 The four conjuncts, and their definitions

Taken from [`multimodal-speed.md`](multimodal-speed.md) §12.2 and implemented by
[`../../scripts/mm/a3_voxtral_neartie_gate.py`](../../scripts/mm/a3_voxtral_neartie_gate.py),
whose definitions this harness reuses rather than restating:

```text
result == PASS  &&  n_divergent == 0  &&  over_band_failures == 0  &&  worst_gap <= 0.5
```

- `gap = top_lp - our_lp`, where `top_lp` is the maximum logprob in the returned
  distribution at that position and `our_lp` is our token's logprob there. Both
  are natural logs.
- A step is **divergent** when `gap > 1e-9`. That is the harness's
  `is_argmax = gap <= 1e-9`, negated. **An exact tie is therefore not
  divergent**, which is precisely why the Voxtral precedent passed this band at
  `worst_gap 0.0000`.
- `over_band_failures` are the divergent steps with `gap > 0.5`.
- `result` is the band-only field: `PASS` when `over_band_failures` is empty. It
  is one conjunct of four and is never the verdict on its own.

### 3.2 The two counts, and which one is the gate's

The adjudication spec §3.4 separates them and so does this one:

- **`n_divergent`** — steps where `gap > 1e-9`. **This is the gate's count.**
- **`n_token_mismatch`** — steps where our token id is not the oracle's argmax
  token id. Of these, the ones with `gap <= 1e-9` are **exact ties** and are
  reported separately, because whether this arm's divergences are exact ties is
  the whole difference between its result and Voxtral's.

### 3.3 The denominator, fixed in advance

- Six prompts, 48 steps each, 288 steps per configuration.
- `prompts_sha256 = c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e`,
  asserted in-process from the prompt strings.
- The oracle is fed the llama.cpp oracle's own `PROMPT_IDS`, exactly as #2788
  did, so a tokenizer difference cannot contaminate the score.
- Our arm's stream is the **recorded** `ours_gen_ids_1.json` committed under
  `docs/bench-evidence/oracle-vllm-gfx1151-20260903/`, sha256
  `8b542c718fd38721d5dd3286a77c91ed30ab495b0c604783b9a2681fcc1ad107`. It is not
  re-generated, so nothing about our engine can move under the measurement.
- `K = 20`. The harness asserts our token is present in the returned dict at
  every step, so `top_lp` is the true argmax logprob and not a top-K artefact.

### 3.4 Both configurations, scored separately

#2788 measured that the oracle's own greedy decode differs between `eager` and
`compiled` on 2 of 6 prompts, reproducibly, with both legs byte-identical to
their repeats. **They are scored separately and neither is averaged into the
other.** Which one a ratification would adopt is the open question, and a spec
that picked one would be answering it.

### 3.5 The controls

- **Self-consistency (positive).** Teacher-force each configuration on **its
  own** recorded greedy tokens. This reads the instrument's own floor: the
  teacher-forced distribution comes from a prefill over the whole sequence, and
  the greedy tokens came from step-by-step decode, so the two need not agree
  bit-for-bit. If this control is not 0, `n_divergent == 0` is unattainable for
  anyone on this device and the arm's number must be read against that floor.
- **llama.cpp `b10451` (context).** Teacher-force the secondary oracle's own
  stream under the primary oracle, from the committed `oracle_hip.txt`. This says
  what the same band reads for the engine that currently convicts our arm.
- **Corrupted stream (negative).** One recorded step replaced by a token the
  oracle cannot prefer. It must register a divergence with a gap above 1 nat.
  The harness asserts this and refuses to emit `DONE_MARKER` without it, because
  an instrument that cannot fail cannot pass.

### 3.6 The outcomes, enumerated before the run

| # | condition | reading |
|---|---|---|
| P1 | all four conjuncts hold under a configuration | the band would `PASS` under that configuration; it still needs the maintainer's ratification before it may be quoted |
| P2 | `n_divergent > 0`, every gap inside the band | `FAIL` on the binding limb; report the band field separately and never as the verdict |
| P3 | `over_band_failures > 0` or `worst_gap > 0.5` | `FAIL`, forward divergence |
| P4 | the two configurations disagree on the verdict | report both; the choice of configuration is the maintainer's and this row does not make it |
| P5 | the self-consistency control is not 0 | no verdict is reported without the floor beside it |
| P6 | the negative control does not discriminate | `INSTRUMENT_FAILURE`; no number is reported at all |

## 4. Design

No product code. One harness,
[`../../docs/bench-evidence/q4km-neartie-vllm-oracle-20260903/neartie_vllm.py`](../../docs/bench-evidence/q4km-neartie-vllm-oracle-20260903/neartie_vllm.py),
and one job script beside it. The build recipe is #2788's, reused in shape and
not re-derived: every apt package in it exists because its absence once read as
"vLLM does not run on gfx1151".

## 5. Risks

- **A band figure gets quoted alone.** It reverses the verdict. §3.1 and §3.2
  exist for this, and the evidence document states the four conjuncts beside
  their results every time.
- **The two configurations get conflated.** §3.4.
- **`HSA_OVERRIDE_GFX_VERSION` leaks in.** The job refuses to start with it set,
  and the harness asserts it again in-process before it imports torch.
- **The prefill-versus-decode difference gets read as an engine defect.** §3.5's
  self-consistency control measures it directly.

## 6. Tests

None. No product code changes and no gate changes. Every figure in the evidence
document names the committed artifact and the `rc` job id it came from.

## 7. Gates

`TOKEN_GATE` for this arm is unchanged and stays `FAIL`. This spec adds no gate,
relaxes none, and changes no gate's declared oracle.

## 8. Evidence

`docs/bench-evidence/q4km-neartie-vllm-oracle-20260903.md` and the directory
beside it: the harness, the job script, the verbatim `rc` job log, and the
per-step JSON for both configurations.

## 9. Stop conditions

- Do not widen, weaken or delete the declared token gate.
- Do not report a band result as the distributional gate's verdict.
- Do not take a performance figure from this arm.
- Do not ratify anything, and do not recommend a ratification.

## 10. Owed

- The ratification decision under
  [#2534](https://github.com/mudler/vllm.cpp/issues/2534). This row adds an input
  to it and takes it no further.
- The bigger-model strict limb of the near-tie doctrine, still without a vehicle;
  §7 of the adjudication spec records the search and its three disqualifications.

## Outcome

`rc` job `e1afb349-98c4-4e8b-a684-26fdeaa4ba24`, `strix:gpu0`, 2026-09-03.

### What was measured

| oracle configuration | our arm | the oracle's own decode | llama.cpp `b10451` |
|---|---|---|---|
| compiled | **0 of 288, worst 0.000000, FOUR CONJUNCTS PASS** | 3 of 288 | 1 of 288 |
| eager | 4 of 288, worst 0.250000, FOUR CONJUNCTS FAIL | 6 of 288 | 3 of 288 |

No divergence anywhere is an exact tie. The negative control failed at 21.24
nats in both configurations, so the instrument discriminates.

### What was rejected, and why

**A headline of "token-exact against the primary oracle" was rejected as false.**
The band is teacher-forced: the oracle is fed our own prefix at every step, so
the sequences cannot drift. The declared token gate compares free-running
streams. Under a free-running comparison our arm diverges from vLLM compiled on
**5 of 6** prompts, worse than the 3 of 6 it reads against llama.cpp
(`score-compiled.md`, `VLLMCPP_vs_VLLM_DIVERGENCES=5/6`, re-derived
independently). Substituting the primary oracle into the declared gate makes
this arm's reading worse. The evidence document states this before it can be
inferred.

**Reading compiled as "the configuration the protocol mandates" was rejected.**
`AGENTS.md` §Gates forbids `--enforce-eager` as the denominator for a
**performance** comparison, where eager disables the path a production
deployment uses. It does not nominate a correctness denominator, and no
ratification has. Treating it as one would have converted a maintainer's open
decision (#2534) into an inference, and it would have picked the configuration
under which the number is favourable. Both are reported; neither is averaged.

**A verdict without the self-consistency floor was rejected.** P5 fired. The
floor is 3 of 288 under compiled and 6 of 288 under eager, so our arm's 0 sits
*below* the floor of the instrument scoring it. Our stream agrees with the
compiled prefill argmax at three near-tie positions where the oracle's own
decode does not. A threshold below its instrument's floor can be met by luck,
and that is the strongest limit on the PASS.

### Why each default has its value

- **`K = 20`** because the harness asserts our token is present in the returned
  dict at every step, so `top_lp` is the true argmax and never a top-K artefact.
  The assertion, not the value, is what carries this.
- **`gap > 1e-9` for divergence**, taken from
  `scripts/mm/a3_voxtral_neartie_gate.py` rather than restated, because an exact
  tie must not count and that is why the Voxtral precedent passed at
  `worst_gap 0.0000`.
- **The oracle's own `PROMPT_IDS`, not a re-tokenization**, so a tokenizer
  difference cannot contaminate the score.
- **The recorded `ours_gen_ids_1.json`, never re-generated**, so nothing about
  our engine could move under the measurement.
- **Both configurations, scored separately**, because #2788 measured that they
  disagree on 2 of 6 prompts and each is byte-identical to its own repeat.

### Records this falsified

`.agents/backend-matrix.md`'s `BACKEND-GATE-ROCM-LLAMACPP` row read "vLLM has no
entry on that architecture" and called llama.cpp "the ONLY comparator that runs
on `gfx1151` at all". #2788 falsified that and did not correct the row; this
change does. The row's disposition is unchanged: it still has one side of one
comparison, still has no ratio, and stays `INVENTORIED`.

### What is still owed

- The ratification decision under
  [#2534](https://github.com/mudler/vllm.cpp/issues/2534). This row adds an
  input and takes it no further.
- The bigger-model strict limb of the near-tie doctrine, still without a
  vehicle.
- A free-running token gate against the primary oracle on this device, if anyone
  wants one. The figure exists (5 of 6) but no spec declares that gate, and
  changing this arm's declared oracle is out of scope here.
