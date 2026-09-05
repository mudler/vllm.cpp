# Does the ratified near-tie gate cover the Q4_K_M ROCm arm? NO

Row `QUANT-QWEN38-27B-GGUF-ARM` (`PARTIAL`), issue
[#2854](https://github.com/mudler/vllm.cpp/issues/2854), arm spec
[quantized arms of Qwen3.8-27B](qwen38-27b-quant-arms.md). Determination only.
No product code, no build, no GPU work, no measurement.

**Verdict: NO.** The already-ratified near-tie distributional methodology does
not cover the Qwen3.8-27B Q4_K_M ROCm arm on `strix:gpu0`. Two of its three
limbs are not satisfied. The arm's declared token gate stays `FAIL` and no
speed, latency, memory or ratio figure may be taken from this arm.

This document ratifies nothing, weakens nothing, and rescores nothing. It reads
an existing ratification against existing evidence and reports which limbs hold.

## Scope

**In scope.** One question: do the three limbs of the near-tie methodology,
ratified 2026-07-20 and reaffirmed 2026-07-21, hold for this arm on this board?
Each limb is answered from committed evidence or from a computation over
committed data, with the computation stated so a reader can repeat it.

**Out of scope, and deliberately not done here.** Choosing the oracle
configuration that decides the verdict; that choice belongs to
[#2534](https://github.com/mudler/vllm.cpp/issues/2534). Re-declaring the token
gate; that belongs to [#2546](https://github.com/mudler/vllm.cpp/issues/2546).
Rescoring
[token gate v2](../../docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md).
Any performance figure. Setting `STRIX_ARM_SPEED_RATIFIED_BY`.

## Upstream chain

vLLM is the primary oracle and it now runs on this board. Pinned build
`0.26.0.dev0+g5559679229`, `Qwen3_5ForConditionalGeneration`, `load_format=gguf`,
on `gfx1151`, measured under one `rc` lease with `HSA_OVERRIDE_GFX_VERSION`
unset and asserted absent
([#2788 evidence](../../docs/bench-evidence/oracle-vllm-gfx1151-20260903.md)).
llama.cpp `b10451` is the secondary oracle the declared gate names.

The ratification under test is
[`multimodal-speed.md` §12.2](multimodal-speed.md) and the memory note
`near-tie-distributional-gate`. Its shipped form is
`test_qwen3_paged_engine.cpp`, `kNearTieMnats=500`.

## Our baseline

The arm is the `Qwen3.8-27B-Q4_K_M.gguf` artifact, `gguf_sha256`
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`,
17,106,775,008 B, on the ROCm `gfx1151` backend through the GGUF k-quant
dequantization and GEMM path.

Declared token gate: 6 of 6 free-running greedy prompts, 48 tokens each, against
llama.cpp `b10451`. Recorded reading: `TOKEN_GATE=FAIL` at 3 of 6.

Near-tie band, teacher-forced, published in
[`q4km-neartie-vllm-oracle-20260903.md`](../../docs/bench-evidence/q4km-neartie-vllm-oracle-20260903.md)
and merged at `6a2fefce1`:

| oracle configuration | `n_divergent` | `worst_gap` | over-band | four conjuncts |
|---|---:|---:|---:|---|
| vLLM compiled (`enforce_eager=False`) | 0 of 288 | 0.000000 | 0 | **PASS** |
| vLLM eager (`enforce_eager=True`) | 4 of 288 | 0.250000 | 0 | **FAIL** |

## Port map

Nothing is ported. The three limbs map onto evidence as follows.

| Limb, as ratified | Evidence read | Holds? |
|---|---|---|
| 1. Token-exact stays the bar wherever vLLM is deterministic | #2788 repeat legs | see §Limb 1 |
| 2. Distributional band where vLLM's own greedy is non-deterministic | #2809 band, both configurations | **NO** |
| 3. Also a strict pass on a bigger deterministic model, same forward | model matrix, #915 | **NO** |

## Tests to port

None. This determination adds no test and changes no test. It reports a
computation over two committed token records, and that computation is stated in
§"The counter-argument" so that any reader can repeat it from the tree.

## Limb 1: vLLM's greedy is deterministic here, run to run

**The headline precondition is absent.** The ratification's limb 2 is triggered
by vLLM's *own greedy* being non-deterministic. On this workload it is not.
#2788 ran each configuration twice and the token files are byte-identical:

```text
EAGER1_EQ_EAGER2       = True      (tokens-gguf-eager.json     sha256 350b5fae…)
COMPILED1_EQ_COMPILED2 = True      (tokens-gguf-compiled.json  sha256 034a1e30…)
```

So the K-run distributional gate as first written has nothing to observe. This
is the same situation as the ★ CORRECTION of 2026-07-20, where batched
"non-determinism" resolved into per-request determinism, and the ratification's
own conclusion there was that "strict token-exact IS well-posed per-prompt".

**What is present instead is oracle self-contradiction**, and the ratification
handles that separately. Teacher-forcing each configuration on its own recorded
decode does not return zero:

```text
vLLM compiled, scoring its OWN greedy decode    3 of 288 divergent
vLLM eager,    scoring its OWN greedy decode    6 of 288 divergent
```

That is the named mechanism: a one-shot prefill argmax disagreeing with the same
engine's incremental decode. In the deterministic Qwen3-0.6B/4B regime the
ratification shipped the teacher-forced band for exactly this. So limb 1 does
not by itself stop the work, but it does establish that only the teacher-forced
form is in play, and that it is in play **per configuration**.

**Limb 1 does not stop the determination. It also does not supply the
non-determinism the methodology's headline names.**

## Limb 2: the band passes against one configuration and fails against the other, and nothing chooses between them

Against vLLM compiled the arm reads `n_divergent` 0 of 288, `worst_gap`
0.000000, `over_band_failures` 0, and all four conjuncts PASS. This is the
**strict** case. Our token is the oracle's teacher-forced argmax at every one of
the 288 steps, so the 0.5-nat tolerance does no work at all.

Against vLLM eager the same arm reads 4 of 288 divergent, `worst_gap` 0.250000,
and the four conjuncts **FAIL**. The four divergent steps are `p1/36`, `p1/45`,
`p2/29` and `p3/45`, at ranks 2 and 3, with strictly positive gaps.

**The configuration choice is unjustified, and it decides the verdict.** Stated
plainly, because the task asked for it plainly:

- `AGENTS.md` §Gates forbids `--enforce-eager` as the denominator for a
  **performance** comparison. That rule is about throughput, where eager
  disables the compiled path a production deployment uses.
- §Gates nominates **no** correctness denominator, and no other document does.
- The evidence that measured both configurations declined to choose and assigned
  the choice to #2534. #2534 is open and has not made it.

Selecting `compiled` is therefore selecting, after the fact, the configuration
that passes. That is a new ratification, not the application of an existing one,
and this document has no authority to make it.

**The configuration does not only move the verdict. It moves the near-tie
geometry the methodology reasons about.** Position `p3/45` is the same prompt,
the same step and the same token 25 under both configurations:

| oracle configuration | our token | our logprob | top token | top logprob | gap | counted |
|---|---:|---:|---:|---:|---:|---|
| compiled | 25 | −1.342165708542 | 25 | −1.342165708542 | 0.000000 | not divergent |
| eager | 25 | −1.374806165695 | 393 | −1.249806165695 | 0.125000 | divergent |

Under `compiled` token 25 is in an exact tie with token 393 and no correct token
exists to match. Under `eager` token 393 is ahead by 0.125 nats and 25 is a
rank-2 loss. A compilation flag decides which of those two descriptions is true.
Until #2534 chooses, this arm has two incompatible near-tie readings and neither
binds.

**A second, independent defect: the pass threshold sits below the instrument's
own floor.** The four-conjunct form requires `n_divergent == 0`. The oracle's own
decode scores 3 of 288 and llama.cpp scores 1 of 288, so **both of them fail this
gate**. A gate that the reference implementation does not pass is not measuring
agreement with the reference implementation. It measures agreement with the
reference implementation's *prefill argmax*, which the reference implementation's
own shipped decode does not satisfy.

**Limb 2 is NOT satisfied.**

## Limb 3: the strict pass is on a different forward path

Limb 3 requires a strict token-exact pass on a bigger dense model where vLLM is
deterministic. Its stated rationale is "same forward code across all sizes; the
bigger-model strict pass proves it". The premise is **same forward code**.

The cited passes do not run this arm's forward code, and one of them is not even
this model.

- **`test_qwen27_paged_engine` 235/235 is a different checkpoint.** It gates
  `unsloth/Qwen3.6-27B-NVFP4`, compressed-tensors NVFP4 W4A4
  (`tests/parity/test_qwen27_paged_engine.cpp:3-6`, snapshot pinned at
  `tests/parity/hf_snapshot.h:282-285`). It runs on CUDA GB10 `sm_121a` only,
  throws unless the build carries `VT_CUTLASS_NVFP4` and `VLLM_CPP_TRITON`
  (`:139-163`), and has never been recorded on ROCm. Its GEMM is the CUTLASS
  `sm120a` fp4 tensor-core path. Note also that `235` is the doctest assertion
  count; the token comparison is 16 greedy tokens (`:163-176`).
- **`Qwen/Qwen3.8-27B` bf16 (#915) is CUDA and bf16, and is not a strict pass.**
  Checkpoint `1d4bf0f2`, GB10 `sm_121a`
  (`.agents/specs/qwen38-27b-bf16-gate.md:15`, `:66`, `:255`). It reads 4 of 7
  prompts strict, with the other three adjudicated as exact fp32 ties inside the
  same near-tie band (`.agents/specs/qwen38-27b-bf16-gate.md:258`, recorded on
  the model-matrix row at `.agents/model-matrix.md:96`). A partial strict result
  cannot be the clean strict proof limb 3 asks for. It is itself a near-tie
  adjudication, so limb 3 would be discharged by the same instrument it is
  supposed to independently corroborate.

Neither exercises the GGUF k-quant dequantization or GEMM kernels, and neither
runs on ROCm. The k-quant tier this arm executes compiles only under
`VLLM_CPP_HIP` (`CMakeLists.txt:1709-1736`) and registers against
`DeviceType::kROCM`: `DotQ4K` (`src/vt/rocm/rocm_grouped_gemm.hip:221`),
`KQuantGemmK` (`:446`), the `QuantizeQ8KK` activation quantizer (`:115`), and the
`MatmulBTQuantKernelRocm` entry point (`:830`), reached through
`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`. A CUDA build of the
27B gate does not link any of it, let alone run it.

`AGENTS.md` states the same polarity from the other direction: a token gate
cannot see a dequantization fallback, and a quantized arm needs its own lower
bound.

**Limb 3 is NOT satisfied.** No strict token-exact pass exists for any model on
the ROCm `gfx1151` backend through the GGUF k-quant path. The only recorded
`gfx1151` Q4_K token gate is `FAIL` at 3 of 6
(`docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md:27-29`).
The nearest strict ROCm result is a different board and a different dtype:
gfx1200, Gemma-3-1B-it bf16, 48/48 (`.agents/backend-matrix.md:224`).

## The counter-argument, addressed on mechanism rather than on symptom

The ratification says that chasing free-running token-match past a teacher-forced
pass is "chasing vLLM's own prefill/decode self-inconsistency (unwinnable)". The
task required deciding whether that reasoning genuinely applies here, and the
mechanism must be the same one, not a similar-sounding symptom.

**It does apply, and the landed record understates this.** Free-running, our arm
diverges from vLLM compiled on 5 of 6 prompts. Recomputing the first divergence
per prompt from the two committed streams
(`oracle-vllm-gfx1151-20260903/ours_gen_ids_1.json` and
`tokens-gguf-compiled.json`), and reading the per-step logprobs out of
`q4km-neartie-vllm-oracle-20260903/neartie-compiled.json.gz`, every one of the
five is the named mechanism. The prefix is byte-identical on both sides at each
of these steps, so both teacher-forced runs scored the same distribution:

| position | our token | vLLM decode token | our logprob | vLLM's logprob | mechanism |
|---|---:|---:|---:|---:|---|
| p1/35 | 5844 | 4350 | −1.693231701851 | −1.693231701851 | **exact tie** |
| p2/4 | 29922 | 11995 | −1.004329204559 | −1.004329204559 | **exact tie** |
| p3/45 | 25 | 393 | −1.342165708542 | −1.342165708542 | **exact tie** |
| p4/13 | 19820 | 6165 | −3.175575017929 | −3.238075017929 | prefill argmax vs own decode |
| p5/32 | 16 | 15 | −1.437732696533 | −1.562732696533 | prefill argmax vs own decode |

At the first three the two candidate tokens carry **bit-identical** logprobs, and
vLLM separates them only by an arbitrary tie-break: it assigns them `rank` 2 and
1, then 1 and 2, then 2 and 3. At the last two our token is the oracle's rank-1
prefill argmax and the oracle's own decode took the rank-2 token. In all five,
no fixed decode token exists to match, which is verbatim the condition the
ratification describes.

**This corrects a statement in the landed evidence.** That document says "No
divergence anywhere is an exact tie" and reads it as "the whole difference
between this arm and the Voxtral precedent". The sentence is true of the 17
steps the harness calls divergent, and it cannot be otherwise: a step where our
token is within `argmax_eps` of the top is never counted divergent, so a
two-way exact tie can never appear in `n_exact_tie_mismatch`. That counter is
structurally unable to fire at the ties it is read as excluding. The three exact
ties above are invisible to it. The inference drawn from it does not hold.

**This finding helps the arm, and it does not change the verdict.** The
counter-argument dissolves for the compiled configuration. Limbs 2 and 3 still
fail, for reasons that have nothing to do with free-running divergence: nothing
has chosen the configuration, and no strict pass exists on this forward path.

## Gates

Records only. This change touches no file under `src/`, `include/` or `tests/`,
so it owes no build and no model gate. The gates are:

```sh
scripts/agent-preflight.sh                 # read the printed failures, not the exit code
python3 scripts/check-agent-record.py
python3 scripts/check-commit-trailers.py
python3 scripts/check-commit-style.py
python3 scripts/check-pr-size.py --base <base> --head <head>
python3 scripts/agent-pr-body.py --pr <N>
```

The determination's own factual claims are gated differently, by being
recomputable: the table in §"The counter-argument" is derived from committed
JSON by the steps named there, and a reader who repeats them either reproduces
the six logprob values or falsifies this document.

## Dependencies

- #2534 owns the correctness-denominator choice and the arm's token gate. This
  determination is blocked on it and does not pre-empt it.
- #2546 owns re-declaring the gate's denominator.
- #2497 owns the retake and already carries one retraction for a measurement
  taken ahead of this precondition.
- #2511 (the `gfx1151` hang) is no longer the blocker it was; the board now
  completes a generation.

## Work breakdown

- **W0 (this change).** The determination, its reasoning, its limits, and the
  recomputation that corrects the exact-tie characterization. Records only.
- **W1 (not claimed here).** #2534 chooses the correctness denominator, or
  refuses to, on its own evidence.
- **W2 (not claimed here).** A strict token-exact pass on the ROCm `gfx1151`
  GGUF k-quant path, which is what limb 3 actually asks for and what nothing in
  the tree supplies.

## Risks/decisions

- **The risk this document exists to manage** is a performance figure taken from
  an arm whose gate has not passed. #2497 has already retracted one. The staged
  script `.agents/scripts/rocm-strix-ourarm-staged.sh` stays refusing at rc 3,
  and `STRIX_ARM_SPEED_RATIFIED_BY` stays unset.
- **The guard's regex is weak and that is not the point.** Its `#[0-9]+` pattern
  is satisfiable by any string carrying an issue number (undetected mutation
  M15). The condition it stands for is the declared token gate passing. That
  condition is not met, so the variable is not set. Satisfying the regex would
  be a bypass.
- **Decided: the exact-tie finding is reported even though it favours the arm.**
  Reporting only the findings that support the verdict would make this document
  an argument rather than a determination.
- **Decided: no limb was stretched to reach a benchmark.** Three retractions on
  this arm already came from wanting the result.

## Now

Determination written and landed as its own pull request. Verdict NO. The row
`QUANT-QWEN38-27B-GGUF-ARM` stays `PARTIAL`, the declared token gate stays
`FAIL`, and no benchmark ran.

## Owed

- O1. The correctness-denominator choice between vLLM compiled and vLLM eager,
  owned by [#2534](https://github.com/mudler/vllm.cpp/issues/2534). Until it is
  made, the near-tie band yields two verdicts for this arm and neither binds.
- O2. A strict token-exact pass on the ROCm `gfx1151` GGUF k-quant forward path,
  which limb 3 requires and which no record supplies. Owned by
  [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
- O3. The published sentence "No divergence anywhere is an exact tie" in
  `docs/bench-evidence/q4km-neartie-vllm-oracle-20260903.md`, and the harness's
  `n_exact_tie_mismatch` counter that cannot fire at a two-way tie. Corrected in
  prose here; the counter itself is unchanged and still cannot report a tie.
  Owned by [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
