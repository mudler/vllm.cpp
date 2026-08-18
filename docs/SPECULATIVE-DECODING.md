# Speculative decoding

A draft proposes tokens, the full model verifies them in the same step, and an
accepted proposal skips a full model step. It is **opt-in and off by default**;
with no speculative config the engine is byte-identical to the non-speculative
one.

Selection mirrors vLLM's own CLI: `--speculative-config '<json>'`, taking the
same JSON object vLLM takes, so a config written for vLLM works here.

## Methods

The engine accepts five `method` strings. Anything else is refused at load with
the list of accepted ones (`src/vllm/config/speculative.cpp`).

| `method` | Draft | k | State |
|---|---|---|---|
| `mtp` | the target's own `mtp.*` head | 1 | Gated, and the default recommendation. Token-exact against vLLM at concurrency 1, ~1.04x its speculative-on decode |
| `dflash` | a separate z-lab block-diffusion checkpoint | block size, e.g. 16 | Gated. 2.9x over speculative-off at concurrency 1, at or above vLLM's own DFlash-on |
| `dspark` | a separate DSpark checkpoint | block size, at least the draft's own | Runs end to end, **not gated**. Token-identical to speculative-off on the 35B. Speculation works (drafts proposed and accepted), but the cross-engine ratio is UNSETTLED: the last matched-and-warm paired measurement is 0.834x of the pinned oracle. No speed win is claimed |
| `ngram` | none, drafts come from the prompt's own suffix | required, no default | Accepted and wired; no published measurement |
| `draft_model` | a separate full model | from the draft | **Config only.** The JSON parses, but the engine has no branch for it and refuses the load |

MTP and DFlash are the two with binding numbers behind them; the per-method
detail below and in [BENCHMARKS.md](BENCHMARKS.md) says which is which.

## Which keys the JSON accepts

The object is checked key by key and **nothing is dropped**. A name this engine
cannot honour is refused at startup, by name, before a byte of weights is read.
Until [#1160](https://github.com/mudler/vllm.cpp/issues/1160) an unrecognised key
was silently discarded, so `"draft_sample_method":"probabilistic"` started a
server that drafted greedily and reported nothing, and a typo such as
`"num_speculatve_tokens"` quietly took the resolved default instead of the value
that was typed.

| Key | Accepted |
|---|---|
| `method` | `mtp`, `dflash`, `dspark`, `ngram`, `draft_model` |
| `num_speculative_tokens` | a positive integer, or absent for the method's own default |
| `model` | the draft checkpoint path or HF repo id |
| `prompt_lookup_min`, `prompt_lookup_max` | an integer of at least 1, `ngram` only |
| `draft_sample_method` | `greedy` only, which is upstream's default and what this engine does |
| `rejection_sample_method` | `standard` only, which is upstream's default and what this engine does |

There are two kinds of refusal, worded differently on purpose. A name vLLM's own
`SpeculativeConfig` declares, such as `quantization` or `max_model_len`, is
reported as a real vLLM field this engine does not implement. Any other name is
reported as unknown, together with the list above, because that one is usually a
typo. `draft_sample_method: probabilistic` and the `synthetic` and `block`
acceptance variants name row `SPEC-ACCEPT-VARIANTS`, which owes them.

The distinction matters beyond ergonomics. Draft sampling and verify are greedy
here, so a dropped `probabilistic` produced a **deterministic** run when a sampled
draft was asked for, and a deterministic run is adjudicable by the token-exact
greedy gate while the configuration the user actually requested is not. A
silently downgraded flag therefore lets a parity or benchmark number be taken
under a configuration nobody chose.

## MTP

- **Models:** the Qwen3.5 / 3.6 gate checkpoints that ship an `mtp.*` draft head
  in their safetensors (Qwen3.6-27B and Qwen3.6-35B-A3B). Both are GDN hybrids,
  and the speculative path is wired through the linear-attention (GDN) recurrence
  and short causal convolution as well as the attention layers.
- **Depth is configurable.** `num_speculative_tokens` sets how many tokens the
  MTP head drafts per step, by looping the single head autoregressively. It
  defaults to the checkpoint's `mtp_num_hidden_layers`, which is 1 on both gate
  checkpoints, so the default is unchanged. A value above `n_predict` must be a
  multiple of it, mirroring vLLM. Depth is a pure throughput lever: greedy
  decoding plus accept-if-equal rejection makes the emitted tokens identical at
  every k, which is proven on CPU for k=1..4 against speculative-off. That
  identity is exactly why a token gate cannot see a clamped depth, so each CPU arm
  asserts TWO positive witnesses instead, because neither one alone is enough. The
  DRAFT DECODE FORWARDS the propose ran, `k-1` per propose call, catch a propose
  that never entered the loop. They do NOT catch padding: a propose that runs
  every forward and then overwrites all k columns with its step-0 draft reports
  `k-1` truthfully. The second witness counts the propose calls whose DELIVERED
  draft row was not a pure function of its own first column, which is what a
  padded row is, and it is measured non-zero at every k >= 2. Neither witness
  proves PER-COLUMN PROVENANCE, that column j came from forward j. What the
  CPU tier does NOT show is acceptance at depth: no draft is accepted at any depth
  on the synthetic gate model, so the accept path at k>1 is unexercised there. A
  non-zero acceptance COUNT at depth would not close provenance either, because a
  padded row is accepted at column 1 whenever the target's own greedy continuation
  repeats a token, which real text does routinely. Closing it needs an acceptance
  RATE measured against a PADDED CONTROL on the same workload. The cross-engine
  speed comparison at k>1 and the DGX three-way at k=2..4, which must run that
  control, are still owed
  ([#81](https://github.com/mudler/vllm.cpp/issues/81) M1/M2), as are
  batch-size-keyed dynamic depth and acceptance-driven adaptive depth.
- **Correctness:** at concurrency 1 the speculative-on greedy output is
  token-for-token identical to both the speculative-off output and vLLM's own MTP
  speculative greedy output on the same prompt.
- **Speed:** measured about 1.04x faster than vLLM's own speculative-on decode at
  concurrency 1 (see [Measured result](#measured-result)).

## DFlash (block diffusion)

DFlash drafts a whole block in one non-autoregressive pass from a separate
z-lab draft checkpoint over the same target. `num_speculative_tokens` is
required and is the draft's block size, e.g. 16.

```bash
server --model /models/Qwen3.6-27B \
  --speculative-config '{"method":"dflash","model":"<draft-path>","num_speculative_tokens":16}'
```

Gated on the 27B: end-to-end output matches the vLLM DFlash-on golden under the
ratified near-tie rule, and at concurrency 1 it is 2.9x our own speculative-off
throughput and at or above vLLM's own DFlash-on decode. Speculative-off stays
byte-identical.

## n-gram

The draft-free proposer: candidates come from matching the sequence's own
suffix, so there is no draft checkpoint to load. `num_speculative_tokens` is
required and the prompt-lookup window defaults to 5/5.

```bash
server --model /models/Qwen3.6-27B \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4}'
```

It is wired end to end and reuses the same verify machinery as MTP. No
throughput number is published for it.

## DSpark (semi-autoregressive block drafting) — in progress

DSpark drafts a whole block in one parallel pass and then adds intra-block
dependency with a small sequential head, so a block draft stops being k
conditionally independent guesses. It is the DFlash drafter plus a low-rank
Markov transition bias.

Current state (`SPEC-DSPARK`): the config, the Markov head and draft model, both
published checkpoint layouts (native `deepseek-ai/dspark_qwen3_*` and
Speculators `RedHatAI/*.dspark`), the sequential sampler and the runner wiring
have landed, and the path runs end to end on the Qwen3.6-35B-A3B gate model.
DSpark now genuinely speculates on the Qwen3.6-35B-A3B gate model: draft tokens
are proposed and accepted, and throughput went from 6.78 to 41.89 tok/s once an
engine-wide bug was fixed that had been silently discarding EVERY speculator's
drafts on the CLI and server path (`check_for_draft_tokens` was never threaded
into `EngineCoreProc`, so `post_step` returned before installing anything).

On that model its greedy output is **token-identical to speculative-off across
all 48 tokens and reproducible** (both arms run on the synchronous path; the
speculative path forces async scheduling off, so a fair comparison has to force
it off on the other side too).

**It is still not gated**, and the cross-engine number is UNSETTLED rather than
merely owed. The host-side sequential stage described above was since moved on
device, the T=1+k verify was captured, and the draft chain made sync-free; all
three are byte-identical and landed. What is not settled is the ratio against
vLLM's own DSpark. Ratios recorded through 2026-08-13 (0.957x-0.989x) were taken
with a SINGLE COLD oracle invocation per paired run, so the denominator paid
compile-JIT the numerator did not; with the oracle warm and generation length
matched, the paired measurement is **0.834x**. Those two cannot be differenced
directly, because the gate host was reimaged in between and is no longer the same
machine. The deciding experiment — a single cold oracle invocation on the CURRENT
box — is specified in the benchmark record and has not yet run. Until it does, no
speed claim in either direction is supportable. The acceptance-rate band and the
other target families remain owed. A GGUF target, and a target architecture with no aux
multi-tap, are both refused by name.

```bash
main --model /models/Qwen3-4B \
  --speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7","num_speculative_tokens":7}'
```

`num_speculative_tokens` is required for a DSpark draft (a native DSpark config
carries no `n_predict`), and it must be at least the checkpoint's block size —
a smaller value produces incorrect output rather than merely lower acceptance,
so it is rejected.

## The flag

On the OpenAI server:

```bash
server --model /models/Qwen3.6-27B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

On the example CLI:

```bash
vllm-cli --model /models/Qwen3.6-27B \
  --prompt "The capital of France is" --max-tokens 64 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Over the C ABI (`include/vllm.h`, ABI v6):

```c
vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/models/Qwen3.6-27B";
mp.speculative_config = "{\"method\":\"mtp\",\"num_speculative_tokens\":1}";
vllm_engine *engine = NULL;
vllm_engine_load(&mp, &engine);   /* NULL/"" speculative_config => no speculation */
```

The JSON is parsed into the same `vllm::SpeculativeConfig` the C++ API takes
programmatically (`EngineParams::speculative_config`). The examples above use
`mtp`, where `num_speculative_tokens` defaults to the checkpoint's head depth.
See [Methods](#methods) for the other spellings and what each one requires. A malformed document, an unsupported
method, a missing `num_speculative_tokens` where the method needs one, or a
checkpoint with no `mtp.*` head fails the load loudly at startup rather than
running silently without speculation.

## Measured result

Concurrency 1, Qwen3.6-27B (NVFP4) on GB10, our speculative-on decode versus vLLM
0.25.0 running the same speculative config:

| Metric | Ours (spec on) | vLLM (spec on) |
|---|---:|---:|
| Time per output token, prose | 66.2 ms | 69.1 ms |
| Time per output token, code | 62.95 ms | 65.3 ms |
| Decode throughput, prose | 15.1 tok/s | 14.4 tok/s |
| Decode throughput, code | 15.7 tok/s | 15.1 tok/s |
| Drafter acceptance rate | 0.85 prose / 0.92 code | 0.84 |

Speculation helps both engines about 1.5 to 1.6x at this operating point, and our
engine is already about 4% faster than vLLM with speculation off, so the lead is
preserved with it on. The extra state speculation needs (a doubled recurrent-state
slot at k=1, plus the draft cache and head) costs about 3.6 GB, well inside the
box's unified memory. The full A/B, including the higher-concurrency numbers, is
in [BENCHMARKS.md](BENCHMARKS.md).

## Concurrency above 1

The concurrent (multi-request) path, where a speculative request shares a
scheduler step with an ordinary prefill, is implemented. The GDN layer splits the
batch into its speculative and ordinary rows, runs each through its own recurrence
over disjoint per-request state, and merges the results back; because the state is
disjoint, the mixed batch's output equals the two requests run alone, bit-for-bit,
at the real 27B and 35B dimensions. Speculation keeps helping at concurrency 2, 4
and 8 on the 27B (about 1.5x our own speculative-off throughput).

## Limitations

Each of these names the method it applies to.

- **MTP depth above 1 has no speed number yet.** The multi-step propose is
  built and token-exactness at k=1..4 is proven on CPU, but the DGX three-way at
  k=2..4 on the 27B and 35B and the cross-engine throughput A/B at matched k are
  both owed ([#81](https://github.com/mudler/vllm.cpp/issues/81) M1/M2), so no
  speed claim is made for k>1. Batch-size-keyed dynamic depth and
  acceptance-driven adaptive depth are unbuilt. A step whose actual draft count
  differs from the configured k (the scheduler clamps drafts to the step's token
  budget) falls out of the captured verify graph silently,
  [#1020](https://github.com/mudler/vllm.cpp/issues/1020).
- **EAGLE, EAGLE3, Medusa and the rest are not wired.** Of vLLM's thirteen
  `SpeculativeMethod` strings the engine accepts five, and `draft_model` among
  those is config-only. The remainder are inventoried, not implemented.
- **DSpark is not gated.** It runs end to end and is token-identical to
  speculative-off on the 35B, but it is about 2% slower than plain decode at
  concurrency 1, and the comparison against vLLM's own DSpark, the
  acceptance-rate band and the other target families are all still owed.
- **MTP needs Qwen3.5/3.6 with an `mtp.*` head.** The target may be a safetensors
  directory or a `.gguf` converted WITH the head (llama.cpp's layer-indexed
  `nextn` block); a GGUF exported `--no-mtp` is refused and says so.
- **On an NVFP4 safetensors target the concurrency-1 identity above is not
  currently reliable.** Measured on the 35B A3B NVFP4 safetensors: its logits
  land on a coarse grid that yields EXACT ties between distinct tokens, and at
  such a tie speculative-on and speculative-off, and even two speculative-off
  runs, can pick differently. The same weights loaded from GGUF, which expands
  to bf16, show no such ties and are token-identical. Open; see
  [docs/STATUS.md](STATUS.md).
- **Concurrency above 1 is not token-stable for the 27B.** Its greedy output is
  not bit-stable across batch shapes even with speculation off (changing the batch
  size flips a few near-tie tokens), so exact token-for-token agreement between
  speculative-on and speculative-off is a concurrency-1 property only. Correctness
  above concurrency 1 rests on the model-independent bit-exact GDN split-merge
  proof and on matching drafter acceptance, not on identical token streams.

## Consuming it programmatically

The flag is a thin wrapper over the library surface:

```cpp
vllm::entrypoints::EngineParams params;
params.speculative_config =
    vllm::ParseSpeculativeConfigJson(R"({"method":"mtp","num_speculative_tokens":1})");
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, params);
```

Leaving `speculative_config` unset (the default) loads the ordinary engine, which
is byte-identical to a build with no speculative code compiled in.
