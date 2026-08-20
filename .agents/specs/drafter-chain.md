# SPEC-DRAFTER-CHAIN — a preference-ordered chain of speculators

**Row:** `SPEC-DRAFTER-CHAIN` (engine-matrix, speculative decoding).
**Issue:** [#1522](https://github.com/mudler/vllm.cpp/issues/1522).
**Kind:** structured spec. No production code lands with this spec.
**Secondary oracle:** llama.cpp (`llama-cpp`), for the SEMANTICS of the chain
only, because vLLM implements nothing here. It never becomes the mirror source.

**One winner per sequence, not an ensemble.** This row adds a fallback chain: try
the first speculator, and if it yields no draft for a sequence, try the next.
Nothing merges proposals, nothing combines distributions, and the lossless verify
is untouched because only one proposal ever reaches it.

**Additive by construction.** The chain is a NEW OPTIONAL FIELD on
`--speculative-config`. Absent, the engine behaves exactly as it does today, key
for key, and every `--speculative-config` document vLLM accepts continues to
parse and behave identically. That constraint is the developer's, recorded
2026-08-21, and it is what keeps this a superset of vLLM's surface rather than a
divergence from it.

## Scope

In scope: the new field and its validation; per-sequence resolution of the chain;
per-drafter attribution and acceptance accounting; and the gates that make an
unfired fallback distinguishable from a helpful one. Entries are methods this
engine already implements — `mtp`, `dflash`, `dspark`, `ngram` — each with its own
checkpoint and its own `num_speculative_tokens`.

Out of scope: merging or ensembling proposals; any change to the verify or to the
rejection sampler; any change to single-method behaviour; llama.cpp's flag
spelling (`--spec-type`, comma-separated names), which cannot carry per-entry
configuration; and adding any speculator this engine does not already have.

**Not to be started before `SPEC-DFLASH2` W6 produces a measured number.** A new
speculator-composition surface landing mid-campaign puts the ratio that row is
chasing behind an extra variable.

## Upstream chain

**vLLM implements nothing here, verified at two revisions.**
`SpeculativeMethod` is a single `Literal` — `ngram`, `medusa`, `mlp_speculator`,
`draft_model`, `suffix`, `custom_class`, plus the Eagle, n-gram-GPU and DSpark
families — and `SpeculativeConfig.method` is one value of it. Checked at the
parity pin `555967922` and at `origin/main` `c20572610`: no composition, no
preference list, no fallback anywhere in that config. So there is nothing to
mirror, and AGENTS.md `## When vLLM has no implementation` admits a secondary
oracle.

**llama.cpp, read from the code rather than the flag.** `--spec-type` splits on
`,` into `params.speculative.types` (`common/arg.cpp:3754-3763`).
`common_speculative_init` computes "the implementations to use based on the config
and their order of preference" — configs *to try*. The resolution is per sequence
(`common/speculative.cpp:2164-2186`):

- `struct common_speculative` holds `std::vector<std::unique_ptr<common_speculative_impl>> impls`
  and `impl_last`, commented "which implementation was used for a given seq_id".
- A sequence carries a `drafting` flag. When an implementation returns a non-empty
  draft it sets `dp.drafting = false`, records `impl_last[seq_id]`, and increments
  that implementation's `n_gen_drafts` and `n_gen_tokens`.
- Sequences still `drafting` are counted into `n_drafting`; the loop breaks when
  `n_drafting == 0`. What falls out is handled as "these sequences failed to
  generate a draft".
- `common_speculative_n_max` takes the `max` over enabled types, sizing the buffer
  for whichever one wins.

Its type set is `DRAFT_SIMPLE`, `DRAFT_EAGLE3`, `DRAFT_MTP` and the n-gram family.
**No `draft-dflash` type exists there**, so the pairing that prompted this row is
not something llama.cpp can demonstrate; the chain mechanism is what is being
mirrored, not any particular pair.

Pin the oracle in `.agents/oracles/llama-cpp.md` before any gate cites it.

## Our baseline

- `--speculative-config` is parsed by `ParseSpeculativeConfigJson`, which honours
  `method`, `num_speculative_tokens`, `model`, `prompt_lookup_min`,
  `prompt_lookup_max`, `draft_sample_method` and `rejection_sample_method`,
  **refuses any other name outright with the accepted list**, and refuses a name
  vLLM declares but this engine does not implement as exactly that (#1160). A new
  key is therefore inert until deliberately admitted — the mechanism that makes
  this row additive rather than accidental.
- `LoadedEngine::ResolveSpecConfig` resolves ONE method today, and
  `GPUModelRunner::propose_drafts` dispatches to one propose path per step.
- The methods that would populate a chain already exist and are separately gated:
  `SPEC-MTP` (DONE), `SPEC-DFLASH` (DONE), `SPEC-DSPARK` (ACTIVE), `SPEC-NGRAM`,
  and `SPEC-DFLASH2` (ACTIVE, mid-campaign).
- Nothing in the engine records WHICH speculator answered a given sequence. There
  is no analogue of `impl_last`, `n_gen_drafts` or `n_gen_tokens`, and no
  per-drafter acceptance counter. That absence is the row's main obstacle, not
  the chain logic.

## Port map

| llama.cpp | Ours | Note |
|---|---|---|
| `params.speculative.types` (vector) | a new optional list field on `--speculative-config` | objects, not names: each entry carries its own `model` and `num_speculative_tokens` |
| `common_speculative_init` preference order | chain construction in `LoadedEngine::ResolveSpecConfig` | one resolved config per entry, in the order given |
| `impls` + `impl_last[seq_id]` | per-sequence winner recorded in the runner's speculative state | the attribution `## Gates` needs |
| the `drafting` flag loop | per-sequence resolution in `propose_drafts` | first non-empty draft wins; later entries are not consulted for that sequence |
| `n_gen_drafts` / `n_gen_tokens` | per-drafter counters, readable from outside | plus per-drafter ACCEPTANCE, which llama.cpp does not keep |
| `common_speculative_n_max` | draft buffer sized by the max over entries | mirrors the same rule |

New files mirror the existing speculative layout; nothing forks a parallel propose
path.

## Tests to port

llama.cpp has no unit tests for this that are portable here, so the tests are
ours and are written against the semantics enumerated above rather than
transcribed. Required:

- an entry removed from the chain reddens a gate;
- the chain ORDER reversed reddens a different gate — order is the semantics, and
  a suite that passes under both orders has not tested a preference list;
- a sequence for which the first entry declines is shown to be served by the
  second, by attribution and not by inference from token counts;
- an absent field leaves every existing single-method path byte-identical,
  asserted against a real config rather than a synthetic one;
- an unknown or malformed entry is refused BY NAME, with the accepted list, as
  #1160 established for every other key;
- a chain naming a method this engine does not implement is refused by name rather
  than silently skipped.

## Gates

- **G1 — additivity.** With the field absent, every existing speculative gate is
  byte-identical. This is the gate that protects vLLM's surface, and it must run
  on a real `--speculative-config` document, not a hand-built struct.
- **G2 — attribution.** For a constructed workload where the first entry provably
  declines, the per-sequence winner is the second entry, read from the engine's
  own counters. **A token gate cannot see this**, so the counters are the gate.
- **G3 — order.** Reversing the chain changes which drafter answers, measured the
  same way.
- **G4 — acceptance accounting.** Per-drafter accepted-token counts sum to the
  engine's total, and each is non-zero for a workload constructed to exercise it.
- **G5 — refusal.** Every malformed shape refused by name, before any weight I/O.

**No throughput claim is admissible from this row until G2 and G4 read.** The
premise — that a cheap fallback covers what an expensive drafter declines — is an
acceptance claim, and acceptance is invisible to a token gate because the verify
is lossless.

## Dependencies

- The methods a chain composes are already implemented; this row adds none.
- `SPEC-DFLASH2` W6 first, by the scope note above.
- `.agents/oracles/llama-cpp.md` must carry a recorded pin before any gate cites
  llama.cpp semantics.
- No GPU is required for G1, G3 or G5. G2 and G4 need whichever checkpoints the
  chosen chain entries use, on a leased device if those are GPU-only.

## Work breakdown

- **W1 — the field.** Parse, validate, refuse by name; inert when absent. G1 and
  G5. Lands no chain behaviour.
- **W2 — attribution.** Per-sequence winner and per-drafter counters, reachable
  from a production entry point. G2. This is deliberately BEFORE the chain runs,
  so the instrument exists before the thing it measures.
- **W3 — resolution.** The chain itself: first non-empty draft wins, later entries
  not consulted for that sequence. G3.
- **W4 — acceptance accounting.** Per-drafter accepted counts. G4.
- **W5 — measurement.** Whether a chain helps, on a workload where it should,
  against single-method baselines.

## Risks/decisions

- **D1 — a new field, not a new flag, and inert when absent.** Developer decision,
  2026-08-21. Keeps `--speculative-config` a superset of vLLM's document; every
  vLLM-valid config keeps its exact meaning. Rejected: llama.cpp's comma-separated
  `--spec-type` string, which cannot carry per-entry `model` and
  `num_speculative_tokens`.
- **D2 — fallback, not ensemble.** This mirrors what llama.cpp actually does,
  which the code shows is one winner per sequence. An ensemble would touch the
  verify and the rejection sampler; this touches neither.
- **D3 — attribution lands before resolution (W2 before W3).** A fallback that
  never fires is indistinguishable from one that fires and helps, because tokens
  are identical either way. Building the instrument first is what makes W3
  gateable at all; the reverse order would ship an unfalsifiable feature.
- **D4 — llama.cpp is cited for semantics only.** It never outranks vLLM and never
  becomes the mirror source. Where vLLM later grows a composition surface, this row
  reconciles onto vLLM and the divergence is recorded.
- **D5 — no ceiling and no ratio is claimed here.** If a chain does not help on a
  workload where it should, that is an open result with a named next hypothesis,
  not a conclusion about chains.

## Now

`SPEC-DRAFTER-CHAIN` is `READY`. The spec is committed and base-reachable, which
is what a helper dispatch needs; no production code has landed. Next action: W1,
the field, dispatched to a fresh implementer against this spec, red-first on the
refusal cases — and not before `SPEC-DFLASH2` W6 reads.
