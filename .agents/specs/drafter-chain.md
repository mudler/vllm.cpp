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
- an absent field leaves every existing single-method path RESOLVING IDENTICALLY,
  key for key, asserted against a real config rather than a synthetic one;
- an unknown or malformed entry is refused BY NAME, with the accepted list, as
  #1160 established for every other key;
- a chain naming a method this engine does not implement is refused by name rather
  than silently skipped.

## Gates

- **G1 — additivity.** With the field absent, every document vLLM accepts
  resolves to the same `SpeculativeConfig` it resolved to before, KEY FOR KEY,
  and every existing speculative gate stays green. This is the gate that protects
  vLLM's surface, and it must run on a real `--speculative-config` document, not
  a hand-built struct.

  **It is not a byte-identity claim, and the one delta is named rather than
  waived.** Measured at W1: 152 refusal messages differ, all of them by the same
  appended tail, because the accepted-key list at
  `src/vllm/config/speculative.cpp` gained `vllm_cpp.drafter_chain`. That change
  is required and not incidental — a list that omits an accepted key stops
  closing the user's search, which is the reason #1160 appends one at all. No
  parse result moves and no guard changes. A gate written as "byte-identical"
  would be a gate this row cannot pass and should not try to, so G1 is written as
  what is actually true.
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
  G5. Lands no chain behaviour. **LANDED**, issue
  [#1522](https://github.com/mudler/vllm.cpp/issues/1522): the document key is
  `vllm_cpp.drafter_chain` (D6), it is parsed and validated by
  `ParseSpeculativeConfigJson` and stored on
  `SpeculativeConfig::drafter_chain`, and `LoadedEngine::ResolveSpecConfig`
  refuses a well-formed chain by name because nothing resolves one yet.
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
- **D6 — the field is `vllm_cpp.drafter_chain`, nested under one extension
  object, not a bare top-level `drafter_chain`.** W1 decision. The constraint is
  that the name must not collide with a key vLLM's `SpeculativeConfig` declares
  "now or plausibly". A bare `drafter_chain` satisfies only the first half:
  nothing prevents upstream from later choosing that spelling, and if it did,
  one document would mean two different things in the two engines with no error
  anywhere. Nesting reduces the collision surface to exactly ONE name, and a
  collision on `vllm_cpp` would be loud rather than silent. It is also the
  landed convention of this tree: `--offload-config` already carries this
  engine's non-vLLM residency tier under exactly the same key
  (`src/vllm/config/weight_residency.cpp:451-478`), so the two flags spell an
  extension the same way. Rejected: a bare `drafter_chain`; and a `vllm_cpp_`
  name prefix, which buys the same guarantee while making the document read as
  though the prefix were part of each field's name.
- **D7 — a top-level `method` and a chain are mutually exclusive, and so is
  every top-level PER-SPECULATOR key.** W1 decision. `method` names one
  speculator and the chain names an ordered list. No rule here makes a top-level
  `method` the head of the chain, its tail, or a default, and inventing one
  would silently reinterpret a document. The same argument holds one key at a
  time for `model`, `num_speculative_tokens`, `prompt_lookup_min` and
  `prompt_lookup_max`: beside a two-entry chain, `model` names a checkpoint with
  no entry to belong to, and applying it to the first entry, to all of them, or
  to none are three different engines. `draft_sample_method` and
  `rejection_sample_method` DO coexist, because they describe the draft sampling
  rule and the verify, and the verify is one verify however many speculators
  propose into it. A parsed chain therefore leaves `SpeculativeConfig::method`
  EMPTY; promoting an entry into it would make a chain document parse into a
  config a one-drafter engine runs without complaint.
- **D8 — a SINGLE-entry chain is legal.** W1 decision. It is the degenerate
  preference list, exactly as `--spec-type mtp` is in llama.cpp, and refusing it
  would force every generator of these documents to special-case `n == 1` into a
  different spelling. It is not equivalent to a top-level `method`: the entry
  lands in the chain, `method` stays empty, and the loader's chain refusal still
  sees it. **The loader half of that sentence is EXECUTED**, by
  `tests/vllm/entrypoints/test_drafter_chain_reach.cpp` "a ONE-entry chain is
  seen by the loader too (D8)", which drives a one-entry document through both
  `ResolveSpecConfig` and `FromModelDir`. It was added by the W1 repair because
  every other reach case used a TWO-entry chain, and a two-entry chain cannot
  tell `!drafter_chain.empty()` apart from `drafter_chain.size() > 1`. Under that
  narrowing the parse still succeeds and `method` is still empty, so
  `ResolveSpecConfig` falls through to its bottom and answers a legal one-drafter
  document with `only methods "mtp", "dflash", "dspark", "ngram" and
  "draft_model" are supported (got "")` — a message about a method the user never
  wrote. That is the observed red, not a predicted one.
- **D9 — an EMPTY chain, and a `vllm_cpp` object with no `drafter_chain`, are
  refused.** W1 decision. Both ask for a preference order over nothing. The only
  two readings are "no speculation" and "whatever the engine would otherwise
  have done", and both are silently different from what the document says, which
  is the class of defect [#1160](https://github.com/mudler/vllm.cpp/issues/1160)
  exists to end. Omitting `vllm_cpp` already means "no chain", so neither
  spelling is needed for anything.
- **D10 — the four chain entry methods are `mtp`, `dflash`, `dspark` and
  `ngram`; `draft_model` is refused with that fact said out loud.** W1 decision,
  from `## Scope`. `draft_model` IS an accepted top-level method here, so a
  message calling it unknown would send a user hunting for a typo that is not
  there — the same distinction #1160 draws for keys, applied to methods. Its
  chain arm is listed under `## Owed`. Every other name, including a real vLLM
  `SpeculativeMethod` such as `eagle3`, is refused by name rather than skipped,
  because a silently shortened chain is a different feature running under the
  user's document.
- **D11 — a method named twice is refused.** W1 decision. W2 keys the
  per-drafter counters and the per-sequence winner on the method name, so two
  entries with the same method cannot be told apart in the numbers that decide
  whether a chain helps at all — and those numbers are G2, G3 and G4. Refusing
  now is reversible; shipping ambiguous counters is not. Two checkpoints of one
  method become expressible when the attribution key is a `(method, model)` pair
  rather than a method, which is a W2 design question and is listed under
  `## Owed`.
- **D12 — a well-formed chain is REFUSED at the loader, not accepted and
  ignored.** W1 decision, and the reason W1 lands nothing dead. W1 stores a
  field that nothing resolves. A malformed document was never the risk; a
  WELL-FORMED one that parses, stores and is then ignored is, because the engine
  would draft with one speculator, or with none, under a document whose author
  believes it configures several — and a measurement taken there is a
  measurement of a configuration nobody chose. `LoadedEngine::ResolveSpecConfig`
  is therefore the production READER of the stored vector and refuses by name,
  and `LoadedEngine::FromModelDir` calls it ahead of its path resolution so the
  refusal precedes every weight operation. W3 lifts the refusal; until then the
  stored field has a live consumer, which is what a reachability mutation can
  delete.

- **D5 — no ceiling and no ratio is claimed here.** If a chain does not help on a
  workload where it should, that is an open result with a named next hypothesis,
  not a conclusion about chains.

## Owed

Each item is refused BY NAME today, so none of them is a silent gap. All are
owned by this row and tracked by
[#1522](https://github.com/mudler/vllm.cpp/issues/1522).

- **Chain RESOLUTION (W3).** `LoadedEngine::ResolveSpecConfig` refuses a
  well-formed `vllm_cpp.drafter_chain` because nothing resolves one. This is the
  one thing W1 landed that is not yet reached beyond its own refusal: the parsed
  entries are read only to be named in that message. D12 records why the refusal
  is the correct shape and W3 lifts it.
- **Per-drafter ATTRIBUTION and ACCEPTANCE counters (W2, W4).** No analogue of
  llama.cpp's `impl_last`, `n_gen_drafts` or `n_gen_tokens` exists, so G2, G3 and
  G4 have no instrument yet.
- **`draft_model` as a chain entry (D10).** Accepted at the top level, refused as
  a chain entry by name, with the row named in the message.
- **Two entries of the SAME method, with different checkpoints (D11).** Refused
  by name. It becomes expressible when W2 keys attribution on a `(method, model)`
  pair rather than on the method.
- **`.agents/oracles/llama-cpp.md` must carry a recorded pin** before any gate
  cites llama.cpp semantics. W1 cites none: its rules are this engine's
  document-shape decisions, and the llama.cpp reading in `## Upstream chain` is
  design context rather than a gate input.

Four more, filed by the W1 fresh review and owned by this row. The first was
fixed in flow; the other three are refused-or-inherited today and are recorded
here so that none of them is a silent gap.

- **[#1598](https://github.com/mudler/vllm.cpp/issues/1598) — FIXED IN FLOW.** A
  chain entry called `draft_sample_method` and `rejection_sample_method` a typo,
  when this engine honours both at the top level of the same document. Class 2 of
  #1160's split was missing from `CheckEntryKeys`. It now has its own refusal,
  which names the key, says the engine honours it, and says to spell it at the
  top level.
- **[#1599](https://github.com/mudler/vllm.cpp/issues/1599) — a non-string
  `model` is dropped in silence, and the required-key message that follows calls
  the key MISSING when it was given.** INHERITED: `src/vllm/config/speculative.cpp`
  has the same `is_string()` shape at the top level, so the entry is faithful to
  the landed contract and repairing only the entry would leave two spellings of
  one rule disagreeing. Out of W1's scope because the fix moves a landed
  top-level refusal that other suites assert on.
- **[#1600](https://github.com/mudler/vllm.cpp/issues/1600) — a misspelled or
  mis-cased `vllm_cpp` on a chain-only document is answered with `a string
  "method" is required`,** the one key such a document must not have. The
  ORDERING is inherited from #1160 and deliberate; the document CLASS that hits
  it is invented by this wave, which is why it is filed against this row. D9
  argues the user must not be misled here, and this is the one shape where the
  landed code misleads. Out of W1's scope because both candidate repairs change a
  landed error ordering that existing suites assert on.
- **[#1601](https://github.com/mudler/vllm.cpp/issues/1601) — the llama.cpp
  citations carry no revision.** `include/vllm/config/speculative.h` and
  `## Upstream chain` cite `common/arg.cpp:3754-3763` and
  `common/speculative.cpp:2164-2186` by bare line, while
  `.agents/oracles/llama-cpp.md` pins `10bf611e` (`b10451`, `gateable = no`).
  Design context only, per the bullet above, but a bare line anchor on a moving
  upstream goes stale silently and `scripts/check-symbol-anchors.py` states it
  cannot verify a line citation.

## Now

`SPEC-DRAFTER-CHAIN` is `ACTIVE`. **W1 has landed**: the
`vllm_cpp.drafter_chain` field parses, validates and refuses by name, and a
well-formed chain is refused at the loader before any weight I/O because
nothing resolves one yet (D12). No chain behaviour landed, nothing changed which
speculator drafts, and with the field absent every document vLLM accepts
resolves exactly as it did, key for key — the one measured delta being the
accepted-key list quoted in an unknown-key refusal, which now also names
`vllm_cpp.drafter_chain` (G1) —
which G1 asserts through real `--speculative-config` documents driven into
`LoadedEngine::ResolveSpecConfig` and `LoadedEngine::FromModelDir`, not through
a hand-built struct.

Gates: **G1 PASS** and **G5 PASS**, both CPU-only, in
`tests/vllm/config/test_speculative_drafter_chain.cpp` and
`tests/vllm/entrypoints/test_drafter_chain_reach.cpp`. G2, G3 and G4 remain
PENDING with no instrument, which is what W2 exists to build. No throughput
number is claimed or claimable from this row.

Next action: W2, per-drafter attribution — and still not before `SPEC-DFLASH2`
W6 produces a measured number, which W3 in particular must wait for.
