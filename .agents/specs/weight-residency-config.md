# The disk-residency tier as a config surface (`ENG-RESIDENCY-CONFIG`)

Make the host-RAM→disk weight-residency tier reachable from the server's JSON
config surface instead of only from `VT_*` environment variables. Issue
[#1110](https://github.com/mudler/vllm.cpp/issues/1110). Also closes
[#1109](https://github.com/mudler/vllm.cpp/issues/1109), the documented default
of `VT_GGUF_PREFAULT` being the opposite of the code's, because this row writes
that default into a resolver and a config key and cannot ship a document that
contradicts its own new code.

**Verdict up front.** vllm.cpp offloads weights at two tiers and only one is
configurable. The device→host tier takes
`--offload-config '{"uva":{"cpu_offload_gb":N,"cpu_offload_params":["experts"]}}'`
and reaches the loader through `server_main.cpp` → `EngineParams` →
`LoadedEngine::FromModelDir`. The host→disk tier — the one that makes
`Qwen3.8-2.4T-A95B UD-Q1_0` (370 GiB) serve on a 119 GB box, because ~330 GiB of
experts stay borrowed in the mapping — is `VT_GGUF_MMAP`, `VT_GGUF_PREFAULT`,
`VT_MOE_EXPERT_STREAM`, `VT_MOE_EXPERT_STREAM_SLOTS`,
`VT_MOE_EXPERT_STREAM_SLOT_BYTES`, and nothing else. A user reaches for
`--offload-config` because its first tier is already expert-aware
(`cpu_offload_params: ["experts"]` targets expert weights by name segment), gets
the GPU tier, and finds nothing for the tier the big-model case needs.

Three findings shape the change:

- **The mirror is not negotiable and does not need to be broken.**
  `include/vllm/config/offload.h` is a transcription of upstream
  `vllm/config/offload.py` @ `555967922`, cited line-for-line
  (`UVAOffloadConfig:16-44`, `PrefetchOffloadConfig:48-76`, `OffloadConfig:80-93`,
  `validate_offload_config:96-136`). Upstream has no disk tier, so there is
  nothing to mirror and the extension is vllm.cpp-original by construction. It
  therefore lives in its own struct in its own file, under its own namespaced
  JSON key, and `OffloadConfig` gains no field.
- **An unknown top-level key is already accepted, so option 2 of the issue is
  workable — and closing that tolerance is this row's job.**
  `parse_offload_config_json` (`src/vllm/config/offload.cpp:234-283`)
  looks up `offload_backend`, `uva` and `prefetch` by name and never enumerates
  the document's keys; `validate_offload_config`'s mirror
  (`OffloadConfig::Validate`) reads fields, not keys. A `vllm_cpp` sibling is
  invisible to both. This is what makes the namespaced key possible rather than a
  sibling flag, and it is also the one hazard the extension parser has to close
  itself: a *misspelled* key is silently ignored by both parsers, so the extension
  parser enumerates the **whole document** — the four legal top-level keys, its own
  two levels, and the fields of the two MIRRORED sub-objects `uva` and `prefetch` —
  and refuses anything else by name. It enumerates NAMES only inside the mirrored
  pair; `parse_offload_config_json` keeps sole ownership of their types, defaults and
  bounds. Enumerating only inside
  `vllm_cpp` was the first shape and it left the worst spelling accepted:
  `{"vllm-cpp":{…}}` with a hyphen parsed to an empty config and started a server
  running this tier at its defaults — prefault ON, streaming OFF (#1122 H1). Refusing is also the mirror-faithful
  polarity: upstream has no `--offload-config` flag at the pin — the string appears
  nowhere in the tree, so the whole JSON document is vllm.cpp's own — and vLLM builds
  its config dataclasses with the `@config` decorator, whose body sets
  `ConfigDict(extra="forbid")` (`vllm/config/utils.py:68-69`). `OffloadConfig`
  (`offload.py:80`) and `KVTransferConfig` (`kv_transfer.py:22-23`) both carry it.
- **Two of these five knobs latch a decision; three do not, and the difference is
  the whole contract of the install.** `ResolveExpertStreamRequested()` caches its
  answer in a function-local static — `Qwen35ExpertStreamRequested()` is the
  model-side name and is a pure delegation to it — and the slot store is built once
  per process so its `slots x slot_bytes` reservation cannot be resized. Those two
  freeze. `mmap` does not, because
  `GgufLoadPolicy::FromEnv()` is called per load; nor does `prefault`, because this
  row deliberately removes `PrefaultBorrowedSpan`'s `enabled` static
  (`qwen3_5_gguf_weights.cpp:37-44`). A config that would change a *frozen* field is
  not merely late — honouring it would record a configuration the engine is not
  running, the invisible-fallback shape this tree refuses everywhere else — so
  `SetWeightResidencyConfig` **throws** when a document SETS one of those two fields
  to a value that would resolve differently from the decision already taken, and
  accepts everything else: a second engine's `mmap`/`prefault` document, a document
  that OMITS the frozen field, one that asks for what was decided, and one the
  environment overrides anyway. Everything accepted is MERGED field by field, so an
  omitted field keeps the value already installed. Marking
  one process-wide flag in the shared resolvers instead made an ordinary first load
  refuse a legal second load (#1122 M1); comparing against the stored document rather
  than the decision, and replacing rather than merging, then produced the same hard
  failure and a silent field drop (#1133 H1, H2). The install site is the first
  statement block of `LoadedEngine::FromModelDir`, beside the weight-offloader
  install, which is already documented as being before any weight I/O.

## Scope

| Field | Content |
|---|---|
| Row ID | `ENG-RESIDENCY-CONFIG` (engine-matrix, KV cache and memory). Issue [#1110](https://github.com/mudler/vllm.cpp/issues/1110); fixes [#1109](https://github.com/mudler/vllm.cpp/issues/1109) in flow |
| In | A vllm.cpp-original `WeightResidencyConfig` under the `vllm_cpp` key of the existing `--offload-config` document; its parser, which refuses an unknown key at every level of the document (the four legal top-level keys included) and a wrong-typed or non-positive field; a process-global install/resolve seam with a defined config-vs-env precedence and a late-install refusal; the three call sites that resolve these knobs today (`GgufLoadPolicy::FromEnv` for `mmap`, `PrefaultBorrowedSpan` for `prefault`, `Qwen35ExpertStreamRequested` + the `Qwen35ExpertStream` constructor for the streaming lane); the flag→`EngineParams`→install chain through both production entry points (`server_main.cpp` and the C ABI's `offload_config`); `docs/USAGE.md` and `docs/ENVIRONMENT.md` |
| Out | Any change to `OffloadConfig`, `UVAOffloadConfig`, `PrefetchOffloadConfig` or their validator — the mirror stays byte-faithful. Any change to what the knobs *do*: this row moves where their value comes from and nothing else. A new flag. `VT_MOE_EXPERT_STREAM_STATS_EVERY` (see below). `VT_GGUF_KEEP_QUANT`, `VT_CPU_REF`, `VT_GGUF_KEEP_F16` and the rest of the load-transform family — they are a different tier and a different row |
| Supported modes | `{"vllm_cpp":{"mmap":{"enabled":bool,"prefault":bool},"expert_stream":{"enabled":bool,"slots":int,"slot_bytes":int}}}`. Every field is optional and every absent field means "unchanged", so an absent `vllm_cpp` key is byte-identical to today |
| Dispatch behavior | Resolved from **env var if set, else config if set, else the built-in default**, at each read. `mmap` and `prefault` are read per load and per span; `expert_stream` is cached on first read and the two sizes are fixed when the slot store is built. Nothing is resolved when neither input is set, so the default engine path is byte-identical |
| Regimes served | A checkpoint larger than host RAM on a single box: the mmap-borrowed weight tower plus the bounded expert slot cache. CPU keep-quant expert towers today; a device platform serves the slice device-resident and is unaffected |

## Upstream chain

Pin `555967922` (`.agents/upstream-sync.md`), verified in the local checkout.

**Upstream implements the mirrored half and nothing else.** `vllm/config/offload.py`
defines `OffloadBackend = Literal["auto", "uva", "prefetch"]` (`:12`) and the two
sub-configs; there is no third arm and no disk tier. `create_offloader`
(`vllm/model_executor/offloader/base.py:139-162`) selects prefetch, uva, or
`NoopOffloader`. `offloader/uva.py:21` is a CPU-blanket UVA offloader and
`offloader/prefetch.py:557-560` is cpu-only. Neither reads a file at inference
time.

So there is nothing to mirror for this tier, and this row must not invent
anything inside the mirrored structs. What it does mirror is the **shape** of the
surface next door: `--kv-transfer-config` and `--offload-config` both take a JSON
object parsed and validated at startup, before the multi-GB load, and refuse a
typo rather than defaulting it (`server_main.cpp:1096-1109`). The extension
follows that contract exactly, including the refuse-rather-than-default rule,
which is why an unknown key inside `vllm_cpp` is an error.

The knobs themselves are ports and already carry their anchors: the load-time
prefault mirrors llama.cpp's mmap prefetch under `use_mmap`
(`src/llama-mmap.cpp:451` @ `237ad9b96`), recorded at
`qwen3_5_gguf_weights.cpp:28-36`; the expert slot cache is the surpass-track
`ENG-EXPERT-STREAM` lane, whose absence upstream is recorded in the engine
matrix. This row adds no upstream behavior, so it inherits their anchors rather
than claiming new ones.

## Our baseline

Measured on this tree at `281e6a120`.

| Knob | Read at | Latches | Default |
|---|---|---|---|
| `VT_GGUF_MMAP` | `gguf_keep_quant.cpp:163` (`GgufLoadPolicy::FromEnv`) | no — `FromEnv()` is called per load | `p.keep_quant`, i.e. on wherever the device can execute the quantized GEMM; forced off by `VT_CPU_REF` |
| `VT_GGUF_PREFAULT` | `qwen3_5_gguf_weights.cpp:39` (`PrefaultBorrowedSpan`) | **yes**, function-local static | **on** — unset reads as enabled — and `docs/ENVIRONMENT.md:50` says off, which is [#1109](https://github.com/mudler/vllm.cpp/issues/1109) |
| `VT_MOE_EXPERT_STREAM` | `qwen3_5.cpp:5149` (`Qwen35ExpertStreamRequested`) | **yes**, function-local static | off |
| `VT_MOE_EXPERT_STREAM_SLOTS` | `qwen3_5.cpp:5471` (`Qwen35ExpertStream` ctor) | effectively — the store is a process-lifetime singleton built once | 64 |
| `VT_MOE_EXPERT_STREAM_SLOT_BYTES` | `qwen3_5.cpp:5470` (same ctor) | same | the largest of the first MoE layer's gate/up/down slices |
| `VT_MOE_EXPERT_STREAM_STATS_EVERY` | `qwen3_5.cpp:5476` (same ctor) | same | 16 |

The config side already exists and is wired: `--offload-config`
(`server_main.cpp:579`) → `parse_offload_config_json` + `Validate`
(`:1101-1108`) → `EngineParams::offload_config`
(`include/vllm/entrypoints/model_loader.h:156`) → `LoadedEngine::FromModelDir`
(`model_loader.cpp:1255-1318`), which installs the offloader **before any weight
I/O** and says so. The C ABI carries the same string
(`include/vllm.h:413-470` documents it, `:471` is the field, `src/capi/vllm_c.cpp:638-645` parses it). So the plumbing this row
needs is one field wide, and the install point is already chosen and already
documented for exactly this ordering reason.

The row's own gap: `EngineParams` has nowhere to put the extension, and the three
resolve sites have no input but `getenv`.

## Port map

| Piece | Where |
|---|---|
| `WeightResidencyConfig` + the resolve/install seam | new `include/vllm/config/weight_residency.h`, `src/vllm/config/weight_residency.cpp` |
| Extension parser | `parse_weight_residency_extension_json` in the same pair, reading the `vllm_cpp` key of the SAME document `--offload-config` carries |
| Engine field | `EngineParams::weight_residency` in `include/vllm/entrypoints/model_loader.h` |
| Install (production call site) | `LoadedEngine::FromModelDir`, `src/vllm/entrypoints/model_loader.cpp`, in the block that already installs the weight offloader |
| Server flag | `src/vllm/entrypoints/openai/server_main.cpp`, inside the existing `if (!args.offload_config.empty())` block |
| C ABI | `src/capi/vllm_c.cpp`, inside the existing `offload_config` block |
| `mmap` resolve | `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp`, `GgufLoadPolicy::FromEnv` |
| `prefault` resolve | `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`, `PrefaultBorrowedSpan` |
| `expert_stream`, `slots`, `slot_bytes` resolve | `src/vllm/model_executor/models/qwen3_5.cpp`, `Qwen35ExpertStreamRequested` and the `Qwen35ExpertStream` constructor |
| Docs | `docs/USAGE.md` (the streaming section gains the config form), `docs/ENVIRONMENT.md` (precedence note + the #1109 default correction) |
| Named resolvers | `ResolveGgufMmap`, `ResolveGgufPrefault`, `ResolveExpertStreamRequested` (+ its pure `ExpertStreamRequestedFrom`), `ResolveExpertStreamSlots`, `ResolveExpertStreamSlotBytes` — one per knob, each the sole reader of its variable |

**Five knobs, five named resolvers, and the polarities are not the same.** Each
knob gets one function in the new header that owns its environment NAME and its
exact historical rule, and each becomes the sole reader of its variable. That is
not decoration. `VT_GGUF_MMAP` and `VT_GGUF_PREFAULT` compare the whole value
against `""`, `"0"`, `"false"` and `"off"` (the tree's `EnvOn`,
`gguf_keep_quant.cpp:61-66`). `VT_MOE_EXPERT_STREAM` examines only the **first
character** — `v[0] != '0' && v[0] != '\0'` — so `VT_MOE_EXPERT_STREAM=false`
reads as ON, and `docs/ENVIRONMENT.md` states that explicitly. Routing all five
through one generic helper would silently normalise the odd one, and a row whose
subject is *where a value comes from* must not also change *what a value means*.
So the odd rule is transcribed, and it is additionally exposed in a pure form
(`ExpertStreamRequestedFrom(env_value, configured)`) because its wrapper latches
and can be exercised only once per process — which is exactly how a
normalisation there would have escaped a test.

That once-per-process budget also decides where the WRAPPER is gated. The pure form
covers the decision and the wrapper's environment name is covered by the precedence
cases; the FIELD the wrapper reads was covered by nothing, and rewiring it to the
adjacent `.mmap` left all four suites green (#1122 M2). So the one observation a
process gets is spent on that, in a binary of its own:
`tests/vllm/config/test_expert_stream_latch.cpp` installs
`{"expert_stream":{"enabled":true}}` with no variable set, leaves `mmap` unset so the
mutated form resolves OFF, and asserts `ResolveExpertStreamRequested() == true`.

**The prefault resolve stops latching, deliberately.** `PrefaultBorrowedSpan`
cached its answer in a function-local static. Dropping the cache costs one
`getenv` per prefaulted span, against the megabytes of pages the function then
reads, and it removes two defects: a config installed at load could be ignored by
whichever caller asked first, and the existing A/B case
(`tests/vllm/test_gguf_keep_quant.cpp`, "L7 load-time prefault is
byte-transparent") was **silently vacuous** — its second `setenv` could not
change an already-latched value, so both arms ran identically. The
expert-stream resolve keeps its latch, because that answer decides whether an
~18 GiB slot store is built and whether the grouped-MoE path is disabled, and
those two must not be able to disagree later in one process.

**Precedence, stated once and pinned by a test: environment variable > JSON
config > built-in default.** The environment keeps winning because several of
these variables exist so a benchmark arm is switchable without restarting with a
new config, and an A/B in flight depends on that. The config is the *documented*
surface; the environment is the *override*. Any other polarity would break a
running measurement to make a document tidier.

**`VT_MOE_EXPERT_STREAM_STATS_EVERY` stays environment-only, deliberately.**
Every other knob here changes what memory the process reserves or where a weight
lives — a deployment decision, which is what a config file is for. This one
changes only how often a diagnostic line is printed to stderr. It moves no byte,
reserves nothing, and changes no number; it is the instrument, not the
configuration. Putting it in the config surface would invite it into deployment
manifests where it means nothing, and it is exactly the kind of thing an operator
flips while staring at a run. Recorded here so a later reader sees a decision
rather than an omission.

**`slot_bytes` is IN, and that is not obvious.** It looks like an internal sizing
detail, but the code refuses a slice that exceeds it **by name** and tells the
operator to raise it (`qwen3_5.cpp:5223-5227`), and a dynamic (UD) quant is
precisely the case where the computed default is wrong. A knob a documented error
message tells you to change is a user surface.

## Tests to port

There is nothing to port: upstream has no disk tier, so there is no upstream test
for this surface. Upstream's `tests/basic_correctness/test_cpu_offload.py:19-21`
covers the mirrored tier only, and `test_offload_config.cpp` already carries it.

New tests, each red before its implementation:

- `tests/vllm/config/test_weight_residency_config.cpp`
  - the parser: every field, an absent `vllm_cpp` key, an empty document, a
    `vllm_cpp` that is not an object, an unknown key inside `vllm_cpp`, an
    unknown key inside `mmap`/`expert_stream`, a wrong-typed field, a
    non-positive `slots`/`slot_bytes`;
  - the mirror is untouched: the SAME document that carries a `vllm_cpp` key
    still parses through `parse_offload_config_json` to a byte-identical
    `OffloadConfig`, and a document with only `vllm_cpp` leaves the offload
    config inert;
  - **precedence**: env-set + config-set ⇒ env; env-unset + config-set ⇒ config;
    both unset ⇒ built-in default; env set to `0` beats a config `true` (an
    override has to be able to turn a thing OFF, which is the direction a
    benchmark arm usually needs);
  - **the latch, and its scope**: an install that would CHANGE `expert_stream`
    after `ResolveExpertStreamRequested` has read it throws, as does one that would
    change `slots`/`slot_bytes` after the store's geometry was noted; an install
    that touches only `mmap`/`prefault` does not, whatever has been resolved,
    because neither of those freezes anything and refusing them would fail a legal
    two-model load; an empty install does not (it is the no-op the default path
    performs); a re-install of the same config does not.
- `tests/vllm/config/test_expert_stream_latch.cpp` — the expert-stream knob's own
  binary, because `ResolveExpertStreamRequested` caches its answer and a process can
  therefore observe what it resolved exactly once. It installs
  `{"expert_stream":{"enabled":true}}` through the parser with no variable set and
  `mmap` deliberately unset, and asserts the resolver returns true and marked its
  latch. Added by the #1122 repair, after rewiring the resolver to the adjacent
  `.mmap` left all four other suites green.
- `tests/vllm/entrypoints/test_weight_residency_reach.cpp` — reachability
  through `LoadedEngine::FromModelDir` on a nonexistent model directory: the
  install happens before the load fails, so the process-global carries the
  parsed values afterwards. It also carries the TWO-MODEL case: a first load's
  ordinary `GgufLoadPolicy::FromEnv()` must not stop a second engine installing a
  document, because reading a knob is not taking a decision.
- `tests/vllm/entrypoints/openai/test_serve_residency_config.cpp` — reachability
  through the REAL `VllmServerMain`, re-exec'ing the test binary as
  `test_serve_recipe_args.cpp` does, asserting the install line on the child's
  stderr for a `--offload-config` document carrying only a `vllm_cpp` key. This
  is the test the reachability mutation deletes the call site under.

**Two existing suites gain the observable their knob never had.** Both were found
by the mutation pass, not by reading, and both are gaps that predate this row —
the old inline `getenv` calls were equally unwatched:

- `tests/vllm/test_gguf_keep_quant.cpp`, the prefault case. With the prefault site
  mutated to never consult its resolver, `test_gguf_keep_quant` (39/39),
  `test_gguf_qwen36_loader` (6/6) and `test_gguf_expert_span` (11/11) all stayed
  GREEN, because byte-transparency is equally true of a prefault that never ran.
  A span counter (`GgufPrefaultedSpanCount`) is the observable: the operation reads
  pages and writes nothing, so there is nothing else to see. Its two arms also
  genuinely differ now that the site no longer latches.
- `tests/vllm/model_executor/test_expert_stream_mixed_slot.cpp`, the slot count.
  It set `VT_MOE_EXPERT_STREAM_SLOTS=64`, which is the default, so the value was
  indistinguishable from the site ignoring the variable — mutating the site to a
  hardcoded 64 left it green. It now sets a non-default 96 and asserts the geometry
  the store was built with (`BuiltExpertStreamGeometry`).

## Gates

Correctness only. This row moves the *source* of a value; it changes no kernel,
no dtype, no allocation and no token, so it has no throughput axis of its own and
claims none.

1. Full gate, the documented CPU recipe:

   ```sh
   cmake -S . -B build -G Ninja
   cmake --build build -j 8
   ctest --test-dir build -j 6
   ```

   Exit 0, and no case-count regression against the pre-change baseline taken on
   the same tree.
2. The focused suites, by name, green with non-zero case counts, each red first:

   ```sh
   ./build/tests/test_weight_residency_config
   ./build/tests/test_expert_stream_latch
   ./build/tests/test_weight_residency_reach
   ./build/tests/test_serve_residency_config
   ./build/tests/test_gguf_keep_quant
   ./build/tests/test_expert_stream_mixed_slot
   ```
3. Every guarantee mutation-proven: for each added test, delete or invert the
   behavior it names, rebuild, require its suite red with a non-zero case count,
   restore by byte copy and verify by sha256. A mutation that fails to compile is
   INVALID, not a pass, and the compile status is recorded beside every row.
4. The reachability mutation: delete the install call site in
   `LoadedEngine::FromModelDir` in a scratch copy and require the server-level
   suite red.
5. `scripts/agent-preflight.sh --staged` green, including `check-env-doc`
   (the environment table is edited here) and `check-agent-record`.
6. Inertness: with no `vllm_cpp` key and no environment variable set, the
   resolvers return exactly what `getenv` returned before. Pinned by the
   both-unset precedence case rather than asserted.

## Dependencies

- `ENG-WEIGHT-OFFLOAD` ([#797](https://github.com/mudler/vllm.cpp/issues/797))
  owns `OffloadConfig` and its mirror. This row must not touch it, and the
  "mirror is untouched" test case is what keeps that honest. If a future vLLM
  release adds a disk arm upstream, the reconciliation is that row's: the
  extension is then superseded by a mirrored field and this row's spec records
  the migration.
- `ENG-EXPERT-STREAM` ([#912](https://github.com/mudler/vllm.cpp/issues/912))
  owns the streaming mechanism, and
  [`specs/expert-streaming.md:193`](expert-streaming.md) already anticipated
  routing through `offload_config` for exactly this reason. This row does that
  and changes nothing about the mechanism.
- No hardware dependency for the gate. The measured big-model reproduction is a
  GB10 job and is owed (see `## Owed`).

## Work breakdown

One wave. The change is one field wide at every hop, and splitting it would land
a parser nothing reaches.

| Step | Content |
|---|---|
| W1a | The struct, the parser, the install/resolve seam, and the config suite — red first |
| W1b | The three resolve sites, switched from `getenv` to the resolver |
| W1c | `EngineParams`, `server_main.cpp`, the C ABI, the install site, and both reachability suites |
| W1d | `docs/USAGE.md`, `docs/ENVIRONMENT.md` (including the #1109 correction), the records |

## Risks and decisions

- **A namespaced key inside a mirrored flag can read as "vLLM takes this".** It
  does not. The key is literally `vllm_cpp`, which is the cheapest possible
  signal that the contents are not upstream, and the docs say so in the same
  sentence they introduce it. The alternative — a second flag — costs a user two
  flags for one concept and still needs the same disclaimer.
- **Env-wins precedence can surprise a user whose config is being overridden by
  an exported variable they forgot.** Accepted, and mitigated where it is
  cheapest to see: the install prints the DOCUMENT it installed on one line and, on a
  second, every variable that would win over a field of it, so a run whose config was
  overridden says so on stderr at startup. It does not print resolved values — the
  streaming answer is cached the first time it is asked, so resolving it at install
  would move that decision ahead of the weight load. That constraint binds
  `expert_stream` alone; `prefault` and `slots` could be resolved at install and
  `mmap`/`slot_bytes` need a default only their caller has, so reporting the document
  for all five is a consistency choice on top of the one real constraint. The opposite
  precedence polarity was rejected because it breaks a measurement in flight.
- **The late-install throw must not fire on a legal second load, and the first
  shape of it did.** A second `FromModelDir` in one process cannot be honored for a
  field a taken decision has already frozen — `expert_stream` once the streaming
  answer has been read, the two sizes once the store is built — so throwing for
  those is the correct answer. It is NOT correct for `mmap` or `prefault`:
  `GgufLoadPolicy::FromEnv()` runs per load and this row removed the prefault
  static, so nothing about them is fixed. The first implementation marked one
  process-wide flag inside the shared resolvers, so the ordinary `FromEnv()` of a
  first load made a second engine's whole document throw — measured through
  `vllm_engine_load` (#1122 M1). The refusal is therefore scoped per latched field:
  `ResidencyLatch::kExpertStream` is marked by `ResolveExpertStreamRequested`,
  `kExpertStreamGeometry` by `NoteExpertStreamGeometry` when the store is built, and
  the shared resolvers mark nothing. An equal re-install and an empty install remain
  accepted, the first so the ordinary two-engine test binaries do not trip on it.
- **It fired on a legal second load a second time, and the same misuse of `optional`
  also silently dropped the first engine's fields.** Both are one root cause: an
  absent field means UNCHANGED, and the code meant two other things by it (#1133 H1,
  H2, both measured through `LoadedEngine::FromModelDir`). `FrozenFields` compared
  `in.expert_stream` against the stored optional, and `nullopt != engaged` is true, so
  `{"vllm_cpp":{"mmap":{"enabled":true}}}` on a second engine threw
  `std::logic_error` and returned `VLLM_ERR_MODEL_LOAD` — while the message asserted
  the engine was not running the configuration it was in fact running. The install
  then did `g.config = config`, so a partial second document turned
  `expert_stream=on slots=8000` into OFF with 64, with no diagnostic and read lazily
  by a store built long afterwards. Repaired as ONE thing: a field is scored only
  when the document SETS it, the comparison is against the DECISION taken (resolved
  through the same rule the production resolver uses, so a document the environment
  overrides is not a change either), and the install MERGES field by field. The
  wholesale replace predated the per-field narrowing; the narrowing widened its reach,
  because before it any differing document after a decision threw. What made both
  survive a fresh review and eight mutations was the test shape: every latch case
  installed a COPY of the first document or the empty one, so the second install always
  restated the frozen field at the value it already had. "Two DIFFERENT partial
  documents in one process" is the case shape that distinguishes them, at the unit
  level and again through `FromModelDir` and `vllm_engine_load`.
- **The unknown-key enumeration stopped one level short of what the ABI promised.**
  `include/vllm.h` claimed an unknown key ANYWHERE in the document is refused, and the
  top level plus the inside of `vllm_cpp` were closed while the inside of the mirrored
  `uva` and `prefetch` objects was not: `{"uva":{"cpu_offload_GB":10}}` and
  `{"prefetch":{"offload_groupsize":8}}` were ACCEPTED, giving a 0 GiB budget or a
  group size of 0 under a document the operator believes configures offloading
  (#1133 H3). Closed rather than scoped down, because refusing is the mirror-faithful
  polarity here too: `UVAOffloadConfig` (`offload.py:15-16`) and
  `PrefetchOffloadConfig` (`:47-48`) each carry `@config`, whose body sets
  `ConfigDict(extra="forbid")` (`utils.py:68-69`), so upstream refuses a nested typo
  and the tolerance was the deviation. The enumeration stays in the EXTENSION parser
  and lists names only: `parse_offload_config_json` is untouched and keeps sole
  ownership of those fields' types, defaults and bounds, so no type rule exists in two
  places.
- **`GgufLoadPolicy::FromEnv` keeps its name while no longer being env-only.**
  Renaming it touches 30 call sites across tests and four model loaders for no
  behavior change, which is a bigger diff than this row's whole subject. The
  function's doc comment states that it resolves env-over-config-over-default,
  and the resolver it calls is named for what it does. Recorded as debt, not
  hidden.

## Evidence

CPU host, documented recipe (`cmake -S . -B build -G Ninja`, no build type, so
asserts are live), 20 cores. Executables are linked with `-Wl,-s`; that strips
symbol tables only, and it was needed because the box had 8 GB free at the time.

**Red first.** Every declaration in the new header was stubbed — parser returns an
empty config, install records nothing, every resolver ignores the config — and the
three suites were built and run against it. All three RED for the intended reason,
with non-zero case counts: `test_weight_residency_config` rc 1, 11 cases / 1
passed / 10 failed, 111 assertions / 53 failed; `test_weight_residency_reach` rc
1, 4 cases / 3 failed; `test_serve_residency_config` rc 1, 5 cases / 3 failed.
Then the real implementation was restored (sha256
`fa1373d2e6acf2ead050bd1a435dafa52abe3351263e0d0be09068d1aa45103a`, verified),
rebuilt, and all three went GREEN.

**Mutations.** Each applied alone, the file's sha256 printed before and after so a
never-applied edit cannot read as a pass, the build's own exit status printed
beside every result so a non-building mutation is INVALID rather than a pass, a
non-zero doctest case count required, and the tree restored by byte copy from a
pristine snapshot with the sha256 compared against the pre-mutation value.

| # | Mutation | Result |
|---|---|---|
| M1 | delete the install call site in `FromModelDir` (the reachability mutation) | reach RED 5 cases / 2 failed; serve RED 5 / 3 |
| M2 | the server flag parses the extension and drops it | serve RED 5 / 3 |
| M3 | the C ABI parses the extension and drops it | reach RED 5 / 1 |
| M4 | unknown-key refusal removed (ignore them, as the mirrored parser does) | config RED 11 / 1; reach RED 5 / 1; serve RED 5 / 1 |
| M5 | precedence inverted (config beats the environment) | config RED 11 / 2; reach RED 5 / 1 |
| M6 | the late-install throw removed (accept and silently ignore) | config RED 11 / 2 |
| M7 | the expert-stream first-character rule normalised onto whole-value | config RED 11 / 1 |
| M8 | `ResolveGgufMmap` drops the config half | config RED 11 / 1; reach RED 5 / 1 |
| M9 | `ResolveGgufPrefault` drops the config half | config RED 11 / 1 |
| M10 | a non-positive `slots`/`slot_bytes` is accepted instead of refused | config RED 11 / 1; reach RED 5 / 1 |
| M11 | the mmap SITE reverts to `EnvOnOr`, bypassing the resolver | reach RED 5 / 1 (`test_gguf_keep_quant` GREEN, it has no config arm) |
| M12 | the prefault SITE never consults its resolver | **first run GREEN across three suites** (39/39, 6/6, 11/11); RED 39 / 1 after the span counter was added |
| M13 | the expert-stream SITE hardcoded off | `test_expert_stream_mixed_slot` RED 1 / 1 |
| M14 | the slot-count SITE hardcoded to the default 64 | **first run GREEN** 1/1; RED 1 / 1 after the suite was moved to a non-default 96 and made to assert the built geometry |
| M15 | the slot-bytes SITE bypasses its resolver | GREEN, and correctly so: with neither the variable nor the config set the resolver returns the caller's computed default, so the mutation is behaviour-identical. The defect-detecting forms are M10 and M16 |
| M16 | `ResolveExpertStreamSlots` drops the config half | config RED 11 / 1 |
| M17 | the latch is never marked, so a late install is accepted | config RED 11 / 2 |

A fourth finding came from reading rather than from a mutation, and is repaired
here too. `ResolveExpertStreamRequested` marked the latch under the process-wide
mutex on EVERY call, and that function sits on the per-expert-slice decode path
(`KqExpertSlice` -> `Qwen35ExpertStream::Get`). A row whose whole claim is that it
changes no kernel, no allocation and no performance axis must not put a lock in
the loop of the lane it exists to configure, so the flag is a relaxed
`std::atomic<bool>`. M6 and M17 were re-run against the atomic form and both stay
RED.

M1, M12 and M14 were findings rather than confirmations, and each was repaired in
this change rather than recorded: the log line now reads the installed global back
instead of `params`, the prefault gained a span counter, and the mixed-slot suite
gained a non-default slot count plus a geometry assertion.

**Focused green, first pass** (the repair pass's counts are below).
`test_weight_residency_config` 11 cases / 138 assertions,
`test_weight_residency_reach` 5 / 39, `test_serve_residency_config` 5 / 55 (1
skipped, the re-exec'd child), `test_gguf_keep_quant` 39 / 6093,
`test_expert_stream_mixed_slot` 1 / 181. All rc 0, all case counts read from the
LAST `test cases:` match so a failing child's summary cannot be mistaken for the
parent's, and every rc captured directly rather than through a pipe.

**Full gate.** See `## Now`.

### The fresh review, and the repair pass (#1122)

The fresh review returned **FAIL**: 2 high, 3 medium, 8 low, and **14 of 69
sentences about the code wrong**. It reproduced this table, including the three
findings above, and then found what the table could not see. The two that mattered:

- **H1.** `RejectUnknownKeys` enumerated only INSIDE `vllm_cpp`, so a misspelled
  TOP-LEVEL key was accepted with an empty config and no refusal, while the header
  said a typo "is an error". Measured through the product parser: `{"vllm-cpp":…}`,
  `{"VLLM_CPP":…}`, `{"vllm_ccp":…}` and `{"vllmcpp":…}` all ACCEPTED. The
  hyphenated spelling is the likeliest of all, because every flag around it is
  hyphenated. **Decision: make the claim true** rather than delete it. The
  precondition was checked first, as the finding required: there is no
  `--offload-config` flag anywhere in the vLLM tree at the pin, so the whole JSON
  document is vllm.cpp's own and no upstream-legal document can be broken; and vLLM
  builds its config dataclasses with `@config`, whose body sets
  `ConfigDict(extra="forbid")` (`vllm/config/utils.py:68-69`), so refusing is the
  mirror-faithful polarity and the tolerance was the deviation. The enumeration
  lives in the EXTENSION parser, which reads the same string at both entry points,
  so `parse_offload_config_json` stays a byte-faithful transcription.
- **M2.** Nothing gated the headline knob reaching its decision: rewiring
  `ResolveExpertStreamRequested` to read `.mmap` instead of `.expert_stream` left
  all four suites green. Repaired with a binary of its own, which the
  once-per-process observability of that latch requires (see Port map).

**M1 was a decision, not only a repair.** The late-install throw fired on a
legitimate two-model process — load A with no residency config, load B carrying
`vllm_cpp`, and B could not load — and its stated reason was false for two of the
five knobs. **Decision: narrow it**, because a hard failure on a legal load is
worse than the thing it prevents, and because the reason can then be true. The
latch became per-decision (`ResidencyLatch::kExpertStream`,
`kExpertStreamGeometry`), the shared resolvers mark nothing, and the refusal fires
only on a field a taken decision has frozen. The alternative — keep the throw and
correct three sentences plus a test name — would have kept a refusal that no longer
had a reason for `mmap` or `prefault`.

**M3 was a sentence, not a mechanism.** The install line prints the installed
DOCUMENT, and reading the global back is what proves the install ran; three
sentences claimed it reported what the engine will use. The sentences are corrected
and the mechanism is unchanged.

Mutations for the repair, each applied alone, with `git diff --stat` and the
file's sha256 printed before and after so a never-applied edit cannot read as a
pass, the build's exit status and an ENOSPC count printed beside every result so a
non-building mutation is INVALID rather than a pass, the LAST `test cases:` match
read, every rc captured directly, and the tree restored by byte copy with the
sha256 compared:

| # | Mutation | Result |
|---|---|---|
| N1 | `ResolveExpertStreamRequested` reads `.mmap`, the adjacent field (the surviving mutation) | latch RED 1 / 1, 9 assertions / 2 failed |
| N2 | the top-level enumeration removed | config RED 14 / 1 (10 assertions), reach RED 6 / 1, serve RED 6 / 1 |
| N3 | the top-level path built from a hardcoded `vllm_cpp` prefix | config RED 14 / 1 |
| N4 | the coarse latch restored, marked inside `ResolveResidencyBool` | config RED 14 / 1, reach RED 6 / 1 |
| N5 | the geometry latch never marked when the store is built | config RED 14 / 1 |
| N6 | `DescribeEnvOverrides` reports env PRESENCE again instead of "would win" | config RED 14 / 1 (4 assertions) |
| N7 | the install call site in `FromModelDir` deleted (the reachability mutation, re-run) | reach RED 6 / 3, serve RED 6 / 3 |
| N8 | the narrowed refusal never throws | config RED 14 / 2, latch RED 1 / 1 |

N4 first left the reach suite GREEN while the config suite went red, because the
per-field comparison is insensitive to WHERE the flag is marked unless the incoming
document changes the frozen field. The reach case was widened to carry
`expert_stream` as well — which is the shape the reviewer actually measured, a
second engine arriving with a `vllm_cpp` document — and it then goes red too. The
first result is recorded because it is the interesting one: a green mutation is a
statement about the test, not about the code.

**L3 has no mutation, and that is a limit rather than an omission.**
`ActiveWeightResidencyConfig` returning by value instead of by reference removes an
unsynchronised read behind a lock that looked like it covered one; a single-threaded
suite cannot distinguish the two, and a test that could would be a race detector
rather than a gate. The relaxed-atomic reason (L2) is the same shape: the code was
already sound and only the stated reason was wrong, so the repair is the sentence.
The comment now also records the window that ordering cannot close — a second engine
installing at the instant a first starts streaming — which the narrowing made
reachable in principle.

**Focused green after the repair.** `test_weight_residency_config` 14 cases / 188
assertions, `test_expert_stream_latch` 1 / 9, `test_weight_residency_reach` 6 / 54,
`test_serve_residency_config` 6 / 64 (1 skipped, the re-exec'd child). All rc 0.

### The second fresh review, and the second repair pass (#1133)

**FAIL again**: 2 behaviour defects, 1 false ABI guarantee, 2 medium, 8 low, and 21
of 55 sentences about the code wrong. Both behaviour defects were ONE root cause —
"absent means unchanged", implemented as "absent is a change" in `FrozenFields` and
as "absent is a clear" in the install — and both are recorded under Risks above.

**Red first, and each red is the review's own measurement reproduced through product
code.** Seven documents with a typo inside `uva` or `prefetch` were added to the
parser's `refused[]` list and each was ACCEPTED (`CHECK_THROWS_AS ... did NOT throw at
all`), which is H3. The new two-partial-documents case threw
`std::logic_error: expert_stream, expert_stream_slots cannot be changed ... (mmap=on
prefault=off expert_stream=on expert_stream_slots=8000) ... accepting this would record
a configuration the engine is not running` on an `mmap`-only second document, which is
H1 — and the message is the sentence the review called false. The reach case, driven
through `vllm_engine_load`, found `mmap` still at engine A's `true` after engine B
asked for `false`, which is H1 at the production entry point and the proof that the
return code cannot discriminate. `test_weight_residency_config` rc 1, 16 cases / 3
failed / 19 assertions failed; `test_weight_residency_reach` rc 1, 7 cases / 1 failed.

**Sixteen mutations for the second repair.** Same harness discipline as above, with one
addition: `git diff --stat` is printed beside the sha256 pair, because a
never-applied edit and an applied one can share an unchanged-looking result line.
Every row below reports `applied`, the build's own rc, an ENOSPC count, a NON-ZERO
doctest case count taken from the LAST `test cases:` match, and each binary's rc
captured directly. `config` = `test_weight_residency_config` (17 cases, 250 assertions green),
`reach` = `test_weight_residency_reach` (7), `latch` = `test_expert_stream_latch` (1),
`serve` = `test_serve_residency_config` (6), `mixed` =
`test_expert_stream_mixed_slot` (1).

| # | Mutation | Result |
|---|---|---|
| R1 | `FrozenFields` compares `in.expert_stream` against the stored optional again (the H1 defect) | config RED 3 failed; reach RED 1; latch GREEN (it never installs a partial document) |
| R2 | the comparison targets the stored document instead of the decision | config RED 2 (the AGREES and ENVIRONMENT cases, exactly the two guarantees it targets) |
| R3 | the install replaces wholesale instead of merging (the H2 defect) | config RED 2; reach RED 1 |
| R4 | the nested `uva` enumeration removed | config RED 1 |
| R5 | the nested `prefetch` enumeration removed | config RED 1 |
| R6 | the refusal never fires | **INVALID**: `-Werror=unused-function` on `FrozenFields`. Re-run as R6b keeping the call and discarding the result: config RED 5, latch RED 1 |
| R7 | the store never records the geometry it built | config RED 3; mixed RED 1 |
| R8 | the flag records the fact but the wrong answer (always "decided off") | config RED 4; latch RED 1 |
| R9 | the decision is never recorded at all | config RED 4; latch RED 1 |
| R10 | `FrozenFields` ignores `VT_MOE_EXPERT_STREAM` | config RED 1 |
| R11 | the count comparison ignores `VT_MOE_EXPERT_STREAM_SLOTS` | config RED 1 |
| R12 | the refusal message quotes the stored document again | **INVALID** (`-Werror` on `DecisionSummary`), then **GREEN** as R12b — see below — then RED 1 as R12c |
| R13 | the install call site in `FromModelDir` deleted (the reachability mutation) | reach RED 4; serve RED 3 |
| R14 | `ResolveExpertStreamRequested` reads `.mmap`, the adjacent field (round two's survivor, re-run) | latch RED 1; config RED 1 |
| R15 | the reset leaves a stale built geometry | **GREEN**, and recorded as such — see below |
| R16 | the nested `uva` path built from a phantom `vllm_cpp` prefix | config RED 1 |

**R12 is the one worth reading.** The first message assertion was
`Mentions(e.what(), "expert_stream_slots=8000")`, and it passed with the mutation
applied: at that point the stored document also held `expert_stream_slots=8000`, so
quoting the document produced a string containing the same substring. The assertion
was insensitive, not the code wrong. It moved to the one moment where the two differ —
a refusal taken while nothing is stored, where quoting the document produced
`environment/default` — and now asserts the decision is named and that phrase is
absent. R12c is that mutation against that assertion.

**R15 is GREEN and correctly so.** The reset clearing `built_geometry` is coherence
with `BuiltExpertStreamGeometry()`'s own "both zero until something builds one", not a
behavioural guarantee: the numbers are read only while the geometry latch is set and
the reset clears that too, so nothing can observe the difference. The comment says so
in the code rather than leaving a reader to assume a gate exists. A green mutation is a
statement about the tests, not about the code.

**TEN stale or imprecise `path:line` anchors in this spec, found by checking them last
against the final tree, as the round required.** All ten were already wrong at the
reviewed head `c7fa7084b`, so none is damage the repair did; it is the class
`ENG-RECORD-ANCHOR-RATCHET` (#632) exists for, found in this row's own Port map. For
`qwen3_5.cpp` and `include/vllm.h` that was verified by reading the same line numbers
out of `git show HEAD~1:<path>`; for `gguf_keep_quant.cpp`,
`include/vllm/entrypoints/model_loader.h` and `qwen3_5_gguf_weights.cpp` it follows
from those files being untouched by this change, so their lines cannot have moved. The
count was written as eight before this paragraph was audited against the actual
corrections, which is the same defect class the round exists to stop.

The Port map's four `VT_MOE_EXPERT_STREAM*` rows pointed at a comment line and at three
lines of the statistics printer rather than at the constructor beside them;
`slot_bytes`' "refuses by name" citation pointed at magic-static commentary;
`GgufLoadPolicy::FromEnv` was cited 85 lines past itself; the
`EngineParams::offload_config` field was off by one; `include/vllm.h:436` fell
mid-sentence in a paragraph rather than on the field; the `FromModelDir` range started
one line late and ended 30 lines early; and the llama.cpp-provenance range opened on a
blank line. Corrected to `qwen3_5.cpp:5149`, `:5470`, `:5471`, `:5476`, `:5223-5227`,
`model_loader.cpp:1255-1318`, `gguf_keep_quant.cpp:163`, `model_loader.h:156`,
`qwen3_5_gguf_weights.cpp:28-36`, and `include/vllm.h:413-470` plus
`:471`. The four upstream anchors this change ADDS were verified in the local checkout
at the pin before being written: `offload.py:15-16` and `:47-48` are the two `@config`
decorators, `utils.py:68-69` is the comment and the `ConfigDict(extra="forbid")` line,
and the six mirrored field names sit at `offload.py:23`, `:34`, `:54`, `:62`, `:66`,
`:70`.

**Three stale anchors OUTSIDE this row are left alone, deliberately, and named so
nobody has to rediscover them.** `qwen3_5.cpp:8907` cites `qwen3_5.cpp:7151` for
`BuildPaddedDecode`, which is at `:9303`. Four places cite `include/vllm.h:912` for
`vllm_video_params::ref_video`, which is at `:972`. `docs/USAGE.md:962` cites
`include/vllm.h:1072`. All three were already stale at `c7fa7084b` — 53 lines off for
`ref_video` there — and each belongs to another row's prose (spec decode, `LTX25-RETAKE`,
MiniMax-H3). This change makes the last two worse by exactly the seven lines its
`include/vllm.h` prose repair adds, and the first by six, which is unavoidable: any edit
to a file shifts every bare `path:line` citation below it, and that is #632's whole
subject. Correcting three other rows' comments would widen this diff into three
subsystems it has no reason to touch, so they are reported rather than edited.

**Two low findings have no mutation, and the reason is that they are prose.** L3's
stale `ctest` count is in a landed commit body, which cannot be corrected in place
without rewriting the reviewed head; the correct count is in this change's body and in
the pull request body, which is what the squash lands. The other prose repairs (M1,
M2, L1, L2, L4-L7) change comments and documents, and their gate is a reader.

**Focused green after the second repair.** `test_weight_residency_config` 17 cases /
250 assertions, `test_weight_residency_reach` 7 / 76, `test_expert_stream_latch` 1 / 9,
`test_serve_residency_config` 6 / 64 (1 skipped, the re-exec'd child). All rc 0, every
count from the last `test cases:` match.

## Owed

- **The measured big-model reproduction.** The headline case
  (`Qwen3.8-2.4T-A95B UD-Q1_0`, 370 GiB on a 119 GB GB10) has to be re-run once
  through the config surface to prove the JSON form reaches the same state the
  environment form did, with the startup line and the `[expert-stream]`
  statistics as the evidence. It was NOT run for this row: dgx.casa is
  unreachable at the SSH layer (TCP/22 accepts, then times out during banner
  exchange), and everything else here is CPU-local. Owned by
  `ENG-RESIDENCY-CONFIG`, issue
  [#1110](https://github.com/mudler/vllm.cpp/issues/1110).
- **`GgufLoadPolicy::FromEnv`'s name.** See Risks. Owned by this row.
- **A residency-level observation of the prefault.** The span counter this row
  added proves the prefault RAN or did not; it does not prove the pages ended up
  resident. `test_load_direct_upload` already does the harder version with
  `mincore()` over a host mirror, and the same instrument would apply here. Owned
  by `ENG-RESIDENCY-CONFIG`, issue
  [#1110](https://github.com/mudler/vllm.cpp/issues/1110).
- **The config form does not reach three entry points, and two of them are
  server-side.** `--offload-config` is parsed once, after the architecture
  resolution, so the server's POOLING/embedding path
  (`server_main.cpp`, the `if (pooling_model)` block) and its transcription-only
  path build their `EngineParams` without it — the MIRRORED `uva`/`prefetch` half is
  dropped there too, and has been since before this key existed, so this is a
  pre-existing gap that the new key inherits rather than a regression. `vllm-cli`
  has no such flag at all, which is a deliberate scope line (this row adds no new
  flag) but is unrecorded. Both are documented in `docs/USAGE.md` beside the config
  form so a reader is not left to discover it. Fixing the server half means moving
  the offload parse ahead of the architecture branch, which is
  `ENG-WEIGHT-OFFLOAD`'s surface as much as this one's. Owned by
  `ENG-RESIDENCY-CONFIG`, issue
  [#1135](https://github.com/mudler/vllm.cpp/issues/1135) — filed for this gap
  specifically, because it was previously listed against
  [#1122](https://github.com/mudler/vllm.cpp/issues/1122), the review issue this pull
  request closes, so on landing the gap would have had no open issue (#1133 L8).

## Now

`ACTIVE`. The config surface exists, parses, refuses a typo ANYWHERE in the
document, defines its precedence, and is reachable from both production entry
points — the server flag and the C ABI — with the reachability proven by deleting
the install call site and watching the server-level suite go red. The mmap,
prefault, expert-stream, slots and slot-bytes knobs all resolve through it, and the
expert-stream knob's own wiring to its own field now has a gate of its own.
`stats_every` stays environment-only by decision, recorded in Port map.
`docs/ENVIRONMENT.md`'s `VT_GGUF_PREFAULT` default is corrected here, closing
#1109. The first fresh review's findings (#1122) are repaired: the top-level typo is
refused, the late-install refusal is scoped to the two decisions that genuinely
freeze, and the public ABI comment says which half of `offload_config` moves
weights.

The second fresh review's findings (#1133) are repaired too, and its two behaviour
defects were one root cause. "Absent means unchanged" now binds both halves of the
install: a field is scored frozen only when the document SETS it and only when the
value would resolve differently from the decision already taken, and the stored config
is MERGED field by field. So a second engine's partial document installs instead of
throwing, and it no longer drops the first engine's fields. The unknown-key
enumeration reaches the mirrored `uva`/`prefetch` sub-objects, which makes the ABI's
"anywhere in the document" true. `#1135` now owns the unreached-entry-point gap that
was listed against the closing review issue.

What keeps it `ACTIVE` rather than `DONE` is what is above under `## Owed`: nobody
has yet driven the 370 GiB checkpoint through the JSON form on the box that can
hold it, and the config form still does not reach the pooling, transcription or
`vllm-cli` entry points.
