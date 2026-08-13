# TOOLS-PARSER-BREADTH — close the tool-parser mirror, and record the 41 already shipped

Issue: [#608](https://github.com/mudler/vllm.cpp/issues/608)
Row: `TOOLS-PARSER-BREADTH` ([engine-matrix.md](../engine-matrix.md))
Sweep that found it: `roadmap_v1.md` § Recipe-surface sweep (2026-08-13)

## Two problems, and the record one comes first

The row reads:

> Qwen-Coder XML, Mistral, pythonic, and remaining parsers · Our code: `-` ·
> Our tests/evidence: `-` · `INVENTORIED` · spec `planned:`

All three parsers it names by title are **implemented**. `qwen3_coder`, `mistral`
and `pythonic` are all in
`src/vllm/entrypoints/openai/tool_parsers/abstract.cpp`, alongside 38 others —
**41 registered names** over 37 families, with **38 test files** under
`tests/vllm/entrypoints/openai/tool_parsers/`. The row records none of it, and
its spec was never written. (This paragraph originally said 40; W0 re-derived it
and corrected it — see `## Now`.)

This is the shape `SAMPLE-REASONING` carried before its W0 ("the seam shipped
under `da933828` but this row was never advanced"). **Backfill precedes
extension**: without it, adding five parsers looks like a change that delivers
forty-one.

## The actual gap, from the registries — not from usage

Upstream `vllm/tool_parsers/__init__.py:24-201` at the pin `5559679` registers
**44** names. We register **41**. Exactly **five** are upstream-only (and two of
ours are not in upstream's registry there — see `## Now`):

| Name | Upstream module · class | Recipe uses | What backs it upstream |
|---|---|---:|---|
| `openai` | `gptoss_tool_parser` · `GptOssToolParser` | 2 | Harmony — the class is a declared **stub** that raises |
| `inkling` | `inkling_tool_parser` · `InklingEngineToolParser` | 2 | ParserEngine adapter — **the only one portable from vLLM source** |
| `minimax_m3` | `minimax_m3_tool_parser` · `MinimaxM3ToolParser` | 1 | the **Rust** tool-parser crate |
| `cohere_command3` | `cohere_command_tool_parser` · `CohereCommand3ToolParser` | **0** | out-of-tree **`cohere_melody`** package |
| `cohere_command4` | `cohere_command_tool_parser` · `CohereCommand4ToolParser` | **0** | out-of-tree **`cohere_melody`** package |

### Only one of the five is a port. Read this before scoping W1 or W2

"Upstream-only name" does not mean "an upstream text parser we have not typed
out yet". For four of these five there is **no grammar in vLLM's Python source
at all**. Verified by reading the pinned checkout at
`5559679229bc961848b121ccdeaa8fa5d79bec98`:

- **`openai` is Harmony-backed by explicit declaration.** `GptOssToolParser`
  (`vllm/tool_parsers/gptoss_tool_parser.py:17`) is a stub whose docstring says
  "All output parsing is handled by HarmonyParser. This stub exists as a
  capability declaration via HarmonyParser.tool_parser_cls." **Both** methods
  raise — `:31` and `:45` — with `NotImplementedError("GptOssToolParser is a
  stub. Use HarmonyParser for tool parsing.")`. Registering the name is not
  mirroring the behaviour; the behaviour lives in Harmony. **W1 correction:
  Harmony is IN vLLM** — `vllm/parser/harmony.py` (358 lines,
  `HarmonyParser(DelegatingParser)`) with `tests/parser/test_harmony.py` — so
  this is a delegation inside vLLM, not an absence of it. What is out-of-tree is
  only the `openai_harmony` package the wrapper drives, which makes this a
  W2-shaped grammar decision rather than a missing parser. See the W1 `openai`
  bullet for the secondary-oracle check this triggered and why it was refused.
- **Both Cohere classes are shims over `cohere_melody`.**
  `vllm/tool_parsers/cohere_command_tool_parser.py:6-13` imports
  `PyFilter`/`PyFilterOptions` from that out-of-tree package and raises a hard
  `ImportError` when it is absent; `BaseCohereCommandToolParser` (`:34`),
  `CohereCommand3ToolParser` (`:125`) and `CohereCommand4ToolParser` (`:138`)
  delegate all parsing to it. **Nothing in vLLM source describes the dialect.**
- **`minimax_m3` is Rust-backed.** `minimax_m3_tool_parser.py:7` subclasses
  `RustToolParser` and only sets `rust_parser_name = "MinimaxM3ToolParser"`
  (`:18`); the grammar lives in the Rust crate. Its class docstring and its
  upstream test are the only executable descriptions available to us.
- **`inkling` is a ParserEngine adapter** over `InklingParserToolAdapter`
  (`inkling_tool_parser.py:4,7`), and we already carry the `inkling` engine
  config. This is the one ordinary port of the five.

Three names the recipes DO reference are in **neither** registry —
`nemotron_json`, `kimi_k3`, `ling3`. They are post-pin and land with the pin
advance, not here.

**The two Cohere parsers are the point of doing this from the registry.** No
recipe references them, so a usage-driven audit cannot see them; they are a
mirror gap regardless. An earlier revision of #608 was built from recipe usage
alone and consequently listed `nemotron_json` as portable (it is not registered
anywhere at the pin) while missing both Cohere entries.

## Upstream has no test for four of the five

`tests/tool_parsers/` carries 43 files, but only **`minimax_m3`** among our five
has one (`test_minimax_m3_tool_parser.py`). There is no `gptoss`, `inkling`, or
`cohere` tool-parser test upstream.

`common_tests.py` does not close that: it is a **config-driven suite each parser
instantiates** via `ToolParserTestConfig` (`common_tests.py:17`), not a sweep over
`ToolParserManager`. A parser with no test file gets **zero** coverage from it.

**Corrected 2026-08-13 by W1: that is true of `tests/tool_parsers/` and false of
the repository.** `inkling`'s test lives in the OTHER place, next to the engine
it is built on — `tests/parser/engine/test_inkling.py` @ `5559679` — because
upstream's `InklingEngineToolParser` is a bare `make_adapters(InklingParser)`
subclass with no behaviour of its own to test. It ported, so `inkling` owed no
authored test after all. Searching one test directory is not searching upstream;
`gptoss` and `cohere` were re-checked the same way and genuinely have none.

So for three of five, "port its tests in the same change" has nothing to port.
The test is ours to author, its fidelity rests entirely on reading the parser
source, and it is recorded as from-scratch in `porting-inventory.md` §9. Say so
on the row rather than implying an inherited test.

### The harness is worth porting on its own

`ToolParserTestConfig` encodes what upstream considers the minimum bar for ANY
tool parser: no-tool-calls, single call, parallel calls, various data types,
empty arguments, surrounding text, escaped strings, and a list of malformed
inputs. We have no equivalent — our 38 test files are hand-written per parser, so
the floor differs per parser and nothing enforces one.

Porting that harness raises the floor for all 41 existing parsers and gives the
five new ones a bar that is upstream's rather than invented. That is W3 below; it
is the highest-value item here and deliberately does not block W1.

## Work breakdown

- **W0 — backfill the row to reality. DONE 2026-08-13, see `## Now`.** Record the
  41 registered names, the factory anchor (`tool_parsers/abstract.cpp`), the
  autodetect table (`tool_parsers/detect.cpp`), and the 38 existing test files.
  Move the state off `INVENTORIED` to what that evidence backs, and reconcile the
  summary counts in the same change (`scripts/check-agent-record.py` is
  CI-enforced). No new parser.
- **W1 — the three with recipe demand**, which are three different jobs, not one
  batch (see "Only one of the five is a port" above):
  - `inkling` — the ordinary one. **LANDED 2026-08-13; see `## Now`.** A
    ParserEngine adapter port over the `inkling` engine config we already carry.
    Two things this bullet had wrong, both corrected by doing it: the work was
    smaller than "port a parser" (the engine, its config and its arg carver were
    already ported and golden-gated; only the tool-parser REGISTRY face was
    missing, so `--tool-call-parser inkling` threw at startup), and the test was
    **not** authored from scratch — upstream has
    `tests/parser/engine/test_inkling.py`, which is the executable description of
    the dialect and ports directly.
  - `minimax_m3` — the grammar is in the Rust crate, so the implementation is
    written from the wire format its docstring describes and is recorded
    from-scratch in `porting-inventory.md` §9. Its upstream test
    (`test_minimax_m3_tool_parser.py`) ports and is the fidelity gate.
  - `openai` — **there is no gpt-oss text parser upstream to port.** The
    registry entry is a stub that raises on both methods. Mirroring the name
    alone ships a name that resolves and then refuses. Decide before writing
    code, and record the decision here: either mirror the Harmony seam (large,
    and a different row), or register the name with an explicit refusal naming
    the missing piece, per AGENTS.md's "an arm that is not implemented is
    refused with a message naming the missing piece and recorded as owed". Do
    **not** plan it as an ordinary text parser.

    **SECONDARY-ORACLE CHECK, run under AGENTS.md §"When vLLM has no
    implementation" and answered NO (W1, 2026-08-13).** The question was whether
    SGLang — a registered oracle (id `sglang`,
    [`.agents/oracles/sglang.md`](../oracles/sglang.md), pin
    `f63458b5beaceabbd9d749b9fc956370e1b649e6` / `v0.5.15`, pinned 2026-07-27;
    the local checkout at `/home/mudler/_git/sglang` is clean AT that SHA) —
    should be the source for gpt-oss tool parsing, since it carries
    `python/sglang/srt/parser/harmony_parser.py` and
    `test/registered/unit/parser/test_harmony_parser.py`. **It should not, and
    reaching for it would have been a rule violation.** That section admits a
    secondary oracle only "where it implements nothing" and only for "a path
    vLLM cannot produce at all". vLLM produces this one: `vllm/parser/harmony.py`
    is a 358-line vLLM-owned `HarmonyParser(DelegatingParser)` — it is what
    `GptOssToolParser`'s docstring defers TO — and it has its own upstream test,
    `tests/parser/test_harmony.py`. The stub is a delegation inside vLLM, not an
    absence of vLLM. **`openai` therefore mirrors `vllm/parser/harmony.py`, and
    SGLang is not consulted.**

    Two things the check did establish, and they are the real W1-remaining
    scoping:
    1. **The residual gap is the same SHAPE as Cohere's, not a missing parser.**
       `HarmonyParser` wraps the third-party `openai_harmony` package
       (`get_streamable_parser_for_assistant()` / `StreamableParser`), so the
       `<|channel|>` token grammar itself is out-of-tree, exactly as
       `cohere_melody` is for W2. vLLM's glue ports; the grammar under it is a
       from-scratch or vendor decision this spec still owes — and it is a W2-class
       decision, not the "register the name with an explicit refusal" the bullet
       above assumed.
    2. **`openai` is a HARMONY-SEAM row, which this spec already said is a
       different row.** It is materially larger than `inkling` was, which is why
       W1 shipped `inkling` alone.

    Had the answer been yes, SGLang would still have been **source-only**: its
    oracle file records `gateable = no` (evidence: #647) — no SGLang run has ever
    been recorded on this project's hardware — so no gate may be taken by
    executing it. Recorded here so the next agent does not re-run the check.
- **W2 — complete the mirror**: `cohere_command3`, `cohere_command4` — **also
  not a text port.** All the grammar lives in `cohere_melody`, an out-of-tree
  package we do not vendor and upstream does not ship; there is nothing in vLLM
  source to read, so "port its tests in the same change" and "cite the upstream
  `file:line` you ported from" both have no referent for the dialect itself. W2
  therefore owes a decision recorded here before code: reimplement the dialect
  from the model's own chat template / wire format (from-scratch, §9), or
  refuse both names explicitly and record the external dependency as the
  blocker. It stays in scope as a mirror gap either way.
- **W3 — port `ToolParserTestConfig`** as a shared C++ harness and retrofit it
  across the existing parsers, one batch per change.
- **Deferred to the pin advance**: `nemotron_json`, `kimi_k3`, `ling3`.

## Per-parser porting rules

Each parser is a direct text port into
`src/vllm/entrypoints/openai/tool_parsers/`, citing the upstream `file:line` it
came from, registered in `abstract.cpp` **and** added to `tool_parser_names()` in
the same change (that file's own comment requires it).

`detect.cpp` gets a marker row **only** where the family has a template-stable
literal. Read its `ORDER MATTERS` comment block first: it explains why generic
markers route to the safe default and why some families stay EXPLICIT-ONLY. A
parser whose marker collides with an existing row must stay explicit — a wrong
autodetect is worse than no autodetect, because it silently mis-parses a model
the user never named.

`docs/USAGE.md:1126` states the parser-name count (already correct at 41 over 37
families) and must move with each wave, as must the `docs/STATUS.md` row.
Re-derive that line rather than trusting it: it read `:902` when W0 was written
and rotted to `:1126` when this branch rebased onto `cefacd2d0` (#641), which
changed `docs/USAGE.md` by 572 lines. Nothing gates anchor line numbers (#632),
so a rebase moves them silently and a re-run preflight still passes.

## Gates

Focused: the new test file per parser. Full: `scripts/agent-preflight.sh
--staged` plus the serve conformance suite. No GPU and no oracle run — these are
text parsers.

Mutation: for each parser, break its extraction in a scratch copy and prove the
new test goes RED; restore the tree byte-for-byte. Report which assertion caught
it, and if an existing guard catches it first, say so rather than claiming the
new test did.

## Risks / decisions

- **Risk**: W0 makes the row look like a large jump with no code. It is a record
  repair, not an achievement, and the commit must say so — the alternative is a
  row that permanently understates the surface it owns.
- **Decision**: `cohere_command3/4` are in scope despite zero recipe demand.
  Upstream registers them and we mirror upstream; demand orders the work, it does
  not define the surface.
- **Decision**: `openai` is the registry name for `GptOssToolParser`. Keep
  upstream's name, however generic it reads — renaming it would break the
  recipes this whole sweep exists to make runnable.
- **Out of scope**: reasoning parsers (`SAMPLE-REASONING`, #605) and the
  accepted-and-inert serve flags (`SERVE-RECIPE-ARGS`, #606).

## Now

`PARTIAL` — **W0 and the `inkling` half of W1 have landed (2026-08-13, #608).**
The live position is **42 accepted names over 38 families**; the W1 subsection
below is the current front, and `openai` / `minimax_m3` / both Cohere names
remain owed, which is what keeps the row `PARTIAL`. The two paragraphs that
follow are W0's record, kept for provenance.

**W0 landed 2026-08-13 (#608).** The row now records the shipped
surface (41 names / 37 families, the factory and `tool_parser_names()` anchors,
the 27-row autodetect table, the 38 test files and the count-pinning registry
test), and moved `INVENTORIED` → `PARTIAL`; the engine-matrix summary counts and
`docs/STATUS.md` moved with it. No parser was ported and no `src/` behaviour
changed — this was a record repair.

`PARTIAL` rather than `ANCHOR-BACKFILL` because that state means "code and tests
but no leaf spike", and this spec is committed; what remains true is the other
half, "the implementation is also known to omit upstream behavior" — the five
upstream-only names below.

**W0 corrected three counts this spec had wrong**, from the registries at
`5559679`:

- We register **41** names, not 40. `muse_glimmer` took it 40 → 41 on
  2026-08-10 under `MODEL-MUSE-GLIMMER-W7`; `test_detect.cpp:221` already pinned
  41, so the spec was stale against a committed test.
- Upstream registers **44**, not 43.
- The five upstream-only names are confirmed exactly as listed. But **two of our
  names are absent from upstream's registry at the pin** — `qwen3` (a local
  alias) and `muse_glimmer`. 44 − 5 = 39 shared; 39 + 2 = 41. The earlier
  "40 = 43 − 3" arithmetic worked only because two errors cancelled.

`muse_glimmer` is not a drifted anchor but the **off-pin exception** recorded in
[porting-inventory.md](../porting-inventory.md) §16: it exists in NO vLLM
revision in the pin's ancestry (absent at `555967922` AND at the older
`e24d1b24`), and is decorator-registered only at
`muse_glimmer_tool_parser.py:183` on the unmerged
[vllm#51655](https://github.com/vllm-project/vllm/pull/51655), head `075d645af`.
So the honest reading of "44 vs 41" is that our 41 spans two upstream sources,
and a future pin advance that lands #51655 folds `muse_glimmer` into the shared
set rather than adding a name.

**W0's characterisation of the five was itself wrong, and is corrected here
(2026-08-13, review of #643).** W0 wrote "only `minimax_m3` is Rust-backed;
`inkling` is a ParserEngine adapter and `openai`/`cohere_command3`/
`cohere_command4` are plain Python". The COUNT correction it made stands — five
upstream-only names, not the "three Rust/Harmony-backed ones" `docs/STATUS.md`
claimed. The characterisation does not: **plain Python is not the same as
portable**, and `openai` plus both Cohere names delegate outside vLLM's source
entirely. `openai` is Harmony-backed by explicit declaration (a stub that raises
on both methods) and the Cohere pair are shims over the out-of-tree
`cohere_melody` package. Of the five, **only `inkling` is portable from vLLM
source**. See "Only one of the five is a port" above for the read anchors; W1
and W2 are scoped to that reading, not to the "plain Python" one.

**W1 in turn corrects one word of that.** "`openai` … delegate[s] outside vLLM's
source entirely" is too strong: the delegation target `vllm/parser/harmony.py`
IS vLLM source and is tested upstream; only the `openai_harmony` package under it
is out-of-tree. The conclusion the paragraph draws survives — of the five, only
`inkling` was a straight port, and it is the one W1 shipped.

### W1 · `inkling` LANDED 2026-08-13 (#608)

`--tool-call-parser inkling` now resolves. **41 → 42 accepted names, 37 → 38
families**, re-derived from `abstract.cpp` rather than incremented (42 names, 38
distinct factory classes, 27 marker rows unchanged); `test_detect.cpp:222`,
`docs/USAGE.md:1126`, `docs/STATUS.md` and the engine-matrix row moved with it.
`README.md` and `docs/FEATURES.md` carry no parser count, so neither is owed.

**The gap was smaller and stranger than the spec assumed.** The Inkling engine
was ALREADY fully ported — `src/vllm/parser/inkling.cpp`, `inkling_config()` with
the JSON-span arg carver in `engine/configs.cpp`, dispatched by
`parser::get_parser_engine("inkling")`, and golden-gated against the pinned
oracle in `test_parser_engine_assembly` (`inkling_think_tool_text_*`,
`inkling_nonobject_args_*`). What was missing was only the tool-parser REGISTRY
face, so `ResolveToolParserName("inkling", …)` threw at startup and a fully
ported, oracle-gated dialect was unreachable from the flag.

Landed:

- `tool_parsers/parser_engine_adapter.{h,cpp}` — `ParserEngineToolAdapter`
  (`vllm/parser/engine/adapters.py:128`, the TOOL half of `make_adapters`, twin
  of the reasoning adapter we already carried) + `InklingEngineToolParser`
  (`vllm/tool_parsers/inkling_tool_parser.py:7`).
- The factory branch and `tool_parser_names()` entry, in the same change, as
  `abstract.cpp`'s own comment demands.
- `ParserRequestFromChatCompletion` moved out of `serving_chat.cpp`'s anonymous
  namespace to sit next to `ParserRequest`, so the adapter and the serving path
  share ONE request projection instead of two copies of it.

**`inkling` is EXPLICIT-ONLY in `detect.cpp` — no autodetect row, and for a
reason no other row has.** Every other EXPLICIT-ONLY family is excluded for
marker collision; Inkling is excluded because there is nothing to sniff. That
table matches a CHAT TEMPLATE string, and Inkling has no jinja template at the
pin — rendering is `vllm/renderers/inkling_encoding.py` (mirrored by the Rust
`rust/src/chat/src/renderer/inkling/`), and `examples/` carries no
`tool_chat_template_inkling.jinja`. Its `<|content_invoke_tool_json|>` marker
collides with nothing, so a row would LOOK harmless while being unreachable
through the only input the function gets and claiming a template-stability we
cannot demonstrate.

**No structural-tag row is owed.** `inkling_tool_parser.py:10-11` sets
`structural_tag_model = None` / `supports_required_and_named = False`; our
`ToolChoiceStructuralTagSpecFor` already returns nullopt for every mode of an
unmapped family, which is that behaviour exactly.

**The test PORTS; it was not authored.** This spec said "test authored from
scratch" on the assumption that a missing `tests/tool_parsers/
test_inkling_tool_parser.py` meant no upstream test. There is one, in the other
place: `tests/parser/engine/test_inkling.py` @ `5559679`. **15 of its tool-facing
cases port** into `tests/vllm/entrypoints/openai/tool_parsers/test_inkling.cpp`
keeping their upstream names verbatim, with two documented harness adaptations (a
`<|message_model|>` prefix reproduces upstream's MESSAGE_HEADER initial state on
the CONTENT-seeded tool adapter; the streaming cases use upstream's
`_stream_text_only` character-chunk harness because our ToolParser seam is
text-only, which is a strictly stronger split). Its `TestArgConverter` class and
the token-id/reasoning cases are deliberately NOT re-ported: they gate the engine
layer, which the assembly goldens already gate against the oracle.

**3 of the 18 cases are AUTHORED**, recorded as from-scratch, each saying so at
its site:

- "the registry name resolves and is enumerated" and "EXPLICIT-ONLY — no
  autodetect row" gate OUR packaging surface, which has no upstream analogue to
  port: upstream's registry is a lazy dict and it has no chat-template marker
  table at all, so `DetectToolParser` is an ORIGINAL component.
- "the tool adapter seeds the engine in CONTENT state" gates UPSTREAM behaviour
  (`adapters.py:158/167`) that upstream's own suite never exercises, because it
  drives `InklingParser` directly and never constructs the adapter. It exists
  *because* a mutation survived without it.

These are test-side authorship on an otherwise straight port, recorded here and
at each case rather than as a new numbered entry in `porting-inventory.md` §9 —
that section enumerates forced STRUCTURAL deviations of the port itself, and this
change has none. Flagged for a reviewer who reads §9's scope more broadly.

Mutation results (each restored byte-for-byte, verified by sha256):

| Mutation | Caught by |
|---|---|
| `InklingEngineToolParser` builds a base `ParserEngine` over `inkling_config()` instead of `InklingParser` (drops the trailing-text flush hook, `inkling.py:376`) | NEW `test_inkling.cpp:248` `test_text_after_tool_call` |
| drop `"inkling"` from `tool_parser_names()`, keep the factory branch | NEW `test_inkling.cpp:138`, AND the pre-existing count pin `test_detect.cpp:222` — the existing guard catches it too |
| drop the adapter's `initialize_streaming(CONTENT)` seed | **SURVIVED** the first suite; caught only after adding the authored case, at `test_inkling.cpp:174` |
| `inkling_arg_converter` returns the raw `{"name":…,"args":…}` wrapper | NEW `test_inkling.cpp` (5 cases / 14 assertions) AND the pre-existing `test_parser_engine_assembly` goldens (3 cases / 33 assertions) — the existing gate catches this one first, since the carver is engine-layer |

### Still owed on this row

Next: the rest of W1 — `openai` (a `vllm/parser/harmony.py` port gated by
`tests/parser/test_harmony.py`, with the out-of-tree `openai_harmony` grammar
under it as a W2-shaped decision; the SGLang secondary-oracle check was run and
answered NO, see the W1 bullet) and `minimax_m3` (from the wire format, its
upstream test the fidelity gate) — then W2 (both Cohere), then W3 (the
`ToolParserTestConfig` harness). Each its own change and its own review.
