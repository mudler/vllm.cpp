# TOOLS-PARSER-BREADTH — close the tool-parser mirror, and record the 40 already shipped

Issue: [#608](https://github.com/mudler/vllm.cpp/issues/608)
Row: `TOOLS-PARSER-BREADTH` ([engine-matrix.md](../engine-matrix.md))
Sweep that found it: `roadmap_v1.md` § Recipe-surface sweep (2026-08-13)

## Two problems, and the record one comes first

The row reads:

> Qwen-Coder XML, Mistral, pythonic, and remaining parsers · Our code: `-` ·
> Our tests/evidence: `-` · `INVENTORIED` · spec `planned:`

All three parsers it names by title are **implemented**. `qwen3_coder`, `mistral`
and `pythonic` are all in
`src/vllm/entrypoints/openai/tool_parsers/abstract.cpp`, alongside 37 others —
**40 registered names** with **38 test files** under
`tests/vllm/entrypoints/openai/tool_parsers/`. The row records none of it, and
its spec was never written.

This is the shape `SAMPLE-REASONING` carried before its W0 ("the seam shipped
under `da933828` but this row was never advanced"). **Backfill precedes
extension**: without it, adding five parsers looks like a change that delivers
forty.

## The actual gap, from the registries — not from usage

Upstream `vllm/tool_parsers/__init__.py` at the pin `5559679` registers **43**
names. We register **40**. Exactly **five** are upstream-only:

| Name | Upstream module · class | Recipe uses |
|---|---|---:|
| `openai` | `gptoss_tool_parser` · `GptOssToolParser` | 2 |
| `inkling` | `inkling_tool_parser` · `InklingEngineToolParser` | 2 |
| `minimax_m3` | `minimax_m3_tool_parser` · `MinimaxM3ToolParser` | 1 |
| `cohere_command3` | `cohere_command_tool_parser` · `CohereCommand3ToolParser` | **0** |
| `cohere_command4` | `cohere_command_tool_parser` · `CohereCommand4ToolParser` | **0** |

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

So for four of five, "port its tests in the same change" has nothing to port. The
test is ours to author, its fidelity rests entirely on reading the parser source,
and it is recorded as from-scratch in `porting-inventory.md` §9. Say so on the
row rather than implying an inherited test.

### The harness is worth porting on its own

`ToolParserTestConfig` encodes what upstream considers the minimum bar for ANY
tool parser: no-tool-calls, single call, parallel calls, various data types,
empty arguments, surrounding text, escaped strings, and a list of malformed
inputs. We have no equivalent — our 38 test files are hand-written per parser, so
the floor differs per parser and nothing enforces one.

Porting that harness raises the floor for all 40 existing parsers and gives the
five new ones a bar that is upstream's rather than invented. That is W3 below; it
is the highest-value item here and deliberately does not block W1.

## Work breakdown

- **W0 — backfill the row to reality.** Record the 40 registered names, the
  factory anchor (`tool_parsers/abstract.cpp`), the autodetect table
  (`tool_parsers/detect.cpp`), and the 38 existing test files. Move the state off
  `INVENTORIED` to what that evidence backs, and reconcile the summary counts in
  the same change (`scripts/check-agent-record.py` is CI-enforced). No new parser.
- **W1 — the three with recipe demand**: `openai`, `inkling`, `minimax_m3`.
  `minimax_m3` ports its upstream test; the other two author one.
- **W2 — complete the mirror**: `cohere_command3`, `cohere_command4`.
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

`docs/USAGE.md:878` states the parser-name count and must move with each wave.

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

`SPIKE` — spec committed, implementation not started. Next: a fresh implementer
takes W0 alone, because a backfill mixed with a port cannot be reviewed cleanly.
