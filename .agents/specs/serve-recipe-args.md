# SERVE-RECIPE-ARGS — a published recipe command must reach model load

Issue: [#606](https://github.com/mudler/vllm.cpp/issues/606)
Row: `SERVE-RECIPE-ARGS` ([engine-matrix.md](../engine-matrix.md))
Sweep that found it: `roadmap_v1.md` § Recipe-surface sweep (2026-08-13)

## Scope

One seam in the serve entry point: an **enumerated** accepted-and-inert argument
table, its per-flag startup notice, and the mirrored `cli_args.py:395`
validation. Two entries to start — `--enable-auto-tool-choice` and
`--trust-remote-code`.

Explicitly NOT in scope, each for its own reason:

| Excluded | Why |
|---|---|
| `--language-model-only` | a real capability gap, tracked as [#607](https://github.com/mudler/vllm.cpp/issues/607) — accepting it would hide missing behaviour |
| the `--tool-call-parser` default (`hermes` vs upstream `None`) | pre-existing divergence, worth its own issue; changing it here would mix a behaviour change into a parsing change |
| `--tensor-parallel-size`, the EP flags, `--mm-encoder-tp-mode` | inert only because the CAPABILITY is missing; they must keep aborting |
| any change to a forward pass, kernel, or model path | none is reached — this is argument parsing |

## The defect

`vllm-serve` rejects any argument it does not recognise
(`src/vllm/entrypoints/openai/server_main.cpp:440`):

```cpp
std::cerr << "server: unknown argument '" << flag << "'\n";
```

That is the right default. It is also why two flags that mean *nothing* to us stop
the server before it loads a model. Measured over `vllm-project/recipes`
@ `86c7777aa699482ef1ebd0c5da9fc540ccc00a40`, 157 official model recipes:

| Flag | Recipes passing it | Why it is inert for us |
|---|---:|---|
| `--enable-auto-tool-choice` | **89 / 157** | we parse tool calls whenever `--tool-call-parser` resolves; there is no second gate to open |
| `--trust-remote-code` | **82 / 157** | authorizes executing Python from the checkpoint; we have no Python, so there is nothing to authorize |

The reproducer is a recipe's own copy-paste block —
`recipes/models/Qwen/Qwen3.5-27B.yaml`, `features.tool_calling`:

```bash
vllm serve Qwen/Qwen3.5-27B-FP8 --enable-auto-tool-choice --tool-call-parser qwen3_coder
#                               ^ server: unknown argument '--enable-auto-tool-choice'
```

Qwen3.5-27B is a model we ship **token-exact and gated**. The kernels are not the
thing standing between a user and a running server; argument parsing is.

## Upstream chain

At the pinned oracle `555967922` (vLLM 0.26.0.dev0):

- `vllm/entrypoints/openai/cli_args.py:105` — `enable_auto_tool_choice: bool = False`.
- `vllm/entrypoints/openai/cli_args.py:395` — **`--enable-auto-tool-choice`
  without `--tool-call-parser` is a `TypeError`**, raised by
  `validate_parsed_serve_args` as
  `TypeError("Error: --enable-auto-tool-choice requires --tool-call-parser")`.
  This validation is part of the behaviour and must be mirrored, not dropped.
- `vllm/entrypoints/openai/cli_args.py:111` — `tool_call_parser: str | None = None`,
  the falsy default the check above tests against.
- `vllm/entrypoints/openai/api_server.py:426,441,529,544` — threaded onward as
  `enable_auto_tools`.
- `--trust-remote-code` has no entry in this chain by design: it reaches
  `transformers` to authorize executing checkpoint Python, and the executing
  chain that consumes it does not exist here.

## Our baseline

`src/vllm/entrypoints/openai/server_main.cpp` parses ~57 flags in one
`if/else if` chain and ends in a single rejecting `else`
(`server: unknown argument '<flag>'` → `Usage(argv[0], 2)`). That default is
correct and is kept. What is absent is any way to say "recognised, and
deliberately does nothing": every flag is either fully wired or fatal, so a flag
that is a no-op **by construction** has nowhere to live and lands in the fatal
branch. `--enable-auto-tool-choice` and `--trust-remote-code` are the two that
matter, because published recipes pass them by default.

`--tool-call-parser` already defaults to `hermes` here, so tool parsing is
unconditional and there is nothing for an "enable" flag to switch on.

## Port map

| Upstream | Here |
|---|---|
| `cli_args.py:105` (`enable_auto_tool_choice` flag exists and defaults False) | `Args::enable_auto_tool_choice` + a `kAcceptedInertArgs` entry — recorded, announced, and inert |
| `cli_args.py:395` (`TypeError` when set without a parser) | a post-loop check in `ParseArgs` against `--tool-call-parser none`, our spelling of upstream's falsy `None`, printing upstream's sentence and exiting 2 |
| `api_server.py:426,441,529,544` (`enable_auto_tools` threading) | **not ported, deliberately** — there is no second gate to thread it to, which is exactly why the flag is inert. Recorded here so the omission is a decision, not an oversight |
| no upstream anchor (`--trust-remote-code`) | a `kAcceptedInertArgs` entry whose reason names the missing Python runtime |

Written from scratch rather than ported: the table, its lookup, and the notice.
There is no upstream analogue — upstream implements every flag it accepts, so
"accepted and inert" is a shape this project needs and vLLM does not.

## Design

Add an **accepted-and-inert** table to `server_main.cpp`. Not a catch-all: a fixed
list, one entry per flag, each carrying the reason it is inert and whether it takes
a value.

```
{"--enable-auto-tool-choice", kNoValue, "tool parsing is already unconditional once --tool-call-parser resolves"},
{"--trust-remote-code",       kNoValue, "no Python runtime: there is no remote code to trust"},
```

Three rules the table must obey:

1. **A flag not in the table still aborts.** Silently swallowing
   `--tensor-parallel-size` would let a user believe they got tensor parallelism.
   The whole value of the seam is that it is enumerated.
2. **Accepting is announced.** On use, emit one notice per accepted flag naming it
   and its reason, so a user reading the log learns the flag did nothing rather
   than inferring that it worked.
3. **Mirrored validation still fires.** `--enable-auto-tool-choice` with
   `--tool-call-parser none` must fail exactly as `cli_args.py:395` does. Inert is
   not the same as unvalidated.

### The one place we knowingly differ

Upstream defaults `--tool-call-parser` to `None`; we default it to `hermes`
(`docs/USAGE.md:878`). So upstream's flag genuinely gates something and ours cannot.
This spec does **not** change that default — it is pre-existing, out of scope here,
and worth its own issue if a reviewer wants it reconciled. The notice text must not
imply our behaviour matches upstream's when the parser is unset.

## Tests to port

vLLM's coverage of this flag is `tests/entrypoints/openai/test_cli_args.py:149`
(`test_enable_auto_choice_passes_without_tool_call_parser`, a `pytest.raises
(TypeError)`) and `:157`
(`test_enable_auto_choice_passes_with_tool_call_parser`), both driving
`validate_parsed_serve_args` over an `argparse.Namespace`. The harness does not
port — it asserts on a Python parser this project does not have — so the two
CASES carry over as behaviour, into the third and first rows of the table below,
with the upstream revision anchor recorded in the test header.

`:169` (`test_enable_auto_choice_fails_with_enable_reasoning`) is the same rule
reached by a different route: it fails because `--tool-call-parser` is still
unset, not because `--reasoning-parser` conflicts. It does NOT carry over as a
failing case, and the reason is the divergence recorded below — our
`--tool-call-parser` defaults to `hermes`, so that invocation legitimately
succeeds here. Mirroring the assertion rather than the rule would encode
upstream's default, not upstream's behaviour.

RED-first, in `tests/vllm/entrypoints/openai/`:

| Case | Asserts |
|---|---|
| each listed flag starts the server | argument parsing succeeds and the engine reaches load |
| an unlisted unknown flag | still aborts with the existing message — the guard is not widened |
| `--enable-auto-tool-choice` + `--tool-call-parser none` | fails, mirroring `cli_args.py:395` |
| notice emission | each accepted flag names itself and its reason on stderr |

The second row is the one that matters. A mutation that turns the table into a
catch-all must turn that test RED; if it does not, the test is not defending the
guarantee. Prove it by mutating in a scratch copy, then restore the tree
byte-for-byte.

## Dependencies

None blocking. The seam is self-contained inside `ParseArgs`, needs no new
header, no ABI change (`vllm_server_main` already takes argv), no GPU, no
checkpoint and no oracle run.

| Depends on | State |
|---|---|
| `VLLM_CPP_SERVER` build gate (owns `server_main.cpp` and the new test) | already ON in the CPU lane |
| `--tool-call-parser` resolution (`ResolveToolParserName`) | already shipped; only read, not changed |
| [#607](https://github.com/mudler/vllm.cpp/issues/607) `--language-model-only` | independent — deliberately NOT a dependency, and must not be folded in |

## Work breakdown

| Step | Deliverable |
|---|---|
| W1 | RED-first test file with the four cases, driving the real `VllmServerMain` from a re-exec'd child; capture the red |
| W2 | the enumerated table, its lookup, the parse branch, and the per-flag notice |
| W3 | the mirrored `cli_args.py:395` validation, plus the usage-text line |
| W4 | `docs/USAGE.md` entries stating the flags are accepted for recipe compatibility and have NO effect |
| W5 | mutation proof in a scratch copy — catch-all must turn W1's second case RED — then the row, the counts, and this spec |

Small enough for one implementer; the steps are ordered, not parallel.

## Gates

Focused: the new test file. Full: `scripts/agent-preflight.sh --staged` plus the
serve conformance suite. No GPU, no oracle run — this is an argument-parsing
change and does not touch a forward pass.

## Risks / decisions

- **Risk**: the table becomes a dumping ground for anything that fails to parse.
  Mitigated by rule 1 plus the per-entry reason string — an entry with no honest
  reason cannot be written.
- **Decision**: `--trust-remote-code` is accepted rather than rejected-with-advice.
  Rejecting it with "we don't need this" would still abort the recipe command,
  which is the entire defect.
- **Decision**: flags that are inert *because we lack the capability* — TP, EP,
  `--mm-encoder-tp-mode` — are **NOT** in this table. They keep aborting. Accepting
  them would be the failure mode this seam exists to prevent.
- **Out of scope**: `--language-model-only` (#607) is a real capability gap, not an
  inert flag, and must not be quietly added here.

## Now

`ACTIVE` — implemented, RED-first, mutation-proven; the claim
`CLAIM-SERVE-RECIPE-ARGS` is still open and the change is awaiting a fresh scoped
review plus the operator's own rerun of the row's gate.

The table, the notice, and the mirrored `cli_args.py:395` validation landed in
`src/vllm/entrypoints/openai/server_main.cpp` (`kAcceptedInertArgs:289`,
`FindAcceptedInertArg:308`, the parse branch `:501`, the validation `:557`), with
`docs/USAGE.md` § "Accepted for recipe compatibility" stating that both flags
have **no** effect and that the list is enumerated rather than a catch-all.

`tests/vllm/entrypoints/openai/test_serve_recipe_args.cpp` carries the four
cases. Each re-execs the test binary into the REAL `VllmServerMain` — `ParseArgs`
reports a bad argument through `Usage()`, which calls `std::exit`, so the abort
cases are unobservable in-process — against a deliberately nonexistent model
directory, which makes "parsing succeeded and the engine reached load" visible
(`server: loading model from ...`) without a checkpoint or a bound port.

Evidence:

- RED, final test against the pre-change `server_main.cpp` (built in a scratch
  copy): `test cases: 4 | 1 passed | 3 failed`, `assertions: 58 | 41 passed |
  17 failed`, `Status: FAILURE!`. The one already-passing case is "an unlisted
  unknown serve flag still aborts" — it must hold both before and after.
- GREEN: `test cases: 4 | 4 passed | 0 failed`, `assertions: 58 | 58 passed |
  0 failed`, `Status: SUCCESS!`.
- MUTATION: widening `FindAcceptedInertArg` into a catch-all (scratch copy)
  turns exactly "an unlisted unknown serve flag still aborts" RED — 8 asserts,
  `server: unknown argument` gone, the post-parse banner and `SERVE_RC=` now
  present, exit status 0 instead of 2 — and leaves the other three GREEN. The
  guard is defended by the test, not by inspection.

Not claimed: `GATING` or `DONE`. `GATING` would require closing
`CLAIM-SERVE-RECIPE-ARGS`, which this implementer does not own; `DONE`
additionally needs the fresh review, the operator gate rerun, and a
parity-ledger anchor. This section becomes an `## Outcome` at that point.
