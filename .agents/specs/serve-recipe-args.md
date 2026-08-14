# SERVE-RECIPE-ARGS — a published recipe command must reach model load

Issue: [#606](https://github.com/mudler/vllm.cpp/issues/606)
Row: `SERVE-RECIPE-ARGS` ([engine-matrix.md](../engine-matrix.md))
Sweep that found it: the 2026-08-13 `recipes.vllm.ai` recipe-surface sweep. Its
write-up is `roadmap_v1.md` § "Recipe-surface sweep (2026-08-13,
`recipes.vllm.ai`)", which arrives with
[#612](https://github.com/mudler/vllm.cpp/pull/612) together with the intake rows
for #605–#608. Until that PR merges this is a **forward reference**, and it is
deliberately not duplicated here: #612 owns those keys, and two branches adding
the same keyed record merge cleanly and then define it twice.

## Scope

One seam in the serve entry point: an **enumerated** accepted-and-inert argument
table, its per-flag startup notice, and the mirrored `cli_args.py:395`
validation. Two entries to start — `--enable-auto-tool-choice` and
`--trust-remote-code`.

Explicitly NOT in scope, each for its own reason:

| Excluded | Why |
|---|---|
| `--language-model-only` | a real capability gap, tracked as [#607](https://github.com/mudler/vllm.cpp/issues/607) — accepting it would hide missing behaviour. **Resolved the right way 2026-08-14**: #607 wave L2 IMPLEMENTED it (and `--limit-mm-per-prompt`) rather than adding it here, so it is a real flag with real enforcement and `kAcceptedInertArgs` still holds only genuine no-ops |
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
- **Named cold path**: `InertArg::takes_value` and its `NextArg` consumption are
  correct but UNEXERCISED — both shipped entries are `kNoValue`, so no test drives
  the value-taking branch and none can without inventing an entry. Kept because
  the alternative is a retrofit at the moment a value-taking recipe flag arrives,
  which is exactly when swallowing the flag but not its value would re-parse the
  value as the next flag. **The first entry with `takes_value == true` owes a test
  in the same change** — the value is consumed and the following flag still
  parses. There is no `kTakesValue` constant yet either; only `kNoValue`. That
  obligation is recorded here rather than paid with speculative code now.

## Now

`ACTIVE` — implemented, RED-first, mutation-proven; the claim
`CLAIM-SERVE-RECIPE-ARGS` is still open and the change is awaiting a fresh scoped
review plus the operator's own rerun of the row's gate.

The table, the notice, and the mirrored `cli_args.py:395` validation landed in
`src/vllm/entrypoints/openai/server_main.cpp` (`kAcceptedInertArgs:289`,
`FindAcceptedInertArg:312`, the parse branch `:505`, the validation `:560`), with
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

Two things a reviewer or the operator will meet and should not re-derive.

**The PR cannot go fully green, and not because of this change. THREE lanes are
red, not two** — each named here, because a record that lists only some of them
tells a reader the rest are green.

`windows-msvc-cpu` and `windows-msvc-vulkan` fail with `test_openai_api_server.exe
exited with status -1073740791` (`0xC0000409`, `STATUS_STACK_BUFFER_OVERRUN`),
which is [#584](https://github.com/mudler/vllm.cpp/issues/584). Attributed rather
than assumed: [#625](https://github.com/mudler/vllm.cpp/pull/625) fails with the
byte-identical exit status and touches no `src/` or `include/` path at all, so a
records-only PR reproduces it. `test_openai_api_server` is also not a consumer of
this seam — it never calls `ParseArgs`.

`build-test-cpu` fails on `test_cpu_threadpool`, 1 of 404, and that is
[#631](https://github.com/mudler/vllm.cpp/issues/631):

```
tests/vt/test_cpu_threadpool.cpp:536: MESSAGE: empty-op dispatch: 2 threads 0.48 us, 5 threads 48.752 us, ratio 101.567
tests/vt/test_cpu_threadpool.cpp:539: ERROR: CHECK( ratio < 100.0 ) is NOT correct!
```

Attributed rather than assumed, to the same standard as #584. The guard divides
two wall-clock medians (`over_us / fits_us`) and compares the ratio to a fixed
100, so both ends are machine-shape dependent and the denominator is the problem:
on the 4-core runner `fits_us` collapsed to 0.48 us, small enough that ordinary
scheduler noise in the numerator moves the ratio by tens. Same commit, same code,
three observations:

| Box | `fits` | `over` | ratio |
|---|---|---|---|
| 4-core CI runner | 2 threads, 0.48 us | 5 threads, 48.752 us | **101.567** RED |
| 20-core box (#631's table) | 10 threads, 7.213 us | 21 threads, 19.467 us | **2.699** GREEN |
| 20-core box, this branch rebuilt at the reviewed head | 10 threads, 13.256 us | 21 threads, 13.135 us | **0.990872** GREEN |

The runner's core count is read off the failure itself, not assumed: the case
returns early at `test_cpu_threadpool.cpp:501` when `cores < 4`, so a box that
produced a ratio at all has at least 4; and `fits = cores / 2` (`:512`) reporting
2 threads with `over = cores + 1` (`:513`) reporting 5 pins it at exactly 4 —
which is what GitHub gives `ubuntu-latest` on a public repo. An earlier revision
of this section said 2-core, a number no run on that lane can produce.

The last row is the one measured while repairing this record — `test_cpu_threadpool`
9 cases / 9 passed, 19602 assertions / 0 failed, `Status: SUCCESS!`. That it sits
2.7x below the middle row on the *same class of box* is itself the finding: the
statistic is not stable enough to carry a fixed threshold. `build-test-cpu` is
also green on the scheduled `main` lane at baseline `7572b0f4e`, and
`test_serve_recipe_args` passed on the very runner that went red. This diff is
argument parsing inside `ParseArgs`; there is no path from it to a threadpool
dispatch ratio. The test's own comment (`test_cpu_threadpool.cpp:496`) claims it
can "be trusted ... never to fail spuriously on a busy one"; the CI run falsifies
that, which is what #631 carries. Not fixed in flow deliberately: changing that
guard changes a gate's semantics, so it takes its own spec and red-before
evidence, and raising 100 to a larger number would be widening a scope to turn a
red gate green.

**A trap this change walked into, recorded so the next person does not.**
`check-windows-portability.py` scans the shipped server sources with comments
stripped but STRING LITERALS INTACT, and its POSIX-call pattern matches a bare
`open` before a parenthesis. The notice text originally read "...no second gate to
open (note --tool-call-parser defaults to hermes...)", and `open (` inside a
user-facing message was reported as an unguarded POSIX call reaching Windows. It
is a semicolon clause now, with a comment at the site. The gate is Windows-only
and is NOT part of `scripts/agent-preflight.sh`, so a fully green local preflight
says nothing about it.

Not claimed: `GATING` or `DONE`. `GATING` would require closing
`CLAIM-SERVE-RECIPE-ARGS`, which this implementer does not own; `DONE`
additionally needs the fresh review, the operator gate rerun, and a
parity-ledger anchor. This section becomes an `## Outcome` at that point.
