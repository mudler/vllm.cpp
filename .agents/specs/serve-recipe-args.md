# SERVE-RECIPE-ARGS — a published recipe command must reach model load

Issue: [#606](https://github.com/mudler/vllm.cpp/issues/606)
Row: `SERVE-RECIPE-ARGS` ([engine-matrix.md](../engine-matrix.md))
Sweep that found it: `roadmap_v1.md` § Recipe-surface sweep (2026-08-13)

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

## Upstream anchors

- `vllm/entrypoints/openai/cli_args.py:105` — `enable_auto_tool_choice: bool = False`.
- `vllm/entrypoints/openai/cli_args.py:395` — **`--enable-auto-tool-choice`
  without `--tool-call-parser` is a `TypeError`.** This validation is part of the
  behaviour and must be mirrored, not dropped.
- `vllm/entrypoints/openai/api_server.py:426,441,529,544` — threaded onward as
  `enable_auto_tools`.

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

## Tests

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

`SPIKE` — spec committed, implementation not started. Next: a fresh implementer
takes the table, the four tests, and the mutation proof.
