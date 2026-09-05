# vllm-bench cannot stop on EOS, and cannot apply a chat template

| Field | Value |
|---|---|
| Issue | [#2759](https://github.com/mudler/vllm.cpp/issues/2759) |
| Owning row | `BENCH-QWEN38-27B-SOTA` ([backend-matrix](../backend-matrix.md)) |
| Discharges | the `## Owed` entry "A harness that can honour EOS and apply a chat template" in [bench-qwen38-27b-nvfp4-matched.md](bench-qwen38-27b-nvfp4-matched.md) |
| Sibling, NOT in scope | [#2770](https://github.com/mudler/vllm.cpp/issues/2770), the server-side acceptance metric, owed under [dflash2-spec-decode.md](dflash2-spec-decode.md) `## Owed` |
| vLLM pin | `e126687a9a828d513c01a07cd69f025f27d63280` (`0.28.1rc1.dev132`), from [upstream-sync.md](../upstream-sync.md). Issue #2759 and the `dflash2-spec-decode.md` `## Owed` entry both cite `5559679229`, which the pin left behind on 2026-09-03 |
| Host | none. CPU only, synthetic engine, no checkpoint, no device, no lease |
| Status | `ACTIVE` |

## 1. What is broken

`examples/bench/bench_core.h` sets `sp.ignore_eos = true` with no field, no flag
and no environment variable, so every `vllm-bench` request runs to exactly
`--output-len` tokens and no request in the history of this harness has ever
terminated on EOS. The same harness sends the raw prompt string, so no chat
template is ever applied.

That is a measurement problem, not a convenience one. The comparator for this
row, `pangoleen/qwen3.8-27b-dgx-spark-dflash2`, posts to `/v1/chat/completions`
with a chat template and never sets `ignore_eos`; across the 25 rows of its
`data/ctxsweep-recommended.csv` `out_tokens` is min 170, median 269, max 333
against a 512 cap, so **not one of 25 reaches the cap** and every figure they
publish ended on EOS. Suppressing EOS decodes past the point where the model
would have stopped, into lower-entropy continuation, and lower-entropy text is
easier for a drafter to predict. The bias is on our side of the comparison, so
it flatters us.

The engine is not the problem. `include/vllm/sampling_params.h` defaults
`ignore_eos = false`, `src/vllm/v1/core/sched/utils.cpp:30-33` mirrors vLLM's
`_eos_token_id` property gate, and `src/vllm/entrypoints/chat_template.cpp`
already renders templates for the server. What is missing is a client that puts
them together.

## 2. Upstream anchors, all at pin `e126687a9`

| What | Upstream `file:line` | Value |
|---|---|---|
| `--ignore-eos`, the benchmark client's flag | `vllm/benchmarks/serve.py:1757-1762` | `action="store_true"`, so upstream's default is FALSE |
| `--skip-chat-template`, the benchmark client's flag | `vllm/benchmarks/datasets/datasets.py:1654-1658` | `action="store_true"`, so upstream's default is APPLY |
| the render itself | `vllm/benchmarks/datasets/datasets.py:2609-2616` | `tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, **chat_template_kwargs)` |
| `--chat-template`, path or single-line literal | `vllm/entrypoints/launchers/cli_args.py:80-82` | "The file path to the chat template, or the template in single-line form for the specified model" |
| `ignore_eos` on the wire | `vllm/benchmarks/lib/endpoint_request_func.py:79,133-134` | sent only when true |

Our local anchors: `server_main.cpp` resolves the template with
`LoadChatTemplateForModel(dir/"tokenizer_config.json", model_dir, source)` and
builds the seam with
`MakeChatTemplatePromptFn(chat_template, bos, eos, DefaultChatTemplateKwargs(args.enable_thinking))`.
The bench takes that same pair verbatim.

## 3. Design

### 3.1 The flag names are upstream's, and the negations are this tree's

Four `BenchConfig` fields, four flag spellings, every one of them already in use
somewhere this change did not invent:

| Field | Default | Flags | Where the spelling comes from |
|---|---|---|---|
| `bool ignore_eos` | `true` | `--ignore-eos` / `--no-ignore-eos` | name from `serve.py:1758`; the `--no-` negation from `server_main.cpp` `--no-enable-thinking` and from `scripts/nemotron-h-oracle-capture.py:617`, which already spells `--no-ignore-eos` |
| `bool skip_chat_template` | `true` | `--skip-chat-template` / `--no-skip-chat-template` | name from `datasets.py:1655`, same negation convention |
| `std::string chat_template` | `""` | `--chat-template <path or literal>` | `cli_args.py:80-82` |
| `std::optional<bool> enable_thinking` | unset | `--enable-thinking` / `--no-enable-thinking` | `server_main.cpp`, verbatim |

**The defaults are today's behaviour, exactly.** `ignore_eos = true` and
`skip_chat_template = true` reproduce the current harness byte for byte, so no
landed figure silently changes meaning. That is why the polarity is inverted
against upstream's: upstream's default is a fresh client's default, ours is a
compatibility constraint, and the alternative — flipping the default and
re-measuring every published row — is a different piece of work.

**`--no-skip-chat-template` is a double negative and it is still the right
name.** The alternative, `--apply-chat-template`, is a second vocabulary for a
flag vLLM already named, and AGENTS.md §"vLLM is the reference" forbids that. A
reader who knows `vllm bench serve` can read our command line.

### 3.2 The EOS half

`MakeSampling` reads the field instead of the literal:

```cpp
sp.ignore_eos = cfg.ignore_eos;
```

`BenchResult` gains `bool resolved_ignore_eos`, set in `RunBench` from
`MakeSampling(cfg, 0).ignore_eos` — a read-back of the object the admission path
actually builds, not a copy of `cfg`. This is the pattern
`resolved_kv_cache_dtype` established for #2619: the report carries what was
asked for and what was resolved as two lines, because they answer different
questions.

### 3.3 The chat-template half

One render site, in `RunBench`, after the prompts exist and **before**
`PretokenizeBenchPromptsThenStartClock`, so both admission paths — the
pretokenized default and the `VT_BENCH_PRETOKENIZE=0` timed-string path — carry
the identical rendered prompt and the prompt-token accounting reflects it:

```cpp
prompts[i] = prompt_fn({{"user", prompts[i]}}, /*add_generation_prompt=*/true,
                       /*tools=*/{}, /*chat_template_kwargs=*/{});
```

One user message, `add_generation_prompt=true`, no tools — `datasets.py:2610-2615`
exactly. The per-request kwargs object is empty because the bench has no
per-request kwargs; the server-level default carries `enable_thinking` through
`DefaultChatTemplateKwargs`, which is where `server_main.cpp` puts it too.

Template source resolution, in order:

1. `--chat-template` non-empty: if it names an existing file, read the file;
   otherwise take the string as the template itself. `cli_args.py:80-82`.
2. Otherwise `LoadChatTemplateForModel(dir/"tokenizer_config.json", model_dir,
   source)`, the server's own selection path, which covers a safetensors
   directory and falls back to a `.gguf`'s `tokenizer.chat_template` metadata.

### 3.4 Three refusals, each naming the missing part

- `--chat-template` given while the template is skipped: refuse. A flag that
  parses and is then dropped is the defect #2759 is about, and accepting it
  silently would reproduce it inside the fix.
- `--no-skip-chat-template` with no `--chat-template` and no model directory
  (the synthetic engine): refuse, naming that the synthetic engine ships no
  chat template and that `--chat-template` supplies one.
- `--no-skip-chat-template` on a model that carries no template: the
  `ChatTemplateError` from `LoadChatTemplateForModel` propagates with its own
  message. Not caught, not downgraded to the role-join fallback — the server
  falls back because a served model must answer, and a benchmark that silently
  measured a different prompt shape than the one requested is worse than one
  that stops.

### 3.5 The report says which run this was

Four lines in `PrintReport`, beside the two `kv_cache_dtype` lines that exist
for the same reason:

```
Ignore EOS (requested):                    1
Ignore EOS (resolved sampling):            1
Chat template:                             skipped (raw prompt)
Chat template kwargs:                      {}
```

and the same values in the `vllm-bench:` stderr banner. A benchmark that does
not state whether it stopped on EOS is not reproducible, and every number this
harness has published so far was taken with EOS suppressed without saying so.

## 4. Not in scope

- **#2770**, the server-side `vllm:spec_decode_*` family. It is a different
  surface, it needs a per-iteration stats path this tree does not have, and it
  stays owed under `dflash2-spec-decode.md` `## Owed` with its issue named.
- `tools/bench/online_gate.py`, which raises unless `--ignore-eos` and
  `--skip-chat-template` are each present exactly once in the vLLM client
  command. That enforcement is about the ORACLE's client, not ours, and the
  matched-protocol gate that would relax it is the next wave.
- Any re-measurement. No published figure moves, because no default moves.

## 5. Risks

- **A default that drifts.** The whole value of this change is that
  `vllm-bench` with no new flag is the old `vllm-bench`. G4 below asserts the
  no-flag run still reports `ignore_eos=1` and a skipped template.
- **A rendered prompt the synthetic tokenizer cannot encode.** The synthetic
  BPE vocabulary is nineteen pieces; a real Qwen template renders `<|im_start|>`
  and friends, which are not in it. The gate therefore renders through a
  template written in that vocabulary. This is a fixture constraint, not a
  behaviour difference: the render seam is the production one either way.
- **No CPU EOS observation exists.** `bench_core.h` builds the synthetic config
  with `c.raw = nlohmann::json::object();  // no eos => runs to max_tokens`, and
  the synthetic tokenizer declares no EOS either, so `check_stop`'s EOS arm can
  never fire on the CPU fixture whatever `ignore_eos` says. §7 records exactly
  what that costs the gate and what remains owed.

## 6. Gates

| ID | What it asserts | How |
|---|---|---|
| G1 | `--no-ignore-eos` reaches `SamplingParams` | run the binary, read `Ignore EOS (resolved sampling)` |
| G2 | the resolved line is not a transcription of the request | `--ignore-eos` and `--no-ignore-eos` produce different resolved values from the same requested-line machinery |
| G3 | `--no-skip-chat-template --chat-template <file>` changes the prompt the engine tokenizes | `Total input tokens` strictly greater than the same run without, on an identical workload |
| G4 | the default run is unchanged | no new flag, and the report reads `ignore_eos` requested 1 / resolved 1 and a skipped template |
| G5 | each refusal fires and names its part | non-zero exit and the named message for the three §3.4 cases |
| G6 | a literal template and a file template resolve identically | same `Total input tokens` from `--chat-template <file>` and `--chat-template <the same string>` |
| G7 | the existing bench harness is inert | `test_bench` unchanged |

Focused gate: `ctest -R 'test_bench'`, driven through the built `vllm-bench`
binary rather than through a hand-built `BenchConfig`, because for a benchmark
harness the production entry point is `argv`
([reachability.md](../reachability.md)); a unit test that constructs
`BenchConfig` by hand stayed green on every day this flag did not exist.

## 7. What this gate cannot see, and what it costs

**The end-to-end EOS stop is not observed anywhere on CPU.** G1 and G2 prove the
flag reaches the `SamplingParams` the admission path builds, and `check_stop`'s
EOS arm is already gated by `ignore_eos` and already covered
(`src/vllm/v1/core/sched/utils.cpp:30-33`). What no test in this change does is
watch a request stop early because EOS came out, because the synthetic fixture
has no EOS id to emit. Making one emit deterministically would mean pinning the
toy model's greedy output as a constant, which is the "assert against the
constant" tautology, or declaring every vocabulary id a stop token, which routes
through `check_stop`'s arm (3) and is not gated by `ignore_eos` at all — it would
be a gate that passes with the flag deleted. Neither is worth having. The
honest statement is: this change makes the harness able to honour EOS and proves
the setting arrives; that it then stops a real generation is owed to the first
checkpoint run, and §8 records it.

## 7a. Evidence: red before, green after, and one mutation nothing catches

Built CPU-only, no device, no lease:
`cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
-DVLLM_CPP_METAL=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_SERVER=ON
-DVLLM_CPP_BUILD_EXAMPLES=ON -DVLLM_CPP_BUILD_TESTS=ON`, `ninja -j 4`.

| Stage | Result |
|---|---|
| RED before, against the unchanged harness | `test_bench_eos_chat_template` exit 1, **9 of 9 cases failed**, 16 of 51 assertions |
| GREEN after | exit 0, 9 of 9 passed, **73 assertions** |
| `test_bench` (inert) | exit 0, 11 of 11, 80 assertions |
| `test_bench_kv_cache_dtype` (inert) | exit 0, 9 of 9, 47 assertions |

Every mutation below was applied to the committed tree, rebuilt, run, and
reverted; each row was checked for a changed source sha256, a build that
succeeded (a build failure reads as a passing test), a `vllm-bench` binary whose
mtime MOVED, and a working tree restored byte-for-byte afterwards. All nine rows
satisfied all four.

| # | Mutation | Cases failed | Verdict |
|---|---|---:|---|
| M1 | `sp.ignore_eos = cfg.ignore_eos` back to `= true` — the pass-through drop | 3 / 9 | **RED** |
| M2 | delete the `--no-ignore-eos` arm from `ParseArgs` | 3 / 9 | **RED** |
| M3 | delete the render call site, keeping the resolution and the report line | 1 / 9 | **RED** |
| M4 | resolved EOS line echoes `cfg.ignore_eos` instead of reading `MakeSampling` back | 0 / 9 | **GREEN — see below** |
| M5 | delete the "`--chat-template` while skipping" refusal | 1 / 9 | **RED** |
| M6 | delete the "synthetic engine ships no template" refusal | 1 / 9 | **RED** |
| M7 | `--chat-template` parses its value and drops it | 4 / 9 | **RED** |
| M8 | `--no-enable-thinking` parses and leaves the kwargs unset | 1 / 9 | **RED** |
| M9 | **M1 and M4 together** | 0 / 9 | **GREEN — the flag is dead and the gate is silent** |

**M9 is the honest hole in this gate, and it is structural.** M1 is detectable
only because the report reads the value back out of `MakeSampling`; M4 removes
that read-back without changing any value, so it is harmless alone and lethal in
combination. Closing it needs an observation that the EOS setting CHANGED A
GENERATION, and §7 records why no such observation exists on the CPU fixture.
The two candidate fixes are both worse than the hole: pinning the toy model's
greedy argmax as the fixture's eos id is the "assert against the constant"
tautology and would redden on any unrelated change to the synthetic weights, and
a `--synthetic-eos-token-id` knob would be production surface that exists only
for a test. This is named here rather than closed, and the first checkpoint run
closes it for real.

## 8. Owed

- An EOS-terminated `vllm-bench` run on a real checkpoint, which is the first
  measurement that observes the stop rather than the setting. Owed to
  `BENCH-QWEN38-27B-SOTA`'s next lease.
- The server-side acceptance metric,
  [#2770](https://github.com/mudler/vllm.cpp/issues/2770), owed under
  [dflash2-spec-decode.md](dflash2-spec-decode.md) `## Owed`.
- The `## Outcome` section this spec owes at `DONE`.

## Now

`ACTIVE`. The flags, the render seam and the report lines land together with the
gate above. Nothing here is measured on a device.
