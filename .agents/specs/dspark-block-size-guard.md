# The DSpark block floor reaches no production caller (`SPEC-DSPARK-BLOCK-SIZE-GUARD`)

| Field | Value |
|---|---|
| Row | `SPEC-DSPARK-BLOCK-SIZE-GUARD` (engine-matrix, Speculative decoding) |
| Issue | [#1225](https://github.com/mudler/vllm.cpp/issues/1225) |
| Scope | Make the landed `k >= block` floor in `SpeculativeConfig::ResolveDspark` reachable from the loader, so a speculative length below a DSpark draft's block is refused instead of silently drafting a structurally wrong block. Three parts: (A) read the draft `config.json` for `n_predict` and the block key; (B) pass them at both `ResolveDspark` call sites instead of `std::nullopt`; (C) a red-first test that enters through `LoadedEngine::ResolveSpecConfig`. **Excluded:** the DSpark speculator itself and its block layout (landed under `SPEC-DSPARK`), draft architecture classification and `IsDsparkDraft` (owned by `SPEC-DSPARK-QWEN3-ROUTING`, [#1193](https://github.com/mudler/vllm.cpp/issues/1193), in flight), the DeepSeek-V4 DSpark runtime (hardware-blocked), and the GPU run gate that would exhibit the garbling (§6, owed). |
| Upstream chain | At the pin `555967922`: `vllm/config/speculative.py:945-961` (the Gemma4 `block_size` → `n_predict` normalization) → `:973-988` (the `n_predict` default and the divisibility rule) → `:990-994` (k required) → `:1003-1027` (the `k >= dspark_block_size` hard error and its wording). Beyond the pin: PR [vllm#52197](https://github.com/vllm-project/vllm/pull/52197) at `7075ddac`, whose two hunks are architecture routing only and leave every line above unchanged. |
| Our baseline | `include/vllm/config/speculative.h:156-188` (`ResolveDspark`, the floor at `:179-185`). Call sites `src/vllm/entrypoints/model_loader.cpp:881-883` and `:1675-1677`, both passing `std::nullopt` twice. The draft config is already read at `:471-474` inside `LoadDsparkDraft`. Tests: `tests/vllm/config/test_speculative_dspark.cpp:99-107`, which call `ResolveDspark` directly. |
| Port map | §3. |
| Tests to port | §5. |
| Gates | §6. |
| Dependencies | Landed: `SPEC-DSPARK` W1-W8 ([dspark-spec-decode.md](dspark-spec-decode.md), `ACTIVE`). In flight and adjacent: `SPEC-DSPARK-QWEN3-ROUTING` ([dspark-qwen3-routing.md](dspark-qwen3-routing.md)), which edits the same `ResolveSpecConfig` branch for classification. §7 R5 records the seam between them. |
| Work breakdown | §4. |
| Risks/decisions | §7. |
| Pin policy | Mirror the pin, plus two recorded divergences: the block-key fallback argued in §2, and the error ORDER argued in §2a. Both are repeated in the commit body. |
| Role / claim | fresh implementer, branch `row/DSPARK-BLOCK-SIZE-GUARD` |
| Base | `65d6cdaed3e20e9bc70b4f9374fccafefefa7bd0` (origin/main, 2026-08-18) |
| Parity pin | vLLM `555967922` (0.26.0.dev0) at `$VLLM_SOURCE` |

## 0. Verdict

Two findings, both verified in a clean worktree at the base above. The second
one changes what the fix has to do, so it is stated in full rather than folded
into the design.

**1. The floor is unreachable.** `ResolveDspark` implements upstream's hard
error at `include/vllm/config/speculative.h:179-185`, and both production call
sites pass `std::nullopt` for `n_predict` and for `dspark_block_size`:

```
src/vllm/entrypoints/model_loader.cpp:881-883   LoadedEngine::ResolveSpecConfig
src/vllm/entrypoints/model_loader.cpp:1675-1677 the draft load inside FromModelDir
```

Every other reference to those parameters is in
`tests/vllm/config/test_speculative_dspark.cpp`, which supplies the values by
hand. This is the unpassed-parameter shape of `.agents/reachability.md`: the
argument exists, the test drives it, and no user can arrive at it.

The consequence is silent. Our draft step is sized only by `k` —
`DsparkBlockLayout::num_speculative_steps` is `N (k)`
(`include/vllm/v1/worker/gpu/spec_decode/dspark/speculator.h:56`) — and nothing
under `src/vllm/v1/worker/gpu/spec_decode/dspark/` or in the loader reads the
draft's block key. No weight is shaped by the block, so a short `k` trips no
`VT_CHECK`. It drafts a structurally wrong block and the tokens keep flowing.

**2. A literal port of the floor would still not fire.** Upstream reads
`getattr(hf_config, "dspark_block_size", None)` (`:1011-1015`). A grep of the
whole pinned checkout finds that identifier in `speculative.py` and in no other
file, so no vLLM config class defines it and it can only arrive from a draft
`config.json`. Both published Qwen3 DSpark drafts were read live on 2026-08-18:

| Draft | `architectures` | `model_type` | block key | `n_predict` | `dspark_block_size` |
|---|---|---|---|---|---|
| `deepseek-ai/dspark_qwen3_4b_block7`, upstream's own e2e draft | `["Qwen3DSparkModel"]` | `qwen3` | `block_size: 7` | absent | absent |
| `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153be924f17ce4bf62726954eeaa4a73e854` | `["DSparkDraftModel"]` | `qwen3` | `block_size: 7` | absent | absent |

Upstream's `block_size` → `n_predict` normalization at `:945-961` is guarded by
`"Gemma4DSparkModel" in architectures`, so it does not reach either row. #52197
rewrites the second row's `architectures` to `Qwen3DSparkModel` and changes
nothing else.

Trace `k=6` against either draft through the pinned `__post_init__` and the
result is the same on both sides of #52197: `n_predict` is `None`, so `:973-988`
does nothing; `dspark_block_size` is `None`, so `:1003-1027` does nothing. **The
value is accepted.** Upstream's floor is, in practice, a DeepSeek-V4 guard, and
DeepSeek-V4 DSpark is the one lane we do not implement.

So the literal port is not a fix. It would land a branch keyed on a field no
supported checkpoint sets, which is the unselected-branch shape of
`.agents/reachability.md`, and it would leave the silent path exactly as it is.

## 1. Scope

In: the two call sites, the config read that feeds them, and the test that
enters through the loader.

Out: everything in the `Scope` row's exclusion list. In particular this row does
not touch `IsDsparkDraft`, `ResolveDsparkArchitecture`, or the method
classification in `ResolveSpecConfig`. Those belong to
`SPEC-DSPARK-QWEN3-ROUTING`, which is in flight against the same branch of the
same function.

## 2. The divergence, and why it is argued rather than smuggled

`AGENTS.md` says to mirror vLLM wherever vLLM defines the behavior, and it also
says that nothing lands dead. §0 finding 2 puts those two in direct tension: the
faithful port is dead on arrival for every checkpoint we ship, and the shipped
lane keeps a silent-wrong-output path.

**Decision: port the pin exactly, and add one fallback.** The floor is
`dspark_block_size` when the draft config carries it, and `block_size` when it
does not. Everything else — the threshold, the comparison, the error wording,
the `n_predict` default and divisibility rules — is upstream's.

The reasons, in the order they carry weight:

1. **The intent is upstream's own.** The comment at `speculative.py:1004-1010`
   says to require `num_speculative_tokens` to be at least the block size, and
   gives 5 or 7 as the examples. For a Qwen3 draft the block size is spelled
   `block_size`, and it is 7 on both published drafts. We are supplying the
   value upstream's own sentence names, not choosing a different rule.
2. **Upstream already treats `block_size` as the block depth.** The Gemma4
   normalization at `:945-961` maps exactly that key onto `n_predict` for a
   self-contained draft. The Qwen3 drafts are self-contained in the same sense.
3. **Silent wrong output is the worst failure class we have.** A token gate does
   not necessarily catch it, because the drafter still produces tokens and the
   target still verifies them.
4. **It is the minimum that changes nothing already working.** Mapping
   `block_size` onto the *floor* rather than onto `n_predict` leaves every
   currently-accepted configuration byte-identical: `k >= 7` behaves exactly as
   today, `k` stays required, and the divisibility rule is untouched. Only
   `k < block` changes, from silent acceptance to a refusal.

This is one tracked exception under `AGENTS.md` `## Changing the rules or a
checker`: argued in the commit body, visible in the diff, and rejectable by the
reviewer. It is not a waiver and it is not a registry entry.

The honest residual is that upstream accepts `k=6` here and we will refuse it.
That is a deliberate stricter-than-upstream refusal on a path upstream leaves
unguarded, and §6 G3 owes the oracle run that shows which side is right about
the output.

## 2a. The second divergence: where the refusal falls in the error order

§6b moved the DSpark resolution to the top of `LoadedEngine::FromModelDir`,
above the `fs::is_directory` refusal. That changes an ORDER upstream fixes, so it
is a second divergence and it is recorded with the same rigour as §2 rather than
left inside a design paragraph.

**What upstream does.** `EngineArgs.create_engine_config` builds `ModelConfig` at
`vllm/engine/arg_utils.py:1899` and `SpeculativeConfig` at `:2218`, in that
order. A run that names both a target the loader cannot open and a `k` below the
draft's block therefore meets the MODEL error first. The speculative refusal
never runs.

**What we do.** The same run meets the `k` error. `FromModelDir` resolves the
DSpark config before it touches the model directory, so
`DSpark requires num_speculative_tokens >= block_size (7); got 6` is what the
user reads, and `model path is not a directory` is what they read after they fix
the `k`.

**Why the internal precedent does not carry it.** §6b argued the placement from
the explicit device resolution that already runs before `const fs::path
dir(model_dir)`. That precedent is real but it points the other way: vLLM builds
`DeviceConfig` at `:1878`, BEFORE `ModelConfig` at `:1899`, so our device
ordering mirrors upstream and says nothing about where a speculative config
belongs. Upstream's own answer for the speculative config is `:2218`, after.

**Why ours is better here, and why the mirror is not available.** Upstream splits
the work: `create_engine_config` builds config objects only, and the target
weights are loaded much later by the executor, so upstream never pays for the
target when it refuses a `k`. Our `FromModelDir` is one function that validates
the path, parses the config, builds the tokenizer AND loads the weights. Placing
the resolution after the model-directory work would spend the whole target load
before refusing a `k` the draft can never serve, which on a 27B target is minutes
of mapping for a run that cannot start. There is no placement here that both
mirrors upstream's order and keeps upstream's property that the refusal is cheap.
We keep the property and diverge on the order.

**The user-visible consequence, exactly.** It shows only when BOTH inputs are
wrong, and then it is which message comes first — the `k` here, the model path
upstream. Either way the user has to fix both. When only the `k` is wrong, which
is the case the guard exists for, upstream and this loader agree on the message
and differ only in how much work preceded it.

**What would reverse it.** A `FromModelDir` split into a config phase and a load
phase would let the model error keep its upstream position and still refuse the
`k` before any weight is read. That is a restructuring this row does not own; §8
carries it.

## 3. Port map

| Upstream, at the pin | Lands in | Shape |
|---|---|---|
| `speculative.py:973-979`, the `n_predict` read and default | `src/vllm/entrypoints/model_loader.cpp`, a new file-local `ReadDsparkDraftKeys` | Read `n_predict` from the draft `config.json` when present. |
| `speculative.py:945-961`, the Gemma4 normalization | the same helper | When `n_predict` is absent and `architectures` contains `Gemma4DSparkModel`, take `block_size` as `n_predict`. Guarded by the architecture name exactly as upstream guards it. |
| `speculative.py:1011-1015`, the floor read | the same helper | Read `dspark_block_size` when present. **Divergence (§2):** fall back to `block_size` when it is not. |
| `speculative.py:1003-1027`, the hard error | `include/vllm/config/speculative.h:179-185` | Already landed and already worded from upstream. Not re-implemented; only reached. |
| `speculative.py:990-994`, k required | `model_loader.cpp`, `LoadedEngine::ResolveSpecConfig` | The existing early throw fires before `ResolveDspark` sees `n_predict`, which would make the threaded `n_predict` default unreachable. Narrow it to fire only when the draft carries no `n_predict` either, and let `ResolveDspark` raise otherwise. §7 R4 records what this claim was worth before §6b. |
| `speculative.py:1021-1026`, the refusal's wording | `include/vllm/config/speculative.h`, `SpeculativeConfig::ResolveDspark` | Upstream names one key literally, because it reads one. We read two (§2), so the key is carried on `DsparkDraftKeys` and passed in; the parameter defaults to upstream's key. Upstream's third sentence, `Use num_speculative_tokens={n} or larger (e.g. 7).`, was dropped when the check was first ported and is restored (§6b, finding 2). |

`ReadDsparkDraftKeys` resolves the draft directory with the existing
`ResolveDflashDraftDir` (`model_loader.cpp:279`) and returns both values empty
when no `config.json` is there, so `ResolveSpecConfig` keeps working on a path
that `LoadDsparkDraft` will later reject with its own message. It applies
`TranslateSpeculatorsDsparkConfig` first, exactly as `LoadDsparkDraft` does, so
both config layouts are read through the same normalized shape. On the two
shipped layouts that translation moves neither key the guard reads, so the call
is consistency insurance and not a measured requirement (§6c).

This is the second read of the draft config, alongside the one in
`LoadDsparkDraft`. `dspark-qwen3-routing.md` §3 already weighs a second read of a
five-layer draft config and records it as not a measured cost. Hoisting the read
would mean restructuring `FromModelDir`, which collides with the in-flight row
for no correctness gain. §8 records the total, and the number the two rows reach
together, as owed.

## 4. Work breakdown

| Step | Content | Gateable on this host |
|---|---|---|
| W1 | The red test: a draft config carrying `dspark_block_size` and one carrying only `block_size`, each with `k` below it, through `LoadedEngine::ResolveSpecConfig`. Capture both reds. | yes, CPU |
| W2 | `ReadDsparkDraftKeys` plus the two call sites plus the narrowed early throw. Focused green. | yes, CPU |
| W3 | The reachability mutation of §6: restore `std::nullopt` at `:881-883` in a scratch copy and confirm the focused gate goes red. | yes, CPU |
| W4 | The GPU run gate of §6 G3. | needs a GPU lease |

W1-W3 need no GPU. W4 does, and it does not block W1-W3.

## 5. Tests to port

Upstream has no unit test for this resolution. It is covered end to end by
`tests/v1/e2e/spec_decode/test_spec_decode.py::test_dspark_correctness_and_acceptance_rate`,
which needs a GPU and two checkpoints. The existing
`tests/vllm/config/test_speculative_dspark.cpp:99-107` already pins the floor at
the function; it stays, because it localizes a failure, and it is not the proof.

New file `tests/vllm/entrypoints/test_dspark_block_size_guard.cpp`, entering at
`LoadedEngine::ResolveSpecConfig`:

| Case | Today | After |
|---|---|---|
| Draft with `dspark_block_size: 7`, `k = 6` | accepted | throws, naming 7 and 6 |
| Draft with `block_size: 7` and no `dspark_block_size`, `k = 6` | accepted | throws, naming 7 and 6 |
| Either draft, `k = 7` | accepted | accepted, `num_speculative_tokens == 7` |
| Either draft, `k = 14` | accepted | accepted |
| Gemma4 draft with `block_size: 7`, no `k` | throws "requires num_speculative_tokens" | accepted, `k` defaults to 7 (`:973-979`) |
| Draft path that does not exist, `k = 7` | accepted | accepted, unchanged |

The last row is the regression guard for §3's filesystem-independence note.

§6b adds five cases to the same file. Three enter at `LoadedEngine::FromModelDir`
with a model directory that does not exist, which is where the second call site
now resolves:

| Case | Before §6b | After |
|---|---|---|
| Qwen3 draft, `block_size: 7`, `k = 6` | `model path is not a directory` — the floor was never consulted on the path a user takes | refused, naming the block and the `k` |
| Gemma4 draft, no `k` | refused with `got 0` against a `k` nobody supplied | resolves; the missing target directory is the failure |
| Qwen3 draft, no `k` | refused with `got 0` | refused with the shipped `requires num_speculative_tokens` message |

Two more cover the speculators config layout at `ResolveSpecConfig`. Nothing
committed covered that layout, although it is one of the two this engine ships,
and that is what those two cases pin. They do NOT drive the
`TranslateSpeculatorsDsparkConfig` call in `ReadDsparkDraftKeys`: `block_size`
sits at the TOP LEVEL of the raw speculators document, which is where the
translation's copied-key list reads it FROM, so a read of the untranslated
document finds the same 7. §6c measures it.

## 6. Gates

| Gate | Content | State |
|---|---|---|
| G1, focused | `test_dspark_block_size_guard` and `test_speculative_dspark` green, both reds captured first. | required, CPU |
| G2a, reachability of the `ResolveSpecConfig` site | Restore `std::nullopt, std::nullopt` at the `ResolveSpecConfig` call (`src/vllm/entrypoints/model_loader.cpp::ResolveSpecConfig`) in a scratch copy. `test_dspark_block_size_guard` must go red. A green here is the finding, not a pass. **This mutates ONE of the two sites**, which the row did not say until the fresh review found the other one unmeasured. | required, CPU |
| G2b, reachability of the `FromModelDir` site | Restore the pre-repair resolution inside `maybe_load_dflash` (`src/vllm/entrypoints/model_loader.cpp::FromModelDir`) in a scratch copy, and separately restore only its `ResolvedNumSpeculativeTokens()` argument. Each must red `test_dspark_block_size_guard`. Before §6b neither did: the site sat behind `LoadShards`, so no CPU test could reach it and reverting it left the suite 8/8 GREEN. | required, CPU |
| G3, run | On `Qwen3.8-27B` + `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153b`, decode at `k = 6` against the pinned oracle at the same `k`, greedy, same prompts. This is the gate that shows whether a short `k` garbles, and therefore whether §2's stricter-than-upstream refusal is right. | **owed**, needs a GPU lease |
| G4, spec-off | Decode with speculation off byte-identical before and after. The change touches only the dspark branch, so a difference is a defect in the change. | owed |

G3 is the one that settles §2. Until it runs, the divergence rests on upstream's
own comment and on the absence of any block-shaped weight in our draft path, not
on a measurement of our output. The row does not claim it.

## 6a. G1 and G2 RESULTS (2026-08-18)

CPU-only build, `-DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo`, at base
`65d6cdaed`. No GPU on this host, so G3 and G4 stay owed.

**G1 red, before the fix.** `test_dspark_block_size_guard` compiled cleanly
(`compile_rc=0`, so this is a real result and not a build failure reading as a
pass) and reported 8 cases with 4 passed and 4 failed, 17 assertions with 14 passed and 3
failed, `Status: FAILURE!`. The four failures were the intended ones:

```
TEST CASE:  the loader refuses k below the draft's dspark_block_size
  ERROR: CHECK_THROWS_AS( LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{}),
         std::invalid_argument ) did NOT throw at all!
TEST CASE:  the loader refuses k below the draft's block_size
  ERROR: CHECK_THROWS_AS( ... ) did NOT throw at all!
TEST CASE:  the refusal names the block and the k that was asked for
  FATAL ERROR: expected a refusal for k=6 against a block-7 DSpark draft
TEST CASE:  a Gemma4 draft's block_size defaults k
  ERROR: test case THREW exception: speculative-config: method "dspark" requires
         num_speculative_tokens (a DSpark draft config carries no n_predict)
```

The first two are the hole itself: a `k` below the block resolved without
complaint on both the upstream-keyed and the published-checkpoint config shape.

**G1 green, after the fix.** 8 cases with 8 passed and 0 failed, 22 assertions
with 22 passed and 0 failed, `Status: SUCCESS!`. The assertion count rises from 17 to 22
because the three previously-throwing cases now reach the checks past the
refusal, which is the shape a real green has here rather than a muted one.

No regression in the neighbours, each rebuilt and rerun at the same base:

| Suite | Result |
|---|---|
| `test_speculative_dspark` | 9 cases, 9 passed; 30 assertions, 30 passed |
| `test_loaded_engine_dense` | 19 cases, 19 passed; 87 assertions, 87 passed |
| `test_speculative_unknown_keys` | 9 cases, 9 passed; 63 assertions, 63 passed |
| `test_qwen3_dspark_config` | 8 cases, 8 passed; 25 assertions, 25 passed |

**G2, the reachability mutation.** `keys.n_predict, keys.block_floor` restored to
`std::nullopt, std::nullopt` at the `ResolveSpecConfig` call site only. The
mutation compiled (`compile_rc=0`) and `git diff --stat` confirmed it applied, so
neither of the two ways a mutation reads as a false pass is in play. The focused
gate went back to 8 cases with 4 passed and 4 failed, `Status: FAILURE!`. The test
therefore enters through the production call site rather than measuring the
class. The file was restored and its sha256 checked equal to the pre-mutation
value (`181250a7c130db41...`), and the suite returned to 8/8.


## 6b. The fresh review's findings, and the repair (2026-08-18)

The fresh review confirmed the fix, the divergence and the mutations of §6a, and
returned two findings. Both are repaired here, on the same CPU-only build
(`-DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo`), on top of a merge of
`origin/main` at `c20018f8d`.

**Finding 1, the second call site landed unreached and the record claimed the
opposite.** Diagnosis and consequences: §7 R4. The repair does not patch the
site's argument list. It DELETES the second resolution:
`LoadedEngine::FromModelDir` now resolves the DSpark config once, at the top of
the function, through `LoadedEngine::ResolveSpecConfig` — the same function the
constructor calls — and `maybe_load_dflash` consumes that result. One resolution,
one set of messages, one place the floor is applied. Patching the argument would
have left two implementations of one rule, which is the shape that produced the
finding.

The resolution is placed after the `.gguf` branch, which keeps its own named
refusal, and before every path, config, tokenizer and weight operation. Two
things follow. A `k` the draft cannot serve is refused before the loader maps the
target rather than after, which is the ordering the device resolution above it
already has. And the site becomes reachable from a CPU test: `FromModelDir` with
a model directory that does not exist now reaches the speculative resolution,
which is the shape `tests/vllm/entrypoints/test_loaded_engine_dense.cpp` uses to
pin that device resolution.

**Finding 2, the refusal named a key the checkpoint does not carry.** The floor
falls back to `block_size` (§2), and on BOTH published Qwen3 drafts that fallback
is what supplies it, yet the message always said `dspark_block_size`. The key the
value was read from is now carried on `DsparkDraftKeys` and passed to
`SpeculativeConfig::ResolveDspark`, whose new `block_size_key` parameter defaults
to upstream's key — so a caller supplying the upstream field gets upstream's
wording unchanged. Upstream's third sentence, `Use num_speculative_tokens={n} or
larger (e.g. 7).` (`speculative.py:1021-1026`), was dropped when the check was
first ported and is restored in the same edit: the first two sentences say the
`k` is wrong, and only the third says what to type.

### Evidence

**RED first, before the repair**, with the new cases compiled against the
unrepaired tree (`compile_rc=0`, so this is a result and not a build failure):
13 cases, 11 passed, 2 failed; 35 assertions, 30 passed, 5 failed;
`Status: FAILURE!`. Both failures name the same thing — `FromModelDir` reached
the missing directory before it had any opinion about the speculative config:

```
TEST CASE:  the loader refuses a short k before it touches the model directory
  ERROR: CHECK( what.find("block_size (7)") != std::string::npos ) is NOT correct!
  logged: what: model path is not a directory: /nonexistent/vllm-cpp/dspark/target
TEST CASE:  the loader keeps the native Qwen3 no-k message at the draft load
  ERROR: CHECK( what.find("requires num_speculative_tokens") != std::string::npos ) is NOT correct!
  logged: what: model path is not a directory: /nonexistent/vllm-cpp/dspark/target
```

**GREEN after**, `test_dspark_block_size_guard`: 14 cases, 14 passed; 39
assertions, 39 passed; `Status: SUCCESS!`. Neighbours, rebuilt and rerun at the
same base: `test_speculative_dspark` 9/9 (30 assertions),
`test_loaded_engine_dense` 19/19 (87), `test_speculative_unknown_keys` 9/9 (63),
`test_qwen3_dspark_config` 8/8 (25).

**The mutations.** Each compiled (`compile_rc=0`), each was confirmed applied
with `git diff --stat`, and the file was restored from a byte copy and its
sha256 rechecked equal afterwards.

| Mutation | What it does | Result |
|---|---|---|
| G2b, the review's M4b | Restore the pre-repair resolution inside `maybe_load_dflash` and ignore the hoisted one | 14 cases, **2 failed**; 39 assertions, 5 failed. It was 8/8 GREEN before this repair. |
| G2b', the `k`-of-zero in isolation | Keep the hoist, resolve with the pre-repair argument list (`ResolvedNumSpeculativeTokens()`) | 14 cases, **2 failed**. Message observed: `DSpark requires num_speculative_tokens >= block_size (7); got 0` — the reviewer's defect, reproduced and caught. |
| G2a, the review's M1, which is also the §7 R5 landing hazard | Restore `std::nullopt, std::nullopt` at the `ResolveSpecConfig` site, which is what `row/DSPARK-QWEN3-ROUTING-IMPL` still carries there | 14 cases, **8 failed**; 31 assertions, 10 failed. The total is BELOW the green run's 39 because a `REQUIRE` abort ends three cases early, so a reader who expects 39 here is reading a shortened run and not a muted one. A conflict resolution that takes the routing branch's copy of that call cannot land silently. |
| The key-naming half | Drop `keys.block_floor_key` from the call | 14 cases, **1 failed**: the refusal says `dspark_block_size` where the 7 came from `block_size`. |

G3 and G4 remain owed. No GPU on this host, and no run gate is claimed.

## 6c. The second review's record findings, and what they measured (2026-08-18)

The second fresh review returned `PASS` on the code: both §6b findings closed,
the `FromModelDir` reordering verified safe, and every mutation of §6b's table
reproduced. Its remaining findings were about this record, and they are repaired
here. No behaviour changed.

**The speculators cases were credited with a guarantee they do not carry.** §5
and the test file both said that a read which did not translate first would find
no block in the speculators document and accept every `k`. That is false.
`block_size` sits at the TOP LEVEL of the raw speculators document — it is a
copied key, so `Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig` reads it
from the top level and writes it to the top level — and the verbatim
`RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` fixture in
`tests/vllm/models/test_qwen3_dspark_config.cpp` carries it there.

Measured, on the same CPU-only build as §6a and §6b: with the
`IsSpeculatorsDsparkConfig` / `TranslateSpeculatorsDsparkConfig` call DELETED
from `ReadDsparkDraftKeys` (`compile_rc=0`, `git diff --stat` confirming one file
and four deleted lines, so neither false-pass mode is in play), all five focused
suites stay GREEN: `test_dspark_block_size_guard` 14/14 (39 assertions),
`test_speculative_dspark` 9/9 (30), `test_loaded_engine_dense` 19/19 (87),
`test_speculative_unknown_keys` 9/9 (63), `test_qwen3_dspark_config` 8/8 (25).
The file was restored from a byte copy and its sha256 rechecked equal
(`c9f16c45e34cf0cc...`).

So the translate call is NOT load-bearing for the block floor on either shipped
layout, and no other reason was found that makes it so. It stays for the reason
§3 gives — the guard reads the same normalized shape `LoadDsparkDraft` builds,
so the two cannot disagree about a future layout — and that reason is now
recorded as unmeasured rather than asserted as tested. The two speculators cases
keep their value: they pin the floor on the second shipped layout, which nothing
committed covered.

**The G2a assertion total did not reproduce.** §6b recorded `39 assertions, 10
failed`; the run gives `31 assertions, 10 failed` over the same 14 cases with 8
failed. A `REQUIRE` abort ends three cases before their later assertions run, so
the total FALLS below the green run's 39. The failure counts were correct and are
load-bearing; the total is corrected in place. §6b's M4b row was re-derived at
the same time and reproduces exactly (14 cases, 2 failed; 39 assertions, 5
failed).

**Two records described the pre-repair shape.** The `SPEC-DSPARK-BLOCK-SIZE-GUARD`
row in `.agents/engine-matrix.md` and
`.agents/claims/CLAIM-SPEC-DSPARK-BLOCK-SIZE-GUARD.md` still said BOTH call sites
pass `std::nullopt`, scoped the claim to the `ResolveDspark` ARGUMENTS at two
line anchors that the repair deleted, and reported the pre-§6b `8/8, 22
assertions`. Both are rewritten to the landed shape, without line anchors, which
is what let them go stale inside one pull request.
`.agents/issue-index.md` carries the same stale anchors and is deliberately NOT
edited: it is append-only and carries `merge=union`, so an edited row duplicates
rather than merges. `docs/STATUS.md` describes the historical defect and is left
alone, because as history it is still true. `docs/USAGE.md` said "one recorded
divergence" beside the `--speculative-config` row and now names both, because
§2a's ordering is user-visible on that flag.

**The upstream citation for the refusal's wording was off by three lines.** The
message is at `speculative.py:1021-1026`; `:1018` is the comparison in the `if`.
Corrected in §3, in §6b and in the comment above the throw in
`include/vllm/config/speculative.h`. The `(e.g. 7)` that reads like a duplicate
of the `num_speculative_tokens={n}` before it is upstream's own, at `:1025-1026`,
and is mirrored deliberately.

## 7. Risks and decisions

**R1. The floor could be wrong about our implementation.** Upstream says a short
`k` garbles for DeepSeek-V4 DSpark. Nobody here has shown it garbles for a Qwen3
draft in *our* speculator. The refusal is therefore conservative: it turns a
possibly-wrong output into a named refusal. If G3 shows a short `k` is merely
worse and not wrong, the right follow-up is to relax the fallback to a warning,
not to delete the threading. Recorded so the next reader does not re-derive it.

**R2. The `block_size` fallback could mask a future upstream key.** If a later
pin adds `dspark_block_size` to the Qwen3 drafts, the explicit key still wins,
because the fallback only applies when it is absent. The precedence order is the
mitigation.

**R3. Reading the config in `ResolveSpecConfig` adds a filesystem touch to a
function that had none.** Mitigated by returning empty on a missing
`config.json` rather than throwing, and pinned by the last row of §5.

**R4. Narrowing the early throw changes an error path.** Today a Gemma4 draft
with `block_size` and no `k` is refused; upstream defaults `k`. After the change
`ResolveDspark` raises the equivalent error when there is genuinely no
`n_predict`, so the native Qwen3 case keeps today's message. This is a mirror
repair.

**It was justified in `c9282725` with a claim that was FALSE when it was
written**, and the correction is kept here rather than quietly dropped. That
commit said the narrowing was "required for the threaded `n_predict` to be
reachable at all". It was not, because the threaded `n_predict` was unreachable
end to end for an unrelated reason: the SECOND call site, inside `FromModelDir`,
runs before the `LoadedEngine` constructor reaches `ResolveSpecConfig`, and it
passed `ResolvedNumSpeculativeTokens()` — `num_speculative_tokens.value_or(n_predict)`,
and therefore `0` for a user who named no `k`, because nothing fills `n_predict`
on the CLI-side config. A no-`k` run was refused THERE, with
`>= dspark_block_size (7); got 0`, before any default could apply. The fresh
review measured it with a linked probe: `ResolvedNumSpeculativeTokens() with no
user k = 0`, and both the Gemma4 and the native Qwen3 configs refused at site 2.

Two consequences followed and both are repaired in §6b. The `## Tests` row
"a Gemma4 draft's block_size defaults k" asserted a default production never
reached. And the shipped native-Qwen3 no-`k` lane silently changed message, from
`method "dspark" requires num_speculative_tokens (a DSpark draft config carries
no n_predict)` to a block-floor refusal quoting a `k` the user never supplied
against a key their checkpoint does not carry.

With site 2 delegating to `ResolveSpecConfig`, the sentence above is true rather
than aspirational: one resolution runs, it runs first, it applies the default,
and the narrowing is what lets it.

**R5. Collision with `SPEC-DSPARK-QWEN3-ROUTING`.** That row edits the same
`ResolveSpecConfig` dspark branch to add classification. The two changes are
semantically independent — it owns `IsDsparkDraft` and the architecture
normalization, this row owns the `ResolveDspark` arguments — but they will
conflict textually. The mitigation is shape: this row's footprint in that branch
is one helper call and the two argument lists, with the helper defined elsewhere
in the file. Whichever lands second resolves by taking the other's classification
lines whole and re-applying its own argument lines. Both rows read the same
`config.json`, so a later cleanup can hoist one read; neither row should do it
(§8 carries the count as owed).

**The hazard is specific, and it is now measured.** `row/DSPARK-QWEN3-ROUTING-IMPL`
still carries `ResolveDspark(std::nullopt, std::nullopt, cli.num_speculative_tokens)`
at that call, because it branched before this row. A resolution that takes its
side of the conflict whole therefore DELETES this guard while resolving cleanly.
§6b's G2a mutation is exactly that outcome, and it reds 8 of 14 cases in
`test_dspark_block_size_guard`, so the deletion cannot land silently. This row's
footprint in that branch stays one helper call and one argument list, and §6b
adds nothing to it: the second call site is in `FromModelDir`, which the routing
row does not touch.

## 8. Owed

- [#1225](https://github.com/mudler/vllm.cpp/issues/1225) — this row. Open,
  owned here.
- G3 and G4 of §6, both needing a GPU lease. G3 is the measurement that decides
  whether §2's divergence should stay a refusal or become a warning.
- `docs/USAGE.md` owes the `RadixArk/Qwen3.8-27B-DSpark` weight pin (repo,
  revision `85ef153be924f17ce4bf62726954eeaa4a73e854`, sha256
  `9d26d5e637551c244d543c67c790bd0947f360e005c569e5851a185ffe692786`). It is
  owed by `SPEC-DSPARK-QWEN3-ROUTING` W4, which reads the same checkpoint, and
  is recorded here so it is not written twice.
- If `SPEC-DSPARK-QWEN3-ROUTING` lands a named refusal for the DeepSeek-V4
  shape, the `dspark_block_size` half of this row's read becomes unreachable in
  practice, and only the `block_size` fallback carries. That does not make the
  read wrong, and it is the reason §2 argues the fallback rather than relying on
  the upstream key.
- **No `VT_CHECK` guards the `*dspark_spec` dereference in `maybe_load_dflash`,
  and that is deliberate.** The optional is filled by exactly the condition the
  dereference sits behind — `params.speculative_config->method == "dspark"` — so
  the dspark branch of `ResolveSpecConfig` cannot return `nullopt` and no input
  reaches an empty one. A check there would make the coupling local, but nothing
  can drive it, and an unreachable check is dead code under `AGENTS.md`
  `## Nothing lands dead`. The second review raised it and accepted this answer.
  If a later row lets that branch return `nullopt` — a classification that
  declines the method, say — the check becomes testable and is owed with the
  test that reds without it.
- **Splitting `FromModelDir` into a config phase and a load phase** would let the
  model-directory error keep its upstream position (§2a) while the `k` refusal
  still runs before any weight is read, which is the one shape that needs no
  divergence. It restructures a function two rows are editing, so it is not this
  row's work. Unowned; it needs an issue before anyone starts it.
- **The draft `config.json` is opened and parsed more than once per load, and no
  row owns hoisting it.** Today: `ReadDsparkDraftKeys` on each of the two
  `ResolveSpecConfig` calls a load makes — `FromModelDir`'s and the
  `LoadedEngine` constructor's — plus `LoadDsparkDraft`, so three. Once
  `SPEC-DSPARK-QWEN3-ROUTING` ([#1193](https://github.com/mudler/vllm.cpp/issues/1193))
  lands, `ReadDsparkDraftIdentity` rides the same two resolutions and the count
  is five. It is a five-layer config and nobody has measured a cost, which is
  why neither row hoists it: the hoist means restructuring `FromModelDir` while
  a second row edits the same branch. Recorded here so whoever lands second does
  not discover it, and so it is not fixed twice. §7 R5 owns the seam.

## 9. Stop conditions

- Stop and report if closing the hole turns out to require touching
  `IsDsparkDraft` or the method classification. Those belong to the in-flight
  row and a silent overlap is worse than a round trip.
- Stop if the fix cannot be built or exercised without CUDA. It can: every
  gate in §6 except G3 and G4 is CPU.
- Do not claim G3. A GPU run gate is owed until it runs.

## Now

`ACTIVE`, not `DONE`. The spec is committed,
[#1225](https://github.com/mudler/vllm.cpp/issues/1225) is open, and W1-W3 have
landed: the floor is threaded into the resolution the loader runs, the red was
captured before the fix, and the reachability mutations are recorded in §6a and
§6b. W4 is owed and needs a GPU lease, so the row cannot reach `DONE` here.

A fresh review then found the second call site landed unreached while §7 R4
claimed otherwise, and found the refusal naming a key the published checkpoints
do not carry. Both are repaired in §6b: the second resolution is deleted rather
than patched, `FromModelDir` resolves once through `ResolveSpecConfig` ahead of
the model load, and the message names the key the block was read from. Nothing
in this row is unreached, so nothing here takes the staged-slice exception.

A second fresh review returned `PASS` on the code and left three record
findings, repaired in §6c: a guarantee the speculators cases were credited with
and do not carry, the error-order divergence now argued in §2a instead of left in
a design paragraph, and one mutation total that did not reproduce.

The divergences of §2 and §2a are decided and recorded rather than deferred, and
G3 is the measurement that can overturn §2's. Until G3 runs, the claim this row makes is that
a `k` below the draft's block is REFUSED, not that a `k` below the block would
have garbled: the second is upstream's statement and our structural reading of
the draft path, not our measurement.
