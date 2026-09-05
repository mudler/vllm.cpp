# `MODEL-DSV4-PAGED-ENTRY` — an EXL3 DeepSeek-V4 checkpoint reaches a paged KV cache from a production entry point

Issue: [#2447](https://github.com/mudler/vllm.cpp/issues/2447).
Owning row: `MODEL-DSV4-EXL3` (`.agents/specs/model-dsv4-exl3.md`), whose
`## Owed` carries the residuals this wave does not close.
Oracle: vLLM at the parity pin `5559679229`, `vllm/models/deepseek_v4/`.

This file is a scoped wave spec rather than a section of the row spec because
`model-dsv4-exl3.md` is 2625 lines and three worktrees write it concurrently.
Per §"Records" a spec read by a glob is the sanctioned shape; a second file for
one row is not a second row.

## Now

`ACTIVE` — the route, the shared predicate and the reachability gate LANDED and
are green; see `## Evidence`. The route is proven on **f32 pages only**. The
published topology's `kI8` / `fp8_ds_mla` SWA pages stay refused by name until
`KV-DSV4-MULTICACHE` W8 lands the packed 584-byte store; see `## Owed`.

## The defect

`DeepseekV4ForwardExl3Paged` (`deepseek_v4.cpp:3823`) is the only DeepSeek-V4
forward that takes BOTH a paged KV cache and a `DeepseekV4CompressorState*`.
Every caller it has is a test. The registered `ForwardDeepseekV4ForCausalLM`
(`deepseek_v4_registry.cpp:110-141`) has three branches and none reaches it:

| branch | why it does not reach the paged arm |
|---|---|
| `input.gather_logits` | `DeepseekV4Model::ForwardDevice` opens `(void)attn_meta; (void)attn_kv;` and binds neither `paged_kv` nor `compressor` |
| `input.multi_kv != nullptr` | `DeepseekV4ForwardGgufPaged` refuses by name on `!weights.has_gguf_weights`; an EXL3 load sets `has_exl3_weights` and never that |
| otherwise | `DeepseekV4Model::Forward` -> `DeepseekV4ForwardExl3`, which binds neither |

**`ModelForwardInput::gather_logits` DEFAULTS TO TRUE**
(`model_registry.h:609`), and the runner sets it from
`LogitsGatherEnabled() && step.prompt_logprob_indices.empty()`
(`runner.cpp:2694`, passed at `:2724`). Only the two multimodal drivers pass
`false`. So the FIRST branch wins on every default step, and the second branch —
the GGUF paged one — is itself unreached on a default configuration. That is a
second instance of this defect and it is NOT this wave's; it is recorded under
`## Owed`.

### Severity: CONFIRMED as wrong tokens, not slow

`token_ids` is `step.input_token_ids` (`runner.cpp:2543`, passed at `:2724`) —
the step's SCHEDULED tokens, one per request on a decode step, not the prefix.
`ForwardComposeImpl`'s attention sets `kv_base = 0` and `n_keys = T` whenever
`be.paged_kv == nullptr` (`deepseek_v4.cpp:977-978`). So a decode step with no
paged cache bound attends over exactly the one new token at global position 0,
while `positions` still carries the true absolute position for RoPE. The output
is finite, correctly shaped, and computed over the wrong key set.

This is a code-reading conclusion over the complete chain, not a measurement:
nobody has run an engine on this path, because #2455 stops the artifact loading
first. The reachability gate below runs the same chain at synthetic geometry and
shows the page storage staying all-zero when the route binds no pages, which is
the observable form of the same claim.

**One qualification the gate MEASURED, and it narrows the severity.** On a
CPU-only build `DeepseekV4Model::ForwardDevice` refuses first, on
`VT_CHECK(deepseek_v4::V4DeviceKernelsAvailable(), kDevicePending)`. So on THAT
build the pre-fix behaviour was a loud refusal, not a silent wrong answer: the
mutation that deletes this row's branch reds by that refusal and never reaches
the page assertion (`## Evidence`, M1). The silent case is a CUDA-build
property, where `V4DeviceKernelsAvailable()` is true, `ForwardDevice` runs, and
nothing writes or reads a page. That build is exactly the one an engine runs on,
so the severity stands — but it is not observable on the gate host, and saying
otherwise would be claiming a measurement nobody took.

## Scope

**In scope.** One new branch in `ForwardDeepseekV4ForCausalLM` that routes an
EXL3 load with a published name-keyed cache set into
`DeepseekV4ForwardExl3Paged`; the compressor state that branch needs, persisted
on the `LoadedModel`; the narrowing of `ResolveDeepseekV4SwaPages` so it stops
refusing the very layers this arm exists to serve; the shared predicate that
makes the route and the composition ONE expression; and the reachability gate.

**Out of scope, with owners.** The packed `fp8_ds_mla` store
(`KV-DSV4-MULTICACHE` W8, in flight on `row/KV-DSV4-MULTICACHE-W8B`); a
per-request compressor state (`## Owed`); a `T > 1` prefill on the composed arm
(`## Owed`); the GGUF paged branch's own unreachability (`## Owed`); the
real-artifact token gate, which is weight-blocked by #2455.

## Upstream chain

Read at `5559679229`, under `vllm/models/deepseek_v4/` — **not**
`vllm/model_executor/models/`, where a `find` returns nothing.

| upstream | what it defines |
|---|---|
| `attention.py:140` | `use_fp8_ds_mla_layout: ClassVar[bool] = True` — the default cache format this tree publishes |
| `attention.py:454-533` | `attention_impl` — the three layer shapes the composed arm mirrors |
| `compressor.py:240-248` | `overlap = compress_ratio == 4`; `coff = 1 + overlap` |
| `sparse_swa.py:82,99` | the SWA block size and 576-byte alignment `MakeDeepseekV4KVCache` publishes |

Upstream keeps the compressor's carried state in the runner's KV-cache pool, as
three published groups. `MakeDeepseekV4KVCache` already publishes those three
groups here (`deepseek_v4_registry.cpp:374-381`). This wave does NOT consume
them; see `## The staged shortcut`.

## Design

### 1. One predicate, two readers

`ResolveDeepseekV4SwaPages` refuses ANY layer carrying a compressor
(`deepseek_v4.cpp:3604-3615`), which is 41 of the real artifact's 43 layers —
exactly the set this arm exists to serve. That refusal is stale relative to
`deepseek_v4.cpp:996-1002`, where a `cr == 4` or `cr == 128` layer IS admitted
when compressor state is supplied. Two derivations of one rule, now disagreeing.

The narrowing does not copy the composition's clauses into the resolver. It
EXTRACTS them:

```c++
bool DeepseekV4PagedArmComposesCompressor(const DeepseekV4Params& params,
                                          int64_t layer, bool dsa_dense,
                                          bool have_compressor_state);
```

`AttentionBlock` uses the return value as its `comp_arm`, so the expression it
enforces and the expression the route applies are the same bytes. The rule at
`model_registry.cpp:545-549` states the polarity: an inline copy is a second
derivation, and a second derivation is what can disagree with the one a test
pins.

`dsa_dense` is a parameter rather than a constant because the two paged arms
differ on it and must keep differing. On the GGUF arm `dsa_dense` is true, every
layer runs DENSE, and a compressor layer would attend the raw prefix — so the
resolver must keep refusing it. Passing `dsa_dense = true, have_state = false`
from the GGUF branch makes the helper return false for every compressor layer,
which is that branch's behaviour byte for byte.

### 2. Three refusals the route now owns

Each guards a silent wrong answer, not a crash.

- **`num_reqs != 1`** already lived in the resolver and is reused as-is. The
  route does not re-derive it: it calls the resolver and `VT_CHECK`s the string.
  `DeepseekV4CompressorState` has NO request dimension
  (`deepseek_v4.h:467-488`) — it is one sequence's state machine. At
  `num_reqs > 1` the shared state is not reliably loud: two requests at equal
  length sail past `CompressorLayerStep`'s `seen == kv_base` guard
  (`deepseek_v4_dsa.cpp:417-424`) into a plausible answer over a mixed history.

- **`T != 1` on the composed arm** is NEW. `MergeWindowAndCompressed` asserts
  `num_tokens == 1 || num_heads == 1` (`deepseek_v4_dsa.cpp:349-351`), so a
  prefill refuses at real geometry anyway — but inside the composition, with a
  message about LSE layouts. Refusing at the route names the row instead.

  **Chosen: refuse, not loop.** The alternative was to drive a `T`-token prefill
  as `T` single-token calls from the registry adapter. Rejected for two reasons.
  It is not gateable: there is no oracle for the composed arm at `T > 1` in this
  tree, so a loop would ship as an untested numeric claim. And routing prefill
  to the stateless path instead is not an escape either — `CompressorLayerStep`
  then fires `seen == kv_base` on the first decode step, because the state never
  saw the prompt. A refusal is limiting and correct; a loop would be neither
  proven nor cheap. Recorded under `## Owed`.

- **A non-float SWA page** is NEW, and it applies to BOTH paged arms.
  `vt::ConcatAndCacheMla` refuses a non-float cache dtype by name
  (`src/vt/ops.cpp:4082`) and the published SWA pages are `kI8` with
  `cache_dtype_str == "fp8_ds_mla"` (`deepseek_v4_registry.cpp:227,374`). The
  write would abort either way; refusing at the route says WHICH row owns the
  gap. `ApplyCacheDType`'s guard is NOT widened — widening it would let a
  packed page be written as if it were float, which is the wrong-tokens shape
  this whole row is about.

### 3. Where the branch goes, and what it returns

The new branch is placed FIRST, before `input.gather_logits`. That ordering is
the fix: `gather_logits` is true on every default step, so any branch after it
is unreachable on a default configuration.

It returns through `WrapV4DeviceLogits`, not `HostLogits`. Returning
`HostLogits` would NOT red `check-runner-routing-consistency.py` — the delegate
hop keeps `deepseek_v4` classified DEVICE — so the checker would keep asserting
device logits while the arm that runs downloads them. That is a false green, and
it is why the new branch calls a wrapper-returning entry
(`DeepseekV4ForwardExl3PagedLogits`) rather than wrapping at the registry.

On a CPU queue `WrapV4DeviceLogits` and `HostLogits` produce the same value, so
this half of the design is a code-reading claim that the CPU gate cannot
separate. Said plainly rather than asserted.

### 4. The staged shortcut: state on the `LoadedModel`

`DeepseekV4CompressorState` must PERSIST across steps, and
`DeepseekV4LoadedModel` (`deepseek_v4_registry.cpp:58-67`) held only `weights_`.
It gains the state as a member, sized on first use. No `mutable` is needed: the
forward hook takes `LoadedModel&` non-const and `ModelAs<T>` returns non-const
(`model_registry.h:306`). Precedent: `Qwen3MoeLoadedModel::decode_graph()`.

**This is a deliberate divergence from upstream and is declared as one.**
Upstream's home for that state is the runner's KV-cache pool, and
`MakeDeepseekV4KVCache` ALREADY publishes three compressor-state groups
(`c4_attn_state`, `c4_indexer_state`, `c128_attn_state`) that this wave does not
read. A model-object member is one sequence's state by construction, which is
also why `num_reqs > 1` must refuse. Consuming the published groups is the
correct end state and is owed.

## Risks

| risk | mitigation |
|---|---|
| The narrowed resolver admits a layer the composition then refuses | the two share one expression; a mutation that widens the route reds the gate |
| The GGUF branch's behaviour moves | it passes `dsa_dense = true, have_state = false`, for which the helper is false on every compressor layer — its existing refusal, unchanged |
| The state member silently serves two requests | `num_reqs != 1` refuses, and the refusal is the resolver's own, not a copy |
| A test proves the member exists rather than that it persists | the gate never downcasts. It reads page bytes and a second step's survival |

## Tests

`tests/vllm/models/test_deepseek_v4_exl3_loader.cpp`, in the suite that already
owns the hermetic EXL3 fixture.

**The reachability gate.** Load the fixture through `ModelRegistry::Load`,
publish f32 SWA pages under the names `MakeDeepseekV4KVCache` uses, and call
`ModelRegistry::Forward` TWICE with `gather_logits` LEFT AT ITS DEFAULT — that
default is the defect — and `num_computed_tokens_cpu` `{0}` then `{1}`.

1. After step 1 the page storage for the COMPRESSOR layer is non-zero. Only
   `vt::ConcatAndCacheMla` writes it, and only the paged arm calls that.
2. Step 2 SUCCEEDS at `kv_base == 1`. That passes only if the state PERSISTED on
   the `LoadedModel` across calls; a per-call state has `seen == 0` and
   `CompressorLayerStep` throws.
3. `num_reqs == 2` throws, naming "one request per step only".

Assertion 1 is red if the new call site is deleted. Assertion 2 is red if the
state is constructed per call. Two production defects, one assertion each.

`DeepseekV4LoadedModel` is in an anonymous namespace, so a test cannot downcast
to read the state — and it must not gain a public accessor for the purpose. An
accessor would weaken the proof to "the member exists". The page bytes and the
second step's survival are the reachable observables.

## Gates

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_CUDA=OFF
cmake --build build -j 4 --target test_deepseek_v4_exl3_loader \
      test_deepseek_v4_paged_equiv test_deepseek_v4_gguf_load
ctest --test-dir build -R 'test_deepseek_v4_(exl3_loader|paged_equiv|gguf_load)' --output-on-failure
```

Mutations, each restored byte-for-byte (sha256) and each rebuilt with the binary
mtime asserted to have MOVED — a mutation that fails to compile reads as a
passing test:

| mutation | must red |
|---|---|
| delete the new registry branch | the route is gone |
| the branch is PRESENT but binds no pages | assertion 1 (pages stay zero) |
| construct the compressor state per call | assertion 2 (step 2 throws) |
| widen the route predicate to admit `num_reqs > 1` | assertion 3 |

The second row exists because the first cannot reach assertion 1 on a CPU-only
build: deleting the branch drops the step into `ForwardDevice`, which refuses on
`kDevicePending` before any page is touched. `M1b` keeps the branch and routes
it to the STATELESS EXL3 forward instead — the same defect shape, reachable on
this build — so assertion 1 is the only thing that can catch it.

## Evidence

### 2026-09-02, CPU-only build, `RelWithDebInfo`, head `3e2d8e42b`

Host `x86_64` Linux 6.8, no `nvcc`, no GPU. Build tree
`/home/mudler/_git/vllm.cpp/.wt/dsv4-paged/build`, configured

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_CUDA=OFF
```

**Green.** `ctest -R 'test_deepseek_v4_(exl3_loader|paged_equiv|gguf_load)'`:

```
1/3 Test #177: test_deepseek_v4_gguf_load .......   Passed    0.17 sec
2/3 Test #181: test_deepseek_v4_exl3_loader .....   Passed    0.86 sec
3/3 Test #295: test_deepseek_v4_paged_equiv .....   Passed    0.03 sec
100% tests passed, 0 tests failed out of 3
```

The reachability case alone, `--test-case='PAGED-ENTRY*' -s`: **18 assertions,
18 passed**, and the two observables read

```
compressor-layer page: 512 non-zero of 16384, L1 420.02
CHECK( nonzero_l1 > 0 )      values: CHECK( 512 >  0 )
CHECK( nonzero_beyond == 0 ) values: CHECK( 0 == 0 )
CHECK( nonzero_slot1 > 0 )   values: CHECK( 512 >  0 )
CHECK_THROWS( step( 3, 2, 2, 2) ) threw as expected!
```

512 is `head_dim`, i.e. EXACTLY one cache row after step 1 and one more after
step 2 — not "the page is dirty".

`REQUIRE( in.gather_logits )` reads `true` on every step, so the case drives the
default the runner sets, which is the whole point.

**Mutations.** Harness rebuilds after each edit, asserts the binary mtime MOVED
(a mutation that fails to compile reads as a passing test), runs both suites,
restores with `git checkout --` and asserts the five touched files hash back to
the same sha256. Baseline and final rebuild both green.

| mutation | mtime moved | result |
|---|---|---|
| M1 delete the new registry branch | both | `exl3_loader` rc=1, 21 cases / 20 passed, THREW `DeepseekV4 DEVICE forward (W7-device) not implemented` at `deepseek_v4.cpp:4001` |
| M1b branch present, binds no pages | both | `exl3_loader` rc=1, **602 assertions / 2 failed**: `CHECK( nonzero_l1 > 0 )` reads `CHECK( 0 > 0 )`, and `CHECK( nonzero_slot1 > 0 )` reads `CHECK( 0 > 0 )` |
| M2 construct the compressor state per call | both | `exl3_loader` rc=1, THREW `the carried state has seen 0 tokens but this step resumes at kv_base 1` at `deepseek_v4_dsa.cpp:417` — AFTER step 1 wrote its 512 page values, so step 1 passed and only the persistence failed |
| M3 widen the route predicate to `num_reqs < 1` | both | `exl3_loader` rc=1 (`CHECK_THROWS( step(3,2,2,2) ) did NOT throw at all!`) AND `paged_equiv` rc=1, 170646 assertions / 2 failed |

Every mutation printed `restored: sha256 identical`.

**M1 is the one that did not land where it was aimed**, and the reason is in
`## The defect` above: on a CPU-only build the deleted route falls into
`ForwardDevice`, which refuses on `kDevicePending` before touching a page. That
makes M1 a proof that the branch is REACHED, not a proof that assertion 1 is
load-bearing. M1b supplies the second proof. Both are recorded; neither is
presented as the other.

**What this evidence does NOT cover.**

- The published `kI8` topology. It is refused by name, and `KV-DSV4-MULTICACHE`
  W8 owns the store. Nothing here ran on a packed page.
- The device-resident return. `WrapV4DeviceLogits` and `HostLogits` are the same
  value on a CPU queue, so this host cannot separate them. The reason for
  choosing the former is static: `check-runner-routing-consistency.py` keeps
  `deepseek_v4` classified DEVICE across the delegate hop, so a `HostLogits`
  return would be a false green rather than a red.
- Any token-exactness claim against the oracle. Weight-blocked by #2455.
- Any GPU, any real checkpoint, any throughput number.

## Owed

- **The packed `fp8_ds_mla` SWA store.** The published pages are `kI8`; the
  route refuses them by name. `KV-DSV4-MULTICACHE` W8 owns the 584-byte packed
  store and read. Until it lands this route is proven on f32 pages only and
  cannot claim a working production route on the published topology.
- **A per-request compressor state.** Owned by the runner's pool upstream, and
  the three published state groups already exist. Until then `num_reqs > 1`
  refuses.
- **`T > 1` prefill on the composed arm.** Refused by name. Needs either a
  general `MergeWindowAndCompressed` (a transpose where the LSE layouts do not
  coincide) or a gated single-token loop with an oracle to gate it against.
- **The GGUF paged branch is unreachable on a default configuration** for the
  same reason this row's arm was: it sits after `if (input.gather_logits)`,
  which is true on every default step. Not fixed here — moving it is a change to
  the GGUF arm's reachability with its own gate — and tracked on #2447.
- **The real-artifact token gate.** Weight-blocked by #2455.

## Stop conditions

- Stop if closing the gap needs `ApplyCacheDType`'s guard widened. That admits a
  packed page to a float write and is a wrong-tokens change, not a fix.
- Stop if the reachability gate can only be made green by a public accessor on
  `DeepseekV4LoadedModel`. That proves the member exists, not that anything
  reaches it.
- Stop if the route and the composition need two expressions to agree. They are
  one expression or the row is not done.
