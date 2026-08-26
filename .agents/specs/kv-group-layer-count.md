# FIX-KV-GROUP-LAYER-COUNT — the KV byte accounting counts placeholder names, so the engine allocates what it was never given

Work row `FIX-KV-GROUP-LAYER-COUNT`, owned by `ROAD-V1-MEM`
([`kv-sizing.md`](kv-sizing.md), issue
[#83](https://github.com/mudler/vllm.cpp/issues/83)) — which is the row the two
issue-index entries name, because the KV pool sizing is its surface.
Issues: [#1963](https://github.com/mudler/vllm.cpp/issues/1963),
[#1966](https://github.com/mudler/vllm.cpp/issues/1966).
Base: `d9a528528`. **Every `file:line` below is read at that base**, including the ones in files this row edits, so a reader diffing them against the head will find them moved.

## Now

`ACTIVE` in [#2000](https://github.com/mudler/vllm.cpp/pull/2000). One pull
request carries the spec and the implementation, in that commit order.

Three things are still outstanding and none of them is this row's to do alone:
a fresh scoped review of the immutable head, the operator rerunning the gate,
and the device confirmation below, which needs a lease this row does not hold.

**The device confirmation, with its predicted values stated first.** Same
launch as [#1963](https://github.com/mudler/vllm.cpp/issues/1963) —
`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` plus the `qwen3.8-27b-dflash2` draft,
`--max-model-len 32768 --max-num-seqs 32 --no-enable-prefix-caching
--speculative-config '{"method":"dflash","num_speculative_tokens":8}'`, with
`VT_KV_ALLOC_LOG=1`.

| Run | Base measured | Predicted with this change |
|---|---|---|
| `--kv-cache-memory 1073741824` | `page_size_bytes=131072 num_blocks=4096` | `page_size_bytes=131072 num_blocks=481` |
| paged bytes allocated, same run | 17 x 4096 x 131072 = 9126805504 B = 8.50 GiB | 17 x 481 x 131072 = 1071775744 B = 0.998 GiB |
| `--kv-cache-memory 6442450944` | `num_blocks=24576`, 51.00 GiB paged | `num_blocks=2891`, 5.999 GiB paged |
| recurrent state, either run | allocated 43.40 GiB, guard REPORTED 0.90 GiB | allocated 43.40 GiB, guard reports 43.40 GiB |

The recurrent allocation does not change; only what the guard says about it
does, so on a box with more than 43.40 GiB free the guard still does not
refuse — which is correct. The observable is the number in the refusal message
when it is forced (lower `MemAvailable`, or raise `--max-num-seqs`).

`num_blocks` is written for **16** target full-attention layers, which is what
the base measurement implies (the base divisor was two pages, one `fa` and one
`fa_draft`, and 1 GiB / 262144 = 4096). If the checkpoint has N rather than 16,
the prediction is `1073741824 / ((N + 1) * 131072)`, and a mismatch is a fact
about the checkpoint rather than about this change — but say so, because it
would mean the 8.5x in the issue is also the wrong multiple.

## The defect

`KVCacheGroupSpec::layer_names` is upstream's per-layer name list
(`vllm/v1/core/kv_cache_utils.py:1209-1211` builds it by appending EVERY layer
that shares a spec object, and `:1399` reads `len(group.layer_names)` as the
layer count that bounds the allocation). Thirty-three of our thirty-four
registries publish a single PLACEHOLDER string instead —
`std::vector<std::string>{"fa"}`, `{"gdn"}`, `{"mla"}`, `{"kda"}`,
`{"fa_draft"}`, `{"encoder"}`. `NemotronHForCausalLM` is the only one that
publishes real names (`nemotron_h_registry.cpp:274-277`), and it does so because
[#810](https://github.com/mudler/vllm.cpp/issues/810) found the same class of
bug from the other side.

The runner states the convention in its own source
(`src/vllm/v1/worker/gpu/runner.cpp:826-829`): *"Every registry shipping today
publishes a single PLACEHOLDER name per group"*. Three consumers then read
`layer_names.size()` as if it were the layer count.

| Consumer | `file:line` at the base | Verdict |
|---|---|---|
| `KVBytesPerBlock` | `src/vllm/v1/kv_cache_interface.cpp:263-264` | **WRONG** — under by the real layer count |
| `recurrent_state_bytes` | `src/vllm/v1/core/kv_cache_utils.cpp:975-981` | **WRONG** — under by the real layer count |
| `GroupLayerMask` (runner) | `src/vllm/v1/worker/gpu/runner.cpp:369-371` | correct — it refuses a placeholder by design and falls back |

`KVBytesPerBlock` itself has exactly two product call sites, and only one of
them is defective:

| Call site | `file:line` | Verdict |
|---|---|---|
| `ResolveNumBlocks` arm 2 | `src/vllm/entrypoints/model_loader.cpp:1416` | **WRONG** — divides an ABSOLUTE byte budget by a per-layer page |
| `ResolveMaxModelLen` | `src/vllm/entrypoints/model_loader.cpp:1564` | correct — the factor appears on BOTH sides (`available = num_blocks * bpb`, `needed = ceil(len/bs) * bpb`, `estimate_max_model_len(available, bpb, bs)`, `auto_fit_max_model_len(derived, available, bpb, bs)`) and cancels, reducing to a comparison in blocks |

The narrowing the dispatch stated therefore HELD, and no fourth product consumer
exists (`grep -rn 'KVBytesPerBlock' src include` returns those two plus header
prose and one comment cross-reference at `kv_cache_utils.cpp:1018`).

### #1963 — the pool is allocated N times over

`ResolveNumBlocks` computes `num_blocks = budget / KVBytesPerBlock(probe)`.
`GPUModelRunner::initialize_kv_cache` then allocates one `CacheBuffer` of
`num_blocks * page_size_bytes()` **per full-attention layer**
(`runner.cpp:974-977`) plus one for the draft layer (`:1131-1133`). With
placeholder names the divisor counts one layer per group, so the allocation
overshoots the budget by the real layer count.

Measured on device before this row (operator run, `--kv-cache-memory
1073741824`): `[kv-alloc] ... page_size_bytes=131072 num_blocks=4096`. Divisor
= 131072 (`fa`) + 131072 (`fa_draft`) = 262144, and 1 GiB / 262144 = 4096.
The 27B has 64 layers, 48 of them `linear_attention`, so 16 full-attention
layers plus 1 draft layer: `17 * 4096 * 131072` = **8.5 GiB allocated for a
1 GiB budget**.

### #1966 — the #371 recurrent-state OOM guard is 48x under

`recurrent_state_bytes` multiplies the Mamba page by `layer_names.size()` == 1
where the runner allocates a conv and an SSM buffer for each of the 48 GDN
layers (`runner.cpp:920-931`). At `--max-num-seqs 32`, `k=8` it reports 0.90 GiB
against a 43.40 GiB allocation. That guard exists because an oversized recurrent
state *"took the machine down rather than failing, which is exactly what it did
four times on 2026-08-11"* (`model_loader.cpp:1885-1890`), and on the only
family it can fire for it does not fire.

The two together account for most of the #1963 collapse arithmetically: at
`--kv-cache-memory 6GiB`, `k=8`, `--max-num-seqs 32` the base tree allocates
`17 * (6GiB / 262144) * 131072` = 54,760,833,024 B = **51.00 GiB** of paged
pool plus **43.40 GiB** of recurrent state = **94.40 GiB**, against the ~108 GB
(100.6 GiB) the watchdog saw. The residue is the weights and the transients the
issue's third candidate names, and this row does not claim it.

## What upstream does, and why it cannot have this bug

`vllm/v1/core/kv_cache_utils.py` at the pin `555967922`:

- `:1399` `group_size = max(len(group.layer_names) for group in kv_cache_groups)`
- `:1400-1402` one uniform `page_size` for every group
- `:1005-1008` `get_num_blocks` -> `available_memory // page_size // num_layers`
- `:1409-1416` allocates exactly `group_size` tensors of `page_size * num_blocks`

The count that divides the budget and the count that multiplies the allocation
are **the same expression over the same list**, so the product is bounded by
construction. Ours are two independent derivations of one number, which is the
shape that can disagree — and did.

## Design

**Make `layer_names` mean what upstream means by it, in ONE place, derived from
the same predicate the runner allocates by.**

A new `vllm::v1::ResolveKVCacheGroupLayerNames(KVCacheConfig&, num_hidden_layers,
layer_types)` rewrites placeholder group names into the real per-layer module
names the runner's own classification implies. It is called from
`LoadedEngine::MakeKVCacheMaybeSpec` (`model_loader.cpp`), the single funnel
through which every production `KVCacheConfig` passes — the spec-on branch
(`MakeQwen3_5KVCacheSpec`) and the spec-off branch
(`ModelRegistry::MakeKVCache`, all 34 registries) both return through it, and
`MakeKVCacheResolved` calls it for both the probe and the resized config.

The classification is a **mirror of `runner.cpp`'s fallback**, not a second
derivation of it:

- `is_gdn(l)` == `has_mamba_group && !layer_types.empty() && layer_types[l] ==
  "linear_attention"` — character for character the runner's own predicate
  (`runner.cpp:905-908`).
- the TARGET attention group is the first non-eagle attention-kind group, which
  is the runner's own first-wins selection (`runner.cpp:586-592`).
- a SECOND attention group is the draft layer: exactly one layer, at index
  `num_hidden_layers`, mirroring upstream's MTP head index
  (`qwen3_5_mtp.py:105-112`) and the runner's single draft buffer
  (`runner.cpp:1128-1134`, which `break`s after one).
- a THIRD or later attention group gets an EMPTY list, because the runner
  allocates no buffer for it. Zero is the honest count, and no registry emits
  one today.

Because the resolver reproduces the allocator's predicate rather than
re-deriving the model's shape, the accounting cannot disagree with the
allocation. That equality is what the gate asserts, in bytes, from a real
registry config.

**A registry that already publishes real names is never overwritten.** If any
group carries a name that `KVCacheLayerIndexOfName` resolves, the resolver
returns untouched. NemotronH is that case, and it knows more than the fallback
can: its `layer_types` is empty and its MoE blocks cache nothing, which is the
whole point of #810.

`LayerIndexOfName` moves out of `runner.cpp`'s anonymous namespace into
`vllm/v1/kv_cache_interface.{h,cpp}` as `KVCacheLayerIndexOfName`, so the
resolver and the runner parse names with one function rather than two copies.

`is_eagle_group` is deliberately NOT set on the `fa_draft` group. It would be
semantically apt, but `KVCacheCoordinator` reads that flag
(`kv_cache_coordinator.cpp:104-114`) and flipping it changes which groups the
eagle path treats as draft groups. That is a different change with a different
blast radius.

### The instrument

`GPUModelRunner::kv_cache_allocated_bytes()` sums the byte size of every
`CacheBuffer` `initialize_kv_cache` created (`full_attn_buf_`, `draft_attn_buf_`,
`ssm_buf_`, `conv_buf_`), and `kv_cache_allocated_paged_bytes()` sums only the
block-scaled half. `CacheBuffer` records the size it was constructed with. These
report what the allocator DID, in its own output, so the gate compares an
accounting number against a measurement rather than against a second
transcription of the same formula.

## Tests

The cases live where their harness already lives, not in a new file with a
copied harness: the engine-level ones in
`tests/vllm/entrypoints/test_loaded_engine_dense.cpp` (which builds a
`LoadedEngine` over synthetic in-memory weights, the production entry point),
and the "real names survive" one in
`tests/vllm/models/test_nemotron_h_scaffold.cpp` (which builds the real
NemotronH KV config). Every case is named `kv-group-layer-count: ...` so one
doctest filter runs the set.

Every engine case is driven from `MakeQwen3_5KVCacheSpec` — the shipping
registry helper — through the loader, never from a hand-built
`KVCacheGroupSpec`. The pre-existing
`tests/vllm/v1/test_kv_cache_interface.cpp:425-433` is exactly the trap this row
exists to close: it hands the function `KVCacheGroupSpec{{"layer1", "layer2"},
ref}`, a shape no registry emits, so the multiplier under test was never the
multiplier in production. That case stays — it pins the formula — and is no
longer the only coverage.

The config is a 6-layer Qwen3.5 hybrid, `layer_types` = [LA, LA, FA, LA, LA,
FA]: **two** full-attention layers and **four** GDN layers, so the placeholder
count is wrong on both halves and by different factors. `MakeDenseConfig`, which
every other case in that file uses, has [LA, LA, LA, FA] — exactly ONE
full-attention layer, which is the config in which this bug is invisible.

1. **`--kv-cache-memory` bounds the bytes the runner allocates.** `LoadedEngine`
   with `kv_cache_memory_bytes = 1 MiB` ->
   `runner().kv_cache_allocated_paged_bytes() <= 1 MiB`.
2. **`KVBytesPerBlock` equals the per-block cost the allocator paid.**
   `KVBytesPerBlock(cfg) * cfg.num_blocks ==
   runner().kv_cache_allocated_paged_bytes()`.
3. **The loader resolves the placeholders.** The raw registry config carries one
   name per group; the engine's carries 2 and 4.
4. **`recurrent_state_bytes` equals the recurrent bytes the runner allocates.**
   `recurrent_state_bytes(cfg, max_num_seqs) == allocated - paged`.
5. **The `fa_draft` group weighs exactly one layer.**
   `MakeQwen3_5KVCacheSpec(num_spec=4)` -> `KVBytesPerBlock == page * 3`, and
   the draft name is `model.layers.<num_hidden_layers>.self_attn.attn`.
6. **NemotronH is not overwritten.** Its 6 attention names and 23 Mamba names
   survive the resolver and `KVBytesPerBlock` stays `page * 6`.

Cases 1 to 4 enter through the `LoadedEngine` constructor, so they are also the
reachability gate: the chain is `LoadedEngine` ctor -> `MakeKVCacheResolved` ->
`MakeKVCacheMaybeSpec` -> `ResolveKVCacheGroupLayerNames` -> `ResolveNumBlocks`
-> `GPUModelRunner::initialize_kv_cache`, and deleting the resolver call site
reds all four. Case 5 calls the resolver directly and stays green under that
mutation, which is correct and is why it is not the reachability case.

### Red before, green after

Mutation: the `ResolveKVCacheGroupLayerNames` call in `MakeKVCacheMaybeSpec`
replaced by `// MUTATION: production call site deleted.`, rebuilt, rerun.

```
[doctest] test cases:  5 |  1 passed | 4 failed | 19 skipped
[doctest] assertions: 21 | 16 passed | 5 failed |

CHECK( allocated <= params.kv_cache_memory_bytes )        2097152 <= 1048576
CHECK( KVBytesPerBlock(kv) * kv.num_blocks == ...paged )  1048576 == 2097152
CHECK( kv_cache_groups[0].layer_names.size() == 2 )       1 == 2
CHECK( kv_cache_groups[1].layer_names.size() == 4 )       1 == 4
CHECK( recurrent_state_bytes(cfg, 4) == recurrent )       4992 == 19968
```

Exactly 2x on the paged half (2 full-attention layers) and exactly 4x on the
recurrent half (4 GDN layers), which is the defect stated in bytes. Restored,
rebuilt, rerun: `5 passed | 0 failed`, `21 assertions | 21 passed`.

## Gates

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build -j 4
./build/tests/test_loaded_engine_dense -tc='kv-group-layer-count*'
./build/tests/test_nemotron_h_scaffold
ctest --test-dir build --output-on-failure
scripts/agent-preflight.sh --staged
```

**Results.** On the pre-merge head: `627/627 tests passed`, `CTEST rc=0`. On
the head after merging `origin/main` and with the box under heavy contention
from other worktrees' builds: `625/627`, with `test_dflash2_ctx_capacity` and
`test_async_llm` red under `ctest -j 4`. Both are green when re-run serially on
the same binary — `6/6` and `77 assertions` for the first, `15/15` and `494
assertions` for the second — which is the starvation case
[`verification.md`](../verification.md) names, not a regression. Neither test
reads `layer_names`, `KVBytesPerBlock` or `recurrent_state_bytes`.

The build type is left unset, exactly as the CPU CI job configures it
(`.github/workflows/ci.yml`). A `RelWithDebInfo` tree of this repository links
about 170 test executables against a static `libvllm.a` carrying debug info and
takes **86 GB**; that filled the disk twice during this row and made unrelated
checkers emit ENOSPC as policy refusals. The CI configuration builds the same
sources into about 4 GB.

## Risks

- **The runner's by-name membership path now engages for hybrids.** With real
  names published, `GroupLayerMask` resolves and `membership_by_name` becomes
  true for Qwen3.5 and Kimi-Linear, where it was false. The masks are derived
  from `layer_types` — the exact predicate the fallback uses — so the
  classification is identical by construction. Case 1 and case 2 measure the
  resulting allocation in bytes rather than asserting the claim.
- **Dense models are untouched in the runner.** `gdn_layer_mask` is only
  computed when the config has a Mamba group, so a dense or MLA model keeps the
  fallback classification whatever its names are. Only the byte accounting
  changes, from 1 layer to `num_hidden_layers`.
- **A pool sized from a budget now buys N times fewer blocks.** That is the
  fix, not a regression: the blocks it stops promising were never allocated
  within the budget. A user who had tuned `--kv-cache-memory` against the old
  behaviour will see a shorter auto-fitted context, and `ResolveMaxModelLen`
  reports that on stderr already.

## Out of scope

- `--gpu-memory-utilization` being inert ([#83](https://github.com/mudler/vllm.cpp/issues/83))
  and `kFallbackNumBlocks = 256`. Neither reads `layer_names`.
- Making the registries publish real names at their own call sites. The central
  resolver reaches all 34 without 33 hand-written loops that could each be
  wrong; a registry that wants finer classification than `layer_types` gives
  publishes its own names and the resolver stands aside, exactly as NemotronH
  does today.

## Stop conditions

Stop and report if the resolver changes the allocated byte total for any
existing model other than through the layer count — that would mean the mirror
of the runner's predicate is not a mirror.
