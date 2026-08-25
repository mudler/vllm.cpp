# SPEC-DFLASH2 embed device dedup — one device copy of the target's embedding table

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding).
**Issue:** [#1946](https://github.com/mudler/vllm.cpp/issues/1946).
**Parent spec:** [dflash2-spec-decode.md](dflash2-spec-decode.md). W9 (#1849) fixed the HOST
half of this duplication and said so in its own comment; this is the DEVICE half.
**Kind:** bug fix. One pull request carries the spec and the implementation, which is the
repository default; the commit order proves the spec came first.

## Why

The DFlash/DFlash2 draft does not own an embedding table. It runs the TARGET's table over its
own hidden states, which is why the z-lab checkpoint ships no `embed_tokens` at all. Our loader
expresses that by READING the target's table a second time into the draft's own `OwnedTensor`.

`ResidentWeight` caches the device upload on the `OwnedTensor` itself — the `if (!w.d_dev)`
guard in `include/vllm/model_executor/models/dense_attn_block.h::ResidentWeight`. Two `OwnedTensor`s
therefore mean two `d_dev` allocations of the same bytes, whatever the host residency is.

On `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` the table is BF16 `[248320, 5120]`
(`tests/vllm/models/qwen38_27b_modelopt_mtp_s1_manifest.inc:22`, generated from a range read of
the real shard header), which is 248320 x 5120 x 2 = **2,542,796,800 B (2.543 GB)** held twice.

W9 (#1849) made both HOST reads borrow-first, and the comment it left at
`src/vllm/entrypoints/model_loader.cpp:358-360` scopes itself to the host explicitly. The device
half was never touched. GB10 is unified memory, so a device allocation IS host RAM: this is
2.543 GB of a 119 GiB box that a `--max-num-seqs 32` ladder cannot use, on a box that has
OOM-rebooted under load.

## Upstream chain

vLLM does not implement DFlash2 at our parity pin (`5559679229`). It is implemented on vLLM
`main`, merged 2026-08-21 at `b389ac29465b33f9e9c534df221ea3c129e9793f` (vllm#52816). This is
**beyond-pin vLLM**, not a secondary oracle, and it is the same head the parent row's `## Upstream
chain` already anchors.

`vllm/v1/worker/gpu/spec_decode/dflash/utils.py:64-74 @ b389ac29` REBINDS the module by
reference rather than loading a second copy:

```python
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed
```

`_should_share` (`vllm/v1/worker/gpu/spec_decode/eagle/utils.py:12-25 @ b389ac29`) shares
UNCONDITIONALLY when the draft declares no own table (`draft is None` ⇒ `return True`), and only
compares bytes when the draft does declare one. The DFlash draft declares none, so upstream's
answer on this lane is an unconditional rebind. Our `Qwen3_5MTPModel` already mirrors exactly
this shape for the MTP lane — it holds `const OwnedTensor* embed_tokens_` pointing at the
target's tensor (`include/vllm/model_executor/models/qwen3_5_mtp.h:81-86`, `has_own_embed_tokens()
== false`). The DFlash lane is the one that did not.

The PP guard is not portable: this tree has no pipeline-parallel rank split, so the `world_size
== 1` arm is the only arm that exists here.

## Scope

**In scope:** the draft's EMBEDDING TABLE, on both draft containers (safetensors and GGUF) and
both target containers, made to share ONE `OwnedTensor` — and therefore ONE `d_dev` — with the
target.

**Out of scope, deliberately:** the `lm_head` (~0.715 GB each side after Marlin repack). Its
dedup is blocked on the O29 Marlin-body convergence recorded under the parent spec's `## Owed`
O3, and nothing here touches `lm_head`, `lm_head_fp4` or `lm_head_dequantized`.

**Also out of scope:** the DSpark lane's own shared embed
(`LoadDsparkDraft`'s `draft->dspark->backbone.embed_tokens` fallback). A
DSpark checkpoint usually SHIPS its own table, so the sharing is a fallback rather than the rule,
and the bind below is guarded to leave it alone. Recorded under `## Owed`.

## Design

### The seam: rebind at the ONE place both loaders meet

The draft is built by three different callers — the GGUF branch
(the `method == "dflash"` arm of `FromModelDir`'s GGUF branch), the two safetensors branches
(`maybe_load_dflash`), and the in-memory `LoadedEngine` overload W3 added for the runner gate. All
three funnel into the ONE private `LoadedEngine` constructor
(the `std::unique_ptr<LoadedModel>` overload in `model_loader.cpp`), which is the first point where the target
`LoadedModel` and the `DflashDraft` both exist. That is where the rebind goes. Putting it in
`LoadDflashDraft` instead would cover the two disk paths and miss the in-memory one — which is
the path every DFlash2 gate in this repository drives.

Three pieces:

1. **`LoadedModel::shared_embed_tokens()`** — a new virtual on the type-erased base, defaulting
   to `nullptr`, returning the target's embedding `OwnedTensor` when the concrete model has one.
   Overridden by `Qwen3_5DenseLoadedModel` and `Qwen3_5MoeLoadedModel`, which are exactly the two
   models that override `supports_aux_multi_tap()` and therefore exactly the targets the DFlash
   loader admits (`FromModelDir`'s `!model_->supports_aux_multi_tap()` guard refuses any other by name). This
   mirrors `BuildMtpDraft`'s existing polarity on the same base: typed access to a target-owned
   tensor without breaking the erasure.

2. **`Qwen3DFlashWeights::shared_embed_tokens`** — a borrowed `const OwnedTensor*`, null by
   default, plus an `EmbedTable()` accessor returning `*shared_embed_tokens` when set and the
   owned `embed_tokens` otherwise. All four draft embed sites
   (`src/vllm/model_executor/models/qwen3_dflash.cpp::EmbedTable`) call
   `ResidentWeight(d, weights.EmbedTable(), ...)`, so a rebind cannot reach three of four.

3. **`BindDflashDraftSharedEmbed(DflashDraft&, const LoadedModel&)`** — the rebind itself,
   EXPORTED from `include/vllm/entrypoints/model_loader.h` rather than left file-local. That is
   the same argument W9 made when it exported `LoadDflashSharedEmbedBf16`: the lever is one that
   a deletion mutation can remove silently, so the gate must reach the exact function production
   calls. It sets `weights.shared_embed_tokens` and then CLEARS `weights.embed_tokens`, which is
   the literal mirror of upstream's `del draft_inner.embed_tokens`.

### Why clearing the draft's own tensor is part of the fix, not tidiness

Leaving the draft's `OwnedTensor` populated leaves a second `d_dev` FIELD reachable. Any future
call site that reads `weights.embed_tokens` directly instead of `EmbedTable()` would re-open the
2.543 GB, silently and with every gate green. Clearing it makes that call site upload from an
EMPTY table and fail loudly instead. It also drops the second host borrow into the target's
safetensors mapping, which is the W9 residency this change supersedes for the embed.

### The device pointer is stable, and does not depend on who touches it first

This is the hazard the row was dispatched with, and the design answers it structurally rather
than by ordering: **after the rebind there is exactly ONE `OwnedTensor`**, so there is exactly one
`d_dev` field, allocated once by whichever side calls `ResidentWeight` first and never
reallocated (`include/vllm/model_executor/models/dense_attn_block.h::ResidentWeight`'s `if (!w.d_dev)`). Load order cannot change which side
allocates, because there is no second thing to allocate. The pointer is freed only when the
TARGET's `OwnedTensor` dies.

That last clause is a real lifetime change, so the constructor's member-declaration order moves
with it. `dflash_draft_` was declared BEFORE `model_`
in `include/vllm/entrypoints/model_loader.h`, which destroys `model_` FIRST and leaves the
draft holding a pointer into a dead target through its own destructor. `model_` now precedes
`dflash_draft_`, so the target outlives the draft. Both stay ahead of `runner_`, which is what
their existing comments require. Nothing in the initializer list depends on the swapped order:
both are plain `std::move` of a constructor parameter.

**The embed is never read inside a captured CUDA graph, on either side.** Draft: the graph
driver in `src/vllm/model_executor/models/qwen3_dflash.cpp::ForwardBlockLogitsWithDeviceKV` refreshes
the persistent graph inputs "ALWAYS OUTSIDE the captured region", and the embed precedes both
its `vt::BreakableGraph::Replay` and its `vt::GraphCaptureScope`. Target:
`src/vllm/model_executor/models/qwen3_5.cpp::EmbedInto` is host-fed and
"illegal inside a capture region"; the graph driver runs it per step into a persistent hidden
buffer and captures only `ForwardLayers` over that fixed address. So no graph bakes the table's
address today. Even if one did, this change makes the address STRICTLY more stable than before,
because the surviving allocation is owned by the target and now outlives the draft rather than
the other way round.

### When the rebind does NOT fire, and why that is fail-safe

`BindDflashDraftSharedEmbed` binds only when the target's table and the draft's own read agree on
dtype, rank and shape. That guard is not decorative: on the GGUF arm the target's table can be
kept **F16** in place (`LoadEmbedAndHead`'s `kKeepF16` arm in
`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`) while the draft's shared read
`LoadGgufSharedEmbedAndHeadBf16` always produces **BF16**. Those are different bytes and must not
be aliased. A mismatch keeps today's behaviour exactly — two tables, correct tokens — and prints
one line naming both dtypes and shapes, so a lane that silently stopped deduping is visible
rather than inferred. It is not a refusal: this is a memory change and must not turn a working
load into a failed one.

It also skips a DSpark draft (`draft.dspark != nullptr`) and an empty draft table, so the
out-of-scope lane is untouched by construction rather than by luck.

### What the production path says

The bind prints one line on `std::cerr`, beside the "DFlash draft loaded from ..." line the
loader already writes:

```
vllm.cpp: DFlash draft embed SHARED with the target (one device copy, 2542796800 B saved)
```

The byte count comes from the tensor that was actually bound, so it is a measurement rather than
a message. On a GB10 that number is the whole point of the change and a user should not have to
infer it.

## Risks and decisions

**D1 — a virtual on `LoadedModel` rather than a `dynamic_cast` in the loader.** The loader would
otherwise have to try `Qwen3_5DenseLoadedModel` then `Qwen3_5MoeLoadedModel` by hand, which is a
second dispatch table beside the registration and goes stale the moment a third target gains the
aux tap. The base already carries `BuildMtpDraft` for the same reason and in the same words.

**D2 — bind in the `LoadedEngine` constructor, not in `LoadDflashDraft`.** Three loader call
sites build a draft and only two of them are inside `LoadDflashDraft`. The constructor is the one
seam all three cross, and it is also where the target model is available as a `LoadedModel`
rather than as a concrete weights struct.

**D3 — no byte comparison of the two tables.** Upstream does not compare when the draft declares
no own table, which is this lane. Comparing 2.543 GB at load to prove what provenance already
guarantees — both reads name the same tensor in the same file — would cost a full pass over the
mapping and fault in every page of a table the load has otherwise never touched. The dtype/rank/
shape check is what discriminates the one case where provenance does NOT hold (GGUF keep-f16),
and it is cheap.

**D4 — the mismatch arm warns and continues.** Refusing would convert a memory optimisation into
a load failure on the GGUF keep-f16 lane, which works today. The line is loud enough to find.

**D5 — member reorder over an explicit destructor.** A user-declared `~LoadedEngine` would also
fix the order, but this file already expresses every lifetime relation it has through declaration
order and says so in the comments beside each member. A second mechanism for the same rule is the
"two descriptions" failure `AGENTS.md` `## Changing the rules or a checker` names.

## Tests

New binary `tests/vllm/v1/spec_decode/test_dflash2_embed_dedup.cpp`, two legs. Neither can pass
before the change and neither is satisfied by the host sharing W9 already landed.

**T1 — the DEVICE bytes, counted.** A fake platform in the otherwise-unused `kXPU` slot over a
counting `vt::Backend` — the machinery `tests/vllm/model_executor/test_resident_weight_host_addressable.cpp`
established for #1299, which is how this tree measures a device allocation without a GPU. Build the
synthetic dense target's weights, borrow them into a real `LoadedModel` through the exported
`BorrowQwen3_5DenseLoadedModel`, build the DFlash2 draft with its own copy of the table, call the
EXPORTED `BindDflashDraftSharedEmbed`, then run `ResidentWeight` over the target's table and over
`draft.weights.EmbedTable()` exactly as the two forwards do, and count.

The assertion is bounded on BOTH sides: total device bytes must be `>= nbytes` (an upload really
happened — a zero would be a mute switch) and `== nbytes` (not `2 * nbytes`), with `allocs == 1`.
`nbytes` is derived in the test from the fixture's own `V` and `H`, not read back from the tensor
the code bound, so it is not a tautology against the code's own number. Before the change the
same case reads `2 * nbytes` and `allocs == 2`.

**The fixture moves with this, and the move is part of the change.**
`dflash2_runner_fixture.h` built the draft's shared table from seed 950 while `MakeDenseWeights`
built the target's from seed 11 — so the fixture's draft gathered from a table its target does
not have, which no production load can produce, under a comment claiming the two are SHARED. Left
alone it would have made the rebind change what every DFlash2 gate in this tree drafts from, for
a reason belonging to the fixture rather than to the engine. Both now use seed 11, so the
existing DFlash2 gates draft exactly what they drafted before this change.

**T2 — the PRODUCTION path reaches it.** The `dflash2_runner_fixture.h` engine — the production
`LoadedEngine` constructor, the production `ResolveSpecConfig`, the production
`CheckDflash2DraftArm` — under a real-fd-2 stderr capture, asserting the bind line is present and
carries the exact byte count `V * H * 2`. This is the reachability leg: it enters at the engine
constructor, not at the type.

## Mutations (named in advance for the fresh reviewer)

**M1 (reachability, the one `.agents/reachability.md` requires).** Delete the
`BindDflashDraftSharedEmbed(...)` call from the `LoadedEngine` constructor's initializer list.
T2 must go red. A green T2 would mean the gate measures the function and not the capability.

**M2.** Make `EmbedTable()` return `embed_tokens` unconditionally. T1 must go red at `2 * nbytes`.

**M3.** Remove the `weights.embed_tokens = OwnedTensor{}` clear from the bind. T1's byte count
does NOT move (one `d_dev` is still allocated), and that is the correct answer — the clear is
defence in depth, not the dedup. T2 does not move either. Recorded so the reviewer does not read
a surviving green as an ungated line: the line's job is to make a FUTURE direct read of
`embed_tokens` fail loudly, and no test in this tree can observe a call site that does not exist.

**M4.** Widen the bind guard to bind on a dtype mismatch. The dedicated dtype-mismatch case in
T1 must go red.

**M5.** Restore `dflash_draft_` ahead of `model_` in the header. Nothing goes red — C++ member
order is not observable from a test that does not read freed memory — so this is recorded as a
REVIEW obligation rather than a gate, with the argument in `## Design` as its evidence.

## Gates

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVLLM_BUILD_TESTS=ON
cmake --build build -j 4 --target test_dflash2_embed_dedup test_dflash2_runner_reach \
  test_dflash2_draft_phase_trace test_resident_weight_host_addressable
./build/tests/test_dflash2_embed_dedup
./build/tests/test_dflash2_runner_reach
ctest --test-dir build --output-on-failure
scripts/agent-preflight.sh --staged
```

## Evidence

**CPU, this box.** Recorded in the pull request body: the red output of both legs before the
change, the green after, and the M1 reachability mutation's red.

**GPU, owed to the operator (this session holds no lease).** One measurement answers the whole
row and nothing here needs a token gate, because the tokens are identical by construction — the
same bytes are gathered either way:

1. Load `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` with `--speculative-config` naming the DFlash2
   draft, on `dgx:gpu0` inside an `rc` lease.
2. Read the `device_upload=` field of the loader's own accounting line
   (`ReportLoadBytes` in `src/vllm/entrypoints/model_loader.cpp`) before and after this change, on the same
   checkpoint and the same flags. The delta must be **2,542,796,800 B (2.368 GiB)**.
3. Confirm the new `DFlash draft embed SHARED` line names that same number.
4. Generate once and confirm the emitted tokens are unchanged against the pre-change run. This is
   a same-binary A/B, so it is a byte comparison of the two token streams rather than a new gate.

## Owed

- **O1** — the `lm_head` device dedup, ~0.715 GB each side. Owned by the parent spec's `## Owed`
  O3 (the O29 Marlin-body convergence), untouched here.
- **O2** — the DSpark lane's shared embed
  (`LoadDsparkDraft`'s shared fallback) takes the same second copy when the DSpark
  checkpoint omits its own table. The bind skips a DSpark draft explicitly, so this is a named
  gap rather than an accident. It needs its own issue and row: the DSpark backbone owns its
  table by value in `Qwen3DSparkWeights`, so the accessor shape here does not carry over
  unchanged.
- **O3** — the GGUF keep-f16 target arm still holds two tables (one F16 for the target, one BF16
  for the draft). Deduping it means teaching the draft's gather to read F16, which is a forward
  change and not a loader change.

## Now

`ACTIVE` — implementation in `row/SPEC-DFLASH2-embed-dedup`.
