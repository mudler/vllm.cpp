# SPEC-DFLASH2 embed device dedup — one device copy of the target's embedding table

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding).
**Issue:** [#1946](https://github.com/mudler/vllm.cpp/issues/1946), and [#1953](https://github.com/mudler/vllm.cpp/issues/1953) — the missing `ResidentWeight` empty
guard, found by this row's fresh review and fixed in the same flow, because the false claim
and the missing check are one defect and one gate covers both.
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
2.543 GB, silently and with every gate green. Clearing it makes that call site fail instead of
duplicating. It also drops the second host borrow into the target's safetensors mapping, which is
the W9 residency this change supersedes for the embed.

**"Fail loudly" was a claim, and it was false; it is now a check.** The first version of this
section said the emptied table would be refused "by name" by `vt::Embedding`. It would not have
been. `vt::Embedding` (`src/vt/ops.cpp`, `void Embedding(`) validates ranks, shapes, dtypes,
contiguity and device and never the data pointer or the byte length, and `ResidentWeight` takes
the shape from the CALLER (`{config.vocab_size, H}`), so an emptied tensor satisfied every
`VT_CHECK` on the way down. What actually happened was worse than a duplicate upload: the host
alias arm handed a kernel a null pointer (SIGSEGV) and a device arm did `d.b.Alloc(0)` and then
viewed `[vocab, H]` over a zero-byte allocation — out-of-bounds device reads, which is exactly
the silently-wrong-tokens outcome the clear is supposed to prevent.

The repair makes the claim true rather than deleting it. `dense_attn::ResidentWeight` now refuses
an empty weight on both arms, naming the seam and the arm
(`resident weight: EMPTY tensor has no host bytes to alias|upload`). Two details matter. The
predicate is `bytes.empty()` and NOT `OwnedTensor::Empty()`, because a weight whose host buffer
was reclaimed after upload (`host_released`) is populated and is served by the `d_dev` branch.
And the staging assert sits INSIDE `if (!w.d_dev)`, so the repeated hot-path read of an
already-resident weight pays nothing; the alias arm's assert is one `size() == 0` compare against
a `GetPlatform` virtual call that was already there. `test_dflash2_embed_dedup.cpp`'s last case
drives the exact hypothetical — rebind, then read the raw field — and gates the refusal.

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

Two new binaries, and they are two binaries rather than two cases because the byte count needs a
fake platform registered in the PROCESS-GLOBAL `kXPU` slot: a binary that also builds a real
`LoadedEngine` would be resolving its device against a registration that exists only for the
measurement. Neither leg can pass before the change, and neither is satisfied by the host sharing
W9 already landed.

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

**The fixture moves with this, and an earlier draft of this section got the reason wrong.**
`dflash2_runner_fixture.h` built the draft's shared table from seed 950 while `MakeDenseWeights`
built the target's from seed 11 — so the fixture's draft gathered from a table its target does
not have, which no production load can produce, under a comment claiming the two are SHARED. Both
now use seed 11.

This section first claimed the seed change is what keeps the DFlash2 gates drafting "exactly what
they drafted before". **That is false, and measuring it is what showed so.** Drafted blocks, read
off the production `VT_SPEC_TRACE` line:

| Tree | drafted blocks |
|---|---|
| pre-change (seed 950, no rebind) | `[19 19 19]` x8 |
| as landed (seed 11, rebind) | `[12 12 12][19 12 12][19 12 19][12 12 12][12 12 12][12 12 12]` |
| seed 950 WITH the rebind | identical to as-landed, byte for byte |

**Which trees these were measured on, because two of them have since moved under it.** A drafted
block is produced by the DFlash2 selector and the draft's context store, and BOTH have been
replaced since this table was first written — so it has been re-read at every tree rather than
carried forward.

| Tree | contains | as-landed row |
|---|---|---|
| `8537aac7b` | neither #1929 nor #1932 | `[12 12 12][19 12 12][19 12 19][12 12 12][12 12 12][12 12 12]` |
| merge of `2c4c2005d` | #1929 (radix top-k, `dc3bbe8cd`) | **byte-identical** |
| merge of `c113886dc` | #1929 **and** #1932 (draft context store) | **byte-identical** |

#1929 replaces the selector's top-k and #1932 rewrites the context store's sizing; neither
reaches this path on a CPU tier. The invariance is corroborated rather than assumed, by two
instruments that are demonstrably NOT blind to it: `test_dflash2_runner_reach` reads 8 cases /
144 assertions and `test_dflash2_draft_phase_trace` 3 / 160 at all three trees, and the repair
commit measured those same counts MOVING with the drafted blocks (8/162 at seed 950 against
8/144 at seed 11). A table that names no tree is one merge away from describing something nobody
runs, which is why this one names all three.

Two things follow, and both are the opposite of what the earlier claim said. The REBIND changes
the drafted tokens, necessarily and correctly: the draft now gathers from the target's table
instead of from a table nothing else in the fixture shares, which is what production always did.
And the SEED CHANGE contributes exactly NOTHING to behaviour — the rebind overwrites the draft's
own table either way, so the third row is byte-identical to the second.

What the seed change is actually for is the RED runs. With seed 950 the pre-change state is "two
DIFFERENT tables", which is not a state any production load can reach; with seed 11 it is "two
copies of the SAME table", which is the defect #1946 describes. So RED-A's `allocs=2
bytes=4096 table=2048` measures the real duplication rather than an artefact of the fixture. No
DFlash2 gate pins exact tokens — they pin block width, id range, and that D9's scalars MOVE the
drafts — so all of them stay green across the token change.

**T2 — the PRODUCTION path reaches it**
(`tests/vllm/v1/spec_decode/test_dflash2_embed_dedup_reach.cpp`). The `dflash2_runner_fixture.h` engine — the production
`LoadedEngine` constructor, the production `ResolveSpecConfig`, the production
`CheckDflash2DraftArm` — under a real-fd-2 stderr capture, asserting the bind line is present and
carries the exact byte count `V * H * 2`. This is the reachability leg: it enters at the engine
constructor, not at the type.

## Mutations (named in advance for the fresh reviewer)

**M1 (reachability, the one `.agents/reachability.md` requires).** Delete the
`BindDflashDraftSharedEmbed(...)` call from the `LoadedEngine` constructor's initializer list.
T2 must go red. A green T2 would mean the gate measures the function and not the capability.
Re-run after the review repair: T2 fails at both of its stderr checks, `rc=1`, and T1 stays
`SUCCESS` — unchanged, as it should be.

**M1 got a second, earlier detector for free**, and it is worth knowing before someone reads a
build failure as a pass. Once `BindSharedEmbed` became `static`, deleting its only call site is
`-Werror=unused-function` at `model_loader.cpp`, so the mutation does not COMPILE. Running M1
therefore needs `[[maybe_unused]]` on the wrapper for the duration of the mutation, and a run that
skips that step reports the STALE binaries as green. Capture the build's exit status, not the
test's.

**M2.** Make `EmbedTable()` return `embed_tokens` unconditionally. T1 must go red at `2 * nbytes`.

**M3.** Remove the `weights.embed_tokens = OwnedTensor{}` clear from the bind. **T1 goes RED**, at
`test_dflash2_embed_dedup.cpp`'s `CHECK(draft->weights.embed_tokens.Empty())` and again at the
`CHECK_THROWS_WITH_AS` in the post-rebind-read case, which requires the field to be empty before
it can be refused. This row previously said the clear was ungated and that no test in this tree
could observe it; both were wrong, and the second was wrong twice over once the refusal became a
check the gate drives. What does NOT move is T1's BYTE count — one `d_dev` is still allocated
either way — so a reader comparing byte totals alone would still see no difference. That is the
distinction the row exists to record.

**M4.** Widen the bind guard to bind on a dtype mismatch. The dedicated dtype-mismatch case in
T1 must go red.

**M5.** Restore `dflash_draft_` ahead of `model_` in the header. Nothing goes red — C++ member
order is not observable from a test that does not read freed memory — so this is recorded as a
REVIEW obligation rather than a gate, with the argument in `## Design` as its evidence. The
argument itself is narrower than it was first written: the OLD order was not a use-after-free,
because `DflashDraft` has no user-declared destructor and its teardown never dereferences
`shared_embed_tokens`. The old order left a WINDOW that any future teardown work would have
fallen into. The new order closes the window rather than fixing a live bug, and the header comment
now says so.

## Gates

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -j 8
./build/tests/test_dflash2_embed_dedup
./build/tests/test_dflash2_embed_dedup_reach
./build/tests/test_dflash2_runner_reach
./build/tests/test_dflash2_draft_phase_trace
ctest --test-dir build --output-on-failure
scripts/agent-preflight.sh --staged
```

## Evidence

**CPU, this box (measured 2026-08-25, gcc, Release, `-Werror`).** The synthetic table is BF16
[64, 16] = 2048 B, so the ratio is what the case reads, not the magnitude.

| State | `test_dflash2_embed_dedup` | `test_dflash2_embed_dedup_reach` |
|---|---|---|
| RED-A: `BindDflashDraftSharedEmbed` returns false at the top — the pre-#1946 behaviour with the API intact | FAILURE, `allocs=2 bytes=4096 table=2048` | FAILURE, no `SHARED` line on the engine's stderr |
| GREEN: as landed | SUCCESS, 6 cases / 29 assertions, `allocs=1 bytes=2048` | SUCCESS, 2 cases / 4 assertions |
| M1 (reachability): the `BindSharedEmbed(...)` member initialiser deleted from the `LoadedEngine` constructor | SUCCESS — it calls the function directly, which is exactly why it cannot be the reachability gate | **FAILURE** — the production path is what this binary measures |
| M2: `EmbedTable()` returns `embed_tokens` unconditionally | FAILURE, `allocs=2` | unaffected |

M2's byte total does NOT move (`bytes=2048`), because the bind has already cleared the draft's
own tensor and an empty table allocates nothing. The allocation COUNT is what catches M2, and
both bounds are in the case for that reason.

`test_dflash2_runner_reach` (8 cases / 144 assertions) is green on the landed tree. Its drafted
tokens DO move — see the measured table under `## Tests`, and note that the rebind rather than
the fixture seed is what moves them — and no case there pins a token value. The full suite is
`100% tests passed, 0 tests failed out of 624` (727 s, 5 skipped for absent checkpoints).

**A second instrument agrees, from two builds rather than from a trace.** The drafted-token table
under `## Tests` was read off `VT_SPEC_TRACE`; a fresh session read the same conclusion off
doctest's assertion counter, which counts the per-drafted-token assertions and so moves when the
drafts move. Release, `-Werror`, CPU, and every cell `Status: SUCCESS`
(`test_dflash2_runner_reach` / `test_dflash2_draft_phase_trace`):

| | fixture draft-embed seed 950 | fixture draft-embed seed 11 |
|---|---|---|
| **parent `18a4e23f4`** (no rebind) | 8 / 162 and 3 / 202 | 8 / 144 and 3 / 160 |
| **head** (rebind) | 8 / 144 and 3 / 160 | 8 / 144 and 3 / 160 |

Row 2 is the no-op: at head both seeds give 8 / 144 and 3 / 160, and the two runs' full `-s`
output is identical once process ids and `[spec-phase]` timings are normalised. `144` is therefore
the AFTER number on both axes, and the implementation commit's "unchanged at 8 cases / 144
assertions" was the head reading quoted as the parent's. **The suite is not blind to that table,
it is simply not golden-pinned:** changing the TARGET seed 11 → 7 reds
`test_dflash2_runner_reach.cpp:225`. Not pinning token VALUES is what let an 18-assertion
behaviour change pass silently.

**The refusal `## Design` claims for the clear, red-before and green-after.** The claim was first
written as a property of `vt::Embedding` and it was false there; it is now a `VT_CHECK` in
`dense_attn::ResidentWeight` and `test_dflash2_embed_dedup.cpp`'s last case drives it.

| State | `test_dflash2_embed_dedup` |
|---|---|
| RED-BEFORE: the case added, no check in `ResidentWeight` | **FAILURE**, 7 cases / 34 assertions, 1 failed — `CHECK_THROWS_WITH_AS( Upload(draft->weights.embed_tokens), "resident weight: EMPTY tensor", std::runtime_error ) did NOT throw at all!` The emptied table went all the way through `ResidentWeight`, and `Upload`'s own `REQUIRE(t.data != nullptr)` passed, because `Alloc(0)` returns a valid one-byte pointer |
| GREEN-AFTER: the two `VT_CHECK`s in place | SUCCESS, 7 cases / 33 assertions (one fewer, because the throw now precedes `Upload`'s `REQUIRE`) |

**Preflight: `scripts/agent-preflight.sh --fail-on-skip` is `All gates green.` — RC=0, 106 gates
`ok`, 0 FAIL, 0 SKIP.** Re-taken after the review repair on a clean `-Werror` Release rebuild
(1733 targets, `rc=0`) with `--staged --fail-on-skip`: `All gates green.`, RC=0, **107** gates
`ok`, 0 FAIL, 0 SKIP, and `ctest` `100% tests passed, 0 tests failed out of 624` in 106 s at
loadavg 24. That `ctest` run is what says the new `ResidentWeight` guard breaks no existing
caller: nothing in the tree reaches the seam with an empty tensor.

`test_cpu_x86_llamacpp_floor` passed in both repair runs, which is the fourth observation on the
#618 lane below and agrees with it.

Getting there is worth recording, because the first run was RED and the reason was the box.
`test_cpu_x86_llamacpp_floor` failed while a second agent's `test_ltx2_video` held 524% CPU: its
contended-leg case is [#618](https://github.com/mudler/vllm.cpp/issues/618), which exits
`NO_QUIET_WINDOW` (4) instead of `GIVING_UP` (2) when no quiet window exists. That was not waved
through on the issue number. Three observations, in the order they were taken:

1. The failing SET MOVED WITH LOAD — one case at loadavg 64.78, a DIFFERENT one
   (`test_the_quiet_gate_does_not_see_the_harnesss_own_process_tree`) at 41.72, both with the
   same `busy=124-130%` signature. A code defect does not choose its case by loadavg.
2. The CONTROL: the same gate, run from a detached worktree at the base SHA `f44077b51` carrying
   none of this change, failed the same way at the same load
   (`NO_QUIET_WINDOW after 30s, busy=104% load=47.35`).
3. THE GATE THEN PASSED. Once the box drained to loadavg 4.2, all 10 of its cases passed in
   **3.7 s**, against 233 s and 310 s of failing runs under load — and the full preflight went
   green with no skips.

The record keeps all three rather than only the last, because a green obtained by waiting is
worth less to the next reader than the reason the red was not the diff.

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

### The three-way tree, and two contention flakes that are not this change

`origin/main` at `c113886dc` carries BOTH other DFlash2 changes in flight — #1929's radix top-k
and #1932's draft context store sizing — and this row adds a `VT_CHECK` to
`dense_attn::ResidentWeight`, a seam 368 call sites reach. That combination existed on no tree
until this merge, so it was gated rather than assumed.

**The `ResidentWeight` emptiness check fires NOWHERE in either newly-merged change.** Statically,
neither can reach it: #1929 touches `include/vt/radix_topk.h`, `include/vt/ops.h` and
`src/vt/cuda/cuda_sample.cu`, and `vt::` deals in `vt::Tensor` VIEWS, strictly below the
`OwnedTensor` seam — a grep of its touched files for `ResidentWeight|OwnedTensor` returns
nothing. #1932 IS in the layer that consumes `OwnedTensor`s, which is why it was checked rather
than waved through, but it adds no `ResidentWeight`, `OwnedTensor`, `EmbedTable` or
`embed_tokens` line at all: it sizes a device KV store, not a weight. Empirically, `resident
weight: EMPTY tensor` occurs **zero** times across the full 626-test suite.

**Two tests failed across the full runs, and neither is this change.** They are DISJOINT and each
passes on its own, which is the signature of contention rather than code:

| run | `test_ltx2_video` | `test_engine_core_proc` |
|---|---|---|
| `-j 4` #1 | FAILED | passed |
| `-j 4` #2 | passed | FAILED |
| alone | passed (106 cases / 4792 assertions) | passed 3 of 3 |

Neither is code this row, #1929 or #1932 touches — `git diff --name-only` across all three
returns no LTX-2.5 file. Both are RECORDED flakes rather than new findings.
`test_engine_core_proc`'s abort race is written down in [`environment.md`](../environment.md) as
"a timing flake under heavy parallel ctest ... passes on rerun", measured about 1 in 5 under
load. The LTX-2.5 failure is the ratio `covered >= c.min_coverage * leaf_seconds` with a margin
of **0.000138 s** (`CHECK( 0.000336164 >= 0.000474467 )`) — the quantity
[#1494](https://github.com/mudler/vllm.cpp/issues/1494) measured deciding by loadavg on one
unchanged binary. #1494 CLOSED on the `denoise` leaf and this is `artifacts.frames`, whose short
records #1559's span-slack bound deliberately does not hold, so it was filed rather than assumed
covered: [#1957](https://github.com/mudler/vllm.cpp/issues/1957), owned by
`LTX25-DEVICE-RESIDENCY` and not fixed in flow for the reason #1439, #1494 and #1576 each
reached — a seconds bound beside the ratio changes a gate's semantics. The box carried other
agents at loadavg 18-35 throughout, and `-j 2` on the same binary is
`100% tests passed, 0 tests failed out of 626`.


## Owed

- **O1** — the `lm_head` device dedup, ~0.715 GB each side. Owned by the parent spec's `## Owed`
  O3 (the O29 Marlin-body convergence), untouched here.
- **O2** — [#1951](https://github.com/mudler/vllm.cpp/issues/1951): the DSpark lane's shared
  embed (`LoadDsparkDraft`'s shared fallback) takes the same second copy when the DSpark
  checkpoint omits its own table. The bind skips a DSpark draft explicitly and
  `test_dflash2_embed_dedup` pins the skip, so this is a named gap rather than an accident. It
  is not an in-flow fix because the shape does not carry over: the DSpark backbone owns its
  table BY VALUE in `Qwen3DSparkWeights`, so it needs its own borrowed pointer and its own
  accessor over its own gather sites. Both published DSpark drafts ship a table, so nothing on
  the default published path duplicates today.
- **O3** — the GGUF keep-f16 target arm still holds two tables (one F16 for the target, one BF16
  for the draft). Deduping it means teaching the draft's gather to read F16, which is a forward
  change and not a loader change.
- **O4** — [#1953](https://github.com/mudler/vllm.cpp/issues/1953)'s residue:
  `dense_attn::ResidentWeightF32`, in the same header, has the same shape and no empty guard. It
  upcasts `w.bytes` into a `std::vector<float>` first, so an empty weight there yields a
  zero-length vector rather than a null alias — a different failure, not the same one — and it is
  not on any path this row touches. Fixing it needs its own red-first case over the f32 arm.

## Now

`ACTIVE` — implementation on `row/SPEC-DFLASH2-embed-dedup` ([#1952](https://github.com/mudler/vllm.cpp/pull/1952)).
Every CPU gate is green: `ctest` 624/624, preflight `All gates green.` with no skips, and both
red-before legs plus the reachability mutation are recorded above. The GPU measurement in
`## Evidence` is owed to the operator; this session held no lease.
