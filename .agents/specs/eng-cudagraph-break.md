# ENG-CUDAGRAPH-BREAK — a shared `vt` capture seam that accepts break points

**Row:** `ENG-CUDAGRAPH-BREAK` (engine-matrix, engine core and scheduling).
**Issue:** [#1163](https://github.com/mudler/vllm.cpp/issues/1163), with
[#1020](https://github.com/mudler/vllm.cpp/issues/1020) as one reachable instance.
**Parent analysis:** [sglang-breakable-cuda-graph.md](sglang-breakable-cuda-graph.md) `## Owed`.
**Kind:** scoping spike plus structured spec. No production code lands with this spec.

**This is a coverage row, not a throughput row.** The spike claims no speed and
measures none. `## Gates` below states what would have to be named and measured
before any speed claim is admissible, and `## Risks/decisions` D5 records why no
such path is named today.

## Scope

One shared capture seam under `vt` that captures a forward as a SEQUENCE of graph
segments split at break points, so a forward containing a host-dependent operation
is still graphed except at that operation, instead of running eager for the whole
step. The seam replaces the per-model capture drivers listed in `## Our baseline`.

In scope:

- The break-point primitive and the capture scope that collects segments.
- The segment container, its replay order, and its ownership of instantiated graphs.
- The lifetime contract for every buffer a segment or a break function reads.
- A staged migration of the existing drivers onto the seam, one driver per stage.
- The predicate that decides whether a step is graph-eligible, moved off
  `pure_decode` and onto "eligible except at the break points".

Out of scope, and each names its owner:

- Graph-executable dedup through `cudaGraphExecUpdate`. That is
  `ENG-CUDAGRAPH-DEDUP`, [#1162](https://github.com/mudler/vllm.cpp/issues/1162),
  owned by a parallel agent. See `## Risks/decisions` D4 for the interaction.
- The LTX-2.5 denoise loop. That is `ENG-CUDAGRAPH-DIFFUSION`,
  [#1164](https://github.com/mudler/vllm.cpp/issues/1164), blocked on its own defects.
- Any change to a model's numerics. Every migrated model must stay bit-exact
  against its own eager forward.

## Upstream chain

**The boundary comes from vLLM. The construction comes from SGLang.** These are two
different questions and this spec keeps them apart.

**Primary oracle, vLLM at pin `5559679229` (`.agents/upstream-sync.md`).** vLLM's
`CUDAGraphMode` enum defines `PIECEWISE = 1` and the composite
`FULL_AND_PIECEWISE = (FULL, PIECEWISE)` at
`/home/mudler/_git/vllm/vllm/config/compilation.py:60-63`. The docstring at
`:608-635` states that `FULL_AND_PIECEWISE` is the v1 default, that it captures a
full graph for decode batches and a piecewise graph for prefill and mixed batches,
and that piecewise mode keeps "the cudagraph incompatible ops (i.e. some attention
ops) outside the cudagraph". The split points themselves are `splitting_ops`,
declared at `:517`, and defaulted at `:1145` to `list(self._attention_ops)`. That
class-level list is at `:764-772` and names, among others,
`vllm::unified_attention_with_output`, `vllm::unified_mla_attention_with_output`,
`vllm::mamba_mixer2`, `vllm::mamba_mixer`, `vllm::short_conv`,
`vllm::linear_attention` and `vllm::qwen_gdn_attention_core`.

So the boundary vLLM chose is: **the attention family is the break point**. Dense
attention, multi-head latent attention (MLA), the Mamba mixers, the short
convolution and the gated delta network (GDN) core. This spec adopts that boundary
and does not invent one.

**Where vLLM's construction does not transfer.** vLLM obtains the split from
`torch.compile`: `set_splitting_ops_for_v1` at `:1123-1145` returns early unless
`self.mode == CompilationMode.VLLM_COMPILE`, and the split is applied by Dynamo at
the FX level or by an Inductor graph partition. We have no compiler, no Dynamo and
no FX graph. The boundary transfers; the mechanism cannot.

**Secondary oracle, SGLang at pin `f63458b5be` (`.agents/oracles/sglang.md`).**
SGLang reaches the same coverage with no compiler, which is the construction this
spec ports. Verified in the pinned tree at
`/home/mudler/_git/sglang` (`git rev-parse HEAD` =
`f63458b5beaceabbd9d749b9fc956370e1b649e6`):

| Mechanism | Anchor in `python/sglang/srt/` |
|---|---|
| Break-point wrapper | `.../breakable_cuda_graph/breakable_cuda_graph.py:204-241` (`eager_on_graph`) |
| Active-capture context | `:60` (`_current_capture_var`, a `ContextVar`) |
| Segment container | `:246-253` (`BreakableCUDAGraph`, `_segments` plus `_break_fns`) |
| Interleaved replay | `:255-263` |
| Capture scope enter and exit | `:307-331` (`__enter__`, `__exit__`) |
| Segment open and close | `:335-350`, `:352-365` |
| Bare break marker | `:370-375` (`break_graph`, an empty body under the decorator) |
| Shared mempool and weak refs | `:14-23` doc, `:156-169` (`_weak_ref_if_tensor`) |
| No-compiler backend | `.../runner_backend/breakable_cuda_graph_backend.py:14-17` |

The break points SGLang registers, all through the same one-line wrapper:

| Site | Anchor |
|---|---|
| Dense attention | `layers/radix_attention.py:256` |
| MLA | `models/deepseek_common/attention_forward_methods/forward_mla.py:1092` |
| Linear attention | `layers/radix_linear_attention.py:159` |
| Nemotron Mamba2 | `models/nemotron_h.py:1240` |
| DeepSeek sparse-attention indexer | `layers/attention/dsa/dsa_indexer.py:2414` |
| DeepSeek V4 attention | `models/deepseek_v4.py:284` |

That set is the same family vLLM lists in `_attention_ops`. The two oracles agree
on the boundary, which is the strongest evidence available that the boundary is
right and not an artefact of either implementation.

## Our baseline

### The seam that exists

`include/vt/backend.h:208-222` already declares the whole capture vocabulary:
`SupportsGraphCapture`, `BeginCapture`, `EndCapture`, `Replay`, and the multi-graph
handle trio `EndCaptureGraph`, `ReplayGraph`, `DestroyGraph`. The base definitions
in `src/vt/backend.cpp:29-34` throw for a backend that does not implement capture,
except `DestroyGraph`, which is a no-op.

Three backends implement it: CUDA at `src/vt/cuda/cuda_backend.cu:198,204,225,233,288`,
ROCm at `src/vt/rocm/rocm_backend.hip:248`, and Tenstorrent at
`src/vt/tenstorrent/tenstorrent_backend.cpp:75-81`, which maps the calls onto ttnn
mesh-trace capture. Vulkan and Metal return false and document why, at
`src/vt/vulkan/vulkan_backend.cpp:16` and `src/vt/metal/metal_backend.mm:13`.

**`EndCaptureGraph` stores nothing in the backend** (`cuda_backend.cu:225-232`). It
ends the stream capture, instantiates the graph, and returns the executable as an
opaque handle the caller owns. That property is what makes a segmented capture
expressible with **no new backend virtual**: a segment is one `BeginCapture` and
`EndCaptureGraph` pair, and N break points produce N+1 such pairs on the same queue.
The spike confirmed this shape end to end; see `## Work breakdown` W0.

### Defect 1: capture is all or nothing

`src/vllm/v1/worker/gpu/runner.cpp:1341` computes the only predicate that admits a
step to a graph:

```
const bool pure_decode = attn_meta.num_actual_tokens == num_reqs &&
                         gdn_meta.num_prefill_tokens == 0;
```

It travels to the model as `ModelForwardInput::pure_decode`
(`include/vllm/model_executor/models/model_registry.h:281`, set at
`src/vllm/v1/worker/gpu/runner.cpp:1378`), and each model's registered forward
tests it before reaching its driver, for example
`src/vllm/model_executor/models/qwen3.cpp:961`. A step that fails the predicate
runs eager in full. There is no way to express "capture this forward except at
these three calls", so coverage is a cliff and not a slope.

[#1020](https://github.com/mudler/vllm.cpp/issues/1020) is one reachable instance:
a speculative verify step whose actual draft depth differs from the configured
`num_speculative_tokens` (`model_registry.h:288`) is admitted by no graph and falls
out to eager with no log and no counter.

### Defect 2: the drivers, enumerated

Eight production drivers exist, and a ninth already landed. Six are batched decode
drivers with a padded-bucket ring; three are single-shape drivers. Every one of
them re-derives the same machinery.

| Driver | Declaration | Implementation | Shape |
|---|---|---|---|
| `Qwen3_5DecodeGraph` | `include/vllm/model_executor/models/qwen3_5.h:275` | `src/vllm/model_executor/models/qwen3_5.cpp:9540,9563,9712,9893,9918,9922` | batched, padded ring |
| `Qwen3_5DenseDecodeGraph` | `include/vllm/model_executor/models/qwen3_5_dense.h:391` | `src/vllm/model_executor/models/qwen3_5.cpp:9974,9986,10321,10340,10348` | batched, padded ring |
| `Qwen3MoeDecodeGraph` | `include/vllm/model_executor/models/qwen3_moe.h:117` | `src/vllm/model_executor/models/qwen3_moe.cpp:385,404,469,512,515,524` | batched, padded ring |
| `Qwen3DenseDecodeGraph` | `include/vllm/model_executor/models/qwen3.h:243` | `src/vllm/model_executor/models/qwen3.cpp:583,613,682,866,888,897` | batched, padded ring |
| `DeepseekV2DecodeGraph` | `include/vllm/model_executor/models/deepseek_v2.h:324` | `src/vllm/model_executor/models/deepseek_v2.cpp:887,902,963,980` | batched, padded ring |
| `VoxtralDecodeGraph` | `include/vllm/model_executor/models/voxtral.h:126` | `src/vllm/model_executor/models/voxtral.cpp:448,469,535,582,585,594` | batched, padded ring |
| DeepSeek V4 `V4Graph` | none, file-local | `src/vllm/model_executor/models/deepseek_v4.cpp:1673,1719,1898,1900,1904` | single shape, `gstate` 0/1/2 |
| Laguna decode graph | none, file-local | `src/vllm/model_executor/models/laguna.cpp:2373,2689,2691,2692,2695` | single shape |
| DFlash draft graph (the ninth) | none, file-local | `src/vllm/model_executor/models/qwen3_dflash.cpp:771,870,1038,1091,1095,1106` | single shape, `g_state` 0/1/2 |

**What every driver re-derives.** Each item below was read in at least two drivers
and the anchors give one instance each.

1. The environment kill switch. Five drivers read `VLLM_CPP_CUDAGRAPH` themselves:
   `qwen3_5.cpp:9540`, `qwen3_5.cpp:9974`, `qwen3_moe.cpp:385`, `qwen3.cpp:583`,
   `deepseek_v2.cpp:887`, `voxtral.cpp:448`. The three single-shape drivers each
   invented their own instead: `VT_V4_DECODE_GRAPH` at `deepseek_v4.cpp:1680`,
   `VT_DFLASH_GRAPH` at `qwen3_dflash.cpp:870`, and the Laguna cluster at
   `laguna.cpp:104,116,140`. There is no one switch that turns capture off.
2. The cold, warm, captured state machine. `qwen3_5.cpp:9563` carries `warm` and
   `captured` flags on the slot; `deepseek_v4.cpp:1719` spells the same thing as
   `int gstate = 0`; `qwen3_dflash.cpp:771` spells it as `int g_state = 0`. Three
   spellings of one three-state machine.
3. The pad-to-bucket selection. Five call sites of `PadToCaptureSize`:
   `qwen3_5.cpp:9712`, `qwen3_moe.cpp:469`, `qwen3.cpp:682`, `deepseek_v2.cpp:963`,
   `voxtral.cpp:535`. The function itself is already shared, at
   `include/vllm/model_executor/models/decode_graph_sizes.h:47-56`, which is the
   proof that extraction works and is the pattern this row extends.
4. The per-slot persistent buffer set. `qwen3_5.cpp:9563-9600` holds `token_ids`,
   `positions`, the two metadata structs, `hidden`, `logits`, `aux`, `graph`,
   `fa_cols`, `captured`, `warm`, `replays`, `reuse_event`, `dev` and `pin`. Every
   other batched driver holds a subset of the same fields under the same name
   `SizeSlot`.
5. The diagnostics pair `captured()` and `replay_count()`, at `qwen3_5.cpp:9680`,
   `qwen3_moe.cpp:454`, `qwen3.cpp:667`, `deepseek_v2.cpp:948`, `voxtral.cpp:519`.
6. The capture-failure recovery. `qwen3_5.cpp:9913`, `qwen3_5.cpp:10335` and
   `qwen3_dflash.cpp:1106` each carry their own copy of the same
   `try { g = EndCaptureGraph(); } catch (...) {}` drain, because a skipped
   `EndCaptureGraph` poisons the stream permanently.
7. The invalidate-and-recapture rule on a shape change. `qwen3_5.cpp:9816-9826`
   destroys the graph when the block-table column count moves;
   `qwen3_dflash.cpp:1038-1047` does the same when the block width moves.

**The divergence that proves the cost.** The persistent DEVICE input path, called
"Option A" in the tree, exists in exactly one driver. `StepDevInputs`
(`qwen3_5.cpp:3894`) is referenced 33 times in `qwen3_5.cpp` and **zero** times in
`qwen3_moe.cpp`, `qwen3.cpp`, `deepseek_v2.cpp` and `voxtral.cpp` (`grep -c`, all
four returned `0`). The other four replay against host vectors. One of them,
`qwen3.cpp:961-986`, therefore has to DECLINE the graph outright whenever the
asynchronous device-token mirror is live, and its comment records the measured
failure that forced it: "depth-2, graph ON FAIL, slots 1-3 degenerate". The same
comment names the real fix as reading the identifiers at replay time from a stable
device buffer, which is precisely what `StepDevInputs` already does in the sibling
driver. One capability, written once, unavailable to four models. That is the cost
of eight drivers stated as a defect rather than as an aesthetic complaint.

**What is genuinely model-specific**, and must stay in the model:

- Which operations are break points for this architecture. A GDN hybrid breaks in
  different places from a dense attention stack.
- The shape key. `qwen3_5.cpp:9816` keys on the block-table column count as well
  as the batch size; `qwen3_dflash.cpp:1036` keys on the draft block width.
- What the forward writes into the persistent output buffers: logits alone, or
  logits plus the auxiliary hidden taps (`qwen3_5.cpp:9570-9577`).
- The pre-warm forward itself, because only the model knows its own allocation set.
- Any model-specific fusion that must run inside a segment.

### The lifetime contract that already exists

`src/vt/cuda/graph_safe_scratch.h:1-32` records the rule this seam must enforce
rather than re-learn: a captured graph BAKES the device pointer a grow-on-demand
scratch helper returned at capture time, so freeing that block on growth makes the
next replay read freed memory. The fix in the tree is retire-on-grow. The full
root-cause is `.agents/specs/decode-graph-scratch-uaf-2026-07-18.md`.

The `DevicePool` (`include/vllm/model_executor/models/device_pool.h:1-38,71`)
never returns a block to the driver. That is a materially better position than
SGLang's, and `## Risks/decisions` D1 states why it does not make the hazard go
away.

## Port map

### The primitive

Three types, all under `vt`, all backend-agnostic, none of them a new virtual on
`vt::Backend`.

1. **`vt::BreakableGraph`** — the container. Holds `std::vector<void*> segments`
   (each an opaque handle from `Backend::EndCaptureGraph`) and
   `std::vector<std::function<void()>> break_fns`. Invariant, asserted:
   `segments.size() == break_fns.size() + 1`. `Replay(Queue&)` walks
   `ReplayGraph(segments[i])` then `break_fns[i]()` for each `i`, mirroring
   `breakable_cuda_graph.py:255-263`. The destructor calls `DestroyGraph` on every
   segment. This is the direct analogue of `BreakableCUDAGraph` at `:246`.

2. **`vt::GraphCaptureScope`** — a resource-acquisition-is-initialization (RAII)
   scope. Its constructor sets one `thread_local GraphCaptureScope*` and calls
   `Backend::BeginCapture`; its destructor calls `Backend::EndCaptureGraph`, pushes
   the final segment, and clears the pointer. This is `BreakableCUDAGraphCapture`
   at `:277-331`, with the destructor doing what `__exit__` does at `:320-331`. The
   thread-local replaces SGLang's `ContextVar` at `:60`. It must be
   `thread_local`, not a global, for the same reason SGLang chose a `ContextVar`
   and CUDA chose `cudaStreamCaptureModeThreadLocal` (`cuda_backend.cu:204-206`):
   capture is a property of one thread's stream, and a process-wide flag would
   make an unrelated thread's forward observe a capture it is not part of.

3. **`vt::GraphBreak(fn)`** — the break point. Outside a capture scope it calls
   `fn` and returns, so a non-capturing forward is byte-identical to today. Inside
   a scope it closes the current segment, runs `fn` once eagerly so the outputs
   hold real data, appends `fn` to `break_fns`, and opens a new segment. This is
   `eager_on_graph` at `:204-241`, minus the decorator syntax we cannot write.

**Why a thread-local and not a parameter.** The break site sits inside an attention
call, several frames below the forward entry point. Threading a capture context
through every intermediate signature would touch every model. SGLang made the same
call for the same reason. The spike proved the thread-local carries the context
across three frames with no intermediate signature change; see `## Work breakdown` W0.

**A bare marker.** `vt::GraphBreak()` with no argument splits the segment and runs
nothing, mirroring `break_graph` at `breakable_cuda_graph.py:370-375`. It is the
form a model uses when the host-dependent work already ran outside the forward.

### How the boundary is registered

vLLM registers `splitting_ops` by operation NAME because it has an FX graph to
match names against. We have neither. Our equivalent is a call at the break site
itself, exactly as SGLang's is: one line at the attention entry point, mirroring
`radix_attention.py:256` and `radix_linear_attention.py:159`. The site is the
registration.

The break-point set is the vLLM `_attention_ops` family
(`compilation.py:764-772`), reached through our own dispatch: `dense_attn::AttnBlock`
for dense attention and MLA, and the GDN and Mamba entry points for the recurrent
families. Naming the exact call sites is W1 work, not spike work, because the set
must be established per migrated model against that model's own break inventory.

### How segments and intermediates stay valid

The question SGLang answers with a shared memory pool and weak references
(`breakable_cuda_graph.py:14-23,156-169`) has a different answer here, because our
allocator is different.

- **Device intermediates.** A `DBuf` allocated during segment `i` is returned to
  the `DevicePool` free list when its scope ends. The pool never returns a block to
  the driver (`device_pool.h:16`), so the POINTER segment `i` baked stays mapped
  for the life of the process. SGLang needs `use_count` pinning because torch's
  caching allocator can release a segment to the driver; we do not. What we DO need
  is protection against REUSE, which `## Risks/decisions` D1 covers.
- **Break-function inputs.** The closure must capture device pointers or `Tensor`
  views that are stable across replays, never a reference to a host temporary that
  dies at the end of the capturing call. This is the same contract the existing
  drivers already meet for their persistent slot buffers
  (`qwen3_5.cpp:9563-9600`), lifted from a per-driver convention to a seam rule.
- **Host sources of a captured upload.** Capture bakes the HOST SOURCE ADDRESS of
  an upload as well as the device destination. `## Risks/decisions` D2 is the rule
  the seam must enforce.

### What is retired, and what replaces it

The seam absorbs items 1, 2, 5, 6 and 7 of the re-derivation list, plus the
segment bookkeeping. `decode_graph_sizes.h` already owns item 3 and is unchanged.
Item 4 becomes a model-supplied persistent buffer set the seam holds but does not
interpret. Each migrated driver keeps only its shape key, its pre-warm forward,
its break-point set and its output buffers.

## Tests to port

vLLM's own tests do not transfer. Its piecewise split is verified through
`torch.compile` configuration tests, and we have no compiler to configure. SGLang's
BCG carries no unit test at the pinned revision that exercises the segment
container in isolation. So the primary tests here are written against OUR seam, and
the anchors above are the behavioral specification they encode.

The existing test surface the migration must keep green:

| Existing test | Path | Why it binds |
|---|---|---|
| Capture-size set and pad selector | `tests/vllm/models/test_decode_graph_sizes.cpp` | the shared piece the seam must not change |
| CUDA capture capability | `tests/vt/test_cuda_backend.cpp:108` | the seam's backend contract |
| ROCm capture capability | `tests/vt/test_rocm_backend.cpp` | proves the seam is not CUDA-only |
| Tenstorrent trace capture | `tests/vt/test_tenstorrent_backend.cpp:1370,1376` | a third capture backend with different semantics |
| Retire-on-grow bookkeeping | `tests/vt/test_graph_safe_scratch.cpp` | the lifetime rule the seam enforces |
| 35B paged engine gate | `tests/parity/test_qwen36_paged_engine.cpp:140` | the model gate the first migration must hold |

New tests this row owes, each written red first:

1. **Segment arithmetic.** N break points in one capture scope produce exactly
   N+1 segments and N break functions. Red first by asserting N segments.
2. **Replay order.** Replay emits segment 0, break 0, segment 1, break 1, and so
   on, in that exact order, and the order is identical on the second and third
   replay. This is the multi-replay requirement stated in `## Gates`.
3. **Pass-through outside capture.** With no scope active, `vt::GraphBreak(fn)`
   runs `fn` and makes zero backend calls. Red first by asserting a backend call.
4. **Capture-failure drain.** A break function that throws leaves the stream
   un-poisoned and the partially built `BreakableGraph` destroyed with every
   already-instantiated segment released. This ports the recovery that
   `qwen3_5.cpp:9913` and `qwen3_dflash.cpp:1106` each hand-rolled.
5. **Non-capturing backend.** On a backend where `SupportsGraphCapture()` is
   false, the scope is inert and the forward runs eager. Vulkan
   (`vulkan_backend.cpp:16`) and Metal (`metal_backend.mm:13`) are the live cases.
6. **Bit-exactness per migrated model**, over more than one replay. See `## Gates`.

Tests 1 through 5 need no graphics processing unit (GPU). They run against a test
backend that records the call sequence. Test 6 needs a GPU and an `rc` lease.

## Gates

**Correctness first. No performance result is accepted before the token-exact gate
for that model is established.**

**G1, bit-exactness against eager, per migrated model, over MORE than one replay.**
For each migrated model, run the same inputs through the segmented capture path and
through the eager forward with capture disabled, and require the logits to be
bit-identical. The comparison must cover at least three consecutive replays of the
same captured graph, not one. A single replay cannot distinguish a correct segment
boundary from one that happens to read a buffer that has not yet been overwritten;
the class of defect this row's own history contains
(`.agents/specs/decode-graph-scratch-uaf-2026-07-18.md`) only appears on a LATER
replay. The existing 35B gate `tests/parity/test_qwen36_paged_engine.cpp:140` is
the model-level anchor for the first migration.

**G2, reachability mutation.** Per `.agents/reachability.md` and the AGENTS.md
"Nothing lands dead" rule. In a scratch copy, DELETE the production call site that
routes a step into the seam — the model's registered forward entry, reached from
`ModelRegistry::Forward`, not an example and not a test — and rerun the focused
gate. The gate MUST go red. A gate that stays green without the call site measures
a class and not a capability. The reviewer restores the tree byte for byte
afterwards. Report the diff stat and the compiler status alongside the mutation
result, because a mutation that never applied and a mutation that failed to build
both read as a passing test.

**G3, segment-count observability.** A log line or counter that reports segments
captured, break functions registered, and replays run, so the ratio is observable
in a run rather than inferred. Each driver today has a private
`replay_count()` (`qwen3_5.cpp:9680` and four siblings) and no segment count exists
at all. Without G3 there is no way to tell a two-segment capture from a fully eager
step, and "the graph ran" is exactly the claim a broken instrument fabricates.

**G4, the eager arm is byte-identical.** With capture disabled, every migrated
model produces output bit-identical to its pre-migration eager output. This is what
makes each migration stage reversible.

**G5, the non-CUDA arm.** The ROCm capture path
(`rocm_backend.hip:248`) runs the seam's unit tests. Tenstorrent's trace capture
(`tenstorrent_backend.cpp:75-81`) is exercised at least to the level its existing
test reaches (`tests/vt/test_tenstorrent_backend.cpp:1370`). A design that only
works on CUDA does not land.

**Speed: what would have to be true.** No throughput gate is declared, because no
qualifying path is named. A speed claim from this row is admissible only when it
first names a path that is BOTH currently eager AND currently host-bound, and
states how the host-bound part is measured. Our prefill is neither: GB10 measured
prefill idle between launches at 3.8% with GPU-busy above 96%, and the 27B prefill
gap at 92.5% non-GEMM glue GPU work
([sglang-breakable-cuda-graph.md](sglang-breakable-cuda-graph.md) §3). Decode is
already captured and already banked its launch-overhead win. If a migration
produces a measurement, it is recorded as a coverage result plus a neutral
same-binary A/B, not as a win.

## Dependencies

| Dependency | State | Effect on this row |
|---|---|---|
| `vt::Backend` capture vocabulary | landed, `include/vt/backend.h:208-222` | the seam needs no new virtual |
| `DecodeGraphSizes` and `PadToCaptureSize` | landed, `decode_graph_sizes.h:47-56` | reused unchanged |
| Retire-on-grow scratch | landed, `src/vt/cuda/graph_safe_scratch.h` | the lifetime rule the seam enforces |
| `DevicePool` never-free discipline | landed, `device_pool.h:16` | replaces SGLang's mempool pinning |
| `ENG-CUDAGRAPH-DEDUP` (#1162) | in flight, parallel agent | touches the same handles; see D4 |
| `SPEC-MTP-K-GT-1` (#1020) | open | one instance this row's coverage closes |
| `ENG-CUDAGRAPH-DIFFUSION` (#1164) | blocked | explicitly excluded |
| An `rc` GPU lease | required from W1 onward | W0 needed none; every model gate does |

## Work breakdown

Each stage lands on its own, is reachable from a production entry point at its own
merge commit, and is revertible without touching the next stage. Each stage owes an
issue before it starts.

**W0, the scoping spike. DONE in this change.** No production code. Its result is
recorded here and in `## Risks/decisions`. What it established:

- The existing backend vocabulary expresses a segmented capture with no new
  virtual. Verified by reading `cuda_backend.cu:225-232`: `EndCaptureGraph` stores
  nothing, so `BeginCapture` and `EndCaptureGraph` compose into N+1 pairs.
- A break point is expressible in C++ with no compiler and no decorator, through
  one `thread_local` capture pointer plus a free function. Proven with a
  throwaway proof-of-concept, not committed: a two-layer forward with the break
  site three frames below the entry produced 3 segments, 2 break functions, a
  capture call log of `Begin End` three times, an identical replay order on two
  consecutive replays, and a pure pass-through with zero backend calls when no
  scope was active. Compiled and ran at exit status 0.
- The inventory in `## Our baseline` is the real one: eight drivers plus a ninth
  already written, and the persistent-device-input capability present in exactly
  one of them.

**W0 did NOT establish** that CUDA permits `cudaStreamEndCapture` followed by
`cudaStreamBeginCapture` on the same stream mid-forward with eager work between
them, on our stream configuration. This box has no NVIDIA GPU, and the fleet is
leased through `rc`. SGLang's `_end_current_segment` and `_begin_new_segment`
(`breakable_cuda_graph.py:335-365`) do exactly this on a production path at the
pinned revision, which is strong evidence but is not our measurement. **This is
W1's exit criterion and it needs a GPU lease.** It is named rather than skipped.

**W1, the seam plus its unit gate, and one break point on one model.** Land
`vt::BreakableGraph`, `vt::GraphCaptureScope` and `vt::GraphBreak` with tests 1
through 5 of `## Tests to port`. Register exactly one break point, on one model,
reached from `ModelRegistry::Forward`. Confirm the CUDA re-begin behavior on a
leased GPU first; if it does not hold, W1 stops and reports rather than working
around it. G2's mutation applies from this stage on, because W1 is the first stage
that can be dead.

**W2, migrate `Qwen3DenseDecodeGraph` first.** It goes first for three reasons,
in order. It is the SMALLEST batched driver by machinery, holding no
`StepDevInputs` and no auxiliary tap buffer. It carries the DECLINE mitigation at
`qwen3.cpp:961-986`, so the migration has a documented, measured failure mode to
gate against rather than a hypothetical one. Its gate models, Qwen3-0.6B and
Qwen3-4B, are the cheapest to run of any driver's. It is deliberately NOT the 35B
MoE driver: `Qwen3_5DecodeGraph` is the richest and the most load-bearing, and
migrating it first would put the hardest correctness surface behind the least
seam experience.

**W3, migrate the three remaining plain batched drivers.** `Qwen3MoeDecodeGraph`,
`DeepseekV2DecodeGraph` and `VoxtralDecodeGraph`. They share W2's shape, so each
is a repeat with its own G1 and G2. Landable one at a time.

**W4, migrate the two Qwen3.5 drivers.** `Qwen3_5DecodeGraph` and
`Qwen3_5DenseDecodeGraph`. These carry `StepDevInputs`, the auxiliary taps and the
speculative-decode predicate. W4 is where the persistent device input path becomes
a seam capability rather than one driver's private code, which is what makes it
available to the four drivers that lack it.

**W5, migrate the three single-shape drivers.** DeepSeek V4, Laguna and DFlash.
`laguna.cpp:2116-2119` already carries the note that its capture class is waiting
for exactly this seam, and it names DeepSeek V4's driver as the sibling that moves
with it.

**W6, close the coverage gap the row exists for.** Only after W1 through W5 does
the predicate at `runner.cpp:1341` move from `pure_decode` to "eligible except at
the break points". This is LAST, not first: replacing the predicate before every
driver can survive a segmented capture would admit steps that no driver can serve.
[#1020](https://github.com/mudler/vllm.cpp/issues/1020) closes here or is
explicitly re-scoped here.

## Risks/decisions

**D1, the capture lifetime hazard, and why our position is different but not safe.**
A captured graph bakes the device pointer that an allocation returned at capture
time (`src/vt/cuda/graph_safe_scratch.h:9-20`). The tree already carries one
production incident from this: the 35B online-serving illegal memory access, where
a later, larger forward grew a scratch buffer and freed the block a captured graph
still referenced, root-caused in
`.agents/specs/decode-graph-scratch-uaf-2026-07-18.md`. Our `DevicePool` never
returns a block to the driver (`device_pool.h:16`), so a baked pointer stays
mapped. **That removes the free hazard and not the reuse hazard.** Between segment
`i` and segment `i+1` an eager break function runs and can take a block from the
free list that segment `i` still reads on every replay, and the result is silently
wrong numerics rather than a fault. Two segments make this strictly worse than the
single-capture case, because the window between them is host code by construction.
The seam must therefore make the intermediates a segment reads unavailable to the
free list for the life of the `BreakableGraph`, which is the same guarantee
SGLang gets from `use_count` pinning (`breakable_cuda_graph.py:14-23`) by a
different route.

**A clean `compute-sanitizer` run is NOT evidence of safety for this class.** The
recorded incident reproduced 5 out of 5 times under normal operation and 0 out of 2
under memcheck, because memcheck's serialization keeps the freed block from being
reused before the replay. Any W-stage that reports a clean sanitizer run as its
safety evidence has reported nothing. The evidence is G1 over more than one
replay, plus an adversarial case that allocates between segments.

**D2, capture bakes the host source address of an upload.** `qwen3_5.h:262-268`
records that the persistent inputs are the HOST step vectors, mutated in place,
because on GB10 the host-to-device copies are capturable and a replay re-reads
from the fixed host address. A break function that captures a reference to a host
temporary, or a model that reallocates a host vector between capture and replay,
produces a replay that reads freed host memory. The seam owns this rule and must
state it at the `GraphBreak` declaration, because it is the rule a new model author
is least likely to know.

**D3, the model-specific parts must not migrate.** The shape key, the pre-warm
forward and the output buffer set are model knowledge. A seam that tries to own
them becomes a second dispatcher with a per-model branch, which is the shape this
row exists to remove. The test of a correct split is that adding a tenth model
adds a break-point registration and a shape key, and no capture machinery.

**D4, the interaction with `ENG-CUDAGRAPH-DEDUP` (#1162), owned in parallel.**
That row re-points ONE `cudaGraphExec` through `cudaGraphExecUpdate` on a topology
signature hit, instead of instantiating one executable per padded bucket. The two
rows touch the same handle: dedup changes what `EndCaptureGraph` hands back and
who owns it, and this row multiplies the number of such handles by the segment
count. They compose in SGLang, where `BreakableCUDAGraph._append_segment`
(`breakable_cuda_graph.py:266-276`) registers each segment with an optional
`_deduped_cuda_graph`, and `_begin_new_segment` (`:335-350`) asks for
`keep_graph=True` only when dedup is active. **Decision: this row lands first and
treats a segment handle as opaque.** `vt::BreakableGraph` must store the handle,
never inspect it, and must route every acquisition and release through
`Backend::EndCaptureGraph` and `Backend::DestroyGraph`, so #1162 can later
interpose at the backend without editing the container. Coordination is in writing
only, through this paragraph and the issues; neither agent edits the other's files.
The ordering risk if #1162 lands first is that segment count multiplies its
signature table; that is a sizing question for #1162, not a correctness question
for this row.

**D5, the framing risk, and the decision on it.** The strongest failure mode for
this row is that somebody sells it as a speed row. It is not one, and the
supporting measurements are recorded and dated: prefill idle between launches at
3.8% with GPU-busy above 96%, and the 27B prefill gap at 92.5% non-GEMM glue GPU
work, both on GB10, both in
[sglang-breakable-cuda-graph.md](sglang-breakable-cuda-graph.md) §3. Decode already
banked its launch-overhead win. **Decision: this spec declares no throughput gate,
and `## Gates` states the two conditions any later speed claim must satisfy
first.** A negative result is regime-dependent, so the refutation is dated and
hardware-specific rather than permanent; the burden is on the claimant to name the
path and the measurement, not on this spec to pre-refute an unnamed one.

**D6, the non-CUDA backends.** Tenstorrent implements capture as ttnn mesh-trace
(`tenstorrent_backend.cpp:18,75-81`), which is a different runtime concept from a
CUDA stream capture. A segmented capture assumes that ending a trace and starting
another on the same queue mid-forward is legal and cheap. That holds for CUDA by
SGLang's production use and is UNVERIFIED for ttnn traces. **Decision: the seam is
defined against the existing `vt::Backend` vocabulary and asks nothing new of any
backend, so a backend that cannot segment reports `SupportsGraphCapture()` and
runs one segment, which is exactly today's behavior.** No backend is required to
change for W1.

**D8, this row states no runnable gate command, and that is the honest record.**
Moving the row to `READY` puts it in the gated population, where the gate-command
classifier asks whether its `## Gates` section names a command that can FAIL. It
does not, because no test exists to run until W1 lands one, so the row classifies
`gates-no-command` and the pinned runnable set does not move for this change.

The classifier reads a backticked command out of the `## Gates` section, and the
first draft of this spec tripped it: `git diff --stat`, written as advice on how to
read a mutation result, was extracted as this row's gate command and made the row
claim gate coverage it does not have. The advice moved here and the section now
names no command. **Decision: do not manufacture a command to satisfy the
classifier, and do not re-pin the runnable set for a row with nothing to run.** W1
earns the entry and re-pins the baseline in the same change.

**D7, a break point is a correctness boundary, not only a coverage knob.** Every
break point converts a graphed operation into a host-dispatched one, so a
mis-registered break point is invisible in a token gate and visible only as a
segment count. G3 exists for this reason. A model whose break-point set is wrong
still produces correct tokens; it produces them from more segments than it needs,
or from a graph that was never entered at all.

## Now

`READY`. The spec is committed, the design is grounded in both oracles, the
inventory is enumerated with anchors, and the work is decomposed into six landable
stages. No production code exists for this row. W1 needs an `rc` GPU lease to
confirm its exit criterion before implementation starts.
