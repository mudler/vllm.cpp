# ENG-CUDAGRAPH-BREAK — a shared `vt` capture seam that accepts break points

**Row:** `ENG-CUDAGRAPH-BREAK` (engine-matrix, engine core and scheduling).
**Issue:** [#1163](https://github.com/mudler/vllm.cpp/issues/1163), with
[#1020](https://github.com/mudler/vllm.cpp/issues/1020) as one reachable instance.
**Parent analysis:** [sglang-breakable-cuda-graph.md](sglang-breakable-cuda-graph.md) `## Owed`.
**Kind:** scoping spike plus structured spec. No production code lands with this spec.

**This is a coverage row and a CORRECTNESS row, and it is not a throughput row.**
The spike claims no speed and measures none. `## Gates` below states what would have
to be named and measured before any speed claim is admissible, and
`## Risks/decisions` D5 records why no such path is named today.

The correctness half is not a framing device, it is already shipped damage
([#1179](https://github.com/mudler/vllm.cpp/issues/1179)). `src/vllm/model_executor/models/qwen3.cpp`'s `DenseDecodeGraphForward`
DECLINES its decode graph outright whenever the asynchronous device-token mirror is
live, and its comment records the measurement that forced it: `depth-1, graph ON
PASS 78/78`, `depth-2, graph OFF PASS 82/82`, `depth-2, graph ON FAIL, slots 1-3
degenerate`. The comment names the real fix — read the identifiers at REPLAY time
from a stable device buffer — and that fix already exists, in the sibling driver,
as `StepDevInputs` (`qwen3_5.cpp:3894`). One driver has the capability and a shipped
model lost its decode graph because it does not. That is duplication producing a
correctness regression, not a tidiness complaint, and it is the second reason this
row exists.

**THE SENTENCE ABOVE ABOUT `StepDevInputs` IS FALSE, and W4 measured it false
([#1307](https://github.com/mudler/vllm.cpp/issues/1307)).** It is kept as
written because it is what this row was scoped on and `## Owed` has to be read
against it, not because it is true. `StepDevInputs` has no token-id member; its
pinned sibling's `token_ids` block was filled every step and never uploaded and
never read; every batched driver embeds OUTSIDE the captured region from the HOST
vector. No driver has the capability, so the decline was never one refactor away
from removable. The decline's own recorded CAUSE is falsified too. `## Owed` and
`## Now` carry the corrected record, and the decline's comment in `qwen3.cpp`
was repaired in the same pull request.

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

Every bare `:N` anchor below is a line in the one file that implements BCG,
spelled here once in full:
`python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`.
That file is 374 lines at the pin (`wc -l` = 374), so no anchor past `:374` exists.

| Mechanism | Anchor |
|---|---|
| Break-point wrapper | `:204-243` (`eager_on_graph`), whose `wrapper` body is `:209-241` |
| Output writeback on replay | `:231-235` (`replay_fn`), `:172-201` (`_copy_output`) |
| Active-capture context | `:63` (`_current_capture_var`, a `ContextVar`) |
| Segment container | `:246-274` (`BreakableCUDAGraph`); fields at `:251-252`; `_append_segment` at `:266-274` |
| Interleaved replay | `:255-264` |
| Capture scope enter and exit | `:309-320` (`__enter__`), `:322-333` (`__exit__`) |
| Segment open and close | `:335-350` (`_begin_new_segment`), `:352-367` (`_end_current_segment`) |
| Auxiliary-stream fork tracking | `:101-153` (the `wait_stream` hook), installed at `:310` and removed at `:332` |
| Auxiliary-stream auto-join before a segment ends | `:353-361` |
| Bare break marker | `:370-374` (`break_graph`, an empty body under the decorator) |
| Shared mempool and weak refs | `:14-23` doc, `:156-169` (`_weak_ref_if_tensor`) |
| No-compiler backend | `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py:14-17` |

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
The spike confirmed this SHAPE in C++, on a recording backend with no GPU
attached; whether CUDA permits the re-begin mid-forward is W1's exit criterion, not
a W0 result. See `## Work breakdown` W0.

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

NINE production drivers exist. Six are batched decode drivers with a padded-bucket
ring; three are single-shape drivers. Every one of them re-derives the same
machinery. The count is nine and not the eight recorded in `9bc4d7f44`
([#1179](https://github.com/mudler/vllm.cpp/issues/1179)): the DFlash draft graph is
file-local with no header declaration, which is why the parent analysis missed it.

| Driver | Declaration | Implementation | Shape |
|---|---|---|---|
| `Qwen3_5DecodeGraph` | `include/vllm/model_executor/models/qwen3_5.h:275` | `src/vllm/model_executor/models/qwen3_5.cpp:9540,9563,9712,9893,9918,9922` | batched, padded ring |
| `Qwen3_5DenseDecodeGraph` | `include/vllm/model_executor/models/qwen3_5_dense.h:391` | `src/vllm/model_executor/models/qwen3_5.cpp:9969,9986,10321,10340,10348` | batched, padded ring |
| `Qwen3MoeDecodeGraph` | `include/vllm/model_executor/models/qwen3_moe.h:117` | `src/vllm/model_executor/models/qwen3_moe.cpp:385,404,469,512,515,524` | batched, padded ring |
| `Qwen3DenseDecodeGraph` | `include/vllm/model_executor/models/qwen3.h:243` | `src/vllm/model_executor/models/qwen3.cpp:583,613,682,866,888,897` | batched, padded ring |
| `DeepseekV2DecodeGraph` | `include/vllm/model_executor/models/deepseek_v2.h:324` | `src/vllm/model_executor/models/deepseek_v2.cpp:887,902,963,980` | batched, padded ring |
| `VoxtralDecodeGraph` | `include/vllm/model_executor/models/voxtral.h:126` | `src/vllm/model_executor/models/voxtral.cpp:448,469,535,582,585,594` | batched, padded ring |
| DeepSeek V4 `V4Graph` | none, file-local | `src/vllm/model_executor/models/deepseek_v4.cpp:1673,1719,1898,1900,1904` | single shape, `gstate` 0/1/2 |
| Laguna decode graph | none, file-local | `src/vllm/model_executor/models/laguna.cpp:2373,2689,2691,2692,2695` | single shape |
| DFlash draft graph (the ninth) | none, file-local | `src/vllm/model_executor/models/qwen3_dflash.cpp:771,870,1038,1091,1095,1106` | single shape, `g_state` 0/1/2 |

**What every driver re-derives.** Each item below was read in at least two drivers
and the anchors give one instance each.

1. The environment kill switch. SIX drivers read `VLLM_CPP_CUDAGRAPH` themselves:
   `qwen3_5.cpp:9540`, `qwen3_5.cpp:9969`, `qwen3_moe.cpp:385`, `qwen3.cpp:583`,
   `deepseek_v2.cpp:887`, `voxtral.cpp:448`. Instrument:
   `grep -rn 'std::getenv("VLLM_CPP_CUDAGRAPH")' src/` returns exactly those six.
   (`grep -rn VLLM_CPP_CUDAGRAPH src/` returns seven, because `qwen3.cpp:947`
   mentions the name in a comment and is not a read site.) The three single-shape
   drivers each invented their own switch instead: `VT_V4_DECODE_GRAPH` at
   `deepseek_v4.cpp:1680`, `VT_DFLASH_GRAPH` at `qwen3_dflash.cpp:870`, and
   `VT_LAGUNA_DECODE_GRAPH` at `laguna.cpp:1684`. There is no one switch that
   turns capture off.
2. The cold, warm, captured state machine. `qwen3_5.cpp:9563` carries `warm` and
   `captured` flags on the slot; `deepseek_v4.cpp:1719` spells the same thing as
   `int gstate = 0`; `qwen3_dflash.cpp:771` spells it as `int g_state = 0`. Three
   spellings of one three-state machine.
3. The pad-to-bucket selection. SIX call sites of `PadToCaptureSize`:
   `qwen3_5.cpp:9712`, `qwen3_5.cpp:10140`, `qwen3_moe.cpp:469`, `qwen3.cpp:682`,
   `deepseek_v2.cpp:963`, `voxtral.cpp:535`. Instrument: `grep -rn PadToCaptureSize
   src/` returns 16 lines, of which four are `#include` comments and six are prose
   comments; the six above are the invocations. The function itself is already
   shared, at `include/vllm/model_executor/models/decode_graph_sizes.h:47-54`, which
   is the proof that extraction works and is the pattern this row extends.
4. The per-slot persistent buffer set. `qwen3_5.cpp:9563-9600` holds `token_ids`,
   `positions`, the two metadata structs, `hidden`, `logits`, `aux`, `graph`,
   `fa_cols`, `captured`, `warm`, `replays`, `reuse_event`, `dev` and `pin`. Every
   other batched driver holds a subset of the same fields under the same name
   `SizeSlot`.
5. The diagnostics pair `captured()` and `replay_count()`, SIX times. The
   `replay_count()` definitions are at `qwen3_5.cpp:9680`, `qwen3_5.cpp:10110`,
   `qwen3_moe.cpp:454`, `qwen3.cpp:667`, `deepseek_v2.cpp:948`, `voxtral.cpp:519`,
   each preceded one line above by its `captured()` sibling
   (`qwen3_5.cpp:9679`, `qwen3_5.cpp:10109`, `qwen3_moe.cpp:453`, `qwen3.cpp:666`,
   `deepseek_v2.cpp:947`, `voxtral.cpp:518`). Instrument:
   `grep -rn '::captured() const' src/vllm/model_executor/models/*.cpp` returns six.
6. The capture-failure recovery. `qwen3_5.cpp:9913`, `qwen3_5.cpp:10335` and
   `qwen3_dflash.cpp:1106` each carry their own copy of the same
   `try { g = EndCaptureGraph(); } catch (...) {}` drain, because a skipped
   `EndCaptureGraph` poisons the stream permanently.
7. The invalidate-and-recapture rule on a shape change. `qwen3_5.cpp:9816-9826`
   destroys the graph when the block-table column count moves;
   `qwen3_dflash.cpp:1038-1047` does the same when the block width moves.

**The divergence that proves the cost.** The persistent DEVICE input path, called
"Option A" in the tree, exists in exactly one driver. `StepDevInputs`
(`qwen3_5.cpp:3894`) appears on **41** lines of `qwen3_5.cpp` and **zero** lines of
`qwen3_moe.cpp`, `qwen3.cpp`, `deepseek_v2.cpp` and `voxtral.cpp`. Instrument:
`grep -c StepDevInputs <file>`, which counts MATCHING LINES, not occurrences; the
occurrence count on the same file is 47 (`grep -o StepDevInputs qwen3_5.cpp | wc -l`).
An earlier draft of this spec recorded 33, which is neither number and is withdrawn.
The four zeros reproduce under both instruments and the conclusion is unchanged. The other four replay against host vectors. One of them,
`qwen3.cpp:961-986`, therefore has to DECLINE the graph outright whenever the
asynchronous device-token mirror is live, and its comment records the measured
failure that forced it: "depth-2, graph ON FAIL, slots 1-3 degenerate". The same
comment names the real fix as reading the identifiers at replay time from a stable
device buffer, which is precisely what `StepDevInputs` already does in the sibling
driver. One capability, written once, unavailable to four models. That is the cost
of nine drivers stated as a defect rather than as an aesthetic complaint. It is also
why this row is a CORRECTNESS row and not only a coverage one.

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
   `std::vector<std::function<void()>> break_fns` (each a seam-built replay
   closure, see item 3, never a caller's raw `fn`). Invariant:
   `segments.size() == break_fns.size() + 1`. It is ASSERTED, in the two places
   where violating it would otherwise be silent: `GraphCaptureScope` refuses to
   open on a container that already holds a capture (re-entry appended to it and
   produced `break_count() == segment_count()`), and `Replay` refuses counts that
   disagree (such a graph replays with its LAST BREAK DROPPED, whose only symptom
   is a forward that skips one host-dependent operation). Documenting it was not
   enough: the first draft said "asserted" in three records and asserted it
   nowhere. `Replay(Queue&)` walks `ReplayGraph(segments[i])` then
   `break_fns[i]()` for each `i`, mirroring `breakable_cuda_graph.py:255-263`. The destructor calls `DestroyGraph` on every
   segment. This is the direct analogue of `BreakableCUDAGraph` at `:246`.

2. **`vt::GraphCaptureScope`** — a resource-acquisition-is-initialization (RAII)
   scope. Its constructor sets one `thread_local GraphCaptureScope*` and calls
   `Backend::BeginCapture`; its destructor calls `Backend::EndCaptureGraph`, pushes
   the final segment, and clears the pointer. This is `BreakableCUDAGraphCapture`
   at `:277-367`, with the destructor doing what `__exit__` does at `:322-333`. The
   thread-local replaces SGLang's `ContextVar` at `:60`. It must be
   `thread_local`, not a global, for the same reason SGLang chose a `ContextVar`
   and CUDA chose `cudaStreamCaptureModeThreadLocal` (`cuda_backend.cu:204-206`):
   capture is a property of one thread's stream, and a process-wide flag would
   make an unrelated thread's forward observe a capture it is not part of.

3. **`vt::GraphBreak(fn, out)`** — the break point. Outside a capture scope it
   calls `fn` and returns, so a non-capturing forward is byte-identical to today.
   Inside a scope it closes the current segment, runs `fn` once eagerly so the
   outputs hold real data, appends a REPLAY CLOSURE to `break_fns`, and opens a
   new segment. This is `eager_on_graph` at `:204-243`.

   **The appended function is not `fn`.** Upstream builds a `replay_fn` closure at
   `:231-235` that calls the captured inner function and then writes its result
   back into the capture-time destination:

   ```python
   def replay_fn():
       new_out = captured_inner(*captured_args, **captured_kwargs)
       return _copy_output(captured_output, new_out)
   ```

   `_copy_output` (`:172-201`) does an in-place `dst.copy_(src)` for a tensor, and
   recurses field-wise over an object's `__dict__` or a dict's values; it returns
   `src` only when neither side is copyable. The companion `_weak_ref_if_tensor`
   (`:156-169`) is what makes the capture-time destination outlive the eager call.

   **This is load-bearing, not decorator sugar.** On replay N the eager operation
   returns a FRESH allocation, whose address is not the one segment `i+1` baked at
   capture time. A container that replays the raw `fn` and discards its return value
   drops that guarantee silently: the next segment keeps reading the capture-time
   address while the break function writes somewhere else. The failure is WRONG
   NUMERICS, not a fault, which is exactly the class `## Risks/decisions` D1 records
   `compute-sanitizer` cannot see.

   **Contract this seam adopts, stated at the `vt::GraphBreak` declaration.** One of
   the two, and the seam must make the choice explicit rather than leave it to the
   caller:

   - `GraphBreak(fn, slot)` takes the destination the following segment reads, and
     the seam copies `fn`'s result into it on every replay. This is the direct
     port of `replay_fn` plus `_copy_output`. **The destination is a
     `vt::BreakSlot<T>`, never a bare `Out&`,** and that is a correctness
     requirement rather than a style choice. The destination must outlive the
     `BreakableGraph`, because the following segment bakes its address; upstream
     gets that from holding `captured_output` BY VALUE with its storage pinned by
     the segment graphs' mempool (`:156-169,225-227`). A C++ seam taking a
     reference gets the opposite, because the natural call site declares the
     destination as a local of the function containing the break — which is
     exactly what the first W1 site did, registering a stack slot that died on
     the next `return` while the pooled block it named went back on the free
     list. `BreakSlot` moves the value into a cell the replay closure owns, so
     the rule holds by construction and the obvious call site cannot break it.
     A destination type with no `CopyOutput` overload is a COMPILE error naming
     the type, not a silent downgrade into the fallback below.
   - `GraphBreak(fn)` with no destination requires `fn` to write IN PLACE into a
     persistent buffer that the model owns and that no replay reallocates. The
     no-destination form is then only legal for a break function with no return
     value; the bare marker below is its degenerate case. Upstream's own
     non-copyable fallback (`:201`, "return `src`, leave the destination alone")
     stays reachable, but it has to be ASKED FOR by passing `vt::NoWriteback{}`,
     because a fallback that is the DEFAULT makes a renamed, moved or misspelled
     `CopyOutput` overload compile clean and be wrong only on replay N.

   `vt::BreakableGraph` therefore holds `std::vector<std::function<void()>>` whose
   elements are the SEAM's closures, never the caller's `fn` — and D9 records the
   hazard the distinction removes.

**The fork-join rule the scope owns.** Closing a segment is illegal while a side
stream forked inside it is still participating in the capture. Upstream solves this
inside `_end_current_segment` (`:353-361`): before `capture_end()` it walks the set
of streams forked but not rejoined and issues the join itself. Tracking that set is
the entire purpose of the `torch.cuda.Stream.wait_stream` hook (`:101-153`), which
`__enter__` installs at `:310` and `__exit__` removes at `:332`.

**This is live for us, in both drivers W4 and W5 migrate.** Our CUDA backend reports
`SupportsAuxStream() == true` (`src/vt/cuda/cuda_backend.cu:200`, against the base
`false` at `include/vt/backend.h:166`), and two model paths fork on it. Qwen3.5's MoE
shared-expert overlap records a fork event on the main queue and makes the auxiliary
queue wait it at `src/vllm/model_executor/models/qwen3_5.cpp:6254-6255`, records the
completion event at `:6261`, and joins the main queue back at `:6384` — so any break
point placed between `:6255` and `:6384` closes a segment with the fork outstanding.
Laguna's decode-graph class owns its own auxiliary queue and event pair
(`src/vllm/model_executor/models/laguna.cpp:2362-2367`, torn down at `:2374-2378`),
forks at `:2572-2573`, records at `:2576` and joins at `:2612` — and unlike Qwen3.5's
that fork is INSIDE the captured region by construction, because the class is the
capture driver.

**Rule: `GraphCaptureScope` owns the set of queues forked since the current segment
opened, and joins every outstanding one before it calls `EndCaptureGraph`.** We need
no `wait_stream` monkey-patch to populate that set, because our fork and join are
explicit `Backend::RecordEvent` and `Backend::QueueWaitEvent` calls on a seam we
control, not an implicit torch API. D10 records what happens if the rule is omitted.

**Why a thread-local and not a parameter.** The break site sits inside an attention
call, several frames below the forward entry point. Threading a capture context
through every intermediate signature would touch every model. SGLang made the same
call for the same reason. The spike proved the thread-local carries the context
across three frames with no intermediate signature change; see `## Work breakdown` W0.

**A bare marker.** `vt::GraphBreak()` with no argument splits the segment and runs
nothing, mirroring `break_graph` at `breakable_cuda_graph.py:370-374`. It is the
form a model uses when the host-dependent work already ran outside the forward.
It has no output, so the writeback contract above is vacuous for it.

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
`torch.compile` configuration tests, and we have no compiler to configure.

**SGLang's BCG DOES carry a unit suite at the pin, and it ports.** An earlier draft
of this spec asserted there was nothing to port, which was wrong and is withdrawn.
The file is
`test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py` at
`f63458b5be`, 305 lines, four classes. AGENTS.md requires the upstream tests in the
same change that ports the behavior, so the table below maps every upstream case to
the local test that owes it. Parameters, modes and failure cases are preserved; the
harness adaptation is stated once at the bottom.

**`TestBreakableCUDAGraphBasic` (`:30`) — the capture and replay mechanism.**

| Upstream case | Anchor | What it pins | Local test |
|---|---|---|---|
| `test_no_break_capture_replay` | `:49-63` | zero breaks captures and replays exactly like a plain graph: `y = x + 1`, refill `x` to 5.0, replay, expect 6.0 | T1: a scope with no `GraphBreak` yields 1 segment, 0 break functions, and one `Begin`/`EndCaptureGraph` pair |
| `test_single_break` | `:65-87` | one break splits into two segments and the chain composes: `x=10 -> +1=11 -> eager *2=22 -> +3=25` | T2: 1 break yields 2 segments and 1 break function, and the arithmetic chain holds |
| `test_multiple_breaks` | `:89-115` | two breaks, three segments, chained: `x=5 -> +1=6 -> +1=7 -> +1=8 -> *2=16` | T3: N breaks yield N+1 segments and N break functions, asserted at N=2 with the same values |
| `test_eager_on_graph_disabled` | `:117-129` | with the wrapper DISABLED the function is returned unchanged and runs normally | T4: `VLLM_CPP_CUDAGRAPH=0` (item 1 of `## Our baseline`) makes `GraphBreak(fn)` call `fn` and make zero backend calls |
| `test_eager_on_graph_outside_capture` | `:131-142` | outside any capture the wrapper is a pass-through | T5: with no scope active, `GraphBreak(fn)` runs `fn` and makes zero backend calls |
| `test_replay_updates_output` | `:144-169` | TWO replays with different inputs give different outputs: `3.0` then, after `x.fill_(10)`, `33.0` | T6: the multi-replay case. This is the upstream anchor for G1's "more than one replay" requirement, and the one case that can see a break function writing to a stale address |

**`TestCopyOutput` (`:172`) — the output-writeback contract of `## Port map` §3.**
This class is why F2's writeback is not optional: upstream tests `_copy_output`
separately from the capture machinery, because it is a separate guarantee.

| Upstream case | Anchor | What it pins | Local test |
|---|---|---|---|
| `test_tensor_copy` | `:187-192` | a tensor destination is written IN PLACE and the SAME object is returned (`assertIs(result, dst)`) | T7: `GraphBreak(fn, out)` copies into `out`; the device pointer `out` held before the call is the one holding the data after it |
| `test_dict_copy` | `:194-208` | a keyed set of destinations is copied field by field | T8: a multi-output break (logits plus an auxiliary tap, the `qwen3_5.cpp:9570-9577` shape) writes every destination |
| `test_object_copy` | `:210-223` | a struct destination copies its tensor fields in place and ASSIGNS its non-tensor fields (`dst.label == "new"`) | T9: a struct-valued break preserves the in-place/assign split |
| `test_non_tensor_fallback` | `:225-227` | with nothing copyable, `_copy_output` returns `src`, the documented fallback | T10: a break function with no device output is legal and the seam does not fabricate a copy |

**`TestBreakGraphHelper` (`:230`).**

| Upstream case | Anchor | What it pins | Local test |
|---|---|---|---|
| `test_break_graph_inserts_segment` | `:249-265` | the BARE marker splits the segment even though its body does nothing: `x=10 -> +1=11 -> break -> +2=13` | T11: `vt::GraphBreak()` with no argument yields 2 segments and 1 (empty) break function, and the value chain is unaffected |

**`TestBreakableCudaGraph` (`:268`) — `test_gsm8k_accuracy` (`:288`) is DELIBERATELY
EXCLUDED from this row**, and the reason is not cost. It launches a Qwen3-8B server
with `--cuda-graph-backend-prefill=breakable` (`:279-281`) and asserts `mgsm_en >=
0.80` over 1319 examples (`:289-301`). That is a DISTRIBUTIONAL accuracy floor on a
PREFILL capture path. Both halves are wrong for us. Our gate polarity for a migrated
model is BIT-EXACTNESS against the model's own eager forward (`## Gates` G1), which
is strictly stronger than an accuracy floor and is the gate AGENTS.md requires when a
greedy path exists. And prefill capture is REFUTED as a lever here and is not what
this row builds (`## Risks/decisions` D5). The upstream case's PURPOSE — an
end-to-end model-level check that a segmented capture does not corrupt generation —
is carried by G1 against `tests/parity/test_qwen36_paged_engine.cpp:140`, per
migrated model. Excluded as redundant to a stronger gate, not as unaffordable.

**The one unavoidable harness adaptation.** Every upstream class's `setUpClass`
raises `unittest.SkipTest` without CUDA (`:34-36`, `:177-178`, `:235-236`), so
upstream runs all eleven cases on a real device. T1 through T5, and T7 through T11,
run instead against a test backend that RECORDS the call sequence and SIMULATES the
graph: `Record(fn)` files one captured operation against the open segment and does
NOT execute it, exactly as a real stream capture files a kernel without running it,
and `ReplayGraph` runs the filed work in order. That is what keeps upstream's
POST-REPLAY assertions literal — `x.fill_(5); graph.replay(); y == 6` is the same
assertion here — rather than degrading them into capture-time value checks, which
assert something upstream never asserted. The same break counts, the same segment
counts, the same arithmetic chains (`x=10 -> 11 -> 22 -> 25`, `x=5 -> 6 -> 7 -> 8 ->
16`, `3.0` then `33.0`, `x=10 -> 11 -> 13`), the same in-place versus assign split
and the same non-copyable fallback.

**Two things the harness does NOT model, named rather than claimed away.** Real
device memory, asynchrony, and any error a real runtime would raise are absent; a
case that needs those is a GPU case and is owed rather than approximated. And
upstream's `eager_on_graph` always returns a fresh tensor, so every upstream break
is the DESTINATION form — the in-place form has no upstream case and is exercised
beside T2 and by the bare marker.

The existing test surface the migration must keep green:

| Existing test | Path | Why it binds |
|---|---|---|
| Capture-size set and pad selector | `tests/vllm/models/test_decode_graph_sizes.cpp` | the shared piece the seam must not change |
| CUDA capture capability | `tests/vt/test_cuda_backend.cpp:108` | the seam's backend contract |
| ROCm capture capability | `tests/vt/test_rocm_backend.cpp` | proves the seam is not CUDA-only |
| Tenstorrent trace capture | `tests/vt/test_tenstorrent_backend.cpp:1370,1376` | a third capture backend with different semantics |
| Retire-on-grow bookkeeping | `tests/vt/test_graph_safe_scratch.cpp` | the lifetime rule the seam enforces |
| 35B paged engine gate | `tests/parity/test_qwen36_paged_engine.cpp:140` | the model gate the first migration must hold |

**Tests this row owes with no upstream counterpart**, each written red first,
numbered on from the ported set:

12. **Replay ORDER, not only replay arithmetic.** Replay emits segment 0, break 0,
    segment 1, break 1, and so on, in that exact order, and the order is identical on
    the second and third replay. Upstream asserts the composed VALUE (T2, T3), which
    a wrong order could in principle still satisfy for a commutative chain; this
    asserts the sequence directly.
13. **Capture-failure drain.** A break function that throws leaves the stream
    un-poisoned and the partially built `BreakableGraph` destroyed with every
    already-instantiated segment released. This ports the recovery that
    `qwen3_5.cpp:9913` and `qwen3_dflash.cpp:1106` each hand-rolled. Upstream has no
    equivalent because Python's `finally` in `__exit__` (`:323-332`) covers it.
    **The BEHAVIOUR was owed too, not only the test**, and an earlier draft of
    `## Owed` said otherwise. The destructor's `catch` guarded a throwing
    `EndCaptureGraph` and nothing else: an exception from a break function, or
    from ordinary model code between two break points, left a PARTIAL capture
    reporting `captured() == true`, which is replayable as half a forward.
    Comparing `std::uncaught_exceptions()` against the depth recorded at scope
    entry is what separates the two, and all three arms are gated (13a break
    throws, 13b ordinary code throws mid-segment, 13c `EndCaptureGraph` throws).
14. **Non-capturing backend.** On a backend where `SupportsGraphCapture()` is
    false, the scope is inert and the forward runs eager. Vulkan
    (`vulkan_backend.cpp:16`) and Metal (`metal_backend.mm:13`) are the live cases.
    Upstream has no equivalent because BCG is CUDA and HIP only.
15. **Auxiliary-stream auto-join.** A segment that forks a queue and does not join
    it before a break point is joined by the scope, per `## Port map`. This is the
    port of `_end_current_segment` `:353-361`, which upstream exercises only
    indirectly through its model-level integration test.
16. **Bit-exactness per migrated model**, over more than one replay. See `## Gates`.

Tests 1 through 15 need no graphics processing unit (GPU), except T6's value arm.
They run against a test backend that records the call sequence. Test 16 needs a GPU
and an `rc` lease.

## Gates

**Correctness first. No performance result is accepted before the token-exact gate
for that model is established.**

**G1, bit-exactness against eager, per migrated model, over MORE than one replay.
RE-RUN AND EXTENDED for W4, 2026-08-19 ([#1307](https://github.com/mudler/vllm.cpp/issues/1307)),
on `thor:gpu0` through an `rc` lease** — NVIDIA Thor, sm_110, driver 595.78,
nvcc 13.0.88, source `4ea38eccbf82bbe6b8a227753fe64463fb840b78`, CUDA-ON build
(`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110`), 32 `.cu.o` objects, the
46.3 MB binary resolving `libcudart.so.13` and `libcublasLt.so.13` out of
`/usr/local/cuda-13.0/targets/sbsa-linux/lib`.
`tests/vllm/models/test_decode_graph_seam_g1_cuda.cpp` ran **5 cases, 2066
assertions, 0 failed, exit 0**:

```
G1 Qwen3MoeDecodeGraph    on CUDA: 5 steps x 100 logits, 0 differing, 4 replays
G1 VoxtralDecodeGraph     on CUDA: 5 steps x 100 logits, 0 differing, 4 replays
G1 DeepseekV2DecodeGraph  on CUDA: 5 steps x 100 logits, 0 differing, 4 replays
G1 Qwen3_5DecodeGraph     on CUDA: 5 steps x  40 logits, 0 differing, 4 replays
G1 Qwen3_5DenseDecodeGraph on CUDA: 5 steps x 40 logits, 0 differing, 4 replays
```

**G1 WAS RE-RUN AT W5's HEAD on `thor:gpu0` through an `rc` lease, and it is a
re-run rather than a new case.** The seam changed UNDER the five measured
drivers: D10 put a `JoinOutstandingForks()` call on the path of EVERY segment
close, so each migrated driver's capture now executes seam code W4's run did not,
and a change to the shared close path is exactly what a CPU harness cannot clear.
Same box and same provenance as W3 and W4 — NVIDIA Thor, sm_110, driver 595.78,
nvcc 13.0.88, `-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110`, 32 `.cu.o`
objects — at source `79dc6b5bd`:
`tests/vllm/models/test_decode_graph_seam_g1_cuda.cpp` ran **5 cases, 2066
assertions, 0 failed, exit 0**, every driver again reading `0 differing, 4
replays`. On the same device and in the same job: the full CUDA library built
clean (`lib_rc=0`, 742 targets), `tests/vt/test_breakable_graph.cpp` ran 30 cases
and 265 assertions, and `tests/vllm/models/test_qwen3_dflash_decode_graph_seam.cpp`
ran **4 cases, 23 assertions, exit 0** — so W5's own driver gate, including the
#1352 child-process arm, is green on a CUDA build and not only on this box's CPU
one. That last line is a build-and-routing result rather than a replay one: the
DFlash case still runs against the CPU harness inside that binary, which is why
G1 for that driver stays owed below.

**THE ONE THING A GREEN BUILD COULD NOT HAVE TOLD US was measured separately.**
Laguna's capture class is inside `#ifdef VT_MARLIN_NVFP4`, so "the CUDA build
compiled `laguna.cpp`" is satisfied just as well by a build that compiled the
migrated region OUT — success and failure are the SAME OBSERVATION. Two facts
settle it. `-DVT_MARLIN_NVFP4=1` is on `laguna.cpp`'s own compile command in
`compile_commands.json`, and CMake reported `CUDA feature marlin-nvfp4: ENABLED
for [110]`. And the region itself was MUTATED: an undeclared identifier inserted
immediately after `vt::GraphCaptureScope scope(b, q, graph, kFull)` FAILED the
object build under `-Werror` (`laguna.cpp:2735`,
`'VT_W5_MUTATION_THIS_MUST_NOT_COMPILE' was not declared in this scope`) against
a baseline object build of rc 0, and the tree restored byte for byte to an empty
`git diff`. The identical mutation on DeepSeek V4 failed at
`deepseek_v4.cpp:1921`. **Both migrated regions are compiled.** That is the half
of their coverage answerable without the models' own kernels and checkpoints; G1
and G2 for those two remain owed, and `## Owed` states what each needs.

**W5 ADDED NO G1 CASE OF ITS OWN, and says so rather than leaving it to be
inferred from a count.** Its three drivers are single-shape and each refuses the
harness for its own reason.

The two Qwen3.5 cases needed a cache pool the other three did not:
`CudaGdnCachePool` allocates the RECURRENT ssm and conv state on device beside
the paged KV, and each arm gets its own — the GDN recurrence advances its state
every step, so two arms sharing one state would step each other's recurrence and
the agreement would measure nothing. **Five of the NINE migrated drivers are
covered by measurement**; W2's `Qwen3DenseDecodeGraph` and W5's three
single-shape drivers are not, and share the seam by argument rather than by
measurement.

**The W3 run this extends, kept because it is the earlier evidence** (W3, #1291,
2026-08-19, the same `thor:gpu0`) — NVIDIA Thor, sm_110, driver 595.78, nvcc
13.0.88, source at `c905bb536`, CUDA-ON build (`-DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=110`), 32 `.cu.o` objects, the binary resolving
`libcudart.so.13` and `libcublasLt.so.13`.
`tests/vllm/models/test_decode_graph_seam_g1_cuda.cpp` ran **3 cases, 1600
assertions, exit 0**, with per-driver readings `5 steps x 100 logits, 0
differing, 4 replays` for `Qwen3MoeDecodeGraph`, `VoxtralDecodeGraph` and
`DeepseekV2DecodeGraph`. The assertion count is the load-bearing half of that
statement: on a box with no CUDA backend the same file reports `Status:
SUCCESS!` over `assertions: 0`, which is a skip wearing a pass, so a G1 claim
from it is admissible only with a non-zero count and the device named. W2's
driver is covered by construction — its gate ran on the identical seam — but it
is not one of the three cases and is NOT claimed here.

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
| `DecodeGraphSizes` and `PadToCaptureSize` | landed, `decode_graph_sizes.h:47-54` | reused unchanged |
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
- The inventory in `## Our baseline` is the real one: NINE drivers, not the eight
  the parent analysis recorded, and the persistent-device-input capability present
  in exactly one of them.

**W0 did NOT establish** that CUDA permits `cudaStreamEndCapture` followed by
`cudaStreamBeginCapture` on the same stream mid-forward with eager work between
them, on our stream configuration. This box has no NVIDIA GPU, and the fleet is
leased through `rc`. SGLang's `_end_current_segment` and `_begin_new_segment`
(`breakable_cuda_graph.py:335-365`) do exactly this on a production path at the
pinned revision, which is strong evidence but is not our measurement. **This is
W1's exit criterion and it needs a GPU lease.** It is named rather than skipped.

**W1, the seam plus its unit gate, and one break point on one model. DONE
2026-08-18, [#1192](https://github.com/mudler/vllm.cpp/issues/1192).** Landed
`vt::BreakableGraph`, `vt::GraphCaptureScope`, `vt::BreakSlot` and
`vt::GraphBreak` in `include/vt/breakable_graph.h` and
`src/vt/breakable_graph.cpp`, with tests 1 through 5 of `## Tests to port` plus
the `TestCopyOutput` set (T7 through T10), the bare marker (T11), the
replay-order case (test 12), the capture-failure drain (test 13, all three
arms), the non-capturing backend (test 14), the aliasing refusal with its two controls,
and the ownership case that proves every segment is released through
`Backend::DestroyGraph`. `tests/vt/test_breakable_graph.cpp`, 24 cases, 163
assertions, exit 0, re-derived by `ninja test_breakable_graph &&
./build/tests/test_breakable_graph`.

**A fresh review returned FAIL on the first head and the repairs are part of
W1.** What the review found, and what closed it, because each one is a class of
defect a later stage can repeat:

- The production break point's destination was a function-local
  `std::optional<DBuf>`, so both scoped layers registered the SAME stack slot in
  a frame that was gone before the scope closed, and the pooled block it named
  went back on the `DevicePool` free list. Inert at W1 because no driver opens a
  scope, and wrong the moment W2 does. Closed structurally: the destination is a
  `vt::BreakSlot` the seam owns and there is no reference-taking overload left.

  **That closed the LIFETIME half of the rule, and this spec claimed the whole
  rule was "unexpressible to violate". It was not, and a scoped re-review
  measured the other half still open.** One slot reused for TWO break points in
  one capture compiled: `PinForCapture` returns the cell it already made, so
  both replay closures bind to the SAME address, `&*slot` is byte identical
  after break 0 and after break 1, and after `Replay` the slot holds only break
  1's value — break 0's writeback overwritten, and any segment that baked break
  0's destination reading break 1's data. The production site is correct today
  because its slot is a `RunLayer` local, but W2 through W5 add nine more
  callers and the prose was telling them the shape was unwritable. Closed by a
  REFUSAL rather than by the type, and the prose now names which half is which:
  `GraphCaptureScope::AppendBreak` takes the destination as a REQUIRED
  parameter, so no form can register without stating where it lands, and it
  throws when a second break in the same capture names a cell already
  registered. Two things the repair had to get right, both measured rather than
  reasoned. The identity is the CELL and not the SLOT: the production slot is a
  per-call local, every layer's slot lands at the same STACK address, and
  slot-address identity therefore refused the correct program on layer 2 and
  reddened the G2 gate. And the non-copyable fallback registers no destination,
  because it pins no cell and its replay writes back nowhere. Red-first: the
  case failed on the unrepaired head with the aliased capture ACCEPTED,
  `break_count() == 2` into one slot. Two mutations, both detected — deleting
  the refusal reds only the new case (23 of 24 still pass and G2 stays green,
  which is exactly why nothing caught this before), and making it over-fire reds
  two unit cases AND the G2 reachability gate, so the controls are not
  vacuous.
- Interleaved replay was claimed, spec'd, and NOT gated. Replacing the
  interleaved loop with "replay all segments, then run all breaks" left the suite
  green, because break markers went into a vector the backend log knew nothing
  about and the two sequences were asserted independently. Closed by putting the
  markers into the backend's OWN log and asserting one trace.
- The writeback branch degraded SILENTLY: renaming `CopyOutput` dropped the site
  into the non-copyable fallback, which is the D9 wrong-numerics path, with both
  suites green. Closed by a `static_assert` in the destination form plus the
  explicit `vt::NoWriteback` opt-in, and by moving the model's overload into a
  header so the G2 gate can `static_assert` that the writeback branch is the one
  selected.
- The `VLLM_CPP_CUDAGRAPH=0` kill switch was ungated (the existing case
  substituted the OTHER conjunct), nesting was accepted and traced an illegal
  capture sequence, the invariant three records called "asserted" was asserted
  nowhere, and `Reset()` left the replay count stale. Each is now gated, the
  switch through a child process because it is read once per process.
- Upstream fidelity was overstated. T3 had lost its arithmetic chain and T1 its
  post-replay value, because a recording backend cannot re-execute a segment. The
  backend now SIMULATES the graph, so every upstream chain and every post-replay
  assertion ports literally, and the two remaining deviations are named above.

**The exit criterion was answered FIRST, and it holds.** `cudaStreamEndCapture`
followed by `cudaStreamBeginCapture` on the SAME stream mid-forward with EAGER
work between them is legal under `cudaStreamCaptureModeThreadLocal`, which is the
mode `src/vt/cuda/cuda_backend.cu:204-206` uses. Measured on `orin:gpu0` through
an `rc` lease, driver `12060`: segment, host-dependent break, re-begin, segment,
bare zero-work re-begin, segment; then three replays with fresh inputs and
0 mismatches on every one. The BARE re-begin — end then immediately begin with no
work between, the degenerate case of `breakable_cuda_graph.py:370-374` — is legal
too. See `## Outcome` for what the first, refused probe actually measured.

**The break point is the dense attention entry of `Qwen3ForCausalLM`**
(`src/vllm/model_executor/models/qwen3.cpp`, inside `RunLayer`), in its
destination-carrying form because `AttnBlock` returns a fresh pooled buffer on
every call. G2's mutation applies from this stage on and was performed; see
`## Outcome`.

**W2, migrate `Qwen3DenseDecodeGraph` first. DONE 2026-08-18,
[#1261](https://github.com/mudler/vllm.cpp/issues/1261).** The driver's capture
is now the seam's: `Qwen3DenseDecodeGraph::Step` opens a `vt::GraphCaptureScope`
over a per-slot `vt::BreakableGraph` and replays through
`BreakableGraph::Replay`, so the hand-rolled `BeginCapture`/`EndCaptureGraph`
pair, the raw `void*` handle, the `bool captured` half of the state machine, the
`DestroyGraph` loop and the driver's own `VLLM_CPP_CUDAGRAPH` read are gone —
items 1, 2, 5, 6 and part of 4 of `## Our baseline`, absorbed. **This is the
first stage at which the seam is ENTERED from a production step**, which retires
W1's staged slice.

**The migration added one thing the seam did not have, and mirroring is the
reason.** `vt::GraphCaptureMode` ports vLLM's `CUDAGraphMode`
(`vllm/config/compilation.py:59-63`), whose v1 default `FULL_AND_PIECEWISE`
(`:63`) is documented at `:630-632` as "Capture full cudagraph for DECODE
batches and piecewise cudagraph for prefill and mixed prefill-decode batches";
`decode_mode()` (`:65-66`) returns the FULL half and the runtime reads it per
batch at `vllm/v1/worker/gpu/cudagraph_utils.py:185-186`. W1 built the piecewise
mode alone. **A decode driver opened in `kPiecewise` would convert a fully
graphed decode step into one EAGER attention call per layer between two graph
replays** — twenty-eight of them on Qwen3-4B — which is not what vLLM does for a
decode batch, which the secondary oracle does not do either (SGLang reaches BCG
through `--cuda-graph-backend-prefill=breakable`,
`test_breakable_cuda_graph.py:279-281`), and which nothing in this record
supports. `GraphBreak` inside a `kFull` scope therefore takes the same
pass-through arm it takes outside a scope, so the break's work is captured
INSIDE the single segment, and `GraphCaptureScope::AppendBreak` REFUSES a
registration in that mode rather than leaving a container with
`break_count() == segment_count()`. **This is a correctness and mirroring
argument, not a performance one.** No throughput is claimed and none was
measured; `kFull` is what keeps the migrated step's shape the one it already
had.

**That degradation is a CONDITIONAL, and the order matters.** It is what a
piecewise decode capture costs once the capture WORKS. Today it does not: the
same one-token change faults on the first replay for the lifetime reason in the
next paragraph, measured as mutation D in `## Outcome`. Both are true and only
the second is currently observable; W2's first head stated the first as though
the second did not exist.

**The piecewise arm does not reach this driver at W2, and the reason is a
lifetime, not a preference.** The registered break point's closure captures
`RunLayer`'s frame by reference — `dhn`, `si`, `meta`, `T` — every one of which
is a per-step local. On replay N those references name a frame that returned,
which is lifetime rule 2 at the `GraphBreak` declaration. Making the closure
replay-safe means holding the layer's inputs in persistent storage the driver
owns, which is the same capability `StepDevInputs` is and which W4 owns. W1's
own G2 gate never saw this because it counts segments and never calls `Replay`.
Recorded under `## Owed` with W4 as the owner.

**The async decline at `qwen3.cpp` STANDS, and W2 says so rather than leaving
it.** Migrating the CAPTURE machinery does not move the INPUTS: `Step` still
refreshes the persistent HOST vectors the captured copies read, so the depth-2
race the decline avoids is untouched by this stage and removing the decline
would restore the measured failure. The fix is `StepDevInputs`
(`qwen3_5.cpp:3894`) as a SEAM capability, which is W4 by this section's own
words and which is the only version of the fix that also reaches
`qwen3_moe.cpp`, `deepseek_v2.cpp` and `voxtral.cpp`. Hand-copying it into this
driver would be the tenth copy of the capability this row exists to stop
copying. Both arms are now GATED
(`tests/vllm/models/test_qwen3_decode_graph_seam.cpp`) so neither can change
silently, and the mutation that deletes the decline reds that case.

**Gates run.** G2, the reachability mutation, is
`tests/vllm/models/test_qwen3_decode_graph_seam.cpp`: it drives
`Qwen3DenseDecodeGraph::Step` through four decode steps (cold, capture, replay,
replay) and asserts the seam's OWN counters, because a driver calling
`Backend::ReplayGraph` directly leaves a backend log identical to the seam's.
Replacing the scope and `Replay` with the pre-W2 raw pair (18 changed lines,
compiled clean, exit 0) left `test_breakable_graph` 27/27, W1's
`test_qwen3_break_point` 2/2 and `test_qwen3_forward` 10/10 GREEN and turned
this file RED on four assertions — W1's contrast reproduced one stage later.
G4 holds in the same file: the capture step's logits are bit-identical to
`Qwen3DenseModel::Forward`, 100 values, 0 differing. **G1 is NOT met and is
owed**, because a CPU harness cannot replay a captured segment; see `## Owed`.

It went first for three reasons, in order. It is the SMALLEST batched driver by machinery, holding no
`StepDevInputs` and no auxiliary tap buffer. It carries the DECLINE mitigation at
`qwen3.cpp:961-986`, so the migration has a documented, measured failure mode to
gate against rather than a hypothetical one. Its gate models, Qwen3-0.6B and
Qwen3-4B, are the cheapest to run of any driver's. It is deliberately NOT the 35B
MoE driver: `Qwen3_5DecodeGraph` is the richest and the most load-bearing, and
migrating it first would put the hardest correctness surface behind the least
seam experience.

**W3, migrate the three remaining plain batched drivers. DONE 2026-08-19,
[#1291](https://github.com/mudler/vllm.cpp/issues/1291).** `Qwen3MoeDecodeGraph`,
`DeepseekV2DecodeGraph` and `VoxtralDecodeGraph` each open a
`vt::GraphCaptureScope` over a per-slot `vt::BreakableGraph` in `kFull` and
replay through `BreakableGraph::Replay`. Each landed as its own commit with its
own red-first G2 gate, because they share W2's shape and nothing about one
depends on another. What is gone from all three: the hand-rolled
`BeginCapture`/`EndCaptureGraph` pair, the raw `void*` handle, the `bool
captured` flag, the `DestroyGraph` loop and the driver's own
`VLLM_CPP_CUDAGRAPH` read. **The six batched-driver reads `## Our baseline` item
1 counted are down to two**, both in `qwen3_5.cpp`, which W4 owns. The two
per-model rollback switches STAY (`VT_QWEN3MOE_CUDAGRAPH`,
`VT_DEEPSEEK_CUDAGRAPH`): each is an A/B lever for exactly one driver, not a copy
of the shared one.

**Each driver owes its own gate, and the reason is that nothing else can see the
difference.** A driver that kept its raw pair produces identical logits, an
identical backend log and an identical `replay_count()`. `segments_captured` and
`replays` are the only observables that separate "captured a graph" from
"captured a graph THROUGH THE SEAM". Each gate drives `Step` through cold,
capture, replay and a second replay and asserts those. Red-first, each on four
assertions before its migration: `test_qwen3_moe_decode_graph_seam` 222/226 exit
1, `test_voxtral_decode_graph_seam` 224/228 exit 1,
`test_deepseek_v2_decode_graph_seam` 224/228 exit 1. Green after: 3/3 each.

**The harness is now SHARED**, which this row should have done at W2 and did not.
W2's gate carried its own capture-capable CPU backend and static-graph CPU
platform; three more copies would have reproduced inside `tests/` the exact
duplication being removed from `src/`, and two copies of a harness diverge
invisibly because both files stay green while measuring different things.
`tests/vllm/models/decode_graph_seam_harness.h` holds them once and W2's file
includes it, unchanged in behaviour at 4/4 and 231 assertions.

**W3 FOUND A GATE THAT COULD NOT FAIL, and closing it is part of this stage.**
The three gates assert `breaks_registered == 0` to hold the capture to vLLM's
decode arm. That assertion is a TAUTOLOGY for these models: it moves only when a
`vt::GraphBreak` registers into a splitting scope, and the one production break
point in the tree is W1's, in `qwen3.cpp`. W2's driver runs through it, so for W2
the assertion was load-bearing; none of W3's three models registers one.
Measured, not reasoned: flipping `kFull` to `kPiecewise` in `qwen3_moe.cpp` — one
token — compiled clean and left that driver's whole gate GREEN at 226/226. The
mode was UNOBSERVABLE from outside a driver: the scope is a `Step` local, the
container is private to the driver's `Impl`, and a token gate cannot see a
segment count. `vt::GraphBreakStats` gains `full_scopes` and `piecewise_scopes`,
counted in `GraphCaptureScope`'s constructor on the ACTIVE path only (an inert
scope makes no backend call in either mode, so counting it would report a mode
that never reached a backend). Gated with its controls in
`tests/vt/test_breakable_graph.cpp`; the same flip now reds each of the three
driver gates on exactly those two assertions.

**NO BREAK POINT IS REGISTERED IN THESE THREE MODELS, and that is a decision.**
Under `kFull` a break point takes the same pass-through arm it takes outside a
scope, so registering one here would land machinery no gate can exercise.
Establishing each model's break-point set is what the PIECEWISE arm needs; that
arm is blocked on replay-safe closure inputs, which W4 owns, and W6 is where the
eligibility predicate moves.

**G1 IS DELIVERED HERE**, which is the item W1 and W2 both carried as owed. See
`## Gates` G1 and the `## Owed` entry it retires.

**The async device-token decline, per driver.** `VoxtralDecodeGraph` needs none:
its only construction site is `VoxtralGenerateGreedy`, the single-sequence
multimodal greedy loop, which is not reached from `runner.cpp` and has no
asynchronous device mirror. `Qwen3MoeDecodeGraph` and `DeepseekV2DecodeGraph` are
a different case and a NEW FINDING rather than an inherited one: their
registrations (`qwen3_moe_registry.cpp:107`, `deepseek_v2_registry.cpp:106`, and
`glm4_moe_lite_registry.cpp:125` which constructs the DeepSeek driver) route a
pure-decode step into a host-vector replay with NO `device_token_ids` check at
all, while `qwen3.cpp`'s `DenseDecodeGraphForward` declines for exactly that condition on a measured
battery. W3 did not add a decline: it would trade a shipped, default-ON
capability away on a measurement this stage cannot make, and the fix
`qwen3.cpp`'s own comment names is `StepDevInputs` as a seam capability, which is
W4 and is the only version that reaches these registrations at all. Filed as
[#1305](https://github.com/mudler/vllm.cpp/issues/1305), owner W4, and recorded
under `## Owed`.

**W4, the persistent device input path as a seam capability, and the two Qwen3.5
drivers. DONE 2026-08-19, [#1307](https://github.com/mudler/vllm.cpp/issues/1307).**

`vt::PersistentStepInput` (`include/vt/persistent_step_input.h`,
`src/vt/persistent_step_input.cpp`) is one capture-stable per-step device input.
It BINDS a destination the driver owns and refreshes it IN PLACE from either a
pinned host staging block or a DEVICE source. What it owns is the part every
driver re-derived: the address-stability rule as a REFUSAL rather than a comment,
the pinned staging block that makes the upload a true asynchronous DMA, and the
refreshing ARM as an observable (`last_source()`, `vt::StepInputStats`) rather
than an inference from which line the driver happened to call. What it
deliberately does NOT own is the device allocation — `Qwen3_5DecodeGraph` draws
its retained inputs from a DEDICATED `DevicePool` so they never pop a block the
captured forward's own scratch then needs, and a seam that took that over would
silently move nine drivers onto one pool (D3).

`Qwen3_5DecodeGraph` and `Qwen3_5DenseDecodeGraph` open a `vt::GraphCaptureScope`
over a per-slot `vt::BreakableGraph` in `kFull` and replay through
`BreakableGraph::Replay`, and their `PinnedStepInputs`/`StageStepInputs` staging
now runs through the capability. **Six of the nine drivers are on the seam**, and
`grep -rn 'std::getenv("VLLM_CPP_CUDAGRAPH")' src/` returns exactly ONE line,
`src/vt/breakable_graph.cpp:61` — there is finally a single switch that turns
capture off. The three single-shape drivers keep their invented switches
(`VT_V4_DECODE_GRAPH`, `VT_DFLASH_GRAPH`, `VT_LAGUNA_DECODE_GRAPH`); they are W5's.

**W4 FOUND THAT ITS OWN PREMISE WAS INCOMPLETE, and this is the stage's most
important result.** The spec said the fix `qwen3.cpp`'s decline names — read the
identifiers at REPLAY time from a stable device buffer — "already exists, in the
sibling driver, as `StepDevInputs`". **It does not.** `StepDevInputs` has no
token-id member. Its pinned sibling did: `PinnedStepInputs::token_ids` was
allocated at capture, filled by `StageStepInputs` every step, zeroed by the
poison hook, and **never uploaded and never read**. The embed runs OUTSIDE the
captured region from the HOST vector (`EmbedInto`), in every batched driver, so
**the decode graph carries no token ids to the device in ANY driver** — including
the one this row cited as having the capability.

So making the persistent device input path a seam capability was never, by
itself, going to close the decline: the one input an asynchronous mirror patches
is precisely the one the path does not carry. What W4 lands is the capability
that makes writing that destination a per-driver two-liner instead of a tenth
hand-rolled copy, and `RefreshFromDevice` is the arm it needs. The dead block is
removed and the finding is recorded rather than quietly tidied.

**Gates.** Red-first throughout. The capability's gate ran against a stub with
the declared API and none of its guarantees: 9 cases, 0 passed, 59 assertions, 32
failed, exit 1; GREEN after at 9/9, 59/59. The driver gate
(`tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp`) ran RED on the MoE
driver's five seam assertions before the migration (3 cases, 1 failed, 62
assertions, 5 failed, exit 1) and GREEN after with both drivers at 7/7 and 129
assertions, G4 reading `40 values, 0 differing` on each.

**Reachability, which the capability could easily have failed.** A unit test that
constructs the type by hand proves the class works and never that anything
reaches it. A case enters through `Qwen3_5DecodeGraph::Step` and asserts
`vt::GetStepInputStats()`: `binds=6` after the capture step, `host_refreshes >= 5`
after the replay step. Deleting the `StageStepInputs(d, s)` call site reds that
case ALONE and leaves `test_persistent_step_input` 59/59 green, which is the
distinction itself. Two limits: the case sets `VT_ASYNC_EXECUTOR=1`, because the
persistent device input path sits behind that lever (default OFF) plus the
speculative-decode arm, and that lever also turns on the 2-slot parity ring, so
slot 0 captures on step THREE.

**The async decline STANDS, and the reason is sharper than "the fix is not built
yet".** See `## Owed`.

**W5, migrate the three single-shape drivers. DONE 2026-08-19,
[#1335](https://github.com/mudler/vllm.cpp/issues/1335).** DeepSeek V4, Laguna
and DFlash. `laguna.cpp:2116-2119` carried the note that its capture class was
waiting for exactly this seam and named DeepSeek V4's driver as the sibling that
moves with it; both moved in this stage and the note now says what is actually
left device-coupled, which is the captured CHAIN and not the capture.

Each driver's `Step` opens a `vt::GraphCaptureScope` over its own
`vt::BreakableGraph` in `kFull` and replays through `BreakableGraph::Replay`.
What is gone from all three: the hand-rolled `BeginCapture`/`EndCaptureGraph`
pair, the raw `void*` handle, two hand-written destructors that released it, and
DFlash's private `try`/drain. **NINE OF NINE DRIVERS ARE ON THE SEAM**, and
`grep -rnE '\.(BeginCapture|EndCaptureGraph|ReplayGraph|DestroyGraph)\s*\('`
over `src/vllm/`, with comment lines excluded, returns NOTHING — every surviving
textual hit is prose about what was removed.

The three per-model rollback switches STAY (`VT_V4_DECODE_GRAPH`,
`VT_DFLASH_GRAPH`, `VT_LAGUNA_DECODE_GRAPH`): each is a same-binary A/B lever for
exactly one driver, not a copy of the shared one. What changes is that
`VLLM_CPP_CUDAGRAPH` reaches all three for the first time, through
`vt::GraphCaptureEnabled()` inside the scope. `gstate`/`g_state` stays too, and
it is not a duplicate of `captured()`: it is each driver's cold/warm/captured
ladder, and the seam has no notion of the eager warm-run that grows the pool so
the capture that follows can allocate nothing.

**D10 IS DISCHARGED HERE, and W5 is the first stage where it could be.**
`GraphCaptureScope` owns the set of side queues forked since the current segment
opened and joins every outstanding one before `Backend::EndCaptureGraph` — the
port of `_end_current_segment` (`:353-361`) plus the `wait_stream` hook
(`:101-153`) whose only purpose is to populate that set. W1 registered its break
point on a model that forks no auxiliary queue; W2, W3 and W4 all opened `kFull`,
which has ONE segment and therefore no segment CLOSE inside a fork window for the
rule to govern, so the machinery would have landed unexercised. W5 owns the only
driver in the tree whose fork is inside the captured region by construction
(`laguna.cpp:2572-2576` fork, `:2612` join), and that driver is the rule's
PRODUCTION CALLER through `vt::GraphNoteFork` and `vt::GraphNoteJoin`.

We need no monkey-patch to populate the set, because our fork and join are
explicit `Backend::RecordEvent` and `Backend::QueueWaitEvent` calls rather than
an implicit torch API — the model TELLS the scope. The retirement call is not
politeness: without it the scope would issue a REDUNDANT second join before every
segment close, which is exactly what the gate's CONTROL arm pins. Both hooks are
no-ops outside an active scope and make ZERO backend calls there, which is the
same pass-through guarantee `GraphBreak` gives.

**The rule is gated as a COUNTER and an ORDER, because W3 proved prose is not a
gate.** W3 measured a mode guard that stayed green at 226/226 under the exact
mutation it named. So `vt::GraphBreakStats` gains `forks_tracked` and
`forks_auto_joined`, and the load-bearing one is the second: it is 0 for a model
that joins its own fork inside the segment and non-zero exactly when the seam did
the work. And the claim is an ORDER rather than an event, because closing a
capture with an unjoined fork FAILS at `cudaStreamEndCapture` — so both ends are
asserted out of ONE backend trace
(`Begin RecordEvent QueueWaitEvent EndCaptureGraph Begin EndCaptureGraph`), for
the same reason W1 had to move break markers into the backend's own log. Test 15
of `## Tests to port` carries five arms: the rule, the CONTROL where the model
joins first and the scope must do nothing, `kFull` (the arm every migrated driver
actually takes), the inert scope, and re-registering one queue. Two mutations,
both compiled clean: deleting `JoinOutstandingForks()` reds ONLY the new case on
5 assertions at 29 of 30 still passing, and making `NoteJoin` fail to retire the
entry reds it on 8 — so neither the rule nor its control is vacuous. Green after:
30 cases, 265 assertions, exit 0.

**ONE of the three drivers is gateable without a GPU, and the reason is each
driver's own admission predicate rather than a choice.** The DFlash draft graph
admits on `VT_DFLASH_GRAPH` plus `Backend::SupportsGraphCapture()` plus
`Platform::support_static_graph_mode()` — it names neither a device type nor a
kernel registry — so the shared harness's two swapped registries reach it.
DeepSeek V4's `CanRunResidentDecode` (`deepseek_v4.cpp:1481-1487`) refuses a CPU
queue outright and refuses again unless `V4DeviceKernelsAvailable()`, the four V4
kernel families registered under `kCUDA` by `cuda_deepseek_v4.cu`. Laguna's whole
capture class is behind `#ifdef VT_MARLIN_NVFP4`, which CMake sets only for a
CUDA build on a marlin-nvfp4 architecture (`sm_12xa` and, since the Thor
bring-up, `sm_110`), so on a box with no `nvcc` the migrated region is not in any
reachable binary at all.

`tests/vllm/models/test_qwen3_dflash_decode_graph_seam.cpp` is therefore the
stage's driver gate. RED FIRST against the unmigrated driver: 3 cases, 0 passed,
16 assertions, 7 failed, exit 1, on `segments_captured`, `full_scopes` and
`replays`. GREEN after: 3 cases, 18 assertions, exit 0. G2's mutation — replacing
the scope and `Replay` with the pre-W5 raw pair, compiled clean at 79 insertions
and 29 deletions — reds ONLY that file (7 assertions) and leaves
`test_breakable_graph` 265, `test_qwen3_decode_graph_seam` 231,
`test_qwen3_moe_decode_graph_seam` 228, `test_voxtral_decode_graph_seam` 230,
`test_deepseek_v2_decode_graph_seam` 230, `test_qwen3_5_decode_graph_seam` 129
and the driver's OWN `test_dflash_propose` 31 all GREEN. That last one is the
whole argument restated as a measurement: a driver's bit-exactness suite cannot
see which capture machinery ran. G4 holds in the same file at 24 values, 0
differing, against the driver's own eager paged arm.

**What W5 does NOT deliver, named rather than implied.** G1 for all three
drivers, and G2 for DeepSeek V4 and Laguna, are OWED on hardware for the
predicate and build-flag reasons above. G5's ROCm and Tenstorrent arms stay
BLOCKED on hardware the fleet does not carry. `## Owed` carries each with its
resource.

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
(`breakable_cuda_graph.py:266-274`) registers each segment with an optional
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
another on the same queue mid-forward is legal. **That assumption is UNVERIFIED on
both runtimes and this spec claims it on neither.** For CUDA, SGLang does it on a
production path at the pinned revision (`breakable_cuda_graph.py:352-367` then
`:335-350`), which is strong evidence and is not our measurement; confirming it on a
leased GPU is W1's exit criterion, in `## Work breakdown` W0's own words. For ttnn
traces there is not even that evidence. Nothing here measures the COST of a re-begin
on either runtime, so no cost claim is made; what would settle it is a same-binary
A/B of an N-segment capture against a 1-segment capture of the same forward, on the
leased GPU that W1 already needs. **Decision: the seam is
defined against the existing `vt::Backend` vocabulary and asks nothing new of any
backend, so a backend that cannot segment reports `SupportsGraphCapture()` and
runs one segment, which is exactly today's behavior.** No backend is required to
change for W1.

**D7, a break point is a correctness boundary, not only a coverage knob.** Every
break point converts a graphed operation into a host-dispatched one, so a
mis-registered break point is invisible in a token gate and visible only as a
segment count. G3 exists for this reason. A model whose break-point set is wrong
still produces correct tokens; it produces them from more segments than it needs,
or from a graph that was never entered at all.

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

**D9, the break function's OUTPUT is a lifetime surface, and D1 and D2 do not cover
it.** D1 covers the device intermediates a SEGMENT reads and D2 covers the host
source of a captured upload. Both are about break-function INPUTS. The output is a
third case and it is the one upstream spends the most machinery on. On replay N the
eager function allocates a fresh result, so unless the seam copies that result back
into the destination the following segment baked at capture time, segment `i+1` reads
capture-time data forever while the break function writes into a buffer nobody reads.
Upstream closes it with `replay_fn` plus `_copy_output`
(`breakable_cuda_graph.py:231-235,172-201`) and keeps the destination alive with
`_weak_ref_if_tensor` (`:156-169`). **Decision: `vt::GraphBreak` states the writeback
contract at its declaration, per `## Port map` §3 — either it takes the destination
and the seam copies into it, or the break function writes in place into a persistent
model-owned buffer and returns nothing.** A `BreakableGraph` that stores the caller's
raw `fn` cannot express either, so it stores the seam's closure instead.

This defect is invisible to every gate that runs ONE replay, and it produces wrong
numbers rather than a fault, so it is in the same detection class D1 describes: a
clean `compute-sanitizer` run says nothing about it. The ported `TestCopyOutput`
cases T7 through T10 and the multi-replay case T6 in `## Tests to port` are what pin
it, and G1's more-than-one-replay requirement is what catches it at model level.

**D10, a segment cannot close while a forked queue is still capturing.** Upstream
auto-joins outstanding side streams inside `_end_current_segment` before
`capture_end()` (`breakable_cuda_graph.py:353-361`) and maintains the whole
`wait_stream` hook (`:101-153`, installed `:310`, removed `:332`) for no other
purpose. This is live here, not hypothetical: `SupportsAuxStream()` is `true` on
CUDA (`src/vt/cuda/cuda_backend.cu:200`) and two model paths fork on it — Qwen3.5's
MoE shared-expert overlap (`src/vllm/model_executor/models/qwen3_5.cpp:6254-6255`
fork, `:6384` join) and Laguna's decode graph, whose auxiliary queue is owned by the
capture class itself (`src/vllm/model_executor/models/laguna.cpp:2362-2367`,
`:2572-2576` fork, `:2612` join). W4 migrates the first and W5 the second.
**Decision: `GraphCaptureScope` owns the outstanding-fork set and joins it before
every `EndCaptureGraph`, and test 15 of `## Tests to port` gates it.** We need no
hook to populate that set, because our fork and join are explicit
`Backend::RecordEvent` and `Backend::QueueWaitEvent` calls rather than an implicit
torch API — which makes this cheaper for us than for SGLang, not harder. A break
point registered inside an unjoined fork window without this rule fails at
`EndCaptureGraph`, which is the one failure mode in this spec that is LOUD.

## Now

`ACTIVE`. W0 (spike), W1 (the seam, its ported unit gate, and one registered
break point), W2 (`Qwen3DenseDecodeGraph`), W3 (`Qwen3MoeDecodeGraph`,
`VoxtralDecodeGraph`, `DeepseekV2DecodeGraph`), W4 (the persistent device input
path as `vt::PersistentStepInput`, plus `Qwen3_5DecodeGraph` and
`Qwen3_5DenseDecodeGraph`) and W5 (the three single-shape drivers, plus D10) have
landed. **NINE OF THE NINE DRIVERS ARE ON THE SEAM**, the migration is complete,
and one `std::getenv("VLLM_CPP_CUDAGRAPH")` remains in `src/` — the seam's own.
`grep -rnE '\.(BeginCapture|EndCaptureGraph|ReplayGraph|DestroyGraph)\s*\('`
over `src/vllm/`, with comment lines excluded, returns NOTHING. **Only W6
remains**, and `## Work breakdown` states it. Owner:
`.agents/claims/CLAIM-ENG-CUDAGRAPH-BREAK-W5.md`.

**D10, the auxiliary-stream auto-join, is discharged and is REACHED.** Every
prior stage opened `kFull`, which has one segment and so no between-segments
window, so the rule could not be exercised and untested machinery was not landed
for it. W5 owns `laguna.cpp`, the only driver whose fork is inside the captured
region by construction, and that driver is the production caller. Gated as a
counter and an ORDER out of one backend trace, with two mutations proving neither
the rule nor its control arm is vacuous.

**W4 corrected a premise this spec had asserted three times.** The decode graph
carries NO token ids to the device in any driver, `StepDevInputs` included, so
the `qwen3.cpp` async decline was never one refactor away from removable. It
STANDS, and `## Owed` now names what is actually missing instead of naming a
stage.

Every migrated step opens its scope in `kFull`, mirroring vLLM's decode arm. The
PIECEWISE arm still has no production driver and `## Owed` names what has to be
true before one exists.

**G1 is MET for five of the NINE migrated drivers and is owed for four.**
Bit-exactness against the eager arm over the capture step plus THREE consecutive
replays, on `thor:gpu0` through an `rc` lease: **5 cases, 2066 assertions, 0
differing**, the two Qwen3.5 drivers added by W4. W2's `Qwen3DenseDecodeGraph`
is still covered by argument rather than by measurement, which `## Gates` G1
says out loud, and W5's three single-shape drivers join it there for reasons
`## Owed` names per driver. This is the only gate in this row a CPU harness could
never have answered.

W1's exit criterion — that CUDA permits `cudaStreamEndCapture` followed by
`cudaStreamBeginCapture` mid-forward on our stream configuration — is
CONFIRMED on a leased GPU and is no longer an open question for any later stage.

## Owed

Each item names the stage that owns it. Nothing here is claimed by W1.

- ~~**The seam is not yet ENTERED from a production step.**~~ RETIRED by **W2**
  ([#1261](https://github.com/mudler/vllm.cpp/issues/1261)).
  `Qwen3DenseDecodeGraph::Step` opens a `vt::GraphCaptureScope` over a per-slot
  `vt::BreakableGraph` and replays through `BreakableGraph::Replay`, and
  `tests/vllm/models/test_qwen3_decode_graph_seam.cpp` holds it through the
  seam's own counters. This was the staged slice AGENTS.md allows; it is closed
  rather than carried.
- ~~**The auxiliary-stream auto-join before every segment close** (D10, the port
  of `breakable_cuda_graph.py:353-361`).~~ RETIRED by **W5**
  ([#1335](https://github.com/mudler/vllm.cpp/issues/1335)).
  `GraphCaptureScope` owns the outstanding-fork set and joins every entry before
  `Backend::EndCaptureGraph`; `vt::GraphNoteFork` and `vt::GraphNoteJoin` are the
  registration, and `laguna.cpp:2572-2576,2612` — the only fork inside a captured
  region by construction — is the production caller. W1 through W4 could not
  discharge it and the reason was the MODE rather than the effort: every stage
  before this one opened `kFull`, which has one segment and therefore no segment
  CLOSE inside a fork window for the rule to govern, so landing the machinery
  would have landed it unexercised. Gated as test 15 of `## Tests to port` with
  five arms and proven non-vacuous by two mutations. **One residual, and it is
  named rather than folded in:** the seam joins a fork the model REGISTERED. A
  model that forks without calling `GraphNoteFork` is exactly as exposed as
  before, and no checker can see that, because the fork is an ordinary pair of
  backend calls. Laguna is the only such site today; the next one is the
  obligation of whoever adds it, and the `SupportsAuxStream()` capability is
  where a reader lands.
- ~~**The capture-failure drain as a GATED case** (test 13).~~ DELIVERED in W1,
  and the record it replaces was wrong twice over. The destructor did NOT already
  behave: its `catch` guarded a throwing `EndCaptureGraph` alone, and an
  exception from a break function or from ordinary model code left a partial
  capture reporting `captured() == true`. The behaviour and all three gated arms
  landed together; see `## Tests to port` test 13.
- **The non-capturing arm on the OTHER capture backends** (G5). The inert path is
  gated here through a recording backend reporting `SupportsGraphCapture()`
  false; ROCm (`rocm_backend.hip:248`) and Tenstorrent
  (`tenstorrent_backend.cpp:75-81`) are not exercised, and D6 records that
  segmenting a ttnn mesh trace is UNVERIFIED on that runtime. **W3 did NOT
  discharge this and says so rather than letting the owner field imply it did.**
  The lease W3 obtained was `thor:gpu0`, which is CUDA; the fleet
  (`rc devices`) carries no ROCm device and no Tenstorrent device, so this is
  BLOCKED on hardware rather than unattempted. What W3 can say is what it
  measured: the seam's CUDA arm now runs on TWO architectures rather than one.

  **THAT SENTENCE NAMED THE WRONG DEVICE AND THE WRONG ARCHITECTURE, and W5
  corrected it ([#1361](https://github.com/mudler/vllm.cpp/issues/1361)).** It
  read "sm_110 here and sm_121a on GB10 for the W1 exit criterion", while this
  same file records at `## Work breakdown` W1 that the exit criterion was
  measured on `orin:gpu0`, driver `12060` — a Jetson AGX Orin, which is neither
  a GB10 nor `sm_121a`. The two architectures are real and the claim survives;
  the attribution did not. What was measured, and where: the W1 exit criterion
  (`cudaStreamEndCapture` then `cudaStreamBeginCapture` mid-forward with eager
  work between) on `orin:gpu0`, and G1 plus the unit suite on `thor:gpu0` at
  sm_110 for W3, W4 and W5. The exit criterion has NOT been re-measured on
  `thor`, and the reason is structural rather than an omission: every migrated
  driver opens `kFull`, so nothing in the tree re-begins a capture mid-forward,
  and G1 exercises capture and replay rather than the re-begin. **`sm_121a` on
  GB10 is OWED, not done** — W5 could not take it because `dgx:gpu0` was held by
  another session for that stage's whole window. This is the shape where a
  number quoted often starts being treated as measured.
  **W5 did NOT discharge it either, and the fleet answer is unchanged**:
  `rc devices` lists `dgx:gpu0`, `orin:gpu0` and `thor:gpu0`, all NVIDIA. This is
  the second stage to inherit the item and find the same wall, which is the
  signal that it is not a stage-sized problem: it needs a ROCm or Tenstorrent
  device on the fleet, and no ordering of the remaining work produces one. Owner:
  row **`ENG-CUDAGRAPH-BREAK`**, discharged by whichever stage first has the
  hardware. The Tenstorrent half additionally has a named entry point when that
  day comes — `qwen3.cpp`'s `VT_TT_RECAPTURE_EVERY` branch is the only place a
  ttnn mesh trace meets this seam.

- **G1 and G2 for the three SINGLE-SHAPE drivers** (W5,
  [#1335](https://github.com/mudler/vllm.cpp/issues/1335)). Each has its own
  reason and none of them is effort, so they are listed per driver rather than as
  one line:

  1. **DFlash** has G2 and G4 on CPU
     (`tests/vllm/models/test_qwen3_dflash_decode_graph_seam.cpp`, red-first,
     3 cases / 18 assertions) and owes only G1, which every CPU-gated driver in
     this row owes for the same reason: a CPU "replay" recomputes nothing, so a
     CPU harness cannot hold that a REPLAYED segment reproduces the eager
     forward.
  2. **DeepSeek V4** owes G1 AND G2. `CanRunResidentDecode`
     (`deepseek_v4.cpp:1481-1487`) refuses `device.type == kCPU` outright and
     refuses again unless `V4DeviceKernelsAvailable()` — the four V4 kernel
     families registered under `kCUDA` by `cuda_deepseek_v4.cu`. The shared
     harness swaps a backend and a platform; it cannot manufacture a device type,
     a CUDA op registry and a whole vt kernel layer, and a stub that did would
     measure the stub. The migration is covered by CONSTRUCTION — same seam, same
     mode, same call shape as the seven gated drivers — which is an argument and
     not a gate, and this entry exists so nobody reads it as one.
  3. **Laguna** owes G1 and G2 for a build reason rather than a predicate one.
     `LagunaGraph` is inside `#ifdef VT_MARLIN_NVFP4`, which CMake sets only for
     a CUDA build on a marlin-nvfp4 architecture, so on a host with no `nvcc` the
     migrated region is in no reachable binary. `laguna.cpp` compiling clean in
     the CPU configuration verifies the FILE and not the region, and
     `test_laguna_scaffold` (167 assertions) and `test_laguna_nvfp4_loader` (63)
     staying green says nothing about the capture at all.

  **ONE HALF OF 2 AND 3 IS NOW CLOSED and the rest is not, so the split is worth
  stating.** W5 measured on `thor:gpu0` that both migrated regions are actually
  COMPILED, by mutating each one and requiring the object build to fail — see
  `## Gates` G1. So "the code might not even be built" is no longer part of what
  these two owe. What they still owe is behavioural: that the routing reaches the
  seam (G2) and that a REPLAYED segment reproduces the eager forward (G1), and
  both need each model's own device kernels — V4's four `kCUDA`-registered
  families, Laguna's NVFP4 Marlin arm — rather than only a compiler.

- **The W1 exit criterion on a THIRD architecture, `sm_121a` on GB10.** Filed as
  its own item because #1361 found the record already claiming it. What has
  actually been measured: the criterion on `orin:gpu0`, and G1 plus the unit
  suite on `thor:gpu0` at sm_110. `dgx:gpu0` is the only `sm_121a` device on the
  fleet and it was held by another session for the whole of W5's window
  (`rc devices`: `busy`, past one hour). **It cannot be discharged by re-running
  G1 there**, and that is the part worth writing down: every migrated driver
  opens `kFull`, so nothing in this tree re-begins a capture mid-forward, and G1
  therefore exercises capture and replay rather than the re-begin the criterion
  is about. Discharging it needs the W1 probe itself run on GB10, or the first
  PIECEWISE production driver — which is W6. Owner: row
  **`ENG-CUDAGRAPH-BREAK`**.

  What settles all three: a CUDA build on a marlin-nvfp4 architecture — the same
  `thor:gpu0` (sm_110) shape W3 and W4 used — which compiles the Laguna and V4
  regions and can extend
  `tests/vllm/models/test_decode_graph_seam_g1_cuda.cpp` with a DFlash case over
  the synthetic draft model this stage's CPU gate already builds. Owner: row
  **`ENG-CUDAGRAPH-BREAK`**, the stage that gets that window.
- ~~**G1, bit-exactness against eager on a real GPU over MORE than one
  replay.**~~ RETIRED by **W3**
  ([#1291](https://github.com/mudler/vllm.cpp/issues/1291)).
  `tests/vllm/models/test_decode_graph_seam_g1_cuda.cpp` runs each migrated
  driver COLD, CAPTURE and THREE consecutive replays against the driver's own
  eager arm, selected by `max_num_reqs == 0` so both arms are the same binary on
  the same device rather than two builds. Measured on `thor:gpu0` through an
  `rc` lease: 3 cases, 1600 assertions, exit 0, `5 steps x 100 logits, 0
  differing, 4 replays` for each of the three W3 drivers. See `## Gates` G1 for
  the build and device provenance, and for why the ASSERTION COUNT and not the
  status line is what carries the claim. **Two limits, named rather than
  claimed away.** The models are the synthetic tiny ones the CPU forward gates
  already use, not Qwen3-0.6B or Qwen3-4B: they exercise the real CUDA kernels
  and the real capture and replay, and they do not exercise a checkpoint's
  weights or a long context. And W2's `Qwen3DenseDecodeGraph` is not one of the
  three cases; it shares the seam and the shape, which is an argument and not a
  measurement.
- **The PIECEWISE arm has no production driver, and the break closure is why.**
  W2's driver opens its scope in `kFull`, so the registered break point takes
  the pass-through arm and the capture is one segment. Opening it `kPiecewise`
  today would register a replay closure that captured `RunLayer`'s frame by
  reference (`dhn`, `si`, `meta`, `T` — every one a per-step local), which is
  lifetime rule 2 at the `GraphBreak` declaration and reads a returned frame on
  replay N. The fix is persistent, driver-owned storage for the layer's inputs,
  which is the capability `StepDevInputs` already is. **W4 landed the storage
  PRIMITIVE and not the arm.** `vt::PersistentStepInput` is persistent,
  driver-owned, capture-stable storage, so what is missing is no longer a
  primitive: it is a driver that holds its layer's inputs there and a break
  closure that reads them instead of `RunLayer`'s frame. Owner: **W6**, which
  cannot move the eligibility predicate before that exists.
- **The async device-token DECLINE at `qwen3.cpp`'s `DenseDecodeGraphForward` STANDS after W4, and the
  reason changed.** W4 owned the decision and did not remove it. What W4
  established, by reading the tree rather than by inheriting the record:

  1. **No driver has a device token-id destination, `StepDevInputs` included.**
     `StepDevInputs` has no token-id member; its pinned sibling
     `PinnedStepInputs::token_ids` was allocated, filled every step and NEVER
     uploaded or read. The embed runs OUTSIDE the captured region from the HOST
     vector (`EmbedInto`) in every batched driver. So the fix the decline's own
     comment names — read the identifiers at REPLAY time from a stable device
     buffer — did not exist anywhere in the tree, and this spec asserted three
     times that it did. `vt::PersistentStepInput::RefreshFromDevice` is the arm
     that fix needs; the destination itself is still owed.
  2. **The decline's recorded CAUSE does not survive contact with the tree.**
     `338cbbfd1` (#323) records "the registry-level `DeviceTokenIdsScope`
     (`60e71a0e`) did not close it: this path returns BEFORE the eager forward
     ever runs". At that commit's own parent the scope was constructed at
     `qwen3_dense.cpp:96`, BEFORE `DenseDecodeGraphForward` at `:103`, and
     `EmbedInto` consumed it through `ApplyDeviceTokenIdsOverride` on all three
     of the driver's arms (`qwen3.cpp:610,621,644` @ `338cbbfd1^`). The override
     was therefore live on the graph path. The measured failure is real; the
     mechanism recorded beside it is not the one that produced it, and nobody
     has since identified what did.
  3. **Therefore the decline could not be removed on this row's evidence.** A
     mitigation whose failure mode is unexplained is not retired by a refactor
     that plausibly addresses the explanation nobody has confirmed. Removing it
     needs the battery, red first with the decline deleted.

  **The battery was NOT run, and this is the plain statement rather than an
  implication.** `tests/parity/test_qwen3_dense_async_serving.cpp` needs a GPU
  **and** the real Qwen3-0.6B/4B snapshots under `~/.cache/huggingface/hub`, and
  its own header restricts it to `dgx.casa`. `dgx:gpu0` was held by another
  session's job for the whole of W4's window (`rc devices`: `busy`, 1h31m and
  climbing); W4's lease was `thor:gpu0`, which has no such checkpoint. Both arms
  of the decline stay gated on CPU at
  `tests/vllm/models/test_qwen3_decode_graph_seam.cpp`. Owner: row
  **`ENG-CUDAGRAPH-BREAK`**, the stage that gets a `dgx` window WITH the
  checkpoints; it owes two runs, not one — the battery as it stands, and the
  battery with the decline deleted, because only the second can fail.
  [#1179](https://github.com/mudler/vllm.cpp/issues/1179)
  and [#323](https://github.com/mudler/vllm.cpp/issues/323) track it.

  **The DECLINE'S OWN COMMENT was left standing verbatim by the head this
  entry landed on, and the fresh review caught it.** It still asserted the
  falsified `DeviceTokenIdsScope` mechanism, still named `StepDevInputs` as the
  fix at the stale anchor `qwen3_5.cpp:3894`, and still read `Owner: W4` — the
  one place a reader lands when they open the decline, disagreeing with the
  spec, `docs/STATUS.md`, the engine-matrix row, the new header and the pull
  request. Repaired in the same pull request that made it stale, per AGENTS.md
  "a record edit rides in the pull request whose change made the record stale".
  The decline itself is UNCHANGED.

  **`RefreshFromDevice` therefore lands with NO production caller, and that is
  the staged slice AGENTS.md admits rather than an oversight.** `grep -rn
  RefreshFromDevice src/ include/` returns the definition alone;
  `last_source()` and `StepInputSource` have no production reader either. The
  arm is unreachable until a driver holds a device token-id destination, which
  is what this entry owes, so writing a caller now would be the tenth
  hand-rolled copy this row exists to remove. Named in the commit body, in the
  pull-request body, at the declaration in
  `include/vt/persistent_step_input.h`, and here. Owner: row
  **`ENG-CUDAGRAPH-BREAK`**, same stage and same two issues as above. The HOST
  arm is reached, and its reach is bounded rather than claimed whole:
  `StageStepInputs` (`qwen3_5.cpp`) routes both Qwen3.5 decode drivers through
  it when `dbuf` is set — `VT_ASYNC_EXECUTOR=1`, default OFF, or a speculative
  verify step — and `tests/vllm/models/test_qwen3_5_decode_graph_seam.cpp`
  holds that call site with a mutation that reds only the reachability case.
- **The async battery has still NOT been re-run, by W2, W3 or W4.** The
  obstacle is the same one and it is a resource rather than an oversight: the
  battery needs a GPU **and** a real checkpoint, W4's lease was `thor:gpu0`
  (clean clone, synthetic weights, no HuggingFace cache), and `dgx:gpu0` — the
  only box its header admits — was held by another session for W4's whole
  window. The decline entry above records what the battery now owes, which is
  two runs and not one.
- **Three decode-graph registrations route an ASYNCHRONOUS step into a
  host-vector replay with NO `device_token_ids` decline**
  ([#1305](https://github.com/mudler/vllm.cpp/issues/1305), found by W3, not
  caused by it — the shape is present at W3's base commit `5d9fe332c`).
  `qwen3_moe_registry.cpp:107`, `deepseek_v2_registry.cpp:106` and
  `glm4_moe_lite_registry.cpp:125` admit a pure-decode step to a driver that
  replays against persistent HOST vectors, while `qwen3.cpp`'s `DenseDecodeGraphForward` declines for
  exactly that condition on a measured battery whose own comment calls the
  hazard "latent for EVERY classic-dense model, since the graph is default-ON".
  W3 did not add a decline: it would trade a shipped, default-ON capability away
  on a measurement this stage cannot make, and the fix `qwen3.cpp` names is
  `StepDevInputs` as a seam capability. `VoxtralDecodeGraph` is unaffected — its
  only construction site is `VoxtralGenerateGreedy`, which the runner does not
  reach.

  **W4's disposition: #1305 STAYS OPEN and W4 did not add a decline either, for
  a reason that is now stronger than W3's.** W3 declined to trade a shipped
  default-ON capability for an unmeasured hazard. W4 adds the finding that makes
  the trade worse: the hazard #1305 describes is the SAME one `qwen3.cpp`
  mitigates, and W4 established that the mitigation's own recorded cause is
  falsified and its real cause unidentified (see the decline entry above).
  Copying a mitigation whose mechanism nobody can name into three more
  registrations would spread an unexplained behaviour, not close a defect. What
  #1305 needs is the same `dgx` window with checkpoints that the decline needs:
  run the battery shape against `Qwen3MoeDecodeGraph` and `DeepseekV2DecodeGraph`
  and find out whether they degenerate at depth 2 at all. Owner: row
  **`ENG-CUDAGRAPH-BREAK`**, the stage that gets that window.
- **An exception CAUGHT INSIDE the capture scope leaves a partial capture the
  drain cannot see.** The `uncaught_exceptions()` comparison in
  `~GraphCaptureScope` detects an exception that is PROPAGATING at scope exit. A
  break function that throws and is caught between two break points inside the
  scope leaves `segment_open_ == false`, so the remainder of the forward is
  never captured, and nothing is unwinding at scope exit, so the drain does not
  fire and `captured()` stays true over a forward that is missing its tail. The
  header and this spec state the guarantee only for the propagating case, so
  this is a RESIDUAL rather than a false claim — but a driver that wraps a break
  in a `try` gets silently wrong numerics on replay, and W2 is the first stage
  that opens a scope from a driver. **W2 did not close it and did not need to:**
  its scope is `kFull`, so it has ONE segment and no `try` anywhere inside it,
  and a `kFull` capture has no between-segments window for a caught exception to
  truncate. The residual becomes live for the first driver that captures
  PIECEWISE. **W4 did not close it either, and for the identical reason:** both
  drivers it migrated open `kFull`. Owner: **W6**, with the piecewise arm it
  unblocks.
  What W2 DID close is the adjacent hole the fresh review found, and it is
  recorded here because the two are easy to confuse: the drain leaves a FAILED
  capture reporting exactly what an INERT scope reports, and the first W2 head
  read that one bit and returned the uncomputed layer output as the step's
  logits. `vt::BreakableGraph::capture_failed()` / `capture_error()` separate the
  two states, the driver rethrows, and
  `tests/vt/test_breakable_graph.cpp` test 13d plus
  `tests/vllm/models/test_qwen3_decode_graph_seam.cpp` gate both.
- **The reuse hazard the seam must close** (D1): making the intermediates a
  segment reads unavailable to the `DevicePool` free list for the life of the
  `BreakableGraph`. W1 states the lifetime rules at the `GraphBreak` declaration
  and enforces the OUTPUT half through the writeback contract; the INPUT half is
  a pool change with its own argument. **W2 did not close it and did not need
  to:** the hazard is the window BETWEEN two segments, which a `kFull` capture
  does not have, so it becomes live for the first driver that captures
  piecewise. **W4 did not close it either, and for the identical reason:** both
  drivers it migrated open `kFull`. Owner: **W6**, with the piecewise arm it
  unblocks.

## Outcome

**What the first probe measured, and what it did not.** The exit-criterion probe
refused twice before it held, and neither refusal was CUDA's. Bound through
`dlsym` on the BARE symbol name, `libcuda` hands back the LEGACY v1 entry points,
which are not capture-aware: `cuMemcpyDtoDAsync` (v1) returned
`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` inside a legal capture, and
`cuStreamBeginCapture` (v1) takes NO capture-mode argument, so the probe believed
it was testing the thread-local mode while testing the global one. Preferring
`_v3` blindly then bound `cuCtxCreate_v3`, which takes two extra parameters, and
context creation failed with `CUDA_ERROR_INVALID_DEVICE`. Both readings presented
as a verdict about the design. Bound by EXACT versioned name the probe passes,
and the rule that survives is the general one: an instrument that can fail toward
a code verdict must assert its own precondition first.

**Why the destination form, and not the in-place one, at the qwen3 site.**
`AttnBlock` returns a fresh `DBuf` from the device pool on every call, so the
in-place form's contract — write into a persistent buffer no replay reallocates —
is one the site cannot meet. Registering it in the in-place form would have
compiled, run, and been wrong on replay in exactly the way D9 describes. The
destination is `std::optional<DBuf>` with a `CopyOutput` overload in
`vllm::dense_attn`, found by argument-dependent lookup.

**Why the writeback is a customization point rather than a fixed signature.**
Upstream's `_copy_output` (`:172-201`) reaches an object's fields through
`__dict__`; C++ has no reflection. Overloading on the output type reproduces the
three branches upstream has — in-place for a tensor, field-wise for a keyed set,
in-place-or-assign for a struct — and reproduces the fourth, the non-copyable
fallback (`:201`), as a type with no overload, for which the seam fabricates no
copy. T10 asserts the ABSENCE of that writeback deliberately: it is the warning a
future author needs, and asserting only its presence would have left the case
mute.

**Why the counters exist.** G3 is not decoration. Without a segment count there
is no way to tell a two-segment capture from a fully eager step, and the second
G2 case uses `break_points_reached` to prove the model REACHES the seam even on
the pass-through arm, which is the only observable that distinguishes a
registered break point from a deleted one while no driver opens a scope.

---

Revised 2026-08-18 after a fresh review returned `FAIL`. What changed: the upstream
test suite is ported rather than declared absent; `## Port map` §3 states the output
writeback contract that the raw-`fn` description had dropped (D9); the auxiliary
stream fork-join rule is stated and gated (D10); four enumerations in
`## Our baseline` and one `grep` count were re-derived and corrected against printed
instruments; D6 now defers to W1's exit criterion instead of asserting it; and the
row is recorded as coverage AND correctness per #1179.

Revised 2026-08-18 again, when W1 landed (#1192): `## Work breakdown` W1 records
the confirmed exit criterion and what shipped, `## Now` moves the row to `ACTIVE`,
and `## Owed` and `## Outcome` are new. The framing is unchanged and is not
negotiable — coverage and correctness, never speed. No throughput gate is
declared and none was measured.

Revised 2026-08-18 a fifth time, after a fresh review of the W2 head returned
`FAIL` (one HIGH, one MEDIUM, two LOW). What changed. **The HIGH:** the driver
read `captured() == false` as "the scope was inert" and returned the layer
output as the step's logits, which on CUDA is uncomputed device memory, because
`~GraphCaptureScope` must swallow a throwing `EndCaptureGraph` and the container
had no way to say WHICH of the two states it was in. `BreakableGraph` now
records `capture_failed()` and `capture_error()`, the driver rethrows the
runtime's own exception, and both the seam-level and driver-level arms are gated
(`tests/vt/test_breakable_graph.cpp` test 13d;
`tests/vllm/models/test_qwen3_decode_graph_seam.cpp`). **The MEDIUM:** the
counterfactual carrying the W2 record was stated as fact, never run, and is
false; `## Outcome` now carries the measured mutation D and separates the two
claims it had merged. **The LOWs:** the caught-inside-scope residual is
reassigned from W2 to W4 with W2's disposition recorded, and a test justification
in the G2 file that could not apply to a `kFull` container is corrected.

**Disclosure, because a reader would look for it here.** The three
`vt::GraphCaptureMode` unit cases in `tests/vt/test_breakable_graph.cpp` were
written AFTER the implementation they cover, not red-first. The fresh review
mutated both load-bearing halves (the `splits()` gate and the `AppendBreak`
refusal) and each was detected, so no test-quality finding follows — but the
order is recorded rather than left to be inferred from a green suite. The
capture-failure gate added in the repair WAS red-first: it failed with
`CHECK_THROWS_AS(...) did NOT throw at all!` before the driver change.

Revised 2026-08-18 a fourth time, when W2 landed (#1261): `## Work breakdown`
W2 records what shipped, `## Now` moves the owner to the W2 claim, `## Owed`
retires the staged slice W1 left and files four new items (the piecewise arm's
lifetime blocker, the standing async decline, the un-re-run async battery, and
G1 still owed on a real GPU), and `## Outcome` records the mode finding. The
framing is unchanged and is not negotiable — coverage and correctness, never
speed. No throughput gate is declared and none was measured; `vt::GraphCaptureMode`
exists to mirror vLLM's decode arm, not to make anything faster.

Revised 2026-08-18 a third time, after a fresh review of the W1 head returned
`FAIL`. What changed: the destination-carrying break point takes a
`vt::BreakSlot` the seam owns instead of a caller reference it cannot outlive
(the rule the header stated and the only site using it broke); the interleaved
replay, the `VLLM_CPP_CUDAGRAPH=0` kill switch, the capture-failure drain, the
`segment == break + 1` invariant, the nesting refusal and the writeback-branch
selection are each GATED rather than described; the recording backend simulates
the graph so upstream's arithmetic chains and post-replay assertions port
literally; and the exit-criterion probe is committed with its recipe and hashes.
The framing is unchanged and is not negotiable — coverage and correctness, never
speed. No throughput gate is declared and none was measured.

**The probe is COMMITTED, because the measurement that produced two false
refusals is the one that most needs a re-runnable form.** Both sources are in the
tree: `scripts/probe_cudagraph_rebegin.c` (the `dlsym` driver-API build, which
needs no CUDA toolkit and no headers, and is the one the exit criterion was
answered with) and `scripts/probe_cudagraph_rebegin.cu` (the runtime-API build,
which needs `nvcc`). `.agents/benchmark-record.md` carries their sha256 sums and
the exact recipe. **What was NOT retained is the leased run's own driver script
and its raw stdout**, so the verdict recorded above is reproducible only by
re-running the committed source under the recorded recipe on a leased GPU — which
is what a later claimant has to do rather than cite this paragraph.

**W2's finding, and it is the one a later stage most needs.** The seam W1 built
is the PIECEWISE mode, and W1's records read as though migrating a driver onto
the seam and splitting its capture were one act. They are two, and the primary
oracle keeps them apart: `splitting_ops` says WHERE a graph may be split,
`cudagraph_mode` says WHETHER this batch's graph is split at all, and for a
decode batch vLLM's own default answers "not at all".

**What a naive piecewise migration actually does, CORRECTED and MEASURED.** W2's
first head asserted that opening the scope `kPiecewise` and changing nothing else
"would have been bit-exact and would have passed every gate in this tree". That
was reasoned, never run, and it is FALSE — the fresh review ran it and so did the
repair. Mutation D, one token (`kFull` -> `kPiecewise`) at
`src/vllm/model_executor/models/qwen3.cpp`, compiles clean and dies on the FIRST
replay, which the capture step itself performs:

```
vt: matmul_bt: rank-2 tensors required   (src/vt/ops.cpp:158)
  vllm::dense_attn::AttnBlock(...)
  vt::GraphBreak<RunLayer::{lambda#1}, std::optional<DBuf>>::{lambda#1}::operator()()
  vt::BreakableGraph::Replay(vt::Queue&)
  vllm::Qwen3DenseDecodeGraph::Step(...)
```

`test_qwen3_decode_graph_seam` goes 4/4 -> 1/4 with three cases THROWN, G2 having
passed only its four cold-step assertions before the capture step's own replay
faulted. So the naive migration is a USE-AFTER-SCOPE, loud, on the first step
that captures — not a quiet degradation. The cause is the very next item below,
which the same section already identified: W1's break closure captures
`RunLayer`'s frame by reference. The two statements contradicted each other two
paragraphs apart, and the false one UNDERSTATED the hazard.

**The part that was true, and is worth keeping separately.** HAD the closure been
replay-safe, the piecewise split would have converted a shipped, default-ON,
fully graphed decode step into twenty-eight eager attention calls per step, and
no gate here could have seen it, because a token gate cannot see a segment count.
Nothing in this row's framing — coverage and correctness, never speed — licenses
that. That is why `kFull` is a MIRRORING requirement and not a preference. But it
is a conditional, not the observed behaviour: today the same change faults first.
D7 therefore stands as a risk with one measured half, not as a lived case.

**And the break point W1 registered is not replay-safe, which is what mutation D
exposes.** Its closure captures `RunLayer`'s frame by reference. W1's G2 gate
opens a scope, runs the forward and counts segments; it never calls `Replay`, so
a closure that reads a returned frame on replay N was invisible to it. W2 found
this by READING the closure's captures at the site rather than by trusting the
gate that was green; mutation D is the run that confirms it, and it is recorded
here so a later stage does not have to re-derive either half.

**The finding worth keeping at the top level.** The seam's most-documented rule —
that the destination must outlive the graph — was broken by the ONLY site that
used it, in the same change that wrote the paragraph explaining it. A rule stated
at a declaration and re-derived by every caller is not a mechanism. Where a
contract can be encoded in a type, encode it: W2 through W5 add nine more callers
and none of them should have to read that paragraph to be correct.

