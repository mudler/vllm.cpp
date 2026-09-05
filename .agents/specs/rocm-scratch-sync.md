# BACKEND-ROCM-SCRATCH-SYNC — hold the lock across the scratch entry, and give the hipBLASLt workspace the pattern its sibling already has

Issue: [#2712](https://github.com/mudler/vllm.cpp/issues/2712).
Base: `ca07f6e94`.
Related: [#2713](https://github.com/mudler/vllm.cpp/issues/2713) touches two of
the same files on its own branch and shares nothing else.

## Scope

Two ROCm scratch allocators. Both still present at the base commit, both read
there rather than from the issue text.

- `EnsureQuantScratch` / `ScratchFor`, `src/vt/rocm/rocm_grouped_gemm.hip:621-637`.
- `LtWorkspace`, `src/vt/rocm/rocm_matmul_hipblaslt.hip:298-311`, consumed at
  `:394`.

IN SCOPE: the synchronisation and the lifetime of those two pools, the portable
bookkeeping both then share, and the unit test that holds it.

OUT OF SCOPE:

- The identical `ScratchFor` / `EnsureScratch` pair on the CUDA side
  (`src/vt/cuda/cuda_quant_dot.cu:1806-1822`). It has the same defect, and it is
  listed under `## Owed` with its own issue rather than repaired here, because
  changing it obliges a CUDA gate this row cannot run.
- `GetLt()`'s single `thread_local` handle across a two-device hop. Also `## Owed`.
- Everything #2713 changes.

## First deliverable: is this latent, or already live?

The issue names this as the first thing worth measuring, so it is answered
before anything is designed.

**In the text-generation path: one host thread. NOT live.** The forward runs on
the single engine busy-loop thread created at
`src/vllm/v1/engine/core_client.cpp:25`. The expert-stream seam states the same
assumption as its own precondition — "the flag is per-thread because a decode
step runs on one host thread"
(`include/vllm/model_executor/expert_stream_seam.h:193-194`), backed by a
`static thread_local` at `src/vllm/model_executor/expert_stream_seam.cpp:376`.
The other production drivers of a forward are serialised: embeddings under
`embed_mutex` (`src/vllm/entrypoints/openai/server_main.cpp:1160,1167`), the
legacy sync engine under `legacy_engine_mutex`
(`src/vllm/entrypoints/openai/api_server.cpp:102,272-274`).

**Process-wide, a second thread can run a device forward at the same time.**
`POST /v1/videos` spawns a job thread at
`src/vllm/entrypoints/openai/api_server.cpp:670`, and `POST /v1/audio/speech`
runs `synthesizer_` inline on the cpp-httplib worker thread
(`ApiServer::handle_audio_speech`, `api_server.cpp:612`). On a ROCm build those runners resolve to `kROCM`, so two
host threads can be inside `vt::MatmulBT` at once.

**That is still not a race on this entry, because they do not share a stream.**
Each engine owns its own `vt::Queue`, so two threads key two different entries
of `ScratchFor`'s map, and the map itself is locked. `unordered_map` is
node-based, so the reference stays valid. So the finding is: **latent, not live**
— which is what #2712 filed it as, now measured rather than assumed.

**One shape would make it live, and only a coincidence prevents it.**
`Music3SpeechEngine` holds one `vt::Queue` created once
(`src/vllm/model_executor/models/minimax_music3_speech.cpp:553-560`) and
`Synthesize` takes no lock — the file contains zero `mutex` and zero
`lock_guard` occurrences — and `handle_audio_speech` adds none. Two concurrent
speech requests therefore enter one stream on two threads. Whether that reaches
`EnsureQuantScratch` today depends on whether that model's device arm hands a
block-quant weight to `vt::MatmulBT`; this row did not establish that it does
not, and nothing in the tree enforces it either way. The repair does not depend
on the answer.

## What actually breaks

`ScratchFor` returns a reference and the `lock_guard` dies with the call. The
read of `sc.bytes`, the `hipMallocAsync` **directly into `sc.buf`**, and the
write of `sc.bytes` are all unlocked. Two callers on one stream write both
fields, and the fields can end up **mismatched**: `bytes` from the larger
request, `buf` from the smaller allocation. Every later caller then passes that
`bytes` check and receives a block smaller than it asked for. The corruption is
not the torn read; it is the surviving pair.

## Design

Mirror the shape `src/vt/cuda/graph_safe_scratch.h` already states for this
tree: the portable, CPU-unit-testable bookkeeping lives in a header, and the
device allocation stays at the call site. The new header
`src/vt/grow_only_stream_scratch.h` is free of HIP and CUDA headers, so it is
compiled and unit-tested in every build including CPU-only CI — the same reason
`include/vt/rocm/rocm_arch.h` is HIP-free.

`Ensure(stream, need, alloc)` holds one lock across the capacity read, the
allocation and the publish. Growth is `O(log(max/min))` per stream over a
process, so serialising it across streams costs nothing measurable, and
correctness is not traded for it. The allocation callable may throw; the entry
is then left exactly as it was, which the current code does not guarantee.

The old block is **retired, never freed** — the pool records it and keeps it
resident. That is the discipline the existing comment at
`rocm_grouped_gemm.hip:606-616` claims ("Retire, never free") while the code
below it merely drops the pointer, so the retire becomes real and countable
instead of a silent leak.

`LtWorkspace` **can** take the same shape, and the answer needs one
qualification. It is `thread_local` today, so it has no cross-thread race; its
defects are the `hipFree` with stream-ordered work possibly in flight and the
illegality of `hipMalloc`/`hipFree` under graph capture. Both are exactly what
the sibling's pattern fixes. But the pattern cannot be applied per-thread:
`hipMallocAsync` is stream-ordered, and one host thread demonstrably submits to
two streams here (`src/vt/rocm/rocm_gemma4_experts.hip:1147,1149` issue on the
expert device's queue while the compute queue is live). So `LtWorkspace` becomes
keyed by stream rather than by thread, which also removes the separate hazard of
one buffer being shared across a two-device hop.

## Tests

`tests/vt/test_grow_only_stream_scratch.cpp`, registered unconditionally, so a
CPU-only runner holds it.

The invariant: **every caller receives a block whose capacity is at least the
size that caller asked for**, and the pool's recorded capacity always describes
the pool's recorded block. Threads contend on one stream key with different
sizes, and the allocator widens the window so the interleaving is reached rather
than hoped for.

RED is taken at the commit that extracts the pool while preserving today's
semantics exactly, so the red result is reproducible at a named SHA rather than
described.

## Gates

- Focused: `ctest -R test_grow_only_stream_scratch`, red before, green after.
- Build: the two `.hip` translation units compile on `strix:gpu0` (gfx1151).
- Reachability: the production call sites at
  `rocm_grouped_gemm.hip:841,861,914,939` reach `Ensure`; deleting the routing
  in a scratch copy must red the focused gate.

## Stop conditions

- Stop if holding the lock across `hipMallocAsync` can deadlock. It cannot: the
  callable makes one HIP allocation call and re-enters nothing.

## Owed

- `src/vt/cuda/cuda_quant_dot.cu:1806-1822` carries the identical defect; the
  ROCm code was copied from it. Not repaired here — see `## Scope`.
- `GetLt()` (`rocm_matmul_hipblaslt.hip:272`) keeps one `thread_local`
  `hipblasLtHandle_t` across both devices of the Gemma4 peer hop, while its
  neighbour `GetBlas` was given an explicit dual slot for that exact hazard.
- [#2836](https://github.com/mudler/vllm.cpp/issues/2836): the speech route
  shares one `vt::Queue` across cpp-httplib worker threads with no lock. This is
  the "one shape would make it live" case in `## First deliverable` above, filed
  because it is a hazard on its own terms and not only as #2712's trigger.
- [#2837](https://github.com/mudler/vllm.cpp/issues/2837): a THIRD grow-only pool
  in `rocm_matmul_hipblaslt.hip` (the batched pointer tables, `:678-680` and
  `:738-740`) frees on growth. Its locking is correct, so it has no race; it
  shares only the lifetime half, and the repair is the same retire-never-free.
  Found by reading the file's remaining `hipFree` occurrences after this row
  removed the one it owned.
- The hipBLASLt arm is opt-in and default OFF (`LtEnabled()`,
  `rocm_matmul_hipblaslt.hip:255`, requires `VT_ROCM_HIPBLASLT=1`), so the
  `LtWorkspace` half of this change is compiled but not exercised at runtime by
  any default-configuration gate this row ran.
