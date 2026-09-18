# QUANT-EXL3 W7: one reconstruct scratch per stream, not one per captured graph

Row: `QUANT-EXL3`, wave W7, a follow-up to W6 (#3150). The row's parent spec is
[`quant-exl3-shared.md`](quant-exl3-shared.md).
Issues: `ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ` (`.agents/issues/ENG-POOL-BEST-FIT/`),
the host OOM this change removes; `ISSUE-LOCAL-01M2BYPW7YTC2B2MY023ETTKQ2`
(`.agents/issues/QUANT-EXL3/`) item 2, the batched decode that crosses M = 144.
Base SHA: `c38ad5b0d`.
Upstream: vLLM implements no EXL3, so the kernel is mirrored from the registered
oracle [`exllamav3`](../oracles/exllamav3.md); the benchmarked revision is
`MiaAI-Lab/exllamav3` @ `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed`.

## Now

W7 `ACTIVE` on `row/QUANT-EXL3-W7`. Spec committed before implementation.

## 1. Scope

`dense_attn::Exl3MatmulD` (`include/vllm/model_executor/models/dense_attn_block.h`)
allocates its reconstruct weight scratch, fp16 `[K, min(N, 32768)]`, as a pool
`DBuf` on every call with M > 144. On a CUDA-graph decode step that scratch is
part of the step's demand profile, so every captured slot pins one block of each
distinct scratch class. This row gives each (device, queue) one persistent
scratch that every reconstruct call on that queue reuses.

In scope: the scratch allocation in `Exl3MatmulD`, the persistent buffer's owner,
and the tests. Out of scope: lazy CUDA-graph capture and its accounting, DevicePool
free-list retention, the DFlash2 context stores, and a memory profile run. Each is
real and recorded in `ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ`, and none is
needed to restore the pre-#3150 plateau.

## 2. The measured problem

On `dgx:gpu0` (GB10, 125 GiB unified memory), Qwen3.8-27B EXL3 3.5bpw with its
DFlash2 EXL3 draft at k = 7, `--max-num-seqs 32 --num-blocks 8192
--max-num-batched-tokens 16384`, 128 requests of the variadic corpus at
concurrency 32:

| binary | MemAvailable at ready | at 130 s | floor |
|---|---|---|---|
| `3cafbcaf` (#3150's parent) | 58.19 GiB | 24.3 GiB | 19.2 GiB, leg completes at 50.84 tok/s |
| `39d3af455` (#3150) | 58.21 GiB | 14.0 GiB | below the 14 GiB watchdog, still falling |

Unwatched, the #3150 binary exhausted host memory and took the machine down.

## 3. Mechanism (read in source; the investigation is in the issue record)

- Spec-decode verify graphs are captured lazily per verify width S = num_reqs x 8,
  on a two-slot ring, and never evicted (`qwen3_5.cpp`, the decode-graph slot map).
  A slot's first step at a shape runs eager and records the pool demand; the next
  pre-grows the pool, captures, and `PinForGraph`s that demand
  (`device_pool.h` `PinForGraph`, `PreGrowForCapture`).
- For S >= 152 (num_reqs 19 to 32) every EXL3 linear takes the reconstruct path,
  and the step's demand holds one scratch block per distinct class. Rounded to
  pool classes for this checkpoint: MLP 176 MiB, q_proj 120 MiB, GDN in_proj_qkv
  100 MiB, 6144 x 5120 projections 60 MiB, k/v 10 MiB, lm_head (sliced at 32768)
  320 MiB, about 786 MiB per captured slot.
- 14 widths x 2 slots bounds that at about 21.5 GiB. The eager path adds under
  1 GiB, because each class peaks at one block.

## 4. Upstream anchors

- exllamav3 `exllamav3/modules/quant/exl3.py:161-217` (`reconstruct_hgemm`)
  allocates the reconstructed weight with `torch.empty` on every call and frees it
  when the call returns. Under torch's caching allocator, and inside a CUDA graph
  pool, that block is SHARED by every call and every graph, because the graphs
  replay sequentially and the tensor is dead after its GEMM.
- vLLM captures all graphs into one shared graph memory pool
  (`vllm/platforms/interface.py` graph pool handle, `gpu_model_runner.py`
  `capture_model`), so a transient that is dead between ops costs one block, not
  one block per graph. This row restores that property for the one transient that
  #3150 added. The general property for all transients is the lazy-capture
  accounting work, which is out of scope here.

## 5. Design

1. A per-(backend, queue) persistent fp16 buffer, owned outside `DevicePool`, so
   it is never pinned, pre-grown, counted in a demand profile or retained as free
   pool bytes. Lifetime: the backend's. Keyed by queue, because two queues on one
   device (the aux stream, the draft) may run reconstruct concurrently.
2. `Exl3MatmulD` borrows a `[K, min(N, 32768)]` view of it instead of drawing a
   `DBuf`. The contents are dead once the GEMM is enqueued, and every consumer on
   a queue runs in that queue's order, so reuse across calls and across replayed
   graphs is safe.
3. GROW-ONLY, and growth is refused while the stream is capturing, with a message
   that names the buffer (the rule `cuda_dropin.cu` applies to its workspace). A
   slot's eager step at a shape runs every linear the captured step will run, so
   the buffer reaches its high-water mark before any capture at that shape.
4. The dispatch predicate, the kernels and the numbers are unchanged.

## 6. Risks

- **A queue shared by two host threads.** Every reconstruct on one queue must be
  serialized by that queue's owner. Verify that no production path enqueues
  EXL3 linears on one queue from two threads; if one does, return
  `NEEDS_DECISION`.
- **A capture at a shape whose eager step never ran.** Growth inside capture is
  refused by name rather than aborting the capture silently; the red test
  covers the refusal.
- **Draft and target on the same queue.** Sharing is correct, because the calls
  are sequential on that queue.
- **Memory held forever.** One buffer per queue at the model's largest
  `[K, min(N, 32768)]`, 320 MiB for this checkpoint.

## 7. Tests (red first, through `dense_attn::Exl3MatmulD`)

- CUDA: repeated `Exl3MatmulD` calls at M > 144 across several `(K, N)` shapes
  draw no `DevicePool` block for the scratch, so the pool's demand profile and
  driver allocations do not include a `[K, min(N, 32768)]` fp16 class. Red at the
  base SHA.
- CUDA: the output of the reconstruct path matches the pre-change output
  byte-for-byte for the same inputs, and matches `Exl3Gemm` within the existing
  cross-path bound.
- CUDA: growth of the buffer while the stream is capturing is refused by name.
- CPU: unchanged dispatch (`test_exl3_matmul_dispatch` stays green).
- Mutation: restore the per-call `DBuf`; the first test must go red.

## 8. Gates (operator, on `dgx:gpu0` through `rc`)

- **G-MEM:** the c = 32 watchdog leg from the issue record. The fixed binary
  completes the leg, and its MemAvailable floor is at least the parent's 19.2 GiB
  minus 1.5 GiB of run-to-run noise.
- **G-TOKENS:** greedy (T = 0) outputs of the fixed binary and `39d3af455` are
  token-identical on the variadic corpus's first 32 prompts at c = 1.
- **G-SPEED:** c = 1 and c = 16 output tok/s and the XL-band median TTFT of the
  fixed binary are within the two binaries' round-to-round spread of
  `39d3af455`, measured interleaved on one boot.

## 9. Evidence

Per-gate logs, binary md5s and the boot id, recorded in the issue record and in
this spec's `## Outcome`.

## 10. Stop conditions

- A production path runs reconstruct on one queue from two threads (return
  `NEEDS_DECISION`).
- G-TOKENS fails.
- G-MEM does not reach the parent plateau: the increment is not the scratch
  pinning alone. Re-open the mechanism rather than widen the change.
