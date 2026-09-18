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

2026-09-13, implementer progress (not a gate result; the operator owes G-MEM,
G-TOKENS and G-SPEED):

- Seam: the scratch lives in the CUDA kernel, not in `Exl3MatmulD`.
  `vt::Exl3ReconstructGemm` gains an overload without `w_scratch`, and
  `src/vt/cuda/cuda_exl3.cu` resolves it to one grow-only fp16 buffer per
  (device index, stream). This is the house per-stream scratch idiom
  (`cuda_mla_attn.cu` `EnsureMidScratch`, `cuda_dropin.cu` workspace), and the
  capture query (`cudaStreamIsCapturing`) is only visible there. The
  explicit-scratch overload is unchanged, and it now refuses an empty scratch.
- Growth under capture throws `exl3 reconstruct scratch growth is forbidden
  during CUDA graph capture`, before the kernel launches anything. On growth, a
  block that was handed out while its stream was capturing is retired
  (`graph_safe_scratch.h`). A block that never was is freed on the stream, so
  the intermediate classes of the first eager step are not kept resident.
- §6 risk check: no production path enqueues EXL3 linears on one queue from two
  host threads. Every model forward, the DFlash2 draft included, runs on the
  engine busy-loop thread. The MoE shared-expert aux queue runs no EXL3.
- Tests, `tests/vt/test_exl3_matmul_dispatch.cpp`, through `Exl3MatmulD` on
  `thor:gpu0` (sm_110). Lease `195340a3-0ade-4dc9-810f-bf9f69fb4c32`, base
  `909125d16`:
  - (1) no pool demand for a scratch class: red at base, green with the change.
  - (2) byte-identical to the per-call-scratch call, and within 1.0e-3 of
    `Exl3Gemm` (unfused) or of the f64 chain (fused): green at base and with the
    change. This is an invariance test, so it has no red arm. The fused path
    measures 9.62e-4 against f64 at M = 1024 and 1.01e-3 against `Exl3Gemm`.
  - (3) named refusal under capture: red at base, where the capture fails in the
    cuBLASLt workspace and poisons the stream. Green with the change.
  - Mutation, a per-call `DBuf` restored in `Exl3MatmulD`: (1) and (3) red. The
    header was restored byte-for-byte, and the rebuilt binary md5 equals the
    unmutated one.
  - CPU: `test_exl3_matmul_dispatch` 4/4, `test_exl3_gemm` 20/20 and
    `test_exl3_linear_method` 7/7.

2026-09-13, review-repair progress (fresh implementer; not a gate result). The
fresh review of `c83521c20` returned FAIL on four findings:

- Retire on growth was untested (M2, growth always frees). New case: eager
  growth to B1, capture G1, eager growth to B2, a second queue allocates B1's
  size, then G1 replays interleaved with that queue's reconstructs.
- The per-stream key was untested (M3, key by device only). New case: two
  queues with different (K, N) enqueue interleaved with no sync, and each output
  is compared with its single-queue reference. B's scratch fits inside A's, so a
  device-only key is a data race on live bytes and not a fault.
- `DestroyQueue` did not release the entry. `ReleaseExl3ReconScratch` in
  `cuda_exl3.cu` now synchronizes the stream, frees the block (or retires it when
  a capture was handed it), and erases the entry. It is called from
  `CudaBackend::DestroyQueue`. New case: destroy, then recreate, a queue.
- The explicit-scratch overload is marked test-only in `include/vt/ops.h`.

Test hooks are in `src/vt/cuda/cuda_exl3_internal.h`. All arms ran on
`dgx:gpu0` (sm_121a), lease `7f670ed1-0e52-4561-b3fe-140e5012e295`. The clone
was at `909125d16` plus the diff, and the code tree hash was
`08b641a2f2e1e5871967b2223b658ad10a11a3e7`:

- FIX: all 7 CUDA dispatch cases green, and `test_exl3_gemm` 20/20.
- M2: the retire case is red. The retired count is 0, not 1. The second queue
  received the freed B1. 2 of 8 replays differ from the pre-growth bytes.
- M3: the per-stream case is red. The hook capacities are wrong, and 1 of 6 of
  A's interleaved outputs differs. The byte race is probabilistic, and the hook
  assertions are deterministic.
- M5 (no release call in `DestroyQueue`): the teardown case is red. The old
  handle keeps 16 MiB, and the live bytes do not return to baseline.
- M1 (no capture refusal) and M4 (per-call `DBuf`): still red, on the same cases
  as before.
- After each restore, the tree hash equals `08b641a2`. A final FIX rebuild was
  7/7 green. The binary md5 is not reproducible, so it is not the evidence.
- CPU: `test_exl3_matmul_dispatch` 4/4, `test_exl3_gemm` 20/20 and
  `test_exl3_linear_method` 7/7.

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

- **G-MEM:** the c = 32 leg from the issue record. The fixed binary completes it,
  and its MemAvailable is within 1.5 GiB of #3150's PARENT **on the same
  statistic**, measured in the same job.

  **The floor was first written as "at least the parent's 19.2 GiB minus 1.5".
  That compared two different statistics** and is corrected here rather than
  restated: 19.2 GiB was the parent's sustained plateau read off an earlier run,
  while the gate computed the fixed binary's transient MINIMUM. Measured
  like for like on 2026-09-18, lease `3a419fc7`, boot `aa8685cb`, two rounds each,
  interleaved, one c = 32 leg of 128 requests per arm:

  | arm | min GiB | p05 | median | output tok/s |
  |---|---|---|---|---|
  | parent `3cafbcaf` | 15.93 / 16.59 | 17.81 / 18.30 | 19.06 / 18.43 | 50.57 / 50.72 |
  | fixed `75984bbf` | 15.10 / 14.96 | 16.13 / 16.59 | 17.59 / 18.16 | 94.26 / 94.37 |

  The parent's own transient minimum is 15.93 GiB, so the 1.5 GiB band admits
  14.43; the fixed binary reads 14.96. No watchdog fired on any of the four legs.
  The fixed binary holds about 1 GiB less at the minimum and 0.6 to 0.9 GiB less
  at the median while serving 1.86x the tokens per second.

- **G-TOKENS:** greedy (T = 0) outputs of the fixed binary and `39d3af455` are
  token-identical on the variadic corpus's first 32 prompts at c = 1.
- **G-SPEED:** c = 1 and c = 16 output tok/s and median TTFT of the fixed binary
  are not below `39d3af455`, measured interleaved on one boot.

## 9. Evidence

Per-gate logs, binary md5s and the boot id, recorded in the issue record and in
this spec's `## Outcome`.

## 10. Stop conditions

- A production path runs reconstruct on one queue from two threads (return
  `NEEDS_DECISION`).
- G-TOKENS fails.
- G-MEM does not reach the parent ON THE SAME STATISTIC: the increment is not
  the scratch pinning alone. Re-open the mechanism rather than widen the change.

## Outcome

`DONE` on 2026-09-18. What the change is, and what it is not.

**Measured.** All three gates ran on `dgx:gpu0`, on a binary built from
`origin/main` plus this branch and verified by tree hash `75984bbf` before it
ran. G-TOKENS: 32 of 32 greedy outputs identical to `39d3af455`. G-SPEED, two
rounds each interleaved on one boot: c = 1 44.09 / 43.98 tok/s against
42.70 / 42.44, and median TTFT 672 / 671 ms against 802 / 803; c = 16
81.18 / 80.67 against 79.41 / 79.33. G-MEM as corrected above. The c = 32 leg,
which `39d3af455` could not finish without a watchdog kill, completes at
94.3 to 95.0 tok/s.

**The TTFT figure is NOT this row's.** 131 ms of it is the chat-template parse
cache (`SERVE-CHAT-TEMPLATE`, landed separately), which this binary also carries
because it was built from main. This row's own contribution to c = 1 is the
throughput, and to c = 32 it is that the leg runs at all.

**Rejected: widening the change.** The issue record attributes about 27 GiB of
the c = 32 growth to lazy CUDA-graph capture and the pool's free list, which this
row does not touch. That was left alone deliberately: the parent's plateau is the
bar, and reaching it does not require them. They stay owed in
`ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ`.

**Rejected: the spec's own §5.2 seam.** The scratch is owned by the CUDA kernel,
not borrowed as a view by `Exl3MatmulD`, because `cudaStreamIsCapturing` is only
visible there and the tree already spells that pattern twice. The reviewer
accepted it; the spec's design text is the draft, this is what shipped.

**Why the retire rule is not simply "always free".** A block handed out while a
stream was capturing is baked into that graph, so freeing it on growth leaves a
replayed graph reading freed memory. A block that no capture saw is freed, which
keeps the first eager step's intermediate classes from staying resident. Two
fresh reviews ran; the first FAILED for leaving that rule, the stream key and the
queue teardown untested, and the repair added a device test for each.

**Open, and visible.** `M7` (the release path frees a capture-exposed block
instead of retiring it) still survives its mutation: no test covers a graph
captured on one queue and replayed after that queue is destroyed. No production
path does that today. Recorded here rather than closed silently.
