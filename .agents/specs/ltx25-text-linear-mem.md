# `LTX25-TEXT-LINEAR-MEM` — #1252 did not cost 26 GiB, and the seam has no memory bound

Row: `LTX25-TEXT-LINEAR-MEM`
Issue: [#1286](https://github.com/mudler/vllm.cpp/issues/1286)
Sibling rows: `LTX25-TEXT-LINEAR-SEAM` (#1252, [spec](ltx25-text-linear-seam.md)),
`LTX25-LORA-FUSE-SEAM` (#1259, [spec](ltx25-lora-fuse-seam.md))

## Scope

[#1286](https://github.com/mudler/vllm.cpp/issues/1286) reports that #1252's
threaded caption projection raises peak host memory from ~79 GiB to 105.85 GiB
on GB10 and aborts a full-model LTX-2.5 render before any denoise step. It names
three candidate causes — the shared threadpool's per-thread tiles, the
`ltx2_text_encoder.cpp` call site's full-size `scaled` copy, or an interaction —
and states plainly that the attribution is not measured and belongs to whoever
takes the row.

This row does that attribution first, and lets the answer choose the fix.

In scope:

- attribute the reported +26 GiB against the retained evidence and against a
  local before/after measurement at the shipped geometry;
- report peak host memory **and** throughput for both arms, because a memory fix
  that costs the 8.57x is a bad trade;
- decide whether a code change is warranted, and make it if so;
- give the shared GEMM seam a **memory bound that a gate can hold**, since the
  tree has none and could not have caught the reported defect had it been real;
- check whether #1259, the sibling seam change that has never run on the full
  model, shares the profile.

Out of scope: the LTX-2.5 speed axis, the unmeasured GB10 per-core ratio `R`
that #1252 carries under `## Owed`, #1254's bf16 add-back, and #1210's
fuse/un-fuse round trip. None becomes reachable from this change.

## Upstream chain

None, and that is a property of the question rather than a gap. Peak resident
footprint is a property of this port's execution strategy: vLLM does not
implement LTX-2.5, vLLM-Omni's recipes stop at 2.3, and the Lightricks
`ltx_core` oracle runs the projection as one `torch.nn.functional.linear` whose
allocator behaviour says nothing about a C++ threadpool's per-worker tiles. The
arithmetic mirror is already decided and unchanged by this row: f32 accumulation
through `vt::MatmulBT`, argued in `ltx2_text_encoder.cpp:64-69` and in
[`ltx25-text-linear-seam.md`](ltx25-text-linear-seam.md).

The parallel-dispatch layer this row measures **is** a 1:1 port, and its anchors
are the ones to read: `src/vt/cpu/cpu_ops.cpp:1-8` and `cpu_threadpool.h` record
llama.cpp (local fork) `ggml/src/ggml-cpu/ggml-cpu.c:1155-1443` @ `237ad9b96` as
the chunk policy's source. `MatmulOneChunk`'s widened-activation buffer is a
recorded *deviation* from that port (`cpu_ops.cpp:99-108`), so its cost is ours
to bound.

## Our baseline

Read out of the tree at `origin/main` `678fc672c`:

1. **The seam allocates exactly one thing per call, and it is thread-local.**
   `src/vt/cpu/cpu_ops.cpp:125` — `static thread_local std::vector<float> af;`,
   resized to `nrows * k` where `nrows = min(16, chunk rows)` and 16 is ggml's
   `blck_1`. Nothing else in `MatmulChunked`, `MatmulOneChunk`,
   `MatmulOneChunkRef` or `vt::MatmulBT` allocates. At the shipped
   `k = 188160` that is `16 x 188160 x 4 = 12.04 MB` per worker, and at
   `hardware_concurrency() = 20` it is **241 MB in total**, not 26 GiB.
2. **`static thread_local` means it is allocated once per worker for the process
   lifetime and never shrinks** — the comment at `cpu_ops.cpp:122-124` says so.
   That is a retention property worth recording; it is not a per-call cost.
3. **The call site's `scaled` copy is pre-existing and unchanged by #1252.**
   `ltx2_text_encoder.cpp:460-465` builds one `std::vector<float>` the size of
   `normed` per `project()` call — at `B*T = 1024`, `flat = 188160` that is
   770.4 MB — and #1252's body records it as deliberately kept on both arms.
   A copy present in the *before* arm cannot explain a *regression*.
4. **The tree has no memory-bounded test at all.** `tests/vt/test_ops_matmul.cpp`
   and `test_ops_matmul_elem.cpp` assert values and byte equality; nothing
   anywhere asserts a peak-RSS or allocation bound. So the reported defect, had
   it been real, would have been invisible to every gate in the repository.

## Port map

| Piece | Where | Change |
|---|---|---|
| Attribution of #1286 | this spec, `## Risks/decisions` | new |
| Local A/B probe (scratch, uncommitted) | scratchpad | new |
| Memory bound for the shared GEMM seam | `tests/vt/test_ops_matmul_mem.cpp` | new |
| Test registration | `tests/CMakeLists.txt` | one line |

No `src/` change is planned. If the measurement contradicts the baseline above,
the fix is chosen from #1286's own list and this table is rewritten before any
code is written.

## Tests to port

None to port: there is no upstream test for a C++ threadpool's peak resident
footprint. The new test is original and its contract is stated where it lives.

It asserts the property the baseline claims and #1286 doubted: **peak RSS growth
across a `vt::MatmulBT` call does not scale with the whole intermediate per
worker.** Concretely, the growth in `VmHWM` attributable to the call, measured
at a large `K` where the per-worker tile is the dominant term, stays under the
documented `nthreads x 16 x K x 4` plus slack — and in particular is nowhere
near `nthreads x rows x K x 4`, which is the shape #1286 hypothesised.

It is red-first by mutation rather than by pre-existing failure, because there
is no defect to start red against. The mutation is the hypothesised defect
itself: widen `af` from the 16-row tile to the chunk's whole row span. Recorded
with `BUILT`, the compiler error count, `git diff --stat`, and the failing
assertion by name.

## Gates

- `test_ops_matmul_mem` — the new bound, plus its mutation.
- `test_ops_matmul`, `test_ops_matmul_elem` — the seam's existing value and
  byte-equality gates, unchanged and still green.
- `test_ltx2_text_encoder` — the caption projection's byte-exactness against the
  recorded goldens. **No tolerance is widened anywhere in this row**; the values
  must not move, and nothing here touches summation order.
- `test_ltx2_lora`, `test_ltx2_loader` — #1259's gates, for the sibling check.
- Full `ctest`, and `scripts/agent-preflight.sh --staged`.

Known-environmental and not chased: `test_engine_core_proc` and `test_async_llm`
starve under `ctest -j` and are re-run serially; `test_cpu_x86_llamacpp_floor`
is #618 under load; `windows-msvc-*` is red on every pull request.

## Dependencies

- The retained GB10 evidence under
  `/mnt/nas_share/rc/ltx25-fullmodel/out/` — both the #1252 run
  (`20260818T220620Z/1024x576-25f/`) and the pre-#1252 run
  (`1024x576-25f/`, 2026-08-18 17:20-18:11), which is what makes a paired
  comparison possible at all.
- `scripts/../runguard.py` on the NAS defines the columns; its
  `runguard.py:236-237,260` fix `used_gib = MemTotal - MemAvailable` and
  `anon_gib = AnonPages`, both **system-wide**, against `rss_gib`, which is the
  child's own. That distinction is what the attribution turns on.
- No GPU and no lease. Allocation sizes are architecture-independent, and the
  local arm is an x86 box.

## Work breakdown

| # | Work | Landable alone |
|---|---|---|
| W1 | Attribute #1286 against the retained paired evidence | yes |
| W2 | Local before/after A/B: peak RSS and throughput at the shipped geometry, swept over threadpool width | yes |
| W3 | Check #1259's profile and either fix in flow or file | yes |
| W4 | The seam's memory bound, red by mutation, green after | yes |

One pull request; no split case applies, and no developer answer is recorded
under `## Git integration`.

## Risks/decisions

**The instrument's own precondition is the first risk, and it is the one that
bites.** #1286 compares two *absolute* peaks of a *system-wide* column. That
comparison is only sound when both runs start from the same occupancy, and
nothing in the issue checks whether they did. This spec's `## Outcome` records
what the check found.

**A scaled probe is never presented as a full-geometry one.** Where the local
arm reduces `rows` or `out_features` to keep a single-threaded arm inside a
sensible wall time, the reduction is stated beside the number, and the
thread-scaling term it isolates is independent of `rows` by construction (`af`
is sized by the 16-row tile, not by the chunk's row count).

**Byte-exactness is not negotiable.** #1252's path is bit-exact against recorded
goldens and its review confirmed a test diff of 150 insertions and zero
deletions. If any candidate fix altered summation order — blocking the GEMM over
`rows` is the one that would — this row stops and puts the question here rather
than deciding it in code. No tolerance is widened to make a red green.

**A memory bound can be flaky.** Peak RSS depends on the allocator, so the test
is written to bound a *difference* at a geometry where the signal (12 MB per
worker) is far above malloc noise, rather than to pin an absolute number. If a
robust bound cannot be written, that is reported as a finding instead of being
weakened until it passes.

## Owed

- [#1286](https://github.com/mudler/vllm.cpp/issues/1286) — this row.

## Now

`ACTIVE`. W1 and W2 measured; see `## Outcome`.

## Outcome

Filled in when the row reaches `DONE`.
