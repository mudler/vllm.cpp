# BACKEND-ROCM-PTR-TABLE-RETIRE — the third grow-only pool takes the seam #2712 landed, and two of the issue's premises do not survive reading the file

Issue: [#2837](https://github.com/mudler/vllm.cpp/issues/2837).
Base: `7b8b480b1`.
Filed by [#2712](https://github.com/mudler/vllm.cpp/issues/2712) and carried
under `## Owed` in [`rocm-scratch-sync.md`](rocm-scratch-sync.md).

## Scope

The batched pointer tables in `src/vt/rocm/rocm_matmul_hipblaslt.hip`: the two
identical `PtrTables` blocks at `:665-687` (`MatmulBTPointerBatchKernelRocm`)
and `:725-747` (`MatmulBTPointerBatchABKernelRocm`).

IN SCOPE: the lifetime and the synchronisation of those tables, the extension of
`vt::GrowOnlyStreamScratch` that carries both, and the unit test that holds it.

OUT OF SCOPE: `src/vt/rocm/rocm_fp8_channel_gemv.hip:509-524`, which
`rocm-gfx1151-q4k-hang.md` records as "the identical shape". It is on a live MoE
path, so it needs a reachability answer this row does not have. `## Owed`.

## First deliverable: what the issue got wrong

Two of #2837's premises do not survive reading the file at the base commit, and
both change what the repair is worth.

**1. It is not "the hipBLASLt arm is default OFF". There is no caller at all.**
Neither `MatmulBTPointerBatchKernelRocm` nor `MatmulBTPointerBatchABKernelRocm`
uses `hipblasLt` — they call `hipblasGemmBatchedEx` through `GetBlas`, and
`LtEnabled()` is not on either path. So the issue's reason for filing rather
than folding in ("the whole hipBLASLt arm is opt-in") names the wrong gate. The
real one is stronger: `grep -rn PointerBatch` over the tree at `7b8b480b1`
returns the two definitions, the two declarations in
`include/vt/rocm/rocm_matmul_batch.h`, and one prose line. `src/vt/fused_ops.cpp`
includes that header and calls neither. This is already on the record —
`.agents/specs/rocm-gfx1151-q4k-hang.md:210` calls it "dead code one call site
away from being real".

**2. "Its locking is correct" is false.** The lock covers the grow. It does not
cover the USE. `tab` is a pointer into the map, and the three
`hipMemcpyAsync(tab->A, ...)` calls that follow are outside the `lock_guard`
scope. A second thread on the same device that grows the tables then frees
`tab->A` and publishes a new pointer WHILE the first thread is reading `tab->A`
for its memcpy. That is an unsynchronised read racing a write under the lock —
a data race by the language's definition, not only a lifetime hazard — and it
frees the block the first thread is about to copy into. So this pool shares the
race half as well as the lifetime half; #2712's "it has no race" applies to the
grow, which is not where the pointer is read.

**Verdict: latent, and more deeply so than the issue says.** Nothing calls it,
so nothing can currently execute either defect. That is the honest reading, and
`## Rejected alternatives` says what follows from it.

## Design

**Extend the seam, do not write a fourth variant.** `src/vt/grow_only_stream_scratch.h`
gains `GrowOnlyStreamTriple<KeyT>`, which holds one `GrowOnlyStreamScratch<KeyT>`
and slices its block into three equal tables. It inherits both of that class's
rules — one lock across the whole operation, and retire-never-free — and adds
the two this site needs:

- **Three tables from ONE block.** `hipblasGemmBatchedEx` wants three device
  arrays of `batch` pointers. Three separate allocations with one shared `cap`
  is what the current code has, and it is the reason a partial allocation
  failure can leave `cap` describing buffers that were not all replaced. One
  block with three slices cannot be partially grown.
- **Returned BY VALUE, under the lock.** The caller receives three pointers, not
  a pointer into a shared entry it must re-read after the lock is gone. That is
  the second defect above, removed by the shape rather than by remembering to
  hold a lock longer.

Keyed by `int` (the device index), because these tables are per-device and the
GEMM is submitted on whichever stream the queue carries. `KeyT` was already a
plain map key with no backend meaning, so this needs no change to the class it
composes.

Growth stays O(log(max/min)) per device over a process, and the retained bytes
are three pointer arrays — kilobytes, not megabytes.

## Rejected alternatives

**Delete both functions.** `## Nothing lands dead` argues for it: they have no
caller, `rocm-gfx1151-q4k-hang.md` already recorded that, and deleting removes
both defects at once with nothing to gate. It is rejected HERE and not on the
merits: `include/vt/rocm/rocm_matmul_batch.h` describes the file as "batched
MatmulBT helpers for MoE top-k fuse", so the functions are staged work with an
intended consumer, and retiring another row's staging is not a helper's call to
make on a bug-fix branch. Recorded under `## Owed` so the decision has an owner
rather than being lost with this branch.

**A lock held across the memcpy calls.** It fixes defect 2 and not defect 1, it
serialises three host-to-device copies that do not need serialising, and it
leaves the free-on-growth that graph capture forbids.

## Tests

`tests/vt/test_grow_only_stream_scratch.cpp` gains the triple's cases, in the
file that already holds the pool it composes.

The invariants: three NON-OVERLAPPING tables, each at least the per-table size
its caller asked for; a table handed to an earlier caller is still the block
that caller was given after a later caller grows the pool (retire, never free);
and under contention on one key with different batch sizes, every caller's three
tables satisfy both.

`VLLM_CPP_SANITIZE=thread` is run on the same case. It is NOT the discriminating
arm here and this row does not pretend otherwise: the composed class already
holds one lock across the whole operation, so a correct implementation is
race-free before this change and after it. The arm proves the extension did not
introduce one.

## What this row does NOT prove

Stated here rather than discovered later.

- **No gate in this project executes the repaired code.** There is no caller, so
  there is no production entry point to enter it through, and
  `.agents/reachability.md`'s mutation test cannot be run on a call site that
  does not exist. The HIP translation unit is compiled on `strix:gpu0` and that
  is all the device side is measured for.
- **There is no red-before that RUNS.** The defect lives in a `.hip` file a CPU
  runner cannot compile and nothing calls. The unit test's red is therefore
  taken by MUTATION on the merged head, rebuilding for each mutation, with the
  restore verified by sha256 — not by a red-then-green pair at two SHAs. This is
  weaker evidence than #2712 had and it is labelled as such.
- **The GPU behaviour is unmeasured.** No hipBLASLt or hipBLAS batched call was
  executed with these tables on any device by this row.

## Gates

- Focused: `ctest -R test_grow_only_stream_scratch`, green, with each guarantee
  mutated and the mutation shown to red it.
- Sanitized: the same target under `-DVLLM_CPP_SANITIZE=thread`, zero races.
- Build: `src/vt/rocm/rocm_matmul_hipblaslt.hip` compiles on `strix:gpu0`
  (gfx1151) under `-Werror`, which reaches HIP translation units since
  `6f6caa725`.

## Stop conditions

- Stop if the three slices can be misaligned for `void**`. They cannot: the
  per-table size is `batch * sizeof(void*)`, so every slice offset is a multiple
  of `alignof(void*)`, and the block itself comes from `hipMalloc`.

## Owed

- Whether `MatmulBTPointerBatchKernelRocm` and `MatmulBTPointerBatchABKernelRocm`
  should exist at all. They have no caller; this row repaired them rather than
  deleting them, and the decision needs the row that staged them.
- `src/vt/rocm/rocm_fp8_channel_gemv.hip:509-524` — the same unsynchronised
  pointer-table upload, on a path that is NOT dead. See `## Scope`.
