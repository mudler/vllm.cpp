# ROCm split sampling scratch ownership

Row: `BACKEND-ROCM`.
Issue: [#3062](https://github.com/mudler/vllm.cpp/issues/3062).
Launch regression: [#3022](https://github.com/mudler/vllm.cpp/issues/3022).
Contribution: [#3010](https://github.com/mudler/vllm.cpp/pull/3010), closing
[#3009](https://github.com/mudler/vllm.cpp/issues/3009).
Base: `066d2fd80f324c647709a56601bb382d526ebfe4`.

## Scope

Repair the split sampler's scratch ownership in `src/vt/rocm/rocm_sample.hip`.
Add dispatch tests and device tests through `vt::RandomSample`.
Do not change sampling arithmetic, defaults, thresholds, or backend lifetime.

## Source and gap

`RandomSampleKernelRocm:522-536` shares `part_score`, `part_idx`, and
`part_rows` across all queues and devices. Two streams can overwrite each
other's partials. Growing a batch discards allocation pointers.
`git log -S part_rows -- src/vt/rocm/rocm_sample.hip` identifies `be3af091b`
as the introducing commit.

The upstream sampling operation stays unchanged. The pinned vLLM source is
specified in `../upstream-sync.md`. The local mirror calls
`GumbelScore` and `ArgReduce` from `include/vt/sample_common.h`.
This repair changes allocator ownership, without porting new arithmetic.
The established local lifetime reference is `src/vt/grow_only_stream_scratch.h`,
which retains allocations that captured graphs can reference.

## Design

Use `GrowOnlyStreamScratch` with a process-unique queue identity.
`Queue::id` distinguishes different devices and reused native stream handles.
Allocate one slab containing 64 rows of 128 scores and indices per queue.
The split dispatch already rejects batches above 64 rows.
The slab uses 98,304 bytes. Its index slice stays aligned to `int64_t`.
Return pointers by value and publish the slab under the existing pool lock.
Keep the slab resident under the existing graph lifetime contract.
Do not free scratch when a queue disappears because a graph can still refer
to it. Queue churn still retains one slab per identity.

### Reject unwarmed capture before allocation

The review of `2bd0ca67f051d7ef7d48b52ad75a8729fe30d0b3` found another
lifetime defect. Removing the sampler warmup produced a device page fault
and exit 134 on gfx1151 with ROCm 7.2.4. A first-use `hipMallocAsync`
inside capture creates a graph-owned allocation. The pool then publishes
that pointer for uncaptured calls before the graph executes.

The established contract requires warmup, not successful cold capture.
`rocm_backend.hip:320-335` requires scratch prewarm and fixed pointers.
`rocm-decode-graph.md:280-291` requires a loud failure if allocation occurs
during capture. `test_rocm_backend.cpp:496-512` gates warmed success.
It leaves a future capture-safe allocator possible, without requiring one.

On a pool miss, query `hipStreamIsCapturing` before allocation. Active
capture refuses with an instruction to pre-warm the sampler on that queue.
Invalidated capture reports that capture must end before prewarming.
A query error propagates without allocation or publication. A pool hit
does not query capture state and keeps the existing warmed path unchanged.
Do not add allocator threads, change the capture API, or alter backend
lifetime. The operator approved this scope on 8 September 2026.

The HIP allocation contract is documented in the
[stream-ordered allocator reference](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/group___stream_o.html).
Its allocation node belongs to the graph, unlike an eager pool allocation.

## Tests and gates

Commit the regression tests before the implementation.
The host dispatch harness compiles the production launcher with fake HIP
allocation and launch functions. It checks queue separation, device separation,
handle reuse, fixed allocation capacity, and production launch reachability.
It tests host ownership only and does not claim device execution.

The ROCm case calls `vt::RandomSample` from two queues with disjoint one-hot
distributions. Concurrent graph replay must preserve every requested token.
Capture at a small batch, run a larger batch, and replay the original graph
to verify pointer lifetime. Run the existing sampler cases unchanged.
Pre-warm that queue before capture, as the backend contract requires.

Add host regressions for active capture, invalidated capture, query errors,
and recovery without cached partial allocations. Check that warmed capture
reuses the slab without querying capture status. Preserve the actual HIP
launch arguments in the host adapter. Phase B must launch one thread per
partial and fit its shared arrays. Replacing its width with `kVocabBlock`
must fail deterministically, even when a numerical GPU test tolerates the
result of the out-of-bounds writes described in #3022.

Add a device case through `vt::RandomSample` that refuses an unwarmed
capture, ends and destroys the empty graph, then warms and retries.
Check eager execution before graph replay and after graph destruction.
Keep the existing concurrent warmed-graph case unchanged. The cold-refusal
case is inapplicable when either fast sampling or split sampling is disabled.

Focused gates: the host dispatch test and `test_ops_sample` on ROCm with
`VT_FAST_RANDOM_SAMPLE=1 VT_SAMPLE_SPLIT=1`.
Repeat device cases with `VT_SAMPLE_SPLIT=0` as the reference path.
The operator runs the HIP build and full CTest suite under its device lease.
Run `scripts/agent-preflight.sh` and inspect its failed-gate count.
The fresh reviewer mutates ownership, allocation capacity, and the production
call site and requires the applicable focused tests to fail.

## Risks and stop conditions

Stop if the repair requires different sampling behavior or backend lifetime.
Device execution stays PENDING until the operator supplies lease evidence.
The pool does not serialize two host callers interleaving phases on one queue.
Callers continue to own submission serialization for a shared queue.

## Now

ACTIVE: the capture-refusal amendment precedes its regression and guard.
The host dispatcher suite failed three of eight cases before the guard and
passes all eight afterward. Seven scratch mutations fail: removed guard,
ignored query error, wrong capture branch, allocation before the query,
1024-thread phase B, missing partial lanes, and a query on warmed hits.
Each source mutation was restored byte-for-byte.

The device recovery case and the existing concurrent replay case remain
PENDING until the operator runs them under a lease. Host execution does not
establish device capture behavior. No performance result is claimed.
