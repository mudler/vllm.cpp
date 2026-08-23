# `apply_logits_processors` gates the staging bounce on the wrong memory property

Row `V1-LOGITSPROC-HOST-ADDRESSABLE`, issue
[#1746](https://github.com/mudler/vllm.cpp/issues/1746). This row repairs one
call site. It adds no matrix entry and changes no checker semantics.

The class is the one
[`vt-reference-tier-host-addressable.md`](vt-reference-tier-host-addressable.md)
already records for the reference tier
([#844](https://github.com/mudler/vllm.cpp/issues/844),
[#1435](https://github.com/mudler/vllm.cpp/issues/1435),
[#960](https://github.com/mudler/vllm.cpp/issues/960)). That row fixed the
reference tier. It did not sweep the tree for other readers of the same wide
predicate, and this is one.

## Now

Spec committed, implementation in the same pull request. The gate is CPU-only
and needs no GPU, because the discriminating backend is a fake one.

## 1. Scope

In scope:

- `src/vllm/v1/sample/logits_processor/builtin.cpp` — `apply_logits_processors`
  asks `Backend::DeviceMemoryIsHostAddressable()` instead of
  `Backend::UnifiedMemory()`, and its comment says why the two differ.
- `tests/vllm/v1/sample/test_logits_processor_host_addressable.cpp` — a new
  executable that carries the GB10 predicate pair on a fake backend.
- `tests/CMakeLists.txt` — one `vllm_cpp_add_test` line.

Out of scope:

- The `else` staging arm. It is already correct and does not change.
- Every other reader of `UnifiedMemory()` in the tree. A sweep is its own row,
  because each site needs its own diagnosis and its own red-before test.
- `CudaBackend::DeviceMemoryIsHostAddressable()`. CUDA keeps the base `false`.
  Overriding it is the #1126 question and is not this row.

## 2. The defect, stated as code

`src/vllm/v1/sample/logits_processor/builtin.cpp` reads:

```cpp
const bool unified = b.UnifiedMemory();
...
if (unified) {
  host = static_cast<float*>(logits.data);
}
```

`host` is then handed to an ABI callback (`vllm_logits_processor`,
`include/vllm.h`), which loads and stores through it on the host.

On CUDA/GB10 that predicate is `true`:

- `src/vt/cuda/cuda_backend.cu` constructs `CudaBackend` with
  `caps.pageable_memory_access && caps.integrated`, which holds on GB10.
- `CudaBackend::Alloc` calls `cudaMalloc`. `cudaMallocManaged` appears zero
  times in that file.
- CUDA never overrides `DeviceMemoryIsHostAddressable()`, so it keeps the base
  `false` from `include/vt/backend.h`. ROCm, Metal and Vulkan all override it.

So the callback receives a plain device allocation and dereferences it. The
header states the rule beside the narrow predicate: a backend must opt in,
"because being wrong here hands a device pointer to a host memcpy and
segfaults".

## 3. Anchors

| What | Where |
|---|---|
| The defect | `src/vllm/v1/sample/logits_processor/builtin.cpp`, `apply_logits_processors` |
| The production call site | `src/vllm/v1/sample/sampler.cpp`, `Sampler::forward` |
| The ABI entry point | `vllm_logits_processor` in `include/vllm.h` |
| The narrow predicate and its contract | `Backend::DeviceMemoryIsHostAddressable()` in `include/vt/backend.h` |
| The prior art, in this tree | `ReferenceTierEligible` in `src/vt/op_provider.cpp` |
| The fake-backend idiom | `tests/vllm/v1/sample/test_host_buffer_staging.cpp`, `tests/vt/test_reference_tier.cpp` |

No upstream anchor applies. vLLM has no memory-residency predicate here,
because torch owns residency there through `.to(device)`.

## 4. Design

One predicate changes. `apply_logits_processors` asks the narrow question,
which is the question it actually has: may the host dereference what
`Backend::Alloc` returned? A backend that answers `false` takes the existing
staging arm, which copies down, runs the callbacks, and copies the edits back.

Vulkan, Metal and integrated ROCm answer the narrow predicate `true` and keep
the zero-copy in-place path they have today.

**CPU does not, and that is measured rather than assumed.** `CpuBackend`
(`src/vt/cpu/cpu_backend.cpp`) overrides `UnifiedMemory()` and returns `true`,
and it never overrides `DeviceMemoryIsHostAddressable()`, so it keeps the base
`false`. After this change the CPU path therefore takes the staging bounce: one
copy down and one copy back per step. That cost is accepted here and is argued
under `## 8. Risks`.

**State the size of that bounce exactly, because it is easy to state too
small.** `builtin.cpp:114` computes `total` as `n * vocab` over the WHOLE
`[n, vocab]` slab and both legs copy `total * sizeof(float)`, and `n` is
`logits.shape[0]` — the BATCH ROW COUNT, not the number of requests carrying a
processor. So the cost is O(batch rows x vocab) f32 down and back per decode
step, and it is paid in full whenever ANY request in the batch carries a
processor: one such request in a 256-row batch stages all 256 rows both ways.
What is true is the narrower statement that the DEFAULT path pays nothing:
`builtin.cpp:79` returns on the empty map, `input_batch.cpp:288` inserts into
that map only when `sp.logits_processor.fn != nullptr`, and that field's default
IS `nullptr` (`include/vllm/sampling_params.h:239`, set explicitly by the C ABI
at `src/capi/vllm_c.cpp:556`). A batch in which nobody registered a processor
therefore never reaches the copy at all.

Making `CpuBackend` opt in is the obvious follow-up and it is NOT this row. Two
other consumers read the same predicate. `ReferenceTierEligible`
(`src/vt/op_provider.cpp`) returns earlier on `device == DeviceType::kCPU` and
never reaches it. The direct-upload adoption in
`src/vllm/model_executor/models/qwen3_5_weights.cpp` reads it at two sites
(`:345`, `:365`), and both are early RETURNS — so on CPU that code never adopts
today, and flipping the predicate would UNBLOCK those returns rather than
redirect anything. Whether adoption is even meaningful on CPU is not
established; establishing it is part of what #1748 owes. Widening a global
backend predicate on the project's default device is its own change with its own
measurement.

## 5. Why the CPU suite cannot see the defect

`tests/vllm/v1/sample/test_logits_processors.cpp` runs on `Device{kCPU, 0}`.
`CpuBackend` answers `UnifiedMemory()` `true` over `std::malloc` allocations, so
the wide predicate selects the in-place arm AND the pointer really is host
memory. The wrong question gets the right answer there, and a CPU-only suite
cannot see the difference. A backend that reports unified memory over
allocations the host may not touch can, and CUDA on GB10 is that backend.

The red-before therefore needs a backend that carries the GB10 pair —
`UnifiedMemory() == true`, `DeviceMemoryIsHostAddressable() == false` — over
ordinary host memory, so the wrong arm is observable rather than fatal. That is
the third fake instance `tests/vt/test_reference_tier.cpp` already builds for
the same pair.

## 6. Tests and gates

`tests/vllm/v1/sample/test_logits_processor_host_addressable.cpp`, its own
executable so the `kXPU` backend registration cannot leak into another binary.

Both cases enter through `Sampler::forward`, the production call site, and not
through `apply_logits_processors` directly.

1. **The defect.** A GB10-shaped fake backend. The callback must not receive
   the tensor's own `data` pointer, one staging copy must have happened before
   it ran, the row must ARRIVE CARRYING THE LOGITS, and the sampled token must
   be the one the callback forced — which only holds if the edits were copied
   back to the device allocation.
2. **The working path.** A fake backend that answers both predicates `true`
   keeps the in-place wrap: the callback receives `logits.data` itself and no
   copy happens. This is the regression guard that stops the fix from turning
   into "always stage", and the `ALWAYS-STAGE` row of `## 10. Evidence` is what
   proves it holds that line.

**The content check in case 1 is there because the other two proxies cannot see
a stage-down that carries nothing.** A copy COUNT says a copy happened; an
ADDRESS says it landed elsewhere. Neither reads the buffer, and the callback
overwrites the row without reading it, so a stage-down leg copying zeroes
satisfies both. Measured: on the pre-assertion tree that mutation left this file
`2 | 2 passed`, `assertions: 16 | 16 passed`, `Status: SUCCESS!` (row `A` of
`## 10. Evidence`). `seen.saw` closes that, and row `C` is its red-before.

**`test_logits_processors` is now the second, independent fidelity gate for the
staging arm's CONTENT, and a later change must not silently drop it.** That
suite runs on `Device{kCPU, 0}`, and from this row onward CPU takes the staging
arm rather than the in-place one, so its `BiasCb` read-modify-write
(`logits[bias_token] += 10.0f`) and its null-`fn` byte-identity case both run
over staged bytes. Under the same stage-down mutation it goes
`15 | 13 passed | 2 failed` (row `A2`). Anyone who moves CPU back to the
in-place arm — the `CpuBackend` opt-in owed under #1748 is exactly that change —
removes that coverage, and owes a replacement rather than a green run.

Focused gate:

```sh
ctest --test-dir build -R 'test_logits_processor_host_addressable|test_logits_processors|test_sampler|test_host_buffer_staging' --output-on-failure
```

Full gate: `scripts/agent-preflight.sh`.

## 7. Reachability

`apply_logits_processors` has one production call site, in `Sampler::forward`,
reached from the ABI callback field `vllm_sampling_params::logits_processor`.
The new test drives `Sampler::forward`, so deleting that call site turns it red.
That mutation is run and recorded in `## Evidence`.

## 8. Risks

- **CPU now pays two copies of the WHOLE batch slab per step whenever any
  request in the batch registered a processor.** That is a real cost, it is
  charged to the most common backend, and it scales with the batch row count and
  not with the number of requests that asked for a processor (`## 4` states the
  arithmetic). It is accepted for three reasons. The alternative is a predicate
  that answers a question the code is not asking, which is the whole defect. The
  default path — no processor anywhere in the batch — still pays nothing,
  because the empty map returns first. And the honest repair, `CpuBackend`
  opting in to the narrow predicate, belongs in a change that can measure it.
  Correctness first, per `AGENTS.md ## Gates`: the arm this fix selects is
  always CORRECT and only slower, so deferring the optimisation is legitimate
  where deferring a correctness gap would not be.
- **Hand-rolling `|| device.type == kCPU` here was REJECTED.** It restores the
  zero copy and it re-creates #844: a call site that decides memory residency
  from something other than the backend's own answer. The backend owns that
  answer.
- **The fake backend's memory is real host memory**, so the wrong arm cannot
  segfault in the test. The observable proxy is the pointer identity and the
  copy count. `tests/vllm/v1/sample/test_host_buffer_staging.cpp` uses the same
  proxy for the same reason.

## 9. Stop conditions

Stop and report if the fix needs `CudaBackend` to override
`DeviceMemoryIsHostAddressable()`. That is a different change with a different
blast radius, and it wakes other consumers of the predicate.

## 10. Evidence

Measured on the authoring host, x86_64 Linux, in worktree
`row/V1-LOGITSPROC-HOST-ADDRESSABLE` at base `6354755ba`. Build:
`cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON`, the same default build
type the `cpu` job in `.github/workflows/ci.yml` uses, so `NDEBUG` is not set and
asserts fire. Rows `A`, `A2`, `C` and `ALWAYS-STAGE` were added in the review
repair pass and were measured on this commit's tree; `RED`, `MUTANT` and `GREEN`
were re-measured in the same pass, because adding the content check moved the
assertion counts.

Every row is `test_logits_processor_host_addressable` unless the row says
otherwise. `STAGE-DOWN` is the mutation that keeps the copy but makes it carry
nothing: the stage-down leg copies from a zeroed scratch vector of the same
length instead of from `logits.data`, so the copy COUNT is unchanged and only
the CONTENT is destroyed.

| Run | Tree | Compile rc | doctest summary |
|---|---|---:|---|
| A | b0ff63eb2's test file (no content check) + `STAGE-DOWN` | 0 | `test cases: 2 \| 2 passed \| 0 failed`, `assertions: 16 \| 16 passed \| 0 failed`, `Status: SUCCESS!` |
| A2 | same tree, `test_logits_processors` | 0 | `test cases: 15 \| 13 passed \| 2 failed`, `assertions: 42 \| 39 passed \| 3 failed`, `Status: FAILURE!` |
| C | this test file (content check) + `STAGE-DOWN` | 0 | `test cases: 2 \| 1 passed \| 1 failed`, `assertions: 26 \| 22 passed \| 4 failed`, `Status: FAILURE!` |
| RED | `builtin.cpp` asks `UnifiedMemory()` | 0 | `test cases: 2 \| 1 passed \| 1 failed`, `assertions: 26 \| 24 passed \| 2 failed`, `Status: FAILURE!` |
| ALWAYS-STAGE | `host_addressable` forced `false` | 0 | `test cases: 2 \| 1 passed \| 1 failed`, `assertions: 26 \| 24 passed \| 2 failed`, `Status: FAILURE!` |
| MUTANT | green tree, `sampler.cpp` call site deleted | 0 | `test cases: 2 \| 0 passed \| 2 failed`, `assertions: 2 \| 0 passed \| 2 failed`, `Status: FAILURE!` |
| GREEN | `builtin.cpp` asks `DeviceMemoryIsHostAddressable()`, no mutation | 0 | `test cases: 2 \| 2 passed \| 0 failed`, `assertions: 26 \| 26 passed \| 0 failed`, `Status: SUCCESS!` |

Row `A` is the one that changed this file. It is the reviewer's mutation, rerun
here: the pre-assertion suite was FULLY GREEN against a staging buffer that
carried none of the logits, because a count is not a content check. Row `C` is
the red-before for the four assertions added to close it:

```text
CHECK( seen.saw[j] == baseline[j] )  values: CHECK( 0 == 0.1 )
CHECK( seen.saw[j] == baseline[j] )  values: CHECK( 0 == 5 )
CHECK( seen.saw[j] == baseline[j] )  values: CHECK( 0 == 0.2 )
CHECK( seen.saw[j] == baseline[j] )  values: CHECK( 0 == 0.3 )
```

Row `ALWAYS-STAGE` fails in CASE 2 only, at exactly its two in-place assertions,
while case 1 stays green:

```text
CHECK( static_cast<const void*>(seen.row) == static_cast<const void*>(dev) )
  values: CHECK( 1 == 1 )
CHECK( seen.copies_at_call == copies_before )
  values: CHECK( 1 == 0 )
```

The `1 == 1` there is doctest stringifying two DISTINCT pointers as `bool`; the
assertion failed, so the two addresses differ. The copy count is the unambiguous
half of the pair.

The `RED` row failed for the intended reason and for no other. Both failing
assertions name the defect:

```text
CHECK( static_cast<const void*>(seen.row) != static_cast<const void*>(dev) )
  values: CHECK( 1 != 1 )
CHECK( seen.copies_at_call == copies_before + 1 )
  values: CHECK( 0 == 1 )
```

The callback received the tensor's own allocation and no staging copy happened.

**Case 2 is not what makes the guard discriminating, and saying so would be
unsound.** Case 2's backend answers BOTH predicates true, so it takes the
in-place arm under the buggy predicate and under the fixed one alike; it is
insensitive to #1746 by construction. Its passing in the red run shows only that
the red is not confounded. Case 1's FAILURE is the whole discrimination. Case 2
earns its place against a different mutation — the fix turning into "always
stage" — and that mutation is the `ALWAYS-STAGE` row of the table above: forcing
`host_addressable` to `false` unconditionally fails case 2 at exactly its two
in-place assertions while case 1 stays green.

The compile rc is recorded for every row because a mutation that fails to BUILD
reads as a passing test here. The first red attempt did fail to compile
(`RED_BUILD_RC=1`, a `vector<vector<int64_t>>` against the
`vector<vector<int32_t>>` `SamplerOutput` carries) and was repaired before the
red was taken; the row above is the compiling tree.

`sampler.cpp` was restored after the mutation and verified byte-for-byte:
`sha256sum -c` returns `src/vllm/v1/sample/sampler.cpp: OK` at
`46b378ba7d18889b611b4bc7472e88a3c460f58b27415e5e1285a3ef9a0a8072`, and the
suite is green again on the restored tree.

Adjacent suites on the green tree, all `Status: SUCCESS!`:
`test_logits_processors` 15 cases / 42 assertions, `test_sampler` 21 / 114,
`test_host_buffer_staging` 4 / 913. The first of those now exercises the staging
arm on CPU rather than the in-place one, and its expecteds are unchanged, which
is the copy-back leg observed a second way — and, per row `A2`, the second
fidelity gate on the staged bytes' content.

## Owed

[#1748](https://github.com/mudler/vllm.cpp/issues/1748) — `CpuBackend` never
opts in to `DeviceMemoryIsHostAddressable()`, although its allocations are
ordinary host memory and the class comment says so. Found while grounding this
fix.

**Not repaired here, and for the reason the index row gives rather than the one
an earlier draft of this section gave.** The override does not "flip" the
direct-upload adoption in
`src/vllm/model_executor/models/qwen3_5_weights.cpp`: both of that file's reads
of the predicate (`:345`, `:365`) are early RETURNS, so CPU never adopts today
and the override would UNBLOCK them, not redirect them. Whether adoption is
meaningful on CPU is not established, and establishing it is part of what #1748
owes. The genuine reason to defer is that flipping a global backend predicate on
the project's DEFAULT device is an unmeasured residency and behaviour change,
while this row is a crash-class correctness repair; and because the arm this fix
selects is always correct and only slower, deferring an optimisation here is
legitimate where deferring a correctness gap would not be.

**What #1748 also owes, so its scope is recorded rather than rediscovered:**
row-scoped staging in `apply_logits_processors`. The staging legs copy the whole
`[n, vocab]` slab (`builtin.cpp:114`), so one request with a processor in a
256-row batch stages 256 rows down and back per step. Copying only the rows that
appear in the `procs` map is the obvious mitigation and is independent of the
`CpuBackend` override — either one alone removes most of the cost this row
introduces.

**The `#1748` index row's cost sentence is superseded by this section.** That
row ends "charged only to a request that registered a processor", which
understates the cost by a factor of the batch row count for the reason given
under `## 4`. `.agents/issue-index.md` is append-only and a row may not be
edited or deleted, so the correction lives here, in the spec the row already
names as its owner.

Nothing else. #1746 closes with this change.
