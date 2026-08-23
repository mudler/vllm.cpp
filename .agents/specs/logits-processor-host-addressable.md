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
copy down and one copy back per step, of `[n, vocab]` f32, and only for a
request that actually registered a processor, because the empty map returns
before any of this. That cost is accepted here and is argued under `## 8.
Risks`. Making `CpuBackend` opt in is the obvious follow-up and it is NOT this
row, because two other consumers read the same predicate and would change
behaviour with it: the direct-upload adoption in
`src/vllm/model_executor/models/qwen3_5_weights.cpp` (two sites) and
`ReferenceTierEligible` in `src/vt/op_provider.cpp`. Widening a predicate that
three call sites read is its own change with its own red-before evidence.

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
   it ran, and the sampled token must be the one the callback forced — which
   only holds if the edits were copied back to the device allocation.
2. **The working path.** A fake backend that answers both predicates `true`
   keeps the in-place wrap: the callback receives `logits.data` itself and no
   copy happens. This is the regression guard that stops the fix from turning
   into "always stage".

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

- **CPU now pays two copies per step when a processor is registered.** That is
  a real cost and it is charged to the most common backend. It is accepted for
  three reasons. The alternative is a predicate that answers a question the code
  is not asking, which is the whole defect. The copy is charged only to requests
  that registered a processor. And the honest repair — `CpuBackend` opting in to
  the narrow predicate — belongs in a change that can measure the two other
  consumers of that predicate. Correctness first, per `AGENTS.md ## Gates`.
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
asserts fire.

| Run | Tree | Compile rc | doctest summary |
|---|---|---:|---|
| RED | `builtin.cpp` asks `UnifiedMemory()` | 0 | `test cases: 2 \| 1 passed \| 1 failed`, `assertions: 16 \| 14 passed \| 2 failed`, `Status: FAILURE!` |
| GREEN | `builtin.cpp` asks `DeviceMemoryIsHostAddressable()` | 0 | `test cases: 2 \| 2 passed \| 0 failed`, `assertions: 16 \| 16 passed \| 0 failed`, `Status: SUCCESS!` |
| MUTANT | green tree, `sampler.cpp` call site deleted | 0 | `test cases: 2 \| 0 passed \| 2 failed`, `assertions: 2 \| 0 passed \| 2 failed`, `Status: FAILURE!` |

The red failed for the intended reason and for no other. Both failing
assertions name the defect:

```text
CHECK( static_cast<const void*>(seen.row) != static_cast<const void*>(dev) )
  values: CHECK( 1 != 1 )
CHECK( seen.copies_at_call == copies_before + 1 )
  values: CHECK( 0 == 1 )
```

The callback received the tensor's own allocation and no staging copy happened.
Case 2, the in-place wrap, passed in the red run, which is what makes the guard
discriminating rather than a blanket refusal.

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
is the copy-back leg observed a second way.

## Owed

[#1748](https://github.com/mudler/vllm.cpp/issues/1748) — `CpuBackend` never
opts in to `DeviceMemoryIsHostAddressable()`, although its allocations are
ordinary host memory and the class comment says so. Found while grounding this
fix. Not repaired here: the override also flips the direct-upload adoption in
`src/vllm/model_executor/models/qwen3_5_weights.cpp`, and that needs its own
red-before test and its own measurement. Until it lands, CPU pays the staging
bounce this row introduces.

Nothing else. #1746 closes with this change.
