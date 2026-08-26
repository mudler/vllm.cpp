# VT-REFTIER-HOST-ADDRESSABLE — the portable reference tier gates on the wrong memory property

Issues: [#844](https://github.com/mudler/vllm.cpp/issues/844) (the class),
[#1435](https://github.com/mudler/vllm.cpp/issues/1435) (the second measured instance).

Row: `VT-REFTIER-HOST-ADDRESSABLE`. Seam: `BACKEND-ACCEL-PROVIDER`
(`.agents/backend-matrix.md`), work row `S5` of
[`accelerator-seam-audit.md`](accelerator-seam-audit.md).

## What is wrong

`ReferenceTierEligible` (`src/vt/op_provider.cpp`) decides whether a device may
receive the portable CPU reference tier. It asks `Backend::UnifiedMemory()`.
That is the wrong question, and `include/vt/backend.h` already says so beside
`Backend::DeviceMemoryIsHostAddressable()`:

> CUDA on GB10 reports unified memory because host and device address the same
> physical RAM, yet a plain `cudaMalloc` pointer is still not
> host-dereferenceable.

`CudaBackend::Alloc` calls `cudaMalloc` and `CudaBackend::UnifiedMemory()`
returns the probed `unified_memory_`, which is true on GB10. So on GB10 every op
that has no native CUDA kernel installs the CPU host kernel as a `vt-cpu-ref`
provider, the host kernel dereferences `cudaMalloc` pointers, and the process
receives SIGSEGV. The banner printed immediately before the crash claims the
opposite:

```text
[vt reference-tier] op=MatmulFp8BlockScaled device=cuda has NO native kernel;
                    running the PORTABLE CPU fallback (correct but slow)
```

Two measured instances, both on `dgx:gpu0` (GB10, compute capability 12.1):

1. `vt::QuantFp8Static` on an arch outside the `cutlass-fp8` cell
   ([#960](https://github.com/mudler/vllm.cpp/issues/960), Thor `sm_110`).
   `.agents/specs/vt-fp8-quant-arch-gate.md` `## Outcome` relocated that one
   registration and recorded that it did **not** close the class: "The portable
   reference tier still accepts `DeviceType::kCUDA` tensors and still calls
   itself 'correct but slow' while dereferencing device pointers."
2. `vt::MatmulFp8BlockScaled` on a CUDA build with no CUTLASS headers
   ([#1435](https://github.com/mudler/vllm.cpp/issues/1435), 2026-08-20).
   `tests/vt/test_ops_matmul_fp8_block_cuda` exits 139 with
   `test cases: 0  assertions: 0`. Evidence logs
   `/mnt/nas_share/rc/fp8block-g2/run.log` and `run2.log`.

The second instance is reachable on a default build: `VLLM_CPP_CUTLASS_FETCH`
defaults `OFF` and CUTLASS is not a submodule, so a plain clone plus
`-DVLLM_CPP_CUDA=ON` reproduces it.

The class is not FP8 and it is not CUDA. Any op whose native kernel is
feature-gated reproduces it on any backend whose `Alloc` returns memory the host
cannot dereference. #844 states this in its own words: "The crash is the lucky
outcome. The silent pass is the dangerous one."

## Scope

In scope:

1. `ReferenceTierEligible` asks the narrow question — may the host dereference
   what `Backend::Alloc` returned — instead of the wide one.
2. The refusal that follows names why the portable tier did not rescue the
   caller, so a reader does not have to know that the tier exists.
3. The reference-tier banner stops asserting correctness it cannot hold. It
   states the precondition that now gates it.
4. `MetalBackend` and `RocmBackend` answer the narrow question truthfully, so
   both keep the tier they have today. Neither answer changes any behaviour;
   see [Design](#design).
5. A host-tier test that reproduces the condition without a GPU.

Out of scope, and why:

- **The `vt_cuda_report_feature` half of #1435.** `CMakeLists.txt` reports
  `CUDA feature cutlass-fp8: ENABLED for [121a]` from the architecture
  intersection alone, before the CUTLASS-header detection later in the same
  file, so it reported a capability the build did not contain. That is a
  configure-time reporting defect on a different surface, with a different
  failure mode, and it cannot be exercised at all without a CUDA compiler — the
  authoring host has no `nvcc` and no GPU. Bundling it here would put a change
  that only a leased box can verify inside a change that a CPU box verifies
  completely. Recorded under [`## Owed`](#owed).
- **Copy-in and copy-out instead of refusal.** Rejected under
  [Design](#design).
- **Registering a CUDA kernel for `kMatmulFp8BlockScaled` on a CUTLASS-less
  build.** That is the FP8 row's question, not this one. This row makes the
  missing kernel a named refusal; it does not supply one.

## Upstream anchors

vLLM has no mirror for this seam and cannot have one: `CustomOp.forward_native`
(`vllm/model_executor/custom_op.py:138` at the pinned oracle) is pure torch, and
torch tensors carry their device, so torch never dispatches a CPU body against
CUDA storage — it raises. The reference tier is the vllm.cpp original that
stands in for `forward_native` (inventory deviation §9.1), and this row restores
the property torch gets for free: a portable body never runs on storage it
cannot address.

## Design

**The predicate already exists.** `Backend::DeviceMemoryIsHostAddressable()`
(`include/vt/backend.h`) is documented as "STRICTLY NARROWER than
`UnifiedMemory()`, and the difference is the whole reason it exists", and its
default is `false` because "being wrong here hands a device pointer to a host
memcpy and segfaults". That is precisely the reference tier's precondition. The
fix is to ask it. No new seam, no new virtual, no `DeviceType` branch — the
audit's rule that eligibility must never key on the device type is kept.

**What each backend answers, and what changes.**

| Backend | `Alloc` | `UnifiedMemory()` | host addressable | tier before | tier after |
|---|---|---|---|---|---|
| CPU | host | true | n/a — the tier's source device is never a target | no | no |
| CUDA | `cudaMalloc` | probed; true on GB10 | **false** | **yes** | **no** |
| Vulkan | host-visible and host-coherent, persistently mapped | true only where a combined host-visible, host-coherent, DEVICE_LOCAL type exists | true, unconditionally (already overridden) | yes, **except** where unified is false | yes, always |
| Metal | `MTLResourceStorageModeShared` | `hasUnifiedMemory` | true, once it answers | yes | yes |
| ROCm | `hipMallocManaged` when integrated and managed-capable, else `hipMalloc` | `managed_alloc` or `pageable_memory_access && integrated` | equal to `UnifiedMemory()` | yes | yes |

Two cells move. CUDA is the crash, and it loses the tier. **Vulkan widens**, and
this is not a side effect worth hiding: `VulkanBackend::DeviceMemoryIsHostAddressable()`
already returns `true` unconditionally, while `VulkanBackend::UnifiedMemory()`
reads `VulkanContext::unified_memory()`, which the context sets from
`FindMemoryType(mem, ~0u, kHostFlags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) >= 0`.
On a device with no such combined type that flag is false, the context falls back
to `FindMemoryType(mem, ~0u, kHostFlags)` and then `VT_CHECK`s that one was
found. So every allocation is host-visible and host-coherent regardless, and the
tier that used to be withheld on that device is now installed. The widening is
sound because the `VT_CHECK` is what guarantees it, not because the tier is
harmless.

Metal and ROCm need the one-line override so that they do not lose the tier as a
side effect; both overrides are by construction equal to what those backends
answer today, so their behaviour is byte-identical. ROCm's answer is its own `unified_memory_` rather than
`managed_alloc_`: the registrar widens `unified_memory_` with
`pageable_memory_access && integrated`, and on that branch `hipMalloc` memory is
host-dereferenceable by the same attribute, which the ROCm backend's own comment
records as measured on gfx1151 and gfx1103.

**Why refusing, and not copying in and out.** The tier's contract is that it
returns the SAME host kernel pointer the CPU device dispatches
(`ref.fn = src->fn`). A staging fallback cannot live behind that contract: it
would have to wrap every op's argument pack, which is per-op and untyped at this
seam, and it would turn a missing kernel into a silent bandwidth cliff on the
model path. AGENTS.md already settles the polarity — "Refuse an unimplemented
arm with a message that names the missing part" — and #844's "What done looks
like" asks for the refusal. A staging tier is a larger, separate design, and
nothing in the tree asks for it.

**Why the refusal message grows a clause.** With the gate corrected, the caller
gets `no kernel for op MatmulFp8BlockScaled (id N) on device cuda (type 1)`.
That is already a named refusal, and it is the message #1435 asked for. It does
not say why the portable tier — which the reader may know exists, and which used
to answer — did not run. One clause names the reason and keeps the reader out of
`op_provider.cpp`.

**Why the banner's wording changes.** After the gate is corrected the banner
only ever prints where the host may address device memory, so "correct" is true
again. It is still the wrong sentence to leave behind: it asserts correctness
without naming what makes it correct, and both #844 and `docs/USAGE.md` quoted
those three words as the thing that misled a reader. `docs/USAGE.md` no longer
quotes them: [#1491](https://github.com/mudler/vllm.cpp/pull/1491) retired the
section that did, which is why this row edits the banner and not that file. #844
still carries the quote. The replacement names the precondition, which is a fact
the reader can check.

## Risks

1. **A backend that is host-addressable but does not say so loses the tier.**
   The measured population is five backends and this spec enumerates all five
   above. Metal and ROCm gain the override in this change and are byte-identical;
   CPU is not a target; CUDA is the defect. **Vulkan is not byte-identical** — it
   widens, on a device with no combined host-visible, host-coherent, DEVICE_LOCAL
   memory type, for the reason argued under [Design](#design). A backend added
   later gets the safe default, which is the polarity `backend.h` chose
   deliberately.
2. **Metal and ROCm are not compiled anywhere this change can reach.** CI builds
   CPU, CUDA, Vulkan and Windows; there is no macOS or ROCm job, and this host
   has neither toolchain. The two edits are therefore one line each, placed
   beside the existing `UnifiedMemory()` override in the same class, with the
   same signature shape. Their existing tests
   (`tests/vt/test_metal_backend.cpp`, `tests/vt/test_rocm_backend.cpp`) assert
   tier eligibility and continue to hold, because the override reproduces the
   value those tests already compare against.
3. **A CUDA op that relied on the tier starts refusing.** That is the intent,
   and it cannot regress a working path: the tier on CUDA either crashed or, on
   an op whose operands happened to be host tensors, silently compared the CPU
   against itself. #844 names that second outcome as the dangerous one. No op is
   known to depend on it; `RefTierHits()` is documented as MUST-be-zero in any
   performance arm.
4. **This does not make the block-wise FP8 arm more finished.** It has no token
   gate, no speed claim, and correctness only on the seven shapes run under
   #1437. This row changes a crash into a refusal and claims nothing else.

## Tests and gates

1. **Red first.** `tests/vt/test_reference_tier.cpp` gains a fake backend that
   reports `UnifiedMemory() == true` and
   `DeviceMemoryIsHostAddressable() == false` — the GB10 CUDA shape, expressed
   without a GPU. The new case requires that the tier is refused, that eager
   registration installs nothing, that `GetOp` throws, and that the message
   names the op and the device. On the unfixed tree the tier installs and
   `GetOp` returns the host kernel, so the case fails on the assertion that
   states the invariant.
2. **Focused green.** `ctest --test-dir build -R test_reference_tier -V`, read
   with ANSI stripped, checking `Status:` and `assertions:` together.
3. **Mutation.** Revert the predicate in a scratch copy and rerun the focused
   gate. The new case must go red. Print `git diff --stat` and any compiler
   error with the result, because a mutation that never applied and a mutation
   that failed to build both read as a passing test.
4. **Reachability, mutated at the production call site.** The new refusal clause
   is not proved by calling `ReferenceTierRefusalReason` directly. Delete the
   clause from the `VT_CHECK` inside `Resolve` — the production path every
   `vt::GetOp` takes — and `tests/vt/test_reference_tier.cpp` goes red on the
   `host-addressable` assertion. Run by the fresh reviewer, recorded here because
   "nothing lands dead" asks for exactly this and the implementer did not run it.

5. **No regression in the seam's own suites.**
   `ctest --test-dir build -R 'test_op_provider|test_reference_tier|test_backend_cross_device'`.
6. **Record gates.** `scripts/check-agent-record.py`,
   `scripts/check-doc-checkpoint.py`, `scripts/check-public-doc-tables.py`,
   `scripts/check-gate-commands.py --check`,
   `scripts/check-commit-style.py --range origin/main..HEAD`,
   `scripts/check-commit-trailers.py --range origin/main..HEAD`.
7. **Not run here, and named as such.** No CUDA build, no GPU. The defect this
   row fixes was measured on GB10 and the fix is verified on the host tier
   against a fake backend that reproduces the exact property pair. A CUDA
   re-measurement — the block-wise FP8 test refusing instead of exiting 139 —
   is owed and is listed under `## Owed`.

## Stop conditions

- Stop and return `NEEDS_DECISION` if a backend in the tree is found to be
  host-addressable and cannot say so through
  `Backend::DeviceMemoryIsHostAddressable()`.
- Stop if the focused gate cannot be made red before the fix. A test that is
  green on the unfixed tree is measuring something else.
- Do not widen what the reference tier accepts to keep a test green.

## Evidence

Red, fixed, and mutation evidence are recorded in the pull request body, which
is the landed commit message. The two GB10 logs that motivated the row are
`/mnt/nas_share/rc/fp8block-g2/run.log` and `run2.log`.

## Owed

- [#1435](https://github.com/mudler/vllm.cpp/issues/1435) stays OPEN after this
  row. Its second defect — `vt_cuda_report_feature` reporting
  `CUDA feature cutlass-fp8: ENABLED` from the architecture intersection alone,
  before the CUTLASS-header detection later in `CMakeLists.txt` — is not fixed
  here. It needs a CUDA configure to exercise, which this host cannot run.
  `cmake/CudaArchFeatures.cmake` opens by naming this exact class as what the
  feature table exists to prevent, so the diagnostic asserts the opposite of the
  truth.

  REPRODUCED AGAIN, on a second box, while this row's `nvcc` check ran. Job
  `645bf395-23fc-408f-a9ad-b9823885622c` on `thor:gpu0` (NVIDIA Thor, CUDA 13.0
  V13.0.88, aarch64) configured this branch at `817e1769c` with
  `-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a` and NO CUTLASS at all,
  and the configure printed:

  ```text
  CUDA feature cutlass-nvfp4: ENABLED for [121a]
  CUDA feature cutlass-fp8: ENABLED for [121a]
  ```

  So the defect is not specific to `dgx:gpu0` and not specific to `cutlass-fp8`.
  Every CUTLASS-dependent feature cell reports ENABLED on a build that contains
  no CUTLASS. Evidence: `/mnt/nas_share/rc/reftier-1435/nvcc-check.sh` and
  `/workspace/reftier-1435/nvcc-configure.log` on that worker.
- [#1482](https://github.com/mudler/vllm.cpp/issues/1482) carries the remainder
  of [#844](https://github.com/mudler/vllm.cpp/issues/844), and this row owns it.
  Two of #844's four "what done looks like" items are not done here: item 1 also
  wants the refusal to name the BUILD FEATURE that would have provided the
  kernel, and item 4 wants a warning at engine construction rather than only at
  configure time. The pull request carries `Fixes #844`, because the defect in
  that issue's title — the SIGSEGV, and the banner that claimed correctness — is
  fixed, and leaving the issue open would misreport that. The alternative,
  `Refs #844` with the issue left open, was rejected for that reason. Splitting
  the remainder is what keeps it owned by an OPEN issue instead of a closed one.
- A CUDA re-run of `tests/vt/test_ops_matmul_fp8_block_cuda` on a CUTLASS-less
  build, to record the refusal replacing exit 139. Owned by this row.
- **This row widened the reach of `VT_ADOPT_DEVICE_BYTES`, and nobody has
  measured the new arms.** Moving `ReferenceTierEligible` onto
  `Backend::DeviceMemoryIsHostAddressable()` required truthful overrides so no
  backend silently lost the tier, and `MetalBackend` and `RocmBackend` now answer
  the predicate. That predicate is also what gates the weight-loader lever, at
  both `AdoptDeviceBytesAsHost` branches in
  `src/vllm/model_executor/models/qwen3_5_weights.cpp` — so since this row landed,
  the lever ACTS on Apple silicon (`MetalContext::unified_memory()`, i.e.
  `dev.hasUnifiedMemory`) and on an integrated ROCm part (`unified_memory_`, i.e.
  a managed allocator or `PageableMemoryAccess` on an integrated device). Every
  number recorded for that lever is GB10 through Vulkan.
  [#1502](https://github.com/mudler/vllm.cpp/issues/1502) is the DOCUMENT half and
  is fixed: `docs/ENVIRONMENT.md` had said "Vulkan today" and "No effect on
  CUDA/CPU/Metal", and it now separates the backends the lever is MEASURED on from
  the backends that merely satisfy the predicate. What stays owed is the
  MEASUREMENT itself on the two new arms, which needs an Apple-silicon box or an
  integrated AMD part and cannot be taken here. Owned by this row.
- **`CudaBackend`'s `DeviceMemoryIsHostAddressable()` is now CHECKED, structurally
  rather than by observation, and
  [#1635](https://github.com/mudler/vllm.cpp/issues/1635) is discharged by that.**
  `tests/vllm/platforms/test_platform.cpp` was cited as the pin and is not one:
  `FakeUnifiedAddressablePlatform` reports `device_type() == kCUDA` while its
  `backend()` returns `vt::GetBackend(DeviceType::kCPU)`, so the `CHECK_FALSE`
  reads the CPU backend. That fixture's comment now says so
  ([#1639](https://github.com/mudler/vllm.cpp/pull/1639)), and the `#1502` row in
  `.agents/issue-index.md` keeps the wrong citation, because that index is
  append-only and can never be edited.

  **A runtime pin was rejected, and the reason is a CI fact rather than a
  preference.** No job in `.github/workflows/ci.yml` has a GPU. `cuda-fat-build`
  is the only job with a CUDA toolchain; it runs the `nvidia/cuda:13.3.0-devel`
  container on `ubuntu-latest`, it configures `-DVLLM_CPP_BUILD_TESTS=OFF`, and
  it builds the `vllm` target alone. `CudaBackend`'s registrar returns early when
  `cudaGetDeviceCount` reports no device, so `vt::GetBackend(kCUDA)` throws on
  every machine this project's CI owns. A `TEST_CASE` that reads the real backend
  would therefore report a skip on every lane forever, and a skip reads as a
  pass — the same shape of evidence #1635 was filed about.

  **What landed instead is one claim split into two halves, each checked on a
  surface that executes it.**

  1. *`CudaBackend`'s answer IS the base default.* A `static_assert` beside the
     class in `src/vt/cuda/cuda_backend.cu` requires
     `decltype(&CudaBackend::DeviceMemoryIsHostAddressable)` to be
     `bool (Backend::*)() const`. Taking the address of an INHERITED member
     through a derived class yields a pointer-to-member of the class that
     DECLARES it, so that type holds exactly while `CudaBackend` declares no
     override, and it becomes `bool (CudaBackend::*)() const` the moment somebody
     adds one. `cuda-fat-build` compiles this translation unit on every push, so
     the check runs where the type exists. It fires on ANY override, including
     one that returns `false`, because an override invalidates the record's
     reasoning whatever it returns.
  2. *The base default is `false`.* `tests/vt/test_backend.cpp` gains a `Backend`
     subclass that implements the pure virtuals and deliberately declares no
     `DeviceMemoryIsHostAddressable`, and requires `false`. Every other fake in
     the tree overrides that method and therefore measures its own override, so
     this subclass is the tree's only reader of the default itself. It runs on
     every host lane.

  Neither half restates the thing it measures, and both were mutation-proven:
  adding an override to `CudaBackend` fails the compile, and flipping the base
  default in `include/vt/backend.h` fails the test. The evidence is in the pull
  request body, which is the landed commit message.

  **What is still NOT pinned, stated precisely.** No CI surface observes a live
  `CudaBackend` object answering the question, because no CI surface has a
  device. `tests/vt/test_cuda_backend.cpp` gains that observation, following the
  skip convention every case in that file already uses, and it asserts nothing
  until somebody runs it on a leased GPU. It is the empirical belt to the two
  structural braces above. It is not what holds the answer on a push, and citing
  it as though it were would repeat #1635 exactly.

## Now

`ACTIVE`. The spec is committed before the implementation, in the same pull
request, so the commit order proves the order of work. The row is deliberately
not added to an area matrix: it is a bug fix to the `BACKEND-ACCEL-PROVIDER`
seam that already carries an `ACTIVE` matrix row, and adding a second row for
one predicate would put a shared-file edit in the path of a one-line fix.
