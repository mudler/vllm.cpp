# SPEC — `Platform::get_device_capability()` is a CUDA-SM value, or it is absent

Issue: [#1823](https://github.com/mudler/vllm.cpp/issues/1823)
Owning row: `BACKEND-ATTN-REGISTRY` (`.agents/backend-matrix.md`), which owns the
shared attention selector seam and filed this defect against itself under
`## Found in flow, filed, not fixed here` in
[attn-validate-configuration.md](attn-validate-configuration.md).
Scope of this change: the UNIT of the Platform capability value, and the two
platforms that answered it in a foreign unit. Nothing else.

## Now

`BACKEND-ATTN-REGISTRY` stays in its recorded state. This change repairs a
defect in the capability layer that row landed; it does not move the row's
lifecycle, add a backend, or change the selector's algorithm.

## M0 — reconciliation, performed before any code

Checked at `origin/main` `df1ee2058`:

- `git log --oneline --grep '1823'` returns no landing for this fix.
- `git log -S'get_device_capability' --oneline -- src/vllm/platforms/metal.cpp
  src/vllm/platforms/vulkan.cpp` returns exactly the two skeleton commits that
  introduced the values (`13bb7241f` Metal W0, `1cb5f643f` Vulkan W0). Neither
  value has ever been revised.
- `gh pr list --state open --limit 100`: 21 open pull requests, none touching
  `src/vllm/platforms/**`, `src/vllm/v1/attention/**` or the Metal/Vulkan tests.
- `git branch -r | grep -iE '1823|capability-unit'`: no match.
- The issue index already carries #1823 at line 673 owned by
  `BACKEND-ATTN-REGISTRY`. The index is append-only and a row is never edited, so
  this spec is linked from the existing row rather than appended again.

## The oracle determination

**vLLM owns attention-backend selection, and it owns the unit of the capability
value.** Read at the parity pin `555967922`.

1. `vllm/platforms/interface.py:420-431 Platform.get_device_capability` — the
   docstring defines the unit outright: *"Stateless version of
   [torch.cuda.get_device_capability][]"*. The base returns `None`.
2. `vllm/v1/attention/backend.py:366-367
   AttentionBackend.validate_configuration` calls
   `cls.supports_compute_capability(device_capability)` **unconditionally** —
   there is no `is not None` guard, because the caller guarantees the unit.
3. The callers are the guarantee. `validate_configuration` is reached from
   exactly three places, and every one of them is an NVIDIA or AMD selector:
   `vllm/platforms/cuda.py:381` (`CudaPlatform.get_valid_backends`) and
   `cuda.py:410` (`CudaPlatform.get_attn_backend_cls`, immediately after
   `assert device_capability is not None` at `cuda.py:404-405`);
   `vllm/platforms/rocm.py:531` and `:558`; and
   `vllm/v1/attention/backends/mla/prefill/selector.py:125,167`, the CUDA MLA
   prefill selector. **Every other platform implements `get_attn_backend_cls`
   itself and never evaluates the predicate at all** — `CpuPlatform.get_attn_backend_cls`
   (`vllm/platforms/cpu.py:75-87`) returns `CPU_ATTN` outright.
4. `vllm/v1/attention/backends/flash_attn.py:200-202
   FlashAttentionBackend.supports_compute_capability` is therefore a statement
   about NVIDIA SM versions, by construction of who may ask it.
5. **Upstream met this exact defect and recorded the answer.**
   `vllm/platforms/xpu.py:228-236 XpuPlatform.get_device_capability`:

   ```python
   def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
       # capacity format differs from cuda's and will cause unexpected
       # failure, so use None directly
       return None
   ```

   A platform whose capability is not in CUDA's unit reports **None**. That is
   the mirror, in upstream's own words, for the situation this issue describes.

**What this tree does differently, and why the rule still lands where upstream
puts it.** `SelectAttentionBackendName`
(`src/vllm/v1/attention/registry.cpp:90-140`) is ONE selector shared by every
`DeviceType`, which is the recorded attn-registry seam. Under that shape the SM
predicate is asked of every platform, so upstream's per-caller guarantee has to
become a per-platform contract on the value: **absent means "no CUDA-SM answer",
and the predicate is skipped — which is what upstream's CPU/XPU/TPU platforms get
by never reaching `validate_configuration`; present means a CUDA-SM value — which
is what upstream's CUDA/ROCm platforms get from `assert device_capability is not
None`.** With that contract the shared selector is behaviourally identical to
upstream's per-platform selectors. Without it the comparison is undefined, which
is the bug.

Rejected: making `FlashAttentionBackend` device-type aware. Upstream's predicate
takes a capability and nothing else (`flash_attn.py:201`), and a backend that
branches on `DeviceType` would be an invention with no upstream anchor, in the
one class that is a 1:1 port.

Rejected: dropping the `capability.present()` guard at
`src/vllm/v1/attention/backend.cpp:197` to match upstream's unguarded call. The
guard is what stands in for upstream's caller-side `assert`; removing it would
refuse FLASH_ATTN on kCPU and kTENSTORRENT, which upstream serves.

## Design

`MetalPlatform::get_device_capability` and `VulkanPlatform::get_device_capability`
return an ABSENT `DeviceCapability`, mirroring `XpuPlatform` and citing it.

The device-specific numbers are not lost and were never the Platform seam's to
report. The Apple GPU family stays on
`vt::Backend::DeviceCapabilityMajor/Minor` (`src/vt/metal/metal_backend.mm:127-130`)
and the Vulkan API version stays on `vt::Backend::DeviceCapabilityMajor/Minor`
(`src/vt/vulkan/vulkan_backend.cpp:144-145`) and `VulkanContext::api_major/minor`.
Nothing in the tree consumed the Platform-level value for either device: the only
consumer of `Platform::has_device_capability` is
`CudaPlatform::supports_fp8` (`src/vllm/platforms/cuda.cpp:47`,
`has_device_capability(8, 9)`), an unmistakably CUDA-SM question.

`include/vllm/platforms/interface.h` states the unit contract on the virtual, so
the next platform does not rediscover it, and
`src/vllm/v1/attention/backend.cpp` replaces the premise the issue identified as
false with the one that is now true and enforced.

## What each device type selects, before and after

| DeviceType | capability before | predicate | selects before | selects after |
|---|---|---|---|---|
| kCUDA | SM, e.g. `{12,1}` | applied, correct unit | FLASH_ATTN | UNCHANGED |
| kROCM | gfx-derived, e.g. `{9,4}` | applied, correct unit | per priority list | UNCHANGED |
| kCPU | absent `{-1,-1}` | skipped | FLASH_ATTN / CPU_ATTN | UNCHANGED |
| kTENSTORRENT | absent `{-1,-1}` | skipped | per priority list | UNCHANGED |
| kMETAL | Apple family `{N,0}` | applied to a FOREIGN unit | FLASH_ATTN only when family >= 8, by coincidence; THROWS below it | absent, skipped, FLASH_ATTN on every Apple family |
| kVULKAN | Vulkan API `{1,4}` | applied to a FOREIGN unit | **THROWS ALWAYS** (`1 >= 8` is false on every Vulkan device that will ever exist) | absent, skipped, FLASH_ATTN |

Two device types change, and both changes are the repair. The issue estimated
four; measured, kCPU and kTENSTORRENT already report an absent capability
(`src/vllm/platforms/cpu.cpp:18`, `src/vllm/platforms/tenstorrent.cpp:34`) and do
not move at all. No selection rework is required, so this row does not need to
return `NEEDS_DECISION`.

## The second instance, found while grounding the premise

The issue asked for `vulkan.cpp` and `tenstorrent.cpp` to be checked against the
same premise. Tenstorrent is clean. **Vulkan carries the identical defect and is
strictly worse than Metal's**: `VulkanPlatform::get_attn_backend_priority`
returns `{"FLASH_ATTN"}` while the platform reports the Vulkan API version as its
capability, so `major=1` fails an SM-8.0 bar on *every* device, always — there is
no coincidence to save it. It went unseen because
`tests/vt/test_vulkan_backend.cpp` asserted that FLASH_ATTN is NAMED in the
priority list and never that the selector REACHES it. That is the same gap the
Metal test had, and closing it on both lanes is part of this change.

## Tests

1. `tests/vt/test_vulkan_backend.cpp` — the red-before. The platform case now
   prints the capability it reports and asserts
   `SelectAttentionBackendName(p) == "FLASH_ATTN"`. Runs on the existing
   `build-test-vulkan` lane, GPU-free on lavapipe, and locally.
2. `tests/vt/test_metal_backend.cpp` — the assertion at :153-154 that Metal
   reports a `present()` capability is INVERTED to the corrected contract, and
   the Apple family is asserted where it actually lives, on `vt::Backend`. The
   existing `SelectAttentionBackendName(p) == "FLASH_ATTN"` at :170 is the
   green-after. Runs on `macos-metal-mlx`.
3. `tests/vllm/platforms/test_platform.cpp` — the class gate. Every REGISTERED
   platform either reports an absent capability or is a platform whose
   capability is a CUDA-SM value (kCUDA/kROCM). A future platform that reports a
   foreign unit reds on whichever lane registers it. This is deliberately a
   contract test rather than a per-platform list, so it does not have to be
   edited when a platform is added.

## Gates

- `build-test-vulkan` (the red-before/green-after this change can run locally).
- `macos-metal-mlx` — RED on `main` at `df1ee2058`; clearing it is the
  deliverable, read at JOB level from a dispatched run.
- `build-test-cpu` full ctest, for the shared `interface.h`/`backend.cpp` edits.
- `scripts/agent-preflight.sh`.

## Evidence

Host: this Linux workstation, GCC, `-DVLLM_CPP_VULKAN=ON`, lavapipe (`lvp_icd.json`)
software ICD — the same GPU-free arrangement `build-test-vulkan` uses.

**RED-BEFORE, run locally at base `df1ee2058` with only the missing assertion
added and no fix applied.** The capability numbers are printed by the case
itself, so the mechanism is in the log rather than in the argument:

```
tests/vt/test_vulkan_backend.cpp:487: MESSAGE: kVULKAN Platform::get_device_capability() present=true major=1 minor=4
tests/vt/test_vulkan_backend.cpp:489: ERROR: CHECK( vllm::v1::SelectAttentionBackendName(p) == "FLASH_ATTN" ) THREW exception:
  "No valid attention backend for device type 3 from {FLASH_ATTN: [compute capability not supported]} (use_mla=false, use_sparse=false)"
[doctest] test cases:  1 |  0 passed | 1 failed | 34 skipped
[doctest] assertions: 15 | 14 passed | 1 failed |
[doctest] Status: FAILURE!
```

`major=1 minor=4` is the Vulkan API version, compared against an SM-8.0 bar. The
message is the same one `test_metal_backend.cpp:170` produced in run
[32668677681](https://github.com/mudler/vllm.cpp/actions/runs/32668677681), with
device type 3 (kVULKAN) instead of 2 (kMETAL). One root cause, two platforms.

**GREEN-AFTER, same host, same build directory.**

```
tests/vt/test_vulkan_backend.cpp:503: MESSAGE: kVULKAN Platform::get_device_capability() present=false major=-1 minor=-1; vt::Backend API version 1.4
tests/vt/test_vulkan_backend.cpp:507: SUCCESS: CHECK( vllm::v1::SelectAttentionBackendName(p) == "FLASH_ATTN" ) is correct!
```

The second half of that line is the point: the API version is still 1.4 and still
probed. It moved to the seam that owns Vulkan-unit questions; it was not deleted
to make a test pass.

| Suite | compile rc | test cases | assertions | Status | `[SKIP]` lines |
|---|---|---|---|---|---|
| `test_vulkan_backend` | 0 | 35 / 35 passed / 0 failed / 0 skipped | 2109 / 2109 passed | SUCCESS! | 0 |
| `test_backend_cross_device` | 0 | 25 / 25 passed / 0 failed / 0 skipped | 80140 / 80140 passed | SUCCESS! | 0 |
| `test_platform` (Vulkan tier) | 0 | 15 / 15 passed / 0 failed / 0 skipped | 118 / 118 passed | SUCCESS! | 0 |

`src/vllm/platforms/metal.cpp` and `tests/vt/test_metal_backend.cpp` are plain
C++ and both pass `g++ -fsyntax-only -std=c++20` on this Linux host, rc 0. That
is a syntax and type gate, not a build against the real Metal SDK; the executing
verdict comes from `macos-metal-mlx`.

**Mutation.** Restoring the defective `VulkanPlatform::get_device_capability`
body (1 hunk, `git diff --stat` 21 insertions / 7 deletions, compile rc 0) reds
BOTH new gates, and the tree was restored byte-identical afterwards
(`diff -q` IDENTICAL, both suites green again):

```
test_platform:        15 cases | 14 passed | 1 failed        Status: FAILURE!
test_vulkan_backend:  tests/vt/test_vulkan_backend.cpp:458: ERROR: CHECK_FALSE( p.get_device_capability().present() ) is NOT correct!
                      tests/vt/test_vulkan_backend.cpp:507: ERROR: ... THREW "compute capability not supported"
```

## Outcome

What the bug actually was, for the next reader: **not a wrong threshold, and not
a platform that "cannot answer".** `DeviceCapability::present()` was doing a job
it cannot do. It reports WHETHER a platform answered and can say nothing about
the UNIT of the answer, and the comment at `backend.cpp:197` read the first as
the second. Two platforms answered helpfully, in units natural to their own
device, and helpfulness was the defect.

The reason Metal looked fine for four days is worth keeping: Apple family 9 on
the M4 gate box clears `>= (8, 0)`. A gate that passes because two unrelated
integers happened to order correctly is not a passing gate, and the value that
made it pass is the one that hid it.

Two smaller things fell out of the same discipline. The doctest `-tc` filter
splits on commas, so the new case is named without one — with a comma it reported
`0 cases ran ... SUCCESS!`. And doctest stringifies a bare `char*` as a bool, so
`vt::DeviceTypeName(type)` printed `1` for every platform until it was wrapped in
`std::string`; a diagnostic line that cannot tell two platforms apart is worse
than no line.

## Stop conditions

- Stop and return `NEEDS_DECISION` if repairing the unit turns out to require
  changing what selection means for kCPU or kTENSTORRENT. It does not: both
  already report an absent capability and neither moves.
- Stop if any consumer of `Platform::get_device_capability()` for kMETAL or
  kVULKAN is found beyond the selector. None exists at `df1ee2058`.
