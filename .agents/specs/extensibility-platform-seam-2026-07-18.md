# Extensibility-first plan — the Platform seam (order-1 priority, 2026-07-18)

**STATUS 2026-07-18 — Item 1 LANDED (CPU-gated, DGX model-gate pending),
`CLAIM-BACKEND-PLATFORM-1` / row `BACKEND-PLATFORM` `ACTIVE`.** The Platform seam
is extracted: `include/vllm/platforms/interface.h` (`class Platform` composes
`vt::Backend`) + `src/vllm/platforms/{platform,cpu,cuda}.cpp` (`CudaPlatform`/
`CpuPlatform`, `RegisterPlatform`/`CurrentPlatform` self-registered per
`DeviceType`) + `tests/vllm/platforms/test_platform.cpp`. The 7 true
platform/memory-model/residency `device.type == kCUDA` sites (runner KV-residency
+ async combine/scatter, model_registry decode-graph gate, qwen3_5
`ResidentWeight`/`ResidentWeightF32`) route through
**`GetPlatform(<obj>.device.type).is_cuda()`** — keyed on the OBJECT's own device;
kernel-shape dispatch branches deliberately left for items 4/5;
`get_attn_backend_priority()` is a stub for item 4; `qwen3_5.cpp` NOT split
(item 5). Behavior-preserving: clean CPU `-Werror` build, `test_platform` PASS,
full CPU CTest green, tools 164/164, checkers green; DGX 27B 235/235 + 35B
315/315 confirmed. Items 2/4/5/6 build on this.

**REGRESSION FIX 2026-07-18 (`CLAIM-BACKEND-PLATFORM-1`).** The 7 per-object
sites were wrongly using the process-global `CurrentPlatform().is_cuda()`, which
is accelerator-first and so mis-routes a CPU queue/tensor on a GPU box into the
CUDA branch (red DGX CPU tests). Fixed to per-object
`GetPlatform(<obj>.device.type).is_cuda()`; `test_platform` fallback assertion
tier-guarded; see the Risks/decisions correction below and the parity ledger.


**STATUS 2026-07-19 — Item 5 (model self-registration + per-arch entry-point TU
split) LANDED (CPU-gated, DGX model-gate pending), `CLAIM-MODEL-SELFREG-1`,
`MODEL-FACTORY-registry`.** The fixed `constexpr std::array<ModelRegistration,2>
kRegistrations` (`model_registry.cpp`) is replaced by a `REGISTER_VLLM_MODEL(...)`
static-`Registrar` self-registration idiom
(`include/vllm/model_executor/models/model_registry.h:167-189`), copying the
proven `RegisterOp`/`RegisterBackend`/`RegisterPlatform` pattern. The Qwen
dense/MoE arch-specific registry entry points (LoadedModel subclass +
load/prepare/forward + factory + Make/Borrow adapters) moved out of the
`model_registry.cpp` monolith into NEW per-variant TUs
`src/vllm/model_executor/models/qwen3_5_dense.cpp` + `qwen3_5_moe.cpp`, over a NEW
shared `qwen3_5_common.{h,cpp}` (ModelInfo, config hook, KV-cache builder,
host-logits carrier, borrowed-weights tag); `model_registry.{h,cpp}` is now the
GENERIC family-agnostic registry. **Measured additive-ness: adding the next model
= 1 new TU + 1 `REGISTER_VLLM_MODEL` line, ZERO edit to a shared array or to
`model_registry.cpp` (before: edit the fixed `kRegistrations` array AND add glue
inside the `model_registry.cpp`/`qwen3_5.cpp` monoliths).** Scope-disciplined: the
DEEP `qwen3_5.cpp` shared-machinery factoring (DevicePool/matmul/GDN +
`Qwen3_5Model::`/`Qwen3_5DenseModel::` bodies into `qwen3_5_common`) is a deferred
follow-up — the prioritized deliverable was the self-registration + arch-entry-
point separation, kept as a reviewable, byte-identical, no-numeric-change refactor
(`qwen3_5.cpp` UNTOUCHED). Deviation: C++ static-init order across TUs is
unspecified → the registry canonically sorts by architecture name on first query
(cosmetic; resolution is order-independent). Behavior-preserving: clean CPU
`-Werror` build, `test_model_registry` self-registration extension PASS, full CPU
CTest green, tools 164/164, checkers green; **DGX-confirmed 2026-07-19 @ `669679a`
(production flags CUTLASS sm120a + FA2 sm_121a + Triton AOT): clean CUDA `-Werror`
build 0 warnings, 27B `test_qwen27_paged_engine` 235/235 + 35B
`test_qwen36_paged_engine` 315/315 token-exact, `compute-sanitizer memcheck` 35B 0
errors.** Items 2/4 remain.

**STATUS 2026-07-19 — Item 2 (residency/memory-model as a CONSUMED
`Platform::residency_policy()` capability) LANDED CPU-side (DGX gate PENDING),
`CLAIM-BACKEND-PLATFORM-2`, `BACKEND-PLATFORM`.** The residency seam is
COMPLETE: the host-weight-release / load-stream-interleave / DevicePool-cap
decisions are now CONSUMED FROM the platform capability, not decided inline.
- **Consumption (file:line).** `qwen3_5.cpp` host-free
  (`BuildMoeMarlinResident`, the `release_host` decision) reads
  `GetPlatform(d.q.device.type).residency_policy()` through the new pure helper
  `vllm::platforms::ShouldReleaseHostWeights(policy, marlin_committed,
  host_free_env)` (`include/vllm/platforms/interface.h`); load-stream
  (`Qwen3_5Model::PrepareMarlinResident`, the per-layer interleave gate) reads it
  through `ShouldInterleaveLoadStream(policy, marlin_committed)`; the DevicePool
  scratch soft cap reads `residency_policy().device_pool_cap_bytes` (memoized
  per-device `ResolveDevicePoolPolicy` → `DBuf`/`DevicePool::Put`). All key on the
  OBJECT's own `device.type` (the per-tensor lesson), never process-global
  `CurrentPlatform()`.
- **POLICY vs KERNEL-PATH split (the design point).** The two are ORTHOGONAL and
  kept separate: the RESIDENCY POLICY (free host after upload? pool cap?) comes
  from the platform; whether Marlin is the committed compute path (so the wmma
  fallback that re-reads the freed host bytes can never run) stays the KERNEL gate
  `MarlinMoeEnabled()`. `VT_MOE_HOST_FREE`/`VT_MOE_LOADSTREAM` env rollbacks stay
  as house-convention overrides layered over the policy.
- **Reproduces today EXACTLY on GB10.** `CudaPlatform::residency_policy()` now
  returns `release_host_weights_after_upload = true` (was `false`, un-consumed —
  the routed MoE experts' host mirror IS freed after the Marlin build today),
  `uses_device_memory_pool = true`, `device_pool_cap_bytes = 0` (uncapped). With
  these, the consuming code takes the IDENTICAL branch it took inline: host-free
  fires (`true && MarlinMoeEnabled() && env`), interleave fires (`policy && Marlin`
  == old `kCUDA && Marlin`), pool uncapped. CPU advertises retain/no-pool ⇒ the
  materialize-all fallback, exactly as `device.type != kCUDA` before.
- **Additive-ness (measured).** BEFORE: a discrete GPU's residency behavior meant
  editing the inline `device.type`/env gates in `qwen3_5.cpp`. AFTER: it sets
  `CudaPlatform`/its-own-platform `residency_policy()` field values (e.g.
  `device_pool_cap_bytes`, or `release_host_weights_after_upload=false` to retain)
  and the model code is UNCHANGED — "a new GPU's residency behavior = set
  `residency_policy()` values, zero model edit."
- **Gate.** CPU: clean `-Werror` build 0 warnings; `test_platform` residency
  CONSUMPTION cases (7 cases / 43 assertions — helper truth-tables +
  discrete-GPU policy round-trip + CPU retain) PASS; full CPU CTest (8 HTTP/engine
  parallel-port timeouts all pass isolated → 126/126); tools 164/164; record +
  doc-checkpoint checkers green. DGX behavior-preserving gate (27B 235/235 + 35B
  315/315 token-exact + 35B ~4 GiB load VmHWM preserved + memcheck) **PASSED**
  (`~/work/vllm.cpp-residency-policy` @ `62fc0e0`, production flags CUTLASS sm120a
  + FA2 sm_121a + Triton AOT verified, one flock): clean CUDA `-Werror` 0 warn; 27B
  **235/235** + 35B **315/315** token-exact; 35B load-to-ready peak RSS (VmHWM)
  **4,201,632 KB ≈ 4.0 GiB** — the `ENG-MOE-LOADSTREAM` ~4.19 GiB win PRESERVED;
  `compute-sanitizer memcheck` 35B **0 errors / 315**. Completes the residency
  seam; the 3 portability seams (Platform item 1 + attn-registry item 4 + model
  self-reg item 5) plus this residency capability are now realized.

**User directive (post-27B-parity):** make adding a new GPU/arch/model an
ADDITIVE change (new files + one registration), not scattered edits — mirror
vLLM so ports are mechanical copy-paste. HW-prioritized: GB10 (sm_121a) + M4
(both unified-memory); NO multi-GPU; SGLang/LMCache (external images) deferred.
See [[extensibility-first-additive-hw-models]] (memory).

## Root cause (grounded in PR #4 scatter + our tree vs vLLM `e24d1b24`)

`backends.md` names three portability seams; only ONE is realized well:
- **Seam 3 — `vt::` op tables:** NEAR-IDEAL. `src/vt/ops.cpp:87-113` `[OpId][DeviceType]`
  table; kernels self-register via static `Registrar` (`cpu_backend.cpp:28-32`,
  `cuda_backend.cu:264`); open `DeviceType` enum (`device.h:16`, reserved
  Metal/Vulkan/XPU slots); CMake gates each backend in its own `target_sources`.
  Adding a device's op = a new TU + a `Registrar`, ZERO shared-file edits. **This
  is the model to copy for the other seams.**
- **Seam 1 — Platform (`vllm/platforms/interface.py:134-229`): MISSING.** No
  `platforms/` tree. Its responsibilities (unified-vs-discrete memory model,
  residency policy, attn-backend selection, capability advertisement) are smeared
  as **37 inline `device.type==kCUDA` / `!backend.UnifiedMemory()` conditionals**
  across `runner.cpp`, `model_registry.cpp`, `qwen3_5.cpp`. THIS is the scatter.
- **Seam 2 — attention/GDN backend registry: HALF-BUILT.** The ABC + concretes
  exist (`include/vllm/v1/attention/backend.h:203`) but there is no registry and
  no platform-driven priority selection (`interface.py:363 get_attn_backend_cls`
  unmirrored). Lower urgency — kernel dispatch already routes through the
  device-agnostic vt table.
- **Seam 4 — model registry: WORKS with 2 frictions.** `model_registry.cpp:373`
  is a fixed `constexpr array` (edit-to-add, not self-register); `qwen3_5.cpp` is
  a dense+MoE monolith so variant work lands in a shared TU.

**PR #4 scatter ledger conclusion:** the extensibility debt (category "missing
seam", ~200 lines) is ENTIRELY discrete-memory/residency and ENTIRELY absorbable
by a Platform memory-model + residency seam (`model_registry.cpp` +48 residency,
`qwen3_5.cpp` DevicePool/residency, the 37 conditionals). Everything else in PR #4
is genuine kernel/perf work (`cuda_sample.cu` +92, `cuda_matmul.cu` +33) or a
GDN a/b stride bugfix (5 files) that would land regardless of architecture.

## Prioritized plan (leverage÷effort; all items 1–5 developable on GB10+CPU alone)

| # | Item | Seam | Additive win (what does adding the next GPU/model then TOUCH?) | Effort | Deps |
|---|------|------|----------------------------------------------------------------|--------|------|
| **1** | **Extract the Platform seam** — `platforms/interface.h` + `{cuda,cpu}.cpp`, `CurrentPlatform()` self-registered per DeviceType; migrate the 37 `device.type`/`UnifiedMemory()` sites to capability calls | 1 | New GPU touches ONLY its `platforms/<gpu>.cpp` + kernel TUs; zero engine/model edits for memory-model/residency (~40 sites → 1 file) | M | none |
| 2 | Residency/memory-model as a `Platform::residency_policy()` capability (folds PR #4's host-weight-release + DevicePool caps) | 1 | discrete/unified split = one policy object; new discrete GPU sets a flag, edits no model code | S–M | 1 |
| 3 | Drop-in kernel ABI: finish `BACKEND-ABI-VT` W0 debts + migrate GEMM families | 3 (ABI leaf) | upstream/ROCm kernels drop in at the raw-pointer boundary | M–L | ABI spine `1141b79` |
| 4 | Attention-backend registry + platform priority (mirror `platforms/cuda.py:89-160`) — **LANDED 2026-07-19 (`BACKEND-ATTN-REGISTRY`), see the Item 4 section below** | 2 | new attn impl self-registers; selection is data, not a code edit | M | 1 |
| 5 | Model self-registration (`REGISTER_VLLM_MODEL` static-init) + per-arch TU split of `qwen3_5.cpp` **— LANDED CPU 2026-07-19 (`CLAIM-MODEL-SELFREG-1`); registration + arch-entry-point separation done, deep `qwen3_5.cpp` machinery factoring deferred** | 4 | "add a model = one additive TU + one registration" | M | none |
| 6 | Metal/MLX bring-up (first non-CUDA PROOF the seams work) | 1+2+3 | the payoff: proves "add a GPU = add files only" | L | 1–4, **needs M4 Mac — defer** |

Do-now (GB10/CPU): 1,2,3,4,5. Defer: 6 (M4 Mac), discrete-memory *validation* of #2, ROCm leg of #3.

## Item 1 — dispatch-ready grounding (the FIRST thing to build)

Behavior-preserving refactor, GB10+CPU-testable, absorbs all PR #4 category-(i) scatter.
- **Mirror:** `vllm/platforms/interface.py:134-229` (`class Platform`: `is_cuda/is_cpu`,
  `supported_dtypes`, `uses_host_device_handling`, capability checks) + `:363
  get_attn_backend_cls` (stub for item 4).
- **Build on:** `include/vt/device.h:16` (DeviceType), `include/vt/backend.h:22-94`
  (`vt::Backend`, `UnifiedMemory()` @ :45 — Platform COMPOSES Backend), the
  `RegisterBackend`/`GetBackend` static-init in `src/vt/backend.cpp` +
  `backend.h:118-126` (copy for `RegisterPlatform`/`CurrentPlatform`), the CUDA
  integrated-GPU probe in `cuda_backend.cu` → `CudaPlatform::is_unified_memory()`.
- **Migrate (the 37 sites):** `grep -n "UnifiedMemory()\|device.type == vt::DeviceType::kCUDA\|device.type != vt::DeviceType::kCUDA"`
  in `runner.cpp`, `model_registry.cpp`, `qwen3_5.cpp` → `CurrentPlatform().is_unified_memory()`
  etc. Start with `model_registry.cpp:230-320` residency + `qwen3_5.cpp` DevicePool
  (the PR #4 debt).
- **Scope discipline:** ADDITIVE first — land `Platform` + CUDA/CPU impls + migrate
  the memory-model/residency call sites; leave `get_attn_backend_priority()` a stub
  for item 4. PURE refactor — no kernel/numeric change.
- **Gate:** clean CPU build + 94/94 CPU unchanged; both model gates
  (`test_qwen27_paged_engine` 235/235, `test_qwen36_paged_engine` 315/315)
  token-exact on GB10 (any delta = regression).
- **Records:** `porting-inventory §9` (faithful port of an upstream seam, not a
  deviation); flip `backend-matrix.md:91-93` Metal/Vulkan/ANE realization cells
  from "enum slot only" to the new Platform anchor.

**Measured additive-ness after item 1:** the next GPU's memory model touches ONE
new file instead of ~40 edit sites — the PR-#4-shaped-but-additive outcome.

## Sequencing note

Item 1 touches `runner.cpp`/`model_registry.cpp`/`qwen3_5.cpp` heavily — it must
NOT race the concurrent 35B-IMA fix (which may touch `qwen3_5.cpp` MoE forward).
Dispatch item 1 on clean main AFTER the 35B-IMA fix lands. Items 3/5 also touch
shared files (op registry / model monolith) — sequence, don't stack.

## Item 1 spike contract — `BACKEND-PLATFORM`

### Scope

In: the C++ Platform capability/memory-model seam for `BACKEND-PLATFORM` — a
`Platform` interface composing `vt::Backend`, `CudaPlatform`/`CpuPlatform`
impls, `RegisterPlatform`/`CurrentPlatform`/`GetPlatform` self-registered per
`DeviceType`, and migration of the TRUE platform/memory-model/residency
`device.type == kCUDA` branches (7 sites) to `CurrentPlatform()` calls.
Behavior-preserving: no kernel/numeric/dispatch change. Out: kernel-shape
dispatch branches (nvfp4/fp8 GEMM selection, fused-kernel gates — items 4/5),
the attention-backend registry (`get_attn_backend_priority()` is a STUB), the
`qwen3_5.cpp` per-arch TU split (item 5), and any residency behavior change
(`residency_policy()` is advertisement only until item 2).

### Upstream chain

`vllm/platforms/interface.py:134-229` (`class Platform`: `is_cuda`/`is_cpu`
:189-202, `supported_dtypes` :181-187, `get_device_capability`/
`has_device_capability` :409-439, `uses_host_device_handling`), `:69-131`
(`class DeviceCapability`) @ pin `e24d1b24`. `current_platform` resolution =
accelerator-probe-then-CPU-fallback (`platforms/__init__.py`), mirrored by
`CurrentPlatform()`. Composes the `vt::Backend` (`include/vt/backend.h:22-94`,
`UnifiedMemory()` :45) and copies the `RegisterBackend`/`GetBackend` static-init
idiom (`src/vt/backend.cpp`, registrars `cpu_backend.cpp:28-32` /
`cuda_backend.cu:255-266`). No runtime-dynamic dispatch here (pure capability
metadata), so no nsys trace plan is required.

### Our baseline

Before: ~37 inline `device.type`/`UnifiedMemory()` conditionals across
`runner.cpp`, `model_registry.cpp`, `qwen3_5.cpp` (the PR #4 scatter). No
`platforms/` tree existed. `vt::Backend` already provided `UnifiedMemory()` and
`SupportsGraphCapture()`; the CUDA integrated-GPU probe lived in the backend
registrar. Gap: the capability/residency/memory-model advertisement was not a
seam — a new GPU's memory model meant editing every scattered site.

### Port map

`vllm/platforms/interface.py:134-229` -> `include/vllm/platforms/interface.h`
(`Platform` + `DeviceCapability` + `ResidencyPolicy`);
`platforms/__init__.py` current-platform resolution + registry ->
`src/vllm/platforms/platform.cpp` (`RegisterPlatform`/`GetPlatform`/
`CurrentPlatform`/`has_device_capability`); `platforms/cpu.py` ->
`src/vllm/platforms/cpu.cpp`; `platforms/cuda.py` ->
`src/vllm/platforms/cuda.cpp`. Migration sites -> `CurrentPlatform().is_cuda()`:
`runner.cpp` (KV-cache device residency, async device combine/scatter),
`model_registry.cpp` (`fp4_cuda` decode-graph gate), `qwen3_5.cpp`
(`ResidentWeight`/`ResidentWeightF32`). Deviation: none new — this is a faithful
port organizing existing seams (porting-inventory §9 note 8).

### Tests to port

vLLM has no standalone `Platform` unit test (it is exercised through
integration); the executable spec here is the capability contract. Ported as a
new CPU-tier `tests/vllm/platforms/test_platform.cpp` mirroring the
backend-registry test style (`tests/vt/test_backend.cpp`): registration +
`GetPlatform`/`HasPlatform`/`CurrentPlatform` resolution, the CPU capability
values (`is_cpu`, `is_unified_memory`, `supported_dtypes` order,
`residency_policy` defaults, `supports_graph_capture` false), the
`get_attn_backend_priority` stub, and the `has_device_capability` lexicographic
`(major, minor)` logic via a synthetic capability platform (no GPU on the CPU
tier). Nothing is skipped.

### Gates

Correctness/behavior: clean CPU `-Werror` build 0 warnings; new `test_platform`
PASS; full CPU CTest green (the 2 HTTP/engine tests pass in isolation —
parallel-port-contention flake); tools `unittest discover` 164/164;
`check-agent-record.py` + `check-doc-checkpoint.py` green. Behavior-preserving
DGX model gates (production flags `-DVLLM_CPP_CUTLASS_DIR=… -DCMAKE_CUDA_COMPILER=
/usr/local/cuda-13.0/bin/nvcc -DVLLM_CPP_TRITON=ON`, CUTLASS+FA2 verified):
`test_qwen27_paged_engine` 235/235 + `test_qwen36_paged_engine` 315/315 token-
exact (any delta = regression). Performance/memory: NOT APPLICABLE (no compute
change).

### Dependencies

None blocking (developable on GB10+CPU). Toolchain: CUDA 13.0 for the CUDA leg.
Sequences AFTER the 35B-IMA fix (`CLAIM-35B-GRAPH-SCRATCH-1`) to avoid racing
`qwen3_5.cpp`. Foundation for item 2 (residency-as-capability), item 4 (attn
registry, fills `get_attn_backend_priority()`), item 5 (model self-registration
+ per-arch TU split), item 6 (Metal/MLX bring-up, needs M4).

### Work breakdown

W1: `interface.h` (`Platform`/`DeviceCapability`/`ResidencyPolicy`) + registry
`platform.cpp`. W2: `cpu.cpp` + `cuda.cpp` impls + registrars + CMake. W3: new
`test_platform.cpp` + CMake row. W4: migrate the 7 memory-model/residency sites.
W5: records (porting-inventory §9, backend-matrix `BACKEND-PLATFORM` +
Metal/Vulkan/ANE anchors, ledger, README, BENCHMARKS, roadmap, coordination) +
DGX model-gate confirmation. All landed in one change except the DGX gate.

### Risks/decisions

**CORRECTION (2026-07-18, regression fix under `CLAIM-BACKEND-PLATFORM-1`).** The
original migration routed the 7 per-object sites through
`CurrentPlatform().is_cuda()` on the false premise that it is byte-equivalent to
the old per-queue `device.type == kCUDA`. It is NOT: `CurrentPlatform()` is
accelerator-first and process-global, so on ANY GPU box it returns the CUDA
platform regardless of the tensor/queue being operated on — a CPU queue/tensor on
a GPU box wrongly took the CUDA branch (red CPU tests on DGX: `test_platform`
no-accelerator fallback, `test_qwen27_dense_forward` maxd=0). FIX: the 7 sites
now key on the OBJECT's device via `GetPlatform(<obj>.device.type).is_cuda()`
(per-tensor-correct, KEEPS the seam). `CurrentPlatform()` is reserved for genuine
process-level "which accelerator is this process on" questions (none among the 7).
The `test_platform` no-accelerator fallback assertion is now tier-guarded
(CPU-only tier asserts CPU fallback; a GPU box asserts accelerator-first), with a
device-correct `GetPlatform(kCPU).is_cpu()` invariant on every tier. Decision (not a product call): only
memory-model/residency branches migrate now; kernel-shape dispatch is left for
the attention/kernel-registry items — no vLLM-defined behavior is reopened.
`residency_policy()` is advertisement only (item 2 wires it), so current
retain-host-weights behavior is unchanged.

## Item 4 — LANDED 2026-07-19 (`CLAIM-ATTN-REGISTRY-1`, `BACKEND-ATTN-REGISTRY`)

The attention-backend registry + Platform-driven priority — the SECOND
portability seam — is realized. Behavior-preserving: no numeric/kernel/dispatch
change; the SAME FA2 attention runs.

**Mirror.** `vllm/v1/attention/backends/registry.py` (self-registration via
`@register_backend(AttentionBackendEnum.X)`) + `vllm/platforms/cuda.py:361-470`
(`get_valid_backends` / `get_attn_backend_cls`) + `:84-166`
(`_get_backend_priorities`, non-MLA branch) + `cpu.py:75-87` (CPU_ATTN) @ pin
`e24d1b24`.

**Mechanism (file:line).**
- Registry keyed on `(DeviceType, name)`:
  `include/vllm/v1/attention/registry.h` (`RegisterAttentionBackend`,
  `HasAttentionBackend`, `MakeAttentionBackend`, `AttentionBackendRegistrar`) +
  `src/vllm/v1/attention/registry.cpp` (the `std::array<map>` table, copying the
  vt-op `Table()` / platform `Registry()` static-init idiom).
- Selector `SelectAttentionBackendName` / `SelectAttentionBackend`
  (`registry.cpp:60`) — walks the platform's priority list and returns the first
  REGISTERED name (upstream's min-priority valid backend; "registered" IS the
  validity check, an unregistered name is skipped exactly as an ImportError-ing
  backend is). Supports an explicit override arg (upstream `selected_backend` /
  `VLLM_ATTENTION_BACKEND`).
- Priority filled (was the item-1 STUB returning int 0):
  `Platform::get_attn_backend_priority()` now `std::vector<std::string>`
  (`interface.h:92`); `CudaPlatform` (`cuda.cpp`) returns the capability-ordered
  non-MLA list — `major==10`: FLASHINFER, FLASH_ATTN, TRITON_ATTN,
  FLEX_ATTENTION, TURBOQUANT; else (incl. GB10 sm_121 major 12): FLASH_ATTN
  first. `CpuPlatform` (`cpu.cpp`): CPU_ATTN, FLASH_ATTN.
- Self-registration: FLASH_ATTN for kCUDA+kCPU in `backend.cpp`; GDN_ATTN for
  kCUDA+kCPU in `gdn_attn.cpp` (both retained past the linker via the vllm
  `--whole-archive` INTERFACE option, same as the platform/model registrars).

**Behavior-preserving proof.** On CUDA (every capability we ship on) the walk
returns "FLASH_ATTN": on GB10 (major 12 → else branch) it is first; on major-10
FLASHINFER is preferred but UNREGISTERED so the walk falls through to
FLASH_ATTN. On CPU, CPU_ATTN is unregistered (our CPU paged-attn reuses the
FlashAttention NHD layout — the recorded `cpu_paged_attn.cpp:6` deviation, not
upstream's `[N,H,block,head]` CPU_ATTN layout) so the walk returns FLASH_ATTN.
FLASHINFER/TRITON_ATTN/FLEX_ATTENTION/TURBOQUANT are named for upstream fidelity
but unimplemented; each becomes auto-selected on the appropriate capability the
moment its self-registering TU lands — ZERO selector edit. The MLA-branch
priorities are deferred until an MLA model ports (our gate models are non-MLA).

**Scope discipline.** PURE ENGINE-level SELECTION seam. The concrete attention
KERNEL stays selected at seam 3 (the vt:: op table, `vt::PagedAttention` →
`GetOp(kPagedAttention, device.type)`), already device-additive — untouched. No
model/runner attention code edited; the runtime path is byte-identical, which is
what the DGX token-exact gate proves.

**Additive-ness (measured).** Adding a new backend's attention BEFORE: edit the
inline selection + wherever the backend is constructed. AFTER: one
self-registering TU (`AttentionBackendRegistrar`) + one slot in the owning
platform's `get_attn_backend_priority()` — ZERO edit to the selector, model, or
runner. Same "add files only" outcome as the Platform (item 1) and model-registry
(item 5) seams; the 3-seam extensibility foundation is complete.

**Gate.** CPU: clean `-Werror` build 0 warnings; new
`tests/vllm/v1/attention/test_attn_backend_registry.cpp` (8 cases / 25
assertions: self-register, Make/throw, CUDA major-10/else + CPU priority order,
first-registered walk, explicit override, empty-priority throw) + updated
`test_platform.cpp` (priority assertion) PASS; full CPU CTest green; tools
164/164 + scripts 18/18; record/doc-checkpoint checkers green. **DGX
behavior-preserving model gates PASSED** (`~/work/vllm.cpp-attn-registry` @
`2c732e7`, production flags CUTLASS sm120a + FA2 sm_121a + Triton AOT verified,
one flock): clean CUDA `-Werror` build 0 warnings; `test_qwen27_paged_engine`
**235/235** + `test_qwen36_paged_engine` **315/315** token-exact (same FA2
sm_121a path — byte-identical tokens prove attention unchanged);
`compute-sanitizer memcheck` 35B **0 errors / 315**. Evidence
`dgx:/tmp/attnreg_{cfg,build,gates,detail,memcheck}.log`.
