# Extensibility-first plan — the Platform seam (order-1 priority, 2026-07-18)

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
| 4 | Attention-backend registry + platform priority (mirror `platforms/cuda.py:89-160`) | 2 | new attn impl self-registers; selection is data, not a code edit | M | 1 |
| 5 | Model self-registration (`REGISTER_VLLM_MODEL` static-init) + per-arch TU split of `qwen3_5.cpp` | 4 | "add a model = one additive TU + one registration" | M | none |
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
