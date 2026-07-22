# Non-CUDA backend fan-out — Metal/MLX, Vulkan, Intel XPU (SPIKE)

Status: **SPIKE ONLY — no implementation.** Owner: `CLAIM-BACKEND-FANOUT-1`.
Rows: **`BACKEND-METAL-MLX`**, **`BACKEND-VULKAN`**, **`BACKEND-XPU`**
(`INVENTORIED` -> `SPIKE`).
Roadmap wiring: **`ROAD-V1-D1`** (NVIDIA target fan-out, ROCm, MLX, Vulkan, XPU, ANE)
— this spike covers the **non-CUDA, non-ROCm** half. `BACKEND-ROCM` and
`BACKEND-ANE` are NOT covered here and stay `INVENTORIED`.

Pins: vLLM `/home/mudler/_git/vllm` @ `e24d1b24`; llama.cpp
`/home/mudler/_git/llama.cpp` @ `237ad9b96` (the same pin the `vt::` GGUF ports
already cite — the Vulkan/Metal port source is byte-checkable locally); local
base `429e19d`.

**Why ONE spec for three backends.** The dominant near-term work is the SHARED
seam repair (§ Port map W0) — it is identical for all three and is a
prerequisite for each. Splitting would triplicate W0 and hide that Metal's
blocker is also CPU-on-macOS's blocker. Per-backend divergence starts at W1 and
is sectioned below; if any backend outgrows this file it forks with its rows.

**What this change is NOT.** No code, no kernels, no shaders, no benchmark, no
model downloaded. Nothing claims `READY`/`ACTIVE`/`DONE`, and no competitor-gate
row (`BACKEND-GATE-METAL-*`, `BACKEND-GATE-VULKAN-LLAMACPP`,
`BACKEND-GATE-XPU-VLLM`) moves — those need a pinned harness an implementation
does not yet exist to drive.

---

### Scope

**In.** The three non-CUDA platform rows, each answered with a MEASURED hardware
verdict, a file-by-file port map against the current tree, a correctness bar we
can actually run, and a row-sized work breakdown.

**Out.** ROCm (`BACKEND-ROCM` — mirrors CUDA csrc signatures, a different and
easier port, unclaimed here), ANE (`BACKEND-ANE` — CoreML, not a paged-decode
backend, per [backends.md](../backends.md) E3), TPU, and every native-competitor
performance gate row.

**Hardware verdict summary (all four rows MEASURED this session, not inferred):**

| Backend | Hardware reachable? | Toolchain | e2e gateable? |
|---|---|---|---|
| Metal | **YES** — `ssh 192.168.68.103`, Apple M4, 16 GB unified, macOS 26.5.2 | CLT-only clang 21; **no full Xcode, so no offline `metal` compiler**; brew cmake 4.1.0 present; MLX **not** installed | **YES** for op parity + small models. **NO** for 27B/35B (16 GB) |
| Vulkan | **YES** — GB10 on `dgx.casa` enumerates as a real Vulkan device; **plus llvmpipe on the dev box** | loader 1.4.328 + NVIDIA ICD present; **no `glslc`/headers/`vulkaninfo` yet** (apt-installable) | **YES**, and uniquely well: same box runs our CUDA backend as the oracle |
| Intel XPU | **NO** — no Intel GPU on any box | oneAPI not installed | **NO — HW-BLOCKED.** Unit/build-only remains possible (§ Gates) |

**Ranked recommendation — (value x testability) / effort:**

| Rank | Backend | Why |
|---|---|---|
| **1** | **Metal (native MSL first, MLX deferred)** | Real HW; **108,952 assertions already pass on it today** (measured below); the only blocker is a ONE-LINE CMake fix that this spike already verified; runtime MSL compilation works without Xcode, so E2 (hand-written Metal) needs **zero installs** and outranks E1 (MLX) for bring-up |
| **2** | **Vulkan** | Real HW with `VK_KHR_cooperative_matrix` v2 AND `VK_NV_cooperative_matrix2`; llama.cpp's mature backend is locally readable at our own pin; a software ICD gives GPU-free CI. Costlier than Metal only because of the shader toolchain + 132-shader reference surface |
| **3** | **Intel XPU** | **HW-BLOCKED**, and worse: vLLM has **no in-tree SYCL source to mirror** — its XPU kernels live in the external `vllm_xpu_kernels` package. Loyalty has no local target. Do not schedule ahead of 1/2 |

---

### Upstream chain

**Seam source (all three backends) — vLLM's own three surfaces**, per
[backends.md](../backends.md):
`vllm/platforms/interface.py:134-229` (`class Platform`), `:409-439`
(`get/has_device_capability`), `:181-187` (`supported_dtypes`);
`vllm/v1/attention/backends/registry.py` (self-registration);
`vllm/platforms/cpu.py:42-125` as the reference minimal platform.

**Metal.** No upstream vLLM platform — a recorded extension
([porting-inventory.md](../porting-inventory.md) §9). Port-FROM references:
- **llama.cpp Metal** `ggml/src/ggml-metal/` @ `237ad9b96` — the proven
  Apple-Silicon design, and specifically its **runtime-MSL-compilation** model
  (embed MSL source, `newLibraryWithSource:`), which this spike VERIFIED works
  with Command-Line-Tools only.
- MLX (`mlx-c`) for E1, deferred — see § Risks/decisions.

**Vulkan.** No upstream vLLM path. Port-FROM is **llama.cpp
`ggml/src/ggml-vulkan/` @ `237ad9b96`**, verified present locally:
`ggml-vulkan.cpp` (18,706 lines, single TU, ~215 `vk_pipeline` members),
`vulkan-shaders/` (132 `.comp` + 26 shared `.glsl` + `vulkan-shaders-gen.cpp`
1,293 lines). Key structures for the port map:
- matmul `mul_mm.comp` (466, scalar **and** coopmat1 via defines),
  `mul_mm_cm2.comp` (658, NV coopmat2), `mul_mmq.comp` (311, quant x quant);
- quant matvec `mul_mat_vecq.comp` (141) + `mul_mat_vec_{q2..q6}_k.comp`;
- flash attention `flash_attn.comp` (758) / `_cm1` (645) / `_cm2` (481);
- build = host tool `vulkan-shaders-gen` shelling out to `glslc`, emitting
  SPIR-V as C arrays into `ggml-vulkan-shaders.hpp`, with **trial-compile
  feature probes** (`CMakeLists.txt:38-58`) for coopmat/coopmat2/bf16/integer-dot.

**Intel XPU.** vLLM HAS `vllm/platforms/xpu.py` (487 lines) — but the loyalty
target is **incomplete by inspection**: `:11-13` imports
`vllm_xpu_kernels._C` / `._moe_C` / `._xpu_C`, an **external package**, and
`find csrc -name '*.sycl' -o -name '*.dp.cpp'` returns **nothing**. There is no
SYCL kernel source in the pinned vLLM tree to mirror. What IS portable 1:1 is
the **policy**: `xpu.py:121-167 get_attn_backend_cls` (forced `NHD` KV layout;
TurboQuant -> sparse-MLA -> MLA -> Triton -> fp32-falls-back-to-Triton ->
FlashAttention), `:206-250` device/capability/memory accessors, `:251-328`
`check_and_update_config`, `:329-380` block-size and hybrid-KV rules.

---

### Our baseline

**MEASURED THIS SESSION — the portable core already runs on Apple Silicon.**
Source archived from `429e19d`, extracted to `~/vllmcpp-probe` on the M4, built
with brew cmake 4.1.0 + AppleClang 21:

| Probe | Result |
|---|---|
| `cmake -S . -B build-cfg` on macOS arm64 | **configures clean, exit 0**, no CUDA, no patch |
| Full `vllm` lib + `test_backend` compile | builds with exactly **THREE** Clang-only `-Werror` suppressions (below) |
| `test_backend` as-shipped | **5 of 7 FAIL** — `vt: no backend registered for device type 0` |
| after the one-line Apple force-load fix | **7/7 pass, 18/18 assertions** |
| `test_platform` | 43/43 |
| `test_attn_backend_registry` | 56/56 |
| `test_fused_chain_additivity` | 17/17 |
| `test_ops_quant_dot` | **78,052/78,052** |
| `test_gguf_keep_quant` | **5,574/5,574** |
| `test_cpu_threadpool` | 19,595/19,595 |
| `test_ops_quant_traits` | 5,615/5,615 |
| **Total on Apple arm64** | **108,952 assertions, 0 failures** |

This is a **third-architecture** confirmation of the GGUF quant tier's
architecture-independence (x86-64 Linux, dgx aarch64 Linux, now Apple arm64
macOS) and of the fusion additivity proof.

**The macOS build delta is exactly three Clang-only diagnostics**, and one is a
REAL latent defect, not pedantry:
1. `-Wunused-private-field` — `include/vllm/v1/core/kv_cache_coordinator.h:283`
   (`hash_block_size_h_` unused).
2. `-Wpotentially-evaluated-expression` —
   `src/vllm/model_executor/layers/attention/chunked_local_attention.cpp:92`
   (`typeid` on an expression with side effects).
3. `-Wdelete-non-abstract-non-virtual-dtor` — **genuine UB**:
   `vllm::v1::Scheduler` and `vllm::v1::AsyncScheduler` are deleted through
   `unique_ptr` with virtual functions but a **non-virtual destructor**. GCC
   does not warn; this is a correctness fix owed regardless of Metal.

**THE blocker, now MEASURED rather than inferred.** `CMakeLists.txt:304-306`:

```cmake
if(UNIX AND NOT APPLE)
  target_link_options(vllm INTERFACE "LINKER:--whole-archive,...,--no-whole-archive")
endif()
```

Static-init registrars (backend, platform, op table, attention backends) live in
archive members nothing references directly. On Apple the force-link is `if`'d
out, so **every registrar is silently dropped** — which is exactly the observed
`no backend registered for device type 0`. **Verified fix** (applied and re-run
on the M4, 7/7 green): `if(APPLE) target_link_options(vllm INTERFACE
"LINKER:-force_load,$<TARGET_FILE:vllm>") elseif(UNIX) ...`. Same guard at
`CMakeLists.txt:738` and `tests/CMakeLists.txt:175`.

**The `vt::` seam is genuinely backend-agnostic.**
- `include/vt/device.h:16-17` — `DeviceType{kCPU,kCUDA,kMETAL,kVULKAN,kXPU}`,
  `kNumDeviceTypes = 5`. **All three target slots already exist**; every
  registry array sizes on it, so adding a backend needs **zero enum edit**.
- `src/vllm/platforms/platform.cpp:38-40` — `CurrentPlatform()` priority is
  already `{kCUDA, kXPU, kVULKAN, kMETAL, kCPU}`.
- Op table `src/vt/ops.cpp:10-15` is `[OpId][DeviceType]` type-erased;
  `RegisterOp` `:98-102`; `GetOp` `:104-111` **throws** on a missing kernel —
  partial backends are a supported, tested state.
- **Every kernel signature's first parameter is `Queue&`**
  (`include/vt/ops.h:439-602`) — there is no ambient stream anywhere. This is
  the single property that makes the Metal command-queue and SYCL-queue
  mappings work at all.
- `vt::Tensor` (`include/vt/tensor.h:15-37`) and `vt::Queue`
  (`device.h:29-33`) contain no CUDA types.
- No `cudaStream_t`/`<cuda_runtime.h>`/`__half` in any `include/` header outside
  `include/vt/cuda/`; `src/vllm/` has exactly ONE `#include <cuda_runtime.h>`
  (`src/vllm/platforms/cuda.cpp:6`, where it belongs).

**CUDA leakage that a non-CUDA build DOES hit** (4 unconditional includes plus
one public header):
`src/vllm/entrypoints/model_loader.cpp:22` (`vt/cuda/nvfp4_autotune.h`),
`src/vllm/v1/worker/gpu/runner.cpp:30` (`vt/cuda/combine_tokens.h`),
`src/vllm/model_executor/models/qwen3_5.cpp:40,46`, and — worst —
`include/vllm/model_executor/models/dense_nvfp4_gemm.h:66`
(`vt/cuda/marlin_repack.h`) **in a public engine header**. Also
`src/vllm/entrypoints/model_loader.cpp:39` hardcodes
`vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue()`.

**Leakage that a new backend can IGNORE** (structurally opt-in `if (cuda) fast
else portable`): 32 `#ifdef VT_*` gates and 43 raw
`device.type == kCUDA` comparisons — **37 of the 43 and ~25 of the 32 are inside
`src/vllm/model_executor/models/qwen3_5.cpp` alone**. This is why the first
bring-up model must NOT be Qwen3.5-Next (§ Risks/decisions).

**Op coverage today:** CPU registers **64 of 75** `OpId`s; CUDA **73**. The 11
CPU-missing ops are 10 NVIDIA-specific NVFP4/FP8/Marlin paths plus the test-only
`kDropinProbe`. **So the CPU 64 is the proven-sufficient target for a new
backend**, and the ~55 non-quant subset is a legitimate first milestone.

**What a new backend gets FREE:** the entire engine (~80 TUs,
`CMakeLists.txt:176-279`, zero device code); 20 of 26 `vt::Backend` virtuals have
working defaults (`src/vt/backend.cpp:19-32`), including all 7 async-output
primitives documented as **already correct for unified-memory backends** — which
Metal and GB10-Vulkan both are; all op argument validation (`src/vt/ops.cpp`,
2,630 lines, validates before `GetOp`); the **entire 10-recipe fusion catalog**
from ONE `kFusedChain` registration; attention selection as data (an unregistered
name is skipped, `include/vllm/v1/attention/registry.h:59-78`); and residency
policy as platform data (`include/vllm/platforms/interface.h:43-75`).

**Minimum mandatory surface, measured against the CPU reference:**
`src/vt/cpu/cpu_backend.cpp` is **36 lines** (the 6 pure virtuals);
`src/vllm/platforms/cpu.cpp` is **57 lines** (5 pure virtuals + registrar).

---

### Port map

**W0 — shared seam repair (blocks all three; no backend code).**

| # | File | Change |
|---|---|---|
| 1 | `CMakeLists.txt:304-306`, `:738`; `tests/CMakeLists.txt:175` | `APPLE` -> `-force_load` branch. **Fix verified on M4 this session** |
| 2 | `include/vllm/v1/core/kv_cache_coordinator.h:283` | drop/consume `hash_block_size_h_` |
| 3 | `src/vllm/model_executor/layers/attention/chunked_local_attention.cpp:92` | hoist the side-effecting expression out of `typeid` |
| 4 | `src/vllm/v1/core/sched/scheduler.h`, `async_scheduler.h` | **add virtual destructors** (real UB fix) |
| 5 | `include/vllm/model_executor/models/dense_nvfp4_gemm.h:66` + the 4 `src/vllm/` `vt/cuda/` includes | guard behind `VLLM_CPP_CUDA` |
| 6 | `src/vllm/entrypoints/model_loader.cpp:39` | `CurrentPlatform().device_type()` instead of hardcoded `kCUDA` |
| 7 | `include/vt/quant.h` | lift `QuantTypeTraits` out of `namespace vt::cpu` into `vt::` — it is format description, and Metal/Vulkan both need it ([backends.md](../backends.md):55-57) |
| 8 | new `include/vt/arch_tactics.h` | generalize the `src/vt/cuda/cuda_arch_tactics.h` registry (123 lines; only 3 shallow CUDA couplings) to a backend-agnostic capability struct; `vt::cuda` becomes its first consumer. Cheapest before three backends each invent one |

**W1+ — per backend.** Each creates the same five artifacts; sizes are the
measured CPU/CUDA references.

| Artifact | Metal | Vulkan | XPU |
|---|---|---|---|
| `vt::Backend` + registrar (6 pure virtuals) | `src/vt/metal/metal_backend.mm` | `src/vt/vulkan/vulkan_backend.cpp` | `src/vt/xpu/xpu_backend.cpp` |
| Op kernels + `RegisterOp` | `src/vt/metal/metal_ops.mm` + embedded MSL | `src/vt/vulkan/vulkan_ops.cpp` + `.comp` shaders | `src/vt/xpu/xpu_ops.cpp` (SYCL) |
| `Platform` subclass (5 pure virtuals) | `src/vllm/platforms/metal.cpp` | `src/vllm/platforms/vulkan.cpp` | `src/vllm/platforms/xpu.cpp` |
| Attention registration + priority slot | 1 line + 1 fn | 1 line + 1 fn | 1 line + 1 fn (port `xpu.py:121-167`) |
| `kFusedChain` registration | 2 lines -> all 10 recipes | 2 lines | 2 lines |
| CMake | `enable_language(OBJCXX)`, `-framework Metal/Foundation`, tri-state `VLLM_CPP_METAL` modeled on `CMakeLists.txt:36,42-63,314-339` | `VLLM_CPP_VULKAN` + `find_package(Vulkan COMPONENTS glslc)` + a `vulkan-shaders-gen` host tool mirroring llama.cpp `CMakeLists.txt:21-31,141-188` | `VLLM_CPP_XPU` + icpx |

**Metal specifics.** MEASURED on the M4: `MTLCreateSystemDefaultDevice` works,
`hasUnifiedMemory=YES`, `MTLGPUFamilyApple9` + `Metal3` YES,
`maxThreadgroupMemoryLength=32768`, `threadExecutionWidth=32`,
`maxTotalThreadsPerThreadgroup=1024`, `recommendedMaxWorkingSetSize=11.84 GiB`.
**`newLibraryWithSource:` compiles MSL at runtime and a dispatched compute
kernel returned numerically correct results** — so the missing offline `metal`
compiler does **not** block native MSL. Graph capture maps to
`MTLIndirectCommandBuffer` (`include/vt/backend.h:92` already names it).

**Vulkan specifics.** MEASURED on GB10 under `$HOME/gpu.lock`: instance creation
OK; **`NVIDIA GB10`, `INTEGRATED_GPU`, Vulkan API 1.4.312, vendor 0x10de**; 249
device extensions including **`VK_KHR_cooperative_matrix` v2**,
**`VK_NV_cooperative_matrix2`**, `VK_KHR_shader_float16_int8`,
`VK_KHR_{8,16}bit_storage`, `VK_KHR_shader_integer_dot_product`,
`VK_KHR_timeline_semaphore`, `VK_EXT_memory_budget`,
`VK_KHR_buffer_device_address`. Memory: **one 89.72 GiB `DEVICE_LOCAL` heap with
a `DEVICE_LOCAL|HOST_VISIBLE` type present** — i.e. unified, matching the CUDA
side's `cudaDevAttrIntegrated` classification, so `UnifiedMemory()` returns true
and the async-output defaults stay correct. Dev box additionally enumerates
`llvmpipe` (Vulkan 1.4.318, CPU) from `mesa-vulkan-drivers`, giving GPU-free CI.
Both `coopmat` tiers llama.cpp implements are therefore reachable on our GPU.

**XPU specifics.** Port the POLICY from `xpu.py` (listed in § Upstream chain);
write SYCL kernels ourselves against the `vt::` contracts, since no upstream
SYCL source exists to mirror.

---

### Tests to port

Per [test-porting.md](../test-porting.md). Upstream-test provenance is thin here
by nature — Metal and Vulkan have no vLLM tests to port — so the inventory is
explicit about which suites are ported vs newly authored.

| Source | Ours | Note |
|---|---|---|
| `vllm/tests/kernels/**` (device-parameterized cases) | extend existing `tests/vt/test_ops_*.cpp` with the new `DeviceType` | Most `tests/vt/` files are already device-parameterized |
| `tests/vt/test_fused_chain_additivity.cpp` | **reuse verbatim** as the additivity proof for each new backend | It deliberately uses a REAL second backend, not a mock; a third slots in |
| `tests/vt/test_ops_quant_dot.cpp` (f64 dequant-then-dot oracle, NMSE <= 5e-4 ported unwidened from llama.cpp `test-quantize-fns:17-28`, `test-backend-ops:4277`) | **reuse verbatim** — device-neutral math, the right oracle for any `vec_dot`-equivalent shader | 78,052 assertions; already green on Apple arm64 |
| `tests/vllm/test_gguf_keep_quant.cpp` (losslessness per encoding) | **reuse verbatim** for a device upload path | 5,574 assertions; already green on Apple arm64 |
| `vllm/tests/test_zen_cpu_platform_detection.py:8-37` shape | `tests/vllm/platforms/test_platform.cpp` — add per-backend registration/capability cases | Mirrors the existing CPU/CUDA cases |
| upstream XPU selector behavior (`xpu.py:121-167`) | `tests/vllm/v1/attention/test_attn_backend_registry.cpp` — XPU priority-order cases | **Runnable with no Intel GPU** (selection is pure data) |
| NEW | `tests/vt/test_backend_cross_device.cpp` | The gap: no CPU-vs-device op-equality harness exists today (§ Gates) |

---

### Gates

**The bar transfers, but the ORACLE changes per backend.** Our ratified standard
is token-exact against vLLM where vLLM is deterministic
([near-tie distributional gate](../gates.md)). vLLM **cannot run on the M4** and
has **no Vulkan or Apple path at all**, so proposing a vLLM oracle there would be
a gate we cannot run. Substitutions, in order of strength:

| Backend | Correctness bar we CAN run |
|---|---|
| **Vulkan** | **Strongest available anywhere in this project: GB10 runs our CUDA backend, our CPU backend AND Vulkan on ONE box.** Gate Vulkan against **our own CUDA backend**, same weights, same prompts, same binary — plus vLLM still reachable on that box as the transitive oracle. Token-exact e2e is a legitimate ask here |
| **Metal** | No vLLM on the M4. Gate against **our own CPU backend** on the same host (already proven green there, 108,952 assertions) for op parity, and token-exact e2e vs a CPU-backend reference run on a small model. `llama.cpp` Metal is the applicable NATIVE FLOOR for speed (`BACKEND-GATE-METAL-LLAMACPP`) |
| **XPU** | **e2e is HW-BLOCKED — stated plainly, no gate proposed.** Still runnable without Intel hardware: (a) compile coverage under oneAPI DPC++; (b) attention-selector/priority unit tests, since selection is pure data; (c) **SYCL kernels executed on the oneAPI OpenCL CPU device**, which gives real unit-level numeric gating on the dev box. Nothing beyond that may be claimed |

**Numeric tolerance — do NOT promise bit-exactness across CPU and GPU.** The CPU
tier's reproducibility comes from a **fixed sequential reduction order**
(`src/vt/cpu/cpu_quant_dot.cpp:22-28`, deliberate); no GPU cross-lane/subgroup
reduction preserves it. Gate on the already-ported NMSE <= 5e-4 thresholds, not
`memcmp`. Bit-exactness IS the correct bar for the pure-copy/layout ops
(`kReshapeAndCache`, `kConcatAndCacheMla`, dequant-vs-file losslessness).

**Harness gap to build (W-item, not free):** `RegisterOp(kMatmulBTQuant,
kCPU, ...)` is the only registration for that op today; the seam for a second
`DeviceType` exists but **nothing exercises CPU-vs-device equality**.
`VT_CPU_REF` (`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:97,80`)
is a **loader-residency** switch that forces dequant-to-bf16 — it is NOT a
cross-device op harness, and this spike does not pretend otherwise.

**Performance.** No performance claim is owed until a backend runs a model.
Native floors when they are: llama.cpp Vulkan
(`BACKEND-GATE-VULKAN-LLAMACPP`), llama.cpp Metal / MLX-LM / oMLX
(`BACKEND-GATE-METAL-*`) — all still `INVENTORIED`.

---

### Dependencies

| Need | Status |
|---|---|
| W0 seam repair | **none — self-contained**, and item 1 is already verified |
| Metal bring-up | **NOTHING TO INSTALL.** Contrary to [environment.md](../environment.md):56, **cmake 4.1.0 IS already installed** (brew, at `/opt/homebrew/bin`, absent from non-interactive PATH). MLX is NOT installed and is **not required** for E2/native-MSL. `ninja` absent (make works) |
| MLX (E1 only, deferred) | `brew install mlx` -> 0.32.0, pulls `python@3.14`. 30 GiB free on the M4 |
| Vulkan dev toolchain | `libvulkan-dev`, `vulkan-tools`, `glslang-tools` are apt-available. **Caveat: Ubuntu `glslc` is shaderc 2023.8 — too old for llama.cpp's coopmat2 feature probe, which greps for "extension not supported" and would silently disable the fastest path.** Use a current Vulkan SDK glslc |
| Vulkan runtime | **already present** on dgx (loader 1.4.328 + NVIDIA ICD) and dev box (llvmpipe) |
| XPU | oneAPI DPC++ (not installed); **Intel GPU hardware — UNAVAILABLE, no acquisition path recorded** |
| GGUF compute-in-quant | `QUANT-GGUF-CIQ-GEMM` G4 (routing call sites) is open; a GPU quant path should not overtake it |
| Cross-claim | Touching `CMakeLists.txt` overlaps active CUDA claims — W0 must be a small, separately reviewable change |

**Quant-tier reuse (USER PRIORITY 4 overlap) — the explicit answer.**

*Reusable VERBATIM on Metal/Vulkan/XPU (device-neutral):* the block geometry
table `src/vt/dtype.cpp:36-73` (7 dtypes: Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K —
this IS `ggml_blck_size`/`ggml_type_size`, and a shader's stride math needs
exactly these numbers); the **block struct layouts + `static_assert`s**
`src/vt/cpu/cpu_quant_blocks.h:36-94` (the single highest-value artifact for a
shader port — directly transcribable to GLSL/MSL); the **dequant reference
math** `src/vt/cpu/cpu_quant_dequant.cpp` (6-bit scale unpack, hi-bit merge,
super-block scaling — the loop form changes, the math does not); the
**keep-quant routing policy** `gguf_keep_quant.h:37-90` with its
`-Werror=switch` totality contract; `OwnGgufQuantBlocks`
(`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:20-46`) whose
`[N,K]`/`nk=true`/no-transpose residency holds on any device; and the f64 test
oracle `tests/vt/test_ops_quant_dot.cpp`.

*CPU-tier-ONLY, must be rewritten as shaders/kernels:* all six `vec_dot` bodies
(`src/vt/cpu/cpu_quant_dot.cpp:42-428` — their `aux8`/`aux16`/`aux32`/`sums[8]`
shape exists **specifically** to coax CPU auto-vectorization and fix reduction
order, and is actively wrong on a GPU); host-side activation staging
(`cpu_quant_gemm.cpp:131-142`); the GEMM driver + threadpool chunking
(`cpu_quant_gemm.cpp:111-158`, `ParallelForRows`); and the planned G5/G6 SIMD +
repack tiers.

*Seam to plan NOW, not discover later:* llama.cpp's Vulkan backend quantizes
activations to **Q8_1** (`quantize_q8_1.comp`; every variant named
`*_q8_1_f32`), while our CPU tier uses Q8_0/Q8_K. The `vec_dot_type` column of
`QuantTypeTraits` therefore becomes **device-dependent** on a GPU port — which
is a second, independent reason for W0 item 7.

---

### Work breakdown

Row-sized; W0 is shared and must land first. Implementation can start on W0+M1
immediately.

| ID | Work | Rows | Blocked by | Deliverable |
|---|---|---|---|---|
| **W0** | Shared seam repair: 8 items in § Port map. Behavior-preserving on Linux/CUDA by construction | all three | — | macOS builds `-Werror`-clean and `test_backend` 7/7; Linux 27B/35B gates byte-identical |
| **M1** | Metal `vt::Backend` (6 pure virtuals) + `Platform` + registrars; NO kernels | `BACKEND-METAL-MLX` | W0 | `test_backend`/`test_platform` pass for `kMETAL` on the M4 |
| **M2** | Metal op tier 1: the ~20 elementwise/norm/rope/activation ops as runtime-compiled MSL; `kFusedChain` registration | `BACKEND-METAL-MLX` | M1 | op parity vs our CPU backend on the M4; all 10 fusion recipes inherited |
| **M3** | Metal GEMM + paged attention (`kMatmul`, `kMatmulBT`, `kPagedAttention`, `kReshapeAndCache`) | `BACKEND-METAL-MLX` | M2 | first small model (OPT or Qwen3-dense) token-exact vs our CPU reference |
| **M4** | Metal GGUF k-quant shaders (reusing the W0-lifted traits) | `BACKEND-METAL-MLX`, `QUANT-GGUF-CIQ-GEMM` | M3, G4 | NMSE <= 5e-4 vs the f64 oracle |
| **M5** | MLX (E1) evaluation as an ALTERNATIVE kernel source, measured against M2/M3 | `BACKEND-METAL-MLX` | M3 | a decision record, not necessarily an adoption |
| **V1** | Vulkan device/queue/buffer bring-up + `vt::Backend` + `Platform`; shader build plumbing (`vulkan-shaders-gen` equivalent) | `BACKEND-VULKAN` | W0 | `test_backend` passes for `kVULKAN` on **both** llvmpipe (dev box) and GB10 |
| **V2** | Vulkan op tier 1 (elementwise/norm/rope/activation `.comp` shaders) + `kFusedChain` | `BACKEND-VULKAN` | V1 | op parity **vs our own CUDA backend on the same GB10 box** |
| **V3** | Vulkan GEMM: scalar `mul_mm` first, then the `coopmat`/`coopmat2` tactic via the W0-generalized arch-tactic registry | `BACKEND-VULKAN` | V2 | measured coopmat selection, with the `launch()->false` portable fallback proven |
| **V4** | Vulkan paged attention (port `flash_attn.comp` family) | `BACKEND-VULKAN` | V3 | small model token-exact vs our CUDA backend |
| **X1** | XPU `Platform` + attention-selector policy port (`xpu.py:121-167`) — **DATA ONLY, no kernels** | `BACKEND-XPU` | W0 | selector unit tests pass with **no Intel hardware** |
| **X2** | SYCL op skeleton + oneAPI CPU-device execution for unit gating | `BACKEND-XPU` | X1 | compile coverage + unit numerics; **e2e stays HW-BLOCKED** |

---

### Risks/decisions

1. **DECISION — Metal bring-up is E2 (native MSL), not E1 (MLX), reversing the
   default reading of [backends.md](../backends.md) E1.** Grounds: this spike
   MEASURED that runtime MSL compilation + dispatch work on the M4 with
   Command-Line-Tools only and **zero installs**, while MLX would add a
   dependency, pull `python@3.14`, and re-introduce the **lazy-evaluation
   impedance** that backends.md itself flags as "the same impedance class that
   disqualified ggml". E1 is not abandoned — it becomes **M5**, evaluated
   against a working M2/M3 baseline instead of ahead of one. This is a
   reversible sequencing call, recorded rather than silently taken.
2. **RISK — no offline `metal` compiler (CLT only).** Mitigated by runtime
   compilation (verified). If AOT `.metallib` is later wanted, it needs full
   Xcode or `xcodebuild -downloadComponent MetalToolchain`. Not on the critical
   path.
3. **RISK — first bring-up model.** `qwen3_5.cpp` holds 37 of 43 raw `kCUDA`
   comparisons and ~25 of 32 `#ifdef VT_*` gates across 5,700 lines. It will
   compile and take portable arms, but it is not a model a new backend can
   cleanly optimize. **Decision: bring up OPT or Qwen3-dense first**, never
   Qwen3.5-Next.
4. **RISK — Ubuntu `glslc` is shaderc 2023.8**, older than llama.cpp's
   coopmat2 probe expects; the probe fails **silently** into the slow path. Pin a
   current Vulkan SDK glslc and assert the probe result.
5. **RISK — cross-backend bit-exactness is not achievable** for reductions
   (§ Gates). Stating an NMSE bar up front prevents a repeat of the RMSNorm-saga
   pattern of discovering tolerance late.
6. **RISK — W0 touches `CMakeLists.txt`**, which active CUDA claims also touch.
   Keep W0 minimal and separately reviewable; re-run both Linux gate models to
   prove it behavior-preserving.
7. **HONEST LIMIT — Apple gate scale.** The M4 has 16 GB (11.84 GiB
   recommended working set). It **cannot** hold the 27B/35B gate models; no
   Apple result may ever be extrapolated to them. Gate-scale Apple numbers need
   a larger-memory Mac, which we do not have.
8. **HONEST LIMIT — XPU has no hardware AND no upstream source.** Both halves of
   the usual justification are missing. It is ranked third for that reason and
   should not be scheduled until 1/2 land or Intel hardware appears.
9. **NOTE — [environment.md](../environment.md):56 and :106 are stale**: cmake
   IS installed on the M4 and our tree configures and builds there. Corrected in
   the same change as this spike.
