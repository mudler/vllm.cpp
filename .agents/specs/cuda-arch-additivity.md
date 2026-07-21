# CUDA-architecture additivity — implementation spike

Status: **IMPLEMENTED (framework seams only)**. Owner: `CLAIM-CUDA-ARCH-ADDITIVITY`.
Rows: **`BACKEND-CUDA-ARCH-ADDITIVITY`** (new, the seam row), advancing
`BACKEND-CUDA-SM120` / `BACKEND-CUDA-SM121` / `BACKEND-CUDA-COMP-FP4` /
`BACKEND-CUDA-COMP-MARLIN` / `BACKEND-CUDA-COMP-SCALEDMM-C3X` / `BACKEND-CUDA-COMP-FA`.
Roadmap wiring: **`ROAD-V1-D1`** (NVIDIA target fan-out).

Pins: vLLM `/home/mudler/_git/vllm` @ `e24d1b24`; local base `c56ab28`.

This spec executes the four SEAM-GAPS the completed audit already named in
[breadth-sweep-plan.md](breadth-sweep-plan.md) §A.2. It does **not** re-audit and
it does **not** bring up any architecture. Its whole purpose is that adding a
CUDA architecture vLLM supports becomes a **table-row edit plus a tactic
registration**, instead of a hunt through four CMake regexes and three host
launchers.

**What this change is NOT.** It does not make any untested architecture
supported. No non-`sm_121` target has hardware here, so nothing moves off
`INVENTORIED`. In the audit's own words, cross-family bring-up "is a kernel
campaign, not an additive drop-in" (§A.4); this spec builds the seams that
campaign will plug into, and nothing more.

---

### Scope

**In.** The four §A.2 seam-gaps, in dependency order:

| Gap | Defect | Fix landed here |
|---|---|---|
| #1 | Four CMake guards hardcode `MATCHES "12[01]a"` over the WHOLE arch list | per-arch FEATURE TABLE + vLLM's loose-intersection resolution |
| #4 | Kernel layer cannot see the capability source of truth | one cached device-capability probe, threaded to the kernel layer and carried by the backend |
| #2 | Host launchers take no capability and have no SM selector | a type-erased tactic registry; the launcher selects, tactics register |
| #3 | `cuda_paged_attn.cu` hardcodes the GB10 100 KiB opt-in smem ceiling | the queried `cudaDevAttrMaxSharedMemoryPerBlockOptin`, cached |

**Dispatch behavior.** Exactly ONE tactic is registered: the existing
`sm_12x` native block-scaled fp4xfp4 path. On GB10 the new selector reduces to
the old `#if VT_FP4_MMA_SM120A && NativeFp4MmaEnabled()` test — same kernel, same
grid, same stream. This change is **behavior-preserving by construction**; it is
structural, not a perf or numerics change.

**Out.** Hopper `wgmma` / `sm_100` `tcgen05` kernel bodies (a separate,
hardware-blocked campaign); per-source `-gencode` narrowing (see Risks); the
`d == 256` tensor-core shape gate in `cuda_paged_attn.cu:2529`, deliberately left
alone because generalizing it is a numerics change on the gate models' hot path
and this change may not touch numerics; the model/quantization sweep
(Deliverable B of the breadth-sweep plan).

### Upstream chain

Every piece is grounded in a pinned upstream source, per the ground-every-impl rule.

| Ours | Ported FROM | What was taken |
|---|---|---|
| `cmake/CudaArchFeatures.cmake` `cuda_archs_loose_intersection()` | `/home/mudler/_git/vllm/cmake/utils.cmake:376-485` | 1:1 port, comments included: `+PTX` handling, `a`/`f` suffix cross-matching, same-major SASS fallback, symmetric TGT-suffix preservation |
| the per-feature `<F>_ARCHS` + `if(<F>_ARCHS)` idiom | `/home/mudler/_git/vllm/CMakeLists.txt:949-953,963` (`FP4_SM120_ARCHS`), `:775-787` (sm120 `SCALED_MM_ARCHS`), `:556-558` (`MARLIN_ARCHS`), `:918-926` (`CUTLASS_MOE_DATA_ARCHS`) | vLLM asks each feature which of the requested archs support it, then gates on the non-empty result — never a regex over the whole list |
| `src/vt/cuda/cuda_device_caps.h` | `vllm/platforms/cuda.py::CudaPlatform.get_device_capability` @ `e24d1b24`, mirrored locally at `src/vllm/platforms/cuda.cpp:88-91` | one cached `(major, minor)` probe as the single source of truth, plus the device properties (`maxSharedMemoryPerBlockOptin`, `multiProcessorCount`) torch caches alongside it |
| `src/vt/cuda/cuda_arch_tactics.h` | FlashInfer's per-arch tactic registry + runtime selection, `flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h:187-220` (the 32 SM12 tactics already mirrored in `src/vt/cuda/nvfp4_tactic_ids.h`); vLLM's runtime capability gate `vllm/platforms/cuda.py:154-166` → `src/vllm/platforms/cuda.cpp:64-71` | kernel identity is DATA selected at runtime from device properties, not a compile-time `#if` |
| `include/vt/backend.h` `DeviceCapabilityMajor/Minor()` | `vllm/platforms/interface.py` `Platform.get_device_capability` | one cached probe exposed to everything downstream |

**Runtime trace plan.** Dispatch here is dynamic, so source reading is not
sufficient evidence. The tactic registry carries its own counters
(`ArchTacticStats`) and a `VT_ARCH_TACTIC_STATS=1` announcement; the gate asserts
on those counters rather than inferring from a passing test. See Gates.

### Our baseline

| Anchor | State before |
|---|---|
| `CMakeLists.txt:71,88,97,137` | four literal `MATCHES "12[01]a"` guards, evaluated over the whole arch string |
| `src/vt/cuda/cuda_backend.cu:255-273` | probed `cudaDevAttrPageableMemoryAccess` / `cudaDevAttrIntegrated` only; **no SM state at all** |
| `src/vllm/platforms/cuda.cpp:88-91` | authoritative `(major, minor)`, invisible below `vt::` |
| `src/vt/cuda/cuda_matmul_nvfp4.cu` `LaunchFp4Fp4` | no capability argument; `#if VT_FP4_MMA_SM120A` straight into sm120a-only `mma.sync kind::mxf4nvf4` PTX |
| `src/vt/cuda/cuda_matmul_fp8_cutlass.cu:164` | `ArchTag = cutlass::arch::Sm120` baked |
| `src/vt/cuda/cuda_flash_attn_fa2.cu:428` | "all instantiations compiled for sm_121a" |
| `cuda_paged_attn.cu:870,2401` | the 100 KiB opt-in ceiling as a COMMENT assumption; nothing checked it |
| `cuda_paged_attn.cu` × 7 sites | `if (shmem > 48u*1024u) Check(cudaFuncSetAttribute(...))` — an over-budget tile failed opaquely inside the driver |

**Honest gaps that remain after this change** are listed under Risks/decisions.

**Correction to the audit (§A.2 item 1), measured not assumed.** The audit states
that adding `90a` to the arch list "silently disables fp4/fp8-cutlass/Marlin for
*all* archs". That specific claim is **FALSE** and was disproven before
implementing: `if(A MATCHES "12[01]a")` is a SUBSTRING search, so
`A = "90a;121a"` still matches and the features stayed enabled. The real defects
of the old guards, all reproduced by `cmake -P` over the literal expression, are:

1. **No per-arch resolution and no report.** The guard answers "does the list
   contain something spelled like sm_12xa?", not "which archs support this?" — a
   fat build enabled every feature globally, for architectures that cannot run
   them, and printed nothing you could audit.
2. **Silent capability drop on legitimate spellings.** `"121"` (base target) and
   `"12.1a"` (vLLM's own gencode spelling) both FAIL the regex and silently
   disable fp4 + cutlass-nvfp4 + cutlass-fp8 + Marlin for the entire build, with
   no diagnostic. This is the real silent regression; the audit attributed it to
   the wrong trigger.
3. **Nothing enforced that the arch-specific `a` target was requested** for the
   feature that requires it.
4. **No place to declare arch support**, so adding an architecture meant editing
   four regexes — the forward-looking problem this row exists to fix.

The fix addresses all four; the honest evidence is reported as such, not as a
confirmation of the audit's wording.

### Port map

| Upstream / origin | Local file | Notes |
|---|---|---|
| `vllm/cmake/utils.cmake:376-485` | `cmake/CudaArchFeatures.cmake` (new) | 1:1 port + `vt_cuda_archs_normalize`/`_denormalize` bridging CMake's `121a` form to vLLM's `12.1a` gencode form, + `VT_CUDA_FEATURE_TABLE`, + `vt_cuda_feature_archs`, + `vt_cuda_report_feature` |
| `vllm/CMakeLists.txt` per-feature intersections | `CMakeLists.txt:64-98` (new block) | resolves and REPORTS `fp4-mma`, `cutlass-nvfp4`, `cutlass-fp8`, `marlin-nvfp4`, `fa2`; the four regexes are gone |
| — | `src/vt/cuda/cuda_device_caps.h` (new) | `DeviceCaps` + `GetDeviceCaps()` + `DynamicSmemFits()`; CUDA-header-free so tests and CPU TUs can include it |
| — | `src/vt/cuda/cuda_arch_tactics.h` (new) | `TacticFamily`, `ArchTactic`, `Nvfp4Fp4MmaArgs`, `RegisterArchTactic`, `SelectArchTactic`, `ArchTacticStats` |
| — | `src/vt/cuda/cuda_arch_tactics.cu` (new) | the cached probe + the registry; a new arch never edits this file |
| — | `include/vt/backend.h` | `DeviceCapabilityMajor/Minor()` virtuals, default `{0,0}` (CPU) |
| — | `src/vt/cuda/cuda_backend.cu` | registrar now uses the ONE shared probe and the backend CARRIES the capability |
| — | `src/vt/cuda/cuda_matmul_nvfp4.cu` | `LaunchFp4Fp4` takes `const DeviceCaps&` and dispatches through the registry; the sm_12x tactic + its registrar live next to the kernel body |
| — | `src/vt/cuda/cuda_paged_attn.cu` | `SetDynamicSmemOptIn()` replaces 7 open-coded opt-in blocks; comment assumptions replaced by the queried attribute |

**Deviations from upstream, recorded.**
1. `vt_cuda_feature_archs` post-filters the loose intersection to archs literally
   present in the requested target list. vLLM can be cross-suffix lenient because
   it emits per-source `-gencode` itself; we pass the list straight to
   `CMAKE_CUDA_ARCHITECTURES`, and for the fp4 rows the `a` suffix is
   load-bearing (`mma.sync kind::mxf4nvf4` is rejected on base `sm_121`). Without
   the filter, `-DVLLM_CPP_CUDA_ARCHITECTURES=121` would define
   `VT_FP4_MMA_SM120A` for a build that cannot emit the instruction — a behavior
   change vs the old guard, and the wrong one.
2. The FEATURE TABLE cells list architectures **we have a built and validated
   kernel body for**, not vLLM's full supported set. Each row's comment records
   vLLM's superset so widening a cell is mechanical once the tactic lands. In
   particular `marlin-nvfp4` does NOT claim vLLM's `8.0+PTX` leg: our vendored
   slice is the bf16 NVFP4 instantiation only and has never been built or run
   outside `sm_12x`. A green fatbinary link is not execution evidence.

### Tests to port

vLLM's arch-selection logic lives in CMake and in `vllm/platforms/cuda.py`;
upstream has no unit test module for either (it is validated by its build matrix
and by `tests/kernels/` skipping on capability). The equivalent local coverage:

| Case | Local tier | Anchor |
|---|---|---|
| CMake feature resolution across single-arch, multi-arch, unsupported-arch and base-suffix target lists | configure-tier, run as the GAP #1 gate | `cmake -P` over `cmake/CudaArchFeatures.cmake`, plus the real `cmake -S . -B` configures in Gates |
| backend carries the driver-reported capability | doctest | `tests/vt/test_cuda_backend.cpp` — "CUDA backend reports the device compute capability" |
| capability probe is real; registry is registered, consulted, and SELECTS | doctest | `tests/vt/test_ops_nvfp4_fp4.cpp` — "CUDA arch tactic registry is exercised by the fp4xfp4 launcher" |
| the smem opt-in ceiling is the queried attribute | exercised by every WMMA path in | `tests/vt/test_ops_paged_attn.cpp` |
| behavior preservation on the gate models | parity | `tests/parity/test_qwen27_paged_engine.cpp`, `test_qwen36_paged_engine.cpp`, `test_qwen3coder_paged_engine.cpp` |

No test is checked in SKIPPED by this change.

### Gates

All on dgx (GB10 sm_121a), clean build from scratch, transferred with
`git archive <commit> | ssh dgx tar -x -C <scratch>` — **never rsync** (an rsync
of `tests/` overwrote regenerated goldens twice and produced FALSE passes).
Goldens md5-verified before AND after the battery.

```
cmake -S <scratch> -B <scratch>/build -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DVLLM_CPP_TRITON=ON -DCMAKE_CUDA_ARCHITECTURES=121a
cmake --build <scratch>/build -j20
```

| Gate | Requirement |
|---|---|
| CUDA `-Werror` | 0 warnings / 0 errors on the clean full build |
| 27B `test_qwen27_paged_engine` | 235/235, unchanged |
| 35B `test_qwen36_paged_engine` | 315/315, unchanged |
| Qwen3-Coder `test_qwen3coder_paged_engine` | 6/6, unchanged |
| unit | `test_ops_nvfp4_fp4`, `test_ops_fp8_cutlass`, `test_ops_moe_grouped`, `test_ops_moe_grouped_bf16`, `test_ops_paged_attn`, `test_cuda_backend` |
| memcheck | 0 errors on the engine path |
| record checkers | `scripts/check-agent-record.py`, `scripts/check-doc-checkpoint.py` |
| **GAP #1 evidence** | configure with `-DVLLM_CPP_CUDA_ARCHITECTURES="90a;121a"` and capture the STATUS block showing fp4-mma / cutlass-nvfp4 / cutlass-fp8 / marlin-nvfp4 / fa2 **ENABLED for [121a]** with a named WARNING for `90a`; and a single-arch `121a` configure producing an identical feature set to before |
| **seam-exercised evidence** | the registry counters must MOVE — a passing gate is not proof a new path ran (the W7 decode graph: graph-ON and graph-OFF logs were byte-identical until a stats counter proved capture). `VT_ARCH_TACTIC_STATS=1` prints the selected tactic; the doctest asserts `fallbacks` advances on the production default and `selections` advances with `last_selected == "nvfp4-fp4-mma/sm12x"` under `VT_NVFP4_FP4_NATIVE=1` |

**Performance.** `benchmark_binding=false`. This change adds no kernel and
changes no numerics; it is not a speed claim and takes no speed credit. The
per-launch cost added to `LaunchFp4Fp4` is one cached-struct read plus a
predicate over an 8-entry table, off the gate models' default path entirely
(no tactic supports the device unless `VT_NVFP4_FP4_NATIVE=1`).

### Dependencies

| Kind | Item |
|---|---|
| Rows | none blocking; advances `BACKEND-CUDA-SM120/SM121` and the `BACKEND-CUDA-COMP-*` rows listed above |
| Toolchain | CMake >= 3.24 (`CMP0057` NEW, required by `IN_LIST` in the ported intersection); nvcc 13.0 |
| Hardware | GB10 `sm_121a` for the behavior-preservation battery. **Every other CUDA target is HW-blocked**: dgx has no `sm_90`/`sm_100`/`sm_80` board, so the fat-build path can be configured but not executed |
| Models | 27B / 35B / Qwen3-Coder gate checkpoints, existing goldens |
| Licenses | none new (no new vendored code) |

### Work breakdown

| # | Row | Files | State |
|---|---|---|---|
| W1 | GAP #1 — feature table + 4 guard replacements | `cmake/CudaArchFeatures.cmake`, `CMakeLists.txt` | DONE |
| W2 | GAP #4 — cached capability probe, threaded to kernel layer + carried by the backend | `cuda_device_caps.h`, `cuda_arch_tactics.cu`, `include/vt/backend.h`, `cuda_backend.cu` | DONE |
| W3 | GAP #2 — tactic registry + `LaunchFp4Fp4` wiring + ONE registered tactic | `cuda_arch_tactics.h/.cu`, `cuda_matmul_nvfp4.cu` | DONE |
| W4 | GAP #3 — queried smem ceiling | `cuda_paged_attn.cu` | DONE |
| W5 | positive-signal tests | `tests/vt/test_cuda_backend.cpp`, `tests/vt/test_ops_nvfp4_fp4.cpp` | DONE |
| W6 | dgx gate battery + multi-arch configure evidence | — | see ledger |
| W7 | **FUTURE, HW-BLOCKED** — per-source `-gencode` narrowing + the first cross-family tactic body | `cmake/CudaArchFeatures.cmake`, a new per-arch TU | NOT STARTED |

### Risks/decisions

1. **A heterogeneous fat build still cannot COMPILE, and that is now loud rather
   than silent.** Feature ENABLEMENT is resolved per arch, but the sources are
   still compiled for the whole `CMAKE_CUDA_ARCHITECTURES` list, so a
   `"90a;121a"` *build* fails on the sm12x-only PTX. Narrowing gencode per source
   requires vLLM's `set_gencode_flags_for_srcs`
   (`/home/mudler/_git/vllm/cmake/utils.cmake:265-345`), which only works with
   target-level `CUDA_ARCHITECTURES OFF` and fully manual gencode — CMake has no
   per-source `CUDA_ARCHITECTURES` property. That is a build restructure with
   real risk to a behavior-preserving change, and it buys nothing until a
   cross-family tactic body exists to compile. Deferred to W7 and reported by a
   configure-time WARNING naming the archs with no tactic. **This is a known
   limitation, not a fixed gap.**
2. **The `d == 256` tensor-core shape gate is untouched.** Generalizing it would
   change which kernel the gate models select. Behavior preservation wins; a
   different arch's MMA/smem shape belongs in that arch's tactic.
3. **`kGqaQG` and the WMMA tile constants remain compile-time**, tuned to GB10's
   measured 101376 B ceiling. The ceiling is now queried and ENFORCED (an
   over-budget tile gets a diagnosis naming the device instead of an opaque
   driver error), but choosing a different tile for a different budget is a
   per-arch tactic, not a constant to retune in shared code.
4. **Registered-tactic count is 1, and that is the honest number.** The registry
   makes a second architecture additive; it does not make one exist. Nothing here
   moves any `INVENTORIED` row.
5. **No product calls were reopened.** Where vLLM has an answer (the intersection
   algorithm, the capability-keyed dispatch shape) it was mirrored.
