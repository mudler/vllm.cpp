# ROCM-GLM5-NEXT-KPOOL — `kGlm5NextKpoolCompress` and `kGlm5NextKpoolSelect` on ROCm

Row: `BACKEND-ROCM`
Issue: [#2942](https://github.com/mudler/vllm.cpp/issues/2942)

## Now

`ACTIVE`, spec committed. Implementation pending.

## Scope

Register native `kGlm5NextKpoolCompress` and `kGlm5NextKpoolSelect` kernels for
`DeviceType::kROCM` in a new translation unit `src/vt/rocm/rocm_glm5_next_kpool.hip`,
add it to the two `VLLM_CPP_HIP` source lists in `CMakeLists.txt`, register both
ops in `rocm_ops.hip`, and add the cross-device test case that proves the ROCm
kernels match the host reference.

Out of scope: lifting the CPU-only refusal at `glm5_next_kda.cpp:322` or the
compose (W9c-0) that would consult these ops from the model forward. Both are
owned by later rows and tracked by #2942.

## The reached set

These are ops 3 and 4 of #2942's four ops. Op 1 (`kMoeGateUpSwiGLuGrouped`)
landed in PR #3001. Op 2 (`kKdaGatedDeltaRule`) landed in PR #3120. These two
ops complete the set.

Like KDA, these ops are **staged slices**: they are unreached by the model's
forward on its default configuration. The compose (W9c-0, #2415) that would
call them does not exist yet. The commit body and PR body name this, the row
that owns the wiring (W9c-0), and the issue that tracks it (#2942). The sibling
spec (`rocm-kda-gated-delta-rule.md`) already lists these ops under `## Owed`.

## Upstream anchors

| Role | Path |
|---|---|
| CUDA kernel (the port donor) | `src/vt/cuda/cuda_glm5_next.cu` (571 lines, 4 kernels + 2 launchers) |
| Host reference (the oracle) | `src/vllm/model_executor/models/glm5_next_dsa.cpp` (`GetPooledStates`, `SelectIndexerTopkFromPacked`) |
| ROCm KDA kernel (the template) | `src/vt/rocm/rocm_kda_scan.hip` (157 lines) |
| Seam contract | `include/vt/ops.h:2436-2446` (`Glm5NextKpoolCompressFn`, `Glm5NextKpoolSelectFn`) |
| Op wrappers | `include/vt/ops.h:5174-5200` (`vt::Glm5NextKpoolCompress`, `vt::Glm5NextKpoolSelect`) |
| CUDA registration | `src/vt/cuda/cuda_glm5_next.cu:559-568` |
| ROCm registration pattern | `src/vt/rocm/rocm_ops.hip:293-295` (`kKdaGatedDeltaRule`) |
| Existing CUDA device test | `tests/vllm/models/test_glm5_next_kpool_device.cpp` (626 lines) |
| Transformers reference | `modular_glm5_next.py:821-1022` @ pin v5.16.1 |

## Design

**A mechanical CUDA-to-HIP translation.** The CUDA donor
(`cuda_glm5_next.cu`) is all-`float` and all-`int32_t` — no template dispatch,
no bf16/f16 variants, no dtype-conditional paths. The port is simpler than the
KDA port in every dimension: one concrete type per parameter, no `Ld`/`St`
overloads, no shared-memory staging of typed vectors.

The translation table:

| CUDA | HIP |
|---|---|
| `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` |
| `#include <math_constants.h>` | removed (not needed) |
| `cudaError_t` | `hipError_t` |
| `cudaSuccess` | `hipSuccess` |
| `cudaGetErrorString` | `hipGetErrorString` |
| `cudaStream_t` | `hipStream_t` |
| `cudaMemsetAsync` | `hipMemsetAsync` |
| `cudaMallocAsync` | `hipMallocAsync` |
| `cudaFreeAsync` | `hipFreeAsync` |
| `cudaGetLastError` | `hipGetLastError` |
| `CUDART_INF_F` | `INFINITY` (from `<cmath>`) |
| `__fadd_rn` / `__fmul_rn` / `__fdiv_rn` | same names in HIP |
| `<<<grid, block, shmem, stream>>>` | same syntax in HIP |
| `__global__` / `__shared__` / `__device__` / `__forceinline__` / `__launch_bounds__` | same in HIP |
| `FLT_MAX` | same (from `<cfloat>`) |

**Four kernels, two launchers, two entry points:**

1. `KpoolMetaKernel` (CUDA :101) — per-pool validity + `first_key` (argmax over
   valid keys). One block per batch row, `kMetaThreads = 256`.

2. `KpoolKeepScanKernel` (CUDA :147) — exclusive scan to compact the `keep`
   predicate, outputs `num_pools` as a device scalar. One block,
   `kScanThreads = 1024`.

3. `KpoolPoolKernel` (CUDA :202) — learned pool weighting: per-channel softmax
   over `index_kpool` members with APE, weighted sum of keys. Grid `(np, batch)`,
   `kPoolThreads = 128`.

4. `KpoolSelectKernel` (CUDA :291) — per-head dot score, ReLU, head mix,
   top-k selection reproducing `torch.topk` CPU rule (score desc, index asc
   tiebreak), then `append_visible_tail`. Grid `(seq_len, batch)`,
   `kSelectThreads = 256`.

**Scratch allocation.** The compress launcher uses `cudaMallocAsync` for
`first_key[batch] + raw_valid[batch*np] + dst[np]`, freed after. HIP has
`hipMallocAsync` / `hipFreeAsync` with the same signature.

**No CPU oracle.** The CUDA file header says so explicitly (line 6-7):
"There is no CPU provider for these two ops on purpose: registering the oracle
under the same OpId would make the seam its own golden." The host reference
(`glm5_next_dsa.cpp`) IS the oracle, but it is not registered under the OpId.
The cross-device test calls the host reference functions directly and compares
device output against them.

## Risks

- **Untestable here.** This host has no ROCm toolchain (`hipcc` absent, no
  `/opt/rocm`) and no AMD device. The `.hip` TU cannot be compiled and the
  kernels cannot be run from this session. The ROCm device gate is `PENDING`.
- **The kernels are unreached by the model's forward.** The compose (W9c-0,
  #2415) that would call them does not exist yet. This is a staged slice
  (AGENTS.md §"Nothing lands dead"): the commit body and PR body name what is
  unreached, the owning row (W9c-0), and the tracking issue (#2942). The
  cross-device test exercises the kernels directly through
  `vt::Glm5NextKpoolCompress` / `vt::Glm5NextKpoolSelect` → `Queue` → `GetOp`,
  which is the production entry point for the ops.
- **`hipMallocAsync` availability.** HIP 7.2 (gfx1151) supports
  `hipMallocAsync`. If an older HIP version does not, the fallback is
  `hipMalloc` + `hipFree` (the scratch is small and freed before return).

## Tests

One new case in `tests/vt/test_backend_cross_device.cpp`, placed beside the
KDA case. The test:

1. Builds the same fixture inputs as the CUDA device test
   (`test_glm5_next_kpool_device.cpp`): packed indexer states, APE, q_states,
   head_weights, valid_keys, q_mask — all from the golden fixtures in
   `glm5_next_dsa_goldens.inc`.
2. Calls the host reference (`GetPooledStates`, `SelectIndexerTopkFromPacked`)
   to get the expected output.
3. Iterates `RegisteredDevices()`, skips devices where
   `OpAvailable(kGlm5NextKpoolCompress, dt)` is false.
4. Runs both ops on the device queue.
5. Compares device output against the host reference:
   - `pool_keys`: `MaxAbsDiff <= 2e-5` (same bound as the CUDA test)
   - `pool_indices`: exact match
   - `pool_valid`: exact match
   - `num_pools`: exact match
   - `topk_indices`: exact match (set equality, as in the CUDA test)
   - `index_scores`: `MaxAbsDiff` bounded by the decision margin (same as CUDA test)

**Three assertions per device**, mirroring the KDA precedent:

1. the device result matches the host reference within the tolerance;
2. `vt::OpRegistered(op, DeviceType::kROCM)` is true — the native-only probe;
3. `vt::GetReferenceTierHits()` does not increase across the call.

Assertion 2 is unconditional on a ROCm build.

## Git integration

One pull request for spec and implementation (the default in AGENTS.md).

## Gates

| Gate | Result |
|---|---|
| CPU-only configure + build, `vt_tests` | run in this session |
| New case, CPU-only run | run in this session — the ROCm arm does not execute |
| New case, `gfx1151` ROCm build | `PENDING` — no toolchain and no device in this session |
| `.hip` TU compiles | `PENDING` — `hipcc` is not on this host |
| `scripts/agent-preflight.sh` | run in this session |

## Evidence

- `git log -S'kGlm5NextKpoolCompress' -- src/vt/rocm/` was empty before this
  change: no prior attempt to register the op on ROCm.
- `git log --oneline --grep 'BACKEND-ROCM-KPOOL'` was empty: no prior row.

## Owed

- **The device gate.** A `gfx1151` run of the new test case, proving all
  assertions with a non-zero assertion count. Tracked by #2942.
- **The compose (W9c-0).** The forward path that would call these ops from
  the model. Tracked by #2415 and #2942.

## Stop conditions

- Stop and report `PENDING` rather than claim a device result that was not
  measured on a gfx1151 device.
- Do not lift the CPU-only refusal at `glm5_next_kda.cpp:322` in this row.
- Do not port the compose (W9c-0) in this row.

## What this does NOT do

Registering these ops does **not** put GLM-5.3-Flash's k-pool indexer on a ROCm
queue. The compose (W9c-0, #2415) that would call them does not exist yet. This
change makes the op-table entries exist on ROCm and proves the kernels are
numerically correct against the host reference. It is a staged slice toward a
ROCm k-pool arm, not that arm.

With all four ops of #2942 registered on ROCm (`kMoeGateUpSwiGLuGrouped`,
`kKdaGatedDeltaRule`, `kGlm5NextKpoolCompress`, `kGlm5NextKpoolSelect`), the
op-table is complete. The remaining work is the compose that reaches them.
