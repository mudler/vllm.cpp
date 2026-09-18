# ROCM-KDA-GATED-DELTA-RULE — `vt::OpId::kKdaGatedDeltaRule` on ROCm, the per-K-channel-decay recurrence

Row: `BACKEND-ROCM`
Issue: [#2942](https://github.com/mudler/vllm.cpp/issues/2942)

## Now

`ACTIVE`, implementation committed (`8d0186d79`), device gate `PENDING`.
CPU-only build and test pass; the ROCm `.hip` TU and the `gfx1151` test
case are pending a HIP toolchain and AMD device. Base
`ed03e50ee8e8e3723f9b8c5fae1f1a5f6e8e8a5d`.

## Scope

Register a native `kKdaGatedDeltaRule` kernel for `DeviceType::kROCM` in a new
translation unit `src/vt/rocm/rocm_kda_scan.hip`, add it to the two
`VLLM_CPP_HIP` source lists in `CMakeLists.txt`, register the op in
`rocm_ops.hip`, and add the cross-device test case that proves the ROCm kernel
matches the CPU oracle.

Out of scope: the other two unregistered ops of #2942 (`kGlm5NextKpoolCompress`,
`kGlm5NextKpoolSelect`), which have no CPU provider and land dead without a
compose that does not exist yet (W9c-0). Also out of scope: lifting the
CPU-only refusal at `glm5_next_kda.cpp:322` that prevents the model's forward
from reaching this op on a device queue. That is W9c-2's job.

## The reached set

`kKdaGatedDeltaRule` is the second of #2942's four ops. The first,
`kMoeGateUpSwiGLUGrouped`, landed in PR #3001 (spec
`rocm-moe-gateup-swiglu-grouped.md`). This op is different from that one in one
critical way: **it is not reached on a device queue today.**

The model's only call site is `glm5_next_kda.cpp:404`, inside
`Glm5NextKdaLayerForward`. That function begins with a CPU-only refusal at
`:322`:

```cpp
VT_CHECK(queue.device.type == vt::DeviceType::kCPU,
         "glm5_next kda: Glm5NextKdaLayerForward is the HOST reference and "
         "needs a CPU queue; the device arm is the assembled text forward's "
         "(W5, .agents/specs/glm5-next-flash.md)");
```

The entire layer — q/k/v projections, short conv, L2 norm, forget gate — is a
host reference over `std::vector<float>`. The `vt::KdaGatedDeltaRule` call at
`:404` is the only `vt::` op in the layer. The refusal exists because handing
host pointers to a device queue is a crash, not a fallback.

This op therefore lands as a **staged slice**: the `RegisterOp` line and the
kernel are unreached by the model's forward on its default configuration. The
commit body and PR body name this, the row that owns the wiring (W9c-2), and the
issue that tracks it (#2942). The sibling spec
(`rocm-moe-gateup-swiglu-grouped.md`) already lists this op under `## Owed`.

On `gfx1151` (Strix Halo, an APU), `DeviceMemoryIsHostAddressable()` is true, so
the reference tier installs the CPU kernel as a provider and the op does not
throw. Registering a native ROCm kernel makes the op run on the GPU instead of
the host CPU, and makes it available on discrete AMD cards where the reference
tier is dead.

## Upstream anchors

| Role | Path |
|---|---|
| Numerics donor (the golden) | `src/vt/cpu/cpu_ops.cpp:2241`, `KdaGatedDeltaRuleKernel` (recurrence math: `KdaHeadTokenStep` :2208) |
| CUDA kernel (the port donor) | `src/vt/cuda/cuda_gdn.cu:3139-3235`, `KdaScanKernel` + `LaunchKdaScan` + `KdaGatedDeltaRuleKernelCuda` |
| ROCm GDN kernel (the template) | `src/vt/rocm/rocm_gdn_scan.hip` (181 lines, `GdnScanK` + launchers) |
| Seam contract | `include/vt/ops.h:2348-2350` (`KdaGatedDeltaRuleFn`), `include/vt/ops.h:3862` (`vt::KdaGatedDeltaRule`) |
| CUDA registration | `src/vt/cuda/cuda_gdn.cu:6648-6649` |
| ROCm registration pattern | `src/vt/rocm/rocm_ops.hip:282-285` (`kGdnPrefill`/`kGdnDecode`) |
| FLA reference | `third_party/flash_linear_attention/ops/fused_recurrent.py:88-175` @ pin `555967922`, `IS_KDA=True` |

## Design

**The KDA scan is the GDN scan with per-K-channel decay.** The CUDA donor says
so in its own header (`cuda_gdn.cu:3131-3134`): "Byte-for-byte GdnScanKernel
EXCEPT the state decay is per-K-channel: GDN uses one scalar
`decay = expf(g[t*hv_n+hv])` for the whole [Dv,Dk] state; KDA stages a per-K
vector `decay[ki] = expf(g[(t*hv_n+hv)*dk + ki])` in shared memory and applies
`s_row[ki] *= decay[ki]`."

The ROCm GDN kernel (`rocm_gdn_scan.hip`) is a clean hand-translation of the
CUDA GDN kernel, using HIP idioms (`Ld`/`St` helpers instead of `Load`/`Store`,
`__hip_bfloat16` instead of `__nv_bfloat16`, `hipStream_t` instead of
`cudaStream_t`). The KDA port mirrors this pattern: it is the same
hand-translation, one step further.

**The three differences from `GdnScanK`** (all visible in the CUDA donor at
`cuda_gdn.cu:3157-3186`):

1. **Shared memory: 3\*dk, not 2\*dk.** GDN stages `[dk] q'` and `[dk] k` (2*dk
   floats). KDA adds `[dk] per-K decay` (3*dk floats). The line is
   `float* d_sh = smem + 2 * dk;`.

2. **Per-K-channel decay, not scalar.** GDN: `const float decay = expf(g[t *
   hv_n + hv]);` — one scalar, indexed `[t * hv_n + hv]`. KDA:
   `d_sh[i] = expf(g[(t * hv_n + hv) * dk + i]);` — a vector of `dk` floats,
   indexed `[(t * hv_n + hv) * dk + i]`. The gate tensor `g` has shape
   `[T, Hv, Dk]` in KDA vs `[T, Hv]` in GDN.

3. **State math uses the vector.** GDN: `const float decayed = Ld(s_row, ki) *
   decay;` (scalar multiply). KDA: `const float decayed = Ld(s_row, ki) *
   d_sh[ki];` (per-element multiply). The rank-1 update is the same change:
   `Load(s_row, ki) * d_sh[ki]` instead of `Load(s_row, ki) * decay`.

Everything else — the grid `(hv_n, n)`, the block size `kBlock = 256`, the
NULL-slot zero-out path, the `qsl` varlen logic, the token loop, the
`__syncthreads` discipline — is identical to `GdnScanK`.

**State dtype: float only.** The CUDA KDA launcher hardcodes `float` as
`TState` (`LaunchKdaScan<float, Tout, float>`). The GDN ROCm launcher dispatches
on state dtype (f16/bf16/f32) because GDN supports compressed recurrent state.
KDA does not — the CUDA donor has no f16/bf16 state path. The ROCm port mirrors
this: `KdaScanK<Tin, Tout, float>` only, no state-dtype dispatch.

**No `state_idx` in the launcher.** The CUDA KDA launcher passes `nullptr` for
`state_idx` (decode with state indexing is handled by the GDN decode path, not
KDA). The ROCm port does the same. The kernel retains the `state_idx` parameter
and the NULL-slot zero-out path for structural parity with `GdnScanK`, but the
launcher always passes `nullptr`.

**Dtype dispatch.** The entry point `KdaGatedDeltaRuleKernelRocm` dispatches on
`q_in.dtype` (f32/bf16) and `out.dtype` (f32/bf16), yielding four template
instantiations — the same 2×2 matrix as the CUDA donor
(`cuda_gdn.cu:3223-3234`). State is always f32.

**Shared-memory limit check.** The CUDA donor checks
`3 * dk * sizeof(float) <= 48 * 1024`. The ROCm port mirrors this. At GLM-5.3
geometry (Dk=128), this is 1536 bytes, well within any limit.

## Risks

- **Untestable here.** This host has no ROCm toolchain (`hipcc` absent, no
  `/opt/rocm`) and no AMD device. The `.hip` TU cannot be compiled and the
  kernel cannot be run from this session. The ROCm device gate is `PENDING`.
- **The kernel is unreached by the model's forward.** The CPU-only refusal at
  `glm5_next_kda.cpp:322` blocks the call site at `:404` from running on any
  non-CPU queue. This is a staged slice (AGENTS.md §"Nothing lands dead"): the
  commit body and PR body name what is unreached, the owning row (W9c-2), and
  the tracking issue (#2942). The cross-device test exercises the kernel
  directly through `vt::KdaGatedDeltaRule` → `Queue` → `GetOp`, which is the
  production entry point for the op.
- **No MFMA.** gfx1151 is RDNA and has no matrix units. The scan is a
  sequential recurrence with no matrix multiply; no MFMA is needed or wanted.

## Tests

One new case in `tests/vt/test_backend_cross_device.cpp`, placed beside the GDN
prefill/decode case (`:1677`) and using the same fixture pattern: CPU oracle
first, then iterate `RegisteredDevices()`, skip devices where
`OpAvailable(vt::OpId::kKdaGatedDeltaRule, dt)` is false, and check
`Nmse(ref_out, dev_out) <= kNmseTol` on both output and in-place state.

The KDA-specific shape difference from the GDN test: the gate `g` has shape
`[T, Hv, Dk]` (per-K-channel), not `[T, Hv]` (scalar). This is the one shape
that distinguishes KDA from GDN and is the load-bearing assertion: if the
kernel reads `g` at the wrong stride, the per-channel decay is wrong and the
output diverges.

**Three assertions per device**, mirroring PR #3001's precedent:

1. the device result matches the CPU oracle at `Nmse <= kNmseTol`;
2. `vt::OpRegistered(op, DeviceType::kROCM)` is true — the native-only probe,
   the only one that can tell a native kernel from the reference tier;
3. `vt::GetReferenceTierHits()` does not increase across the call.

Assertion 2 is unconditional on a ROCm build and is not `if (!OpAvailable)
continue` — a missing registration is the defect under test.

## Gates

| Gate | Result |
|---|---|
| CPU-only configure + build, `vt_tests` | run in this session |
| New case, CPU-only run | run in this session — the ROCm arm does not execute |
| New case, `gfx1151` ROCm build | `PENDING` — no toolchain and no device in this session |
| `.hip` TU compiles | `PENDING` — `hipcc` is not on this host |
| `scripts/agent-preflight.sh` | run in this session |

## Evidence

- `git log -S'kKdaGatedDeltaRule' -- src/vt/rocm/` was empty before this change:
  no prior attempt to register the op on ROCm.
- `git log --oneline --grep 'BACKEND-ROCM-KDA'` was empty: no prior row.

## Owed

- **The device gate.** A `gfx1151` run of the new test case, proving all three
  assertions with a non-zero assertion count. Tracked by #2942.
- **Lifting the CPU-only refusal.** `glm5_next_kda.cpp:322` blocks the model's
  forward from reaching this op on a device queue. That is W9c-2's job (spec
  `glm5-next-flash.md`), and it depends on W9b (keep-quant residency) making
  the operands device-resident first. Tracked by #2942 and #2410.
- **`kGlm5NextKpoolCompress` / `kGlm5NextKpoolSelect` on ROCm.** The remaining
  two ops of #2942. Blocked behind W9c-0 (#2415), which owns the compose that
  would consult them. Tracked by #2942.

## Stop conditions

- Stop and report `PENDING` rather than claim a device result that was not
  measured on a gfx1151 device.
- Stop if the per-K-channel decay vector cannot be staged in shared memory at
  GLM-5.3 geometry (Dk=128). It can: 3*128*4 = 1536 bytes, well under 48 KiB.
- Do not lift the CPU-only refusal at `glm5_next_kda.cpp:322` in this row. That
  is W9c-2's job and depends on W9b.
- Do not port the k-pool pair in this row.

## What this does NOT do

Registering this op does **not** put GLM-5.3-Flash's KDA layers on a ROCm queue.
Two more gates stand after it:

1. The CPU-only refusal at `glm5_next_kda.cpp:322` (W9c-2).
2. The keep-quant residency that makes the KDA layer's operands device-resident
   (W9b), which the spec (`glm5-next-flash.md` §W10) already showed cannot fit
   on `gfx1151` at the published artifact's encoding.

This change makes the op-table entry exist on ROCm and proves the kernel is
numerically correct against the CPU oracle. It is a staged slice toward a ROCm
KDA arm, not that arm.
