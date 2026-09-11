# ROCM-MOE-GATEUP-SWIGLU-GROUPED — `vt::OpId::kMoeGateUpSwiGLUGrouped` on ROCm, the one op GLM-5.3-Flash asks for and does not get

Row: `BACKEND-ROCM`
Issue: [#2942](https://github.com/mudler/vllm.cpp/issues/2942)

## Now

`ACTIVE`, implementation written, device gate `PENDING`. Base
`32f83ac0c64f6539c4a709ae7e49664f0158d630`.

## Scope

Register a native `kMoeGateUpSwiGLUGrouped` kernel for `DeviceType::kROCM` in a
new translation unit `src/vt/rocm/rocm_moe_gate_up_swiglu.hip`, add it to the
two `VLLM_CPP_HIP` source lists in `CMakeLists.txt`, and add the cross-device
test case that separates a native registration from the portable reference tier.

Out of scope, and deliberately so — see "The reached set" below:
`kKdaGatedDeltaRule`, `kGlm5NextKpoolCompress`, `kGlm5NextKpoolSelect`. #2942
lists all four as unregistered on ROCm. Only this one is reached on a device
queue by this model.

## The reached set, established against the tree and not taken from the issue

`src/vllm/model_executor/models/glm5_next_forward.cpp:307-310`, on any non-CPU
queue and before a single operand is built:

```cpp
    const bool gate_up_here =
        vt::OpRegistered(vt::OpId::kMoeGateUpSwiGLUGrouped, queue.device.type);
    const bool down_here =
        vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped, queue.device.type);
```

`src/vt/rocm/rocm_ops.hip:261` registers `kMatmulBTQuantGrouped` on
`DeviceType::kROCM`. Nothing under `src/vt/rocm/` registers
`kMoeGateUpSwiGLUGrouped`; the only registrations are
`src/vt/cuda/cuda_quant_dot.cu:2867` (`kCUDA`) and
`src/vt/cpu/cpu_quant_gemm.cpp:310` (`kCPU`). `gate_up_here` is therefore the
one half of that pair that is false on ROCm, and the forward throws at
`:315-327` naming exactly it.

The other three are **not** reached on a device queue:

- `kKdaGatedDeltaRule` — `src/vllm/model_executor/models/glm5_next_kda.cpp:322`
  is `VT_CHECK(queue.device.type == vt::DeviceType::kCPU, ...)`, so the only
  call site this model has (`:404`) can never run on a ROCm queue. It is a host
  arm by construction. The other callers of `vt::KdaGatedDeltaRule` are Kimi
  Linear's, which this row does not own.
- `kGlm5NextKpoolCompress` / `kGlm5NextKpoolSelect` — the only probe is
  `src/vllm/model_executor/models/glm5_next_device.cpp:12-13`, and
  `include/vllm/model_executor/models/glm5_next_device.h` says in its own words
  that **no production path consults it yet** and records the debt as the flash
  row's O36 (#2415). Porting them would land dead.

That is the same economy #2942 predicted from the non-flash row: the reached set
is smaller than the missing set.

## Upstream anchors

| Role | Path |
|---|---|
| Numerics donor (the golden) | `src/vt/cpu/cpu_quant_gemm.cpp:282-299`, `MoeGateUpSwiGLUGroupedKernel` |
| Seam contract | `src/vt/ops.cpp:256-285` (`vt::MoeGateUpSwiGLUGrouped`), `include/vt/ops.h:2197-2199` (`MoeGateUpSwiGLUGroupedFn`) |
| Epilogue definition | `src/vt/cuda/cuda_quant_dot.cu:1090-1142`, `QuantDotGemmGroupedFusedSwiGLUKernel` — read for the epilogue formula only |
| Reused ROCm GEMM | `src/vt/rocm/rocm_grouped_gemm.hip:896`, `MatmulBTQuantGroupedKernelRocm` |
| HIP idiom, TU shape | `src/vt/rocm/rocm_moe_chain.hip` |
| Wavefront width, and why no warp primitive | `src/vt/rocm/rocm_rmsnorm.hip:39-44` |
| Grow-only stream scratch | `src/vt/grow_only_stream_scratch.h`, used as `src/vt/rocm/rocm_grouped_gemm.hip:620-635` does |

## Design

**Costed from the CPU reference, as #2942 requires.** The CPU golden is not a
monolithic kernel. It is a composite, and its own comment says so: two grouped
keep-quant GEMMs through `MatmulBTQuantGroupedKernel` into f32 temporaries, then
one elementwise clamped-SwiGLU pass

```text
gate = min(g, limit);  up = clamp(u, -limit, limit);  out = gate*sigmoid(gate)*up
```

with no extra scale, because the grouped GEMM has already folded the weight
`FinalFactor` into `g` and `u`. `include/vt/merged_gemm.h:117` states the same
polarity for the bf16 twin: the fused op is defined as **bit-identical** to that
composite.

The ROCm arm is therefore that composite:

1. `MatmulBTQuantGroupedKernelRocm(q, gate_out, act, gate_w, expert_ids)` into a
   device f32 temporary;
2. the same call again with `up_w` into a second device f32 temporary;
3. **one new HIP kernel**, `MoeGateUpSwiGLUEpilogueK`, over the `P*N`
   elements.

This is one kernel, one entry point, one `RegisterOp` line, and it reuses the
already-gated ROCm grouped GEMM instead of writing a second copy of the Q8_0 /
Q4_K / Q5_K / Q6_K dot cores — which is what "never write a parallel path by
hand" forbids.

**No warp-level primitive, and none needed.** The epilogue is elementwise: no
cross-lane reduction exists in it, so there is no `warpSize` assumption to get
wrong. AMD's wavefront is 64 lanes where NVIDIA's warp is 32
(`rocm_rmsnorm.hip:39-44`), and the CUDA donor's `__shfl_down_sync(0xffffffffu,
…)` tree is exactly the construct that does not survive the move. Sidestepping
it rather than translating it is the point of costing from the CPU reference.
**No MFMA**: gfx1151 is RDNA and has none.

**Scratch.** `2*P*N` floats from a grow-only per-stream pool
(`vt::GrowOnlyStreamScratch<hipStream_t>` + `hipMallocAsync`), the discipline
`rocm_grouped_gemm.hip:606-635` sets out: `hipMalloc`/`hipFree`/
`hipStreamSynchronize` per call are illegal under `hipGraph` capture, and a
captured graph may have baked the pointer, so the pool never frees. This TU
keeps its **own** pool: the grouped GEMM's `EnsureQuantScratch` block holds the
quantized activation and is rewritten by each of the two GEMM calls, so sharing
it would clobber the gate result while the up GEMM runs.

The two GEMM calls each re-quantize the same activation into that shared block.
That is redundant work, not a defect — the calls are stream-ordered, so the
second sees a correctly rewritten buffer. It is priced under `## Owed`.

**Dtype refusals.** The seam (`ops.cpp:256-285`) already validates rank, shapes,
same-block-quant dtype for both towers, f32 output, `K % BlockElems == 0`,
packed activation rows, contiguity and device match. The kernel re-asserts the
f32 output and the equal gate/up dtype, because `Tensor::Ptr<T>()` is an
unchecked cast — the hazard `rocm_moe_chain.hip:141-146` names from the #509
review sweep.

## Risks

- **Untestable here.** This host has no ROCm toolchain (`hipcc` absent, no
  `/opt/rocm`) and no AMD device. The `.hip` TU cannot be compiled and the
  kernel cannot be run from this session. Every ROCm-side result is `PENDING`.
- **The redundant activation quantization** doubles the quantize cost relative
  to the fused CUDA kernel and writes `2*P*N` f32 to HBM that the fused kernel
  keeps in registers. This is a speed property. It is owed, not a correctness
  risk.
- **`docs/ROCM.md:60-61` still disqualifies a GLM-5.3-Flash ROCm speed number**
  while any reached op serves from the reference tier. This change removes one
  such op from the Flash path's device-gate pair; it does not on its own make a
  speed axis valid.

## Tests

One new case in `tests/vt/test_backend_cross_device.cpp`, placed beside the
`kMatmulBTQuantGrouped` case it is modelled on (`:3301`) and using that case's
fixture shape, block builders and tolerance — `kNmseTol = 5e-4` (`:58`). No
wider tolerance is invented.

**Three assertions per device, and the middle one is the load-bearing one.**
The file's own header note at `:3845-3858` says why:

1. the device result matches the CPU oracle at `Nmse <= kNmseTol`;
2. `vt::OpRegistered(op, DeviceType::kROCM)` is true — the native-only probe
   (`src/vt/op_provider.cpp:801-825`), the **only** one of the three that can
   tell a native kernel from the reference tier, because the tier computes the
   same answer and an oracle-equality assertion is green with no kernel at all;
3. `vt::GetReferenceTierHits()` does not increase across the call.

Assertion 2 is unconditional on a ROCm build and is not `if (!OpAvailable)
continue` — a missing registration is the defect under test, so it must fail
rather than skip.

## Gates

| Gate | Result |
|---|---|
| CPU-only configure + build, `vt_tests` | run in this session |
| New case, CPU-only run | run in this session — **the ROCm arm does not execute**, see below |
| New case, `gfx1151` ROCm build | `PENDING` — no toolchain and no device in this session |
| `.hip` TU compiles | `PENDING` — `hipcc` is not on this host |
| `scripts/agent-preflight.sh` | run once in this session |

**The RED is only observable on a HIP build, and this is stated rather than
worked around.** `RegisteredDevices()` (`:88-101`) probes with
`vt::GetBackend(kROCM)`, which throws unless the backend TU was compiled and
linked. On a CPU-only build the new case returns early and executes **zero**
device assertions. An assertion count is what proves a case ran
(`.agents/verification.md`), and on this host that count is zero for the ROCm
arm. Reporting a CPU-only green as a RED-then-GREEN would be the exact failure
the brief names.

## Evidence

- `git log -S'kMoeGateUpSwiGLUGrouped' -- src/vt/rocm/` was empty before this
  change: no prior attempt to reconcile.
- `rc devices` on 2026-09-06 reported `strix:gpu0 busy
  mudler@glm5flash-2942`, so the one gfx1151 device is held by another
  session for this same issue.

## Owed

- **The fused single-kernel arm.** One HIP kernel computing both dots against
  one shared quantized activation with a `__syncthreads()` shared-memory
  reduction tree, removing the `2*P*N` f32 HBM round trip and the duplicate
  activation quantization. This is #2942's split-KV analogue: a performance
  wave, not a prerequisite. Tracked by #2942.
- **The device gate.** A `gfx1151` run of the new case, proving all three
  assertions with a non-zero assertion count. Tracked by #2942.
- **The KERNEL BODY is unreached on any default configuration.** The
  `RegisterOp` line IS reached by default — `glm5_next_forward.cpp:307-310`
  probes it on every non-CPU queue before any environment variable is read, so
  the refusal at `:315-327` stops firing on ROCm the moment this lands. The
  kernel itself is another matter: both production call sites are behind an
  opt-in that defaults OFF.
  `glm5_next_forward.cpp:331` needs `VT_GLM5_NEXT_DEVICE_EXPERTS=1` (#2464), and
  Laguna's fused gate/up arm (`laguna.cpp:1032`, reached from `:1247`) needs
  `VT_LAGUNA_FUSED_GATEUP=1` (`:1052-1058`). `vt::MergedGemm`
  (`merged_gemm.cpp:38-40`) refuses a non-CPU device without this op and has no
  caller under `src/vllm/` today. So on a default ROCm run nothing executes
  `MoeGateUpSwiGLUEpilogueK`. Tracked by #2942, and by #2464 for the GLM-5.3
  half, which owns the SIGSEGV that made its gate default off.
- **`kKdaGatedDeltaRule` on ROCm.** Not reached today
  (`glm5_next_kda.cpp:322`); it becomes reachable the day the KDA arm takes a
  device queue. Tracked by #2942.
- **`kGlm5NextKpoolCompress` / `kGlm5NextKpoolSelect` on ROCm.** Blocked behind
  the flash row's O36 (#2415), which owns the compose that would consult the
  probe at all. Tracked by #2942.

## Stop conditions

- Stop and report `PENDING` rather than claim a device result that was not
  measured on a gfx1151 device.
- Stop if the composite turns out not to be admissible as the ROCm arm — if a
  reviewer holds that the op's contract requires register-resident
  intermediates rather than the CPU golden's composite semantics. That is a
  `NEEDS_DECISION`, not a silent re-scope.
- Do not port the other three ops of #2942 in this row.

## What this does NOT do

Registering this op does **not** put GLM-5.3-Flash on a ROCm queue. A second
gate stands after it:
`src/vllm/model_executor/models/glm5_next_forward.cpp:49-55` reads
`VT_GLM5_NEXT_DEVICE_EXPERTS` and admits `"1"` and nothing else, and
`:331-341` refuses the device arm when it is unset. It defaults off because both
`--device cuda` legs against the published 101.24 GiB UD-Q2_K_XL artifact died
with SIGSEGV having emitted no token, 3 of 3 (#2464, root-caused in #2480). This
change makes the op-table half of the refusal stop firing on ROCm. The opt-in
half is untouched and still defaults off.
