# BACKEND-ROCM MoE Silu-Gate Rounding Repair

## Scope

Fix the ROCm `SiluAndMul` / `MoeSiluMul` kernels that compute `silu(gate)` in
f32 and multiply by `up` WITHOUT first narrowing the silu result to the gate
tensor's dtype. The CPU oracle (`src/vt/cpu/cpu_ops.cpp`) narrows via
`RoundThrough(in_dt, ...)` before the multiply, and upstream vLLM's
`silu_kernel` does the same intermediate cast. On exact-equality checks the
bf16 arm diverges.

## Upstream Anchors

- vLLM `csrc/libtorch_stable/activation_kernels.cu::silu_kernel` (line ~158
  at vLLM e126687a9): returns `(T)(((float)x) / (1.0f + expf(...)))` — the
  intermediate is narrowed to `T` (the gate/input scalar type) BEFORE
  `compute` (line ~36) multiplies: `(scalar_t)(ACT_FN(gate, alpha) * ((float)up + beta))`.
  The vectorized `packed_compute` (line ~72) narrows identically via
  `cast_to_packed<packed_t>`.
- CPU oracle: `src/vt/cpu/cpu_ops.cpp::SiluAndMulKernel` (line ~669):
  `float silu = RoundThrough(in_dt, gate / (1.0f + std::exp(-gate)));`
  then `StoreF32(out, ..., silu * up);`
- CPU oracle: `src/vt/cpu/cpu_ops.cpp::MoeSiluMulKernel` (line ~733):
  `const float silu = RoundThrough(in_dt, g / (1.0f + std::exp(-g)));`
  then `StoreF32(out, i, silu * LoadF32(up, i));`
- `RoundThrough` (cpu_ops.cpp:2355): `kF32` → identity; `kBF16` →
  `BF16ToF32(F32ToBF16(v))` (round-trip through bf16); `kF16` →
  `F16ToF32(F32ToF16(v))`.

## Defect

### Variant 1: Dense `SiluMulK` — `src/vt/rocm/rocm_dense_basic.hip:99`

```cpp
St(out, idx, (g / (1.0f + expf(-g))) * up);
```

Computes silu in f32, multiplies by up in f32, stores. No narrowing to the
gate dtype before the multiply. When `Tin = __hip_bfloat16`, the CPU oracle
rounds `silu(gate)` to bf16 precision first, then multiplies — the ROCm kernel
keeps full f32 precision through the multiply, producing different low bits.

### Variant 2: MoE `MoeSiluMulK` — `src/vt/rocm/rocm_moe_router.hip:32`

```cpp
St(out, i, Silu(Ld(gate, i)) * Ld(up, i));
```

Same defect: `Silu()` returns f32, multiplied by `Ld(up, i)` (f32) without
narrowing to the gate dtype first.

## Design

After computing `silu(gate)` in f32, narrow to the gate tensor's dtype BEFORE
multiplying by `up`, mirroring upstream's `silu_kernel` intermediate cast then
`compute` multiply.

### Narrowing helper

Add a `__device__` `NarrowTo` template that round-trips an f32 value through
the gate dtype, matching `RoundThrough` semantics:
- `float` → identity (no narrowing)
- `__hip_bfloat16` → `__bfloat162float(__float2bfloat16(v))`
- `__half` → `__half2float(__float2half(v))` (for future f16 support)

### Dense `SiluMulK` fix

The kernel is templated `<typename Tin, typename Tout>`. The gate dtype is
`Tin`. After computing `g / (1.0f + expf(-g))`, narrow via
`NarrowTo<Tin>(...)` before multiplying by `up`:

```cpp
const float silu = NarrowTo<Tin>(g / (1.0f + expf(-g)));
St(out, idx, silu * up);
```

### MoE `MoeSiluMulK` fix

The kernel is templated `<typename Tout, typename Tg, typename Tu>`. The gate
dtype is `Tg`. After computing `Silu(Ld(gate, i))`, narrow via
`NarrowTo<Tg>(...)` before multiplying by `Ld(up, i)`:

```cpp
St(out, i, NarrowTo<Tg>(Silu(Ld(gate, i))) * Ld(up, i));
```

## Risks

- **f32 paths unchanged**: `NarrowTo<float>` is identity, so f32-in/f32-out
  paths are bit-identical to before.
- **bf16 paths now match CPU oracle**: the narrowing round-trip is exactly
  what `RoundThrough(kBF16, ...)` does on the CPU side.
- **No new dtype arms**: the dense kernel currently only dispatches f32 and
  bf16. The MoE kernel dispatches f32 and bf16 for all three slots. No f16
  path is live, but `NarrowTo<__half>` is defined for completeness.
- **Performance**: one extra cast per element on bf16 paths — negligible
  (already loading/storing at that width).

## Tests

A focused self-skipping test `test_ops_rocm_silu_rounding.cpp` that:
- Skips when no ROCm device is available (mirrors `test_rocm_backend.cpp`
  guard pattern via `vt::rocm::DeviceAvailable()`).
- Runs both `SiluAndMul` and `MoeSiluMul` on the ROCm backend across f32 and
  bf16 dtypes and multiple shapes.
- Compares ROCm output against the CPU oracle (run through the same `vt::`
  entry points) with EXACT equality on the bf16 arms (raw uint16 bits) and
  exact f32 equality on the f32 arms.
- Named after the defect class: `test_ops_rocm_silu_rounding`.

Registered in `tests/CMakeLists.txt` inside the `if(VLLM_CPP_HIP)` block,
following the `vllm_cpp_add_test` pattern.

## Gates

- `python3 scripts/check-env-doc.py` — must stay green.
- `python3 scripts/check-agent-record.py --check` — must stay green.
- Docker HIP compile (no GPU): `cmake -G Ninja -DVLLM_CPP_HIP=ON
  -DVLLM_CPP_HIP_ARCHITECTURES=gfx1100 && ninja vllm-cli
  test_ops_rocm_silu_rounding` — must compile clean.

## Evidence

- Commit list (spec + fix + test).
- Every silu-mul variant found and fixed (file:line each).
- Upstream citation (file:line).
- Test name + CMakeLists registration line.
- Docker compile result.
- Gate scripts output.

## Stop Conditions

NEEDS_DECISION if the ROCm silu-mul path's dtype contract is not a
narrow-to-gate-dtype (e.g. the kernel has no dtype parameter to narrow to).
Current signatures: `SiluMulK<Tin, Tout>` and `MoeSiluMulK<Tout, Tg, Tu>` —
both carry the gate dtype as a template parameter, so narrowing is
well-defined. No stop condition triggered.

## Issue Linkage

- #2889 (open): names three pre-existing bugs including MoE silu rounding.
- #1954: records the ROCm exactness failure.
- The PR that lands this closes #2889's silu item. Do NOT close any issue
  in the commit.
