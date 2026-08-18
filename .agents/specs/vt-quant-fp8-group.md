# VT-QUANT-FP8-GROUP — dynamic per-token, per-group FP8 activation quantization

Issue: [#1189](https://github.com/mudler/vllm.cpp/issues/1189), milestone **M1**.
Row: `VT-QUANT-FP8-GROUP`.
Pinned oracle: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`
(`.agents/upstream-sync.md`), HEAD of the local checkout verified before every
`file:line` below was read.

## Scope

Add one `vt` op, `vt::QuantFp8Group`, with a CPU kernel and a CUDA kernel:

```c++
void QuantFp8Group(Queue& q, Tensor& out_fp8, Tensor& out_scale,
                   const Tensor& x, int group_size);
```

`x` is `[M, K]` f32 or bf16. `out_fp8` is `[M, K]` i8 that carries raw
fp8-e4m3fn bytes. `out_scale` is `[M, K / group_size]` **f32**. The op refuses
`K % group_size != 0` by name.

This is the activation half of block-wise FP8. It is milestone M1 of #1189 and
it deliberately stops there.

**Out of scope, each owned by a later milestone of #1189**: the block-scaled
GEMM (M2), `Fp8BlockWeight` and the loader rung (M3), `Fp8BlockLinearMethod`
and the Qwen3.5 wiring (M4), the mainloop-scaled CUTLASS kernel (M5), and the
merged `gate_up` / QKV projections (M6).

## Upstream anchors

Read the whole executing chain, not the top-level Python. On a CUDA platform
with a contiguous input, `per_token_group_quant_fp8` does **not** run its Triton
kernel: it calls the C++ custom op and returns
(`vllm/model_executor/layers/quantization/utils/fp8_utils.py:635-650`). The
Triton kernel below it is the fallback for every other platform.

| What | Where |
|---|---|
| Python entry point, defaults, refusals | `vllm/model_executor/layers/quantization/utils/fp8_utils.py:567-650` |
| the divisibility refusal this op mirrors | `fp8_utils.py:596-599` |
| the contiguity refusal | `fp8_utils.py:600` |
| row-major scale allocation `[M, K/group_size]` f32 | `fp8_utils.py:629-631` |
| CUDA dispatch, taken whenever the platform is CUDA-alike and `x` is contiguous | `fp8_utils.py:635-650` |
| **the executing CUDA kernel**: group absmax and scale | `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu:42-74` |
| **the executing CUDA kernel**: divide, clamp, e4m3 store | `per_token_group_quant.cu:76-96` |
| the Triton fallback, for contrast only | `fp8_utils.py:95-150` |
| `fp8_min`, `fp8_max` = `finfo(e4m3fn)` = -448, 448 | `vllm/model_executor/layers/quantization/utils/quant_utils.py:27-35` |
| the native torch reference the upstream test compares against | `tests/kernels/quant_utils.py:157-180` |
| the ported test | `tests/kernels/quantization/test_block_fp8.py:82-118`, grid at `:42-46` |

## Design

### Numerics

Per group of `group_size` contiguous elements of one row:

```text
amax = max(eps, max |x_f32| over the group)      eps = 1e-10
y_s  = amax / 448.0f
q    = min(max(x_f32 / y_s, -448.0f), 448.0f)
out  = e4m3fn(q), round to nearest even, saturating
```

Three details are deliberate, and each is a divergence risk if a later reader
"corrects" it.

**A divide, not a reciprocal multiply.** `vt::QuantFp8Static` multiplies by a
hoisted reciprocal because that is the form upstream ships for the per-tensor
static path (`csrc/quantization/w8a8/fp8/common.cuh:62`). The per-group path is
the opposite. The executing CUDA kernel writes `float y_s = local_absmax /
max_8bit` (`per_token_group_quant.cu:68`) and `static_cast<float>(src) / y_s`
(`per_token_group_quant.cu:85`). Both are true divisions, and the scale changes
per group, so there is no loop-invariant reciprocal to hoist. The Triton
fallback instead writes `scale_raw = _absmax * (1.0 / fp8_max)`
(`fp8_utils.py:145`) with a comment that names the 1-ULP difference. The two
upstream arms therefore disagree by up to one f32 ULP in `y_s`, and upstream's
own test admits that with `rtol=0.15` on the values
(`test_block_fp8.py:112-114`). We mirror the CUDA arm, because that is the arm
that executes on the target architecture.

**`eps` is the initial value of the reduction, not a post-clamp.** The kernel
sets `float local_absmax = eps` (`per_token_group_quant.cu:47`) and reduces
`fmaxf` over it. That is identical to `clamp(min=eps)` in the native reference
(`quant_utils.py:176`), and it makes an all-zero group produce
`y_s = 1e-10 / 448`, not a division by zero.

**The load widens to f32 before the absolute value and before the divide.**
`fabsf(static_cast<float>(src))` (`per_token_group_quant.cu:53`) and
`static_cast<float>(src) / y_s` (`:85`). A bf16 input therefore rounds at
exactly one point, as it does in `vt::QuantFp8Static`.

### Layout and what it excludes

`out_scale` is row-major `[M, K/group_size]` f32, upstream's `column_major_scales
= False` branch (`fp8_utils.py:629-631`). The column-major and TMA-aligned
layouts (`fp8_utils.py:610-628`) exist for the DeepGEMM and CUTLASS GEMMs. No
consumer in this tree can read them yet, so shipping them now would ship a
parameter no caller passes. They are owed below.

`use_ue8m0` is excluded, not forgotten. It rounds the scale up to a power of two
for DeepGEMM (`per_token_group_quant.cu:69-71`). Issue #1189 established that
upstream excludes `qwen3_5_text` from DeepGEMM on family 120
(`vllm/utils/deep_gemm.py:27-46`) and dispatches CUTLASS, so the target path
never sets it.

`eps` stays a named constant rather than a parameter. Upstream exposes it, and
every upstream call site takes the `1e-10` default (`fp8_utils.py:570`).

### Structure

`vt::ScaledFp4Quant` (`include/vt/ops.h:1425-1427`) is the closest existing op:
it is dynamic, per-token, per-group, and it emits a 2-D scale beside the packed
values. `QuantFp8Group` follows its shape. `OpId::kQuantFp8Group` is appended
before `kCount`, the additive convention documented at `include/vt/ops.h:363-368`,
so no existing op id shifts.

The CUDA arm lives in `src/vt/cuda/cuda_quant_fp8.cu`. That file exists because
`vt::QuantFp8Static` once lived in the cutlass-gated translation unit and was
therefore unregistered on every CUDA architecture outside `VT_CUTLASS_FP8_ARCHS`
(#960, and #844 is the same defect seen from the fallback's end). This kernel
has the same property: it is a divide and a convert, with no cutlass dependency,
and it must resolve on every CUDA architecture. It is added to
`scripts/check-cuda-op-arch-gate.py`'s `REQUIRED` set for that reason.

## Risks

| Risk | Control |
|---|---|
| a later reader converts the divide to a hoisted reciprocal, matching the Triton arm and moving emitted bytes near an e4m3 tie | the prose above, plus G1, which compares bytes against a reference that spells the divide out |
| the scale silently widens or narrows | G1 asserts the f32 dtype and the `[M, K/group_size]` shape; the op refuses any other dtype |
| an all-zero group divides by zero | `eps` is the reduction's initial value; G4 feeds an all-zero row |
| a ragged `K` is accepted and reads past the row | the op refuses `K % group_size != 0` by name; G5 asserts the refusal |
| an all-zero output passes every value comparison | G2 and G3 carry a `nonzero == numel` vacuity guard |
| the CPU and CUDA arms drift apart | G6, which is CUDA-gated and currently **owed** |

## Tests

`tests/vt/test_ops_quant_fp8_group_cpu.cpp`, registered in
`tests/CMakeLists.txt`.

- **G1** bitwise, zero tolerance. The CPU op equals an independently written
  reference, byte for byte, over a grid that includes saturation in both signs,
  the subnormal ladder, exact e4m3 ties, and both zeros. The reference encodes
  by an exhaustive nearest-value scan over the 128 finite e4m3fn magnitudes,
  which is a different algorithm from the tree's `F32ToFp8`, so agreement is
  evidence rather than a tautology.
- **G2** the ported upstream case. `test_block_fp8.py:82-118` with its grid
  (`num_tokens` in {7, 2050}, `d` in {512, 4096, 5120, 13824}, `group_size` in
  {64, 128, 512}, bf16 and f32), against the native reference transcribed from
  `quant_utils.py:157-180`, at upstream's tolerances: `rtol=0.15` on the
  dequantized values and `allclose` on the scale. Carries the vacuity guard.
- **G3** the scale is exactly `amax / 448` for a group whose amax is known by
  construction, and `1e-10 / 448` for an all-zero group.
- **G4** shape and dtype contract: `out_scale` is f32 `[M, K/group_size]`, and
  a wrong scale dtype or shape is refused.
- **G5** the refusals: `K % group_size != 0`, a non-contiguous input, a
  device mismatch, each by name.
- **G6** CPU against CUDA, byte for byte, on the identical input. **Owed**: this
  host has no GPU and the row took no lease. The case is written and it never
  reports a silent skip; without a device it asserts the CPU registration and
  prints a banner that names what was not measured, following the G2 precedent
  at `tests/vt/test_ops_fp8_cpu.cpp:279`.

## Gates

| Gate | Command |
|---|---|
| focused | `ctest -R test_ops_quant_fp8_group_cpu --output-on-failure` |
| op provider totality | `ctest -R test_op_provider` |
| the per-tensor sibling, unchanged | `ctest -R test_ops_fp8_cpu` |
| structural | `python3 scripts/check-cuda-op-arch-gate.py` |
| record | `scripts/agent-preflight.sh --fail-on-skip` |

The CUDA arm compiles in this change. It does not run in this change.

## Owed

- The CUDA arm's on-hardware leg. The kernel is written and it compiles, and
  G6 measures nothing without a device. Owed by #1189 milestone M5, which needs
  a GPU anyway; the case is already written, so the debt is a run and not a
  test.
- **Nothing reaches this op yet.** `vt::QuantFp8Group` is dispatched by no
  production entry point at this merge commit: `include/vllm.h` does not expose
  it, no loader builds a `Fp8BlockWeight`, and `ModelRegistry::Forward` has no
  block-FP8 linear method to call it from. The wiring is owned by #1189
  milestone M4 (`Fp8BlockLinearMethod` and the Qwen3.5 dense forward), which
  needs M2 and M3 first. This is the staged-slice exception of
  `.agents/reachability.md`, named here, in the commit body, and in the pull
  request body.
- The column-major and TMA-aligned scale layouts (`fp8_utils.py:610-628`).
  Owed by #1189 milestone M5, which is the first consumer that can read them.
- `use_ue8m0` scale rounding (`per_token_group_quant.cu:69-71`). Not owed by a
  milestone: the target architecture never selects DeepGEMM
  (`vllm/utils/deep_gemm.py:27-46`). It becomes owed when a DeepGEMM path lands.

## Stop conditions

Stop and report `NEEDS_DECISION` if any of the following holds.

- The pinned oracle's checkout is not at
  `5559679229bc961848b121ccdeaa8fa5d79bec98`. Every anchor above was read at
  that revision.
- G1 cannot be made bitwise green without widening its tolerance. A tolerance
  on a byte comparison is the defect, not the fix.
- The op cannot express upstream's contract without a parameter that no caller
  in this tree passes.

Stop and report `NEEDS_CONTEXT` if the work requires a GPU lease. The row is
scoped so that it does not.

## Evidence

Recorded in the pull request body: the RED capture before the implementation
existed, the GREEN capture after, and the per-block gate counts from
`scripts/agent-preflight.sh --fail-on-skip`.
