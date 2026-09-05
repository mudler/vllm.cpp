# qwen4_exp: the hyper-connection inject weight's residency

Row: `MODEL-MM-QWEN4-EXP`
Issue: [#2449](https://github.com/mudler/vllm.cpp/issues/2449)
Base: `origin/main` `7a1a98760`

This is a scoped repair spec, not a wave spec. It owns two call sites in one
file. `.agents/specs/qwen4-exp-flash-next.md` remains the row's spec, and this
file exists beside it rather than inside it because that file is edited by three
live waves at once and a repair does not need to take that lock.

## 1. Scope

**In scope.** The two `block_inject_weight` operands of
`src/vllm/model_executor/models/qwen4_exp_forward.cpp` — line 418 (the attention
hyper-connection) and line 476 (the MLP hyper-connection) — and a gate that
enters through `ModelRegistry::Forward`.

**Out of scope, deliberately.** `src/vt/ops.cpp` (the validation is correct and
the file is contended), `qwen4_exp_qsa_block.cpp` (#2422), `cuda_quant_dot.cu`
(#2423), `gguf_keep_quant.cpp` and `model_loader.cpp` (#2397). Whatever stops
the forward AFTER layer 0 is out of scope by construction: this row's claim is
about one operand, and the gate says so in its own assertion.

## 2. What was measured

`ModelRegistry::Forward` on a CUDA queue over the tiny `qwen4exp` GGUF fixture
stops at decoder **layer 0** with

```text
vt: qwen4_exp_gated_residual: block_inject_weight device mismatch
```

Measured on `thor:gpu0` under an `rc` lease, twice, in two legs of one job that
differ only in the QSA block's source (#2449). It fires before PLE, before the
MoE, and long before the first `qwen_sparse_attention` layer, so it is the first
stop for every CUDA forward of this architecture.

## 3. Cause, read from the tree

`vt::Qwen4ExpGatedResidual` takes four weight operands. Three go through
`dense_attn::ResidentWeight`, which is the seam that puts a weight where the
queue's kernels can read it: on a CPU queue it aliases the host bytes, on any
device platform it allocates, uploads once, and memoizes on `w.d_dev`
(`include/vllm/model_executor/models/dense_attn_block.h:181-269`).

The fourth went through `OwnedTensor::View()`, whose last act on the device
field is

```cpp
t.device = vt::Device{};  // default = CPU host
```

(`src/vllm/model_executor/models/qwen3_5_weights.cpp:500`). So the tensor carries
a host pointer stamped `kCPU` whatever the queue is, and the op's own validation
refuses it.

`ResidentWeight`'s body already names this exact pair as a hazard, for the
`repacked` marker rather than for the device:

> `qwen4_exp_forward.cpp` hands `hc_*_down`/`hc_*_up` to
> `vt::Qwen4ExpGatedResidual` through THIS function (:421-422, :479-480,
> :538-539) while it hands `hc_*_inject` through `OwnedTensor::View()`
> (:417, :475)

so the split between the two halves of one op's operands was known and its
device consequence was not.

### 3.1 The withdrawn half of #2449

The issue as filed said `check_projection` skips the device check for a
block-quantized operand, and therefore that the released Q8_0 checkpoint would
hand a device kernel a host pointer instead of producing a message. That is
false and is withdrawn on the issue. Reading `src/vt/ops.cpp:2590-2599`, the
`VT_CHECK(t.device == q.device, ...)` at `:2598` is INSIDE the block-quant
branch, and `check_operand` carries its own identical check for the float path.
Both arms check; the observed message IS the block-quant arm firing. This is a
clean refusal on both arms, not silent memory corruption. Priority is unchanged.

## 4. Design

Two lines. Route both operands through the seam their siblings already use:

```cpp
Tensor bi = dense_attn::ResidentWeight(d, lw.attn_hc.inject, {hc, W});
Tensor bi = dense_attn::ResidentWeight(d, lw.mlp_hc.inject, {hc, W});
```

`{hc, W}` is the shape `vt::Qwen4ExpGatedResidual` validates for this operand
(`ops.cpp`: `shape[0] == args.hc_count && shape[1] == flat`, `flat = hc * H =
p.stream_width()`), and it is the shape the loader required at load
(`qwen4_exp_weights.cpp:269-270`, `LoadMatmul(..., p.hc_count, stream)`), so the
explicit shape restates the contract rather than reinterpreting the bytes.

No new mechanism, no second staging path, no change to `vt::ops`. AGENTS.md
§"Shared seams" asks for exactly this: extend the seam or route through it,
never write a parallel path by hand.

### 4.1 Why the CPU arm is unchanged

`ResidentWeight`'s CPU arm builds the tensor over the same host bytes and copies
`repacked` and `elem_kn_repacked`, which is what `View()` does. The one marker
`View()` carries and `ResidentWeight` drops is `q8_0_aligned`, and that marker is
never set on this model's weights: `LoadMatmul`
(`src/vllm/model_executor/models/qwen4_exp_weights.cpp:130-141`) calls
`OwnGgufQuantBlocks(t, n, k, 0, MmapSrc(g, pol), pol.quant_repack)` and does not
pass `cuda_align`, which defaults `false`
(`include/vllm/model_executor/models/qwen3_5_gguf_weights.h:100`). The CPU forward
is therefore expected to be byte-identical, and §6's CPU case asserts it still
completes rather than merely compiles.

### 4.2 What upstream says

vLLM registers no `qwen4_exp` at the pin, so it defines nothing here; this is a
residency detail of THIS tree's device glue and has no upstream analogue in
either direction. The secondary oracle for the architecture, `llama-cpp-qwen4exp`
(`ggml-org/llama.cpp` PR #27742), declares all six hyper-connection projections
`GGML_OP_MUL_MAT` — down, up AND inject, on both sides
(`src/llama-arch.cpp:759-765`) — and consumes each with a plain `build_lora_mm`
on the file-typed tensor (`src/models/qwen4exp.cpp:237-241`). It draws no
distinction between inject and its two siblings, which is the reading this change
makes true here.

## 5. Risks

* **The fix exposes the next stop.** Getting past layer 0 is progress, not
  completion. The gate asserts only that the stop is not this operand and prints
  what the stop actually is.
* **`repacked` on a device-staged weight.** `ResidentWeight`'s staging arm
  guards `elem_kn_repacked` and does NOT guard `repacked`, so an aarch64 i8mm
  host that sets `GgufLoadPolicy::quant_repack` (`gguf_keep_quant.cpp:367`,
  which is device-blind) would upload i8mm-interleaved bytes and drop the
  marker. That hazard is PRE-EXISTING and already applies to `hc_*_down` and
  `hc_*_up`, which take this seam today; this change makes `inject` share it
  rather than creating it. It belongs to #2397, which owns the residency policy's
  device resolution. Recorded under `## Owed`.

## 6. Tests

`tests/vllm/models/test_qwen4_exp_inject_residency.cpp`, its own target
(`test_qwen4_exp_inject_residency`) rather than another case in
`test_qwen4_exp_layer_loop.cpp`, which #2422 is concurrently appending to.

| Case | Queue | What it asserts |
|---|---|---|
| `a CUDA ModelRegistry::Forward gets past the layer-0 hyper-connection inject weight` | CUDA | The forward does not stop with `block_inject_weight device mismatch`. Prints where it DID stop. Asserts the loader's own precondition (`CurrentPlatform().device_type() == kCUDA`) first, so a CPU load cannot report a pass. |
| `routing the inject weight through ResidentWeight leaves the CPU forward intact` | CPU | The forward completes with finite, non-constant logits. A regression guard: it cannot see the defect, because on a CPU queue `View()`'s `kCPU` stamp is the right answer. |

Both enter through `ModelRegistry::Forward`. Neither constructs the
hyper-connection by hand.

## 7. Gates

```sh
ninja -C build test_qwen4_exp_inject_residency test_qwen4_exp_layer_loop \
                test_qwen4_exp_runner test_qwen4_exp_forward
./build/tests/test_qwen4_exp_inject_residency
./build/tests/test_qwen4_exp_layer_loop
./build/tests/test_qwen4_exp_runner
./build/tests/test_qwen4_exp_forward
scripts/agent-preflight.sh --staged
```

The CUDA leg runs the same binary inside an `rc` lease on a fleet device, built
with `-DVLLM_CPP_CUDA=ON`.

## 8. Reachability

The production call site is the `dense_attn::ResidentWeight(d, lw.attn_hc.inject,
{hc, W})` line inside `ForwardQwen4ExpDecoderLayers`, reached from
`ModelRegistry::Forward` through `qwen4_exp_registry.cpp`'s forward hook. The
mutation is to restore `lw.attn_hc.inject.View()` there and rerun the focused
gate; a green gate would mean the CUDA case never reached the call site.

## 9. Stop conditions

Stop and report if the CUDA leg cannot be built or leased, if the fix does not
move the stop point, or if the forward's next stop is inside another wave's
files. Do not repair another wave's stop.

## 10. Evidence

Measured on `thor:gpu0` (`rc-worker-n8smh`, aarch64 Tegra), nvcc 13.0,
`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`,
inside an `rc` lease, job `134ab22f-e541-4409-9933-174efdcec393`, tree
`df8b5d3a0`. `CMAKE_RC=0`, `BUILD_RC=0`, 41 `.cu.o` objects, `libcudart.so.13`
and `libcublasLt.so.13` linked into the test binary, and the case's own
`REQUIRE(CurrentPlatform().device_type() == kCUDA)` passed, which is what
separates a CUDA run from a CPU load reporting a pass.

**Leg A, the branch as it lands.** `test_qwen4_exp_inject_residency` rc 0,
2 cases, 13 assertions, 13 passed. The CUDA forward gets past layer 0 and stops
elsewhere:

```text
CUDA ModelRegistry::Forward stopped with: vt cuda: matmul_bt:
  unsupported dtype combo (f32,bf16)->f32;
  supported: (bf16,bf16)->f32|bf16, (f32,f32)->f32|bf16
```

**Leg B, the reachability mutation.** Both production call sites restored to
`.View()` in place, applied-ness proved by a counted property rather than by a
patch exit code — `ResidentWeight(d, lw.*_hc.inject` 2 -> 0 and
`_hc.inject.View()` 0 -> 2 — and `BUILD_B_RC=0`, so this is not a build failure
reading as a pass. `test_qwen4_exp_inject_residency` rc 1, 12 of 13 assertions,
the one failure being

```text
CUDA ModelRegistry::Forward stopped with: vt: qwen4_exp_gated_residual:
  block_inject_weight device mismatch at src/vt/ops.cpp:2562
ERROR: CHECK_FALSE( IsInjectResidencyRefusal(stopped_with) ) is NOT correct!
```

Red to green through `ModelRegistry::Forward` on a CUDA queue, on one binary,
one box, one run.

**CPU parity, measured not argued.** The same mutation applied on an x86_64 CPU
build leaves all four suites green — `test_qwen4_exp_inject_residency` 8/8,
`test_qwen4_exp_layer_loop` 309/309, `test_qwen4_exp_runner` 136/136,
`test_qwen4_exp_forward` 429/429 — and the CPU forward's logit range reads
`[3290.84, 95090.7]` in both arms, as does the layer loop's
`max|diff| = 0.00982457` against the transformers 5.16.0 goldens. So the change
is invisible on the CPU arm, and the CPU tier cannot gate it. The CUDA leg is
the only real gate here, and §6 says so rather than leaving a green CPU run to
be read as one.

**`test_qwen4_exp_runner` fails on a CUDA build in BOTH legs**, at
`test_qwen4_exp_runner.cpp:464`, the `LoadedEngine` case. Leg B throws
`block_inject_weight device mismatch`; leg A throws the `matmul_bt` dtype
refusal. It is a pre-existing CUDA gap that this change moves forward rather
than a regression it introduces, and #2452 owns it.

## Owed

* `ResidentWeight`'s device-staging arm drops `repacked` without refusing it, so
  an aarch64 i8mm host loading for a CUDA queue can upload interleaved bytes that
  the device GEMM reads as plain. Pre-existing for `hc_*_down`/`hc_*_up`; owned
  by `ENG-GGUF-RESIDENCY-RESOLVED-DEVICE` (#2397), which is making the residency
  policy read the engine's resolved device.
* The next stop, `vt cuda: matmul_bt: unsupported dtype combo (f32,bf16)->f32`,
  is filed as [#2452](https://github.com/mudler/vllm.cpp/issues/2452) against
  this row. It was measured, not localized: the call site is unnamed, and
  `MatmulF32D` in the MoE seam is a reading rather than a measurement.

## Now

`MODEL-MM-QWEN4-EXP` is unchanged by this repair. No lifecycle state moves.
