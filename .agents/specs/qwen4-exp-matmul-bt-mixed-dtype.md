# `qwen4_exp`: the `(f32, bf16)` `matmul_bt` a CUDA forward stops on

Row: `MODEL-MM-QWEN4-EXP`
Issue: [#2452](https://github.com/mudler/vllm.cpp/issues/2452)
Campaign: [#1978](https://github.com/mudler/vllm.cpp/issues/1978), spec
[`qwen4-exp-flash-next.md`](qwen4-exp-flash-next.md)

## Scope

One question and one fix. `ModelRegistry::Forward` on a CUDA queue over the tiny
`qwen4exp` GGUF fixture stops with

```text
vt cuda: matmul_bt: unsupported dtype combo (f32,bf16)->f32;
    supported: (bf16,bf16)->f32|bf16, (f32,f32)->f32|bf16
```

thrown from `src/vt/cuda/cuda_matmul.cu:425`. `ComboName(a, b, out)` names the
ACTIVATION first, so the operands are an **f32 activation** against a **bf16
weight**, producing f32. #2452 measured that stop and deliberately did not
localize it.

In scope: naming the call site, deciding which side is wrong, and the smallest
change that makes a CUDA forward pass this point. Out of scope: `qwen4_exp`'s
remaining walls, which are other waves' files —
`qwen4_exp_forward.cpp` (#2453), `qwen4_exp_qsa_block.cpp` (#2422),
`cuda_quant_dot.cu` (#2423), `gguf_keep_quant.cpp` / `model_loader.cpp` (#2397).

## The question this spec exists to answer

`AGENTS.md` "vLLM is the reference" states the polarity: vLLM resolves one model
dtype and every layer inherits it, and an `f32` value is a rare annotated
exception. A mixed `(f32, bf16)` GEMM therefore reports that one side did not
inherit. There are three shapes it could be, and they take three different fixes:

1. **An activation was widened to f32 that should have stayed bf16.** The fix is
   at the producer and the CUDA combo stays unsupported. A token gate cannot see
   this: the tokens still match while the path moves twice the bytes.
2. **The reference genuinely runs a mixed-precision GEMM here.** Then mirror it,
   cite the `file:line`, and match the accumulate and output dtypes.
3. **The CPU arm is merely laxer.** `vt::MatmulBT` (`src/vt/ops.cpp`) checks only
   `IsFloat(a) && IsFloat(b)`, and `MatmulBTKernel` on CPU reads both operands
   through a dtype-generic getter, so the mix runs there and answers. That makes
   CPU a value oracle and never a dtype-policy oracle.

## The oracle

vLLM registers no `qwen4_exp` at its pin, so the algorithm oracle is the row's
accepted lane pin, `transformers` **5.16.0**
(`.agents/oracles/transformers.md`), at
`src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`. Read at that tag:

* `Qwen4ExpTextRMSNorm.forward` (`:173-178`) computes in `float()` and returns
  `output.type_as(x)`.
* `Qwen4ExpTextRMSNormGated.forward` (`:192-201`) records `input_dtype`, upcasts,
  and returns `hidden_states.to(input_dtype)`.
* `Qwen4ExpTextTopKRouter.forward` (`:907-916`) runs `F.linear` at model dtype
  and upcasts only the SOFTMAX (`dtype=torch.float`), then casts the top-k values
  back with `router_top_value.to(router_logits.dtype)`.
* `Qwen4ExpTextGatedResidual.forward` (`:955-969`) applies `hc_norm` and feeds
  `hyper_input_normed` — a `type_as(x)` result — to all three `nn.Linear`s.

Every f32 in the reference is an interior accumulation that is cast back before
the next parameterized layer. **No `nn.Linear` in this architecture ever receives
an f32 activation while its weight is model dtype.** Shape 2 is therefore ruled
out by the oracle rather than by preference.

## Method

The refusal is CUDA-only, but the composition that produces the operand dtypes is
not: `vt::MatmulBT` is device-agnostic and the CPU arm runs the same call. So the
call site is localized on the CPU tier, where a full `LoadedEngine` forward over
the same fixture already completes
(`tests/vllm/models/test_qwen4_exp_runner.cpp`, case 5), by an uncommitted
instrumentation in `vt::MatmulBT` that prints a backtrace for every
`a.dtype != b.dtype` dispatch. That names the site without a lease. The FIX is
then gated on a GPU, because the CPU tier cannot gate this refusal at all.

## The call site, measured

`Qwen4ExpGatedResidualKernelCuda`, `src/vt/cuda/cuda_qwen4_exp.cu:414` (then
`:418` and `:428`). The CUDA arm of `vt::Qwen4ExpGatedResidual` is four kernels
and THREE `vt::MatmulBT` calls where the CPU arm is one fused loop, because at
the released config `hc*H` is thousands of floats and a device block cannot hold
the intermediates. It keeps the grouped-RMSNorm result in an f32 device scratch
(`ScratchF32`, `:361`) and projects that scratch through `mix_down`, `mix_up` and
`block_inject`, which on this fixture are bf16.

**Localized on the CPU tier, then confirmed on the device.** An uncommitted
`vt::MatmulBT` trace over the CPU `LoadedEngine` forward of the same fixture
(`test_qwen4_exp_runner`, case 5) printed 260 dispatches and **not one mixed
pair** — every CPU GEMM is `(bf16,bf16)`. That negative is the finding: the mix
is a device-only branch, and the only device-only decomposition of an op into
`vt::MatmulBT` calls on this forward is this one. The same trace on `thor:gpu0`
then named it directly: dispatch **#0** of the whole CUDA forward is
`(f32,bf16)->f32 m=4 k=128 n=8` — `t_normed [T, hc*H=128]` against
`mix_down [R=8, 128]` — with `vt::MatmulBT` <- `Qwen4ExpTextModelForward` <-
`ModelRegistry::Forward` in its backtrace.

That refines #2452, which placed the stop "after the layer-0 MLP hyper-
connection". It is the layer-0 **attention** hyper-connection, and it is the
FIRST GEMM the forward issues — before PLE, before the GDN input projection.

## Which of the three shapes this is

**Shape 3: the CPU arm is not laxer by accident; the CUDA elementwise lane is
narrower than every sibling.** `vt::MatmulBT`'s contract is
`IsFloat(a) && IsFloat(b)` (`src/vt/ops.cpp`), and three of the four
implementations behind it already answer an f32 activation against a typed
weight — the CPU elementwise kernel (`cpu_ops.cpp` `MatmulChunked<true>`, a
dtype-generic getter per operand), the CPU block-quant kernel, and **this
device's own `kMatmulBTQuant`**, whose validation admits any
`IsFloat(a.dtype)` and which is the arm the RELEASED `unsloth/…-GGUF`
checkpoint's Q8_0 mix weights take at these three call sites. Only the cuBLASLt
elementwise lane refused. The tree had already paid for that: `RouterGateKernel`
(`src/vt/cuda/cuda_deepseek_v4.cu:1877-1882`) is a hand-written GEMV whose own
comment says it exists because "the CUDA elementwise MatmulBT lacks" this pair —
a private path beside the seam, which is the failure `AGENTS.md`
"Shared seams" names.

**Shape 2 is ruled out by the oracle.** `transformers` 5.16.0 does NOT run a
mixed-precision GEMM here. `Qwen4ExpTextRMSNorm.forward`
(`modeling_qwen4_exp.py:173-178`) computes in `float()` and returns
`output.type_as(x)`; `Qwen4ExpTextGatedResidual.forward` (`:955-969`) then feeds
that narrowed tensor to all three `nn.Linear`s. Every `f32` in that file is an
interior accumulation cast back before the next parameterized layer, and
`Qwen4ExpTextRMSNormGated` (`:192-201`) does the same through `input_dtype`. So
the oracle cannot be cited in favour of the mixed pair, and it is not cited.

**Shape 1 is ruled out because the widening is ratified and GATED, not
accidental.** Upstream narrows and this tree deliberately does not:
`.agents/specs/qwen4-exp-flash-next.md` `## Owed` records it as "One deliberate
divergence from upstream, in the bf16 arm … This op does not: it widens on load,
computes in f32 and rounds once on the store, which is this tree's house contract
and what `vt::RmsNorm` says of itself in the same terms." Reproducing upstream's
rounding is that row's mutation **M13**, RED in the host suite and the device
suite alike (1 of 9 cases, 8 assertions). Narrowing `t_normed` here to satisfy
cuBLASLt would BE M13. What upstream's `.type_as(x)` costs is already owed there
— "that term stated in whatever first compares a bf16 arm to the oracle" — and
that debt belongs to the mixer's dtype policy, which spans both arms and the
committed goldens. It is not this GEMM's, and this row does not reopen it.

## Design

`MatmulBTKernelCuda` grows one arm: `a.dtype == kF32 && b.dtype == kBF16`.

cuBLASLt takes ONE data type for A and B, so a mixed pair has to be made
uniform, and there are exactly two ways. Rounding the ACTIVATION to bf16 is M13,
so the WEIGHT is upcast instead: `bf16 -> f32` is exact (`bits << 16`, the same
widening `RouterGateKernel` and the CPU `LoadF32` perform), so the existing f32
lane then computes the CPU arm's values with only the GEMM's own
K-reassociation between them — the divergence this file already owns for every
other lane.

The cost is stated in the code rather than discovered later: one `[N,K]` f32
scratch per call, so the weight is touched at 10 bytes/element instead of 2.
Today's only caller projects the hyper-connection mix weights, which are small
(`k=128 n=8`, `k=8 n=128`, `k=128 n=2` on the fixture); a large-weight caller
wants a native mixed kernel, and that is a performance row, not a correctness
one. Recorded under `## Owed`.

`cudaMallocAsync`/`cudaFreeAsync` rather than `cudaMalloc`/`cudaFree`: the latter
pair is ILLEGAL during CUDA-graph capture and implicitly synchronizes the device,
and this is a shared GEMM seam that capture paths reach.

## Risks

* **Adding the CUDA combo would close the symptom and keep the defect.** If the
  activation was wrongly widened, the combo makes the forward run while the path
  keeps moving twice the bytes, and no token gate would ever see it.
* **A CPU-green gate proves nothing here.** CI has no GPU, and a sibling wave
  measured a mutation that left all four CPU suites green with identical logits.
* **A mutation kill against a red baseline, or one whose build failed, is void.**
  Applied-ness is proven by a counted property, not by a patch exit code.

## Tests

Red first, through the production entry point (`ModelRegistry::Forward` /
`LoadedEngine`) on a CUDA queue. A unit test that constructs a GEMM by hand
proves the kernel works and never that anything reaches it. Reachability is
proven by deleting the production call site in a scratch copy and showing the
focused gate go red.

## Gates

* The focused `qwen4_exp` device case, on a leased GPU, red before and green
  after.
* CPU-vs-CUDA parity on the localized GEMM's output, reported as `max|diff|`.
* `scripts/agent-preflight.sh --staged`.

## Owed

* **A native mixed kernel for a large weight.** The arm upcasts the `[N,K]`
  weight into an f32 scratch per call, which is correct and costs 5x the weight
  traffic of an ideal `(f32 x bf16)` GEMM. Nothing on a hot path takes it today
  — the only production caller is the hyper-connection mixer's three small
  projections, and the released Q8_0 checkpoint routes to `kMatmulBTQuant`
  instead — so this is a performance item, not a correctness one. It becomes a
  question the moment a wide projection reaches this pair.
* **`RouterGateKernel`'s private GEMV** (`cuda_deepseek_v4.cu:1877-1882`) exists
  only because this arm did not. It can now route through the seam. Out of scope
  here: it is a DeepSeek-V4 decode path with its own bit-identity claim against
  the CPU `MatmulBT`, and re-pointing it needs that claim re-measured.
* **`vt::Matmul` (the row-major NN lane) still refuses the pair.** This row
  closes `matmul_bt` only, which is what the forward reaches; the NN lane's
  callers pass `nk == false` weights and none of them hands it an f32
  activation today. Named rather than silently left.
* **The `.type_as(x)` term** upstream applies and this tree does not is already
  owed by `.agents/specs/qwen4-exp-flash-next.md` `## Owed`; this row does not
  move or discharge it.

## Evidence

Two `rc` leases on `thor:gpu0` (`sm_110`, aarch64 Tegra), nvcc **13.0.88**,
`-DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF
-DVLLM_CPP_BUILD_TESTS=ON`. Every rc below was read from the job's own output,
never derived. Jobs `ef3f2da0-3148-4365-9c2c-b5f08e5f3033` (localization, tree
`31bd51e9b`) and `ce266e08-d053-44d7-9ea6-b10b1dd9bb9d` (gate, tree
`aa3ce2f30`).

**Both trees are this branch MERGED with `row/2449-inject-resident`.** The
mixer's GEMM sits inside the op whose operand #2453 stages, so it is
unreachable without it, and a measurement on this branch alone would have been
of an earlier stop. The suite says so itself rather than passing quietly: the
reach case detects that stop by name and reports `UNMEASURED`.

### Job 1 — the localization, and the A/B on the stop point

| Leg | `test_qwen4_exp_inject_residency` reports the forward stopped with |
|---|---|
| A1, arm absent | `vt cuda: matmul_bt: unsupported dtype combo (f32,bf16)->f32` |
| A3, arm absent, traced | `[MIXED_BT] #0 (f32,bf16)->f32 m=4 k=128 n=8`, backtrace `vt::MatmulBT` <- `Qwen4ExpTextModelForward` <- `ModelRegistry::Forward` |
| B1, arm present (`PATCH_RC=0`, `BUILD_B_RC=0`) | `qwen4_exp qsa block: … both must be CPU-resident … qwen4_exp_qsa_block.cpp:147` |
| C1, arm removed again (`MUT_PATCH_RC=0`, `BUILD_C_RC=0`) | back to `matmul_bt: unsupported dtype combo (f32,bf16)->f32` |

The B1 trace shows the three mixed projections (`k=128 n=8`, `k=8 n=128`,
`k=128 n=2`) twice per layer for layers 0-2, beside the GDN, PLE and MoE GEMMs,
so the arm is exercised 18 times per step on this fixture rather than once.

### Job 2 — the committed gate, red first, with two mutations

Target `test_qwen4_exp_matmul_bt_dtype`. Applied-ness is proven by a COUNTED
property, not by a patch exit code, and each leg's binary mtime is printed so a
build that did not happen cannot read as a result.

| Leg | counted property | build rc | run rc |
|---|---|---|---|
| P0, arm removed | `f32_act_bf16_w` occurrences **0** | 0, mtime 03:12:22 | **1 (RED)** |
| P1, arm restored | occurrences **3** | 0, mtime 03:12:45 | **0 (GREEN)** |
| M-B, upcast zeroed | `MUTANT` 1, exact-upcast lines **0** | 0, mtime 03:13:10 | **1 (RED)** |
| Restore | bytes `IDENTICAL` to P1 | 0, mtime 03:13:34 | **0 (GREEN)** |

P0 is both the red baseline and the reachability mutation: removing the arm is
removing the production call site of this change, and it reds the forward case
(`CHECK_FALSE(IsMixedComboRefusal(...))`) and the value case
(`REQUIRE_NOTHROW` throws naming the combo) together, 8 assertions with 2 failed.

**CPU-vs-CUDA parity, and the margin.** The value case's oracle is a host
`double` dot over the same bytes, not the CPU arm, so it measures correctness
rather than agreement. Measured at P1:

| m, k, n | max abs diff | relative | bound | margin |
|---|---|---|---|---|
| 4, 64, 8 | 4.3994e-07 | 7.50e-08 | 7.63e-06 | 102x |
| 6, 1024, 16 | 2.63019e-06 | 1.01e-07 | 3.05e-05 | 303x |
| 6, 16, 1024 | 5.67081e-07 | 1.14e-07 | 3.81e-06 | 33x |
| 1, 4096, 4 | 3.92919e-06 | 3.66e-07 | 6.10e-05 | 167x |

The bound is `8 * sqrt(K) * eps_f32`, DERIVED from the reduction-order term
rather than fitted: the measured relative error grows more slowly than `sqrt(K)`
across those four rows, so the bound does not point the wrong way as K scales.
Under M-B the relative error is exactly **1.0** on all four, four to six orders
over. The CPU arm reads 6.21905e-07 against the same oracle at K=32.

### What is NOT proven

* **No token came out on a GPU.** The forward still stops at layer 3
  (`qwen4_exp_qsa_block.cpp:147`, #2422), so there are no device logits and no
  logit comparison. The gate asserts the stop MOVED and prints where.
* **The value case cannot be killed by the reach case and vice versa.** M-B
  leaves the reach case green, because that case is one-sided by construction.
  Both are needed and neither is claimed to cover the other.
* **`check-cuda-fat-gencode.py` did not run.** It needs a built library plus
  `cuobjdump` over a FAT build, and both jobs built one architecture on purpose.
  CI's `cuda-fat-build` is the reader for it.
* **CI cannot gate any of this.** CI has no GPU; the CPU case in this suite is a
  regression guard that says so in its own output.

## Stop conditions

* The forward stops at a NEW wall that belongs to another wave's files. Report
  the new stop point precisely and do not chase it.
* No fleet device is leasable. Then the fix is unproven on a device and says so.

## Now

`DONE` pending review — fix committed and gated on `thor:gpu0`, red first, with
a reachability mutation and a value mutation both killed. The forward now stops
at the QSA rope host-residency check (#2422), which is another wave's file. This
change must land AFTER #2453; before it does, the reach case reports UNMEASURED
on a device rather than passing.

No public document changes: `src/` and `tests/` on their own owe none, and this
adds no command, C API, config key, feature, model, backend or quantization
surface.
