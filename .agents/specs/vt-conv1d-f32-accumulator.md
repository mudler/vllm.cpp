# VT-CONV1D-F32-ACC — the vocoder convolutions accumulate wider than their own oracle

Row: `VT-CONV1D-F32-ACC`.
Issue: [#1474](https://github.com/mudler/vllm.cpp/issues/1474).
Precedent: [#1008](https://github.com/mudler/vllm.cpp/issues/1008) /
`LTX25-DECODE-DTYPE` ([`ltx25-decode-dtype.md`](ltx25-decode-dtype.md)), which
made the identical argument for LTX-2.5's conv VIDEO VAE and landed at
`d1b0ea3a8` via [PR #1036](https://github.com/mudler/vllm.cpp/pull/1036). This
row is the audio half of the same polarity defect, one op layer lower.
Filed as owed by [#1334](https://github.com/mudler/vllm.cpp/issues/1334) /
[`minimax-music3.md`](minimax-music3.md) §18.9, which narrowed the *chain* at
the f64 width and left the *width* explicitly untaken.

## Now

`ACTIVE`.

## 0. Scope

**In scope.** The output accumulator of `vt::Conv1d` and `vt::ConvTranspose1d`
moves from `double` to `float` on every provider, and every operand widened per
multiply narrows with it:

* `src/vt/cpu/cpu_conv1d_general.cpp::Conv1dKernel` and
  `::ConvTranspose1dKernel`,
* `src/vt/cuda/cuda_conv1d_general.cu::Conv1dKernelCudaImpl` and
  `::ConvTranspose1dKernelCudaImpl`, where `__dmul_rn`/`__dadd_rn` become
  `__fmul_rn`/`__fadd_rn` for the same reason they were f64 intrinsics rather
  than bare `+`/`*`: nvcc contracts, and the host is `-ffp-contract=off`,
* the three in-test serial references that pin the order bitwise —
  `SerialConv1d` and `SerialConvTranspose1d` in
  `tests/vt/test_ops_conv1d_general.cpp`, and `SerialConv1d` /
  `SerialConvTranspose1d` in `tests/vllm/models/test_host_parallel.cpp`.
  These narrow in LOCKSTEP or the bitwise gates go red for the wrong reason.

**Explicitly NOT in scope.** `vt::DepthwiseConv1d`
(`src/vt/cpu/cpu_conv1d_depthwise.cpp:80`) is untouched. It already accumulates
`float` and its byte-exactness gate pins that width; it is the sibling that was
already correct, and §13.2's sibling argument survives this row intact — it just
stops being a statement about two different widths.

Storage stays f32 everywhere. No tensor dtype moves. The `f16`/`bf16` refusal
stays a refusal.

## 1. What torch accumulates, measured rather than read

The reduction is engineered so the two widths are separable: 27 taps over a
uniform-`1.0` input with weights `[+1e8, 0.1 x 25, -1e8]`. Every partial sum
`1e8 + j*0.1` for `j <= 25` is below half an ulp of `1e8` (which is 8.0 in f32
and 1.49e-8 in f64), so an f32 accumulator holds exactly `1e8` until the final
`-1e8` and lands on exactly zero in ANY summation order, while an f64
accumulator lands near 2.5. torch 2.11.0+cu130, reproduced for this row:

| call | dtype | result |
|---|---|---|
| `F.conv1d` | f32 | **0.0** |
| `F.conv1d` | bf16 — the dtype an omni checkpoint actually carries | **0.0** |
| `F.conv1d` | f64 | 2.4999998510 |
| `F.conv_transpose1d` | f32 | **0.0** |
| `F.conv_transpose1d` | f64 | 2.4999998510 |

Both ops accumulate f32. The port accumulates f64.

## 2. Why torch is the reference here at all

vLLM owns no such op at the parity pin `555967922`. There is no `Conv1dLayer`
and no `ConvTranspose*Layer`; the only `ConvTranspose` string in the tree names
`torch.nn` classes inside an NVTX annotation hook
(`vllm/utils/nvtx_pytorch_hooks.py:47-49`), which annotates a module it does not
implement. vLLM deliberately drops the vocoder it would otherwise own —
`vllm/model_executor/models/qwen3_omni_moe_thinker.py:1975` loads with
`skip_prefixes=["talker.", "code2wav."]`.

And where vLLM DOES own a convolution it states this row's polarity itself:
`csrc/cpu/mamba_kernels.hpp` — `// Accumulate in float32 for precision`, with
`float acc` seeded from the bias and both operands `static_cast<float>`.

So ownership falls to torch through the per-consumer secondary oracles
(`transformers`, `diffusers`), which is what every one of the four consumers'
golden generators already runs.

## 3. The recorded provenance of the f64 is wrong, in two separate ways

This is why the row exists rather than being a preference.

**3a. The goldens were NOT taken at f64.** `.agents/specs/minimax-music3.md`
§13.2 and §18.3 both state that f64 "is what every committed golden for all four
consumers was taken with". The generators run torch in f32 end to end:

* `scripts/gen-bigvgan-goldens.py:48` — builds `dtype=torch.float64` and then
  calls `.float()`, so every parameter enters torch as f32.
* `scripts/gen-ltx2-vae-goldens.py:223,234` — `values.astype(np.float32)` for
  every parameter and every input.
* `scripts/gen-minimax-music3-acoustic-goldens.py:81,134` — the same.

The goldens are therefore the output of an **f32-accumulating** reference, and
the f64 arm has been wider than the oracle its own goldens came from since the
op landed. Narrowing moves the port TOWARD its goldens' generator, which is why
§6 expects `max|diff|` to hold or improve and treats a regression as a finding
rather than as a cost.

**3b. `include/vt/ops.h` cites a directory with no such golden in it.** Clause
(1) of the `vt::Conv1d` numeric contract says "every committed golden under
`tests/parity/goldens/` for all four consumers was taken through it". That
directory holds 101 entries and not one is a vocoder, BigVGAN, LTX-2.5 VAE, FVQ
or general-conv1d golden. The goldens for all four consumers are `.inc` headers
beside their tests (`tests/vllm/models/bigvgan_goldens.inc`,
`ltx2_vae_goldens.inc`, `minimax_music3_acoustic_goldens.inc`). The citation was
never checkable, which is how the claim in 3a survived.

Both records are corrected in this change, because this row is what makes them
load-bearing.

## 4. The sites

Anchors at `5870cb2bf`, this row's base.

| site | what it is | this row |
|---|---|---|
| `cpu_conv1d_general.cpp` `Conv1dKernel` `alignas(64) double acc[kConv1dPosTile]` | the tile of per-output-cell accumulators (#1334) | `float` |
| the same kernel's `double seed`, `double wv`, and the three `static_cast<double>(x)` accumulate paths | operands widened per multiply | `float` |
| `cpu_conv1d_general.cpp` `ConvTranspose1dKernel` `std::vector<double> acc(full)` | the scatter line, one per output channel, tens of MB at vocoder shapes | `float` |
| the same kernel's `double value`, the tap `static_cast<double>`, and the crop-and-bias tail | operands and the tail | `float` |
| `cuda_conv1d_general.cu` both kernels' `double acc` and `__dmul_rn`/`__dadd_rn` | the device arm | `float`, `__fmul_rn`/`__fadd_rn` |
| `tests/vt/test_ops_conv1d_general.cpp` `SerialConv1d`, `SerialConvTranspose1d` | the bitwise order references | `float` |
| `tests/vllm/models/test_host_parallel.cpp` `SerialConv1d`, `SerialConvTranspose1d` | the same, end to end | `float` |

Halving the scatter accumulator also halves a multi-tens-of-MB buffer and
doubles the vector width per operation, but **no speed number is claimed by this
row.** The box is contended for its whole duration and correctness is the gate.

## 5. Risks

* **A golden moves past its tolerance.** This is the risk that can change the
  design, and in #1008 it BOUND: narrowing the width while keeping the naive
  per-cell serial order pushed one arm from 2.09e-6 to 5.00679e-6 against a
  5e-6 tolerance — over by 0.14%, a genuine red. Tightest exposure here, in
  order: LTX-2.5's audio arms at `kLtx2GoldenTol = 5e-6` (the same constant),
  MiniMax-Music3 at rel `1e-5` with a `1e-6` absolute floor, BigVGAN at `2e-4`.
  **Mitigation, and it is not a tolerance.** #1008's remedy was upstream's own
  summation ORDER — per-input-channel blocked partial sums, which is what
  torch's blocked-GEMM convolution does, and which this repository already
  ships as a declared contract on a sibling op (`vt::Conv3d`: "one f32 partial
  per INPUT CHANNEL, bias seeded FIRST", `.agents/kernel-matrix.md`
  `KERNEL-CONV3D`). #1334's tile hoists the `(ic, k)` sweep outside the position
  loop but does NOT block the reduction across `ic`, so that remedy is
  half-built here and is the budgeted fallback. **No tolerance is widened and no
  golden is re-captured from our own output.** If an arm cannot be made green by
  the order, the row stops and reports (§8).
* **The bitwise gates lose standing, and this row says so rather than implying
  otherwise.** `SerialConv1d` / `SerialConvTranspose1d` are verbatim copies of
  the pre-op `vocoder1d` host loops at `8fa405bb7` and `d9441ef3`. Once the
  width moves they can no longer assert "identical to the pre-op host loop";
  they assert "the shipped kernel's order is the declared order, at every thread
  count". §18.9 already concedes this. If §5's fallback is taken they also stop
  being a transcription of any historical loop at all, and the file comments must
  say that in the same change.
* **A `float` accumulator silently re-promoted.** `acc += float * double`
  promotes the whole expression back to `double`, and the change then reads as
  done while being a no-op. Every operand at every site narrows, and the
  mutation table proves each one is observed.
* **An f64 accumulator stored through f32 cannot see a reassociation on benign
  data.** This is the measured reason the two `CATASTROPHIC CANCELLATION` cases
  exist in `test_host_parallel.cpp` (mutations M1 and M7 left the ordinary
  shapes entirely green). Any order-related mutation for this row is taken on
  those cases, never on the well-scaled ones.
* **The CUDA arm cannot be built here.** No CUDA toolkit and no lease. The
  device change is made so that the `memcmp` contract survives by the same
  construction it does today — one accumulator per output cell, the same order,
  non-contracted intrinsics at the narrower width — and it is reported
  **UNVERIFIED**, not claimed as parity. §7 carries it as owed.

## 6. Gates and evidence

1. **The torch probe reproduced independently**, with the torch version and
   `__file__` printed, before any code is written.
2. **`max|diff|` recorded per golden arm, before and after**, for every affected
   consumer, by temporarily setting each suite's tolerance to `0.0` so every
   `CHECK` reports its `INFO` (#1008's instrument). A golden that gets WORSE is
   reported as a finding, never absorbed.
3. **Every suite reports `test cases:`, `assertions:` AND `Status:`.**
   `assertions: 0` is a skip wearing a pass.
4. **Every mutation reports the compiler exit code, `git diff --stat`, and the
   sha256 of the BINARY.** A mutation that fails to build reads as a passing
   test, and a mutation that never applied reads as one too.
5. Suites re-gated: `test_ops_conv1d_general`, `test_host_parallel`,
   `test_minimax_music3_acoustic`, `test_ltx2_vae`, `test_minimax_h3`,
   `test_bigvgan`, `test_fvq`, `test_minimax_music3_loader`, `test_vocoder1d`.
6. **Reachability.** The op is reached from production through
   `vllm::vocoder1d::Conv1d` / `ConvTranspose1d`, which `test_host_parallel.cpp`
   enters end to end; the narrowing is observable there and not only in the
   `vt` unit test.

## 7. Owed

| owed | what would settle it |
|---|---|
| **The CUDA provider is UNBUILT and UNRUN at the new width.** No toolkit on this box, no lease taken. The `memcmp` argument is §5's construction argument, and an argument is not a measurement. | A build on a box with a CUDA toolkit, running `test_ops_conv1d_general`'s CPU-vs-CUDA cases. This is the SAME debt `minimax-music3.md` §18.9 already carries for #1334's tile, which has also never been compiled against the device arm. |
| **The speed magnitude. UNMEASURED, and no number is claimed.** | One `Conv1dKernel` wall at the vocoder's own shapes on an idle host, same binary, f64 arm against f32 arm. The box was contended throughout. |
| `ltx2_audio_vae.cpp`'s own 2-D host convolution loop still routes through no op ([#1114](https://github.com/mudler/vllm.cpp/issues/1114)) | unchanged by this row |
| [#1208](https://github.com/mudler/vllm.cpp/issues/1208) — the same f64 polarity on LTX-2.5's text tower `Linear` | its own row |

## 8. Stop conditions

Stop and report `NEEDS_DECISION` rather than widening scope if: a golden arm
cannot be brought inside its EXISTING tolerance by the width plus the summation
order; any golden gets measurably worse without an order that repairs it; or a
tolerance would have to move. Never widen a band and never re-capture a golden
from this port's own new output.

## 9. Outcome — what was measured

Release, x86_64, CPU-only, one build directory, `-ffp-contract=off`. The box was
contended throughout (load average above 200 from another session's build), which
is why this row claims no speed number.

### 9.1 The torch probe, reproduced

torch `2.11.0+cu130`,
`/home/mudler/.local/lib/python3.12/site-packages/torch/__init__.py`. 27 taps
`[+1e8, 0.1 x 25, -1e8]` over a uniform-`1.0` input:

| call | dtype | result |
|---|---|---|
| `F.conv1d` | f32 | **0.0** |
| `F.conv1d` | bf16 | **0.0** |
| `F.conv1d` | f64 | 2.4999998510 |
| `F.conv_transpose1d` | f32 | **0.0** |
| `F.conv_transpose1d` | f64 | 2.4999998510 |

The f64 arm of this port answers the same probe with **2.5**, printed by mutation
W1 as `CHECK( 2.5 == 0 )`. So the port and torch were separated by this probe by
construction, and the new gate asserts torch's value rather than ours.

### 9.2 The goldens, per arm, before and after

Measured by printing `max|diff|` from `vllm_test::MaxAbsDiff` and from
`test_minimax_music3_acoustic`'s own `ExpectClose` on every call, tagged with
`__builtin_FILE()`/`__builtin_LINE()`, on both arms of the same build recipe.
The instrumentation was reverted before the commit and is not in the tree.

**194 golden arms across the four consumers' suites. 182 did not move at all.**

| suite | arms | unchanged | moved toward the golden | moved away |
|---|---:|---:|---:|---:|
| `test_ltx2_vae` | 56 | 52 | 4 | 0 |
| `test_minimax_h3` | 108 | 104 | 3 | 1 |
| `test_minimax_music3_acoustic` | 29 | 25 | 3 | 1 |
| `test_bigvgan` | 1 | 1 | 0 | 0 |

Every arm that moved, with its bound:

| arm | bound | f64 (before) | f32 (after) |
|---|---|---|---|
| `test_ltx2_vae.cpp:725` snake-arm vocoder | 5e-06 | 2.23517e-08 | **1.49012e-08** |
| `test_ltx2_vae.cpp:753` legacy vocoder | 5e-06 | 2.23517e-08 | **1.49012e-08** |
| `test_ltx2_vae.cpp:812` BWE vocoder | 5e-06 | 1.49012e-07 | **1.04308e-07** |
| `test_ltx2_vae.cpp:1793` saturated-clamp BWE | 5e-06 | 4.07919e-07 | **2.23517e-07** |
| `test_minimax_h3.cpp:2098` audio VAE encoder | 1e-05 | 2.98023e-08 | **1.49012e-08** |
| `test_minimax_h3.cpp:2131` audio VAE encode-to-latent | 1e-05 | 1.86265e-08 | **1.11759e-08** |
| `test_minimax_h3.cpp:2248` noise-aug delegation, a NEGATIVE control (`> 1e-3`) | > 1e-03 | 0.468997002 | **0.468997031** |
| music3 `dit conditional` | rel 1e-05 / abs 1e-06 | 1.56462e-07 | **1.19209e-07** |
| music3 `cpu-backend dit conditional` | rel 1e-05 / abs 1e-06 | 1.56462e-07 | **1.19209e-07** |
| music3 `residual unit` | rel 1e-05 / abs 1e-06 | 5.96046e-08 | **0** (exact) |
| **`test_minimax_h3.cpp:1927` audio VAE waveform** | 1e-05 | 4.19095e-09 | **4.65661e-09** |
| **music3 `vocoder`** | rel 1e-05 / abs 1e-06 | 2.98023e-08 | **5.96046e-08** |

**The two arms that got worse are reported rather than absorbed, as §8 requires.**
Both move by one unit in the last place. The MiniMax-H3 waveform goes from
0.042 % of its bound to 0.047 % of it. The MiniMax-Music3 vocoder arm goes from
2.98 % of its absolute floor to 5.96 % of it — the largest single move in the
table and still a 16.8x margin. Neither is a design finding, no tolerance was
touched, and no golden was re-captured.

`test_minimax_h3.cpp:2248` is a **negative control** and its polarity is the
opposite of every other row: it asserts the mix really moved, `> 1e-3`. A larger
number there is more margin, not less. It is listed among the improvements for
that reason, and it is called out because a comparator that assumes "bigger is
worse" would have reported it as this row's largest regression.

### 9.3 `ic`-blocking was budgeted and NOT needed

§5 named per-input-channel blocked partial sums as the fallback, because in
#1008 the identical narrowing pushed an LTX-2.5 arm to 5.00679e-06 against a
5e-06 bound. **That did not recur, and the naive per-cell order was kept.** The
tightest LTX-2.5 audio arm sits at 2.24e-07 against the same 5e-06 constant — a
22x margin, and it IMPROVED. The reductions differ in depth by more than an
order: #1008's `CausalConv3d` sums `ci * kernel^3`, where the deepest arm here is
a Music3 residual unit at `384 * 7 = 2688` and the LTX-2.5 audio arms are
shallower still.

Not blocking is also what preserves §5's second point at its narrowest: the
per-cell visit order is untouched, so the three serial references lose only
their WIDTH claim and keep their order claim, and the CUDA provider's `memcmp`
argument survives by the same construction it had.

### 9.4 Mutations

Every one from a clean tree, sources `touch`ed before rebuilding, restored
byte-for-byte afterwards. `compile rc` and `git diff --stat` are reported
because a mutation that fails to build and a mutation that never applied both
read as passing tests.

| id | mutation | diff | compile rc / errors | detected by |
|---|---|---|---|---|
| W1 | `Conv1dKernel` accumulator widened back to `double` | 6 lines | 0 / 0 | width case (`2.5 == 0`), 63 assertions in `test_host_parallel`, 17 in `test_ops_conv1d_general` |
| W2 | `ConvTranspose1dKernel` accumulator widened back to `double` | 8 lines | 0 / 0 | 31 assertions in `test_host_parallel`, 9 in `test_ops_conv1d_general` |
| M1 | `Conv1dKernel` `ic` sweep reversed (order, not width) | 1 line | 0 / 0 | 25 + 16 assertions, on the CANCELLATION cases |
| M2 | `ConvTranspose1dKernel` `ic` sweep reversed | 1 line | 0 / 0 | 25 + 8 assertions |
| W3b | **kernel AND both serial references widened CONSISTENTLY** | 14 lines | 0 / 0 | **only the width case.** `test_ops_conv1d_general` stays 379/379 `SUCCESS!` |
| R | the `vt::Conv1d` / `vt::ConvTranspose1d` production call sites in `vocoder1d.cpp` emptied | 2 lines | 0 / 0 | `test_host_parallel` 55, `test_vocoder1d` 9, `test_bigvgan` 23, `test_minimax_music3_acoustic` 6, `test_ltx2_vae` 5 |

**W3b is the result that justifies the new gate existing.** It is the blind spot
a self-referential bitwise gate has by construction, and before this row the
width was ungated against it: with the kernel and both references widened
together, `test_ops_conv1d_general` reports 379 of 379 assertions passed and
`SUCCESS!`, and every bitwise and cancellation case in `test_host_parallel`
stays green. The only thing that fires is
`vocoder1d Conv1d accumulates in f32, which is what torch does`, which reads its
expectation off torch rather than off us.

A first attempt (W3) widened the accumulator and the store but not the products,
so the two sides no longer agreed on product rounding and the bitwise gates fired
for a reason that was not the width. It is recorded because it would have
supported the opposite conclusion.

### 9.5 Suites

Every suite, final tree, with `test cases:`, `assertions:` and `Status:`:

| suite | cases | assertions | Status |
|---|---|---|---|
| `test_ops_conv1d_general` | 10 / 10 passed | 379 / 379 | `SUCCESS!` |
| `test_host_parallel` | 10 / 10 passed | 920 / 920 | `SUCCESS!` |
| `test_minimax_music3_acoustic` | 32 / 32 passed | 283 / 283 | `SUCCESS!` |
| `test_ltx2_vae` | 44 / 44 passed | 3131 / 3131 | `SUCCESS!` |
| `test_minimax_h3` | 79 / 79 passed | 57395 / 57395 | `SUCCESS!` |
| `test_bigvgan` | 6 / 6 passed | 65 / 65 | `SUCCESS!` |
| `test_fvq` | 3 / 3 passed | 17 / 17 | `SUCCESS!` |
| `test_minimax_music3_loader` | 21 / 21 passed | 1393 / 1393 | `SUCCESS!` |
| `test_vocoder1d` | 10 / 10 passed | 58 / 58 | `SUCCESS!` |

`test_host_parallel` gains 2 cases and 43 assertions, which are the two width
cases. Nothing else moved. `scripts/agent-preflight.sh --staged` is green except
`test_cpu_x86_llamacpp_floor`, which is
[#618](https://github.com/mudler/vllm.cpp/issues/618) under box load and not this
row's.

### 9.6 What is NOT verified, stated as unverified

**The CUDA provider was neither compiled nor run.** This box has no CUDA
toolkit, and no `rc` lease was taken because the row's gates are correctness
gates that run on the host. `__dmul_rn`/`__dadd_rn` became `__fmul_rn`/`__fadd_rn`
and the accumulators became `float`, so the `memcmp` contract survives by the
same construction it had — one accumulator per output cell, the same order,
non-contracted intrinsics, now at the same narrower width on both arms. That is
an argument, not a measurement, and §7 carries it as owed. It is named in the
file header, in the `vt::Conv1d` contract clause (4) and in `docs/USAGE.md` so a
reader meets it wherever they arrive.

**No speed number is claimed**, although the change halves a multi-tens-of-MB
scatter accumulator and doubles the vector width per operation. The box was
contended for the row's whole duration.

**`kConv1dPosTile` was not re-swept.** 32 measured best of {8, 16, 32, 64} while
these accumulators were f64; halving their width doubles how many fit a vector
register and may move the optimum. Recorded beside the constant rather than
assumed away.
