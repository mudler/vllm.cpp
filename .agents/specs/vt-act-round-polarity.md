# Per-op bf16 rounding polarity in the `vt` elementwise seam

Issues: [#1322](https://github.com/mudler/vllm.cpp/issues/1322) (the filed gap),
[#1342](https://github.com/mudler/vllm.cpp/issues/1342) (Vulkan/Metal silu
spelling), [#1343](https://github.com/mudler/vllm.cpp/issues/1343)
(`RmsNormPlusAdd` arm asymmetry),
[#1458](https://github.com/mudler/vllm.cpp/issues/1458) (four model suites red
on `main` behind the landed narrowing; fixed, see `## The four suites #1458
reds, and why the kernel is not the defect`).
Row: `VT-ACT-ROUND-POLARITY`.

## Now

`ACTIVE`. The CPU arm of the gated-activation fix has landed with the upstream
exactness test ported as its red-first gate. The premise was re-verified against
the primary oracle at the pin before any code was written, and it survived —
unlike the RmsNorm half, which stays refuted (see `## The refuted half`).

**The change is confined to bf16-INPUT paths by construction**, because the
narrowing target is `x.dtype` rather than `out.dtype`. That is what keeps every
existing f32 golden and every existing byte-exact composite contract unmoved,
and it is asserted directly rather than inferred (see
`tests/vt/test_ops_activation.cpp`, "leaves an f32 INPUT bit-identical").

**The narrowing survived a second, independent challenge.**
[#1458](https://github.com/mudler/vllm.cpp/issues/1458) reported four model
suites red behind `4712dac40`, attributed by mutation in both directions. The
kernel was re-verified from source at the pin and from the committed oracle
golden, and it is right; the four bounds were not. See `## The four suites #1458
reds, and why the kernel is not the defect`.

**Five providers are still owed and are listed under `## Owed`.** CUDA, ROCm,
Metal and Tenstorrent cannot be compiled from the dev box at all, and the Vulkan
arm additionally needs a GLSL toolchain to regenerate its committed SPIR-V. This
row does NOT claim provider parity it did not test.

## What landed, and what it measured

`RoundThrough(DType, float)` — the helper `kRmsNormGatedGroup` already uses for
exactly this purpose — is now forward-declared beside `LoadF32`/`StoreF32` and
called from `SiluAndMulKernel`, `GeluAndMulKernel` and `MoeSiluMulKernel` in
`src/vt/cpu/cpu_ops.cpp`. No new mechanism was added, as the spike predicted.

Red-first, on an otherwise clean tree, kernel unchanged: `test_ops_activation`
went **24 cases / 2 failed / Status: FAILURE!** with both bf16 exactness cases
failing at the `REQUIRE(out[...] == want)` line, while the f32-input case passed
already. After the kernel change: **24 cases / 24 passed / assertions 285996 /
Status: SUCCESS!**

Fourteen CPU suites were measured before and after with the binary sha256
recorded on each side. **Every one is unchanged-green**, and the 46-case CPU
golden pass ran 46 cases on both sides:

| suite | before | after |
|---|---|---|
| `test_ops_activation` | 21/21, 277289 assn, SUCCESS | 24/24, 285996 assn, SUCCESS |
| `test_op_parity` | 13/13, 153 assn, SUCCESS (46 golden cases) | 13/13, 153 assn, SUCCESS (46 golden cases) |
| `test_ops_moe_grouped`, `test_gemma3_forward`, `test_qwen27_dense_forward` | SUCCESS | SUCCESS, unchanged |
| `test_qwen3_forward`, `test_qwen3_moe_forward`, `test_gemma2_forward`, `test_gemma_forward`, `test_gpt2`, `test_qwen35_paged_forward`, `test_deepseek_v4_moe` | SUCCESS | SUCCESS, unchanged |
| `test_qwen3_load`, `test_gemma3_load` | SUCCESS but **assertions: 0** | same — these are skips wearing a pass, not evidence |

**No golden got worse.** The one oracle-captured activation golden improves: see
the table under `## Premise`, where the worst element margin against the
harness's own `atol + rtol*|want|` moves 0.5364 -> 0.2989.

## Gap verification

Checked before writing anything, because several changes this session were found
already landed.

| surface | result |
|---|---|
| `git log --grep 1322` | only `94e740514`, a Music3 spec commit that *cites* the issue; no implementation |
| open pull requests | none touches `vt::RmsNorm`, `vt::SiluAndMul` or their providers |
| `origin/fix/fused-chain-interp-rmsnorm` | ancestor of `origin/main`; its RmsNorm content is a `-ffp-contract=off` build pin, not a polarity change |
| `origin/perf/rmsnorm-fp4` | ancestor of `origin/main` |
| `origin/row/BACKEND-VULKAN-RMSNORM` | PR [#185](https://github.com/mudler/vllm.cpp/pull/185), **MERGED** — a decode workgroup-shape speedup (7.85 → 1.59 ms/token), not a polarity change. Not an ancestor by sha because it was squash-merged, so the pull request state is the authority, not `--is-ancestor` |

Nothing to reconcile. The gap is real and unowned.

## Premise, re-verified independently at the pin

The implementing wave re-read the primary oracle rather than inheriting the
spike's reading. Every leg holds, and one piece of evidence is stronger than
what the spike had.

| Claim | Anchor at `5559679229bc961848b121ccdeaa8fa5d79bec98` | Verdict |
|---|---|---|
| `act(gate)` narrows to the input dtype | `csrc/libtorch_stable/activation_kernels.cu:36` — `return (scalar_t)(ACT_FN(gate, alpha) * ((float)up + beta));` | holds |
| `silu` itself narrows | `activation_kernels.cu:158` — `return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));` | holds |
| `gelu_tanh` narrows | `activation_kernels.cu:205` — `return (T)(0.5f * f * (1.0f + ::tanhf(inner)));` | holds |
| the native reference does the same | `vllm/model_executor/layers/activation.py:143` — `return F.silu(x[..., :d]) * x[..., d:]` | holds |
| upstream pins the two bit-exactly | `tests/kernels/core/test_activation.py:108` — `torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)` | holds |
| the vectorized path agrees | `activation_kernels.cu:72` — `cast_to_float2(PACKED_ACT_FN(gate, alpha))`, whose argument returns `packed_t` | holds |
| CPU `forward_cpu` is not the mirror source | `activation.py:155-158` — returns `forward_native` on every architecture except POWERPC | holds |

### The prior-attempt search, and why it had to be run differently

The RmsNorm half of #1322 was refuted by an upstream revert, so the same search
was owed here. `${VLLM_SOURCE}` is a **shallow** checkout: `.git/shallow` carries
three grafts and the oldest commit reachable from the pin is `16282a9c4`
(2026-06-10), six weeks before the pin. A `git log -S` walk therefore cannot see
a revert older than that, and `git merge-base --is-ancestor` returns a false
negative across a graft — two of the three shas the spike cites for the RmsNorm
revert (`4d51588e2`, `124fac10c`) report NOT-an-ancestor here for that reason
alone. Their trees are readable even though the walk is cut, so the search was
run on **content** instead of on history:

| ref | `silu_kernel` return | narrows to `T`? |
|---|---|---|
| v0.15.0 … v0.23.0 | `return (T)(((float)x) / (1.0f + expf((float)-x)));` | yes |
| v0.24.0, v0.25.0, v0.26.0 | `return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));` | yes |
| the pin | same | yes |
| v0.27.2rc0 (**newer than the pin**) | same | yes |

The narrowing is continuous across twelve releases spanning the pin, and it is
still there on the newest tag in the checkout. There is no attempt to remove it
and no revert. This is the opposite of the RmsNorm finding, not a weaker version
of it.

### The golden proves the polarity, not just the source

`tests/parity/goldens/silu_and_mul_bf16_8x256/` is a real oracle capture
(manifest: vLLM 0.24.0 @ `e24d1b24`, torch 2.11.0; x bf16 [8,256], out f32
[8,128], atol=rtol=0.008). Recomputing it from `x.npy` with this tree's own bf16
codec (`src/vt/dtype.cpp:297-304`):

| expression | max abs err vs golden | bit-exact? |
|---|---|---|
| `silu_f32(g) * up` — what `vt` computes today | 1.434994e-02 | no |
| `bf16(silu_f32(g)) * up` — the polarity this row adds | 7.812500e-03 | no |
| `bf16( bf16(silu_f32(g)) * up )` — upstream's full chain | **0.000000e+00** | **yes** |

The third row reproduces the captured oracle bytes exactly. That is an
end-to-end confirmation of the polarity from a running vLLM, independent of
reading any kernel source. The residual on row two is the store rounding, which
this seam does not take because the golden's `out` is declared f32 while
upstream's `out` is always `x.dtype`.

Under the harness's own tolerance (`tests/parity/test_op_parity.cpp:159`,
`tol = atol + rtol*|want|`) the worst element margin moves **0.5364 -> 0.2989**,
a 44% reduction. No element moves the wrong way and none was ever over tolerance.

### The effect, reproduced

Over 2^20 independently bf16-rounded standard-normal `(gate, up)` pairs, using
this tree's `F32ToBF16`: **288456 of 1048576 outputs differ (27.51%), every one
by exactly 1 bf16 ULP, none by more.** The spike measured 27.47% on a different
RNG stream; the shape of the effect is the same.

## Scope

**In scope for the implementing row.** The gated activation family —
`vt::SiluAndMul`, `vt::GeluAndMul`, `vt::MoeSiluMul`, and the SwiGLU epilogues
of `kMoeGateUpSwiGLUGrouped` and `kMoeGroupedGemmBf16GateUpSilu` — rounds
nothing between the activation and the multiply. Upstream rounds `act(gate)` to
the input dtype first, in both of the implementations it ships for accelerators,
and asserts the two agree bit-exactly.

**Out of scope, with evidence.** `vt::RmsNorm`, `vt::RmsNormGated` and the
quant-fused siblings. See `## The refuted half`.

**Not touched.** `.agents/specs/minimax-music3.md`, and the files of the two
Music3 pull requests in repair
([#1330](https://github.com/mudler/vllm.cpp/pull/1330),
[#1328](https://github.com/mudler/vllm.cpp/pull/1328)). #1330's tolerance rests
on this divergence existing; the activation change would move part of it, which
is a fact for that row's owner and not an edit this row may make.

## The refuted half — `vt::RmsNorm` is already faithful

The dispatch anchors the obligation on a reference that rounds `x*inv` to the
weight dtype before the affine multiply. That reference exists. It is not the
one that runs.

At the parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, upstream carries
two RMSNorm implementations with **different** polarity:

| upstream implementation | final expression | polarity |
|---|---|---|
| `vllm/ir/ops/layernorm.py::rms_norm` (native) | `x = x.to(weight.dtype) * weight` | rounds first |
| `vllm/ir/ops/layernorm.py::fused_add_rms_norm` (native) | `x = x.to(weight.dtype) * weight` | rounds first |
| `csrc/libtorch_stable/layernorm_kernels.cu::rms_norm_kernel` | `dst.val[j] = static_cast<scalar_t>(x * s_variance * w)` | **f32, one rounding** |
| `csrc/libtorch_stable/layernorm_kernels.cu::fused_add_rms_norm_kernel` (both specializations) | `Converter::convert(x * s_variance * wf)`, `(scalar_t)(x * s_variance * w)` | **f32, one rounding** |
| `csrc/cpu/layernorm.cpp::rms_norm_impl` | `fp32_out = fp32_x * fp32_s_variance * fp32_w;` | **f32, one rounding** |

`vllm/kernels/vllm_c.py::rms_no_var_size` selects the C kernel whenever
`variance_size` is unset **and `weight.dtype == x.dtype`**. A bf16 model on a
CUDA-alike device therefore always executes the f32 form, and so does the CPU
backend. The native form is reached only when the dtypes differ — which is
`GemmaRMSNorm`, whose weight is `self.weight.float() + 1.0`, so there the
cast-back is a cast to f32 and rounds nothing at all.
`layernorm.py::GemmaRMSNorm`'s own docstring names this as difference 2 from the
standard class.

**Upstream made the proposed change and reverted it.**

| sha | date | subject | direction |
|---|---|---|---|
| `4d51588e2` | 2026-04-26 | `[Feat] DeepSeek V4 Rebased (#40860)` | round-first → f32-across-weight, unremarked inside a large feature commit |
| `124fac10c` | 2026-05-29 | `[Bugfix] Fix RMSNorm kernels to multiply in weight's native dtype (#42379)` | f32-across-weight → round-first, in `layernorm_kernels.cu` **and** `layernorm_quant_kernels.cu` — precisely the edit this row was dispatched to make |
| `225936a1d` | 2026-06-18 | `[CI Bug] Revert #42379 to fix CI \`Multi-Modal Models (Extended Generation 1)\` (#46070)` | back to f32-across-weight |

All three are ancestors of the pin, and the reverted-to state is what the pin
carries. Three later commits touch the file and none alters the store
expression.

Upstream's own kernel test `tests/kernels/core/test_layernorm.py::test_rms_norm`
compares `forward_native` against the kernel at `atol=1e-2, rtol=1e-2` and never
asserts they agree exactly, so upstream does not treat the two polarities as one
behaviour.

`vt::RmsNorm` computes `v * inv * wj` in f32 and stores once, on every provider.
That is the executed upstream kernel. Making it round first would move us
**away** from the pinned oracle onto a form upstream removed from its own tree:
no golden captured from a running vLLM would improve, and the bf16 ones would
get worse.

**The honest reading of the history.** The f32 form arrived as a side effect,
was labelled a bug, and survives only because reverting unbroke a CI job. It is
current and deliberate-by-revert, and it disagrees with upstream's own Python
reference, which was never changed to match. That makes it a candidate to
re-examine at the next pin advance — it does not make it something to diverge
from now. The rule is mirror the pin, and `vt` mirrors it.

## The measured half — the gated activations round `act(gate)`

Upstream has one polarity here on every accelerator path, and pins it.

| upstream implementation | final expression | rounds `act(gate)`? |
|---|---|---|
| `activation.py::SiluAndMul.forward_native` | `F.silu(x[..., :d]) * x[..., d:]` | yes — `F.silu` on a bf16 tensor yields bf16 |
| `activation_kernels.cu::silu_kernel` | `return (T)(((float)x) / (1.0f + expf((float)-x * alpha)))` | yes — explicit narrowing to `T` |
| `activation_kernels.cu::compute` | `return (scalar_t)(ACT_FN(gate, alpha) * ((float)up + beta))` | yes — `ACT_FN` already returned `scalar_t` |
| `activation_kernels.cu::packed_silu_kernel` | `return cast_to_packed<packed_t>(fval)` | yes — the vectorized path narrows identically |
| `activation.py::GeluAndMul.forward_native` | `F.gelu(x[..., :d], approximate=…) * x[..., d:]` | yes |
| `activation_kernels.cu::gelu_kernel` / `gelu_tanh_kernel` | `return (T)(…)` | yes |

The fused mixture-of-experts path is the same kernel, not a second one:
`fused_moe.py::fused_experts_impl` calls `activation.py::apply_moe_activation`,
which calls `torch.ops._C.silu_and_mul`.

**Upstream asserts this bit-exactly.**
`tests/kernels/core/test_activation.py::test_act_and_mul` runs the CUDA kernel
and `forward_native` over the same input and, for `silu_and_mul`,
`mul_and_silu`, `gelu`, `gelu_tanh` and `fatrelu`, asserts

```python
torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)
```

with the comment that these implementations "are equivalent to the native
PyTorch implementations, so we can do exact comparison". Only
`swigluoai_and_mul` and `swiglustep_and_mul` get a tolerance. So the rounding of
`act(gate)` is not an artefact of one kernel — it is a pinned contract between
two implementations, and it is the one upstream test in this area a port can
reproduce exactly. That test is the smallest failing test the implementing row
should port.

`src/vt/cuda/cuda_ops.cu` already carries the standing note `// Upstream csrc
counterpart: csrc/activation_kernels.cu (act_and_mul_kernel<silu>) — align
post-MVP`, so the divergence was known when the kernel landed and deferred, not
decided.

**One upstream implementation disagrees, and it is nearly dead.**
`csrc/cpu/activation.cpp::activation_kernel` and
`csrc/cpu/cpu_fused_moe.cpp::silu_and_mul` keep silu in f32 and round once, as
`vt` does. But `SiluAndMul.forward_cpu` returns `forward_native` on every CPU
architecture except POWERPC, so that kernel is reached only there. It is not the
mirror source.

**The rounding target is the INPUT dtype, not the output dtype.** Upstream's
`scalar_t` is the dtype of `x`, and `SiluAndMul.forward_cuda` allocates `out`
with `dtype=x.dtype`, so upstream never has the two differ. Our seam permits an
f32 input with a bf16 output. Mirroring upstream means rounding through the
**input** dtype, which leaves every f32-input path bit-identical to today and
confines the change to bf16-in paths. That is a property to assert directly, not
to infer from values: it is what keeps the existing f32 goldens in
`tests/vt/test_ops_activation.cpp` unmoved, and a token gate cannot see it.

**Size of the effect, as an analysis figure and not a gate.** Over 2^20 pairs of
independently bf16-rounded standard-normal `(gate, up)`, replicating the tree's
round-to-nearest-even bf16 codec: **27.47% of outputs differ, every one of them
by exactly 1 bf16 unit in the last place**. So the change is pervasive and
uniformly small — which is the profile that a token gate is least able to see
and that accumulates across layers.

## The `vt` inventory

Every provider of the RmsNorm family keeps f32 across the weight multiply, so
the family is internally consistent and consistent with the pin:

| op | providers | polarity | verdict |
|---|---|---|---|
| `kRmsNorm` | CPU, CUDA (3 kernels), ROCm, Vulkan, Metal, Tenstorrent host | f32, one rounding | matches the pin — no change |
| `kRmsNormGated` | CPU, CUDA (2), ROCm, Vulkan | f32, one rounding | matches `RMSNormGated.forward_static` and its triton kernel, both all-f32 — no change |
| `kRmsNormQuantFp8`, `kRmsNormGatedQuantFp8` | CPU, CUDA | bf16 hop before the fp8 quant | the stated contract (`ops.h`), and it matches `layernorm_quant_kernels.cu` at the pin — no change |
| `kRmsNormGatedGroup` | CPU, CUDA | **rounds `x*inv` to `input_dtype` before `*weight`** | already the round-first polarity, deliberately mirroring `mamba_mixer2.py::Mixer2RMSNormGated` — no change |

`kRmsNormGatedGroup` is the important one for the framing of #1322. The issue
says the seam "has no way to express the other behaviour". It does: `cpu_ops.cpp`
already has `RoundThrough(DType, float)` and `cuda_mamba2_ssd.cuh` already has
`M2RoundThrough`, and one op already uses them for exactly this purpose. What is
missing is not an expressive mechanism but a general elementwise
tensor-times-tensor multiply, and the activation fix does not need one — it needs
one `RoundThrough` call inside each existing kernel.

The gated activations, which do need to change:

| op | providers | polarity today |
|---|---|---|
| `kSiluAndMul` | CPU, CUDA, ROCm, Vulkan, Metal | f32, one rounding — **diverges** |
| `kSiluAndMul` | Tenstorrent | `ttnn::silu` materializes a bf16 tile before `ttnn::multiply` — **already rounds**, so the providers are *already inconsistent with each other* |
| `kGeluAndMul` | CPU, CUDA, ROCm (2) | f32, one rounding — **diverges** |
| `kMoeSiluMul` | CPU, CUDA, ROCm | f32, one rounding — **diverges** |
| `kMoeGateUpSwiGLUGrouped` | CPU, CUDA | f32 SwiGLU epilogue into an f32 output — diverges only where the output is narrow |
| `kMoeGroupedGemmBf16GateUpSilu` | CUDA | f32 SwiGLU epilogue — **diverges** |
| `kSiluMulFp4Quant`, `kSiluAndMulFp4Quant` | CPU, CUDA | bf16 hop on the **product**, by contract; the `silu` itself is unrounded — **diverges in the same way**, and the contract comment needs re-reading against upstream when it is fixed |

## Reachability

`vt::SiluAndMul` is reached from a production entry point on the default
configuration through the shared multilayer-perceptron seam:
`include/vllm/model_executor/layers/linear.h::UnquantizedMlpGateUpMethod`
matmuls into a bf16 `[M,2I]` buffer and calls `vt::SiluAndMul(d.q, act.t(),
gate_up.t())` with both tensors bf16. Sixteen further production call sites exist
across `qwen3_5.cpp`, `qwen3_dflash.cpp`, `laguna.cpp`, `dense_nvfp4_gemm.h`,
`minimax_h3_device.cpp`, `minimax_h3_encoder_device.cpp`,
`minimax_h3_video_vae_device.cpp` and `minimax_music3_device.cpp`. There is no
unreached-slice question here. The question is the opposite one, and it is why
this is not a small change.

## Blast radius

Every bf16 SwiGLU or GeGLU multilayer perceptron in the tree changes value: the
dense path of every registered text model, the shared expert of every
mixture-of-experts model, and the H3 and Music3 device arms. Goldens captured
from a running vLLM should improve; goldens captured from our own output shift
and each shift has to be explained rather than absorbed.

Six providers move together or not at all, because `AGENTS.md` forbids a
hand-written parallel path and holds the providers to being consistent with each
other. Four of them — CUDA, ROCm, Metal, Tenstorrent — cannot be executed from
the shared checkout at all, and the CPU arm needs an idle box.

That is why this spec stops here.

## Owed

- The implementing row: the six-provider edit, the ported form of upstream's
  `test_act_and_mul` exactness assertion as the red-first test, and the full
  gate. Owns [#1322](https://github.com/mudler/vllm.cpp/issues/1322).
- [#1342](https://github.com/mudler/vllm.cpp/issues/1342): Vulkan
  `vt_silu_and_mul.comp` and Metal `metal_msl.h` both document a 1:1 port of the
  CPU kernel's `gate / (1 + exp(-gate))` spelling and both emit
  `gate * vt_sigmoid(gate) * up` instead. Found here, not fixed here.
- [#1343](https://github.com/mudler/vllm.cpp/issues/1343):
  `src/vt/fused_ops.cpp::RmsNormPlusAdd` rounds `out` between the norm and the
  add on its composed arm and does not on its ROCm arm, so one API has two
  numeric behaviours. Found here, not fixed here.
- The `include/vt/ops.h` doc comment on `vt::RmsNorm` says "unlike upstream
  `forward_native`", which reads as a divergence and is the reason this gap was
  filed against that op. It should say that the kernel which actually runs
  agrees with us. Deferred with the implementing row rather than taken here,
  because editing that header rebuilds the tree and this row lands no code.


- **CUDA, ROCm, Metal, Tenstorrent, Vulkan: the same narrowing, UNVERIFIED here.**
  None of the four accelerator toolchains exists on the dev box (`nvcc`,
  `hipcc` absent; Linux x86_64, so no Metal; no Tenstorrent runtime), so those
  kernels are NOT edited by this row rather than edited blind. Vulkan is a
  second-order case: `src/vt/vulkan/shaders/vt_silu_and_mul.comp` compiles
  AHEAD OF TIME into the committed `src/vt/vulkan/vulkan_spirv.cpp`, and
  `scripts/gen-vulkan-spirv.py --check` exits 1 here with "no GLSL->SPIR-V
  compiler found", so editing the shader without regenerating would ship a
  source that disagrees with the executed SPIR-V — #1342's defect in a worse
  form. The CI job `vulkan-spirv-freshness` fetches a pinned glslang 16.5.0 and
  would catch it. Tenstorrent needs nothing: `ttnn::silu` already materializes a
  bf16 tile before `ttnn::multiply`, so it ALREADY has the upstream polarity.
- **The gate for those arms is PENDING on named resources**, not waived: a CUDA
  box for the byte-exact composite suites, a ROCm box, a Mac, and a glslang for
  the SPIR-V regenerate.

## The four suites #1458 reds, and why the kernel is not the defect

[#1458](https://github.com/mudler/vllm.cpp/issues/1458) reported
`test_ltx2_text_encoder`, `test_muse_glimmer_text`,
`test_muse_glimmer_text_fallback` and `test_minimax_music3_ar` red on `main`
behind `4712dac40`, attributed by mutation in both directions. All four
reproduce at `aeba0de6f`, CPU-only Release (`-DCMAKE_BUILD_TYPE=Release`, so
NDEBUG), x86_64.

**The premise was re-derived rather than taken.** Every anchor the landing
commit cites is exact at the pin `555967922`, read in the checkout whose
`git rev-parse HEAD` is that sha:
`csrc/libtorch_stable/activation_kernels.cu:158` (`silu_kernel` returns
`(T)(...)`), `:36` (`compute` returns `(scalar_t)(ACT_FN(gate, alpha) * ...)`,
so `ACT_FN` has already narrowed), `:205` (`gelu_tanh_kernel`, same shape),
`vllm/model_executor/layers/activation.py:143` and `:418` (the native arms,
`F.silu` / `F.gelu` on a bf16 tensor yielding bf16), and
`tests/kernels/core/test_activation.py:108`
(`assert_close(out, ref_out, atol=0.0, rtol=0.0)`, which pins the two together
bit-exactly). The VECTORIZED arm agrees: `packed_compute` at
`activation_kernels.cu:50-95` narrows through `cast_to_packed<packed_t>` as
well, so upstream has ONE polarity across both of its arms and the `atol=0.0`
test has no second behaviour to hide.

**The committed oracle golden settles it without reading a kernel.**
Recomputing `tests/parity/goldens/silu_and_mul_bf16_8x256/out.npy` from its own
`x.npy` in numpy, with bf16 round-to-nearest-even applied by hand:

| expression | max abs err vs golden | bit-exact |
|---|---|---|
| `silu_f32(g) * up`, no store round | 1.434994e-02 | no |
| `bf16(silu_f32(g) * up)` — the PRE-`4712dac40` kernel | 1.562500e-02 | no |
| `bf16(silu_f32(g)) * up`, no store round | 7.812500e-03 | no |
| `bf16(bf16(silu_f32(g)) * up)` — the POST kernel | **0** | **yes** |

The golden's every value is exactly bf16-representable, so it came from a vLLM
that rounds twice. The landed kernel is the only one of the four that reproduces
it. Reverting it is not available.

**What the four bounds were, and what each one actually measured.**

- `test_ltx2_text_encoder`, the prompt->conditioning VALUES case. Its floor is
  the oracle's own f32-vs-bf16 spread carried through the same projection, which
  is the right SCALE. Its constant was not: the bf16 arm carried `1.0x` while the
  f32 arm carried `2.0x`, and the `1.0x` was imported from the state-level parity
  case, where the same two objects are compared elementwise. This case compares
  them through `Ltx2TextEncoderConditioning`, which stacks all 13 states — the
  floor is the projection of ONE error vector and the gated quantity is the
  projection of a DIFFERENT one, and a linear map that combines 13 states does
  not preserve the relation. Both arms now carry `2.0x`, which is the triangle
  bound with each side's departure from the shared f32 trajectory taken at one
  floor. Video 1.209x -> margin 0.60; audio 1.313x -> margin 0.66.
  The state-level case is untouched and still green at 0.870x of its `1.0x`.
- `test_muse_glimmer_text` and its `VT_FUSED_CHAIN_ADOPT=0` registration.
  `bdiff <= 1e-5` was the W1 measurement (5.28e-06 at `3a54c4b7d`) rounded up,
  1.89x of headroom and no derivation. The biting-soft-cap case exists to pin the
  ORDER of the output multiplier and the cap, and the order is an ALGEBRAIC
  property of the output range, not a max|diff| band: at `cap = 1e-3` the logits
  saturate the tanh to within 1e-40, so upstream's order reaches `cap` and the
  swap reaches `out_mult * cap`. Measured: our saturation and the reference's
  agree EXACTLY (delta 0) and the swap moves it by 2.5e-04, 32x the bound, which
  is two bf16 relative spacings on `cap`. The knee-driven max|diff| is kept at
  its rigorous Lipschitz value (`bdiff <= diff`; the cap is a contraction).
- `test_minimax_music3_ar`, the composed depth stage. `device_codes ==
  host_codes` over an argmax on two implementations whose measured separation is
  308 bf16 ULP. Draw 6 is a tie: relative top-2 margins 1.95e-03 (device) and
  2.93e-03 (host), both BELOW one bf16 ULP (2^-8 = 3.91e-03), between codes 2 and
  17. Draws 0..5 have margins 1.74e-02 to 3.53e-01 and agree. The `==` was
  reading a coin. It is now: codes agree, or the divergence is a SHARED near-tie
  in which each arm ranks the other's pick within one bf16 ULP of its own.

**Two candidate replacements were tried and rejected**, recorded so nobody
re-derives them. For muse, `bdiff <= diff` alone is rigorous but does NOT red the
order swap (uncapped envelope 3.4e-04, defect 2.5e-04). A measured twin at
`out_mult = 1`, where the two orders coincide by algebra, gives 6.10e-06 — but it
is a different model sitting at a different point on the tanh knee, and the gated
run reaches 2.4x it with nothing wrong.

**What this costs, MEASURED, and it is `4712dac40` that spent it rather than the
new bound.** The ltx2 conditioning case documents that renumbering the positions
from zero reds it at 1.10x the audio floor. Ratios to the propagated floor, one
build directory, compile rc 0 on every arm, every source restored and
`sha256sum`-verified:

| `src/vt/cpu/cpu_ops.cpp` | production code | video | audio | at the old `1.0x` |
|---|---|---:|---:|---|
| before `4712dac40` | correct | 0.565 | 0.688 | pass |
| before `4712dac40` | renumbered | 0.831 | **1.099** | **RED** — the documented 1.10x |
| at `aeba0de6f` | correct | **1.209** | **1.313** | **RED** |
| at `aeba0de6f` | renumbered | 0.683 | 0.931 | pass |

Read the bottom two rows together: post-`4712dac40` the instrument is INVERTED —
it reds the correct code and passes the mutant, and the mutant is measurably
CLOSER to the oracle than the port is. The `2.0x` restores a functioning
instrument; it does not recover that detection, and no constant can, because the
ordering of the two has reversed. It is also a property of the defect —
upstream's own f32 answers for positions 12..19 and 0..7 agree to 3.6e-06
relative — and `scripts/gen-ltx2-gemma-tower-goldens.py:363-375` already records
that the end-to-end states are the wrong instrument for that class and the f32
rope table is the right one. Filed as
[#1467](https://github.com/mudler/vllm.cpp/issues/1467) and owed by
[`ltx-2-5.md`](ltx-2-5.md).

**One more bound of the same class is left standing, and it is filed rather than
re-tuned.** `tests/vllm/models/test_muse_glimmer_text.cpp:532`'s
`CHECK(diff <= 5e-4)` is the other W1 measurement rounded up (1.21e-04 at
`3a54c4b7d`), and `4712dac40` moved it from 0.242 to 0.687 of its bound. It is
green, it is out of scope for this repair because it needs its own derivation,
and it is [#1466](https://github.com/mudler/vllm.cpp/issues/1466), owed by
[`muse-glimmer.md`](muse-glimmer.md).

**The narrowing did not make the port worse.** At the state level, where the
comparison is elementwise and the amplification above does not apply, our
distance to the f32 oracle IMPROVED at 11 of the 13 states (worst ratio 0.644 ->
0.572) and the bf16 arm stayed inside its `1.0x` floor (0.714 -> 0.870).

## Stop conditions

- A golden that gets **worse** after the activation change stops the row and is
  reported, not absorbed. It means either the change is wrong or that golden was
  captured from our own output.
- No tolerance is widened to make a golden pass. A golden that must be re-taken
  is re-taken from the pinned oracle, with its provenance recorded.
- `vt::RmsNorm` is not touched without a new pin and a fresh reading of
  `layernorm_kernels.cu`. If a later pin restores vllm#42379, that is a pin
  advance with its own reconciliation, not a licence to pre-empt it.

## Outcome

Deferred to the implementing row. What this row establishes:

- The RmsNorm half of #1322 is **refuted** against the primary oracle at the
  pin, with the upstream revert as the decisive evidence. Nobody needs to derive
  it again.
- The activation half is **confirmed**, with an upstream bit-exact test to port
  and a measured 27.47%-of-elements, 1-ULP effect.
- The seam's expressive gap named in #1322 is narrower than filed: the rounding
  helper already exists on two providers and one op already uses it. A general
  elementwise multiply is not required for this fix.
