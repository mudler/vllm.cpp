# CUDA `RmsNorm` refuses an f32 gamma against a bf16 activation

**Row:** `MODEL-MM-QWEN4-EXP` (wave SANITIZE)
**Issue:** [#2477](https://github.com/mudler/vllm.cpp/issues/2477)
**State:** `ACTIVE`
**Base:** `origin/main` at `855905f59`

## Scope

Make the CUDA `RmsNorm` kernel read its gamma in the gamma's own dtype, so a
`--device cuda` forward of `qwen4_exp` on the released GGUF artifact stops
refusing at `src/vt/cuda/cuda_ops.cu:463`.

Out of scope, recorded under `## Owed`: the ROCm twin of the same weld, and
issue #2476.

## What is wrong, and which side of the mismatch it is

`VT_CHECK(w.dtype == x.dtype)` in `RmsNormKernelCuda` is not an arbitrary
restriction. `RmsNormRowKernel` is declared
`template <typename Tin, typename Tout, typename Tres>` and takes the gamma as
`const Tin* w` — **the activation's type**. The dispatcher's equality check
mirrors that weld exactly, so widening the check on its own would read a bf16
gamma buffer through a `const float*` and walk off the end of a `[128]`
allocation. The check is honest; the kernel behind it is too narrow.

**Neither side of the mismatch is the defect.** Both operands are the value
upstream says they should be:

- **The activation is bf16, and must be.** `qwen4_exp_qsa_block.cpp:446`
  refuses anything else (`VT_CHECK(hidden.dtype == DType::kBF16, ...)`), which
  mirrors vLLM's own refusal at
  `vllm/models/qwen4_exp/nvidia/qsa.py:188-189`
  (`if model_config.dtype != torch.bfloat16: raise NotImplementedError(
  "Qwen4Exp QSA currently requires BF16")`), read at vLLM `origin/main`
  `25efcfa788`. vLLM reaches `q_norm` through a plain `torch.chunk`
  (`vllm/model_executor/models/qwen3_next.py:426-435`), which does not change
  dtype, so upstream's norm input is bf16 too.
- **The gamma is f32, and must be.** Every norm tensor in the released
  `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact is stored `F32`:
  `blk.N.attn_q_norm.weight` `[256]`, `blk.N.attn_k_norm.weight` `[256]`,
  `blk.N.indexer.q_norm.weight` `[128]`, `blk.N.indexer.k_norm.weight` `[128]`,
  `blk.N.hc_*_norm.weight` `[10240]`, `blk.N.ple_norm_*.weight` `[10240]`.
  Measured by parsing the three shards' GGUF tensor tables directly.

### Which of the three shapes this is

`qwen4-exp-matmul-bt-mixed-dtype.md` set this row's taxonomy for a mixed-dtype
refusal, and it applies unchanged here. This is its **shape 2**: the reference
genuinely runs the op mixed, so mirror it and match the accumulate and output
dtypes.

It is explicitly **not shape 1**, an activation widened to f32 that should have
stayed bf16. At the site that refuses first (`:632`) the activation is
`q_index_raw`, allocated at `hidden.dtype` (`:592`) and therefore bf16. Nothing
on that path was widened.

It is also **not shape 3**, and this matters. That spec records that
`vt::MatmulBT` on CPU reads both operands through a dtype-generic getter, "which
makes CPU a value oracle and never a dtype-policy oracle". The CPU `RmsNorm`
being laxer is therefore corroboration and never the argument. The argument is
vLLM: `GemmaRMSNorm` upcasts the gamma itself.

vLLM never requires the two to agree. `GemmaRMSNorm` upcasts the gamma on its
own (`ple_layer.py:80`: `normalized * (1.0 + self.weight.float())`), and the
`vllm.cpp` **CPU** kernel already does exactly that: `cpu_ops.cpp:554-557`
widens `w` through `WidenRowToF32(w.dtype, ...)` for f32/f16/bf16, and
`cpu_ops.cpp:576` widens `x` separately. That is why the CPU control produces
tokens on the same checkpoint where CUDA refuses.

The tree also already carries the correct CUDA pattern one file over:
`RmsNormGroupKernelCuda` passes an **independent** weight tag
(`cuda_rms_norm_group.cu:194`, `w_tag = TagOf(weight.dtype, "weight")`). That
is why the PLE and hyper-connection grouped norms — the same f32 gamma against
the same bf16 stream — pass, and why only the plain `RmsNorm` refuses.

`RmsNormRowKernel` is therefore the outlier among three siblings that all get
it right.

**This row had already written the rule down.** `src/vt/cuda/cuda_qwen4_exp.cu:60-62`,
from an earlier wave of this same campaign, states it in as many words:

> a device arm that refused a dtype its CPU sibling accepts would be a divergence
> to record, and the tag design below makes admitting it free.

That file names `cuda_ops.cu`'s `<Tin, Tout>` house style as the thing it
deliberately diverged from, and carries each operand's dtype as an independent
runtime tag instead — `HcGroupedNormKernel(float* normed, const void* hyper,
DTag hyper_tag, const void* hc_norm_w, DTag w_tag, ...)` at `:258-259`. So the
qwen4_exp-specific CUDA ops already accept the f32 gamma this checkpoint stores.
`RmsNormRowKernel` is the op that had not caught up, and the fix below is that
principle applied in `cuda_ops.cu`'s own template idiom rather than a new one.

### Why `ResidentWeightF32` is not the fix here

`qwen3_5.cpp:1226` exists for the mirror-image pairing (f32 activation, gamma
on disk at the checkpoint dtype) and `qwen3_5.cpp:5329-5337` selects between
the raw and the upcast weight per site. That precedent does not reach this
case. It can only widen a gamma to f32; the failing site has a **bf16**
activation, so matching it would mean *narrowing* an f32 gamma to bf16 — a
precision loss vLLM does not have, applied to the value `GemmaRMSNorm`
explicitly calls `.float()` on.

## Which call site fires first

All three plain-`vt::RmsNorm` call sites in `qwen4_exp` are in the QSA block,
and the block is layer 3 of the repeating `3 x linear -> 1 x QSA` pattern:

| line | activation | dtype | gamma | verdict |
|---|---|---|---|---|
| `qwen4_exp_qsa_block.cpp:632` | `q_index_raw` (`:592`, `hidden.dtype`) | bf16 | f32 | **refuses, and is first** |
| `qwen4_exp_qsa_block.cpp:705` | `q_f32` (`:694`, `DType::kF32`) | f32 | f32 | passes |
| `qwen4_exp_qsa_block.cpp:736` | `k_raw` (`:732`, `hidden.dtype`) | bf16 | f32 | refuses |

`idx_k_norm` reaches the same kernel through `Qwen4ExpQsaIndex` (`:690`).

## Design

Give `RmsNormRowKernel` its own gamma type:

```
template <typename Tin, typename Tw, typename Tout, typename Tres>
__global__ void RmsNormRowKernel(Tout* out, const Tin* x, const Tw* w, ...)
```

`Load()` is already overloaded for `float` and `__nv_bfloat16`
(`cuda_ops.cu:51-52`), so the kernel body is unchanged. `LaunchRmsNormRes` and
`LaunchRmsNorm` forward `Tw`; `RmsNormKernelCuda` replaces the equality check
with a dispatch over the gamma's dtype and refuses an unsupported one by name.

Neither vectorized fast path changes. `TryLaunchRmsNormDecodeFast` already
requires `w.dtype == DType::kBF16` and `TryLaunchRmsNormDecodeFastF32` requires
`w.dtype == DType::kF32`, so both self-guard and keep their bit-identity.

## Risks

- **A too-wide dtype is invisible to a token gate.** This change widens no
  buffer: it reads the gamma the checkpoint already stores, at its own width,
  and adds no `f32` model-path allocation. The activation stays bf16.
- **Silently reading the wrong bytes.** This is the failure the current check
  prevents, so the check is narrowed rather than deleted: an unsupported gamma
  dtype is still refused, by name.
- Instantiation count goes from 8 to 16. Compile-time only.

## Tests

Red before green: a `RmsNorm` with bf16 `x`/`out` and an f32 `w` must produce
the CPU kernel's values rather than a refusal, on every registered device.

## The precondition that refused a good tree

Run 1 on `thor:gpu0` staged the artifact in 1766 s, asserted the fix was in the
tarball it was about to compile, and refused:

```text
RESULT FIX IN SOURCE: Tw=1 dispatch=3 old_check=2 test=1
FATAL: the tarball does not carry the fix. REFUSING to measure it.
```

The tarball did carry the fix. `old_check` grepped the bare phrase
`weight dtype must match x`, which also appears at `cuda_ops.cu:578`
(`rmsnorm_quant_fp8`) and `:3689` (`fused_chain`) — two different ops this change
deliberately does not touch. The correct count of the RmsNorm message is 0 and
was 0. The grep was too broad, so it counted its own neighbours.

A grep precondition also cannot detect the defect that actually mattered. The
same tree failed `cuda-fat-build` on
`cuda_ops.cu:485: error: macro "VT_CHECK" passed 3 arguments, but takes just 2` —
the refusal message was written in a 3-argument form the 2-argument macro does
not accept. Every `FIX IN SOURCE` grep passed on that tree, because the strings
they look for were all present. **Only a compiler can say a change compiles**, so
the CI build is the precondition that matters and the greps only assert which
change is being compiled.

It is recorded because of the direction it failed in. A precondition that is too
broad **refuses a good tree**, which costs a lease and a staging run and is
visible immediately. A precondition that is too narrow **passes a bad tree**, and
then every arm below it reports a measurement of something other than what it
names. This project's standing hazard is the second one, so the cost here bought
the right polarity. The repair is to match the op's own message rather than a
phrase several ops share, and to stage persistently so a refusal costs the lease
and not the 1766 s as well.

## Gates

- `tests/vt/test_ops_rmsnorm*` focused, then the full gate.
- End to end: `--device cuda` on the released UD-IQ1_S artifact, token ids
  compared against the CPU control `11751 13 15767 411 2029 11 1092 369`.

## Stop conditions

A third wall after #2477 and #2476 clear is a reportable result, not a failure.

## Owed

- **Two unreached twins of this same weld**, tracked by
  [#2492](https://github.com/mudler/vllm.cpp/issues/2492):
  - `src/vt/rocm/rocm_rmsnorm.hip:105-177` is a line-for-line twin
    (`RmsNormRowKernel<Tin, Tout, ...>` with `const Tin* w`, and
    `VT_CHECK(w.dtype == x.dtype)` at `:170`), so it will refuse a GGUF f32 gamma
    exactly as CUDA did. No ROCm device is in the `rc` fleet, so this wave can
    neither build nor gate it, and editing an untestable backend blind is worse
    than recording it.
  - `src/vt/cuda/cuda_ops.cu:578`, `RmsNormQuantFp8KernelCuda`, welds the same two
    dtypes over `LaunchRmsNormQuantFp8<Tin>`. It is latent rather than live: its
    consumers load safetensors checkpoints whose gammas already sit at the model
    dtype, and `qwen4_exp` does not reach the op at all. Fixing it here would add
    untested surface to a path this row cannot exercise.
- [#2476](https://github.com/mudler/vllm.cpp/issues/2476), the illegal memory
  access, whose ordering against this refusal this wave measures.
- [#2488](https://github.com/mudler/vllm.cpp/issues/2488) tracks the widening below.
- **A genuine shape-1 widening, found while reading this path and not fixed
  here.** `qwen4_exp_qsa_block.cpp:694-696` allocates `q_f32` and `gate` at
  `DType::kF32` and `vt::AttnGateSplit` fills them, where vLLM reaches the same
  point through `torch.chunk` (`qwen3_next.py:426-435`), which does not change
  dtype and leaves `q` at the bf16 model dtype. The value is already bf16-rounded
  by the `qgate` store at `:691`, so the widening costs bandwidth rather than
  precision — `T * Hq * Dh * 4` bytes where upstream moves half that, for `Hq=24`,
  `Dh=256`. It is exactly the too-wide dtype a token gate cannot see.

  It is not fixed in this wave for two reasons. It is not a wall: that site
  pairs an f32 activation with the f32 gamma and passes today. And
  `AttnGateSplit`'s f32 output is welded into the op's own CUDA signature
  (`cuda_glue.cu:181`, `AttnGateSplitKernel(float* q_out, float* gate_out, ...)`)
  and is shared with `qwen3_5.cpp:5328` and `:5501`, so narrowing it changes a
  second model's path and needs that model's gate, not this one's.

## Evidence

Measured on `thor:gpu0` (NVIDIA Thor, `compute_cap 11.0`, driver 595.78, CUDA
13.0.88, aarch64), serving the released `unsloth/Qwen3.8-Flash-Next-GGUF`
UD-IQ1_S artifact: 72,546,461,344 bytes, shard-1 sha256
`88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`, staged in
2195 s and verified by both byte count and hash before any arm ran.

Binary sha256 `70e522df28d7748d8925ce9fc54dffd4909b3e8fd53d72fafd87bb13bed62056`,
built for `sm_110` from `e934fb0020ba099125994e9a44e93e8c89d87977` (1662 s,
`cuda_libs=2`, `cu_objects=41`). The source was asserted before compiling:
`Tw=1 dispatch=3 old_check=0 test=1`.

**Focused test, on the device:** `rc=0`, `assertions=4`, `skip_lines=0`,
`FOCUSED CUDA ARM EXERCISED: YES`. The assertion count is reported because
`assertions: 0 ... SUCCESS!` at rc 0 is a skip wearing a pass; four assertions
and zero `[SKIP]` lines mean the CUDA arm ran.

**End to end**, prompt `The capital of France is`, `max_tokens=8`,
`temperature=0`, `VT_CPU_QUANT_REPACK=0`:

| arm | `CUDA_LAUNCH_BLOCKING` | outcome | wall | IMA | refusal |
|---|---|---|---|---|---|
| ARM FIX | 0 | HTTP 500, #2476 | 237 s | 3 | **0** |
| ARM FIXLB | 1 | **HTTP 200, 8 tokens** | 200 s | 0 | **0** |

`rmsnorm-dtype refusal present: 0` in both arms is this change's own result: the
wall at `cuda_ops.cu:463` is gone from a real forward, not only from a unit test.

Host `sys_used` peaked at 40.0 GiB (ARM FIX) and 42.5 GiB (ARM FIXLB), so the
n-gram table stayed quantized and residency is not implicated.

## What this did NOT fix

A token now comes out of a GPU, and it is **wrong after the first one**:

```
GPU : 11751 271 271 271 271 271 0 0        " Paris\n\n\n\n\n\n\n\n\n\n!!"
CPU : 11751 13 15767 411 2029 11 1092 369  " Paris. Given this fact, what is"
```

Token 0 agrees; tokens 1-7 do not. That is [#2496](https://github.com/mudler/vllm.cpp/issues/2496),
a third wall this change does not touch and does not claim to. The CPU sequence is
the control recorded in the wave brief for this artifact and was **not** re-measured
in this run, so re-taking it is the first step on that issue.

[#2476](https://github.com/mudler/vllm.cpp/issues/2476) also did not dissolve. It
is independent, it is FIRST in program order — its faulting kernel runs in layers
0-2, ahead of the layer-3 QSA block that holds the first plain `vt::RmsNorm` — and
it is timing-dependent: serialising launches suppresses it entirely, which is why
the pre-fix binary showed this refusal under `CUDA_LAUNCH_BLOCKING=1` and the
illegal access without it. It blocks the gate, because the only arm that completes
a forward is not a production configuration.

## Now

`ACTIVE`. The row's CUDA forward is unblocked at this wall and stops at #2476.
