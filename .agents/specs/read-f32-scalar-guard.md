# Read a per-tensor scale as a per-tensor scale

Row `FIX-READ-F32-SCALAR-GUARD`. Issue
[#1181](https://github.com/mudler/vllm.cpp/issues/1181).

## Scope

`ReadF32Scalar` copies four bytes out of a safetensors tensor and calls them a
float. It checks neither how many elements the tensor holds nor what dtype it
declares. Give it both checks, in one shared implementation the whole tree
reads through, and make each refusal name the offending tensor, its shape, and
its dtype.

Out of scope, and recorded under `## Owed`: block-wise FP8, per-output-channel
FP8 on the per-tensor arm, and any explicit narrow-dtype conversion. Also out of
scope: the `#1166` refusal that landed in `469f38395`, which is untouched here.

## 0. What is wrong today

`src/vllm/model_executor/models/qwen3_5_weights.cpp:312-318` at `ab6e65216`:

```cpp
float ReadF32Scalar(const StTensor& t) {
  VT_CHECK(t.data != nullptr && t.nbytes >= sizeof(float),
           "qwen3_5 weights: scalar tensor too small for f32");
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  return v;
}
```

`nbytes >= sizeof(float)` is a LOWER bound, so two wrong-value paths follow and
neither of them fails.

1. **An array is silently reduced to element 0.** A block-wise FP8 scale grid of
   shape `[ceil(N/128), ceil(K/128)]` passes trivially and is read as block
   `(0, 0)`, which then stands in for the whole weight. Measured under #1166 on
   `Qwen/Qwen3.8-27B-FP8` at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`:
   `q_proj.weight_scale_inv` is `[96, 40]`.
2. **Any dtype is reinterpreted.** The function copies four bytes whatever
   `t.dtype` says. That same measured tensor is `BF16`, so the four bytes are
   two bf16 values read as one float, and the result is not a scale at all.

Both return a finite, plausible float. A wrong scale produces fluent, plausible,
wrong tokens, which is the failure mode a token gate cannot see. The tree
already argues exactly this at
`include/vllm/model_executor/models/dense_weight_loaders.h:78-80`, where a
per-output-channel scale read as per-tensor is called out as "silently WRONG
rather than loud".

## 1. What upstream does, with anchors

Pinned vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`, the parity pin in
[`upstream-sync.md`](../upstream-sync.md), verified with `git rev-parse` in the
oracle checkout before citing.

| Anchor | What it does |
|---|---|
| `vllm/model_executor/parameter.py:260-272` | `PerTensorScaleParameter` is a distinct parameter TYPE. A per-tensor scale is not a raw tensor read, it is a typed slot |
| `vllm/model_executor/parameter.py:304-309` | `_load_into_shard_id` asserts `loaded_weight.shape[0] == 1` for any non-rank-0 scale, then asserts the shapes match. This is the element-count check we lack |
| `vllm/model_executor/parameter.py:93-96` | `_assert_and_load` asserts `self.data.shape == loaded_weight.shape` or the rank-0 to `[1]` case. Same rule on the non-sharded path |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py:1276` | the scale parameter is allocated `torch.float32`, so `copy_` VALUE-converts a narrower on-disk dtype rather than reinterpreting its bytes |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:63,84,128` | the declared STRATEGY (`TENSOR` / `CHANNEL` / `BLOCK`) picks the parameter type before a byte is read. Shape is never inferred from bytes |
| `vllm/model_executor/layers/quantization/fp8.py:358-366` | non-block FP8 registers `weight_scale` as a `PerTensorScaleParameter`, and the block arm registers a `BlockQuantScaleParameter` instead |
| `vllm/model_executor/layers/quantization/fp8.py:469-483` | at apply time upstream still branches on `weight_scale.numel() == 1` versus per-row. The element count is load-bearing on both sides |

The mirror is exact and it is not a design question. `ReadF32Scalar` is this
tree's `PerTensorScaleParameter`. It kept the copy and dropped the assertions.

## 2. The call-site audit

`grep -rn ReadF32Scalar src include` over the five files the issue names returns
27 hits. Reconciled, those 27 are **5 definitions, 20 call sites, and 2 comment
references**, and the file list is INCOMPLETE. A sixth definition is spelled
differently and a sixth model file reaches it.

### Definitions

| Definition | dtype checked | element count checked |
|---|---|---|
| `src/vllm/model_executor/models/qwen3_5_weights.cpp:312` `ReadF32Scalar` | no | no |
| `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:37` `ReadF32Scalar` | no | no |
| `include/vllm/model_executor/models/dense_weight_loaders.h:376` `ReadCtF32Scalar` | no | no |
| `src/vllm/model_executor/models/laguna_weights.cpp:249` `LnReadF32Scalar` | yes | no |
| `src/vllm/model_executor/models/laguna_shared_fp4.cpp:73` `ShReadF32Scalar` | yes | no |
| `src/vllm/model_executor/models/nemotron_h_weights.cpp:557` `ReadF32Scalar` | yes | yes |

Six hand-written copies of one function is the parallel path AGENTS.md forbids,
and the table is what a parallel path costs. Three copies check nothing, two
check half, and one is correct. `nemotron_h_weights.cpp:557-573` is the correct
one and is the model this row generalizes: it refuses a non-`F32` dtype by name,
refuses any shape that is neither rank-0 nor `[1]` by name, and requires exactly
four bytes.

### Call sites, and what each passes

Every site reads a scale that is per-tensor BY CONTRACT, so no site
LEGITIMATELY passes a multi-element or non-`F32` tensor. Two sites can be
REACHED by one, and that is the second finding below.

| Site | Tensor | Expected on disk |
|---|---|---|
| `qwen3_5_weights.cpp:458` `LoadFp8Raw` | `<proj>.weight_scale` | F32 scalar |
| `qwen3_5_weights.cpp:459` `LoadFp8Raw` | `<proj>.input_scale` | F32 scalar |
| `qwen3_5_weights.cpp:480` `LoadFp8Transposed` | `<proj>.weight_scale` | F32 scalar |
| `qwen3_5_weights.cpp:510` `LoadNvfp4Raw` | `<proj>.weight_scale_2` | F32 scalar (ModelOpt) |
| `qwen3_5_dense_weights.cpp:180` `LoadCtNvfp4Raw` | `<proj>.weight_global_scale` | F32 scalar (CT divisor) |
| `qwen3_5_dense_weights.cpp:193` `LoadCtNvfp4Raw` | `<proj>.input_global_scale` | F32 scalar (CT divisor) |
| `qwen3_5_dense_weights.cpp:305` `LoadLmHeadAnyDtype` | `<name>_scale_2` | F32 scalar (ModelOpt) |
| `qwen3_5_dense_weights.cpp:312` `LoadLmHeadAnyDtype` | `<name>_global_scale` | F32 scalar (CT) |
| `qwen3_5_dense_weights.cpp:377` `LoadModelOptNvfp4Raw` | `<proj>.weight_scale_2` | F32 scalar |
| `qwen3_5_dense_weights.cpp:393` `LoadModelOptNvfp4Raw` | `<proj>.input_scale` | F32 scalar, behind `VT_MODELOPT_W4A4` |
| `qwen3_5_dense_weights.cpp:611` | `<proj>.weight_global_scale` | F32 scalar (CT) |
| `dense_weight_loaders.h:421` `LoadCtNvfp4W4A16` | `<proj>.weight_global_scale` | F32 scalar (CT) |
| `laguna_weights.cpp:296` `LnLoadCtNvfp4Raw` | `<proj>.weight_global_scale` | F32 scalar |
| `laguna_weights.cpp:303` `LnLoadCtNvfp4Raw` | `<proj>.input_global_scale` | F32 scalar |
| `laguna_weights.cpp:326` `LnLoadSharedExpertBf16` | `<proj>.weight_global_scale` | F32 scalar |
| `laguna_shared_fp4.cpp:95` `ShLoadSharedNvfp4W4A16` | `<proj>.weight_global_scale` | F32 scalar |
| `nemotron_h_weights.cpp:604` | `<prefix>.weight_scale_2` | F32 scalar, already guarded |
| `nemotron_h_weights.cpp:626` | `<prefix>.weight_scale` | F32 scalar, already guarded |
| `nemotron_h_weights.cpp:627` | `<prefix>.input_scale` | F32 scalar, already guarded |
| `nemotron_h_weights.cpp:691` | `<mixer>.k_proj.k_scale` | F32 scalar, already guarded |
| `nemotron_h_weights.cpp:692` | `<mixer>.v_proj.v_scale` | F32 scalar, already guarded |

Every existing fixture agrees. `tests/vllm/models/test_qwen3_8_text_only.cpp:345,346,355`
emits `{1}` `F32`. `tests/vllm/models/test_laguna_nvfp4_loader.cpp:166,167` and
`tests/vllm/models/test_qwen27_dense_forward.cpp:331,333,398` emit rank-0 or
`{1}` `F32`. `tests/vllm/models/ltx2_nvfp4_te_manifest.inc` is rank-0 `F32`
throughout. Nothing in the tree needs the leniency, so nothing in the tree
breaks when it is removed.

### Finding A, named not absorbed: a SIXTH definition and a SIXTH model file

`ReadCtF32Scalar` (`dense_weight_loaders.h:376`) is the same defect under
another name, it already lives in the shared header, and the issue does not list
it. Its one caller `LoadCtNvfp4W4A16` is reached from
`src/vllm/model_executor/models/qwen3_weights.cpp:100,126,127,128`, the Qwen3
dense additive model, which the issue's five-file list also omits. Folded into
this row: the whole point is one implementation, and leaving a sixth copy out
would leave the class defect open in the file that is supposed to be the seam.

### Finding B, named not absorbed: a live reachable wrong scale, not only a latent one

The issue calls the defect latent because #1166's name miss stops the measured
`Qwen/Qwen3.8-27B-FP8` load first. That is true for THAT checkpoint. It is not
true for the class, and the tree already documents the counterexample.

`dense_weight_loaders.h:73-74` records that `unsloth/Qwen3.6-27B-NVFP4`
@`ccdaab7e` "went FP8 across the whole tower with BF16 per-output-channel
scales", and `docs/BENCHMARKS.md:52` records the same revision as "the same repo
name re-quantized to FP8 W8A8 throughout, not NVFP4".
`qwen3_5_dense_weights.cpp:258` names that layout again for `lm_head`, where it
is handled correctly.

`LoadAttnDense` branches on the weight dtype alone
(`qwen3_5_dense_weights.cpp:478-480`), so an `F8_E4M3` projection of that
checkpoint enters the per-tensor arm and `LoadFp8Raw` reads
`<proj>.weight_scale` through the unchecked reader. A `BF16 [out, 1]` scale
there is read as element `(0, 0)` of the wrong dtype: both defects at once, on a
real published revision, with the tensor NAME the loader asked for. Naming luck
does not protect this one, because nothing is misspelled.

No recorded gate covers it. Every recorded 27B NVFP4 measurement ran
@`890bdef7` (`qwen3_5_dense_weights.cpp:227-229`, `docs/BENCHMARKS.md:52`), and
@`ccdaab7e` is documented as historically rejected by the bf16 dense loader.
So this row turns a silent wrong scale into a named refusal on a path no gate
was reading. The per-output-channel FP8 arm itself is a separate capability and
is recorded under `## Owed`.

## 3. Design

One function, in the shared seam, with the name in the signature so the message
can never be anonymous:

```cpp
// include/vllm/model_executor/models/dense_weight_loaders.h
inline float ReadF32Scalar(const TensorResolver& get, const std::string& name);
```

It refuses in this order, and each refusal names `name`:

1. `numel(t.shape) != 1`, reporting the shape it actually got and the element
   count. Shape comes first because the array reduction is the defect that
   survives a correct dtype.
2. `t.dtype != "F32"`, reporting the dtype it actually got.
3. `t.data == nullptr || t.nbytes != sizeof(float)`, an exact byte count rather
   than a floor.

The five other definitions are deleted and their 16 call sites route here.
`nemotron_h_weights.cpp:557` stays as it is: its checks are already a superset,
and it carries `Loader` bookkeeping (`Need`, `RefuseLoad`, `Consumed`) that the
resolver-based helper has no access to. That is the one exact tracked exception,
recorded here rather than in a registry.

**A narrow dtype is refused, not converted.** Upstream converts by value
(`fp8_utils.py:1276` plus `copy_`), so conversion would be defensible. It is not
implemented, for two reasons that are evidence and not taste. No caller can be
shown to need it: a one-element BF16 scale has never been read correctly here,
because the four-byte copy takes two bytes of the scale and two bytes of
whatever follows it, so there is no working behavior to preserve. And the BF16
scale layout that IS shipped is per-output-channel, which the element-count
check refuses first whatever the dtype rule says. Adding a conversion would be
untested code on a path no checkpoint reaches. Recorded under `## Owed`.

Message prefix is `dense loader:`, matching the rest of the shared header, which
already states that a shared helper must not name one architecture
(`dense_weight_loaders.h:7-9`). The tensor name carries the architecture.

## 4. Port map

| Upstream | Local |
|---|---|
| `parameter.py:304-309` element-count assert | the `numel != 1` refusal |
| `parameter.py:93-96` shape assert | same, on the non-sharded path |
| `fp8_utils.py:1276` f32 scale slot | the `dtype != "F32"` refusal |
| `compressed_tensors_w8a8_fp8.py:128` strategy-to-type map | why the shape is never inferred from bytes |

## 5. Tests

`tests/vllm/models/test_qwen3_8_text_only.cpp`, which already builds synthetic
safetensors and drives the PRODUCTION entry point `vllm::LoadQwen3_5Moe`. The
file is registered at `tests/CMakeLists.txt:476`, so no registration edit is
owed and `scripts/check-test-registration.py` stays satisfied.

RED first, and each case must fail for its own reason:

1. a per-output-channel `weight_scale` (`{kMoeQ}` `F32`) on `q_proj` is refused,
   and the message names the tensor, its shape and its element count.
2. a one-element `BF16` `weight_scale` is refused, and the message names the
   tensor and `BF16`.
3. a multi-element `weight_scale_2` on a routed expert is refused, which reaches
   `LoadNvfp4Raw` rather than the FP8 loaders.
4. positive control: the unmodified fixture, one element and `F32`, still loads.

A happy-path case cannot detect either defect, because both defects ARE the
happy path. Case 4 exists only to prove the guard did not refuse everything.

Reachability, per [`reachability.md`](../reachability.md): the guard is reached
from `LoadQwen3_5Moe`, not from a hand-built `StTensor`. Deleting the production
call site in a scratch copy must turn cases 1 to 3 red.

## 6. Gates

`scripts/agent-preflight.sh --fail-on-skip`, plus the focused
`test_qwen3_8_text_only` run recorded with its case and assertion counts. CPU
only. No GPU lease, no checkpoint download.

## 7. Risks and decisions

- **A currently loading checkpoint could start refusing.** Intended, and the
  only shape that can do it is the one in finding B, which is loading with a
  wrong scale today. A named refusal is strictly better than fluent wrong
  tokens, and it is the same trade #1166 made. If a GPU run of
  `unsloth/Qwen3.6-27B-NVFP4` @`ccdaab7e` starts refusing after this lands, that
  refusal is the DISCOVERY, not the regression, and it belongs to the owed
  per-channel FP8 arm.
- **Six copies into one changes six messages.** No test, document or spec
  asserts any of the five deleted strings, verified by grep over `tests/`,
  `docs/` and `.agents/`.
- **A shared helper in a header the SACRED 27B TU now includes.** The
  qwen3_5_weights.cpp include is new. It adds no symbol that collides: the
  file's own anonymous-namespace `MakeOwned` and `TransposeBf16` stay
  unqualified and keep winning, since the shared ones need
  `dense_loaders::`.

## Owed

- Per-output-channel FP8 on the per-tensor arm, for
  `unsloth/Qwen3.6-27B-NVFP4` @`ccdaab7e` and any republish like it. Needs its
  own row, a GPU gate and a scale-aware `Fp8Weight`.
- Block-wise FP8, still owed by
  [`fp8-blockwise-refusal.md`](fp8-blockwise-refusal.md) and #1166.
- Explicit narrow-dtype conversion for a one-element `BF16` or `F16` scale, if
  a checkpoint is ever measured that ships one.

## Outcome

Measured on the CPU gate at `ab6e65216` plus this branch, `x86_64`, Ninja,
default `CMAKE_BUILD_TYPE`.

### RED before, at commit `8cff76113`

`test_qwen3_8_text_only`, one case, 29 assertions: **16 failed, 6 of 6 refusal
subcases failed**, the positive control passed.

The SHAPE of that red is the finding, and it is stronger than the issue
predicted. Eleven of the sixteen failed assertions logged `message := ` with
nothing after it. A per-output-channel `weight_scale [8] F32`, a block-wise
grid `[2, 4] F32`, a per-output-channel `[8] BF16` and a multi-element
`weight_scale_2 [2] F32` did not merely read the wrong element. They LOADED, and
`LoadQwen3_5Moe` returned a complete model built on a number nobody wrote.

The two one-element BF16 cases did stop, on
`vt: qwen3_5 weights: scalar tensor too small for f32 at
src/vllm/model_executor/models/qwen3_5_weights.cpp:313`. That message names no
tensor and describes a truncation rather than a dtype, so it cannot tell a
reader which of hundreds of scales was wrong. And it only fires because ONE bf16
value is two bytes. Eight bf16 values are sixteen bytes, which is the layout
that is actually published, and the floor never saw it.

### GREEN after, at commit `60aefbca6`

- focused: `test_qwen3_8_text_only -tc="...REFUSED by name"`, 1 case, **29 of 29
  assertions**, `SUCCESS`
- whole file: 19 cases, **67,845 of 67,845 assertions**, `SUCCESS`
- full suite: `ctest --test-dir build -j 2`, **511 of 511 passed, 0 failed**,
  181 s, with `test_modelopt_mixed_precision_checkpoint` and `test_voxtral_e2e`
  skipped for missing checkpoints as they are on an unmodified tree
- `scripts/agent-preflight.sh --fail-on-skip`: 81 gates, **81 ok, 0 FAIL, 0
  SKIP**

### Mutations, each with its build status and its applied diff

Run in place against a clean tree, restored with `git checkout --` and verified
by `sha256sum`. A mutation that fails to build and a mutation that never applied
both read as a passing test, so both are printed rather than assumed.

| Mutation | diff stat | `compile_rc` | Result | Restore |
|---|---|---|---|---|
| R1 reachability: `LoadFp8Transposed`'s call site replaced by `1.0F` | 1 file, +1 -1 | 0 | RED, 11 of 29, the 4 `weight_scale` subcases | byte-identical |
| R2 reachability: `LoadNvfp4Raw`'s call site replaced by `1.0F` | 1 file, +1 -1 | 0 | RED, 5 of 29, the 2 `weight_scale_2` subcases | byte-identical |
| G1: the element-count check deleted | 1 file, +1 -4 | 0 | RED, 7 of 29, the 4 multi-element subcases | byte-identical |
| G2: the dtype check deleted | 1 file, -3 | 0 | RED, 2 of 29, the 2 one-element BF16 subcases | byte-identical |
| G3: the exact-byte-count check deleted | 1 file, -3 | 0 | **GREEN, 29 of 29** | byte-identical |

R1 is the reachability proof the `Nothing lands dead` rule asks for. The guard
is reached from `vllm::LoadQwen3_5Moe` over a real synthetic checkpoint, and
`LoadFp8Transposed` rather than `LoadFp8Raw` is the arm a CPU build takes,
because `DenseNativeEnabled()` needs `VT_CUTLASS_FP8`. R2 proves the same
through a second, unrelated loader, which is what makes the guard shared rather
than local.

### The negative result, recorded rather than quietly kept

**G3 stayed green, so the exact-byte-count check is NOT independently gated.**
No fixture built by `BuildSafetensors` can reach it, because the safetensors
reader already rejects a header whose shape times dtype width does not equal its
data span, so `numel == 1` and `dtype == "F32"` together imply four bytes for
any file that parses. What the check still buys is the `t.data != nullptr` half,
against a default-constructed `StTensor`, which
`safetensors_reader.h:20-33` documents as a real state in tests. It is kept
because the in-tree reference `nemotron_h_weights.cpp:566-568` carries the same
check, and because a `memcpy` from `nullptr` is worse than a refusal. It is
recorded here as an ungated line rather than presented as a tested one.

### What surprised us

The issue framed this as latent, protected by #1166's name miss. The audit says
otherwise on two counts, and both are in the tree already rather than inferred.
`dense_weight_loaders.h:73-74` and `docs/BENCHMARKS.md:52` both record
`unsloth/Qwen3.6-27B-NVFP4` @`ccdaab7e` as FP8 W8A8 throughout with BF16
per-output-channel scales, and `LoadAttnDense` branches on the weight dtype
alone, so that revision reaches both defects at once under the tensor name the
loader asked for. Nothing was misspelled and nothing stopped it.

And the reader had SIX copies, not five. `ReadCtF32Scalar` carried the identical
defect under a different name, inside the shared header that exists to stop
exactly that, and it was reached from a sixth model file the issue's file list
did not name. A grep for one identifier measured five sixths of a class defect.

## Now

`ACTIVE`, PR open, awaiting fresh review. Spec committed before implementation
at `30f704f6c`, red tests at `8cff76113`, guard at `60aefbca6`.
