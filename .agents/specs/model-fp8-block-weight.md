# MODEL-FP8-BLOCK-WEIGHT — the block-wise FP8 weight, its loader rung, and its config reader

Issue: [#1189](https://github.com/mudler/vllm.cpp/issues/1189), milestone **M3**.
Row: `MODEL-FP8-BLOCK-WEIGHT`.
Pinned oracle: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`
(`.agents/upstream-sync.md`), asserted as the HEAD of the local checkout before
any `file:line` below was read.

## Scope

Three things, and the refusal they replace.

1. **`Fp8BlockWeight`**, a sibling of `Fp8Weight` in
   `include/vllm/model_executor/models/qwen3_5_weights.h`.
2. **The loader rung**: a probe on `<proj>.weight_scale_inv` inserted *before*
   the existing `dtype == "F8_E4M3"` per-tensor rung in
   `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`, plus the shared
   `dense_loaders::LoadFp8BlockRaw` it calls.
3. **The config reader**, `ReadFp8BlockQuantConfig`, which reads
   `weight_block_size`, `activation_scheme` and `modules_to_not_convert`, and
   validates them against upstream's own rules before any tensor is touched.

`RefuseUnsupportedFp8BlockQuant` (landed `469f38395`, #1166) stops refusing the
whole scheme and refuses only what M3 does not cover. The one thing M3 still
cannot do — *consume* the weight — is refused by name at
`ModelRegistry::Prepare`, not left to produce wrong numbers.

**Out of scope, each owned by a later milestone of #1189**:
`layers::Fp8BlockLinearMethod` and the Qwen3.5 dense forward wiring (M4); the
mainloop-scaled CUTLASS kernel for `sm_121a` (M5); merged `gate_up` and QKV
(M6). No kernel lands here, no GPU is leased, and no checkpoint is downloaded.

## What the target checkpoint actually is

`Qwen/Qwen3.8-27B-FP8` at revision `017b9c7a`, measured by HTTP range request
for #1166 and not re-fetched here:

| Fact | Value |
|---|---|
| `quant_method` | `fp8` |
| `weight_block_size` | `[128, 128]` |
| `activation_scheme` | `dynamic` |
| architecture | `Qwen3_5ForConditionalGeneration` |
| `self_attn.q_proj.weight` | `F8_E4M3` `[12288, 5120]` |
| `self_attn.q_proj.weight_scale_inv` | **`BF16`** `[96, 40]` = `[12288/128, 5120/128]` |
| `input_scale` tensors | **0** |

Every projection in this checkpoint divides by 128 on both axes. That is a
property of this checkpoint, not of the scheme, and the loader is written and
tested against `cdiv` on both axes for that reason.

## The BF16 scale dtype, established rather than assumed

The scale ships `BF16` and upstream allocates the parameter `float32`. Both are
true, and the resolution is that **torch's `copy_` converts**:

| Step | Where |
|---|---|
| the block scale parameter is allocated `[cdiv(N,block_n), cdiv(K,block_k)]` with `dtype = scale_dtype if scale_dtype is not None else torch.float32` | `vllm/model_executor/layers/quantization/utils/fp8_utils.py:1276,1283-1296` |
| `scale_dtype` is `torch.float8_e8m0fnu if self.is_scale_e8m0 else None` | `fp8.py:376` |
| `is_scale_e8m0` is `getattr(quant_config, "is_scale_e8m0", False)`, and `Fp8Config` defines no such attribute (`fp8.py:96-134`), so it is **False** and the parameter is `float32` | `fp8.py:282` |
| the checkpoint tensor is written into that parameter by `self.data.copy_(loaded_weight)`, a **dtype-converting** copy, not a reinterpretation | `vllm/model_executor/parameter.py:95-108`; `BlockQuantScaleParameter` inherits it at `:397-403` |
| downstream code then asserts the scale is `float32` (or E8M0/uint8 for MXFP8), which is only consistent because the widening already happened at load | `fp8_utils.py:1103-1112` |

So upstream's answer to a narrower on-disk scale dtype is: **widen it to f32 at
load, once, losslessly.** `bf16 -> f32` is exact — bf16 is the top 16 bits of an
f32 — so nothing is invented and nothing is lost.

We mirror that. `Fp8BlockWeight::scale` is **f32**, and this is not a
`.agents/porting.md` dtype widening to be justified: f32 *is* the resident dtype
upstream carries. The bytes are negligible (`[96,40]` f32 is 15 360 B per
projection against 60 MiB of weight) and `vt::MatmulFp8BlockScaled` refuses any
scale that is not f32 (`.agents/specs/vt-matmul-fp8-block-ref.md`), so a bf16
resident scale would need a conversion at every GEMM instead of one at load.

**What we must not do**, and the reason the rule exists: `f22c6cc82` (#1181)
landed a guard because six copies of the per-tensor scale reader bounded their
input with `nbytes >= sizeof(float)` — a floor — and then `memcpy`d four bytes
whatever the dtype was. `LoadFp8BlockRaw` therefore switches on `t.dtype`
explicitly, decodes `BF16` through `vt::BF16ToF32(vt::LoadUnaligned<uint16_t>)`
and `F32` through `vt::LoadUnaligned<float>`, and **refuses every other dtype by
name**. There is no default branch that reinterprets bytes.

`vt::LoadUnaligned` rather than a typed pointer, because a safetensors tensor's
offset is the running byte total of everything ahead of it and can be odd
(#627), exactly as `dense_loaders::ReadF32Scalar` and `TransposeBf16` already
do.

## Upstream anchors

| What | Where |
|---|---|
| `weight_block_size`, `activation_scheme`, `ignored_layers`, and the `modules_to_not_convert` fallback are read together | `vllm/model_executor/layers/quantization/fp8.py:157-172` |
| the validation this mirrors: fp8-serialized, exactly 2 dimensions, `dynamic` only | `fp8.py:115-131` |
| `block_quant = self.weight_block_size is not None` is the whole dispatch | `fp8.py:297-298` |
| the scale registers as `weight_scale_inv`, strictly conditional on block quant | `fp8.py:378-379`, `:511` |
| scale allocation, `cdiv` on **both** axes | `fp8_utils.py:1283-1296` |
| the shape assertion the load performs | `vllm/model_executor/parameter.py:95-98` |
| `is_layer_skipped`: default `prefix_full_match`, i.e. exact membership of the module prefix in the ignore list | `vllm/model_executor/layers/quantization/utils/quant_utils.py:510-524,568-569` |
| the ignore list is rewritten into vLLM module naming before matching | `fp8.py:151-153` (`apply_vllm_mapper`) |
| the activation quant this pairs with (M1, landed `ad5f175e7`) | `.agents/specs/vt-quant-fp8-group.md` |
| the GEMM this pairs with (M2, landed `770e49486`) | `.agents/specs/vt-matmul-fp8-block-ref.md` |

## Design

### `Fp8BlockWeight` is a sibling, not an extension

```c++
struct Fp8BlockWeight {
  OwnedTensor packed;   // i8  [N, K]                    raw fp8-e4m3fn bytes
  OwnedTensor scale;    // f32 [cdiv(N,bn), cdiv(K,bk)]  widened from disk
  int64_t n = 0, k = 0;
  int64_t block_n = 0, block_k = 0;
  bool Empty() const { return packed.Empty(); }
  mutable std::shared_ptr<void> d_packed;
  mutable std::shared_ptr<void> d_scale;
};
```

`Fp8Weight` (`qwen3_5_weights.h:563-576`) is three host floats — `weight_scale`,
`input_scale`, and the `alpha = input_scale * weight_scale` folded at load. A
block scheme has **no `input_scale` at all** (the activation scheme is dynamic;
the target checkpoint ships zero such tensors) and its weight scale is a 2-D
tensor. There is no value `alpha` could take.

Adding an optional scale tensor to `Fp8Weight` was rejected. Every existing
reader of `Fp8Weight::alpha` — the cutlass and cuBLASLt fp8 GEMM wrappers, the
merged-QKV alpha vector, `PrepareGdnFp8Resident` — would then need a silent
which-arm branch, and the arm that forgets one produces a plausible number
rather than an error. A distinct type makes the wrong call site fail to compile.

`Nvfp4Weight` (`qwen3_5_weights.h:243-304`) is the shape that already works
here: an `OwnedTensor scale` beside the packed values, plus lazily-populated
device handles. `Fp8BlockWeight` follows it, minus everything NVFP4-specific.

`block_n` and `block_k` are carried **on the weight**, not looked up from the
config at use time. The consumer needs them per GEMM, and a weight that knows
its own geometry cannot be paired with the wrong one.

### The loader rung

`load_projection` in `LoadAttnDense` (`qwen3_5_dense_weights.cpp:471-480`)
probes NVFP4, then `dtype == "F8_E4M3"`, then bf16. A block-wise weight **is**
`F8_E4M3`, so it fell into the per-tensor arm and asked for a `weight_scale`
that a block-wise checkpoint spells `weight_scale_inv`. That is #1166.

The block probe goes **before** the per-tensor rung, at every site that probes
`F8_E4M3` today: `LoadAttnDense` (q/k/v/o), `LoadGdnDense`
(`in_proj_qkv`, `in_proj_z`, `out_proj`), and `LoadDenseMlp`, which had no fp8
rung at all and would otherwise have sent a block-wise MLP into
`LoadMergedBf16RawNK`.

### Read the config, then cross-check it against the tensors

A dtype probe alone is not enough, for two measured reasons.

**`modules_to_not_convert` is a ~400-entry list** that a probe reproduces only
by accident. A projection this checkpoint deliberately left unquantized is
`BF16` on disk and a probe agrees with the config by luck; the moment a
checkpoint ships an `F8_E4M3` tensor for a module it also lists as excluded, the
probe and the config disagree and only one of them is right.

**A probe cannot see a disagreement at all.** It sees a tensor and picks an arm.
That is precisely where a silent-wrong-scale bug lives, and #1166's own commit
message records the near miss: a `[96,40]` scale passed the old `nbytes >=
sizeof(float)` floor and would have been read as element `(0,0)` and applied to
the whole `[N,K]` weight. Only the tensor *name* stopped it.

So `ResolveFp8Arm` decides the arm from **both** sources and refuses each
disagreement by name:

| Config says | Tensors say | Result |
|---|---|---|
| block-wise, module not excluded | `F8_E4M3` + `weight_scale_inv` | block arm |
| block-wise, module not excluded | `F8_E4M3`, no `weight_scale_inv` | **refuse**, naming the tensor that is missing |
| block-wise, module **excluded** | `weight_scale_inv` present | **refuse**, naming the module and the list |
| block-wise, module excluded | no `weight_scale_inv` | falls through to the existing rungs |
| not block-wise | `weight_scale_inv` present | **refuse**, naming the config key that is missing |
| not block-wise | no `weight_scale_inv` | falls through, byte-identical to today |

Module exclusion mirrors `is_layer_skipped`'s default `prefix_full_match`: the
tensor name with its `.weight` suffix removed, compared for **exact equality**
against each list entry. Upstream first rewrites the list into vLLM module
naming (`fp8.py:151-153`); we match in *checkpoint* naming, which is what our
loader has, and the two coincide for every entry that names a real checkpoint
module. The substring form (`skip_with_substr=True`) is not the default at any
fp8 call site and is not mirrored.

The shape cross-check lives in `LoadFp8BlockRaw` and is upstream's own: the
scale must be exactly `[cdiv(N, block_n), cdiv(K, block_k)]`, mirroring the
allocation at `fp8_utils.py:1283-1296` and the `param.data.shape ==
loaded_weight.shape` assertion at `parameter.py:95-98`. A floor-sized scale, a
transposed scale, and a per-tensor scalar are each refused with both shapes in
the message.

An `<proj>.input_scale` present while the config declares `dynamic` is also a
disagreement, and it is refused: upstream registers an input scale only when
`act_q_static` (`fp8.py:381-384`), which block quant asserts against outright
(`fp8.py:367`).

### Ragged edges are supported, not refused

`cdiv` on both axes, everywhere. Upstream's shape contract admits a short final
block (`fp8_utils.py:935-936`) and M2's reference arm already handles one. The
target checkpoint has none, so the tests carry `N=576` (`4*128 + 64`) and
`K=3884` (`30*128 + 44`) — upstream's own ragged shapes from
`tests/kernels/quantization/test_block_fp8.py:49-50` — separately and together.
M2 measured that a grid of round shapes stays green through two different
floor-vs-ceil defects; that measurement is the reason these shapes are here.

### What is still refused, and where

| Refused | Where | Why |
|---|---|---|
| `activation_scheme != "dynamic"` | `ReadFp8BlockQuantConfig`, reached from `ModelRegistry::Load` | upstream refuses it too (`fp8.py:127-131`); nothing here quantizes activations statically for a block scheme |
| `weight_block_size` with other than 2 dimensions | same | upstream refuses it (`fp8.py:121-126`) |
| a block shape other than `[128, 128]` | same | M5's kernel is 128x128 and M2's reference is the only other consumer; a `[64, 128]` checkpoint would load into a weight nothing can execute |
| `quant_method` without `fp8` | same | mirrors `is_checkpoint_fp8_serialized` (`fp8.py:117-120`, `:159`) |
| a scale dtype that is neither `BF16` nor `F32` | `LoadFp8BlockRaw` | the #1181 rule: never reinterpret bytes across a dtype |
| a config/tensor disagreement | `ResolveFp8Arm` / `LoadFp8BlockRaw` | the table above |
| **a loaded block weight that nothing consumes** | `PrepareQwen3_5Dense`, reached from `ModelRegistry::Prepare` | M4 owns the linear method. See below |

The last row is the M3/M4 seam and it is deliberate. `ModelRegistry::Load` now
*succeeds* on a supported block-wise checkpoint, which is what makes the loader
rung reachable from a production entry point at this merge commit. Nothing
consumes an `Fp8BlockWeight` yet, so the dense `project` lambda
(`qwen3_5.cpp:2464-2486`) would fall through to an empty bf16 tensor. Rather
than let that happen, `ModelRegistry::Prepare` — which every runner calls
unconditionally before the first forward and before graph capture
(`src/vllm/v1/worker/gpu/runner.cpp:414,455`) — refuses by name, quotes the
projection, and names #1189 M4. A block-wise checkpoint therefore loads and then
declines to run, and it never runs wrong.

## Risks

| Risk | Control |
|---|---|
| the BF16 scale is reinterpreted as f32 rather than converted | `LoadFp8BlockRaw` switches on dtype with no default branch; G2 asserts the decoded VALUES against a hand-computed table, which a reinterpretation cannot pass |
| the scale silently widens or narrows | `Fp8BlockWeight::scale` is f32 by construction and G2 asserts the dtype; the reason it is f32 is recorded above with the upstream anchor |
| a ragged dimension uses floor tiling and misindexes or drops a block | `cdiv` everywhere; G3 runs `N=576`, `K=3884`, and both together |
| a config/tensor disagreement picks an arm silently | G4 asserts four distinct disagreements, each refused by name |
| the block rung is inserted after the per-tensor rung and never selected | G1 loads a block-wise checkpoint through the production loader and asserts the block slot is populated and the per-tensor slot is empty |
| a per-tensor checkpoint regresses into the block arm | G6 is the negative control: a per-tensor fixture still lands in `Fp8Weight` with its `alpha` folded, and the existing `test_fp8_block_quant` cases are retained |
| an unsupported block shape or scheme loads silently | G5 asserts each refusal by name through `ModelRegistry::Load` |
| **nothing consumes the weight, so the whole rung is dead** | acknowledged under `## Owed`; the M4 gap is refused by name at `Prepare` and G7 asserts that refusal |

## Tests

`tests/vllm/model_executor/models/test_fp8_block_weight_load.cpp`, registered in
`tests/CMakeLists.txt`. A synthetic safetensors fixture is written to a temp
directory — no checkpoint download, no GPU, no snapshot. The fixture builder
follows `tests/vllm/test_safetensors.cpp:57-89` (u64-LE header length + JSON
header + payload).

The fixture is a **complete, minimal** `Qwen3_5ForConditionalGeneration` dense
checkpoint: one `full_attention` layer, tiny dimensions, tied `lm_head`. That is
what lets every case enter through the production loader rather than through
`LoadFp8BlockRaw`.

- **G1** the rung is selected: a block-wise fixture loaded through
  `LoadQwen3_5Dense` — the loader `ModelRegistry::Load` reaches at
  `qwen3_5_dense.cpp:101` — populates `q_proj_fp8_block` and leaves
  `q_proj_fp8` and `q_proj` empty, for attn, GDN and MLP projections alike.
- **G2** the scale tensor: dtype f32, shape `[cdiv(N,128), cdiv(K,128)]`, and
  **values** equal to the BF16 bytes written into the fixture, decoded
  independently. The fixture writes a value that is exactly representable in
  bf16 and one that is not, so a reinterpretation reads as a wildly wrong
  number rather than a rounding difference. An `F32` scale fixture is loaded in
  the same case, and a `F16` one is refused by name.
- **G3** the ragged grid: `N=576`, `K=3884`, and both together, with the scale
  sized by `cdiv`. A floor-sized scale for the same weight is refused.
- **G4** the config/tensor disagreements, each by name: `weight_scale_inv`
  present with no `weight_block_size` in the config; `weight_block_size`
  declared with no `weight_scale_inv` beside an `F8_E4M3` weight; a module named
  in `modules_to_not_convert` that nevertheless ships a `weight_scale_inv`; and
  an `input_scale` present under `activation_scheme = dynamic`.
- **G5** the config refusals through `ModelRegistry::Load`, each by name:
  `activation_scheme = "static"`, `weight_block_size = [128]`,
  `weight_block_size = [64, 128]`, and a `quant_method` that is not fp8.
- **G6** the negative controls: a per-tensor fp8 fixture still loads into
  `Fp8Weight` with `alpha = input_scale * weight_scale`, and a bf16 fixture is
  untouched. Without this the gate passes for a rung that captures every fp8
  checkpoint.
- **G7** the M4 gap: `ModelRegistry::Prepare` on a loaded block-wise model
  refuses by name, quoting the projection and #1189.

`tests/vllm/model_executor/layers/test_fp8_block_quant.cpp` is updated in the
same change: the cases that asserted the whole scheme was refused now assert the
narrowed refusals, and the negative controls are kept verbatim.

## Gates

| Gate | Command |
|---|---|
| focused | `ctest -R test_fp8_block_weight_load --output-on-failure` |
| the narrowed refusal | `ctest -R test_fp8_block_quant --output-on-failure` |
| the M1/M2 siblings, unchanged | `ctest -R "test_ops_quant_fp8_group_cpu\|test_ops_matmul_fp8_block_cpu"` |
| the per-tensor arm, unchanged | `ctest -R "test_ops_fp8_cpu\|test_qwen36_weights\|test_linear_method"` |
| record | `scripts/agent-preflight.sh --fail-on-skip` |

No GPU lease is taken and none is needed.

## Owed

- **Nothing consumes `Fp8BlockWeight`.** At this merge commit the loader
  populates it and no forward path reads it: `include/vllm.h` exposes no block
  linear method, `layers::MakeLinearMethod` has no block arm, and the dense
  `project` lambda (`src/vllm/model_executor/models/qwen3_5.cpp:2464-2486`)
  knows only fp4, per-tensor fp8 and bf16. Owed by **#1189 milestone M4**
  (`layers::Fp8BlockLinearMethod` and the Qwen3.5 dense forward wiring). This is
  the staged-slice exception of `.agents/reachability.md`, named here, in the
  commit body, and in the pull request body. It is refused by name at
  `ModelRegistry::Prepare` rather than left to produce a wrong number, and G7
  asserts that refusal.
- **The device handles `d_packed` and `d_scale` are declared and never
  populated.** They mirror `Nvfp4Weight`'s and `Fp8Weight`'s shape so that M4
  and M5 upload through the same lazily-populated seam every other quantized
  weight uses. Owed by #1189 M5, which is the first arm with a device kernel.
- **Merged `gate_up` and QKV.** The MLP rung loads gate, up and down as three
  independent block weights. Block scales concatenate losslessly along N, so
  merging is *simpler* here than in the per-tensor case, and #1189 M6 owns it.
- **The MoE and GGUF loaders.** `qwen3_5_weights.cpp`'s MoE path and
  `qwen3_5_gguf_weights.cpp` are untouched: no block-wise MoE or GGUF checkpoint
  is in play for #1189, and `ReadFp8BlockQuantConfig` refuses an unsupported
  block config for every architecture at `ModelRegistry::Load` regardless. A
  block-wise MoE safetensors checkpoint would reach the MoE loader's per-tensor
  rung and fail on the missing `weight_scale` name, which is #1166's original
  sentence and is not made worse here. Owed by whichever row ports one.
- **`store_dtype`** (`fp8.py:104`, `:167`) is read by upstream and ignored here.
  No checkpoint in play sets it. It becomes owed when one does.

## Stop conditions

Stop and report `NEEDS_DECISION` if any of the following holds.

- The pinned oracle's checkout is not at
  `5559679229bc961848b121ccdeaa8fa5d79bec98`. Every anchor above was read at
  that revision, asserted before the first read.
- The BF16 scale cannot be shown to widen rather than reinterpret. A byte
  reinterpretation that happens to pass a shape check is the #1181 defect and
  widening a tolerance is not the fix.
- The loader rung cannot be reached from `ModelRegistry::Load` without wiring a
  forward path, which is M4's scope. If reaching it requires M4, the milestone
  boundary is wrong and the operator decides where to move it.

Stop and report `NEEDS_CONTEXT` if the work requires a GPU lease or a checkpoint
download. The row is scoped so that it needs neither.

## Evidence

Taken on the merged tree, `origin/main` at `65d6cdaed`. No GPU lease, no
checkpoint download, no snapshot.

**RED.** With both test files present and no implementation, the focused build
fails, `compile_rc=1`, with **50 errors**: 6 ``'Fp8BlockWeight' does not name a
type``, 4 + 2 + 1 + 1 ``has no member named 'q_proj_fp8_block'`` /
`'o_proj_fp8_block'` / `'v_proj_fp8_block'` / `'k_proj_fp8_block'` on
`FullAttnLayerWeights`, 3 more on `DenseMlpWeights`, and the cascade of
undeclared locals that follows. `test_fp8_block_quant` compiled but reported
**4 cases, 2 failed, 19 assertions, 9 failed** against the narrowed refusal,
which is the same red seen from the other side.

**GREEN.** `test_fp8_block_weight_load` reports **7 cases, 102 assertions, 0
failed**; `test_fp8_block_quant` reports **8 cases, 35 assertions, 0 failed**.
Per block, each run through a `-tc` prefix filter that contains no comma,
because doctest splits `-tc` on commas and a name that contains one yields
`0 cases ran` under a `SUCCESS!` banner:

| Block | Cases | Assertions |
|---|---:|---:|
| G1 the rung is selected | 1 | 24 |
| G2 the scale is widened not reinterpreted | 1 | 17 |
| G3 the ragged grid | 1 | 11 |
| G4 the config/tensor disagreements | 1 | 19 |
| G5 the unsupported configs | 1 | 17 |
| G6 the per-tensor and bf16 negative controls | 1 | 9 |
| G7 the M4 gap at `Prepare` | 1 | 5 |
| **sum** | **7** | **102** |

The buckets sum to the whole-run count, so no block is silently empty and no
filter selected nothing.

The other declared gates on the same tree, all passing:
`test_ops_quant_fp8_group_cpu` (M1 unchanged), `test_ops_matmul_fp8_block_cpu`
(M2 unchanged), `test_ops_fp8_cpu`, `test_qwen36_weights`, `test_linear_method`,
`test_model_registry`, `test_op_provider`. The whole tree builds clean, 609
targets, `build_rc=0`.

### The reachability mutation

`.agents/reachability.md`: delete the **production call site**, not the
implementation, and rerun the focused gate.

The chain is `ModelRegistry::Load` -> `LoadQwen3_5DenseModel`
(`qwen3_5_dense.cpp:101`) -> `LoadQwen3_5Dense` -> `LoadQwen3_5DenseLayer` ->
`LoadAttnDense`'s `load_projection`. Deleting the `LoadFp8BlockRaw` call there
reds **6 of 7 cases**, and the sentence it reds with is issue #1166 verbatim:

```text
vt: qwen3_5 dense: tensor not found: model.layers.0.self_attn.q_proj.weight_scale
```

That is the whole point of the rung. Without it the block-wise projection falls
into the per-tensor arm and asks for a tensor the checkpoint spells
`weight_scale_inv`.

### Mutation results

Every mutation printed `git diff --stat` **and** `compile_rc` before the run,
because a mutation that fails to build and a mutation that never applied both
read as a passing test, and M1 and M2 of #1189 each hit one. Each was restored
with `git checkout -- .` and the restore verified by comparing
`git ls-files -s | sha256sum` before and after; every row below restored to
`fdb713f0...`.

| Mutation | `compile_rc` | Result |
|---|---|---|
| the attn block rung call site deleted | **1** | proves nothing: `-Werror=unused-parameter` on `block` |
| the same with `block` kept live | 0 | **6 of 7 cases fail**, 10 assertions; the failure is `tensor not found: ...q_proj.weight_scale` |
| the block rung moved AFTER the per-tensor rung, i.e. the #1166 ordering | 0 | 6 of 7 cases fail, 10 assertions, same sentence. Ordering is load-bearing and measured |
| the BF16 scale `memcpy`'d instead of converted, i.e. the #1181 defect | 0 | 2 cases fail, 5 assertions, all in G2 |
| the scale-dtype refusal widened to a silent f32 read for any dtype | 0 | 1 case fails, 1 assertion (G2's F16 refusal) |
| `cdiv` replaced by floor on both axes | 0 | **only G3 fails**, and by THROWING: the run reports 91 assertions instead of 102 with `Status: FAILURE!` and `run_rc=1`. G1, G2 and G6 stay green, because every dimension in them is a multiple of 128 |
| the config/tensor cross-check reduced to a probe | 0 | **only G4 fails**, 5 assertions |
| the 128x128 and `dynamic` refusals removed | 0 | G5 fails 1 assertion and `test_fp8_block_quant` fails **3 of 8 cases, 12 assertions** |
| the `Prepare` refusal's `is_linear_attention` branch flipped to `if (false)` | 0 | **everything stays green** — see below |
| the `Prepare` call site deleted | 0 | 1 case fails (G7), 1 assertion |
| the packed fp8 bytes zeroed instead of copied | 0 | 1 case fails, 1 assertion (G1's verbatim-bytes check) |

Three results are worth keeping.

**A mutation that applies cleanly can still change nothing.** Guarding the
`if (layer.is_linear_attention)` arm with `if (false)` looked like deleting the
refusal. It took the `else` arm instead, which runs every attention check, and
the MLP checks sit outside the branch entirely — so the refusal still fired and
the gate stayed green over a two-line diff that `git diff --stat` happily
reported. The verdict was re-taken against the actual call site, where it reds.
`compile_rc` and `git diff --stat` do not catch this class; only asking what the
mutated code now does catches it.

**`-Werror` turns the natural reachability mutation into a non-event.** Deleting
the rung call orphans the `block` parameter, `-Werror=unused-parameter` fires,
the build fails, and the STALE binary from the previous link then prints
`SUCCESS!`. That is a green over a mutation that never ran. The re-run keeps
`block.block_n` live.

**A grid of round shapes is blind to the ragged defect.** Replacing `cdiv` with
floor left G1, G2, G4, G5, G6 and both of `test_fp8_block_quant`'s halves
entirely green and failed only G3, which is why `N=576` and `K=3884` are in the
grid rather than the shapes the target checkpoint happens to use — all of which
are multiples of 128. This is the same measurement M2 recorded for its own
kernel, reproduced one layer up.

### The known-red gate on this host

`scripts/agent-preflight.sh --staged --fail-on-skip` reports
`test_cpu_x86_llamacpp_floor` FAIL:
`NO_QUIET_WINDOW after 30s (busy=114% builders=0 load=134.83 146.63 122.42)`,
the harness exiting 4 where the case expects 2. That is
[#618](https://github.com/mudler/vllm.cpp/issues/618): the harness refuses to
measure while the box is loaded, and the refusal itself reads as a defect in
whatever diff is in flight.

It is a property of the host and not of this change, and that was **measured
rather than argued**. A pristine detached worktree at `origin/main`
`65d6cdaed`, with no part of this row applied, ran the same suite: `Ran 10
tests`, `FAILED (failures=2)`, both `NO_QUIET_WINDOW after 30s` at `busy=154%`
and `busy=177%`, `builders=0`, 1-minute load 50 to 60. The worktree was removed
afterwards. Every other preflight gate is `ok`.

## Now

`ACTIVE` — M3 of #1189. M1 (`ad5f175e7`) and M2 (`770e49486`) are `DONE`; M4,
M5 and M6 are open.
