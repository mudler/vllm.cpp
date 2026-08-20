# Refuse block-wise FP8 by name

Row `FIX-FP8-BLOCKWISE-REFUSAL`. Issue
[#1166](https://github.com/mudler/vllm.cpp/issues/1166).

## Scope

Detect a block-wise (fine-grained) FP8 checkpoint at load and refuse it with a
message that names `weight_block_size`, states that block-wise FP8 is not
implemented, and names issue #1166.

Out of scope, and recorded under `## Owed`: reading `weight_scale_inv`,
dequantizing a 128x128 block scale, and the block-wise GEMM. That work needs a
GPU gate and its own row.

## 0. What is wrong today, measured

`Qwen/Qwen3.8-27B-FP8` at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`,
read over HTTP on 2026-08-17. The config declares `quant_method` `fp8`,
`weight_block_size` `[128, 128]`, and `activation_scheme` `dynamic`, with
`quantization_config` at the top level and none under `text_config`.

The safetensors header of `layers-3.safetensors`, read by range request, gives
`self_attn.q_proj.weight` as `F8_E4M3` `[12288, 5120]` and
`self_attn.q_proj.weight_scale_inv` as `BF16` `[96, 40]`. `[96, 40]` equals
`[12288/128, 5120/128]`, so the scale carries one value for each 128x128 block.
The shard holds zero `input_scale` tensors, which is what the dynamic activation
scheme means.

This tree stops on that checkpoint, so the defect is not wrong numerics. It
stops on the wrong sentence. `LoadAttnDense` branches on the weight dtype alone
(`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:480`), so an
`F8_E4M3` block-wise projection enters the per-tensor arm at `:480`, and
`LoadFp8Raw` (`src/vllm/model_executor/models/qwen3_5_weights.cpp:449`) asks for
`<proj>.weight_scale` at `:458`. The checkpoint spells that tensor
`weight_scale_inv`, so the resolver lambda at `qwen3_5_dense_weights.cpp:683`
raises

```
qwen3_5 dense: tensor not found: model.language_model.layers.N.self_attn.q_proj.weight_scale
```

Nothing is missing from the checkpoint. The reader is sent to look for a tensor
that upstream never writes in this mode, instead of being told that the
fine-grained arm is absent.

`git grep weight_block_size -- src/ include/` returns nothing.
`git grep weight_scale_inv` returns one hit in an unrelated name predicate
(`src/vllm/model_executor/models/minimax_music3_quant.cpp:276`) and no reader in
any load path.

## 1. What upstream does, with anchors

Pinned vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`, the parity pin in
[`upstream-sync.md`](../upstream-sync.md), verified with `git rev-parse` in the
oracle checkout before citing.

| Anchor | What it does |
|---|---|
| `vllm/model_executor/layers/quantization/fp8.py:161` | `Fp8Config.from_config` reads `weight_block_size` out of the quantization config |
| `vllm/model_executor/layers/quantization/fp8.py:115-132` | `Fp8Config.__init__` validates it. It requires an fp8-serialized checkpoint, exactly 2 dimensions, and a dynamic activation scheme |
| `vllm/model_executor/layers/quantization/fp8.py:297-298` | `Fp8LinearMethod` sets `self.block_quant = self.weight_block_size is not None`, which is the dispatch |
| `vllm/model_executor/layers/quantization/fp8.py:378-379` | the block scale registers as `weight_scale_inv`, with the comment that the name is intentional for deepseekv3 |
| `vllm/model_executor/layers/quantization/fp8.py:511` | `"weight_scale_inv" if self.block_quant else "weight_scale"`, the MoE spelling of the same rule |

The last anchor is why the failure is a name miss rather than a shape miss.
Upstream makes the scale name strictly conditional on block quant, so a
block-wise checkpoint that spells its scale `weight_scale` is not a shape
upstream produces.

## 2. Our baseline

`include/vllm/model_executor/layers/quantization/fp8.h:1` describes itself as
"Per-tensor FP8 (W8A8)" and mirrors the per-tensor arm only. The quantization
matrix already records the gap. `QUANT-FP8-GENERIC`
([`quantization-matrix.md:125`](../quantization-matrix.md)) covers
"static/dynamic, tensor/channel/token/block" at `PARTIAL`, and
`QUANT-FP8-PB-WO` at `:126` is `INVENTORIED`. No matrix edit is owed, because
neither row changes lifecycle state here.

## 3. Design

Refuse at the one place every model load passes through with an `HfConfig` in
hand. `ModelRegistry::Load`
(`src/vllm/model_executor/models/model_registry.cpp:322`) resolves the
architecture, then calls `factory.parse_config` and `factory.load_weights`. The
check goes after `Resolve` and before `parse_config`, so an unsupported
architecture still reports the architecture, and a block-wise checkpoint of a
supported architecture reports the quantization before any tensor lookup runs.

This placement is deliberately architecture-independent. `weight_block_size` is
a property of the checkpoint's quantization config, not of one model, so a
refusal per model loader would have to be written again for every architecture
and would be absent from whichever one is added next.

The tree has a second pre-load refusal site. `LoadedEngine::FromModelDir` calls
`RefuseUnsupportedWeightOffload` at
`src/vllm/entrypoints/model_loader.cpp:1536-1541`, after `LoadHfConfig` at
`:1529` and before the shards are mapped at `:1550`. It was rejected for two
reasons. It covers the safetensors path only, so the GGUF arm that reaches
`ModelRegistry::Load` at `:1448` would keep the old message. It also needs an
on-disk model directory to reach, which would make the gate depend on a
checkpoint this row is not allowed to download. `ModelRegistry::Load` takes the
`HfConfig` and the `ModelSource` directly, so the test drives the real
production function with no model directory and no weights.

The predicate mirrors `Fp8Config.from_config`. It reads
`quantization_config.weight_block_size` from `HfConfig::raw`, treats a non-empty
array as block quant, and ignores the key when it is absent or null. `raw` holds
the full top-level document (`src/vllm/transformers_utils/hf_config.cpp:564`),
so the top-level `quantization_config` this checkpoint uses is reachable. The
nested `text_config.quantization_config` spelling is read as well, because a
multimodal wrapper is exactly the shape in play.

New files, mirroring the upstream module that owns the key:

- `include/vllm/model_executor/layers/quantization/fp8_block_quant.h`
- `src/vllm/model_executor/layers/quantization/fp8_block_quant.cpp`

The header forward-declares `HfConfig` rather than including
`transformers_utils/hf_config.h`, so no JSON dependency enters the quantization
headers.

## 4. Port map

| Upstream | Here |
|---|---|
| `fp8.py:161` `from_config` reads the key | `Fp8BlockQuantOf` in `fp8_block_quant.cpp` |
| `fp8.py:115-132` `__init__` validates 2 dimensions | `RefuseUnsupportedFp8BlockQuant` reports the dimensions it found |
| `fp8.py:297-298` `block_quant` dispatch | the refusal. This tree has no block-wise arm to dispatch to |

## 5. Tests to port

Upstream has no test for refusing block-wise FP8, because upstream implements
it. There is nothing to port, and this section records that rather than leaving
it blank. The tests below are this tree's own, and they are written red first.

`tests/vllm/model_executor/layers/test_fp8_block_quant.cpp`, registered in
`tests/CMakeLists.txt` with `vllm_cpp_add_test`:

1. The refusal fires through `ModelRegistry::Load` on a config carrying
   `quantization_config.weight_block_size` `[128, 128]` and a registered
   architecture. This is the reachability case. It enters at the production
   entry point rather than at the predicate.
2. The message names `weight_block_size`, names block-wise FP8, and names
   `1166`.
3. A per-tensor FP8 config, meaning `quant_method` `fp8` with no
   `weight_block_size`, does NOT trip the refusal. Without this case the gate
   passes for a refusal that fires on every FP8 checkpoint.
4. The nested `text_config.quantization_config` spelling trips it too.
5. A null or empty `weight_block_size` does not trip it.

## 6. Gates

CPU only. No GPU lease, and no checkpoint download.

- Focused: the new target, red before the change and green after.
- Full: `scripts/agent-preflight.sh` with real per-block counts.
- Reachability mutation: delete the call site in `ModelRegistry::Load` in a
  scratch copy, rerun the focused target, and require red. Restore
  byte-for-byte.

## 7. Dependencies

None. The change adds two files, one call site, one test target, and the record
edits the change makes stale.

## 8. Work breakdown

One unit. Spec, then the refusal with its tests, then the record edits, in one
pull request with the spec committed first.

## 9. Risks and decisions

| Risk | Decision |
|---|---|
| The refusal fires on a checkpoint that loads today | Measured false. `weight_block_size` appears in no test and no example, and no load path reads `weight_scale_inv`, so nothing in the tree can be loading a block-wise checkpoint now |
| Refusing before `parse_config` changes error precedence | Accepted and bounded. The check sits AFTER `Resolve`, so an unsupported architecture still reports the architecture. Only a supported architecture with a block-wise config changes its message, which is the point |
| A GGUF load carries no `quantization_config` | Inert by construction. The key is absent, so the predicate is false |
| The message drifts from the issue it names | The test asserts the issue number, so deleting it reds the gate |

## Owed

- ~~Block-wise FP8 execution itself.~~ **DELIVERED** by the #1189 milestones,
  which this row's refusal made legible: `ad5f175e7` (M1, the dynamic per-token
  group quant upstream pairs with it, `fp8.py:301-310`), `770e49486` (M2, the CPU
  reference GEMM), `09597106e` (M3, the `weight_scale_inv` load),
  `281b4bc76` (M4, the linear method and the dense forward that reads it),
  `489a9a4c0` (M5, the mainloop-scaled CUTLASS kernel) and `836c13c35` (M6, the
  merged `gate_up` and QKV). What this row refused by name is now executed.
  [#1166](https://github.com/mudler/vllm.cpp/issues/1166) and
  [#1189](https://github.com/mudler/vllm.cpp/issues/1189) both stay OPEN, because
  the remaining debt is not the code:
  **the CUDA kernel has never executed on hardware and there is no token gate
  against `Qwen/Qwen3.8-27B-FP8`.** That is recorded in
  `.agents/specs/vt-matmul-fp8-block-cuda.md` `## Owed` and is not narrowed here.
- Eight more places still say the block-wise FP8 CUDA kernel is owed, outside
  the eight comment anchors [#1396](https://github.com/mudler/vllm.cpp/issues/1396)
  corrected. Three are LIVE refusal messages that tell a user on an unsupported
  CUDA arch to wait for milestone M5, which landed at `489a9a4c0`
  (`layers/quantization/fp8_block_quant.cpp:188-190`;
  `models/dense_fp8_block_gemm.h:200-202` in the `MatmulFp8BlockScaledD`
  guard; and `:428-430` in the shared `CheckFp8BlockMergedActivation` helper
  that both `MatmulFp8BlockMergedD` and `Fp8BlockGateUpSwiGLUD` call, so that
  message also fires on the merged `gate_up` path and not only on QKV -- a
  fixer who looks in `MatmulFp8BlockMergedD` will not find it). Five are
  comments (`include/vt/ops.h:1637,1658`, `src/vt/cpu/cpu_ops.cpp:668`,
  `include/vt/merged_gemm.h:92`,
  `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:605`,
  `tests/vllm/model_executor/models/test_fp8_block_linear.cpp:267`). The
  measurement `include/vt/ops.h:1637` names is still genuinely owed, so that one
  is a tense defect and not a retired obligation. Found by the fresh review of
  #1396 and left unfixed there rather than widening a comment-only change.
  A ninth place, the `RefuseUnrunnableQwen3_5DenseFp8Block` comment block in
  `qwen3_5_dense_weights.cpp`, WAS fixed in flow by #1396 and is no longer owed.
  It carries no line anchor here because #1396 edits that same block. It
  claimed `MatmulFp8BlockScaledD` reads each of the ten projections when it
  reads eight, because `qwen3_5.cpp` reaches those ten through three entry
  points -- `Fp8BlockGateUpSwiGLUD` is the only reader of `gate_proj_fp8_block`
  and `up_proj_fp8_block`, and `MatmulFp8BlockMergedD` reads q/k/v as one
  operand.
  Tracked by [#1411](https://github.com/mudler/vllm.cpp/issues/1411), which this
  row owns.
- `ReadF32Scalar` (`src/vllm/model_executor/models/qwen3_5_weights.cpp:312`)
  bounds its input with `t.nbytes >= sizeof(float)`, a lower bound, and returns
  the first 4 bytes. A multi-element scale passes that check and reads as block
  `(0,0)`. It is unreachable by the block-wise route, because the name miss
  described in section 0 stops the load first, and it is recorded here rather
  than fixed so a later change that adds a `weight_scale` alias cannot
  reintroduce it silently. Tracked under
  [#1166](https://github.com/mudler/vllm.cpp/issues/1166).

## Now

`ACTIVE`. The refusal and its tests land in this change. The block-wise arm
stays owed.
