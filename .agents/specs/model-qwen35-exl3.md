# MODEL-QWEN35-EXL3 — the EXL3 arm of the DENSE Qwen3.5 loader and forward

Row: `MODEL-QWEN35-EXL3`
Issues: [#2495](https://github.com/mudler/vllm.cpp/issues/2495) (items 3 and 5)
Base SHA: `c5df2b1ed` (branch `row/QUANT-EXL3-MUL1`)
Parent row: [`QUANT-EXL3-MUL1`](quant-exl3-mul1.md) →
[`QUANT-EXL3`](quant-exl3-shared.md)
Matrix: [`.agents/model-matrix.md`](../model-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`. vLLM defines the
MODEL (`vllm/model_executor/models/qwen3_5.py`, which this tree already mirrors)
and the SEAM (`layers/linear.py`, `layers/quantization/base_config.py`). vLLM
implements no EXL3 at the pin, so the trellis FORMAT comes from the registered
secondary oracle [`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (MIT), exactly as the parent row
records. Nothing here re-derives either half.

## Now

`ACTIVE`. The dense half of an EXL3 `Qwen3_5ForConditionalGeneration` checkpoint
loads and runs; the GDN linear-attention tower refuses by name and is #2495
item 4.

## The gap

`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` declares
`architectures: ["Qwen3_5ForConditionalGeneration"]`, which
`src/vllm/model_executor/models/qwen3_5_dense.cpp` registers and which this tree
already runs in bf16, per-tensor FP8 and NVFP4. Before this row,
`grep -c exl3 src/vllm/model_executor/models/qwen3_5_dense_weights.cpp` returned
`0`.

The dense loader resolves a projection's scheme by TENSOR PRESENCE among three
shapes: bf16 `.weight`; per-tensor FP8 (`.weight` F8_E4M3 + `.weight_scale` +
`.input_scale`); ModelOpt/compressed-tensors NVFP4 (`.weight` U8 +
`.weight_scale` + `.weight_scale_2`). An EXL3 projection has **no `.weight` at
all** — it has `.trellis`, `.suh`, `.svh` and, in this checkpoint, `.mul1`. So
every probe in that resolver either mis-routes it or dies asking for a tensor
the checkpoint correctly does not ship, and the failure a user saw was
`tensor not found: ...q_proj.weight` on a complete artifact.

## Scope

IN: a fourth rung in the dense scheme resolver for `self_attn.{q,k,v,o}_proj`
and `mlp.{gate,up,down}_proj`; the `lm_head` (#2495 item 5); the F16
unquantized remainder an EXL3 artifact stores; the forward that reads them; a
by-name refusal for an EXL3 GDN tower.

OUT, and each is named rather than left to be discovered:

- **The GDN / `linear_attn` tower** (`in_proj_qkv`, `in_proj_z`, `out_proj`).
  48 of the real model's 64 layers are `linear_attention` and NOTHING in
  `ProjectGdnQkvz`/`ProjectGdnBA`/`ProjectGdnOut` consumes an `Exl3Weight`.
  That is #2495 item 4 and it carries its own dispatch, because the GDN
  in-projections are MERGED owners and a trellis merge on the output dim is a
  real transform. This row REFUSES it by name.
- The vision tower (ships unquantized and is unaffected).
- The DFlash2 draft, the shared head and the MTP head.
- Any end-to-end run or benchmark of the published checkpoint, which item 4
  blocks.

## Design

Every piece already existed; this row is the BINDING and never a second copy.

| Need | Reused |
|---|---|
| storage predicate | `dense_loaders::IsExl3Projection` (`Linear.is_exl3_storage`, `modules/linear.py:385-389`) |
| reader | `dense_loaders::LoadExl3` |
| container | `Exl3Weight` (`qwen3_5_weights.h`) |
| compute | `layers::Exl3LinearMethod` / `layers::Exl3MlpGateUpMethod` → `dense_attn::Exl3MatmulD` |
| F16 remainder | `dense_loaders::LoadF16AsBf16Direct` |

The working exemplar is `src/vllm/model_executor/models/llama_weights.cpp:73-85`
(projections) and `:152-161` (the head preferred over a tied embedding table),
and the shape is mirrored rather than re-invented.

Three decisions are worth recording:

1. **The EXL3 rung goes FIRST in each resolver**, before the NVFP4 probe and
   before any `get(name + ".weight").dtype` read, for the same reason #1189 M3
   put the block-FP8 rung before the per-tensor one: the later probes read a
   tensor this arm does not have.
2. **q/k/v and gate/up stay SEPARATE.** Every other arm holds or builds one
   merged owner. A trellis is `[k/16, n/16, 32*bits]`, so joining on the output
   dim interleaves per input tile — a real transform, valid for this family and
   owed its own gate (`## Owed` in `quant-exl3-shared.md`).
3. **F16 acceptance is scoped to an EXL3 load** (`LoadModelVectorForScheme`).
   Teaching `LoadModelBf16Direct` F16 outright would widen acceptance for every
   dense model through a conversion that drops three mantissa bits, which is
   the argument `LoadF16AsBf16Direct` already makes at its own declaration.

## Tests

`tests/vllm/models/test_qwen35_exl3.cpp`, four cases over one synthetic
safetensors checkpoint written per arm from ONE set of trellis bytes:

- **G1** the EXL3 checkpoint LOADS: codebook 2, 4 bits, the right
  `in/out_features` on all nine projections and the head, EXACTLY ONE
  representation populated, the trellis still trellis bytes, the GDN tower bf16
  beside it, and `IsPlainBf16Qwen3_5Dense` false.
- **G2** `ModelRegistry::Forward` over the LOADED weights agrees with the same
  bytes decoded into the bf16 fields (`rel_rms <= 5e-2`), with a non-vacuity
  guard on the reference energy.
- **G3** an EXL3 GDN tower refuses by name, through `LoadQwen3_5Dense` AND
  through `ModelRegistry::Load`, naming `in_proj_qkv`, `EXL3`, `2495` and
  `item 4`; and the same dense arm with a bf16 GDN tower still loads, so the
  case measures the GDN projections rather than the presence of any EXL3
  weight.
- **G4** the bf16, per-tensor FP8 and ModelOpt NVFP4 arms land in their own
  fields with the EXL3 ones empty, and `IsPlainBf16Qwen3_5Dense` still claims
  the plain bf16 model.

## Gates

```sh
cmake -S . -B build -GNinja && ninja -C build -j 4 test_qwen35_exl3 test_llama_exl3_forward test_fp8_block_weight_load test_fp8_block_linear
ctest --test-dir build -R '^(test_qwen35_exl3|test_llama_exl3_forward|test_fp8_block_weight_load|test_fp8_block_linear)$' --output-on-failure
scripts/agent-preflight.sh --staged
```

## Owed

- **#2495 item 4, the GDN tower.** Refused by name here. Until it lands, the
  published checkpoint does not load end to end.
- **The `mtp.*` head, the other half of #2495 item 5.** This row wires
  `lm_head`. The MTP head is loaded on demand by `LoadQwen3_5MTP` and is not
  part of the always-resident target weights, so it is a separate container and
  a separate dispatch; #2495 items 6 and 7 (the shared head and the DFlash2
  draft loader) are in the same neighbourhood and are likewise out.
- **`docs/USAGE.md` owes the checkpoint's file names, sizes, repo and
  REVISION** once an arm of it actually loads end to end. Inherited verbatim
  from the parent row, and still owed for the same reason: nothing can be fed
  the published artifact yet.
- **No device run.** Every case here is CPU. The EXL3 device kernels are the
  parent row's slice C and carry their own `## Owed` entry.
- **No merged EXL3 QKV or gate_up.** Named in `quant-exl3-shared.md`; this row
  inherits it rather than opening it.
- **`ReleaseResidentQwen3_5DenseHostWeights` does not release an EXL3 host
  mirror.** It walks `OwnedTensor` members it names one by one and knows
  nothing about `Exl3Weight`, so a trellis, `suh` and `svh` stay host-resident
  after their device upload. That is a residency cost on a host-addressable
  device, not a correctness one, and it is left to the row that measures it
  rather than guessed at here.
- **`fp4_attn`, the per-arch fused-preamble key, reads false for an EXL3
  model.** It is a performance default and not a correctness switch, and an
  EXL3 model is genuinely not fp4; recorded so the next reader does not read
  the `false` as an oversight.

## Stop conditions

Stop and report rather than widen scope if: the GDN refusal would have to be
weakened to make a case pass; a merged trellis operand is needed; or the F16
remainder acceptance would have to leak outside an EXL3 load.
