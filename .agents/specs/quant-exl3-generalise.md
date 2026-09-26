# EXL3 loader generalisation — quant spec (QUANT-EXL3-GENERALISE)

> **Scope:** detach the EXL3 quantized-linear loader from the DeepSeek-V4
> weight schema so any architecture can consume EXL3 checkpoints, then open
> the door for `MiMoV2ForCausalLM` (and any future model) to run EXL3.

## Now

The EXL3 format is mature: the `mcg` codebook (cb 0/1) and the `mul1` codebook
(cb 2) are both decoded on host and on every device (CUDA, ROCm, Vulkan, CPU
reference). The kernels exist and are tested. The problem is the **loader**:
`deepseek_v4_weights.cpp:1010-1046` is the only consumer, and it
hard-rejects anything that is not a DeepSeek-V4 checkpoint.

## The blocker — two hard rejects

`src/vllm/model_executor/models/deepseek_v4_weights.cpp` around line 1010:

```
const std::string version = RawString(qc, "version", "");
// rejects anything but "rank-sliced-deepseek-v4-v1"
const std::string codebook = RawString(qc, "codebook", "mcg");
// rejects anything but "mcg"
```

The checkpoint `vcruz305/MiMo-V2.6-Flash-RL-EXL3` carries:

```json
"quantization_config": {
  "quant_method": "exl3",
  "version": "1.5.1",
  "bits": 2.2,
  "head_bits": 6,
  "codebook": "mul1",
  ...
}
```

Two `VT_CHECK` failures before any tensor loads:
1. `version != "rank-sliced-deepseek-v4-v1"` → rejected.
2. `codebook != "mcg"` → rejected.

The `mul1` kernel and the `mul1` marker acceptance have already landed (per
`.agents/specs/quant-exl3-mul1.md` slices A/B). The kernel is general; the
loader is not.

## What the checkpoint carries

From the safetensors index of `MiMo-V2.6-Flash-RL-EXL3` (2.20bpw variant):

- EXL3 tensors use suffixes `.trellis`, `.suh` (scale-up vector), `.svh`
  (scale-down vector) — the same schema as DeepSeek-V4 EXL3.
- `embed_tokens.weight` is bf16 (unquantized).
- `lm_head.weight` carries `.trellis`/`.suh`/`.svh` (quantized at `head_bits=6`).
- Per-layer linear weights: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`, and MoE expert weights
  (`experts.{i}.gate_proj`, etc.) all carry `.trellis`/`.suh`/`.svh`.
- `attention_sink_bias` is a plain bf16 tensor (not quantized).
- Router gate weights (`mlp.gate.weight`) are bf16 (not quantized).
- Norm weights are bf16.
- MTP weights (`model.mtp.*`) carry EXL3 tensors.

The trellis shapes confirm `mul1` codebook and fractional (2.2 bpw) trellis
widths — the same physical layout the kernel already decodes.

## Scope

Generalise the EXL3 loading path so it is **not gated to DeepSeek-V4**. The
kernels stay as-is; only the loader dispatch changes.

### What changes

1. **Move the EXL3 checkpoint detection out of `deepseek_v4_weights.cpp`.**
   Today `IsExl3Checkpoint` lives in the DSV4 weights file. It should move to
   a shared location (e.g. `include/vllm/model_executor/layers/quantization/exl3.h`
   or a new `exl3_checkpoint.h`) so any model's weight loader can call it.

2. **Generalise the version/codebook acceptance.** The loader must accept:
   - `version: "rank-sliced-deepseek-v4-v1"` (existing DSV4 checkpoints)
   - `version: "1.5.1"` (new upstream exllamav3 format version)
   - `codebook: "mcg"` (existing)
   - `codebook: "mul1"` (existing kernel, new loader acceptance)
   - Any future version/codebook the kernel supports.

3. **Provide a shared `Exl3LinearMethod` / `Exl3MlpGateUpMethod` dispatch.**
   The `Exl3LinearMethod` and `Exl3MlpGateUpMethod` classes already exist in
   `include/vllm/model_executor/layers/quantization/exl3.h`. They are the
   public interface. The problem is that no model loader outside DSV4 calls
   them. The generalisation makes them callable from any model's weight loader.

4. **Per-model opt-in.** Each model that wants EXL3 support registers its
   linear layers with `Exl3LinearMethod` when `IsExl3Checkpoint(config)` is
   true. The model's own weight loader maps tensor names to its architecture;
   the EXL3 method handles the dequant/repack.

### What does NOT change

- The EXL3 kernels (`cpu_exl3_dequant.cpp`, `cuda_exl3.cu`,
  `exl3_policy.cpp`).
- The `Exl3LinearMethod` / `Exl3MlpGateUpMethod` class interfaces.
- The DeepSeek-V4 weight loader (it keeps working with the same path).
- The `quantization_config` parsing (the `version` and `codebook` fields are
  read; they just stop being reject gates for non-DSV4 models).

## How it interacts with the model port

The MiMoV2 architecture port (`MODEL-TEXT-mimo-v2`) writes its own weight
loader (`mimo_v2_weights.cpp`). When that loader detects
`IsExl3Checkpoint(config)`, it routes each linear layer through
`Exl3LinearMethod` instead of the bf16 path. The tensor names are
MiMoV2-specific (`model.layers.N.self_attn.q_proj.trellis` etc.), but the
dequant kernel is shared.

The model port and the quant port are **independent issues**:
- The bf16 architecture port works without this row.
- This row works without the MiMoV2 port (it unblocks any future model).
- The motivating checkpoint (`MiMo-V2.6-Flash-RL-EXL3`) needs both.

## Work breakdown

### W1 — extract shared EXL3 checkpoint detection

- Move `IsExl3Checkpoint` from `deepseek_v4_weights.cpp` to
  `include/vllm/model_executor/layers/quantization/exl3.h` (or a new
  `exl3_checkpoint.h`).
- The function checks `quantization_config.quant_method == "exl3"` only —
  it does not gate on `version` or `codebook`.
- DeepSeek-V4 keeps its own `version == "rank-sliced-deepseek-v4-v1"` check
  in its own loader (that is a DSV4-specific invariant, not an EXL3 one).
- Tests: unit test that `IsExl3Checkpoint` returns true for both DSV4 and
  MiMoV2 quantization configs.

### W2 — generalise version/codebook acceptance

- The shared EXL3 path accepts `version` and `codebook` as informational
  fields, not reject gates.
- The `codebook` field selects the kernel decode path (`mcg` → cb 0/1,
  `mul1` → cb 2). Both are already implemented.
- The `version` field is logged but not rejected. If a future version
  introduces a breaking format change, a version gate can be added then.
- Tests: unit test that both `codebook: "mcg"` and `codebook: "mul1"` are
  accepted; the correct decode path is selected.

### W3 — wire EXL3 into the MiMoV2 loader

- `mimo_v2_weights.cpp` calls `IsExl3Checkpoint(config)` and, if true,
  routes linear layers through `Exl3LinearMethod`.
- The MiMoV2 loader maps tensor names (`q_proj.trellis`, `q_proj.suh`,
  `q_proj.svh`) to the EXL3 method.
- `head_bits` is read from `quantization_config` and applied to the
  `lm_head` and `embed_tokens` (if quantized).
- Tests: weight-load test on the EXL3 checkpoint (all tensors consumed,
  no missing/extra).

### W4 — end-to-end EXL3 gate

- Load `vcruz305/MiMo-V2.6-Flash-RL-EXL3` and run forward.
- Parity: compare dequantized weights against the bf16 reference (if
  available) or compare output tokens against the upstream exllamav3
  reference.
- Record in the quantization matrix `E` column for `QUANT-EXL3` and
  `QUANT-EXL3-MUL1`.

## Gates

1. **Loader generalisation.** `IsExl3Checkpoint` is callable from any model
   loader, not just DSV4.
2. **Version/codebook acceptance.** `version: "1.5.1"` and
   `codebook: "mul1"` no longer cause a hard reject outside DSV4.
3. **MiMoV2 EXL3 load.** The `MiMo-V2.6-Flash-RL-EXL3` checkpoint loads
   without missing/extra tensors.
4. **MiMoV2 EXL3 forward.** Forward produces correct output (parity against
   bf16 or upstream reference).
5. **DeepSeek-V4 regression.** Existing DSV4 EXL3 checkpoints still load
   and run (no regression).

## Dependencies

- **`MODEL-TEXT-mimo-v2`** (separate issue): the architecture port must land
  first. This row makes the EXL3 path available to it; the architecture port
  provides the weight loader that calls it.
- The EXL3 kernels (`QUANT-EXL3`, `QUANT-EXL3-MUL1`) are already landed.

## Risks / decisions

1. **`out_scales: "always"`.** The checkpoint config sets
   `out_scales: "always"`, meaning every linear layer carries an `svh`
   (scale-down) vector. The existing DSV4 path handles this; the
  generalised path must too.

2. **`head_bits` vs `bits`.** The `lm_head` and `embed_tokens` use
   `head_bits=6` (higher precision); body layers use `bits=2.2`. The loader
   must read both fields and apply the correct trellis width per tensor.

3. **`original_quantization_config`.** The config carries an
   `original_quantization_config` with `activation_scheme: "dynamic"` and
   `fmt: "e4m3"` — this is metadata from the FP8 → EXL3 conversion path.
   It is informational; the EXL3 loader ignores it.

4. **Fractional bits (2.2 bpw).** The trellis width for a 2.2 bpw layer is
   `ceil(out_features * 2.2 / 32) * 32`. The kernel handles this; the loader
   must pass the correct trellis shape.

## Stop conditions

- This row is DONE when:
  1. `IsExl3Checkpoint` is callable from any model loader.
  2. `version: "1.5.1"` and `codebook: "mul1"` are accepted (not rejected).
  3. `MiMo-V2.6-Flash-RL-EXL3` loads and runs (after the MiMoV2 arch port
     lands).
  4. DSV4 EXL3 checkpoints are unaffected.
- This row does NOT require:
  - The MiMoV2 architecture port to be complete (but the end-to-end gate
    does).
  - Performance gates (that is `QUANT-EXL3-PERF`).

## Owed

- `QUANT-EXL3-PERF`: EXL3 GEMV throughput optimisation — separate issue.
- `MODEL-MM-mimo-v2`: multimodal port — separate issue.
- Future EXL3 codebooks or versions: accepted as informational; a version
  gate can be added when a breaking format change ships.
