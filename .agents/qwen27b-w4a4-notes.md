# Qwen3.6-27B (dense, NVFP4 W4A4) — pinned semantics + bring-up plan

The 27B is the SECOND MVP gate model (co-equal with the 35B; gates.md). This
doc is its pinned bring-up reference, in the style of gdn-semantics.md /
moe-semantics.md: every fact below was read from source — the actual checkpoint
config.json + safetensors manifest (read read-only on dgx 2026-07-04) and the
pinned upstream checkout `/home/mudler/_git/vllm` @ `e24d1b24`. Cites are
`file:line` relative to `vllm/` in that tree, or the checkpoint file.

Checkpoint (read-only on dgx):
`~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4`.

STATUS (2026-07-04): CPU-first correctness path LANDED (§5 steps 1-4b). Delivered:
the CPU W4A4 dequant + activation-quant reference (`nvfp4_emulation.h`,
unit-tested); the dense loader (`LoadQwen3_5Dense` → `Qwen3_5DenseWeights`,
`qwen3_5_dense.h`/`qwen3_5_dense_weights.cpp`) that routes each Linear bf16 vs
W4A4-materialized-to-bf16 by name (`IsQwen27QuantizedLinear` +
`MaterializeCtNvfp4Bf16Transposed`); the dense TEXT forward
(`Qwen3_5DenseModel::ForwardDense` in `qwen3_5.cpp`, reusing the 35B GDN +
gated-attn + norm helpers with the MoE block swapped for the dense SwiGLU MLP);
the batched PAGED dense forward (`Qwen3_5DenseModel::Forward`, same signature/
structure as the 35B `Qwen3_5Model::Forward` — paged KV cache for full-attn,
batched GDN state for GDN, via a new `RunDenseLayerPaged` reusing the 35B
`GdnBlockPaged`/`FullAttnBlockPaged` VERBATIM) + the runner route (a dense-arch
`GPUModelRunner` constructor overload holding `Qwen3_5DenseWeights` and routing
`execute_model` to the dense forward; the MoE reference member became a
`{moe,dense}_weights_` pointer pair); and CPU unit tests
(`test_qwen27_dense_forward.cpp`: routing + materialization + finite/deterministic
forward + MLP-wired, 4 cases / 280 assertions; `test_qwen27_paged_forward.cpp`:
the paged==dense anchor + multi-block + decode-via-cache + GDN-state-zeroing, 4
cases / 8 assertions, CPU-green). The full LoadedEngine dense dispatch is now
wired (arch-select in `model_loader.cpp` + a `{moe,dense}` weights pair on
`LoadedEngine`, CPU-proven by `test_loaded_engine_dense.cpp`) — the 27B is FULLY
CPU-wired end-to-end. The full CPU ctest suite (87 targets) stays
green. What is NOT done and is GPU-gated: the pip-vLLM oracle greedy golden
capture (step 5), the W4A4 matmul kernel wiring (step 6), and flipping
`kW4A4ForwardReady` to close the gate (step 7). §5 is the ordered plan, GPU steps
marked.

---

## 0. TWO surprises that change the plan (read first)

1. **The 27B is a VISION-LANGUAGE multimodal model**, not a plain text model.
   `architectures = ["Qwen3_5ForConditionalGeneration"]`, `model_type
   "qwen3_5"`, with a full `vision_config` (27-layer ViT, hidden 1152 →
   out_hidden 5120, patch 16, spatial_merge 2), `image_token_id 248056`,
   `video_token_id 248057`, `vision_start/end_token_id`, and
   `language_model_only: false` (config.json top level). The text backbone is
   `text_config` / `model_type "qwen3_5_text"`. **For text throughput parity the
   vision tower is inert** (instantiated only when image/video tokens are
   present) — but the arch CLASS, the 248320-vocab embed table (includes the
   vision special tokens), and mRoPE are multimodal-shaped. Bring-up should
   implement the TEXT path first and leave the ViT + merger + deepstack as a
   later (T1/T2) multimodal task. Upstream registry:
   `model_executor/models/registry.py:556`
   (`"Qwen3_5ForConditionalGeneration": ("qwen3_5", ...)`); the MoE sibling is
   `:557` (`Qwen3_5MoeForConditionalGeneration`).

2. **Quant is compressed-tensors NVFP4 W4A4** (both weights AND activations
   fp4), NOT the 35B's modelopt W4A16 (bf16 activations). This is a genuinely
   different quant path (§3). BUT the on-disk WEIGHT encoding is byte-identical
   to modelopt NVFP4 (E2M1 nibbles + fp8-e4m3 group-16 scale + one f32 global),
   differing only in the tensor NAMES and that the global scale is stored as a
   DIVISOR — so the existing W4A16 tensor-core GEMM can carry the 27B weights
   with a one-line reciprocal fix (see §5, the recommended fast path).

The task's framing ("dense, W4A4") is correct on both the MLP (dense, §2) and
the quant (W4A4, §3); the VL wrapper was the unexpected part.

---

## 1. Text-backbone dims + layer pattern (config.json `text_config`)

| field | value | note |
|---|---|---|
| hidden_size | 5120 | |
| num_hidden_layers | 64 | |
| intermediate_size | 17408 | DENSE SwiGLU MLP (§2) |
| vocab_size | 248320 | incl. vision special tokens |
| head_dim | 256 | attention head dim |
| num_attention_heads | 24 | full-attn Q heads |
| num_key_value_heads | 4 | full-attn KV heads (GQA 6:1) |
| attn_output_gate | true | q_proj emits Q **and** an output gate |
| full_attention_interval | 4 | every 4th layer is full attention |
| layer_types | `[L,L,L,F] × 16` | full-attn at layers 3,7,…,63 (16 full, 48 GDN) |
| linear_num_key_heads (Hk) | 16 | GDN |
| linear_num_value_heads (Hv) | **48** | **GQA ratio Hv/Hk = 3** (35B is 2) |
| linear_key_head_dim (Dk) | 128 | |
| linear_value_head_dim (Dv) | 128 | |
| linear_conv_kernel_dim | 4 | |
| output_gate_type | "swish" | → silu gate (gdn-semantics.md §5) |
| mamba_ssm_dtype | float32 | |
| partial_rotary_factor | 0.25 | rotary_dim = 0.25·256 = **64** |
| rope_parameters.rope_theta | 1e7 | |
| rope_parameters.mrope_section | [11,11,10] | sums to 32 = rotary_dim/2; mrope_interleaved true |
| rms_norm_eps | 1e-6 | |
| tie_word_embeddings | false | separate lm_head |
| mtp_num_hidden_layers | 1 | MTP head (ignored in quant; not needed for base greedy) |

Derived GDN dims (gdn-semantics.md §1): key_dim = Hk·Dk = 2048, value_dim =
Hv·Dv = **6144**, conv_dim = 2·key_dim + value_dim = **10240**. Confirmed by the
manifest (§4).

---

## 2. What is SHARED with the 35B vs NEW

The 27B REUSES the 35B's hybrid-backbone op contract wholesale — only the MLP
and the quant path change.

**SHARED (reuse the qwen3_5.cpp components as-is, dims from §1):**
- The **GDN (Gated DeltaNet) linear-attention** layers — conv1d + l2norm +
  gated-delta-rule recurrence + RMSNormGated, g/beta from A_log/dt_bias, all per
  gdn-semantics.md §1-§8. Same Dk=Dv=128, conv 4, silu gating. Only Hv differs
  (48 vs 35B's 32; the GQA broadcast `i_h = i_hv // (Hv//Hk)` already handles any
  ratio, gdn-semantics.md §1).
- The **gated full-attention** layers — GQA, q/k RMSNorm, partial RoPE (64d),
  the attention output gate (`attn_output_gate=true`), per
  qwen36-forward-notes.md §5. Same head_dim 256.
- **mRoPE → NeoX** partial rotary (qwen36-forward-notes.md §2); for text-only
  input the three mrope position streams are identical so it degenerates to 1-D
  RoPE.
- **Gemma-style `(1+w)` RMSNorm**, per-layer input_layernorm /
  post_attention_layernorm, final norm.

**NEW (must be written for the 27B):**
- **Dense SwiGLU MLP** replaces the 256-expert MoE block (no router, no shared
  expert, no expert gather). Per layer: `gate_proj [17408,5120]`, `up_proj
  [17408,5120]`, `down_proj [5120,17408]`; `down( silu(gate(x)) * up(x) )`.
  Upstream: `model_executor/models/qwen3_5.py` dense-MLP path (the
  `Qwen3_5MLP` used when the layer is not MoE). This is the ONLY structural
  change vs the 35B backbone.
- **compressed-tensors NVFP4 W4A4** load + GEMM (§3, §5) instead of modelopt
  W4A16. Activations are dynamically fp4-quantized per token.
- **GQA ratio 3** (Hv48) — a dims change only; no new code.
- **The VL wrapper** (§0.1) — deferred; text path first.

---

## 3. compressed-tensors NVFP4 W4A4 format (from the checkpoint + upstream)

### 3.1 quantization_config (config.json)

`quant_method: "compressed-tensors"`, `format: "nvfp4-pack-quantized"`,
`quantization_status: "compressed"`, one `config_groups.group_0` targeting
`["Linear"]`:

- **weights**: `num_bits 4`, `type "float"` (fp4/E2M1), `group_size 16`,
  `strategy "tensor_group"`, `symmetric true`, `dynamic false`,
  `scale_dtype torch.float8_e4m3fn`, `observer "memoryless_minmax"`.
- **input_activations**: `num_bits 4`, `type "float"`, `group_size 16`,
  `strategy "tensor_group"`, `symmetric true`, `dynamic "local"` (⇐ **per-token
  dynamic quant at runtime**), `scale_dtype torch.float8_e4m3fn`,
  `observer "static_minmax"` (the per-tensor global scale is calibrated; the
  per-group scales are computed dynamically at runtime).
- **output_activations: null** (outputs stay high precision).

`num_bits 4` on BOTH weights and input_activations ⇒ **W4A4**.

### 3.2 On-disk tensor set per quantized Linear (manifest, read on dgx)

Example `model.language_model.layers.0.mlp.gate_proj` (in=5120, out=17408):

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `weight_packed` | U8 | [17408, 2560] | two E2M1 fp4 per byte, in/2 cols; elem 2i = low nibble, 2i+1 = high |
| `weight_scale` | F8_E4M3 | [17408, 320] | one fp8-e4m3 scale per 16-elem group (in/16), **LINEAR layout on disk (no swizzle)** |
| `weight_global_scale` | F32 | [1] | per-tensor global scale, stored as a **DIVISOR** (≈ FP8_MAX·FP4_MAX/amax = 2688/amax_w) |
| `input_global_scale` | F32 | [1] | per-tensor activation global scale, also a **DIVISOR** (2688/amax_act) |

Name/semantics contrast with the modelopt W4A16 path (nvfp4_dequant.h):
`weight_packed`/`weight_scale`/`weight_global_scale`/`input_global_scale` (CT)
vs `weight`/`weight_scale`/`weight_scale_2`/`input_scale` (modelopt). And CT
stores the global scales as **reciprocals** of the modelopt multipliers
(§3.3). Create-weights: `compressed_tensors_w4a4_nvfp4.py:38-93`.

### 3.3 Weight dequant (the CPU-truth)

`dequantize_to_dtype(swizzle=False)` (`nvfp4_emulation_utils.py:346-400`), with
the global scale pre-reciprocated at load
(`compressed_tensors_w4a4_nvfp4.py:110-113`: `weight_global_scale = 1.0 /
weight_global_scale.max()`):

    global = 1.0 / weight_global_scale_ondisk          # CT stores 1/scale
    w_dq[o,i] = e2m1(nibble[o,i]) * ( f32(weight_scale[o, i//16]) * global )

Identical E2M1 LUT `{0,.5,1,1.5,2,3,4,6}` and fp8-e4m3 decode as the modelopt
path; the ONLY math delta is the reciprocal on the global scale. Our C++:
`DequantCtNvfp4WeightToF32` (nvfp4_emulation.h) — reuses `F8E4M3ToF32` +
`kE2M1Lut` + `kNvfp4GroupSize`, reciprocates internally.

### 3.4 Activation quant-dequant (the NEW W4A4 piece, the CPU-truth)

`ref_nvfp4_quant` + dequant (`nvfp4_emulation_utils.py:427-466`), driven by the
emulation kernel with `input_global_scale = layer.input_global_scale_inv` — the
**ON-DISK divisor, used DIRECTLY (NOT reciprocated)**
(`kernels/linear/nvfp4/emulation.py:41`). Per row, per 16-group:

    FP4_MAX = 6.0                                       # e2m1 max
    vec_max = max_j |x[j]|                              # f32, over the group
    scale   = input_global_scale * (vec_max / 6.0)
    scale   = clamp(scale, -448, 448)                  # fp8-e4m3 range
    scale_f8 = f8e4m3_round(scale) -> f32              # the stored per-group scale
    block_scale  = scale_f8 / input_global_scale       # = x_blockscale (dequant)
    output_scale = 1 / block_scale                     # (0 stays 0)
    for each element x_j:
        fp4  = cast_to_fp4( clamp(x_j * output_scale, -6, 6) )   # bucket-round
        x_dq = fp4 * block_scale                        # x_dq ≈ x_j

`cast_to_fp4` = fixed half-open bucket boundaries
(`nvfp4_emulation_utils.py:413-424`). The **asymmetry is intentional**: weight
dequant reciprocates the global, activation quant does not — both on-disk values
are the same "divisor" form (2688/amax), used where each formula needs it. Our
C++: `RefNvfp4QuantDequant` + `CastToFp4` + `F32ToF8E4M3` (nvfp4_emulation.h).

### 3.5 The emulated W4A4 apply (`run_nvfp4_emulations`, :469-495)

    out = (activation round-trip x_dq) @ (weight dequant w_dq)^T      # f32

`alpha = input_global_scale * weight_global_scale` is precomputed for the real
fp4×fp4 GEMM (`compressed_tensors_w4a4_nvfp4.py:135-138`) — the true kernel
multiplies fp4 activations by fp4 weights (integer-ish) and applies the two
global scales + the two per-group fp8 scales; the emulation above is the
numeric reference it must match. Our C++: `RunNvfp4Emulation`.

### 3.6 Which Linears are quantized (config.json `ignore` + manifest)

QUANTIZED (W4A4): every dense-MLP `gate/up/down_proj`, every full-attn
`q/k/v/o_proj`, and the GDN **`linear_attn.out_proj`**.

NOT quantized (bf16 on disk, in the `ignore` list): the GDN **input**
projections `linear_attn.in_proj_{qkv,z,a,b}` (regex
`re:.*\.linear_attn\.in_proj_*$`), `conv1d`, `A_log`, `dt_bias`, all norms, the
embed table, **`lm_head`**, `re:^mtp\..*`, and **all `model.visual.*`**. The
loader must route by name: bf16 for the ignore set, W4A4-dequant for the rest.
(nvfp4_recipe.json: 304 quantized / 303 ignored Linear modules.)

---

## 4. Manifest cross-check (dgx, `model.safetensors`, 2111 tensors)

GDN layer 0 (`layers.0.linear_attn.*`), bf16 in-projs + quantized out_proj:
- `in_proj_qkv.weight` bf16 [10240, 5120] — conv_dim 10240 = 2·2048+6144 ✓
- `in_proj_z.weight` bf16 [6144, 5120] — value_dim 6144 (z gate, Dv per v-head) ✓
- `in_proj_a.weight` / `in_proj_b.weight` bf16 [48, 5120] — Hv=48 ✓
- `conv1d.weight` bf16 [10240, 1, 4] — conv_dim × K=4 ✓; `A_log`/`dt_bias` bf16 [48]
- `norm.weight` bf16 [128] — RMSNormGated over Dv=128 ✓
- `out_proj.weight_packed` U8 [5120, 3072] (in=6144=value_dim), `weight_scale`
  F8_E4M3 [5120, 384] (=6144/16), `weight_global_scale`/`input_global_scale`
  F32 [1] — W4A4 ✓

Full-attn layer 3 (`layers.3.self_attn.*`), all W4A4:
- `q_proj.weight_packed` U8 [12288, 2560] (in=5120, out=12288 = 24·256·2 — Q +
  output gate) ✓; `q_norm.weight`/`k_norm.weight` bf16 [256]
- `k_proj`/`v_proj` [1024, 2560] (out=4·256=1024, GQA KV) ✓
- `o_proj` [5120, 3072] (in=6144=24·256) ✓

Dense MLP (both layer types): `gate_proj`/`up_proj` [17408, 2560] (in=5120),
`down_proj` [5120, 8704] (in=17408) — all W4A4 ✓.

---

## 5. Bring-up plan (ordered; correctness → GPU kernels → throughput)

Legend: **[CPU]** doable on the dev box now; **[GPU]** needs the free GB10.

1. **[CPU] ✅ DONE — CPU W4A4 dequant + activation-quant reference.**
   `nvfp4_emulation.h/.cpp` (`DequantCtNvfp4WeightToF32`, `RefNvfp4QuantDequant`,
   `CastToFp4`, `F32ToF8E4M3`, `RunNvfp4Emulation`), unit-tested hand-computed vs
   the pinned emulation math (`test_ct_nvfp4_emulation.cpp`, 6 cases / 81
   assertions, CPU-green). This is the CPU-truth the GPU GEMM validates against.

2. **[CPU] ✅ DONE — Config + loader plumbing.** `LoadHfConfig` already
   recognizes `Qwen3_5ForConditionalGeneration` via the `text_config`/
   `qwen3_5_text` resolution (partial_rotary_factor default 0.25 → rotary_dim 64;
   `num_experts` reads 0 = dense). `LoadQwen3_5Dense`
   (`qwen3_5_dense.h`/`qwen3_5_dense_weights.cpp`) mirrors `LoadQwen3_5Moe`:
   `Qwen3_5DenseWeights` = embed + final_norm + bf16 `lm_head` + per-layer
   `Qwen3_5DenseLayerWeights` (reused `GdnLayerWeights`/`FullAttnLayerWeights` +
   new `DenseMlpWeights`). Each Linear is routed bf16 vs W4A4-materialized-to-bf16
   by `IsQwen27QuantizedLinear` (encodes the §3.6 `ignore` list) and materialized
   via `MaterializeCtNvfp4Bf16Transposed` → `DequantCtNvfp4WeightToF32` (reads the
   CT `weight_packed`/`weight_scale`/`weight_global_scale` names; the on-disk
   `input_global_scale` activation divisor is IGNORED on this bf16-activation path
   — the step-6a fast path). Tensor-name remap recorded in porting-inventory §9.7.

3. **[CPU] ✅ DONE — Dense forward assembly.** `Qwen3_5DenseModel::ForwardDense`
   (in `qwen3_5.cpp`, so it reuses the file-local `GdnBlock`/`FullAttnBlock`/norm/
   `DBuf`/matmul helpers verbatim) mirrors `Qwen3_5Model::ForwardDense`: embed →
   N `RunDenseLayer` (input_layernorm → GDN|full-attn → post_attn_layernorm →
   `DenseMlpBlock`) → final RMSNorm → bf16 `lm_head`. `DenseMlpBlock` =
   `down( silu(gate(x)) * up(x) )` (the shared-expert silu-mul, no router/gate).
   Text-only: the three mRoPE streams coincide so partial NeoX RoPE degenerates to
   1-D over `positions` (§2). CPU-validated on a synthetic small hybrid model
   (finite + deterministic logits; MLP-perturbation moves the output). The ViT +
   image/video merger + MTP head are DEFERRED (text-first, §0.1) — no stub code,
   the loader simply does not request `model.visual.*`/`mtp.*`.
   `ForwardDense` is retained as the single-sequence parity reference (the
   paged==dense anchor), exactly as the 35B's `ForwardDense`.
   ⚠ REMAINING before step 7 flips the gate (all GPU-gated now): (a) the W4A4
   matmul kernel (step 6); (b) mrope_section handling for genuine multimodal
   positions (inert for text).

3b. **[CPU] ✅ DONE — Paged dense path + runner wiring.**
   `Qwen3_5DenseModel::Forward` (in `qwen3_5.cpp`) is the batched/paged 27B text
   forward with the SAME signature/structure as `Qwen3_5Model::Forward`: paged KV
   cache (`PagedKvCache`) for the full-attn layers + batched GDN recurrent state
   (`GdnStateCache`) for the GDN layers + the f32 residual thread, per-layer
   `fa_idx`/`gdn_idx` indexing identical to the 35B. It reuses the 35B
   `GdnBlockPaged`/`FullAttnBlockPaged` + paged machinery VERBATIM via a new
   `RunDenseLayerPaged` (a copy of `RunLayerPaged` with `DenseMlpBlock` in place
   of `MoeBlock` and `Qwen3_5DenseLayerWeights` in place of the MoE weights). No
   dense CUDA-graph driver (the MoE `Qwen3_5DecodeGraph` is an fp4/CUDA decode
   optimization; the dense path runs eager, as its GPU GEMM is step 6). Runner
   route: `GPUModelRunner` gained a `Qwen3_5DenseWeights` constructor overload;
   its MoE reference member became a `{moe,dense}_weights_` pointer pair, and
   `execute_model` routes to `Qwen3_5DenseModel::Forward` when `dense_weights_` is
   set (the MoE-only fp4 decode-graph fast path stays inert on the dense arch).
   `initialize_kv_cache` is unchanged (config-driven; same hybrid backbone).
   CPU-validated by `test_qwen27_paged_forward.cpp` (the 27B analogue of the 35B
   `test_qwen35_paged_forward.cpp`): paged==dense full-prefill, multi-block
   (block_size<T non-contiguous), decode-via-KV-cache, and GDN-state-zeroing on a
   garbage-seeded mamba block — all within tolerance (`max|diff|` 0 on the
   zeroing/mixed-batch gate). Deviations recorded in porting-inventory §9.
   ✅ Full LoadedEngine dense loading is now WIRED (CPU): `model_loader.cpp`
   arch-selects on `LoadedEngine::IsDenseArch` (`num_experts==0` → `LoadQwen3_5Dense`,
   else `LoadQwen3_5Moe`) and `LoadedEngine` carries the SAME `{moe,dense}` weights
   pair (two `std::optional<...>` members + a `Qwen3_5DenseWeights` constructor
   overload driving the runner's dense overload). The Executor/EngineCore/processors
   were already arch-agnostic (they touch the runner only through `ModelRunnerBase`),
   so the MoE-typing was confined to the runner (previous commit) + the loader — no
   deeper MoE assumption resisted the pointer-pair. CPU-proven by
   `tests/vllm/entrypoints/test_loaded_engine_dense.cpp` (IsDenseArch dispatch + the
   dense stack generating deterministically through the whole LLMEngine loop), the
   dense sibling of the 35B `test_llm_engine.cpp`. The 27B is now FULLY CPU-wired
   end-to-end; the only remaining steps are the GPU ones (steps 5-7).

4. **[CPU] ✅ DONE — greedy-parity gate scaffold.**
   `test_qwen27_paged_engine.cpp` resolves the `unsloth/Qwen3.6-27B-NVFP4`
   snapshot and SKIPS (checkpoint-gated + `kW4A4ForwardReady=false`). Compiles
   + links on CPU; flip the flag when step 6 lands (the paged forward is now
   ready — step 3b).

5. **[GPU] Capture the pip-vLLM oracle greedy golden** (AGENTS.md STANDING
   DIRECTIVE; gates.md §PROTOCOL). Run pip-vLLM (`~/venvs/vllm-oracle`) on the
   SAME checkpoint, greedy (temperature 0) continuation of the pinned M0-exit
   prompt → commit `greedy_ids` (+ optional per-layer/logits goldens), exactly
   as the 35B goldens were captured. **Uses the GPU** — do only when the box is
   free and serialized behind the 35B kernel jobs.

6. **[GPU] The W4A4 matmul kernel.** TWO options:
   - **6a. FAST PATH (recommended first): reuse the existing W4A16 tensor-core
     GEMM.** The 27B weight encoding == modelopt NVFP4 (§0.2), so
     `MatmulNvfp4Wmma` / `MoeGroupedGemmNvfp4Wmma` can carry the 27B **dense**
     linears with just (i) reciprocating the global scale (`scale2 = 1 /
     weight_global_scale`) and (ii) the CT tensor-name remap — keeping bf16
     activations (ignoring the checkpoint's activation-quant). Numerically this
     is a tiny accuracy deviation from true W4A4; validate greedy vs the oracle
     (step 5). This gets the 27B to correctness + throughput WITHOUT a new
     kernel, reusing the whole M2.7 tensor-core path.
   - **6b. TRUE W4A4 fp4×fp4 MMA (throughput follow-up).** A dynamic per-token
     activation-quant kernel (§3.4) + an fp4×fp4 tensor-core GEMM with
     `alpha = input_global_scale·weight_global_scale`. Halves activation
     bandwidth and uses Blackwell fp4 tensor cores, but **adds the per-token
     activation-quant tax**. ⚠ RISK: the killgate 0034/0035 prior art
     (`~/killgate_series`, and parity-ledger M2.7) found **W4A4 FP4-MMA
     REGRESSED on GB10** for the 35B (activation-quant overhead > the fp4-MMA
     win at these shapes). So 6a is likely the throughput winner near-term;
     6b is a measured experiment, not an assumed speedup.

7. **[GPU] Flip `kW4A4ForwardReady` and close both gates.** Greedy
   token-for-token vs the step-5 oracle golden (correctness); then `vllm bench
   throughput` vs ours on the IDENTICAL workload on GB10, record both numbers +
   ratio in parity-ledger (STANDING DIRECTIVE). The 27B GDN/attn kernels are the
   SAME as the 35B's — so the 27B rides the 35B's kernel campaign; its only
   unique kernel work is the W4A4 GEMM (step 6) and the dense-MLP wiring.

---

## 6. How the oracle parity is checked (recorded, not run here)

- **Correctness oracle** = pip-vLLM (`~/venvs/vllm-oracle`) on the SAME
  `unsloth/Qwen3.6-27B-NVFP4` checkpoint, greedy/temperature-0. The C++ paged
  engine must reproduce the oracle's continuation **token-for-token**
  (`test_qwen27_paged_engine.cpp`, once wired). Optional per-layer activation +
  final-logits goldens for localizing a divergence, as with the 35B.
- **Throughput oracle** = `vllm bench throughput` on the identical workload on
  GB10; record ours + vLLM + ratio per commit (parity-ledger).
- Both are **GPU-gated** and run LATER (box currently reserved for the 35B
  kernel job + kept clean for throughput measurement). This session set them up
  (the CPU reference + the skipping gate) but ran neither.
