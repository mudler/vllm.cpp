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

---

## 7. TRUE W4A4 (fp4×fp4) — kernel selection + impl spec (grounded 2026-07-04)

All cites `file:line` relative to `/home/mudler/_git/vllm` @ `e24d1b24`.
This section supersedes §5's "6a fast path is the recommended primary" framing:
**the 6a fast path is NOT what vLLM runs, and the parity-ledger proved it is not
token-exact** (`parity-ledger.md` 27B row: 6a greedy **6/16 mismatch** — tokens
0-5 exact, diverges at tok6 on a "\n"/"\n\n" near-tie; prefill argmax 9/9 exact,
top-1000 logit gap 1.25; the divergence is *bf16 activations vs the oracle's
true fp4 activations* tipping a near-tie). To be token-for-token exact vs vLLM we
MUST mirror the **true W4A4 fp4×fp4** path (§5 step 6b), below.

### 7.1 DEFINITIVE ANSWER — what vLLM runs for THIS checkpoint on GB10/sm_121

**Mode.** `use_a16` is decided purely by the checkpoint config, NOT hardware:
`compressed_tensors.py:696-705` — `if input_quant is None: return
CompressedTensorsW4A4Fp4(use_a16=True)` (weight-only, Marlin), `else return
CompressedTensorsW4A4Fp4()` (`use_a16=False`, true W4A4). The 27B config HAS
`input_activations` (§3.1, num_bits 4, dynamic "local") ⇒ `input_quant` is not
None ⇒ **`use_a16=False` → TRUE W4A4 fp4×fp4** is what this checkpoint selects.
The `use_a16` shortcut we currently have corresponds to vLLM's *other* branch
(`use_a16=True`), which this checkpoint does **not** take.

**Kernel selection** (`kernels/linear/__init__.py:842-943`,
`init_nvfp4_linear_kernel`; CUDA priority list `:407-424`). For `use_a16=False`,
`linear_backend="auto"` (default), it walks `_POSSIBLE_NVFP4_KERNELS[CUDA]` in
order, taking the first whose `is_supported()` passes. On sm_121 (compute
capability 121):

1. `FlashInferCuteDslNvFp4LinearKernel` — `flashinfer.py:34-42` requires
   `is_device_capability_family(100)`. `interface.py:465-478`: family = cc//10;
   121//10=12 ≠ 100//10=10 ⇒ **SKIP** (it is sm_10x only).
2. `FlashInferCutlassNvFp4LinearKernel` — `flashinfer.py:105-119` requires
   `cutlass_fp4_supported() AND has_device_capability(100) AND has_flashinfer()`.
   `has_device_capability(100)` = cc≥100 (`interface.py:417-439`) → 121≥100 True.
   ⇒ **WINS iff FlashInfer is installed** in the venv.
3. `CutlassNvFp4LinearKernel` — `cutlass.py:23-29` requires
   `cutlass_fp4_supported()` only. ⇒ **WINS if FlashInfer is absent.**
4. Marlin / FlashInferTrtllm / FlashInferCudnn / Fbgemm / Emulation — fallbacks.

`cutlass_fp4_supported()` (`nvfp4_utils.py:56-61`) → C++
`cutlass_scaled_mm_supports_fp4(121)` (`nvfp4_scaled_mm_entry.cu:71-87`): returns
true for cc∈[120,130) **iff the wheel was built with `ENABLE_NVFP4_SM120`** (the
Blackwell/SM120a fp4 kernel). The GEMM itself dispatches sm∈[120,130) →
`cutlass_scaled_fp4_mm_sm120a` (`nvfp4_scaled_mm_entry.cu:59-64`). So the
**authoritative kernel to mirror is the CUTLASS SM120a fp4×fp4 GEMM**
(`cutlass_scaled_fp4_mm` → `..._sm120a`); the FlashInfer-cutlass wrapper, when
present, calls a numerically equivalent cutlass fp4 GEMM (same operand/scale/
alpha contract, `flashinfer.py:135-176`) — mirroring the native cutlass path
covers both. (Note `FlashInferB12x`, the SM120 CuTe-DSL warp-MMA kernel, is
**excluded from auto-selection**, `__init__.py:410-412` — opt-in only.) Whether
the measured 47.5 tok/s vLLM run used FlashInfer-cutlass or native cutlass is a
one-line probe (log the kernel class in `init_nvfp4_linear_kernel:937`); both are
the same math, so it does not change our spec.

**For `use_a16=True` (the OTHER branch, for reference):**
`__init__.py:881-883` — `linear_backend=="auto" and use_a16` **forces
`MarlinNvFp4LinearKernel`** (`marlin.py:18-57`, W4A16 weight-only, bf16
activations, no activation-quant). That is the mode our current
`MatmulNvfp4Wmma` shortcut approximates; this checkpoint does not use it.

### 7.2 The activation-quant op (the NEW W4A4 piece) — grounded on the REAL kernel

vLLM dynamically quantizes bf16/fp16 activations → fp4 **per token, per 16-elem
group** via `scaled_fp4_quant(x, layer.input_global_scale_inv, ...)` (called by
every real kernel, e.g. `cutlass.py:56-62`). The global scale passed is
`input_global_scale_inv` = the **on-disk activation divisor, used DIRECTLY (not
reciprocated)** (`compressed_tensors_w4a4_nvfp4.py:132-134`; same as the
emulation path `emulation.py:39-46`). The C++ kernel math
(`csrc/.../fp4/nvfp4_utils.cuh:241-297`, `cvt_warp_fp16_to_fp4`) is:

    SFScaleVal = input_global_scale_inv            # on-disk divisor (2688/amax_act)
    vecMax  = max_j |x[j]|                          # over the 16-elem group
    SFValue = SFScaleVal * (vecMax / 6.0)          # 6.0 = FP4_E2M1_MAX
    fp8SF   = fp8_e4m3(SFValue)                     # the STORED per-group block scale
    SFValue = float(fp8SF)                          # round-trip through fp8
    outputScale = SFValue!=0 ? SFScaleVal / SFValue : 0    # via fast reciprocal
    for each element:  fp4 = e2m1_round( x_j * outputScale )   # cast_to_fp4 buckets

This is **byte-identical in intent** to `ref_nvfp4_quant`
(`nvfp4_emulation_utils.py:427-441`) + `cast_to_fp4` (`:413-424`) — which is what
our CPU `RefNvfp4QuantDequant` already mirrors (§3.4). PARITY NUANCE: the real
CUDA kernel uses `reciprocal_approximate_ftz` (`nvfp4_utils.cuh:157-166,283-286`)
where the emulation uses exact division; the difference is sub-bucket and almost
never flips an e2m1 bucket, but it is the one place our fp4-quant op could
diverge from the oracle by 1 ULP → keep it in mind if a token near-ties.
Output layout: packed uint8 `[m, k/2]` + fp8-e4m3 block scales `[m, k/16]` in a
**swizzled** (128×4 / 8×4) layout for cutlass (`_custom_ops.py:73-92, 1492-1560`);
our own WMMA GEMM can keep the block scales in **linear** layout (we own both
producer and consumer) and skip the swizzle.

### 7.3 The fp4×fp4 GEMM contract to mirror (cutlass sm120a)

`cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_blockscale, b_blockscale, alpha,
out_dtype)` (`cutlass.py:64-71`, `_custom_ops.py:702-714`) computes, per output
`[m,n]`:

    out[m,n] = alpha * Σ_k ( a_fp4[m,k] · a_scale_fp8[m, k/16] )
                             · ( b_fp4[n,k] · b_scale_fp8[n, k/16] )

- `a_fp4` = per-token-quantized activations (§7.2); `a_scale_fp8` = its fp8 block
  scales (RAW stored fp8 values as f32, NOT divided by any global).
- `b_fp4` = the on-disk `weight_packed`; `b_scale_fp8` = the on-disk
  `weight_scale` (fp8, group-16), swizzled for cutlass (`cutlass.py:35-43`,
  `swizzle_blockscale` `nvfp4_utils.py:13-53`) — linear for our WMMA.
- **`alpha = input_global_scale · weight_global_scale`** — the product of the two
  *reciprocated* globals (`compressed_tensors_w4a4_nvfp4.py:135-138`;
  `weight_global_scale = 1/divisor` `:110-114`, `input_global_scale = 1/divisor`
  `:126-129`). Algebraically this folds both on-disk divisors: since both
  x_dq = a_fp4·a_scale_fp8/input_div and w_dq = b_fp4·b_scale_fp8/weight_div,
  `Σ x_dq·w_dq = alpha · Σ (a_fp4·a_scale_fp8)(b_fp4·b_scale_fp8)` with
  `alpha = 1/(input_div·weight_div)`. This is the exact `run_nvfp4_emulations`
  numeric result (§3.5) — our CPU `RunNvfp4Emulation` is its reference.

**Δ vs our current M2.7 `MatmulNvfp4Wmma`** (fp4×bf16, W4A16): (a) the A operand is
now *fp4 with its own per-token fp8 group scales*, not bf16 — so the MMA is
fp4×fp4 (Blackwell fp4 tensor cores) with TWO block-scale streams, not one; (b) a
single scalar `alpha` folds both globals (our W4A16 kernel applies only the
weight global as `scale2`); (c) A must be produced by the §7.2 activation-quant op
each forward (the per-token-quant tax).

### 7.4 How it plugs into our `vt` runtime (names chosen to mirror vLLM)

- **Activation-quant op** `ScaledFp4Quant` (mirror `scaled_fp4_quant`): bf16
  `[m,k]` + scalar `input_global_scale_inv` → (packed-fp4 `[m,k/2]`, fp8 block
  scale `[m,k/16]` linear). CUDA kernel = §7.2 math; the CPU reference is the
  existing `RefNvfp4QuantDequant` split into quant-only. Reuse `F32ToF8E4M3`,
  `CastToFp4`, `kNvfp4GroupSize`.
- **GEMM op** `MatmulNvfp4Fp4Wmma` (mirror `cutlass_scaled_fp4_mm` /
  `cutlass_scaled_fp4_mm_sm120a`): fp4 A + fp8 A-scale + fp4 B (`Nvfp4Weight`) +
  fp8 B-scale + scalar `alpha` → out `[m,n]`, per §7.3. Sibling to
  `MatmulNvfp4Wmma`; unit-test vs `RunNvfp4Emulation`.
- **Weight side** reuses `Nvfp4Weight` / `LoadCtNvfp4Raw` (already carries CT
  `weight_packed`/`weight_scale` + `scale2 = 1/weight_global_scale`). ADD: load
  `input_global_scale` (on-disk divisor) → store `input_global_scale_inv` and
  precompute `alpha = (1/input_divisor)·(1/weight_divisor)` at load (mirror
  `process_weights_after_loading`), so the forward passes `alpha` + `inv` to the
  two ops. The bf16-activation `MaterializeCtNvfp4Bf16Transposed` path stays as
  the correctness-grade fallback.
- **27B dense forward**: each quantized Linear (§3.6) becomes
  `MatmulNvfp4Fp4Wmma( ScaledFp4Quant(x, inv), W.fp4, W.wscale, alpha )` instead
  of `MatmulNvfp4Bf16D`. Gate on `Nvfp4Weight` presence exactly as today so the
  35B/bf16 paths are untouched.

### 7.5 killgate "W4A4 FP4-MMA regressed on GB10" — source-grounded caveat

The regression claim is cited in `parity-ledger.md` (M2.7 row) as *killgate
0034/0035: W4A4 FP4-MMA regressed on GB10*. `~/killgate_series/` does **not exist
on this dev box** (it lived on dgx), so I could only cross-check via the ledger.
Two things de-rate that claim for OUR decision: (1) it was measured on the **35B
modelopt W4A16** checkpoint, which has **no fp4 activations** — i.e. an *artificial*
fp4-activation experiment, not a real W4A4 checkpoint; and (2) it was a
hand-rolled / Marlin-grouped fp4-MMA, **not** vLLM's selected
`cutlass_scaled_fp4_mm_sm120a`. So it is evidence that *a naive fp4-MMA lost to
the W4A16 WMMA at 35B shapes*, NOT that vLLM's tuned sm120a path is slow. The 27B
question is different: vLLM *actually runs* sm120a fp4×fp4 here (§7.1) and we owe
token-exactness (6a already fails 6/16, §7 intro). Treat 6b throughput as a
measured experiment vs the sm120a-class kernel, not an assumed loss.

### 7.6 Ordered implementation plan (correctness-first; GPU steps marked)

STATUS (2026-07-05, dev-box session — NO GPU here: `mudler-ubuntu-box` has no
CUDA toolkit and the GB10/dgx is unreachable, so steps that BUILD/RUN CUDA or the
oracle were WRITTEN but not executed). Landed on branch
`worktree-agent-afef1d015a3960d60` (NOT merged to main — the token-for-token gate
must pass on GB10 first):
- **Step 1 ✅ (CPU, verified).** `vllm::RefScaledFp4Quant` + `Fp4ToNibble`
  (nvfp4_emulation.{h,cpp}) — the quant-only half, emits packed fp4 + fp8 block
  scale. vt ops `ScaledFp4Quant` + `MatmulNvfp4Fp4` added (ops.h/ops.cpp, OpIds
  `kScaledFp4Quant`/`kMatmulNvfp4Fp4`) with **CPU kernels** (cpu_ops.cpp,
  self-contained fp8/fp4 codec). Unit test `tests/vt/test_ops_nvfp4_fp4.cpp`:
  ScaledFp4Quant BYTE-EXACT vs `RefScaledFp4Quant` and its decode == the
  `RefNvfp4QuantDequant` x_dq; `MatmulNvfp4Fp4(ScaledFp4Quant(x),W)` ==
  `RunNvfp4Emulation` (529 assertions, CPU-green). Full CPU ctest 86/86.
- **Steps 2+3 ⚠ (CUDA WRITTEN, not built — no nvcc here).** `ScaledFp4QuantKernel`
  + `MatmulNvfp4Fp4{Naive,Wmma}` in cuda_matmul_nvfp4.cu, registered for kCUDA.
  The GEMM is the CORRECTNESS-FIRST compute (dequant BOTH fp4 operands to bf16 in
  shared → bf16 WMMA → ×alpha = the exact `RunNvfp4Emulation` numeric result); the
  native block-scaled fp4×fp4 (Blackwell mxf4) MMA is the throughput follow-up
  (step 6, not attempted — would need the sm120a mxf4 `mma.sync` PTX / cutlass
  drop-in). test_ops_nvfp4_fp4 has a HasCuda()-guarded device-vs-CPU case ready to
  run on GB10 (quant byte-exact; GEMM within matmul tol). **MUST clean-rebuild +
  run on GB10 to confirm nvcc compiles it.**

---

## 7.7 NATIVE sm120a fp4×fp4 MMA — implemented + GB10-grounded (2026-07-05)

Executed §7.6 step 6 (native block-scaled fp4×fp4) end-to-end on dgx (GB10,
sm_121, CUDA 13.0.88, nvcc V13.0.88). All results below are measured, not assumed.

**STEP 0 — vLLM native vs emulation on the gate prompt (hardware A/B).**
Prompt `"The capital of France is Paris, and the"`, greedy 16, `unsloth/Qwen3.6-27B-NVFP4`,
`~/venvs/vllm-oracle` (vLLM 0.24.0, flashinfer 0.6.12), enforce_eager.
- NATIVE (auto-selected `FlashInferCutlassNvFp4LinearKernel` →
  `get_gemm_sm120_module_cutlass_fp4`, confirms §7.1): greedy tok6 = **198**
  (== the oracle golden `greedy_ids.npy`).
- EMULATION (forced via `VLLM_DISABLED_KERNELS=` the 7 higher-priority nvfp4
  kernels → `EmulationNvFp4LinearKernel`): greedy tok6 = **271**, full 16-tok
  stream `{...13, 271, 248068, 271, 248069, 271, 4639, 369, 4252, 13, 271}`.
- ⇒ **PROVEN on hardware**: vLLM's own emulation ≠ its native cutlass on this
  near-tie. Our C++ true-W4A4 path reproduces the EMULATION stream token-for-token
  (all 16), i.e. we faithfully mirror vLLM's reference; only the flashinfer-cutlass
  sm120a kernel yields 198.

**STEP 1 — feasibility of an ORIGINAL native block-scaled fp4×fp4 MMA (no cutlass link).**
FEASIBLE, but architecture-specific. The NVFP4 warp MMA
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`
(e2m1 operands, ue4m3 group-16 scales) **assembles and RUNS correctly on
`sm_121a`/`sm_120a`** but is **rejected on base `sm_121`**: ptxas
`error: Instruction 'mma with block scale' not supported on .target 'sm_121'`.
So the TU must be built for the arch-specific target (`-gencode
arch=compute_121a,code=sm_121a`; `-arch=sm_121a` alone drops the `a` for exe
builds). Guard macro = `__CUDA_ARCH_SPECIFIC__` (==1210 for sm_121a, undefined for
sm_121). No WMMA C++ API for fp4 (`mma.h` has no e2m1 fragment) — inline PTX only.
The exact m16n8k64 fragment + `scale_vec::4X` scale-operand layout was
reverse-engineered device-vs-CPU on GB10 and validated **bit-exact** (maxrel 0.0)
on random full GEMMs incl. K/M/N tails:
  - lane g=lane/4 (0..7), t=lane%4 (0..3); nibble j → k-elem j (low nibble low k).
  - A frags a0(row g,k=t·8+j) a1(row g+8,·) a2(row g,32+·) a3(row g+8,·); B b0(n g,k=t·8+j) b1(n g,32+·).
  - D d0(g,t·2) d1(g,t·2+1) d2(g+8,t·2) d3(g+8,t·2+1).
  - A-scale: row r → lane (r%8)·4 + (r≥8?1:0), byte b = k-block b. B-scale: col n → lane n·4, byte b = k-block b. thread_id=byte_id=0.

**STEP 2 — implemented.** `MatmulNvfp4Fp4Native` (warp-per-16×8-tile, the above
layout, K looped in 64-chunks, tails zero-padded) + fast-reciprocal
(`rcp.approx.ftz.f32`) activation quant, both gated behind runtime env
`VT_NVFP4_FP4_NATIVE=1` (**default OFF**) and build define `VT_FP4_MMA_SM120A`
(set when `CMAKE_CUDA_ARCHITECTURES` matches `12[01]a`; repo default bumped to
`121a`). `test_ops_nvfp4_fp4` passes both default and native (1850/1850).
- **27B gate with native ON: still tok6 = 271** (identical 16-tok stream to vLLM
  emulation). Our independent, bit-correct native MMA lands on the SAME side of the
  near-tie as emulation; it does NOT bit-reproduce flashinfer-cutlass's specific
  accumulation order + quant → 198 is not recovered. This is the documented
  acceptable outcome (§7 intro / task STEP 2 caveat): matching cutlass's 198
  requires bit-matching its exact reduction tree + flashinfer's scaled_fp4_quant,
  which an original kernel cannot without the cutlass/flashinfer drop-in.
- **35B gate: 16/16 SUCCESS** on the sm_121a build (W4A16 path untouched; arch
  bump is numerically safe).

**STEP 3 — throughput: NOT measured (vllm-bench).** The native kernel is an
unoptimized warp-per-(16×8) tile (no shared-memory operand reuse) so it is
expected SLOWER than the existing shared-mem-tiled bf16 WMMA path for prefill;
it needs tiling work before it is a throughput lever. Left as follow-up.

**Decision.** Native path kept **opt-in, not default**: it gives identical
correctness (271, == vLLM emulation) to the faster default bf16 path and unknown
(likely worse) throughput, so defaulting it would risk a throughput regression
with no correctness gain. NOT merged to main; committed to this branch. To recover
the 198 golden, the remaining honest option is the cutlass/flashinfer sm120a
drop-in (post-MVP); the correctness bar for our path is logit/emulation parity,
which we meet token-for-token.
- **Weight load ✅ (plumbed).** `Nvfp4Weight` gained `input_global_scale_inv` +
  `alpha` (+ `IsTrueW4A4()`); `LoadCtNvfp4Raw` now reads `<proj>.input_global_scale`
  and precomputes `alpha = scale2/input_global_scale_inv` (notes §7.3).
- **Forward ✅ (wired, GPU-untested).** New `MatmulNvfp4Fp4D` (qwen3_5.cpp) =
  `ScaledFp4Quant(x)`→`MatmulNvfp4Fp4(…alpha)`; `MatmulNvfp4F32D`/`MatmulNvfp4Bf16D`
  route to it when `w.IsTrueW4A4() && TrueW4A4Enabled()` (CUDA only). Env toggle
  `VT_W4A4_TRUE` (default ON, mirrors vLLM; `=0` → 6a W4A16 A/B). Gated on `alpha>0`
  so the **35B path is byte-identical** (its modelopt weights have alpha==0).
- **Steps 4–6 ✗ (GPU-gated, NOT done):** oracle true-W4A4 golden capture,
  flipping `kW4A4ForwardReady` + the token-for-token gate, and the throughput A/B
  all require the GB10 + checkpoint + `~/venvs/vllm-oracle`. `kW4A4ForwardReady`
  stays `false`. THIS is what remains to close the gate 6a failed (§7 intro).

1. **[CPU]** Split `ScaledFp4Quant` CPU reference out of `RefNvfp4QuantDequant`
   (quant-only: emit packed fp4 + fp8 block scale). Unit-test vs the existing
   emulation numbers + a hand golden (fast-reciprocal nuance noted §7.2).
2. **[GPU]** `ScaledFp4Quant` CUDA kernel (§7.2). Unit-test device-vs-CPU on
   random bf16 (allow ≤1 ULP on the fast-reciprocal path).
3. **[GPU]** `MatmulNvfp4Fp4Wmma` CUDA kernel (§7.3, fp4×fp4 + two fp8 scale
   streams + scalar alpha). Unit-test vs `RunNvfp4Emulation` (the CPU truth) on
   random shapes incl. odd N/K tails — same rig as `test_ops_nvfp4_matmul`.
4. **[GPU]** Capture the pip-vLLM **true-W4A4** greedy golden on
   `unsloth/Qwen3.6-27B-NVFP4` (§5 step 5; supersedes the 6a golden). Log the
   selected kernel class (`init_nvfp4_linear_kernel:937`) to confirm
   flashinfer-cutlass vs native cutlass — one light dispatch line, no throughput
   run.
5. **[GPU]** Wire the two ops into the 27B dense forward (§7.4); flip
   `kW4A4ForwardReady`; run `test_qwen27_paged_engine` **token-for-token** vs the
   step-4 true-W4A4 golden (this is what closes the correctness gate that 6a
   failed).
6. **[GPU]** `vllm bench throughput` vs ours on the identical GB10 workload;
   record both + ratio in parity-ledger. Compare 6b (true fp4×fp4) against the 6a
   fast path as a measured A/B — keep whichever wins throughput *among the
   token-exact options* (6a is not token-exact, so if 6b is slower we still need
   6b for the correctness gate, and 6a can only remain as an opt-in speed mode).
