# MiMoV2 — architecture port spec (MODEL-TEXT-mimo-v2)

> **Scope:** port the `MiMoV2ForCausalLM` architecture (text-only LLM arm) so
> the server loads, forward runs, and parity holds against the upstream
> reference. The EXL3 quant row is a separate issue (see Dependencies).

## Now

No `mimo_v2` code exists in the tree. `grep -ri mimo src/ include/` returns
nothing. The architecture is not registered, not loaded, not forwarded.

## The checkpoint

Source: `vcruz305/MiMo-V2.6-Flash-RL-EXL3` (HuggingFace). The config ships a
`configuration_mimo_v2.py` and `modeling_mimo_v2.py` with
`auto_map` entries for `AutoConfig`, `AutoModel`, `AutoModelForCausalLM`.

The EXL3 quant is a separate row (`QUANT-EXL3-GENERALISE`). This spec targets
the bf16 reference; the quant row feeds into it.

## Config — what the model is

| Field | Value |
|---|---|
| `model_type` | `mimo_v2` |
| `architectures` | `["MiMoV2ForCausalLM"]` |
| `hidden_size` | 4096 |
| `num_hidden_layers` | 48 |
| `vocab_size` | 152576 |
| `tie_word_embeddings` | `false` |
| `layernorm_epsilon` | `1e-06` |
| `hidden_act` | `silu` |

### Attention — hybrid full + sliding-window, per-layer

MiMoV2 alternates two attention geometries per layer, driven by
`hybrid_layer_pattern` (a 48-element list):

| Field | Full attention (pattern==0) | SWA (pattern==1) |
|---|---|---|
| `num_attention_heads` | 64 | 64 |
| `num_key_value_heads` | 4 | 8 |
| `head_dim` | 192 | 192 (`swa_head_dim`) |
| `v_head_dim` | 128 | 128 (`swa_v_head_dim`) |
| `rope_theta` | 10 000 000 | 10 000 (`swa_rope_theta`) |
| `partial_rotary_factor` | 0.334 | 0.334 |
| `sliding_window` | — | 128 (`attention_chunk_size`) |
| `attention_projection_layout` | `fused_qkv` | `fused_qkv` |
| `attention_value_scale` | 0.707 | 0.707 |

`hybrid_layer_pattern = [0,1,1,1,1,0, 1,1,1,1,1,0, ...]` — a full-attention
layer every 6th layer (indices 0, 6, 12, 18, 24, 30, 36, 42), the rest SWA.

`add_swa_attention_sink_bias = true`, `add_full_attention_sink_bias = false`:
SWA layers carry an `attention_sink_bias` parameter (shape
`[num_key_value_heads, head_dim]` = `[8, 192]`); full-attention layers do
not. This is a learned additive bias on the attention sink position, not the
DeepSeek-V4 attention-sink cache mechanism — it is a plain `nn.Parameter`
added to the sink-token logits before softmax.

`partial_rotary_factor = 0.334` means `rotary_dim = floor(192 * 0.334) = 64`.
The first 64 channels of each head get RoPE; the last 128 are passthrough.

`attention_value_scale = 0.707` scales the V projection output (sqrt(0.5)).

### MoE — sigmoid routing, noaux_tc, no shared experts

| Field | Value |
|---|---|
| `n_routed_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 2048 |
| `n_shared_experts` | `None` (no shared expert) |
| `scoring_func` | `sigmoid` |
| `topk_method` | `noaux_tc` (top-k with bias correction) |
| `n_group` | 1 |
| `topk_group` | 1 |
| `norm_topk_prob` | `true` |
| `routed_scaling_factor` | `None` |
| `moe_router_dtype` | `bfloat16` |
| `intermediate_size` (dense) | 16384 |
| `moe_layer_freq` | `[0,1,1,...]` — layer 0 is dense, layers 1–47 are MoE |

`moe_layer_freq[0] = 0` means layer 0 runs a dense `MiMoV2MLP` (gate/up/down,
`intermediate_size=16384`). Layers 1–47 run `MiMoV2MoE` (256 experts,
`moe_intermediate_size=2048`, no shared expert).

Routing: `sigmoid(gate_logits) → noaux_tc top-k → normalize`. The `noaux_tc`
method (from DeepSeek-V2) adds a bias correction term to the top-k selection:
`weights = softmax(sigmoid(scores[topk_indices]) + correction_bias)`, where
`correction_bias` is the mean score of the top-k candidates. This is the same
routing used by Qwen3-MoE and DeepSeek-V2, already implemented in the tree.

### MTP — 3 nextn predict layers

`num_nextn_predict_layers = 3`. The MTP draft model shares the embedding and
has its own per-layer weights (`model.mtp.{0,1,2}.*`). The modeling code
registers `model.mtp.*` in the weight map cleanup regex. This is the same MTP
pattern as DeepSeek-V4 / Dots3-Note / GLM5 — the existing MTP infra applies,
but the per-layer composition (shared embedding + per-layer transformer block +
shared lm_head) must be wired for MiMoV2's geometry.

### Multimodal — out of scope for this row

The config ships `vision_config` (28-layer ViT, `mimovl` type,
`out_hidden_size=4096`) and `audio_config` (20-channel audio encoder with
local + full attention). These are multimodal arms and belong to a future
`MODEL-MM-mimo-v2` row. This spec ports the text-only `MiMoV2ForCausalLM`.

## What is NEW vs the closest existing model

The closest existing model is Qwen3.5-MoE (hybrid linear+full attention,
sigmoid MoE, partial rotary). The deltas:

1. **Hybrid full-attention + SWA** (not linear attention). Qwen3.5 uses
   GDN/linear attention for non-full layers. MiMoV2 uses sliding-window
   attention — different KV geometry (8 KV heads, `v_head_dim=128` for SWA vs
   4 KV heads for full) and a `sliding_window=128` constraint. DeepSeek-V4 and
   Gemma2 have SWA but not with per-layer different KV head counts.

2. **Per-layer KV head split.** Full-attention layers have 4 KV heads;
   SWA layers have 8 KV heads. `v_head_dim=128` for both, but `head_dim=192`
   means the Q/O projection is 192-dim while K/V are different shapes per layer
   type. The fused QKV projection must be aware of which geometry the layer
   uses.

3. **`attention_sink_bias` on SWA layers.** A learned additive bias on the
   sink position, present only on SWA layers. No existing model has this.

4. **`attention_value_scale = 0.707`.** V projection output is scaled by
   sqrt(0.5) before the attention computation. This is a post-projection
   scalar, not a softmax temperature.

5. **`noaux_tc` routing without shared experts.** Qwen3-MoE and DeepSeek-V2
   have `noaux_tc` but with shared experts. MiMoV2 has none. The MoE block
   is simpler (no shared-expert branch), but the routing path must handle
   `n_shared_experts=None`.

6. **`moe_layer_freq[0] = 0`.** Layer 0 is dense. This is a config-driven
   per-layer MLP type switch, not a global "all layers are MoE" flag.

## Upstream chain

- `modeling_mimo_v2.py` (1875 lines) ships in the checkpoint repo.
- The architecture is registered via `auto_map` (not a native vLLM
  registry architecture at the parity pin `555967922`).
- No vLLM registry entry for `MiMoV2ForCausalLM` exists at the pin.
- This is a **beyond-pin** architecture: the port follows
  AGENTS.md "When vLLM has no implementation" — the upstream modeling code is
  the reference, and we implement from the Python source.

## Port map

### Files to create

| File | Purpose |
|---|---|
| `include/vllm/model_executor/models/mimo_v2.h` | `MiMoV2Weights` struct, `MiMoV2Model::Forward/ForwardDevice` decls |
| `src/vllm/model_executor/models/mimo_v2.cpp` | forward pass (attention + MoE + MLP per layer) |
| `src/vllm/model_executor/models/mimo_v2_weights.cpp` | bf16 weight loader |
| `src/vllm/model_executor/models/mimo_v2_registry.cpp` | `REGISTER_VLLM_MODEL(mimo_v2, "MiMoV2ForCausalLM")` + config parse + KV-cache spec |

### Reuse from existing code

| Component | Reuse from | Delta |
|---|---|---|
| RMSNorm | `vt::RmsNorm` | none (`layernorm_epsilon=1e-06` is standard) |
| RoPE (partial) | `vt::RotaryEmbedding` with `partial_rotary_factor` | already supported (Qwen3.5 path); `rotary_dim=64` |
| Fused QKV | `vt::MatmulBT` | Q is `[hidden, 64*192]`, K/V shape depends on layer type |
| Paged attention | `vt::PagedAttention` | SWA layers need `sliding_window=128` in the attention spec |
| MoE routing (sigmoid + noaux_tc) | existing `vt::MoeCombine` | no shared expert (`n_shared_experts=None` path) |
| MoE experts | `vt::MatmulBT` per-expert | 256 experts × (gate, up, down) at `moe_intermediate_size=2048` |
| Dense MLP (layer 0) | `vt::MatmulBT` | SwiGLU at `intermediate_size=16384` |
| MTP | existing MTP infra | 3 nextn layers, shared embedding + shared lm_head |
| lm_head | `vt::MatmulNB` | untied (`tie_word_embeddings=false`) |

### New code

1. **Hybrid attention layer dispatch.** A per-layer function that reads
   `hybrid_layer_pattern[layer]` and selects full-attention vs SWA geometry:
   different KV head count, different rope_theta, different sliding_window.

2. **`attention_sink_bias` parameter.** Loaded as a per-SWA-layer bias tensor
   `[8, 192]`, added to the sink-position logits before softmax. No existing
   model has this — it is a new weight in the attention block.

3. **`attention_value_scale` post-projection scalar.** Multiply V projection
   output by `0.707` before reshaping. A one-line scalar multiply, but it
   must be applied at the right point (after the V projection matmul, before
   the attention score computation).

4. **`moe_layer_freq` per-layer MLP type switch.** Layer 0 loads a dense MLP
   (`gate_proj`, `up_proj`, `down_proj` at `intermediate_size=16384`); layers
   1–47 load a MoE block (256 experts). The loader must read
   `moe_layer_freq` and dispatch.

## Tests to port

- Upstream: none at the pin (no vLLM test for `MiMoV2ForCausalLM`).
- From the checkpoint: the `auto_map` modeling code is the reference. We
  write our own parity test from a bf16 reference run.
- Token gate: generate N tokens from the bf16 checkpoint and compare
  token-exact against the upstream `modeling_mimo_v2.py` reference.
- Registry test: `tests/vllm/models/test_model_registry.cpp` — resolve
  `MiMoV2ForCausalLM`, check `is_dense_model=false`, check hybrid KV spec.

## Gates

1. **Config parse.** `model_type: "mimo_v2"` is recognised; all config fields
   are read; unknown fields are rejected (not silently dropped).
2. **Registry.** `MiMoV2ForCausalLM` resolves to the `mimo_v2` factory.
3. **Weight load.** All safetensors tensors are consumed; no missing or
   extra tensors (modulo the multimodal arms, which are skipped for the
   text-only port).
4. **Forward parity.** Token-exact match against the upstream reference on
   at least one prompt at short context (within sliding window) and one at
   long context (beyond sliding window, exercising the hybrid pattern).
5. **MTP.** If MTP is enabled, the draft model produces the same logits as
   the upstream reference for the first nextn layer.

## Dependencies

- **`QUANT-EXL3-GENERALISE`** (separate issue): the EXL3 loader currently
  hard-rejects `version: "1.5.1"` and `codebook: "mul1"` outside the
  DeepSeek-V4 path. The bf16 port does not depend on this — but running the
  motivating checkpoint (`MiMo-V2.6-Flash-RL-EXL3`) does. The quant row is
  a prerequisite for the end-to-end EXL3 run, not for the architecture port.
- **`noaux_tc` routing**: already implemented (DeepSeek-V2, Qwen3-MoE).
- **MTP infra**: already implemented (DeepSeek-V4, Dots3-Note, GLM5).
- **Partial rotary**: already implemented (Qwen3.5).
- **Sliding-window attention**: already implemented (DeepSeek-V4, Gemma2).

## Work breakdown

### W1 — registry + config + KV-cache spec

- `mimo_v2_registry.cpp`: `REGISTER_VLLM_MODEL(mimo_v2, "MiMoV2ForCausalLM")`
- `ParseMiMoV2Config`: read all fields; validate `hybrid_layer_pattern`
  length == `num_hidden_layers`; validate `moe_layer_freq` length ==
  `num_hidden_layers`; validate `num_nextn_predict_layers`; reject if
  multimodal configs are present and `--allow-multimodal` is not set (text-only
  port).
- KV-cache spec: hybrid — full-attention layers get a standard paged spec
  (4 KV heads, `head_dim=192`, `v_head_dim=128`); SWA layers get a
  sliding-window spec (8 KV heads, `head_dim=192`, `v_head_dim=128`,
  `sliding_window=128`).
- Tests: `test_model_registry.cpp` — resolve, config, KV spec.

### W2 — weight loader (bf16)

- `mimo_v2_weights.cpp`: load all tensors for the text-only arm.
- Per layer: fused QKV (` fused_qkv`), O proj, `attention_sink_bias` (SWA
  layers only), input/output RMSNorm, MLP (dense for layer 0, MoE for 1–47).
- MoE: router gate `[hidden, 256]`, 256 × (gate, up, down) at
  `[hidden, 2048]`.
- MTP: `model.mtp.{0,1,2}.*` weights.
- Skip: `model.visual.*`, `model.audio.*`, `model.vision_tower.*`.
- Tests: weight-count test (all tensors consumed).

### W3 — forward pass

- `mimo_v2.cpp`: `MiMoV2Model::Forward` / `ForwardDevice`.
- Per layer:
  1. Input RMSNorm + residual.
  2. Fused QKV matmul → reshape per layer geometry.
  3. `attention_value_scale` on V.
  4. Partial RoPE (rotary_dim=64, theta per layer type).
  5. `attention_sink_bias` added to sink position (SWA layers only).
  6. Paged attention (full or SWA per `hybrid_layer_pattern`).
  7. O proj.
  8. Output RMSNorm + residual.
  9. MLP: dense (layer 0) or MoE (layers 1–47).
- MTP forward (if enabled).
- lm_head (untied).
- Tests: forward parity against upstream reference.

### W4 — MTP wiring

- Wire the 3 nextn layers into the existing MTP infra.
- Shared embedding, shared lm_head, per-layer transformer block.
- Tests: MTP draft logits parity.

### W5 — parity gate

- Token-exact match on short + long context.
- Record in the parity ledger.

## Risks / decisions

1. **`attention_sink_bias` semantics.** The upstream code adds it to the
   sink position before softmax. We must confirm the exact position in the
   attention computation (after QK^T, before softmax, at the sink index
   only). This is a new mechanism — no existing model has it.

2. **Fused QKV with per-layer KV geometry.** Full-attention layers have
   `num_key_value_heads=4`; SWA layers have `num_key_value_heads=8`. The fused
   QKV projection produces Q (64×192), K (kv_heads×192), V (kv_heads×128).
   The projection weight shape changes per layer type. We must either load
   separate projections per layer type or pad — the upstream code uses a
   single `fused_qkv` weight per layer with the K/V shape determined by the
   layer type at config time.

3. **`v_head_dim != head_dim`.** `head_dim=192` but `v_head_dim=128`. The V
   projection produces 128-dim heads; the attention score is computed on
   `head_dim=192` (Q·K) but the value is 128-dim. This is unusual — most
   models have `v_head_dim == head_dim`. The O projection maps
   `num_heads × v_head_dim = 64 × 128 = 8192` back to `hidden_size=4096`.

   Wait — that does not work: `64 × 128 = 8192 ≠ 4096`. Let me re-check.
   Actually `v_head_dim=128` and `num_attention_heads=64`:
   `64 × 128 = 8192`. The O projection input is `8192` → `4096`. That is a
   valid matmul. The attention output is `[batch, num_heads, seq, v_head_dim]`
   reshaped to `[batch, seq, 8192]`, then O proj maps to 4096.

   But the attention score is `Q @ K^T` where Q is `[64, 192]` and K is
   `[kv_heads, 192]` — the score is on `head_dim=192`. The value is 128-dim.
   So the attention output per head is 128-dim, not 192-dim. This is a
   non-standard split: Q/K use `head_dim=192`, V uses `v_head_dim=128`.

   This is a NEW pattern not present in any existing model. It must be
   handled explicitly in the attention forward.

4. **SWA `v_head_dim=128` vs full `v_head_dim=128`.** Both are 128, so V
   is consistent. But `head_dim=192` for Q/K in both. The difference is
   only in KV head count (4 vs 8) and rope_theta (10M vs 10K).

5. **`moe_router_dtype = bfloat16`.** The router runs in bf16, not float32.
   Some existing MoE implementations upcast the router to fp32 — we must
   not do that here, or confirm parity allows it.

## Stop conditions

- The architecture port is DONE when:
  1. `MiMoV2ForCausalLM` resolves in the registry.
  2. A bf16 checkpoint loads without missing/extra tensors.
  3. Forward produces token-exact output against the upstream reference.
  4. MTP (if enabled) produces matching draft logits.
  5. The registry test passes.
- The architecture port does NOT require:
  - EXL3 quantization (that is `QUANT-EXL3-GENERALISE`).
  - Multimodal arms (vision/audio — future `MODEL-MM-mimo-v2`).
  - Performance gates (that is a separate row after parity).

## Owed

- `MODEL-MM-mimo-v2`: multimodal port (vision + audio) — not started.
- `QUANT-EXL3-GENERALISE`: EXL3 loader generalisation — separate issue.
- Performance gate: no perf target until parity is green.
