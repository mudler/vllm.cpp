// Nemotron-H (`NemotronHForCausalLM`) — the W3 STRUCTURAL bring-up of the row
// `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
// ([spec](../../../../.agents/specs/nemotron-h-model.md) §4 W3, issue #517).
//
// W3 SCOPE, exactly: make the architecture KNOWN, PARSED, ENUMERATED and
// KV-SHAPED. Config descent over the released `config.json`, the on-disk weight
// name map, the registry entry, and the heterogeneous KV-cache topology (one
// full-attention group over the 6 attention layers + one Mamba2 recurrent-state
// group over the 23 mamba layers). There is **no forward path here** — that is
// W4, and `ForwardNemotronHForCausalLM` REFUSES BY NAME rather than returning a
// silent wrong answer. The row stays `INVENTORIED`; this file changes no
// lifecycle state.
//
// This header lives in `src/`, not `include/vllm/`, deliberately: `include/vllm/`
// is a `USER_USAGE_PREFIXES` surface (#515) and W3 ships nothing on the public
// ABI. Same call as the W1 resolver header.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
//   OURS                       <-  UPSTREAM
//   NemotronHParams            <-  transformers `NemotronHConfig`
//                                  (models/nemotron_h/configuration_nemotron_h.py
//                                  @ transformers 7d06b1a5 / 5.10.0.dev0):
//                                  field defaults :86-140, the legacy-alias
//                                  normalization + pattern fallbacks in
//                                  `__post_init__` :142-190, `validate_layer_type`
//                                  :195-223, `num_hidden_layers` as a PROPERTY
//                                  over `layers_block_type` :225-238.
//                                  vLLM vendors its OWN NemotronHConfig
//                                  (transformers_utils/configs/nemotron_h.py
//                                  @ 555967922) in which the polarity is
//                                  REVERSED — `hybrid_override_pattern` is the
//                                  ctor arg and `layers_block_type` the derived
//                                  property (:277-287) — but that class is
//                                  imported by `nemotron_h.py:83` for TYPE
//                                  ANNOTATION only; the object that actually
//                                  reaches the model at runtime comes from
//                                  transformers `AutoConfig`. The spec's live
//                                  oracle run (§5a) settles it on the real
//                                  checkpoint: `CONFIG pattern =
//                                  MEMEM*EMEMEM*EM...` was DERIVED from
//                                  `layers_block_type`, which is what the
//                                  released config.json ships. So
//                                  `layers_block_type` is the source of truth
//                                  here and `hybrid_override_pattern` is the
//                                  legacy alias, not the other way round.
//   EnumerateNemotronHTensors  <-  nemotron_h.py:86-123 (NemotronHMLP up/down),
//                                  :126-256 (NemotronHMoE gate +
//                                  e_score_correction_bias :158, shared_experts
//                                  :181, FusedMoE ckpt_names
//                                  ("up_proj","down_proj","") :220),
//                                  :373-389 (MambaMixer2 fed
//                                  mamba_num_heads*mamba_head_dim), :503
//                                  (NemotronHAttention), the `.mixer` prefix on
//                                  every block (:290, :335, :373, :503), and the
//                                  on-disk mapper `hf_to_vllm_mapper`
//                                  (:716-724: prefix `backbone.` -> `model.`,
//                                  substr `A_log` -> `A`, `embeddings` ->
//                                  `embed_tokens`, and q/k/v STACKED into
//                                  qkv_proj at load — so the CHECKPOINT ships
//                                  them SEPARATE, which is what we enumerate).
//                                  MTP: nemotron_h_mtp.py:40-90
//                                  (enorm/hnorm/eh_proj under
//                                  has_start_projections, final_layernorm under
//                                  has_end_norm) + :241-275 (total_layers =
//                                  num_nextn_predict_layers * pattern_len, char
//                                  '*' -> attention, 'E' -> moe).
//   MakeNemotronHKVCache       <-  nemotron_h.py:761-798
//                                  (get_mamba_state_shape_from_config:
//                                  intermediate_size = mamba_num_heads *
//                                  mamba_head_dim) ->
//                                  mamba_utils.py:173-198 (mamba2_state_shape)
//                                  and :743-758 (get_mamba_state_dtype_from_config)
//                                  -> mamba_utils.py:93-106 (_mamba_state_dtype).
//   NemotronHModel::Forward    <-  nemotron_h.py:604-660 — REFUSE-by-name (W4).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {

// The four decoder block kinds. Mirror of `ALL_DECODER_LAYER_TYPES`
// (nemotron_h.py:531-536) and of the pattern mapping
// {"M": mamba, "E": moe, "*": attention, "-": mlp}
// (configuration_nemotron_h.py:259-268).
enum class NemotronHBlock { kMamba, kAttention, kMoe, kMlp };

// The `layers_block_type` spelling of a block, as it appears in config.json.
std::string_view NemotronHBlockName(NemotronHBlock block);

// The COARSE quantization surface — only the signals W3 needs to know WHICH
// tensors the checkpoint ships, never which kernel consumes them.
//
// Scope boundary, deliberate: resolving a per-module `quant_algo` out of
// `quantization_config.{config_groups,quantized_layers}` (5981 entries) is W1
// of this row (#517 W1) and is NOT on `main` at this commit. W3 therefore reads
// four small, unambiguous, individually falsifiable keys and derives the scale
// companions STRUCTURALLY (mamba projections -> FP8 pair, expert/lm_head ->
// NVFP4 pair, attention k/v -> fp8-KV scale, `mtp.*` -> unquantized). The
// enumeration gate then proves that structural derivation against the released
// 18487-tensor index, so a wrong rule fails loudly instead of silently.
struct NemotronHQuantSurface {
  bool present = false;      // a `quantization_config` object exists at all
  std::string quant_method;  // "modelopt"
  std::string quant_algo;    // "MIXED_PRECISION"
  // `kv_cache_scheme` = {"dynamic": false, "num_bits": 8, "type": "float"} —
  // the fp8 KV scheme, which is what puts `k_proj.k_scale` / `v_proj.v_scale`
  // on a quantized attention layer.
  bool fp8_kv_cache = false;
  // The `ignore` list carries the `mtp*` wildcard, so the whole MTP tower ships
  // UNQUANTIZED (bf16, no scale companions) even though its backbone twin does
  // not. Enumerating scales for it would over-claim 270 tensors.
  bool mtp_ignored = false;
};

// Every NemotronH config field the enumeration / KV-cache builder / (W4)
// forward consume, resolved ONCE from the standalone HfConfig. There is no
// `text_config` nesting: `NemotronHForCausalLM` is the top-level architecture.
struct NemotronHParams {
  // --- shared geometry ---
  int64_t hidden_size = 0;              // 2688
  int64_t vocab_size = 0;               // 131072
  int64_t max_position_embeddings = 0;  // 1048576
  double layer_norm_epsilon = 1e-5;
  bool tie_word_embeddings = false;

  // --- the layer schedule (the SOURCE OF TRUTH for depth) ---
  // 52 entries: 23 mamba, 23 moe, 6 attention at indices 5,12,19,26,33,42.
  // `num_hidden_layers` in the released config is DEPRECATED and ignored by
  // the runtime config's setter (configuration_nemotron_h.py:233-238); we
  // mirror that and only WARN-equivalent (refuse) on a genuine conflict.
  std::vector<NemotronHBlock> layers_block_type;
  // The MTP head's own schedule; ["attention","moe"] on this checkpoint and by
  // default (configuration_nemotron_h.py:179-180).
  std::vector<NemotronHBlock> mtp_layers_block_type;
  int64_t num_nextn_predict_layers = 0;  // 1

  // --- attention (32 q / 2 kv heads, head_dim 128) ---
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  double rope_theta = 10000.0;
  double partial_rotary_factor = 1.0;
  bool attention_bias = false;
  std::optional<int64_t> sliding_window = std::nullopt;

  // --- Mamba2 ---
  // Each `legacy alias` below WINS over the modern spelling when both ship —
  // `self.n_groups = kwargs.pop("mamba_n_groups") if "mamba_n_groups" in kwargs
  // else self.n_groups` (configuration_nemotron_h.py:145-155) overwrites the
  // already-populated dataclass field. That is the OPPOSITE of the two SCHEDULE
  // pairs (`layers_block_type`/`hybrid_override_pattern` and the `mtp_` pair,
  // :158-184), where the legacy pattern is consulted only when the modern list
  // is absent. Upstream disagrees with itself between the two families; both
  // polarities are mirrored as-is and pinned by the "PER-FAMILY" test case.
  int64_t mamba_num_heads = 0;  // 64
  int64_t mamba_head_dim = 0;   // 64
  int64_t n_groups = 0;         // 8   (legacy alias `mamba_n_groups`)
  int64_t ssm_state_size = 0;   // 128
  int64_t conv_kernel = 0;      // 4   (legacy alias `mamba_d_conv`)
  int64_t chunk_size = 0;       // 128 (legacy alias `mamba_chunk_size`)
  int64_t expand = 0;           // 2   (legacy alias `mamba_expand`)
  std::string mamba_hidden_act = "silu";
  // The recurrent (SSM) state cache dtype, resolved INDEPENDENTLY of the
  // convolution-state dtype (mamba_utils.py:99-104). "float32" here.
  std::string mamba_ssm_cache_dtype;
  bool use_conv_bias = true;    // legacy alias `mamba_conv_bias`
  bool use_bias = false;
  bool mamba_proj_bias = false;
  double time_step_min = 1e-3;   // legacy alias `mamba_dt_min`
  double time_step_max = 1e-1;   // legacy alias `mamba_dt_max`
  double time_step_floor = 1e-4; // legacy alias `mamba_dt_init_floor`

  // --- MoE ---
  int64_t n_routed_experts = 0;                   // 128
  int64_t num_experts_per_tok = 0;                // 6
  int64_t moe_intermediate_size = 0;              // 1856
  int64_t n_shared_experts = 0;                   // 1
  int64_t moe_shared_expert_intermediate_size = 0;  // 3712
  int64_t n_group = 1;
  int64_t topk_group = 1;
  double routed_scaling_factor = 1.0;  // 2.5, applied to the OUTPUT (:246)
  bool norm_topk_prob = true;
  bool moe_shared_expert_overlap = true;
  // `moe_latent_size` is `null` in this checkpoint. Upstream's predicate is
  // `getattr(config, "moe_latent_size", None) is not None` (nemotron_h.py:143),
  // so an ABSENT key and an explicit `null` are the SAME state — both mean "no
  // latent MoE". nullopt covers both; a real value REFUSES (spec §0 puts
  // `fc1_latent_proj`/`fc2_latent_proj` out of scope).
  std::optional<int64_t> moe_latent_size = std::nullopt;

  // --- dense MLP block (`-`); this checkpoint has none ---
  int64_t intermediate_size = 0;
  bool mlp_bias = false;
  std::string mlp_hidden_act = "relu2";

  NemotronHQuantSurface quant;

  // Depth is the SCHEDULE's length, never the deprecated scalar
  // (configuration_nemotron_h.py:225-238).
  int64_t num_hidden_layers() const {
    return static_cast<int64_t>(layers_block_type.size());
  }
  // MambaMixer2 is fed `mamba_num_heads * mamba_head_dim` (nemotron_h.py:377):
  // 64 * 64 = 4096. This is the SSM `intermediate_size`, NOT the MLP one.
  int64_t mamba_intermediate_size() const {
    return mamba_num_heads * mamba_head_dim;
  }
  // mamba_utils.py:190 `conv_dim = intermediate_size + 2 * n_groups *
  // state_size` = 4096 + 2*8*128 = 6144. Falsifiable straight off disk: the
  // released `mixer.conv1d.weight` is BF16 [6144, 1, 4].
  int64_t conv_dim() const {
    return mamba_intermediate_size() + 2 * n_groups * ssm_state_size;
  }
  // The fused zxbcdt projection width: z (intermediate) + xBC (conv_dim) + dt
  // (num_heads) = 4096 + 6144 + 64 = 10304. Released `mixer.in_proj.weight` is
  // F8_E4M3 [10304, 2688].
  int64_t in_proj_out_features() const {
    return mamba_intermediate_size() + conv_dim() + mamba_num_heads;
  }
  int64_t q_proj_out_features() const { return num_attention_heads * head_dim; }
  int64_t kv_proj_out_features() const { return num_key_value_heads * head_dim; }
  // The layer indices carrying `block`, in ascending order.
  std::vector<int64_t> LayerIndices(NemotronHBlock block) const;
};

// Resolve + validate NemotronHParams from a standalone HfConfig. Pure/host, so
// it is unit-testable against the committed released-config fixture with no
// checkpoint present. Throws with a precise message naming the offending key on
// anything this bring-up cannot represent.
NemotronHParams ParseNemotronHParams(const HfConfig& config);

// Per-family config hook (registry `parse_config`). The resolve IS the
// validation.
void ParseNemotronHConfig(const HfConfig& config);

// The TEMPORAL/SSM state cache dtype — mirror of
// `MambaStateDtypeCalculator._mamba_state_dtype` (mamba_utils.py:93-106): it is
// resolved from `mamba_ssm_cache_dtype` INDEPENDENTLY of the convolution-state
// dtype, and only "auto"/absent falls back to `conv_dtype`.
//
// It is resolved from `NemotronHParams` rather than through the shared
// `detail::ResolveMambaSsmCacheDType` on purpose. That helper reads
// `HfConfig::mamba_ssm_dtype`, which `hf_config.cpp:439` parses from the key
// **`mamba_ssm_dtype`** — Qwen3.5/3.6's spelling. NemotronH ships
// **`mamba_ssm_cache_dtype`** (transformers' own field name,
// configuration_nemotron_h.py:121), so the shared helper sees an empty string
// here and silently returns the CONVOLUTION dtype. That is precisely the
// "recurrent state collapsed to the activation dtype" defect: numerically
// plausible, half the bytes, and invisible to a token gate.
vt::DType NemotronHSsmCacheDType(const NemotronHParams& params,
                                 vt::DType conv_dtype);

// One enumerated checkpoint tensor and the named thing that consumes it. A
// tensor with no entry here is UNCLAIMED, which the enumeration gate reports as
// a failure rather than tolerating — "nobody thought of it" is not a state.
struct NemotronHTensor {
  std::string name;      // the name as it ships on disk
  std::string consumer;  // stable tag: what will read it
};

// The on-disk weight name map of `NemotronHForCausalLM`, by the names the
// checkpoint SHIPS (q/k/v separate, `backbone.` prefix, `A_log` not `A`) rather
// than the upstream module names. Ordered: root, backbone layers ascending,
// then the MTP tower.
std::vector<NemotronHTensor> EnumerateNemotronHTensors(
    const NemotronHParams& params);

// The HETEROGENEOUS KV topology: TWO groups.
//   (1) a full-attention paged group over the 6 GQA layers
//       (num_kv_heads=2, head_size=128);
//   (2) a Mamba2 recurrent-state group over the 23 mamba layers
//       (mamba_utils.py:173-198 / :93-106).
// Both groups carry their REAL per-layer names, not a single tag: a mamba
// group's page size is multiplied by `layer_names.size()` when the runner sizes
// recurrent state (kv_cache_utils.cpp:979), and an attention group's page by
// the same count when the block budget is computed
// (kv_cache_interface.cpp:151-158). A one-element tag would under-count both by
// 23x and 6x.
v1::KVCacheConfig MakeNemotronHKVCache(const HfConfig& config, int block_size,
                                       int num_blocks);

}  // namespace vllm
