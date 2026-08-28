// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — W1 config surface.
//
// Model-private header, deliberately not under include/: nothing outside this
// model needs these types yet, and `include/vllm.h` is the ABI seam a shipped
// capability is exposed through. W1 ships no capability above the config layer.
//
// ORACLE. vLLM implements `glm5_next` at NO revision — not at our parity pin
// `555967922` and not at vLLM `main`; `vllm-project/vllm#53906` would register
// it and is OPEN, and an unmerged pull request is not a revision. SGLang,
// vllm-omni and llama.cpp implement nothing either. Under AGENTS.md "When vLLM
// has no implementation" the sole admissible reference for this surface is
// **transformers `v5.16.1`** (implementing commit `eb4d9e2a64`; `v5.16.0` is
// 404 for this model), and every anchor below is
// `transformers/models/glm5_next/configuration_glm5_next.py` at that tag,
// which is the flattened, EXECUTED class rather than the modular file. Reading
// the modular file and assuming its inheritance survives into the generated
// class is the error that cost `MODEL-MM-QWEN4-EXP` a review cycle. The lane
// pin itself is W0's deliverable and is not yet recorded; see
// `.agents/specs/glm5-next-flash.md` `## Oracles`.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_H_

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Per-layer attention kind AFTER upstream's rewrite. `__post_init__` replaces
// every `full_attention` entry with `deepseek_sparse_attention`
// (configuration_glm5_next.py, the list comprehension immediately after the
// `layer_types is None` default). There is deliberately no `kFullAttention`
// enumerator: a reader that takes `full_attention` at face value wires a dense
// attention block on a layer that actually runs the DSA indexer, and does it
// WITHOUT SAYING SO. The state is unrepresentable rather than merely unused.
enum class Glm5NextLayerKind {
  kLinearAttention,         // KDA — 34 of 45 layers
  kDeepseekSparseAttention  // NoPE MLA + the k-pool DSA indexer — 11 layers
};

// Per-layer feed-forward kind. `dense` on layers 0-2, `sparse` thereafter on
// the published checkpoint. NOTE that `first_k_dense_replace` is NOT the
// source: the RUNTIME class `Glm5NextTextConfig(PreTrainedConfig)` in
// `configuration_glm5_next.py` @ v5.16.1 does not declare the field and
// `__post_init__` never reads it -- the name does not occur in that file at
// all. It is removed one level up, in `modular_glm5_next.py:169`, where the
// class still has a `GlmMoeDsaConfig` parent that carries the field and
// `first_k_dense_replace = AttributeError()` deletes the inherited attribute;
// that is the MODULAR CONVERTER's convention and not a `@strict` one
// (`utils/modular_model_converter.py:1064-1066`, "delete unnecessary cls
// attribute, especially in configs"), and `mlp_bias` and `rope_parameters` go
// the same way on the two lines above it. Either way the checkpoint's
// `first_k_dense_replace: 3` is an inert extra kwarg that merely happens to
// agree with the `min(3, num_hidden_layers)` default.
enum class Glm5NextMlpKind { kDense, kSparse };

// Per-layer DSA indexer mode. `full` runs the indexer; `shared` reuses the
// previous full layer's top-k selection. All 45 are `full` on the published
// checkpoint, and the schedule that produces that is upstream's freq/offset
// default (`index_topk_freq` 1), not a hardcode.
enum class Glm5NextIndexerKind { kFull, kShared };

// The NoPE MLA geometry. Every field is the published checkpoint's value and
// none of them is a default this port chose.
struct Glm5NextMlaParams {
  int64_t q_lora_rank = 0;        // 1536 — the supported DeepSeek-V3 branch
  int64_t kv_lora_rank = 0;       // 512
  int64_t qk_nope_head_dim = 0;   // 256
  // ZERO, and upstream REQUIRES it to be zero: `validate_architecture` raises
  // "Expecting NoPE for the DSA attention layers, but got {n} as RoPE dim."
  // for any positive value. `MlaBlockDims::Validate` raises for any value that
  // is not positive. The two validators are exact complements over this field
  // and no value satisfies both; W1 mirrors upstream, and the relaxation of
  // ours is W3's (O11 in the spec).
  int64_t qk_rope_head_dim = 0;
  int64_t v_head_dim = 0;         // 256
  // Upstream's forced overrides, computed and never read from the config:
  // `head_dim = qk_rope_head_dim` and
  // `qk_head_dim = qk_rope_head_dim + qk_nope_head_dim`. `head_dim` is
  // therefore 0 here, which is what the checkpoint's `head_dim: 0` records.
  int64_t qk_head_dim = 0;        // 256
  int64_t head_dim = 0;           // 0
};

// The DSA lightning indexer plus this model's net-new k-pool compression stage.
struct Glm5NextIndexerParams {
  int64_t head_dim = 0;   // index_head_dim = 128
  int64_t n_heads = 0;    // index_n_heads = 32
  int64_t topk = 0;       // index_topk = 2048
  // FOUR on the published checkpoint and SIXTEEN as the class default, so a
  // reader that defaults instead of reading is wrong by a factor of four.
  int64_t kpool = 0;
  bool kpool_always_select_tail = true;

  // topk / kpool = 512. Derived, never read. Refuses rather than dividing:
  // `index_kpool` is validated `>= 1` at parse, but this type is also built by
  // the GGUF path and by tests, and `x / 0` is SIGFPE on x86 — a crash, not a
  // refusal, and one no downstream gate would attribute to this config.
  int64_t max_select_pools() const {
    if (kpool <= 0) {
      throw std::runtime_error(
          "glm5_next: max_select_pools() needs a positive `index_kpool`, got " +
          std::to_string(kpool) + ".");
    }
    return topk / kpool;
  }
};

// KDA linear attention. Its geometry arrives through the `linear_attn_config`
// SUB-OBJECT under four different spellings, remapped by `__post_init__`.
struct Glm5NextKdaParams {
  int64_t num_heads = 0;         // linear_num_heads     <- `num_heads` 64
  int64_t head_dim = 0;          // linear_head_dim      <- `head_dim` 128
  int64_t conv_kernel_dim = 0;   // linear_conv_kernel_dim <- `short_conv_kernel_size` 4

  // `linear_lower_bound` <- `gate_lower_bound`, and it is OPTIONAL upstream
  // (`float | None`). Its presence SELECTS THE FORGET-GATE FORMULA, which is
  // why it is an optional here and not a double with a sentinel:
  //
  //   present (-5.0 here): -bound * sigmoid(exp(A_log) * (g + dt_bias))
  //   absent:              -exp(A_log) * softplus(g + dt_bias)
  //
  // Those are different functions of the same inputs, both smooth, both
  // fluent, and the SIGN of `decay_rate` differs between them. Our
  // Kimi-Linear KDA (`kimi_kda.cpp:60`) implements the second. W2 owes the
  // first. A token gate cannot see which one ran.
  std::optional<double> lower_bound;

  // Upstream's `safe_gate` rule, mirrored: when the `linear_attn_config` dict
  // is present, `safe_gate` defaults to TRUE, and a true `safe_gate` with an
  // absent lower bound installs -5.0. So an explicit `"gate_lower_bound": null`
  // does NOT reach the softplus branch unless `safe_gate` is explicitly false.
  bool takes_sigmoid_branch() const { return lower_bound.has_value(); }
};

// The multi-stream hyper-connection residual manifold.
struct Glm5NextMhcParams {
  int64_t mult = 0;            // hc_mult = 4 residual streams
  int64_t sinkhorn_iters = 0;  // hc_sinkhorn_iters = 20
  // 1e-6, and it is a DIFFERENT constant from `rms_norm_eps` (1e-5). It is
  // ADDED to every Sinkhorn denominator rather than used as a floor.
  double eps = 0.0;
};

struct Glm5NextMoeParams {
  int64_t n_routed_experts = 0;      // 288
  int64_t n_shared_experts = 0;      // 1
  int64_t num_experts_per_tok = 0;   // 8
  int64_t moe_intermediate_size = 0; // 2048
  int64_t n_group = 0;               // 1 — with topk_group 1 the group stage
  int64_t topk_group = 0;            //     is a NO-OP; do not implement it
  double routed_scaling_factor = 0.0;  // 2.5
  bool norm_topk_prob = true;
};

struct Glm5NextVisionParams {
  int64_t depth = 0;                       // 24
  int64_t hidden_size = 0;                 // 1024
  int64_t intermediate_size = 0;           // 4096
  int64_t num_heads = 0;                   // 16
  int64_t in_channels = 0;                 // 3
  int64_t patch_size = 0;                  // 14
  int64_t image_size = 0;                  // 448 — the class default is 336
  int64_t spatial_merge_size = 0;          // 2
  int64_t temporal_patch_size = 0;         // 2
  int64_t out_hidden_size = 0;             // 4096 — the class default is 1536
  int64_t projection_intermediate_size = 0;  // 10240
  double rms_norm_eps = 0.0;               // 1e-5
  double swiglu_limit = 0.0;               // 10.0
  bool attention_bias = true;              // TRUE here, false in the text stack
};

// Placeholder token ids. Image and video SHARE one id: the processor emits
// `image_token_id` for video frames too and disambiguates by the
// begin/end-of-video SPAN, so all six travel together or a reader classifies
// every frame as an image.
struct Glm5NextMmTokens {
  int64_t image = 0;        // 154854
  int64_t video = 0;        // 154855
  int64_t image_start = 0;  // 154830
  int64_t image_end = 0;    // 154831
  int64_t video_start = 0;  // 154832
  int64_t video_end = 0;    // 154833
};

struct Glm5NextParams {
  // --- geometry ---
  int64_t hidden_size = 0;        // 4096
  int64_t num_hidden_layers = 0;  // 45
  int64_t intermediate_size = 0;  // 12288 — the DENSE MLP width
  int64_t vocab_size = 0;         // 154880
  int64_t num_attention_heads = 0;  // 64
  int64_t num_key_value_heads = 0;  // 64 — upstream REQUIRES these two equal
  int64_t max_position_embeddings = 0;  // 1048576
  double rms_norm_eps = 1e-5;     // 1e-5, NOT the 1e-6 the GlmMoeDsa parent uses
  double swiglu_limit = 10.0;     // the SwiGLU clamp, in five places
  bool tie_word_embeddings = false;

  // --- the per-layer schedules, all three exactly num_hidden_layers long ---
  std::vector<Glm5NextLayerKind> layer_types;
  std::vector<Glm5NextMlpKind> mlp_layer_types;
  std::vector<Glm5NextIndexerKind> indexer_types;

  Glm5NextMlaParams mla;
  Glm5NextIndexerParams indexer;
  Glm5NextKdaParams kda;
  Glm5NextMhcParams mhc;
  Glm5NextMoeParams moe;

  // --- multimodal ---
  bool has_vision = false;
  Glm5NextVisionParams vision;
  Glm5NextMmTokens mm_tokens;

  // --- quantization, from the top-level `quantization_config` ---------------
  // The published checkpoint is FP8 e4m3 block-quantized at [128, 128] with
  // `weight_scale_inv` companions and a `modules_to_not_convert` list that
  // includes `hyper_connection`, so the mHC parameters ship unquantised.
  std::string quant_method;       // "fp8" on the published checkpoint, else ""
  std::string quant_fmt;          // "e4m3"
  std::vector<int64_t> weight_block_size;  // {128, 128}
  std::vector<std::string> modules_to_not_convert;

  int64_t num_kda_layers() const;
  int64_t num_dsa_layers() const;
  // The residual manifold is `hc_mult * hidden_size` wide through the WHOLE
  // stack: 4 * 4096 = 16384 here.
  int64_t residual_stream_width() const { return mhc.mult * hidden_size; }
};

// Resolves and VALIDATES. The resolve IS the validation: it throws by name on
// everything unrepresentable, mirroring upstream `__post_init__` and all five
// `validate_architecture` rejections.
Glm5NextParams ParseGlm5NextParams(const HfConfig& config);

// ModelFactory::parse_config hook. Delegates to ParseGlm5NextParams and
// discards the result, so a malformed config is refused at load rather than at
// first forward.
void ParseGlm5NextConfig(const HfConfig& config);

// The names, for messages and for the GGUF metadata round trip. One spelling,
// one definition; `Kind(Name(k)) == k` for every enumerator.
const char* Glm5NextLayerKindName(Glm5NextLayerKind kind);
const char* Glm5NextMlpKindName(Glm5NextMlpKind kind);
const char* Glm5NextIndexerKindName(Glm5NextIndexerKind kind);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_H_
