// CLM (Contrastive-LM) System 1 decision model — registry (MODEL-CLM).
//
// Bi-encoder: Qwen3-8B frozen backbone (last-token pooling) + two MLP
// projection heads (state head + action head), L2-normalized, scored via
// scaled cosine similarity.
//
// Ported from Contrastive-LM/CLM-v0.1-8B (spec .agents/specs/clm.md).

#include "vllm/model_executor/models/clm.h"
#include "vllm/model_executor/models/clm_inference.h"

#include "vllm/model_executor/model_loader/safetensors_reader.h"  // SafetensorsFile, StTensor
#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/tokenizer/tokenizer.h"  // tok::Tokenizer
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"
#include "vt/tensor.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace vllm {
namespace {

// ── ModelInfo ────────────────────────────────────────────────────────────
inline constexpr ModelInfo kClmInfo{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// ── Loaded model ──────────────────────────────────────────────────────────
class ClmLoadedModel final : public LoadedModel {
 public:
  ClmLoadedModel(const ModelRegistration& registration,
                 Qwen3DenseWeights weights,
                 clm::HeadWeights head_weights,
                 clm::HeadParams head_params,
                 HfConfig config)
      : LoadedModel(registration),
        weights_(std::move(weights)),
        head_weights_(std::move(head_weights)),
        head_params_(head_params),
        config_(std::move(config)) {}

  const Qwen3DenseWeights& weights() const { return weights_; }
  const clm::HeadWeights& head_weights() const { return head_weights_; }
  const clm::HeadParams& head_params() const { return head_params_; }
  const HfConfig& config() const { return config_; }

  void set_queue(vt::Queue q) { queue_ = q; }
  vt::Queue queue() const { return queue_; }

 private:
  Qwen3DenseWeights weights_;
  clm::HeadWeights head_weights_;
  clm::HeadParams head_params_;
  HfConfig config_;
  vt::Queue queue_;
};

// ── Head weight loading ───────────────────────────────────────────────────
// Load the dual MLP head weights from head.safetensors. The tensor names
// follow the PyTorch module attribute names:
//   state_head.0.weight / state_head.0.bias  (Linear 4096→1536)
//   state_head.2.weight / state_head.2.bias  (LayerNorm 1536)
//   state_head.4.weight / state_head.4.bias  (Linear 1536→1536)
//   state_head.6.weight / state_head.6.bias  (Linear 1536→512)
//   action_head.* (same layout)
// All tensors are F32, row-major (PyTorch nn.Linear convention).
clm::HeadWeights LoadHeadWeights(const std::vector<SafetensorsFile>& shards) {
  clm::HeadWeights hw;

  auto load_mlp = [&](const std::string& prefix, clm::MlpHeadWeights& m) {
    for (const SafetensorsFile& shard : shards) {
      for (const std::string& name : shard.Names()) {
        const StTensor& t = shard.Get(name);
        if (t.dtype != "F32") continue;
        size_t count = t.nbytes / sizeof(float);
        std::vector<float> dst(count);
        std::memcpy(dst.data(), t.data, t.nbytes);

        if (name == prefix + ".0.weight") m.w0 = std::move(dst);
        else if (name == prefix + ".0.bias") m.b0 = std::move(dst);
        else if (name == prefix + ".2.weight") m.w2 = std::move(dst);
        else if (name == prefix + ".2.bias") m.b2 = std::move(dst);
        else if (name == prefix + ".4.weight") m.w4 = std::move(dst);
        else if (name == prefix + ".4.bias") m.b4 = std::move(dst);
        else if (name == prefix + ".6.weight") m.w6 = std::move(dst);
        else if (name == prefix + ".6.bias") m.b6 = std::move(dst);
      }
    }
  };

  load_mlp("state_head", hw.state_head);
  load_mlp("action_head", hw.action_head);

  // Validate all tensors are present
  auto check = [](const clm::MlpHeadWeights& m, const char* name) {
    if (m.w0.empty() || m.b0.empty() || m.w2.empty() || m.b2.empty() ||
        m.w4.empty() || m.b4.empty() || m.w6.empty() || m.b6.empty()) {
      throw std::runtime_error(
          std::string("clm: head.safetensors incomplete for ") + name +
          " (expected state_head/action_head .0/.2/.4/.6 weight+bias). "
          "Run scripts/convert-clm.py first.");
    }
  };
  check(hw.state_head, "state_head");
  check(hw.action_head, "action_head");

  return hw;
}

// ── Load ──────────────────────────────────────────────────────────────────
std::unique_ptr<LoadedModel> LoadClm(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture ClmModel does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }

  // Load merged Qwen3 dense weights (Qwen3-8B backbone).
  Qwen3DenseWeights weights =
      LoadQwen3ForCausalLMWeights(*source.safetensors, config);

  // Load dual MLP head weights from head.safetensors.
  clm::HeadWeights head_weights = LoadHeadWeights(*source.safetensors);

  // Read head params from config (written by convert-clm.py).
  clm::HeadParams head_params;
  head_params.hidden_size = config.hidden_size;
  if (config.raw.contains("clm_proj_dim")) {
    head_params.proj_dim = config.raw["clm_proj_dim"].get<int64_t>();
  }
  if (config.raw.contains("clm_embed_dim")) {
    head_params.embed_dim = config.raw["clm_embed_dim"].get<int64_t>();
  }
  if (config.raw.contains("clm_logit_scale")) {
    head_params.logit_scale = config.raw["clm_logit_scale"].get<float>();
  }

  return std::make_unique<ClmLoadedModel>(
      registration, std::move(weights), std::move(head_weights),
      head_params, config);
}

// ── Prepare ──────────────────────────────────────────────────────────────
void PrepareClm(LoadedModel& model, const HfConfig& config,
                vt::Queue& queue) {
  (void)config;
  ModelAs<ClmLoadedModel>(model, "ClmModel").set_queue(queue);
}

// ── Forward (factory) ─────────────────────────────────────────────────────
// The factory forward delegates to Qwen3DenseModel::ForwardHidden, using the
// runner-provided attn_meta / attn_kv (same pattern as llama_embedding_registry).
ForwardLogits ForwardClm(LoadedModel& model, const ModelForwardInput& input) {
  auto& clm = ModelAs<ClmLoadedModel>(model, "ClmModel");
  return Qwen3DenseModel::ForwardHidden(
      input.token_ids, input.positions, input.attn_meta, input.attn_kv,
      clm.weights(), input.config, input.queue, input.logits_indices);
}

const ModelFactory kClmFactory{
    .parse_config = &ParseQwen3ForCausalLMConfig,
    .load_weights = &LoadClm,
    .prepare = &PrepareClm,
    .forward = &ForwardClm,
    .make_kv_cache = &MakeQwen3ForCausalLMKVCache,
    .is_dense_model = true,
};

// ── Single-seq hidden forward for inference ──────────────────────────────
// Constructs minimal CommonAttentionMetadata + PagedKvCache for a single
// prefill, then calls Qwen3DenseModel::ForwardHidden. This is the CLM
// analogue of kev's direct ForwardDenseHidden call — needed because Qwen3
// dense (unlike Qwen3.5) has no single-seq ForwardDenseHidden helper.
std::vector<float> ForwardClmHidden(
    const std::vector<int32_t>& token_ids,
    const Qwen3DenseWeights& weights,
    const HfConfig& config, vt::Queue& queue) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T == 0) return {};

  // Positions [0, T)
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);

  // Construct CommonAttentionMetadata for a single-seq prefill.
  const int block_size = 32;
  const int num_blocks =
      static_cast<int>((T + block_size - 1) / block_size);
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.max_query_len = static_cast<int>(T);
  am.max_seq_len = static_cast<int>(T);
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.seq_lens = {static_cast<int32_t>(T)};
  am.causal = true;
  am.block_table_num_cols = num_blocks;
  am.block_table_tensor.resize(static_cast<size_t>(num_blocks));
  for (int b = 0; b < num_blocks; ++b)
    am.block_table_tensor[static_cast<size_t>(b)] = b;
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    am.slot_mapping[static_cast<size_t>(i)] = i;

  // Allocate PagedKvCache (one per layer). Each cache holds K and V for
  // [num_blocks, block_size, num_kv_heads, head_dim].
  dense_attn::Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t kv_elems_per_layer =
      static_cast<int64_t>(num_blocks) * block_size *
      num_kv_heads * head_dim * 2;  // *2 for K and V
  std::vector<PagedKvCache> attn_kv(
      static_cast<size_t>(config.num_hidden_layers));
  std::vector<dense_attn::DBuf> kv_bufs(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    kv_bufs[static_cast<size_t>(l)] =
        dense_attn::DBuf(d, dense_attn::DType::kBF16, {kv_elems_per_layer});
    kv_bufs[static_cast<size_t>(l)].Zero(d);
    auto& kv = attn_kv[static_cast<size_t>(l)];
    kv.data = kv_bufs[static_cast<size_t>(l)].ptr();
    kv.dtype = vt::DType::kBF16;
    kv.num_blocks = num_blocks;
    kv.block_size = block_size;
    kv.num_kv_heads = num_kv_heads;
    kv.head_size = head_dim;
  }

  // Forward and extract the last-token hidden [H] from [T, H].
  ForwardLogits fl = Qwen3DenseModel::ForwardHidden(
      token_ids, positions, am, attn_kv, weights, config, queue);
  return fl.host;  // [T, H] f32
}

}  // namespace

// ── Decision inference ────────────────────────────────────────────────────
// Full bi-encoder decision pipeline:
//   1. Tokenize state text and each option text separately.
//   2. Forward the Qwen3 dense backbone for the state text → last-token hidden [H].
//   3. Forward the Qwen3 dense backbone for each option text → last-token hidden [H].
//   4. Apply state-head MLP → L2-normalized [E] for the state hidden.
//   5. Apply action-head MLP → L2-normalized [E] for each option hidden.
//   6. Score: logit_k = exp(logit_scale) * dot(h_state, h_option_k).
//   7. Return pre-softmax scores; the server applies softmax + CLM confidence.
ClmDecisionResult ClmInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    const std::string& state,
    const std::string& qtype,
    const std::string& instructions,
    const std::vector<std::string>& options) {
  (void)qtype;  // raw logits are qtype-independent
  (void)instructions;  // CLM does not use a shared instruction field

  ClmDecisionResult result;
  if (options.empty()) return result;

  auto& m = ModelAs<ClmLoadedModel>(model, "ClmModel");
  vt::Queue queue = m.queue();
  const auto& params = m.head_params();

  // 1. Tokenize state text.
  std::vector<int32_t> state_tokens = tokenizer.Encode(state);

  // 2. Forward state text through backbone → last-token hidden [H].
  std::vector<float> state_hidden_all = ForwardClmHidden(
      state_tokens, m.weights(), m.config(), queue);
  const int64_t H = m.config().hidden_size;
  const int64_t T_state = static_cast<int64_t>(state_tokens.size());
  if (T_state == 0) return result;

  // Extract last-token hidden: row (T_state - 1)
  std::vector<float> state_hidden(static_cast<size_t>(H));
  const float* last_row = state_hidden_all.data() +
      static_cast<size_t>(T_state - 1) * static_cast<size_t>(H);
  std::copy_n(last_row, static_cast<size_t>(H), state_hidden.data());

  // 3. Apply state-head MLP → L2-normalized [E].
  std::vector<float> h_state =
      clm::ClmMlpHeadForward(m.head_weights().state_head, params, state_hidden);

  // 4. Forward each option, extract last-token hidden, apply action-head MLP.
  std::vector<std::vector<float>> h_options;
  h_options.reserve(options.size());
  int64_t total_tokens = T_state;

  for (const std::string& opt : options) {
    std::vector<int32_t> opt_tokens = tokenizer.Encode(opt);
    int64_t T_opt = static_cast<int64_t>(opt_tokens.size());
    if (T_opt == 0) {
      h_options.emplace_back(static_cast<size_t>(params.embed_dim), 0.0F);
      continue;
    }

    std::vector<float> opt_hidden_all = ForwardClmHidden(
        opt_tokens, m.weights(), m.config(), queue);
    total_tokens += T_opt;

    // Extract last-token hidden
    std::vector<float> opt_hidden(static_cast<size_t>(H));
    const float* opt_last = opt_hidden_all.data() +
        static_cast<size_t>(T_opt - 1) * static_cast<size_t>(H);
    std::copy_n(opt_last, static_cast<size_t>(H), opt_hidden.data());

    // Apply action-head MLP → L2-normalized [E]
    h_options.push_back(
        clm::ClmMlpHeadForward(m.head_weights().action_head, params, opt_hidden));
  }

  // 5. Score: logit_k = exp(logit_scale) * dot(h_state, h_option_k)
  result.scores = clm::ClmScoreOptions(h_state, h_options, params.logit_scale);
  result.prompt_tokens = total_tokens;
  return result;
}

REGISTER_VLLM_MODEL(clm_model, "ClmModel", kClmFactory, kClmInfo)

}  // namespace vllm
