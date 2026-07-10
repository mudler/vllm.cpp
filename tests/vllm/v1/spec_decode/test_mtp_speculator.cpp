// Ported from upstream vLLM @ e24d1b24fe96a56ba8b0d653efa076d03eb95d6c:
//   tests/v1/spec_decode/test_mtp.py:67-221
//   tests/v1/worker/test_gpu_autoregressive_speculator.py:52-82
// Loader sharing and direct-hidden-return assertions land with M-mtp-0. The
// propose-loop-only assertions remain explicitly skipped until M-mtp-1 adds the
// scheduler/speculator plumbing, as required by .agents/test-porting.md rule 6.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseWeights;
using vllm::Qwen3_5MoeWeights;
using vllm::Qwen3_5MTPKind;
using vllm::Qwen3_5MTPModel;
using vllm::Qwen3_5MTPWeights;
using vllm::StTensor;
using vllm::TensorResolver;

struct StoredTensor {
  std::vector<uint16_t> values;
  StTensor view;
};

class TensorStore {
 public:
  void Add(const std::string& name, std::vector<int64_t> shape,
           uint16_t seed) {
    int64_t numel = 1;
    for (int64_t dim : shape) numel *= dim;
    StoredTensor& stored = tensors_[name];
    stored.values.resize(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
      const int centered = static_cast<int>((i + seed) % 19) - 9;
      stored.values[static_cast<size_t>(i)] =
          vt::F32ToBF16(static_cast<float>(centered) * 0.0078125F);
    }
    stored.view.dtype = "BF16";
    stored.view.shape = std::move(shape);
    stored.view.data =
        reinterpret_cast<const uint8_t*>(stored.values.data());
    stored.view.nbytes = stored.values.size() * sizeof(uint16_t);
  }

  const StTensor& Get(const std::string& name) const {
    return tensors_.at(name).view;
  }

  const StoredTensor& Stored(const std::string& name) const {
    return tensors_.at(name);
  }

  StTensor& MutableView(const std::string& name) {
    return tensors_.at(name).view;
  }

  TensorResolver Resolver() const {
    return [this](const std::string& name) -> const StTensor& {
      return Get(name);
    };
  }

 private:
  std::map<std::string, StoredTensor> tensors_;
};

OwnedTensor MakeOwned(std::vector<int64_t> shape, uint16_t seed,
                      bool raw_nk = false) {
  OwnedTensor out;
  out.dtype = vt::DType::kBF16;
  out.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < out.rank; ++i) {
    out.shape[i] = shape[static_cast<size_t>(i)];
    numel *= out.shape[i];
  }
  out.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* values = reinterpret_cast<uint16_t*>(out.bytes.data());
  for (int64_t i = 0; i < numel; ++i) {
    const int centered = static_cast<int>((i + seed) % 23) - 11;
    values[i] = vt::F32ToBF16(static_cast<float>(centered) * 0.005F);
  }
  out.nk = raw_nk;
  return out;
}

HfConfig MakeConfig(Qwen3_5MTPKind kind) {
  HfConfig config;
  config.model_type = kind == Qwen3_5MTPKind::kDense ? "qwen3_5" : "qwen3_5_moe";
  config.hidden_size = 4;
  config.num_hidden_layers = 2;
  config.vocab_size = 16;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.head_dim = 2;
  config.rotary_dim = 2;
  config.rope_theta = 10000.0;
  config.rms_norm_eps = 1e-5;
  config.max_position_embeddings = 32;
  config.intermediate_size = 6;
  if (kind == Qwen3_5MTPKind::kMoe) {
    config.num_experts = 2;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 3;
    config.shared_expert_intermediate_size = 3;
  }
  config.raw = {
      {"text_config",
       {{"mtp_num_hidden_layers", 1},
        {"mtp_use_dedicated_embeddings", false}}}};
  return config;
}

void AddCommonMtp(TensorStore& store, const HfConfig& config) {
  const int64_t hidden = config.hidden_size;
  const int64_t q_out =
      2 * config.num_attention_heads * config.head_dim;
  const int64_t kv_out = config.num_key_value_heads * config.head_dim;
  store.Add("mtp.fc.weight", {hidden, 2 * hidden}, 1);
  store.Add("mtp.pre_fc_norm_embedding.weight", {hidden}, 2);
  store.Add("mtp.pre_fc_norm_hidden.weight", {hidden}, 3);
  store.Add("mtp.layers.0.input_layernorm.weight", {hidden}, 4);
  store.Add("mtp.layers.0.self_attn.q_proj.weight", {q_out, hidden}, 5);
  store.Add("mtp.layers.0.self_attn.k_proj.weight", {kv_out, hidden}, 6);
  store.Add("mtp.layers.0.self_attn.v_proj.weight", {kv_out, hidden}, 7);
  store.Add("mtp.layers.0.self_attn.o_proj.weight",
            {hidden, config.num_attention_heads * config.head_dim}, 8);
  store.Add("mtp.layers.0.self_attn.q_norm.weight", {config.head_dim}, 9);
  store.Add("mtp.layers.0.self_attn.k_norm.weight", {config.head_dim}, 10);
  store.Add("mtp.layers.0.post_attention_layernorm.weight", {hidden}, 11);
  store.Add("mtp.norm.weight", {hidden}, 12);
}

void AddDenseMtp(TensorStore& store, const HfConfig& config) {
  AddCommonMtp(store, config);
  store.Add("mtp.layers.0.mlp.gate_proj.weight",
            {config.intermediate_size, config.hidden_size}, 13);
  store.Add("mtp.layers.0.mlp.up_proj.weight",
            {config.intermediate_size, config.hidden_size}, 14);
  store.Add("mtp.layers.0.mlp.down_proj.weight",
            {config.hidden_size, config.intermediate_size}, 15);
}

void AddMoeMtp(TensorStore& store, const HfConfig& config) {
  AddCommonMtp(store, config);
  store.Add("mtp.layers.0.mlp.gate.weight",
            {config.num_experts, config.hidden_size}, 13);
  store.Add("mtp.layers.0.mlp.experts.gate_up_proj",
            {config.num_experts, 2 * config.moe_intermediate_size,
             config.hidden_size},
            14);
  store.Add("mtp.layers.0.mlp.experts.down_proj",
            {config.num_experts, config.hidden_size,
             config.moe_intermediate_size},
            15);
  store.Add("mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            {config.shared_expert_intermediate_size, config.hidden_size}, 16);
  store.Add("mtp.layers.0.mlp.shared_expert.up_proj.weight",
            {config.shared_expert_intermediate_size, config.hidden_size}, 17);
  store.Add("mtp.layers.0.mlp.shared_expert.down_proj.weight",
            {config.hidden_size, config.shared_expert_intermediate_size}, 18);
  store.Add("mtp.layers.0.mlp.shared_expert_gate.weight",
            {1, config.hidden_size}, 19);
}

Qwen3_5DenseWeights MakeDenseTarget(const HfConfig& config) {
  Qwen3_5DenseWeights target;
  target.embed_tokens =
      MakeOwned({config.vocab_size, config.hidden_size}, 21);
  target.lm_head = MakeOwned({config.hidden_size, config.vocab_size}, 22);
  return target;
}

Qwen3_5MoeWeights MakeMoeTarget(const HfConfig& config) {
  Qwen3_5MoeWeights target;
  target.embed_tokens =
      MakeOwned({config.vocab_size, config.hidden_size}, 23);
  target.lm_head = MakeOwned({config.hidden_size, config.vocab_size}, 24);
  return target;
}

void CheckFinite(const std::vector<float>& values) {
  REQUIRE_FALSE(values.empty());
  for (float value : values) CHECK(std::isfinite(value));
}

}  // namespace

TEST_CASE("test_mtp_load_model_unified: dense MTP shares target embedding and lm_head") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  CHECK_FALSE(model.has_own_embed_tokens());
  CHECK_FALSE(model.has_own_lm_head());
  CHECK(&model.embed_tokens() == &target.embed_tokens);
  CHECK(model.lm_head() == &target.lm_head);
  CHECK(model.lm_head_fp4() == nullptr);
  CHECK(weights.NumLayers() == 1);
  REQUIRE(weights.dense_layers.size() == 1);
  CHECK(weights.fc.nk);
  CHECK(weights.fc.shape[0] == config.hidden_size);
  CHECK(weights.fc.shape[1] == 2 * config.hidden_size);
  CHECK_FALSE(weights.dense_layers[0].is_linear_attention);
  CHECK(weights.dense_layers[0].attn.q_proj.nk);
  CHECK(weights.dense_layers[0].mlp.down_proj.nk);
}

TEST_CASE("test_mtp_load_model_unified: MoE fused stacks split per expert and share target") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kMoe);
  TensorStore store;
  AddMoeMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), config, Qwen3_5MTPKind::kMoe);
  const Qwen3_5MoeWeights target = MakeMoeTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  CHECK(&model.embed_tokens() == &target.embed_tokens);
  CHECK(model.lm_head() == &target.lm_head);
  CHECK(model.lm_head_fp4() == &target.lm_head_fp4);
  REQUIRE(weights.moe_layers.size() == 1);
  const auto& moe = weights.moe_layers[0].moe;
  REQUIRE(moe.expert_gate.size() ==
          static_cast<size_t>(config.num_experts));
  REQUIRE(moe.expert_up.size() ==
          static_cast<size_t>(config.num_experts));
  REQUIRE(moe.expert_down.size() ==
          static_cast<size_t>(config.num_experts));
  CHECK(moe.expert_gate[0].shape[0] == config.moe_intermediate_size);
  CHECK(moe.expert_gate[0].shape[1] == config.hidden_size);
  CHECK(moe.expert_down[0].shape[0] == config.hidden_size);
  CHECK(moe.expert_down[0].shape[1] == config.moe_intermediate_size);

  const auto& gate_up =
      store.Stored("mtp.layers.0.mlp.experts.gate_up_proj").values;
  const int64_t hidden = config.hidden_size;
  const int64_t intermediate = config.moe_intermediate_size;
  const int64_t stride = 2 * intermediate * hidden;
  const auto* gate1 = reinterpret_cast<const uint16_t*>(
      moe.expert_gate[1].bytes.data());
  const auto* up0 =
      reinterpret_cast<const uint16_t*>(moe.expert_up[0].bytes.data());
  CHECK(gate1[0] == gate_up[static_cast<size_t>(stride)]);
  CHECK(up0[0] == gate_up[static_cast<size_t>(intermediate * hidden)]);
}

TEST_CASE("test_mtp_load_model_unified: every mtp tensor is strictly BF16") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  store.MutableView("mtp.fc.weight").dtype = "F16";
  CHECK_THROWS_AS(
      vllm::LoadQwen3_5MTP(store.Resolver(), config,
                           Qwen3_5MTPKind::kDense),
      std::runtime_error);
}

TEST_CASE("test_mtp_load_model_unified: wrong same-byte-count shape is rejected") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  // [q_out,H] is [8,4] for this config. Reversing the dimensions preserves
  // nbytes, proving the loader checks the upstream semantic shape and not just
  // storage size.
  store.MutableView("mtp.layers.0.self_attn.q_proj.weight").shape = {4, 8};
  CHECK_THROWS_AS(
      vllm::LoadQwen3_5MTP(store.Resolver(), config,
                           Qwen3_5MTPKind::kDense),
      std::runtime_error);
}

TEST_CASE("test_mtp_load_model_unified: dedicated embeddings are rejected for gate checkpoints") {
  HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  config.raw["text_config"]["mtp_use_dedicated_embeddings"] = true;
  TensorStore store;
  AddDenseMtp(store, config);
  CHECK_THROWS_AS(
      vllm::LoadQwen3_5MTP(store.Resolver(), config,
                           Qwen3_5MTPKind::kDense),
      std::runtime_error);
}

TEST_CASE("test_mtp_propose k=1: MTP forward returns hidden states directly") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  OwnedTensor target_hidden = MakeOwned({3, config.hidden_size}, 31);
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const std::vector<int32_t> input_ids = {1, 2, 3};
  const std::vector<int32_t> positions = {0, 1, 2};
  const auto hidden =
      model.Forward(input_ids, positions, target_hidden.View(), queue);
  REQUIRE(hidden.storage != nullptr);
  CHECK(hidden.tensor.rank == 2);
  CHECK(hidden.tensor.shape[0] == 3);
  CHECK(hidden.tensor.shape[1] == config.hidden_size);
  CHECK(hidden.tensor.dtype == vt::DType::kBF16);

  const vllm::ForwardLogits logits = model.ComputeLogits(hidden.tensor, queue);
  CHECK(logits.rows == 3);
  CHECK(logits.vocab == config.vocab_size);
  const std::vector<float> host = model.ForwardLogitsHost(
      input_ids, positions, target_hidden.View(), queue);
  CHECK(host.size() == static_cast<size_t>(3 * config.vocab_size));
  CheckFinite(host);
  backend.DestroyQueue(queue);
}

TEST_CASE("test_mtp_propose k=1: MoE MTP forward returns hidden states directly") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kMoe);
  TensorStore store;
  AddMoeMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), config, Qwen3_5MTPKind::kMoe);
  const Qwen3_5MoeWeights target = MakeMoeTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  OwnedTensor target_hidden = MakeOwned({3, config.hidden_size}, 37);
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const std::vector<int32_t> input_ids = {3, 4, 5};
  const std::vector<int32_t> positions = {0, 1, 2};
  const auto hidden =
      model.Forward(input_ids, positions, target_hidden.View(), queue);
  REQUIRE(hidden.storage != nullptr);
  CHECK(hidden.tensor.rank == 2);
  CHECK(hidden.tensor.shape[0] == 3);
  CHECK(hidden.tensor.shape[1] == config.hidden_size);
  CHECK(hidden.tensor.dtype == vt::DType::kBF16);

  const std::vector<float> host = model.ForwardLogitsHost(
      input_ids, positions, target_hidden.View(), queue);
  CHECK(host.size() == static_cast<size_t>(3 * config.vocab_size));
  CheckFinite(host);
  backend.DestroyQueue(queue);
}

TEST_CASE("test_run_model_reuses_tensor_return_for_mtp" * doctest::skip(true)) {
  MESSAGE("SKIP: _run_model tensor reuse belongs to M-mtp-1 AutoRegressiveSpeculator");
}

TEST_CASE("test_run_model_unpacks_tuple_return_for_mtp" * doctest::skip(true)) {
  MESSAGE("SKIP: tuple-vs-tensor dispatch belongs to M-mtp-1 AutoRegressiveSpeculator");
}

TEST_CASE("test_mtp_propose shape" * doctest::skip(true)) {
  MESSAGE("SKIP: propose() [batch,k] flow lands with M-mtp-1, not loader-only M-mtp-0");
}
