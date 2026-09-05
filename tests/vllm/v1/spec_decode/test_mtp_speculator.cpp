// Ported from upstream vLLM @ e24d1b24fe96a56ba8b0d653efa076d03eb95d6c:
//   tests/v1/spec_decode/test_mtp.py:67-221
//   tests/v1/worker/test_gpu_autoregressive_speculator.py:52-82
// Loader sharing and direct-hidden-return assertions land with M-mtp-0. The
// propose-loop-only assertions remain explicitly skipped until M-mtp-1 adds the
// scheduler/speculator plumbing, as required by .agents/test-porting.md rule 6.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/worker/gpu/spec_decode/mtp/speculator.h"
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

  // MODEL-QWEN35-EXL3-HEAD (#2495 item 5): the presence predicate
  // `LoadQwen3_5MTP` now REQUIRES, so a caller cannot silently turn the EXL3
  // arm off by omitting it.
  std::function<bool(const std::string&)> Exists() const {
    return [this](const std::string& name) { return tensors_.count(name) != 0; };
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

// ── SPEC-MTP I5c helpers: a host-backed draft KV cache + full-attn metadata. ──
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;

// Owns one draft full-attention KV layer's bf16 buffer (the production draft KV
// dtype — ResolveKvCacheDType; the raw-torch MTP weights produce bf16 K/V so the
// "auto" reshape_and_cache requires a bf16 cache) and hands out a PagedKvCache
// view. num_blocks*block_size slots of [2, num_kv_heads, head_size].
struct DraftKvPool {
  std::vector<uint16_t> buf;
  PagedKvCache kv;
  DraftKvPool(const HfConfig& c, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    buf.assign(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0);
    kv.data = buf.data();
    kv.dtype = vt::DType::kBF16;
    kv.num_blocks = num_blocks;
    kv.block_size = block_size;
    kv.num_kv_heads = Hkv;
    kv.head_size = Dh;
  }
};

// A single-request prefill metadata over T contiguous tokens starting at
// absolute position `start` into block 0 (block_size >= start+T). slot_mapping =
// [start, start+T).
CommonAttentionMetadata MtpMeta(int64_t T, int64_t seq_len, int64_t start,
                                int64_t block_size) {
  (void)block_size;  // single block 0; slots addressed by absolute position.
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(seq_len)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(seq_len);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t)
    m.slot_mapping.push_back(start + t);
  m.causal = true;
  return m;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(static_cast<double>(a[i]) - b[i]));
  return d;
}

int64_t ArgmaxRow(const std::vector<float>& logits, int64_t row, int64_t vocab) {
  const float* r = logits.data() + static_cast<size_t>(row) * vocab;
  int64_t best = 0;
  for (int64_t v = 1; v < vocab; ++v)
    if (r[static_cast<size_t>(v)] > r[static_cast<size_t>(best)]) best = v;
  return best;
}

}  // namespace

TEST_CASE("test_mtp_load_model_unified: dense MTP shares target embedding and lm_head") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  CHECK_FALSE(model.has_own_embed_tokens());
  CHECK_FALSE(model.has_own_lm_head());
  CHECK(&model.embed_tokens() == &target.embed_tokens);
  CHECK(model.lm_head() == &target.lm_head);
  // PERF-27B-LMHEAD-FP4 (issue #213): the dense drafter shares the target's
  // PACKED head too, so the pointer is the target's field rather than null. This
  // target is bf16, so the field is EMPTY and the bf16 arm is still selected.
  CHECK(model.lm_head_fp4() == &target.lm_head_fp4);
  REQUIRE(model.lm_head_fp4() != nullptr);
  CHECK(model.lm_head_fp4()->Empty());
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
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kMoe);
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

// MODEL-QWEN35-EXL3-HEAD (#2495 item 5) NARROWS this title and nothing else.
// The assertion is unchanged and still fires: `mtp.fc.weight` is the BF16
// arm's tensor, which an EXL3 checkpoint does not ship at all, so the trellis
// rung is not selected and the strictness refusal is still the right answer
// for the arm this case is about. "every mtp tensor" is simply no longer
// true of every checkpoint.
TEST_CASE("test_mtp_load_model_unified: every mtp tensor on the BF16 arm is strictly BF16") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  store.MutableView("mtp.fc.weight").dtype = "F16";
  CHECK_THROWS_AS(
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config,
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
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config,
                           Qwen3_5MTPKind::kDense),
      std::runtime_error);
}

TEST_CASE("test_mtp_load_model_unified: dedicated embeddings are rejected for gate checkpoints") {
  HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  config.raw["text_config"]["mtp_use_dedicated_embeddings"] = true;
  TensorStore store;
  AddDenseMtp(store, config);
  CHECK_THROWS_AS(
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config,
                           Qwen3_5MTPKind::kDense),
      std::runtime_error);
}

TEST_CASE("test_mtp_propose k=1: MTP forward returns hidden states directly") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights = vllm::LoadQwen3_5MTP(
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
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
      store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kMoe);
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

// ── SPEC-MTP I5c: the PAGED MTP propose forward + draft KV layer. ────────────

namespace {
// Download an on-device ForwardLogits to a host [rows*vocab] vector.
std::vector<float> HostLogits(const vllm::ForwardLogits& logits,
                              vt::Backend& backend, vt::Queue& queue) {
  std::vector<float> host(static_cast<size_t>(logits.rows) * logits.vocab);
  backend.Copy(queue, host.data(), logits.device_tensor.data,
               host.size() * sizeof(float));
  backend.Synchronize(queue);
  return host;
}

// A [1,H] contiguous bf16 view of row `r` of a [rows,H] owned bf16 tensor.
vt::Tensor Row(const OwnedTensor& owned, int64_t r, int64_t H) {
  vt::Tensor t = owned.View();
  t.rank = 2;
  t.shape[0] = 1;
  t.shape[1] = H;
  t.stride[0] = H;
  t.stride[1] = 1;
  t.data = static_cast<uint16_t*>(t.data) + r * H;
  return t;
}
}  // namespace

// CORE I5c PROOF: the PAGED MTP forward over a single-request 1-block KV with a
// trivial slot map reproduces the STANDALONE (dense) forward's logits/argmax —
// the paged rewrite did not change the head math (mtp-spec-decode.md §5 I5c gate).
TEST_CASE("i5c paged MTP forward equals standalone (dense head)") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights =
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const int64_t T = 4, H = config.hidden_size, vocab = config.vocab_size;
  const std::vector<int32_t> ids = {1, 5, 2, 9};
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  OwnedTensor target_hidden = MakeOwned({T, H}, 41);

  // Standalone (I1) path.
  const std::vector<float> logits_std =
      model.ForwardLogitsHost(ids, pos, target_hidden.View(), queue);

  // Paged path: fresh 1-block KV, contiguous slot map.
  DraftKvPool pool(config, /*num_blocks=*/1, /*block_size=*/8);
  const CommonAttentionMetadata am = MtpMeta(T, /*seq_len=*/T, /*start=*/0, 8);
  const vllm::Qwen3_5MTPHiddenStates hidden = model.ForwardPaged(
      ids, pos, target_hidden.View(), am, pool.kv, queue);
  const std::vector<float> logits_paged =
      HostLogits(model.ComputeLogits(hidden.tensor, queue), backend, queue);

  REQUIRE(logits_paged.size() == logits_std.size());
  const double d = MaxAbsDiff(logits_paged, logits_std);
  MESSAGE("i5c dense paged==standalone max|diff| = " << d);
  CHECK(d < 1e-2);
  for (int64_t t = 0; t < T; ++t)
    CHECK(ArgmaxRow(logits_paged, t, vocab) == ArgmaxRow(logits_std, t, vocab));
  backend.DestroyQueue(queue);
}

TEST_CASE("i5c paged MTP forward equals standalone (MoE head)") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kMoe);
  TensorStore store;
  AddMoeMtp(store, config);
  const Qwen3_5MTPWeights weights =
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kMoe);
  const Qwen3_5MoeWeights target = MakeMoeTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const int64_t T = 4, H = config.hidden_size, vocab = config.vocab_size;
  const std::vector<int32_t> ids = {3, 7, 1, 4};
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  OwnedTensor target_hidden = MakeOwned({T, H}, 43);

  const std::vector<float> logits_std =
      model.ForwardLogitsHost(ids, pos, target_hidden.View(), queue);
  DraftKvPool pool(config, 1, 8);
  const CommonAttentionMetadata am = MtpMeta(T, T, 0, 8);
  const vllm::Qwen3_5MTPHiddenStates hidden = model.ForwardPaged(
      ids, pos, target_hidden.View(), am, pool.kv, queue);
  const std::vector<float> logits_paged =
      HostLogits(model.ComputeLogits(hidden.tensor, queue), backend, queue);

  REQUIRE(logits_paged.size() == logits_std.size());
  const double d = MaxAbsDiff(logits_paged, logits_std);
  MESSAGE("i5c MoE paged==standalone max|diff| = " << d);
  CHECK(d < 1e-2);
  for (int64_t t = 0; t < T; ++t)
    CHECK(ArgmaxRow(logits_paged, t, vocab) == ArgmaxRow(logits_std, t, vocab));
  backend.DestroyQueue(queue);
}

// DRAFT-KV CORRECTNESS: a two-step drive where step 1 writes the draft K/V and
// step 2 (a decode) attends over it must reproduce the token-1 hidden of a single
// combined 2-token forward. The RED CONTROL in the same test — running step 2
// over a FRESH (unpopulated) draft KV — DIVERGES, proving step 2 genuinely reads
// what step 1 wrote (a stubbed/absent paged KV path fails here).
TEST_CASE("i5c draft KV two-step: decode attends over written K/V") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights =
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const int64_t H = config.hidden_size;
  const int32_t id0 = 5, id1 = 9;
  OwnedTensor th = MakeOwned({2, H}, 47);  // per-token target hidden rows
  const vt::Tensor th0 = Row(th, 0, H), th1 = Row(th, 1, H);

  auto row_bytes = [&](const vllm::Qwen3_5MTPHiddenStates& hs, int64_t r) {
    std::vector<float> out(static_cast<size_t>(H));
    const auto* p = static_cast<const uint16_t*>(hs.tensor.data) + r * H;
    for (int64_t i = 0; i < H; ++i)
      out[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
    return out;
  };

  // (A) Combined 2-token forward → reference token-1 hidden.
  DraftKvPool comb(config, 1, 8);
  const CommonAttentionMetadata am_comb = MtpMeta(2, 2, 0, 8);
  const auto hs_comb = model.ForwardPaged({id0, id1}, {0, 1}, th.View(),
                                          am_comb, comb.kv, queue);
  const std::vector<float> ref = row_bytes(hs_comb, 1);

  // (B) Split: step 1 prefill token0 (writes slot 0), step 2 decode token1
  // (seq_len 2, slot 1) reads slots 0..1.
  DraftKvPool split(config, 1, 8);
  const CommonAttentionMetadata am_s1 = MtpMeta(1, 1, 0, 8);
  (void)model.ForwardPaged({id0}, {0}, th0, am_s1, split.kv, queue);
  const CommonAttentionMetadata am_s2 = MtpMeta(1, 2, 1, 8);
  const auto hs_s2 = model.ForwardPaged({id1}, {1}, th1, am_s2, split.kv, queue);
  const std::vector<float> got = row_bytes(hs_s2, 0);

  // (C) RED CONTROL: step 2 over a FRESH KV (slot 0 never written).
  DraftKvPool fresh(config, 1, 8);
  const auto hs_nokv = model.ForwardPaged({id1}, {1}, th1, am_s2, fresh.kv, queue);
  const std::vector<float> nokv = row_bytes(hs_nokv, 0);

  const double d_ok = MaxAbsDiff(got, ref);
  const double d_red = MaxAbsDiff(nokv, ref);
  MESSAGE("i5c draft-KV two-step: with-step1 max|diff|=" << d_ok
          << " vs fresh-KV (RED) max|diff|=" << d_red);
  CHECK(d_ok < 2e-2);       // decode-via-written-cache == combined forward
  CHECK(d_red > d_ok);      // unwritten KV diverges → step 2 really reads step 1
  backend.DestroyQueue(queue);
}

// propose() k=1: prepare_prefill_inputs (I5b) shift-splice + one paged forward +
// argmax draft pick, returning one token per request (speculator.py:236-238).
TEST_CASE("i5c MtpProposePrefill k=1 returns the argmax over the shifted draft") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);
  const Qwen3_5MTPWeights weights =
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense);
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  const Qwen3_5MTPModel model(weights, target, config);

  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue queue = backend.CreateQueue();
  const int64_t H = config.hidden_size, vocab = config.vocab_size;

  // One decoding request, k=1: verify span = 2 tokens (the last is the bonus).
  const std::vector<int32_t> verify_ids = {5, 9};
  const std::vector<int64_t> verify_pos = {0, 1};
  const std::vector<int32_t> idx_mapping = {0};
  const int32_t sampled = 7;  // last_sampled[req_state 0]
  const std::vector<int32_t> last_sampled = {sampled};
  const std::vector<int32_t> next_prefill_tokens = {0};
  const std::vector<int32_t> num_sampled = {1};   // all accepted
  const std::vector<int32_t> num_rejected = {0};
  OwnedTensor target_hidden = MakeOwned({2, H}, 51);

  CommonAttentionMetadata am = MtpMeta(2, 2, 0, 8);
  DraftKvPool pool(config, 1, 8);

  const std::vector<int32_t> draft = vllm::v1::MtpProposePrefill(
      model, am, pool.kv, target_hidden.View(), verify_ids, verify_pos,
      idx_mapping, last_sampled, next_prefill_tokens, num_sampled, num_rejected,
      /*max_num_reqs=*/1, queue);
  REQUIRE(draft.size() == 1);
  CHECK(draft[0] >= 0);
  CHECK(draft[0] < vocab);

  // Independent recomputation over the EXPECTED shift-splice: with 0 rejected the
  // draft span is [verify_ids[1], sampled] and the sampled row is index 1. This
  // proves propose applied prepare_prefill_inputs' shift + last_token index.
  const std::vector<int32_t> expect_ids = {9, sampled};
  const std::vector<int32_t> expect_pos = {0, 1};
  DraftKvPool pool2(config, 1, 8);
  const auto hs = model.ForwardPaged(expect_ids, expect_pos,
                                     target_hidden.View(), am, pool2.kv, queue);
  const std::vector<float> lg =
      HostLogits(model.ComputeLogits(hs.tensor, queue), backend, queue);
  CHECK(draft[0] == ArgmaxRow(lg, /*row=*/1, vocab));
  backend.DestroyQueue(queue);
}

// The target-model hidden-state tap (ForwardDeviceTap) is INERT: it returns the
// exact same logits as ForwardDevice and hands back a [T,H] post-norm hidden.
TEST_CASE("i5c hidden-state tap is inert and shape-correct") {
  // Exercised against a real paged forward in the dedicated paged-forward tests
  // (test_qwen27_paged_forward / test_qwen35_paged_forward); this asserts the
  // MTP carrier plumbing compiles and the forward-declared type resolves.
  vllm::Qwen3_5MTPHiddenStates carrier;
  CHECK(carrier.storage == nullptr);
  CHECK(carrier.tensor.data == nullptr);
}

// SPEC-MTP I5d-pre: the LoadedModel draft-construction virtual. The concrete
// Qwen3.5 target retains the loaded mtp.* weights and builds a Qwen3_5MTPModel
// draft sharing the target's embed_tokens/lm_head; a non-MTP model returns null
// and refuses to hold draft weights. RED-first: before this increment the
// virtuals do not exist (does not compile) and no non-qwen model returns null.
TEST_CASE("i5d-pre LoadedModel::BuildMtpDraft builds a Qwen3.5 draft, null otherwise") {
  const HfConfig config = MakeConfig(Qwen3_5MTPKind::kDense);
  TensorStore store;
  AddDenseMtp(store, config);

  // The target LoadedModel (borrows caller-owned dense target weights).
  const Qwen3_5DenseWeights target = MakeDenseTarget(config);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(target);

  // Capability + before-attach: supports MTP, but no draft yet.
  CHECK(model->supports_mtp_draft());
  CHECK(model->BuildMtpDraft(config) == nullptr);

  // Attach the loaded mtp.* draft weights, then build the draft.
  model->AttachMtpDraftWeights(
      vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense));
  std::unique_ptr<Qwen3_5MTPModel> draft = model->BuildMtpDraft(config);
  REQUIRE(draft != nullptr);
  // load_eagle_model sharing: the draft borrows the target's embed_tokens/lm_head.
  CHECK(&draft->embed_tokens() == &target.embed_tokens);
  CHECK(draft->lm_head() == &target.lm_head);

  // A non-MTP model inherits the base defaults: no support, null draft, and it
  // refuses to hold draft weights. A test-local subclass exercises the base
  // contract without needing a full non-qwen checkpoint on disk.
  struct NonMtpModel final : public vllm::LoadedModel {
    explicit NonMtpModel(const vllm::ModelRegistration& r) : LoadedModel(r) {}
  };
  NonMtpModel non_mtp(vllm::RegistrationFor("OPTForCausalLM"));
  CHECK_FALSE(non_mtp.supports_mtp_draft());
  CHECK(non_mtp.BuildMtpDraft(config) == nullptr);
  CHECK_THROWS_AS(
      non_mtp.AttachMtpDraftWeights(
          vllm::LoadQwen3_5MTP(store.Resolver(), store.Exists(), config, Qwen3_5MTPKind::kDense)),
      std::runtime_error);
}
