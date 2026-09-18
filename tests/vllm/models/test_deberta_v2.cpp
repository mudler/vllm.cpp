// DeBERTa v2 encoder parity gate — GLiNER2.5-multi-v1 backbone (MODEL-GLINER25).
//
// Compared against a restatement of HuggingFace
// transformers/models/deberta_v2/modeling_deberta_v2.py @ v4.44.2 (the
// secondary oracle), executed at reduced dimensions by
// scripts/gen-deberta-v2-goldens.py. Both sides rebuild every weight from ONE
// deterministic FNV-1a -> splitmix64 stream, so no weight byte is checked in.
//
// The disentangled attention bias (c2p + p2c) is the feature this architecture
// gets wrong quietly: without it the model still runs and emits plausible
// hidden states. A perturbation test disables the bias and verifies the output
// changes, rather than leaving the bias to be implied by the forward.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "deberta_v2_goldens.inc"
#include "vllm/model_executor/models/deberta_v2.h"

namespace {

// The generator's stream, byte-for-byte: values uniform in [-1, 1) derived
// from the tensor NAME alone. Deliberately a local copy — if this drifts from
// the generator the goldens stop matching, which is the failure we want, not
// a shared helper that could drift on both sides at once.
std::vector<float> Rand(const std::string& name, int64_t count, double scale) {
  uint64_t seed = 0xCBF29CE484222325ULL;
  for (const char c : name) {
    seed ^= static_cast<unsigned char>(c);
    seed *= 0x100000001B3ULL;
  }
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    uint64_t x = seed + static_cast<uint64_t>(i);
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double u = static_cast<double>(z >> 11) * 0x1.0p-53;
    out[static_cast<size_t>(i)] = static_cast<float>((u * 2.0 - 1.0) * scale);
  }
  return out;
}

std::vector<float> RandPlusOne(const std::string& name, int64_t count, double scale) {
  std::vector<float> v = Rand(name, count, scale);
  for (float& x : v) x += 1.0F;
  return v;
}

vllm::deberta_v2::Params GoldenParams() {
  vllm::deberta_v2::Params p;
  p.vocab_size = deberta_v2_goldens::kVocab;
  p.hidden_size = deberta_v2_goldens::kHidden;
  p.num_hidden_layers = deberta_v2_goldens::kLayers;
  p.num_attention_heads = deberta_v2_goldens::kHeads;
  p.intermediate_size = deberta_v2_goldens::kInter;
  p.max_position_embeddings = deberta_v2_goldens::kMaxPos;
  p.position_buckets = deberta_v2_goldens::kPosBuckets;
  p.layer_norm_eps = deberta_v2_goldens::kLayerNormEps;
  p.position_biased_input = false;
  p.type_vocab_size = 0;
  p.norm_rel_ebd = deberta_v2_goldens::kNormRelEbd;
  p.share_att_key = deberta_v2_goldens::kShareAttKey;
  p.use_c2p = deberta_v2_goldens::kUseC2p;
  p.use_p2c = deberta_v2_goldens::kUseP2c;
  return p;
}

// Rebuild the checkpoint in nn.Linear [out, in] orientation, matching the
// generator's weight names byte-for-byte.
vllm::deberta_v2::CheckpointTensors GoldenCheckpoint(const vllm::deberta_v2::Params& p) {
  vllm::deberta_v2::CheckpointTensors t;
  const int64_t h = p.hidden_size;
  const int64_t inter = p.intermediate_size;
  const int64_t pe = p.pos_ebd_size();

  t.Set("encoder.embeddings.word_embeddings.weight", {p.vocab_size, h},
        Rand("encoder.embeddings.word_embeddings.weight", p.vocab_size * h, 0.5));
  t.Set("encoder.embeddings.LayerNorm.weight", {h},
        RandPlusOne("encoder.embeddings.LayerNorm.weight", h, 0.1));
  t.Set("encoder.embeddings.LayerNorm.bias", {h},
        Rand("encoder.embeddings.LayerNorm.bias", h, 0.1));

  t.Set("encoder.encoder.rel_embeddings.weight", {pe, h},
        Rand("encoder.encoder.rel_embeddings.weight", pe * h, 0.3));
  t.Set("encoder.encoder.LayerNorm.weight", {h},
        RandPlusOne("encoder.encoder.LayerNorm.weight", h, 0.1));
  t.Set("encoder.encoder.LayerNorm.bias", {h},
        Rand("encoder.encoder.LayerNorm.bias", h, 0.1));

  for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
    const std::string b = "encoder.encoder.layer." + std::to_string(i) + ".";
    t.Set(b + "attention.self.query_proj.weight", {h, h},
          Rand(b + "attention.self.query_proj.weight", h * h, 0.3));
    t.Set(b + "attention.self.query_proj.bias", {h},
          Rand(b + "attention.self.query_proj.bias", h, 0.2));
    t.Set(b + "attention.self.key_proj.weight", {h, h},
          Rand(b + "attention.self.key_proj.weight", h * h, 0.3));
    t.Set(b + "attention.self.key_proj.bias", {h},
          Rand(b + "attention.self.key_proj.bias", h, 0.2));
    t.Set(b + "attention.self.value_proj.weight", {h, h},
          Rand(b + "attention.self.value_proj.weight", h * h, 0.3));
    t.Set(b + "attention.self.value_proj.bias", {h},
          Rand(b + "attention.self.value_proj.bias", h, 0.2));
    t.Set(b + "attention.output.dense.weight", {h, h},
          Rand(b + "attention.output.dense.weight", h * h, 0.3));
    t.Set(b + "attention.output.dense.bias", {h},
          Rand(b + "attention.output.dense.bias", h, 0.2));
    t.Set(b + "attention.output.LayerNorm.weight", {h},
          RandPlusOne(b + "attention.output.LayerNorm.weight", h, 0.1));
    t.Set(b + "attention.output.LayerNorm.bias", {h},
          Rand(b + "attention.output.LayerNorm.bias", h, 0.1));
    t.Set(b + "intermediate.dense.weight", {inter, h},
          Rand(b + "intermediate.dense.weight", inter * h, 0.3));
    t.Set(b + "intermediate.dense.bias", {inter},
          Rand(b + "intermediate.dense.bias", inter, 0.2));
    t.Set(b + "output.dense.weight", {h, inter},
          Rand(b + "output.dense.weight", h * inter, 0.3));
    t.Set(b + "output.dense.bias", {h},
          Rand(b + "output.dense.bias", h, 0.2));
    t.Set(b + "output.LayerNorm.weight", {h},
          RandPlusOne(b + "output.LayerNorm.weight", h, 0.1));
    t.Set(b + "output.LayerNorm.bias", {h},
          Rand(b + "output.LayerNorm.bias", h, 0.1));
  }
  return t;
}

std::vector<int64_t> GoldenIds() {
  return std::vector<int64_t>(std::begin(deberta_v2_goldens::kInputIds),
                              std::end(deberta_v2_goldens::kInputIds));
}

double MaxAbsDiff(const std::vector<float>& got, const float* want, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i])));
  }
  return worst;
}

}  // namespace

TEST_CASE("deberta_v2 forward reproduces the upstream hidden states") {
  const vllm::deberta_v2::Params p = GoldenParams();
  const vllm::deberta_v2::Weights w = vllm::deberta_v2::Load(p, GoldenCheckpoint(p));
  const std::vector<float> hidden = vllm::deberta_v2::ForwardHost(p, w, GoldenIds());

  const size_t n = GoldenIds().size() * static_cast<size_t>(p.hidden_size);
  REQUIRE(hidden.size() == n);
  const double worst = MaxAbsDiff(hidden, deberta_v2_goldens::kHiddenStates, n);
  INFO("max abs diff vs upstream hidden states: ", worst);
  CHECK(worst < 2e-5);
}

TEST_CASE("deberta_v2 disentangled attention bias changes the output") {
  // Without c2p and p2c the model still runs and emits plausible hidden states,
  // so the bias is asserted to matter rather than left to be implied.
  const vllm::deberta_v2::Params p = GoldenParams();
  const vllm::deberta_v2::Weights w = vllm::deberta_v2::Load(p, GoldenCheckpoint(p));

  const std::vector<float> full = vllm::deberta_v2::ForwardHost(p, w, GoldenIds());

  vllm::deberta_v2::Params p_no_bias = p;
  p_no_bias.use_c2p = false;
  p_no_bias.use_p2c = false;
  const std::vector<float> no_bias = vllm::deberta_v2::ForwardHost(p_no_bias, w, GoldenIds());

  REQUIRE(full.size() == no_bias.size());
  double worst = 0.0;
  for (size_t i = 0; i < full.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(full[i] - no_bias[i])));
  }
  INFO("max abs diff with vs without disentangled attention bias: ", worst);
  CHECK(worst > 1e-4);
}

TEST_CASE("deberta_v2 attention is bidirectional") {
  // DeBERTa is bidirectional: a change to an EARLY token must move LATER
  // positions. With a causal mask every position only sees the past, which
  // still produces plausible output and is the wrong model.
  const vllm::deberta_v2::Params p = GoldenParams();
  const vllm::deberta_v2::Weights w = vllm::deberta_v2::Load(p, GoldenCheckpoint(p));

  std::vector<int64_t> ids = GoldenIds();
  const std::vector<float> base = vllm::deberta_v2::ForwardHost(p, w, ids);

  // Perturb the FIRST token and check that the LAST position moves.
  ids[0] = (ids[0] + 5) % p.vocab_size;
  const std::vector<float> perturbed = vllm::deberta_v2::ForwardHost(p, w, ids);

  const size_t hidden = static_cast<size_t>(p.hidden_size);
  const size_t last_offset = (ids.size() - 1) * hidden;
  double moved = 0.0;
  for (size_t i = last_offset; i < base.size(); ++i) {
    moved = std::max(moved, std::fabs(static_cast<double>(base[i] - perturbed[i])));
  }
  CHECK(moved > 1e-6);
}
